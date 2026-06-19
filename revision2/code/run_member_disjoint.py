#!/usr/bin/env python3
"""
Member-disjoint re-analysis for the Medical Care second revision.

This runner partitions the member population into disjoint training, validation,
and test sets so that no member contributes observations to more than one split.
It preserves the prospective forecast structure of the original analysis: for
every member, cross-sectional features end at the training cutoff and the forecast
target is the prospective May-Jul 2025 three-month total. Models are fit on
training members only, the hybrid calibration factor and hyperparameters are
selected on validation members, and all reported metrics are computed on the
held-out test members.

This addresses the second reviewer's concern that the original temporal-only split
reused the same members across the training, validation, and test windows.

Metric conventions (consistent across every model):
  - Mean absolute error (MAE) and RMSE are per-member-per-month: for each test
    member the absolute error is averaged across the three forecast months, then
    averaged across members. Cross-sectional models, which predict a three-month
    total, are allocated evenly across the three months; time-series and hybrid
    models use their native monthly predictions. This is the natural scale for a
    per-member-per-month cost study and is identical across model classes.
  - Calibrated R-squared and the predictive ratio are computed on the patient-level
    three-month total (sum of the three monthly values), the actuarial discrimination
    and calibration scale.

The torch-dependent Chronos and panel-conditioned forecasts run in a separate
process (chronos_stage.py) because torch and LightGBM cannot share an OpenMP
runtime on macOS.

Usage:
  DATA_ROOT=/path/to/real_inputs python3 run_member_disjoint.py
  DATA_ROOT=/path/to/real_inputs python3 run_member_disjoint.py --quick 5000
  DATA_ROOT=/path/to/real_inputs python3 run_member_disjoint.py --reuse-forecasts
"""
import os
import sys
import json
import time
import argparse
import warnings
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# This process must NOT import torch: on macOS, torch's OpenMP runtime and
# LightGBM's bundled OpenMP runtime cannot safely coexist in one process. All
# torch work runs in the chronos_stage.py subprocess invoked below.
PKG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG_ROOT))
THIS_DIR = Path(__file__).resolve().parent

from src.data.load_claims import build_cross_sectional_features, build_patient_time_series
from src.models.baselines import (
    XGBoostBaseline, RandomForestBaseline, TwoPartModel, DemographicGLM,
    StackingEnsemble, build_targets,
)
from src.models.ts_baselines import NaiveBaselines
from src.evaluation.metrics import (
    r_squared, r_squared_calibrated, predictive_ratio, high_cost_identification, decile_analysis,
)

try:
    from src.models.baselines import LightGBMBaseline
except Exception:
    LightGBMBaseline = None

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SPLIT_SEED = 20260617
FRAC_TRAIN, FRAC_VAL, FRAC_TEST = 0.70, 0.15, 0.15
HORIZON = 3
TRAIN_CUTOFF = pd.Timestamp("2025-01-01")
CONTEXT_CUTOFF = pd.Timestamp("2025-04-01")
TARGET_START = pd.Timestamp("2025-04-01")    # predicts the 3 months after this -> May-Jul 2025
LOOKBACK_MONTHS = 12
MIN_HISTORY_MONTHS = 6
MIN_ENROLLED_DAYS = 15
MIN_TOTAL_MONTHS = 9
CHRONOS_VAL_CAP = 6000
PANELFM_TRAIN_CAP = 2000
N_BOOT = 2000

DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(PKG_ROOT / "data" / "real_inputs")))
OUT_DIR = PKG_ROOT / "revision2" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CFG = {
    "xgboost": dict(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0),
    "random_forest": dict(n_estimators=500, max_depth=12, min_samples_leaf=20, max_features="sqrt"),
    "lightgbm": dict(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                     colsample_bytree=0.8, min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0),
}


def log(msg):
    print(msg, flush=True)


# -----------------------------------------------------------------------------
# Data loading and cohort construction
# -----------------------------------------------------------------------------
def load_cohort():
    log(f"Loading data from {DATA_ROOT}")
    op = DATA_ROOT / "outcomes_monthly.parquet"
    outcomes = pd.read_parquet(op) if op.exists() else pd.read_csv(
        DATA_ROOT / "outcomes_monthly.csv", parse_dates=["month_year"])
    outcomes.columns = outcomes.columns.str.lower().str.strip()
    outcomes["month_year"] = pd.to_datetime(outcomes["month_year"])

    ap = DATA_ROOT / "member_attributes.parquet"
    attributes = pd.read_parquet(ap) if ap.exists() else pd.read_csv(DATA_ROOT / "member_attributes.csv")
    attributes.columns = attributes.columns.str.lower().str.strip()

    ep = DATA_ROOT / "eligibility.parquet"
    eligibility = pd.read_parquet(ep) if ep.exists() else pd.read_csv(DATA_ROOT / "eligibility.csv")
    eligibility.columns = eligibility.columns.str.lower().str.strip()

    if "enrolled_days" in outcomes.columns:
        outcomes = outcomes[outcomes["enrolled_days"] >= MIN_ENROLLED_DAYS]
    pm = outcomes.groupby("person_id").size()
    keep = pm[pm >= MIN_TOTAL_MONTHS].index
    outcomes = outcomes[outcomes["person_id"].isin(keep)].copy()
    outcomes["person_id"] = outcomes["person_id"].astype(str)
    for col in ["total_paid", "emergency_department_ct", "acute_inpatient_ct"]:
        if col in outcomes.columns:
            outcomes[col] = outcomes[col].fillna(0)
    log(f"Cohort after filters: {outcomes.shape[0]:,} patient-months, "
        f"{outcomes['person_id'].nunique():,} members")
    return outcomes, attributes, eligibility


def member_disjoint_split(members_with_cost):
    df = members_with_cost.copy()
    df["stratum"] = 0
    pos = df["lookback_cost"] > 0
    if pos.sum() > 5:
        df.loc[pos, "stratum"] = pd.qcut(df.loc[pos, "lookback_cost"], q=5,
                                         labels=False, duplicates="drop") + 1
    rng = np.random.default_rng(SPLIT_SEED)
    train_ids, val_ids, test_ids = [], [], []
    for _, grp in df.groupby("stratum"):
        ids = grp["person_id"].values.copy()
        rng.shuffle(ids)
        n = len(ids); n_tr = int(round(n * FRAC_TRAIN)); n_va = int(round(n * FRAC_VAL))
        train_ids.extend(ids[:n_tr]); val_ids.extend(ids[n_tr:n_tr + n_va]); test_ids.extend(ids[n_tr + n_va:])
    train_ids, val_ids, test_ids = set(train_ids), set(val_ids), set(test_ids)
    assert not (train_ids & val_ids) and not (train_ids & test_ids) and not (val_ids & test_ids)
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def temporal_split(eligible, outcomes):
    """Non-random temporal (entry-cohort) split per Steyerberg 17.3: earliest-entry members
    to train, most-recent-entry members to test. Entry = each member's first observed month."""
    entry = outcomes.groupby("person_id")["month_year"].min()
    e = sorted(eligible, key=lambda p: (entry.get(p, pd.Timestamp.max), p))
    n = len(e); n_tr = int(round(n * FRAC_TRAIN)); n_va = int(round(n * FRAC_VAL))
    return {"train": set(e[:n_tr]), "val": set(e[n_tr:n_tr + n_va]), "test": set(e[n_tr + n_va:])}


def prepare(outcomes, attributes, eligibility, quick=None, temporal=False):
    if quick:
        rng0 = np.random.default_rng(SPLIT_SEED)
        members = outcomes["person_id"].unique()
        keep = set(rng0.choice(members, size=min(quick, len(members)), replace=False))
        outcomes = outcomes[outcomes["person_id"].isin(keep)].copy()
        log(f"QUICK pilot: restricted to {outcomes['person_id'].nunique():,} members, "
            f"{outcomes.shape[0]:,} patient-months")

    feats = build_cross_sectional_features(
        outcomes, attributes, eligibility, lookback_end=TRAIN_CUTOFF, lookback_months=LOOKBACK_MONTHS)
    feats["person_id"] = feats["person_id"].astype(str)
    for col in list(feats.columns):
        if feats[col].dtype == object and col != "person_id":
            d = pd.get_dummies(feats[col], prefix=col, drop_first=True)
            feats = pd.concat([feats.drop(columns=[col]), d], axis=1)
    feats = feats.fillna(0)

    targets = build_targets(outcomes, prediction_start=TARGET_START, horizon_months=HORIZON)
    targets["person_id"] = targets["person_id"].astype(str)

    pre = outcomes[outcomes["month_year"] <= CONTEXT_CUTOFF]
    context, _ = build_patient_time_series(pre, min_months=MIN_HISTORY_MONTHS)
    context = {str(k): np.asarray(v, dtype=np.float32) for k, v in context.items()}

    target_window = outcomes[(outcomes["month_year"] > TARGET_START) &
                             (outcomes["month_year"] <= TARGET_START + pd.DateOffset(months=HORIZON))]
    actuals = {}
    for pid, g in target_window.groupby("person_id"):
        vals = g.sort_values("month_year")["total_paid"].fillna(0).values.astype(np.float32)
        if len(vals) == HORIZON:   # complete-case: members enrolled across the full target window
            actuals[str(pid)] = vals

    eligible = sorted(set(feats["person_id"]) & set(targets["person_id"]) & set(context) & set(actuals))
    log(f"Members eligible for the disjoint analysis: {len(eligible):,}")

    lb_cost = feats.set_index("person_id")["mean_total_paid"].reindex(eligible).fillna(0)
    mwc = pd.DataFrame({"person_id": eligible, "lookback_cost": lb_cost.values})
    if quick:
        eligible = set(eligible)
        mwc = mwc[mwc["person_id"].isin(eligible)].reset_index(drop=True)
    if temporal:
        splits = temporal_split(list(mwc["person_id"]), outcomes)
        log(f"TEMPORAL entry-cohort split: train={len(splits['train']):,} val={len(splits['val']):,} test={len(splits['test']):,}")
    else:
        splits = member_disjoint_split(mwc)
        log(f"Disjoint split: train={len(splits['train']):,} val={len(splits['val']):,} test={len(splits['test']):,}")
    return {"feats": feats, "targets": targets, "context": context, "actuals": actuals,
            "splits": splits, "outcomes": outcomes}


# -----------------------------------------------------------------------------
# Evaluation: per-month MAE/RMSE + patient-total R^2/PR/discrimination
# -----------------------------------------------------------------------------
def eval3(pred3, act3):
    """pred3/act3: dict person_id -> length-H monthly array. Returns metric dict."""
    pids = sorted(set(pred3) & set(act3))
    pm_mae = np.array([np.mean(np.abs(pred3[p] - act3[p])) for p in pids])
    pm_mse = np.array([np.mean((pred3[p] - act3[p]) ** 2) for p in pids])
    yt = np.array([float(np.sum(act3[p])) for p in pids])
    yp = np.array([float(np.sum(pred3[p])) for p in pids])
    hc = high_cost_identification(yt, yp).get("top_10pct", {})
    return {
        "mae": float(pm_mae.mean()),
        "rmse": float(np.sqrt(pm_mse.mean())),
        "r_squared": r_squared(yt, yp),
        "r_squared_calibrated": r_squared_calibrated(yt, yp),
        "predictive_ratio": predictive_ratio(yt, yp)["overall"],
        "auroc": hc.get("c_statistic", np.nan),
        "ppv_at_top10pct": hc.get("ppv", np.nan),
        "lift": hc.get("lift", np.nan),
        "n_patients": len(pids),
    }


def flat3(total_map, act3, horizon=HORIZON):
    """Allocate a 3-month total prediction evenly across the months."""
    return {p: np.full(len(act3[p]), float(total_map[p]) / len(act3[p]))
            for p in total_map if p in act3}


def vectors_from_forecasts(forecasts, actuals, horizon=HORIZON):
    pred3, act3 = {}, {}
    for pid, f in forecasts.items():
        if pid in actuals:
            a = np.asarray(actuals[pid], dtype=float)[:horizon]
            p = np.asarray(f, dtype=float)[:len(a)]
            if len(a) >= 1 and len(p) == len(a):
                pred3[pid] = p
                act3[pid] = a
    return pred3, act3


# -----------------------------------------------------------------------------
# Cross-sectional models (fit on train, calibrate on val, score on test)
# -----------------------------------------------------------------------------
def feature_columns(df):
    drop = {"person_id", "total_paid_sum", "ed_visits_sum", "ip_admits_sum",
            "high_cost_flag", "any_ed_flag", "any_ip_flag", "n_months"}
    return [c for c in df.columns if c not in drop and
            df[c].dtype in (np.float64, np.int64, np.float32, np.int32, np.uint8, bool)]


def run_cross_sectional(data):
    feats, targets, splits, actuals = data["feats"], data["targets"], data["splits"], data["actuals"]
    merged = feats.merge(targets, on="person_id", how="inner")
    tr = merged[merged["person_id"].isin(splits["train"])].copy()
    va = merged[merged["person_id"].isin(splits["val"])].copy()
    te = merged[merged["person_id"].isin(splits["test"]) & merged["person_id"].isin(set(actuals))].copy()
    fcols = feature_columns(tr)
    thr_train = tr["total_paid_sum"].quantile(0.90)
    y_tr_cls = (tr["total_paid_sum"] >= thr_train).astype(int)
    Xtr, Xva, Xte = tr[fcols], va[fcols], te[fcols]
    ytr = tr["total_paid_sum"].values
    te_ids = te["person_id"].tolist()
    act3 = {p: actuals[p] for p in te_ids}

    specs = [("demographic_glm", DemographicGLM, {}),
             ("xgboost", XGBoostBaseline, MODEL_CFG["xgboost"]),
             ("random_forest", RandomForestBaseline, MODEL_CFG["random_forest"]),
             ("two_part", TwoPartModel, MODEL_CFG["xgboost"])]
    if LightGBMBaseline is not None:
        specs.append(("lightgbm", LightGBMBaseline, MODEL_CFG["lightgbm"]))

    results, perpatient, fitted = {}, {}, {}
    for name, cls, cfg in specs:
        reg = cls(task="regression", config=cfg)
        reg.fit(Xtr, pd.Series(ytr))
        total_te = {pid: max(float(v), 0.0) for pid, v in zip(te_ids, reg.predict(Xte))}
        pred3 = flat3(total_te, act3)
        results[name] = eval3(pred3, act3)
        perpatient[name] = {"pred3": pred3, "act3": act3}
        fitted[name] = reg
        log(f"  {name:16s} MAE={results[name]['mae']:.1f} R2cal={results[name]['r_squared_calibrated']:.4f} "
            f"PR={results[name]['predictive_ratio']:.3f} AUROC={results[name]['auroc']:.3f}")

    sreg = StackingEnsemble(task="regression", config=MODEL_CFG)
    sreg.fit(Xtr, pd.Series(ytr))
    total_te = {pid: max(float(v), 0.0) for pid, v in zip(te_ids, sreg.predict(Xte))}
    pred3 = flat3(total_te, act3)
    results["stacking"] = eval3(pred3, act3)
    perpatient["stacking"] = {"pred3": pred3, "act3": act3}
    log(f"  {'stacking':16s} MAE={results['stacking']['mae']:.1f} "
        f"R2cal={results['stacking']['r_squared_calibrated']:.4f} PR={results['stacking']['predictive_ratio']:.3f}")

    # Tweedie gradient boosting: the principled single-model approach for zero-inflated,
    # right-skewed cost (compound Poisson-gamma); targets the conditional mean.
    import xgboost as xgb
    tw = xgb.XGBRegressor(objective="reg:tweedie", tweedie_variance_power=1.5,
                          n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0,
                          random_state=42, n_jobs=-1)
    tw.fit(Xtr.values, ytr)
    total_te = {pid: max(float(v), 0.0) for pid, v in zip(te_ids, tw.predict(Xte.values))}
    pred3 = flat3(total_te, act3)
    results["tweedie"] = eval3(pred3, act3)
    perpatient["tweedie"] = {"pred3": pred3, "act3": act3}
    fitted["tweedie"] = tw
    log(f"  {'tweedie':16s} MAE={results['tweedie']['mae']:.1f} "
        f"R2cal={results['tweedie']['r_squared_calibrated']:.4f} PR={results['tweedie']['predictive_ratio']:.3f}")

    # Quantile gradient boosting at the conditional median: a cross-sectional model that, like
    # the foundation model, targets the median rather than the mean, isolating the loss-function
    # (median-vs-mean) origin of the MAE/calibration trade-off within one model class.
    import lightgbm as lgb
    qm = lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=500, max_depth=6,
                           learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                           min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0, random_state=42,
                           n_jobs=-1, verbose=-1)
    qm.fit(Xtr.values, ytr)
    total_te = {pid: max(float(v), 0.0) for pid, v in zip(te_ids, qm.predict(Xte.values))}
    pred3 = flat3(total_te, act3)
    results["quantile_median_cs"] = eval3(pred3, act3)
    perpatient["quantile_median_cs"] = {"pred3": pred3, "act3": act3}
    log(f"  {'quantile_median':16s} MAE={results['quantile_median_cs']['mae']:.1f} "
        f"R2cal={results['quantile_median_cs']['r_squared_calibrated']:.4f} PR={results['quantile_median_cs']['predictive_ratio']:.3f}")

    ctx = {"test_ids": te_ids, "fitted": fitted, "Xtr": Xtr, "fcols": fcols, "perpatient": perpatient}
    return results, ctx


# -----------------------------------------------------------------------------
# Time-series + foundation models (torch-isolated subprocess)
# -----------------------------------------------------------------------------
def run_time_series_and_fm(data, cs_ctx, reuse_forecasts=False):
    context, actuals, splits = data["context"], data["actuals"], data["splits"]
    test_ids, val_ids, train_ids = set(splits["test"]), set(splits["val"]), set(splits["train"])
    test_ctx = {p: context[p] for p in context if p in test_ids and p in actuals}
    val_ctx = {p: context[p] for p in context if p in val_ids and p in actuals}
    train_ctx = {p: context[p] for p in context if p in train_ids and p in actuals}

    rng = np.random.default_rng(SPLIT_SEED)
    if len(val_ctx) > CHRONOS_VAL_CAP:
        keep = set(rng.choice(list(val_ctx), CHRONOS_VAL_CAP, replace=False))
        val_ctx = {p: val_ctx[p] for p in keep}
    train_pids = list(train_ctx)
    if len(train_pids) > PANELFM_TRAIN_CAP:
        train_pids = list(rng.choice(train_pids, PANELFM_TRAIN_CAP, replace=False))
    train_ctx_sub = {p: train_ctx[p] for p in train_pids}

    results, perpatient = {}, {}
    for name, fn in [("naive_last", lambda s: NaiveBaselines.last_value(s, HORIZON)),
                     ("naive_mean3", lambda s: NaiveBaselines.trailing_mean(s, HORIZON, window=3))]:
        pred3, act3 = vectors_from_forecasts(fn(test_ctx), actuals)
        results[name] = eval3(pred3, act3)
        perpatient[name] = {"pred3": pred3, "act3": act3}
        log(f"  {name:16s} MAE={results[name]['mae']:.1f} R2cal={results[name]['r_squared_calibrated']:.4f} "
            f"PR={results[name]['predictive_ratio']:.3f}")

    wd = OUT_DIR / "_torch_workdir"
    if not (reuse_forecasts and (wd / "forecasts.joblib").exists()):
        wd.mkdir(parents=True, exist_ok=True)
        union_pids = set(train_ctx_sub) | set(val_ctx) | set(test_ctx)
        joblib.dump({p: context[p] for p in union_pids}, wd / "context.joblib")
        joblib.dump({p: np.asarray(actuals[p], dtype=np.float32) for p in union_pids if p in actuals},
                    wd / "actuals.joblib")
        (wd / "ids.json").write_text(json.dumps({
            "train_sub": list(train_ctx_sub), "val": list(val_ctx), "test": list(test_ctx)}))
        from src.models.patient_encoder import PatientEncoder, PatientEmbeddingStore
        xgb_model = cs_ctx["fitted"]["xgboost"]
        encoder = PatientEncoder(embedding_dim=8, method="leaf", include_risk_score=True)
        encoder.fit(xgb_model, cs_ctx["Xtr"], pd.Series(np.zeros(len(cs_ctx["Xtr"]))))
        fe = data["feats"].copy()
        for c in xgb_model.feature_names:
            if c not in fe.columns:
                fe[c] = 0
        fe = fe[["person_id"] + xgb_model.feature_names].fillna(0)
        fe = fe[fe["person_id"].isin(union_pids)]
        store = PatientEmbeddingStore(); store.build(encoder, fe)
        joblib.dump({"embeddings": {p: np.asarray(v).tolist() for p, v in store.embeddings.items()},
                     "embedding_dim": int(store.embedding_dim)}, wd / "embeddings.joblib")
        log("  launching torch-isolated chronos_stage subprocess...")
        t0 = time.time()
        proc = subprocess.run([sys.executable, str(THIS_DIR / "chronos_stage.py"), "--workdir", str(wd)],
                              capture_output=True, text=True)
        log(proc.stdout.strip())
        if proc.returncode != 0:
            log(f"  chronos_stage stderr:\n{proc.stderr[-2000:]}")
            raise RuntimeError(f"chronos_stage failed (exit {proc.returncode})")
        log(f"  chronos_stage done in {time.time()-t0:.0f}s")
    else:
        log("  reusing existing forecasts.joblib")

    fc_all = joblib.load(wd / "forecasts.joblib")
    chronos_fc = {p: np.asarray(v, dtype=np.float32) for p, v in fc_all["chronos"].items()}

    pred3, act3 = vectors_from_forecasts({p: chronos_fc[p] for p in test_ctx if p in chronos_fc}, actuals)
    results["chronos_zeroshot"] = eval3(pred3, act3)
    perpatient["chronos_zeroshot"] = {"pred3": pred3, "act3": act3}
    log(f"  {'chronos_zeroshot':16s} MAE={results['chronos_zeroshot']['mae']:.1f} "
        f"R2cal={results['chronos_zeroshot']['r_squared_calibrated']:.4f} PR={results['chronos_zeroshot']['predictive_ratio']:.3f}")

    for name in ["panelfm_xreg", "panelfm_adapter"]:
        if name not in fc_all:
            continue
        fc = {p: np.asarray(v, dtype=np.float32) for p, v in fc_all[name].items()}
        pred3, act3 = vectors_from_forecasts(fc, actuals)
        results[name] = eval3(pred3, act3)
        perpatient[name] = {"pred3": pred3, "act3": act3}
        log(f"  {name:16s} MAE={results[name]['mae']:.1f} R2cal={results[name]['r_squared_calibrated']:.4f} "
            f"PR={results[name]['predictive_ratio']:.3f}")

    # Chronos monthly forecasts on validation members (for the hybrid calibration factor check).
    ts_ctx = {"perpatient": perpatient,
              "chronos_fc": chronos_fc, "val_ctx_ids": list(val_ctx)}
    return results, ts_ctx


# -----------------------------------------------------------------------------
# Hybrid: validation-calibrated cross-sectional budget x time-series monthly shape
# -----------------------------------------------------------------------------
def run_hybrid(data, cs_ctx, ts_ctx):
    fitted = cs_ctx["fitted"]["two_part"]
    fcols = cs_ctx["fcols"]
    feats, targets, splits, actuals = data["feats"], data["targets"], data["splits"], data["actuals"]
    merged = feats.merge(targets, on="person_id", how="inner")
    va = merged[merged["person_id"].isin(splits["val"]) & merged["person_id"].isin(set(actuals))]
    te = merged[merged["person_id"].isin(splits["test"]) & merged["person_id"].isin(set(actuals))]

    cs_pred_va = np.maximum(fitted.predict(va[fcols]), 0)
    cs_factor = float(np.mean(va["total_paid_sum"].values) / np.mean(cs_pred_va)) if np.mean(cs_pred_va) > 0 else 1.0
    log(f"  hybrid calibration factor (from validation members): {cs_factor:.4f}")
    cs_pred_te = np.maximum(fitted.predict(te[fcols]), 0)
    budget = {pid: float(v) * cs_factor for pid, v in zip(te["person_id"].values, cs_pred_te)}

    results, perpatient = {}, {}
    for base in ["chronos_zeroshot", "naive_mean3", "panelfm_adapter"]:
        if base not in ts_ctx["perpatient"]:
            continue
        base_pred3 = ts_ctx["perpatient"][base]["pred3"]
        base_act3 = ts_ctx["perpatient"][base]["act3"]
        common = sorted(set(base_pred3) & set(budget))
        if len(common) < 100:
            continue
        pred3, act3 = {}, {}
        for pid in common:
            ts_vec = np.asarray(base_pred3[pid], dtype=float)
            s = ts_vec.sum()
            b = max(budget[pid], 0.0)
            pred3[pid] = (ts_vec / s) * b if s > 0 else np.zeros(len(ts_vec))
            act3[pid] = base_act3[pid]
        results[f"hybrid_{base}"] = eval3(pred3, act3)
        perpatient[f"hybrid_{base}"] = {"pred3": pred3, "act3": act3}
        log(f"  hybrid_{base:14s} MAE={results['hybrid_'+base]['mae']:.1f} "
            f"R2cal={results['hybrid_'+base]['r_squared_calibrated']:.4f} PR={results['hybrid_'+base]['predictive_ratio']:.3f}")

    # Gated hybrid: the foundation-model 3-month total gates between the calibrated
    # cross-sectional budget (members predicted to incur cost) and the foundation-model
    # level (members predicted near-zero), allocated by the foundation-model monthly shape.
    # The sigmoid gate (tau, k) is tuned on validation members to minimize per-member-per-
    # month MAE subject to calibrated R-squared >= 0.15, then applied to held-out test members.
    chronos_fc = ts_ctx.get("chronos_fc", {})
    val_ids = [p for p in ts_ctx.get("val_ctx_ids", []) if p in chronos_fc and p in actuals]
    if val_ids and "chronos_zeroshot" in ts_ctx["perpatient"]:
        cs_pred_va_full = np.maximum(fitted.predict(va[fcols]), 0)
        bud_va = {pid: float(v) * cs_factor for pid, v in zip(va["person_id"].values, cs_pred_va_full)}

        def gated(pids, ch_src, bud_map, tau, k):
            out, act = {}, {}
            for p in pids:
                if p not in bud_map or p not in actuals:
                    continue
                ch = np.asarray(ch_src[p], dtype=float)[:HORIZON]
                s = ch.sum(); shape = (ch / s) if s > 0 else np.ones(len(ch)) / len(ch)
                ct = float(ch.sum())
                g = 1.0 / (1.0 + np.exp(-k * (ct - tau)))
                out[p] = shape * (g * bud_map[p] + (1 - g) * ct)
                act[p] = np.asarray(actuals[p], dtype=float)[:HORIZON]
            return out, act

        best = None
        for tau in [50, 100, 200, 300, 400, 600, 800, 1200, 1800]:
            for k in [0.0005, 0.001, 0.002, 0.004, 0.01]:
                vp, va_act = gated(val_ids, chronos_fc, bud_va, tau, k)
                mv = eval3(vp, va_act)
                if mv["r_squared_calibrated"] >= 0.15 and (best is None or mv["mae"] < best[2]):
                    best = (tau, k, mv["mae"])
        if best:
            tau, k, _ = best
            te_chronos = {p: chronos_fc[p] for p in te["person_id"].values if p in chronos_fc}
            tp3, ta3 = gated(list(te_chronos), chronos_fc, budget, tau, k)
            results["hybrid_gated"] = eval3(tp3, ta3)
            results["hybrid_gated"]["gate_tau"] = tau
            results["hybrid_gated"]["gate_k"] = k
            perpatient["hybrid_gated"] = {"pred3": tp3, "act3": ta3}
            log(f"  hybrid_gated      MAE={results['hybrid_gated']['mae']:.1f} "
                f"R2cal={results['hybrid_gated']['r_squared_calibrated']:.4f} "
                f"PR={results['hybrid_gated']['predictive_ratio']:.3f} (tau={tau}, k={k})")
    return results, perpatient, cs_factor


# -----------------------------------------------------------------------------
# Concurrent evaluation (explanatory ceiling)
# -----------------------------------------------------------------------------
def run_concurrent(data):
    outcomes, splits, targets, actuals = data["outcomes"], data["splits"], data["targets"], data["actuals"]
    conc_end = TARGET_START + pd.DateOffset(months=HORIZON)
    conc = build_cross_sectional_features(
        outcomes[(outcomes["month_year"] > TARGET_START) & (outcomes["month_year"] <= conc_end)],
        None, None, lookback_end=conc_end, lookback_months=HORIZON)
    conc["person_id"] = conc["person_id"].astype(str)
    conc = conc.fillna(0)
    merged = conc.merge(targets, on="person_id", how="inner")
    tr = merged[merged["person_id"].isin(splits["train"])]
    te = merged[merged["person_id"].isin(splits["test"]) & merged["person_id"].isin(set(actuals))]
    fcols = feature_columns(tr)
    if len(fcols) < 3 or len(tr) < 50 or len(te) < 50:
        log("  concurrent: insufficient features/rows, skipping")
        return {}
    Xtr, Xte = tr[fcols], te[fcols]
    ytr = tr["total_paid_sum"].values
    te_ids = te["person_id"].tolist()
    act3 = {p: actuals[p] for p in te_ids}
    out = {}
    specs = [("concurrent_xgboost", XGBoostBaseline, MODEL_CFG["xgboost"]),
             ("concurrent_random_forest", RandomForestBaseline, MODEL_CFG["random_forest"]),
             ("concurrent_two_part", TwoPartModel, MODEL_CFG["xgboost"])]
    if LightGBMBaseline is not None:
        specs.append(("concurrent_lightgbm", LightGBMBaseline, MODEL_CFG["lightgbm"]))
    for name, cls, cfg in specs:
        reg = cls(task="regression", config=cfg)
        reg.fit(Xtr, pd.Series(ytr))
        total_te = {pid: max(float(v), 0.0) for pid, v in zip(te_ids, reg.predict(Xte))}
        pred3 = flat3(total_te, act3)
        out[name] = eval3(pred3, act3)
        log(f"  {name:24s} R2cal={out[name]['r_squared_calibrated']:.4f} MAE={out[name]['mae']:.1f}")
    return out


# -----------------------------------------------------------------------------
# Downstream analyses
# -----------------------------------------------------------------------------
def bootstrap_ci(pred3, act3, n_boot=N_BOOT, seed=SPLIT_SEED):
    pids = sorted(set(pred3) & set(act3))
    pm = np.array([np.mean(np.abs(pred3[p] - act3[p])) for p in pids])
    yt = np.array([float(np.sum(act3[p])) for p in pids])
    yp = np.array([float(np.sum(pred3[p])) for p in pids])
    n = len(pids); rng = np.random.default_rng(seed)
    maes, r2s = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        maes.append(pm[idx].mean())
        r2s.append(r_squared_calibrated(yt[idx], yp[idx]))
    return {"mae": float(pm.mean()), "mae_ci": [float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))],
            "r2_cal": r_squared_calibrated(yt, yp),
            "r2_cal_ci": [float(np.percentile(r2s, 2.5)), float(np.percentile(r2s, 97.5))], "n": n}


def stratified_mae(perpatient_all):
    rows = {}
    for model, d in perpatient_all.items():
        pred3, act3 = d["pred3"], d["act3"]
        pids = sorted(set(pred3) & set(act3))
        tot = np.array([float(np.sum(act3[p])) for p in pids])
        err = np.array([np.mean(np.abs(pred3[p] - act3[p])) for p in pids])
        zero = tot == 0
        rows[model] = {"n_total": int(len(tot)), "n_zero": int(zero.sum()), "n_pos": int((~zero).sum()),
                       "mae_overall": float(err.mean()),
                       "mae_zero": float(err[zero].mean()) if zero.any() else None,
                       "mae_pos": float(err[~zero].mean()) if (~zero).any() else None}
    return rows


def decile_table(pred3, act3):
    pids = sorted(set(pred3) & set(act3))
    yt = np.array([float(np.sum(act3[p])) for p in pids])
    yp = np.array([float(np.sum(pred3[p])) for p in pids])
    return decile_analysis(yt, yp)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", type=int, default=None)
    ap.add_argument("--reuse-forecasts", action="store_true",
                    help="reuse existing forecasts.joblib instead of re-running the torch stage")
    ap.add_argument("--temporal", action="store_true",
                    help="use a non-random temporal entry-cohort split (Steyerberg 17.3) instead of random")
    args = ap.parse_args()

    outcomes, attributes, eligibility = load_cohort()
    data = prepare(outcomes, attributes, eligibility, quick=args.quick, temporal=args.temporal)
    n_members = outcomes["person_id"].nunique()
    n_months = outcomes.shape[0]
    zero_frac = float((outcomes["total_paid"] == 0).mean())

    log("\n=== Cross-sectional (train-fit, val-calibrate, test-eval) ===")
    cs_results, cs_ctx = run_cross_sectional(data)
    log("\n=== Time-series + foundation models ===")
    ts_results, ts_ctx = run_time_series_and_fm(data, cs_ctx, reuse_forecasts=args.reuse_forecasts)
    log("\n=== Hybrid (validation-calibrated budget x temporal allocation) ===")
    hyb_results, hyb_pp, cs_factor = run_hybrid(data, cs_ctx, ts_ctx)
    log("\n=== Concurrent (explanatory ceiling) ===")
    conc_results = run_concurrent(data)

    perpatient_all = dict(ts_ctx["perpatient"])
    perpatient_all.update(hyb_pp)
    perpatient_all.update(cs_ctx["perpatient"])

    all_metrics = {}
    all_metrics.update(cs_results); all_metrics.update(ts_results)
    all_metrics.update(hyb_results); all_metrics.update(conc_results)

    log("\n=== Bootstrap CIs ===")
    ci = {}
    for model in ["chronos_zeroshot", "naive_mean3", "panelfm_adapter", "two_part", "stacking",
                  "random_forest", "xgboost", "lightgbm", "demographic_glm", "naive_last",
                  "tweedie", "quantile_median_cs",
                  "hybrid_chronos_zeroshot", "hybrid_panelfm_adapter", "hybrid_naive_mean3", "hybrid_gated"]:
        if model in perpatient_all:
            ci[model] = bootstrap_ci(perpatient_all[model]["pred3"], perpatient_all[model]["act3"])
            log(f"  {model:24s} MAE {ci[model]['mae']:.1f} [{ci[model]['mae_ci'][0]:.1f},{ci[model]['mae_ci'][1]:.1f}] "
                f"R2cal {ci[model]['r2_cal']:.3f} [{ci[model]['r2_cal_ci'][0]:.3f},{ci[model]['r2_cal_ci'][1]:.3f}]")

    strat = stratified_mae(perpatient_all)
    deciles = {m: decile_table(perpatient_all[m]["pred3"], perpatient_all[m]["act3"])
               for m in ["two_part", "chronos_zeroshot", "panelfm_adapter", "hybrid_chronos_zeroshot",
                         "hybrid_gated", "naive_mean3", "stacking", "demographic_glm"] if m in perpatient_all}

    tw = outcomes[(outcomes["month_year"] > TARGET_START) & (outcomes["month_year"] <= pd.Timestamp("2025-10-01"))]
    month_cost = tw.groupby(tw["month_year"].dt.strftime("%Y-%m"))["total_paid"].mean().round(2).to_dict()

    split_info = {
        "design": "member-disjoint (no member in more than one split); prospective May-Jul 2025 3-month target",
        "metric_scale": "MAE/RMSE are per-member-per-month; R2/PR are on the patient 3-month total",
        "seed": SPLIT_SEED, "fractions": {"train": FRAC_TRAIN, "val": FRAC_VAL, "test": FRAC_TEST},
        "n_members_cohort": int(n_members), "n_patient_months_cohort": int(n_months),
        "zero_cost_month_frac": zero_frac,
        "n_eligible_members": len(data["splits"]["train"]) + len(data["splits"]["val"]) + len(data["splits"]["test"]),
        "n_train_members": len(data["splits"]["train"]), "n_val_members": len(data["splits"]["val"]),
        "n_test_members": len(data["splits"]["test"]),
        "n_test_scored": all_metrics.get("chronos_zeroshot", {}).get("n_patients"),
        "hybrid_calibration_factor_from_validation": cs_factor,
        "train_cutoff": str(TRAIN_CUTOFF.date()), "context_cutoff": str(CONTEXT_CUTOFF.date()),
        "target_window": "2025-05 to 2025-07",
    }

    def jsonable(o):
        if isinstance(o, dict):
            return {k: jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [jsonable(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if (isinstance(o, float) and np.isnan(o)) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o

    # Per-patient test predictions (3-month totals) + actuals + high-cost label, for the
    # downstream decision-curve / net-benefit / recalibration analyses across all models.
    pp_save = {"actual_total": {}, "models": {}}
    for mname, d in perpatient_all.items():
        pp_save["models"][mname] = {pid: float(np.sum(d["pred3"][pid])) for pid in d["pred3"]}
        for pid in d["act3"]:
            pp_save["actual_total"][pid] = float(np.sum(d["act3"][pid]))
    hc_thr = float(np.quantile(list(pp_save["actual_total"].values()), 0.90)) if pp_save["actual_total"] else 0.0
    pp_save["high_cost_label"] = {pid: int(v >= hc_thr) for pid, v in pp_save["actual_total"].items()}
    pp_save["high_cost_threshold"] = hc_thr

    tag = "quick" if args.quick else ("temporal" if args.temporal else "disjoint")
    (OUT_DIR / f"per_patient_test_{tag}.json").write_text(json.dumps(jsonable(pp_save)))
    (OUT_DIR / f"all_metrics_{tag}.json").write_text(json.dumps(jsonable(all_metrics), indent=2))
    (OUT_DIR / f"split_info_{tag}.json").write_text(json.dumps(jsonable(split_info), indent=2))
    (OUT_DIR / f"bootstrap_ci_{tag}.json").write_text(json.dumps(jsonable(ci), indent=2))
    (OUT_DIR / f"stratified_mae_{tag}.json").write_text(json.dumps(jsonable(strat), indent=2))
    (OUT_DIR / f"decile_{tag}.json").write_text(json.dumps(jsonable(deciles), indent=2))
    (OUT_DIR / f"month_cost_{tag}.json").write_text(json.dumps(jsonable(month_cost), indent=2))

    log("\n=== SUMMARY (test members; MAE per-member-per-month, R2/PR on 3-month total) ===")
    log(f"{'model':26s} {'MAE':>9} {'R2cal':>8} {'PR':>7} {'AUROC':>7} {'n':>8}")
    for k, m in sorted(all_metrics.items(), key=lambda x: x[1].get("mae", 9e9)):
        log(f"{k:26s} {m.get('mae', float('nan')):9.1f} {m.get('r_squared_calibrated', float('nan')):8.4f} "
            f"{m.get('predictive_ratio', float('nan')):7.3f} {(m.get('auroc') or float('nan')):7.3f} "
            f"{m.get('n_patients', 0):8d}")
    log(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
