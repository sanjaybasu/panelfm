#!/usr/bin/env python3
"""
TabPFN-v2 (tabular foundation model) on the cross-sectional task, for the second revision.

TabPFN-v2 (Hollmann et al., Nature 2025) is a state-of-the-art foundation model for tabular
data. This evaluates whether a tabular foundation model improves between-member cost
discrimination over gradient-boosted trees, on the same member-disjoint design.

Runs torch-isolated and does NOT import xgboost/lightgbm (OpenMP conflict on macOS); it
re-derives the identical member-disjoint split from the data (same seed and eligibility as
run_member_disjoint.py), then fits TabPFN on a stratified training subsample (TabPFN-v2's
practical sample ceiling) and scores the held-out test members.

Output: results/all_metrics_tabpfn.json, results/per_patient_tabpfn.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(4)

PKG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG_ROOT))
from src.data.load_claims import build_cross_sectional_features, build_patient_time_series
from src.evaluation.metrics import r_squared, r_squared_calibrated, predictive_ratio, high_cost_identification

# Constants must match run_member_disjoint.py exactly to reproduce the split.
SPLIT_SEED = 20260617
FRAC_TRAIN, FRAC_VAL, FRAC_TEST = 0.70, 0.15, 0.15
HORIZON = 3
TRAIN_CUTOFF = pd.Timestamp("2025-01-01"); CONTEXT_CUTOFF = pd.Timestamp("2025-04-01")
TARGET_START = pd.Timestamp("2025-04-01")
LOOKBACK_MONTHS, MIN_HISTORY_MONTHS, MIN_ENROLLED_DAYS, MIN_TOTAL_MONTHS = 12, 6, 15, 9
TABPFN_TRAIN_CAP = 10000
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(PKG_ROOT / "data" / "real_inputs")))
OUT = PKG_ROOT / "revision2" / "results"


def log(m): print(m, flush=True)


def build_targets_inline(outcomes, start, horizon):
    end = start + pd.DateOffset(months=horizon)
    w = outcomes[(outcomes["month_year"] > start) & (outcomes["month_year"] <= end)].copy()
    w["total_paid"] = w["total_paid"].fillna(0)
    t = w.groupby("person_id").agg(total_paid_sum=("total_paid", "sum"),
                                   n_months=("month_year", "count")).reset_index()
    return t


def member_disjoint_split(mwc):
    df = mwc.copy(); df["stratum"] = 0
    pos = df["lookback_cost"] > 0
    if pos.sum() > 5:
        df.loc[pos, "stratum"] = pd.qcut(df.loc[pos, "lookback_cost"], q=5, labels=False, duplicates="drop") + 1
    rng = np.random.default_rng(SPLIT_SEED); tr, va, te = [], [], []
    for _, g in df.groupby("stratum"):
        ids = g["person_id"].values.copy(); rng.shuffle(ids)
        n = len(ids); ntr = int(round(n * FRAC_TRAIN)); nva = int(round(n * FRAC_VAL))
        tr.extend(ids[:ntr]); va.extend(ids[ntr:ntr + nva]); te.extend(ids[ntr + nva:])
    return set(tr), set(va), set(te)


def prepare():
    op = DATA_ROOT / "outcomes_monthly.parquet"
    outcomes = pd.read_parquet(op); outcomes.columns = outcomes.columns.str.lower().str.strip()
    outcomes["month_year"] = pd.to_datetime(outcomes["month_year"])
    ap = DATA_ROOT / "member_attributes.parquet"; attributes = pd.read_parquet(ap); attributes.columns = attributes.columns.str.lower().str.strip()
    ep = DATA_ROOT / "eligibility.parquet"; eligibility = pd.read_parquet(ep); eligibility.columns = eligibility.columns.str.lower().str.strip()
    outcomes = outcomes[outcomes["enrolled_days"] >= MIN_ENROLLED_DAYS]
    pm = outcomes.groupby("person_id").size(); keep = pm[pm >= MIN_TOTAL_MONTHS].index
    outcomes = outcomes[outcomes["person_id"].isin(keep)].copy(); outcomes["person_id"] = outcomes["person_id"].astype(str)
    for c in ["total_paid", "emergency_department_ct", "acute_inpatient_ct"]:
        outcomes[c] = outcomes[c].fillna(0)

    feats = build_cross_sectional_features(outcomes, attributes, eligibility, lookback_end=TRAIN_CUTOFF, lookback_months=LOOKBACK_MONTHS)
    feats["person_id"] = feats["person_id"].astype(str)
    for col in list(feats.columns):
        if feats[col].dtype == object and col != "person_id":
            d = pd.get_dummies(feats[col], prefix=col, drop_first=True); feats = pd.concat([feats.drop(columns=[col]), d], axis=1)
    feats = feats.fillna(0)
    targets = build_targets_inline(outcomes, TARGET_START, HORIZON); targets["person_id"] = targets["person_id"].astype(str)
    pre = outcomes[outcomes["month_year"] <= CONTEXT_CUTOFF]
    context, _ = build_patient_time_series(pre, min_months=MIN_HISTORY_MONTHS); context = {str(k): v for k, v in context.items()}
    tw = outcomes[(outcomes["month_year"] > TARGET_START) & (outcomes["month_year"] <= TARGET_START + pd.DateOffset(months=HORIZON))]
    actuals = {str(p): g.sort_values("month_year")["total_paid"].fillna(0).values for p, g in tw.groupby("person_id")}
    actuals = {p: v for p, v in actuals.items() if len(v) == HORIZON}
    eligible = sorted(set(feats["person_id"]) & set(targets["person_id"]) & set(context) & set(actuals))
    lb = feats.set_index("person_id")["mean_total_paid"].reindex(eligible).fillna(0)
    mwc = pd.DataFrame({"person_id": eligible, "lookback_cost": lb.values})
    tr, va, te = member_disjoint_split(mwc)
    return feats, targets, actuals, tr, te


def main():
    feats, targets, actuals, train_ids, test_ids = prepare()
    merged = feats.merge(targets, on="person_id", how="inner")
    drop = {"person_id", "total_paid_sum", "n_months"}
    fcols = [c for c in merged.columns if c not in drop and merged[c].dtype.kind in "fiub"]
    tr = merged[merged["person_id"].isin(train_ids)]
    te = merged[merged["person_id"].isin(test_ids) & merged["person_id"].isin(set(actuals))]
    log(f"train {len(tr)}  test {len(te)}  features {len(fcols)}")

    thr = tr["total_paid_sum"].quantile(0.90)
    ytr_cls = (tr["total_paid_sum"] >= thr).astype(int).values
    ytr_reg = tr["total_paid_sum"].values
    Xtr = tr[fcols].values.astype(np.float32); Xte = te[fcols].values.astype(np.float32)
    te_ids = te["person_id"].tolist()
    act3 = {p: np.asarray(actuals[p], dtype=float) for p in te_ids}

    # Stratified subsample of training rows to TabPFN's practical ceiling.
    rng = np.random.default_rng(SPLIT_SEED)
    if len(Xtr) > TABPFN_TRAIN_CAP:
        pos = np.where(ytr_cls == 1)[0]; neg = np.where(ytr_cls == 0)[0]
        n_pos = min(len(pos), TABPFN_TRAIN_CAP // 2); n_neg = TABPFN_TRAIN_CAP - n_pos
        idx = np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, min(n_neg, len(neg)), replace=False)])
        rng.shuffle(idx)
    else:
        idx = np.arange(len(Xtr))
    log(f"TabPFN training subsample: {len(idx)}")

    results = {}; pp = {}
    t0 = time.time()
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    clf = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
    clf.fit(Xtr[idx], ytr_cls[idx])
    score = clf.predict_proba(Xte)[:, 1]
    y_te_total = np.array([float(np.sum(act3[p])) for p in te_ids])
    hc = high_cost_identification(y_te_total, score).get("top_10pct", {})
    results["tabpfn_classifier"] = {"auroc": hc.get("c_statistic", np.nan), "ppv_at_top10pct": hc.get("ppv", np.nan),
                                    "lift": hc.get("lift", np.nan), "n_patients": len(te_ids)}
    log(f"  tabpfn_classifier AUROC={results['tabpfn_classifier']['auroc']:.3f} ({time.time()-t0:.0f}s)")

    t0 = time.time()
    reg = TabPFNRegressor(device="cpu", ignore_pretraining_limits=True)
    reg.fit(Xtr[idx], ytr_reg[idx])
    pred_total = np.maximum(reg.predict(Xte), 0)
    pred3 = {p: np.full(len(act3[p]), pred_total[i] / len(act3[p])) for i, p in enumerate(te_ids)}
    pm = np.array([np.mean(np.abs(pred3[p] - act3[p])) for p in te_ids])
    yt = y_te_total; yp = np.array([float(np.sum(pred3[p])) for p in te_ids])
    hcr = high_cost_identification(yt, yp).get("top_10pct", {})
    results["tabpfn_regressor"] = {"mae": float(pm.mean()), "r_squared_calibrated": r_squared_calibrated(yt, yp),
                                   "predictive_ratio": predictive_ratio(yt, yp)["overall"],
                                   "auroc": hcr.get("c_statistic", np.nan), "n_patients": len(te_ids)}
    pp["tabpfn_regressor"] = {p: float(yp[i]) for i, p in enumerate(te_ids)}
    log(f"  tabpfn_regressor MAE={results['tabpfn_regressor']['mae']:.1f} R2cal={results['tabpfn_regressor']['r_squared_calibrated']:.3f} "
        f"PR={results['tabpfn_regressor']['predictive_ratio']:.3f} ({time.time()-t0:.0f}s)")

    def js(o):
        if isinstance(o, dict): return {k: js(v) for k, v in o.items()}
        if isinstance(o, (np.floating, float)): return None if (isinstance(o, float) and np.isnan(o)) else float(o)
        if isinstance(o, (np.integer,)): return int(o)
        return o
    (OUT / "all_metrics_tabpfn.json").write_text(json.dumps(js(results), indent=2))
    (OUT / "per_patient_tabpfn.json").write_text(json.dumps(js(pp)))
    log("Wrote all_metrics_tabpfn.json")


if __name__ == "__main__":
    main()
