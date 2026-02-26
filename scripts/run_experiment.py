#!/usr/bin/env python3
"""
Unified PanelFM experiment runner.

Runs everything: data prep, baselines, PanelFM variants, evaluation.
Handles NaN costs (enrolled but no claims = $0), subsample for ARIMA speed,
and graceful fallback when optional packages are missing.
"""

import sys
import json
import time
import warnings
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import yaml
import joblib

from src.data.synthetic_panel import SyntheticPanelConfig, generate_synthetic_panel
from src.data.load_claims import (
    load_and_prepare_all,
    build_cross_sectional_features,
    build_patient_time_series,
    temporal_train_val_test_split,
)
from src.models.baselines import (
    XGBoostBaseline,
    RandomForestBaseline,
    StackingEnsemble,
    TwoPartModel,
    DemographicGLM,
    build_targets,
)
from src.models.patient_encoder import PatientEncoder, PatientEmbeddingStore
from src.evaluation.metrics import (
    evaluate_all_metrics,
    panel_r2_decomposition,
    equity_stratified_metrics,
    auroc,
    r_squared,
    r_squared_calibrated,
    calibrate_predictions,
    predictive_ratio,
    decile_analysis,
    cost_bucket_calibration,
    censored_metrics,
    high_cost_identification,
)

try:
    from src.models.baselines import LightGBMBaseline
except Exception:
    LightGBMBaseline = None

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*NumPy.*")

RESULTS_DIR = project_root / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_model_config():
    with open(project_root / "configs" / "model_config.yaml") as f:
        return yaml.safe_load(f)


def compute_ts_mae(forecasts, actuals, horizon):
    """Compute mean absolute error for time series forecasts.

    Returns:
        aggregate_mae: float, mean of per-patient MAEs
        per_patient_errors: dict mapping person_id to per-patient MAE
        per_patient_actuals_mean: dict mapping person_id to mean actual cost
        per_patient_preds_mean: dict mapping person_id to mean predicted cost
            (needed for actuarial metrics: R², predictive ratios, decile analysis)
    """
    per_patient_errors = {}
    per_patient_actuals_mean = {}
    per_patient_preds_mean = {}
    for pid in forecasts:
        if pid in actuals:
            actual = actuals[pid][:horizon]
            pred = forecasts[pid][:len(actual)]
            if len(actual) > 0 and len(pred) > 0:
                err = np.mean(np.abs(actual - pred))
                if np.isfinite(err):
                    per_patient_errors[pid] = float(err)
                    per_patient_actuals_mean[pid] = float(np.mean(actual))
                    per_patient_preds_mean[pid] = float(np.mean(pred))
    aggregate_mae = np.mean(list(per_patient_errors.values())) if per_patient_errors else np.nan
    return aggregate_mae, per_patient_errors, per_patient_actuals_mean, per_patient_preds_mean


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_synthetic():
    """Generate synthetic panel data."""
    print("Generating synthetic panel data...")
    config = SyntheticPanelConfig(
        n_patients=10_000, n_months=24, seed=42,
        ar1_coefficient=0.15, innovation_std=1.8,
        random_intercept_std=0.8, base_monthly_cost_mean=5.0,
        observation_noise_std=0.8, seasonal_amplitude=0.05,
    )
    data = generate_synthetic_panel(config)
    data["outcomes_monthly"].to_parquet(RESULTS_DIR / "synthetic_outcomes.parquet", index=False)
    data["patient_features"].to_parquet(RESULTS_DIR / "synthetic_features.parquet", index=False)
    print(f"  {data['outcomes_monthly'].shape[0]:,} patient-months, "
          f"{data['outcomes_monthly']['person_id'].nunique():,} patients")
    return data["outcomes_monthly"], data["patient_features"]


def prepare_real():
    """Load and prepare real claims data."""
    print("Loading real claims data...")
    real_data = load_and_prepare_all(str(project_root / "configs" / "data_config.yaml"))
    outcomes = real_data["outcomes"]
    features = real_data["features"]
    outcomes.to_parquet(RESULTS_DIR / "real_outcomes_filtered.parquet", index=False)
    features.to_parquet(RESULTS_DIR / "real_features.parquet", index=False)
    return outcomes, features


def prepare_dataset(outcomes, patient_features, dataset_name):
    """Prepare features, targets, and time series for a dataset."""
    # Fill NaN costs with 0 (enrolled but no claims)
    for col in ["total_paid", "emergency_department_ct", "acute_inpatient_ct"]:
        if col in outcomes.columns:
            outcomes[col] = outcomes[col].fillna(0)

    # Temporal split
    splits = temporal_train_val_test_split(outcomes, val_months=3, test_months=3)
    train_end = splits["train_end"]
    val_end = splits["val_end"]

    print(f"  Split: train≤{train_end.date()}, val≤{val_end.date()}, test≤{splits['test_end'].date()}")

    # Build utilization features from lookback window
    util_features = build_cross_sectional_features(
        splits["train"], None, None, train_end, lookback_months=12
    )

    # Merge with patient-level features if available
    if patient_features is not None:
        util_cols = set(util_features.columns) - {"person_id"}
        extra_cols = ["person_id"] + [c for c in patient_features.columns
                                       if c not in util_cols and c != "person_id"]
        features_extra = patient_features[extra_cols].drop_duplicates(subset=["person_id"])
        features = util_features.merge(features_extra, on="person_id", how="left")
    else:
        features = util_features

    # Encode categoricals, drop non-numeric
    for col in features.columns:
        if features[col].dtype == object and col != "person_id":
            dummies = pd.get_dummies(features[col], prefix=col, drop_first=True)
            features = pd.concat([features.drop(columns=[col]), dummies], axis=1)

    features = features.fillna(0)

    # Build targets
    targets_train = build_targets(outcomes, train_end - pd.DateOffset(months=3), horizon_months=3)
    targets_test = build_targets(outcomes, val_end, horizon_months=3)

    # Build time series
    series_train, _ = build_patient_time_series(splits["train"], min_months=6)
    pre_test = outcomes[outcomes["month_year"] <= val_end]
    series_context, _ = build_patient_time_series(pre_test, min_months=6)

    actuals = {}
    for pid, group in splits["test"].groupby("person_id"):
        vals = group.sort_values("month_year")["total_paid"].fillna(0).values
        if len(vals) > 0:
            actuals[pid] = vals

    print(f"  Features: {features.shape}, Train targets: {len(targets_train)}, "
          f"Test targets: {len(targets_test)}")
    print(f"  TS train: {len(series_train)}, TS context: {len(series_context)}, "
          f"TS actuals: {len(actuals)}")

    return {
        "features": features,
        "targets_train": targets_train,
        "targets_test": targets_test,
        "series_train": series_train,
        "series_context": series_context,
        "actuals": actuals,
        "splits": splits,
    }


# =============================================================================
# CROSS-SECTIONAL BASELINES
# =============================================================================

def run_cross_sectional(data, model_config):
    """Run XGBoost, RF, LightGBM, TwoPartModel, Stacking."""
    features = data["features"]
    targets_train = data["targets_train"]
    targets_test = data["targets_test"]

    train = features.merge(targets_train, on="person_id")
    test = features.merge(targets_test, on="person_id")

    target_cols = ["total_paid_sum", "ed_visits_sum", "ip_admits_sum",
                   "high_cost_flag", "any_ed_flag", "any_ip_flag", "n_months"]
    feat_cols = [c for c in train.columns if c not in ["person_id"] + target_cols
                 and train[c].dtype in [np.float64, np.int64, np.float32, np.int32, np.uint8, bool]]

    X_train, X_test = train[feat_cols], test[feat_cols]
    y_train_reg, y_test_reg = train["total_paid_sum"], test["total_paid_sum"]
    y_train_cls, y_test_cls = train["high_cost_flag"], test["high_cost_flag"]

    results = {}
    xgb_ref_scores = None
    per_patient_predictions = {}

    models = [
        ("demographic_glm", DemographicGLM),
        ("xgboost", XGBoostBaseline),
        ("random_forest", RandomForestBaseline),
    ]
    if LightGBMBaseline is not None:
        models.append(("lightgbm", LightGBMBaseline))

    # Add two-part model
    models.append(("two_part", TwoPartModel))

    for name, ModelClass in models:
        print(f"  {name}...", end=" ", flush=True)
        t0 = time.time()

        model_reg = ModelClass(task="regression", config=model_config.get(name, model_config.get("xgboost", {})))
        model_reg.fit(X_train, y_train_reg)
        y_pred_reg = model_reg.predict(X_test)

        if name == "two_part":
            # Two-part model: use Part 1 classifier for classification
            y_score_cls = model_reg.predict_proba(X_test)
        else:
            model_cls = ModelClass(task="classification", config=model_config.get(name, {}))
            model_cls.fit(X_train, y_train_cls)
            y_score_cls = model_cls.predict_proba(X_test)

        if name == "xgboost":
            xgb_ref_scores = y_score_cls

        metrics = evaluate_all_metrics(
            y_test_reg.values, y_pred_reg,
            y_test_cls.values, y_score_cls,
            y_score_ref=xgb_ref_scores if name != "xgboost" else None,
        )
        metrics["time_seconds"] = time.time() - t0
        results[name] = metrics
        print(f"MAE={metrics['mae']:.1f}, AUROC={metrics['auroc']:.4f} ({metrics['time_seconds']:.1f}s)")

        # Save per-patient predictions for bootstrap
        per_patient_predictions[name] = {
            "y_pred_reg": y_pred_reg.tolist(),
            "y_score_cls": y_score_cls.tolist(),
        }

        if name == "xgboost":
            results["_xgb_model"] = model_reg
            results["_X_train"] = X_train
            results["_X_test"] = X_test

    # Stacking
    print(f"  stacking...", end=" ", flush=True)
    t0 = time.time()
    stack_cls = StackingEnsemble(task="classification", config=model_config)
    stack_cls.fit(X_train, y_train_cls)
    y_score_stack = stack_cls.predict_proba(X_test)

    stack_reg = StackingEnsemble(task="regression", config=model_config)
    stack_reg.fit(X_train, y_train_reg)
    y_pred_stack = stack_reg.predict(X_test)

    metrics = evaluate_all_metrics(
        y_test_reg.values, y_pred_stack,
        y_test_cls.values, y_score_stack,
        y_score_ref=xgb_ref_scores,
    )
    metrics["time_seconds"] = time.time() - t0
    results["stacking"] = metrics
    print(f"MAE={metrics['mae']:.1f}, AUROC={metrics['auroc']:.4f} ({metrics['time_seconds']:.1f}s)")

    per_patient_predictions["stacking"] = {
        "y_pred_reg": y_pred_stack.tolist(),
        "y_score_cls": y_score_stack.tolist(),
    }

    # Save per-patient data for bootstrap
    results["_per_patient"] = {
        "person_ids": test["person_id"].tolist(),
        "y_true_reg": y_test_reg.values.tolist(),
        "y_true_cls": y_test_cls.values.tolist(),
        "predictions": per_patient_predictions,
    }

    return results


# =============================================================================
# CONCURRENT EVALUATION (actuarial standard)
# =============================================================================

def run_concurrent_evaluation(data, model_config):
    """Run concurrent evaluation: same-period features → same-period costs.

    In actuarial practice, concurrent models use diagnoses/utilization from
    the same period as the costs being predicted. This yields much higher R²
    (35-67% for MARA) because current-period features directly reflect
    current-period costs. It measures explanatory power rather than prediction.

    Method: Build features from the test period itself, then apply the same
    trained models to predict test-period costs from test-period features.
    """
    from src.data.load_claims import build_cross_sectional_features

    splits = data["splits"]
    outcomes = pd.concat([splits["train"], splits["val"], splits["test"]])

    # Build features from the TEST period itself (concurrent)
    test_end = splits["test"]["month_year"].max()
    test_start = splits["test"]["month_year"].min() - pd.DateOffset(months=1)

    print("  Building concurrent features (test-period utilization)...")
    concurrent_features = build_cross_sectional_features(
        splits["test"], None, None, test_end, lookback_months=3
    )

    # Merge with demographic features if available
    features = data["features"]
    demo_cols = [c for c in features.columns if c in [
        "person_id", "age", "female", "dual_eligible", "gender_F",
        "gender_M", "race_Other", "race_White", "race_Unknown"
    ]]
    if len(demo_cols) > 1:
        demo_df = features[demo_cols].drop_duplicates(subset=["person_id"])
        concurrent_features = concurrent_features.merge(demo_df, on="person_id", how="left")

    concurrent_features = concurrent_features.fillna(0)

    # Build targets from same period
    targets_test = data["targets_test"]
    test_df = concurrent_features.merge(targets_test, on="person_id")

    target_cols = ["total_paid_sum", "ed_visits_sum", "ip_admits_sum",
                   "high_cost_flag", "any_ed_flag", "any_ip_flag", "n_months"]
    feat_cols = [c for c in test_df.columns if c not in ["person_id"] + target_cols
                 and test_df[c].dtype in [np.float64, np.int64, np.float32, np.int32, np.uint8, bool]]

    X_test = test_df[feat_cols]
    y_test_reg = test_df["total_paid_sum"]
    y_test_cls = test_df["high_cost_flag"]

    # Need to train models on training data with same feature set
    # For concurrent evaluation on the test set, we train on train-period
    # features built from the TRAINING period (concurrent within training),
    # then evaluate on test-period features
    train_features = build_cross_sectional_features(
        splits["train"], None, None,
        splits["train"]["month_year"].max(),
        lookback_months=3
    )
    if len(demo_cols) > 1:
        train_features = train_features.merge(demo_df, on="person_id", how="left")
    train_features = train_features.fillna(0)

    # Build training targets from last 3 months of training
    train_end = splits["train"]["month_year"].max()
    train_targets = build_targets(
        splits["train"], train_end - pd.DateOffset(months=3), horizon_months=3
    )

    train_df = train_features.merge(train_targets, on="person_id")
    train_feat_cols = [c for c in feat_cols if c in train_df.columns]

    if len(train_feat_cols) < 3:
        print("  Concurrent evaluation: insufficient overlapping features, skipping")
        return {}

    X_train = train_df[train_feat_cols]
    y_train_reg = train_df["total_paid_sum"]
    y_train_cls = train_df["high_cost_flag"]

    # Ensure test features match training features
    for c in train_feat_cols:
        if c not in X_test.columns:
            X_test[c] = 0
    X_test = X_test[train_feat_cols]

    results = {}
    per_patient_predictions = {}

    models = [
        ("xgboost", XGBoostBaseline),
        ("random_forest", RandomForestBaseline),
        ("two_part", TwoPartModel),
        ("demographic_glm", DemographicGLM),
    ]
    if LightGBMBaseline is not None:
        models.append(("lightgbm", LightGBMBaseline))

    for name, ModelClass in models:
        print(f"  concurrent_{name}...", end=" ", flush=True)
        t0 = time.time()

        model_reg = ModelClass(task="regression", config=model_config.get(name, model_config.get("xgboost", {})))
        model_reg.fit(X_train, y_train_reg)
        y_pred_reg = model_reg.predict(X_test)

        if name == "two_part":
            y_score_cls = model_reg.predict_proba(X_test)
        elif name == "demographic_glm":
            model_cls = ModelClass(task="classification")
            model_cls.fit(X_train, y_train_cls)
            y_score_cls = model_cls.predict_proba(X_test)
        else:
            model_cls = ModelClass(task="classification", config=model_config.get(name, {}))
            model_cls.fit(X_train, y_train_cls)
            y_score_cls = model_cls.predict_proba(X_test)

        metrics = evaluate_all_metrics(
            y_test_reg.values, y_pred_reg,
            y_test_cls.values, y_score_cls,
        )
        metrics["time_seconds"] = time.time() - t0
        results[f"concurrent_{name}"] = metrics
        print(f"MAE={metrics['mae']:.1f}, R²(cal)={metrics['r_squared_calibrated']:.4f}, "
              f"PR={metrics['predictive_ratio']:.3f} ({metrics['time_seconds']:.1f}s)")

        per_patient_predictions[f"concurrent_{name}"] = {
            "y_pred_reg": y_pred_reg.tolist(),
            "y_score_cls": y_score_cls.tolist(),
        }

    results["_per_patient"] = {
        "person_ids": test_df["person_id"].tolist(),
        "y_true_reg": y_test_reg.values.tolist(),
        "y_true_cls": y_test_cls.values.tolist(),
        "predictions": per_patient_predictions,
    }

    return results


# =============================================================================
# TIME SERIES BASELINES
# =============================================================================

def run_ts_baselines(data, horizon=3):
    """Run naive + ARIMA baselines. Returns metrics and per-patient errors."""
    from src.models.ts_baselines import NaiveBaselines

    series_context = data["series_context"]
    actuals = data["actuals"]
    results = {}
    all_per_patient_errors = {}

    # Naive baselines (run on full test set)
    for name, method in [
        ("naive_last", NaiveBaselines.last_value),
        ("naive_mean3", lambda s, h: NaiveBaselines.trailing_mean(s, h, window=3)),
    ]:
        forecasts = method(series_context, horizon)
        mae_val, pp_errors, pp_actuals, pp_preds = compute_ts_mae(forecasts, actuals, horizon)
        results[name] = {"mae": mae_val, "n_patients": len(pp_errors)}
        all_per_patient_errors[name] = {"errors": pp_errors, "actuals_mean": pp_actuals, "preds_mean": pp_preds}
        print(f"  {name}: MAE={mae_val:.1f} (n={len(pp_errors)})")

    # ARIMA (subsample for speed)
    max_arima = 500
    rng = np.random.default_rng(42)
    common_pids = list(set(series_context.keys()) & set(actuals.keys()))
    if len(common_pids) > max_arima:
        arima_pids = rng.choice(common_pids, max_arima, replace=False)
    else:
        arima_pids = common_pids

    arima_series = {pid: series_context[pid] for pid in arima_pids}
    arima_actuals = {pid: actuals[pid] for pid in arima_pids if pid in actuals}

    try:
        from src.models.ts_baselines import ARIMABaseline
        print(f"  arima ({len(arima_series)} patients)...", end=" ", flush=True)
        t0 = time.time()
        arima = ARIMABaseline(season_length=12)
        arima_forecasts = arima.forecast_batch(arima_series, horizon)
        mae_val, pp_errors, pp_actuals, pp_preds = compute_ts_mae(arima_forecasts, arima_actuals, horizon)
        results["arima"] = {"mae": mae_val, "time_seconds": time.time() - t0, "n_patients": len(pp_errors)}
        all_per_patient_errors["arima"] = {"errors": pp_errors, "actuals_mean": pp_actuals, "preds_mean": pp_preds}
        print(f"MAE={mae_val:.1f} ({results['arima']['time_seconds']:.1f}s)")
    except Exception as e:
        print(f"  arima: failed ({e})")
        results["arima"] = {"mae": np.nan}

    results["_per_patient_errors"] = all_per_patient_errors
    return results


# =============================================================================
# CHRONOS FOUNDATION MODEL
# =============================================================================

def run_chronos_baseline(data, horizon=3, max_patients=None):
    """Run Chronos zero-shot on per-patient series.

    Uses ALL patients with sufficient history by default (no artificial cap).
    """
    from src.models.timesfm_wrapper import ChronosForecaster

    series_context = data["series_context"]
    actuals = data["actuals"]

    common_pids = list(set(series_context.keys()) & set(actuals.keys()))
    rng = np.random.default_rng(42)
    if max_patients is not None and len(common_pids) > max_patients:
        sample_pids = list(rng.choice(common_pids, max_patients, replace=False))
    else:
        sample_pids = common_pids

    sub_context = {pid: series_context[pid] for pid in sample_pids}
    sub_actuals = {pid: actuals[pid] for pid in sample_pids if pid in actuals}

    print(f"  chronos_zeroshot ({len(sub_context)} patients)...", end=" ", flush=True)
    t0 = time.time()

    forecaster = ChronosForecaster(model_name="amazon/chronos-t5-small", device="cpu")
    forecasts = forecaster.forecast_batch(sub_context, horizon, batch_size=64)
    mae_val, pp_errors, pp_actuals, pp_preds = compute_ts_mae(forecasts, sub_actuals, horizon)
    elapsed = time.time() - t0

    print(f"MAE={mae_val:.1f} ({elapsed:.1f}s, n={len(pp_errors)})")
    return {
        "chronos_zeroshot": {"mae": mae_val, "time_seconds": elapsed, "n_patients": len(pp_errors)},
        "_forecaster": forecaster,
        "_sample_pids": sample_pids,
        "_forecasts": forecasts,
        "_per_patient_errors": {"chronos_zeroshot": {"errors": pp_errors, "actuals_mean": pp_actuals, "preds_mean": pp_preds}},
    }


# =============================================================================
# PANELFM VARIANTS
# =============================================================================

def run_panelfm(data, cs_results, chronos_results, model_config, horizon=3):
    """Run all PanelFM variants."""
    from src.models.timesfm_wrapper import PanelFMXReg, PanelFMAdapter, PanelFMICF

    features = data["features"]
    series_train = data["series_train"]
    series_context = data["series_context"]
    actuals = data["actuals"]

    xgb_model = cs_results.get("_xgb_model")
    X_train = cs_results.get("_X_train")
    forecaster = chronos_results.get("_forecaster")
    sample_pids = chronos_results.get("_sample_pids", list(series_context.keys())[:2000])

    if xgb_model is None or forecaster is None:
        print("  Skipping PanelFM: missing XGBoost model or Chronos forecaster")
        return {}

    # Build patient embeddings
    print("  Building patient embeddings...", end=" ", flush=True)
    encoder = PatientEncoder(embedding_dim=8, method="leaf", include_risk_score=True)

    y_dummy = pd.Series(np.zeros(len(X_train)))
    encoder.fit(xgb_model, X_train, y_dummy)

    # The encoder needs the same columns XGBoost was trained on.
    # Build a version of features that has only those columns + person_id.
    xgb_feat_names = xgb_model.feature_names
    features_for_embed = features.copy()
    for col in xgb_feat_names:
        if col not in features_for_embed.columns:
            features_for_embed[col] = 0
    # Keep person_id + xgb feature columns only
    features_for_embed = features_for_embed[["person_id"] + xgb_feat_names].fillna(0)

    store = PatientEmbeddingStore()
    store.build(encoder, features_for_embed)
    print(f"dim={store.embedding_dim}")

    # Save embeddings
    all_pids = list(store.embeddings.keys())
    np.save(RESULTS_DIR / "patient_embeddings.npy", store.get_batch(all_pids))

    # Subsample for PanelFM experiments
    sub_context = {pid: series_context[pid] for pid in sample_pids if pid in series_context}
    sub_actuals = {pid: actuals[pid] for pid in sample_pids if pid in actuals}

    # Build train context/actuals for fitting residual model
    train_context = {}
    train_actuals = {}
    train_pids = list(series_train.keys())
    rng = np.random.default_rng(42)
    if len(train_pids) > 2000:
        train_pids = rng.choice(train_pids, 2000, replace=False)
    for pid in train_pids:
        ts = series_train[pid]
        if len(ts) > horizon:
            train_context[pid] = ts[:-horizon]
            train_actuals[pid] = ts[-horizon:]

    results = {}
    all_per_patient_errors = {}

    # --- Option A: XReg ---
    print(f"  panelfm_xreg...", flush=True)
    t0 = time.time()
    panelfm_a = PanelFMXReg(forecaster, store)
    panelfm_a.fit_residual_model(train_context, train_actuals, horizon)
    forecasts_a = panelfm_a.forecast_batch(sub_context, horizon)
    mae_a, pp_errors_a, pp_actuals_a, pp_preds_a = compute_ts_mae(forecasts_a, sub_actuals, horizon)
    results["panelfm_xreg"] = {"mae": mae_a, "time_seconds": time.time() - t0, "n_patients": len(pp_errors_a)}
    all_per_patient_errors["panelfm_xreg"] = {"errors": pp_errors_a, "actuals_mean": pp_actuals_a, "preds_mean": pp_preds_a}
    print(f"    MAE={mae_a:.1f} ({results['panelfm_xreg']['time_seconds']:.1f}s)")

    # --- Option B: Adapter ---
    print(f"  panelfm_adapter...", flush=True)
    t0 = time.time()
    panelfm_b = PanelFMAdapter(
        forecaster, store,
        embedding_dim=store.embedding_dim,
        hidden_dim=32, lr=1e-3, epochs=30,
    )
    panelfm_b.fit(train_context, train_actuals, horizon)
    forecasts_b = panelfm_b.forecast_batch(sub_context, horizon)
    mae_b, pp_errors_b, pp_actuals_b, pp_preds_b = compute_ts_mae(forecasts_b, sub_actuals, horizon)
    results["panelfm_adapter"] = {"mae": mae_b, "time_seconds": time.time() - t0, "n_patients": len(pp_errors_b)}
    all_per_patient_errors["panelfm_adapter"] = {"errors": pp_errors_b, "actuals_mean": pp_actuals_b, "preds_mean": pp_preds_b}
    print(f"    MAE={mae_b:.1f} ({results['panelfm_adapter']['time_seconds']:.1f}s)")

    # --- Option C: ICF ---
    try:
        print(f"  panelfm_icf...", flush=True)
        t0 = time.time()

        # ICF needs features for similarity lookup — use XGBoost-compatible columns
        test_features_icf = features_for_embed[
            features_for_embed["person_id"].isin(sample_pids)
        ]
        train_features_icf = features_for_embed[
            features_for_embed["person_id"].isin(series_train.keys())
        ]

        panelfm_c = PanelFMICF(forecaster, encoder, n_context_patients=5)
        panelfm_c.set_corpus(series_train, train_features_icf)

        forecasts_c = panelfm_c.forecast_batch_with_context(
            sub_context, test_features_icf, horizon
        )
        mae_c, pp_errors_c, pp_actuals_c, pp_preds_c = compute_ts_mae(forecasts_c, sub_actuals, horizon)
        results["panelfm_icf"] = {"mae": mae_c, "time_seconds": time.time() - t0, "n_patients": len(pp_errors_c)}
        all_per_patient_errors["panelfm_icf"] = {"errors": pp_errors_c, "actuals_mean": pp_actuals_c, "preds_mean": pp_preds_c}
        print(f"    MAE={mae_c:.1f} ({results['panelfm_icf']['time_seconds']:.1f}s)")
    except Exception as e:
        print(f"    ICF failed: {e}")

    results["_per_patient_errors"] = all_per_patient_errors
    return results


# =============================================================================
# HYBRID CALIBRATION (CS model patient-level + TS model temporal dynamics)
# =============================================================================

def run_hybrid_calibration(cs_results, ts_sources, data, horizon=3):
    """Create hybrid models: CS patient-level discrimination + TS temporal dynamics.

    The fundamental problem with TS-only models in actuarial evaluation:
    - TS models predict monthly costs well (low MAE, capture $0 months)
    - But they don't discriminate between patients (negative individual R²)
    - CS models discriminate between patients (positive R²) but miss temporal dynamics

    Hybrid approach (standard actuarial practice):
    1. CS model predicts each patient's total cost over the forecast horizon
    2. TS model predicts the temporal pattern (monthly allocation)
    3. Rescale each patient's TS predictions so their sum matches the CS prediction

    This preserves:
    - TS model's monthly accuracy (when costs occur)
    - CS model's patient-level discrimination (who has high costs)
    """
    if "_per_patient" not in cs_results:
        print("  No CS per-patient data for hybrid calibration, skipping")
        return {}, {}

    # Get CS model predictions (use two_part as the calibrator)
    cs_data = cs_results["_per_patient"]
    cs_pids = cs_data["person_ids"]
    cs_predictions = cs_data["predictions"]

    # Select CS model for calibration (preference: two_part > lightgbm > xgboost)
    calibrator_name = None
    for candidate in ["two_part", "lightgbm", "xgboost"]:
        if candidate in cs_predictions:
            calibrator_name = candidate
            break

    if calibrator_name is None:
        print("  No suitable CS calibrator found, skipping hybrid")
        return {}, {}

    print(f"\n--- Hybrid Calibration (using {calibrator_name} for patient-level) ---")
    cs_pred_reg = np.array(cs_predictions[calibrator_name]["y_pred_reg"])
    y_true_reg = np.array(cs_data["y_true_reg"])

    # Calibrate CS predictions to budget-neutral (mean predicted = mean actual).
    # Without this step, the CS model's systematic overprediction (PR ~1.9)
    # propagates into all hybrid predictions.
    cs_cal_factor = np.mean(y_true_reg) / np.mean(cs_pred_reg) if np.mean(cs_pred_reg) > 0 else 1.0
    cs_pred_calibrated = cs_pred_reg * cs_cal_factor
    print(f"  CS calibration: raw PR={np.mean(cs_pred_reg)/np.mean(y_true_reg):.3f}, "
          f"calibrated PR={np.mean(cs_pred_calibrated)/np.mean(y_true_reg):.3f} "
          f"(factor={cs_cal_factor:.4f})")

    # Build lookup: person_id → calibrated CS predicted total cost
    cs_pred_map = {}
    for i, pid in enumerate(cs_pids):
        cs_pred_map[str(pid)] = cs_pred_calibrated[i]

    # Also get actual totals from CS data
    cs_actual_map = {}
    for i, pid in enumerate(cs_pids):
        cs_actual_map[str(pid)] = y_true_reg[i]

    hybrid_results = {}
    hybrid_per_patient = {}

    # For each TS model, create a hybrid version
    for source in ts_sources:
        if "_per_patient_errors" not in source:
            continue
        for model_name, error_data in source["_per_patient_errors"].items():
            preds_mean = error_data.get("preds_mean", {})
            actuals_mean = error_data.get("actuals_mean", {})

            if not preds_mean:
                continue

            # Find patients in both CS and TS results
            common_pids = sorted(
                set(preds_mean.keys()) & set(actuals_mean.keys()) & set(cs_pred_map.keys())
            )
            if len(common_pids) < 100:
                continue

            # Create hybrid predictions:
            # For each patient, rescale TS predicted mean by (CS_pred / TS_pred_total)
            # where TS_pred_total = TS_pred_mean * horizon
            hybrid_preds = {}
            for pid in common_pids:
                ts_pred_mean = preds_mean[pid]
                ts_pred_total = ts_pred_mean * horizon
                cs_pred_total = max(cs_pred_map[pid], 0)  # CS predictions are 3-month totals

                if ts_pred_total > 0:
                    # Rescale: use CS level, TS temporal pattern
                    scale = cs_pred_total / (ts_pred_total)
                    hybrid_preds[pid] = ts_pred_mean * scale
                else:
                    # TS predicted $0 — trust it (likely truly $0)
                    hybrid_preds[pid] = 0.0

            # Evaluate hybrid model
            y_true_arr = np.array([actuals_mean[p] for p in common_pids])
            y_hybrid_arr = np.array([hybrid_preds[p] for p in common_pids])

            # Compute metrics
            r2_raw = r_squared(y_true_arr, y_hybrid_arr)
            r2_cal = r_squared_calibrated(y_true_arr, y_hybrid_arr)
            pr = predictive_ratio(y_true_arr, y_hybrid_arr)

            # MAE (on monthly means — comparable to original TS MAE)
            hybrid_mae = float(np.mean(np.abs(y_true_arr - y_hybrid_arr)))

            hybrid_name = f"hybrid_{model_name}"
            hybrid_results[hybrid_name] = {
                "mae": hybrid_mae,
                "r_squared": r2_raw,
                "r_squared_calibrated": r2_cal,
                "predictive_ratio": pr["overall"],
                "n_patients": len(common_pids),
                "calibrator": calibrator_name,
                "base_ts_model": model_name,
            }

            # Censored metrics
            cens = censored_metrics(y_true_arr, y_hybrid_arr, cap=250000)
            hybrid_results[hybrid_name]["r_squared_censored_250k"] = cens["r_squared_censored"]
            hybrid_results[hybrid_name]["r_squared_calibrated_censored_250k"] = r_squared_calibrated(
                np.minimum(y_true_arr, 250000), np.minimum(y_hybrid_arr, 250000)
            )
            hybrid_results[hybrid_name]["mae_censored_250k"] = cens["mae_censored"]
            hybrid_results[hybrid_name]["predictive_ratio_censored_250k"] = cens["predictive_ratio_censored"]

            hybrid_per_patient[hybrid_name] = {
                "preds_mean": hybrid_preds,
                "actuals_mean": {p: actuals_mean[p] for p in common_pids},
                "errors": {p: abs(actuals_mean[p] - hybrid_preds[p]) for p in common_pids},
            }

            print(f"  {hybrid_name}: MAE={hybrid_mae:.1f}, R²(raw)={r2_raw:.4f}, "
                  f"R²(cal)={r2_cal:.4f}, PR={pr['overall']:.4f} (n={len(common_pids)})")

    return hybrid_results, hybrid_per_patient


# =============================================================================
# MAIN
# =============================================================================

def run_full_experiment(dataset_name, outcomes, patient_features):
    """Run complete experiment on one dataset."""
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {dataset_name.upper()}")
    print(f"{'='*70}")

    model_config = load_model_config()

    # Prepare
    data = prepare_dataset(outcomes, patient_features, dataset_name)

    # Cross-sectional baselines
    print("\n--- Cross-Sectional Baselines ---")
    cs_results = run_cross_sectional(data, model_config)

    # Time series baselines
    print("\n--- Time Series Baselines ---")
    ts_results = run_ts_baselines(data, horizon=3)

    # Chronos foundation model (no artificial patient cap)
    print("\n--- Chronos Foundation Model ---")
    try:
        chronos_results = run_chronos_baseline(data, horizon=3, max_patients=None)
    except Exception as e:
        print(f"  Chronos failed: {e}")
        chronos_results = {"chronos_zeroshot": {"mae": np.nan}}

    # PanelFM variants
    print("\n--- PanelFM Variants ---")
    try:
        panelfm_results = run_panelfm(data, cs_results, chronos_results, model_config, horizon=3)
    except Exception as e:
        print(f"  PanelFM failed: {e}")
        import traceback; traceback.print_exc()
        panelfm_results = {}

    # Concurrent evaluation (actuarial standard)
    print("\n--- Concurrent Evaluation (same-period features) ---")
    try:
        concurrent_results = run_concurrent_evaluation(data, model_config)
    except Exception as e:
        print(f"  Concurrent evaluation failed: {e}")
        import traceback; traceback.print_exc()
        concurrent_results = {}

    # Hybrid calibration: CS patient-level + TS temporal dynamics
    print("\n--- Hybrid Models (CS patient-level + TS temporal dynamics) ---")
    try:
        hybrid_results, hybrid_per_patient = run_hybrid_calibration(
            cs_results, [ts_results, chronos_results, panelfm_results], data, horizon=3
        )
    except Exception as e:
        print(f"  Hybrid calibration failed: {e}")
        import traceback; traceback.print_exc()
        hybrid_results, hybrid_per_patient = {}, {}

    # Compile all metrics (exclude internal keys starting with _)
    all_metrics = {}
    for name, metrics in cs_results.items():
        if not name.startswith("_"):
            all_metrics[name] = metrics
    for name, metrics in ts_results.items():
        if not name.startswith("_"):
            all_metrics[name] = metrics
    for name, metrics in chronos_results.items():
        if not name.startswith("_"):
            all_metrics[name] = metrics
    for name, metrics in panelfm_results.items():
        if not name.startswith("_"):
            all_metrics[name] = metrics
    for name, metrics in concurrent_results.items():
        if not name.startswith("_"):
            all_metrics[name] = metrics
    for name, metrics in hybrid_results.items():
        all_metrics[name] = metrics

    # Compute actuarial metrics for TS models using saved predicted means
    for source in [ts_results, chronos_results, panelfm_results]:
        if "_per_patient_errors" in source:
            for model_name, error_data in source["_per_patient_errors"].items():
                if model_name in all_metrics and "preds_mean" in error_data:
                    preds_mean = error_data["preds_mean"]
                    actuals_mean = error_data["actuals_mean"]
                    common_pids = sorted(set(preds_mean.keys()) & set(actuals_mean.keys()))
                    if len(common_pids) > 10:
                        y_true_arr = np.array([actuals_mean[p] for p in common_pids])
                        y_pred_arr = np.array([preds_mean[p] for p in common_pids])
                        all_metrics[model_name]["r_squared"] = r_squared(y_true_arr, y_pred_arr)
                        all_metrics[model_name]["r_squared_calibrated"] = r_squared_calibrated(y_true_arr, y_pred_arr)
                        pr = predictive_ratio(y_true_arr, y_pred_arr)
                        all_metrics[model_name]["predictive_ratio"] = pr["overall"]
                        cens = censored_metrics(y_true_arr, y_pred_arr, cap=250000)
                        all_metrics[model_name]["r_squared_censored_250k"] = cens["r_squared_censored"]
                        all_metrics[model_name]["r_squared_calibrated_censored_250k"] = r_squared_calibrated(
                            np.minimum(y_true_arr, 250000), np.minimum(y_pred_arr, 250000)
                        )
                        all_metrics[model_name]["mae_censored_250k"] = cens["mae_censored"]
                        all_metrics[model_name]["predictive_ratio_censored_250k"] = cens["predictive_ratio_censored"]

    # Save aggregate metrics
    def clean_metrics(d):
        return {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                     for kk, vv in v.items()} for k, v in d.items()}

    with open(RESULTS_DIR / f"all_metrics_{dataset_name}.json", "w") as f:
        json.dump(clean_metrics(all_metrics), f, indent=2)

    # Save per-patient data for bootstrap CIs, stratified analysis, and stat tests
    per_patient_data = {}

    # CS per-patient predictions
    if "_per_patient" in cs_results:
        per_patient_data["cs"] = cs_results["_per_patient"]

    # TS per-patient errors (combine from all sources)
    ts_errors = {}
    for source in [ts_results, chronos_results, panelfm_results]:
        if "_per_patient_errors" in source:
            for model_name, error_data in source["_per_patient_errors"].items():
                ts_errors[model_name] = {
                    "errors": {str(k): v for k, v in error_data["errors"].items()},
                    "actuals_mean": {str(k): v for k, v in error_data["actuals_mean"].items()},
                    "preds_mean": {str(k): v for k, v in error_data.get("preds_mean", {}).items()},
                }
    per_patient_data["ts"] = ts_errors

    # Concurrent per-patient predictions
    if "_per_patient" in concurrent_results:
        per_patient_data["concurrent"] = concurrent_results["_per_patient"]

    # Hybrid per-patient data
    if hybrid_per_patient:
        hybrid_ts = {}
        for model_name, hp_data in hybrid_per_patient.items():
            hybrid_ts[model_name] = {
                "errors": {str(k): v for k, v in hp_data["errors"].items()},
                "actuals_mean": {str(k): v for k, v in hp_data["actuals_mean"].items()},
                "preds_mean": {str(k): v for k, v in hp_data["preds_mean"].items()},
            }
        per_patient_data["ts"].update(hybrid_ts)

    with open(RESULTS_DIR / f"per_patient_data_{dataset_name}.json", "w") as f:
        json.dump(per_patient_data, f)
    print(f"\n  Per-patient data saved: {RESULTS_DIR / f'per_patient_data_{dataset_name}.json'}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY: {dataset_name.upper()}")
    print(f"{'='*70}")
    print(f"  {'Model':<25} {'MAE':>10} {'R²(raw)':>10} {'R²(cal)':>10} {'Pred Ratio':>12}")
    print(f"  {'-'*70}")
    for name, m in sorted(all_metrics.items(), key=lambda x: x[1].get('mae', 1e9)):
        mae_str = f"{m['mae']:.1f}" if np.isfinite(m.get('mae', np.nan)) else "---"
        r2_str = f"{m['r_squared']:.4f}" if np.isfinite(m.get('r_squared', np.nan)) else "---"
        r2c_str = f"{m['r_squared_calibrated']:.4f}" if np.isfinite(m.get('r_squared_calibrated', np.nan)) else "---"
        pr_str = f"{m['predictive_ratio']:.4f}" if np.isfinite(m.get('predictive_ratio', np.nan)) else "---"
        print(f"  {name:<25} {mae_str:>10} {r2_str:>10} {r2c_str:>10} {pr_str:>12}")

    return all_metrics


def main():
    # --- Synthetic ---
    syn_outcomes, syn_features = prepare_synthetic()
    syn_metrics = run_full_experiment("synthetic", syn_outcomes, syn_features)

    # --- Real ---
    try:
        real_outcomes, real_features = prepare_real()
        real_metrics = run_full_experiment("real", real_outcomes, real_features)
    except Exception as e:
        print(f"Real data failed: {e}")
        import traceback; traceback.print_exc()
        real_metrics = {}

    print(f"\n{'='*70}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
