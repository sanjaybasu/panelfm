#!/usr/bin/env python3
"""
Generate publication tables and figures for JAMIA Open manuscript.

Outputs:
  - results/table1_demographics.csv
  - results/bootstrap_cis.json           (CIs for ALL models)
  - results/stratified_analysis.json     (zero vs non-zero cost months)
  - results/statistical_tests.json       (paired bootstrap tests)
  - results/figure1_mae_comparison.pdf
  - results/figure2_auroc_comparison.pdf

Requires real data outputs from the panelfm experiment, including
per_patient_data_real.json for bootstrap CIs.
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PANELFM_DIR = Path(__file__).parent.parent.parent / "panelfm"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PANELFM_RESULTS = PANELFM_DIR / "results"

sys.path.insert(0, str(PANELFM_DIR))


def load_real_data():
    """Load real outcomes and features from panelfm results."""
    outcomes = pd.read_parquet(PANELFM_RESULTS / "real_outcomes_filtered.parquet")
    features = pd.read_parquet(PANELFM_RESULTS / "real_features.parquet")
    return outcomes, features


def load_metrics():
    """Load all metrics from panelfm results."""
    with open(PANELFM_RESULTS / "all_metrics_real.json") as f:
        real = json.load(f)
    with open(PANELFM_RESULTS / "all_metrics_synthetic.json") as f:
        synthetic = json.load(f)
    return real, synthetic


def load_per_patient_data():
    """Load per-patient predictions/errors saved by run_experiment.py."""
    path = PANELFM_RESULTS / "per_patient_data_real.json"
    if not path.exists():
        print(f"  WARNING: {path} not found. Bootstrap CIs will be limited.")
        return None
    with open(path) as f:
        return json.load(f)


def generate_table1(outcomes, features):
    """
    Table 1: Demographic and clinical characteristics of the study population.
    Stratified by temporal split (train / validation / test).
    """
    from src.data.load_claims import temporal_train_val_test_split

    for col in ["total_paid", "emergency_department_ct", "acute_inpatient_ct"]:
        if col in outcomes.columns:
            outcomes[col] = outcomes[col].fillna(0)

    splits = temporal_train_val_test_split(outcomes, val_months=3, test_months=3)

    rows = []

    for split_name, split_df in [
        ("Training", splits["train"]),
        ("Validation", splits["val"]),
        ("Test", splits["test"]),
        ("Overall", outcomes),
    ]:
        n_patients = split_df["person_id"].nunique()
        n_patient_months = len(split_df)

        cost = split_df["total_paid"]
        ed = split_df["emergency_department_ct"] if "emergency_department_ct" in split_df.columns else pd.Series()
        ip = split_df["acute_inpatient_ct"] if "acute_inpatient_ct" in split_df.columns else pd.Series()

        patient_months = split_df.groupby("person_id").size()

        row = {
            "Split": split_name,
            "N patients": f"{n_patients:,}",
            "N patient-months": f"{n_patient_months:,}",
            "Months per patient, median (IQR)": (
                f"{patient_months.median():.0f} "
                f"({patient_months.quantile(0.25):.0f}-{patient_months.quantile(0.75):.0f})"
            ),
            "Monthly cost ($), median (IQR)": (
                f"{cost.median():.0f} "
                f"({cost.quantile(0.25):.0f}-{cost.quantile(0.75):.0f})"
            ),
            "Monthly cost ($), mean (SD)": f"{cost.mean():.0f} ({cost.std():.0f})",
            "Months with $0 cost, %": f"{(cost == 0).mean() * 100:.1f}",
            "Any ED visit, %": (
                f"{(ed > 0).mean() * 100:.1f}" if len(ed) > 0 else "N/A"
            ),
            "Any IP admission, %": (
                f"{(ip > 0).mean() * 100:.1f}" if len(ip) > 0 else "N/A"
            ),
        }

        if features is not None and len(features) > 0:
            pids_in_split = set(split_df["person_id"].unique())
            feat_sub = features[features["person_id"].isin(pids_in_split)]

            if "age" in feat_sub.columns:
                age = feat_sub["age"].dropna()
                row["Age, mean (SD)"] = f"{age.mean():.1f} ({age.std():.1f})"

            if "female" in feat_sub.columns:
                row["Female, %"] = f"{feat_sub['female'].mean() * 100:.1f}"

            if "dual_eligible" in feat_sub.columns:
                row["Dual eligible, %"] = f"{feat_sub['dual_eligible'].mean() * 100:.1f}"

        rows.append(row)

    table1 = pd.DataFrame(rows)
    table1.to_csv(RESULTS_DIR / "table1_demographics.csv", index=False)
    print(f"Table 1 saved: {RESULTS_DIR / 'table1_demographics.csv'}")
    return table1


# =============================================================================
# BOOTSTRAP CI COMPUTATION (from saved per-patient data)
# =============================================================================

def bootstrap_ci(values, stat_fn=np.mean, n_boot=2000, alpha=0.05, seed=42):
    """Compute bootstrap confidence interval for a statistic."""
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_stats = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_stats[b] = stat_fn(values[idx])
    lo = np.percentile(boot_stats, 100 * alpha / 2)
    hi = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    point = stat_fn(values)
    return point, lo, hi


def compute_bootstrap_cis_all_models(per_patient_data, n_boot=2000):
    """
    Compute bootstrap CIs for ALL models using saved per-patient data.

    For CS models: bootstrap over per-patient absolute errors and classification scores.
    For TS models: bootstrap over per-patient MAE values saved from the experiment.

    This ensures CIs are computed on the EXACT same evaluation as the original
    experiment (no re-running models, no different patient samples).
    """
    from sklearn.metrics import mean_absolute_error, roc_auc_score, average_precision_score

    results = {}

    # --- Cross-sectional models ---
    if per_patient_data and "cs" in per_patient_data:
        cs_data = per_patient_data["cs"]
        y_true_reg = np.array(cs_data["y_true_reg"])
        y_true_cls = np.array(cs_data["y_true_cls"])
        n_test = len(y_true_reg)

        rng = np.random.default_rng(42)

        for model_name, preds in cs_data.get("predictions", {}).items():
            y_pred_reg = np.array(preds["y_pred_reg"])
            y_score_cls = np.array(preds["y_score_cls"])

            boot_mae = np.zeros(n_boot)
            boot_rmse = np.zeros(n_boot)
            boot_auroc = np.zeros(n_boot)
            boot_auprc = np.zeros(n_boot)

            for b in range(n_boot):
                idx = rng.integers(0, n_test, size=n_test)
                boot_mae[b] = mean_absolute_error(y_true_reg[idx], y_pred_reg[idx])
                boot_rmse[b] = np.sqrt(np.mean((y_true_reg[idx] - y_pred_reg[idx]) ** 2))
                try:
                    boot_auroc[b] = roc_auc_score(y_true_cls[idx], y_score_cls[idx])
                except ValueError:
                    boot_auroc[b] = np.nan
                try:
                    boot_auprc[b] = average_precision_score(y_true_cls[idx], y_score_cls[idx])
                except ValueError:
                    boot_auprc[b] = np.nan

            results[model_name] = {
                "mae": round(float(mean_absolute_error(y_true_reg, y_pred_reg)), 1),
                "mae_ci_lower": round(float(np.percentile(boot_mae, 2.5)), 1),
                "mae_ci_upper": round(float(np.percentile(boot_mae, 97.5)), 1),
                "rmse": round(float(np.sqrt(np.mean((y_true_reg - y_pred_reg) ** 2))), 1),
                "rmse_ci_lower": round(float(np.percentile(boot_rmse, 2.5)), 1),
                "rmse_ci_upper": round(float(np.percentile(boot_rmse, 97.5)), 1),
                "auroc": round(float(np.nanmean(boot_auroc)), 4),
                "auroc_ci_lower": round(float(np.nanpercentile(boot_auroc, 2.5)), 4),
                "auroc_ci_upper": round(float(np.nanpercentile(boot_auroc, 97.5)), 4),
                "auprc": round(float(np.nanmean(boot_auprc)), 4),
                "auprc_ci_lower": round(float(np.nanpercentile(boot_auprc, 2.5)), 4),
                "auprc_ci_upper": round(float(np.nanpercentile(boot_auprc, 97.5)), 4),
                "n_test": n_test,
            }
            print(
                f"  {model_name}: MAE={results[model_name]['mae']:.1f} "
                f"({results[model_name]['mae_ci_lower']:.1f}-{results[model_name]['mae_ci_upper']:.1f}), "
                f"AUROC={results[model_name]['auroc']:.4f} "
                f"({results[model_name]['auroc_ci_lower']:.4f}-{results[model_name]['auroc_ci_upper']:.4f})"
            )

    # --- Time series models (including PanelFM) ---
    if per_patient_data and "ts" in per_patient_data:
        for model_name, error_data in per_patient_data["ts"].items():
            errors = np.array(list(error_data["errors"].values()))
            if len(errors) == 0:
                continue
            point, lo, hi = bootstrap_ci(errors, np.mean, n_boot=n_boot, seed=42)
            results[model_name] = {
                "mae": round(float(point), 1),
                "mae_ci_lower": round(float(lo), 1),
                "mae_ci_upper": round(float(hi), 1),
                "n_patients": len(errors),
            }
            print(f"  {model_name}: MAE={point:.1f} (95% CI: {lo:.1f}-{hi:.1f}, n={len(errors)})")

    return results


# =============================================================================
# STRATIFIED ANALYSIS (zero vs non-zero cost months)
# =============================================================================

def compute_stratified_analysis(per_patient_data, real_metrics):
    """
    Stratified error analysis: patients with all-zero cost vs patients
    with any non-zero cost in the test period.

    For TS models: stratify by mean actual cost (zero vs non-zero).
    For CS models: stratify by actual total_paid_sum (zero vs non-zero).
    """
    results = {}

    # TS models
    if per_patient_data and "ts" in per_patient_data:
        print("\n  Time series models - stratified by test-period cost:")
        for model_name, error_data in per_patient_data["ts"].items():
            errors = error_data["errors"]
            actuals_mean = error_data["actuals_mean"]

            zero_errors = []
            nonzero_errors = []

            for pid, err in errors.items():
                mean_actual = actuals_mean.get(pid, 0)
                if mean_actual == 0:
                    zero_errors.append(err)
                else:
                    nonzero_errors.append(err)

            zero_errors = np.array(zero_errors) if zero_errors else np.array([])
            nonzero_errors = np.array(nonzero_errors) if nonzero_errors else np.array([])

            model_results = {
                "overall_mae": round(float(np.mean(list(errors.values()))), 1),
                "n_total": len(errors),
            }

            if len(zero_errors) > 0:
                model_results["zero_cost_mae"] = round(float(np.mean(zero_errors)), 1)
                model_results["zero_cost_n"] = len(zero_errors)
                model_results["zero_cost_pct"] = round(100 * len(zero_errors) / len(errors), 1)
            if len(nonzero_errors) > 0:
                model_results["nonzero_cost_mae"] = round(float(np.mean(nonzero_errors)), 1)
                model_results["nonzero_cost_n"] = len(nonzero_errors)
                model_results["nonzero_cost_pct"] = round(100 * len(nonzero_errors) / len(errors), 1)

            results[model_name] = model_results
            z_mae = model_results.get("zero_cost_mae", "N/A")
            nz_mae = model_results.get("nonzero_cost_mae", "N/A")
            print(f"    {model_name}: zero-cost MAE={z_mae}, nonzero-cost MAE={nz_mae}")

    # CS models
    if per_patient_data and "cs" in per_patient_data:
        cs_data = per_patient_data["cs"]
        y_true_reg = np.array(cs_data["y_true_reg"])
        zero_mask = y_true_reg == 0
        nonzero_mask = y_true_reg > 0

        print(f"\n  Cross-sectional models - stratified by 3-month total cost:")
        print(f"    Zero-cost patients: {zero_mask.sum()} ({100*zero_mask.mean():.1f}%)")
        print(f"    Non-zero cost patients: {nonzero_mask.sum()} ({100*nonzero_mask.mean():.1f}%)")

        for model_name, preds in cs_data.get("predictions", {}).items():
            y_pred_reg = np.array(preds["y_pred_reg"])

            abs_errors = np.abs(y_true_reg - y_pred_reg)

            model_results = {
                "overall_mae": round(float(np.mean(abs_errors)), 1),
                "n_total": len(abs_errors),
            }

            if zero_mask.sum() > 0:
                model_results["zero_cost_mae"] = round(float(np.mean(abs_errors[zero_mask])), 1)
                model_results["zero_cost_n"] = int(zero_mask.sum())
                model_results["zero_cost_pct"] = round(100 * zero_mask.mean(), 1)
            if nonzero_mask.sum() > 0:
                model_results["nonzero_cost_mae"] = round(float(np.mean(abs_errors[nonzero_mask])), 1)
                model_results["nonzero_cost_n"] = int(nonzero_mask.sum())
                model_results["nonzero_cost_pct"] = round(100 * nonzero_mask.mean(), 1)

            results[model_name] = model_results
            z_mae = model_results.get("zero_cost_mae", "N/A")
            nz_mae = model_results.get("nonzero_cost_mae", "N/A")
            print(f"    {model_name}: zero-cost MAE={z_mae}, nonzero-cost MAE={nz_mae}")

    return results


# =============================================================================
# FORMAL STATISTICAL TESTS (paired bootstrap)
# =============================================================================

def paired_bootstrap_test(errors_a, errors_b, n_boot=10000, seed=42):
    """
    Paired bootstrap test for difference in mean MAE.
    H0: mean(errors_a) = mean(errors_b)

    Uses paired resampling: for each bootstrap, the same patients are
    sampled for both models, preserving the correlation structure.

    Returns observed difference, 95% CI for difference, and p-value.
    """
    rng = np.random.default_rng(seed)
    n = len(errors_a)
    obs_diff = np.mean(errors_a) - np.mean(errors_b)

    boot_diffs = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_diffs[b] = np.mean(errors_a[idx]) - np.mean(errors_b[idx])

    # Two-sided p-value: fraction of bootstrap diffs that are more extreme
    # than the observed diff in magnitude (under H0, center at 0)
    centered_diffs = boot_diffs - np.mean(boot_diffs)
    p_value = float(np.mean(np.abs(centered_diffs) >= np.abs(obs_diff)))

    return {
        "diff_mean": round(float(obs_diff), 2),
        "diff_ci_lower": round(float(np.percentile(boot_diffs, 2.5)), 2),
        "diff_ci_upper": round(float(np.percentile(boot_diffs, 97.5)), 2),
        "p_value": round(float(p_value), 4),
        "significant_005": p_value < 0.05,
        "n_patients": n,
    }


def compute_statistical_tests(per_patient_data, real_metrics):
    """
    Compute formal pairwise statistical tests between models.

    Key comparisons:
    1. Best TS model (Chronos) vs best CS model (LightGBM) — main claim
    2. PanelFM variants vs Chronos — does conditioning help?
    3. Two-part model vs standard XGBoost — does zero-inflation modeling help?
    """
    results = {}

    # TS model comparisons (paired on common patients)
    if per_patient_data and "ts" in per_patient_data:
        ts_data = per_patient_data["ts"]

        # Find common patients across all TS models
        ts_models = list(ts_data.keys())
        if len(ts_models) >= 2:
            # Get patient IDs for each model
            model_pids = {m: set(ts_data[m]["errors"].keys()) for m in ts_models}

            # Pairwise comparisons among TS models
            comparisons = [
                ("chronos_zeroshot", "panelfm_icf"),
                ("chronos_zeroshot", "panelfm_adapter"),
                ("chronos_zeroshot", "panelfm_xreg"),
                ("chronos_zeroshot", "naive_mean3"),
                ("chronos_zeroshot", "arima"),
                ("panelfm_icf", "panelfm_adapter"),
            ]

            for model_a, model_b in comparisons:
                if model_a not in ts_data or model_b not in ts_data:
                    continue

                common = sorted(model_pids[model_a] & model_pids[model_b])
                if len(common) < 30:
                    continue

                errors_a = np.array([ts_data[model_a]["errors"][pid] for pid in common])
                errors_b = np.array([ts_data[model_b]["errors"][pid] for pid in common])

                test_result = paired_bootstrap_test(errors_a, errors_b)
                test_result["model_a"] = model_a
                test_result["model_b"] = model_b
                results[f"{model_a}_vs_{model_b}"] = test_result

                sig = "*" if test_result["significant_005"] else ""
                print(
                    f"  {model_a} vs {model_b}: "
                    f"diff={test_result['diff_mean']:.1f} "
                    f"(95% CI: {test_result['diff_ci_lower']:.1f} to {test_result['diff_ci_upper']:.1f}), "
                    f"p={test_result['p_value']:.4f}{sig}"
                )

    # CS model comparisons (paired on all test patients)
    if per_patient_data and "cs" in per_patient_data:
        cs_data = per_patient_data["cs"]
        y_true_reg = np.array(cs_data["y_true_reg"])

        cs_models = list(cs_data.get("predictions", {}).keys())

        cs_comparisons = [
            ("demographic_glm", "xgboost"),
            ("demographic_glm", "two_part"),
            ("two_part", "xgboost"),
            ("two_part", "lightgbm"),
            ("lightgbm", "xgboost"),
            ("lightgbm", "random_forest"),
            ("lightgbm", "stacking"),
        ]

        for model_a, model_b in cs_comparisons:
            if model_a not in cs_data["predictions"] or model_b not in cs_data["predictions"]:
                continue

            preds_a = np.array(cs_data["predictions"][model_a]["y_pred_reg"])
            preds_b = np.array(cs_data["predictions"][model_b]["y_pred_reg"])

            errors_a = np.abs(y_true_reg - preds_a)
            errors_b = np.abs(y_true_reg - preds_b)

            test_result = paired_bootstrap_test(errors_a, errors_b)
            test_result["model_a"] = model_a
            test_result["model_b"] = model_b
            results[f"{model_a}_vs_{model_b}"] = test_result

            sig = "*" if test_result["significant_005"] else ""
            print(
                f"  {model_a} vs {model_b}: "
                f"diff={test_result['diff_mean']:.1f} "
                f"(95% CI: {test_result['diff_ci_lower']:.1f} to {test_result['diff_ci_upper']:.1f}), "
                f"p={test_result['p_value']:.4f}{sig}"
            )

    return results


# =============================================================================
# ACTUARIAL METRICS (SOA standard evaluation)
# =============================================================================

def compute_actuarial_metrics(per_patient_data, real_metrics):
    """
    Compute actuarial-standard metrics for all models:
    - R² (individual-level coefficient of determination)
    - Predictive ratio (predicted/actual, overall and by subgroup)
    - Decile analysis (predictive ratios by actual cost decile)
    - Cost bucket calibration
    - High-cost identification (top 1%, 5%, 10%)
    - Cost-censored evaluation ($250K cap, SOA standard)

    These are the metrics an SOA reviewer would expect to see.
    """
    from src.evaluation.metrics import (
        r_squared, r_squared_calibrated, calibrate_predictions,
        predictive_ratio, decile_analysis,
        cost_bucket_calibration, censored_metrics, high_cost_identification,
    )

    results = {}

    # --- Cross-sectional models ---
    if per_patient_data and "cs" in per_patient_data:
        cs_data = per_patient_data["cs"]
        y_true = np.array(cs_data["y_true_reg"])

        for model_name, preds in cs_data.get("predictions", {}).items():
            y_pred = np.array(preds["y_pred_reg"])
            model_results = {
                "r_squared": r_squared(y_true, y_pred),
                "r_squared_calibrated": r_squared_calibrated(y_true, y_pred),
                "predictive_ratio": predictive_ratio(y_true, y_pred),
                "decile_analysis": decile_analysis(y_true, calibrate_predictions(y_true, y_pred)),
                "cost_buckets": cost_bucket_calibration(y_true, calibrate_predictions(y_true, y_pred)),
                "high_cost": high_cost_identification(y_true, y_pred),
                "censored_250k": censored_metrics(y_true, y_pred, cap=250000),
                "r_squared_calibrated_censored_250k": r_squared_calibrated(
                    np.minimum(y_true, 250000), np.minimum(y_pred, 250000)
                ),
                "n_patients": len(y_true),
                "model_type": "cross_sectional",
            }
            results[model_name] = model_results
            print(f"  {model_name}: R²(raw)={model_results['r_squared']:.4f}, "
                  f"R²(cal)={model_results['r_squared_calibrated']:.4f}, "
                  f"PR={model_results['predictive_ratio']['overall']:.4f}")

    # --- Time series models (using saved predicted means) ---
    if per_patient_data and "ts" in per_patient_data:
        for model_name, error_data in per_patient_data["ts"].items():
            preds_mean = error_data.get("preds_mean", {})
            actuals_mean = error_data.get("actuals_mean", {})

            if not preds_mean:
                print(f"  {model_name}: no predicted means saved, skipping actuarial metrics")
                continue

            common_pids = sorted(set(preds_mean.keys()) & set(actuals_mean.keys()))
            if len(common_pids) < 10:
                continue

            y_true = np.array([actuals_mean[p] for p in common_pids])
            y_pred = np.array([preds_mean[p] for p in common_pids])

            model_results = {
                "r_squared": r_squared(y_true, y_pred),
                "r_squared_calibrated": r_squared_calibrated(y_true, y_pred),
                "predictive_ratio": predictive_ratio(y_true, y_pred),
                "decile_analysis": decile_analysis(y_true, calibrate_predictions(y_true, y_pred)),
                "cost_buckets": cost_bucket_calibration(y_true, calibrate_predictions(y_true, y_pred)),
                "high_cost": high_cost_identification(y_true, y_pred),
                "censored_250k": censored_metrics(y_true, y_pred, cap=250000),
                "r_squared_calibrated_censored_250k": r_squared_calibrated(
                    np.minimum(y_true, 250000), np.minimum(y_pred, 250000)
                ),
                "n_patients": len(common_pids),
                "model_type": "time_series",
            }
            results[model_name] = model_results
            print(f"  {model_name}: R²(raw)={model_results['r_squared']:.4f}, "
                  f"R²(cal)={model_results['r_squared_calibrated']:.4f}, "
                  f"PR={model_results['predictive_ratio']['overall']:.4f}")

    # --- Concurrent models ---
    if per_patient_data and "concurrent" in per_patient_data:
        conc_data = per_patient_data["concurrent"]
        y_true = np.array(conc_data["y_true_reg"])

        for model_name, preds in conc_data.get("predictions", {}).items():
            y_pred = np.array(preds["y_pred_reg"])
            model_results = {
                "r_squared": r_squared(y_true, y_pred),
                "r_squared_calibrated": r_squared_calibrated(y_true, y_pred),
                "predictive_ratio": predictive_ratio(y_true, y_pred),
                "decile_analysis": decile_analysis(y_true, calibrate_predictions(y_true, y_pred)),
                "cost_buckets": cost_bucket_calibration(y_true, calibrate_predictions(y_true, y_pred)),
                "high_cost": high_cost_identification(y_true, y_pred),
                "censored_250k": censored_metrics(y_true, y_pred, cap=250000),
                "r_squared_calibrated_censored_250k": r_squared_calibrated(
                    np.minimum(y_true, 250000), np.minimum(y_pred, 250000)
                ),
                "n_patients": len(y_true),
                "model_type": "concurrent",
            }
            results[model_name] = model_results
            print(f"  {model_name}: R²(raw)={model_results['r_squared']:.4f}, "
                  f"R²(cal)={model_results['r_squared_calibrated']:.4f}, "
                  f"PR={model_results['predictive_ratio']['overall']:.4f}")

    return results


def generate_actuarial_summary_table(actuarial_results, real_metrics):
    """
    Generate the main actuarial summary table (Table 2 in actuarial framing):
    Model | MAE | R² | R² (censored) | Pred Ratio | Top 10% Sensitivity | Top 10% PPV | Lift
    """
    rows = []
    model_order = [
        "demographic_glm", "xgboost", "random_forest", "lightgbm",
        "two_part", "stacking",
        "naive_last", "naive_mean3", "arima",
        "chronos_zeroshot", "panelfm_xreg", "panelfm_adapter", "panelfm_icf",
        # Hybrid models (CS patient-level + TS temporal dynamics)
        "hybrid_chronos_zeroshot", "hybrid_panelfm_adapter", "hybrid_panelfm_icf",
        "hybrid_panelfm_xreg",
        # Concurrent models
        "concurrent_demographic_glm", "concurrent_xgboost",
        "concurrent_random_forest", "concurrent_lightgbm", "concurrent_two_part",
    ]

    for model_name in model_order:
        if model_name not in actuarial_results:
            continue
        ar = actuarial_results[model_name]
        rm = real_metrics.get(model_name, {})
        hc = ar.get("high_cost", {}).get("top_10pct", {})

        rows.append({
            "Model": model_name,
            "MAE": round(rm.get("mae", np.nan), 1),
            "R² (raw)": round(ar["r_squared"], 4),
            "R² (calibrated)": round(ar.get("r_squared_calibrated", np.nan), 4),
            "R² (cal+cens $250K)": round(ar.get("r_squared_calibrated_censored_250k", np.nan), 4),
            "Predictive Ratio": round(ar["predictive_ratio"]["overall"], 4),
            "Top 10% Sensitivity": round(hc.get("sensitivity", np.nan), 3),
            "Top 10% PPV": round(hc.get("ppv", np.nan), 3),
            "Top 10% Lift": round(hc.get("lift", np.nan), 1),
            "C-statistic (top 10%)": hc.get("c_statistic", np.nan),
            "N": ar["n_patients"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "actuarial_summary.csv", index=False)
    print(f"\nActuarial summary saved: {RESULTS_DIR / 'actuarial_summary.csv'}")
    return df


def generate_decile_table(actuarial_results):
    """
    Generate decile analysis table (SOA standard):
    Decile | Actual Mean | Model1 PR | Model2 PR | ...

    This is the table an actuary looks at first.
    """
    key_models = [
        "demographic_glm", "xgboost", "lightgbm", "two_part",
        "chronos_zeroshot", "panelfm_adapter", "panelfm_icf",
        "hybrid_chronos_zeroshot", "hybrid_panelfm_adapter",
    ]

    rows = []
    for decile_key in [f"decile_{i}" for i in range(1, 11)] + ["top_10pct", "top_5pct", "top_1pct"]:
        row = {"Segment": decile_key}
        for model_name in key_models:
            if model_name in actuarial_results:
                da = actuarial_results[model_name].get("decile_analysis", {})
                if decile_key in da:
                    row[f"{model_name}_actual"] = da[decile_key].get("actual_mean")
                    row[f"{model_name}_pred"] = da[decile_key].get("pred_mean")
                    row[f"{model_name}_pr"] = da[decile_key].get("predictive_ratio")
                    row[f"{model_name}_n"] = da[decile_key].get("n")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "decile_analysis.csv", index=False)
    print(f"Decile analysis saved: {RESULTS_DIR / 'decile_analysis.csv'}")
    return df


# =============================================================================
# FIGURES
# =============================================================================

def generate_figures(real_metrics, bootstrap_results):
    """Generate publication figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("matplotlib not available; skipping figures")
        return

    plt.rcParams.update({
        "font.size": 10,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
    })

    # Figure 1: MAE comparison bar chart
    model_order = [
        ("Chronos", "chronos_zeroshot"),
        ("PanelFM-ICF", "panelfm_icf"),
        ("PanelFM-Adapter", "panelfm_adapter"),
        ("ARIMA", "arima"),
        ("Naive Mean", "naive_mean3"),
        ("Naive Last", "naive_last"),
        ("PanelFM-XReg", "panelfm_xreg"),
        ("Two-Part", "two_part"),
        ("LightGBM", "lightgbm"),
        ("XGBoost", "xgboost"),
        ("Random Forest", "random_forest"),
        ("Stacking", "stacking"),
    ]

    labels = []
    maes = []
    ci_lo = []
    ci_hi = []
    colors = []

    color_map = {
        "chronos_zeroshot": "#2196F3",
        "panelfm_icf": "#4CAF50",
        "panelfm_adapter": "#4CAF50",
        "panelfm_xreg": "#4CAF50",
        "arima": "#9E9E9E",
        "naive_mean3": "#9E9E9E",
        "naive_last": "#9E9E9E",
        "two_part": "#E91E63",
        "lightgbm": "#FF9800",
        "xgboost": "#FF9800",
        "random_forest": "#FF9800",
        "stacking": "#FF9800",
    }

    for label, key in model_order:
        if key in real_metrics:
            mae_val = real_metrics[key].get("mae", np.nan)
            if np.isfinite(mae_val):
                labels.append(label)
                maes.append(mae_val)
                colors.append(color_map.get(key, "#9E9E9E"))

                if key in bootstrap_results:
                    ci_lo.append(mae_val - bootstrap_results[key].get("mae_ci_lower", mae_val))
                    ci_hi.append(bootstrap_results[key].get("mae_ci_upper", mae_val) - mae_val)
                else:
                    ci_lo.append(0)
                    ci_hi.append(0)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, maes, color=colors, edgecolor="white", height=0.6)
    ax.errorbar(
        maes, y_pos, xerr=[ci_lo, ci_hi],
        fmt="none", ecolor="black", capsize=3, linewidth=1,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean Absolute Error ($ per patient per month)")
    ax.set_title("Three-Month Cost Forecasting Performance")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="Foundation model (zero-shot)"),
        Patch(facecolor="#4CAF50", label="PanelFM (foundation + embedding)"),
        Patch(facecolor="#E91E63", label="Two-part model"),
        Patch(facecolor="#FF9800", label="Cross-sectional ML"),
        Patch(facecolor="#9E9E9E", label="Classical time series / naive"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "figure1_mae_comparison.pdf", bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "figure1_mae_comparison.png", bbox_inches="tight")
    plt.close()
    print(f"Figure 1 saved: {RESULTS_DIR / 'figure1_mae_comparison.pdf'}")

    # Figure 2: AUROC comparison (cross-sectional models only)
    cs_models = [
        ("Two-Part", "two_part"),
        ("XGBoost", "xgboost"),
        ("LightGBM", "lightgbm"),
        ("Random Forest", "random_forest"),
        ("Stacking", "stacking"),
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    labels_cs = []
    aurocs = []
    auroc_ci_lo = []
    auroc_ci_hi = []

    for label, key in cs_models:
        if key in bootstrap_results and "auroc" in bootstrap_results[key]:
            b = bootstrap_results[key]
            labels_cs.append(label)
            aurocs.append(b["auroc"])
            auroc_ci_lo.append(b["auroc"] - b["auroc_ci_lower"])
            auroc_ci_hi.append(b["auroc_ci_upper"] - b["auroc"])

    if labels_cs:
        y_pos = np.arange(len(labels_cs))
        cs_colors = ["#E91E63" if l == "Two-Part" else "#FF9800" for l in labels_cs]
        ax.barh(y_pos, aurocs, color=cs_colors, edgecolor="white", height=0.5)
        ax.errorbar(
            aurocs, y_pos, xerr=[auroc_ci_lo, auroc_ci_hi],
            fmt="none", ecolor="black", capsize=3, linewidth=1,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_cs)
        ax.set_xlabel("AUROC (high-cost classification)")
        ax.set_title("High-Cost Patient Classification (top 10%)")
        ax.invert_yaxis()
        ax.set_xlim(0.80, 0.95)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "figure2_auroc_comparison.pdf", bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "figure2_auroc_comparison.png", bbox_inches="tight")
    plt.close()
    print(f"Figure 2 saved: {RESULTS_DIR / 'figure2_auroc_comparison.pdf'}")


def generate_actuarial_figures(actuarial_results):
    """Generate actuarial-specific figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("matplotlib not available; skipping actuarial figures")
        return

    plt.rcParams.update({
        "font.size": 10,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
    })

    # Figure 3: R² comparison bar chart with CDPS/MARA benchmarks
    model_order = [
        ("Demo GLM", "demographic_glm"),
        ("XGBoost", "xgboost"),
        ("LightGBM", "lightgbm"),
        ("Two-Part", "two_part"),
        ("RF", "random_forest"),
        ("Stacking", "stacking"),
        ("Naive Last", "naive_last"),
        ("Naive Mean", "naive_mean3"),
        ("ARIMA", "arima"),
        ("Chronos", "chronos_zeroshot"),
        ("PanelFM-XReg", "panelfm_xreg"),
        ("PanelFM-Adapter", "panelfm_adapter"),
        ("PanelFM-ICF", "panelfm_icf"),
    ]

    color_map = {
        "demographic_glm": "#795548",
        "chronos_zeroshot": "#2196F3",
        "panelfm_icf": "#4CAF50", "panelfm_adapter": "#4CAF50", "panelfm_xreg": "#4CAF50",
        "arima": "#9E9E9E", "naive_mean3": "#9E9E9E", "naive_last": "#9E9E9E",
        "two_part": "#E91E63",
        "lightgbm": "#FF9800", "xgboost": "#FF9800",
        "random_forest": "#FF9800", "stacking": "#FF9800",
    }

    labels, r2s, colors = [], [], []
    for label, key in model_order:
        if key in actuarial_results:
            r2_val = actuarial_results[key]["r_squared"]
            if np.isfinite(r2_val):
                labels.append(label)
                r2s.append(r2_val * 100)  # Convert to percentage
                colors.append(color_map.get(key, "#9E9E9E"))

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, r2s, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("R² (%)")
    ax.set_title("Individual-Level R² — Comparison with Actuarial Benchmarks")
    ax.invert_yaxis()

    # Add benchmark lines
    ax.axvline(x=3, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(x=15, color="orange", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(x=24, color="green", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(3, len(labels) - 0.5, "Demo-only\n(1-3%)", fontsize=7, color="red", alpha=0.7)
    ax.text(15, len(labels) - 0.5, "CDPS\n(8-24%)", fontsize=7, color="orange", alpha=0.7)
    ax.text(24, len(labels) - 0.5, "CDPS best\n(24%)", fontsize=7, color="green", alpha=0.7)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#795548", label="Demographic-only GLM"),
        Patch(facecolor="#2196F3", label="Foundation model (Chronos)"),
        Patch(facecolor="#4CAF50", label="PanelFM (foundation + embedding)"),
        Patch(facecolor="#E91E63", label="Two-part (hurdle) model"),
        Patch(facecolor="#FF9800", label="Cross-sectional ML"),
        Patch(facecolor="#9E9E9E", label="Classical TS / naive"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "figure3_r2_comparison.pdf", bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "figure3_r2_comparison.png", bbox_inches="tight")
    plt.close()
    print(f"Figure 3 saved: {RESULTS_DIR / 'figure3_r2_comparison.pdf'}")

    # Figure 4: Predictive ratio by cost decile (key models)
    key_models = [
        ("Demo GLM", "demographic_glm", "#795548"),
        ("XGBoost", "xgboost", "#FF9800"),
        ("Two-Part", "two_part", "#E91E63"),
        ("Chronos", "chronos_zeroshot", "#2196F3"),
        ("PanelFM-Adapter", "panelfm_adapter", "#4CAF50"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    decile_x = np.arange(1, 11)

    for label, key, color in key_models:
        if key not in actuarial_results:
            continue
        da = actuarial_results[key].get("decile_analysis", {})
        prs = []
        for d in range(1, 11):
            dk = f"decile_{d}"
            if dk in da and da[dk]["predictive_ratio"] is not None:
                prs.append(da[dk]["predictive_ratio"])
            else:
                prs.append(np.nan)
        ax.plot(decile_x, prs, marker="o", label=label, color=color, linewidth=2)

    ax.axhline(y=1.0, color="black", linestyle="-", linewidth=0.5)
    ax.axhline(y=0.9, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axhline(y=1.1, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.fill_between(decile_x, 0.9, 1.1, alpha=0.1, color="green")
    ax.set_xlabel("Actual Cost Decile (1=lowest, 10=highest)")
    ax.set_ylabel("Predictive Ratio (predicted/actual)")
    ax.set_title("Predictive Ratio by Actual Cost Decile (SOA Standard)")
    ax.set_xticks(decile_x)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, max(3.0, ax.get_ylim()[1]))

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "figure4_decile_calibration.pdf", bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "figure4_decile_calibration.png", bbox_inches="tight")
    plt.close()
    print(f"Figure 4 saved: {RESULTS_DIR / 'figure4_decile_calibration.pdf'}")

    # Figure 5: Cost bucket calibration
    key_models_buckets = [
        ("XGBoost", "xgboost", "#FF9800"),
        ("Two-Part", "two_part", "#E91E63"),
        ("Chronos", "chronos_zeroshot", "#2196F3"),
        ("PanelFM-Adapter", "panelfm_adapter", "#4CAF50"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bucket_labels = ["$0", "$1-100", "$100-500", "$500-1K", "$1K-5K", "$5K-10K", "$10K+"]
    x_pos = np.arange(len(bucket_labels))
    bar_width = 0.2

    for i, (label, key, color) in enumerate(key_models_buckets):
        if key not in actuarial_results:
            continue
        cb = actuarial_results[key].get("cost_buckets", {})
        prs = []
        for bl in bucket_labels:
            if bl in cb and cb[bl]["predictive_ratio"] is not None:
                prs.append(cb[bl]["predictive_ratio"])
            else:
                prs.append(np.nan)
        ax.bar(x_pos + i * bar_width, prs, bar_width, label=label, color=color, alpha=0.8)

    ax.axhline(y=1.0, color="black", linestyle="-", linewidth=0.5)
    ax.axhline(y=0.9, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axhline(y=1.1, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xticks(x_pos + bar_width * 1.5)
    ax.set_xticklabels(bucket_labels, rotation=45)
    ax.set_ylabel("Predictive Ratio")
    ax.set_title("Calibration by Cost Bucket")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(3.0, ax.get_ylim()[1]))

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "figure5_cost_bucket_calibration.pdf", bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "figure5_cost_bucket_calibration.png", bbox_inches="tight")
    plt.close()
    print(f"Figure 5 saved: {RESULTS_DIR / 'figure5_cost_bucket_calibration.pdf'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Generating publication tables and figures")
    print("=" * 60)

    outcomes, features = load_real_data()
    real_metrics, synthetic_metrics = load_metrics()
    per_patient_data = load_per_patient_data()

    # Table 1: Demographics
    print("\n--- Table 1: Demographics ---")
    table1 = generate_table1(outcomes.copy(), features)
    print(table1.to_string(index=False))

    # Bootstrap CIs for ALL models (from saved per-patient data)
    print("\n--- Bootstrap CIs: All Models ---")
    all_bootstrap = compute_bootstrap_cis_all_models(per_patient_data, n_boot=2000)

    with open(RESULTS_DIR / "bootstrap_cis.json", "w") as f:
        json.dump(all_bootstrap, f, indent=2)
    print(f"\nBootstrap CIs saved: {RESULTS_DIR / 'bootstrap_cis.json'}")

    # Stratified analysis
    print("\n--- Stratified Analysis: Zero vs Non-Zero Cost ---")
    stratified = compute_stratified_analysis(per_patient_data, real_metrics)

    with open(RESULTS_DIR / "stratified_analysis.json", "w") as f:
        json.dump(stratified, f, indent=2)
    print(f"Stratified analysis saved: {RESULTS_DIR / 'stratified_analysis.json'}")

    # Statistical tests
    print("\n--- Statistical Tests: Pairwise Model Comparisons ---")
    stat_tests = compute_statistical_tests(per_patient_data, real_metrics)

    with open(RESULTS_DIR / "statistical_tests.json", "w") as f:
        json.dump(stat_tests, f, indent=2)
    print(f"Statistical tests saved: {RESULTS_DIR / 'statistical_tests.json'}")

    # Actuarial metrics (SOA standard evaluation)
    print("\n--- Actuarial Metrics (SOA Standard) ---")
    actuarial = compute_actuarial_metrics(per_patient_data, real_metrics)

    with open(RESULTS_DIR / "actuarial_metrics.json", "w") as f:
        # Convert numpy types for JSON serialization
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (np.floating, float)):
                return round(float(obj), 6) if np.isfinite(obj) else None
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        json.dump(clean_for_json(actuarial), f, indent=2)
    print(f"Actuarial metrics saved: {RESULTS_DIR / 'actuarial_metrics.json'}")

    # Actuarial summary table
    print("\n--- Actuarial Summary Table ---")
    act_summary = generate_actuarial_summary_table(actuarial, real_metrics)
    print(act_summary.to_string(index=False))

    # Decile analysis table
    print("\n--- Decile Analysis Table ---")
    decile_tbl = generate_decile_table(actuarial)

    # Generate figures
    print("\n--- Generating Figures ---")
    generate_figures(real_metrics, all_bootstrap)

    # Actuarial-specific figures
    print("\n--- Generating Actuarial Figures ---")
    generate_actuarial_figures(actuarial)

    print("\n" + "=" * 60)
    print("All publication tables, figures, and statistical tests generated.")
    print(f"Results in: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
