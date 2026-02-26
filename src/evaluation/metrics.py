"""
Evaluation metrics for PanelFM.

Includes standard forecasting metrics, classification metrics,
and the novel panel decomposition (between/within R²).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    brier_score_loss,
)
from typing import Dict, Optional


# =============================================================================
# Regression Metrics
# =============================================================================

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_absolute_error(y_true, y_pred)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    """Mean Absolute Percentage Error. epsilon avoids division by zero."""
    return np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))


def weighted_quantile_loss(
    y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.9
) -> float:
    """Quantile loss (pinball loss) at a given quantile."""
    residual = y_true - y_pred
    return np.mean(np.maximum(quantile * residual, (quantile - 1) * residual))


def crps_gaussian(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
) -> float:
    """
    Continuous Ranked Probability Score for Gaussian predictive distribution.
    Lower is better.
    """
    from scipy.stats import norm

    z = (y_true - y_pred_mean) / np.maximum(y_pred_std, 1e-6)
    crps = y_pred_std * (
        z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)
    )
    return np.mean(crps)


# =============================================================================
# Classification Metrics
# =============================================================================

def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, y_score)


def sensitivity_at_specificity(
    y_true: np.ndarray, y_score: np.ndarray, target_specificity: float = 0.90
) -> float:
    """Sensitivity (recall) at a given specificity threshold."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # Specificity = 1 - FPR
    specificities = 1 - fpr
    # Find threshold closest to target specificity
    idx = np.argmin(np.abs(specificities - target_specificity))
    return tpr[idx]


def ppv_at_top_k_percent(
    y_true: np.ndarray, y_score: np.ndarray, top_k_pct: float = 0.05
) -> float:
    """
    Positive Predictive Value among top-k% highest-risk patients.
    This is the metric that matters for care management targeting.
    """
    n = len(y_true)
    k = max(1, int(n * top_k_pct))
    top_indices = np.argsort(y_score)[-k:]
    return np.mean(y_true[top_indices])


def net_reclassification_index(
    y_true: np.ndarray,
    y_score_new: np.ndarray,
    y_score_ref: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Net Reclassification Index (NRI) of new model vs. reference.

    Returns dict with:
        nri_events: NRI among events (correctly reclassified up)
        nri_nonevents: NRI among non-events (correctly reclassified down)
        nri_total: Total NRI
    """
    class_new = (y_score_new >= threshold).astype(int)
    class_ref = (y_score_ref >= threshold).astype(int)

    events = y_true == 1
    nonevents = y_true == 0

    # Among events: fraction reclassified up minus fraction reclassified down
    if events.sum() > 0:
        up_events = ((class_new == 1) & (class_ref == 0) & events).sum() / events.sum()
        down_events = ((class_new == 0) & (class_ref == 1) & events).sum() / events.sum()
        nri_events = up_events - down_events
    else:
        nri_events = 0.0

    # Among non-events: fraction reclassified down minus fraction reclassified up
    if nonevents.sum() > 0:
        down_nonevents = ((class_new == 0) & (class_ref == 1) & nonevents).sum() / nonevents.sum()
        up_nonevents = ((class_new == 1) & (class_ref == 0) & nonevents).sum() / nonevents.sum()
        nri_nonevents = down_nonevents - up_nonevents
    else:
        nri_nonevents = 0.0

    return {
        "nri_events": nri_events,
        "nri_nonevents": nri_nonevents,
        "nri_total": nri_events + nri_nonevents,
    }


# =============================================================================
# Panel Decomposition (Novel)
# =============================================================================

def panel_r2_decomposition(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    id_col: str = "person_id",
    value_col: str = "total_paid",
    pred_col: str = "predicted",
) -> dict:
    """
    Decompose prediction R² into between-patient and within-patient components.

    This is analogous to the between/within decomposition in panel econometrics:
      Total R² = Between R² + Within R²

    Between R²: How well does the model rank patients by their average cost?
    Within R²: How well does the model track each patient's trajectory over time?

    This directly addresses the paper's core claim: PanelFM improves on BOTH
    dimensions simultaneously, while XGBoost wins only on between and TimesFM
    wins only on within.
    """
    df = y_true.merge(y_pred, on=[id_col, "month_year"], suffixes=("", "_pred"))

    grand_mean_true = df[value_col].mean()
    grand_mean_pred = df[pred_col].mean()

    # Total R²
    ss_total = ((df[value_col] - grand_mean_true) ** 2).sum()
    ss_residual = ((df[value_col] - df[pred_col]) ** 2).sum()
    total_r2 = 1 - ss_residual / ss_total if ss_total > 0 else 0

    # Between R²: R² on patient-level means
    patient_means = df.groupby(id_col).agg(
        mean_true=(value_col, "mean"),
        mean_pred=(pred_col, "mean"),
    ).reset_index()

    ss_between_total = ((patient_means["mean_true"] - grand_mean_true) ** 2).sum()
    ss_between_residual = ((patient_means["mean_true"] - patient_means["mean_pred"]) ** 2).sum()
    between_r2 = 1 - ss_between_residual / ss_between_total if ss_between_total > 0 else 0

    # Within R²: R² on demeaned values (patient fixed-effects removed)
    patient_mean_map_true = df.groupby(id_col)[value_col].transform("mean")
    patient_mean_map_pred = df.groupby(id_col)[pred_col].transform("mean")

    within_true = df[value_col] - patient_mean_map_true
    within_pred = df[pred_col] - patient_mean_map_pred

    ss_within_total = (within_true ** 2).sum()
    ss_within_residual = ((within_true - within_pred) ** 2).sum()
    within_r2 = 1 - ss_within_residual / ss_within_total if ss_within_total > 0 else 0

    # Variance decomposition
    between_var_share = ss_between_total / ss_total if ss_total > 0 else 0

    return {
        "total_r2": total_r2,
        "between_r2": between_r2,
        "within_r2": within_r2,
        "between_var_share": between_var_share,
        "n_patients": patient_means.shape[0],
        "n_observations": df.shape[0],
    }


# =============================================================================
# Equity Audit
# =============================================================================

def equity_stratified_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_labels: np.ndarray,
    metric_fn=auroc,
) -> dict:
    """
    Compute a metric stratified by demographic group.

    Returns dict mapping group_label → metric_value.
    Also computes disparity ratio (min/max across groups).
    """
    results = {}
    group_values = []

    for group in np.unique(group_labels):
        mask = group_labels == group
        if mask.sum() < 20:
            continue
        y_true_g = y_true[mask]
        y_score_g = y_score[mask]

        val = metric_fn(y_true_g, y_score_g)
        results[f"group_{group}"] = val
        if np.isfinite(val):
            group_values.append(val)

    if len(group_values) >= 2:
        results["disparity_ratio"] = min(group_values) / max(group_values)
        results["disparity_range"] = max(group_values) - min(group_values)
    else:
        results["disparity_ratio"] = np.nan
        results["disparity_range"] = np.nan

    return results


def calibration_by_group(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_labels: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute calibration slope and intercept by demographic group.
    A well-calibrated model has slope ≈ 1 and intercept ≈ 0 for all groups.
    """
    from sklearn.linear_model import LogisticRegression

    results = {}

    for group in np.unique(group_labels):
        mask = group_labels == group
        if mask.sum() < 50:
            continue
        y_true_g = y_true[mask]
        y_score_g = y_score[mask]

        if len(np.unique(y_true_g)) < 2:
            continue

        # Fit logistic recalibration
        lr = LogisticRegression(max_iter=1000)
        lr.fit(y_score_g.reshape(-1, 1), y_true_g)

        results[f"group_{group}_cal_slope"] = lr.coef_[0][0]
        results[f"group_{group}_cal_intercept"] = lr.intercept_[0]
        results[f"group_{group}_brier"] = brier_score_loss(y_true_g, y_score_g)

    return results


# =============================================================================
# Aggregate evaluation
# =============================================================================

# =============================================================================
# Actuarial Metrics (SOA standard evaluation)
# =============================================================================

def calibrate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calibrate predictions so mean(predicted) = mean(actual).

    This is standard actuarial practice: risk adjustment models are always
    calibrated before evaluation. Without calibration, R² can be negative
    even for models with good discrimination, because the scale is wrong.

    The calibration factor is simply: actual_mean / pred_mean.
    """
    pred_mean = np.mean(y_pred)
    actual_mean = np.mean(y_true)
    if pred_mean == 0:
        return y_pred
    return y_pred * (actual_mean / pred_mean)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Individual-level R² (coefficient of determination).

    The primary metric in actuarial cost model evaluation.
    Typical benchmarks (Medicaid prospective):
      - Demographic-only: 1-3%
      - CDPS: 8-24% (depending on eligibility category)
      - Best concurrent (MARA Dx+Rx): 55-67%
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def r_squared_calibrated(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² after calibrating predictions to match actual mean.

    This is the R² metric used in SOA studies and CMS risk adjustment.
    Models are always calibrated (budget-neutral) before R² is computed.
    """
    y_pred_cal = calibrate_predictions(y_true, y_pred)
    return r_squared(y_true, y_pred_cal)


def predictive_ratio(y_true: np.ndarray, y_pred: np.ndarray,
                     groups: np.ndarray = None) -> dict:
    """Predictive ratio: mean(predicted) / mean(actual) by subgroup.

    The central calibration metric in actuarial practice (CMS, SOA).
    PR = 1.00 is perfect. Acceptable range: 0.90 - 1.10.

    If groups is None, returns overall PR. Otherwise returns PR per group.
    """
    results = {}
    actual_mean = float(np.mean(y_true))
    pred_mean = float(np.mean(y_pred))
    results["overall"] = pred_mean / actual_mean if actual_mean > 0 else np.nan

    if groups is not None:
        for g in np.unique(groups):
            mask = groups == g
            a_mean = float(np.mean(y_true[mask]))
            p_mean = float(np.mean(y_pred[mask]))
            results[str(g)] = p_mean / a_mean if a_mean > 0 else np.nan

    return results


def decile_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Predictive ratios by actual cost decile (SOA standard).

    Segments population into deciles of actual cost and computes
    predicted/actual ratio for each. Also includes top 5% and top 1%.
    """
    results = {}
    n = len(y_true)

    # Deciles
    decile_labels = np.zeros(n, dtype=int)
    sorted_idx = np.argsort(y_true)
    for i in range(10):
        start = int(n * i / 10)
        end = int(n * (i + 1) / 10)
        decile_labels[sorted_idx[start:end]] = i + 1

    for d in range(1, 11):
        mask = decile_labels == d
        if mask.sum() == 0:
            continue
        actual_mean = float(np.mean(y_true[mask]))
        pred_mean = float(np.mean(y_pred[mask]))
        pr = pred_mean / actual_mean if actual_mean > 0 else np.nan
        results[f"decile_{d}"] = {
            "n": int(mask.sum()),
            "actual_mean": round(actual_mean, 2),
            "pred_mean": round(pred_mean, 2),
            "predictive_ratio": round(pr, 4) if np.isfinite(pr) else None,
        }

    # Top percentiles (5%, 1%)
    for pct_label, pct in [("top_10pct", 0.90), ("top_5pct", 0.95), ("top_1pct", 0.99)]:
        threshold = np.percentile(y_true, pct * 100)
        mask = y_true >= threshold
        if mask.sum() == 0:
            continue
        actual_mean = float(np.mean(y_true[mask]))
        pred_mean = float(np.mean(y_pred[mask]))
        pr = pred_mean / actual_mean if actual_mean > 0 else np.nan
        results[pct_label] = {
            "n": int(mask.sum()),
            "actual_mean": round(actual_mean, 2),
            "pred_mean": round(pred_mean, 2),
            "predictive_ratio": round(pr, 4) if np.isfinite(pr) else None,
            "threshold": round(float(threshold), 2),
        }

    return results


def cost_bucket_calibration(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calibration by cost bucket (actuarial standard).

    Buckets: $0, $1-100, $100-500, $500-1000, $1000-5000, $5000-10000, $10000+
    """
    buckets = [
        ("$0", 0, 0),
        ("$1-100", 1, 100),
        ("$100-500", 100, 500),
        ("$500-1K", 500, 1000),
        ("$1K-5K", 1000, 5000),
        ("$5K-10K", 5000, 10000),
        ("$10K+", 10000, np.inf),
    ]
    results = {}
    for label, lo, hi in buckets:
        if lo == 0 and hi == 0:
            mask = y_true == 0
        elif hi == np.inf:
            mask = y_true >= lo
        else:
            mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() == 0:
            continue
        actual_mean = float(np.mean(y_true[mask]))
        pred_mean = float(np.mean(y_pred[mask]))
        pr = pred_mean / actual_mean if actual_mean > 0 else np.nan
        results[label] = {
            "n": int(mask.sum()),
            "pct": round(100 * mask.mean(), 1),
            "actual_mean": round(actual_mean, 2),
            "pred_mean": round(pred_mean, 2),
            "predictive_ratio": round(pr, 4) if np.isfinite(pr) else None,
            "mae": round(float(np.mean(np.abs(y_true[mask] - y_pred[mask]))), 2),
        }
    return results


def censored_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     cap: float = 250000) -> dict:
    """Evaluate with cost censoring at cap (SOA standard: $250K).

    Censoring reduces the influence of extreme outliers, which are
    inherently unpredictable. MARA R² improved from 55% to 67% with
    $250K censoring in the SOA 2016 study.
    """
    y_true_c = np.minimum(y_true, cap)
    y_pred_c = np.minimum(y_pred, cap)
    n_censored = int(np.sum(y_true > cap))
    return {
        "r_squared_censored": r_squared(y_true_c, y_pred_c),
        "mae_censored": float(np.mean(np.abs(y_true_c - y_pred_c))),
        "rmse_censored": float(np.sqrt(np.mean((y_true_c - y_pred_c) ** 2))),
        "predictive_ratio_censored": (
            float(np.mean(y_pred_c) / np.mean(y_true_c))
            if np.mean(y_true_c) > 0 else np.nan
        ),
        "n_censored": n_censored,
        "pct_censored": round(100 * n_censored / len(y_true), 3),
        "cap": cap,
    }


def high_cost_identification(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evaluate high-cost patient identification at multiple thresholds.

    SOA standard: evaluate at top 1%, 5%, 10% of actual cost.
    Reports sensitivity (how many true high-cost are flagged) and
    PPV (what fraction of flagged are truly high-cost).
    """
    results = {}
    for label, pct in [("top_1pct", 0.01), ("top_5pct", 0.05), ("top_10pct", 0.10)]:
        threshold_actual = np.percentile(y_true, (1 - pct) * 100)
        threshold_pred = np.percentile(y_pred, (1 - pct) * 100)

        true_high = y_true >= threshold_actual
        pred_high = y_pred >= threshold_pred

        tp = (true_high & pred_high).sum()
        fp = (~true_high & pred_high).sum()
        fn = (true_high & ~pred_high).sum()
        tn = (~true_high & ~pred_high).sum()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Lift: how much better than random at identifying high-cost
        lift = ppv / pct if pct > 0 else np.nan

        results[label] = {
            "threshold_actual": round(float(threshold_actual), 2),
            "n_true_high": int(true_high.sum()),
            "sensitivity": round(float(sensitivity), 4),
            "ppv": round(float(ppv), 4),
            "specificity": round(float(specificity), 4),
            "lift": round(float(lift), 2),
            "c_statistic": round(float(roc_auc_score(
                true_high.astype(int),
                y_pred
            )), 4) if len(np.unique(true_high)) > 1 else np.nan,
        }
    return results


def calibration_slope_intercept(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10,
) -> dict:
    """
    Compute calibration slope, intercept, and expected calibration error (ECE).

    Calibration slope ~1.0 and intercept ~0.0 indicate good calibration.
    ECE is the mean absolute difference between predicted and observed event rates.
    """
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(y_true)) < 2 or len(y_score) < 50:
        return {"cal_slope": np.nan, "cal_intercept": np.nan, "ece": np.nan}

    # Logistic calibration curve
    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr.fit(y_score.reshape(-1, 1), y_true)
    cal_slope = float(lr.coef_[0][0])
    cal_intercept = float(lr.intercept_[0])

    # Expected Calibration Error (binned)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_score >= bin_edges[i]) & (y_score < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_pred_mean = y_score[mask].mean()
        bin_true_mean = y_true[mask].mean()
        ece += mask.sum() * np.abs(bin_pred_mean - bin_true_mean)
    ece /= len(y_true)

    return {"cal_slope": cal_slope, "cal_intercept": cal_intercept, "ece": ece}


def ppv_npv_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5,
) -> dict:
    """
    Compute PPV (precision), NPV, sensitivity, specificity at a given threshold.
    Also computes Number Needed to Evaluate (NNE = 1/PPV).
    """
    y_pred = (y_score >= threshold).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    nne = 1 / ppv if ppv > 0 else np.nan  # Number Needed to Evaluate

    return {
        "ppv": ppv,
        "npv": npv,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "nne": nne,
        "n_flagged": int(y_pred.sum()),
        "prevalence": float(y_true.mean()),
    }


def evaluate_all_metrics(
    y_true_reg: np.ndarray,
    y_pred_reg: np.ndarray,
    y_true_cls: np.ndarray,
    y_score_cls: np.ndarray,
    y_score_ref: Optional[np.ndarray] = None,
    y_pred_std: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute all metrics for a single model. Returns a flat dict.

    Includes clinical relevance metrics: calibration, PPV/NPV at
    clinically meaningful thresholds, and Number Needed to Evaluate.
    """
    results = {}

    # Regression
    results["mae"] = mae(y_true_reg, y_pred_reg)
    results["rmse"] = rmse(y_true_reg, y_pred_reg)
    results["mape"] = mape(y_true_reg, y_pred_reg)

    for q in [0.5, 0.9, 0.99]:
        results[f"quantile_loss_{int(q*100)}"] = weighted_quantile_loss(
            y_true_reg, y_pred_reg, quantile=q
        )

    if y_pred_std is not None:
        results["crps"] = crps_gaussian(y_true_reg, y_pred_reg, y_pred_std)

    # Classification
    results["auroc"] = auroc(y_true_cls, y_score_cls)
    results["auprc"] = auprc(y_true_cls, y_score_cls)
    results["sensitivity_at_90spec"] = sensitivity_at_specificity(y_true_cls, y_score_cls, 0.90)
    results["ppv_at_top5pct"] = ppv_at_top_k_percent(y_true_cls, y_score_cls, 0.05)

    # Calibration
    cal = calibration_slope_intercept(y_true_cls, y_score_cls)
    results["cal_slope"] = cal["cal_slope"]
    results["cal_intercept"] = cal["cal_intercept"]
    results["ece"] = cal["ece"]

    # PPV/NPV at clinically meaningful thresholds
    # Threshold at score that captures top 10% (matches high_cost_flag definition)
    top10_threshold = np.percentile(y_score_cls, 90)
    ppv_npv_10 = ppv_npv_at_threshold(y_true_cls, y_score_cls, threshold=top10_threshold)
    results["ppv_at_top10pct"] = ppv_npv_10["ppv"]
    results["npv_at_top10pct"] = ppv_npv_10["npv"]
    results["nne_at_top10pct"] = ppv_npv_10["nne"]

    # Also at top 5% (more aggressive targeting)
    top5_threshold = np.percentile(y_score_cls, 95)
    ppv_npv_5 = ppv_npv_at_threshold(y_true_cls, y_score_cls, threshold=top5_threshold)
    results["ppv_at_top5pct_threshold"] = ppv_npv_5["ppv"]
    results["npv_at_top5pct_threshold"] = ppv_npv_5["npv"]
    results["nne_at_top5pct_threshold"] = ppv_npv_5["nne"]
    results["sensitivity_at_top5pct"] = ppv_npv_5["sensitivity"]

    # Brier score (calibration-in-the-large)
    results["brier_score"] = float(brier_score_loss(y_true_cls, y_score_cls)) if len(np.unique(y_true_cls)) >= 2 else np.nan

    # NRI vs reference (XGBoost)
    if y_score_ref is not None:
        nri = net_reclassification_index(y_true_cls, y_score_cls, y_score_ref)
        results.update(nri)

    # Actuarial metrics (SOA standard)
    results["r_squared"] = r_squared(y_true_reg, y_pred_reg)
    results["r_squared_calibrated"] = r_squared_calibrated(y_true_reg, y_pred_reg)
    pr = predictive_ratio(y_true_reg, y_pred_reg)
    results["predictive_ratio"] = pr["overall"]

    # Cost-censored evaluation ($250K cap, SOA standard)
    cens = censored_metrics(y_true_reg, y_pred_reg, cap=250000)
    results["r_squared_censored_250k"] = cens["r_squared_censored"]
    results["r_squared_calibrated_censored_250k"] = r_squared_calibrated(
        np.minimum(y_true_reg, 250000), np.minimum(y_pred_reg, 250000)
    )
    results["mae_censored_250k"] = cens["mae_censored"]
    results["predictive_ratio_censored_250k"] = cens["predictive_ratio_censored"]
    results["n_censored_250k"] = cens["n_censored"]

    return results
