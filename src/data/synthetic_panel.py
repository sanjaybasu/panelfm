"""
Synthetic panel data generator for PanelFM development and validation.

Generates realistic claims-like panel data with known ground truth:
- Patient-level heterogeneity (random intercepts)
- Temporal dynamics (AR(1) within-patient correlation)
- Seasonality (annual cycle)
- Cross-sectional feature effects (demographics → baseline cost)
- Intervention effects (for causal sub-analysis)
- Unbalanced panels (staggered enrollment/disenrollment)

Use this to validate PanelFM before running on real claims data.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SyntheticPanelConfig:
    """Configuration for synthetic panel data generation."""
    n_patients: int = 10_000
    n_months: int = 24
    seed: int = 42

    # Patient heterogeneity
    random_intercept_std: float = 1.5      # log-scale std of patient baseline cost
    base_monthly_cost_mean: float = 6.0    # log-scale mean (~$400/mo)

    # Temporal dynamics
    ar1_coefficient: float = 0.6           # Within-patient autocorrelation
    innovation_std: float = 0.5            # Shock std (log-scale)

    # Seasonality (flu season peaks in Jan-Feb)
    seasonal_amplitude: float = 0.15       # Fractional amplitude

    # Demographics
    age_mean: float = 45.0
    age_std: float = 15.0
    age_min: float = 18.0
    age_max: float = 89.0
    female_frac: float = 0.58             # Medicaid skews female
    race_probs: dict = field(default_factory=lambda: {
        "White": 0.40, "Black": 0.30, "Hispanic": 0.20, "Other": 0.10
    })
    dual_eligible_frac: float = 0.25

    # Feature effects (log-scale coefficients)
    age_effect: float = 0.015              # Per year of age
    female_effect: float = -0.05           # Females slightly lower cost
    dual_effect: float = 0.4              # Dual-eligible much higher cost
    race_effects: dict = field(default_factory=lambda: {
        "White": 0.0, "Black": 0.05, "Hispanic": -0.05, "Other": 0.0
    })

    # Comorbidity features (latent)
    n_comorbidity_factors: int = 3         # Latent comorbidity dimensions
    comorbidity_effect_range: tuple = (0.1, 0.5)  # Effect range per factor

    # SDOH features
    sdoh_prevalence: float = 0.30          # Fraction with any SDOH barrier
    sdoh_cost_effect: float = 0.2          # Additional log-cost from SDOH

    # Utilization components (correlated with cost)
    ed_rate_per_1000_cost: float = 0.002   # ED visits scale with cost
    ip_rate_per_1000_cost: float = 0.0005  # IP admits scale with cost

    # Intervention (treatment)
    treated_frac: float = 0.15            # Fraction receiving intervention
    treatment_effect: float = -0.10       # Log-cost reduction from treatment
    treatment_start_month_range: tuple = (6, 18)  # When treatment starts

    # Unbalanced panel
    staggered_enrollment: bool = True
    enrollment_start_std_months: float = 3.0  # Std of enrollment start month
    disenrollment_monthly_prob: float = 0.01  # Monthly prob of disenrolling

    # Noise
    observation_noise_std: float = 0.3     # Measurement noise (log-scale)


def generate_synthetic_panel(config: Optional[SyntheticPanelConfig] = None) -> dict:
    """
    Generate synthetic panel data mimicking Medicaid claims.

    Returns:
        dict with keys:
            'outcomes_monthly': pd.DataFrame — panel of monthly outcomes
            'patient_features': pd.DataFrame — cross-sectional patient features
            'ground_truth': dict — true parameters for validation
    """
    if config is None:
        config = SyntheticPanelConfig()

    rng = np.random.default_rng(config.seed)

    # =========================================================================
    # 1. Generate patient-level (cross-sectional) features
    # =========================================================================
    n = config.n_patients

    # Demographics
    age = rng.normal(config.age_mean, config.age_std, n).clip(config.age_min, config.age_max)
    female = rng.binomial(1, config.female_frac, n)

    race_labels = list(config.race_probs.keys())
    race_probs = list(config.race_probs.values())
    race = rng.choice(race_labels, size=n, p=race_probs)

    dual_eligible = rng.binomial(1, config.dual_eligible_frac, n)

    # Latent comorbidity factors (continuous, standard normal)
    comorbidity_factors = rng.standard_normal((n, config.n_comorbidity_factors))
    comorbidity_effects = rng.uniform(
        config.comorbidity_effect_range[0],
        config.comorbidity_effect_range[1],
        config.n_comorbidity_factors
    )

    # SDOH barrier flag
    sdoh_any = rng.binomial(1, config.sdoh_prevalence, n)

    # Random intercept (patient-level baseline, log-scale)
    random_intercept = rng.normal(0, config.random_intercept_std, n)

    # Compute deterministic component of log-cost baseline
    log_cost_baseline = (
        config.base_monthly_cost_mean
        + config.age_effect * (age - config.age_mean)
        + config.female_effect * female
        + config.dual_effect * dual_eligible
        + np.array([config.race_effects.get(r, 0.0) for r in race])
        + comorbidity_factors @ comorbidity_effects
        + config.sdoh_cost_effect * sdoh_any
        + random_intercept
    )

    # Risk score = normalized cross-sectional prediction (what XGBoost would learn)
    risk_score = (log_cost_baseline - log_cost_baseline.mean()) / log_cost_baseline.std()

    # Treatment assignment (correlated with risk — higher risk patients more likely treated)
    treatment_prob = 1 / (1 + np.exp(-(risk_score * 0.5 + rng.normal(0, 1, n))))
    treatment_prob = treatment_prob * config.treated_frac / treatment_prob.mean()
    treatment_prob = treatment_prob.clip(0.01, 0.99)
    treated = rng.binomial(1, treatment_prob)

    treatment_start_month = np.where(
        treated,
        rng.integers(
            config.treatment_start_month_range[0],
            config.treatment_start_month_range[1],
            n
        ),
        config.n_months + 1  # Never starts
    )

    # =========================================================================
    # 2. Generate enrollment windows (unbalanced panel)
    # =========================================================================
    if config.staggered_enrollment:
        enrollment_start = rng.normal(0, config.enrollment_start_std_months, n)
        enrollment_start = np.maximum(0, np.round(enrollment_start)).astype(int)
    else:
        enrollment_start = np.zeros(n, dtype=int)

    # Disenrollment: each month after minimum 6 months, prob of leaving
    enrollment_end = np.full(n, config.n_months)
    for i in range(n):
        for t in range(max(enrollment_start[i] + 6, 6), config.n_months):
            if rng.random() < config.disenrollment_monthly_prob:
                enrollment_end[i] = t
                break

    # =========================================================================
    # 3. Generate monthly time series for each patient
    # =========================================================================
    records = []

    for i in range(n):
        t_start = enrollment_start[i]
        t_end = enrollment_end[i]
        T = t_end - t_start

        if T < config.min_months_implicit():
            continue

        # AR(1) process for within-patient dynamics
        innovations = rng.normal(0, config.innovation_std, T)
        log_cost_ts = np.zeros(T)
        log_cost_ts[0] = log_cost_baseline[i] + innovations[0]

        for t in range(1, T):
            log_cost_ts[t] = (
                (1 - config.ar1_coefficient) * log_cost_baseline[i]
                + config.ar1_coefficient * log_cost_ts[t - 1]
                + innovations[t]
            )

        # Add seasonality (peaks in month 1 = January)
        months_absolute = np.arange(t_start, t_end)
        calendar_month = months_absolute % 12  # 0=Jan, 11=Dec
        seasonal = config.seasonal_amplitude * np.cos(
            2 * np.pi * (calendar_month - 1) / 12  # Peak in Jan-Feb
        )
        log_cost_ts += seasonal

        # Add treatment effect (step function after treatment start)
        if treated[i]:
            treatment_active = months_absolute >= treatment_start_month[i]
            log_cost_ts[treatment_active] += config.treatment_effect

        # Add observation noise
        log_cost_ts += rng.normal(0, config.observation_noise_std, T)

        # Convert to dollar costs (exponentiate)
        total_paid = np.exp(log_cost_ts).clip(0, 100_000)

        # Generate correlated utilization counts
        ed_rate = config.ed_rate_per_1000_cost * total_paid
        ed_visits = rng.poisson(ed_rate.clip(0.01, 10))

        ip_rate = config.ip_rate_per_1000_cost * total_paid
        ip_admits = rng.poisson(ip_rate.clip(0.001, 2))

        # Enrolled days (mostly full months, some partial)
        enrolled_days = np.full(T, 30)
        if T > 0:
            enrolled_days[0] = rng.integers(15, 31)  # Partial first month
            enrolled_days[-1] = rng.integers(15, 31)  # Partial last month

        for t_idx in range(T):
            month_abs = months_absolute[t_idx]
            records.append({
                "person_id": f"SYN_{i:06d}",
                "month_idx": month_abs,
                "month_year": pd.Timestamp("2023-01-01") + pd.DateOffset(months=int(month_abs)),
                "total_paid": round(total_paid[t_idx], 2),
                "emergency_department_ct": int(ed_visits[t_idx]),
                "acute_inpatient_ct": int(ip_admits[t_idx]),
                "enrolled_days": int(enrolled_days[t_idx]),
                "treatment_active": bool(
                    treated[i] and month_abs >= treatment_start_month[i]
                ),
            })

    outcomes_monthly = pd.DataFrame(records)

    # =========================================================================
    # 4. Build cross-sectional patient features table
    # =========================================================================
    patient_ids = [f"SYN_{i:06d}" for i in range(n)]
    patient_features = pd.DataFrame({
        "person_id": patient_ids,
        "age": age,
        "female": female,
        "race": race,
        "dual_eligible": dual_eligible,
        "sdoh_any": sdoh_any,
        "risk_score_true": risk_score,
        "treated": treated,
        "treatment_start_month": treatment_start_month,
        "enrollment_start": enrollment_start,
        "enrollment_end": enrollment_end,
    })

    # Add comorbidity factors as observable features
    for k in range(config.n_comorbidity_factors):
        patient_features[f"comorbidity_factor_{k}"] = comorbidity_factors[:, k]

    # Filter to patients actually in outcomes
    valid_ids = set(outcomes_monthly["person_id"].unique())
    patient_features = patient_features[
        patient_features["person_id"].isin(valid_ids)
    ].reset_index(drop=True)

    # =========================================================================
    # 5. Ground truth for validation
    # =========================================================================
    ground_truth = {
        "ar1_coefficient": config.ar1_coefficient,
        "treatment_effect_log": config.treatment_effect,
        "age_effect_log": config.age_effect,
        "dual_effect_log": config.dual_effect,
        "random_intercept_std": config.random_intercept_std,
        "comorbidity_effects": comorbidity_effects.tolist(),
        "seasonal_amplitude": config.seasonal_amplitude,
    }

    return {
        "outcomes_monthly": outcomes_monthly,
        "patient_features": patient_features,
        "ground_truth": ground_truth,
    }


# Add helper to config (can't be in dataclass default)
def _min_months_implicit(self):
    return 6

SyntheticPanelConfig.min_months_implicit = _min_months_implicit


def validate_synthetic_vs_real(synthetic_outcomes, real_outcomes, target_col="total_paid"):
    """
    Compare marginal distributions and autocorrelation between synthetic and real data.
    Returns a dict of diagnostic statistics.
    """
    diagnostics = {}

    for label, df in [("synthetic", synthetic_outcomes), ("real", real_outcomes)]:
        vals = df[target_col].dropna()
        diagnostics[f"{label}_mean"] = vals.mean()
        diagnostics[f"{label}_median"] = vals.median()
        diagnostics[f"{label}_std"] = vals.std()
        diagnostics[f"{label}_skew"] = vals.skew()
        diagnostics[f"{label}_p95"] = vals.quantile(0.95)
        diagnostics[f"{label}_p99"] = vals.quantile(0.99)
        diagnostics[f"{label}_zero_frac"] = (vals == 0).mean()

        # Per-patient autocorrelation
        acorrs = []
        for pid, grp in df.groupby("person_id"):
            ts = grp.sort_values("month_year")[target_col].values
            if len(ts) >= 4:
                acorr = np.corrcoef(ts[:-1], ts[1:])[0, 1]
                if np.isfinite(acorr):
                    acorrs.append(acorr)
        diagnostics[f"{label}_median_autocorr"] = np.median(acorrs) if acorrs else np.nan

        # Between-patient variance share
        grand_mean = vals.mean()
        patient_means = df.groupby("person_id")[target_col].mean()
        between_var = patient_means.var()
        total_var = vals.var()
        diagnostics[f"{label}_between_var_share"] = between_var / total_var if total_var > 0 else np.nan

    return diagnostics


if __name__ == "__main__":
    print("Generating synthetic panel data...")
    config = SyntheticPanelConfig(n_patients=10_000, n_months=24, seed=42)
    data = generate_synthetic_panel(config)

    outcomes = data["outcomes_monthly"]
    features = data["patient_features"]
    truth = data["ground_truth"]

    print(f"\nOutcomes: {outcomes.shape[0]:,} patient-months, "
          f"{outcomes['person_id'].nunique():,} patients")
    print(f"Features: {features.shape[0]:,} patients, {features.shape[1]} columns")
    print(f"\nOutcome summary:")
    print(outcomes[["total_paid", "emergency_department_ct", "acute_inpatient_ct"]].describe())
    print(f"\nGround truth: {truth}")
