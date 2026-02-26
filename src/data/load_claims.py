"""
Load and prepare real Waymark claims data for PanelFM experiments.

Joins outcomes_monthly with member_attributes and eligibility,
engineers cross-sectional features, and constructs per-patient time series.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import yaml
import warnings


def load_config(config_path: str = "configs/data_config.yaml") -> dict:
    """Load data configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_outcomes(data_root: str, filename: str) -> pd.DataFrame:
    """Load monthly outcomes panel data."""
    path = Path(data_root) / filename
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["month_year"])

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()
    return df


def load_attributes(data_root: str, filename: str) -> pd.DataFrame:
    """Load member attributes (demographics)."""
    path = Path(data_root) / filename
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    return df


def load_eligibility(data_root: str, filename: str) -> pd.DataFrame:
    """Load eligibility/enrollment data."""
    path = Path(data_root) / filename
    df = pd.read_csv(path, parse_dates=["enrollment_start_date", "enrollment_end_date"])
    df.columns = df.columns.str.lower().str.strip()
    return df


def compute_age(birth_date_series: pd.Series, reference_date: str = "2024-01-01") -> pd.Series:
    """Compute age from birth_date column."""
    birth_dates = pd.to_datetime(birth_date_series, errors="coerce")
    ref = pd.Timestamp(reference_date)
    age = (ref - birth_dates).dt.days / 365.25
    return age.clip(0, 120)


def build_cross_sectional_features(
    outcomes: pd.DataFrame,
    attributes: pd.DataFrame,
    eligibility: pd.DataFrame,
    lookback_end: pd.Timestamp,
    lookback_months: int = 12,
    id_col: str = "person_id",
) -> pd.DataFrame:
    """
    Build cross-sectional feature matrix from lookback window.

    Features:
    - Demographics (age, sex, race, dual-eligible)
    - Utilization summary stats (mean/max/std of costs, ED counts, IP counts)
    - Trend features (slope of cost over lookback)
    - Intervention flags
    """
    lookback_start = lookback_end - pd.DateOffset(months=lookback_months)

    # Filter outcomes to lookback window
    mask = (outcomes["month_year"] >= lookback_start) & (outcomes["month_year"] < lookback_end)
    lb = outcomes[mask].copy()

    # Fill NaN costs/counts with 0 (enrolled month with no claims)
    for col in ["total_paid", "emergency_department_ct", "acute_inpatient_ct"]:
        if col in lb.columns:
            lb[col] = lb[col].fillna(0)

    # --- Utilization summary statistics per patient ---
    agg = lb.groupby(id_col).agg(
        n_months=("month_year", "count"),
        mean_total_paid=("total_paid", "mean"),
        max_total_paid=("total_paid", "max"),
        std_total_paid=("total_paid", "std"),
        mean_ed_visits=("emergency_department_ct", "mean"),
        total_ed_visits=("emergency_department_ct", "sum"),
        mean_ip_admits=("acute_inpatient_ct", "mean"),
        total_ip_admits=("acute_inpatient_ct", "sum"),
        mean_enrolled_days=("enrolled_days", "mean"),
        total_enrolled_days=("enrolled_days", "sum"),
    ).reset_index()

    # Fill NaN std (patients with 1 month)
    agg["std_total_paid"] = agg["std_total_paid"].fillna(0)

    # Months with any ED / IP
    months_ed = lb[lb["emergency_department_ct"] > 0].groupby(id_col).size()
    months_ip = lb[lb["acute_inpatient_ct"] > 0].groupby(id_col).size()
    agg["months_with_any_ed"] = agg[id_col].map(months_ed).fillna(0).astype(int)
    agg["months_with_any_ip"] = agg[id_col].map(months_ip).fillna(0).astype(int)

    # Cost trend slope (OLS on month index vs log1p(cost))
    def _cost_slope(group):
        if len(group) < 3:
            return 0.0
        group = group.sort_values("month_year")
        x = np.arange(len(group), dtype=float)
        y = np.log1p(group["total_paid"].values)
        x_centered = x - x.mean()
        denom = (x_centered ** 2).sum()
        if denom == 0:
            return 0.0
        return float((x_centered * (y - y.mean())).sum() / denom)

    slopes = lb.groupby(id_col).apply(_cost_slope, include_groups=False)
    slopes.name = "cost_trend_slope"
    agg = agg.merge(slopes.reset_index(), on=id_col, how="left")
    agg["cost_trend_slope"] = agg["cost_trend_slope"].fillna(0)

    # --- Intervention flags (if columns exist) ---
    for col in ["ever_targeted", "ever_activated"]:
        if col in outcomes.columns:
            flags = outcomes.groupby(id_col)[col].max().reset_index()
            agg = agg.merge(flags, on=id_col, how="left")
            agg[col] = agg[col].fillna(0).astype(int)

    # --- Demographics from attributes ---
    if attributes is not None and len(attributes) > 0:
        # Determine join key
        attr_id = "member_id" if "member_id" in attributes.columns else id_col
        demo_cols = [attr_id]
        rename_map = {}
        if attr_id != id_col:
            rename_map[attr_id] = id_col

        for col in ["gender", "race", "ethnicity", "birth_date", "risk_score"]:
            if col in attributes.columns:
                demo_cols.append(col)

        demo = attributes[demo_cols].drop_duplicates(subset=[attr_id]).copy()
        if rename_map:
            demo = demo.rename(columns=rename_map)

        if "birth_date" in demo.columns:
            demo["age"] = compute_age(demo["birth_date"])
            demo = demo.drop(columns=["birth_date"])

        if "gender" in demo.columns:
            demo["female"] = (demo["gender"].str.upper().isin(["F", "FEMALE"])).astype(int)
            demo = demo.drop(columns=["gender"])

        # Encode race as dummies
        if "race" in demo.columns:
            race_dummies = pd.get_dummies(demo["race"], prefix="race", drop_first=True)
            demo = pd.concat([demo.drop(columns=["race"]), race_dummies], axis=1)

        if "ethnicity" in demo.columns:
            demo["hispanic"] = (
                demo["ethnicity"].str.upper().str.contains("HISPANIC", na=False)
            ).astype(int)
            demo = demo.drop(columns=["ethnicity"])

        agg = agg.merge(demo, on=id_col, how="left")

    # --- Dual-eligible from eligibility ---
    if eligibility is not None and len(eligibility) > 0:
        elig_id = id_col if id_col in eligibility.columns else "source_member_id"
        if "dual_eligible" in eligibility.columns:
            dual = eligibility.groupby(elig_id)["dual_eligible"].max().reset_index()
            if elig_id != id_col:
                dual = dual.rename(columns={elig_id: id_col})
            agg = agg.merge(dual, on=id_col, how="left")

    return agg


def build_patient_time_series(
    outcomes: pd.DataFrame,
    id_col: str = "person_id",
    time_col: str = "month_year",
    target_col: str = "total_paid",
    min_months: int = 6,
) -> dict:
    """
    Construct per-patient time series dict for TimesFM.

    Returns:
        dict mapping person_id → np.ndarray of monthly target values (chronological).
        Also returns a metadata dict with start dates and lengths.
    """
    series = {}
    metadata = {}

    for pid, group in outcomes.groupby(id_col):
        group = group.sort_values(time_col)
        values = group[target_col].fillna(0).values
        if len(values) < min_months:
            continue

        series[pid] = values.astype(np.float32)
        metadata[pid] = {
            "start_date": group[time_col].iloc[0],
            "n_months": len(values),
            "mean_enrolled_days": group["enrolled_days"].mean() if "enrolled_days" in group else 30,
        }

    return series, metadata


def temporal_train_val_test_split(
    outcomes: pd.DataFrame,
    time_col: str = "month_year",
    val_months: int = 3,
    test_months: int = 3,
) -> dict:
    """
    Split panel data by time: train | validation | test.

    Returns dict with 'train_end', 'val_end', 'test_end' timestamps
    and filtered DataFrames.
    """
    dates = sorted(outcomes[time_col].unique())
    n_dates = len(dates)

    test_start_idx = n_dates - test_months
    val_start_idx = test_start_idx - val_months

    train_end = dates[val_start_idx - 1]
    val_end = dates[test_start_idx - 1]
    test_end = dates[-1]

    return {
        "train": outcomes[outcomes[time_col] <= train_end],
        "val": outcomes[
            (outcomes[time_col] > train_end) & (outcomes[time_col] <= val_end)
        ],
        "test": outcomes[outcomes[time_col] > val_end],
        "train_end": train_end,
        "val_end": val_end,
        "test_end": test_end,
    }


def load_and_prepare_all(config_path: str = "configs/data_config.yaml") -> dict:
    """
    End-to-end data loading and preparation.

    Returns dict with:
        'outcomes': full outcomes DataFrame
        'features': cross-sectional features DataFrame
        'series': per-patient time series dict
        'splits': temporal train/val/test split
    """
    config = load_config(config_path)
    data_root = config["data_root"]
    tables = config["tables"]
    temporal = config["temporal"]

    # Load raw tables
    outcomes = load_outcomes(data_root, tables["outcomes"])
    attributes = load_attributes(data_root, tables["attributes"])
    eligibility = load_eligibility(data_root, tables["eligibility"])

    # Apply panel filters
    filters = config["filters"]
    if "enrolled_days" in outcomes.columns:
        outcomes = outcomes[outcomes["enrolled_days"] >= filters["min_enrolled_days_per_month"]]

    patient_months = outcomes.groupby("person_id").size()
    valid_patients = patient_months[patient_months >= filters["min_total_months"]].index
    outcomes = outcomes[outcomes["person_id"].isin(valid_patients)]

    print(f"After filtering: {outcomes.shape[0]:,} patient-months, "
          f"{outcomes['person_id'].nunique():,} patients")

    # Temporal split
    splits = temporal_train_val_test_split(
        outcomes,
        val_months=temporal["forecast_horizons"][0],
        test_months=temporal["forecast_horizons"][-1],
    )

    # Cross-sectional features (from training period only)
    features = build_cross_sectional_features(
        splits["train"],
        attributes,
        eligibility,
        lookback_end=splits["train_end"],
        lookback_months=temporal["lookback_months"],
    )

    # Per-patient time series (training period only for model fitting)
    series, series_meta = build_patient_time_series(
        splits["train"],
        min_months=temporal["min_history_months"],
    )

    print(f"Cross-sectional features: {features.shape}")
    print(f"Patient time series: {len(series)} patients")

    return {
        "outcomes": outcomes,
        "features": features,
        "series": series,
        "series_meta": series_meta,
        "splits": splits,
        "config": config,
    }
