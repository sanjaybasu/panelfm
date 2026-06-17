# PanelFM

Time series foundation models for per-member per-month (PMPM) cost prediction in Medicaid managed care, evaluated against actuarial risk adjustment benchmarks.

## Overview

This repository contains the analysis code for:

> **Time Series Foundation Models for Per-Member Per-Month Cost Prediction: Prospective and Concurrent Evaluation Against Actuarial Benchmarks in Medicaid Claims**
>
> Sanjay Basu, Aaron Baum, Sadiq Patel

The study evaluates 22 models across five categories — cross-sectional ML, classical time-series baselines, a pretrained foundation model (Chronos-T5-Small), panel-conditioned foundation-model variants (PanelFM), and hybrid models — on deidentified Medicaid managed care claims data from 122,849 members contributing 2.4 million patient-months.

This repository contains **code only**. Manuscript artifacts (figures, tables, paper text, supplementary appendix, response letters) and computed outputs (results JSONs, derived metrics, figure files) are not committed; they regenerate from the scripts below using the source data described in *Data setup*.

## Repository structure

```
panelfm/
├── configs/
│   ├── data_config.yaml         # Data source paths, column mappings, temporal settings
│   ├── experiment_config.yaml   # Experiment parameters
│   └── model_config.yaml        # Model hyperparameters
├── scripts/
│   ├── run_experiment.py                # Main pipeline (all 22 models, synthetic + real)
│   ├── generate_publication_tables.py   # Bootstrap CIs, paired tests, actuarial tables
│   ├── generate_figures.py              # Publication figures (Figures 1, 2, 3, S2, S3)
│   ├── derive_stratified_analyses.py    # Stratified MAE, IBNR sensitivity, R²/PR CIs
│   ├── derive_rmse.py                   # Analytical RMSE for all 22 models
│   └── concordance_check.py             # Pre-submission cross-file audit
├── src/
│   ├── data/
│   │   ├── load_claims.py       # Claims loading, feature engineering, temporal splits
│   │   └── synthetic_panel.py   # Synthetic panel data generator
│   ├── evaluation/
│   │   └── metrics.py           # Calibrated R², predictive ratios, decile analysis
│   ├── models/
│   │   ├── baselines.py         # XGBoost, RF, LightGBM, Two-Part hurdle, demographics GLM
│   │   ├── ts_baselines.py      # Naive trailing mean, naive last-value, ARIMA
│   │   ├── timesfm_wrapper.py   # Chronos foundation model + PanelFM conditioning
│   │   └── patient_encoder.py   # XGBoost leaf-node embeddings + PCA
│   └── utils/
├── requirements.txt
└── README.md (this file)
```

## Reproducing the analyses

### Prerequisites

```bash
pip install -r requirements.txt
```

Python 3.12 was used for all experiments. The Chronos model checkpoint (`amazon/chronos-t5-small`) is downloaded automatically from Hugging Face on first run. CPU inference is sufficient; total runtime is approximately 2 hours.

### Data setup

The experiments require deidentified Medicaid managed care claims data in CSV format. Set the `DATA_ROOT` environment variable to point to your data directory, or place files under `./data/real_inputs/`. The required tables and column mappings are listed in `configs/data_config.yaml`.

The deidentified claims used in the published study are not publicly redistributable under the data-use agreement with the participating care management organization; reasonable requests for collaborative re-analysis can be directed to the corresponding author. The synthetic data generator (`src/data/synthetic_panel.py`) requires no external data and produces a complete synthetic panel sufficient to run the full pipeline end-to-end.

### Full pipeline

```bash
# 1. Run all models (synthetic + real data; outputs to ./results/, not committed)
python scripts/run_experiment.py

# 2. Generate publication tables (bootstrap CIs, statistical tests, decile analysis)
python scripts/generate_publication_tables.py

# 3. Derive stratified, RMSE, and IBNR sensitivity analyses
python scripts/derive_stratified_analyses.py
python scripts/derive_rmse.py

# 4. Render publication figures
python scripts/generate_figures.py

# 5. (Optional) Run the pre-submission concordance audit
python scripts/concordance_check.py
```

## Models

| Category | Models | Manuscript section |
|---|---|---|
| Demographics-only | Generalized linear model (age, sex, dual-eligible) | Methods §Models (floor) |
| Cross-sectional ML | XGBoost, Random Forest, LightGBM, Stacking Ensemble, Two-Part Hurdle | §S2 |
| Classical time series | Naive trailing mean, Naive last-value, ARIMA | §S3 |
| Foundation model | Chronos-T5-Small (46M parameters, zero-shot) | §S3.3 |
| Panel-conditioned foundation model | PanelFM-XReg (linear), PanelFM-Adapter (2-layer MLP, 1,475 params), PanelFM-ICF (5-NN context) | §S4–S5 |
| Hybrid | Calibrated cross-sectional 3-month budget × foundation-model temporal allocation | §S9 |
| Concurrent | Same-period features applied to XGBoost, Random Forest, LightGBM, Two-Part, GLM | §S8.6 |

## Evaluation metrics

Following Society of Actuaries and CMS risk-adjustment standards:

- Mean absolute error (MAE) at the patient × 3-month level
- Root mean squared error (RMSE) and RMSE-to-MAE ratio
- Calibrated R² (budget-neutral)
- Predictive ratio (target 0.90–1.10)
- Decile analysis (D5–D10 PR plots; D1–D4 raw $ predictions)
- Cost-censored R² ($250,000 cap)
- Quantile losses at q=0.50 and q=0.90
- C-statistic, PPV, and lift for high-cost identification
- Bootstrap 95% CIs (2,000 patient-level iterations); paired bootstrap pairwise tests (10,000 iterations); block-bootstrap and seed-sensitivity validation
- Stratified MAE by zero-cost vs positive-cost stratum
- IBNR sensitivity rerun on a mature 3-month claims window

## Citation

If you use this code, please cite the published manuscript (citation will appear here on acceptance).

## License

See `LICENSE`.

## Revision 2 — member-disjoint validation

The second-round revision re-runs the analysis under a member-disjoint split: the
member population is partitioned into non-overlapping training (70%), validation
(15%), and test (15%) sets, so that no member contributes observations to more than
one set. Mean absolute error is reported per member-month for every model class, and
calibrated R-squared and the predictive ratio are computed on the patient-level
three-month total. A gated hybrid uses the foundation model to gate and time costs
while the cross-sectional model sets the level for members predicted to incur cost.

```
revision2/code/
  run_member_disjoint.py   # member-disjoint pipeline (CS, TS, foundation, hybrid, concurrent)
  chronos_stage.py         # torch-isolated Chronos + panel-conditioned forecasting (subprocess)
  hybrid_experiments.py    # gated-hybrid gate search across the MAE-discrimination frontier
  render_figures.py        # figures from the member-disjoint results
  make_tables.py           # cohort and performance tables
```

Run:

```bash
DATA_ROOT=/path/to/real_inputs python revision2/code/run_member_disjoint.py
# reuse saved Chronos forecasts when iterating on metrics:
DATA_ROOT=/path/to/real_inputs python revision2/code/run_member_disjoint.py --reuse-forecasts
```

The torch-based Chronos stage runs in a separate process from the gradient-boosted
models to avoid an OpenMP runtime conflict on macOS. Results, figures, tables, and
manuscript text are not committed; they regenerate from the source data.
