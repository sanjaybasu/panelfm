# revision1/code — Code for the Medical Care revision (MDC-D-26-00128R1)

This directory contains the analysis scripts added during the major revision. None of the outputs (figures, tables, derived metrics) are committed — they regenerate from these scripts plus the source data described in `../../README.md`.

## Scripts

| Script | Purpose |
|---|---|
| `derive_revision_analyses.py` | Computes derived outputs from `results/all_metrics_real.json` and `decile_analysis.csv`: stratified MAE by zero-cost vs positive-cost stratum (Table S9b), hybrid all-zero trigger rate (§S9.4), IBNR sensitivity logic (§S12), subgroup data-availability summary (§S13), bootstrap/k-fold rationale (§S6), and R²/PR confidence intervals for headline models. |
| `derive_rmse_for_all_models.py` | Derives RMSE analytically for time-series, foundation-model, and hybrid models using the identity RMSE = √[(1 − R²\_raw) × σ²\_y], where σ²\_y is estimated from the six cross-sectional models with stored RMSE (reproduces stored RMSE for those six exactly, Δ < 0.01%). Populates Supplementary Table S8. |
| `render_revision_figures.py` | Renders Figures 1, 2, 3, S2, and S3. Reads `results/all_metrics_real.json` and `decile_analysis.csv`; writes `.pdf` and `.png` to the local figures folder (not committed). |
| `concordance_check_pmpm.py` | Nine-check pre-submission audit: canonical-number cross-file consistency, citation completeness, no orphan bibliography, sequential Vancouver numbering, word counts within Medical Care limits, no revision-mode artifacts, response-letter blinding, figure/table reference completeness, and reviewer-comment coverage. Emits a markdown report and a JSON summary. |

## Reproduction

```bash
# 1. Ensure the primary pipeline has been run (see ../../README.md)
#    — produces results/all_metrics_real.json, results/all_metrics_synthetic.json,
#      and the decile_analysis.csv consumed by the revision scripts.

# 2. Re-derive the revision analyses
python revision1/code/derive_revision_analyses.py
python revision1/code/derive_rmse_for_all_models.py

# 3. Re-render the revision figures
python revision1/code/render_revision_figures.py

# 4. (Optional) Run the concordance gate against the manuscript artifacts
#    — assumes the manuscript .md files are present locally; not committed here.
python revision1/code/concordance_check_pmpm.py
```

The scripts are read-only against the primary results and write outputs to project-local folders (not committed to this repository).
