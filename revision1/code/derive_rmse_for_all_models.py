"""
Derive RMSE for all 22 models (including TS, FM, Hybrid) using the
identity R² = 1 − SS_res/SS_total, where SS_total = Σ(y_i − ȳ)² is the
same constant across models evaluated on the same test set.

For models without retained per-patient predictions (TS, FM, Hybrid),
the JSON contains R²_raw and MAE but no RMSE. We reconstruct:

    SS_res = (1 − R²_raw) × SS_total
    RMSE   = sqrt(SS_res / n)

SS_total is estimated as the mean across the 6 cross-sectional models
(which all have RMSE in the JSON):
    SS_total = mean_k( n_k × RMSE_k² / (1 − R²_raw_k) )

Also adds the log-MAE analogue when derivable, and the relative penalty
of switching from MAE to RMSE (which signals outlier sensitivity).

Run:
  python /Users/sanjaybasu/waymark-local/packaging/panelfm/revision1/code/derive_rmse_for_all_models.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path("/Users/sanjaybasu/waymark-local/packaging/panelfm")
RESULTS = ROOT / "results"
OUT = ROOT / "revision1" / "results"

with open(RESULTS / "all_metrics_real.json") as f:
    M = json.load(f)


def estimate_ss_total() -> float:
    """Mean SS_total estimate from CS models with known RMSE + R²_raw."""
    ss_totals = []
    for name in ("xgboost", "random_forest", "lightgbm", "two_part",
                 "stacking", "demographic_glm"):
        m = M[name]
        n = 64047  # CS test set size
        rmse = m["rmse"]
        r2 = m["r_squared"]
        ss_res = n * rmse * rmse
        ss_total = ss_res / (1 - r2)
        ss_totals.append(ss_total)
    return sum(ss_totals) / len(ss_totals)


SS_TOTAL_PER_N = None  # computed once


def variance_y() -> float:
    """σ²_y on the test set, common across all models."""
    global SS_TOTAL_PER_N
    if SS_TOTAL_PER_N is None:
        SS_TOTAL_PER_N = estimate_ss_total() / 64047
    return SS_TOTAL_PER_N


def derive_rmse(model_name: str, n_override: int | None = None) -> float | None:
    """Derive RMSE from R²_raw + variance_y for any model."""
    m = M.get(model_name)
    if m is None:
        return None
    if "rmse" in m and m["rmse"] is not None and "r_squared" in m:
        # Use stored RMSE if available
        return m["rmse"]
    if "r_squared" not in m:
        return None
    n = n_override or m.get("n_patients", 64141)
    var_y = variance_y()
    ss_total = var_y * n
    ss_res = (1 - m["r_squared"]) * ss_total
    mse = ss_res / n
    if mse < 0:
        return None
    return math.sqrt(mse)


def main():
    rows = []
    # Ordering matches Table S8
    order = [
        ("demographic_glm", "CS (floor)"),
        ("xgboost", "CS"),
        ("random_forest", "CS"),
        ("lightgbm", "CS"),
        ("two_part", "CS"),
        ("stacking", "CS"),
        ("naive_mean3", "TS"),
        ("naive_last", "TS"),
        ("arima", "TS (n=500)"),
        ("chronos_zeroshot", "FM"),
        ("panelfm_xreg", "FM+panel"),
        ("panelfm_adapter", "FM+panel"),
        ("panelfm_icf", "FM+panel"),
        ("hybrid_chronos_zeroshot", "Hybrid"),
        ("hybrid_panelfm_adapter", "Hybrid"),
        ("hybrid_panelfm_icf", "Hybrid"),
        ("hybrid_panelfm_xreg", "Hybrid"),
        ("concurrent_xgboost", "Concurrent"),
        ("concurrent_random_forest", "Concurrent"),
        ("concurrent_lightgbm", "Concurrent"),
        ("concurrent_two_part", "Concurrent"),
    ]
    for name, cat in order:
        m = M.get(name)
        if m is None:
            continue
        mae = m["mae"]
        rmse = derive_rmse(name)
        rmse_to_mae = (rmse / mae) if (rmse and mae) else None
        q50 = m.get("quantile_loss_50")
        q90 = m.get("quantile_loss_90")
        # If q50 missing, approximate as MAE/2 (exact for point-median forecast)
        if q50 is None:
            q50 = mae / 2.0
        rows.append({
            "model": name,
            "category": cat,
            "mae": round(mae, 1),
            "rmse_derived": round(rmse, 1) if rmse else None,
            "rmse_to_mae_ratio": round(rmse_to_mae, 2) if rmse_to_mae else None,
            "q_loss_50": round(q50, 1) if q50 else None,
            "q_loss_90": round(q90, 1) if q90 else None,
            "r2_raw": round(m.get("r_squared", float("nan")), 3),
            "n_patients": m.get("n_patients", 64047),
            "rmse_source": ("stored in JSON" if "rmse" in m and m["rmse"] is not None
                            else "derived from R²_raw + Var(y)"),
        })

    out_csv = OUT / "rmse_alternative_losses_all_models.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv}")

    # Also write a markdown table fragment for direct paste into Table S8
    out_md = OUT / "rmse_alternative_losses_table_fragment.md"
    lines = [
        "| Model | Category | MAE ($) | RMSE ($) | RMSE/MAE | q-Loss 0.50 ($) | q-Loss 0.90 ($) | R²_raw |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['category']} | "
            f"{r['mae']:,.0f} | "
            f"{r['rmse_derived']:,.0f} | "
            f"{r['rmse_to_mae_ratio']:.2f} | "
            f"{r['q_loss_50']:,.0f} | "
            f"{(str(r['q_loss_90']) + ' (CS only)') if r['q_loss_90'] else 'n/a'} | "
            f"{r['r2_raw']:.2f} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_md}")

    # Summary stats: does RMSE preserve the model ordering?
    print("\n--- Sanity check: RMSE preserves headline ordering ---")
    headline = ["chronos_zeroshot", "hybrid_chronos_zeroshot", "hybrid_panelfm_adapter",
                "two_part", "stacking", "demographic_glm"]
    for name in headline:
        m = next((r for r in rows if r["model"] == name), None)
        if m:
            print(f"  {m['model']:30s}  MAE=${m['mae']:>7,.0f}  "
                  f"RMSE=${m['rmse_derived']:>7,.0f}  ratio={m['rmse_to_mae_ratio']}")

    # Validate by recomputing for CS models (where we have stored RMSE)
    print("\n--- Validation: derived RMSE vs JSON RMSE for CS models ---")
    for name in ("xgboost", "random_forest", "lightgbm", "two_part", "stacking",
                 "demographic_glm"):
        m = M[name]
        derived = math.sqrt((1 - m["r_squared"]) * variance_y())
        stored = m["rmse"]
        delta_pct = abs(derived - stored) / stored * 100
        print(f"  {name:25s}  derived=${derived:>7,.0f}  stored=${stored:>7,.0f}  "
              f"Δ={delta_pct:.2f}%")


if __name__ == "__main__":
    main()
