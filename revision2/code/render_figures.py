#!/usr/bin/env python3
"""
Figure renderer for the member-disjoint second revision.

Reads the JSON outputs of run_member_disjoint.py and produces the manuscript and
supplementary figures. Schematic panels (Figure 1A, Figure S1) are drawn the same
way as the prior revision; data-driven panels use the member-disjoint results.

Design rule: no floating text on data plots; all annotation lives in captions.

Outputs to notebooks/panelfm/revision2/figures/.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PKG_ROOT = Path(__file__).resolve().parents[2]
RES = PKG_ROOT / "revision2" / "results"
FIG_OUT = PKG_ROOT / "revision2" / "figures"
FIG_OUT.mkdir(parents=True, exist_ok=True)

METRICS = json.loads((RES / "all_metrics_disjoint.json").read_text())
DECILES = json.loads((RES / "decile_disjoint.json").read_text())
MONTHCOST = json.loads((RES / "month_cost_disjoint.json").read_text())
SPLIT = json.loads((RES / "split_info_disjoint.json").read_text())

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 11,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
})
CAT_COLORS = {
    "Cross-sectional": "#1f77b4", "Time series (classical)": "#7f7f7f",
    "Foundation model": "#d62728", "Panel-conditioned FM": "#ff7f0e", "Hybrid": "#2ca02c",
}


def _box(ax, x, y, w, h, label, fc, ec, fontsize=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                linewidth=1.2, facecolor=fc, edgecolor=ec))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize, wrap=True)


def _arrow_h(ax, x0, x1, y, color="black"):
    ax.add_patch(FancyArrowPatch((x0 + 0.05, y), (x1 - 0.05, y), arrowstyle="-|>",
                                 mutation_scale=12, linewidth=1.4, color=color))


def render_figure_1():
    fig = plt.figure(figsize=(14.5, 6.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.22)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    axA.set_xlim(0, 14); axA.set_ylim(0, 9); axA.set_axis_off()
    axA.set_title("A. Hybrid Construction", loc="left", fontweight="bold", pad=12)
    BOX_W, BOX_H, GAP, LEFT = 4.0, 1.3, 0.5, 0.3
    c1, c2, c3 = LEFT, LEFT + BOX_W + GAP, LEFT + 2 * (BOX_W + GAP)
    rt, rm, rf = 7.4, 5.2, 2.6
    _box(axA, c1, rt, BOX_W, BOX_H, "20 patient features\n(12-month lookback,\ntraining members)", "#e6f0ff", "#1f77b4")
    _box(axA, c2, rt, BOX_W, BOX_H, "Two-part hurdle\ncross-sectional model", "#e6f0ff", "#1f77b4")
    _box(axA, c3, rt, BOX_W, BOX_H, "Calibrated 3-month\ncost budget  Ŷ$^{CS}$\n(validation-calibrated)", "#cfe2f3", "#1f77b4")
    _arrow_h(axA, c1 + BOX_W, c2, rt + BOX_H / 2); _arrow_h(axA, c2 + BOX_W, c3, rt + BOX_H / 2)
    _box(axA, c1, rm, BOX_W, BOX_H, "Monthly cost history", "#ffe6e6", "#d62728")
    _box(axA, c2, rm, BOX_W, BOX_H, "Chronos-T5-Small\nfoundation model\n(46M, zero-shot)", "#ffe6e6", "#d62728")
    _box(axA, c3, rm, BOX_W, BOX_H, "Three monthly\nforecasts\nŷ$^{TS}_1$, ŷ$^{TS}_2$, ŷ$^{TS}_3$", "#fcd0d0", "#d62728")
    _arrow_h(axA, c1 + BOX_W, c2, rm + BOX_H / 2); _arrow_h(axA, c2 + BOX_W, c3, rm + BOX_H / 2)
    _box(axA, c1, rf, (c3 + BOX_W) - c1, 1.6,
         "Hybrid prediction for month h:\n\n"
         "ŷ$^{hybrid}_h$  =  ( ŷ$^{TS}_h$ / Σ$_m$ ŷ$^{TS}_m$ )  ×  Ŷ$^{CS}$    "
         "[ = 0 if Σ$_m$ ŷ$^{TS}_m$ = 0 ]", "#e8f5e9", "#2ca02c", 9.5)

    def cat(n):
        if n in ("naive_last", "naive_mean3"): return "Time series (classical)"
        if n == "chronos_zeroshot": return "Foundation model"
        if n.startswith("panelfm"): return "Panel-conditioned FM"
        if n.startswith("hybrid"): return "Hybrid"
        return "Cross-sectional"

    ordered = [
        ("hybrid_gated", "Hybrid (gated)"),
        ("hybrid_chronos_zeroshot", "Hybrid (proportional)"),
        ("hybrid_naive_mean3", "Hybrid (naive base)"),
        ("stacking", "Stacking ensemble"),
        ("random_forest", "Random forest"),
        ("lightgbm", "LightGBM"),
        ("two_part", "Two-part hurdle"),
        ("xgboost", "XGBoost"),
        ("naive_mean3", "Naive trailing mean"),
        ("naive_last", "Naive last-value"),
        ("demographic_glm", "Demographics-only GLM"),
        ("panelfm_xreg", "PanelFM-XReg"),
        ("panelfm_adapter", "PanelFM-Adapter"),
        ("chronos_zeroshot", "Chronos (zero-shot)"),
    ]
    pts = [(i, n, lab, METRICS[n]["mae"], METRICS[n]["r_squared_calibrated"], cat(n))
           for i, (n, lab) in enumerate(ordered, 1) if n in METRICS]
    axB.axhspan(0.08, 0.24, color="#1f77b4", alpha=0.06, zorder=1)
    axB.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    for c, color in CAT_COLORS.items():
        xs = [p[3] for p in pts if p[5] == c]; ys = [p[4] for p in pts if p[5] == c]
        axB.scatter(xs, ys, s=140, color=color, edgecolor="white", linewidth=1.0, label=c, zorder=3)
    for i, n, lab, mae, r2, c in pts:
        axB.text(mae, r2, str(i), ha="center", va="center", fontsize=8, fontweight="bold", color="white", zorder=4)
    maxmae = max(p[3] for p in pts)
    axB.set_xlabel("MAE ($/member-month, log scale)"); axB.set_ylabel("Calibrated R²")
    axB.set_title("B. MAE vs Calibrated R²", loc="left", fontweight="bold", pad=12)
    axB.set_xscale("log"); axB.set_xlim(0.8 * min(p[3] for p in pts), 1.25 * maxmae)
    axB.set_ylim(-1.5, 0.45)
    ymin = min(p[4] for p in pts)
    axB.set_ylim(min(-1.5, ymin - 0.1), 0.45)
    key = [f"{p[0]:>2}.  {p[2]}" for p in pts]
    axB.legend(loc="lower right", fontsize=8, framealpha=1.0, title="Model category", title_fontsize=8.5)
    axB.text(1.04, 1.0, "\n".join(key), transform=axB.transAxes, fontsize=7.8, va="top", ha="left", linespacing=1.6)
    fig.subplots_adjust(left=0.05, right=0.72, top=0.92, bottom=0.10)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_OUT / f"figure1_hybrid_and_scatter.{ext}", bbox_inches="tight",
                    dpi=(200 if ext == "png" else None))
    plt.close(fig); print("Wrote figure1")


def _decile_series(model, key):
    d = DECILES.get(model, {})
    return [d.get(f"decile_{i}", {}).get(key, np.nan) for i in range(1, 11)]


def render_figure_2():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.05, 1.0]})
    modelsA = {"demographic_glm": ("Demographics-only GLM", "#7f7f7f"),
               "two_part": ("Two-part hurdle", "#1f77b4"),
               "chronos_zeroshot": ("Chronos (zero-shot)", "#d62728"),
               "naive_mean3": ("Naive trailing mean", "#ff7f0e"),
               "hybrid_gated": ("Hybrid (gated)", "#2ca02c")}
    axA.axhspan(0.9, 1.1, color="#cccccc", alpha=0.18, zorder=1)
    axA.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7, zorder=2)
    # Plot deciles where the predictive ratio is defined (actual mean > 0).
    for m, (lab, color) in modelsA.items():
        if m not in DECILES:
            continue
        pr = _decile_series(m, "predictive_ratio")
        xs = [i for i in range(1, 11) if pr[i - 1] is not None and np.isfinite(pr[i - 1] or np.nan)]
        ys = [pr[i - 1] for i in xs]
        axA.plot(xs, ys, "o-", color=color, label=lab, linewidth=1.8, markersize=6, zorder=3)
    axA.set_xlabel("Decile of actual 3-month cost")
    axA.set_ylabel("Predictive ratio (predicted / actual, log scale)")
    axA.set_title("A. Decile calibration (deciles with positive actual cost)", loc="left", fontweight="bold")
    axA.set_yscale("log")
    axA.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8, framealpha=1.0)

    # Panel B: raw $ predictions in the all-zero-actual deciles.
    zero_deciles = [i for i in range(1, 11)
                    if (DECILES.get("two_part", {}).get(f"decile_{i}", {}).get("actual_mean", 1) or 0) == 0]
    if not zero_deciles:
        zero_deciles = [1, 2, 3, 4]
    modelsB = {"demographic_glm": ("Demographics-only GLM", "#7f7f7f"),
               "two_part": ("Two-part hurdle", "#1f77b4"),
               "chronos_zeroshot": ("Chronos (zero-shot)", "#d62728"),
               "naive_mean3": ("Naive trailing mean", "#ff7f0e"),
               "hybrid_gated": ("Hybrid (gated)", "#2ca02c")}
    x = np.arange(len(zero_deciles)); n = len(modelsB); bw = 0.85 / n
    for i, (m, (lab, color)) in enumerate(modelsB.items()):
        if m not in DECILES:
            continue
        vals = [DECILES[m].get(f"decile_{d}", {}).get("pred_mean", 0) for d in zero_deciles]
        axB.bar(x + (i - (n - 1) / 2) * bw, vals, width=bw, color=color, label=lab, edgecolor="white", linewidth=0.4)
    axB.set_xticks(x); axB.set_xticklabels([f"D{d}" for d in zero_deciles])
    axB.set_xlabel("Decile of actual 3-month cost  (all actual = $0)")
    axB.set_ylabel("Mean predicted 3-month cost ($)")
    axB.set_title("B. Low-cost deciles: raw $ predictions", loc="left", fontweight="bold")
    axB.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8, framealpha=1.0)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.30)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_OUT / f"figure2_decile_calibration.{ext}", bbox_inches="tight",
                    dpi=(200 if ext == "png" else None))
    plt.close(fig); print("Wrote figure2")


def render_figure_3():
    prospective = [("Demo GLM", "demographic_glm", "Demographics-only"),
                   ("XGBoost", "xgboost", "Cross-sectional ML"),
                   ("Stacking", "stacking", "Cross-sectional ML"),
                   ("Chronos\n(raw)", "chronos_zeroshot", "Foundation (raw)"),
                   ("Hybrid\n(proportional)", "hybrid_chronos_zeroshot", "Hybrid"),
                   ("Hybrid\n(gated)", "hybrid_gated", "Hybrid")]
    concurrent = [("Conc.\nXGBoost", "concurrent_xgboost", "Concurrent"),
                  ("Conc.\nRF", "concurrent_random_forest", "Concurrent")]
    colors = {"Demographics-only": "#a6cee3", "Cross-sectional ML": "#e69a00",
              "Foundation (raw)": "#1f77b4", "Hybrid": "#2ca02c", "Concurrent": "#c75ba1"}
    rows = [(lab, METRICS[k]["r_squared_calibrated"] * 100, cat) for lab, k, cat in (prospective + concurrent) if k in METRICS]
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    x = np.arange(len(rows)); npro = sum(1 for r in (prospective) if r[1] in METRICS)
    xadj = np.array([xi if i < npro else xi + 0.6 for i, xi in enumerate(x)])
    ax.axhspan(1, 3, color="#a6cee3", alpha=0.25); ax.axhspan(8, 24, color="#1f77b4", alpha=0.10)
    ax.axhspan(55, 67, color="#2ca02c", alpha=0.10); ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.axvline((xadj[npro - 1] + xadj[npro]) / 2, color="#888888", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.bar(xadj, [r[1] for r in rows], color=[colors[r[2]] for r in rows], edgecolor="white", linewidth=0.6, width=0.7, zorder=3)
    for xi, v in zip(xadj, [r[1] for r in rows]):
        ax.text(xi, v + (1.8 if v >= 0 else -3.5), f"{v:.1f}%", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8.5, fontweight="bold",
                color=("#222222" if v >= 0 else "#d62728"))
    ax.set_xticks(xadj); ax.set_xticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_ylabel("Calibrated R² (%)")
    lo = min(r[1] for r in rows)
    ax.set_ylim(min(-100, lo - 10), 100)
    legend = [mpatches.Patch(color=colors[k], label=k) for k in colors]
    legend += [mpatches.Patch(facecolor="#a6cee3", alpha=0.35, label="Demographics-only band (1-3%)"),
               mpatches.Patch(facecolor="#1f77b4", alpha=0.15, label="CDPS prospective band (8-24%)"),
               mpatches.Patch(facecolor="#2ca02c", alpha=0.15, label="MARA concurrent band (55-67%)")]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=8, framealpha=1.0)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.26)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_OUT / f"figure3_r2_benchmarks.{ext}", bbox_inches="tight",
                    dpi=(200 if ext == "png" else None))
    plt.close(fig); print("Wrote figure3")


def render_figure_s2():
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.set_xlim(-1, 35); ax.set_ylim(-2.0, 4.2); ax.set_axis_off()
    ax.set_title("Supplementary Figure S2. Member-disjoint design with a prospective forecast window",
                 loc="left", fontweight="bold", fontsize=12, pad=10)
    ax.plot([0, 33], [0, 0], "k-", linewidth=1.0)
    quarters = [(0, "Jan\n2023"), (6, "Jul"), (12, "Jan\n2024"), (18, "Jul"),
                (24, "Jan\n2025"), (27, "Apr"), (30, "Jul"), (33, "Oct\n2025")]
    for xq, lab in quarters:
        ax.plot([xq, xq], [-0.18, 0.18], "k-", linewidth=0.7)
        ax.text(xq, -0.55, lab, ha="center", va="top", fontsize=8)
    ax.add_patch(mpatches.Rectangle((13, 1.6), 11, 0.6, facecolor="#fff4cc", edgecolor="#e8a800", linewidth=1.0))
    ax.text(18.5, 1.9, "12-month feature lookback (Feb 2024 - Jan 2025)", ha="center", va="center", fontsize=9)
    ax.add_patch(mpatches.Rectangle((0, 0.4), 24, 0.8, facecolor="#eeeeee", edgecolor="#666666", linewidth=1.0))
    ax.text(12, 0.8, "Member history used as features / context", ha="center", va="center", fontsize=9.5)
    ax.add_patch(mpatches.Rectangle((27, 0.4), 3, 0.8, facecolor="#d6f0d6", edgecolor="#2ca02c", linewidth=1.0))
    ax.text(28.5, 0.8, "Prospective\ntarget\nMay-Jul 2025", ha="center", va="center", fontsize=8, fontweight="bold")
    # Disjoint member partition shown as three stacked tracks.
    n_tr, n_va, n_te = SPLIT["n_train_members"], SPLIT["n_val_members"], SPLIT["n_test_members"]
    ax.text(-0.5, 3.4, "Disjoint member sets (no member in >1 set):", fontsize=9.5, fontweight="bold", va="center")
    ax.add_patch(mpatches.Rectangle((0, 2.7), 23.1, 0.45, facecolor="#cfe2f3", edgecolor="#1f77b4"))
    ax.text(11.5, 2.925, f"Training members  (n={n_tr:,}; fit all models)", ha="center", va="center", fontsize=8)
    ax.add_patch(mpatches.Rectangle((23.4, 2.7), 4.9, 0.45, facecolor="#e8d6f5", edgecolor="#6a4ea6"))
    ax.text(25.85, 2.925, f"Validation\n(n={n_va:,})", ha="center", va="center", fontsize=7)
    ax.add_patch(mpatches.Rectangle((28.6, 2.7), 4.4, 0.45, facecolor="#d6f0d6", edgecolor="#2ca02c"))
    ax.text(30.8, 2.925, f"Test\n(n={n_te:,})", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_OUT / f"figure_s2_temporal_setup.{ext}", bbox_inches="tight",
                    dpi=(200 if ext == "png" else None))
    plt.close(fig); print("Wrote figureS2")


def render_figure_s3():
    items = sorted(MONTHCOST.items())
    months = [m for m, _ in items]; vals = [v for _, v in items]
    x = np.arange(len(months))
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.bar(x, vals, color="#1f77b4", edgecolor="white", linewidth=0.6, width=0.7, zorder=3)
    coeffs = np.polyfit(x, vals, 1)
    ax.plot(x, np.polyval(coeffs, x), color="#d62728", linestyle="--", linewidth=1.6, zorder=4,
            label=f"Linear trend: ${coeffs[0]:.1f}/month")
    for xi, c in zip(x, vals):
        ax.text(xi, c + max(vals) * 0.02, f"${c:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(months, fontsize=9, rotation=0)
    ax.set_ylabel("Mean reported cost ($)"); ax.set_xlabel("Month of incurrence")
    ax.set_title("Supplementary Figure S3. Test-period cost by month of incurrence", loc="left", fontweight="bold", fontsize=11, pad=10)
    ax.set_ylim(0, max(vals) * 1.18); ax.legend(loc="upper right", fontsize=8.5, framealpha=1.0)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_OUT / f"figure_s3_month_of_incurrence.{ext}", bbox_inches="tight",
                    dpi=(200 if ext == "png" else None))
    plt.close(fig); print("Wrote figureS3")


if __name__ == "__main__":
    render_figure_1()
    render_figure_2()
    render_figure_3()
    render_figure_s2()
    render_figure_s3()
    print("All figures rendered to", FIG_OUT)
