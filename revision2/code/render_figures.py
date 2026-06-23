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
FIG_OUT = Path(__file__).resolve().parents[4] / "notebooks" / "panelfm" / "revision2" / "figures"
FIG_OUT.mkdir(parents=True, exist_ok=True)

METRICS = json.loads((RES / "all_metrics_disjoint.json").read_text())
DECILES = json.loads((RES / "decile_disjoint.json").read_text())
MONTHCOST = json.loads((RES / "month_cost_disjoint.json").read_text())
SPLIT = json.loads((RES / "split_info_disjoint.json").read_text())
METRICS_12MO = json.loads((RES / "all_metrics_12mo.json").read_text()) if (RES / "all_metrics_12mo.json").exists() else {}
SPLIT_12MO = json.loads((RES / "split_info_12mo.json").read_text()) if (RES / "split_info_12mo.json").exists() else {}
DECILES_12MO = (json.loads((RES / "decile_12mo.json").read_text())
                if (RES / "decile_12mo.json").exists() else {})

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
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.30, wspace=0.22)
    axA = fig.add_subplot(gs[0, :])
    axB, axC = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

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
    def _scatter(ax, src, title, show_cat_legend):
        pts = [(i, n, lab, src[n]["mae"], src[n]["r_squared_calibrated"], cat(n))
               for i, (n, lab) in enumerate(ordered, 1) if n in src]
        ax.axhspan(0.08, 0.24, color="#1f77b4", alpha=0.06, zorder=1)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
        for c, color in CAT_COLORS.items():
            xs = [p[3] for p in pts if p[5] == c]; ys = [p[4] for p in pts if p[5] == c]
            ax.scatter(xs, ys, s=140, color=color, edgecolor="white", linewidth=1.0, label=c, zorder=3)
        for i, n, lab, mae, r2, c in pts:
            ax.text(mae, r2, str(i), ha="center", va="center", fontsize=8, fontweight="bold", color="white", zorder=4)
        maxmae = max(p[3] for p in pts)
        ax.set_xlabel("MAE ($/member-month, log scale)"); ax.set_ylabel("Calibrated R²")
        ax.set_title(title, loc="left", fontweight="bold", pad=12)
        ax.set_xscale("log"); ax.set_xlim(0.8 * min(p[3] for p in pts), 1.25 * maxmae)
        ymin = min(p[4] for p in pts); ymax = max(p[4] for p in pts)
        ax.set_ylim(min(-1.5, ymin - 0.1), max(0.45, ymax + 0.1))
        if show_cat_legend:
            ax.legend(loc="lower right", fontsize=8, framealpha=1.0, title="Model category", title_fontsize=8.5)
        return pts

    _scatter(axB, METRICS, "B. MAE vs Calibrated R² (three-month horizon)", True)
    pts = _scatter(axC, METRICS_12MO, "C. MAE vs Calibrated R² (twelve-month horizon)", False)
    key = [f"{p[0]:>2}.  {p[2]}" for p in pts]
    axC.text(1.04, 1.0, "\n".join(key), transform=axC.transAxes, fontsize=7.8, va="top", ha="left", linespacing=1.6)
    fig.subplots_adjust(left=0.05, right=0.80, top=0.94, bottom=0.07)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_OUT / f"figure1_hybrid_and_scatter.{ext}", bbox_inches="tight",
                    dpi=(200 if ext == "png" else None))
    plt.close(fig); print("Wrote figure1")


def _decile_series(model, key, src=None):
    d = (src if src is not None else DECILES).get(model, {})
    return [d.get(f"decile_{i}", {}).get(key, np.nan) for i in range(1, 11)]


def render_figure_2():
    fig, ((axA, axB), (axC, axD)) = plt.subplots(2, 2, figsize=(13, 9))
    models_cal = {"demographic_glm": ("Demographics-only GLM", "#7f7f7f"),
                  "two_part": ("Two-part hurdle", "#1f77b4"),
                  "chronos_zeroshot": ("Chronos (zero-shot)", "#d62728"),
                  "naive_mean3": ("Naive trailing mean", "#ff7f0e"),
                  "hybrid_gated": ("Hybrid (gated)", "#2ca02c")}

    # Panels A (3-month) and B (12-month): predictive ratio by decile of actual cost,
    # for deciles where the ratio is defined (actual mean > 0). Identical series and scale.
    def _calibration_panel(ax, src, horizon):
        ax.axhspan(0.9, 1.1, color="#cccccc", alpha=0.18, zorder=1)
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7, zorder=2)
        for m, (lab, color) in models_cal.items():
            if m not in src:
                continue
            pr = _decile_series(m, "predictive_ratio", src)
            xs = [i for i in range(1, 11) if pr[i - 1] is not None and np.isfinite(pr[i - 1] or np.nan)]
            ys = [pr[i - 1] for i in xs]
            ax.plot(xs, ys, "o-", color=color, label=lab, linewidth=1.8, markersize=6, zorder=3)
        ax.set_xlabel(f"Decile of actual {horizon} cost")
        ax.set_yscale("log")

    _calibration_panel(axA, DECILES, "3-month")
    axA.set_ylabel("Predictive ratio (predicted / actual, log scale)")
    axA.set_title("A. Decile calibration, three-month horizon", loc="left", fontweight="bold")
    _calibration_panel(axB, DECILES_12MO, "12-month")
    axB.set_title("B. Decile calibration, twelve-month horizon", loc="left", fontweight="bold")

    # Panels C (3-month) and D (12-month): raw $ predictions in the lowest-cost deciles.
    def _lowcost_panel(ax, src, deciles, horizon, xlabel):
        x = np.arange(len(deciles)); n = len(models_cal); bw = 0.85 / n
        for i, (m, (lab, color)) in enumerate(models_cal.items()):
            if m not in src:
                continue
            vals = [src[m].get(f"decile_{d}", {}).get("pred_mean", 0) for d in deciles]
            ax.bar(x + (i - (n - 1) / 2) * bw, vals, width=bw, color=color, label=lab, edgecolor="white", linewidth=0.4)
        ax.set_xticks(x); ax.set_xticklabels([f"D{d}" for d in deciles])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"Mean predicted {horizon} cost ($)")

    # Three-month: deciles where actual cost is all $0.
    zero_deciles = [i for i in range(1, 11)
                    if (DECILES.get("two_part", {}).get(f"decile_{i}", {}).get("actual_mean", 1) or 0) == 0]
    if not zero_deciles:
        zero_deciles = [1, 2, 3, 4]
    _lowcost_panel(axC, DECILES, zero_deciles, "3-month",
                   "Decile of actual 3-month cost  (all actual = $0)")
    axC.set_title("C. Low-cost deciles: raw $ predictions (three-month)", loc="left", fontweight="bold")
    # Twelve-month: only decile 1 is all-zero, so show the lowest three deciles.
    _lowcost_panel(axD, DECILES_12MO, [1, 2, 3], "12-month",
                   "Decile of actual 12-month cost  (lowest three)")
    axD.set_title("D. Low-cost deciles: raw $ predictions (twelve-month)", loc="left", fontweight="bold")

    # One shared legend (identical five series across all four panels).
    handles, labels = axA.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8.5, framealpha=1.0,
               bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.12, wspace=0.24, hspace=0.30)
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

    def _benchmark_panel(ax, src, title):
        rows = [(lab, src[k]["r_squared_calibrated"] * 100, cat) for lab, k, cat in (prospective + concurrent) if k in src]
        x = np.arange(len(rows)); npro = sum(1 for r in (prospective) if r[1] in src)
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
        ax.set_title(title, loc="left", fontweight="bold")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6.0))
    _benchmark_panel(axA, METRICS, "A. Calibrated R² vs published benchmarks (three-month horizon)")
    _benchmark_panel(axB, METRICS_12MO, "B. Calibrated R² vs published benchmarks (twelve-month horizon)")
    legend = [mpatches.Patch(color=colors[k], label=k) for k in colors]
    legend += [mpatches.Patch(facecolor="#a6cee3", alpha=0.35, label="Demographics-only band (1-3%)"),
               mpatches.Patch(facecolor="#1f77b4", alpha=0.15, label="CDPS prospective band (8-24%)"),
               mpatches.Patch(facecolor="#2ca02c", alpha=0.15, label="MARA concurrent band (55-67%)")]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=8, framealpha=1.0)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.20, wspace=0.18)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_OUT / f"figure3_r2_benchmarks.{ext}", bbox_inches="tight",
                    dpi=(200 if ext == "png" else None))
    plt.close(fig); print("Wrote figure3")


def render_figure_s1():
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(13, 7))
    fig.suptitle("Supplementary Figure S1. Member-disjoint design with prospective forecast windows",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)

    # Panel A: three-month prospective target (the primary design).
    axA.set_xlim(-1, 35); axA.set_ylim(-2.0, 4.2); axA.set_axis_off()
    axA.set_title("A. Three-month prospective target (primary)", loc="left", fontweight="bold", fontsize=11, pad=8)
    axA.plot([0, 33], [0, 0], "k-", linewidth=1.0)
    quarters = [(0, "Jan\n2023"), (6, "Jul"), (12, "Jan\n2024"), (18, "Jul"),
                (24, "Jan\n2025"), (27, "Apr"), (30, "Jul"), (33, "Oct\n2025")]
    for xq, lab in quarters:
        axA.plot([xq, xq], [-0.18, 0.18], "k-", linewidth=0.7)
        axA.text(xq, -0.55, lab, ha="center", va="top", fontsize=8)
    axA.add_patch(mpatches.Rectangle((13, 1.6), 11, 0.6, facecolor="#fff4cc", edgecolor="#e8a800", linewidth=1.0))
    axA.text(18.5, 1.9, "12-month feature lookback (Feb 2024 - Jan 2025)", ha="center", va="center", fontsize=9)
    axA.add_patch(mpatches.Rectangle((0, 0.4), 24, 0.8, facecolor="#eeeeee", edgecolor="#666666", linewidth=1.0))
    axA.text(12, 0.8, "Member history used as features / context", ha="center", va="center", fontsize=9.5)
    axA.add_patch(mpatches.Rectangle((27, 0.4), 3, 0.8, facecolor="#d6f0d6", edgecolor="#2ca02c", linewidth=1.0))
    axA.text(28.5, 0.8, "Prospective\ntarget\nMay-Jul 2025", ha="center", va="center", fontsize=8, fontweight="bold")
    # Disjoint member partition shown as three stacked tracks.
    n_tr, n_va, n_te = SPLIT["n_train_members"], SPLIT["n_val_members"], SPLIT["n_test_members"]
    axA.text(-0.5, 3.4, "Disjoint member sets (no member in >1 set):", fontsize=9.5, fontweight="bold", va="center")
    axA.add_patch(mpatches.Rectangle((0, 2.7), 23.1, 0.45, facecolor="#cfe2f3", edgecolor="#1f77b4"))
    axA.text(11.5, 2.925, f"Training members  (n={n_tr:,}; fit all models)", ha="center", va="center", fontsize=8)
    axA.add_patch(mpatches.Rectangle((23.4, 2.7), 4.9, 0.45, facecolor="#e8d6f5", edgecolor="#6a4ea6"))
    axA.text(25.85, 2.925, f"Validation\n(n={n_va:,})", ha="center", va="center", fontsize=7)
    axA.add_patch(mpatches.Rectangle((28.6, 2.7), 4.4, 0.45, facecolor="#d6f0d6", edgecolor="#2ca02c"))
    axA.text(30.8, 2.925, f"Test\n(n={n_te:,})", ha="center", va="center", fontsize=7)

    # Panel B: twelve-month prospective target (calendar-year CDPS convention).
    axB.set_xlim(-1, 35); axB.set_ylim(-2.0, 4.2); axB.set_axis_off()
    axB.set_title("B. Twelve-month prospective target (CDPS-convention sensitivity)",
                  loc="left", fontweight="bold", fontsize=11, pad=8)
    axB.plot([0, 33], [0, 0], "k-", linewidth=1.0)
    # Calendar axis: x=0 -> Jan 2023, x=11 -> Jan 2024, x=22 -> Jan 2025, x=33 -> Jan 2026.
    quarters_b = [(0, "Jan\n2023"), (5.5, "Jul"), (11, "Jan\n2024"), (16.5, "Jul"),
                  (22, "Jan\n2025"), (27.5, "Jul"), (33, "Dec\n2025")]
    for xq, lab in quarters_b:
        axB.plot([xq, xq], [-0.18, 0.18], "k-", linewidth=0.7)
        axB.text(xq, -0.55, lab, ha="center", va="top", fontsize=8)
    # 12-month feature lookback = calendar-year 2024 (x=11 to x=22), ending Dec 2024.
    axB.add_patch(mpatches.Rectangle((11, 1.6), 11, 0.6, facecolor="#fff4cc", edgecolor="#e8a800", linewidth=1.0))
    axB.text(16.5, 1.9, "12-month feature lookback (Jan - Dec 2024)", ha="center", va="center", fontsize=9)
    # Time-series context also through Dec 2024 (aligned cutoff at x=22).
    axB.add_patch(mpatches.Rectangle((0, 0.4), 22, 0.8, facecolor="#eeeeee", edgecolor="#666666", linewidth=1.0))
    axB.text(11, 0.8, "Member history used as features / context", ha="center", va="center", fontsize=9.5)
    # Prospective target = calendar-year 2025 (x=22 to x=33).
    axB.add_patch(mpatches.Rectangle((22, 0.4), 11, 0.8, facecolor="#d6f0d6", edgecolor="#2ca02c", linewidth=1.0))
    axB.text(27.5, 0.8, "Prospective target\nJan - Dec 2025", ha="center", va="center", fontsize=8, fontweight="bold")
    # Disjoint member partition (12-month split).
    n_tr2, n_va2, n_te2 = SPLIT_12MO["n_train_members"], SPLIT_12MO["n_val_members"], SPLIT_12MO["n_test_members"]
    axB.text(-0.5, 3.4, "Disjoint member sets (no member in >1 set):", fontsize=9.5, fontweight="bold", va="center")
    axB.add_patch(mpatches.Rectangle((0, 2.7), 23.1, 0.45, facecolor="#cfe2f3", edgecolor="#1f77b4"))
    axB.text(11.5, 2.925, f"Training members  (n={n_tr2:,}; fit all models)", ha="center", va="center", fontsize=8)
    axB.add_patch(mpatches.Rectangle((23.4, 2.7), 4.9, 0.45, facecolor="#e8d6f5", edgecolor="#6a4ea6"))
    axB.text(25.85, 2.925, f"Validation\n(n={n_va2:,})", ha="center", va="center", fontsize=7)
    axB.add_patch(mpatches.Rectangle((28.6, 2.7), 4.4, 0.45, facecolor="#d6f0d6", edgecolor="#2ca02c"))
    axB.text(30.8, 2.925, f"Test\n(n={n_te2:,})", ha="center", va="center", fontsize=7)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(FIG_OUT / f"figure_s1_temporal_setup.{ext}", bbox_inches="tight",
                    dpi=(200 if ext == "png" else None))
    plt.close(fig); print("Wrote figureS1")


def render_figure_s5():
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
    ax.set_title("Supplementary Figure S5. Test-period cost by month of incurrence", loc="left", fontweight="bold", fontsize=11, pad=10)
    ax.set_ylim(0, max(vals) * 1.18); ax.legend(loc="upper right", fontsize=8.5, framealpha=1.0)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_OUT / f"figure_s5_month_of_incurrence.{ext}", bbox_inches="tight",
                    dpi=(200 if ext == "png" else None))
    plt.close(fig); print("Wrote figureS3")


def _load_sota():
    """Merge the core, TSFM, and TabPFN metric files into one dict with category tags
    for the frontier figure. Returns list of (label, mae, pr, r2cal, category)."""
    tsfm = json.loads((RES / "all_metrics_tsfm.json").read_text()) if (RES / "all_metrics_tsfm.json").exists() else {}
    tab = json.loads((RES / "all_metrics_tabpfn.json").read_text()) if (RES / "all_metrics_tabpfn.json").exists() else {}
    rows = []

    def add(key, src, label, cat):
        if key in src and "mae" in src[key]:
            v = src[key]
            rows.append((label, v["mae"], v.get("predictive_ratio", np.nan),
                         v.get("r_squared_calibrated", np.nan), cat))
    # Cross-sectional gradient-boosted and actuarial baselines.
    for k, lab in [("two_part", "Two-part"), ("stacking", "Stacking"), ("random_forest", "Random forest"),
                   ("tweedie", "Tweedie GBM")]:
        add(k, METRICS, lab, "Cross-sectional (mean target)")
    add("quantile_median_cs", METRICS, "Quantile-median GBM", "Cross-sectional (median target)")
    add("naive_mean3", METRICS, "Naive mean", "Classical time series")
    # Time series foundation models.
    add("chronos_zeroshot", METRICS, "Chronos-small", "TS foundation (median)")
    for k, lab in [("chronos_t5_base_median", "Chronos-base"), ("chronos_t5_large_median", "Chronos-large"),
                   ("chronos_bolt_base_median", "Chronos-Bolt")]:
        add(k, tsfm, lab, "TS foundation (median)")
    add("timesfm_2p5", tsfm, "TimesFM-2.5", "TS foundation (median)")
    # Tabular foundation model.
    add("tabpfn_regressor", tab, "TabPFN-v2", "Tabular foundation (mean)")
    # Concurrent upper bound.
    add("concurrent_lightgbm", METRICS, "Concurrent LightGBM", "Concurrent (reference)")
    return rows


def render_figure_s4():
    """Decision-relevant evaluation: cost captured at capacity, and net-benefit (DCA)."""
    da = json.loads((RES / "decision_analysis.json").read_text())
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.0, 1.0]})
    show = {"two_part": ("Two-part hurdle", "#1f77b4"),
            "tweedie": ("Tweedie GBM", "#9467bd"),
            "timesfm_2p5": ("TimesFM-2.5", "#d62728"),
            "tabpfn_regressor": ("TabPFN-v2", "#8c564b"),
            "naive_mean3": ("Naive trailing mean", "#ff7f0e")}
    caps = [5, 10, 15, 20]
    for m, (lab, color) in show.items():
        cc = da["cost_captured"].get(m)
        if not cc:
            continue
        ys = [cc[f"cap_{c}pct"]["cost_captured_frac"] * 100 for c in caps]
        axA.plot(caps, ys, "o-", color=color, label=lab, linewidth=1.8, markersize=6)
    axA.plot(caps, caps, "k--", linewidth=0.9, alpha=0.6, label="Random targeting")
    axA.set_xlabel("Care-management capacity (% of members enrolled)")
    axA.set_ylabel("Realized cost captured (%)")
    axA.set_title("A. Cost captured at capacity", loc="left", fontweight="bold")
    axA.set_xticks(caps); axA.legend(loc="lower right", fontsize=8, framealpha=1.0)

    thr = [0.05, 0.10, 0.15, 0.20, 0.30]
    for m, (lab, color) in show.items():
        nb = da["net_benefit"].get(m)
        if not nb:
            continue
        ys = [nb[f"pt_{t:.2f}"] for t in thr]
        axB.plot([t * 100 for t in thr], ys, "o-", color=color, label=lab, linewidth=1.8, markersize=6)
    ta = da["net_benefit"]["_treat_all"]
    axB.plot([t * 100 for t in thr], [ta[f"pt_{t:.2f}"] for t in thr], color="#555555",
             linestyle="--", linewidth=1.2, label="Enroll all")
    axB.axhline(0, color="black", linewidth=0.9, linestyle=":", alpha=0.7, label="Enroll none")
    axB.set_xlabel("Threshold probability of high cost (%)")
    axB.set_ylabel("Net benefit")
    axB.set_title("B. Decision-curve analysis (net benefit)", loc="left", fontweight="bold")
    axB.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8, framealpha=1.0)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.26)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_OUT / f"figure_s4_decision_curves.{ext}", bbox_inches="tight",
                    dpi=(200 if ext == "png" else None))
    plt.close(fig); print("Wrote figureS4")


def render_figure_s3():
    """MAE vs predictive ratio frontier across the full state-of-the-art panel."""
    rows = _load_sota()
    cats = ["Cross-sectional (mean target)", "Cross-sectional (median target)",
            "Classical time series", "TS foundation (median)", "Tabular foundation (mean)", "Concurrent (reference)"]
    colors = {"Cross-sectional (mean target)": "#1f77b4", "Cross-sectional (median target)": "#17becf",
              "Classical time series": "#7f7f7f", "TS foundation (median)": "#d62728",
              "Tabular foundation (mean)": "#8c564b", "Concurrent (reference)": "#c75ba1"}
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.axhline(1.0, color="black", linewidth=0.9, linestyle="--", alpha=0.7, zorder=2)
    ax.axhspan(0.9, 1.1, color="#cccccc", alpha=0.18, zorder=1)
    for c in cats:
        xs = [r[1] for r in rows if r[4] == c]; ys = [r[2] for r in rows if r[4] == c]
        ax.scatter(xs, ys, s=130, color=colors[c], edgecolor="white", linewidth=1.0, label=c, zorder=3)
    for lab, mae, pr, r2, c in rows:
        ax.annotate(lab, (mae, pr), textcoords="offset points", xytext=(6, 4), fontsize=7.5, zorder=4)
    ax.set_xlabel("MAE ($/member-month)"); ax.set_ylabel("Predictive ratio (predicted / actual, log scale)")
    ax.set_yscale("log")
    ax.set_title("Supplementary Figure S3. Forecasting accuracy vs calibration across the model panel",
                 loc="left", fontweight="bold", fontsize=11, pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8, framealpha=1.0)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.22)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_OUT / f"figure_s3_frontier_panel.{ext}", bbox_inches="tight",
                    dpi=(200 if ext == "png" else None))
    plt.close(fig); print("Wrote figureS5")


if __name__ == "__main__":
    render_figure_1()
    render_figure_2()
    render_figure_3()
    render_figure_s1()
    render_figure_s5()
    render_figure_s4()
    render_figure_s3()
    print("All figures rendered to", FIG_OUT)
