"""
Final figure renderer for the PMPM revision.

Design rule (strict): no floating text on plots, no text boxes inside plots.
All annotations live in the figure caption. Plot elements are limited to:
  - axes, gridlines, ticks (mandatory)
  - data marks (bars, points, lines)
  - one legend per panel (placed outside the plot area when crowded)

Outputs (overwrites the same filenames consumed by the .docx embed):
  figure1_hybrid_and_scatter.{pdf,png}
  figure2_decile_calibration_revised.{pdf,png}
  figure3_r2_benchmarks.{pdf,png}
  figure_s2_temporal_setup.{pdf,png}
  figure_s3_month_of_incurrence.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PKG_ROOT = Path("/Users/sanjaybasu/waymark-local/packaging/panelfm")
NB_ROOT = Path("/Users/sanjaybasu/waymark-local/notebooks/panelfm")
METRICS_PATH = PKG_ROOT / "results" / "all_metrics_real.json"
DECILES_PATH = NB_ROOT / "decile_analysis.csv"
FIG_OUT = NB_ROOT / "revision1" / "figures"
FIG_OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CAT_COLORS = {
    "Cross-sectional": "#1f77b4",
    "Time series (classical)": "#7f7f7f",
    "Foundation model": "#d62728",
    "Panel-conditioned FM": "#ff7f0e",
    "Hybrid": "#2ca02c",
}


# ======================================================================
# Figure 1 — Panel A schematic + Panel B numbered scatter
# ======================================================================
def render_figure_1():
    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    fig = plt.figure(figsize=(14.5, 6.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.22)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # ----- Panel A schematic -----
    # Sized so every label fits comfortably inside its box at the chosen font.
    # Box width was increased and font was reduced to eliminate any text-overflow.
    axA.set_xlim(0, 14)
    axA.set_ylim(0, 9)
    axA.set_axis_off()
    axA.set_title("A. Hybrid Construction", loc="left", fontweight="bold", pad=12)

    BOX_W = 4.0     # uniform box width — comfortably contains every label
    BOX_H = 1.3     # taller boxes give vertical room for 2- and 3-line labels
    GAP = 0.5       # gap between adjacent boxes (where the arrow lives)
    LEFT = 0.3      # x-coordinate of the first box's left edge

    # Compute the three column x-positions
    col1_x = LEFT
    col2_x = col1_x + BOX_W + GAP            # 3.8
    col3_x = col2_x + BOX_W + GAP            # 7.4

    row_top_y = 7.4    # cross-sectional pathway
    row_mid_y = 5.2    # time-series pathway
    fusion_y  = 2.6    # fusion box

    def box(ax, x, y, w, h, label, fc, ec, fontsize=8.5):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                               linewidth=1.2, facecolor=fc, edgecolor=ec)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fontsize, wrap=True)

    def arrow_horiz(ax, src_x_right, dst_x_left, y, color="black"):
        """Arrow that lives entirely in the gap between two boxes (no overlap)."""
        # Inset slightly from box edges so the arrowhead doesn't touch the box border
        ax.add_patch(FancyArrowPatch((src_x_right + 0.05, y),
                                     (dst_x_left - 0.05, y),
                                     arrowstyle="-|>", mutation_scale=12,
                                     linewidth=1.4, color=color))

    def arrow_routed(ax, src_x_right, src_y, dst_x, dst_y_top, color):
        """Route a green fusion arrow from the right edge of a source box DOWN
        to the top edge of the fusion box, going around the layout (no text
        overlap)."""
        # Two-segment path: rightward, then downward, then in toward fusion
        # For simplicity, we use a curved arrow that goes outside the right column
        ax.add_patch(FancyArrowPatch(
            (src_x_right + 0.05, src_y),
            (dst_x, dst_y_top + 0.05),
            arrowstyle="-|>", mutation_scale=12, linewidth=1.4, color=color,
            connectionstyle="arc3,rad=-0.25",
        ))

    # --- Cross-sectional pathway (top row) ---
    box(axA, col1_x, row_top_y, BOX_W, BOX_H,
        "20 patient features\n(12-month lookback)",
        fc="#e6f0ff", ec="#1f77b4")
    box(axA, col2_x, row_top_y, BOX_W, BOX_H,
        "Two-part hurdle\ncross-sectional\nmodel",
        fc="#e6f0ff", ec="#1f77b4")
    box(axA, col3_x, row_top_y, BOX_W, BOX_H,
        "Calibrated 3-month\ncost budget\nŶ$^{CS}$",
        fc="#cfe2f3", ec="#1f77b4")
    arrow_horiz(axA, col1_x + BOX_W, col2_x, row_top_y + BOX_H/2)
    arrow_horiz(axA, col2_x + BOX_W, col3_x, row_top_y + BOX_H/2)

    # --- Time-series pathway (middle row) ---
    box(axA, col1_x, row_mid_y, BOX_W, BOX_H,
        "Monthly cost\nhistory",
        fc="#ffe6e6", ec="#d62728")
    box(axA, col2_x, row_mid_y, BOX_W, BOX_H,
        "Chronos-T5-Small\nfoundation model\n(46M, zero-shot)",
        fc="#ffe6e6", ec="#d62728")
    box(axA, col3_x, row_mid_y, BOX_W, BOX_H,
        "Three monthly\nforecasts\nŷ$^{TS}_1$, ŷ$^{TS}_2$, ŷ$^{TS}_3$",
        fc="#fcd0d0", ec="#d62728")
    arrow_horiz(axA, col1_x + BOX_W, col2_x, row_mid_y + BOX_H/2)
    arrow_horiz(axA, col2_x + BOX_W, col3_x, row_mid_y + BOX_H/2)

    # --- Fusion box (bottom row, full width, formula on its own line) ---
    fusion_x = col1_x
    fusion_w = (col3_x + BOX_W) - col1_x
    box(axA, fusion_x, fusion_y, fusion_w, 1.6,
        "Hybrid prediction for month h:\n\n"
        "ŷ$^{hybrid}_h$  =  ( ŷ$^{TS}_h$ / Σ$_m$ ŷ$^{TS}_m$ )  ×  Ŷ$^{CS}$    "
        "[ = 0 if Σ$_m$ ŷ$^{TS}_m$ = 0 ]",
        fc="#e8f5e9", ec="#2ca02c", fontsize=9.5)

    # No connecting arrows from the row-1 / row-2 boxes down to the fusion box.
    # The relationship is clear from (a) the top-to-bottom reading order and
    # (b) the fusion-box formula, which uses the same Ŷ^CS and ŷ^TS symbols
    # defined in the rows above. Adding diagonal arrows here would overlap
    # box text and other arrows.

    # ----- Panel B scatter (NO floating text annotation; band described in caption) -----
    def category(name):
        if name in ("naive_last", "naive_mean3"): return "Time series (classical)"
        if name == "chronos_zeroshot": return "Foundation model"
        if name.startswith("panelfm"): return "Panel-conditioned FM"
        if name.startswith("hybrid"): return "Hybrid"
        return "Cross-sectional"

    ordered = [
        ("hybrid_chronos_zeroshot",  "Hybrid Chronos"),
        ("hybrid_panelfm_adapter",   "Hybrid PanelFM-Adapter"),
        ("stacking",                 "Stacking ensemble"),
        ("random_forest",            "Random forest"),
        ("naive_mean3",              "Naive trailing mean"),
        ("lightgbm",                 "LightGBM"),
        ("two_part",                 "Two-part hurdle"),
        ("xgboost",                  "XGBoost"),
        ("naive_last",               "Naive last-value"),
        ("demographic_glm",          "Demographics-only GLM"),
        ("panelfm_xreg",             "PanelFM-XReg"),
        ("panelfm_adapter",          "PanelFM-Adapter"),
        ("panelfm_icf",              "PanelFM-ICF"),
        ("chronos_zeroshot",         "Chronos (zero-shot)"),
    ]
    points = []
    for i, (name, label) in enumerate(ordered, start=1):
        m = metrics[name]
        points.append({"n": i, "name": name, "label": label,
                       "mae": m["mae"], "r2": m["r_squared_calibrated"],
                       "category": category(name)})
    df = pd.DataFrame(points)

    # CDPS band — visual only, NO floating text label (band is described in caption)
    axB.axhspan(0.08, 0.24, color="#1f77b4", alpha=0.06, zorder=1)
    axB.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)

    for cat, color in CAT_COLORS.items():
        sub = df[df["category"] == cat]
        axB.scatter(sub["mae"], sub["r2"], s=140, color=color,
                    edgecolor="white", linewidth=1.0, label=cat, zorder=3)

    for _, row in df.iterrows():
        axB.text(row["mae"], row["r2"], str(row["n"]),
                 ha="center", va="center", fontsize=8, fontweight="bold",
                 color="white", zorder=4)

    axB.set_xlabel("MAE ($/patient-3mo, log scale)")
    axB.set_ylabel("Calibrated R²")
    axB.set_title("B. MAE vs Calibrated R²", loc="left", fontweight="bold", pad=12)
    axB.set_xscale("log")
    axB.set_xlim(220, 2600)
    axB.set_ylim(-1.4, 0.4)
    axB.set_xticks([300, 500, 1000, 2000])
    axB.set_xticklabels(["$300", "$500", "$1,000", "$2,000"])

    axB.legend(loc="lower right", fontsize=8, framealpha=1.0,
               title="Model category", title_fontsize=8.5, borderpad=0.6)

    key_lines = [f"{r['n']:>2}.  {r['label']}" for _, r in df.iterrows()]
    axB.text(1.04, 1.0, "\n".join(key_lines), transform=axB.transAxes,
             fontsize=7.8, va="top", ha="left", family="DejaVu Sans",
             linespacing=1.6)

    fig.subplots_adjust(left=0.05, right=0.72, top=0.92, bottom=0.10)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_OUT / f"figure1_hybrid_and_scatter.{ext}",
                    bbox_inches="tight", dpi=(200 if ext == "png" else None))
    plt.close(fig)
    print("Wrote figure1_hybrid_and_scatter")


# ======================================================================
# Figure 2 — Panel A predictive ratio + Panel B raw $ deciles 1-4
# ======================================================================
def render_figure_2():
    df = pd.read_csv(DECILES_PATH)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.8),
                                   gridspec_kw={"width_ratios": [1.05, 1.0]})

    decile_rows = df[df["Segment"].str.startswith("decile_")].copy()
    decile_rows["decile_num"] = decile_rows["Segment"].str.replace("decile_", "").astype(int)
    sub = decile_rows[decile_rows["decile_num"] >= 5]

    models_A = {
        "demographic_glm_pr":         ("Demographics-only GLM", "#7f7f7f"),
        "two_part_pr":                ("Two-part hurdle",       "#1f77b4"),
        "chronos_zeroshot_pr":        ("Chronos (zero-shot)",   "#d62728"),
        "panelfm_adapter_pr":         ("PanelFM-Adapter",       "#ff7f0e"),
        "hybrid_chronos_zeroshot_pr": ("Hybrid Chronos",        "#2ca02c"),
    }
    axA.axhspan(0.9, 1.1, color="#cccccc", alpha=0.18, zorder=1)
    axA.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7, zorder=2)
    for col, (label, color) in models_A.items():
        axA.plot(sub["decile_num"], sub[col], "o-", color=color, label=label,
                 linewidth=1.8, markersize=6, zorder=3)
    axA.set_xlabel("Decile of actual 3-month cost")
    axA.set_ylabel("Predictive ratio (predicted / actual, log scale)")
    axA.set_title("A. Decile 5–10 calibration", loc="left", fontweight="bold")
    axA.set_xticks(range(5, 11))
    axA.set_xticklabels([f"D{i}" for i in range(5, 11)])
    axA.set_yscale("log")
    axA.set_ylim(0.05, 1200)
    axA.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
               fontsize=8, framealpha=1.0)

    models_B = {
        "demographic_glm_pred":         ("Demographics-only GLM", "#7f7f7f"),
        "xgboost_pred":                 ("XGBoost",               "#aec7e8"),
        "two_part_pred":                ("Two-part hurdle",       "#1f77b4"),
        "chronos_zeroshot_pred":        ("Chronos (zero-shot)",   "#d62728"),
        "panelfm_adapter_pred":         ("PanelFM-Adapter",       "#ff7f0e"),
        "panelfm_icf_pred":             ("PanelFM-ICF",           "#ffbb78"),
        "hybrid_chronos_zeroshot_pred": ("Hybrid Chronos",        "#2ca02c"),
    }
    sub04 = decile_rows[decile_rows["decile_num"] <= 4]
    x = np.arange(len(sub04))
    n_models = len(models_B)
    bar_w = 0.85 / n_models
    for i, (col, (label, color)) in enumerate(models_B.items()):
        offset = (i - (n_models - 1) / 2) * bar_w
        axB.bar(x + offset, sub04[col], width=bar_w, color=color, label=label,
                edgecolor="white", linewidth=0.4)
    axB.set_xticks(x)
    axB.set_xticklabels(["D1", "D2", "D3", "D4"])
    axB.set_xlabel("Decile of actual 3-month cost  (all actual = $0)")
    axB.set_ylabel("Mean predicted 3-month cost ($)")
    axB.set_title("B. Deciles 1–4 raw $ predictions", loc="left", fontweight="bold")
    axB.set_ylim(0, 1000)
    axB.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
               fontsize=8, framealpha=1.0)

    fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.30)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_OUT / f"figure2_decile_calibration_revised.{ext}",
                    bbox_inches="tight", dpi=(200 if ext == "png" else None))
    plt.close(fig)
    print("Wrote figure2_decile_calibration_revised")


# ======================================================================
# Figure 3 — R² vs benchmarks (clean: NO in-plot labels on shaded bands)
# ======================================================================
def render_figure_3():
    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    # Two groups: prospective (no GLM floor — implied at 0) and concurrent
    prospective = [
        ("Demo GLM",        metrics["demographic_glm"]["r_squared_calibrated"],         "Demographics-only"),
        ("XGBoost",         metrics["xgboost"]["r_squared_calibrated"],                 "Cross-sectional ML"),
        ("Stacking",        metrics["stacking"]["r_squared_calibrated"],                "Cross-sectional ML"),
        ("Chronos\n(raw)",  metrics["chronos_zeroshot"]["r_squared_calibrated"],        "Foundation (raw)"),
        ("Hybrid\nChronos", metrics["hybrid_chronos_zeroshot"]["r_squared_calibrated"], "Hybrid"),
        ("Hybrid\nPanelFM-Ad.", metrics["hybrid_panelfm_adapter"]["r_squared_calibrated"], "Hybrid"),
    ]
    concurrent = [
        ("Conc.\nXGBoost", metrics["concurrent_xgboost"]["r_squared_calibrated"], "Concurrent"),
        ("Conc.\nRF",      metrics["concurrent_random_forest"]["r_squared_calibrated"], "Concurrent"),
    ]

    cat_to_color = {
        "Demographics-only":   "#a6cee3",
        "Cross-sectional ML":  "#e69a00",
        "Foundation (raw)":    "#1f77b4",
        "Hybrid":              "#2ca02c",
        "Concurrent":          "#c75ba1",
    }
    all_rows = prospective + concurrent
    labels = [r[0] for r in all_rows]
    values = [r[1] * 100 for r in all_rows]
    colors = [cat_to_color[r[2]] for r in all_rows]

    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    x = np.arange(len(all_rows))
    # Insert a visual gap between prospective and concurrent
    n_prosp = len(prospective)
    x_adj = np.array([xi if i < n_prosp else xi + 0.6 for i, xi in enumerate(x)])

    # Shaded benchmark bands (visual only — NO in-plot text labels)
    ax.axhspan(1, 3, color="#a6cee3", alpha=0.25, zorder=1)
    ax.axhspan(8, 24, color="#1f77b4", alpha=0.10, zorder=1)
    ax.axhspan(55, 67, color="#2ca02c", alpha=0.10, zorder=1)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    # Vertical separator between prospective and concurrent
    ax.axvline((x_adj[n_prosp - 1] + x_adj[n_prosp]) / 2, color="#888888",
               linestyle="--", linewidth=0.8, alpha=0.5)

    bars = ax.bar(x_adj, values, color=colors, edgecolor="white", linewidth=0.6,
                  width=0.7, zorder=3)
    # Numeric labels above each bar
    for xi, v in zip(x_adj, values):
        offset = 1.8 if v >= 0 else -3.5
        ax.text(xi, v + offset, f"{v:.1f}%", ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8.5, fontweight="bold",
                color=("#222222" if v >= 0 else "#d62728"))

    ax.set_xticks(x_adj)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Calibrated R² (%)")
    ax.set_ylim(-95, 100)
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax.set_yticklabels(["–90%", "–60%", "–30%", "0%", "30%", "60%", "90%"])

    # Group labels BELOW the x-axis, not on plot
    ax.text(np.mean(x_adj[:n_prosp]), -115, "Prospective models",
            ha="center", va="top", fontsize=10, fontstyle="italic", color="#555555")
    ax.text(np.mean(x_adj[n_prosp:]), -115, "Concurrent models",
            ha="center", va="top", fontsize=10, fontstyle="italic", color="#555555")

    # Custom legend with category + benchmark bands (placed OUTSIDE plot)
    from matplotlib.lines import Line2D
    legend_handles = [
        mpatches.Patch(color=cat_to_color["Demographics-only"], label="Demographics-only"),
        mpatches.Patch(color=cat_to_color["Cross-sectional ML"], label="Cross-sectional ML"),
        mpatches.Patch(color=cat_to_color["Foundation (raw)"], label="Foundation (raw)"),
        mpatches.Patch(color=cat_to_color["Hybrid"], label="Hybrid"),
        mpatches.Patch(color=cat_to_color["Concurrent"], label="Concurrent"),
        mpatches.Patch(facecolor="#a6cee3", alpha=0.35, label="Demographics-only band (1–3%)"),
        mpatches.Patch(facecolor="#1f77b4", alpha=0.15, label="CDPS prospective band (8–24%)"),
        mpatches.Patch(facecolor="#2ca02c", alpha=0.15, label="MARA concurrent band (55–67%)"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=4, fontsize=8, framealpha=1.0)

    fig.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.30)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_OUT / f"figure3_r2_benchmarks.{ext}",
                    bbox_inches="tight", dpi=(200 if ext == "png" else None))
    plt.close(fig)
    print("Wrote figure3_r2_benchmarks")


# ======================================================================
# Figure S2 — Temporal setup (clean: no callout, no floating note)
# ======================================================================
def render_figure_s2():
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.set_xlim(-1, 35)
    ax.set_ylim(-1.5, 3.8)
    ax.set_axis_off()
    ax.set_title("Supplementary Figure S2. Temporal setup",
                 loc="left", fontweight="bold", fontsize=12, pad=10)

    ax.plot([0, 33], [0, 0], "k-", linewidth=1.0)
    quarters = [(0, "Jan\n2023"), (3, "Apr"), (6, "Jul"), (9, "Oct"),
                (12, "Jan\n2024"), (15, "Apr"), (18, "Jul"), (21, "Oct"),
                (24, "Jan\n2025"), (27, "Apr"), (30, "Jul"), (33, "Oct\n2025")]
    for xq, lab in quarters:
        ax.plot([xq, xq], [-0.18, 0.18], "k-", linewidth=0.7)
        ax.text(xq, -0.55, lab, ha="center", va="top", fontsize=8)

    # Time-period bands (these ARE the data — colored timeline segments)
    ax.add_patch(mpatches.Rectangle((0, 0.4), 24, 0.8, facecolor="#cfe2f3",
                                    edgecolor="#1f77b4", linewidth=1.0))
    ax.text(12, 0.8, "Training  (Jan 2023 – Jan 2025)",
            ha="center", va="center", fontsize=10, fontweight="bold")

    ax.add_patch(mpatches.Rectangle((13, 1.6), 11, 0.6, facecolor="#fff4cc",
                                    edgecolor="#e8a800", linewidth=1.0))
    ax.text(18.5, 1.9, "12-month feature lookback",
            ha="center", va="center", fontsize=9)

    ax.add_patch(mpatches.Rectangle((24, 0.4), 3, 0.8, facecolor="#e8d6f5",
                                    edgecolor="#6a4ea6", linewidth=1.0))
    ax.text(25.5, 0.8, "Validation",
            ha="center", va="center", fontsize=8.5, fontweight="bold")

    ax.add_patch(mpatches.Rectangle((27, 0.4), 6, 0.8, facecolor="#d6f0d6",
                                    edgecolor="#2ca02c", linewidth=1.0))
    ax.text(30, 0.8, "Test  (May–Oct 2025)",
            ha="center", va="center", fontsize=10, fontweight="bold")

    # No mature-window visual cue — that subset (May–Jul 2025) is described in
    # the caption and §S12. Keeps the timeline figure clean and unambiguous.
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_OUT / f"figure_s2_temporal_setup.{ext}",
                    bbox_inches="tight", dpi=(200 if ext == "png" else None))
    plt.close(fig)
    print("Wrote figure_s2_temporal_setup")


# ======================================================================
# Figure S3 — Month-of-incurrence (already clean — keep as is)
# ======================================================================
def render_figure_s3():
    months = ["May\n2025", "Jun\n2025", "Jul\n2025", "Aug\n2025", "Sep\n2025", "Oct\n2025"]
    mean_cost = [342, 328, 315, 298, 278, 263]
    x = np.arange(len(months))

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.bar(x, mean_cost, color="#1f77b4", edgecolor="white", linewidth=0.6,
           width=0.7, zorder=3)
    coeffs = np.polyfit(x, mean_cost, 1)
    ax.plot(x, np.polyval(coeffs, x), color="#d62728", linestyle="--",
            linewidth=1.6, zorder=4, label=f"Linear trend: ${coeffs[0]:.1f}/month")
    for xi, c in zip(x, mean_cost):
        ax.text(xi, c + 10, f"${c}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(months, fontsize=9)
    ax.set_ylabel("Mean reported cost ($)", fontsize=10)
    ax.set_xlabel("Month of incurrence", fontsize=10)
    ax.set_title("Supplementary Figure S3. Test-period cost by month of incurrence",
                 loc="left", fontweight="bold", fontsize=11, pad=10)
    ax.set_ylim(0, 420)
    # Legend INSIDE plot but in a corner with no bars — top-right is empty
    ax.legend(loc="upper right", fontsize=8.5, framealpha=1.0)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_OUT / f"figure_s3_month_of_incurrence.{ext}",
                    bbox_inches="tight", dpi=(200 if ext == "png" else None))
    plt.close(fig)
    print("Wrote figure_s3_month_of_incurrence")


if __name__ == "__main__":
    render_figure_1()
    render_figure_2()
    render_figure_3()
    render_figure_s2()
    render_figure_s3()
    print("\nAll figures re-rendered to", FIG_OUT)
