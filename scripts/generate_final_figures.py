#!/usr/bin/env python3
"""
Generate publication-quality figures for Medical Care manuscript.

Produces exactly 3 figures for the actuarial / health services research paper
on time series foundation models for healthcare cost prediction.

Output (saved to ../results/):
  - figure1_model_performance.pdf / .png   (Panel A: MAE bars, Panel B: MAE vs R² scatter)
  - figure2_decile_calibration.pdf / .png  (Predictive ratios by cost decile)
  - figure3_r2_benchmarks.pdf / .png       (Prospective + concurrent R² with published bands)

Style: Medical Care requirements — 1200+ dpi line art, sans-serif fonts, 7.5-inch width.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# ── global style ─────────────────────────────────────────────────────────────
FONT_FAMILY = "Arial"
# Fallback if Arial is not available
try:
    from matplotlib.font_manager import findfont, FontProperties
    findfont(FontProperties(family=FONT_FAMILY), fallback_to_default=False)
except ValueError:
    FONT_FAMILY = "Helvetica"
    try:
        findfont(FontProperties(family=FONT_FAMILY), fallback_to_default=False)
    except ValueError:
        FONT_FAMILY = "DejaVu Sans"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT_FAMILY, "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "pdf.fonttype": 42,       # TrueType in PDF — required by many journals
    "ps.fonttype": 42,
})

# ── colorblind-friendly palette ──────────────────────────────────────────────
# Adapted from Wong (2011) Nature Methods colorblind-safe palette
COLORS = {
    "cs":          "#E69F00",   # amber / orange — cross-sectional ML
    "ts":          "#999999",   # gray — classical time series
    "foundation":  "#0072B2",   # blue — foundation models (raw)
    "hybrid":      "#009E73",   # teal/green — hybrid models
    "concurrent":  "#CC79A7",   # rose/mauve — concurrent models
    "demo":        "#56B4E9",   # sky blue — demographics-only
}

CATEGORY_LABELS = {
    "cs":          "Cross-sectional ML",
    "ts":          "Time-series baselines",
    "foundation":  "Foundation models (raw)",
    "hybrid":      "Hybrid (foundation + CS)",
    "concurrent":  "Concurrent models",
    "demo":        "Demographics-only",
}

# ── model metadata ───────────────────────────────────────────────────────────
# (display_name, internal_key, category)
ALL_MODELS = [
    ("Demographics GLM",    "demographic_glm",              "demo"),
    ("XGBoost",             "xgboost",                      "cs"),
    ("Random Forest",       "random_forest",                "cs"),
    ("LightGBM",           "lightgbm",                      "cs"),
    ("Two-Part",            "two_part",                      "cs"),
    ("Stacking",            "stacking",                      "cs"),
    ("Naive Last",          "naive_last",                    "ts"),
    ("Naive Mean",          "naive_mean3",                   "ts"),
    ("ARIMA",               "arima",                         "ts"),
    ("Chronos",             "chronos_zeroshot",              "foundation"),
    ("PanelFM-XReg",        "panelfm_xreg",                 "foundation"),
    ("PanelFM-Adapter",     "panelfm_adapter",              "foundation"),
    ("PanelFM-ICF",         "panelfm_icf",                  "foundation"),
    ("Hybrid Chronos",      "hybrid_chronos_zeroshot",       "hybrid"),
    ("Hybrid PanelFM-Adapter","hybrid_panelfm_adapter",     "hybrid"),
    ("Hybrid PanelFM-ICF",  "hybrid_panelfm_icf",           "hybrid"),
    ("Hybrid PanelFM-XReg", "hybrid_panelfm_xreg",          "hybrid"),
    ("Conc. Demo GLM",     "concurrent_demographic_glm",     "concurrent"),
    ("Conc. XGBoost",       "concurrent_xgboost",           "concurrent"),
    ("Conc. RF",            "concurrent_random_forest",      "concurrent"),
    ("Conc. LightGBM",      "concurrent_lightgbm",          "concurrent"),
    ("Conc. Two-Part",      "concurrent_two_part",          "concurrent"),
]

# ── data loading ─────────────────────────────────────────────────────────────

def load_summary():
    df = pd.read_csv(RESULTS_DIR / "actuarial_summary.csv")
    df = df.set_index("Model")
    return df


def load_bootstrap():
    with open(RESULTS_DIR / "bootstrap_cis.json") as f:
        return json.load(f)


def load_actuarial_metrics():
    with open(RESULTS_DIR / "actuarial_metrics.json") as f:
        return json.load(f)


def load_decile_csv():
    return pd.read_csv(RESULTS_DIR / "decile_analysis.csv")


# ── Figure 1: Model Performance Overview (two-panel) ────────────────────────

def figure1(summary, bootstrap, actuarial):
    """
    Panel A — grouped horizontal bar chart of MAE by model category.
    Panel B — scatter of MAE vs calibrated R² with Pareto frontier.
    """
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.5, 5.5),
        gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.38},
    )

    # ── Panel A: MAE bar chart ───────────────────────────────────────────
    # Define display order: categories top to bottom, models sorted within group
    category_order = ["demo", "cs", "ts", "foundation", "hybrid", "concurrent"]
    category_display = {
        "demo": "Demographics",
        "cs": "Cross-sectional ML",
        "ts": "Time-series baselines",
        "foundation": "Foundation (raw)",
        "hybrid": "Hybrid",
        "concurrent": "Concurrent",
    }

    bar_data = []  # (display_name, key, category, mae, ci_lo, ci_hi)
    for cat in category_order:
        group = [(dn, k, c) for dn, k, c in ALL_MODELS if c == cat and k in summary.index]
        # sort within group by MAE ascending
        group_with_mae = [(dn, k, c, summary.loc[k, "MAE"]) for dn, k, c in group]
        group_with_mae.sort(key=lambda x: x[3])
        for dn, k, c, mae_val in group_with_mae:
            ci_lo = mae_val - bootstrap.get(k, {}).get("mae_ci_lower", mae_val)
            ci_hi = bootstrap.get(k, {}).get("mae_ci_upper", mae_val) - mae_val
            bar_data.append((dn, k, c, mae_val, max(ci_lo, 0), max(ci_hi, 0)))

    labels_a = [d[0] for d in bar_data]
    maes_a = [d[3] for d in bar_data]
    ci_lo_a = [d[4] for d in bar_data]
    ci_hi_a = [d[5] for d in bar_data]
    colors_a = [COLORS[d[2]] for d in bar_data]

    y_pos = np.arange(len(labels_a))
    bars = ax_a.barh(y_pos, maes_a, color=colors_a, edgecolor="white",
                     height=0.65, linewidth=0.5, zorder=2)
    ax_a.errorbar(maes_a, y_pos, xerr=[ci_lo_a, ci_hi_a],
                  fmt="none", ecolor="#333333", capsize=2.5, linewidth=0.8, zorder=3)

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(labels_a, fontsize=7.5)
    ax_a.set_xlabel("Mean Absolute Error ($/patient-month)", fontsize=9)
    ax_a.invert_yaxis()
    ax_a.xaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    ax_a.set_xlim(0, max(maes_a) * 1.12)

    # Add thin separators between groups
    cum = 0
    for cat in category_order:
        n = sum(1 for d in bar_data if d[2] == cat)
        if cum > 0:
            ax_a.axhline(y=cum - 0.5, color="#cccccc", linewidth=0.5, linestyle="-", zorder=1)
        cum += n

    # Panel label
    ax_a.text(-0.14, 1.02, "A", transform=ax_a.transAxes,
              fontsize=14, fontweight="bold", va="bottom")

    # Legend for categories (compact)
    legend_patches = []
    seen = set()
    for _, _, cat, _, _, _ in bar_data:
        if cat not in seen:
            seen.add(cat)
            legend_patches.append(Patch(facecolor=COLORS[cat], edgecolor="none",
                                        label=CATEGORY_LABELS[cat]))
    ax_a.legend(handles=legend_patches, loc="lower right", fontsize=6.5,
                frameon=True, framealpha=0.9, edgecolor="#cccccc",
                handlelength=1.0, handletextpad=0.4, borderpad=0.4)

    # ── Panel B: MAE vs Calibrated R² scatter ────────────────────────────
    scatter_data = []  # (display_name, key, category, mae, r2_cal)
    for dn, k, cat in ALL_MODELS:
        if k in summary.index:
            mae_val = summary.loc[k, "MAE"]
            r2_val = summary.loc[k, "R² (calibrated)"]
            if np.isfinite(mae_val) and np.isfinite(r2_val):
                scatter_data.append((dn, k, cat, mae_val, r2_val))

    # Plot all points
    for dn, k, cat, mae_val, r2_val in scatter_data:
        ax_b.scatter(mae_val, r2_val, c=COLORS[cat], s=55, zorder=3,
                     edgecolors="white", linewidth=0.6, alpha=0.92)

    # Label offsets carefully tuned to avoid overlap
    label_offsets = {
        "Demographics GLM": (8, 6),
        "XGBoost": (8, 8),
        "Two-Part": (8, -12),
        "Stacking": (-48, 12),
        "Naive Mean": (-55, 10),
        "Chronos": (8, -10),
        "PanelFM-Adapter": (8, 8),
        "Hybrid Chronos": (-72, -12),
        "Hybrid PanelFM-Adapter": (-100, 12),
        "Conc. XGBoost": (8, -10),
        "Conc. RF": (8, 8),
    }
    # Only label key models to avoid clutter
    key_models_scatter = {
        "Demographics GLM", "XGBoost", "Two-Part", "Stacking",
        "Naive Mean", "Chronos", "PanelFM-Adapter",
        "Hybrid Chronos", "Hybrid PanelFM-Adapter",
        "Conc. XGBoost", "Conc. RF",
    }
    for dn, k, cat, mae_val, r2_val in scatter_data:
        if dn in key_models_scatter:
            dx, dy = label_offsets.get(dn, (8, 0))
            use_arrow = abs(dx) > 20 or abs(dy) > 12
            ax_b.annotate(
                dn, (mae_val, r2_val),
                xytext=(dx, dy), textcoords="offset points",
                fontsize=5.5, color="#444444",
                arrowprops=dict(arrowstyle="-", color="#aaaaaa",
                                linewidth=0.4, shrinkA=0, shrinkB=3)
                if use_arrow else None,
            )

    # CDPS prospective reference band (R² = 0.08 – 0.24)
    ax_b.axhspan(0.08, 0.24, color="#0072B2", alpha=0.07, zorder=0)
    ax_b.text(0.97, 0.98, "CDPS prospective\n(R²=0.08–0.24)",
              transform=ax_b.transAxes, fontsize=5.5, color="#0072B2",
              alpha=0.65, va="top", ha="right")

    # Vertical reference at raw Chronos MAE
    chronos_mae = summary.loc["chronos_zeroshot", "MAE"]
    ax_b.axvline(x=chronos_mae, color="#0072B2", linewidth=0.6, linestyle=":",
                 alpha=0.35, zorder=1)

    # Pareto frontier: models NOT dominated on (lower MAE, higher R²)
    pts = [(mae_val, r2_val, dn) for dn, k, cat, mae_val, r2_val in scatter_data]
    pareto = []
    for m, r, n in pts:
        dominated = False
        for m2, r2, n2 in pts:
            if m2 <= m and r2 >= r and (m2 < m or r2 > r):
                dominated = True
                break
        if not dominated:
            pareto.append((m, r, n))
    if len(pareto) >= 2:
        pareto.sort(key=lambda x: x[0])
        pm = [p[0] for p in pareto]
        pr_vals = [p[1] for p in pareto]
        ax_b.plot(pm, pr_vals, color="#333333", linewidth=1.0, linestyle="--",
                  alpha=0.35, zorder=2)
        # Mark Pareto-optimal points with a subtle ring
        ax_b.scatter(pm, pr_vals, s=120, facecolors="none", edgecolors="#333333",
                     linewidth=0.6, alpha=0.3, zorder=2)

    ax_b.set_xlabel("MAE ($/patient-month)", fontsize=9)
    ax_b.set_ylabel("Calibrated R²", fontsize=9)
    max_mae = max(d[3] for d in scatter_data)
    ax_b.set_xlim(-50, max_mae * 1.08)
    # Set y-axis to focus on the interesting range while showing all points
    all_r2 = [d[4] for d in scatter_data]
    ax_b.set_ylim(min(all_r2) - 0.15, max(all_r2) + 0.12)

    ax_b.text(-0.14, 1.02, "B", transform=ax_b.transAxes,
              fontsize=14, fontweight="bold", va="bottom")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(RESULTS_DIR / f"figure1_model_performance.{ext}",
                    dpi=300 if ext == "png" else None)
    plt.close()
    print(f"  Figure 1 saved: {RESULTS_DIR / 'figure1_model_performance.pdf'}")


# ── Figure 2: Actuarial Calibration (Decile Analysis) ───────────────────────

def figure2(actuarial, decile_csv):
    """
    Predictive ratios by actual-cost decile for key models.
    Includes top-5% and top-1% segments on the right.
    Uses two sub-panels: left for deciles 6-10 (linear), right for top segments.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    # Models to display (display_name, internal_key, color, marker, linewidth, linestyle)
    key_models = [
        ("Demographics GLM",     "demographic_glm",         COLORS["demo"],       "s", 1.0, "-"),
        ("XGBoost",              "xgboost",                 COLORS["cs"],         "^", 1.0, "-"),
        ("Two-Part",             "two_part",                COLORS["cs"],         "v", 1.0, "-"),
        ("Chronos (raw)",        "chronos_zeroshot",        COLORS["foundation"], "o", 1.3, "-"),
        ("PanelFM-Adapter (raw)","panelfm_adapter",         COLORS["foundation"], "D", 1.3, "-"),
        ("Hybrid Chronos",       "hybrid_chronos_zeroshot", COLORS["hybrid"],     "P", 1.5, "-"),
        ("Conc. RF",             "concurrent_random_forest",COLORS["concurrent"], "X", 1.0, "-"),
    ]

    # Show deciles 6-10, plus top 5% and top 1%
    # Skip decile_5 (PR values extremely large / infinite for many models)
    segments = ["decile_6", "decile_7", "decile_8", "decile_9",
                "decile_10", "top_5pct", "top_1pct"]
    x_labels = ["D6", "D7", "D8", "D9", "D10", "Top 5%", "Top 1%"]
    x_pos = np.arange(len(segments))

    # Acceptable range shading (0.80–1.20 broader, 0.90–1.10 ideal)
    ax.axhspan(0.80, 1.20, color="#e8e8e8", alpha=0.5, zorder=0)
    ax.axhspan(0.90, 1.10, color="#d0d0d0", alpha=0.5, zorder=0)
    ax.axhline(y=1.0, color="#555555", linewidth=0.9, linestyle="-", zorder=1)

    for display_name, key, color, marker, lw, ls in key_models:
        if key not in actuarial:
            continue
        decile_data = actuarial[key].get("decile_analysis", {})
        pr_vals = []
        for seg in segments:
            entry = decile_data.get(seg, {})
            pr = entry.get("predictive_ratio", None)
            pr_vals.append(pr)

        # Filter out None/NaN entries but keep array aligned for plotting
        valid_x = []
        valid_pr = []
        for i, pr in enumerate(pr_vals):
            if pr is not None and np.isfinite(pr):
                valid_x.append(x_pos[i])
                valid_pr.append(pr)

        if not valid_pr:
            continue

        ax.plot(valid_x, valid_pr, color=color, marker=marker, markersize=5.5,
                linewidth=lw, linestyle=ls,
                markeredgecolor="white", markeredgewidth=0.5,
                label=display_name, zorder=3, alpha=0.92)

    # Focus on the actuarially meaningful range
    ax.set_ylim(0, 4.0)

    # Add a note about truncation and omitted deciles
    ax.text(0.01, 0.97, "Deciles 1–5 omitted\n(PR undefined or > 10)",
            transform=ax.transAxes, fontsize=5.5, color="#888888",
            va="top", ha="left", fontstyle="italic")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=8.5)
    ax.set_xlabel("Actual Cost Segment", fontsize=9)
    ax.set_ylabel("Predictive Ratio (predicted / actual)", fontsize=9)

    # Vertical separator before top percentiles
    ax.axvline(x=3.5, color="#bbbbbb", linewidth=0.6, linestyle=":", zorder=1)
    ax.text(5.0, 3.85, "High-cost\ntail segments",
            fontsize=6, color="#888888", ha="center", va="top",
            fontstyle="italic")

    # Annotation for the ideal band
    ax.text(0.99, 0.295, "Ideal: PR = 0.90–1.10",
            transform=ax.transAxes, fontsize=5.5, color="#666666",
            va="center", ha="right", fontstyle="italic")

    # Clean legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", fontsize=6.5,
              frameon=True, framealpha=0.95, edgecolor="#cccccc",
              ncol=2, columnspacing=0.8,
              handlelength=2.0, handletextpad=0.4, borderpad=0.5)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(RESULTS_DIR / f"figure2_decile_calibration_final.{ext}",
                    dpi=300 if ext == "png" else None)
    plt.close()
    print(f"  Figure 2 saved: {RESULTS_DIR / 'figure2_decile_calibration_final.pdf'}")


# ── Figure 3: Prospective vs Concurrent R² with Published Benchmarks ────────

def figure3(summary):
    """
    Grouped bar chart: prospective and concurrent R² for key models,
    with horizontal reference bands from published literature.
    Uses a broken y-axis to handle the Chronos negative R² gracefully.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    # Models to show (display_name, key, is_concurrent)
    models_prosp = [
        ("Demo\nGLM",             "demographic_glm",         False),
        ("XGBoost",               "xgboost",                 False),
        ("Stacking",              "stacking",                False),
        ("Chronos\n(raw)",        "chronos_zeroshot",        False),
        ("Hybrid\nChronos",       "hybrid_chronos_zeroshot", False),
        ("Hybrid\nPanelFM-Ad.",   "hybrid_panelfm_adapter",  False),
    ]
    models_conc = [
        ("Conc.\nXGBoost",       "concurrent_xgboost",       True),
        ("Conc.\nRF",            "concurrent_random_forest",  True),
    ]

    all_models_fig3 = models_prosp + models_conc
    n = len(all_models_fig3)
    x = np.arange(n)

    # Get calibrated R² values (as percentages)
    r2_cal = []
    bar_colors = []
    for dn, key, is_conc in all_models_fig3:
        if key in summary.index:
            val = summary.loc[key, "R² (calibrated)"] * 100
            r2_cal.append(val)
        else:
            r2_cal.append(0)

        if is_conc:
            bar_colors.append(COLORS["concurrent"])
        elif "hybrid" in key.lower():
            bar_colors.append(COLORS["hybrid"])
        elif "chronos" in key.lower() or "panelfm" in key.lower():
            bar_colors.append(COLORS["foundation"])
        elif key == "demographic_glm":
            bar_colors.append(COLORS["demo"])
        else:
            bar_colors.append(COLORS["cs"])

    # Clip Chronos raw R² for display; show actual value as annotation
    r2_display = []
    for val in r2_cal:
        if val < -10:
            r2_display.append(-8)  # clip for visual; annotate with real value
        else:
            r2_display.append(val)

    # Published benchmark bands
    # Demographics-only: 1-3%
    ax.axhspan(1, 3, color="#999999", alpha=0.12, zorder=0)
    # Place band labels at left edge of the bands, avoiding bar overlap
    ax.text(0.02, 0.035, "Demographics-only (1\u20133%)", transform=ax.transAxes,
            fontsize=5.5, color="#777777", ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.8))

    # CDPS prospective: 8-24%
    ax.axhspan(8, 24, color="#0072B2", alpha=0.09, zorder=0)
    ax.text(0.02, 0.31, "CDPS prospective (8\u201324%)", transform=ax.transAxes,
            fontsize=5.5, color="#0072B2", ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.8))

    # MARA concurrent: 55-67%
    ax.axhspan(55, 67, color="#009E73", alpha=0.09, zorder=0)
    ax.text(0.02, 0.645, "MARA concurrent (55\u201367%)", transform=ax.transAxes,
            fontsize=5.5, color="#009E73", ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.8))

    bars = ax.bar(x, r2_display, color=bar_colors, width=0.58, edgecolor="white",
                  linewidth=0.5, zorder=2)

    # Add value labels on top of bars
    for i, (val, dval, bar) in enumerate(zip(r2_cal, r2_display, bars)):
        cx = bar.get_x() + bar.get_width() / 2
        if val < -10:
            # Clipped bar: show real value above the bar (outside)
            ax.text(cx, 1.5, f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=6,
                    fontweight="medium", color=COLORS["foundation"])
            # Down-arrow inside bar
            ax.annotate("", xy=(cx, -11.5),
                        xytext=(cx, -4),
                        arrowprops=dict(arrowstyle="->", color="white",
                                        linewidth=1.0))
        elif val < 0:
            ax.text(cx, val - 1.5,
                    f"{val:.1f}%", ha="center", va="top", fontsize=6.5,
                    fontweight="medium", color="#333333")
        else:
            label_y = val + 1.2
            ax.text(cx, label_y,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=6.5,
                    fontweight="medium", color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([dn for dn, _, _ in all_models_fig3], fontsize=7.5,
                       rotation=0, ha="center", linespacing=0.9)
    ax.set_ylabel("Calibrated R² (%)", fontsize=9.5)
    ax.set_ylim(-12, 95)

    # Custom y-axis ticks avoiding the clipped region
    yticks = [-10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t}%" for t in yticks])

    # Horizontal baseline at 0
    ax.axhline(y=0, color="#aaaaaa", linewidth=0.5, zorder=1)

    # Vertical separator between prospective and concurrent
    sep_x = len(models_prosp) - 0.5
    ax.axvline(x=sep_x, color="#999999", linewidth=0.8, linestyle="--", zorder=1)

    # Bracket labels below the x-axis tick labels (use axes transform for y)
    # These will be placed using fig.text after tight_layout
    bracket_info = [
        ((len(models_prosp) - 1) / 2, "Prospective models"),
        (sep_x + len(models_conc) / 2, "Concurrent"),
    ]
    # Store for post-layout placement
    fig._bracket_info = bracket_info
    fig._bracket_ax = ax
    fig._bracket_n_prosp = len(models_prosp)
    fig._bracket_n_conc = len(models_conc)

    # Legend
    legend_handles = [
        Patch(facecolor=COLORS["demo"], edgecolor="none", label="Demographics-only"),
        Patch(facecolor=COLORS["cs"], edgecolor="none", label="Cross-sectional ML"),
        Patch(facecolor=COLORS["foundation"], edgecolor="none", label="Foundation (raw)"),
        Patch(facecolor=COLORS["hybrid"], edgecolor="none", label="Hybrid"),
        Patch(facecolor=COLORS["concurrent"], edgecolor="none", label="Concurrent"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=6.5,
              frameon=True, framealpha=0.95, edgecolor="#cccccc",
              ncol=3, columnspacing=0.8,
              handlelength=1.0, handletextpad=0.4, borderpad=0.4)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22)

    # Add bracket labels below x-tick labels using figure coordinates
    for data_x, label_text in fig._bracket_info:
        # Convert data x to axes fraction, then to figure coords
        disp = ax.transData.transform((data_x, 0))
        fig_coord = fig.transFigure.inverted().transform(disp)
        fig.text(fig_coord[0], 0.02, label_text,
                 fontsize=7.5, ha="center", va="bottom", color="#555555",
                 fontstyle="italic")

    for ext in ("pdf", "png"):
        fig.savefig(RESULTS_DIR / f"figure3_r2_benchmarks.{ext}",
                    dpi=300 if ext == "png" else None)
    plt.close()
    print(f"  Figure 3 saved: {RESULTS_DIR / 'figure3_r2_benchmarks.pdf'}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data ...")
    summary = load_summary()
    bootstrap = load_bootstrap()
    actuarial = load_actuarial_metrics()
    decile_csv = load_decile_csv()

    print("Generating Figure 1: Model Performance Overview ...")
    figure1(summary, bootstrap, actuarial)

    print("Generating Figure 2: Actuarial Calibration (Decile Analysis) ...")
    figure2(actuarial, decile_csv)

    print("Generating Figure 3: Prospective vs Concurrent R² ...")
    figure3(summary)

    print("Done. All figures saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
