#!/usr/bin/env python3
"""
Decision-relevant analyses for the second revision.

Forecasting error, calibration, and discrimination trade off against each other, so
no single dollar metric crowns a winner. But Waymark's operational question is a
decision: which members to enroll in care management under a fixed capacity. For that
decision, rank is what matters. This script reframes the evaluation accordingly:

  1. Cost-captured-at-capacity: at care-management capacities of 5/10/15/20% of members,
     what fraction of the true high-cost members, and of total realized cost, does each
     model's ranking capture?
  2. Decision-curve analysis (net benefit; Vickers 2006): clinical-utility curve over
     threshold probabilities, using 5-fold cross-validated isotonic-calibrated risk.
  3. Isotonic recalibration: any model's predictions can be made budget-neutral
     (predictive ratio -> 1.0) post hoc, at a cost in MAE -- showing calibration is a
     separable axis and confirming the mean-vs-median trade-off.
  4. Pre-specified paired-bootstrap comparisons (3-month-total absolute error) with
     Holm-Bonferroni control for multiplicity.

Reads results/per_patient_test_disjoint.json (+ per_patient_tsfm.json if present).
Output: results/decision_analysis.json
"""
import sys, json
from pathlib import Path
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

PKG_ROOT = Path(__file__).resolve().parents[2]
RES = PKG_ROOT / "revision2" / "results"
SEED = 20260617
CAPACITIES = [0.05, 0.10, 0.15, 0.20]


def load():
    pp = json.loads((RES / "per_patient_test_disjoint.json").read_text())
    models = pp["models"]
    if (RES / "per_patient_tsfm.json").exists():
        for m, d in json.loads((RES / "per_patient_tsfm.json").read_text()).items():
            models[m] = d
    actual = pp["actual_total"]
    label = pp["high_cost_label"]
    return models, actual, label


def cost_captured(score_map, actual, label):
    pids = [p for p in score_map if p in actual]
    s = np.array([score_map[p] for p in pids])
    y = np.array([actual[p] for p in pids])
    hc = np.array([label[p] for p in pids])
    order = np.argsort(-s)
    n = len(pids); total_cost = y.sum(); total_hc = hc.sum()
    out = {}
    for c in CAPACITIES:
        k = max(1, int(round(n * c)))
        idx = order[:k]
        out[f"cap_{int(c*100)}pct"] = {
            "cost_captured_frac": float(y[idx].sum() / total_cost) if total_cost > 0 else None,
            "highcost_recall": float(hc[idx].sum() / total_hc) if total_hc > 0 else None,
            "ppv": float(hc[idx].mean()),
        }
    return out


def cv_isotonic_prob(score_map, label, actual):
    """5-fold CV isotonic calibration of a score to the high-cost label -> honest P(high-cost)."""
    pids = [p for p in score_map if p in label]
    s = np.array([score_map[p] for p in pids]); y = np.array([label[p] for p in pids])
    prob = np.zeros(len(pids))
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, te in kf.split(s):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(s[tr], y[tr]); prob[te] = ir.predict(s[te])
    return pids, prob, y


def net_benefit(score_map, label, actual, thresholds):
    pids, prob, y = cv_isotonic_prob(score_map, label, actual)
    n = len(y); prev = y.mean()
    nb = {}
    for pt in thresholds:
        treat = prob >= pt
        tp = np.sum(treat & (y == 1)); fp = np.sum(treat & (y == 0))
        nb[f"pt_{pt:.2f}"] = float(tp / n - fp / n * (pt / (1 - pt)))
    nb_all = {f"pt_{pt:.2f}": float(prev - (1 - prev) * (pt / (1 - pt))) for pt in thresholds}
    return nb, nb_all, float(prev)


def recalibrate(score_map, actual):
    """Isotonic recalibration of predicted total to actual total (5-fold CV). Returns PR/MAE before/after."""
    pids = [p for p in score_map if p in actual]
    s = np.array([score_map[p] for p in pids]); y = np.array([actual[p] for p in pids])
    pr_before = float(s.mean() / y.mean()) if y.mean() > 0 else None
    mae_before = float(np.mean(np.abs(s - y)))
    cal = np.zeros(len(pids)); kf = KFold(5, shuffle=True, random_state=SEED)
    for tr, te in kf.split(s):
        ir = IsotonicRegression(out_of_bounds="clip"); ir.fit(s[tr], y[tr]); cal[te] = ir.predict(s[te])
    return {"pr_before": pr_before, "pr_after": float(cal.mean() / y.mean()),
            "mae_total_before": mae_before, "mae_total_after": float(np.mean(np.abs(cal - y)))}


def paired_bootstrap(a_map, b_map, actual, n_boot=10000, seed=SEED):
    """Paired bootstrap on 3-month-total absolute error: mean(|b-y|) - mean(|a-y|)."""
    pids = [p for p in a_map if p in b_map and p in actual]
    y = np.array([actual[p] for p in pids])
    ea = np.abs(np.array([a_map[p] for p in pids]) - y)
    eb = np.abs(np.array([b_map[p] for p in pids]) - y)
    diff = eb - ea; n = len(diff); rng = np.random.default_rng(seed)
    boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"mean_diff_total_abs_err": float(diff.mean()),
            "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p_value": float(min(p, 1.0)), "n": n}


def holm(pvals):
    items = sorted(pvals.items(), key=lambda x: x[1]); m = len(items); adj = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min((m - i) * p, 1.0)); adj[k] = running
    return adj


def main():
    models, actual, label = load()
    targeting_models = [m for m in ["two_part", "stacking", "random_forest", "tweedie",
                                    "lightgbm", "xgboost", "quantile_median_cs", "chronos_zeroshot",
                                    "naive_mean3", "hybrid_gated", "chronos_bolt_base_median",
                                    "timesfm_2p5", "tabpfn_regressor"]
                        if m in models]
    out = {"cost_captured": {}, "net_benefit": {}, "recalibration": {}, "paired_tests": {}}

    for m in targeting_models:
        out["cost_captured"][m] = cost_captured(models[m], actual, label)
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.30]
    nb_all_ref = None
    for m in targeting_models:
        nb, nb_all, prev = net_benefit(models[m], label, actual, thresholds)
        out["net_benefit"][m] = nb; nb_all_ref = nb_all
    out["net_benefit"]["_treat_all"] = nb_all_ref
    out["net_benefit"]["_treat_none"] = {f"pt_{pt:.2f}": 0.0 for pt in thresholds}
    out["net_benefit"]["_prevalence"] = prev

    for m in ["chronos_zeroshot", "timesfm_2p5", "tweedie", "tabpfn_regressor", "hybrid_gated", "two_part"]:
        if m in models:
            out["recalibration"][m] = recalibrate(models[m], actual)

    # Pre-specified comparisons (3-month-total absolute error), Holm-corrected.
    pairs = {
        "gated_hybrid_vs_naive_mean3": ("hybrid_gated", "naive_mean3"),
        "gated_hybrid_vs_two_part": ("hybrid_gated", "two_part"),
        "tweedie_vs_two_part": ("tweedie", "two_part"),
        "chronos_vs_best_cs_stacking": ("chronos_zeroshot", "stacking"),
        "tabpfn_vs_tweedie": ("tabpfn_regressor", "tweedie"),
        "tabpfn_vs_quantile_median": ("tabpfn_regressor", "quantile_median_cs"),
        "timesfm_vs_chronos": ("timesfm_2p5", "chronos_zeroshot"),
        "gated_hybrid_vs_stacking": ("hybrid_gated", "stacking"),
        "panelfm_adapter_vs_chronos": ("panelfm_adapter", "chronos_zeroshot"),
        "two_part_vs_lightgbm": ("two_part", "lightgbm"),
    }
    raw = {}
    for name, (a, b) in pairs.items():
        if a in models and b in models:
            r = paired_bootstrap(models[a], models[b], actual)
            out["paired_tests"][name] = r; raw[name] = r["p_value"]
    adj = holm(raw)
    for name in out["paired_tests"]:
        out["paired_tests"][name]["p_holm"] = adj.get(name)

    (RES / "decision_analysis.json").write_text(json.dumps(out, indent=2))

    # console summary
    print("=== Cost captured at 10% capacity (recall of high-cost / % total cost / PPV) ===")
    for m in targeting_models:
        c = out["cost_captured"][m]["cap_10pct"]
        print(f"  {m:26s} recall {c['highcost_recall']:.2f}  cost {c['cost_captured_frac']:.2f}  ppv {c['ppv']:.2f}")
    print("\n=== Net benefit at p_t=0.10 (vs treat-all {:.3f}) ===".format(nb_all_ref["pt_0.10"]))
    for m in targeting_models:
        print(f"  {m:26s} {out['net_benefit'][m]['pt_0.10']:.3f}")
    print("\n=== Recalibration (PR before->after; total-MAE before->after) ===")
    for m, r in out["recalibration"].items():
        print(f"  {m:20s} PR {r['pr_before']:.2f}->{r['pr_after']:.2f}  MAE_total {r['mae_total_before']:.0f}->{r['mae_total_after']:.0f}")
    print("\n=== Paired comparisons (3-mo-total abs err diff; +=second better) ===")
    for name, r in out["paired_tests"].items():
        print(f"  {name:34s} d={r['mean_diff_total_abs_err']:+.1f} [{r['ci'][0]:+.1f},{r['ci'][1]:+.1f}] p_holm={r['p_holm']:.3f}")
    print(f"\nWrote {RES/'decision_analysis.json'}")


if __name__ == "__main__":
    main()
