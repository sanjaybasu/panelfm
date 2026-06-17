#!/usr/bin/env python3
"""
Principled-hybrid search under the member-disjoint design.

The post-processing hybrid in the main analysis distributes the cross-sectional
3-month budget across all members, which overrides the foundation model's correct
near-zero predictions for zero-cost members and so forfeits the foundation model's
mean-absolute-error advantage. This script searches for a hybrid construction that
keeps the foundation model's zero-cost accuracy while inheriting the cross-sectional
model's between-member discrimination, with every tuning decision made on the
validation members and every reported number on the held-out test members.

Reuses the Chronos monthly forecasts already saved in results/_torch_workdir.
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import run_member_disjoint as R
from src.models.baselines import TwoPartModel
from src.evaluation.metrics import r_squared_calibrated, predictive_ratio, high_cost_identification

HORIZON = R.HORIZON
OUT = R.OUT_DIR
WD = OUT / "_torch_workdir"


def permonth_mae(pred3, act3):
    pids = sorted(set(pred3) & set(act3))
    return float(np.mean([np.mean(np.abs(pred3[p] - act3[p])) for p in pids]))


def totals(pred3, act3):
    pids = sorted(set(pred3) & set(act3))
    yt = np.array([float(np.sum(act3[p])) for p in pids])
    yp = np.array([float(np.sum(pred3[p])) for p in pids])
    return yt, yp


def evalp(pred3, act3):
    yt, yp = totals(pred3, act3)
    hc = high_cost_identification(yt, yp).get("top_10pct", {})
    return {"mae": permonth_mae(pred3, act3), "r2cal": r_squared_calibrated(yt, yp),
            "pr": predictive_ratio(yt, yp)["overall"], "auroc": hc.get("c_statistic", np.nan),
            "n": len(yt)}


def shape_from(ts_vec):
    s = ts_vec.sum()
    return (ts_vec / s) if s > 0 else np.ones(len(ts_vec)) / len(ts_vec)


def main():
    outcomes, attributes, eligibility = R.load_cohort()
    data = R.prepare(outcomes, attributes, eligibility)
    feats, targets, splits, actuals = data["feats"], data["targets"], data["splits"], data["actuals"]
    merged = feats.merge(targets, on="person_id", how="inner")
    fcols = R.feature_columns(merged[merged["person_id"].isin(splits["train"])])

    tr = merged[merged["person_id"].isin(splits["train"])]
    va = merged[merged["person_id"].isin(splits["val"]) & merged["person_id"].isin(set(actuals))]
    te = merged[merged["person_id"].isin(splits["test"]) & merged["person_id"].isin(set(actuals))]

    tp = TwoPartModel(task="regression", config=R.MODEL_CFG["xgboost"])
    tp.fit(tr[fcols], pd.Series(tr["total_paid_sum"].values))
    cs_factor = float(np.mean(va["total_paid_sum"].values) / np.mean(np.maximum(tp.predict(va[fcols]), 0)))
    print(f"cs calibration factor (val): {cs_factor:.4f}", flush=True)

    def budgets(df):
        return {pid: max(float(v), 0.0) * cs_factor for pid, v in zip(df["person_id"].values, np.maximum(tp.predict(df[fcols]), 0))}
    bud_va, bud_te = budgets(va), budgets(te)

    fc_all = joblib.load(WD / "forecasts.joblib")
    chronos = {p: np.asarray(v, dtype=float) for p, v in fc_all["chronos"].items()}

    def split_vectors(df, budget):
        pids = [p for p in df["person_id"].values if p in chronos and p in actuals and p in budget]
        ch = {p: chronos[p][:HORIZON] for p in pids}
        ac = {p: np.asarray(actuals[p], dtype=float)[:HORIZON] for p in pids}
        bu = {p: budget[p] for p in pids}
        return pids, ch, ac, bu

    va_pids, va_ch, va_ac, va_bu = split_vectors(va, bud_va)
    te_pids, te_ch, te_ac, te_bu = split_vectors(te, bud_te)
    print(f"val n={len(va_pids)} test n={len(te_pids)}", flush=True)

    # Foundation-model gate signal = predicted 3-month total from Chronos.
    def chronos_total(ch):
        return {p: float(ch[p].sum()) for p in ch}
    va_ct, te_ct = chronos_total(va_ch), chronos_total(te_ch)

    results = {}

    # Reference points (recomputed here on the same members for a fair frontier).
    def cs_flat(pids, ac, bu):
        return {p: np.full(len(ac[p]), bu[p] / len(ac[p])) for p in pids}
    def chronos_pred(pids, ch):
        return {p: ch[p] for p in pids}
    def naive_pred(df, pids):
        # trailing 3-month mean from context
        ctx = data["context"]
        return {p: np.full(HORIZON, float(np.mean(ctx[p][-3:]))) for p in pids if p in ctx}

    results["chronos"] = evalp(chronos_pred(te_pids, te_ch), te_ac)
    results["cs_two_part_calibrated"] = evalp(cs_flat(te_pids, te_ac, te_bu), te_ac)
    np_te = naive_pred(te, te_pids)
    results["naive_mean3"] = evalp(np_te, {p: te_ac[p] for p in np_te})

    # Current proportional hybrid (baseline that fails).
    def hybrid_proportional(pids, ch, bu):
        return {p: shape_from(ch[p]) * bu[p] for p in pids}
    results["hybrid_proportional"] = evalp(hybrid_proportional(te_pids, te_ch, te_bu), te_ac)

    # ---- Variant A: hard gate on the foundation-model total ----
    # If Chronos predicts the member is essentially zero (total <= tau), trust it (predict
    # the Chronos series); else use the calibrated CS budget allocated by the Chronos shape.
    def hybrid_gated(pids, ch, bu, ct, tau):
        out = {}
        for p in pids:
            if ct[p] <= tau:
                out[p] = ch[p]                      # trust the foundation model's near-zero call
            else:
                out[p] = shape_from(ch[p]) * bu[p]  # cross-sectional level, foundation-model timing
        return out

    taus = [0, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1800, 2500]
    best = None
    for tau in taus:
        mv = evalp(hybrid_gated(va_pids, va_ch, va_bu, va_ct, tau), va_ac)
        # Frontier objective on validation: minimize MAE subject to R2 >= cross-sectional-grade (0.15);
        # among those, pick lowest MAE. Track the full val curve too.
        if mv["r2cal"] >= 0.15:
            if best is None or mv["mae"] < best[1]["mae"]:
                best = (tau, mv)
    chosen_tau = best[0] if best else 200
    print(f"gated hybrid: chosen tau={chosen_tau} (val MAE {best[1]['mae']:.1f}, val R2 {best[1]['r2cal']:.3f})" if best
          else "gated hybrid: no tau met val R2>=0.15", flush=True)
    results[f"hybrid_gated_tau{chosen_tau}"] = evalp(hybrid_gated(te_pids, te_ch, te_bu, te_ct, chosen_tau), te_ac)
    # Also record the full validation/test gate sweep for the appendix.
    sweep = []
    for tau in taus:
        tv = evalp(hybrid_gated(te_pids, te_ch, te_bu, te_ct, tau), te_ac)
        vv = evalp(hybrid_gated(va_pids, va_ch, va_bu, va_ct, tau), va_ac)
        sweep.append({"tau": tau, "val_mae": vv["mae"], "val_r2": vv["r2cal"],
                      "test_mae": tv["mae"], "test_r2": tv["r2cal"], "test_pr": tv["pr"]})

    # ---- Variant B: soft gate (sigmoid on Chronos total), tuned slope/threshold on val ----
    # The foundation-model total gates between the calibrated cross-sectional budget (for
    # members predicted to incur cost) and the foundation-model level (for members predicted
    # near-zero). Sweeping the gate traces the MAE-discrimination-calibration frontier.
    def hybrid_soft(pids, ch, bu, ct, tau, k):
        out = {}
        for p in pids:
            g = 1.0 / (1.0 + np.exp(-k * (ct[p] - tau)))
            out[p] = shape_from(ch[p]) * (g * bu[p] + (1 - g) * ct[p])
        return out

    soft_grid = []
    for tau in [50, 100, 200, 300, 400, 600, 800, 1200, 1800]:
        for k in [0.0005, 0.001, 0.002, 0.004, 0.01]:
            vv = evalp(hybrid_soft(va_pids, va_ch, va_bu, va_ct, tau, k), va_ac)
            tv = evalp(hybrid_soft(te_pids, te_ch, te_bu, te_ct, tau, k), te_ac)
            soft_grid.append({"tau": tau, "k": k, "val_mae": vv["mae"], "val_r2": vv["r2cal"], "val_pr": vv["pr"],
                              "test_mae": tv["mae"], "test_r2": tv["r2cal"], "test_pr": tv["pr"]})

    # Pre-registered selection on VALIDATION: minimize MAE subject to (a) calibrated R2 >=
    # 0.15 (cross-sectional grade) and (b) predictive ratio within the actuarial 0.90-1.10 band.
    def pick(constr_pr):
        cand = [g for g in soft_grid if g["val_r2"] >= 0.15 and (not constr_pr or 0.90 <= g["val_pr"] <= 1.10)]
        return min(cand, key=lambda g: g["val_mae"]) if cand else None

    gcal = pick(True)     # calibrated operating point (PR in band)
    gmae = pick(False)    # MAE-leaning operating point (R2>=0.15, PR free)
    if gcal:
        print(f"soft gate CALIBRATED: tau={gcal['tau']} k={gcal['k']} val(MAE {gcal['val_mae']:.1f},R2 {gcal['val_r2']:.3f},PR {gcal['val_pr']:.3f})", flush=True)
        results["hybrid_soft_calibrated"] = evalp(hybrid_soft(te_pids, te_ch, te_bu, te_ct, gcal["tau"], gcal["k"]), te_ac)
    if gmae:
        print(f"soft gate MAE-LEAN: tau={gmae['tau']} k={gmae['k']} val(MAE {gmae['val_mae']:.1f},R2 {gmae['val_r2']:.3f},PR {gmae['val_pr']:.3f})", flush=True)
        results["hybrid_soft_mae_lean"] = evalp(hybrid_soft(te_pids, te_ch, te_bu, te_ct, gmae["tau"], gmae["k"]), te_ac)

    # ---- Variant C: min-rule (predict the smaller of Chronos and CS budget per member) ----
    def hybrid_min(pids, ch, bu):
        out = {}
        for p in pids:
            ts_tot = ch[p].sum()
            tgt = min(ts_tot, bu[p]) if ts_tot > 0 else 0.0
            out[p] = shape_from(ch[p]) * tgt
        return out
    results["hybrid_min"] = evalp(hybrid_min(te_pids, te_ch, te_bu), te_ac)

    # ---- Report ----
    print("\n=== TEST-set comparison (MAE per-member-per-month; R2/PR on 3-month total) ===", flush=True)
    print(f"{'model':32s} {'MAE':>8} {'R2cal':>8} {'PR':>7} {'AUROC':>7}", flush=True)
    for k, m in sorted(results.items(), key=lambda x: x[1]["mae"]):
        print(f"{k:32s} {m['mae']:8.1f} {m['r2cal']:8.3f} {m['pr']:7.3f} {(m['auroc'] or float('nan')):7.3f}", flush=True)

    # Pareto check vs the two reference frontier points.
    ref = [("naive_mean3", results["naive_mean3"]), ("cs_two_part_calibrated", results["cs_two_part_calibrated"]),
           ("chronos", results["chronos"])]
    print("\n=== Pareto status of gated hybrid ===", flush=True)
    gh = results[f"hybrid_gated_tau{chosen_tau}"]
    dominated = any((r["mae"] <= gh["mae"] and r["r2cal"] >= gh["r2cal"] and (r["mae"] < gh["mae"] or r["r2cal"] > gh["r2cal"]))
                    for _, r in ref)
    print(f"gated hybrid (MAE {gh['mae']:.1f}, R2 {gh['r2cal']:.3f}) dominated by a reference model? {dominated}", flush=True)

    out = {"results": results, "chosen_tau": chosen_tau, "gate_sweep": sweep,
           "soft_grid": soft_grid, "soft_calibrated": gcal, "soft_mae_lean": gmae, "cs_factor": cs_factor}
    (OUT / "hybrid_experiments.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {OUT / 'hybrid_experiments.json'}", flush=True)


if __name__ == "__main__":
    main()
