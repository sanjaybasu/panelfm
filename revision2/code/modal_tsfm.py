#!/usr/bin/env python3
"""
Modal GPU runner for the second-revision SOTA model panel.

Runs, on GPU, the time-series foundation models and the tabular foundation model that are
slow or unstable on the local CPU:
  - Chronos-T5 (small / base / large) and Chronos-Bolt-Base: median, mean, quantiles
  - TimesFM-2.0 (500M): point forecast
  - TabPFN-v2 classifier + regressor (cross-sectional discrimination)

Data sent to Modal is strictly numeric and de-identified: per-member monthly cost vectors
and the numeric feature matrix only. NO member identifiers, names, or dates are uploaded;
the index<->member mapping and all metric computation stay on the local machine.

Usage:
  DATA_ROOT=/path modal run revision2/code/modal_tsfm.py
"""
import modal

app = modal.App("panelfm-sota")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "numpy", "scikit-learn",
        "chronos-forecasting", "timesfm[torch]", "tabpfn",
        "huggingface_hub",
    )
)

HORIZON = 3
QLEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@app.function(image=image, gpu="A10G", timeout=3600)
def run_tsfm(context_list, horizon=HORIZON):
    """context_list: list of 1-D float lists (per-member monthly cost). Returns per-model
    forecasts (point=median, mean, quantiles) as index-aligned lists."""
    import numpy as np, torch, time, traceback
    out = {}

    def chronos(name, hf_id):
        from chronos import BaseChronosPipeline
        t0 = time.time()
        pipe = BaseChronosPipeline.from_pretrained(hf_id, device_map="cuda", torch_dtype=torch.bfloat16)
        med, mean, quant = [], [], []
        B = 256
        for i in range(0, len(context_list), B):
            batch = [torch.tensor(np.asarray(c, dtype=np.float32)) for c in context_list[i:i + B]]
            q, m = pipe.predict_quantiles(batch, prediction_length=horizon, quantile_levels=QLEVELS)
            q = q.float().cpu().numpy(); m = m.float().cpu().numpy()
            for bi in range(len(batch)):
                quant.append(q[bi].tolist())
                med.append(q[bi][:, QLEVELS.index(0.5)].tolist())
                mean.append(m[bi].tolist())
        out[name] = {"median": med, "mean": mean, "quantiles": quant}
        print(f"{name}: {time.time()-t0:.0f}s", flush=True)

    for name, hf in [("chronos_t5_small", "amazon/chronos-t5-small"),
                     ("chronos_bolt_base", "amazon/chronos-bolt-base"),
                     ("chronos_t5_base", "amazon/chronos-t5-base"),
                     ("chronos_t5_large", "amazon/chronos-t5-large")]:
        try:
            chronos(name, hf)
        except Exception as e:
            print(f"{name} FAILED: {e}"); print(traceback.format_exc()[-1200:])

    _timesfm(out, context_list, horizon)
    return out


def _timesfm(out, context_list, horizon):
    """TimesFM-2.5 (200M) panel forecast -- a second foundation-model family. Uses the
    timesfm 2.x API (TimesFM_2p5_200M_torch + ForecastConfig)."""
    import numpy as np, time as _t, traceback
    try:
        import timesfm
        t0 = _t.time()
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        model.compile(timesfm.ForecastConfig(
            max_context=512, max_horizon=horizon, normalize_inputs=True,
            use_continuous_quantile_head=True, infer_is_positive=True,
            fix_quantile_crossing=True))
        inputs = [np.asarray(c, dtype=np.float32) for c in context_list]
        point, _q = model.forecast(horizon=horizon, inputs=inputs)
        point = np.asarray(point)
        out["timesfm_2p5"] = {"median": [point[i][:horizon].tolist() for i in range(len(inputs))]}
        print(f"timesfm_2p5: {point.shape} {_t.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"timesfm_2p5 FAILED: {e}"); print(traceback.format_exc()[-1500:])


@app.function(image=image, gpu="A10G", timeout=3600)
def run_timesfm(context_list, horizon=HORIZON):
    """TimesFM-2.5 only (for merging into an existing Chronos panel without re-running it)."""
    out = {}
    _timesfm(out, context_list, horizon)
    return out


@app.function(image=image, gpu="A10G", timeout=3600)
def run_tabpfn(Xtr, ytr_cls, ytr_reg, Xte):
    """All numeric arrays (no identifiers). Returns classifier P(high-cost) and regressor cost."""
    import numpy as np, time, traceback
    Xtr = np.asarray(Xtr, dtype=np.float32); Xte = np.asarray(Xte, dtype=np.float32)
    ytr_cls = np.asarray(ytr_cls); ytr_reg = np.asarray(ytr_reg, dtype=np.float32)
    res = {}
    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        t0 = time.time()
        # Pin to TabPFN-v2 checkpoints (open weights; non-gated). The v2.5/v2.6/v3
        # defaults require one-time browser license acceptance / TABPFN_TOKEN.
        clf = TabPFNClassifier(model_path="tabpfn-v2-classifier-finetuned-zk73skhh.ckpt",
                               device="cuda", ignore_pretraining_limits=True)
        clf.fit(Xtr, ytr_cls); res["clf_proba"] = clf.predict_proba(Xte)[:, 1].tolist()
        reg = TabPFNRegressor(model_path="tabpfn-v2-regressor.ckpt",
                              device="cuda", ignore_pretraining_limits=True)
        reg.fit(Xtr, ytr_reg); res["reg_pred"] = np.maximum(reg.predict(Xte), 0).tolist()
        print(f"tabpfn: {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"tabpfn FAILED: {e}"); print(traceback.format_exc()[-1500:])
    return res


@app.local_entrypoint()
def main():
    import sys, json
    import numpy as np
    from pathlib import Path
    PKG = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PKG / "revision2" / "code"))
    import run_member_disjoint as R
    from src.evaluation.metrics import r_squared_calibrated, predictive_ratio, high_cost_identification

    outcomes, attributes, eligibility = R.load_cohort()
    data = R.prepare(outcomes, attributes, eligibility)
    splits, actuals, context = data["splits"], data["actuals"], data["context"]
    test_ids = [p for p in splits["test"] if p in actuals and p in context]
    ctx_list = [np.asarray(context[p], dtype=np.float32).tolist() for p in test_ids]
    act3 = {p: np.asarray(actuals[p], dtype=float) for p in test_ids}
    OUT = R.OUT_DIR

    # TSFM panel on GPU.
    tsfm = run_tsfm.remote(ctx_list, HORIZON)

    def eval_point(points):  # points: list aligned to test_ids
        pm, yt, yp = [], [], []
        for i, p in enumerate(test_ids):
            a = act3[p][:HORIZON]; f = np.asarray(points[i])[:HORIZON]
            pm.append(np.mean(np.abs(f - a))); yt.append(float(a.sum())); yp.append(float(f.sum()))
        yt = np.array(yt); yp = np.array(yp)
        hc = high_cost_identification(yt, yp).get("top_10pct", {})
        return {"mae": float(np.mean(pm)), "r_squared_calibrated": r_squared_calibrated(yt, yp),
                "predictive_ratio": predictive_ratio(yt, yp)["overall"],
                "auroc": hc.get("c_statistic", float("nan")), "n_patients": len(test_ids)}

    def crps_q(quant):
        vals = []
        for i, p in enumerate(test_ids):
            Q = np.asarray(quant[i]); y = act3[p][:HORIZON]; pl = 0.0
            for j, tau in enumerate(QLEVELS):
                d = y - Q[:HORIZON, j]; pl += np.mean(np.maximum(tau * d, (tau - 1) * d))
            vals.append(2.0 * pl / len(QLEVELS))
        return float(np.mean(vals))

    metrics, pp = {}, {}
    for name, d in tsfm.items():
        if "median" in d:
            m = eval_point(d["median"])
            if "quantiles" in d: m["crps"] = crps_q(d["quantiles"])
            metrics[f"{name}_median"] = m
            pp[f"{name}_median"] = {test_ids[i]: float(np.sum(np.asarray(d['median'][i])[:HORIZON])) for i in range(len(test_ids))}
        if "mean" in d:
            metrics[f"{name}_mean"] = eval_point(d["mean"])
            pp[f"{name}_mean"] = {test_ids[i]: float(np.sum(np.asarray(d['mean'][i])[:HORIZON])) for i in range(len(test_ids))}
    (OUT / "all_metrics_tsfm.json").write_text(json.dumps(metrics, indent=2, default=float))
    (OUT / "per_patient_tsfm.json").write_text(json.dumps(pp, default=float))
    print("TSFM panel:", {k: round(v["mae"], 1) for k, v in metrics.items()})

    # TabPFN on the cross-sectional task. Use the production feature_columns() so the
    # leak-prone target-window columns (high_cost_flag, *_sum, any_*_flag) are excluded,
    # exactly as the gradient-boosted baselines do.
    merged = data["feats"].merge(data["targets"], on="person_id", how="inner")
    fcols = R.feature_columns(merged)
    tr = merged[merged["person_id"].isin(splits["train"])]
    te = merged[merged["person_id"].isin(set(test_ids))]
    thr = tr["total_paid_sum"].quantile(0.90)
    ytr_cls = (tr["total_paid_sum"] >= thr).astype(int).values
    ytr_reg = tr["total_paid_sum"].values
    Xtr = tr[fcols].values.astype(np.float32);
    te = te.set_index("person_id").loc[test_ids].reset_index()
    Xte = te[fcols].values.astype(np.float32)
    rng = np.random.default_rng(R.SPLIT_SEED)
    if len(Xtr) > 10000:
        pos = np.where(ytr_cls == 1)[0]; neg = np.where(ytr_cls == 0)[0]
        npos = min(len(pos), 5000)
        idx = np.concatenate([rng.choice(pos, npos, replace=False), rng.choice(neg, 10000 - npos, replace=False)])
        rng.shuffle(idx)
    else:
        idx = np.arange(len(Xtr))
    tab = run_tabpfn.remote(Xtr[idx].tolist(), ytr_cls[idx].tolist(), ytr_reg[idx].tolist(), Xte.tolist())
    tabm = {}
    yt = np.array([float(act3[p].sum()) for p in test_ids])
    if "clf_proba" in tab:
        hc = high_cost_identification(yt, np.array(tab["clf_proba"])).get("top_10pct", {})
        tabm["tabpfn_classifier"] = {"auroc": hc.get("c_statistic", float("nan")),
                                     "ppv_at_top10pct": hc.get("ppv", float("nan")), "n_patients": len(test_ids)}
    if "reg_pred" in tab:
        yp = np.array(tab["reg_pred"]); pm = np.mean([np.mean(np.abs(yp[i] / HORIZON - act3[test_ids[i]][:HORIZON])) for i in range(len(test_ids))])
        hc = high_cost_identification(yt, yp).get("top_10pct", {})
        tabm["tabpfn_regressor"] = {"mae": float(pm), "r_squared_calibrated": r_squared_calibrated(yt, yp),
                                    "predictive_ratio": predictive_ratio(yt, yp)["overall"],
                                    "auroc": hc.get("c_statistic", float("nan")), "n_patients": len(test_ids)}
        pp["tabpfn_regressor"] = {test_ids[i]: float(yp[i]) for i in range(len(test_ids))}
    (OUT / "all_metrics_tabpfn.json").write_text(json.dumps(tabm, indent=2, default=float))
    (OUT / "per_patient_tsfm.json").write_text(json.dumps(pp, default=float))  # merged TSFM + tabpfn totals
    print("TabPFN:", tabm)


def _prep():
    """Shared data prep for the focused entrypoints. Returns the helper module, the
    prepared data dict, the test ids, the context list, the per-member actuals, and OUT."""
    import sys
    import numpy as np
    from pathlib import Path
    PKG = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PKG / "revision2" / "code"))
    import run_member_disjoint as R
    outcomes, attributes, eligibility = R.load_cohort()
    data = R.prepare(outcomes, attributes, eligibility)
    splits, actuals, context = data["splits"], data["actuals"], data["context"]
    test_ids = [p for p in splits["test"] if p in actuals and p in context]
    ctx_list = [np.asarray(context[p], dtype=np.float32).tolist() for p in test_ids]
    act3 = {p: np.asarray(actuals[p], dtype=float) for p in test_ids}
    return R, data, splits, test_ids, ctx_list, act3, R.OUT_DIR


@app.local_entrypoint()
def timesfm_only():
    """Run TimesFM-2.5 only and MERGE it into the existing Chronos panel results, so the
    GPU Chronos sweep is not paid for again."""
    import json
    import numpy as np
    from src.evaluation.metrics import r_squared_calibrated, predictive_ratio, high_cost_identification
    R, data, splits, test_ids, ctx_list, act3, OUT = _prep()
    res = run_timesfm.remote(ctx_list, HORIZON)
    if "timesfm_2p5" not in res or "median" not in res["timesfm_2p5"]:
        print("TimesFM produced no forecast; nothing merged."); return
    pts = res["timesfm_2p5"]["median"]
    pm, yt, yp = [], [], []
    for i, p in enumerate(test_ids):
        a = act3[p][:HORIZON]; f = np.asarray(pts[i])[:HORIZON]
        pm.append(np.mean(np.abs(f - a))); yt.append(float(a.sum())); yp.append(float(f.sum()))
    yt = np.array(yt); yp = np.array(yp)
    hc = high_cost_identification(yt, yp).get("top_10pct", {})
    m = {"mae": float(np.mean(pm)), "r_squared_calibrated": r_squared_calibrated(yt, yp),
         "predictive_ratio": predictive_ratio(yt, yp)["overall"],
         "auroc": hc.get("c_statistic", float("nan")), "n_patients": len(test_ids)}
    af = OUT / "all_metrics_tsfm.json"; pf = OUT / "per_patient_tsfm.json"
    am = json.loads(af.read_text()) if af.exists() else {}
    ppm = json.loads(pf.read_text()) if pf.exists() else {}
    am["timesfm_2p5"] = m
    ppm["timesfm_2p5"] = {test_ids[i]: float(yp[i]) for i in range(len(test_ids))}
    af.write_text(json.dumps(am, indent=2, default=float)); pf.write_text(json.dumps(ppm, default=float))
    print("Merged timesfm_2p5:", {k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()})


@app.local_entrypoint()
def tabpfn_only():
    """Run TabPFN-v2 (cross-sectional) only and MERGE into the tabpfn + per-patient files."""
    import json
    import numpy as np
    from src.evaluation.metrics import r_squared_calibrated, predictive_ratio, high_cost_identification
    R, data, splits, test_ids, ctx_list, act3, OUT = _prep()
    # Production feature set (excludes the target-window leak columns), identical to the GBMs.
    merged = data["feats"].merge(data["targets"], on="person_id", how="inner")
    fcols = R.feature_columns(merged)
    tr = merged[merged["person_id"].isin(splits["train"])]
    te = merged[merged["person_id"].isin(set(test_ids))].set_index("person_id").loc[test_ids].reset_index()
    thr = tr["total_paid_sum"].quantile(0.90)
    ytr_cls = (tr["total_paid_sum"] >= thr).astype(int).values
    ytr_reg = tr["total_paid_sum"].values
    Xtr = tr[fcols].values.astype(np.float32); Xte = te[fcols].values.astype(np.float32)
    rng = np.random.default_rng(R.SPLIT_SEED)
    if len(Xtr) > 10000:
        pos = np.where(ytr_cls == 1)[0]; neg = np.where(ytr_cls == 0)[0]
        npos = min(len(pos), 5000)
        idx = np.concatenate([rng.choice(pos, npos, replace=False), rng.choice(neg, 10000 - npos, replace=False)])
        rng.shuffle(idx)
    else:
        idx = np.arange(len(Xtr))
    tab = run_tabpfn.remote(Xtr[idx].tolist(), ytr_cls[idx].tolist(), ytr_reg[idx].tolist(), Xte.tolist())
    tabm = {}; yt = np.array([float(act3[p].sum()) for p in test_ids])
    if "clf_proba" in tab:
        hc = high_cost_identification(yt, np.array(tab["clf_proba"])).get("top_10pct", {})
        tabm["tabpfn_classifier"] = {"auroc": hc.get("c_statistic", float("nan")),
                                     "ppv_at_top10pct": hc.get("ppv", float("nan")), "n_patients": len(test_ids)}
    pf = OUT / "per_patient_tsfm.json"; ppm = json.loads(pf.read_text()) if pf.exists() else {}
    if "reg_pred" in tab:
        yp = np.array(tab["reg_pred"])
        pm = np.mean([np.mean(np.abs(yp[i] / HORIZON - act3[test_ids[i]][:HORIZON])) for i in range(len(test_ids))])
        hc = high_cost_identification(yt, yp).get("top_10pct", {})
        tabm["tabpfn_regressor"] = {"mae": float(pm), "r_squared_calibrated": r_squared_calibrated(yt, yp),
                                    "predictive_ratio": predictive_ratio(yt, yp)["overall"],
                                    "auroc": hc.get("c_statistic", float("nan")), "n_patients": len(test_ids)}
        ppm["tabpfn_regressor"] = {test_ids[i]: float(yp[i]) for i in range(len(test_ids))}
    (OUT / "all_metrics_tabpfn.json").write_text(json.dumps(tabm, indent=2, default=float))
    pf.write_text(json.dumps(ppm, default=float))
    print("TabPFN:", tabm)
