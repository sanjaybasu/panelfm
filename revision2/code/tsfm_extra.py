#!/usr/bin/env python3
"""
Additional time-series foundation models (torch-isolated), for the second revision.

Adds, on the same held-out test members as the core run:
  - Chronos-T5-Small predictive MEDIAN, MEAN, and quantiles (the median-vs-mean
    contrast that demonstrates the forecasting-vs-calibration frontier within one model,
    plus CRPS for the probabilistic forecast)
  - Chronos-Bolt-Base (current state-of-the-art zero-shot TSFM)
  - Chronos-T5-Base (model-size sweep)
  - TimesFM-2.0 (a second foundation-model family; the model cited but not previously run)

Reuses the workdir saved by run_member_disjoint.py (context, actuals, member ids), so it
does not re-prepare the data. Runs in a separate process from the gradient-boosted models
(torch / LightGBM OpenMP conflict on macOS). Each model is wrapped in try/except so one
failure does not abort the others.

Output: results/all_metrics_tsfm.json, results/per_patient_tsfm.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys, json, time, traceback
from pathlib import Path
import numpy as np
import joblib
import torch
torch.set_num_threads(4)

PKG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG_ROOT))
from src.evaluation.metrics import r_squared, r_squared_calibrated, predictive_ratio, high_cost_identification

HORIZON = 3
WD = PKG_ROOT / "revision2" / "results" / "_torch_workdir"
OUT = PKG_ROOT / "revision2" / "results"
QLEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def log(m): print(m, flush=True)


def eval_point(point3, act3):
    pids = sorted(set(point3) & set(act3))
    pm = np.array([np.mean(np.abs(np.asarray(point3[p])[:HORIZON] - act3[p][:HORIZON])) for p in pids])
    yt = np.array([float(np.sum(act3[p][:HORIZON])) for p in pids])
    yp = np.array([float(np.sum(np.asarray(point3[p])[:HORIZON])) for p in pids])
    hc = high_cost_identification(yt, yp).get("top_10pct", {})
    return {"mae": float(pm.mean()), "r_squared_calibrated": r_squared_calibrated(yt, yp),
            "predictive_ratio": predictive_ratio(yt, yp)["overall"],
            "auroc": hc.get("c_statistic", np.nan), "n_patients": len(pids)}, pids


def crps_from_quantiles(q3, act3, levels=QLEVELS):
    # CRPS approx = 2 * mean over quantile levels of pinball loss (per month, per member)
    pids = sorted(set(q3) & set(act3))
    vals = []
    for p in pids:
        Q = np.asarray(q3[p])  # (H, n_levels)
        y = np.asarray(act3[p])[:HORIZON]
        pl = 0.0
        for j, tau in enumerate(levels):
            d = y - Q[:HORIZON, j]
            pl += np.mean(np.maximum(tau * d, (tau - 1) * d))
        vals.append(2.0 * pl / len(levels))
    return float(np.mean(vals))


def main():
    context = joblib.load(WD / "context.joblib")
    actuals = {p: np.asarray(v, dtype=float) for p, v in joblib.load(WD / "actuals.joblib").items()}
    ids = json.loads((WD / "ids.json").read_text())
    test_ids = [p for p in ids["test"] if p in context and p in actuals]
    test_ctx = {p: np.asarray(context[p], dtype=np.float32) for p in test_ids}
    log(f"test members: {len(test_ids)}")

    results, perpatient = {}, {}

    def chronos_model(name, hf_id):
        from chronos import BaseChronosPipeline
        t0 = time.time()
        pipe = BaseChronosPipeline.from_pretrained(hf_id, device_map="cpu", torch_dtype=torch.float32)
        pids = list(test_ctx); med, mean, qd = {}, {}, {}
        B = 64
        for i in range(0, len(pids), B):
            batch = pids[i:i + B]
            ctx = [torch.tensor(test_ctx[p]) for p in batch]
            q, m = pipe.predict_quantiles(ctx, prediction_length=HORIZON, quantile_levels=QLEVELS)
            q = q.numpy(); m = m.numpy()
            for bi, p in enumerate(batch):
                qd[p] = q[bi]                          # (H, n_levels)
                med[p] = q[bi][:, QLEVELS.index(0.5)]  # median
                mean[p] = m[bi]                        # mean
        log(f"  {name}: forecast done {time.time()-t0:.0f}s")
        mmed, _ = eval_point(med, actuals); mmed["crps"] = crps_from_quantiles(qd, actuals)
        results[f"{name}_median"] = mmed; perpatient[f"{name}_median"] = med
        mmean, _ = eval_point(mean, actuals); mmean["crps"] = mmed["crps"]
        results[f"{name}_mean"] = mmean; perpatient[f"{name}_mean"] = mean
        log(f"  {name}_median MAE={mmed['mae']:.1f} R2cal={mmed['r_squared_calibrated']:.3f} PR={mmed['predictive_ratio']:.3f} CRPS={mmed['crps']:.1f}")
        log(f"  {name}_mean   MAE={mmean['mae']:.1f} R2cal={mmean['r_squared_calibrated']:.3f} PR={mmean['predictive_ratio']:.3f}")

    for name, hf in [("chronos_t5_small", "amazon/chronos-t5-small"),
                     ("chronos_bolt_base", "amazon/chronos-bolt-base"),
                     ("chronos_t5_base", "amazon/chronos-t5-base")]:
        try:
            chronos_model(name, hf)
        except Exception as e:
            log(f"  {name} FAILED: {e}"); log(traceback.format_exc()[-1500:])

    # TimesFM 2.0 (separate family).
    try:
        t0 = time.time()
        import timesfm
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(backend="cpu", per_core_batch_size=32,
                                           horizon_len=HORIZON, context_len=512),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-pytorch"))
        pids = list(test_ctx)
        inputs = [test_ctx[p].astype(np.float32) for p in pids]
        freq = [0] * len(inputs)
        point, qfc = tfm.forecast(inputs, freq=freq)
        pt = {p: np.asarray(point[i])[:HORIZON] for i, p in enumerate(pids)}
        m, _ = eval_point(pt, actuals)
        results["timesfm_2"] = m; perpatient["timesfm_2"] = pt
        log(f"  timesfm_2 MAE={m['mae']:.1f} R2cal={m['r_squared_calibrated']:.3f} PR={m['predictive_ratio']:.3f} ({time.time()-t0:.0f}s)")
    except Exception as e:
        log(f"  timesfm_2 FAILED: {e}"); log(traceback.format_exc()[-1500:])

    def jsonable(o):
        if isinstance(o, dict): return {k: jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
        if isinstance(o, np.ndarray): return [float(x) for x in o]
        if isinstance(o, (np.floating, float)): return None if (isinstance(o, float) and np.isnan(o)) else float(o)
        if isinstance(o, (np.integer,)): return int(o)
        return o

    (OUT / "all_metrics_tsfm.json").write_text(json.dumps(jsonable(results), indent=2))
    pp_tot = {m: {p: float(np.sum(np.asarray(v)[:HORIZON])) for p, v in d.items()} for m, d in perpatient.items()}
    (OUT / "per_patient_tsfm.json").write_text(json.dumps(jsonable(pp_tot)))
    log(f"\nWrote all_metrics_tsfm.json ({len(results)} model variants)")


if __name__ == "__main__":
    main()
