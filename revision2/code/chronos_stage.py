#!/usr/bin/env python3
"""
Torch-isolated Chronos + panel-conditioned forecasting stage.

Run as a separate process from the cross-sectional stage: on macOS, torch's
OpenMP runtime and LightGBM's bundled OpenMP runtime cannot safely share a
single process. Isolating all torch work here avoids the conflict.

Inputs (in --workdir):
  context.joblib    dict person_id -> np.float32 monthly cost series (through context cutoff)
  ids.json          {"train_sub": [...], "val": [...], "test": [...]}
  embeddings.joblib {"embeddings": {pid: [9-dim]}, "embedding_dim": 9}  (built upstream with XGBoost)

The patient embeddings are computed upstream (XGBoost-based) and passed in as plain
vectors, so this process never imports XGBoost, keeping XGBoost's OpenMP runtime out
of the torch process.

Output (in --workdir):
  forecasts.joblib {"chronos": {pid: arr}, "panelfm_xreg": {pid: arr}, "panelfm_adapter": {pid: arr}}
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
torch.set_num_threads(4)

PKG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG_ROOT))

from src.models.timesfm_wrapper import ChronosForecaster, PanelFMXReg, PanelFMAdapter

HORIZON = int(os.environ.get("FM_HORIZON", "3"))


class DictEmbeddingStore:
    """Minimal embedding store backed by a precomputed {person_id: vector} dict.

    Exposes the .get / .embedding_dim interface that PanelFMXReg and PanelFMAdapter
    require, without importing XGBoost in this torch process.
    """

    def __init__(self, embeddings, embedding_dim):
        self.embeddings = {p: np.asarray(v, dtype=np.float32) for p, v in embeddings.items()}
        self.embedding_dim = embedding_dim

    def get(self, pid):
        return self.embeddings.get(pid)


class CachedChronos(ChronosForecaster):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._cache = {}

    def forecast_batch(self, series_dict, horizon=6, num_samples=20, batch_size=64):
        import hashlib
        todo, out = {}, {}
        for pid, ts in series_dict.items():
            arr = np.asarray(ts, dtype=np.float32)
            key = (horizon, hashlib.md5(arr.tobytes()).hexdigest())
            if key in self._cache:
                out[pid] = self._cache[key]
            else:
                todo[pid] = (arr, key)
        if todo:
            res = super().forecast_batch({p: v[0] for p, v in todo.items()},
                                         horizon, num_samples, batch_size)
            for pid, fc in res.items():
                out[pid] = fc
                self._cache[todo[pid][1]] = fc
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    args = ap.parse_args()
    wd = Path(args.workdir)

    context = joblib.load(wd / "context.joblib")
    ids = json.loads((wd / "ids.json").read_text())
    train_sub = [p for p in ids["train_sub"] if p in context]
    val_ids = [p for p in ids["val"] if p in context]
    test_ids = [p for p in ids["test"] if p in context]

    forecaster = CachedChronos(model_name="amazon/chronos-t5-small", device="cpu")
    forecaster.load_model()

    union = {p: context[p] for p in set(train_sub) | set(val_ids) | set(test_ids)}
    print(f"chronos: forecasting {len(union):,} series "
          f"(test={len(test_ids):,} val={len(val_ids):,} train_sub={len(train_sub):,})", flush=True)
    chronos_fc = forecaster.forecast_batch(union, HORIZON, batch_size=64)

    out = {"chronos": {p: np.asarray(chronos_fc[p]).tolist() for p in chronos_fc}}

    # Patient embeddings (precomputed upstream with XGBoost) for the panel-conditioned variants.
    emb = joblib.load(wd / "embeddings.joblib")
    store = DictEmbeddingStore(emb["embeddings"], emb["embedding_dim"])

    # Train residual/adapter on train-subsample (context split into context/actual).
    tr_ctx, tr_act = {}, {}
    for p in train_sub:
        ts = np.asarray(context[p], dtype=np.float32)
        if len(ts) > HORIZON:
            tr_ctx[p] = ts[:-HORIZON]
            tr_act[p] = ts[-HORIZON:]
    test_ctx = {p: np.asarray(context[p], dtype=np.float32) for p in test_ids}

    try:
        xreg = PanelFMXReg(forecaster, store)
        xreg.fit_residual_model(tr_ctx, tr_act, HORIZON)
        fc = xreg.forecast_batch(test_ctx, HORIZON)
        out["panelfm_xreg"] = {p: np.asarray(fc[p]).tolist() for p in fc}
        print(f"panelfm_xreg: {len(out['panelfm_xreg'])} forecasts", flush=True)
    except Exception as e:
        print(f"panelfm_xreg failed: {e}", flush=True)

    try:
        adapter = PanelFMAdapter(forecaster, store, embedding_dim=store.embedding_dim,
                                 hidden_dim=32, lr=1e-3, epochs=30)
        adapter.fit(tr_ctx, tr_act, HORIZON)
        fc = adapter.forecast_batch(test_ctx, HORIZON)
        out["panelfm_adapter"] = {p: np.asarray(fc[p]).tolist() for p in fc}
        print(f"panelfm_adapter: {len(out['panelfm_adapter'])} forecasts", flush=True)
    except Exception as e:
        print(f"panelfm_adapter failed: {e}", flush=True)

    joblib.dump(out, wd / "forecasts.joblib")
    print(f"Wrote {wd / 'forecasts.joblib'}", flush=True)


if __name__ == "__main__":
    main()
