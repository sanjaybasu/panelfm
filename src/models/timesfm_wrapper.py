"""
Time Series Foundation Model wrappers for per-patient claims forecasting.

Uses Amazon's Chronos (transformer-based TS foundation model) as the core
foundation model. Also supports TimesFM when available.

PanelFM conditioning strategies:
  A) XReg: patient embedding as static covariate (linear residual correction)
  B) Adapter: learned injection block (nonlinear conditioning)
  C) ICF: in-context fine-tuning with similar patients as context
"""

import numpy as np
import pandas as pd
import torch
from typing import Optional, Dict, List
import warnings
from tqdm import tqdm


class ChronosForecaster:
    """
    Wrapper around Amazon's Chronos for per-patient claims forecasting.
    Chronos is a family of pretrained time series foundation models
    based on T5 architecture with tokenized time series.
    """

    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-small",  # 46M params; or -base (200M), -large (710M)
        device: str = "cpu",  # "cpu", "cuda", or "mps"
    ):
        self.model_name = model_name
        self.device = device
        self.pipeline = None

    def load_model(self):
        """Load Chronos pipeline."""
        try:
            from chronos import BaseChronosPipeline
            self.pipeline = BaseChronosPipeline.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=torch.float32,
            )
            print(f"Loaded Chronos model: {self.model_name}")
        except ImportError:
            warnings.warn("chronos-forecasting not installed")
            self.pipeline = None

    def forecast_batch(
        self,
        series_dict: Dict[str, np.ndarray],
        horizon: int = 6,
        num_samples: int = 20,
        batch_size: int = 64,
    ) -> Dict[str, np.ndarray]:
        """
        Forecast for a batch of patient time series.

        Returns point forecasts (median of samples) and optionally
        prediction intervals.
        """
        if self.pipeline is None:
            self.load_model()

        if self.pipeline is None:
            warnings.warn("Chronos not loaded, returning mean forecasts")
            return {pid: np.full(horizon, np.mean(ts)) for pid, ts in series_dict.items()}

        person_ids = list(series_dict.keys())
        results = {}

        # Process in batches
        for batch_start in range(0, len(person_ids), batch_size):
            batch_pids = person_ids[batch_start:batch_start + batch_size]
            batch_series = [
                torch.tensor(series_dict[pid], dtype=torch.float32)
                for pid in batch_pids
            ]

            # Chronos forecast
            forecast_samples = self.pipeline.predict(
                batch_series,
                prediction_length=horizon,
                num_samples=num_samples,
            )  # (batch, num_samples, horizon)

            for i, pid in enumerate(batch_pids):
                samples = forecast_samples[i].numpy()  # (num_samples, horizon)
                results[pid] = np.median(samples, axis=0).astype(np.float32)

        return results

    def forecast_batch_with_quantiles(
        self,
        series_dict: Dict[str, np.ndarray],
        horizon: int = 6,
        num_samples: int = 100,
        quantiles: list = [0.1, 0.5, 0.9],
        batch_size: int = 64,
    ) -> Dict[str, dict]:
        """Forecast with quantile predictions."""
        if self.pipeline is None:
            self.load_model()

        if self.pipeline is None:
            return {}

        person_ids = list(series_dict.keys())
        results = {}

        for batch_start in range(0, len(person_ids), batch_size):
            batch_pids = person_ids[batch_start:batch_start + batch_size]
            batch_series = [
                torch.tensor(series_dict[pid], dtype=torch.float32)
                for pid in batch_pids
            ]

            forecast_samples = self.pipeline.predict(
                batch_series,
                prediction_length=horizon,
                num_samples=num_samples,
            )

            for i, pid in enumerate(batch_pids):
                samples = forecast_samples[i].numpy()
                results[pid] = {
                    "point": np.median(samples, axis=0),
                    "mean": np.mean(samples, axis=0),
                    "std": np.std(samples, axis=0),
                }
                for q in quantiles:
                    results[pid][f"q{int(q*100):02d}"] = np.quantile(samples, q, axis=0)

        return results


class PanelFMXReg:
    """
    PanelFM Option A: Patient embedding as static covariate with
    linear residual correction on top of foundation model forecasts.

    This is the simplest approach — no retraining of the foundation model.
    """

    def __init__(
        self,
        forecaster: ChronosForecaster,
        embedding_store=None,
    ):
        self.forecaster = forecaster
        self.embedding_store = embedding_store
        self.residual_model = None

    def fit_residual_model(
        self,
        series_dict: Dict[str, np.ndarray],
        actuals_dict: Dict[str, np.ndarray],
        horizon: int = 6,
    ):
        """
        Fit the residual correction model.

        1. Get base forecasts from foundation model
        2. Compute residuals = actual - forecast
        3. Fit Ridge regression: patient_embedding → mean_residual
        """
        from sklearn.linear_model import Ridge

        print("    Generating base forecasts for residual fitting...")
        base_forecasts = self.forecaster.forecast_batch(series_dict, horizon)

        embeddings = []
        residuals = []

        for pid in series_dict:
            if pid not in actuals_dict or pid not in base_forecasts:
                continue

            embedding = self.embedding_store.get(pid)
            if embedding is None:
                continue

            actual = actuals_dict[pid][:horizon]
            base = base_forecasts[pid][:len(actual)]

            if len(actual) == 0 or len(base) == 0:
                continue

            residual = np.mean(actual - base)
            if np.isfinite(residual):
                embeddings.append(embedding)
                residuals.append(residual)

        if len(embeddings) < 10:
            warnings.warn(f"Too few patients ({len(embeddings)}) for residual model")
            return self

        X = np.vstack(embeddings)
        y = np.array(residuals)

        self.residual_model = Ridge(alpha=1.0)
        self.residual_model.fit(X, y)

        r2 = self.residual_model.score(X, y)
        print(f"    Residual model R²: {r2:.4f} (n={len(y)})")

        return self

    def forecast_batch(
        self,
        series_dict: Dict[str, np.ndarray],
        horizon: int = 6,
    ) -> Dict[str, np.ndarray]:
        """Forecast with patient embedding residual correction."""
        base_forecasts = self.forecaster.forecast_batch(series_dict, horizon)

        if self.embedding_store is None or self.residual_model is None:
            return base_forecasts

        results = {}
        for pid in series_dict:
            base = base_forecasts.get(pid, np.zeros(horizon))
            embedding = self.embedding_store.get(pid)

            if embedding is not None:
                adjustment = self.residual_model.predict(embedding.reshape(1, -1))[0]
                results[pid] = base + adjustment
            else:
                results[pid] = base

        return results


class PanelFMAdapter:
    """
    PanelFM Option B: Learned adapter that maps patient embeddings
    to forecast adjustments via a small neural network.

    More expressive than linear XReg but requires training data.
    """

    def __init__(
        self,
        forecaster: ChronosForecaster,
        embedding_store=None,
        embedding_dim: int = 9,
        hidden_dim: int = 32,
        lr: float = 1e-3,
        epochs: int = 50,
    ):
        self.forecaster = forecaster
        self.embedding_store = embedding_store
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.adapter = None

    def _build_adapter(self, horizon: int):
        """Build the adapter network: embedding → per-step adjustment."""
        self.adapter = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_dim, horizon),
        )
        n_params = sum(p.numel() for p in self.adapter.parameters())
        print(f"    Adapter parameters: {n_params:,}")

    def fit(
        self,
        series_dict: Dict[str, np.ndarray],
        actuals_dict: Dict[str, np.ndarray],
        horizon: int = 6,
    ):
        """Train the adapter on base forecast residuals."""
        print("    Generating base forecasts for adapter training...")
        base_forecasts = self.forecaster.forecast_batch(series_dict, horizon)

        embeddings = []
        residuals = []

        for pid in series_dict:
            if pid not in actuals_dict or pid not in base_forecasts:
                continue
            embedding = self.embedding_store.get(pid)
            if embedding is None:
                continue

            actual = actuals_dict[pid][:horizon]
            base = base_forecasts[pid][:len(actual)]
            min_len = min(len(actual), len(base), horizon)
            if min_len == 0:
                continue

            # Pad to horizon length
            res = np.zeros(horizon)
            res[:min_len] = actual[:min_len] - base[:min_len]

            if np.all(np.isfinite(res)):
                embeddings.append(embedding)
                residuals.append(res)

        if len(embeddings) < 20:
            warnings.warn(f"Too few patients ({len(embeddings)}) for adapter training")
            return self

        X = torch.tensor(np.vstack(embeddings), dtype=torch.float32)
        Y = torch.tensor(np.vstack(residuals), dtype=torch.float32)

        self._build_adapter(horizon)
        optimizer = torch.optim.Adam(self.adapter.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        # Mini-batch training
        n = len(X)
        batch_size = min(256, n)

        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            epoch_loss = 0
            n_batches = 0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                pred = self.adapter(X[idx])
                loss = loss_fn(pred, Y[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 10 == 0:
                print(f"      Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/n_batches:.4f}")

        self.adapter.eval()
        return self

    def forecast_batch(
        self,
        series_dict: Dict[str, np.ndarray],
        horizon: int = 6,
    ) -> Dict[str, np.ndarray]:
        """Forecast with adapter-based adjustment."""
        base_forecasts = self.forecaster.forecast_batch(series_dict, horizon)

        if self.adapter is None or self.embedding_store is None:
            return base_forecasts

        results = {}
        for pid in series_dict:
            base = base_forecasts.get(pid, np.zeros(horizon))
            embedding = self.embedding_store.get(pid)

            if embedding is not None:
                with torch.no_grad():
                    embed_t = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
                    adjustment = self.adapter(embed_t).squeeze().numpy()
                results[pid] = base + adjustment[:len(base)]
            else:
                results[pid] = base

        return results


class PanelFMICF:
    """
    PanelFM Option C: In-Context Fine-Tuning with similar patients.

    At inference, prepends time series from K similar patients (by embedding
    cosine similarity) as context. The foundation model implicitly learns
    cross-patient patterns from these in-context examples.
    """

    def __init__(
        self,
        forecaster: ChronosForecaster,
        patient_encoder=None,
        n_context_patients: int = 5,
    ):
        self.forecaster = forecaster
        self.patient_encoder = patient_encoder
        self.n_context_patients = n_context_patients
        self.corpus_series = None
        self.corpus_features = None
        self.corpus_ids = None

    def set_corpus(
        self,
        series_dict: Dict[str, np.ndarray],
        features: pd.DataFrame,
        id_col: str = "person_id",
    ):
        self.corpus_series = series_dict
        self.corpus_features = features
        self.corpus_ids = list(series_dict.keys())

    def forecast_batch_with_context(
        self,
        series_dict: Dict[str, np.ndarray],
        features: pd.DataFrame,
        horizon: int = 6,
        id_col: str = "person_id",
        max_context_len: int = 256,
    ) -> Dict[str, np.ndarray]:
        """
        Forecast with similar patients' series prepended as context.

        For each target patient:
        1. Find K nearest neighbors in embedding space
        2. Concatenate their series as context prefix
        3. Forecast from the combined sequence
        """
        if self.patient_encoder is None or self.corpus_features is None:
            return self.forecaster.forecast_batch(series_dict, horizon)

        # Pre-compute nearest neighbors for all test patients
        test_pids = [pid for pid in series_dict if pid in features[id_col].values]

        if len(test_pids) == 0:
            return self.forecaster.forecast_batch(series_dict, horizon)

        test_features = features[features[id_col].isin(test_pids)]
        distances, indices = self.patient_encoder.get_similar_patients(
            test_features, self.corpus_features, n_neighbors=self.n_context_patients
        )

        # Build augmented series for each patient
        augmented_series = {}
        pid_to_idx = {pid: i for i, pid in enumerate(test_features[id_col].values)}

        for pid in series_dict:
            target_ts = series_dict[pid]

            if pid in pid_to_idx:
                idx = pid_to_idx[pid]
                nn_indices = indices[idx]

                # Concatenate similar patients' recent history
                context_parts = []
                remaining_budget = max_context_len - len(target_ts)

                for nn_idx in nn_indices:
                    if remaining_budget <= 0:
                        break
                    nn_pid = self.corpus_ids[nn_idx]
                    if nn_pid in self.corpus_series:
                        nn_ts = self.corpus_series[nn_pid]
                        # Take last portion that fits budget
                        chunk = nn_ts[-min(len(nn_ts), remaining_budget // self.n_context_patients):]
                        context_parts.append(chunk)
                        remaining_budget -= len(chunk)

                if context_parts:
                    context = np.concatenate(context_parts)
                    augmented_series[pid] = np.concatenate([context, target_ts]).astype(np.float32)
                else:
                    augmented_series[pid] = target_ts
            else:
                augmented_series[pid] = target_ts

        return self.forecaster.forecast_batch(augmented_series, horizon)
