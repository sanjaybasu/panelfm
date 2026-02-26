"""
Time series baselines for per-patient claims forecasting.

These models capture temporal dynamics but (mostly) ignore cross-patient structure.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import warnings


class ARIMABaseline:
    """
    Per-patient ARIMA via StatsForecast.
    Fits an auto-ARIMA to each patient's monthly time series independently.
    """

    def __init__(self, season_length: int = 12):
        self.season_length = season_length

    def forecast_batch(
        self,
        series_dict: Dict[str, np.ndarray],
        horizon: int = 6,
    ) -> Dict[str, np.ndarray]:
        """Forecast for each patient independently."""
        try:
            from statsforecast import StatsForecast
            from statsforecast.models import AutoARIMA
        except ImportError:
            warnings.warn("statsforecast not installed")
            return {pid: np.full(horizon, np.mean(ts)) for pid, ts in series_dict.items()}

        results = {}

        for pid, ts in series_dict.items():
            if len(ts) < 6:
                results[pid] = np.full(horizon, np.mean(ts))
                continue

            df = pd.DataFrame({
                "unique_id": pid,
                "ds": pd.date_range("2023-01-01", periods=len(ts), freq="MS"),
                "y": ts,
            })

            try:
                sf = StatsForecast(
                    models=[AutoARIMA(season_length=self.season_length)],
                    freq="MS",
                    n_jobs=1,
                )
                forecast = sf.forecast(df=df, h=horizon)
                results[pid] = forecast["AutoARIMA"].values[:horizon]
            except Exception:
                results[pid] = np.full(horizon, np.mean(ts))

        return results


class NaiveBaselines:
    """
    Simple baselines for reference.
    """

    @staticmethod
    def last_value(series_dict: Dict[str, np.ndarray], horizon: int = 6):
        """Predict last observed value for all future months."""
        return {pid: np.full(horizon, ts[-1]) for pid, ts in series_dict.items()}

    @staticmethod
    def trailing_mean(series_dict: Dict[str, np.ndarray], horizon: int = 6, window: int = 3):
        """Predict trailing mean of last `window` months."""
        return {
            pid: np.full(horizon, np.mean(ts[-window:]))
            for pid, ts in series_dict.items()
        }

    @staticmethod
    def seasonal_naive(series_dict: Dict[str, np.ndarray], horizon: int = 6, period: int = 12):
        """Predict same month from previous year."""
        results = {}
        for pid, ts in series_dict.items():
            forecast = np.zeros(horizon)
            for h in range(horizon):
                idx = len(ts) - period + (h % period)
                if 0 <= idx < len(ts):
                    forecast[h] = ts[idx]
                else:
                    forecast[h] = np.mean(ts)
            results[pid] = forecast
        return results


class DeepARBaseline:
    """
    DeepAR via GluonTS — probabilistic RNN that CAN pool across patients.
    This is the strongest existing baseline because it learns shared
    temporal patterns across the panel.
    """

    def __init__(
        self,
        prediction_length: int = 6,
        context_length: int = 12,
        num_layers: int = 2,
        hidden_size: int = 40,
        dropout_rate: float = 0.1,
        epochs: int = 50,
        batch_size: int = 64,
        num_batches_per_epoch: int = 100,
    ):
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.predictor = None

    def fit(self, series_dict: Dict[str, np.ndarray], start_date: str = "2023-01-01"):
        """Train DeepAR on the panel of patient time series."""
        try:
            from gluonts.dataset.common import ListDataset
            from gluonts.torch.model.deepar import DeepAREstimator
        except ImportError:
            warnings.warn("gluonts not installed")
            return self

        # Build GluonTS dataset
        dataset = ListDataset(
            [
                {
                    "target": ts.tolist(),
                    "start": pd.Timestamp(start_date),
                    "item_id": pid,
                }
                for pid, ts in series_dict.items()
                if len(ts) >= self.context_length + self.prediction_length
            ],
            freq="MS",
        )

        estimator = DeepAREstimator(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            dropout_rate=self.dropout_rate,
            trainer_kwargs={
                "max_epochs": self.epochs,
            },
            batch_size=self.batch_size,
            num_batches_per_epoch=self.num_batches_per_epoch,
        )

        self.predictor = estimator.train(dataset)
        return self

    def forecast_batch(
        self,
        series_dict: Dict[str, np.ndarray],
        horizon: int = 6,
        start_date: str = "2023-01-01",
    ) -> Dict[str, np.ndarray]:
        """Generate point forecasts for each patient."""
        if self.predictor is None:
            warnings.warn("DeepAR not fitted; returning mean forecasts")
            return {pid: np.full(horizon, np.mean(ts)) for pid, ts in series_dict.items()}

        try:
            from gluonts.dataset.common import ListDataset
        except ImportError:
            return {pid: np.full(horizon, np.mean(ts)) for pid, ts in series_dict.items()}

        dataset = ListDataset(
            [
                {
                    "target": ts.tolist(),
                    "start": pd.Timestamp(start_date),
                    "item_id": pid,
                }
                for pid, ts in series_dict.items()
            ],
            freq="MS",
        )

        forecasts = list(self.predictor.predict(dataset))
        results = {}
        pids = list(series_dict.keys())

        for i, forecast in enumerate(forecasts):
            pid = pids[i]
            # DeepAR returns SampleForecast; take median
            results[pid] = np.median(forecast.samples, axis=0)[:horizon]

        return results
