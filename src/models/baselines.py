"""
Cross-sectional ML baselines for claims risk prediction.

These models treat each prediction window as i.i.d. (ignoring temporal dynamics).
They represent the current industry standard for claims risk modeling.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
try:
    import lightgbm as lgb
except Exception:
    lgb = None
from typing import Optional
import yaml
import warnings


def load_model_config(config_path: str = "configs/model_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class CrossSectionalBaseline:
    """
    Base class for cross-sectional claims risk models.

    These models predict a future outcome (cost, utilization, high-cost flag)
    from a feature vector summarizing the patient's lookback window.
    """

    def __init__(self, task: str = "regression"):
        """
        Args:
            task: 'regression' for cost prediction, 'classification' for high-cost flag
        """
        self.task = task
        self.model = None
        self.scaler = None
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Point predictions."""
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability predictions (classification only)."""
        raise NotImplementedError

    def _prepare_X(self, X: pd.DataFrame) -> np.ndarray:
        """Select numeric features and handle NaN."""
        if self.feature_names is None:
            self.feature_names = X.select_dtypes(include=[np.number]).columns.tolist()
        X_num = X[self.feature_names].fillna(0).values
        return X_num


class XGBoostBaseline(CrossSectionalBaseline):
    """XGBoost gradient-boosted tree model for claims risk prediction."""

    def __init__(self, task: str = "regression", config: Optional[dict] = None):
        super().__init__(task)
        if config is None:
            config = {}
        self.config = config

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None):
        X_arr = self._prepare_X(X)

        params = {
            "n_estimators": self.config.get("n_estimators", 500),
            "max_depth": self.config.get("max_depth", 6),
            "learning_rate": self.config.get("learning_rate", 0.05),
            "subsample": self.config.get("subsample", 0.8),
            "colsample_bytree": self.config.get("colsample_bytree", 0.8),
            "min_child_weight": self.config.get("min_child_weight", 10),
            "reg_alpha": self.config.get("reg_alpha", 0.1),
            "reg_lambda": self.config.get("reg_lambda", 1.0),
            "random_state": 42,
            "n_jobs": -1,
        }

        # Only use early_stopping if eval_set will be provided
        es_rounds = self.config.get("early_stopping_rounds", 50) if eval_set is not None else None

        if self.task == "classification":
            self.model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="aucpr",
                early_stopping_rounds=es_rounds,
                **params,
            )
        else:
            self.model = xgb.XGBRegressor(
                objective="reg:squarederror",
                eval_metric="mae",
                early_stopping_rounds=es_rounds,
                **params,
            )

        fit_params = {}
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val_arr = self._prepare_X(X_val)
            fit_params["eval_set"] = [(X_val_arr, y_val)]

        self.model.fit(X_arr, y, **fit_params)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._prepare_X(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba only for classification task")
        return self.model.predict_proba(self._prepare_X(X))[:, 1]

    def get_leaf_embeddings(self, X: pd.DataFrame) -> np.ndarray:
        """
        Extract leaf-node embeddings: for each sample, record which leaf
        it lands in across all trees. Returns (n_samples, n_trees) int array.
        """
        X_arr = self._prepare_X(X)
        return self.model.get_booster().predict(
            xgb.DMatrix(X_arr, feature_names=self.feature_names),
            pred_leaf=True,
        )

    def feature_importance(self) -> pd.Series:
        imp = self.model.feature_importances_
        return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)


class RandomForestBaseline(CrossSectionalBaseline):
    """Random Forest baseline."""

    def __init__(self, task: str = "regression", config: Optional[dict] = None):
        super().__init__(task)
        if config is None:
            config = {}
        self.config = config

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None):
        X_arr = self._prepare_X(X)

        params = {
            "n_estimators": self.config.get("n_estimators", 500),
            "max_depth": self.config.get("max_depth", 12),
            "min_samples_leaf": self.config.get("min_samples_leaf", 20),
            "max_features": self.config.get("max_features", "sqrt"),
            "random_state": 42,
            "n_jobs": -1,
        }

        if self.task == "classification":
            self.model = RandomForestClassifier(**params)
        else:
            self.model = RandomForestRegressor(**params)

        self.model.fit(X_arr, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._prepare_X(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba only for classification task")
        return self.model.predict_proba(self._prepare_X(X))[:, 1]


class LightGBMBaseline(CrossSectionalBaseline):
    """LightGBM baseline."""

    def __init__(self, task: str = "regression", config: Optional[dict] = None):
        super().__init__(task)
        if config is None:
            config = {}
        self.config = config

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None):
        X_arr = self._prepare_X(X)

        params = {
            "n_estimators": self.config.get("n_estimators", 500),
            "max_depth": self.config.get("max_depth", 6),
            "learning_rate": self.config.get("learning_rate", 0.05),
            "subsample": self.config.get("subsample", 0.8),
            "colsample_bytree": self.config.get("colsample_bytree", 0.8),
            "min_child_samples": self.config.get("min_child_samples", 20),
            "reg_alpha": self.config.get("reg_alpha", 0.1),
            "reg_lambda": self.config.get("reg_lambda", 1.0),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        if self.task == "classification":
            self.model = lgb.LGBMClassifier(objective="binary", **params)
        else:
            self.model = lgb.LGBMRegressor(objective="regression", **params)

        fit_params = {}
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val_arr = self._prepare_X(X_val)
            fit_params["eval_set"] = [(X_val_arr, y_val)]

        self.model.fit(X_arr, y, **fit_params)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._prepare_X(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba only for classification task")
        return self.model.predict_proba(self._prepare_X(X))[:, 1]


class StackingEnsemble(CrossSectionalBaseline):
    """
    Stacking ensemble: XGBoost + RF + LightGBM with logistic meta-learner.
    """

    def __init__(self, task: str = "regression", config: Optional[dict] = None):
        super().__init__(task)
        full_config = config or {}
        self.base_models = {
            "xgboost": XGBoostBaseline(task, full_config.get("xgboost", {})),
            "random_forest": RandomForestBaseline(task, full_config.get("random_forest", {})),
            "lightgbm": LightGBMBaseline(task, full_config.get("lightgbm", {})),
        }
        self.meta_model = None

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None):
        # Fit base models
        for name, model in self.base_models.items():
            model.fit(X, y, eval_set=eval_set)

        # Generate stacked features via cross-validation predictions
        X_arr = self._prepare_X(X)
        n = len(X_arr)
        stacked = np.zeros((n, len(self.base_models)))

        # Simple holdout-based stacking (could use k-fold for more rigor)
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_arr)):
            X_fold_train = X.iloc[train_idx]
            y_fold_train = y.iloc[train_idx]
            X_fold_val = X.iloc[val_idx]

            for j, (name, model_class) in enumerate(
                [
                    ("xgb", XGBoostBaseline),
                    ("rf", RandomForestBaseline),
                    ("lgb", LightGBMBaseline),
                ]
            ):
                fold_model = model_class(self.task)
                fold_model._prepare_X(X)  # Set feature names
                fold_model.fit(X_fold_train, y_fold_train)
                if self.task == "classification":
                    stacked[val_idx, j] = fold_model.predict_proba(X_fold_val)
                else:
                    stacked[val_idx, j] = fold_model.predict(X_fold_val)

        # Fit meta-learner
        if self.task == "classification":
            self.meta_model = LogisticRegression(max_iter=1000)
        else:
            self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(stacked, y)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        stacked = self._get_base_predictions(X)
        return self.meta_model.predict(stacked)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba only for classification task")
        stacked = self._get_base_predictions(X)
        return self.meta_model.predict_proba(stacked)[:, 1]

    def _get_base_predictions(self, X: pd.DataFrame) -> np.ndarray:
        preds = []
        for name, model in self.base_models.items():
            if self.task == "classification":
                preds.append(model.predict_proba(X))
            else:
                preds.append(model.predict(X))
        return np.column_stack(preds)


class TwoPartModel(CrossSectionalBaseline):
    """
    Two-part model for zero-inflated cost data.

    Part 1: XGBoost classifier for P(cost > 0)
    Part 2: XGBoost regressor for E[cost | cost > 0]
    Final prediction: P(cost > 0) * E[cost | cost > 0]

    This addresses the zero-inflation problem common in claims data where
    60%+ of patient-months have $0 cost.
    """

    def __init__(self, task: str = "regression", config: Optional[dict] = None):
        super().__init__(task)
        if config is None:
            config = {}
        self.config = config
        self.part1_classifier = None
        self.part2_regressor = None
        self.fallback_mean = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None):
        X_arr = self._prepare_X(X)
        y_arr = y.values if hasattr(y, "values") else np.array(y, dtype=float)

        # Part 1: P(cost > 0)
        y_binary = (y_arr > 0).astype(int)
        zero_frac = 1 - y_binary.mean()

        # Only fit classifier if there are both classes (zero and nonzero)
        if 0 < zero_frac < 1:
            self.part1_classifier = XGBoostBaseline(
                task="classification", config=self.config
            )
            self.part1_classifier.feature_names = self.feature_names
            self.part1_classifier.fit(X, pd.Series(y_binary))
            self._all_nonzero = False
        else:
            # All values are the same class; skip classifier
            self._all_nonzero = (zero_frac == 0)
            self.part1_classifier = None

        # Part 2: E[cost | cost > 0]
        mask = y_arr > 0
        if mask.sum() > 20:
            X_pos = X.iloc[mask]
            y_pos = pd.Series(y_arr[mask])
            self.part2_regressor = XGBoostBaseline(
                task="regression", config=self.config
            )
            self.part2_regressor.feature_names = self.feature_names
            self.part2_regressor.fit(X_pos, y_pos)
            self.fallback_mean = float(y_arr[mask].mean())
        else:
            self.fallback_mean = float(y_arr.mean())

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.part1_classifier is not None:
            prob_nonzero = self.part1_classifier.predict_proba(X)
        else:
            # No classifier fitted (all training values same class)
            prob_nonzero = np.ones(len(X)) if self._all_nonzero else np.zeros(len(X))

        if self.part2_regressor is not None:
            pred_positive = self.part2_regressor.predict(X)
            return prob_nonzero * np.maximum(pred_positive, 0)
        return prob_nonzero * self.fallback_mean

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Use Part 1 classifier probability for classification."""
        if self.part1_classifier is not None:
            return self.part1_classifier.predict_proba(X)
        # Fallback: use normalized predicted cost as risk score
        pred = self.predict(X)
        vmin, vmax = pred.min(), pred.max()
        if vmax > vmin:
            return (pred - vmin) / (vmax - vmin)
        return np.full(len(X), 0.5)


class DemographicGLM(CrossSectionalBaseline):
    """
    Demographics-only GLM baseline (actuarial minimum benchmark).

    Uses only age, sex, and dual-eligible status — the minimum set of
    predictors available in any risk adjustment context. This serves as
    the baseline against which diagnosis-based models are evaluated.

    Typical R² for demographic-only models: 1-3% (SOA 2016 study).
    """

    DEMO_COLS = ["age", "female", "dual_eligible"]

    def __init__(self, task: str = "regression", config: Optional[dict] = None):
        super().__init__(task)
        self.config = config or {}
        self._demo_features = None

    def _find_demo_cols(self, X: pd.DataFrame) -> list:
        """Find demographic columns in features."""
        found = []
        for col in self.DEMO_COLS:
            if col in X.columns:
                found.append(col)
            else:
                # Try common variants
                for variant in [f"{col}_1", f"{col}_True", col.upper(), col.lower()]:
                    if variant in X.columns:
                        found.append(variant)
                        break
        return found

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None):
        self._demo_features = self._find_demo_cols(X)
        if not self._demo_features:
            # Fallback: use age-like and gender-like columns
            for col in X.columns:
                cl = col.lower()
                if any(k in cl for k in ["age", "female", "male", "gender", "sex", "dual"]):
                    self._demo_features.append(col)

        if not self._demo_features:
            raise ValueError("No demographic columns found in features")

        X_demo = X[self._demo_features].fillna(0).values

        if self.task == "classification":
            self.model = LogisticRegression(max_iter=1000, solver="lbfgs")
            self.model.fit(X_demo, y)
        else:
            self.model = Ridge(alpha=1.0)
            self.model.fit(X_demo, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_demo = X[self._demo_features].fillna(0).values
        return self.model.predict(X_demo)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba only for classification task")
        X_demo = X[self._demo_features].fillna(0).values
        return self.model.predict_proba(X_demo)[:, 1]


# =============================================================================
# Target construction helpers
# =============================================================================

def build_targets(
    outcomes: pd.DataFrame,
    prediction_start: pd.Timestamp,
    horizon_months: int = 3,
    id_col: str = "person_id",
    high_cost_percentile: float = 0.90,
) -> pd.DataFrame:
    """
    Build prediction targets from the outcome window.

    Returns DataFrame with columns:
        person_id, total_paid_sum, ed_visits_sum, ip_admits_sum, high_cost_flag
    """
    prediction_end = prediction_start + pd.DateOffset(months=horizon_months)

    window = outcomes[
        (outcomes["month_year"] > prediction_start)
        & (outcomes["month_year"] <= prediction_end)
    ].copy()

    # Fill NaN costs/counts with 0 (enrolled but no claims)
    for col in ["total_paid", "emergency_department_ct", "acute_inpatient_ct"]:
        if col in window.columns:
            window[col] = window[col].fillna(0)

    targets = window.groupby(id_col).agg(
        total_paid_sum=("total_paid", "sum"),
        ed_visits_sum=("emergency_department_ct", "sum"),
        ip_admits_sum=("acute_inpatient_ct", "sum"),
        n_months=("month_year", "count"),
    ).reset_index()

    # High-cost flag
    threshold = targets["total_paid_sum"].quantile(high_cost_percentile)
    targets["high_cost_flag"] = (targets["total_paid_sum"] >= threshold).astype(int)

    # Any ED flag
    targets["any_ed_flag"] = (targets["ed_visits_sum"] > 0).astype(int)

    # Any IP flag
    targets["any_ip_flag"] = (targets["ip_admits_sum"] > 0).astype(int)

    return targets
