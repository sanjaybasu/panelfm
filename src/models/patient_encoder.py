"""
Patient Encoder: Extract compact patient representations from XGBoost.

This is the bridge between cross-sectional ML and temporal foundation models.
It produces a low-dimensional embedding that captures all cross-patient
heterogeneity that XGBoost can learn, which is then used to condition TimesFM.

Two embedding strategies:
1. Leaf-node embedding: Which leaf each patient lands in across all trees → PCA
2. SHAP-value embedding: Per-feature SHAP contributions → PCA
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from typing import Optional
import xgboost as xgb


class PatientEncoder:
    """
    Extracts compact patient embeddings from a trained XGBoost model.

    The embedding captures nonlinear feature interactions (comorbidity patterns,
    demographic × utilization interactions, etc.) in a low-dimensional space
    suitable for conditioning a time series foundation model.
    """

    def __init__(
        self,
        embedding_dim: int = 8,
        method: str = "leaf",  # "leaf" or "shap"
        include_risk_score: bool = True,
    ):
        self.embedding_dim = embedding_dim
        self.method = method
        self.include_risk_score = include_risk_score
        self.pca = None
        self.scaler = None
        self.xgb_model = None

    def fit(self, xgb_baseline, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fit the encoder using a trained XGBoost baseline.

        Args:
            xgb_baseline: Trained XGBoostBaseline instance
            X_train: Training features (same as used to train XGBoost)
            y_train: Training targets
        """
        self.xgb_model = xgb_baseline

        if self.method == "leaf":
            raw_embed = self._get_leaf_encoding(X_train)
        elif self.method == "shap":
            raw_embed = self._get_shap_encoding(X_train)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Standardize then PCA
        self.scaler = StandardScaler()
        raw_scaled = self.scaler.fit_transform(raw_embed)

        n_components = min(self.embedding_dim, raw_scaled.shape[1], raw_scaled.shape[0])
        self.pca = PCA(n_components=n_components, random_state=42)
        self.pca.fit(raw_scaled)

        print(f"PatientEncoder fitted: {raw_embed.shape[1]} raw dims → "
              f"{n_components} PCA dims "
              f"(explained variance: {self.pca.explained_variance_ratio_.sum():.3f})")

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Produce patient embeddings for new data.

        Returns:
            (n_patients, embedding_dim) array.
            If include_risk_score=True, last column is the XGBoost risk score.
        """
        if self.method == "leaf":
            raw = self._get_leaf_encoding(X)
        elif self.method == "shap":
            raw = self._get_shap_encoding(X)

        raw_scaled = self.scaler.transform(raw)
        embedding = self.pca.transform(raw_scaled)

        if self.include_risk_score:
            risk_score = self.xgb_model.predict(X).reshape(-1, 1)
            # Normalize risk score to same scale
            risk_score = (risk_score - risk_score.mean()) / (risk_score.std() + 1e-8)
            embedding = np.hstack([embedding, risk_score])

        return embedding.astype(np.float32)

    def _get_leaf_encoding(self, X: pd.DataFrame) -> np.ndarray:
        """One-hot encode leaf assignments across all trees."""
        leaves = self.xgb_model.get_leaf_embeddings(X)  # (n_samples, n_trees)

        # Convert to dense float (each tree's leaf index is a categorical)
        # Use raw leaf indices; PCA will handle dimensionality
        return leaves.astype(np.float32)

    def _get_shap_encoding(self, X: pd.DataFrame) -> np.ndarray:
        """Use per-feature SHAP values as embedding."""
        X_arr = self.xgb_model._prepare_X(X)
        booster = self.xgb_model.model.get_booster()
        dmat = xgb.DMatrix(X_arr, feature_names=self.xgb_model.feature_names)
        shap_values = booster.predict(dmat, pred_contribs=True)
        # Drop the bias column (last column)
        return shap_values[:, :-1].astype(np.float32)

    def get_similar_patients(
        self,
        X_query: pd.DataFrame,
        X_corpus: pd.DataFrame,
        n_neighbors: int = 5,
    ) -> tuple:
        """
        Find similar patients in embedding space (for ICF context selection).

        Returns:
            distances: (n_query, n_neighbors) array
            indices: (n_query, n_neighbors) array of corpus indices
        """
        query_embed = self.transform(X_query)
        corpus_embed = self.transform(X_corpus)

        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        nn.fit(corpus_embed)
        distances, indices = nn.kneighbors(query_embed)

        return distances, indices


class PatientEmbeddingStore:
    """
    Stores and retrieves patient embeddings keyed by person_id.
    Used to pass embeddings to TimesFM as static covariates.
    """

    def __init__(self):
        self.embeddings = {}  # person_id → np.ndarray
        self.embedding_dim = None

    def build(
        self,
        encoder: PatientEncoder,
        features: pd.DataFrame,
        id_col: str = "person_id",
    ):
        """Compute and store embeddings for all patients."""
        ids = features[id_col].values
        embeddings = encoder.transform(features)
        self.embedding_dim = embeddings.shape[1]

        for i, pid in enumerate(ids):
            self.embeddings[pid] = embeddings[i]

        print(f"Stored embeddings for {len(self.embeddings)} patients "
              f"(dim={self.embedding_dim})")
        return self

    def get(self, person_id: str) -> Optional[np.ndarray]:
        """Retrieve embedding for a single patient."""
        return self.embeddings.get(person_id)

    def get_batch(self, person_ids: list) -> np.ndarray:
        """Retrieve embeddings for a batch of patients."""
        result = np.zeros((len(person_ids), self.embedding_dim), dtype=np.float32)
        for i, pid in enumerate(person_ids):
            if pid in self.embeddings:
                result[i] = self.embeddings[pid]
        return result

    def to_static_covariates(self, person_ids: list) -> pd.DataFrame:
        """
        Format embeddings as a DataFrame suitable for TimesFM XReg.

        Returns DataFrame with columns: person_id, embed_0, embed_1, ..., embed_d
        """
        embeddings = self.get_batch(person_ids)
        cols = {f"embed_{i}": embeddings[:, i] for i in range(self.embedding_dim)}
        cols["person_id"] = person_ids
        return pd.DataFrame(cols)
