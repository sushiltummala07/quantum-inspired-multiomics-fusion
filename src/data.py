"""
data.py — Robust multi-omics data pipeline.

Steps:
  1. Load CSVs or generate synthetic correlated omics data
  2. Quality control  : missing values, low-variance features, outlier clipping
  3. Imputation       : KNN imputer per modality
  4. Feature selection: variance + mutual information filter
  5. Dimensionality reduction to n_qubits via PCA
  6. Stratified train / val / test split
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.decomposition import PCA
from dataclasses import dataclass
from typing import Tuple, Optional
import logging

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Synthetic data generator
# ──────────────────────────────────────────────
def _make_synthetic(n_samples=400, n_features=50, n_classes=2,
                    noise=0.25, missing_rate=0.05, seed=42) -> Tuple:
    rng = np.random.default_rng(seed)
    latent_dim = 8
    latent = rng.standard_normal((n_samples, latent_dim))

    def _modality(latent, rng, n_features, noise):
        W   = rng.standard_normal((latent_dim, n_features))
        sig = latent @ W
        # add structured noise and correlated batch effect
        sig += noise * rng.standard_normal((n_samples, n_features))
        # inject some missing values
        mask = rng.random((n_samples, n_features)) < missing_rate
        sig[mask] = np.nan
        return sig

    G = _modality(latent, rng, n_features, noise)
    T = _modality(latent, rng, n_features, noise)
    P = _modality(latent, rng, n_features, noise)

    # Multi-class labels from latent projection
    proj   = latent[:, :n_classes] @ rng.standard_normal((n_classes, 1))
    thresholds = np.percentile(proj, np.linspace(0, 100, n_classes + 1)[1:-1])
    y = np.digitize(proj.ravel(), thresholds)

    return (pd.DataFrame(G), pd.DataFrame(T), pd.DataFrame(P),
            pd.Series(y, name="label"))


# ──────────────────────────────────────────────
# QC + Imputation + Feature Selection
# ──────────────────────────────────────────────
def _qc_impute_select(df: pd.DataFrame, y: np.ndarray,
                       n_top_features: int = 100) -> np.ndarray:
    """
    Returns cleaned numpy array with at most n_top_features columns.
    """
    # 1. Drop columns with >50% missing
    thresh = int(0.5 * len(df))
    df = df.dropna(axis=1, thresh=thresh)

    # 2. KNN imputation
    imputer = KNNImputer(n_neighbors=5)
    X = imputer.fit_transform(df.values)

    # 3. Remove near-zero variance features
    sel_var = VarianceThreshold(threshold=1e-4)
    X = sel_var.fit_transform(X)

    # 4. Keep top-k features by mutual information (supervised)
    k = min(n_top_features, X.shape[1])
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    top_k = np.argsort(mi)[::-1][:k]
    X = X[:, top_k]

    # 5. Standard scale
    X = StandardScaler().fit_transform(X)
    return X


def _pca_reduce(X: np.ndarray, n_components: int) -> np.ndarray:
    n_comp = min(n_components, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=n_comp, whiten=True, random_state=42)
    return pca.fit_transform(X)


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────
@dataclass
class OmicsDataset:
    G_train: np.ndarray; T_train: np.ndarray; P_train: np.ndarray
    G_val:   np.ndarray; T_val:   np.ndarray; P_val:   np.ndarray
    G_test:  np.ndarray; T_test:  np.ndarray; P_test:  np.ndarray
    y_train: np.ndarray; y_val:   np.ndarray; y_test:  np.ndarray
    n_classes: int
    n_features: int   # per modality after reduction

def load_data(cfg) -> OmicsDataset:
    dc = cfg.data

    import numpy as np
    from sklearn.model_selection import train_test_split

    try:
        # 🔥 Load preprocessed real data
        G = np.load("data/G.npy")
        T = np.load("data/T.npy")
        P = np.load("data/P.npy")
        y = np.load("data/y.npy")

        log.info("Loaded preprocessed multi-omics data (.npy files).")

    except FileNotFoundError:
        raise RuntimeError("Processed data not found. Run prepare_all_data.py first.")

 # Clip to [-π, π] for angle encoding
    G = np.clip(G, -np.pi, np.pi)
    T = np.clip(T, -np.pi, np.pi)
    P = np.clip(P, -np.pi, np.pi)
    # ✅ Ensure correct dtype
    G = G.astype("float32")
    T = T.astype("float32")
    P = P.astype("float32")
    y = y.astype("int64")

    n_classes = len(np.unique(y))
    n_feat = G.shape[1]

    log.info(f"Shapes — G:{G.shape}, T:{T.shape}, P:{P.shape}")

    # =========================
    # Stratified split
    # =========================
    idx = np.arange(len(y))

    idx_tv, idx_test = train_test_split(
        idx, test_size=dc.test_size, stratify=y, random_state=dc.random_seed
    )

    val_rel = dc.val_size / (1.0 - dc.test_size)

    idx_tr, idx_val = train_test_split(
        idx_tv, test_size=val_rel, stratify=y[idx_tv], random_state=dc.random_seed
    )

    def _s(arr, idx): return arr[idx]

    return OmicsDataset(
        G_train=_s(G, idx_tr), T_train=_s(T, idx_tr), P_train=_s(P, idx_tr), y_train=_s(y, idx_tr),
        G_val=_s(G, idx_val),   T_val=_s(T, idx_val),   P_val=_s(P, idx_val),   y_val=_s(y, idx_val),
        G_test=_s(G, idx_test), T_test=_s(T, idx_test), P_test=_s(P, idx_test), y_test=_s(y, idx_test),
        n_classes=n_classes,
        n_features=n_feat
    )
