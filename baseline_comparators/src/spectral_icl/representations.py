"""All fitted representations are fit exclusively on the actual reference pool."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter


class Representation:
    def __init__(self, name="snv", components=8):
        self.name, self.components = name, components
        self.scaler = self.pca = None

    def basic(self, x):
        x = np.asarray(x, dtype=float)
        if self.name == "snv":
            return (x - x.mean(axis=1, keepdims=True)) / np.maximum(x.std(axis=1, keepdims=True), 1e-12)
        if self.name == "sg":
            return savgol_filter(x, 11, 2, axis=1)
        if self.name == "spectral_descriptors":
            blocks = np.array_split(x, min(20, x.shape[1]), axis=1)
            return np.concatenate([np.stack([b.mean(axis=1), b.std(axis=1)], axis=1) for b in blocks], axis=1)
        if self.name not in ("raw", "pca_support_only"):
            raise ValueError(f"Unknown representation {self.name}")
        return x

    def fit(self, x):
        x = self.basic(x)
        if self.name == "pca_support_only":
            self.scaler = StandardScaler().fit(x)
            self.pca = PCA(n_components=min(self.components, len(x)-1, x.shape[1]), svd_solver="full")
            self.pca.fit(self.scaler.transform(x))
        return self

    def transform(self, x):
        x = self.basic(x)
        return self.pca.transform(self.scaler.transform(x)) if self.pca is not None else x
