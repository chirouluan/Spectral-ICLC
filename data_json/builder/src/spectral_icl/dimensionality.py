"""Train-only SNV + dimensionality reduction for the classifier benchmark."""
import numpy as np
from scipy.linalg import eigh
from scipy.signal import savgol_filter
from sklearn.covariance import LedoitWolf
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from .representations import Representation

METHODS = ["snv_only", "pca", "lda", "mnf_estimated_noise", "ica", "pls_scores",
           "isomap", "lle", "tsne_openTSNE", "umap"]


class EstimatedNoiseMNF:
    """Noise-adjusted PCA with a declared spectral-residual noise approximation.

    Noise = SNV spectrum minus SG(11,2) smoothed spectrum. This is not measured
    sensor noise and may contain fine spectral signal. Never difference samples
    of different classes or days to manufacture noise replicates.
    """
    def __init__(self, dimensions):
        self.dimensions = dimensions

    def fit(self, x):
        self.mean_ = x.mean(axis=0)
        residuals = x-savgol_filter(x, 11, 2, axis=1)
        noise = LedoitWolf().fit(residuals).covariance_
        signal = np.cov(x, rowvar=False)
        noise += np.eye(x.shape[1])*max(float(np.trace(noise)/x.shape[1])*1e-8, 1e-12)
        values, vectors = eigh(signal, noise, check_finite=True)
        order = np.argsort(values)[::-1]
        self.eigenvalues_ = values[order]
        self.components_ = vectors[:, order[:self.dimensions]]
        return self

    def transform(self, x):
        return (x-self.mean_) @ self.components_


class SNVReduction:
    def __init__(self, method, dimensions, seed=12345):
        self.method, self.dimensions, self.seed = method, dimensions, seed

    def fit_transform(self, x, labels):
        x = Representation("snv").transform(x)
        d = self.dimensions
        self.classes_ = np.unique(labels)
        if self.method == "snv_only":
            self.model = None
            return x
        if self.method == "pca":
            self.model = PCA(n_components=.95 if d == "95pct" else d, svd_solver="full")
        elif self.method == "lda":
            self.model = LinearDiscriminantAnalysis(n_components=d, solver="svd")
        elif self.method == "mnf_estimated_noise":
            self.model = EstimatedNoiseMNF(d).fit(x)
            return self.model.transform(x)
        elif self.method == "ica":
            self.model = FastICA(n_components=d, whiten="unit-variance", whiten_solver="svd",
                                 random_state=self.seed, max_iter=3000, tol=.001)
        elif self.method == "pls_scores":
            self.model = PLSRegression(n_components=d, scale=False, max_iter=1000, tol=1e-6)
            onehot = (np.asarray(labels)[:, None] == self.classes_[None, :]).astype(float)
            self.model.fit(x, onehot)
            return self.model.transform(x)
        elif self.method == "isomap":
            self.model = Isomap(n_components=d, n_neighbors=min(12,len(x)-1), n_jobs=1)
        elif self.method == "lle":
            self.model = LocallyLinearEmbedding(n_components=d,
                n_neighbors=min(max(12,d+2),len(x)-1), eigen_solver="dense", reg=.001,
                random_state=self.seed, n_jobs=1)
        elif self.method == "tsne_openTSNE":
            if d != 2:
                raise ValueError("The preregistered t-SNE benchmark uses two dimensions")
            from openTSNE import TSNE
            self.model = TSNE(n_components=2, perplexity=min(15,(len(x)-1)/3),
                random_state=self.seed, n_jobs=1, n_iter=500,
                initialization="pca", negative_gradient_method="bh").fit(x)
            return np.asarray(self.model).copy()
        elif self.method == "umap":
            from umap import UMAP
            self.model = UMAP(n_components=d, n_neighbors=min(12,len(x)-1), min_dist=.1,
                n_epochs=300, random_state=self.seed, transform_seed=self.seed, n_jobs=1)
        else:
            raise ValueError(f"Unknown reduction: {self.method}")
        if self.method == "lda":
            return self.model.fit_transform(x, labels)
        return self.model.fit_transform(x)

    def transform(self, x):
        x = Representation("snv").transform(x)
        if self.model is None:
            return x
        if self.method == "tsne_openTSNE":
            return np.asarray(self.model.transform(x, perplexity=5, n_iter=250))
        return self.model.transform(x)


def candidate_dimensions(method, grid, n_samples, n_features, n_classes):
    if method == "snv_only":
        return [n_features]
    if method == "tsne_openTSNE":
        return [2]
    cap = min(n_samples-1,n_features,n_classes-1 if method == "lda" else n_features)
    dims = [d for d in grid if d <= cap]
    return dims + (["95pct"] if method == "pca" else [])
