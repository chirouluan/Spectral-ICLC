"""Conservative fixed classical baselines and discrete unseen-level prototypes."""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

REGISTRY = ["deepseek_icl", "mimo_icl", "lumir_reproduced", "lumir_mul_generate",
            "svm_rbf", "random_forest", "cnn1d", "transformer",
            "spectral_centroid_interpolation", "ordinal_prototype_distance"]




class PrototypeClassifier:
    def __init__(self, candidates, extrapolate=False, ordinal=False):
        self.candidates, self.extrapolate, self.ordinal = np.array(candidates), extrapolate, ordinal

    def fit(self, x, y):
        seen = np.unique(y)
        means = np.stack([x[np.asarray(y) == label].mean(axis=0) for label in seen])
        if len(seen) < 2:
            raise ValueError("Prototype interpolation needs at least two seen classes")
        self.prototypes = []
        for label in self.candidates:
            if label in seen:
                self.prototypes.append(means[list(seen).index(label)])
                continue
            if label < seen[0] or label > seen[-1]:
                if not self.extrapolate:
                    raise ValueError("Held-out endpoint needs explicitly enabled linear extrapolation")
            j = int(np.clip(np.searchsorted(seen, label), 1, len(seen)-1))
            if self.ordinal:
                rank = {c: i for i, c in enumerate(self.candidates)}
                alpha = (rank[label] - rank[seen[j-1]]) / (rank[seen[j]] - rank[seen[j-1]])
            else:
                alpha = (label - seen[j-1]) / (seen[j] - seen[j-1])
            self.prototypes.append(means[j-1] * (1-alpha) + means[j] * alpha)
        self.prototypes = np.array(self.prototypes)
        return self

    def predict(self, x):
        distances = ((x[:, None, :] - self.prototypes[None, :, :]) ** 2).sum(axis=2)
        return self.candidates[np.argmin(distances, axis=1)]


def make_model(name, n, candidates, seed, extrapolate=False):
    if name == "svm_rbf":
        return make_pipeline(StandardScaler(), SVC(C=1, gamma="scale"))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    if name == "spectral_centroid_interpolation":
        return PrototypeClassifier(candidates, extrapolate)
    if name == "ordinal_prototype_distance":
        return PrototypeClassifier(candidates, extrapolate, ordinal=True)
    raise ValueError(f"{name}: verified adapter unavailable")
