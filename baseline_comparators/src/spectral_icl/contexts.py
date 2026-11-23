"""Support sets contain allowed train IDs only; query labels are never inputs."""
import numpy as np


def select_support(meta, x, pool, k, seed, query_x=None, strategy="random"):
    rng = np.random.default_rng(seed)
    selected = []
    for _, group in meta.loc[pool].groupby("concentration_pct", sort=True):
        candidates = np.array(sorted(group.index), dtype=int)
        if strategy in ("random", "class_balanced"):
            order = rng.permutation(candidates)
        elif strategy == "spectral_similarity":
            if query_x is None:
                raise ValueError("Similarity requires query spectrum")
            norms = np.linalg.norm(x[candidates], axis=1) * np.linalg.norm(query_x)
            scores = x[candidates] @ query_x / np.maximum(norms, 1e-15)
            order = candidates[np.argsort(-scores, kind="stable")]
        elif strategy in ("diversity", "hybrid"):
            unit = x[candidates] / np.maximum(np.linalg.norm(x[candidates], axis=1, keepdims=True), 1e-15)
            if strategy == "hybrid":
                if query_x is None:
                    raise ValueError("Hybrid requires query spectrum")
                first = int(np.argmax(unit @ query_x / max(np.linalg.norm(query_x), 1e-15)))
            else:
                first = int(rng.integers(len(candidates)))
            chosen = [first]
            while len(chosen) < min(k, len(candidates)):
                distances = np.min(1 - unit @ unit[chosen].T, axis=1)
                distances[chosen] = -np.inf
                chosen.append(int(np.argmax(distances)))
            order = candidates[chosen]
        else:
            raise ValueError(f"Unknown selector {strategy}")
        selected.extend(order[:k].tolist())
    return selected


def t3_support(meta, pool, source_day, target_day, k, seed, control):
    rng = np.random.default_rng(seed)
    selected = []
    # Source and target IDs allocated once, before query exclusion.
    for _, group in meta.loc[pool].groupby("concentration_pct"):
        ids = rng.permutation(sorted(group.physical_sample_id.unique()))
        source, target = set(ids[:5]), set(ids[5:5+k])
        if control == "source_only":
            wanted = [(source, source_day)]
        elif control == "target_only":
            wanted = [(target, target_day)]
        elif control == "mixed":
            wanted = [(source, source_day), (target, target_day)]
        elif control == "size_matched_source":
            wanted = [(source | target, source_day)]
        else:
            raise ValueError("Unknown T3 control")
        for idset, day in wanted:
            selected.extend(group[(group.day == day) & group.physical_sample_id.isin(idset)].index)
    return list(selected)
