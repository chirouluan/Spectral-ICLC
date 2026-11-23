"""Read-only public food-spectrum loaders and group-safe three-way partitions.

Released spectrum IDs are not silently promoted to physical preparation IDs.
Method provenance and unresolved ambiguities live in configs/public_datasets.yaml.
"""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml

from .io import file_hash


@dataclass
class PublicData:
    dataset_id: str
    x: np.ndarray
    axis: np.ndarray
    meta: pd.DataFrame
    spec: dict
    source_hashes: dict
    audit: dict


def read_public_config(path="configs/public_datasets.yaml"):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    records = cfg["datasets"]
    for name, spec in list(records.items()):
        if "extends" in spec:
            records[name] = {**deepcopy(records[spec["extends"]]), **spec}
    return cfg


def _table(x, labels, ids, groups, **extra):
    return np.asarray(x, dtype=float), pd.DataFrame(dict(
        sample_id=ids, group_id=groups, label=np.asarray(labels, dtype=str), **extra))


def load_public(dataset_id, config=None):
    config = config or read_public_config()
    spec = config["datasets"][dataset_id]
    path = Path(config["raw_root"])/spec["folder"]/spec["file"]
    source_paths = [path, path.parent/spec["paper"]]
    kind = spec["loader"]
    audit = {}
    if kind == "honey":
        frame = pd.read_csv(path)
        spectral = [c for c in frame if c.endswith("nm")]
        axis = np.array([float(c[:-2]) for c in spectral])
        excluded = frame.Concentration_Class.eq(100)
        audit["excluded_pure_sugar_rows"] = int(excluded.sum())
        frame = frame.loc[~excluded].copy()
        groups = frame[["Brand", "Class", "Acquisition"]].astype(str).agg("|".join, axis=1)
        x, meta = _table(frame[spectral], frame.Concentration_Class.astype(int).astype(str),
            [f"honey_row_{i:05d}" for i in frame.index], groups.to_numpy(),
            acquisition=frame.Acquisition.to_numpy(), brand=frame.Brand.to_numpy(),
            botanical_class=frame.Class.to_numpy(), actual_concentration=frame.Concentration.to_numpy())
        audit["input_mean_abs_row_mean"] = float(np.abs(x.mean(axis=1)).mean())
        audit["input_mean_row_sample_sd"] = float(x.std(axis=1, ddof=1).mean())
        audit["independence_scope"] = "acquisition, not independent honey jar"
    elif kind == "turmeric":
        frame = pd.read_excel(path, sheet_name="Raw spectra region 1750-200", header=2)
        axis = np.asarray(frame.columns[2:], dtype=float)
        pct = frame.iloc[:, 1].to_numpy(float)*100
        labels = [format(v, "g") for v in pct]
        if dataset_id == "turmeric_authenticity":
            labels = np.where(pct == 0, "authentic", "adulterated")
        base = frame.iloc[:, 0].astype(str).to_numpy()
        x, meta = _table(frame.iloc[:, 2:], labels,
            [f"turmeric_{i:03d}" for i in range(len(frame))], base,
            base_batch=base, concentration_pct=pct)
        with pd.ExcelFile(path) as book:
            audit["excluded_sheets"] = [s for s in book.sheet_names if s != "Raw spectra region 1750-200"]
        audit["independence_scope"] = "all seven concentrations from a base turmeric batch stay together"
    elif kind == "beef":
        matrices, labels, official, ids = [], [], [], []
        for part, p in [("train", path), ("test", path.parent/spec["test_file"])]:
            source_paths.append(p)
            lines = p.read_text(encoding="utf-8").splitlines()
            if "@univariate true" not in lines or "@seriesLength 470" not in lines:
                raise ValueError("Unsupported Beef TS schema")
            for line in lines[lines.index("@data")+1:]:
                if not line.strip() or line.startswith("#"):
                    continue
                values, label = line.rsplit(":", 1)
                matrices.append([float(v) for v in values.split(",")])
                labels.append(label.strip())
                official.append(part)
                ids.append(f"beef_{part}_{len(ids):03d}")
        x, meta = _table(matrices, labels, ids, ids, official_partition=official)
        # An index is deliberately not called a measured wavenumber.
        axis = np.arange(x.shape[1], dtype=float)
        audit.update(axis_unit="channel_index", physical_group_ids_available=False,
                     input_max_abs_row_mean=float(np.abs(x.mean(axis=1)).max()),
                     input_mean_row_sample_sd=float(x.std(axis=1, ddof=1).mean()))
    else:
        frame = pd.read_csv(path)
        valid = pd.to_numeric(frame.iloc[:, 0], errors="coerce").notna()
        axis = frame.loc[valid].iloc[:, 0].to_numpy(float)
        spectra = frame.loc[valid].iloc[:, 1:].to_numpy(float).T
        headers = [str(c).strip() for c in frame.columns[1:]]
        if kind in {"oil", "coffee"}:
            labels = frame.iloc[1, 1:].astype(str).str.strip().to_numpy()
            ids = [f"{kind}_spectrum_{i:03d}" for i in range(len(headers))]
            # pandas adds .1 to duplicate sample-number column names.
            groups = [f"{kind}_sample_{h.split('.')[0]}" for h in headers]
        elif kind == "meats":
            matches = [re.fullmatch(r"Fresh(Chicken|Pork|Turkey)_(Supplier\w+)_Sample(\d+)_Run([AB])", h) for h in headers]
            if not all(matches):
                raise ValueError("Unrecognized meat sample identity")
            labels = [m[1] for m in matches]
            groups = [f"{m[1]}_{m[2]}_Sample{m[3]}" for m in matches]
            ids = [f"meat_spectrum_{i:03d}" for i in range(len(headers))]
            audit["duplicate_run_headers"] = pd.Series(headers)[pd.Series(headers).duplicated()].tolist()
            audit["header_identity_note"] = "SupplierD Pork Sample1 is labelled RunB twice; preserve both column records with unique spectrum IDs and one shared piece group, without guessing RunA"
        elif kind == "fruit":
            labels = [re.sub(r"\.\d+$", "", h) for h in headers]
            if set(labels) != {"Strawberry", "NON-Strawberry"}:
                raise ValueError("Unexpected fruit labels")
            ids = [f"fruit_spectrum_{i:04d}" for i in range(len(headers))]
            groups = ids
            audit["physical_group_ids_available"] = False
            audit["independence_scope"] = "released spectrum only; shared fruit batches/mixture parents unavailable"
        else:
            raise ValueError(f"Unknown public loader: {kind}")
        x, meta = _table(spectra, labels, ids, groups)
        if kind in {"oil", "meats"}:
            if not meta.groupby("group_id").size().eq(2).all():
                raise ValueError("Expected exactly two acquisitions per physical sample")
    order = np.argsort(axis)
    axis, x = axis[order], x[:, order]
    if x.shape != (len(meta), len(axis)) or not np.isfinite(x).all() or not np.isfinite(axis).all():
        raise ValueError("Nonfinite or misaligned public spectra")
    if (np.diff(axis) <= 0).any() or meta.sample_id.duplicated().any():
        raise ValueError("Duplicate axis or sample identity")
    if kind not in {"honey", "turmeric"} and meta.groupby("group_id").label.nunique().max() != 1:
        raise ValueError("Conflicting labels within a physical group")
    audit.update(observations=len(meta), groups=meta.group_id.nunique(), raw_dimensions=x.shape[1],
        label_counts=meta.label.value_counts().sort_index().to_dict(),
        axis_unit=audit.get("axis_unit", "nm" if kind == "honey" else "cm-1"),
        axis_range=[float(axis.min()), float(axis.max())])
    meta["dataset_id"] = dataset_id
    return PublicData(dataset_id, x, axis, meta.reset_index(drop=True), spec,
                      {str(p): file_hash(p) for p in dict.fromkeys(source_paths)}, audit)


def _counts(n):
    if n < 3:
        raise ValueError("Each class needs at least three independent grouping units")
    ideal = np.asarray([5/8, 1/8, 2/8])*n
    counts = np.maximum(1, np.floor(ideal).astype(int))
    while counts.sum() < n:
        counts[np.argmax(ideal-counts)] += 1
    while counts.sum() > n:
        over = np.where(counts > 1, counts-ideal, -np.inf)
        counts[np.argmax(over)] -= 1
    return tuple(int(v) for v in counts)


def public_split(data, seed):
    """Never split duplicates, image segments, or turmeric base batches."""
    meta = data.meta
    roles = np.full(len(meta), "", dtype=object)
    rng = np.random.default_rng(seed)
    if data.spec["loader"] == "honey":
        roles[meta.acquisition.isin([1, 2, 3, 4])] = "train"
        roles[meta.acquisition.eq(5)] = "val"
        roles[meta.acquisition.eq(6)] = "test"
    elif data.spec["loader"] == "turmeric":
        groups = rng.permutation(sorted(meta.group_id.unique()))
        n_train, n_val, _ = _counts(len(groups))
        lookup = {g: "train" if i < n_train else "val" if i < n_train+n_val else "test" for i, g in enumerate(groups)}
        roles = meta.group_id.map(lookup).to_numpy()
    else:
        eligible = meta if data.spec["loader"] != "beef" else meta[meta.official_partition.eq("train")]
        if data.spec["loader"] == "beef":
            roles[meta.official_partition.eq("test")] = "test"
        for label, group in eligible.groupby("label", sort=True):
            groups = rng.permutation(sorted(group.group_id.unique()))
            if data.spec["loader"] == "beef":
                n_val = max(1, round(len(groups)/4))
                n_train = len(groups)-n_val
            else:
                n_train, n_val, _ = _counts(len(groups))
            for i, gid in enumerate(groups):
                roles[meta.group_id.eq(gid)] = "train" if i < n_train else "val" if i < n_train+n_val else "test"
    split = meta[["sample_id", "group_id", "label"]].copy()
    split["split"] = roles
    validate_public_split(data, split)
    return split


def validate_public_split(data, split):
    if split.sample_id.duplicated().any() or list(split.sample_id) != list(data.meta.sample_id):
        raise ValueError("Split sample identity mismatch")
    if not split.group_id.equals(data.meta.group_id) or not split.label.equals(data.meta.label):
        raise ValueError("Split must preserve original group IDs and labels")
    if set(split.split) != {"train", "val", "test"} or split.groupby("group_id").split.nunique().max() != 1:
        raise ValueError("Group leakage or missing partition")
    labels = set(data.meta.label)
    if any(set(g.label) != labels for _, g in split.groupby("split")):
        raise ValueError("Every partition must contain all candidate classes")
    if data.spec["loader"] == "beef":
        official = data.meta.official_partition.eq("test")
        if not split.loc[official, "split"].eq("test").all() or split.loc[~official, "split"].eq("test").any():
            raise ValueError("Official Beef Test must remain unchanged")
