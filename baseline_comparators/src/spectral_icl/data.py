"""Read-only workbooks, explicit ID linkage and immutable grouped splits."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .io import file_hash, write_json

LEVELS = [0, 1, 5, 10, 15, 20, 30, 40, 50, 100]
DAYS = [0, 7, 14]


def load_surimi(config):
    cfg = config["data"]
    mapping = pd.read_csv(cfg["mapping"])
    required = {"day", "source_record_id", "concentration_pct", "physical_sample_id", "mapping_source"}
    if not required <= set(mapping):
        raise ValueError(f"Mapping missing columns: {required - set(mapping)}")
    if mapping[list(required)].isna().any().any():
        raise ValueError("Incomplete mapping or provenance")
    if mapping.duplicated(["day", "source_record_id"]).any():
        raise ValueError("Duplicate source IDs in mapping")
    frames, matrices, hashes, wavelengths = [], [], {}, None
    for day in DAYS:
        path = Path(cfg["workbook_template"].format(day=day))
        frame = pd.read_excel(path)
        wave = np.asarray([float(re.sub(r"\s*nm\s*$", "", str(c))) for c in frame.columns[2:]], dtype=float)
        if wavelengths is not None and not np.array_equal(wavelengths, wave):
            raise ValueError("Wavelength axes differ across days")
        wavelengths = wave
        x = frame.iloc[:, 2:].to_numpy(dtype=float)
        if not np.isfinite(x).all() or not np.isfinite(wave).all() or np.any(np.diff(wave) <= 0):
            raise ValueError("Nonfinite spectra or invalid wavelength axis")
        obs = pd.DataFrame({"source_record_id": frame.iloc[:, 0].astype(str),
                            "concentration_pct": frame.iloc[:, 1].astype(float), "day": day})
        if obs.duplicated(["day", "source_record_id"]).any():
            raise ValueError("Duplicate workbook observation")
        # Preserve workbook order explicitly; merge must never silently align by label or row order.
        obs["_row"] = np.arange(len(obs))
        dm = mapping[mapping.day == day]
        if set(dm.source_record_id) != set(obs.source_record_id):
            raise ValueError(f"Mapping/workbook records disagree on day {day}")
        merged = obs.merge(dm, on=["day", "source_record_id", "concentration_pct"],
                           how="left", validate="one_to_one").sort_values("_row")
        if merged.physical_sample_id.isna().any():
            raise ValueError("Mapping label mismatch")
        merged["hyperspectral_file"] = str(path.resolve())
        merged["acquisition_id"] = merged.physical_sample_id + "_D" + str(day)
        merged["n_valid_pixels"] = None
        merged["group_batch_id"] = None
        merged["qc_status"] = "pass"
        merged["wavelength_metadata_id"] = "surimi_workbook_axis_v1"
        frames.append(merged.drop(columns="_row"))
        matrices.append(x)
        hashes[str(path)] = file_hash(path)
    meta = pd.concat(frames, ignore_index=True)
    x = np.vstack(matrices)
    if meta.groupby("physical_sample_id").concentration_pct.nunique().max() != 1:
        raise ValueError("Physical ID has conflicting labels")
    if meta.duplicated(["physical_sample_id", "day"]).any():
        raise ValueError("Duplicate sample-time observations; explicit ROI aggregation required")
    if not (meta.groupby("physical_sample_id").day.agg(set).apply(lambda v: v == set(DAYS))).all():
        raise ValueError("Missing sample-time observations")
    ids = meta.drop_duplicates("physical_sample_id")
    counts = ids.groupby("concentration_pct").size()
    expected_levels = cfg.get("levels", LEVELS)
    if set(counts.index) != set(expected_levels) or not (counts == 16).all():
        raise ValueError("Dataset must contain exactly configured classes x 16 physical samples")
    return meta, x, wavelengths, {**hashes, str(cfg["mapping"]): file_hash(cfg["mapping"])}


def audit_data(config, output):
    meta, x, wave, hashes = load_surimi(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    meta.to_csv(output / "metadata.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(output / "spectra.npz", spectra=x, wavelengths_nm=wave)
    report = {"status": "pass", "observations": len(meta),
              "physical_samples": meta.physical_sample_id.nunique(), "bands": len(wave),
              "wavelength_range_nm": [float(wave[0]), float(wave[-1])],
              "counts": meta.groupby(["day", "concentration_pct"]).size().to_string(),
              "source_sha256": hashes,
              "mapping_evidence": sorted(meta.mapping_source.unique()),
              "unknown": ["preparation batch", "ROI pixel counts", "instrument", "ROI averaging provenance"],
              "independence": "Physical IDs use supplied user-confirmed mapping; not independently verified preparation batches"}
    write_json(output / "audit.json", report)
    return report


def validate_split(meta, split):
    ids = meta.drop_duplicates("physical_sample_id")
    if split.physical_sample_id.duplicated().any() or set(ids.physical_sample_id) != set(split.physical_sample_id):
        raise ValueError("Split ID mismatch or duplicates")
    joined = ids[["physical_sample_id", "concentration_pct"]].merge(
        split, on="physical_sample_id", suffixes=("_data", "_split"), validate="one_to_one")
    if not (joined.concentration_pct_data == joined.concentration_pct_split).all():
        raise ValueError("Split labels disagree")
    for _, group in joined.groupby("concentration_pct_data"):
        if group.split.value_counts().to_dict() != {"train": 10, "val": 2, "test": 4}:
            raise ValueError("Split must be 10/2/4 per class")


def create_splits(meta, directory, repeats=5, seed=20261012):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)
        rows = []
        for label, group in meta.drop_duplicates("physical_sample_id").groupby("concentration_pct"):
            ids = rng.permutation(sorted(group.physical_sample_id))
            for i, sid in enumerate(ids):
                rows.append(dict(dataset_id="surimi_2022", physical_sample_id=sid,
                                 concentration_pct=label, split="test" if i < 4 else "val" if i < 6 else "train",
                                 repeat=repeat, split_seed=seed + repeat, split_version="v1"))
        split = pd.DataFrame(rows)
        validate_split(meta, split)
        split.to_csv(directory / f"surimi_split_v1_repeat_{repeat}.csv", index=False)
    write_json(directory / "manifest.json", {"repeats": repeats, "seed": seed,
               "hashes": {p.name: file_hash(p) for p in directory.glob("*.csv")}})
