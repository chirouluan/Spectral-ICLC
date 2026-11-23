"""Read back every exported condition and verify provenance against raw data."""
from collections import Counter
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import yaml

from .data import load_surimi, validate_split
from .io import digest, file_hash, write_json
from .public_data import load_public, read_public_config, validate_public_split
from .public_validation import checksum
from .representations import Representation
from .web_batches import _anonymous
from .web_policy import CONTEXT_ONLY_POLICY


def validate_web_export(root, write_report=True):
    root = Path(root)
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    cfg = manifest["configuration"]
    for relative, expected in [line.split("  ", 1)[::-1] for line in
                               (root/"checksum.sha256").read_text(encoding="utf-8").splitlines()]:
        if file_hash(root/relative) != expected:
            raise ValueError(f"Artifact checksum mismatch: {relative}")
    for p, h in manifest["source_sha256"].items():
        if file_hash(p) != h:
            raise ValueError(f"Raw source changed: {p}")
    source_config = yaml.safe_load(Path(cfg["surimi_config"]).read_text(encoding="utf-8"))
    original, raw, _, _ = load_surimi(source_config)
    fish = original.copy()
    fish["sample_id"] = fish.acquisition_id.astype(str)
    fish["group_id"] = fish.physical_sample_id.astype(str)
    fish["label"] = fish.concentration_pct.map(lambda v: str(int(v)))
    public_cfg = read_public_config(cfg["public_config"])
    public_cache, cache, supports, nested, full_sets, query_sets = {}, {}, {}, {}, {}, {}
    rows = []
    json_observations = 0
    checksums = 0

    def context(info):
        transform = info["transform_id"]
        if transform in cache:
            return cache[transform]
        directory = root/info["transform_provenance"]
        fit = json.loads((directory/"fit.json").read_text(encoding="utf-8"))
        if fit["test_used_for_fitting_or_selection"]:
            raise ValueError("Test used for representation learning")
        for path, expected in fit["source_sha256"].items():
            if file_hash(path) != expected:
                raise ValueError(f"Changed fitting source: {path}")
        if file_hash(directory/"representation.joblib") != fit["representation_sha256"]:
            raise ValueError("Changed fitted transform")
        model = joblib.load(directory/"representation.joblib")
        if info["task"] == "PUBLIC":
            name = info["dataset_id"]
            if name not in public_cache:
                public_cache[name] = load_public(name, public_cfg)
            data = public_cache[name]
            meta, x = data.meta, data.x
            split_path = root/fit["split_path"]
            split = pd.read_csv(split_path, dtype={"label": str})
            validate_public_split(data, split)
            roles = split.split
        else:
            meta, x = fish, raw
            split_path = Path(fit["split_path"])
            split = pd.read_csv(split_path)
            validate_split(original, split)
            roles = meta.group_id.map(split.set_index("physical_sample_id").split)
        if file_hash(split_path) != fit["split_sha256"]:
            raise ValueError("Changed fitted partition")
        by_id = pd.Index(meta.sample_id)
        fit_indices = by_id.get_indexer(fit["train_fit_sample_ids"])
        if (fit_indices < 0).any() or len(set(fit_indices)) != len(fit_indices):
            raise ValueError("Invalid/duplicate fitting identities")
        if not roles.iloc[fit_indices].eq("train").all():
            raise ValueError("A non-Train sample fitted the representation")
        if digest(sorted(meta.loc[fit_indices, "sample_id"])) != fit["train_fit_ids_sha256"]:
            raise ValueError("Train identity hash mismatch")
        val_fit = by_id.get_indexer(fit["validation_selection_sample_ids"])
        if (val_fit < 0).any() or (len(val_fit) and not roles.iloc[val_fit].eq("val").all()):
            raise ValueError("Dimension selection used non-Val IDs")
        if info["task"] != "PUBLIC":
            if not meta.loc[fit_indices, "day"].eq(info["source_day"]).all():
                raise ValueError("Cross-day representation was fitted on target observations")
            expected_count = 70 if info["task"] == "UNSEEN" else 45 if info["task"] == "T3" else 90
            if len(fit_indices) != expected_count:
                raise ValueError("Wrong supervised label budget")
            if info["task"] == "UNSEEN":
                if set(meta.loc[fit_indices, "label"]) & {"15", "30"}:
                    raise ValueError("Held-out labels fitted LDA")
                if set(model.classes_) & {15, 30} or model.model.n_components != 6:
                    raise ValueError("Unseen transform reused nine-class LDA")
            snv_train = Representation("snv").transform(x[fit_indices])
            np.testing.assert_allclose(model.model.xbar_, snv_train.mean(0), atol=1e-10)
        elif isinstance(model.reducer_, PCA):
            np.testing.assert_allclose(model.reducer_.mean_, model.preprocess(x[fit_indices]).mean(0), atol=1e-10)
        z = model.transform(x)
        cache[transform] = (meta, roles, z, fit, by_id)
        return cache[transform]

    for relative in manifest["packages"]:
        package = root/relative
        info = json.loads((package/"metadata.json").read_text(encoding="utf-8"))
        for filename, h in info["file_sha256"].items():
            if file_hash(package/filename) != h:
                raise ValueError(f"Package checksum mismatch: {relative}/{filename}")
            checksums += 1
        meta, roles, z, fit, by_id = context(info)
        mapping = pd.read_csv(package/"private_id_mapping.csv", dtype={"label": str})
        if mapping.id.duplicated().any():
            raise ValueError("Repeated exported anonymous ID")
        mapping = mapping.set_index("id")
        exported_groups, exported_ids = {}, {}
        for part in ["train", "val", "test"]:
            data = json.loads((package/f"{part}.json").read_text(encoding="utf-8"))
            expected_count = info["reference_count" if part == "train" else f"{part}_count"]
            if len(data) != expected_count or len({r["id"] for r in data}) != len(data):
                raise ValueError("Wrong count or duplicate query/reference ID")
            ids = [r["id"] for r in data]
            selected = mapping.loc[ids]
            indices = by_id.get_indexer(selected.sample_id)
            if (indices < 0).any() or not roles.iloc[indices].eq(part).all():
                raise ValueError("Exported record belongs to another partition")
            if not np.isfinite(np.array([r["features"] for r in data], dtype=float)).all():
                raise ValueError("Nonfinite JSON feature")
            np.testing.assert_allclose(np.array([r["features"] for r in data]), z[indices],
                atol=5.1*10**(-info["feature_decimals"]-1), rtol=0)
            if selected.label.tolist() != meta.loc[indices, "label"].tolist():
                raise ValueError("Mapping changed truth labels")
            expected_anon = [_anonymous(info["dataset_id"], info["repeat"], sid) for sid in selected.sample_id]
            if ids != expected_anon:
                raise ValueError("Anonymous identity changed")
            if info["task"] != "PUBLIC":
                if [r["day"] for r in data] != meta.loc[indices, "day"].tolist():
                    raise ValueError("Incorrect day metadata")
            exported_groups[part] = set(selected.group_id)
            exported_ids[part] = set(selected.sample_id)
            if part == "train":
                if [r["label"] for r in data] != selected.label.tolist():
                    raise ValueError("Reference label mismatch")
                classes = info["seen_labels"] if info["task"] == "UNSEEN" else info["candidate_labels"]
                counts = Counter(r["label"] for r in data)
                if info.get("reference_mode") == "full_reference":
                    eligible_train = roles.eq("train")
                    if info["task"] != "PUBLIC":
                        eligible_train &= meta.day.eq(info["source_day"])
                    if info["task"] == "UNSEEN":
                        eligible_train &= ~meta.label.isin(info["heldout_labels"])
                    if exported_ids["train"] != set(meta.loc[eligible_train, "sample_id"]):
                        raise ValueError("Full-reference condition omitted or added Train observations")
                    if info.get("k") is not None or info.get("experiment_role") != "main":
                        raise ValueError("Full-reference condition must be a main experiment without K")
                elif info["task"] != "T3":
                    if counts != {c: info["k"] for c in classes}:
                        raise ValueError("Wrong per-class K")
                    if any(g.group_id.nunique() != info["k"] for _, g in selected.groupby("label")):
                        raise ValueError("K references include duplicate sample groups")
                else:
                    k, control = info["target_k"], info["control"]
                    expected = 5 if control == "source_only" else k if control == "target_only" else 5+k
                    if counts != {c: expected for c in classes}:
                        raise ValueError("Wrong T3 reference budget")
                    source_ids = set(meta.loc[indices].loc[lambda f: f.day.eq(info["source_day"]), "group_id"])
                    target_ids = set(meta.loc[indices].loc[lambda f: f.day.eq(info["target_day"]), "group_id"])
                    if source_ids & target_ids:
                        raise ValueError("T3 source/target IDs overlap")
            else:
                if any("label" in r or set(r)-{"id", "features", "day"} for r in data):
                    raise ValueError("Query contains label or unsupported metadata")
                answers = json.loads((package/f"{part}_labels.json").read_text(encoding="utf-8"))
                if [r["id"] for r in answers] != ids or [r["label"] for r in answers] != selected.label.tolist():
                    raise ValueError("Ground truth order or label mismatch")
                eligible = roles.eq(part)
                if info["task"] != "PUBLIC":
                    eligible &= meta.day.eq(info["target_day"])
                if info["task"] == "UNSEEN" and part == "val":
                    eligible &= ~meta.label.isin(info["heldout_labels"])
                if exported_ids[part] != set(meta.loc[eligible, "sample_id"]):
                    raise ValueError("Query partition was sampled, chunked or truncated")
            json_observations += len(data)
        if any(exported_groups[a] & exported_groups[b] for a, b in [("train", "val"), ("train", "test"), ("val", "test")]):
            raise ValueError("Grouped partition leakage")
        if info["task"] == "UNSEEN":
            if set(mapping.loc[mapping.split.isin(["train", "val"]), "label"]) & {"15", "30"}:
                raise ValueError("Held-out labels shown in reference or validation")
            if info["val_count"] != 14 or info["test_count"] != 36:
                raise ValueError("Unseen evaluation must keep seen Val14 and full Test36")
        prompt = (package/"prompt.txt").read_text(encoding="utf-8")
        if info.get("llm_execution_mode") == "context_only_no_tools":
            if info.get("tool_use_allowed") is not False or CONTEXT_ONLY_POLICY not in prompt:
                raise ValueError("Context-only/tool prohibition contract missing")
        if info["experiment_id"] not in prompt or "predictions.json" not in prompt or "评分阶段" not in prompt:
            raise ValueError("Prompt contract missing")
        if len(info["feature_order"]) != z.shape[1] or info["test_subset_or_chunk"]:
            raise ValueError("Feature dimension mismatch or query chunking")
        supports[relative] = exported_ids["train"]
        if info.get("equivalent_to") and supports.get(info["equivalent_to"]) != supports[relative]:
            raise ValueError("Incorrect T3 anchor equivalence")
        if info["task"] != "T3":
            key = (info["batch"], info["task"], info["dataset_id"], info["repeat"], info.get("source_day"), info.get("target_day"))
            if info.get("reference_mode") == "full_reference":
                full_sets[key] = exported_ids["train"]
            else:
                nested.setdefault(key, {})[info["k"]] = exported_ids["train"]
            signature = (info["transform_id"], tuple(info["file_sha256"][f"{part}{suffix}.json"]
                         for part in ["val", "test"] for suffix in ["", "_labels"]))
            if key in query_sets and signature != query_sets[key]:
                raise ValueError("Reference conditions must share identical transform and Val/Test bytes/order")
            query_sets[key] = signature
        rows.append(dict(package=relative, experiment_id=info["experiment_id"], task=info["task"],
                         status="passed", reference=info["reference_count"], val=info["val_count"], test=info["test_count"]))
    for key, entries in nested.items():
        ks = sorted(entries)
        if any(not entries[a] <= entries[b] for a, b in zip(ks, ks[1:])):
            raise ValueError("K support sets are not nested")
        if cfg.get("include_full_reference", False):
            if key not in full_sets or any(not support <= full_sets[key] for support in entries.values()):
                raise ValueError("A supplementary K condition lacks its complete main reference pool")
    if len({r["experiment_id"] for r in rows}) != len(rows):
        raise ValueError("Repeated experiment identity")
    report = dict(status="passed", packages=len(rows), batch_counts=dict(Counter(Path(r["package"]).parts[0] for r in rows)),
        task_counts=dict(Counter(r["task"] for r in rows)), transforms_verified=len(cache),
        full_reference_packages=len(full_sets), full_reference_coverage_verified=True,
        identical_full_and_K_query_bytes_and_coordinates_verified=True,
        json_observations_verified=json_observations, package_checksums_verified=checksums,
        source_hashes_unchanged=True, train_only_transform_fit=True,
        unseen_quarantine_verified=True, common_cross_day_coordinates_verified=True,
        group_disjointness_verified=True, distinct_group_K_and_nested_support_verified=True,
        complete_unsplit_val_test_verified=True, test_used_for_fit_or_selection=False,
        api_requests=0, actual_llm_predictions=0, package_checks=rows)
    if write_report:
        write_json(root/"validation_report.json", report)
        checksum(root)
    return report
