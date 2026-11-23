"""Fit classical comparators to exactly the references in each web package.

Frozen exported features are never refitted; Test is only used after prediction.
Local numeric computation is separate from the LLM's context-only restriction.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import platform
import time
import uuid
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits
import yaml

from .io import digest, file_hash, write_json
from .models import make_model, PrototypeClassifier
from .lumir import NeuralClassifier
from .public_validation import checksum
from .web_scoring import _partition


class ConcentrationPrototype:
    """Numeric concentration interpolation with string labels at the IO boundary."""
    def __init__(self, candidates):
        self.candidates = sorted(float(v) for v in candidates)

    def fit(self, x, y):
        self.model_ = PrototypeClassifier(self.candidates, extrapolate=False)
        self.model_.fit(x, np.asarray(y, dtype=float))
        return self

    def predict(self, x):
        return np.asarray([format(float(v), "g") for v in self.model_.predict(x)])


def baseline_model(name, count, candidates, seed, settings, checkpoint=None):
    if name in ("cnn1d", "transformer"):
        return NeuralClassifier(name, seed, settings, checkpoint)
    if name == "spectral_centroid_interpolation":
        if settings != dict(extrapolate=False, distance="euclidean_in_exported_space"):
            raise ValueError("Prototype policy must use non-extrapolating Euclidean interpolation")
        return ConcentrationPrototype(candidates)
    estimator = make_model(name, count, candidates, seed)
    if name == "svm_rbf":
        if settings["scale_on_reference_only"] is not True:
            raise ValueError("SVM scaler must fit references only")
        estimator.set_params(svc__C=settings["C"], svc__gamma=settings["gamma"])
    elif name == "random_forest":
        if settings["random_state"] != "package_support_seed":
            raise ValueError("Forest seed must follow package support seed")
        estimator.set_params(n_estimators=settings["n_estimators"], n_jobs=settings["n_jobs"])
    return estimator


def package_data(package):
    package = Path(package)
    info = json.loads((package/"metadata.json").read_text(encoding="utf-8"))
    for filename, expected in info["file_sha256"].items():
        if file_hash(package/filename) != expected:
            raise ValueError(f"Changed web input: {package}/{filename}")
    mapping = pd.read_csv(package/"private_id_mapping.csv", dtype={"label": str}).set_index("id")
    data = {}
    for part in ["train", "val", "test"]:
        rows = json.loads((package/f"{part}.json").read_text(encoding="utf-8"))
        ids = [r["id"] for r in rows]
        frame = mapping.loc[ids].reset_index()
        if frame.id.duplicated().any() or not frame.split.eq(part).all():
            raise ValueError("Invalid identity or partition alignment")
        if part == "train":
            labels = [r["label"] for r in rows]
        else:
            if any("label" in row for row in rows):
                raise ValueError("Query JSON contains labels")
            answers = json.loads((package/f"{part}_labels.json").read_text(encoding="utf-8"))
            if ids != [r["id"] for r in answers]:
                raise ValueError("Answer/query ID order differs")
            labels = [r["label"] for r in answers]
        if labels != frame.label.tolist():
            raise ValueError("Truth/mapping labels differ")
        x = np.asarray([r["features"] for r in rows], dtype=float)
        if not np.isfinite(x).all() or x.shape != (len(rows), len(info["feature_order"])):
            raise ValueError("Invalid exported features")
        data[part] = dict(x=x, y=np.asarray(labels, dtype=str), frame=frame)
    groups = {p:set(v["frame"].group_id) for p, v in data.items()}
    if groups["train"] & (groups["val"] | groups["test"]) or groups["val"] & groups["test"]:
        raise ValueError("Train/Val/Test group overlap")
    if info["task"] == "UNSEEN" and set(data["train"]["y"]) & set(info["heldout_labels"]):
        raise ValueError("Held-out labels in training references")
    return info, data


def metrics_for_frame(frame, info, part, bootstrap_draws, seed):
    answers = frame[["id", "y_true"]].rename(columns={"y_true":"label"}).to_dict("records")
    predictions = frame[["id", "y_pred"]].rename(columns={"y_pred":"label"}).to_dict("records")
    result, issues, _ = _partition(predictions, answers, info["candidate_labels"])
    if issues:
        raise ValueError(f"Classical classifier returned invalid predictions: {issues[:2]}")
    result["accuracy"] = result["correct"]/result["total"]
    result["eval_mode"] = "in_sample_on_reference_records" if part == "train" else "holdout"
    result["evaluation_unit"] = "exported spectrum record"
    correct = frame.assign(correct=frame.y_true.eq(frame.y_pred))
    groups = correct.groupby("group_id").correct.mean()
    result.update(n_reference_or_query_groups=len(groups), mean_group_accuracy=float(groups.mean()),
        cluster_scope="available package group IDs; missing preparation identities remain unresolved")
    if bootstrap_draws:
        rng = np.random.default_rng(seed)
        # CI is for group-equal mean accuracy conditional on this partition.
        values = groups.to_numpy()
        indices = rng.integers(0, len(values), size=(bootstrap_draws, len(values)))
        result["group_mean_accuracy_ci95"] = np.quantile(values[indices].mean(1), [.025, .975]).tolist()
    if info["task"] == "UNSEEN" and part == "test":
        seen = frame.y_true.isin(info["seen_labels"])
        unseen = frame.y_true.isin(info["heldout_labels"])
        sa, ua = float(correct.loc[seen, "correct"].mean()), float(correct.loc[unseen, "correct"].mean())
        result.update(seen_accuracy=sa, unseen_accuracy=ua,
            seen_correct=int(correct.loc[seen, "correct"].sum()), seen_total=int(seen.sum()),
            unseen_correct=int(correct.loc[unseen, "correct"].sum()), unseen_total=int(unseen.sum()),
            h_mean=2*sa*ua/(sa+ua) if sa+ua else 0.)
    return result


def smoke_conditions(index):
    """Exercise low-shot, adaptation, unseen labels, binary classification and largest input."""
    chosen = []
    predicates = [
        (index.task.eq("T1") & index.reference_mode.eq("context_k") & index.k.eq(1)),
        (index.task.eq("T2") & index.reference_mode.eq("context_k") & index.k.eq(1)),
        (index.task.eq("UNSEEN") & index.reference_mode.eq("context_k") & index.k.eq(1)),
        (index.dataset_id.eq("coffee") & index.reference_mode.eq("context_k") & index.k.eq(1)),
        (index.dataset_id.eq("beef") & index.reference_mode.eq("full_reference")),
        (index.dataset_id.eq("olive_oils") & index.reference_mode.eq("full_reference")),
    ]
    for mask in predicates:
        if mask.any():
            chosen.append(index.loc[mask, "directory"].iloc[0])
    return chosen


def current_package_index(root):
    """Build the active index from package metadata, ignoring stale export indexes."""
    root = Path(root)
    rows = []
    for prompt_path in sorted(root.rglob("prompt.txt")):
        package = prompt_path.parent
        metadata_path = package / "metadata.json"
        info = json.loads(metadata_path.read_text(encoding="utf-8"))
        rows.append({
            "batch": package.relative_to(root).parts[0], "task": info["task"], "dataset_id": info["dataset_id"],
            "reference_mode": info["reference_mode"], "experiment_role": info["experiment_role"],
            "repeat": info["repeat"], "k": info.get("k"), "source_day": info.get("source_day"),
            "target_day": info.get("target_day"), "control": info.get("control"),
            "target_k": info.get("target_k"), "reference_count": info["reference_count"],
            "val_count": info["val_count"], "test_count": info["test_count"],
            "dimensions": len(info["feature_order"]), "experiment_id": info["experiment_id"],
            "equivalent_to": info.get("equivalent_to"),
            "directory": package.relative_to(root).as_posix(), "label_budget": info["label_budget"],
        })
    if not rows:
        raise ValueError(f"No prepared experiment packages found under {root}")
    index = pd.DataFrame(rows)
    if index.experiment_id.duplicated().any() or index.directory.duplicated().any():
        raise ValueError("Duplicate experiment IDs or package directories")
    return index


def export_fingerprint(root, index=None):
    """Hash the active packages and their metadata, not stale top-level manifests."""
    root = Path(root)
    index = current_package_index(root) if index is None else index
    entries = []
    for relative in sorted(index.directory.tolist()):
        package = root / relative
        info = json.loads((package / "metadata.json").read_text(encoding="utf-8"))
        files = {name: file_hash(package / name) for name in sorted(info["file_sha256"])}
        entries.append({"directory": relative, "metadata_sha256": file_hash(package / "metadata.json"),
                        "files": files})
    return digest(entries)


def validate_current_export(root, index=None):
    """Validate every active package from its own hashes, IDs, partitions and labels."""
    root = Path(root)
    index = current_package_index(root) if index is None else index
    checked = 0
    for row in index.to_dict("records"):
        info, _ = package_data(root / row["directory"])
        if info["experiment_id"] != row["experiment_id"] or info["task"] != row["task"]:
            raise ValueError(f"Index/metadata mismatch: {row['directory']}")
        checked += len(info["file_sha256"])
    return {"status": "passed", "packages": len(index), "package_files_checked": checked,
            "validation_scope": "metadata-declared hashes, truth/mapping alignment, feature shapes, and group separation"}


def plan_matched(config_path="configs/web_matched_baselines.yaml", packages=None, smoke=False):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    root = Path(cfg["web_export"])
    index = current_package_index(root)
    index = index[~index.dataset_id.isin(cfg.get("excluded_datasets", []))]
    if smoke:
        valid = set(index.directory)
        packages = [p for p in smoke_conditions(index) if p in valid]
    if packages is not None:
        missing = set(packages)-set(index.directory)
        if missing:
            raise ValueError(f"Unknown web packages: {missing}")
        index = index[index.directory.isin(packages)]
    jobs = []
    for row in index.to_dict("records"):
        for model in cfg["models"]+(cfg["unseen_models"] if row["task"] == "UNSEEN" else []):
            jobs.append({**row, "model_id":model})
    return cfg, pd.DataFrame(jobs)


def _save_metrics(path, part, metrics):
    per_class = metrics.pop("per_class")
    confusion = metrics.pop("confusion")
    write_json(path/f"metrics_{part}.json", metrics)
    pd.DataFrame(per_class).to_csv(path/f"per_class_metrics_{part}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame.from_dict(confusion, orient="index").rename_axis("y_true").to_csv(path/f"confusion_{part}.csv", encoding="utf-8-sig")


def verify_tree_checksums(root):
    root = Path(root)
    lines = (root/"checksum.sha256").read_text(encoding="utf-8").splitlines()
    for line in lines:
        expected, relative = line.split("  ", 1)
        if file_hash(root/relative) != expected:
            raise ValueError(f"Checksum mismatch: {root/relative}")
    return len(lines)


def verify_current_export(root):
    report = validate_current_export(root)
    return report["package_files_checked"]


def run_matched(config_path="configs/web_matched_baselines.yaml", packages=None, smoke=False, output=None):
    cfg, plan = plan_matched(config_path, packages, smoke)
    if plan.empty:
        raise ValueError("No selected cases")
    web = Path(cfg["web_export"])
    # Validate the actual package directories; the top-level index/manifest can
    # be historical after packages are pruned or renamed.
    web_validation = validate_current_export(web, current_package_index(web))
    web_manifest_hash = export_fingerprint(web)
    input_checksum_hash = web_manifest_hash
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid.uuid4().hex[:8]
    root = Path(output) if output else Path(cfg["output"])/run_id
    root.mkdir(parents=True, exist_ok=False)
    model_cache = root/"_models"
    model_cache.mkdir()
    source_files = [Path(__file__), Path("src/spectral_icl/models.py"), Path("src/spectral_icl/lumir.py"),
                    Path("src/spectral_icl/architectures.py"), Path("src/spectral_icl/public_validation.py"),
                    Path("src/spectral_icl/web_scoring.py")]
    code_hashes = {str(p):file_hash(p) for p in source_files}
    plan.to_csv(root/"plan.csv", index=False, encoding="utf-8-sig")
    write_json(root/"plan.json", dict(run_id=run_id, config=cfg, config_sha256=file_hash(config_path),
        web_export=str(web.resolve()), web_manifest_sha256=web_manifest_hash,
        web_checksum_sha256=input_checksum_hash, web_validation=web_validation["status"],
        packages=plan.directory.nunique(), model_runs=len(plan), smoke=smoke,
        code_sha256=code_hashes, sklearn=sklearn.__version__, numpy=np.__version__,
        python=platform.python_version(), api_requests=0,
        selection_policy="fixed hyperparameters; neural checkpoints selected on validation; Test never used for selection",
        train_eval_scope="the exact train.json references, in-sample; not LLM leave-sample-out diagnostics"))
    summaries, registry = [], []
    with threadpool_limits(limits=cfg["threads"]):
        for package_no, (relative, jobs) in enumerate(plan.groupby("directory", sort=False), 1):
            package = web/relative
            info, data = package_data(package)
            mapping = data["train"]["frame"]
            for model_name in jobs.model_id:
                path = root/relative/model_name
                path.mkdir(parents=True, exist_ok=False)
                for directory in ["logs", "prompts", "responses"]:
                    (path/directory).mkdir()
                start = time.monotonic()
                fit_warnings, failure, status = [], None, "complete"
                uses_validation = model_name in ("cnn1d", "transformer")
                signature = digest([info["file_sha256"]["train.json"],
                    info["file_sha256"]["val.json"] if uses_validation else None, model_name,
                    info["file_sha256"]["val_labels.json"] if uses_validation else None,
                    info["candidate_labels"], info["support_seed"], cfg["hyperparameters"], code_hashes])
                cached = model_cache/f"{signature}.joblib"
                cache_metadata = model_cache/f"{signature}.json"
                cache_hit = cached.exists()
                estimator = None
                pending_rows = []
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        if cache_hit:
                            estimator = joblib.load(cached)
                            fit_warnings = json.loads(cache_metadata.read_text(encoding="utf-8"))["fit_warnings"]
                        else:
                            estimator = baseline_model(model_name, len(mapping), info["candidate_labels"], info["support_seed"], cfg["hyperparameters"][model_name], cached.with_suffix(".pt"))
                            # Only these exact references are passed to fit.
                            if uses_validation:
                                estimator.fit(data["train"]["x"], data["train"]["y"],
                                    validation=(data["val"]["x"], data["val"]["y"]))
                            else:
                                estimator.fit(data["train"]["x"], data["train"]["y"])
                            fit_warnings = [dict(category=type(w.message).__name__, message=str(w.message)) for w in caught]
                            joblib.dump(estimator, cached)
                            write_json(cache_metadata, dict(signature=signature, model_id=model_name,
                                train_json_sha256=info["file_sha256"]["train.json"],
                                validation_json_sha256=info["file_sha256"]["val.json"] if uses_validation else None,
                                validation_labels_sha256=info["file_sha256"]["val_labels.json"] if uses_validation else None,
                                ordered_fit_ids=mapping.id.tolist(), fit_warnings=fit_warnings))
                            checkpoint_path = cached.with_suffix(".pt")
                            if checkpoint_path.exists():
                                checkpoint_path.unlink()
                    prediction_warnings = []
                    for part in ["train", "val", "test"]:
                        sample = data[part]
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            prediction = np.asarray(estimator.predict(sample["x"]), dtype=str)
                        prediction_warnings.extend(dict(partition=part, category=type(w.message).__name__, message=str(w.message)) for w in caught)
                        frame = sample["frame"][["id", "sample_id", "group_id"]].copy()
                        if prediction.shape != (len(frame),):
                            raise ValueError("Prediction length differs from complete web partition")
                        frame["y_true"], frame["y_pred"] = sample["y"], prediction
                        frame["prediction_valid"] = np.isin(prediction, info["candidate_labels"])
                        frame["eval_mode"] = "in_sample_on_reference_records" if part == "train" else "holdout"
                        frame.to_csv(path/f"predictions_{part}.csv", index=False, encoding="utf-8-sig")
                        metrics = metrics_for_frame(frame, info, part, cfg["bootstrap_draws"], cfg["bootstrap_seed"])
                        _save_metrics(path, part, metrics)
                        pending_rows.append(dict(run_id=f"{root.name}_{info['experiment_id']}_{model_name}",
                            batch=relative.split("/")[0], task=info["task"], dataset_id=info["dataset_id"], repeat=info["repeat"],
                            model_id=model_name, reference_mode=info["reference_mode"], experiment_role=info["experiment_role"],
                            k=info["k"], source_day=info.get("source_day"), target_day=info.get("target_day"),
                            control=info.get("control"), target_k=info.get("target_k"),
                            experiment_id=info["experiment_id"], split=part, status="complete",
                            reference_count=info["reference_count"], dimensions=len(info["feature_order"]),
                            web_directory=relative, result_directory=str(path),
                            equivalent_to=info.get("equivalent_to"), **metrics))
                    fit_warnings.extend(prediction_warnings)
                    summaries.extend(pending_rows)
                except Exception as error:
                    failure, status = f"{type(error).__name__}: {error}", "failed"
                    for part in ["train", "val", "test"]:
                        write_json(path/f"metrics_{part}.json", dict(status="failed", reason=failure, accuracy=None))
                mapping.assign(context_scope="fixed_for_all_queries").to_csv(path/"support_context.csv", index=False, encoding="utf-8-sig")
                comparison = "unseen_candidate_classifier" if model_name in cfg["unseen_models"] else "closed_set_control_only" if info["task"] == "UNSEEN" else "closed_set_classifier"
                details = dict(run_id=f"{root.name}_{info['experiment_id']}_{model_name}",
                    experiment_id=info["experiment_id"], web_package=str(package.resolve()), model_id=model_name,
                    status=status, reason=failure, batch=relative.split("/")[0], task=info["task"], dataset_id=info["dataset_id"],
                    repeat=info["repeat"], reference_mode=info["reference_mode"], control=info.get("control"),
                    reference_count=info["reference_count"], train_fit_ids=mapping.id.tolist(),
                    train_fit_ids_sha256=digest(mapping.id.tolist()),
                    train_json_sha256=info["file_sha256"]["train.json"], web_metadata_sha256=file_hash(package/"metadata.json"),
                    label_budget=info["label_budget"], source_train_representation_fit_count=info["source_train_fit_count"],
                    transform_id=info["transform_id"], feature_order=info["feature_order"],
                    transform_refitted=False, estimator_fit_on_reference_only=True,
                    validation_used_for_representation_dim_selection=info.get("validation_used_for_dimension_selection", False),
                    validation_or_test_used_for_estimator_selection=uses_validation,
                    validation_used_for_early_stopping=uses_validation,
                    test_used_for_estimator_selection=False, comparison_scope=comparison,
                    hyperparameters=cfg["hyperparameters"][model_name], seed=info["support_seed"],
                    fitted_estimator=str(cached.relative_to(root)), fit_cache_hit=cache_hit,
                    validation_ids=data["val"]["frame"].id.tolist() if uses_validation else [],
                    validation_ids_sha256=digest(data["val"]["frame"].id.tolist()) if uses_validation else None,
                    early_stopping_best_epoch=getattr(estimator, "best_epoch", None),
                    early_stopping_best_validation_accuracy=getattr(estimator, "best_validation_accuracy", None),
                    early_stopping_best_validation_loss=getattr(estimator, "best_validation_loss", None),
                    epochs_ran=getattr(estimator, "epochs_ran", None),
                    optimizer_steps=getattr(estimator, "optimizer_steps", None),
                    actual_batch_size=getattr(estimator, "actual_batch_size", None),
                    train_diagnostic="in_sample on train.json references only", fit_warnings=fit_warnings,
                    elapsed_seconds=time.monotonic()-start, api_requests=0, actual_llm_predictions=0,
                    code_sha256=code_hashes, sklearn=sklearn.__version__)
                if cached.is_file():
                    details["fitted_estimator_sha256"] = file_hash(cached)
                if uses_validation and status == "complete":
                    write_json(path/"training_history.json", estimator.history)
                write_json(path/"manifest.json", details)
                (path/"logs/events.jsonl").write_text(json.dumps(dict(event="run_completed",status=status,
                    duration_seconds=details["elapsed_seconds"],fit_cache_hit=cache_hit,reason=failure),ensure_ascii=False)+"\n",encoding="utf-8")
                checksum(path)
                registry.append(dict(experiment_id=info["experiment_id"], batch=relative.split("/")[0], task=info["task"],
                    dataset_id=info["dataset_id"], repeat=info["repeat"], model_id=model_name, status=status,
                    reason=failure, comparison_scope=comparison, result_directory=str(path), web_directory=relative))
            for name in cfg["not_run_models"]:
                registry.append(dict(experiment_id=info["experiment_id"], batch=relative.split("/")[0], task=info["task"],
                    dataset_id=info["dataset_id"], repeat=info["repeat"], model_id=name, status="not_run",
                    reason="No API-backed LLM call or LUMIR prompt reproduction was run", web_directory=relative))
            if info["task"] == "UNSEEN":
                registry.append(dict(experiment_id=info["experiment_id"], batch=relative.split("/")[0], task=info["task"],
                    dataset_id=info["dataset_id"], repeat=info["repeat"], model_id="ordinal_prototype_distance", status="not_run_equivalent",
                    reason="15 and 30 are numeric and ordinal midpoints of adjacent seen classes, giving identical prototypes to spectral_centroid_interpolation", web_directory=relative))
            print(f"{package_no}/{plan.directory.nunique()}: {relative}", flush=True)
    full_index = current_package_index(web)
    for row in full_index[full_index.dataset_id.isin(cfg.get("excluded_datasets", []))].to_dict("records"):
        models = cfg["models"]+(cfg["unseen_models"] if row["task"] == "UNSEEN" else [])
        for name in models:
            registry.append(dict(experiment_id=row["experiment_id"], batch=row["batch"], task=row["task"],
                dataset_id=row["dataset_id"], repeat=row["repeat"], model_id=name, status="not_run_user_excluded",
                reason="Dataset excluded at user direction", web_directory=row["directory"]))
    pd.DataFrame(summaries).to_csv(root/"all_runs.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(registry).to_csv(root/"model_registry.csv", index=False, encoding="utf-8-sig")
    if export_fingerprint(web) != web_manifest_hash or input_checksum_hash != web_manifest_hash:
        raise ValueError("Web export changed during execution")
    write_json(root/"manifest.json", dict(run_id=run_id, web_export=str(web.resolve()),
        web_manifest_sha256=web_manifest_hash, web_checksum_sha256=input_checksum_hash,
        config=cfg, code_sha256=code_hashes, packages=plan.directory.nunique(), model_runs_planned=len(plan),
        excluded_datasets=cfg.get("excluded_datasets", []),
        status_counts=pd.Series([r["status"] for r in registry]).value_counts().to_dict(),
        api_requests=0, actual_llm_predictions=0, input_export_unchanged=True))
    checksum(root)
    return root
