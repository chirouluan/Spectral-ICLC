"""Independent readback of all matched predictions, model fit IDs and metrics."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from .io import digest, file_hash, write_json
from .public_validation import checksum
from .web_baselines import (current_package_index, export_fingerprint, package_data,
                            validate_current_export, verify_tree_checksums)
from .web_scoring import _partition


def audit_matched_run(root):
    root = Path(root)
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    web = Path(manifest["web_export"])
    checked_files = verify_tree_checksums(root)
    checked_files += validate_current_export(web)["package_files_checked"]
    if export_fingerprint(web) != manifest["web_manifest_sha256"]:
        raise ValueError("Input package set changed")
    plan = pd.read_csv(root/"plan.csv")
    registry = pd.read_csv(root/"model_registry.csv")
    summary = pd.read_csv(root/"all_runs.csv")
    expected = set(zip(plan.directory, plan.model_id))
    executed = registry[registry.status.isin(["complete", "failed"])]
    if len(executed) != len(expected) or set(zip(executed.web_directory, executed.model_id)) != expected:
        raise ValueError("Planned classifier omitted or duplicated")
    complete = executed[executed.status.eq("complete")]
    if len(summary) != len(complete)*3:
        raise ValueError("Summary omits successful model partitions")
    cache_checked, source_files_checked = set(), set()
    prediction_rows, cache_hits = 0, 0
    warnings = []
    complete_keys = set(zip(complete.web_directory, complete.model_id))
    for case_no, (relative, runs) in enumerate(complete.groupby("web_directory", sort=False), 1):
        info, data = package_data(web/relative)
        for _, record in runs.iterrows():
            path = Path(record.result_directory)
            details = json.loads((path/"manifest.json").read_text(encoding="utf-8"))
            if details["status"] != "complete" or details["experiment_id"] != info["experiment_id"]:
                raise ValueError("Run identity/status mismatch")
            if details["train_json_sha256"] != info["file_sha256"]["train.json"]:
                raise ValueError("Estimator fit used different references")
            if details["train_fit_ids"] != data["train"]["frame"].id.tolist():
                raise ValueError("Estimator fit reference identity/order mismatch")
            if details["train_fit_ids_sha256"] != digest(data["train"]["frame"].id.tolist()):
                raise ValueError("Reference fit identity hash mismatch")
            uses_validation = record.model_id in ("cnn1d", "transformer")
            if details["transform_refitted"] or details["test_used_for_estimator_selection"]:
                raise ValueError("Unexpected representation refit or Test-driven selection")
            if bool(details["validation_used_for_early_stopping"]) != uses_validation:
                raise ValueError("Validation selection policy differs from registered model policy")
            if bool(details["validation_or_test_used_for_estimator_selection"]) != uses_validation:
                raise ValueError("Validation selection flag is inconsistent")
            if uses_validation:
                if details["validation_ids"] != data["val"]["frame"].id.tolist():
                    raise ValueError("Early stopping used different validation IDs")
                if details["validation_ids_sha256"] != digest(details["validation_ids"]):
                    raise ValueError("Validation identity hash mismatch")
            if details["api_requests"] or details["actual_llm_predictions"]:
                raise ValueError("Unexpected LLM/API activity")
            for filename, expected_hash in details["code_sha256"].items():
                if filename not in source_files_checked:
                    if file_hash(filename) != expected_hash:
                        raise ValueError("Fitted code changed during the run/audit")
                    source_files_checked.add(filename)
            model_path = root/details["fitted_estimator"]
            if model_path not in cache_checked:
                if file_hash(model_path) != details["fitted_estimator_sha256"]:
                    raise ValueError("Fitted estimator hash mismatch")
                fit = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
                if fit["ordered_fit_ids"] != details["train_fit_ids"]:
                    raise ValueError("Fit cache reference IDs differ")
                if uses_validation and fit["validation_json_sha256"] != info["file_sha256"]["val.json"]:
                    raise ValueError("Neural checkpoint selection used different validation data")
                model = joblib.load(model_path)
                if hasattr(model, "named_steps") and "standardscaler" in model.named_steps:
                    scaler = model.named_steps["standardscaler"]
                    np.testing.assert_allclose(scaler.mean_, data["train"]["x"].mean(0), atol=1e-10)
                    np.testing.assert_allclose(scaler.var_, data["train"]["x"].var(0), atol=1e-10)
                elif hasattr(model, "model_") and hasattr(model.model_, "_x_mean"):
                    np.testing.assert_allclose(model.model_._x_mean, data["train"]["x"].mean(0), atol=1e-10)
                elif uses_validation:
                    np.testing.assert_allclose(model.scaler.mean_, data["train"]["x"].mean(0), atol=1e-10)
                    np.testing.assert_allclose(model.scaler.var_, data["train"]["x"].var(0), atol=1e-10)
                    if model.best_epoch is None or not 0 <= model.best_epoch < model.config["epochs"]:
                        raise ValueError("Neural early-stopping checkpoint metadata is invalid")
                cache_checked.add(model_path)
            cache_hits += bool(details["fit_cache_hit"])
            if details["fit_warnings"]:
                warnings.append(dict(web_directory=relative, model_id=record.model_id, warnings=details["fit_warnings"]))
            support = pd.read_csv(path/"support_context.csv")
            if support.id.tolist() != details["train_fit_ids"]:
                raise ValueError("Recorded support context mismatch")
            for part in ["train", "val", "test"]:
                frame = pd.read_csv(path/f"predictions_{part}.csv", dtype={"y_true":str,"y_pred":str,"id":str})
                if frame.id.tolist() != data[part]["frame"].id.tolist() or frame.y_true.tolist() != data[part]["y"].tolist():
                    raise ValueError("Predictions do not cover exact web IDs/truth/order")
                if frame.group_id.tolist() != data[part]["frame"].group_id.tolist():
                    raise ValueError("Physical/available group identity changed")
                metrics = json.loads((path/f"metrics_{part}.json").read_text(encoding="utf-8"))
                answers = frame[["id","y_true"]].rename(columns={"y_true":"label"}).to_dict("records")
                predictions = frame[["id","y_pred"]].rename(columns={"y_pred":"label"}).to_dict("records")
                recalculated, issues, _ = _partition(predictions, answers, info["candidate_labels"])
                if issues or recalculated["correct"] != metrics["correct"] or recalculated["total"] != metrics["total"]:
                    raise ValueError("Invalid outputs or incorrect accuracy counts")
                for metric in ["macro_f1", "balanced_accuracy", "accuracy_pct"]:
                    if not np.isclose(recalculated[metric], metrics[metric], atol=1e-12):
                        raise ValueError("Incorrect saved classification metrics")
                for filename in [f"per_class_metrics_{part}.csv", f"confusion_{part}.csv"]:
                    if not (path/filename).is_file():
                        raise ValueError("Required per-class/confusion artifact missing")
                correct = frame.y_true.eq(frame.y_pred)
                grouped = frame.assign(correct=correct).groupby("group_id").correct.mean()
                if not np.isclose(grouped.mean(), metrics["mean_group_accuracy"]):
                    raise ValueError("Group-equal accuracy mismatch")
                if info["task"] == "UNSEEN" and part == "test":
                    unseen = frame.y_true.isin(info["heldout_labels"])
                    sa, ua = correct[~unseen].mean(), correct[unseen].mean()
                    h = 2*sa*ua/(sa+ua) if sa+ua else 0.
                    if not np.isclose(metrics["seen_accuracy"], sa) or not np.isclose(metrics["unseen_accuracy"], ua) or not np.isclose(metrics["h_mean"], h):
                        raise ValueError("Seen/Unseen/H-mean mismatch")
                    if record.model_id != "spectral_centroid_interpolation" and (set(frame.y_pred) & set(info["heldout_labels"]) or ua != 0):
                        raise ValueError("Closed-set classifier unexpectedly generated an unseen class")
                row = summary[summary.run_id.eq(details["run_id"]) & summary.split.eq(part)]
                if len(row) != 1 or not np.isclose(row.accuracy.iloc[0], correct.mean()):
                    raise ValueError("Summary accuracy mismatch")
                prediction_rows += len(frame)
        if case_no % 25 == 0:
            print(f"Audited {case_no}/{complete.web_directory.nunique()} packages", flush=True)
    # Equal-input conditions must produce equal predictions under each fixed model.
    index = current_package_index(web)
    pairs = [(r.directory, r.equivalent_to) for r in index.itertuples() if pd.notna(r.equivalent_to)]
    comparison_path = web/"previous_export_comparison.json"
    if comparison_path.is_file():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        pairs.extend((r["context_k"],r["full_reference"]) for r in comparison.get("reference_data_equivalent_conditions", []))
    equivalent_checks = 0
    for a,b in pairs:
        for model_name in manifest["config"]["models"]:
            if (a,model_name) not in complete_keys or (b,model_name) not in complete_keys:
                continue
            for part in ["train","val","test"]:
                if file_hash(root/a/model_name/f"predictions_{part}.csv") != file_hash(root/b/model_name/f"predictions_{part}.csv"):
                    raise ValueError("Equal input conditions produced inconsistent deterministic predictions")
                equivalent_checks += 1
    report = dict(status="passed", planned_model_runs=len(expected), completed_model_runs=len(complete),
        failed_model_runs=int(executed.status.eq("failed").sum()), packages=plan.directory.nunique(),
        prediction_rows_verified=prediction_rows, checksum_files_verified=checked_files,
        unique_fitted_estimators_verified=len(cache_checked), fit_cache_hits=cache_hits,
        equivalent_prediction_files_verified=equivalent_checks,
        exact_reference_fit_ids_verified=True, scaler_fit_on_references_only_verified=True,
        neural_validation_early_stopping_verified=True, test_never_used_for_model_selection=True,
        complete_train_val_test_ids_and_truth_verified=True, classification_metrics_recomputed=True,
        unseen_closed_set_limit_verified=True, input_export_unchanged=True,
        warning_runs=len(warnings), warnings=warnings, api_requests=0, actual_llm_predictions=0,
        audit_code_sha256=file_hash(__file__))
    write_json(root/"validation_report.json", report)
    checksum(root)
    return report
