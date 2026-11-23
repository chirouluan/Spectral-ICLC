"""Deterministic verification of frozen browser predictions; no LLM or API."""
from collections import Counter
from pathlib import Path
import json

import numpy as np

from .io import file_hash


def _partition(predictions, answers, candidates):
    if not isinstance(predictions, list):
        raise ValueError("Prediction partitions must be arrays")
    truth = {row["id"]: row["label"] for row in answers}
    if len(truth) != len(answers) or not set(truth.values()) <= set(candidates):
        raise ValueError("Duplicate or illegal ground truth")
    issues, grouped = [], {}
    for row in predictions:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            issues.append(dict(type="malformed_prediction_record", record=row))
            continue
        grouped.setdefault(row["id"], []).append(row)
    for key in sorted(set(grouped)-set(truth)):
        issues.append(dict(type="extra_id", id=key))
    resolved = {}
    invalid = "__invalid__"
    for key in truth:
        rows = grouped.get(key, [])
        if len(rows) != 1:
            issues.append(dict(type="missing_id" if not rows else "duplicate_id", id=key))
            resolved[key] = invalid
        else:
            label = str(rows[0].get("label"))
            if label not in candidates:
                issues.append(dict(type="invalid_label", id=key, label=rows[0].get("label")))
                label = invalid
            resolved[key] = label
    correct = sum(resolved[key] == label for key, label in truth.items())
    supported = [label for label in candidates if label in truth.values()]
    per_class = []
    confusion = {label: {pred: 0 for pred in [*candidates, invalid]} for label in candidates}
    for key, label in truth.items():
        confusion[label][resolved[key]] += 1
    predicted = Counter(resolved.values())
    actual = Counter(truth.values())
    for label in candidates:
        tp = confusion[label][label]
        precision = tp/predicted[label] if predicted[label] else 0.
        recall = tp/actual[label] if actual[label] else 0.
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.
        per_class.append(dict(label=label, support=actual[label], precision=precision, recall=recall, f1=f1))
    result = dict(correct=correct, total=len(truth), accuracy_pct=round(100*correct/len(truth), 2),
        macro_f1=float(np.mean([r["f1"] for r in per_class if r["label"] in supported])),
        balanced_accuracy=float(np.mean([r["recall"] for r in per_class if r["label"] in supported])),
        macro_class_scope="ground-truth-present classes", per_class=per_class,
        confusion=confusion, invalid_predictions=sum(v == invalid for v in resolved.values()))
    return result, issues, resolved


def score_frozen_predictions(package, predictions):
    package = Path(package)
    info = json.loads((package/"metadata.json").read_text(encoding="utf-8"))
    if isinstance(predictions, (str, Path)):
        predictions = json.loads(Path(predictions).read_text(encoding="utf-8-sig"))
    if not isinstance(predictions, dict) or predictions.get("experiment_id") != info["experiment_id"]:
        raise ValueError("experiment_id missing or does not match this package")
    result = dict(experiment_id=info["experiment_id"], issues=[])
    for part in ["val", "test"]:
        for filename in [f"{part}.json", f"{part}_labels.json"]:
            if file_hash(package/filename) != info["file_sha256"][filename]:
                raise ValueError(f"Package truth/query checksum mismatch: {filename}")
        answers = json.loads((package/f"{part}_labels.json").read_text(encoding="utf-8"))
        queries = json.loads((package/f"{part}.json").read_text(encoding="utf-8"))
        if len(answers) != info[f"{part}_count"] or [r["id"] for r in answers] != [r["id"] for r in queries]:
            raise ValueError("Ground truth IDs/count do not match expected queries")
        metrics, issues, resolved = _partition(predictions.get(part, []), answers, info["candidate_labels"])
        result[part] = metrics
        result["issues"].extend(dict(partition=part, **issue) for issue in issues)
        if part == "test" and info["task"] == "UNSEEN":
            truth = {r["id"]: r["label"] for r in answers}
            def score(labels):
                ids = [key for key, label in truth.items() if label in labels]
                n = sum(resolved[key] == truth[key] for key in ids)
                return dict(correct=n, total=len(ids), accuracy_pct=round(100*n/len(ids), 2)), n/len(ids)
            seen, sa = score(info["seen_labels"])
            unseen, ua = score(info["heldout_labels"])
            result["test"].update(seen=seen, unseen=unseen, seen_accuracy_pct=seen["accuracy_pct"],
                unseen_accuracy_pct=unseen["accuracy_pct"], h_mean_pct=round(200*sa*ua/(sa+ua), 2) if sa+ua else 0.,
                heldout_per_class={label: score([label])[0] for label in info["heldout_labels"]})
    return result
