"""Bounded experiment matrices, read-only artifact verification and paired drift."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .runner import run
from .data import load_surimi, validate_split
from .contexts import select_support
from .representations import Representation
from .llm import make_prompt
from .io import file_hash, write_json, write_csv, digest


def matrix(config, suite, repeats, support_seeds, allow_api=False, evaluate_test=False):
    if repeats < 1 or repeats > config["repeats"] or support_seeds < 1:
        raise ValueError("Invalid matrix repeats or seeds")
    if allow_api:
        raise ValueError("Paid API matrices are disabled; review and execute bounded single conditions")
    base = config["experiment"]
    conditions = []
    for repeat in range(repeats):
        for support_seed in range(support_seeds):
            if suite == "t1-t2":
                for source, target in ((0, 0), (7, 7), (14, 14), (0, 7), (0, 14), (7, 14)):
                    for k in (1, 2, 3, 5):
                        conditions.append(dict(id="T1" if source == target else "T2", source_day=source,
                            target_day=target, k=k, repeat=repeat, heldout_levels=[]))
            elif suite == "t3":
                for k in (0, 1, 2, 3, 5):
                    for control in ("source_only", "target_only", "mixed", "size_matched_source"):
                        if control == "target_only" and k == 0:
                            continue
                        conditions.append(dict(id="T3", source_day=0, target_day=14, k=k,
                                               repeat=repeat, t3_control=control, heldout_levels=[]))
            elif suite == "unseen":
                for heldout in ([15], [30], [15, 30], [1]):
                    for day in (0, 7, 14):
                        conditions.append(dict(id="UNSEEN", source_day=day, target_day=day,
                            k=base["k"], repeat=repeat, heldout_levels=heldout))
            else:
                raise ValueError("Unknown suite")
            for c in conditions:
                if "support_seed" not in c:
                    c["support_seed"] = config["seed"] + support_seed
    # Freeze the entire matrix before the first prediction, preventing accidental adaptive grid changes.
    import uuid
    lock = Path(config["output"]) / "locks" / f"matrix_{uuid.uuid4().hex}.json"
    write_json(lock, {"suite": suite, "config": config, "conditions": conditions,
                      "evaluate_test": evaluate_test, "support_seed_count": support_seeds,
                      "note": "Predetermined numerical grid; not evidence of optimal method selection"})
    results = []
    for i, condition in enumerate(conditions):
        cfg = deepcopy(config)
        cfg["seed"] = condition["support_seed"]
        cfg["experiment"].update({k:v for k,v in condition.items() if k != "support_seed"})
        if suite == "unseen" and "spectral_centroid_interpolation" not in cfg["models"]:
            cfg["models"].append("spectral_centroid_interpolation")
        batch = run(cfg, allow_api=False, evaluate_test=evaluate_test)
        results.extend(batch)
        print(json.dumps({"condition": i+1, "total": len(conditions), "statuses": [r["status"] for r in batch]}), flush=True)
    return {"matrix_lock": str(lock), "conditions": len(conditions), "model_runs": len(results),
            "failed": sum(r["status"] == "failed" for r in results)}


def verify_run(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for line in (directory / "checksum.sha256").read_text(encoding="utf-8").splitlines():
        expected, filename = line.split("  ", 1)
        if file_hash(directory / filename) != expected:
            raise ValueError(f"Artifact checksum mismatch: {filename}")
    cfg = manifest["config"]
    meta, _, _, _ = load_surimi(cfg)
    split = pd.read_csv(Path(cfg["splits"]) / f"surimi_split_v1_repeat_{cfg['experiment']['repeat']}.csv")
    if file_hash(Path(cfg["splits"]) / f"surimi_split_v1_repeat_{cfg['experiment']['repeat']}.csv") != manifest["split_hash"]:
        raise ValueError("Frozen split hash changed")
    validate_split(meta, split)
    roles = split.set_index("physical_sample_id").split.to_dict()
    heldout = set(cfg["experiment"].get("heldout_levels", []))
    contexts = pd.read_csv(directory / "support_context.csv")
    queries = {r.acquisition_id: r.physical_sample_id for r in meta.itertuples()}
    query_order = {r.acquisition_id: i for i, r in meta.iterrows()}
    feedback_ids = set(manifest.get("adaptive_reference_ids", []))
    for ref in contexts.itertuples():
        adaptive_feedback = (manifest.get("validation_adaptive_context", False) and
            ref.role == "validation_feedback" and roles[ref.reference_physical_sample_id] == "val" and
            ref.reference_acquisition_id in feedback_ids and
            (ref.query_split == "test" or (ref.query_split == "val" and
            query_order[ref.reference_acquisition_id] < query_order[ref.query_acquisition_id])))
        if (roles[ref.reference_physical_sample_id] != "train" and not adaptive_feedback) or ref.reference_label in heldout:
            raise ValueError("Context escaped legal train pool")
        if ref.role in ("prompt", "validation_feedback") and ref.reference_physical_sample_id == queries[ref.query_acquisition_id]:
            raise ValueError("Query physical ID leaked into prompt")
    for part, count in (("train", 10), ("val", 2), ("test", 4)):
        predictions = pd.read_csv(directory / f"predictions_{part}.csv")
        if len(predictions) != len(cfg["data"]["levels"])*count or predictions.acquisition_id.duplicated().any():
            raise ValueError("Missing or duplicate predictions")
        if any(roles[sid] != part for sid in predictions.physical_sample_id):
            raise ValueError("Prediction partition mismatch")
    return {"run_id": manifest["run_id"], "status": "verified", "checks": ["checksums", "split", "coverage", "quarantine", "self-exclusion"]}


def drift(config, output):
    meta, x, _, hashes = load_surimi(config)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for sid, group in meta.groupby("physical_sample_id"):
        by_day = {int(meta.loc[i, "day"]): x[i] for i in group.index}
        for target in (7, 14):
            a, b = by_day[0], by_day[target]
            sam = float(np.arccos(np.clip(a @ b / max(np.linalg.norm(a)*np.linalg.norm(b), 1e-15), -1, 1)))
            rows.append(dict(physical_sample_id=sid, concentration_pct=float(group.concentration_pct.iloc[0]),
                source_day=0, target_day=target, spectral_angle_radians=sam,
                mean_absolute_spectral_change=float(np.abs(a-b).mean())))
    write_csv(out / "paired_drift.csv", rows)
    write_json(out / "manifest.json", {"source_hashes": hashes, "observations": len(rows),
        "interpretation": "Paired descriptive spectral drift, not proof of chemical mechanism"})
    return {"output": str(out), "paired_comparisons": len(rows)}


def preview(config, output):
    meta, x, wave, _ = load_surimi(config)
    e = config["experiment"]
    split = pd.read_csv(Path(config["splits"]) / f"surimi_split_v1_repeat_{e['repeat']}.csv")
    validate_split(meta, split)
    roles = meta.physical_sample_id.map(split.set_index("physical_sample_id").split)
    pool = meta.index[(roles == "train") & (meta.day == e["source_day"]) &
                      ~meta.concentration_pct.isin(e.get("heldout_levels", []))]
    support = select_support(meta, x, pool, e["k"], config["seed"])
    q = meta.index[(roles == "val") & (meta.day == e["target_day"])][0]
    rep = Representation(e["representation"]).fit(x[support])
    messages = make_prompt(rep.transform(x[support]), meta.loc[support, "concentration_pct"].to_numpy(),
        rep.transform(x[[q]])[0], config["data"]["levels"], e["representation"],
        wave.tolist() if e["representation"] in ("raw", "snv", "sg") else None)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "prompt.json", messages)
    write_json(out / "context.json", {"query_acquisition_id": meta.loc[q, "acquisition_id"],
        "reference_ids": meta.loc[support, "acquisition_id"].tolist(), "prompt_hash": digest(messages),
        "api_requests": 0, "chars": len(json.dumps(messages)),
        "token_estimate": "Unavailable: configure exact provider tokenizer before pricing"})
    return {"output": str(out), "api_requests": 0, "examples": len(support)}
