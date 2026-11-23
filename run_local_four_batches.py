"""Run the four-batch web-evaluation package through the local Qwen model.

Only prompt.txt and train/val/test inputs are sent to the model. Label files,
metadata, private mappings, and provenance are read only after generation (or
never) and are never included in the inference prompt.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE = PROJECT_ROOT / "data_json" / "four_batches"
DEFAULT_MODEL = Path(r"D:\LLM models\Qwen3.5-9B")
MODEL_CONTEXT = 262_144


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def build_content(prompt: str, directory: Path) -> str:
    sections = [prompt.strip()]
    for name in ("train", "val", "test"):
        path = directory / f"{name}.json"
        sections.append(f"\n\n===== {name}.json =====\n{read_text(path).strip()}")
    return "".join(sections)


def parse_model_json(text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    decoder = json.JSONDecoder()
    for i, char in enumerate(cleaned):
        if char not in "[{":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and ("experiment_id" in obj or "val" in obj or "test" in obj):
            return obj
    return None


def score_split(predictions: Any, labels_path: Path, class_labels: list[str]) -> dict[str, Any]:
    from sklearn.metrics import balanced_accuracy_score, f1_score

    truth_rows = json.loads(read_text(labels_path))
    truths = {str(row["id"]): str(row["label"]) for row in truth_rows}
    if not isinstance(predictions, list):
        predictions = []
    pred_by_id: dict[str, str] = {}
    duplicate_ids: list[str] = []
    invalid_rows = 0
    for row in predictions:
        if not isinstance(row, dict) or "id" not in row or "label" not in row:
            invalid_rows += 1
            continue
        sample_id = str(row["id"])
        if sample_id in pred_by_id:
            duplicate_ids.append(sample_id)
            continue
        pred_by_id[sample_id] = str(row["label"])

    expected_ids = set(truths)
    missing = sorted(expected_ids - set(pred_by_id))
    extra = sorted(set(pred_by_id) - expected_ids)
    # A missing prediction, invalid class, or duplicate ID scores as incorrect.
    y_true = [truths[sample_id] for sample_id in truths]
    y_pred = [pred_by_id.get(sample_id, "__MISSING__") for sample_id in truths]
    allowed = [str(x) for x in class_labels]
    y_pred = [label if label in allowed else "__INVALID__" for label in y_pred]
    correct = sum(a == b for a, b in zip(y_true, y_pred))
    present_labels = sorted(set(y_true), key=str)
    return {
        "correct": correct,
        "total": len(y_true),
        "accuracy_pct": round(100 * correct / len(y_true), 2) if y_true else 0.0,
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=allowed, average="macro", zero_division=0)), 6) if y_true else 0.0,
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6) if y_true else 0.0,
        "missing_predictions": len(missing),
        "duplicate_prediction_ids": len(duplicate_ids),
        "extra_prediction_ids": len(extra),
        "invalid_rows": invalid_rows,
        "duplicate_ids": duplicate_ids,
        "missing_ids": missing,
        "extra_ids": extra,
        "support_by_label": {label: sum(y == label for y in y_true) for label in present_labels},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "four_batches_qwen35_local")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N feasible experiments (0 means all).")
    parser.add_argument("--batch", default="", help="Optional exact batch name, e.g. batch_01_T1.")
    parser.add_argument("--max-input-tokens", type=int, default=240_000)
    parser.add_argument("--max-bytes", type=int, default=600_000, help="Skip preflight tokenization above this combined byte size.")
    parser.add_argument("--max-new-tokens-per-sample", type=int, default=32)
    parser.add_argument("--api-base", default=os.environ.get("LOCAL_LLM_API_BASE"),
                        help="OpenAI-compatible endpoint, e.g. http://127.0.0.1:8080/v1 (default: LOCAL_LLM_API_BASE).")
    parser.add_argument("--context", type=int, default=0,
                        help="Context tokens for preflight (default: the server's 65536 in API mode, else config.json).")
    parser.add_argument("--resume", action="store_true", help="Skip experiment IDs already present in the result JSONL.")
    args = parser.parse_args()

    package = args.package.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = package / "index_current.csv"
    with index_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.batch:
        rows = [row for row in rows if row["batch"] == args.batch]
    available = [row for row in rows if (package / row["directory"]).is_dir()]
    missing_count = len(rows) - len(available)
    if missing_count:
        print(f"Skipping {missing_count} experiments whose data directories are missing.", flush=True)
    rows = available
    if not rows:
        raise SystemExit("No available experiments selected.")

    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    result_path = output_dir / "predictions.jsonl"
    done_ids: set[str] = set()
    if args.resume and result_path.exists():
        for line in result_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") in {"completed", "context_skipped"}:
                done_ids.add(record["experiment_id"])

    planned: list[tuple[dict[str, str], Path, str, int, int]] = []
    preflight_records: list[dict[str, Any]] = []
    cfg_path = args.model / "config.json"
    if args.context:
        model_context = args.context
    elif args.api_base:
        model_context = 65_536              # the local Strata server's context (-c 64k)
    elif cfg_path.exists():
        context_cfg = json.loads(cfg_path.read_text(encoding="utf-8")).get("text_config", {})
        model_context = int(context_cfg.get("max_position_embeddings", MODEL_CONTEXT))
    else:
        model_context = MODEL_CONTEXT
    for row in rows:
        exp_id = row["experiment_id"]
        if exp_id in done_ids:
            continue
        directory = package / row["directory"]
        content = build_content(read_text(directory / "prompt.txt"), directory)
        n_output = int(row.get("val_count") or 0) + int(row.get("test_count") or 0)
        max_new_tokens = max(512, n_output * args.max_new_tokens_per_sample + 1024)
        byte_count = len(content.encode("utf-8"))
        if byte_count > args.max_bytes:
            token_count = None
            reason = f"combined_input_bytes={byte_count} exceeds preflight limit {args.max_bytes}"
        else:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            token_ids = ids["input_ids"] if hasattr(ids, "keys") and "input_ids" in ids else ids
            token_count = len(token_ids[0]) if token_ids and isinstance(token_ids[0], list) else len(token_ids)
            reason = ""
            if token_count + max_new_tokens > min(args.max_input_tokens, model_context):
                reason = f"input_tokens={token_count} + max_new_tokens={max_new_tokens} exceeds safe context"
        if reason:
            record = {
                "experiment_id": exp_id,
                "batch": row["batch"],
                "directory": row["directory"],
                "status": "context_skipped",
                "input_bytes": byte_count,
                "input_tokens": token_count,
                "reason": reason,
            }
            preflight_records.append(record)
        else:
            planned.append((row, directory, content, token_count, max_new_tokens))

    print(f"Indexed {len(rows)} experiments; feasible after exact preflight: {len(planned)}; context-skipped: {len(preflight_records)}", flush=True)
    for record in preflight_records:
        print(f"SKIP {record['experiment_id']}: {record['reason']}", flush=True)

    if args.limit > 0:
        planned = planned[:args.limit]
    if not planned:
        print("No feasible experiments selected.", flush=True)
        return

    from utils.local_model import LocalLLM

    if args.api_base:
        print(f"Using the OpenAI-compatible endpoint {args.api_base} (context {model_context} tokens)", flush=True)
    else:
        print(f"Loading local Qwen model: {args.model}", flush=True)
    llm = LocalLLM(
        base_model_path=str(args.model),
        lora_path=None,
        device="auto",
        max_memory={0: "11GiB", "cpu": "16GiB"},
        api_base=args.api_base,
    )
    mode = "a" if args.resume else "w"
    with result_path.open(mode, encoding="utf-8", buffering=1) as out:
        for n, (row, directory, content, input_tokens, max_new_tokens) in enumerate(planned, start=1):
            exp_id = row["experiment_id"]
            started = time.time()
            print(f"[{n}/{len(planned)}] {exp_id} input_tokens={input_tokens} max_new_tokens={max_new_tokens}", flush=True)
            try:
                response = llm.chat.completions.create(
                    model="Qwen3.5-9B-local",
                    messages=[{"role": "user", "content": content}],
                    temperature=0.0,
                    max_tokens=max_new_tokens,
                    prefill_chunk_size=512,
                )
                raw = response.choices[0].message.content or ""
                parsed = parse_model_json(raw)
                elapsed = time.time() - started
                record: dict[str, Any] = {
                    "experiment_id": exp_id,
                    "batch": row["batch"],
                    "directory": row["directory"],
                    "experiment_role": row["experiment_role"],
                    "status": "completed" if parsed is not None else "parse_error",
                    "input_tokens": input_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "elapsed_seconds": round(elapsed, 2),
                    "response": parsed,
                    "raw_response": raw,
                }
                if parsed is not None:
                    meta = json.loads(read_text(directory / "metadata.json"))
                    classes = [str(x) for x in meta.get("candidate_labels", [])]
                    record["experiment_id_matches"] = str(parsed.get("experiment_id", "")) == exp_id
                    record["val_metrics"] = score_split(parsed.get("val"), directory / "val_labels.json", classes)
                    record["test_metrics"] = score_split(parsed.get("test"), directory / "test_labels.json", classes)
                else:
                    (output_dir / f"{exp_id}.raw.txt").write_text(raw, encoding="utf-8")
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"  status={record['status']} elapsed={elapsed:.1f}s", flush=True)
            except Exception as exc:
                record = {
                    "experiment_id": exp_id,
                    "batch": row["batch"],
                    "directory": row["directory"],
                    "status": "error",
                    "input_tokens": input_tokens,
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": round(time.time() - started, 2),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"  ERROR {record['error']}", flush=True)

    skipped_path = output_dir / "context_skipped.json"
    skipped_path.write_text(json.dumps(preflight_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prediction checkpoint: {result_path}", flush=True)
    print(f"Context exclusions: {skipped_path}", flush=True)


if __name__ == "__main__":
    main()
