"""Run bounded OpenAI-compatible API classification on prepared data_json packages.

Inference reads only each experiment's prompt.txt, train.json, val.json and
test.json. Ground-truth labels and private metadata are never opened here.
Live calls require both --execute and a positive --max-requests budget.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import runpy
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = ROOT / "data_json" / "four_batches"
DEFAULT_OUTPUT = ROOT / "outputs" / "api_classification"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_provider(name: str, config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        return {}
    values = runpy.run_path(str(config_path)).get("PROVIDERS", {})
    return dict(values.get(name, {}))


def package_messages(directory: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Prepare API-specific instructions while preserving the package task spec."""
    prompt = read_text(directory / "prompt.txt")
    # The source prompt was written for context-only web inference. Remove only
    # the block that forbids the inference transport itself; retain all task,
    # split, label, coverage, and output requirements.
    prompt = re.sub(r"【执行限制：预测阶段和评分阶段均适用】.*?(?=实验编号：)", "", prompt, flags=re.S)
    prompt = prompt.replace(
        "仅利用上下文进行分类",
        "仅依据本请求中提供的参考数据和光谱特征进行分类",
    )
    sections = [
        "本请求由项目的 API 分类运行器发送给你。API 仅用于调用当前模型，不代表你可以调用工具、代码或其他外部服务。",
        prompt.strip(),
    ]
    loaded: dict[str, Any] = {}
    for name in ("train", "val", "test"):
        raw = read_text(directory / f"{name}.json")
        try:
            loaded[name] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid {name}.json: {exc}") from exc
        sections.append(f"\n\n===== {name}.json =====\n{raw.strip()}")
    messages = [
        {"role": "system", "content": "Follow the user task and return only its requested JSON prediction object. Do not reveal chain-of-thought."},
        {"role": "user", "content": "\n\n".join(sections)},
    ]
    return messages, loaded


def first_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    decoder = json.JSONDecoder()
    for i, char in enumerate(cleaned):
        if char == "{":
            try:
                obj, _ = decoder.raw_decode(cleaned[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    raise ValueError("No JSON object found in model response")


def validate_predictions(obj: dict[str, Any], experiment_id: str,
                         inputs: dict[str, Any], prompt: str) -> dict[str, Any]:
    if str(obj.get("experiment_id", "")) != experiment_id:
        raise ValueError("experiment_id missing or does not match the selected index")
    candidate_match = re.search(r"允许的label仅为以下字符串：\s*(\[[^\n]+\])", prompt)
    if not candidate_match:
        candidate_match = re.search(r"允许的label仅为以下字符串：\s*(\[[^\]]+\])", prompt)
    if not candidate_match:
        raise ValueError("Could not read candidate labels from prompt.txt")
    candidates = {str(x) for x in json.loads(candidate_match.group(1))}
    result: dict[str, Any] = {"experiment_id": experiment_id}
    issues: list[str] = []
    for split in ("val", "test"):
        source = inputs[split]
        if not isinstance(source, list) or any(not isinstance(row, dict) or "id" not in row for row in source):
            raise ValueError(f"{split}.json must be a list of records with ids")
        expected = [str(row["id"]) for row in source]
        rows = obj.get(split)
        if not isinstance(rows, list):
            rows = []
            issues.append(f"{split}: prediction field missing or not a list")
        seen: set[str] = set()
        checked = []
        for row in rows:
            if not isinstance(row, dict) or "id" not in row or "label" not in row:
                issues.append(f"{split}: malformed prediction row")
                continue
            sid, label = str(row["id"]), str(row["label"])
            if sid in seen:
                issues.append(f"{split}: duplicate prediction id {sid}")
            seen.add(sid)
            if label not in candidates:
                issues.append(f"{split}: invalid label for id {sid}: {label}")
            checked.append({"id": sid, "label": label})
        missing = sorted(set(expected) - seen)
        extra = sorted(seen - set(expected))
        if missing:
            issues.append(f"{split}: {len(missing)} missing ids")
        if extra:
            issues.append(f"{split}: {len(extra)} extra ids")
        result[split] = checked
    result["issues"] = issues
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE, help="four_batches package root")
    parser.add_argument("--batch", default="", help="Optional batch name, e.g. batch_01_T1")
    parser.add_argument("--experiment-id", default="", help="Optional exact experiment_id from index_current.csv")
    parser.add_argument("--limit", type=int, default=1, help="Maximum selected experiments; default 1 for cost control")
    parser.add_argument("--provider", default="deepseek", help="Provider key in local_api_config.py")
    parser.add_argument("--config", type=Path, default=ROOT / "local_api_config.py")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true", help="Actually send API requests")
    parser.add_argument("--max-requests", type=int, default=0, help="Hard cap on new API requests")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()

    package = args.package.resolve()
    index_path = package / "index_current.csv"
    if not index_path.is_file():
        parser.error(f"Package index not found: {index_path}")
    import csv
    with index_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.batch:
        rows = [r for r in rows if r.get("batch") == args.batch]
    if args.experiment_id:
        rows = [r for r in rows if r.get("experiment_id") == args.experiment_id]
    rows = [r for r in rows if (package / r["directory"]).is_dir()]
    if args.limit < 1:
        parser.error("--limit must be positive")
    rows = rows[:args.limit]
    if not rows:
        parser.error("No matching experiment packages")

    provider = load_provider(args.provider, args.config.resolve())
    # Provider credentials are never serialized into output or printed.
    ready = bool(provider.get("base_url") and provider.get("model") and provider.get("api_key"))
    preview = [{"experiment_id": r["experiment_id"], "batch": r["batch"], "directory": r["directory"]} for r in rows]
    if not args.execute:
        print(json.dumps({"mode": "preview_only", "provider": args.provider,
                          "model": provider.get("model"), "base_url": provider.get("base_url"),
                          "credentials_filled": ready, "planned_experiments": preview,
                          "api_requests": 0,
                          "next_step": "Configure local_api_config.py and explicitly pass --execute --max-requests N"},
                         ensure_ascii=False, indent=2))
        return 0
    if args.max_requests < 1:
        parser.error("--execute requires a positive --max-requests")
    if not ready:
        parser.error(f"Provider {args.provider!r} needs base_url, exact model and api_key in {args.config}")
    if args.max_requests < len(rows):
        parser.error(f"Selected {len(rows)} experiments but request budget is {args.max_requests}; no calls started")

    try:
        from openai import OpenAI
    except ImportError:
        parser.error("Install API dependency with: python -m pip install -r requirements-api.txt")
    client = OpenAI(api_key=provider["api_key"], base_url=provider["base_url"], max_retries=0, timeout=120)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "cache"
    cache.mkdir(exist_ok=True)
    requests_made = 0
    results_path = output / "predictions.jsonl"
    with results_path.open("a", encoding="utf-8") as out:
        for row in rows:
            experiment_id = row["experiment_id"]
            directory = package / row["directory"]
            messages, inputs = package_messages(directory)
            prompt = read_text(directory / "prompt.txt")
            cache_key = digest({"endpoint": provider["base_url"], "model": provider["model"],
                                "messages": messages, "temperature": args.temperature,
                                "max_tokens": args.max_tokens})
            cache_file = cache / f"{cache_key}.json"
            started = time.time()
            try:
                if cache_file.exists():
                    response = json.loads(read_text(cache_file))
                    cached = True
                else:
                    if requests_made >= args.max_requests:
                        raise RuntimeError("API request budget exhausted")
                    # Count attempts even when the provider raises an error.
                    requests_made += 1
                    response_obj = client.chat.completions.create(
                        model=provider["model"], messages=messages,
                        temperature=args.temperature, max_tokens=args.max_tokens)
                    response = response_obj.model_dump()
                    cache_file.write_text(json.dumps(response, ensure_ascii=False), encoding="utf-8")
                    cached = False
                raw = response["choices"][0]["message"].get("content") or ""
                parsed = first_json_object(raw)
                validated = validate_predictions(parsed, experiment_id, inputs, prompt)
                pred_dir = output / "experiments" / experiment_id
                pred_dir.mkdir(parents=True, exist_ok=True)
                (pred_dir / "predictions.json").write_text(json.dumps(validated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                (pred_dir / "raw_response.txt").write_text(raw, encoding="utf-8")
                record = {"experiment_id": experiment_id, "batch": row.get("batch"),
                          "directory": row.get("directory"), "status": "completed",
                          "cached": cached, "latency_seconds": round(time.time()-started, 2),
                          "model": provider["model"], "response_file": str(pred_dir / "predictions.json"),
                          "issues": validated["issues"]}
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                api_key = provider.get("api_key", "")
                if api_key:
                    message = message.replace(api_key, "[REDACTED]")
                record = {"experiment_id": experiment_id, "batch": row.get("batch"),
                          "directory": row.get("directory"), "status": "error", "cached": False,
                          "latency_seconds": round(time.time()-started, 2), "error": message}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(json.dumps(record, ensure_ascii=False))
    print(json.dumps({"actual_api_requests_this_run": requests_made, "results": str(results_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
