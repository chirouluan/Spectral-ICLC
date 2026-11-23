#!/usr/bin/env python3
"""Independent variant runner for the four-batches evaluation.

All outputs go to outputs/four_batches_strata_variants/<variant>/ and never
touch outputs/four_batches_strata.

Phases
------
P0  prompt variant: grouped/rounded/compact JSON + target-day priority +
    ordinal instruction + thinking.  Selects T2 full_reference (15) and
    T3 target_only/mixed at target_k=3,5 (20): 35 experiments.
P1  representation variant: refit an LDA8 on the target-day references
    (only the reference labels are used), then prompt the model.  Selects
    T3 target_only at target_k=3,5 (10); optionally adds mixed (20).
P2  representation + hint variant: per-day z-score alignment using only
    unlabeled features, plus class centroids / nearest-centroid hints.
    Selects T2 day0->day7 and day7->day14 full_reference (10).
P3  self-consistency: rerun the best variant R times with shuffled grouped
    reference order, then majority-vote per sample.  Default subset:
    T3 target_only k=5 (5) + T2 day0->day7 full (5), R=3.
P4  PUBLIC prompt variant: task-aware ordinal / nominal / K=1 instructions on
    the selected batch_04 datasets (80 experiments in the current index).
ALL raw four-batch rerun: every available experiment (all four batches) with the
    original prompt.txt and train/val/test JSON read verbatim, no variant hints
    and no transforms.  This reproduces the historical P0 baseline protocol.

Robustness
----------
- A chunk whose response ends with ``finish_reason == "length"`` is split and
  only its missing records are retried, bounded by MAX_RETRY_DEPTH.
- Every chunk response is cached under ``<variant>/chunk_checkpoints/`` keyed by
  the exact request content, so ``--resume`` skips chunks already done.
- ``--response-format json_object`` opts into server-side JSON validation.
  Strata has no grammar decoder: it validates the final answer and rejects
  truncated output instead of returning partial JSON.

Examples
--------
python3 run_variants.py --phase P0 --dry-run
python3 run_variants.py --phase P0 --concurrency 2
python3 run_variants.py --phase P1 --concurrency 2
python3 run_variants.py --phase P2 --concurrency 2
python3 run_variants.py --phase P3 --p3-base best --repeats 3 --concurrency 2
python3 run_variants.py --phase P4 --no-thinking --concurrency 2
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import re
import shutil
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGE = PROJECT_ROOT / "data_json" / "four_batches"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "four_batches_strata_variants"
DEFAULT_API = os.environ.get("LOCAL_LLM_API_BASE", "http://127.0.0.1:8080/v1")
DEFAULT_MODEL = os.environ.get("LOCAL_LLM_API_MODEL", "qwen3.8-flash-next-q2_0")
TOKENIZER_PATH = Path(
    os.environ.get("LOCAL_LLM_TOKENIZER", str(PROJECT_ROOT / "strata-kit" / "tokenizer"))
)

# Ordinal datasets: labels are concentration / addition levels.
ORDINAL_LABEL_ORDER: dict[str, list[str]] = {
    "turmeric_levels": ["0", "0.1", "0.5", "1", "2.5", "5", "10"],
}

# Short semantic notes for nominal public datasets (the prompt already carries
# the full description; keep this additive and conservative).
DATASET_NOTES: dict[str, str] = {
    "beef": "UCR Beef五类代码1–5未提供语义次序，按名义类别处理；不要根据编号大小推断相邻关系。",
    "coffee": "Arabica/Robusta为名义类别，没有数值顺序。",
    "fresh_meats": "Chicken/Pork/Turkey为物种名义类别，没有数值顺序。",
    "olive_oils": "产地标签为名义类别，没有数值顺序。",
    "turmeric_authenticity": "authentic/adulterated为二分类（authentic=0%，adulterated>0%）。",
}


def _load_helpers():
    spec = importlib.util.spec_from_file_location(
        "_four_batches_helpers", PROJECT_ROOT / "run_local_four_batches.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _load_helpers()
read_text = H.read_text
parse_model_json = H.parse_model_json
score_split = H.score_split


# ---------------------------------------------------------------------------
# data loading / selection
# ---------------------------------------------------------------------------
def load_index() -> list[dict[str, str]]:
    with (PACKAGE / "index_current.csv").open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _num(value: Any, default: float = 1e9) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_t3(row: dict[str, str]) -> bool:
    return row.get("task") == "T3"


def select_experiments(phase: str, args: argparse.Namespace) -> list[dict[str, str]]:
    rows = load_index()
    selected: list[dict[str, str]] = []
    if phase == "ALL":
        selected = list(rows)
    elif phase == "P0":
        for row in rows:
            if row["task"] == "T2" and row["reference_mode"] == "full_reference":
                selected.append(row)
            elif (
                row["task"] == "T3"
                and row["control"] in {"target_only", "mixed"}
                and int(_num(row.get("target_k"), -1)) in {3, 5}
            ):
                selected.append(row)
    elif phase == "P1":
        controls = {"target_only", "mixed"} if args.include_mixed else {"target_only"}
        for row in rows:
            if (
                row["task"] == "T3"
                and row["control"] in controls
                and int(_num(row.get("target_k"), -1)) in {3, 5}
            ):
                selected.append(row)
    elif phase == "P2":
        for row in rows:
            if row["task"] != "T2" or row["reference_mode"] != "full_reference":
                continue
            pair = (_num(row.get("source_day")), _num(row.get("target_day")))
            if pair in {(0.0, 7.0), (7.0, 14.0)}:
                selected.append(row)
    elif phase == "P4":
        for row in rows:
            if row.get("batch") == "batch_04_public":
                selected.append(row)
    elif phase == "P3":
        base = args.p3_base
        if base in {"best", "p0"}:
            selected.extend(
                row for row in rows
                if row["task"] == "T3" and row["control"] == "target_only"
                and int(_num(row.get("target_k"), -1)) == 5
            )
            selected.extend(
                row for row in rows
                if row["task"] == "T2" and row["reference_mode"] == "full_reference"
                and (_num(row.get("source_day")), _num(row.get("target_day"))) == (0.0, 7.0)
            )
        elif base == "p1":
            selected.extend(
                row for row in rows
                if row["task"] == "T3" and row["control"] == "target_only"
                and int(_num(row.get("target_k"), -1)) == 5
            )
        elif base == "p2":
            selected.extend(
                row for row in rows
                if row["task"] == "T2" and row["reference_mode"] == "full_reference"
                and (_num(row.get("source_day")), _num(row.get("target_day"))) == (0.0, 7.0)
            )
    else:
        raise ValueError(f"unknown phase {phase}")
    # Deleted datasets keep their index rows; never select an experiment whose
    # data directory is gone (honey_adulteration and fruit_purees were removed).
    available = [r for r in selected if (PACKAGE / r["directory"]).is_dir()]
    missing = len(selected) - len(available)
    if missing:
        print(f"Skipping {missing} selected experiments whose data directories are missing.", flush=True)
    return available


def method_for(row: dict[str, str], phase: str, args: argparse.Namespace) -> str:
    if phase == "ALL":
        return "default"
    if phase == "P0":
        return "p0"
    if phase == "P1":
        return "p1"
    if phase == "P2":
        return "p2"
    if phase == "P4":
        return "public"
    # P3
    base = args.p3_base
    if base != "best":
        return base
    return "p1" if row.get("task") == "T3" else "p2"


# ---------------------------------------------------------------------------
# record helpers
# ---------------------------------------------------------------------------
def load_experiment(directory: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    meta = json.loads(read_text(directory / "metadata.json"))
    data = {
        split: json.loads(read_text(directory / f"{split}.json"))
        for split in ("train", "val", "test")
    }
    return meta, data


def clone_split(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(rec, features=list(rec.get("features", []))) for rec in records]


def round_features(records: list[dict[str, Any]], decimals: int) -> list[dict[str, Any]]:
    out = []
    for rec in records:
        rr = dict(rec)
        rr["features"] = [round(float(x), decimals) for x in rec.get("features", [])]
        out.append(rr)
    return out


def group_train(records: list[dict[str, Any]], shuffle: bool, seed: int) -> list[dict[str, Any]]:
    """Group references by day, then label; target-day groups come last."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[(str(rec.get("day", "")), str(rec.get("label", "")))].append(rec)
    by_day: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in groups:
        by_day[key[0]].append(key)
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for day in sorted(by_day, key=_num):
        keys = by_day[day]
        if shuffle:
            rng.shuffle(keys)
        else:
            keys.sort(key=lambda k: _num(k[1]))
        for key in keys:
            bucket = groups[key]
            if shuffle:
                rng.shuffle(bucket)
            else:
                bucket.sort(key=lambda r: str(r.get("id", "")))
            out.extend(bucket)
    return out


def shuffle_records(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    out = list(records)
    random.Random(seed).shuffle(out)
    return out


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------
def zscore_by_day(data: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """Per-day z-score using only feature statistics (no labels).

    The target-day statistics are estimated from the val/test inputs (and any
    target-day references); this is transductive but label-free.
    """
    by_day: dict[str, list[np.ndarray]] = defaultdict(list)
    for split in ("train", "val", "test"):
        for rec in data[split]:
            by_day[str(rec.get("day", ""))].append(np.asarray(rec["features"], float))
    means: dict[str, np.ndarray] = {}
    stds: dict[str, np.ndarray] = {}
    for day, arrays in by_day.items():
        mat = np.vstack(arrays)
        means[day] = mat.mean(0)
        sd = mat.std(0, ddof=1) if len(mat) > 1 else np.ones(mat.shape[1])
        stds[day] = np.where(sd < 1e-9, 1.0, sd)
    out: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        out[split] = []
        for rec in data[split]:
            day = str(rec.get("day", ""))
            rr = dict(rec)
            rr["features"] = ((np.asarray(rec["features"], float) - means[day]) / stds[day]).tolist()
            out[split].append(rr)
    return out


def fit_lda_n(
    data: dict[str, list[dict[str, Any]]], max_components: int = 8
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Refit an LDA on the reference (train.json) labels and transform all splits."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    x_train = np.asarray([rec["features"] for rec in data["train"]], float)
    y_train = np.asarray([str(rec["label"]) for rec in data["train"]])
    n_classes = len(set(y_train.tolist()))
    n_components = min(max_components, n_classes - 1, x_train.shape[1])
    if n_components <= 0:
        return ({k: clone_split(v) for k, v in data.items()}, {"n_components": 0, "solver": "none"})
    try:
        model = LDA(n_components=n_components).fit(x_train, y_train)
        solver = "svd"
    except Exception:
        model = LDA(n_components=n_components, solver="lsqr", shrinkage="auto").fit(x_train, y_train)
        solver = "lsqr+shrinkage"
    out: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        x = np.asarray([rec["features"] for rec in data[split]], float)
        z = model.transform(x)
        out[split] = [dict(rec, features=z[i].tolist()) for i, rec in enumerate(data[split])]
    return out, {"n_components": n_components, "solver": solver, "n_fit": len(y_train), "n_classes": n_classes}


def fit_target_lda(data: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    return fit_lda_n(data, 8)


def fit_pca_n(
    data: dict[str, list[dict[str, Any]]], n_components: int
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Label-free PCA fitted on all provided features (train refs + val/test)."""
    from sklearn.decomposition import PCA

    x_all = np.vstack(
        [np.asarray(rec["features"], float) for split in ("train", "val", "test") for rec in data[split]]
    )
    n = min(n_components, x_all.shape[0] - 1, x_all.shape[1])
    if n <= 0:
        return ({k: clone_split(v) for k, v in data.items()}, {"n_components": 0, "fit_on": "none"})
    model = PCA(n_components=n, random_state=0).fit(x_all)
    out: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        x = np.asarray([rec["features"] for rec in data[split]], float)
        z = model.transform(x)
        out[split] = [dict(rec, features=z[i].tolist()) for i, rec in enumerate(data[split])]
    return out, {
        "n_components": n,
        "fit_on": "all_unlabeled_inputs",
        "explained_variance": round(float(model.explained_variance_ratio_.sum()), 4),
    }


def subsample_per_class(
    records: list[dict[str, Any]], n_per_class: int, seed: int = 12345
) -> list[dict[str, Any]]:
    """Deterministic stratified subsample used only to fit the context budget."""
    if n_per_class <= 0:
        return list(records)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[str(rec.get("label"))].append(rec)
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for label in sorted(groups, key=_num):
        bucket = list(groups[label])
        rng.shuffle(bucket)
        out.extend(bucket[:n_per_class])
    rng.shuffle(out)
    return out


def class_centroids(data: dict[str, list[dict[str, Any]]]) -> dict[str, np.ndarray]:
    x = np.asarray([rec["features"] for rec in data["train"]], float)
    y = [str(rec["label"]) for rec in data["train"]]
    centroids: dict[str, np.ndarray] = {}
    for label in sorted(set(y), key=_num):
        centroids[label] = x[[i for i, v in enumerate(y) if v == label]].mean(0)
    return centroids


def prototype_blocks(data: dict[str, list[dict[str, Any]]], hint: str, decimals: int) -> str:
    if hint == "none":
        return ""
    centroids = class_centroids(data)
    labels = list(centroids)
    lines = ["【辅助信息：按天对齐后的类原型（参考样本均值，%d位小数）】" % decimals]
    for label in labels:
        vec = ", ".join(f"{v:.{decimals}f}" for v in centroids[label])
        lines.append(f"{label}: [{vec}]")
    text = "\n".join(lines)
    if hint in {"nearest", "both"}:
        mat = np.vstack([centroids[label] for label in labels])
        lines = ["【辅助信息：每条待测样本最近的3个类原型（欧氏距离，越小越近）】"]
        for split in ("val", "test"):
            for rec in data[split]:
                x = np.asarray(rec["features"], float)
                dist2 = ((mat - x) ** 2).sum(1)
                order = np.argsort(dist2)[:3]
                cells = ", ".join(f"{labels[i]}({dist2[i] ** 0.5:.2f})" for i in order)
                lines.append(f"{rec['id']}: {cells}")
        text += "\n" + "\n".join(lines)
    return text


# ---------------------------------------------------------------------------
# prompt variants
# ---------------------------------------------------------------------------
FEATURE_DESC = "源日与目标日使用同一个只由合法源日Train拟合的变换"
FEATURE_DESC_P1 = "本变体使用目标日参考样本（仅参考标签）重新拟合的LDA8维变换"
FEATURE_DESC_P2 = (
    "本变体在LDA8维上按采集日做了无标签z-score对齐"
    "（目标日仅使用特征统计，不使用标签）"
)


def insert_before_prediction(prompt: str, block: str) -> str:
    marker = "一、预测阶段"
    if marker in prompt:
        return prompt.replace(marker, block.strip() + "\n\n" + marker, 1)
    return prompt.strip() + "\n\n" + block.strip()


def prompt_variant(
    prompt: str, meta: dict[str, Any], method: str, include_mixed: bool = False
) -> tuple[str, str]:
    """Return (modified_prompt, note_for_metadata)."""
    task = meta.get("task")
    if method == "p0":
        if task == "T2":
            block = (
                "【变体提示】不同采集日的LD绝对偏移可能不同；判断时请比较样本在各类别内部的"
                "相对模式，不要依赖绝对LD值。先估计一个浓度值，再取最接近的允许标签。"
            )
        elif task == "T3" and meta.get("control") == "mixed":
            block = (
                "【变体提示】day=14的参考是目标日，可信度最高；day=0的参考只用于理解类别语义，"
                "两者冲突时以day=14为准。先估计一个浓度值，再取最接近的允许标签。"
            )
        else:
            block = "【变体提示】先估计一个浓度值，再取最接近的允许标签。"
        return insert_before_prediction(prompt, block), "p0_instructions"
    if method == "p1":
        note = (
            "【变体提示】本变体的8维特征由参考样本重新拟合的LDA给出（仅使用参考标签，"
            "val/test不参与拟合）；其余流程不变。"
        )
        out = prompt.replace(FEATURE_DESC, FEATURE_DESC_P1)
        out = out.replace("特征仍使用共同的源日变换", "特征使用本变体重新拟合的变换")
        return insert_before_prediction(out, note), "p1_target_lda"
    if method == "p2":
        note = (
            "【变体提示】特征已按采集日做无标签z-score对齐；下面的类原型/最近邻基于对齐后的"
            "特征计算，可作为判断参考。"
        )
        out = prompt.replace(FEATURE_DESC, FEATURE_DESC_P2)
        return insert_before_prediction(out, note), "p2_zday_proto"
    if method == "public":
        dataset = str(meta.get("dataset_id", ""))
        labels = [str(x) for x in (meta.get("candidate_labels") or [])]
        blocks: list[str] = []
        if dataset in ORDINAL_LABEL_ORDER:
            order = ORDINAL_LABEL_ORDER[dataset]
            blocks.append(
                "【变体提示·有序标签】允许标签是递增的浓度/掺假等级："
                + " < ".join(order)
                + "。请先沿参考的类别均值方向估计样本的连续等级，再取最接近的允许标签；"
                "相邻等级最易混淆，不要凭整体亮度或第一维数值直接跳级。"
            )
        else:
            note = DATASET_NOTES.get(dataset, "")
            blocks.append(
                "【变体提示·名义标签】类别之间没有数值顺序，请比较谱形/主成分的整体模式，"
                "不要按标签编号距离或各类参考数量外推。" + note
            )
        k = int(_num(meta.get("k"), 0))
        if meta.get("reference_mode") == "context_k" and k == 1:
            blocks.append(
                "【变体提示·K=1】每类只有1条参考，无法估计类内分布；请把待测样本与各参考做"
                "相对谱形比较，并结合题目给出的类别语义，不要因为某类参考少就回避该类。"
            )
        return insert_before_prediction(prompt, "\n".join(blocks)), "public_task_aware"
    return prompt, method


# ---------------------------------------------------------------------------
# content building
# ---------------------------------------------------------------------------
def dump_records(records: list[dict[str, Any]], fmt: str) -> str:
    if fmt == "plain":
        return json.dumps(records, ensure_ascii=False, indent=1)
    return json.dumps(records, ensure_ascii=False, separators=(",", ":"))


def build_sections(
    prompt: str,
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
    extra: str,
    fmt: str,
) -> str:
    sections = [prompt.strip()]
    if extra:
        sections.append("\n\n" + extra.strip())
    for split, records in (("train", train), ("val", val), ("test", test)):
        sections.append(f"\n\n===== {split}.json =====\n{dump_records(records, fmt)}")
    return "".join(sections)


def build_content(
    prompt: str,
    data: dict[str, list[dict[str, Any]]],
    extra: str,
    fmt: str,
) -> str:
    return build_sections(prompt, data["train"], data["val"], data["test"], extra, fmt)


def build_content_raw(prompt: str, directory: Path) -> str:
    """Prompt plus train/val/test exactly as stored (historical baseline protocol)."""
    sections = [prompt.strip()]
    for name in ("train", "val", "test"):
        path = directory / f"{name}.json"
        sections.append(f"\n\n===== {name}.json =====\n{read_text(path).strip()}")
    return "".join(sections)


_TOKENIZER: Any = None


def estimate_tokens(text: str) -> int:
    """Exact token count when the local Qwen tokenizer is importable, else a byte estimate."""
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            from transformers import AutoTokenizer

            _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
        except Exception:  # noqa: BLE001 - fall back to bytes
            _TOKENIZER = False
    if _TOKENIZER:
        return len(_TOKENIZER.encode(text, add_special_tokens=False))
    return max(1, len(text.encode("utf-8")) // 3)


CHUNK_NOTE = (
    "【分片说明】为保证上下文完整，本请求只包含待测集合的第 {index}/{total} 个分片"
    "（本片 val={n_val} 条，test={n_test} 条）。请只为本片内提供的记录输出预测；"
    "输出的val/test数组中只包含本片记录，未提供的分区用空数组[]。"
    "不要推测未提供的记录，不要输出占位ID，也不要声称已完成全部分片。"
)

RETRY_NOTE = (
    "【分片重试】上一次请求的输出被截断，本次只包含仍需补测的 {n_val} 条 val 和 "
    "{n_test} 条 test 记录。请只为本片内提供的记录输出预测；未提供的分区用空数组[]。"
    "不要推测未提供的记录，不要重复输出本片之外的记录。"
)

# A truncated chunk is retried by splitting it; stop after this many levels so a
# pathological request cannot recurse forever.
MAX_RETRY_DEPTH = 5


def _balanced_slices(records: list[dict[str, Any]], max_predictions: int) -> list[list[dict[str, Any]]]:
    """Split into ceil(n/max) near-equal slices instead of one tiny remainder slice."""
    n = len(records)
    if n == 0:
        return []
    n_chunks = max(1, -(-n // max_predictions))
    size = -(-n // n_chunks)
    return [records[i : i + size] for i in range(0, n, size)]


def plan_chunk_pairs(
    val: list[dict[str, Any]], test: list[dict[str, Any]], max_predictions: int
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """Split val/test so one call never has to emit too many predictions."""
    total = len(val) + len(test)
    if total <= max_predictions:
        return [(val, test)]
    pairs: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    pairs.extend((chunk, []) for chunk in _balanced_slices(val, max_predictions))
    pairs.extend(([], chunk) for chunk in _balanced_slices(test, max_predictions))
    return pairs


def split_to_budget(
    prompt: str,
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
    extra: str,
    fmt: str,
    input_budget: int,
    depth: int = 0,
) -> list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]]:
    """Recursively split a chunk until its serialized context fits the token budget."""
    content = build_sections(prompt, train, val, test, extra, fmt)
    tokens = estimate_tokens(content)
    if tokens <= input_budget or depth >= 6 or len(val) + len(test) <= 2:
        return [(content, val, test)]
    if len(val) >= len(test) and len(val) > 1:
        mid = len(val) // 2
        return split_to_budget(prompt, train, val[:mid], test, extra, fmt, input_budget, depth + 1) + split_to_budget(
            prompt, train, val[mid:], test, extra, fmt, input_budget, depth + 1
        )
    mid = len(test) // 2
    return split_to_budget(prompt, train, val, test[:mid], extra, fmt, input_budget, depth + 1) + split_to_budget(
        prompt, train, val, test[mid:], extra, fmt, input_budget, depth + 1
    )


def build_chunk_content(
    prompt: str,
    train: list[dict[str, Any]],
    val_chunk: list[dict[str, Any]],
    test_chunk: list[dict[str, Any]],
    extra: str,
    fmt: str,
    note: str | None = None,
) -> str:
    content = build_sections(prompt, train, val_chunk, test_chunk, extra, fmt)
    if note:
        content = content.replace(prompt.strip(), insert_before_prediction(prompt, note).strip(), 1)
    return content


def build_chunk_contents(
    prompt: str,
    work: dict[str, list[dict[str, Any]]],
    extra: str,
    fmt: str,
    max_predictions: int,
    input_budget: int,
) -> list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]]:
    base_pairs = plan_chunk_pairs(work["val"], work["test"], max_predictions)
    calls: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = []
    for val_chunk, test_chunk in base_pairs:
        calls.extend(
            split_to_budget(prompt, work["train"], val_chunk, test_chunk, extra, fmt, input_budget)
        )
    total = len(calls)
    out: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = []
    for index, (_, val_chunk, test_chunk) in enumerate(calls, start=1):
        note = CHUNK_NOTE.format(
            index=index, total=total, n_val=len(val_chunk), n_test=len(test_chunk)
        )
        out.append(
            (
                build_chunk_content(
                    prompt, work["train"], val_chunk, test_chunk, extra, fmt,
                    note if total > 1 else None,
                ),
                val_chunk,
                test_chunk,
            )
        )
    return out


def prepare_experiment(
    meta: dict[str, Any],
    data: dict[str, list[dict[str, Any]]],
    method: str,
    args: argparse.Namespace,
    shuffle_seed: int | None = None,
    prompt_path: Path | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], str, str, dict[str, Any]]:
    """Apply a variant to one experiment.  Returns data, prompt, extra, info."""
    if prompt_path is None:
        prompt_path = PACKAGE / meta["directory"] / "prompt.txt"
    prompt = read_text(prompt_path)
    prompt, prompt_note = prompt_variant(prompt, meta, method, args.include_mixed)
    info: dict[str, Any] = {"method": method, "prompt_variant": prompt_note}

    work = {split: clone_split(data[split]) for split in ("train", "val", "test")}
    extra = ""
    if method == "p1":
        work, lda_info = fit_target_lda(work)
        info.update(lda_info)
    elif method == "p2":
        work = zscore_by_day(work)
        extra = prototype_blocks(work, args.hint, args.decimals)
        info["hint"] = args.hint
        info["zscore"] = "per_day_unlabeled"
    elif method == "public":
        dataset = str(meta.get("dataset_id", ""))
        info["dataset_id"] = dataset
        info["task_type"] = "ordinal" if dataset in ORDINAL_LABEL_ORDER else "nominal"
        info["decimals"] = int(meta.get("feature_decimals") or args.decimals)

    if args.format == "grouped":
        shuffle = shuffle_seed is not None
        work["train"] = group_train(work["train"], shuffle=shuffle, seed=shuffle_seed or 0)
        decimals = int(info.get("decimals", args.decimals))
        for split in ("train", "val", "test"):
            work[split] = round_features(work[split], decimals)
    elif shuffle_seed is not None:
        work["train"] = shuffle_records(work["train"], shuffle_seed)
    return work, prompt, extra, info


# ---------------------------------------------------------------------------
# API / scoring
# ---------------------------------------------------------------------------
class StructuredOutputFailed(RuntimeError):
    """The server's response_format validator rejected the output (e.g. truncation)."""


class FatalAPIError(RuntimeError):
    """A non-retryable client error (4xx other than timeout / rate limit)."""


def get_model_id(api_base: str) -> str:
    try:
        with urllib.request.urlopen(api_base.rstrip("/") + "/models", timeout=10) as resp:
            obj = json.load(resp)
        data = obj.get("data") or []
        if data and data[0].get("id"):
            return str(data[0]["id"])
    except Exception:
        pass
    return DEFAULT_MODEL


def api_chat(
    content: str, args: argparse.Namespace, model_id: str, thinking: bool | None = None
) -> dict[str, Any]:
    enable_thinking = args.thinking if thinking is None else thinking
    response_format = getattr(args, "response_format", "none")

    def make_request() -> urllib.request.Request:
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": content}],
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if response_format == "json_object":
            # Strata validates the final answer against {"type": "object"} and
            # rejects truncated/invalid output instead of returning it.
            payload["response_format"] = {"type": "json_object"}
        return urllib.request.Request(
            args.api_base.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(make_request(), timeout=args.timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", "replace")
            except Exception:  # noqa: BLE001 - body is best-effort context
                body = ""
            if "structured_output_failed" in body or "structured output" in body:
                # Deterministic model failure; retrying the same prompt wastes calls.
                raise StructuredOutputFailed(body[:800]) from None
            if 400 <= exc.code < 500 and exc.code not in (408, 429):
                raise FatalAPIError(f"HTTP {exc.code}: {body[:400]}") from None
            last_exc = exc
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
        except Exception as exc:  # noqa: BLE001 - retry transport/server errors
            last_exc = exc
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"API failed after retries: {type(last_exc).__name__}: {last_exc}")


# ---------------------------------------------------------------------------
# chunk checkpoint + truncation recovery
# ---------------------------------------------------------------------------
def load_chunk_cache(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return obj if isinstance(obj, dict) else {}


def save_chunk_cache(path: Path | None, cache: dict[str, Any]) -> None:
    if path is None:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _seen_ids(parsed: Any) -> dict[str, set[str]]:
    seen: dict[str, set[str]] = {"val": set(), "test": set()}
    if isinstance(parsed, dict):
        for split in ("val", "test"):
            for item in parsed.get(split) or []:
                if isinstance(item, dict) and "id" in item:
                    seen[split].add(str(item["id"]))
    return seen


def _missing_records(
    val_chunk: list[dict[str, Any]], test_chunk: list[dict[str, Any]], parsed: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen = _seen_ids(parsed)
    return (
        [r for r in val_chunk if str(r.get("id")) not in seen["val"]],
        [r for r in test_chunk if str(r.get("id")) not in seen["test"]],
    )


def _split_chunk(
    val_chunk: list[dict[str, Any]], test_chunk: list[dict[str, Any]]
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """Halve the larger side; at 1-val/1-test split into single-record calls."""
    if len(val_chunk) >= len(test_chunk) and len(val_chunk) > 1:
        mid = len(val_chunk) // 2
        return [(val_chunk[:mid], test_chunk), (val_chunk[mid:], test_chunk)]
    if len(test_chunk) > 1:
        mid = len(test_chunk) // 2
        return [(val_chunk, test_chunk[:mid]), (val_chunk, test_chunk[mid:])]
    if len(val_chunk) + len(test_chunk) > 1:
        return [(val_chunk, []), ([], test_chunk)]
    return []


def run_call_tree(
    content: str,
    prompt: str,
    train: list[dict[str, Any]],
    val_chunk: list[dict[str, Any]],
    test_chunk: list[dict[str, Any]],
    extra: str,
    fmt: str,
    args: argparse.Namespace,
    model_id: str,
    thinking: bool,
    depth: int,
    cache: dict[str, Any],
    cache_path: Path | None,
    stats: dict[str, Any],
) -> list[dict[str, Any] | None]:
    """Call one chunk; on truncation, retry missing records in smaller chunks.

    Parsed results are cached by content hash, so ``--resume`` reuses every
    chunk (including retry sub-chunks) that already succeeded.
    """
    key = hashlib.sha256(content.encode("utf-8")).hexdigest()[:24]
    parsed: Any = None
    finish: str | None = None

    if key in cache:
        entry = cache[key]
        parsed = entry.get("parsed")
        finish = entry.get("finish_reason")
        stats["checkpoint_hits"] += 1
        stats["input_tokens"] += int(entry.get("prompt_tokens") or 0)
        stats["completion_tokens"] += int(entry.get("completion_tokens") or 0)
    else:
        try:
            response = api_chat(content, args, model_id, thinking=thinking)
        except StructuredOutputFailed as exc:
            stats["calls"] += 1
            message = str(exc)
            if "incomplete" in message or "finish_reason=length" in message:
                stats["truncated"] += 1
                finish = "length"
                cache[key] = {
                    "parsed": None,
                    "finish_reason": "length",
                    "error": message[:400],
                }
                save_chunk_cache(cache_path, cache)
            else:
                stats["chunk_errors"].append(f"StructuredOutputFailed: {message[:300]}")
                return []
        except Exception as exc:  # noqa: BLE001 - transport / fatal client errors
            stats["chunk_errors"].append(f"{type(exc).__name__}: {exc}")
            return []
        else:
            stats["calls"] += 1
            choice = (response.get("choices") or [{}])[0]
            message_obj = choice.get("message") or {}
            raw = message_obj.get("content") or ""
            reasoning = message_obj.get("reasoning_content") or ""
            parsed = parse_model_json(raw)
            if parsed is None and reasoning:
                parsed = parse_model_json(reasoning)
            finish = choice.get("finish_reason")
            usage = response.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            stats["input_tokens"] += int(prompt_tokens or 0)
            stats["completion_tokens"] += int(completion_tokens or 0)
            if finish == "length":
                stats["truncated"] += 1
            cache[key] = {
                "parsed": parsed,
                "finish_reason": finish,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
            save_chunk_cache(cache_path, cache)

    results: list[dict[str, Any] | None] = [parsed] if isinstance(parsed, dict) else []
    missing_val, missing_test = _missing_records(val_chunk, test_chunk, parsed)
    if not missing_val and not missing_test:
        return results
    if finish != "length" or depth >= MAX_RETRY_DEPTH:
        return results

    # Partial JSON: retry all missing records in one call when the model got
    # through most of the chunk; split when it produced less than half.
    # Unparseable JSON: halve the chunk so the output fits next time.
    if isinstance(parsed, dict):
        missing_n = len(missing_val) + len(missing_test)
        total_n = len(val_chunk) + len(test_chunk)
        if missing_n * 2 > total_n:
            sub_chunks = _split_chunk(missing_val, missing_test) or [(missing_val, missing_test)]
        else:
            sub_chunks = [(missing_val, missing_test)]
    else:
        sub_chunks = _split_chunk(missing_val, missing_test)
        if not sub_chunks:
            # A single record still truncated: a deterministic retry cannot help.
            return results

    stats["retried"] += len(sub_chunks)
    for sub_val, sub_test in sub_chunks:
        if not sub_val and not sub_test:
            continue
        note = RETRY_NOTE.format(n_val=len(sub_val), n_test=len(sub_test))
        sub_content = build_chunk_content(prompt, train, sub_val, sub_test, extra, fmt, note)
        results.extend(
            run_call_tree(
                sub_content,
                prompt,
                train,
                sub_val,
                sub_test,
                extra,
                fmt,
                args,
                model_id,
                thinking,
                depth + 1,
                cache,
                cache_path,
                stats,
            )
        )
    return results


def evaluate(directory: Path, parsed: dict[str, Any] | None, meta: dict[str, Any]) -> dict[str, Any]:
    classes = meta.get("candidate_labels") or []
    result: dict[str, Any] = {}
    for split in ("val", "test"):
        preds = parsed.get(split) if isinstance(parsed, dict) else None
        result[f"{split}_metrics"] = score_split(
            preds if isinstance(preds, list) else [],
            directory / f"{split}_labels.json",
            classes,
        )
    return result


def merge_chunk_predictions(
    parsed_list: list[dict[str, Any] | None], experiment_id: str
) -> dict[str, Any]:
    merged: dict[str, Any] = {"experiment_id": experiment_id, "val": [], "test": []}
    seen: dict[str, set[str]] = {"val": set(), "test": set()}
    for parsed in parsed_list:
        if not isinstance(parsed, dict):
            continue
        for split in ("val", "test"):
            items = parsed.get(split)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or "id" not in item or "label" not in item:
                    continue
                sample_id = str(item["id"])
                if sample_id in seen[split]:
                    continue
                seen[split].add(sample_id)
                merged[split].append(item)
    return merged


def run_one(
    row: dict[str, str],
    args: argparse.Namespace,
    phase: str,
    model_id: str,
    repeat_index: int = 0,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, list[dict[str, Any]]] | None]:
    directory = PACKAGE / row["directory"]
    meta, data = load_experiment(directory)
    method = method_for(row, phase, args)
    seed = None
    if phase == "P3":
        seed = int(hashlib.sha256(f"{row['experiment_id']}:{repeat_index}".encode()).hexdigest()[:16], 16)
    work, prompt, extra, info = prepare_experiment(
        meta, data, method, args, seed, prompt_path=directory / "prompt.txt"
    )

    dataset_id = row.get("dataset_id", "")
    use_thinking = args.thinking or dataset_id in args.thinking_datasets
    record: dict[str, Any] = {
        "experiment_id": row["experiment_id"],
        "batch": row["batch"],
        "task": row["task"],
        "dataset_id": dataset_id,
        "directory": row["directory"],
        "reference_mode": row.get("reference_mode", ""),
        "control": row.get("control", ""),
        "target_k": row.get("target_k", ""),
        "k": row.get("k", ""),
        "variant": args.variant,
        "phase": phase,
        "method": method,
        "prompt_variant": info.get("prompt_variant", ""),
        "thinking": use_thinking,
        "repeat_index": repeat_index,
    }
    for key in (
        "transform",
        "n_components",
        "task_type",
        "refs_available",
        "refs_used",
        "decimals",
    ):
        if key in info:
            record[key] = info[key]

    # Public datasets may need chunked context or a compressed transform.
    if method == "public":
        input_budget = args.n_ctx - args.max_tokens - 512
        calls = build_chunk_contents(
            prompt,
            work,
            extra,
            args.format,
            args.max_predictions_per_call,
            input_budget,
        )
        estimated = [estimate_tokens(content) for content, _, _ in calls]
        record.update(
            n_chunks=len(calls),
            input_bytes=sum(len(content.encode("utf-8")) for content, _, _ in calls),
            estimated_input_tokens_max=max(estimated) if estimated else 0,
        )
        if any(tokens > input_budget for tokens in estimated):
            record.update(
                status="context_skipped",
                reason=f"chunk_tokens={max(estimated)} > budget={input_budget}",
            )
            return record, None, None

        checkpoint_dir = getattr(args, "checkpoint_dir", None)
        cache_path = (
            Path(checkpoint_dir) / f"{row['experiment_id']}__r{repeat_index}.json"
            if checkpoint_dir is not None
            else None
        )
        cache = load_chunk_cache(cache_path)
        stats: dict[str, Any] = {
            "calls": 0,
            "input_tokens": 0,
            "completion_tokens": 0,
            "truncated": 0,
            "retried": 0,
            "checkpoint_hits": 0,
            "chunk_errors": [],
        }

        started = time.time()
        parsed_list: list[dict[str, Any] | None] = []
        top_level_parsed = 0
        for content, val_chunk, test_chunk in calls:
            results = run_call_tree(
                content,
                prompt,
                work["train"],
                val_chunk,
                test_chunk,
                extra,
                args.format,
                args,
                model_id,
                use_thinking,
                0,
                cache,
                cache_path,
                stats,
            )
            if any(isinstance(parsed, dict) for parsed in results):
                top_level_parsed += 1
            parsed_list.extend(results)
        elapsed = time.time() - started
        merged = merge_chunk_predictions(parsed_list, row["experiment_id"])
        chunk_errors = stats["chunk_errors"]
        if top_level_parsed == 0:
            status = "error" if chunk_errors else "parse_error"
        else:
            status = "completed"
        expected_total = len(work["val"]) + len(work["test"])
        got_total = len(merged["val"]) + len(merged["test"])
        record.update(
            status=status,
            input_tokens=stats["input_tokens"] or None,
            completion_tokens=stats["completion_tokens"] or None,
            elapsed_seconds=round(elapsed, 2),
            chunk_parsed=top_level_parsed,
            chunk_calls=stats["calls"],
            chunk_truncated=stats["truncated"],
            chunk_retried=stats["retried"],
            chunk_checkpoint_hits=stats["checkpoint_hits"],
            chunk_errors=chunk_errors,
            missing_predictions=max(0, expected_total - got_total),
            response=merged,
            experiment_id_matches=(
                any(
                    isinstance(parsed, dict)
                    and parsed.get("experiment_id") == row["experiment_id"]
                    for parsed in parsed_list
                )
                if top_level_parsed
                else False
            ),
        )
        if status == "completed":
            record.update(evaluate(directory, merged, meta))
        return record, merged if status == "completed" else None, data

    if phase == "ALL":
        content = build_content_raw(prompt, directory)
    else:
        content = build_content(prompt, work, extra, args.format)
    record["input_bytes"] = len(content.encode("utf-8"))

    if record["input_bytes"] > args.max_bytes:
        record.update(status="context_skipped", reason=f"input_bytes={record['input_bytes']} > {args.max_bytes}")
        return record, None, None

    started = time.time()
    try:
        response = api_chat(content, args, model_id, thinking=use_thinking)
    except Exception as exc:  # noqa: BLE001
        record.update(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=round(time.time() - started, 2),
        )
        return record, None, None

    elapsed = time.time() - started
    choice = response["choices"][0]
    message = choice["message"]
    raw = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    parsed = parse_model_json(raw)
    if parsed is None and reasoning:
        parsed = parse_model_json(reasoning)
    usage = response.get("usage") or {}
    record.update(
        status="completed" if parsed is not None else "parse_error",
        finish_reason=choice.get("finish_reason"),
        input_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        elapsed_seconds=round(elapsed, 2),
        response=parsed,
        raw_response=raw,
        reasoning_chars=len(reasoning),
        experiment_id_matches=(
            isinstance(parsed, dict) and parsed.get("experiment_id") == row["experiment_id"]
        ),
    )
    record.update(evaluate(directory, parsed, meta))
    return record, parsed, data


# ---------------------------------------------------------------------------
# self-consistency voting
# ---------------------------------------------------------------------------
def vote_predictions(
    parsed_list: list[dict[str, Any] | None],
    fallback_order: dict[str, list[str]],
) -> dict[str, Any]:
    out: dict[str, Any] = {"val": [], "test": []}
    experiment_id = None
    for parsed in parsed_list:
        if isinstance(parsed, dict) and parsed.get("experiment_id"):
            experiment_id = parsed["experiment_id"]
            break
    for split in ("val", "test"):
        votes: dict[str, Counter] = defaultdict(Counter)
        first: dict[str, str] = {}
        order: list[str] = list(fallback_order.get(split, []))
        for parsed in parsed_list:
            if not isinstance(parsed, dict):
                continue
            for item in parsed.get(split) or []:
                if not isinstance(item, dict) or "id" not in item:
                    continue
                sid = str(item["id"])
                label = str(item.get("label"))
                if sid not in votes and sid not in order:
                    order.append(sid)
                if sid not in first:
                    first[sid] = label
                votes[sid][label] += 1
        for sid in order:
            if sid not in votes:
                continue
            counts = votes[sid]
            label = max(
                counts.items(),
                key=lambda kv: (kv[1], kv[0] == first.get(sid), -_num(kv[0])),
            )[0]
            out[split].append({"id": sid, "label": label})
    if experiment_id:
        out["experiment_id"] = experiment_id
    return out


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    out: dict[str, Any] = {}
    for split in ("val", "test"):
        correct = sum(int(r[f"{split}_metrics"]["correct"]) for r in records)
        total = sum(int(r[f"{split}_metrics"]["total"]) for r in records)
        f1 = [float(r[f"{split}_metrics"]["macro_f1"]) for r in records]
        out[split] = {
            "n_experiments": len(records),
            "n_samples": total,
            "accuracy_pct": round(100 * correct / total, 2) if total else 0.0,
            "macro_f1_mean": round(float(np.mean(f1)), 4) if f1 else 0.0,
        }
    return out


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [r for r in records if r.get("status") == "completed"]
    groups: dict[str, list[dict[str, Any]]] = {"overall": completed}
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in completed:
        task = rec.get("task") or "?"
        by_task[task].append(rec)
        dataset = str(rec.get("dataset_id") or "")
        if dataset:
            by_dataset[dataset].append(rec)
        if task == "T3":
            cond = f"{rec.get('control','?')}_k{rec.get('target_k','?')}"
        elif task == "PUBLIC":
            cond = str(rec.get("reference_mode") or "?")
            if rec.get("k"):
                cond += f"_k{rec['k']}"
        else:
            cond = str(rec.get("reference_mode") or "?")
        by_condition[f"{task}|{cond}"].append(rec)
    groups.update({f"task:{k}": v for k, v in sorted(by_task.items())})
    groups.update({f"cond:{k}": v for k, v in sorted(by_condition.items())})
    groups.update({f"dataset:{k}": v for k, v in sorted(by_dataset.items())})
    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_records": len(records),
        "n_completed": len(completed),
        "n_error": sum(1 for r in records if r.get("status") == "error"),
        "n_parse_error": sum(1 for r in records if r.get("status") == "parse_error"),
        "n_skipped": sum(1 for r in records if r.get("status") == "context_skipped"),
        "groups": {name: _aggregate(rows) for name, rows in groups.items()},
    }
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("\n%-34s %5s %9s %9s %9s %9s" % ("group", "n", "val_acc", "val_F1", "test_acc", "test_F1"))
    for name, stats in summary["groups"].items():
        val = stats.get("val")
        test = stats.get("test")
        if not val or not test:
            print("%-34s %5d %9s %9s %9s %9s" % (name, 0, "-", "-", "-", "-"))
            continue
        print(
            "%-34s %5d %8.2f%% %9.4f %8.2f%% %9.4f"
            % (
                name,
                val["n_experiments"],
                val["accuracy_pct"],
                val["macro_f1_mean"],
                test["accuracy_pct"],
                test["macro_f1_mean"],
            )
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", required=True, choices=["ALL", "P0", "P1", "P2", "P3", "P4"])
    parser.add_argument("--variant", default="", help="Output subdirectory; defaults to a phase-derived name.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--api-base", default=DEFAULT_API)
    parser.add_argument("--model", default="", help="Model id; auto-detected from /v1/models when empty.")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-bytes", type=int, default=600_000)
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("--format", choices=["grouped", "plain"], default="grouped")
    parser.add_argument("--hint", choices=["both", "centroids", "nearest", "none"], default="both")
    parser.add_argument("--include-mixed", action="store_true", help="P1: also run mixed target_k=3,5.")
    parser.add_argument("--p3-base", choices=["best", "p0", "p1", "p2"], default="best")
    parser.add_argument("--repeats", type=int, default=3, help="P3 repeats per experiment.")
    parser.add_argument("--datasets", default="", help="Comma-separated dataset_id filter (P4 pilots).")
    parser.add_argument(
        "--max-predictions-per-call",
        type=int,
        default=400,
        help="Chunked public inference: max val+test predictions per API call.",
    )
    parser.add_argument("--n-ctx", type=int, default=65536, help="Server context window for token preflight.")
    parser.add_argument(
        "--thinking-datasets",
        default="",
        help="Comma-separated dataset_ids that always use thinking (e.g. turmeric_levels,beef).",
    )
    parser.add_argument(
        "--response-format",
        choices=["none", "json_object"],
        default="none",
        help="Ask the server to validate one JSON object. Strata rejects truncated output "
        "(finish_reason=length) in this mode; 'none' keeps the partial-output retry path.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.temperature is None:
        args.temperature = 0.2 if args.phase == "P3" else 0.0
    args.thinking_datasets = {
        item.strip() for item in args.thinking_datasets.split(",") if item.strip()
    }
    if not args.variant:
        if args.phase == "P4":
            args.variant = "p4_public"
        else:
            args.variant = args.phase.lower() + ("_best" if args.phase == "P3" else "_default")
    out_dir = args.output_root.resolve() / args.variant
    result_path = out_dir / "predictions.jsonl"
    repeats_path = out_dir / "predictions_repeats.jsonl"
    skipped_path = out_dir / "context_skipped.json"
    summary_path = out_dir / "summary.json"
    meta_path = out_dir / "meta.json"

    rows = select_experiments(args.phase, args)
    if args.datasets:
        wanted = {item.strip() for item in args.datasets.split(",") if item.strip()}
        rows = [row for row in rows if row.get("dataset_id") in wanted]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No experiments selected.")

    print(
        f"Phase {args.phase} variant={args.variant}: {len(rows)} experiments | "
        f"thinking={args.thinking} temperature={args.temperature} format={args.format} "
        f"hint={args.hint} concurrency={args.concurrency}",
        flush=True,
    )
    by_task = Counter(r["task"] for r in rows)
    print("  by task:", dict(by_task), flush=True)
    by_dataset = Counter(r.get("dataset_id", "") for r in rows)
    print("  by dataset:", dict(sorted(by_dataset.items())), flush=True)
    print(f"  thinking_datasets={sorted(args.thinking_datasets)}", flush=True)
    print(f"  response_format={args.response_format}", flush=True)
    if args.dry_run:
        for row in rows:
            print(f"  {row['experiment_id']}  {row['directory']}", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "chunk_checkpoints"
    args.checkpoint_dir = checkpoint_dir
    if not args.resume and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_id = args.model or get_model_id(args.api_base)
    config = {
        "variant": args.variant,
        "phase": args.phase,
        "created": datetime.now(timezone.utc).isoformat(),
        "api_base": args.api_base,
        "model": model_id,
        "thinking": args.thinking,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "decimals": args.decimals,
        "format": args.format,
        "hint": args.hint,
        "p3_base": args.p3_base,
        "repeats": args.repeats if args.phase == "P3" else 1,
        "max_bytes": args.max_bytes,
        "max_predictions_per_call": args.max_predictions_per_call,
        "n_ctx": args.n_ctx,
        "thinking_datasets": sorted(args.thinking_datasets),
        "datasets_filter": args.datasets,
        "response_format": args.response_format,
        "n_selected": len(rows),
    }
    meta_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    done: set[str] = set()
    if args.resume and result_path.exists():
        for line in result_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") == "completed":
                done.add(rec["experiment_id"])
        rows = [r for r in rows if r["experiment_id"] not in done]
        print(f"Resume: {len(done)} completed experiments already present; {len(rows)} remaining.", flush=True)
    elif result_path.exists():
        result_path.unlink()
    if args.phase == "P3" and repeats_path.exists() and not args.resume:
        repeats_path.unlink()

    if not rows:
        print("Nothing to do.")
        return

    lock = threading.Lock()
    if not args.resume:
        result_path.write_text("", encoding="utf-8")
        if args.phase == "P3":
            repeats_path.write_text("", encoding="utf-8")

    def emit(path: Path, record: dict[str, Any]) -> None:
        with lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.phase != "P3":
        total = len(rows)
        done_count = 0
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = {
                pool.submit(run_one, row, args, args.phase, model_id): row
                for row in rows
            }
            for future in as_completed(futures):
                row = futures[future]
                done_count += 1
                try:
                    record, _, _ = future.result()
                except Exception as exc:  # noqa: BLE001
                    record = {
                        "experiment_id": row["experiment_id"],
                        "batch": row["batch"],
                        "task": row["task"],
                        "directory": row["directory"],
                        "variant": args.variant,
                        "phase": args.phase,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                emit(result_path, record)
                print(
                    f"  [{done_count}/{total}] {record['experiment_id']} "
                    f"status={record.get('status')} elapsed={record.get('elapsed_seconds')}s "
                    f"in={record.get('input_tokens')} out={record.get('completion_tokens')}",
                    flush=True,
                )
        all_records = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        summary = summarize(all_records)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print_summary(summary)
        print(f"\nPredictions: {result_path}\nSummary: {summary_path}", flush=True)
        return

    # P3: repeats + majority vote
    tasks = [(row, rep) for row in rows for rep in range(max(1, args.repeats))]
    per_experiment: dict[str, list[dict[str, Any] | None]] = defaultdict(list)
    fallback_order: dict[str, dict[str, list[str]]] = {}
    records_by_exp: dict[str, dict[str, str]] = {row["experiment_id"]: row for row in rows}
    total = len(tasks)
    done_count = 0
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = {
            pool.submit(run_one, row, args, "P3", model_id, rep): (row, rep)
            for row, rep in tasks
        }
        for future in as_completed(futures):
            row, rep = futures[future]
            done_count += 1
            try:
                record, parsed, data = future.result()
            except Exception as exc:  # noqa: BLE001
                record = {
                    "experiment_id": row["experiment_id"],
                    "batch": row["batch"],
                    "task": row["task"],
                    "directory": row["directory"],
                    "variant": args.variant,
                    "phase": "P3",
                    "repeat_index": rep,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                parsed, data = None, None
            emit(repeats_path, record)
            per_experiment[row["experiment_id"]].append(parsed)
            if data is not None:
                fallback_order[row["experiment_id"]] = {
                    split: [str(r["id"]) for r in data[split]] for split in ("val", "test")
                }
            print(
                f"  [{done_count}/{total}] {record['experiment_id']} rep={rep} "
                f"status={record.get('status')} elapsed={record.get('elapsed_seconds')}s "
                f"in={record.get('input_tokens')} out={record.get('completion_tokens')}",
                flush=True,
            )

    # Vote and score each experiment.
    for exp_id, parsed_list in per_experiment.items():
        row = records_by_exp[exp_id]
        directory = PACKAGE / row["directory"]
        meta = json.loads(read_text(directory / "metadata.json"))
        voted = vote_predictions(parsed_list, fallback_order.get(exp_id, {}))
        metrics = evaluate(directory, voted, meta)
        repeat_metrics = []
        for parsed in parsed_list:
            repeat_metrics.append(evaluate(directory, parsed, meta))
        votes: dict[str, dict[str, dict[str, int]]] = {}
        for split in ("val", "test"):
            votes[split] = {}
            for parsed in parsed_list:
                if not isinstance(parsed, dict):
                    continue
                for item in parsed.get(split) or []:
                    if isinstance(item, dict) and "id" in item:
                        sid = str(item["id"])
                        votes[split].setdefault(sid, {})
                        label = str(item.get("label"))
                        votes[split][sid][label] = votes[split][sid].get(label, 0) + 1
        record = {
            "experiment_id": exp_id,
            "batch": row["batch"],
            "task": row["task"],
            "directory": row["directory"],
            "reference_mode": row.get("reference_mode", ""),
            "control": row.get("control", ""),
            "target_k": row.get("target_k", ""),
            "variant": args.variant,
            "phase": "P3",
            "method": method_for(row, "P3", args),
            "status": "completed" if voted.get("val") or voted.get("test") else "parse_error",
            "repeats": len(parsed_list),
            "response": voted,
            "repeat_metrics": repeat_metrics,
            "vote_counts": votes,
        }
        record.update(metrics)
        emit(result_path, record)

    all_records = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = summarize(all_records)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_summary(summary)
    print(f"\nVoted predictions: {result_path}\nRaw repeats: {repeats_path}\nSummary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
