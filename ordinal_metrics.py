#!/usr/bin/env python3
"""Recompute ordinal metrics from saved predictions (no model calls).

For every completed experiment whose dataset is a known multi-level ordinal
task, join the saved ``response.test`` predictions with the ground-truth
``test_labels.json`` and report:

  - exact accuracy
  - adjacent accuracy  (|rank error| <= 1)
  - MAE in ordinal rank units
  - fraction with rank error >= 2
  - quadratic weighted kappa (QWK)

The ordinal rank order is taken from ``metadata.json``'s candidate_labels,
sorted numerically, so no hard-coded label order is needed.

Only datasets listed in ORDINAL_DATASETS are evaluated.  Numeric-coded but
nominal datasets (e.g. beef UCR codes) are intentionally excluded.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data_json" / "four_batches"
OUT = ROOT / "outputs" / "four_batches_strata_variants"

# Datasets whose labels form a meaningful ordered scale.
ORDINAL_DATASETS = {"surimi9", "turmeric_levels"}

SOURCES = [
    ("P0", ROOT / "outputs" / "four_batches_strata" / "predictions.jsonl"),
    ("P4", OUT / "p4_public" / "predictions.jsonl"),
    ("P1", OUT / "p1_default" / "predictions.jsonl"),
    ("P2", OUT / "p2_default" / "predictions.jsonl"),
    ("P3", OUT / "p3_best" / "predictions.jsonl"),
    ("TH", OUT / "p4_weak_thinking" / "predictions.jsonl"),
]


def load_completed(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("status") == "completed":
            out.append(rec)
    return out


def mode_of(directory: str) -> str:
    tail = directory.split("/")[-1]
    return "full" if tail == "full_reference" else tail


def ordinal_order(candidate_labels: list[str]) -> list[str] | None:
    try:
        values = {lab: float(lab) for lab in candidate_labels}
    except (TypeError, ValueError):
        return None
    if len(values) < 3:
        return None
    return sorted(values, key=lambda lab: values[lab])


def qwk(pairs: list[tuple[int, int]], k: int) -> float | None:
    n = len(pairs)
    if n == 0:
        return None
    obs = [[0] * k for _ in range(k)]
    for truth, pred in pairs:
        obs[truth][pred] += 1
    hist_t = [0] * k
    hist_p = [0] * k
    for truth, pred in pairs:
        hist_t[truth] += 1
        hist_p[pred] += 1
    expected = [[hist_t[i] * hist_p[j] / n for j in range(k)] for i in range(k)]
    num = sum((i - j) ** 2 * obs[i][j] for i in range(k) for j in range(k))
    den = sum((i - j) ** 2 * expected[i][j] for i in range(k) for j in range(k))
    return None if den == 0 else 1.0 - num / den


def metrics(pairs: list[tuple[int, int]], k: int) -> dict | None:
    n = len(pairs)
    if n == 0:
        return None
    return {
        "n": n,
        "exact_pct": round(100 * sum(t == p for t, p in pairs) / n, 2),
        "adjacent_pct": round(100 * sum(abs(t - p) <= 1 for t, p in pairs) / n, 2),
        "mae": round(sum(abs(t - p) for t, p in pairs) / n, 3),
        "far_pct": round(100 * sum(abs(t - p) >= 2 for t, p in pairs) / n, 2),
        "qwk": None if (q := qwk(pairs, k)) is None else round(q, 4),
    }


def group_tags(meta: dict, directory: str) -> dict:
    task = meta.get("task") or "PUBLIC"
    tags = {
        "task": str(task),
        "dataset_id": meta.get("dataset_id", "?"),
        "mode": mode_of(directory),
    }
    if task == "T1":
        tags["cond"] = f"day{meta.get('target_day')}"
    elif task == "T2":
        tags["cond"] = f"day{meta.get('source_day')}_to_day{meta.get('target_day')}"
    elif task == "T3":
        tags["cond"] = f"{meta.get('control')}_k{meta.get('target_k')}"
    elif task == "UNSEEN":
        tags["cond"] = f"day{meta.get('target_day')}"
    else:
        tags["cond"] = "public"
    return tags


def collect() -> tuple[list[dict], dict[str, list[dict]]]:
    rows: list[dict] = []
    by_eid: dict[str, list[dict]] = defaultdict(list)
    for source, path in SOURCES:
        for rec in load_completed(path):
            directory = rec.get("directory", "")
            meta_path = DATA / directory / "metadata.json"
            labels_path = DATA / directory / "test_labels.json"
            if not meta_path.exists() or not labels_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            dataset = meta.get("dataset_id", "")
            if dataset not in ORDINAL_DATASETS:
                continue
            order = ordinal_order(meta.get("candidate_labels") or [])
            if not order:
                continue
            rank = {lab: i for i, lab in enumerate(order)}
            k = len(order)

            gold = {x["id"]: str(x["label"]) for x in json.loads(labels_path.read_text(encoding="utf-8"))}
            response = rec.get("response") or {}
            pred_rows = response.get("test") or []
            pred = {str(x.get("id")): str(x.get("label", "")).strip() for x in pred_rows}

            pairs: list[tuple[int, int]] = []
            missing = invalid = 0
            for gid, gl in gold.items():
                if gid not in pred:
                    missing += 1
                    continue
                pl = pred[gid]
                if gl not in rank or pl not in rank:
                    invalid += 1
                    continue
                pairs.append((rank[gl], rank[pl]))

            m = metrics(pairs, k)
            if m is None:
                continue
            m.update(group_tags(meta, directory))
            m.update(
                {
                    "source": source,
                    "experiment_id": rec["experiment_id"],
                    "directory": directory,
                    "missing": missing,
                    "invalid": invalid,
                    "n_test": len(gold),
                    "k_levels": k,
                }
            )
            rows.append(m)
            by_eid[rec["experiment_id"]].append(m)
    return rows, by_eid


def pooled(rows: list[dict]) -> dict:
    n = sum(r["n"] for r in rows)
    if n == 0:
        return {"n_exp": len(rows), "n": 0}
    # Reconstruct weighted aggregates from the stored per-experiment metrics.
    def wavg(key: str) -> float:
        return round(sum(r[key] * r["n"] for r in rows) / n, 2)

    qn = sum(1 for r in rows if r["qwk"] is not None)

    def qavg() -> float | None:
        if not qn:
            return None
        return round(sum(r["qwk"] for r in rows if r["qwk"] is not None) / qn, 4)

    return {
        "n_exp": len(rows),
        "n": n,
        "exact_pct": wavg("exact_pct"),
        "adjacent_pct": wavg("adjacent_pct"),
        "mae": round(sum(r["mae"] * r["n"] for r in rows) / n, 3),
        "far_pct": wavg("far_pct"),
        "qwk": qavg(),
        "missing": sum(r["missing"] for r in rows),
        "invalid": sum(r["invalid"] for r in rows),
    }


def fmt(v, width: int = 6, decimals: int = 2) -> str:
    if v is None:
        return " " * (width - 1) + "-"
    if isinstance(v, float):
        return f"{v:{width}.{decimals}f}"
    return f"{v:>{width}d}"


def print_table(title: str, groups: dict[tuple, list[dict]]) -> None:
    print(f"\n== {title} ==")
    names = {key: " | ".join(str(x) for x in key) for key in sorted(groups)}
    width = max([len("group"), *(len(n) for n in names.values())])
    print(f"{'group':{width}s} {'n_exp':>5s} {'n':>6s} {'exact':>7s} {'adj@1':>7s} {'MAE':>6s} {'far>=2':>7s} {'QWK':>7s}")
    for key in sorted(groups):
        rows = groups[key]
        if not rows:
            continue
        s = pooled(rows)
        print(
            f"{names[key]:{width}s} {s['n_exp']:5d} {s['n']:6d} {fmt(s['exact_pct'], 7)} "
            f"{fmt(s['adjacent_pct'], 7)} {fmt(s['mae'], 6)} {fmt(s['far_pct'], 7)} {fmt(s['qwk'], 7, 4)}"
        )


def grouped(rows: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def main() -> None:
    rows, by_eid = collect()
    if not rows:
        raise SystemExit("No ordinal experiments with saved predictions found.")

    # 1. Overall per dataset / task, P0 baseline only (variant subsets would
    #    otherwise over-weight the matched experiments).
    p0_rows = [r for r in rows if r["source"] == "P0"]
    print_table("P0 baseline ordinal metrics by dataset + task", grouped(p0_rows, "dataset_id", "task"))

    # 2. surimi9 detail.
    surimi = [r for r in rows if r["dataset_id"] == "surimi9" and r["source"] == "P0"]
    for task, title in [
        ("T1", "surimi9 T1 (same day): day x mode"),
        ("T2", "surimi9 T2 (cross day): pair x mode"),
        ("T3", "surimi9 T3 (target k-shot): control_k x mode"),
        ("UNSEEN", "surimi9 UNSEEN: day x mode"),
    ]:
        subset = [r for r in surimi if r["task"] == task]
        if subset:
            print_table(title, grouped(subset, "cond", "mode"))

    # 3. turmeric_levels public: P0 vs P4 (and any other source).
    turmeric = [r for r in rows if r["dataset_id"] == "turmeric_levels"]
    if turmeric:
        print_table("turmeric_levels: source x mode", grouped(turmeric, "source", "mode"))

    # 4. Matched variant comparisons against P0.
    p0_eids = {r["experiment_id"] for r in rows if r["source"] == "P0"}
    for source in ("P1", "P2", "P3", "TH"):
        matched = [
            r for r in rows if r["source"] == source and r["experiment_id"] in p0_eids
        ]
        if not matched:
            continue
        eids = {r["experiment_id"] for r in matched}
        base = [r for r in rows if r["source"] == "P0" and r["experiment_id"] in eids]
        combined = [{**r, "source": "P0"} for r in base] + [{**r, "source": source} for r in matched]
        print_table(
            f"matched P0 vs {source} (same experiment ids)",
            grouped(combined, "dataset_id", "task", "cond", "mode", "source"),
        )

    payload = {
        "ordinal_datasets": sorted(ORDINAL_DATASETS),
        "n_experiments": len(rows),
        "per_experiment": rows,
        "overall_p0": {
            f"{ds}|{task}": pooled(rs)
            for (ds, task), rs in grouped(p0_rows, "dataset_id", "task").items()
        },
    }
    out_path = OUT / "ordinal_metrics.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path} ({len(rows)} experiments)")


if __name__ == "__main__":
    main()
