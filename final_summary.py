#!/usr/bin/env python3
"""Consolidate every completed experiment into one offline report.

No model calls.  Reads the saved predictions for:

  - P0 baseline (all four batches, outputs/four_batches_strata)
  - P4 public prompt variant
  - thinking-on arm for the weak public datasets
  - P1/P2/P3 variant subsets

and writes outputs/four_batches_strata_variants/final_results.json, plus
printed tables for: batch overview, P0 vs P4 per public dataset,
thinking on/off, matched variants, best per-dataset
configuration, and confusion matrices for the key failure cases.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ordinal_metrics import DATA, OUT, load_completed, metrics, ordinal_order, pooled

SOURCES = [
    ("P0", Path("outputs/four_batches_strata/predictions.jsonl")),
    ("P4", OUT / "p4_public" / "predictions.jsonl"),
    ("TH", OUT / "p4_weak_thinking" / "predictions.jsonl"),
    ("P1", OUT / "p1_default" / "predictions.jsonl"),
    ("P2", OUT / "p2_default" / "predictions.jsonl"),
    ("P3", OUT / "p3_best" / "predictions.jsonl"),
]

ORDINAL_DATASETS = {"surimi9", "turmeric_levels"}


def mode_from_dir(directory: str) -> str:
    tail = directory.split("/")[-1]
    return "full" if tail == "full_reference" else tail


def load_experiments() -> list[dict]:
    rows: list[dict] = []
    for source, path in SOURCES:
        path = path if path.is_absolute() else Path(__file__).resolve().parent / path
        for rec in load_completed(path):
            directory = rec.get("directory", "")
            exp_dir = DATA / directory
            meta_path = exp_dir / "metadata.json"
            labels_path = exp_dir / "test_labels.json"
            if not meta_path.exists() or not labels_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            m = rec.get("test_metrics") or {}
            if not m:
                continue

            task = str(meta.get("task") or "PUBLIC")
            dataset = str(meta.get("dataset_id") or "?")
            if task == "T3":
                cond = f"{meta.get('control')}_k{meta.get('target_k')}"
                mode = "t3"
            elif task in ("T1", "T2", "UNSEEN"):
                cond = f"day{meta.get('source_day')}_to_day{meta.get('target_day')}" if task == "T2" else f"day{meta.get('target_day')}"
                mode = mode_from_dir(directory)
            else:
                mode = mode_from_dir(directory)
                cond = mode

            labels = meta.get("candidate_labels") or []
            ord_order = ordinal_order(labels)
            order = ord_order or labels
            pairs = None
            om = None
            if order:
                rank = {lab: i for i, lab in enumerate(order)}
                gold = {x["id"]: str(x["label"]) for x in json.loads(labels_path.read_text(encoding="utf-8"))}
                pred_rows = (rec.get("response") or {}).get("test") or []
                pred = {str(x.get("id")): str(x.get("label", "")).strip() for x in pred_rows}
                pairs = []
                missing = invalid = 0
                for gid, gl in gold.items():
                    pl = pred.get(gid)
                    if pl is None:
                        missing += 1
                        continue
                    if gl not in rank or pl not in rank:
                        invalid += 1
                        continue
                    pairs.append((rank[gl], rank[pl]))
                if dataset in ORDINAL_DATASETS and ord_order:
                    om = metrics(pairs, len(order))
                    if om is not None:
                        om.update({"missing": missing, "invalid": invalid})

            rows.append(
                {
                    "source": source,
                    "experiment_id": rec["experiment_id"],
                    "directory": directory,
                    "batch": str(meta.get("batch", "")),
                    "task": task,
                    "dataset_id": dataset,
                    "cond": cond,
                    "mode": mode,
                    "control": meta.get("control"),
                    "target_k": meta.get("target_k"),
                    "source_day": meta.get("source_day"),
                    "target_day": meta.get("target_day"),
                    "repeat": meta.get("repeat"),
                    "total": int(m.get("total", 0)),
                    "correct": int(m.get("correct", 0)),
                    "acc": float(m.get("accuracy_pct", 0.0)),
                    "f1": float(m.get("macro_f1", 0.0)),
                    "input_tokens": int(rec.get("input_tokens") or 0),
                    "completion_tokens": int(rec.get("completion_tokens") or 0),
                    "ordinal": om,
                    "pairs": pairs,
                    "order": order,
                }
            )
    return rows


def agg(rows: list[dict]) -> dict:
    n = len(rows)
    total = sum(r["total"] for r in rows)
    correct = sum(r["correct"] for r in rows)
    out = {
        "n_exp": n,
        "n": total,
        "acc": round(100 * correct / total, 2) if total else None,
        "f1": round(sum(r["f1"] for r in rows) / n, 4) if n else None,
    }
    ord_rows = [r["ordinal"] for r in rows if r["ordinal"]]
    if ord_rows:
        out["ordinal"] = pooled(ord_rows)
    return out


def grouped(rows: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def fmt(v, width=6, decimals=2) -> str:
    if v is None:
        return " " * (width - 1) + "-"
    return f"{v:{width}.{decimals}f}"


def print_overview(rows: list[dict]) -> None:
    p0 = [r for r in rows if r["source"] == "P0"]
    print("\n== P0 baseline: batch overview ==")
    print(f"{'batch/task':28s} {'n_exp':>5s} {'samples':>7s} {'acc':>7s} {'F1':>7s} {'adj@1':>7s} {'QWK':>7s}")
    for (batch, task), rs in sorted(grouped(p0, "batch", "task").items()):
        a = agg(rs)
        o = a.get("ordinal") or {}
        if not {r["dataset_id"] for r in rs} <= ORDINAL_DATASETS:
            o = {}
        print(
            f"{batch + '/' + task:28s} {a['n_exp']:5d} {a['n']:7d} {fmt(a['acc'], 7)} "
            f"{fmt(a['f1'], 7, 4)} {fmt(o.get('adjacent_pct'), 7)} {fmt(o.get('qwk'), 7, 4)}"
        )
    for source, label in (("P4", "batch_04/P4"),):
        rs = [r for r in rows if r["source"] == source]
        if rs:
            a = agg(rs)
            o = a.get("ordinal") or {}
            if not {r["dataset_id"] for r in rs} <= ORDINAL_DATASETS:
                o = {}
            print(
                f"{label:28s} {a['n_exp']:5d} {a['n']:7d} {fmt(a['acc'], 7)} "
                f"{fmt(a['f1'], 7, 4)} {fmt(o.get('adjacent_pct'), 7)} {fmt(o.get('qwk'), 7, 4)}"
            )


def print_public(rows: list[dict]) -> None:
    p0 = [r for r in rows if r["source"] == "P0" and r["task"] == "PUBLIC"]
    p4 = [r for r in rows if r["source"] == "P4"]
    modes = ["full", "k_1", "k_3", "k_5"]
    datasets = sorted({r["dataset_id"] for r in p4})
    print("\n== batch_04 public: P0 -> P4 (pooled test accuracy) ==")
    print(f"{'dataset':22s} {'mode':5s} {'P0':>7s} {'P4':>7s} {'delta':>7s} {'P0 QWK':>7s} {'P4 QWK':>7s}")
    for ds in datasets:
        for md in modes:
            a = agg([r for r in p0 if r["dataset_id"] == ds and r["mode"] == md])
            b = agg([r for r in p4 if r["dataset_id"] == ds and r["mode"] == md])
            if a["n_exp"] == 0 or b["n_exp"] == 0:
                continue
            q0 = (a.get("ordinal") or {}).get("qwk")
            q1 = (b.get("ordinal") or {}).get("qwk")
            print(
                f"{ds:22s} {md:5s} {fmt(a['acc'], 7)} {fmt(b['acc'], 7)} "
                f"{fmt(b['acc'] - a['acc'], 7)} {fmt(q0, 7, 4)} {fmt(q1, 7, 4)}"
            )


def print_thinking(rows: list[dict]) -> None:
    off = [r for r in rows if r["source"] == "P4"]
    on = [r for r in rows if r["source"] == "TH"]
    print("\n== thinking off -> on (weak public datasets) ==")
    print(f"{'dataset':20s} {'mode':5s} {'off':>7s} {'on':>7s} {'delta':>7s} {'off QWK':>8s} {'on QWK':>8s}")
    for ds in ["beef", "turmeric_levels"]:
        for md in ["full", "k_1", "k_3", "k_5"]:
            a = agg([r for r in off if r["dataset_id"] == ds and r["mode"] == md])
            b = agg([r for r in on if r["dataset_id"] == ds and r["mode"] == md])
            if a["n_exp"] and b["n_exp"]:
                q0 = (a.get("ordinal") or {}).get("qwk")
                q1 = (b.get("ordinal") or {}).get("qwk")
                print(
                    f"{ds:20s} {md:5s} {fmt(a['acc'], 7)} {fmt(b['acc'], 7)} "
                    f"{fmt(b['acc'] - a['acc'], 7)} {fmt(q0, 8, 4)} {fmt(q1, 8, 4)}"
                )
        a = agg([r for r in off if r["dataset_id"] == ds])
        b = agg([r for r in on if r["dataset_id"] == ds])
        print(f"{ds:20s} {'all':5s} {fmt(a['acc'], 7)} {fmt(b['acc'], 7)} {fmt(b['acc'] - a['acc'], 7)}")


def print_variants(rows: list[dict]) -> None:
    p0_by_id = {r["experiment_id"]: r for r in rows if r["source"] == "P0"}
    for source in ("P1", "P2", "P3"):
        rs = [r for r in rows if r["source"] == source and r["experiment_id"] in p0_by_id]
        if not rs:
            continue
        print(f"\n== matched P0 -> {source} (same experiment ids) ==")
        print(f"{'task/cond/mode':40s} {'P0':>7s} {'var':>7s} {'delta':>7s} {'P0 QWK':>8s} {'var QWK':>8s}")
        for key, group in sorted(grouped(rs, "task", "cond", "mode").items()):
            base = [p0_by_id[r["experiment_id"]] for r in group]
            a = agg(base)
            b = agg(group)
            q0 = (a.get("ordinal") or {}).get("qwk")
            q1 = (b.get("ordinal") or {}).get("qwk")
            name = "/".join(str(x) for x in key)
            print(
                f"{name:40s} {fmt(a['acc'], 7)} {fmt(b['acc'], 7)} {fmt(b['acc'] - a['acc'], 7)} "
                f"{fmt(q0, 8, 4)} {fmt(q1, 8, 4)}"
            )


def print_best(rows: list[dict]) -> dict:
    print("\n== best configuration per batch_04 dataset ==")
    print(f"{'dataset':22s} {'best overall':>28s} {'best k-shot':>26s}")
    datasets = ["beef", "coffee", "fresh_meats", "olive_oils", "turmeric_authenticity", "turmeric_levels"]
    result = {}
    for ds in datasets:
        cand = [r for r in rows if r["source"] in ("P0", "P4") and r["dataset_id"] == ds]
        if not cand:
            continue
        def pick(rs):
            best = None
            for key, group in grouped(rs, "source", "mode").items():
                a = agg(group)
                if best is None or (a["acc"] or 0) > best[0]:
                    best = (a["acc"], key, a)
            return best
        overall = pick(cand)
        kshot = pick([r for r in cand if r["mode"] != "full"])
        o_txt = f"{overall[1][0]} {overall[1][1]} {overall[0]:.2f}%"
        k_txt = f"{kshot[1][0]} {kshot[1][1]} {kshot[0]:.2f}%" if kshot else "-"
        print(f"{ds:22s} {o_txt:>28s} {k_txt:>26s}")
        result[ds] = {
            "overall": {"source": overall[1][0], "mode": overall[1][1], "acc": overall[0]},
            "k_shot": None
            if not kshot
            else {"source": kshot[1][0], "mode": kshot[1][1], "acc": kshot[0]},
        }
    return result


def confusion(pairs: list[tuple[int, int]], k: int) -> list[list[int]]:
    m = [[0] * k for _ in range(k)]
    for t, p in pairs:
        m[t][p] += 1
    return m


def print_confusion(title: str, rows: list[dict]) -> None:
    if not rows:
        print(f"\n[{title}] no data")
        return
    order = rows[0]["order"]
    pairs = [pair for r in rows for pair in (r["pairs"] or [])]
    if not pairs:
        print(f"\n[{title}] no pairs")
        return
    m = confusion(pairs, len(order))
    n = sum(sum(row) for row in m)
    print(f"\n[{title}] n={n} (rows=true, cols=pred)")
    print("true\\pred " + " ".join(f"{lab:>6s}" for lab in order) + "   acc")
    for i, lab in enumerate(order):
        row = m[i]
        acc = 100 * row[i] / sum(row) if sum(row) else 0
        print(f"{lab:>9s} " + " ".join(f"{v:6d}" for v in row) + f" {acc:6.1f}%")
    off = sorted(
        ((m[i][j], order[i], order[j]) for i in range(len(order)) for j in range(len(order)) if i != j),
        reverse=True,
    )[:6]
    print("top confusions:", ", ".join(f"{t}->{p}:{c}" for c, t, p in off if c))


def main() -> None:
    rows = load_experiments()
    print_overview(rows)
    print_public(rows)
    print_thinking(rows)
    print_variants(rows)
    best_config = print_best(rows)

    print("\n" + "=" * 70)
    print("CONFUSION MATRICES")
    p0 = [r for r in rows if r["source"] == "P0"]
    p4 = [r for r in rows if r["source"] == "P4"]
    print_confusion(
        "surimi9 T2 day0->day14 full (P0)",
        [r for r in p0 if r["task"] == "T2" and r["cond"] == "day0_to_day14" and r["mode"] == "full"],
    )
    print_confusion(
        "surimi9 T3 source_only k5 (P0)",
        [r for r in p0 if r["task"] == "T3" and r["cond"] == "source_only_k5"],
    )
    print_confusion(
        "coffee k1 (P0)",
        [r for r in p0 if r["dataset_id"] == "coffee" and r["mode"] == "k_1"],
    )
    print_confusion(
        "coffee k1 (P4)",
        [r for r in p4 if r["dataset_id"] == "coffee" and r["mode"] == "k_1"],
    )
    print_confusion(
        "turmeric_levels full (P0)",
        [r for r in p0 if r["dataset_id"] == "turmeric_levels" and r["mode"] == "full"],
    )
    print_confusion(
        "turmeric_levels full (P4)",
        [r for r in p4 if r["dataset_id"] == "turmeric_levels" and r["mode"] == "full"],
    )

    payload = {
        "n_experiments": len(rows),
        "best_config": best_config,
        "overview_p0": {
            f"{batch}/{task}": agg(rs)
            for (batch, task), rs in sorted(grouped([r for r in rows if r["source"] == "P0"], "batch", "task").items())
        },
        "public_p0": {
            f"{ds}/{md}": agg(rs)
            for (ds, md), rs in sorted(grouped([r for r in rows if r["source"] == "P0" and r["task"] == "PUBLIC"], "dataset_id", "mode").items())
        },
        "public_p4": {
            f"{ds}/{md}": agg(rs)
            for (ds, md), rs in sorted(grouped([r for r in rows if r["source"] == "P4"], "dataset_id", "mode").items())
        },
        "thinking": {
            f"{ds}/{md}/{src}": agg(rs)
            for (ds, md, src), rs in sorted(
                grouped([r for r in rows if r["source"] in ("P4", "TH")], "dataset_id", "mode", "source").items()
            )
        },
        "confusion": {
            "surimi_T2_day0_to_day14_full_P0": confusion(
                [p for r in p0 if r["task"] == "T2" and r["cond"] == "day0_to_day14" and r["mode"] == "full" for p in (r["pairs"] or [])],
                9,
            ),
            "surimi_T3_source_only_k5_P0": confusion(
                [p for r in p0 if r["task"] == "T3" and r["cond"] == "source_only_k5" for p in (r["pairs"] or [])],
                9,
            ),
            "coffee_k1_P0": confusion([p for r in p0 if r["dataset_id"] == "coffee" and r["mode"] == "k_1" for p in (r["pairs"] or [])], 2),
            "coffee_k1_P4": confusion([p for r in p4 if r["dataset_id"] == "coffee" and r["mode"] == "k_1" for p in (r["pairs"] or [])], 2),
            "turmeric_full_P0": confusion([p for r in p0 if r["dataset_id"] == "turmeric_levels" and r["mode"] == "full" for p in (r["pairs"] or [])], 7),
            "turmeric_full_P4": confusion([p for r in p4 if r["dataset_id"] == "turmeric_levels" and r["mode"] == "full" for p in (r["pairs"] or [])], 7),
        },
    }
    out_path = OUT / "final_results.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path} ({len(rows)} experiments)")


if __name__ == "__main__":
    main()
