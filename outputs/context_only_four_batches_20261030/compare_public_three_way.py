import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "context_only_four_batches_20261030"
DATA = ROOT / "data_json" / "four_batches" / "batch_04_public"
OLD_CONTEXT = ROOT / "outputs" / "context_only_public_20261029"
STRATA = ROOT / "outputs" / "four_batches_strata_rerun" / "predictions.jsonl"
DATASETS = ("beef", "coffee", "fresh_meats", "olive_oils", "turmeric_authenticity", "turmeric_levels")
CONDITIONS = ("full_reference", "k_1", "k_3", "k_5")


def jread(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_predictions(root, nested_response=False):
    rows = {}
    for path in root.rglob("predictions.json"):
        data = jread(path)
        eid = data.get("experiment_id")
        if not eid:
            raise ValueError(f"Missing experiment_id at {path}")
        if eid in rows:
            raise ValueError(f"Duplicate experiment_id {eid} under {root}")
        rows[eid] = data
    return rows


def load_strata():
    rows = {}
    with STRATA.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "completed":
                continue
            if row.get("directory", "").split("/")[0] != "batch_04_public":
                continue
            eid = row["experiment_id"]
            if eid in rows:
                raise ValueError(f"Duplicate strata experiment_id {eid}")
            rows[eid] = row.get("response") or {}
    return rows


def labels_for(dataset, repeat, condition, split):
    return jread(DATA / dataset / f"repeat_{repeat}" / condition / f"{split}_labels.json")


def score(split, truth_rows, pred_rows, context):
    true_ids = [r["id"] for r in truth_rows]
    if not isinstance(pred_rows, list) or [r.get("id") for r in pred_rows] != true_ids:
        raise ValueError(f"ID/order mismatch in {context} {split}")
    truth = {r["id"]: str(r["label"]) for r in truth_rows}
    correct = sum(str(r["label"]) == truth[r["id"]] for r in pred_rows)
    return correct, len(truth_rows)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summarize(records):
    defs = {
        "overall": lambda r: ("PUBLIC",),
        "dataset": lambda r: (r["dataset"],),
        "condition": lambda r: (r["condition"],),
        "dataset_condition": lambda r: (r["dataset"], r["condition"]),
    }
    result = []
    models = ("current", "previous_context", "strata")
    for grouping, fn in defs.items():
        groups = defaultdict(list)
        for r in records:
            groups[fn(r)].append(r)
        for key, rows in groups.items():
            for split in ("val", "test", "combined"):
                entry = {"grouping": grouping, "value": " | ".join(key), "split": split, "experiments": len(rows)}
                for model in models:
                    if split == "combined":
                        n = sum(x[model]["val"][1] + x[model]["test"][1] for x in rows)
                        correct = sum(x[model]["val"][0] + x[model]["test"][0] for x in rows)
                        macro = sum((x[model]["val"][0] + x[model]["test"][0]) / (x[model]["val"][1] + x[model]["test"][1]) for x in rows) / len(rows)
                    else:
                        n = sum(x[model][split][1] for x in rows)
                        correct = sum(x[model][split][0] for x in rows)
                        macro = sum(x[model][split][0] / x[model][split][1] for x in rows) / len(rows)
                    entry[f"{model}_n"] = n
                    entry[f"{model}_correct"] = correct
                    entry[f"{model}_accuracy"] = correct / n
                    entry[f"{model}_macro_accuracy"] = macro
                entry["current_minus_previous_context_pp"] = (entry["current_accuracy"] - entry["previous_context_accuracy"]) * 100
                entry["current_minus_strata_pp"] = (entry["current_accuracy"] - entry["strata_accuracy"]) * 100
                entry["previous_context_minus_strata_pp"] = (entry["previous_context_accuracy"] - entry["strata_accuracy"]) * 100
                result.append(entry)
    return result


def main():
    current_root = OUT / "batch_04_public"
    current = load_predictions(current_root)
    previous = load_predictions(OLD_CONTEXT)
    strata = load_strata()
    ids = set(current)
    if ids != set(previous) or ids != set(strata):
        raise ValueError(f"Experiment ID mismatch: current={len(ids)}, previous={len(set(previous))}, strata={len(set(strata))}")
    records = []
    detail = []
    for eid, cur_pred in current.items():
        # Current output relative path is dataset/repeat_N/condition.
        path = next(p for p in current_root.rglob("predictions.json") if jread(p).get("experiment_id") == eid)
        rel = path.parent.relative_to(current_root).as_posix()
        parts = rel.split("/")
        dataset, repeat, condition = parts[0], int(parts[1].replace("repeat_", "")), parts[2]
        old_pred, strata_pred = previous[eid], strata[eid]
        rec = {"experiment_id": eid, "dataset": dataset, "repeat": repeat, "condition": condition}
        for split in ("val", "test"):
            truth = labels_for(dataset, repeat, condition, split)
            for model, pred in (("current", cur_pred), ("previous_context", old_pred), ("strata", strata_pred)):
                rec.setdefault(model, {})[split] = score(split, truth, pred.get(split), f"{eid}/{model}")
        records.append(rec)
        for split in ("val", "test"):
            n = rec["current"][split][1]
            detail.append({
                "experiment_id": eid, "dataset": dataset, "repeat": repeat, "condition": condition,
                "split": split, "n": n,
                "current_correct": rec["current"][split][0], "current_accuracy": rec["current"][split][0] / n,
                "previous_context_correct": rec["previous_context"][split][0], "previous_context_accuracy": rec["previous_context"][split][0] / n,
                "strata_correct": rec["strata"][split][0], "strata_accuracy": rec["strata"][split][0] / n,
            })
    summary = summarize(records)
    write_csv(OUT / "public_three_way_comparison_by_experiment.csv", detail)
    write_csv(OUT / "public_three_way_comparison_summary.csv", summary)

    def pct(x):
        return f"{x*100:.2f}%"

    overall = {r["split"]: r for r in summary if r["grouping"] == "overall"}
    lines = [
        "# PUBLIC 三方配对比较", "",
        "同一批120套实验（6个数据集 × 5次重复 × 4种条件），按同一源标签重新计分。测试预测的样本 ID 和顺序均与标签及另两种方法逐项匹配。", "",
        "比较对象：本轮上下文预测（2026-10-30）、上一轮上下文预测（context_only_public_20261029）、four_batches_strata_rerun（Qwen3.8 Flash Next Q2_0，thinking=false）。准确率为样本汇总 micro。", "",
        "## 总体", "",
        "| Split | 本轮上下文 | 上轮上下文 | strata rerun | 本轮−上轮 | 本轮−strata |", "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("val", "test", "combined"):
        r = overall[split]
        lines.append(f"| {split} | {r['current_correct']}/{r['current_n']} ({pct(r['current_accuracy'])}) | {r['previous_context_correct']}/{r['previous_context_n']} ({pct(r['previous_context_accuracy'])}) | {r['strata_correct']}/{r['strata_n']} ({pct(r['strata_accuracy'])}) | {r['current_minus_previous_context_pp']:+.2f} pp | {r['current_minus_strata_pp']:+.2f} pp |")
    lines += ["", "## 按数据集（测试集）", "", "| 数据集 | 本轮上下文 | 上轮上下文 | strata rerun | 本轮−上轮 | 本轮−strata |", "|---|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        r = next(x for x in summary if x["grouping"] == "dataset" and x["value"] == dataset and x["split"] == "test")
        lines.append(f"| {dataset} | {r['current_correct']}/{r['current_n']} ({pct(r['current_accuracy'])}) | {r['previous_context_correct']}/{r['previous_context_n']} ({pct(r['previous_context_accuracy'])}) | {r['strata_correct']}/{r['strata_n']} ({pct(r['strata_accuracy'])}) | {r['current_minus_previous_context_pp']:+.2f} | {r['current_minus_strata_pp']:+.2f} |")
    lines += ["", "## 按参考条件（测试集）", "", "| 条件 | 本轮上下文 | 上轮上下文 | strata rerun | 本轮−上轮 | 本轮−strata |", "|---|---:|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        r = next(x for x in summary if x["grouping"] == "condition" and x["value"] == condition and x["split"] == "test")
        lines.append(f"| {condition} | {r['current_correct']}/{r['current_n']} ({pct(r['current_accuracy'])}) | {r['previous_context_correct']}/{r['previous_context_n']} ({pct(r['previous_context_accuracy'])}) | {r['strata_correct']}/{r['strata_n']} ({pct(r['strata_accuracy'])}) | {r['current_minus_previous_context_pp']:+.2f} | {r['current_minus_strata_pp']:+.2f} |")
    lines += ["", "## 数据集 × 参考条件（测试集）", "", "| 数据集 | 条件 | 本轮上下文 | 上轮上下文 | strata rerun | 本轮−上轮 | 本轮−strata |", "|---|---|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for condition in CONDITIONS:
            r = next(x for x in summary if x["grouping"] == "dataset_condition" and x["value"] == f"{dataset} | {condition}" and x["split"] == "test")
            lines.append(f"| {dataset} | {condition} | {r['current_correct']}/{r['current_n']} ({pct(r['current_accuracy'])}) | {r['previous_context_correct']}/{r['previous_context_n']} ({pct(r['previous_context_accuracy'])}) | {r['strata_correct']}/{r['strata_n']} ({pct(r['strata_accuracy'])}) | {r['current_minus_previous_context_pp']:+.2f} | {r['current_minus_strata_pp']:+.2f} |")
    lines += ["", "完整逐实验结果见 `public_three_way_comparison_by_experiment.csv`，分组汇总见 `public_three_way_comparison_summary.csv`。", ""]
    (OUT / "public_three_way_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")
    manifest_path = OUT / "run_manifest.json"
    manifest = jread(manifest_path)
    outputs = manifest.setdefault("scoring_outputs", [])
    for name in ("public_three_way_comparison_report.md", "public_three_way_comparison_by_experiment.csv", "public_three_way_comparison_summary.csv"):
        if name not in outputs:
            outputs.append(name)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"experiments": len(records), "overall": overall["test"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
