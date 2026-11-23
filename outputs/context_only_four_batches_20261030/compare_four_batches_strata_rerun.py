import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "context_only_four_batches_20261030"
DATA = ROOT / "data_json" / "four_batches"
STRATA = ROOT / "outputs" / "four_batches_strata_rerun" / "predictions.jsonl"


def jread(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def current_predictions():
    result = {}
    for path in OUT.rglob("predictions.json"):
        data = jread(path)
        rel = path.parent.relative_to(OUT).as_posix()
        eid = data["experiment_id"]
        if eid in result:
            raise ValueError(f"Duplicate current experiment_id: {eid}")
        result[eid] = (rel, data)
    return result


def old_predictions():
    result = {}
    with STRATA.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "completed":
                continue
            response = row.get("response") or {}
            eid = row.get("experiment_id")
            if eid in result:
                raise ValueError(f"Duplicate strata experiment_id: {eid}")
            result[eid] = (row, response)
    return result


def metadata(rel):
    parts = rel.split("/")
    batch = parts[0]
    dataset = parts[1]
    task = parts[2] if len(parts) >= 5 else ""
    condition = parts[-1]
    return batch, dataset, task, condition


def score_predictions(rel, pred):
    batch = rel.split("/")[0]
    source_dir = DATA / batch / Path(*rel.split("/")[1:])
    results = {}
    for split in ("val", "test"):
        truth_rows = jread(source_dir / f"{split}_labels.json")
        pred_rows = pred.get(split)
        if not isinstance(pred_rows, list):
            raise ValueError(f"Missing prediction array: {rel} {split}")
        truth_ids = [x["id"] for x in truth_rows]
        pred_ids = [x["id"] for x in pred_rows]
        if truth_ids != pred_ids:
            raise ValueError(f"Current prediction IDs/order mismatch: {rel} {split}")
        truth = {x["id"]: str(x["label"]) for x in truth_rows}
        correct = sum(str(x["label"]) == truth[x["id"]] for x in pred_rows)
        results[split] = (correct, len(truth_rows))
    return results


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(records):
    group_defs = {
        "overall": lambda r: ("all",),
        "batch": lambda r: (r["batch"],),
        "task": lambda r: (r["task"],),
        "dataset": lambda r: (r["dataset"],),
        "condition": lambda r: (r["condition"],),
        "dataset_condition": lambda r: (r["dataset"], r["condition"]),
        "batch_task": lambda r: (r["batch"], r["task"]),
        "batch_condition": lambda r: (r["batch"], r["condition"]),
    }
    output = []
    for grouping, keyfn in group_defs.items():
        groups = defaultdict(list)
        for r in records:
            groups[keyfn(r)].append(r)
        for key, rows in groups.items():
            for split in ("val", "test", "combined"):
                if split == "combined":
                    n = sum(x["current"]["val"][1] + x["current"]["test"][1] for x in rows)
                    cur = sum(x["current"]["val"][0] + x["current"]["test"][0] for x in rows)
                    old = sum(x["strata"]["val"][0] + x["strata"]["test"][0] for x in rows)
                    macro_cur = sum((x["current"]["val"][0] + x["current"]["test"][0]) / (x["current"]["val"][1] + x["current"]["test"][1]) for x in rows) / len(rows)
                    macro_old = sum((x["strata"]["val"][0] + x["strata"]["test"][0]) / (x["strata"]["val"][1] + x["strata"]["test"][1]) for x in rows) / len(rows)
                else:
                    n = sum(x["current"][split][1] for x in rows)
                    cur = sum(x["current"][split][0] for x in rows)
                    old = sum(x["strata"][split][0] for x in rows)
                    macro_cur = sum(x["current"][split][0] / x["current"][split][1] for x in rows) / len(rows)
                    macro_old = sum(x["strata"][split][0] / x["strata"][split][1] for x in rows) / len(rows)
                values = {"grouping": grouping, "split": split, "experiments": len(rows), "n": n,
                          "context_correct": cur, "context_accuracy": cur / n,
                          "strata_correct": old, "strata_accuracy": old / n,
                          "delta_pp": (cur - old) / n * 100,
                          "context_macro_accuracy": macro_cur,
                          "strata_macro_accuracy": macro_old,
                          "mean_experiment_delta_pp": (macro_cur - macro_old) * 100}
                if grouping == "overall":
                    values["value"] = key[0]
                elif grouping in ("batch", "task", "dataset", "condition"):
                    values["value"] = key[0]
                else:
                    values["value"] = " | ".join(key)
                output.append(values)
    return output


def main():
    current = current_predictions()
    previous = old_predictions()
    records = []
    for eid, (rel, current_pred) in current.items():
        if eid not in previous:
            raise ValueError(f"No matching four_batches_strata_rerun prediction: {eid}")
        old_row, old_pred = previous[eid]
        if old_row["directory"].split("/", 1)[-1] != "/".join(rel.split("/")[1:]):
            raise ValueError(f"Path mismatch for {eid}: current={rel}; strata={old_row['directory']}")
        batch, dataset, task, condition = metadata(rel)
        if batch == "batch_02_T2" and old_row.get("batch") not in ("batch_02_T2", "batch_02_T2_T3"):
            raise ValueError(f"Unexpected old batch for T2 experiment {eid}: {old_row.get('batch')}")
        cur_scores = score_predictions(rel, current_pred)
        old_scores = score_predictions(rel, old_pred)
        for split in ("val", "test"):
            truth_ids = [x["id"] for x in jread(DATA / batch / Path(*rel.split("/")[1:]) / f"{split}_labels.json")]
            if [x["id"] for x in old_pred.get(split, [])] != truth_ids:
                raise ValueError(f"Strata prediction IDs/order mismatch: {eid} {split}")
        records.append({"experiment_id": eid, "directory": rel, "batch": batch, "dataset": dataset,
                        "task": task, "condition": condition, "current": cur_scores, "strata": old_scores})

    detail = []
    for r in records:
        for split in ("val", "test"):
            c, n = r["current"][split]
            s, _ = r["strata"][split]
            detail.append({"experiment_id": r["experiment_id"], "directory": r["directory"],
                           "batch": r["batch"], "dataset": r["dataset"], "task": r["task"],
                           "condition": r["condition"], "split": split, "n": n,
                           "context_correct": c, "context_accuracy": c / n,
                           "strata_correct": s, "strata_accuracy": s / n,
                           "delta_pp": (c - s) / n * 100})
    summary = summarize(records)
    write_csv(OUT / "four_batches_strata_comparison_by_experiment.csv", detail)
    write_csv(OUT / "four_batches_strata_comparison_summary.csv", summary)

    def pct(value):
        return f"{value * 100:.2f}%"

    lines = [
        "# 上下文预测与 four_batches_strata_rerun 配对比较", "",
        f"共比较 {len(records)} 套同实验结果。仅使用 experiment_id、目录、样本 ID 完全配对的预测，并对两方都按相同 source label files 重新计分；旧结果模型来自 `four_batches_strata_rerun/meta.json`（Qwen3.8 Flash Next Q2_0，thinking=false）。",
        "", "测试集为主结果，准确率按样本汇总（micro）；Δ=上下文预测减数值 rerun，单位为百分点。", "",
        "| Batch | 上下文预测 | strata rerun | Δ (pp) |", "|---|---:|---:|---:|"
    ]
    for batch in ("batch_01_T1", "batch_02_T2", "batch_03_unseen_15_30", "batch_04_public"):
        r = next(x for x in summary if x["grouping"] == "batch" and x["value"] == batch and x["split"] == "test")
        lines.append(f"| {batch} | {r['context_correct']}/{r['n']} ({pct(r['context_accuracy'])}) | {r['strata_correct']}/{r['n']} ({pct(r['strata_accuracy'])}) | {r['delta_pp']:+.2f} |")
    lines += ["", "## 按任务 / 数据集的测试集结果", "", "| Batch | 任务/数据集 | 上下文预测 | strata rerun | Δ (pp) |", "|---|---|---:|---:|---:|"]
    for r in summary:
        if r["split"] != "test":
            continue
        if r["grouping"] == "batch_task" and r["value"].split(" | ")[0] != "batch_04_public":
            batch, task = r["value"].split(" | ")
            lines.append(f"| {batch} | {task} | {r['context_correct']}/{r['n']} ({pct(r['context_accuracy'])}) | {r['strata_correct']}/{r['n']} ({pct(r['strata_accuracy'])}) | {r['delta_pp']:+.2f} |")
        elif r["grouping"] == "dataset" and r["value"] in ("beef", "coffee", "fresh_meats", "olive_oils", "turmeric_authenticity", "turmeric_levels"):
            lines.append(f"| PUBLIC | {r['value']} | {r['context_correct']}/{r['n']} ({pct(r['context_accuracy'])}) | {r['strata_correct']}/{r['n']} ({pct(r['strata_accuracy'])}) | {r['delta_pp']:+.2f} |")
    lines += ["", "## PUBLIC 按数据集 × 参考条件（验证 / 测试）", ""]
    for dataset in ("beef", "coffee", "fresh_meats", "olive_oils", "turmeric_authenticity", "turmeric_levels"):
        lines += [
            f"### {dataset}", "",
            "| 条件 | 验证集：上下文 | 验证集：strata | Δ (pp) | 测试集：上下文 | 测试集：strata | Δ (pp) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for condition in ("full_reference", "k_1", "k_3", "k_5"):
            val = next(x for x in summary if x["grouping"] == "dataset_condition" and x["value"] == f"{dataset} | {condition}" and x["split"] == "val")
            test = next(x for x in summary if x["grouping"] == "dataset_condition" and x["value"] == f"{dataset} | {condition}" and x["split"] == "test")
            lines.append(
                f"| {condition} | {val['context_correct']}/{val['n']} ({pct(val['context_accuracy'])}) | {val['strata_correct']}/{val['n']} ({pct(val['strata_accuracy'])}) | {val['delta_pp']:+.2f} | "
                f"{test['context_correct']}/{test['n']} ({pct(test['context_accuracy'])}) | {test['strata_correct']}/{test['n']} ({pct(test['strata_accuracy'])}) | {test['delta_pp']:+.2f} |"
            )
        lines.append("")
    lines += ["", "## 按条件的测试集结果", "", "| Batch | 条件 | 上下文预测 | strata rerun | Δ (pp) |", "|---|---|---:|---:|---:|"]
    for r in summary:
        if r["grouping"] == "batch_condition" and r["split"] == "test":
            batch, condition = r["value"].split(" | ")
            lines.append(f"| {batch} | {condition} | {r['context_correct']}/{r['n']} ({pct(r['context_accuracy'])}) | {r['strata_correct']}/{r['n']} ({pct(r['strata_accuracy'])}) | {r['delta_pp']:+.2f} |")
    lines += ["", "完整逐实验结果见 `four_batches_strata_comparison_by_experiment.csv`，分组汇总见 `four_batches_strata_comparison_summary.csv`。", ""]
    (OUT / "four_batches_strata_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")
    manifest_path = OUT / "run_manifest.json"
    manifest = jread(manifest_path)
    outputs = manifest.setdefault("scoring_outputs", [])
    for name in ("four_batches_strata_comparison_report.md", "four_batches_strata_comparison_by_experiment.csv", "four_batches_strata_comparison_summary.csv"):
        if name not in outputs:
            outputs.append(name)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"matched_experiments": len(records), "test_by_batch": [r for r in summary if r["grouping"] == "batch" and r["split"] == "test"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
