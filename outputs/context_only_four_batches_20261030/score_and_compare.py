import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "context_only_four_batches_20261030"
DATA = ROOT / "data_json" / "four_batches"
OLD = {
    "batch_02_T2": ROOT / "outputs" / "context_only_T2_20261029",
    "batch_04_public": ROOT / "outputs" / "context_only_public_20261029",
}


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def split_name(batch):
    return {
        "val": "val_labels.json",
        "test": "test_labels.json",
    }


def get_experiment_files(root):
    result = {}
    for p in root.rglob("predictions.json"):
        rel = p.parent.relative_to(root).as_posix()
        data = read_json(p)
        exp_id = data.get("experiment_id")
        if not exp_id:
            raise ValueError(f"Missing experiment_id: {p}")
        if exp_id in result:
            raise ValueError(f"Duplicate experiment_id {exp_id} under {root}")
        result[exp_id] = (rel, data)
    return result


def batch_source(rel):
    first = rel.split("/")[0]
    if first not in {"batch_01_T1", "batch_02_T2", "batch_03_unseen_15_30", "batch_04_public"}:
        raise ValueError(f"Unknown batch in relative path {rel}")
    return first


def source_experiment_path(rel):
    parts = rel.split("/")
    # New output paths are batch-prefixed; raw data paths start after that prefix.
    return Path(*parts[1:])


def validate_and_score(rel, exp_id, pred, source_root, batch_override=None):
    batch = batch_override or batch_source(rel)
    source_rel = source_experiment_path(rel) if batch_override is None else Path(rel)
    src_dir = DATA / batch / source_rel
    # Normalize legacy paths to the current batch-prefixed representation.
    norm_rel = rel if batch_override is None else f"{batch}/{rel}"
    rows = {"experiment_id": exp_id, "directory": norm_rel, "batch": batch}
    for split, label_name in split_name(batch).items():
        labels = read_json(src_dir / label_name)
        pred_rows = pred.get(split)
        if not isinstance(pred_rows, list):
            raise ValueError(f"{rel}: predictions missing list for {split}")
        truth_ids = [x["id"] for x in labels]
        pred_ids = [x["id"] for x in pred_rows]
        if truth_ids != pred_ids:
            raise ValueError(f"{rel}: {split} ID/order mismatch with labels")
        truth = {x["id"]: str(x["label"]) for x in labels}
        correct = sum(str(x["label"]) == truth[x["id"]] for x in pred_rows)
        rows[f"{split}_n"] = len(labels)
        rows[f"{split}_correct"] = correct
        rows[f"{split}_accuracy"] = correct / len(labels) if labels else None
    return rows


def group_keys(row):
    rel_parts = row["directory"].split("/")
    # batch, dataset/task, (day or direction when present), repeat, condition
    batch = row["batch"]
    if batch == "batch_04_public":
        dataset = rel_parts[1]
        task = "public"
    else:
        dataset = rel_parts[1]
        task = rel_parts[2] if len(rel_parts) >= 5 else ""
    cond = rel_parts[-1]
    return batch, dataset, task, cond


def make_summary(rows, group_fields, group_name):
    groups = defaultdict(list)
    for r in rows:
        key = tuple(r[x] for x in group_fields)
        groups[key].append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        label = dict(zip(group_fields, key))
        for split in ("val", "test"):
            n = sum(r[f"{split}_n"] for r in rs)
            correct = sum(r[f"{split}_correct"] for r in rs)
            label.update({
                "grouping": group_name,
                "split": split,
                "experiments": len(rs),
                "n": n,
                "correct": correct,
                "accuracy": correct / n if n else None,
                "mean_experiment_accuracy": sum(r[f"{split}_accuracy"] for r in rs) / len(rs) if rs else None,
            })
            out.append(label.copy())
        # Combined val+test is sample-pooled and is useful as a compact overall figure.
        n = sum(r["val_n"] + r["test_n"] for r in rs)
        correct = sum(r["val_correct"] + r["test_correct"] for r in rs)
        macro = sum(
            (r["val_correct"] + r["test_correct"]) / (r["val_n"] + r["test_n"])
            for r in rs
        ) / len(rs) if rs else None
        label.update({
            "grouping": group_name,
            "split": "combined",
            "experiments": len(rs),
            "n": n,
            "correct": correct,
            "accuracy": correct / n if n else None,
            "mean_experiment_accuracy": macro,
        })
        out.append(label.copy())
    return out


def write_csv(path, rows):
    if not rows:
        return
    cols = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def score_root(root, batch_override=None):
    scored = []
    experiments = get_experiment_files(root)
    for exp_id, (rel, pred) in experiments.items():
        scored.append(validate_and_score(rel, exp_id, pred, root, batch_override))
    return experiments, scored


def paired_compare(current_map, current_scores):
    by_id = {r["experiment_id"]: r for r in current_scores}
    comparison_rows = []
    for batch, old_root in OLD.items():
        old_map, old_scores = score_root(old_root, batch)
        old_scored = {r["experiment_id"]: r for r in old_scores}
        shared = set(current_map) & set(old_map) & set(old_scored)
        for exp_id in sorted(shared):
            new_rel, new_pred = current_map[exp_id]
            old_rel, old_pred = old_map[exp_id]
            if batch_source(new_rel) != batch:
                continue
            if source_experiment_path(new_rel).as_posix() != old_rel:
                raise ValueError(f"Path mismatch for {exp_id}: new={new_rel}, old={old_rel}")
            for split in ("val", "test"):
                new_records = new_pred[split]
                old_records = old_pred[split]
                if [x["id"] for x in new_records] != [x["id"] for x in old_records]:
                    raise ValueError(f"Prediction IDs/order differ for paired {exp_id} {split}")
                nr = by_id[exp_id]
                oldr = old_scored[exp_id]
                comparison_rows.append({
                    "experiment_id": exp_id,
                    "batch": batch,
                    "directory": new_rel,
                    "split": split,
                    "n": nr[f"{split}_n"],
                    "new_correct": nr[f"{split}_correct"],
                    "new_accuracy": nr[f"{split}_accuracy"],
                    "previous_correct": oldr[f"{split}_correct"],
                    "previous_accuracy": oldr[f"{split}_accuracy"],
                    "delta_pp": (nr[f"{split}_accuracy"] - oldr[f"{split}_accuracy"]) * 100,
                })
    # Aggregate paired outcomes with same exact matched experiments and sample IDs.
    summary = []
    groups = defaultdict(list)
    for r in comparison_rows:
        parts = r["directory"].split("/")
        condition = parts[-1]
        dataset = parts[1]
        group_fields = [("overall", "all"), ("batch", r["batch"]), ("dataset", dataset), ("condition", condition), ("batch_condition", r["batch"] + " | " + condition)]
        for g, value in group_fields:
            groups[(g, value, r["split"])].append(r)
    for (grouping, value, split), rs in sorted(groups.items()):
        n = sum(r["n"] for r in rs)
        new_correct = sum(r["new_correct"] for r in rs)
        old_correct = sum(r["previous_correct"] for r in rs)
        summary.append({
            "grouping": grouping,
            "value": value,
            "split": split,
            "experiments": len(rs),
            "n": n,
            "new_correct": new_correct,
            "new_accuracy": new_correct / n if n else None,
            "previous_correct": old_correct,
            "previous_accuracy": old_correct / n if n else None,
            "delta_pp": (new_correct - old_correct) / n * 100 if n else None,
            "mean_experiment_delta_pp": sum(r["delta_pp"] for r in rs) / len(rs) if rs else None,
        })
    return comparison_rows, summary


def main():
    current_map, rows = score_root(OUT)
    if len(rows) != 300:
        raise ValueError(f"Expected 300 current predictions, found {len(rows)}")
    summary = []
    summary += make_summary(rows, ["batch"], "batch")
    summary += make_summary(rows, ["batch", "directory"], "experiment")
    # Dataset/task/condition groupings are directly useful for interpretation.
    for r in rows:
        batch, dataset, task, cond = group_keys(r)
        r["dataset"] = dataset
        r["task"] = task
        r["condition"] = cond
    summary += make_summary(rows, ["batch", "dataset"], "dataset")
    summary += make_summary(rows, ["batch", "task"], "task")
    summary += make_summary(rows, ["batch", "condition"], "condition")
    summary += make_summary(rows, ["batch", "task", "condition"], "task_condition")
    write_csv(OUT / "accuracy_by_experiment.csv", rows)
    write_csv(OUT / "accuracy_summary.csv", summary)

    paired_rows, paired_summary = paired_compare(current_map, rows)
    write_csv(OUT / "paired_comparison_by_experiment.csv", paired_rows)
    write_csv(OUT / "paired_comparison_summary.csv", paired_summary)

    overall = next(r for r in summary if r["grouping"] == "batch" and r["batch"] == "batch_01_T1" and r["split"] == "test")
    report = {
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_result_directory": str(OUT),
        "experiment_count": len(rows),
        "validation": "all 300 predictions matched exact ordered IDs to source val/test label files",
        "accuracy_definition": "exact string label match; sample-pooled accuracy is correct / n; mean_experiment_accuracy is an unweighted macro average",
        "overall_by_batch": [r for r in summary if r["grouping"] == "batch"],
        "paired_comparison_counts": {
            "batch_02_T2": len({r["experiment_id"] for r in paired_rows if r["batch"] == "batch_02_T2"}),
            "batch_04_public": len({r["experiment_id"] for r in paired_rows if r["batch"] == "batch_04_public"}),
        },
        "paired_comparison_summary": [r for r in paired_summary if r["grouping"] in ("batch", "overall")],
        "unpaired_batches": ["batch_01_T1", "batch_03_unseen_15_30"],
    }
    (OUT / "accuracy_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    def pct(x):
        return f"{x * 100:.2f}%"

    lines = [
        "# 四个 batch 上下文预测准确率与历史结果对比",
        "",
        f"计分时间（UTC）：{report['scored_at_utc']}。共计 {len(rows)} 套预测；所有预测均与源 `val_labels.json` / `test_labels.json` 的样本 ID 和顺序逐项核对。标签按字符串精确匹配。",
        "",
        "## 本轮准确率",
        "",
        "准确率采用样本汇总（micro，正确数 / 样本数）；验证集和测试集分开报告，另列二者合并结果。",
        "",
        "| Batch | 验证集 | 测试集 | 验证+测试 |",
        "|---|---:|---:|---:|",
    ]
    batch_names = {
        "batch_01_T1": "T1",
        "batch_02_T2": "T2",
        "batch_03_unseen_15_30": "UNSEEN",
        "batch_04_public": "PUBLIC",
    }
    for batch in batch_names:
        stats = {r["split"]: r for r in report["overall_by_batch"] if r["batch"] == batch}
        lines.append(
            f"| {batch_names[batch]} | {stats['val']['correct']}/{stats['val']['n']} ({pct(stats['val']['accuracy'])}) "
            f"| {stats['test']['correct']}/{stats['test']['n']} ({pct(stats['test']['accuracy'])}) "
            f"| {stats['combined']['correct']}/{stats['combined']['n']} ({pct(stats['combined']['accuracy'])}) |"
        )
    lines += [
        "",
        "## 与上一轮上下文预测的配对比较",
        "",
        "只比较有历史上下文预测的 T2（2026-10-29）和 PUBLIC（2026-10-29）；实验 ID、样本 ID、样本顺序均相同。Δ 为本轮减上一轮，单位为百分点。",
        "",
        "| Batch | Split | 本轮 | 上一轮 | Δ (pp) |",
        "|---|---|---:|---:|---:|",
    ]
    for batch in ("batch_02_T2", "batch_04_public"):
        for split in ("test", "val"):
            r = next(x for x in paired_summary if x["grouping"] == "batch" and x["value"] == batch and x["split"] == split)
            lines.append(
                f"| {batch_names[batch]} | {split} | {r['new_correct']}/{r['n']} ({pct(r['new_accuracy'])}) "
                f"| {r['previous_correct']}/{r['n']} ({pct(r['previous_accuracy'])}) | {r['delta_pp']:+.2f} |"
            )
    lines += [
        "",
        "### T2 按迁移方向的测试集准确率",
        "",
        "| 迁移方向 | 正确数 | 测试样本 | 准确率 |",
        "|---|---:|---:|---:|",
    ]
    for direction in ("day0_to_day14", "day0_to_day7", "day7_to_day14"):
        r = next(x for x in summary if x["grouping"] == "task_condition" and x["batch"] == "batch_02_T2" and x["task"] == direction and x["condition"] == "full_reference" and x["split"] == "test")
        # Direction-level total combines all four conditions.
        direction_row = next(x for x in summary if x["grouping"] == "task" and x["batch"] == "batch_02_T2" and x["task"] == direction and x["split"] == "test")
        lines.append(f"| {direction} | {direction_row['correct']}/{direction_row['n']} | {direction_row['n']} | {pct(direction_row['accuracy'])} |")
    lines += [
        "",
        "### 按条件的测试集配对结果",
        "",
        "| Batch | 条件 | 本轮 | 上一轮 | Δ (pp) |",
        "|---|---|---:|---:|---:|",
    ]
    for batch in ("batch_02_T2", "batch_04_public"):
        for condition in ("full_reference", "k_1", "k_3", "k_5"):
            key = f"{batch} | {condition}"
            r = next(x for x in paired_summary if x["grouping"] == "batch_condition" and x["value"] == key and x["split"] == "test")
            lines.append(
                f"| {batch_names[batch]} | {condition} | {r['new_correct']}/{r['n']} ({pct(r['new_accuracy'])}) "
                f"| {r['previous_correct']}/{r['n']} ({pct(r['previous_accuracy'])}) | {r['delta_pp']:+.2f} |"
            )
    lines += [
        "",
        "### PUBLIC 按数据集测试集配对结果",
        "",
        "| 数据集 | 本轮 | 上一轮 | Δ (pp) |",
        "|---|---:|---:|---:|",
    ]
    for dataset in ("beef", "coffee", "fresh_meats", "olive_oils", "turmeric_authenticity", "turmeric_levels"):
        r = next(x for x in paired_summary if x["grouping"] == "dataset" and x["value"] == dataset and x["split"] == "test")
        lines.append(
            f"| {dataset} | {r['new_correct']}/{r['n']} ({pct(r['new_accuracy'])}) "
            f"| {r['previous_correct']}/{r['n']} ({pct(r['previous_accuracy'])}) | {r['delta_pp']:+.2f} |"
        )
    lines += [
        "",
        "## 说明",
        "",
        "T1 和 UNSEEN 在已保存结果中没有同策略历史预测，因此只报告本轮准确率，未做跨轮差值。完整逐实验数据见 `accuracy_by_experiment.csv`；分组汇总见 `accuracy_summary.csv`；逐实验配对和分组配对见 `paired_comparison_by_experiment.csv` 与 `paired_comparison_summary.csv`。",
        "",
    ]
    (OUT / "accuracy_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# 四个 batch 独立上下文预测\n\n"
        "本轮重新运行四个数据 batch，共300套实验：T1 60、T2 60、UNSEEN 60、PUBLIC 120。每套实验使用独立上下文，仅读取对应的 prompt.txt、train.json、val.json、test.json。预测标签由模型根据上下文自行判断；文件工具只用于输入读取、输出序列化和结构检查，不执行数值预测算法。预测阶段不读取答案文件或其他实验结果。\n\n"
        "每套预测保存在与源 batch 对应的路径下，文件为 predictions.json，包含 experiment_id、val 和 test 的有序 {id,label} 记录。run_manifest.json 和 validation_report.json 记录清单与结构检查。预测完成后，单独读取 val_labels.json 和 test_labels.json 计算准确率，并与有匹配实验 ID 的上一轮上下文结果配对比较。\n\n"
        "准确率与对比报告见 accuracy_comparison_report.md；逐实验得分见 accuracy_by_experiment.csv；汇总见 accuracy_summary.csv；配对明细见 paired_comparison_by_experiment.csv 和 paired_comparison_summary.csv。计分脚本为 score_and_compare.py。\n",
        encoding="utf-8",
    )
    manifest_path = OUT / "run_manifest.json"
    manifest = read_json(manifest_path)
    manifest["scoring_performed"] = True
    manifest["scoring_timestamp_utc"] = report["scored_at_utc"]
    manifest["scoring_outputs"] = [
        "accuracy_by_experiment.csv",
        "accuracy_summary.csv",
        "accuracy_report.json",
        "accuracy_comparison_report.md",
        "paired_comparison_by_experiment.csv",
        "paired_comparison_summary.csv",
    ]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "experiments": len(rows),
        "accuracy_by_batch": report["overall_by_batch"],
        "paired_counts": report["paired_comparison_counts"],
        "paired_summary": report["paired_comparison_summary"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
