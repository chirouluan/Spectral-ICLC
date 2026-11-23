"""Import source-project model outputs matching the currently present packages."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from spectral_icl.io import file_hash, write_json
from spectral_icl.web_baselines import current_package_index, package_data

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT.parents[1] / "Spectral-ICLC"
DEFAULT_SOURCE_RUN = Path("results/web_matched_baselines/20261028T141929_5e72fc07")
DEFAULT_EXPORT = Path("deepseek_web_test/four_batches_20261020T143923_1d231230")
MODELS = ["svm_rbf", "random_forest", "cnn1d", "transformer"]


def verify_tree(root: Path) -> int:
    lines = (root / "checksum.sha256").read_text(encoding="utf-8").splitlines()
    for line in lines:
        expected, relative = line.split("  ", 1)
        if file_hash(root / relative) != expected:
            raise ValueError(f"Source result checksum mismatch: {relative}")
    return len(lines)


def package_file_identity(current: Path, source: Path, info: dict) -> int:
    checked = 0
    for name, expected in info["file_sha256"].items():
        a, b = current / name, source / name
        if not a.is_file() or not b.is_file():
            raise ValueError(f"Missing matched package file: {name}")
        if file_hash(a) != expected or file_hash(b) != expected:
            raise ValueError(f"Current package differs from source experiment data: {current}/{name}")
        checked += 1
    if file_hash(current / "metadata.json") != file_hash(source / "metadata.json"):
        raise ValueError(f"Package metadata differs: {current}")
    return checked + 1


def write_report(root: Path, records: pd.DataFrame, packages: pd.DataFrame, source_path: Path,
                 source_audit: dict, copied_runs: int, checked_files: int) -> None:
    test = records[records.split.eq("test")].copy()
    test["accuracy_pct"] = test.correct / test.total * 100
    full = test[test.reference_mode.eq("full_reference")]
    tasks = ["T1", "T2", "UNSEEN", "PUBLIC"]
    active_datasets = sorted(packages.dataset_id.unique())
    table_rows = []
    for (task, dataset, source_day, target_day, model), group in full.groupby(
        ["task", "dataset_id", "source_day", "target_day", "model_id"], dropna=False
    ):
        if task not in tasks or dataset not in active_datasets:
            continue
        vals = group.accuracy_pct
        table_rows.append((task, dataset, source_day, target_day, model,
                           f"{vals.mean():.2f} ± {vals.std(ddof=1):.2f}" if len(vals) > 1 else f"{vals.mean():.2f}"))
    lines = [
        "# 与当前 data_json 匹配的传统模型对照结果",
        "",
        f"当前项目实际包含 {len(packages)} 套实验包；本结果包含 {copied_runs} 个已完成模型运行。",
        "原项目基线采用的网页实验导出中，这些实验的 prompt、Train/Val/Test、标签、metadata 和私有映射与当前包逐文件校验一致。原项目完整基线包含历史 T3 与额外公共任务；本目录只保留当前存在的 300 套包对应结果。",
        "",
        "模型包括 SVM-RBF、随机森林、CNN1D、Transformer；UNSEEN 任务另有光谱质心浓度内插。结果均来自原项目已完成并审计的运行，没有在迁移时重新拟合或重算。K 条件只取相应数据包的 Train 参考，使用其冻结导出特征。CNN1D/Transformer 的早停使用 Val；Test 仅用于最终评分。",
        "",
        "完整参考条件的 Test 准确率（跨五个 repeat 的均值 ± 样本标准差，单位 %）：",
        "",
        "| task | dataset | source day | target day | model | accuracy |",
        "|---|---|---:|---:|---|---:|",
    ]
    for row in table_rows:
        lines.append("| " + " | ".join("—" if pd.isna(v) else str(v) for v in row) + " |")
    lines += [
        "",
        "标准差反映五次划分间的变动，不是独立制备批次的置信区间。六个公共任务彼此不合并成一个总准确率。闭集分类器不能输出 Train 中未见类别，因此 UNSEEN 任务另报告内插模型；这属于带有序标签信息的传统数值基线，不能与普通闭集分类器等价解释。",
        "",
        "",
        "逐次 Test 指标见 `summary/test_each_run.csv`；按条件聚合见 `summary/test_all_conditions.csv`；完整参考对 K=1/3/5 的配对差值见 `summary/paired_full_minus_k_summary.csv`；逐样本预测与每类指标位于各实验目录下的模型子目录。",
        "",
        "迁移来源：`" + str(source_path) + "`。源结果的全量独立审计状态为 `" + str(source_audit.get("status")) + "`。迁入内容保留逐模型预测、指标、运行 manifest 和校验和；为控制体积，没有复制 1.25 GB 的共享拟合模型缓存。",
        "",
        "迁移不代表对源分析作了新的独立复核；迁移脚本验证了当前包与源包元数据中的每个文件哈希相同，并验证了每个迁入模型目录的文件校验和。",
        "",
    ]
    (root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--source-export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output", type=Path, default=PROJECT / "results/classical_baselines/matched_data_json_20261030")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    source_run = args.source_run if args.source_run.is_absolute() else source_root / args.source_run
    source_export = args.source_export if args.source_export.is_absolute() else source_root / args.source_export
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Output already exists; refusing to overwrite: {output}")

    current_root = PROJECT / "data_json/four_batches"
    packages = current_package_index(current_root)
    source_index = pd.read_csv(source_export / "index.csv", dtype={"experiment_id": str})
    source_by_id = source_index.set_index("experiment_id", drop=False)
    registry = pd.read_csv(source_run / "model_registry.csv", dtype={"experiment_id": str})
    registry_by_key = registry.set_index(["experiment_id", "model_id"], drop=False)
    source_audit = json.loads((source_run / "validation_report.json").read_text(encoding="utf-8"))
    if source_audit.get("status") != "passed" or source_audit.get("failed_model_runs", 0):
        parser.error("Source run does not have a clean full audit")

    matches, missing, mismatched, result_rows = [], [], [], []
    checked_input_files = 0
    for row in packages.to_dict("records"):
        experiment_id = row["experiment_id"]
        if experiment_id not in source_by_id.index:
            missing.append((experiment_id, "source package"))
            continue
        source_row = source_by_id.loc[experiment_id]
        current_package = current_root / row["directory"]
        source_package = source_export / source_row.directory
        info, _ = package_data(current_package)
        try:
            checked_input_files += package_file_identity(current_package, source_package, info)
        except ValueError as exc:
            mismatched.append((experiment_id, str(exc)))
            continue
        expected_models = MODELS + (["spectral_centroid_interpolation"] if info["task"] == "UNSEEN" else [])
        source_rows = []
        for model in expected_models:
            key = (experiment_id, model)
            if key not in registry_by_key.index:
                missing.append(key)
                continue
            record = registry_by_key.loc[key]
            result_dir = Path(record.result_directory)
            if not result_dir.is_absolute():
                result_dir = source_root / result_dir
            if record.status != "complete":
                missing.append((experiment_id, model, record.status))
                continue
            try:
                verify_tree(result_dir)
            except Exception as exc:
                mismatched.append((experiment_id, model, str(exc)))
                continue
            destination = output / row["directory"] / model
            source_rows.append((record, result_dir, destination))
        matches.append((row, info, source_row, source_package, source_rows))

    if missing or mismatched or len(matches) != len(packages):
        print(json.dumps({"packages": len(packages), "matched": len(matches), "missing": missing[:10],
                          "mismatched": mismatched[:10]}, ensure_ascii=False, indent=2))
        raise SystemExit("Cannot import an incomplete or mismatched result set")

    # Create only after the full read-only preflight succeeds.
    output.mkdir(parents=True, exist_ok=False)
    selected_ids = set(packages.experiment_id)
    copied = 0
    normalized_registry = []
    copied_run_dirs = []
    for row, info, source_row, source_package, source_rows in matches:
        for record, result_dir, destination in source_rows:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(result_dir, destination)
            copied += 1
            copied_run_dirs.append(destination.relative_to(output).as_posix())
        normalized = registry[registry.experiment_id.eq(row["experiment_id"])].copy()
        for ix, record in normalized.iterrows():
            if record.status == "complete":
                normalized.loc[ix, "result_directory"] = str((output / row["directory"] / record.model_id).resolve())
            normalized.loc[ix, "web_directory"] = row["directory"]
            normalized.loc[ix, "batch"] = row["batch"]
        normalized_registry.append(normalized)

    plan = pd.read_csv(source_run / "plan.csv", dtype={"experiment_id": str})
    plan = plan[plan.experiment_id.isin(selected_ids)].copy()
    plan["directory"] = plan.experiment_id.map(packages.set_index("experiment_id").directory)
    plan["batch"] = plan.experiment_id.map(packages.set_index("experiment_id").batch)
    plan.to_csv(output / "plan.csv", index=False, encoding="utf-8-sig")

    all_runs = pd.read_csv(source_run / "all_runs.csv", dtype={"experiment_id": str})
    all_runs = all_runs[all_runs.experiment_id.isin(selected_ids)].copy()
    all_runs["web_directory"] = all_runs.experiment_id.map(packages.set_index("experiment_id").directory)
    all_runs["batch"] = all_runs.experiment_id.map(packages.set_index("experiment_id").batch)
    all_runs["result_directory"] = all_runs.apply(
        lambda r: str((output / r.web_directory / r.model_id).resolve()), axis=1)
    all_runs.to_csv(output / "all_runs.csv", index=False, encoding="utf-8-sig")
    pd.concat(normalized_registry, ignore_index=True).to_csv(output / "model_registry.csv", index=False, encoding="utf-8-sig")

    summary = output / "summary"
    summary.mkdir()
    test_each = pd.read_csv(source_run / "summary/test_each_run.csv", dtype={"experiment_id": str})
    test_each = test_each[test_each.experiment_id.isin(selected_ids)].copy()
    test_each["web_directory"] = test_each.experiment_id.map(packages.set_index("experiment_id").directory)
    test_each["batch"] = test_each.experiment_id.map(packages.set_index("experiment_id").batch)
    test_each["result_directory"] = test_each.apply(
        lambda r: str((output / r.web_directory / r.model_id).resolve()), axis=1)
    test_each.to_csv(summary / "test_each_run.csv", index=False, encoding="utf-8-sig")

    test_conditions = pd.read_csv(source_run / "summary/test_all_conditions.csv")
    active_pairs = set(zip(packages.task, packages.dataset_id))
    test_conditions = test_conditions[
        [(task, dataset) in active_pairs for task, dataset in zip(test_conditions.task, test_conditions.dataset_id)]
    ]
    test_conditions.to_csv(summary / "test_all_conditions.csv", index=False, encoding="utf-8-sig")
    pairs = pd.read_csv(source_run / "summary/paired_full_minus_k_each_repeat.csv", dtype={"experiment_id": str})
    pairs = pairs[pairs.experiment_id.isin(selected_ids)].copy()
    pairs.to_csv(summary / "paired_full_minus_k_each_repeat.csv", index=False, encoding="utf-8-sig")
    pair_summary = pd.read_csv(source_run / "summary/paired_full_minus_k_summary.csv")
    pair_summary = pair_summary[
        [(task, dataset) in active_pairs for task, dataset in zip(pair_summary.task, pair_summary.dataset_id)]
    ]
    pair_summary.to_csv(summary / "paired_full_minus_k_summary.csv", index=False, encoding="utf-8-sig")
    unseen = pd.read_csv(source_run / "summary/unseen_full_reference_means_pct.csv")
    unseen.to_csv(summary / "unseen_full_reference_means_pct.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "status": "imported_and_verified",
        "created_from_source_project": str(source_root),
        "source_result_run": str(source_run),
        "source_result_manifest_sha256": file_hash(source_run / "manifest.json"),
        "source_validation_report_sha256": file_hash(source_run / "validation_report.json"),
        "source_validation_status": source_audit["status"],
        "active_current_packages": len(packages),
        "package_data_matches": len(matches),
        "input_files_hash_checked": checked_input_files,
        "complete_model_run_directories_copied": copied,
        "models": MODELS + ["spectral_centroid_interpolation (UNSEEN only)"],
        "fitted_model_cache_copied": False,
        "source_model_cache_size_gb_not_copied": 1.25,
        "source_audit_metrics": {k: source_audit.get(k) for k in (
            "planned_model_runs", "completed_model_runs", "prediction_rows_verified",
            "unique_fitted_estimators_verified", "warning_runs", "api_requests", "actual_llm_predictions")},
        "copied_result_directories": copied_run_dirs,
        "note": "Current package inputs and source package files match by every metadata-declared SHA-256. Result directories retain original manifests and checksums; shared fitted estimator cache was excluded to avoid copying 1.25 GB.",
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "validation_report.json", {
        "status": "passed_import_checks", "packages": len(packages), "copied_model_runs": copied,
        "input_package_files_hash_checked": checked_input_files,
        "per_model_result_checksum_files_verified_during_import": copied,
        "source_run_audit_status": source_audit["status"],
        "source_prediction_rows_verified": source_audit["prediction_rows_verified"],
        "fitted_estimator_cache_included": False,
    })
    write_report(output, test_each, packages, source_run, source_audit, copied, checked_input_files)
    checksum_lines = []
    for path in sorted(p for p in output.rglob("*") if p.is_file() and p.name != "checksum.sha256"):
        checksum_lines.append(f"{file_hash(path)}  {path.relative_to(output).as_posix()}")
    (output / "checksum.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "packages": len(packages), "model_runs_copied": copied,
                      "input_files_hash_checked": checked_input_files}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
