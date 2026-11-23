"""Four task-aware web experiment batches; no APIs and no model predictions.

K controls displayed examples, not the labels used by supervised representations.
Validation and Test remain single complete, unlabelled JSON files per condition.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import platform
import uuid
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits
import yaml

from .contexts import t3_support
from .data import load_surimi, validate_split
from .dimensionality import SNVReduction
from .io import digest, file_hash, write_json
from .public_data import load_public, public_split, read_public_config, validate_public_split
from .public_validation import PaperRepresentation, checksum
from .web_policy import CONTEXT_ONLY_POLICY


PROMPT_VERSION = "four-batches-web-v3-context-only"
PARTS = ("val", "test")


def ordered_labels(labels):
    labels = sorted({str(v) for v in labels})
    try:
        return sorted(labels, key=float)
    except ValueError:
        return labels


def reference_budgets(config):
    """None denotes every legal Train observation; K remains context-only."""
    return ([None] if config.get("include_full_reference", False) else [])+config["context_k"]


def nested_support(meta, pool, k, seed):
    """Distinct available groups per class; K=1 is a subset of K=3/5."""
    rng = np.random.default_rng(seed)
    selected = []
    for label, group in meta.loc[pool].groupby("label", sort=True):
        groups = rng.permutation(sorted(group.group_id.unique()))
        # Generate the full class ordering, so later classes do not change with K.
        representatives = [int(rng.choice(sorted(group.index[group.group_id.eq(g)]))) for g in groups]
        if len(representatives) < k:
            raise ValueError(f"Class {label} has only {len(representatives)} Train groups for K={k}")
        selected.extend(representatives[:k])
    return sorted(selected, key=lambda i: digest([seed, str(meta.loc[i, "sample_id"])]))


def public_web_split(data, seed, settings):
    """Keep official public Tests; carve a feasible development pool inside Train."""
    split = public_split(data, seed)
    rng = np.random.default_rng(seed)
    kind = data.spec["loader"]
    if kind == "beef":
        # Six official Train records/class -> five Train, one Val; Test unchanged.
        for _, group in data.meta[data.meta.official_partition.eq("train")].groupby("label", sort=True):
            ids = rng.permutation(group.index.to_numpy())
            n_val = settings["beef_validation_per_class"]
            split.loc[ids, "split"] = "train"
            split.loc[ids[:n_val], "split"] = "val"
    elif kind == "honey":
        split["split"] = "train"
        split.loc[data.meta.acquisition.eq(settings["honey_test_acquisition"]), "split"] = "test"
        # All concentration levels/segments of each product-acquisition stay together.
        for _, group in data.meta.groupby(["brand", "botanical_class"], sort=True):
            eligible = sorted(v for v in group.acquisition.unique() if v in [1, 2, 3, 4, 5])
            acquisition = rng.choice(eligible)
            split.loc[group.index[group.acquisition.eq(acquisition)], "split"] = "val"
    validate_public_split(data, split)
    return split


def _task_description(info):
    task = info["task"]
    source, target = info.get("source_day"), info.get("target_day")
    if task == "T1":
        return f"任务T1：Day{target}同日鱼糜淀粉添加九分类。参考与待测来自同一天。"
    if task == "T2":
        return (f"任务T2：Day{source}→Day{target}跨日九分类。带标签参考仅来自Day{source}，"
                f"val和test均来自Day{target}。请考虑贮藏时间造成的光谱分布变化；不要请求目标日真实标签。")
    if task == "T3":
        descriptions = {
            "source_only": "只提供源日参考，用于源日上下文对照",
            "target_only": "只展示目标日参考；这是目标日上下文条件，特征仍使用共同的源日变换",
            "mixed": "同时提供源日与目标日参考，比较引入目标日参考光谱的作用",
            "size_matched_source": "提供与混合条件总数量相同的源日参考，用于控制参考数量增加的影响",
        }
        return (f"任务T3：Day{source}→Day{target}目标日适应。当前条件：{descriptions[info['control']]}。"
                f"每条记录的day字段表示采集日，不是类别标签。待预测样本均来自Day{target}。"
                "各条件中的参考物理样品与val/test完全分离；混合条件的源日和目标日参考物理ID也彼此分离。")
    if task == "UNSEEN":
        return ("任务：未见淀粉添加浓度类别识别。参考中只有0、1、5、10、20、40、50%七类；"
                "15%、30%的全部日期记录均未用于表示拟合或提供参考，val也不含这两类。"
                "test可能包含已见七类及未见15%、30%。必须在全部九个候选浓度标签中判断，"
                "不能将预测限制为train.json里出现的标签。可依据参考特征与浓度关系推断中间等级，"
                "但不要假定LDA得分与浓度线性对应，也不要强迫输出某个类别或某种类数量。")
    definitions = {
        "honey_adulteration": "蜂蜜糖浆添加等级五分类（不含100%纯糖）。记录是光谱分块，不能把分块数量当成独立蜂蜜样品数",
        "turmeric_levels": "姜黄铬酸铅添加等级七分类。这是本项目的离散等级任务，不输出连续浓度回归值",
        "turmeric_authenticity": "姜黄真伪二分类：authentic表示0%添加，adulterated表示大于0%的添加；与七等级任务使用同一原始数据",
        "beef": "UCR Beef五分类。标签1–5为发布版类别代码；代码语义未核实，不要自行将其解释为肉种或添加百分比",
        "fruit_purees": "果泥二分类。Strawberry为纯草莓，NON-Strawberry包含其他水果及掺假混合物；不能将后者全解释为同一种掺假物",
        "fresh_meats": "Chicken/Pork/Turkey三种鲜肉物种分类，不是添加浓度分类",
        "olive_oils": "橄榄油国家产地四分类，不是纯度或添加浓度分类",
        "coffee": "Arabica/Robusta咖啡品种二分类，不是掺假浓度回归",
    }
    return "公共数据任务："+definitions[info["dataset_id"]]+"。"


def task_prompt(info):
    labels = json.dumps(info["candidate_labels"], ensure_ascii=False)
    unseen = info["task"] == "UNSEEN"
    extra = ("test还要分别按已见类、未见类统计seen_accuracy_pct、unseen_accuracy_pct，并计算"
             "h_mean_pct=2×seen_accuracy_pct×unseen_accuracy_pct/(seen_accuracy_pct+unseen_accuracy_pct)，"
             "二者都为0时H-mean为0；另外给出15%和30%各自的正确数、总数及召回率。"
             "不能把仅含已见类的val称为未见类别验证。" if unseen else "")
    schema = {"experiment_id": info["experiment_id"], "val": [{"id": "原始ID", "label": "预测标签"}],
              "test": [{"id": "原始ID", "label": "预测标签"}]}
    mode = info.get("reference_mode", "context_k")
    reference_note = ("当前为完整训练参考主实验：train.json包含全部合法Train记录，不是K个参考的子集。请使用全部参考，不得自行抽样。"
                      if mode == "full_reference" else
                      "当前为T3目标日适应的专门预算条件，源日/目标日参考组成按本任务说明执行。"
                      if mode == "t3_adaptation" else
                      f"当前为补充的参考数量实验，每类提供K={info['k']}个不同样品组的参考；K不代表整个表示流程仅使用这些标签。")
    return f'''你要依据带标签参考完成食品光谱分类，并在之后收到答案时核对冻结的预测。同一个Prompt支持预测与评分两阶段。收到材料后直接执行，不要只回复“已学习”，不要等待我说开始。

{CONTEXT_ONLY_POLICY}

实验编号：{info['experiment_id']}
{_task_description(info)}
features按以下顺序排列：{json.dumps(info['feature_order'], ensure_ascii=False)}。
特征说明：{info['feature_description']}
允许的label仅为以下字符串：{labels}。
参考数{info['reference_count']}，val共{info['val_count']}条，test共{info['test_count']}条。{reference_note}你仅利用上下文进行分类，不更新模型参数。id为随机匿名标识，仅用于匹配结果，不包含可用的类别信息。

一、预测阶段
上传的train.json是参考，val.json和test.json是不含答案的完整待测集合。请仅根据features及任务允许的day信息，完成全部val和test预测，每条恰好一个允许标签；保留各分区输入顺序和原始id，不遗漏、不重复、不编造。不要强制各类预测数量相等。不索取真实标签，也不要声称已经计算准确率。
只输出一个合法JSON对象，不加解释、不加Markdown围栏，结构如下；用实际全部预测替换示例记录，不能输出省略号或占位值：
{json.dumps(schema, ensure_ascii=False)}
这份输出保存为predictions.json，之后不得按真实答案修改。如果输出长度受限，明确说明未完成，不要声称已预测所有记录，也不要私自抽样或改变评价集合。

二、评分阶段
我随后提供val_labels.json和test_labels.json时，直接对照已有预测评分。如果是新聊天，我会同时提供predictions.json。新聊天只有答案、没有预测时，请先让我补充预测结果；不要凭真实标签编造准确率。
先确认experiment_id与本Prompt一致，预测的val/test对象也不可互换。分别在每个分区内部按id匹配，不能按行位置比较。预测保持冻结，不重新分类。真实标签应各有{info['val_count']}和{info['test_count']}个唯一id，标签在允许集合内；答案缺失、重复或非法时先报告问题，不生成虚假的评分。
每个分区分别报告correct、total、accuracy_pct、macro_f1、balanced_accuracy，以及每类的support、precision、recall、f1；accuracy_pct=100×correct/total，百分数保留两位小数。macro_f1和balanced_accuracy用0–1小数。宏平均只对该分区真实出现的类别计算；若某类没有预测，precision为0。
缺失预测、非法标签、重复预测id均计为该样本预测错误；不缩小分母。额外id不进入分母，但须列入issues。重复id即使标签相同也不能静默去重。分类错误统计包含无效预测造成的假阴性。若只有有限的预测输出，也必须按完整答案集作分母并报告遗漏。
{extra}
只返回一个合法JSON对象，包含experiment_id、val、test和issues；issues为空时用[]。评分单位为当前文件中的光谱记录，不把技术重复当作独立物理样品，也不把5次划分当作5套新样品。

预测时不要读取任何*_labels.json、metadata.json、私有映射或变换拟合ID文件。若同一条消息同时提供待测数据与真实答案，却没有先前冻结的预测，请让我将预测与评分材料分开。
'''


def _anonymous(dataset, repeat, sample_id):
    return "ID-"+digest(["web-v1", dataset, repeat, str(sample_id)])[:20].upper()


def write_package(root, relative, meta, z, support, queries, info, decimals, order_seed):
    package = root/relative
    package.mkdir(parents=True, exist_ok=False)
    support = list(support)
    if len(support) != len(set(support)):
        raise ValueError("Duplicate reference observation")
    query_indices = set(queries["val"]) | set(queries["test"])
    if query_indices & set(support):
        raise ValueError("Query appears in reference set")
    reference_groups = set(meta.loc[support, "group_id"])
    if reference_groups & set(meta.loc[list(query_indices), "group_id"]):
        raise ValueError("Reference/query group leakage")
    if set(meta.loc[queries["val"], "group_id"]) & set(meta.loc[queries["test"], "group_id"]):
        raise ValueError("Validation/Test group leakage")
    if info["task"] == "UNSEEN":
        if set(meta.loc[support, "label"]) & {"15", "30"} or set(meta.loc[queries["val"], "label"]) & {"15", "30"}:
            raise ValueError("Unseen class quarantine violation")
    order = lambda ids: sorted(ids, key=lambda i: digest([order_seed, str(meta.loc[i, "sample_id"])]))
    partitions = {"train": order(support), **{part: order(queries[part]) for part in PARTS}}
    mappings = []
    for part, ids in partitions.items():
        rows, answers = [], []
        for i in ids:
            features = np.round(z[i], decimals).tolist()
            if len(features) != len(info["feature_order"]) or not np.isfinite(features).all():
                raise ValueError("Invalid feature vector")
            sample = meta.loc[i]
            anon = _anonymous(info["dataset_id"], info["repeat"], sample.sample_id)
            row = dict(id=anon, features=features)
            if "day" in meta:
                row["day"] = int(sample.day)
            if part == "train":
                row["label"] = str(sample.label)
            rows.append(row)
            answers.append(dict(id=anon, label=str(sample.label)))
            mappings.append(dict(id=anon, sample_id=sample.sample_id, group_id=sample.group_id,
                                 label=str(sample.label), split=part))
        write_json(package/f"{part}.json", rows)
        if part != "train":
            write_json(package/f"{part}_labels.json", answers)
    pd.DataFrame(mappings).to_csv(package/"private_id_mapping.csv", index=False, encoding="utf-8-sig")
    info = {**info, "reference_count": len(support), "val_count": len(queries["val"]),
            "test_count": len(queries["test"]), "feature_decimals": decimals,
            "prompt_version": PROMPT_VERSION, "api_requests": 0,
            "llm_execution_mode": "context_only_no_tools", "tool_use_allowed": False,
            "reference_class_counts": meta.loc[support, "label"].value_counts().to_dict(),
            "reference_group_count": len(reference_groups), "test_subset_or_chunk": False}
    input_hashes = {f"{part}.json": file_hash(package/f"{part}.json") for part in ["train", "val", "test"]}
    info["experiment_id"] = f"{info['task']}_"+digest([relative.as_posix(), info["transform_id"], input_hashes, PROMPT_VERSION])[:16]
    (package/"prompt.txt").write_text(task_prompt(info), encoding="utf-8")
    info["prompt_sha256"] = file_hash(package/"prompt.txt")
    info["file_sha256"] = {p.name: file_hash(p) for p in package.iterdir() if p.is_file()}
    write_json(package/"metadata.json", info)
    return dict(batch=info["batch"], task=info["task"], dataset_id=info["dataset_id"],
        reference_mode=info["reference_mode"], experiment_role=info["experiment_role"],
        repeat=info["repeat"], k=info.get("k"), source_day=info.get("source_day"),
        target_day=info.get("target_day"), control=info.get("control"),
        target_k=info.get("target_k"), reference_count=len(support),
        val_count=info["val_count"], test_count=info["test_count"], dimensions=z.shape[1],
        experiment_id=info["experiment_id"], equivalent_to=info.get("equivalent_to"),
        directory=relative.as_posix(), label_budget=info["label_budget"])


def _record_transform(root, transform_id, model, train_ids, val_ids, source_hashes, details):
    directory = root/"_provenance"/"transforms"/transform_id
    directory.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, directory/"representation.joblib")
    record = dict(transform_id=transform_id, train_fit_sample_ids=sorted(train_ids),
        train_fit_ids_sha256=digest(sorted(train_ids)), validation_selection_sample_ids=sorted(val_ids),
        source_sha256=source_hashes, representation_sha256=file_hash(directory/"representation.joblib"),
        test_used_for_fitting_or_selection=False, **details)
    write_json(directory/"fit.json", record)
    return directory.relative_to(root).as_posix()


def _surimi_export(root, config, batches):
    cfg = yaml.safe_load(Path(config["surimi_config"]).read_text(encoding="utf-8"))
    original, x, _, hashes = load_surimi(cfg)
    meta = original.copy()
    meta["sample_id"] = meta.acquisition_id.astype(str)
    meta["group_id"] = meta.physical_sample_id.astype(str)
    meta["label"] = meta.concentration_pct.map(lambda v: str(int(v)))
    candidates = ordered_labels(meta.label)
    records = []
    for repeat in range(config["repeats"]):
        split_path = Path(cfg["splits"])/f"surimi_split_v1_repeat_{repeat}.csv"
        split = pd.read_csv(split_path)
        validate_split(original, split)
        roles = meta.group_id.map(split.set_index("physical_sample_id").split)
        seed = config["support_seed"]+repeat
        train_all = meta.index[roles.eq("train")].tolist()
        transform_cache = {}

        def fit(day, unseen=False, t3=False):
            key = (day, unseen, t3)
            if key in transform_cache:
                return transform_cache[key]
            fit_ids = meta.index[roles.eq("train") & meta.day.eq(day)].tolist()
            if unseen:
                fit_ids = [i for i in fit_ids if meta.loc[i, "concentration_pct"] not in config["batch_03"]["heldout_levels"]]
            if t3:
                fit_ids = t3_support(meta, train_all, day, 14, 0, seed, "source_only")
            dimensions = config["batch_03"]["lda_dimensions"] if unseen else 8
            model = SNVReduction("lda", dimensions, seed)
            model.fit_transform(x[fit_ids], meta.loc[fit_ids, "concentration_pct"].to_numpy())
            z = model.transform(x)
            if z.shape[1] != dimensions:
                raise ValueError("Unexpected LDA rank")
            name = f"surimi_r{repeat+1}_day{day}_"+("unseen_lda6" if unseen else "t3_anchor_lda8" if t3 else "lda8")
            provenance = _record_transform(root, name, model, meta.loc[fit_ids, "sample_id"].tolist(), [],
                {**hashes, str(split_path): file_hash(split_path)},
                dict(method="SNV+LDA", dimensions=dimensions, source_day=day,
                     heldout_levels=config["batch_03"]["heldout_levels"] if unseen else [],
                     train_fit_physical_ids=sorted(meta.loc[fit_ids, "group_id"]),
                     split_path=str(split_path), split_sha256=file_hash(split_path),
                     source_train_fit_count=len(fit_ids), target_train_labels_used_in_fitting=False))
            transform_cache[key] = (z, name, provenance, fit_ids)
            return transform_cache[key]

        def queries(day, unseen=False):
            return {part: meta.index[roles.eq(part) & meta.day.eq(day) &
                (~meta.concentration_pct.isin(config["batch_03"]["heldout_levels"]) if unseen and part == "val" else True)].tolist() for part in PARTS}

        def emit(batch, task, source, target, k, support=None, control=None, target_k=None, equivalent=None):
            unseen, t3 = task == "UNSEEN", task == "T3"
            z, name, provenance, fit_ids = fit(source, unseen=unseen, t3=t3)
            if support is None:
                pool = meta.index[roles.eq("train") & meta.day.eq(source) &
                    (~meta.concentration_pct.isin(config["batch_03"]["heldout_levels"]) if unseen else True)].tolist()
                support = pool if k is None else nested_support(meta, pool, k, seed)
            direction = f"day{source}" if source == target else f"day{source}_to_day{target}"
            if t3:
                relative = Path(batch)/"T3"/direction/f"repeat_{repeat+1}"/f"target_k_{target_k}"/control
                label_budget = "SOURCE5_LDA_BANK_PLUS_CONTEXT_LABELS; target_only is context-only, not source-label-free"
                reference_mode = "t3_adaptation"
            else:
                reference_mode = "full_reference" if k is None else "context_k"
                relative = Path(batch)/task/direction/f"repeat_{repeat+1}"/("full_reference" if k is None else f"k_{k}")
                label_budget = ("FULL_TRAIN_BANK10_REFERENCE" if k is None else "BANK_10_CONTEXT_K") if not unseen else (
                    "SEEN7_FULL_TRAIN_BANK10_REFERENCE; heldout15_30 never fit" if k is None else "SEEN7_BANK10_CONTEXT_K; heldout15_30 never fit")
            heldout = [str(v) for v in config["batch_03"]["heldout_levels"]] if unseen else []
            info = dict(batch=batch, task=task, dataset_id="surimi9", repeat=repeat+1, k=k,
                reference_mode=reference_mode, experiment_role="supplementary" if reference_mode == "context_k" else "main",
                source_day=source, target_day=target, control=control, target_k=target_k,
                candidate_labels=candidates, heldout_labels=heldout,
                seen_labels=[v for v in candidates if v not in heldout],
                feature_order=[f"LD{i+1}" for i in range(z.shape[1])],
                feature_description=f"逐光谱SNV后LDA{z.shape[1]}维；源日与目标日使用同一个只由合法源日Train拟合的变换，LD得分不是波长或浓度。",
                label_budget=label_budget, transform_id=name, transform_provenance=provenance,
                source_train_fit_count=len(fit_ids), fit_reference_overlap_expected=True,
                split_path=str(split_path), split_sha256=file_hash(split_path),
                equivalent_to=equivalent, support_seed=seed,
                train_accuracy="not_evaluated; train.json is reference, not leave-sample-out predictions")
            if t3:
                source_groups = set(meta.loc[support].loc[lambda f: f.day.eq(source), "group_id"])
                target_groups = set(meta.loc[support].loc[lambda f: f.day.eq(target), "group_id"])
                if source_groups & target_groups:
                    raise ValueError("T3 source/target references share physical IDs")
                info["source_reference_count"] = int(meta.loc[support, "day"].eq(source).sum())
                info["target_reference_count"] = int(meta.loc[support, "day"].eq(target).sum())
            records.append(write_package(root, relative, meta, z, support, queries(target, unseen), info,
                config["feature_decimals"], seed))
            return relative.as_posix()

        if "batch_01" in batches:
            for day in config["batch_01"]["days"]:
                for k in reference_budgets(config):
                    emit("batch_01_T1", "T1", day, day, k)
        if "batch_02" in batches:
            for source, target in config["batch_02"]["t2_pairs"]:
                for k in reference_budgets(config):
                    emit("batch_02_T2_T3", "T2", source, target, k)
            for source, target in config["batch_02"]["t3_pairs"]:
                # These conditions mirror the existing 19-condition T3 plan.
                anchor = None
                for k in config["batch_02"]["t3_target_k"]:
                    for control in config["batch_02"]["t3_controls"]:
                        if k == 0 and control == "target_only":
                            continue
                        support = t3_support(meta, train_all, source, target, k, seed, control)
                        equivalent = anchor if control == "source_only" or k == 0 else None
                        relative = emit("batch_02_T2_T3", "T3", source, target, None, support,
                                        control, k, equivalent)
                        if k == 0 and control == "source_only":
                            anchor = relative
        if "batch_03" in batches:
            for day in config["batch_03"]["days"]:
                for k in reference_budgets(config):
                    emit("batch_03_unseen_15_30", "UNSEEN", day, day, k)
        print(f"Surimi repeat {repeat+1}: {len(records)} cumulative packages", flush=True)
    return records, hashes


def _public_export(root, config):
    cfg = read_public_config(config["public_config"])
    records, hashes = [], {}
    for dataset in cfg["datasets"]:
        data = load_public(dataset, cfg)
        hashes.update(data.source_hashes)
        meta = data.meta
        source = root/"_provenance"/"public_sources"/dataset
        source.mkdir(parents=True, exist_ok=False)
        meta.to_csv(source/"metadata.csv", index=False, encoding="utf-8-sig")
        write_json(source/"source_audit.json", dict(**data.audit, spec=data.spec, source_sha256=data.source_hashes))
        for repeat in range(config["repeats"]):
            seed = cfg["seed"]+repeat
            split = public_web_split(data, seed, config["batch_04"])
            roles = split.split
            train, val = roles.eq("train").to_numpy(), roles.eq("val").to_numpy()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                model = PaperRepresentation(data.spec["method"]).fit(data.x[train], meta.label.to_numpy()[train],
                    data.axis, data.x[val], meta.label.to_numpy()[val])
            z = model.transform(data.x)
            name = f"{dataset}_r{repeat+1}_{data.spec['method']}"
            split_path = source/f"split_repeat_{repeat+1}.csv"
            split.to_csv(split_path, index=False, encoding="utf-8-sig")
            fit_val_ids = meta.loc[val, "sample_id"].tolist() if model.selection_trace_ else []
            provenance = _record_transform(root, name, model, meta.loc[train, "sample_id"].tolist(),
                fit_val_ids, data.source_hashes, dict(method=data.spec["method"],
                    feature_order=model.feature_names_, dimensions=z.shape[1],
                    preprocessed_dimensions=model.preprocessed_dimensions_,
                    validation_used_for_dimension_selection=bool(model.selection_trace_),
                    dimension_selection=model.selection_trace_, fidelity=data.spec["fidelity"],
                    caveats=data.spec["caveats"], split_path=split_path.relative_to(root).as_posix(),
                    split_sha256=file_hash(split_path),
                    fit_warnings=[dict(category=type(w.message).__name__, message=str(w.message)) for w in captured]))
            pool = meta.index[train].tolist()
            queries = {part: meta.index[roles.eq(part)].tolist() for part in PARTS}
            for k in reference_budgets(config):
                support = pool if k is None else nested_support(meta, pool, k, config["support_seed"]+repeat)
                relative = Path("batch_04_public")/dataset/f"repeat_{repeat+1}"/("full_reference" if k is None else f"k_{k}")
                info = dict(batch="batch_04_public", task="PUBLIC", dataset_id=dataset,
                    reference_mode="full_reference" if k is None else "context_k",
                    experiment_role="main" if k is None else "supplementary",
                    repeat=repeat+1, k=k, candidate_labels=ordered_labels(meta.label),
                    feature_order=model.feature_names_,
                    feature_description=f"该数据的文献处理分支 {data.spec['method']}，输出{z.shape[1]}维；同一目录中参考与待测共用Train拟合的变换。",
                    label_budget=("FULL_TRAIN_REFERENCE" if k is None else "FULL_TRAIN_REPRESENTATION_BANK_CONTEXT_K")+"; supervised PLS uses full Train labels",
                    transform_id=name, transform_provenance=provenance,
                    source_train_fit_count=int(train.sum()), source_url=data.spec["source_url"],
                    license=data.spec["license"], fidelity=data.spec["fidelity"], caveats=data.spec["caveats"],
                    validation_used_for_dimension_selection=bool(model.selection_trace_),
                    split_path=split_path.relative_to(root).as_posix(), split_sha256=file_hash(split_path),
                    public_split_note="Beef: official Test retained, 5 Train+1 Val per class inside official Train; honey: official Acquisition6 Test, random one eligible acquisition per product for Val; others grouped approximate 5:1:2",
                    support_seed=config["support_seed"]+repeat,
                    train_accuracy="not_evaluated; train.json is reference, not leave-group-out predictions")
                records.append(write_package(root, relative, meta, z, support, queries, info,
                    config["feature_decimals"], config["support_seed"]+repeat))
        print(f"Public {dataset}: {config['repeats']*len(reference_budgets(config))} complete-query packages", flush=True)
    return records, hashes


def _readme(root, records):
    frame = pd.DataFrame(records)
    intro = '''# 四批网页实验数据

每套目录仅使用同目录文件：预测时上传train.json、val.json、test.json，粘贴prompt.txt；模型读完参考直接预测完整val和test。保存输出为predictions.json。

评分时在原聊天上传val_labels.json、test_labels.json；另开聊天则再提供predictions.json和同一个prompt.txt。不要提前上传答案、metadata、私有映射或_provenance。每个Prompt有独立experiment_id，预测文件不可跨条件混用。

优先运行full_reference主实验：T1/T2使用全部合法源日90条参考，未见类别任务使用已见7类全部70条参考，公共数据使用其完整合法Train池（保留其中的技术重复记录）。k_1/k_3/k_5是补充的参考数量实验。T3按原计划使用源日5、目标日0/1/2/3/5及四个对照，不额外套乘K，也不增加full_reference。

K是每类显示的参考数，SNV+LDA或监督PLS使用的训练标签预算单独记录，不能称为STRICT-K。相同任务/repeat的full_reference与K条件共享同一个变换、完整Val/Test及记录顺序，只改变参考集合。训练参考文件不是训练集预测结果。

前三批：T1/T2共用相应源日Train拟合的LDA8；未见15%/30%单独用7类Train拟合LDA6；T3各条件共用源日5例/类（45条）锚点拟合的LDA8，目标日不拟合变换。T3 target_only仅指展示的参考，仍有源日45标签的表示监督。

第三批val仅14条已见样品，test完整36条（已见28、未见8），评分另给Seen/Unseen/H-mean。15%/30%不参与任何日期的Train/Val表示拟合或支持选择。

公共数据按逐数据文献方案，未统一改成LDA。Beef保留官方Test，其官方Train每类6条中留1条Val、5条Train以支持K=5；这与先前本地基线4/2的开发划分不同，不能直接混为同配置对照。蜂蜜保留Acquisition6 Test，每次在每产品Acquisition1–5中随机选1组作Val，其余Train；共享官方Test的五次实验不是五套新独立测试样品。

完整val和test各是一个JSON文件，没有切包或抽取小型评价子集；尤其蜂蜜文件较大，不能将网页未读完或输出截断算作完整有效结果。所有答案仍按完整评价集评分，缺失预测计错。

T3每repeat保留原计划19个目录，其中部分是相同参考的锚点重复；index.csv的equivalent_to标注其等价条件，分析时不得计成独立实验。公共姜黄两个任务共用70条光谱，不能算两套独立数据。

## 目录索引

'''
    descriptions = {
        "batch_01_T1": "第一批：三日期T1同日九分类",
        "batch_02_T2_T3": "第二批：三个T2方向及Day0→Day14的T3四组对照",
        "batch_03_unseen_15_30": "第三批：联合留出15%/30%的未见类别识别",
        "batch_04_public": "第四批：7套公共数据、8项分类任务",
    }
    lines = [intro]
    for batch, group in frame.groupby("batch", sort=True):
        lines.append(f"- [{descriptions[batch]}]({batch}/README.md)：{len(group)}套。")
        rows = [f"# {descriptions[batch]}", "", "优先运行full_reference主实验；k_1/k_3/k_5为补充曲线。T3保留专门预算。每行对应独立聊天，不上传答案。完整参考保留合法Train中的所有观测，不将重复测量宣称为独立样品。", "",
                "| 数据/任务 | repeat | K / 目标日K | 条件 | 参考 | val | test | 维度 | 文件夹 |", "|---|---:|---:|---|---:|---:|---:|---:|---|"]
        for r in group.to_dict("records"):
            relative = Path(r["directory"]).relative_to(batch).as_posix()
            k = "全部Train" if r["reference_mode"] == "full_reference" else int(r["target_k"]) if pd.notna(r.get("target_k")) else int(r["k"])
            direction = f"D{int(r['source_day'])}→D{int(r['target_day'])}" if pd.notna(r.get("source_day")) else r["dataset_id"]
            control = r.get("control") if pd.notna(r.get("control")) else r["task"]
            rows.append(f"| {direction} | {r['repeat']} | {k} | {control} | {r['reference_count']} | {r['val_count']} | {r['test_count']} | {r['dimensions']} | [打开]({relative}/) |")
        (root/batch/"README.md").write_text("\n".join(rows)+"\n", encoding="utf-8")
    lines += ["", "index.csv列出全部条件；_provenance保存分区、变换、拟合ID和原始文件哈希；validation_report.json为实际完整性检查。这里没有真实LLM预测或性能结果。", "",
              "可用本地确定性评分核对网页算术：", "```powershell", "$env:PYTHONPATH = Join-Path (Get-Location) 'src'",
              "& D:\\Anaconda\\envs\\specllm\\python.exe scripts/score_web_predictions.py --package '<当前实验目录>' --predictions '<predictions.json>' --output '<新评分文件.json>'", "```", ""]
    (root/"README.md").write_text("\n".join(lines), encoding="utf-8")


def export_four_batches(config_path="configs/web_batches.yaml", output=None, batches=None):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    batches = set(batches or ["batch_01", "batch_02", "batch_03", "batch_04"])
    if not batches <= {"batch_01", "batch_02", "batch_03", "batch_04"}:
        raise ValueError("Unknown batch")
    if config["batch_02"]["t3_source_k"] != 5 or config["batch_02"]["t3_pairs"] != [[0, 14]]:
        raise ValueError("Exporter implements the approved source5 Day0→Day14 T3 protocol")
    run_id = "four_batches_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid.uuid4().hex[:8]
    root = Path(output) if output else Path(config["output"])/run_id
    root.mkdir(parents=True, exist_ok=False)
    records, hashes = [], {}
    with threadpool_limits(limits=2):
        if batches & {"batch_01", "batch_02", "batch_03"}:
            found, sources = _surimi_export(root, config, batches)
            records.extend(found)
            hashes.update(sources)
        if "batch_04" in batches:
            found, sources = _public_export(root, config)
            records.extend(found)
            hashes.update(sources)
    pd.DataFrame(records).to_csv(root/"index.csv", index=False, encoding="utf-8-sig")
    write_json(root/"manifest.json", dict(run_id=run_id, configuration=config,
        config_sha256=file_hash(config_path), packages=[r["directory"] for r in records],
        source_sha256=hashes, prompt_version=PROMPT_VERSION, api_requests=0,
        actual_llm_predictions=0, python=platform.python_version(), sklearn=sklearn.__version__,
        code_sha256={str(Path(__file__)): file_hash(__file__),
            **{f"src/spectral_icl/{name}.py": file_hash(f"src/spectral_icl/{name}.py")
               for name in ["data", "contexts", "dimensionality", "representations", "public_data", "public_validation"]}},
        configuration_sha256={str(p):file_hash(p) for p in [Path(config_path), Path(config["surimi_config"]), Path(config["public_config"])]}))
    _readme(root, records)
    checksum(root)
    return root
