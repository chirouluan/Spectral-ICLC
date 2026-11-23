"""Paper-specific representations, immutable public-data exports and local baselines.

No API calls. Validation may select PLS dimensionality; Test never fits/selects.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import platform
import time
import uuid
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from threadpoolctl import threadpool_limits

from .io import digest, file_hash, write_json
from .models import make_model
from .public_data import load_public, public_split, read_public_config
from .web_policy import CONTEXT_ONLY_POLICY


def linear_baseline(x, axis):
    fraction = (axis-axis[0])/(axis[-1]-axis[0])
    return x-(x[:, :1]+(x[:, -1:]-x[:, :1])*fraction)


def area_normalize(x, axis):
    area = np.trapezoid(x, axis=1, x=axis) if hasattr(np, "trapezoid") else np.trapz(x, axis=1, x=axis)
    if (np.abs(area) < 1e-14).any():
        raise ValueError("Zero integrated spectral area")
    return x/area[:, None]


class PaperRepresentation:
    def __init__(self, method):
        self.method = method

    def preprocess(self, x):
        x = np.asarray(x, dtype=float)[:, self.mask_].copy()
        axis = self.axis_[self.mask_]
        if self.method == "snv_full":
            sd = x.std(axis=1, ddof=1, keepdims=True)
            if (sd <= 1e-14).any():
                raise ValueError("Constant spectrum cannot be SNV normalized")
            x = (x-x.mean(axis=1, keepdims=True))/sd
        elif self.method == "fruit_pls14":
            x = area_normalize(x-x[:, -1:], axis)
        elif self.method == "meat_area_pca2":
            x = area_normalize(linear_baseline(x, axis), axis)
        elif self.method == "turmeric_selected_pca2":
            # Normalize on the full fingerprint interval BEFORE taking disjoint bands.
            x = linear_baseline(x, axis)
            norm = np.linalg.norm(x, axis=1, keepdims=True)
            if (norm <= 1e-14).any():
                raise ValueError("Zero baseline-corrected Raman signal")
            x /= norm
            x = x[:, self.selected_mask_]
        return x

    def fit(self, train_x, train_y, axis, val_x=None, val_y=None):
        self.axis_ = np.asarray(axis)
        self.mask_ = np.ones(len(axis), dtype=bool)
        if self.method == "meat_area_pca2":
            self.mask_ = (self.axis_ >= 1000) & (self.axis_ <= 1800)
        if self.method == "turmeric_selected_pca2":
            # Workbook named 1750-200 has rounded endpoints 1750.022 and 199.5076.
            self.selected_mask_ = np.zeros(len(axis), dtype=bool)
            for low, high in [(782, 907), (396, 404), (321, 383)]:
                self.selected_mask_ |= (self.axis_ >= low) & (self.axis_ <= high)
        self.classes_ = np.unique(train_y)
        x = self.preprocess(train_x)
        self.preprocessed_dimensions_ = x.shape[1]
        self.selection_trace_ = []
        self.reducer_ = None
        if self.method in {"coffee_pca238", "meat_area_pca2", "turmeric_selected_pca2"}:
            count = 8 if self.method == "coffee_pca238" else 2
            self.reducer_ = PCA(n_components=count, svd_solver="full").fit(x)
            self.take_ = np.array([1, 2, 7]) if self.method == "coffee_pca238" else np.arange(count)
            self.feature_names_ = [f"PC{i+1}" for i in self.take_]
        elif self.method in {"pls_scores", "fruit_pls14"}:
            y = self._targets(train_y)
            if self.method == "fruit_pls14":
                count = 14
            else:
                if val_x is None or val_y is None:
                    raise ValueError("PLS dimension selection requires Validation, never Test")
                vx = self.preprocess(val_x)
                limit = min(15, len(x)-len(self.classes_), x.shape[1], np.linalg.matrix_rank(x-x.mean(0)))
                for n in range(1, limit+1):
                    reducer = PLSRegression(n_components=n, scale=False).fit(x, y)
                    tx, vz = reducer.transform(x), reducer.transform(vx)
                    lda = self.paper_model().fit(tx, train_y)
                    score = float(accuracy_score(val_y, lda.predict(vz)))
                    self.selection_trace_.append(dict(dimensions=n, validation_accuracy=score))
                if not self.selection_trace_:
                    raise ValueError("Insufficient Train rank for PLS")
                count = max(self.selection_trace_, key=lambda row: (row["validation_accuracy"], -row["dimensions"]))["dimensions"]
            self.reducer_ = PLSRegression(n_components=count, scale=False).fit(x, y)
            self.feature_names_ = [f"PLS{i+1}" for i in range(count)]
        elif self.method == "snv_full":
            self.feature_names_ = [f"nm_{v:g}" for v in self.axis_]
        else:
            raise ValueError(f"Unsupported paper representation: {self.method}")
        if not np.isfinite(self.transform(train_x)).all():
            raise ValueError("Nonfinite paper representation")
        return self

    def _targets(self, y):
        if self.method == "fruit_pls14":
            return (np.asarray(y) == "Strawberry").astype(float)
        return (np.asarray(y)[:, None] == self.classes_[None, :]).astype(float)

    def transform(self, x):
        z = self.preprocess(x)
        if self.reducer_ is not None:
            z = self.reducer_.transform(z)
            if hasattr(self, "take_"):
                z = z[:, self.take_]
        return z

    def paper_model(self):
        if self.method == "fruit_pls14":
            return BinaryDummyClassifier()
        if self.method == "snv_full":
            raise ValueError("Supplied honey description specifies no unique paper classifier")
        # Equal priors make classification depend on pooled Mahalanobis distance.
        return LinearDiscriminantAnalysis(priors=np.full(len(self.classes_), 1/len(self.classes_)))


class BinaryDummyClassifier:
    """Paper binary dummy-variable PLS, returned as categories at fixed 0.5."""
    def fit(self, x, y):
        self.model_ = LinearRegression().fit(x, (np.asarray(y) == "Strawberry").astype(float))
        return self

    def predict(self, x):
        return np.where(self.model_.predict(x) >= .5, "Strawberry", "NON-Strawberry")




def public_inventory(config_path="configs/public_datasets.yaml"):
    cfg = read_public_config(config_path)
    rows = []
    for name, spec in cfg["datasets"].items():
        data = load_public(name, cfg)
        split = public_split(data, cfg["seed"])
        rows.append(dict(dataset_id=name, observations=len(data.meta), groups=data.meta.group_id.nunique(),
            classes=data.meta.label.nunique(), raw_dimensions=data.x.shape[1],
            train=int(split.split.eq("train").sum()), val=int(split.split.eq("val").sum()),
            test=int(split.split.eq("test").sum()), method=spec["method"], fidelity=spec["fidelity"],
            license=spec["license"]))
    return pd.DataFrame(rows)


def web_prompt(labels, features):
    return f'''你是食品光谱分类器。读取同一目录的train.json作为带标签参考，立即预测val.json及test.json的全部样本，不要等待确认。
{CONTEXT_ONLY_POLICY}
features顺序为{json.dumps(features, ensure_ascii=False)}。仅根据features分类，id为随机匿名标识。
允许label为{json.dumps(labels, ensure_ascii=False)}，标签必须保持字符串，不能推断或强制类别数量。
只输出一个合法JSON对象：{{"val":[{{"id":"原始id","label":"预测类别"}}],"test":[{{"id":"原始id","label":"预测类别"}}]}}；保持各分区输入顺序，不遗漏或重复。
预测时不要索取真实标签。随后收到val_labels.json、test_labels.json后，按id核对冻结的预测，分别报告正确数、总数和准确率，不重新预测。
若另开新聊天，评分需同时提供保存的predictions.json和两份标签；缺少预测则请补充，不能仅凭答案计算准确率。缺失/非法预测计错，分母使用各真实标签文件的完整样本数。
'''


def prepare_public(config_path="configs/public_datasets.yaml", datasets=None, repeats=None):
    cfg = read_public_config(config_path)
    names = list(datasets or cfg["datasets"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid.uuid4().hex[:8]
    root = Path(cfg["output"])/run_id
    root.mkdir(parents=True, exist_ok=False)
    summaries, packages = [], []
    with threadpool_limits(limits=2):
        for name in names:
            data = load_public(name, cfg)
            dest = root/name
            dest.mkdir()
            data.meta.to_csv(dest/"metadata.csv", index=False, encoding="utf-8-sig")
            np.savez_compressed(dest/"raw_spectra.npz", x=data.x, axis=data.axis)
            write_json(dest/"source_audit.json", dict(**data.audit, source_sha256=data.source_hashes,
                specification=data.spec, paper_path=str(Path(cfg["raw_root"])/data.spec["folder"]/data.spec["paper"])))
            # Honey has one prescribed acquisition split, not five identical repeats.
            count = 1 if data.spec["loader"] == "honey" else (repeats or cfg["repeats"])
            for repeat in range(count):
                seed = cfg["seed"]+repeat
                split = public_split(data, seed)
                roles = split.split.to_numpy()
                train, val = roles == "train", roles == "val"
                model = PaperRepresentation(data.spec["method"]).fit(data.x[train], data.meta.label.to_numpy()[train],
                    data.axis, data.x[val], data.meta.label.to_numpy()[val])
                z = model.transform(data.x)
                if not np.isfinite(z).all():
                    raise ValueError("Nonfinite exported feature")
                package = dest/f"repeat_{repeat+1}"
                package.mkdir()
                split.to_csv(package/"split.csv", index=False, encoding="utf-8-sig")
                joblib.dump(model, package/"representation.joblib")
                np.savez_compressed(package/"features.npz", x=z, y=data.meta.label.to_numpy(dtype=str), split=roles.astype(str))
                exported = []
                rng = np.random.default_rng(seed)
                for part in ["train", "val", "test"]:
                    indices = rng.permutation(np.flatnonzero(roles == part))
                    samples, answers = [], []
                    for i in indices:
                        anon = "ID-"+digest([name, repeat, data.meta.sample_id.iloc[i]])[:16].upper()
                        features = np.round(z[i], 10).tolist()
                        label = str(data.meta.label.iloc[i])
                        sample = dict(id=anon, features=features)
                        if part == "train":
                            sample["label"] = label
                        samples.append(sample)
                        answers.append(dict(id=anon, label=label))
                        exported.append(dict(id=anon, sample_id=data.meta.sample_id.iloc[i], group_id=data.meta.group_id.iloc[i], split=part))
                    write_json(package/f"{part}.json", samples)
                    if part != "train":
                        write_json(package/f"{part}_labels.json", answers)
                pd.DataFrame(exported).to_csv(package/"private_id_mapping.csv", index=False, encoding="utf-8-sig")
                labels = sorted(data.meta.label.unique())
                (package/"prompt.txt").write_text(web_prompt(labels, model.feature_names_), encoding="utf-8")
                info = dict(dataset_id=name, repeat=repeat+1, seed=seed, classes=labels,
                    train=int(train.sum()), val=int(val.sum()), test=int((roles == "test").sum()),
                    raw_dimensions=data.x.shape[1], preprocessed_dimensions=model.preprocessed_dimensions_,
                    output_dimensions=z.shape[1], feature_order=model.feature_names_,
                    method=data.spec["method"], fidelity=data.spec["fidelity"], caveats=data.spec["caveats"],
                    split_protocol=data.spec["split"], source_sha256=data.source_hashes,
                    split_sha256=file_hash(package/"split.csv"), representation_sha256=file_hash(package/"representation.joblib"),
                    train_fit_sample_ids=data.meta.sample_id[train].tolist(),
                    train_fit_ids_sha256=digest(sorted(data.meta.sample_id[train])),
                    validation_used_for_dimension_selection=bool(model.selection_trace_),
                    dimension_selection=model.selection_trace_, test_used_in_fitting_or_selection=False,
                    label_budget="FULL_TRAIN; supervised PLS fits all Train labels; not STRICT_K", api_requests=0)
                write_json(package/"metadata.json", info)
                summaries.append({k: info[k] for k in ["dataset_id", "repeat", "classes", "train", "val", "test", "raw_dimensions", "preprocessed_dimensions", "output_dimensions", "method", "fidelity"]})
                packages.append(str(package.relative_to(root)))
            print(f"Prepared {name}: {len(data.meta)} spectra; {count} partition(s)", flush=True)
    pd.DataFrame(summaries).to_csv(root/"inventory.csv", index=False, encoding="utf-8-sig")
    write_json(root/"manifest.json", dict(run_id=run_id, config=cfg, config_sha256=file_hash(config_path),
        packages=packages, api_requests=0, python=platform.python_version(), sklearn=sklearn.__version__,
        code_sha256={str(Path(__file__)): file_hash(__file__), str(Path(__file__).with_name("public_data.py")): file_hash(Path(__file__).with_name("public_data.py"))}))
    checksum(root)
    return root


def checksum(root):
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "checksum.sha256")
    (root/"checksum.sha256").write_text("".join(f"{file_hash(p)}  {p.relative_to(root).as_posix()}\n" for p in files), encoding="utf-8")


def _score(frame, labels, path, part):
    report = classification_report(frame.y_true, frame.y_pred, labels=labels, output_dict=True, zero_division=0)
    metrics = dict(accuracy=float(accuracy_score(frame.y_true, frame.y_pred)),
        macro_f1=float(np.mean([report[label]["f1-score"] for label in labels])),
        balanced_accuracy=float(np.mean([report[label]["recall"] for label in labels])),
        n_observations=len(frame), n_groups=frame.group_id.nunique(),
        eval_mode="in_sample" if part == "train" else "holdout",
        unit="spectrum; grouped partition; not independent spectrum replicates")
    write_json(path/f"metrics_{part}.json", metrics)
    pd.DataFrame([dict(label=label, **report[label]) for label in labels]).to_csv(path/f"per_class_metrics_{part}.csv", index=False)
    pd.DataFrame(confusion_matrix(frame.y_true, frame.y_pred, labels=labels), index=labels, columns=labels).to_csv(path/f"confusion_{part}.csv")
    # Each grouping unit contributes equally. This complements observation-weighted accuracy.
    group_scores = frame.assign(correct=frame.y_true == frame.y_pred).groupby("group_id").correct.mean()
    metrics["mean_group_accuracy"] = float(group_scores.mean())
    rng = np.random.default_rng(12345)
    boot = [float(rng.choice(group_scores.to_numpy(), len(group_scores), replace=True).mean()) for _ in range(1000)]
    metrics["group_mean_accuracy_ci95"] = np.quantile(boot, [.025, .975]).tolist()
    metrics["bootstrap_scope"] = "available group IDs, conditional on this split; missing preparation IDs remain unresolved"
    write_json(path/f"metrics_{part}.json", metrics)
    return metrics


def run_public_baselines(prepared_root, output=None):
    """Use frozen prepared representations. Record every requested model, including failures."""
    prepared_root = Path(prepared_root)
    for line in (prepared_root/"checksum.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if file_hash(prepared_root/relative) != expected:
            raise ValueError(f"Prepared artifact checksum mismatch: {relative}")
    prepared = json.loads((prepared_root/"manifest.json").read_text(encoding="utf-8"))
    cfg = prepared["config"]
    root = Path(output) if output else Path(cfg["results"])/(prepared["run_id"]+"_"+uuid.uuid4().hex[:6])
    root.mkdir(parents=True, exist_ok=False)
    records, registry = [], []
    with threadpool_limits(limits=2):
        for relative in prepared["packages"]:
            package = prepared_root/relative
            info = json.loads((package/"metadata.json").read_text(encoding="utf-8"))
            if file_hash(package/"split.csv") != info["split_sha256"] or file_hash(package/"representation.joblib") != info["representation_sha256"]:
                raise ValueError("Prepared split/transform checksum mismatch")
            values = np.load(package/"features.npz", allow_pickle=False)
            x, y, roles = values["x"], values["y"], values["split"]
            split = pd.read_csv(package/"split.csv", dtype={"label":str})
            if not np.array_equal(split.label.to_numpy(), y) or not np.array_equal(split.split.to_numpy(), roles):
                raise ValueError("Prepared features/split are misaligned")
            train = roles == "train"
            labels = info["classes"]
            for name in cfg["models"]:
                path = root/relative/name
                path.mkdir(parents=True)
                start = time.monotonic()
                status, reason = "complete", None
                fit_warnings = []
                estimator = None
                try:
                    if name == "paper_classifier":
                        representation = joblib.load(package/"representation.joblib")
                        estimator = representation.paper_model()
                    else:
                        estimator = make_model(name, int(train.sum()), labels, info["seed"])
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter("always")
                        estimator.fit(x[train], y[train])
                    fit_warnings = [dict(category=type(w.message).__name__, message=str(w.message)) for w in captured]
                    joblib.dump(estimator, path/"estimator.joblib")
                    for part in ["train", "val", "test"]:
                        mask = roles == part
                        pred = estimator.predict(x[mask])
                        frame = split.loc[mask, ["sample_id", "group_id"]].copy()
                        frame["y_true"], frame["y_pred"] = y[mask], pred
                        frame["prediction_valid"] = np.isin(pred, labels)
                        frame["eval_mode"] = "in_sample" if part == "train" else "holdout"
                        frame.to_csv(path/f"predictions_{part}.csv", index=False, encoding="utf-8-sig")
                        metrics = _score(frame, labels, path, part)
                        records.append(dict(dataset_id=info["dataset_id"], repeat=info["repeat"], model_id=name,
                            split=part, status="complete", dimensions=x.shape[1], **metrics, result_directory=str(path)))
                except Exception as error:
                    status = "not_applicable" if name == "paper_classifier" and info["method"] == "snv_full" else "failed"
                    reason = f"{type(error).__name__}: {error}"
                    for part in ["train", "val", "test"]:
                        write_json(path/f"metrics_{part}.json", dict(status=status, reason=reason, accuracy=None))
                run_manifest = dict(run_id=f"{prepared['run_id']}_{info['dataset_id']}_r{info['repeat']}_{name}",
                    dataset_id=info["dataset_id"], repeat=info["repeat"], model_id=name, status=status, reason=reason,
                    prepared_package=str(package.resolve()), prepared_metadata_sha256=file_hash(package/"metadata.json"),
                    source_sha256=info["source_sha256"], split_sha256=info["split_sha256"], seed=info["seed"],
                    label_budget=info["label_budget"], feature_order=info["feature_order"],
                    train_score_mode="in_sample", validation_used_for_dimension_selection=info["validation_used_for_dimension_selection"],
                    actual_api_requests=0, elapsed_seconds=time.monotonic()-start, sklearn=sklearn.__version__,
                    python=platform.python_version(), code_sha256=file_hash(__file__), fit_warnings=fit_warnings,
                    estimator_parameters=estimator.get_params() if hasattr(estimator, "get_params") else
                        (estimator.model_.get_params() if hasattr(estimator, "model_") else {}))
                write_json(path/"manifest.json", run_manifest)
                checksum(path)
                registry.append(dict(dataset_id=info["dataset_id"], repeat=info["repeat"], model_id=name,
                    status=status, reason=reason, result_directory=str(path)))
            for name in ["deepseek_icl", "mimo_icl", "lumir_reproduced", "lumir_mul_generate", "cnn1d", "transformer"]:
                registry.append(dict(dataset_id=info["dataset_id"], repeat=info["repeat"], model_id=name,
                    status="not_run", reason="This public runner executes local classical baselines only; no API/neural adapter run", result_directory=None))
            print(f"Scored {relative}", flush=True)
    pd.DataFrame(records).to_csv(root/"all_runs.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(registry).to_csv(root/"model_registry.csv", index=False, encoding="utf-8-sig")
    write_json(root/"manifest.json", dict(prepared_root=str(prepared_root.resolve()),
        prepared_manifest_sha256=file_hash(prepared_root/"manifest.json"), config=cfg, api_requests=0,
        test_evaluated_after_freezing=True, registered_model_status_counts=pd.Series([r["status"] for r in registry]).value_counts().to_dict()))
    checksum(root)
    return root
