"""SNV reduction comparison: full training pool, Val dimension selection, locked Test."""
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import uuid
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import yaml

from .data import load_surimi, validate_split
from .dimensionality import SNVReduction, candidate_dimensions
from .io import digest, file_hash, write_csv, write_json
from .metrics import evaluate
from .models import make_model


def run_benchmark(config_path):
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(config["base_config"]).read_text(encoding="utf-8"))
    meta, x, wave, hashes = load_surimi(base)
    levels = base["data"]["levels"]
    created = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    root = Path(config["output"])/f"{created}_{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=False)
    versions = {p:importlib.metadata.version(p) for p in
        ["numpy","scipy","scikit-learn","pandas","umap-learn","pynndescent","openTSNE"]}
    source_hashes = {p.name:file_hash(p) for p in Path(__file__).parent.glob("*.py")}
    write_json(root/"plan_before_test.json", dict(config=config, dataset_source_hashes=hashes,
        code_hashes=source_hashes, dependencies=versions, python=sys.executable,
        created_at=datetime.now(timezone.utc).isoformat(), classification_only=True,
        note="All methods and dimension candidates declared before evaluation; no Test selection",
        full_training_pool_per_class=10, original_dimensions=x.shape[1]))
    all_scores, candidate_rows, selected_rows = [], [], []
    started = time.perf_counter()
    total = len(config["repeats"])*len(config["days"])*len(config["methods"])
    counter = 0
    with threadpool_limits(limits=2):
        for repeat in config["repeats"]:
            split_path = Path(base["splits"])/f"surimi_split_v1_repeat_{repeat}.csv"
            split = pd.read_csv(split_path)
            validate_split(meta, split)
            roles = meta.physical_sample_id.map(split.set_index("physical_sample_id").split)
            for day in config["days"]:
                indices = {part:meta.index[(meta.day == day) & (roles == part)].to_numpy()
                           for part in ["train","val","test"]}
                train, val = indices["train"], indices["val"]
                labels = meta.loc[train,"concentration_pct"].to_numpy()
                for method in config["methods"]:
                    counter += 1
                    stamp = time.perf_counter()
                    method_dir = root/f"repeat_{repeat}"/f"day_{day}"/method
                    method_dir.mkdir(parents=True)
                    dims = candidate_dimensions(method, config["dimensions_grid"],len(train),x.shape[1],len(levels))
                    local_candidates, best = [], None
                    for position, requested in enumerate(dims):
                        candidate = dict(repeat=repeat, day=day, method=method, requested_dimensions=requested,
                                         actual_dimensions=None, status="failed", failure_reason=None,
                                         warning_messages=[], validation_model_scores={},
                                         selection_score=None)
                        try:
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter("always")
                                reducer = SNVReduction(method,requested,config["seed"]+repeat)
                                z_train = np.asarray(reducer.fit_transform(x[train], labels))
                                z_val = np.asarray(reducer.transform(x[val]))
                                if not np.isfinite(z_train).all() or not np.isfinite(z_val).all():
                                    raise ValueError("Nonfinite reduced coordinates")
                                if z_train.shape[1] == 0 or z_val.shape[1] != z_train.shape[1]:
                                    raise ValueError("Invalid transformed dimension")
                                classifiers, val_scores = {}, []
                                for model in config["models"]:
                                    classifier = make_model(model,len(train),levels,config["seed"]+repeat)
                                    classifier.fit(z_train,labels)
                                    prediction = classifier.predict(z_val)
                                    score = float(np.mean(prediction == meta.loc[val,"concentration_pct"].to_numpy()))
                                    candidate["validation_model_scores"][model] = score
                                    val_scores.append(score)
                                    classifiers[model] = classifier
                                candidate["actual_dimensions"] = z_train.shape[1]
                                candidate["selection_score"] = float(np.mean(val_scores))
                            candidate["warning_messages"] = [str(w.message) for w in caught]
                            if any(issubclass(w.category,ConvergenceWarning) for w in caught):
                                raise RuntimeError("Reduction/model did not converge; candidate excluded from selection")
                            candidate["status"] = "success"
                            rank = (candidate["selection_score"],-z_train.shape[1],-position)
                            if best is None or rank > best[0]:
                                best = (rank,reducer,z_train,z_val,classifiers,dict(candidate))
                        except Exception as exc:
                            candidate["failure_reason"] = f"{type(exc).__name__}: {exc}"
                        local_candidates.append(candidate)
                        candidate_rows.append(candidate)
                    write_json(method_dir/"dimension_candidates.json",local_candidates)
                    selection = dict(repeat=repeat, day=day, method=method,
                        selected_dimensions=best[5]["actual_dimensions"] if best else None,
                        requested_dimensions=best[5]["requested_dimensions"] if best else None,
                        validation_selection_score=best[5]["selection_score"] if best else None,
                        split_sha256=file_hash(split_path),
                        train_fit_id_hash=digest(sorted(meta.loc[train,"physical_sample_id"])),
                        selection_rule=config["dimension_selection"], test_used_for_selection=False,
                        reference_budget="TRAIN_POOL_10_PER_CLASS", n_train=len(train),
                        method_note="SG(11,2) spectral residual + Ledoit-Wolf noise covariance; approximate noise model, not measured noise" if method == "mnf_estimated_noise" else
                            "one-hot class-label PLS scores, not scalar concentration regression" if method == "pls_scores" else
                            "openTSNE inductive mapping; Train reference embedding frozen" if method == "tsne_openTSNE" else "")
                    # Selection is persisted before any call to transform or predict Test.
                    write_json(method_dir/"selection_before_test.json",selection)
                    selected_rows.append(selection)
                    failure = None
                    feature_sets = {}
                    if best:
                        _,reducer,z_train,z_val,classifiers,_ = best
                        feature_sets = {"train":z_train,"val":z_val}
                        try:
                            if config["evaluate_test"]:
                                feature_sets["test"] = reducer.transform(x[indices["test"]])
                                if not np.isfinite(feature_sets["test"]).all():
                                    raise ValueError("Nonfinite Test embedding")
                            joblib.dump(reducer,method_dir/"reducer.joblib")
                        except Exception as exc:
                            failure = f"{type(exc).__name__}: {exc}"
                    else:
                        failure = "No valid converged dimensionality candidate; see dimension_candidates.json"
                    for model in config["models"]:
                        run_id = f"SNV_REDUCTION_{method}_{model}_r{repeat}_d{day}_{uuid.uuid4().hex[:10]}"
                        run_dir = method_dir/model
                        run_dir.mkdir()
                        for sub in ["prompts","responses","logs"]:
                            (run_dir/sub).mkdir()
                        status = "failed" if failure else "success" if config["evaluate_test"] else "development_only"
                        if not failure:
                            joblib.dump(classifiers[model],run_dir/"classifier.joblib")
                        for part, ids in indices.items():
                            pred = None
                            if not failure and part in feature_sets:
                                pred = classifiers[model].predict(feature_sets[part])
                            records = []
                            for j,i in enumerate(ids):
                                obs = meta.loc[i]
                                valid = pred is not None and float(pred[j]) in levels
                                records.append(dict(run_id=run_id, experiment_id="SNV_REDUCTION", dataset_id="surimi_2022",
                                    repeat=repeat, split=part, eval_mode="train_in_sample" if part=="train" else part,
                                    model_id=model, physical_sample_id=obs.physical_sample_id,
                                    acquisition_id=obs.acquisition_id, day=int(day), y_true=float(obs.concentration_pct),
                                    y_pred=float(pred[j]) if valid else None, prediction_valid=bool(valid),
                                    invalid_reason=failure if failure else "deferred_test" if pred is None else None,
                                    is_seen_class=True, n_components=selection["selected_dimensions"],
                                    representation_id="snv+"+method, reference_budget="TRAIN_POOL_10_PER_CLASS"))
                            metrics,per_class,confusion = evaluate(records,levels)
                            if part == "test" and not config["evaluate_test"]:
                                metrics = {"accuracy":None,"status":"deferred"}
                            metrics.update(run_id=run_id,split=part,status=status)
                            write_csv(run_dir/f"predictions_{part}.csv",records)
                            write_json(run_dir/f"metrics_{part}.json",metrics)
                            write_csv(run_dir/f"per_class_metrics_{part}.csv",per_class)
                            write_csv(run_dir/f"confusion_{part}.csv",confusion)
                            all_scores.append(dict(run_id=run_id,repeat=repeat,day=day,method=method,model=model,
                                split=part,status=status,failure_reason=failure,
                                dimensions=selection["selected_dimensions"],n=len(ids),
                                accuracy=metrics.get("accuracy"),macro_f1=metrics.get("macro_f1"),
                                balanced_accuracy=metrics.get("balanced_accuracy"),run_directory=str(run_dir)))
                        write_csv(run_dir/"support_context.csv",[
                            dict(reference_physical_sample_id=meta.loc[i,"physical_sample_id"],
                                 acquisition_id=meta.loc[i,"acquisition_id"],day=day,split="train",role="fitting_pool")
                            for i in train])
                        manifest = dict(run_id=run_id,status=status,failure_reason=failure,config=config,
                            selection=selection,source_hashes=hashes,code_hashes=source_hashes,
                            python=sys.executable,os=platform.platform(),dependencies=versions,
                            model_parameters=classifiers[model].get_params() if not failure and hasattr(classifiers[model],"get_params") else
                                {},
                            created_at=datetime.now(timezone.utc).isoformat(),api_requests=0)
                        write_json(run_dir/"manifest.json",manifest)
                        (run_dir/"logs/events.jsonl").write_text(json.dumps({"status":status,"failure_reason":failure})+"\n",encoding="utf-8")
                        checks = {str(p.relative_to(run_dir)):file_hash(p) for p in run_dir.rglob("*") if p.is_file()}
                        (run_dir/"checksum.sha256").write_text("\n".join(f"{v}  {k}" for k,v in sorted(checks.items())),encoding="utf-8")
                    pd.DataFrame(all_scores).to_csv(root/"all_runs.csv",index=False)
                    print(json.dumps(dict(progress=f"{counter}/{total}",repeat=repeat,day=day,method=method,
                        dimensions=selection["selected_dimensions"],status="failed" if failure else "success")),flush=True)
    pd.DataFrame(selected_rows).to_csv(root/"selected_dimensions.csv",index=False)
    flat_candidates = [{**r,"warning_messages":json.dumps(r["warning_messages"])} for r in candidate_rows]
    pd.DataFrame(flat_candidates).to_csv(root/"all_dimension_candidates.csv",index=False)
    scores = pd.DataFrame(all_scores)
    successful = scores[(scores.status=="success") & (scores.split=="test")]
    daily = successful.groupby(["day","method","model"]).agg(
        accuracy_mean=("accuracy","mean"),accuracy_std=("accuracy","std"),
        dimensions_min=("dimensions","min"),dimensions_max=("dimensions","max"),
        dimensions_selected=("dimensions",lambda s: ",".join(str(int(v)) for v in s)),
        n_repeats=("repeat","nunique")).reset_index()
    daily.to_csv(root/"test_summary_by_day.csv",index=False)
    write_json(root/"execution_summary.json",dict(root=str(root),model_runs=scores.run_id.nunique(),
        success_model_runs=scores[scores.status=="success"].run_id.nunique(),
        failed_model_runs=scores[scores.status=="failed"].run_id.nunique(),
        source_hashes_after={p:file_hash(p) for p in hashes},duration_seconds=time.perf_counter()-started,
        scope="Nine-class supervised full training pool benchmark, not few-shot and not regression",
        stability_note="Repeat standard deviations describe partition stability, not inferential uncertainty"))
    return root
