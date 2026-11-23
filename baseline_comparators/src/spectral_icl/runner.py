"""One immutable model run per condition; all partitions retain full coverage."""
import json
import platform
import sys
import time
import uuid
import importlib.metadata
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from .data import load_surimi, validate_split
from .contexts import select_support, t3_support
from .representations import Representation
from .models import make_model, REGISTRY
from .llm import FrozenAPI, make_prompt, parse_label, PROMPT_VERSION, load_provider, redact_error
from .lumir import lumir_messages, parse_lumir, NeuralClassifier
from .metrics import evaluate
from .io import digest, file_hash, write_json, write_csv, append_csv


def run(config, allow_api=False, evaluate_test=False):
    cfg = deepcopy(config)
    # Credentials belong in the ignored local Python file, never the serialized experiment config.
    for value in cfg.get("llm", {}).get("providers", {}).values():
        if value.get("api_key"):
            raise ValueError("Place api_key in local_llm_config.py, not the experiment YAML")
    resolved_providers = {}
    for name in cfg["models"]:
        if name in ("deepseek_icl", "mimo_icl", "lumir_reproduced", "lumir_mul_generate"):
            spec = load_provider(name, cfg.get("llm", {}).get("providers", {}).get(name, {}),
                                 cfg.get("llm", {}).get("local_config_path", "local_llm_config.py"))
            resolved_providers[name] = spec
            cfg.setdefault("llm", {}).setdefault("providers", {})[name] = {k:v for k,v in spec.items() if k != "api_key"}
    e = cfg["experiment"]
    # Input contract is checked before any experiment artifacts or API request.
    if e["id"] not in ("T1", "T2", "T3", "UNSEEN", "CONTEXT"):
        raise ValueError("Unknown experiment")
    if e["source_day"] not in (0, 7, 14) or e["target_day"] not in (0, 7, 14):
        raise ValueError("Invalid day")
    if e["k"] < 1 and not (e["id"] == "T3" and e["k"] == 0):
        raise ValueError("Invalid k")
    if e["k"] > (5 if e["id"] == "T3" else 10):
        raise ValueError("Support budget exceeds pool")
    if e["reference_pool_mode"] not in ("STRICT_K", "BANK_10_CONTEXT_K"):
        raise ValueError("Unknown reference budget protocol")
    if e["id"] == "T3" and e["reference_pool_mode"] != "STRICT_K":
        raise ValueError("T3 uses its explicit matched reference pools, not bank retrieval")
    if e["reference_pool_mode"] == "STRICT_K" and e["selector"] not in ("random", "class_balanced"):
        raise ValueError("Query-based bank retrieval is not STRICT_K")
    if e.get("mechanism_control", "none") != "none" and any(m not in ("deepseek_icl", "mimo_icl") for m in cfg["models"]):
        raise ValueError("Mechanism controls currently apply only to LLM prompt inputs")
    if allow_api and cfg["llm"].get("dry_run", True):
        raise ValueError("Set llm.dry_run=false as well as --allow-api for live calls")
    meta, x, wave, hashes = load_surimi(cfg)
    split_path = Path(cfg["splits"]) / f"surimi_split_v1_repeat_{e['repeat']}.csv"
    split = pd.read_csv(split_path)
    validate_split(meta, split)
    if set(split["repeat"]) != {e["repeat"]}:
        raise ValueError("Split repeat mismatch")
    meta["split"] = meta.physical_sample_id.map(split.set_index("physical_sample_id")["split"])
    heldout = set(e.get("heldout_levels", []))
    candidates = sorted(meta.concentration_pct.unique().tolist())
    if not heldout <= set(candidates) or len(heldout) == len(candidates):
        raise ValueError("Invalid unseen-level quarantine")
    if heldout and e["id"] != "UNSEEN":
        raise ValueError("Held-out levels require UNSEEN experiment")
    pool = meta.index[(meta.split == "train") & ~meta.concentration_pct.isin(heldout)].tolist()
    source_pool = [i for i in pool if meta.loc[i, "day"] == e["source_day"]]
    if e["id"] == "T3":
        fixed = t3_support(meta, pool, e["source_day"], e["target_day"], e["k"], cfg["seed"], e["t3_control"])
    else:
        fixed = select_support(meta, x, source_pool, e["k"], cfg["seed"])
    bank_mode = e["reference_pool_mode"] == "BANK_10_CONTEXT_K"
    fit_bank = source_pool if bank_mode and e["id"] != "T3" else fixed
    if not fit_bank:
        raise ValueError("Empty reference pool (target-only k=0 is not applicable)")
    root = Path(cfg["output"])
    condition = digest(cfg)
    lock = {"config_hash": condition, "split_hash": file_hash(split_path), "config": cfg,
            "test_evaluation": evaluate_test, "created_at": datetime.now(timezone.utc).isoformat()}
    # This records the exact pre-evaluation config, not a claim that method selection is complete.
    lock_path = root / "locks" / f"{uuid.uuid4().hex}.json"
    write_json(lock_path, lock)
    registry = []
    results = []
    for model_name in cfg["models"]:
        if model_name not in REGISTRY:
            raise ValueError(f"Unregistered model {model_name}")
        run_id = f"{e['id']}_r{e['repeat']}_{model_name}_{condition[:8]}_{uuid.uuid4().hex[:12]}"
        out = root / "runs" / run_id
        out.mkdir(parents=True, exist_ok=False)
        for sub in ("prompts", "responses", "logs"):
            (out / sub).mkdir()
        is_llm = model_name in ("deepseek_icl", "mimo_icl", "lumir_reproduced", "lumir_mul_generate")
        is_neural = model_name in ("cnn1d", "transformer")
        is_adaptive = model_name == "lumir_mul_generate"
        version = None
        provider = resolved_providers.get(model_name, {})
        failure = None
        predictions = {s: [] for s in ("train", "val", "test")}
        contexts, events = [], []
        start = time.perf_counter()
        api = None
        adaptive_refs = []
        source_adapter_hash = None
        fitted_predictions = {}
        try:
            if is_llm:
                if not provider.get("model") or not provider.get("base_url"):
                    raise ValueError("Exact API model ID and endpoint are not configured")
                version = provider["model"]
                api = FrozenAPI(provider, root / "cache", allow_api, cfg["llm"].get("max_requests", 0))
            else:
                version = importlib.metadata.version("scikit-learn")
            rep = Representation(e["representation"]).fit(x[fit_bank])
            features = rep.transform(x)
            if is_llm:
                estimator = None
            elif is_neural:
                estimator = NeuralClassifier(model_name, cfg["seed"], cfg.get("neural", {}), out / "best_checkpoint.pt")
            else:
                estimator = make_model(model_name, len(fit_bank), candidates, cfg["seed"], e.get("allow_endpoint_extrapolation", False))
            if estimator is not None:
                if is_neural:
                    v = meta.index[(meta.split == "val") & (meta.day == e["target_day"]) & ~meta.concentration_pct.isin(heldout)]
                    estimator.fit(features[fit_bank], meta.loc[fit_bank, "concentration_pct"].to_numpy(),
                                  (features[v], meta.loc[v, "concentration_pct"].to_numpy()))
                    write_json(out / "training_history.json", estimator.history)
                    source_adapter_hash = estimator.source_hash
                else:
                    estimator.fit(features[fit_bank], meta.loc[fit_bank, "concentration_pct"].to_numpy())
                # Batched inference avoids invoking a forest once per observation.
                eligible = meta.index[(meta.day == e["target_day"]) &
                    ((meta.split != "test") | evaluate_test) &
                    (~meta.concentration_pct.isin(heldout) | (meta.split == "test"))]
                fitted_predictions = dict(zip(eligible, estimator.predict(features[eligible]).tolist()))
        except Exception as exc:
            failure = redact_error(exc, provider)
        for partition in ("train", "val", "test"):
            queries = meta.index[(meta.split == partition) & (meta.day == e["target_day"])]
            for q in queries:
                obs = meta.loc[q]
                unseen = obs.concentration_pct in heldout
                mode = "train_leave_sample_out" if is_llm and partition == "train" else (
                    "train_in_sample" if q in fit_bank else "train_non_support") if partition == "train" else partition
                row = dict(run_id=run_id, experiment_id=e["id"], dataset_id="surimi_2022", repeat=e["repeat"],
                           split_version="v1", split=partition, eval_mode=mode, model_id=model_name, model_version=version,
                           physical_sample_id=obs.physical_sample_id, acquisition_id=obs.acquisition_id,
                           day=int(obs.day), source_day=e["source_day"], target_day=e["target_day"],
                           is_seen_class=not unseen, y_true=float(obs.concentration_pct), y_pred=None,
                           prediction_valid=False, invalid_reason=failure, k_per_class_requested=e["k"],
                           k_per_class_effective=0, num_context_examples=0, num_unique_context_ids=0,
                           context_hash=None, prompt_hash=None, representation_id=e["representation"],
                           selection_strategy=e["selector"], random_seed=cfg["seed"], latency_ms=0,
                           input_tokens=0, output_tokens=0, estimated_cost_usd=None, response_file=None,
                           prompt_file=None, reference_fit_ids_hash=digest(sorted(meta.loc[fit_bank, "acquisition_id"])),
                           target_k_per_class_requested=e["k"] if e["id"] == "T3" else 0,
                           num_source_context_examples=0, num_target_context_examples=0,
                           budget_exception=False)
                if is_neural and partition == "val":
                    row["eval_mode"] = "validation_for_early_stopping"
                if is_adaptive and partition in ("val", "test"):
                    row["eval_mode"] = "validation_adaptive_context" if partition == "val" else "test_after_validation_adaptation"
                if unseen and partition != "test":
                    row.update(eval_mode="not_applicable_unseen_class", invalid_reason="quarantined_unseen")
                elif partition == "test" and not evaluate_test:
                    row.update(eval_mode="deferred_test", invalid_reason="Test evaluation not enabled; selection must be frozen first")
                elif failure is None:
                    tick = time.perf_counter()
                    support = list(fit_bank)
                    try:
                        if is_llm:
                            legal = [i for i in (source_pool if bank_mode else fixed)
                                     if meta.loc[i, "physical_sample_id"] != obs.physical_sample_id]
                            if bank_mode:
                                support = select_support(meta, x, legal, e["k"], cfg["seed"], x[q], e["selector"])
                            else:
                                support = legal
                            # STRICT-K never replenishes excluded support IDs from unlabeled bank.
                            if not support:
                                raise ValueError("No legal leave-sample-out context")
                            fitting_indices = legal if bank_mode else support
                            local_rep = Representation(e["representation"]).fit(x[fitting_indices])
                            if is_adaptive and partition != "train":
                                support = support + [i for i in adaptive_refs if meta.loc[i, "physical_sample_id"] != obs.physical_sample_id]
                            sx = local_rep.transform(x[support])
                            qx = local_rep.transform(x[[q]])[0]
                            row["reference_fit_ids_hash"] = digest(sorted(meta.loc[fitting_indices, "acquisition_id"]))
                        if is_llm:
                            if model_name in ("lumir_reproduced", "lumir_mul_generate"):
                                messages, source_adapter_hash = lumir_messages(sx, meta.loc[support, "concentration_pct"].to_numpy(), qx, candidates, is_adaptive)
                            else:
                                messages = make_prompt(sx, meta.loc[support, "concentration_pct"].to_numpy(), qx,
                                                   candidates, e["representation"],
                                                   wave.tolist() if e["representation"] in ("raw", "snv", "sg") else None,
                                                   e.get("mechanism_control", "none"), cfg["seed"])
                            prompt_path = out / "prompts" / f"{partition}_{q}.json"
                            write_json(prompt_path, messages)
                            row.update(prompt_hash=digest(messages), prompt_file=str(prompt_path))
                            response, cached = api.predict(messages)
                            response_path = out / "responses" / f"{partition}_{q}.json"
                            write_json(response_path, {"cached": cached, "response": response})
                            row["response_file"] = str(response_path)
                            usage = response.get("usage") or {}
                            row.update(input_tokens=usage.get("prompt_tokens", 0), output_tokens=usage.get("completion_tokens", 0))
                            parser = parse_lumir if model_name in ("lumir_reproduced", "lumir_mul_generate") else parse_label
                            row["y_pred"] = parser(response["choices"][0]["message"]["content"], candidates)
                        else:
                            row["y_pred"] = float(fitted_predictions[q])
                        row["prediction_valid"] = row["y_pred"] in candidates
                    except Exception as exc:
                        row["invalid_reason"] = redact_error(exc, provider)
                        error_path = out / "responses" / f"error_{partition}_{q}.json"
                        write_json(error_path, {"error": row["invalid_reason"]})
                        if not row["response_file"]:
                            row["response_file"] = str(error_path)
                    # Fitted-model records describe fitting pool, not query-specific prompt examples.
                    actual_counts = meta.loc[support].groupby("concentration_pct").size()
                    seen_labels = set(candidates) - heldout
                    effective_k = min((int(actual_counts.get(c, 0)) for c in seen_labels), default=0)
                    row.update(k_per_class_effective=effective_k, num_context_examples=len(support),
                               num_source_context_examples=int((meta.loc[support, "day"] == e["source_day"]).sum()),
                               num_target_context_examples=int((meta.loc[support, "day"] == e["target_day"]).sum()) if e["source_day"] != e["target_day"] else 0,
                               num_unique_context_ids=meta.loc[support].physical_sample_id.nunique(),
                               context_hash=digest(meta.loc[support, "acquisition_id"].tolist()),
                               latency_ms=(time.perf_counter()-tick)*1000,
                               budget_exception=effective_k < (5+e["k"] if e["id"] == "T3" and e["t3_control"] in ("mixed", "size_matched_source") else e["k"]))
                    for order, i in enumerate(support):
                        ref = meta.loc[i]
                        valid_feedback = is_adaptive and i in adaptive_refs and ref.split == "val" and partition != "train"
                        if (ref.split != "train" and not valid_feedback) or ref.concentration_pct in heldout:
                            raise AssertionError("Illegal reference escaped quarantine")
                        if is_llm and ref.physical_sample_id == obs.physical_sample_id:
                            raise AssertionError("Self label leakage")
                        contexts.append(dict(query_acquisition_id=obs.acquisition_id, query_split=partition,
                                             reference_physical_sample_id=ref.physical_sample_id,
                                             reference_acquisition_id=ref.acquisition_id, reference_day=int(ref.day),
                                             reference_label=float(ref.concentration_pct), order=order,
                                             role="validation_feedback" if valid_feedback else "prompt" if is_llm else "fitting_pool"))
                    if is_adaptive and partition == "val" and row["prediction_valid"] and row["y_pred"] != row["y_true"]:
                        adaptive_refs.append(q)
                        events.append({"event": "validation_hard_example_added_after_prediction",
                                       "acquisition_id": obs.acquisition_id, "prediction": row["y_pred"]})
                predictions[partition].append(row)
        failed_rows = [r for rows in predictions.values() for r in rows if not r["prediction_valid"] and r["eval_mode"] not in ("not_applicable_unseen_class", "deferred_test")]
        status = "failed" if failure or failed_rows else "success" if evaluate_test else "development_only"
        if failed_rows and not failure:
            failure = f"{len(failed_rows)} invalid predictions; inspect responses and predictions"
        summary_rows = []
        for partition, rows in predictions.items():
            write_csv(out / f"predictions_{partition}.csv", rows)
            metrics, per_class, confusion = evaluate(rows, candidates)
            if partition == "test" and not evaluate_test:
                metrics = {"accuracy": None, "status": "deferred", "reason": "Test evaluation not enabled"}
                per_class, confusion = [], []
            metrics.update(run_id=run_id, split=partition, status=status,
                           eval_modes=sorted({r["eval_mode"] for r in rows}))
            if partition == "train" and not is_llm:
                metrics["by_eval_mode"] = {m: evaluate([r for r in rows if r["eval_mode"] == m], candidates)[0]
                                           for m in ("train_in_sample", "train_non_support")}
            write_json(out / f"metrics_{partition}.json", metrics)
            write_csv(out / f"per_class_metrics_{partition}.csv", per_class, ["label", "support", "precision", "recall", "f1"])
            write_csv(out / f"confusion_{partition}.csv", confusion, ["y_true", "y_pred", "count"])
            summary_rows.append(dict(run_id=run_id, model_id=model_name, experiment_id=e["id"],
                repeat=e["repeat"], n_repeats_expected=cfg.get("repeats", 5), seed=cfg["seed"], split=partition,
                source_day=e["source_day"], target_day=e["target_day"], k=e["k"],
                representation=e["representation"], selector=e["selector"], reference_pool_mode=e["reference_pool_mode"],
                heldout_levels=json.dumps(sorted(heldout)), t3_control=e.get("t3_control"),
                mechanism_control=e.get("mechanism_control", "none"), status=status, failure_reason=failure,
                config_hash=condition, split_hash=file_hash(split_path),
                **{key: metrics.get(key) for key in ("accuracy", "macro_f1", "balanced_accuracy", "invalid_prediction_rate", "seen_accuracy", "unseen_accuracy", "seen_unseen_harmonic_mean")}))
        write_csv(out / "support_context.csv", contexts, ["query_acquisition_id", "query_split", "reference_physical_sample_id", "reference_acquisition_id", "reference_day", "reference_label", "order", "role"])
        code_files = sorted(Path(__file__).parent.glob("*.py"))
        manifest = dict(run_id=run_id, created_at=datetime.now(timezone.utc).isoformat(), config=cfg,
                        config_hash=condition, split_hash=file_hash(split_path), source_hashes=hashes,
                        code_hashes={p.name: file_hash(p) for p in code_files}, python=sys.executable,
                        python_version=sys.version, os=platform.platform(),
                        dependencies={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "pandas", "openpyxl")},
                        status=status, failure_reason=failure, duration_seconds=time.perf_counter()-start,
                        model_version=version, prompt_version=PROMPT_VERSION,
                        provider={k:v for k,v in provider.items() if k != "api_key"},
                        validation_adaptive_context=is_adaptive,
                        adaptive_reference_ids=meta.loc[adaptive_refs, "acquisition_id"].tolist(),
                        adaptive_protocol="single_pass_post_prediction_hard_example_feedback; not original multi-round replay" if is_adaptive else None,
                        legacy_adapter_source_hash=source_adapter_hash,
                        actual_reference_ids=meta.loc[fit_bank, "acquisition_id"].tolist(),
                        labeled_reference_pool_size=len(fit_bank), frozen_llm=is_llm,
                        test_evaluated=evaluate_test, pre_evaluation_lock=str(lock_path),
                        actual_api_requests=api.calls if api is not None else 0,
                        cost_usd=None, cost_reason="No verified provider tariff configured",
                        comparison_scope="seen-only non-equivalent control" if heldout and model_name in ("svm_rbf", "random_forest") else "standard")
        write_json(out / "manifest.json", manifest)
        events.append({"event": "run_finished", "status": status, "failure_reason": failure})
        (out / "logs" / "events.jsonl").write_text("".join(json.dumps(event)+"\n" for event in events), encoding="utf-8")
        checksums = {str(p.relative_to(out)): file_hash(p) for p in out.rglob("*") if p.is_file()}
        (out / "checksum.sha256").write_text("\n".join(f"{v}  {k}" for k, v in sorted(checksums.items())), encoding="utf-8")
        append_csv(root / "summary" / "all_runs.csv", summary_rows)
        entry = dict(experiment_id=e["id"], repeat=e["repeat"], seed=cfg["seed"], run_id=run_id,
                     model_id=model_name, status=status, failure_reason=failure, config_hash=condition,
                     split_hash=file_hash(split_path))
        entry.update({f"{s}_predictions_path": str(out / f"predictions_{s}.csv") for s in predictions})
        entry.update({f"{s}_metrics_path": str(out / f"metrics_{s}.json") for s in predictions})
        registry.append(entry)
        results.append({"run_id": run_id, "status": status, "failure_reason": failure})
    append_csv(root / "summary" / "model_registry.csv", registry)
    with (root / "summary" / "experiment_ledger.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"lock": str(lock_path), "runs": results})+"\n")
    return results
