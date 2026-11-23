"""Small shared operations for the four batch notebooks."""
from copy import deepcopy
import json
from pathlib import Path
import uuid

import pandas as pd
from .data import load_surimi, validate_split
from .io import digest, file_hash, write_json
from .llm import load_provider
from .models import REGISTRY
from .runner import run

LLM_MODELS = {"deepseek_icl", "mimo_icl", "lumir_reproduced", "lumir_mul_generate"}
METHOD_FIELDS = ["representation", "selector", "reference_pool_mode"]


def check_data(config):
    meta, x, wave, _ = load_surimi(config)
    for repeat in range(config["repeats"]):
        path = Path(config["splits"]) / f"surimi_split_v1_repeat_{repeat}.csv"
        validate_split(meta, pd.read_csv(path))
    return pd.DataFrame([dict(physical_samples=meta.physical_sample_id.nunique(),
        observations=len(meta), bands=x.shape[1], wavelength_min_nm=wave[0],
        wavelength_max_nm=wave[-1], split_repeats=config["repeats"], status="passed")])


def model_status(config, selected):
    rows = []
    for name in REGISTRY:
        row = {"model": name, "selected": name in selected, "type": "frozen_llm" if name in LLM_MODELS else "fitted_baseline"}
        if name in LLM_MODELS:
            spec = load_provider(name, config.get("llm", {}).get("providers", {}).get(name, {}))
            row.update(exact_model=spec.get("model"), url_filled=bool(spec.get("base_url")),
                       local_key_filled=bool(spec.get("api_key")))
        rows.append(row)
    return pd.DataFrame(rows)


def make_plan(config, conditions, models, repeats, seeds):
    if not models or not set(models) <= set(REGISTRY):
        raise ValueError("Select at least one registered model")
    jobs = []
    for repeat in repeats:
        if repeat not in range(config["repeats"]):
            raise ValueError("Repeat outside frozen split range")
        for seed in seeds:
            for condition in conditions:
                for name in models:
                    cfg = deepcopy(config)
                    cfg["experiment"].update(condition, repeat=repeat)
                    cfg.update(models=[name], seed=seed)
                    # Never carry inline credentials into a Notebook plan or saved output.
                    for spec in cfg.get("llm", {}).get("providers", {}).values():
                        if spec.get("api_key"):
                            raise ValueError("Fill credentials in local_llm_config.py")
                    jobs.append(cfg)
    return jobs


def plan_table(jobs, evaluate_test=False):
    rows = []
    for cfg in jobs:
        e = cfg["experiment"]
        all_classes = len(cfg["data"]["levels"])
        seen_classes = all_classes-len(e.get("heldout_levels", []))
        requests = seen_classes*12 + (all_classes*4 if evaluate_test else 0)
        rows.append(dict(experiment=e["id"], model=cfg["models"][0], repeat=e["repeat"],
            seed=cfg["seed"], source_day=e["source_day"], target_day=e["target_day"],
            k=e["k"], representation=e["representation"], selector=e["selector"],
            budget_mode=e["reference_pool_mode"], t3_control=e.get("t3_control"),
            heldout=str(e.get("heldout_levels", [])), mechanism=e.get("mechanism_control", "none"),
            max_new_api_requests=requests if cfg["models"][0] in LLM_MODELS else 0))
    return pd.DataFrame(rows)


def save_method(config, path, *, quarantine=()):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Method lock already exists; choose a new version")
    providers = {}
    for name in config["models"]:
        if name in LLM_MODELS:
            spec = load_provider(name, config["llm"].get("providers", {}).get(name, {}))
            if not spec.get("model") or not spec.get("base_url"):
                raise ValueError(f"Configure exact model and endpoint for {name} before locking")
            providers[name] = {k:v for k,v in spec.items() if k != "api_key"}
    write_json(path, {"method": {key:config["experiment"][key] for key in METHOD_FIELDS},
        "models": config["models"], "providers": providers,
        "selection_quarantined_levels": sorted(quarantine),
        "mapping_sha256": file_hash(config["data"]["mapping"]),
        "split_manifest_sha256": file_hash(Path(config["splits"])/"manifest.json")})
    return str(path)


def apply_method(config, path, *, required=False, quarantine=()):
    cfg = deepcopy(config)
    path = Path(path)
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Create method selection lock first: {path}")
        return cfg
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock["selection_quarantined_levels"] != sorted(quarantine):
        raise ValueError("Method selection quarantine does not match this experiment")
    if lock["mapping_sha256"] != file_hash(cfg["data"]["mapping"]) or lock["split_manifest_sha256"] != file_hash(Path(cfg["splits"])/"manifest.json"):
        raise ValueError("Data mapping or frozen splits changed after method selection")
    for name, saved in lock["providers"].items():
        actual = load_provider(name, cfg["llm"].get("providers", {}).get(name, {}))
        if any(actual.get(k) != saved.get(k) for k in ("model", "base_url")):
            raise ValueError(f"Provider changed after method selection: {name}")
    cfg["experiment"].update(lock["method"])
    cfg["models"] = lock["models"]
    return cfg


def execute_plan(jobs, root, *, execute=False, allow_api=False, api_budget=0,
                 evaluate_test=False, method_confirmed=False):
    """One model/condition per output root: T2/T3 processes cannot collide."""
    if not execute:
        return pd.DataFrame([{"status":"preview_only", "jobs":len(jobs), "api_requests":0}])
    if evaluate_test and not method_confirmed:
        raise ValueError("Confirm a locked, validation-selected method before Test evaluation")
    estimates = plan_table(jobs, evaluate_test)
    needed = int(estimates.max_new_api_requests.sum()) if len(estimates) else 0
    if needed and (not allow_api or api_budget < needed):
        raise ValueError(f"Selected LLM jobs require explicit ALLOW_PAID_API=True and total API budget >= {needed}; no jobs started")
    for cfg in jobs:
        for name in cfg["models"]:
            if name in LLM_MODELS:
                spec = load_provider(name, cfg["llm"].get("providers", {}).get(name, {}))
                if not all(spec.get(k) for k in ("model", "base_url")):
                    raise ValueError(f"Missing exact model or endpoint: {name}")
    session = Path(root)/uuid.uuid4().hex[:12]
    # Freeze all requested conditions before running any model.
    write_json(session/"plan.json", {"jobs":jobs, "evaluate_test":evaluate_test,
        "allow_paid_api":allow_api, "total_api_request_budget":api_budget})
    rows, used = [], 0
    for i, original in enumerate(jobs):
        cfg = deepcopy(original)
        output = session/f"job_{i:04d}_{digest(cfg)[:8]}"
        cfg["output"] = str(output)
        cfg["llm"].update(dry_run=not allow_api, max_requests=max(0, api_budget-used))
        result = run(cfg, allow_api=allow_api, evaluate_test=evaluate_test)
        for record in result:
            manifest_path = output/"runs"/record["run_id"]/"manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            used += manifest["actual_api_requests"] or 0
            rows.append({**record, "output_root":str(output), "api_requests":manifest["actual_api_requests"]})
        print(f"{i+1}/{len(jobs)}: {cfg['experiment']['id']} {cfg['models'][0]} -> {result[0]['status']}", flush=True)
    return pd.DataFrame(rows)


def read_results(roots, include_test=False):
    tables = []
    for root in roots:
        for path in Path(root).rglob("all_runs.csv"):
            frame = pd.read_csv(path)
            frame["result_file"] = str(path)
            tables.append(frame)
    if not tables:
        return pd.DataFrame()
    frame = pd.concat(tables, ignore_index=True).drop_duplicates(["run_id", "split"])
    return frame if include_test else frame[frame.split != "test"].reset_index(drop=True)


def audit_external_manifest(path):
    """Metadata admission only; does not pretend the surimi runner supports external data."""
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame([{"status":"not_configured", "next_step":"Provide audited public dataset manifest; external classifier adapter is pending"}])
    data = json.loads(path.read_text(encoding="utf-8"))
    required = ["dataset_id", "source_url", "license", "classification_label_definition",
                "physical_id_definition", "data_path"]
    rows = []
    for record in data:
        missing = [key for key in required if not record.get(key)]
        if record.get("data_path") and not Path(record["data_path"]).is_file():
            missing.append("existing_data_file")
        rows.append({"dataset_id":record.get("dataset_id"),
                     "status":"metadata_complete_adapter_pending" if not missing else "blocked",
                     "missing":", ".join(missing)})
    return pd.DataFrame(rows)
