"""CLI: run from repo root after editable install, or with PYTHONPATH=src."""
import argparse
import json
from pathlib import Path
import pandas as pd
import yaml
from .data import load_surimi, audit_data, create_splits
from .io import write_json
from .runner import run


def aggregate(root):
    frame = pd.read_csv(Path(root) / "summary" / "all_runs.csv")
    frame = frame[frame.status == "success"]
    keys = ["model_id", "experiment_id", "split", "source_day", "target_day", "k", "representation",
            "selector", "reference_pool_mode", "heldout_levels", "t3_control", "mechanism_control", "seed"]
    rows = []
    for key, group in frame.groupby(keys, dropna=False):
        if group.repeat.duplicated().any():
            raise ValueError("Duplicate runs per repeat/condition; explicitly select runs before aggregation")
        for metric in ["accuracy", "macro_f1", "balanced_accuracy", "seen_accuracy", "unseen_accuracy", "seen_unseen_harmonic_mean"]:
            values = group[metric].dropna()
            if len(values):
                rows.append(dict(zip(keys, key)) | dict(metric=metric, mean=values.mean(),
                    variance=values.var(ddof=1) if len(values)>1 else None,
                    std=values.std(ddof=1) if len(values)>1 else None,
                    min=values.min(), max=values.max(), n=len(values),
                    n_repeats_expected=int(group.n_repeats_expected.max()),
                    missing_reason="missing or failed repeats" if len(values)<group.n_repeats_expected.max() else "",
                    interpretation="partition stability only; not inferential intervals"))
    out = Path(root) / "summary" / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "repeated_aggregate_runs.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "repeated_aggregate_summary.csv", index=False)
    return {"rows": len(rows), "output": str(out)}


def main():
    parser = argparse.ArgumentParser(description="Surimi classification pipeline (no regression)")
    parser.add_argument("command", choices=["audit", "create-splits", "run", "aggregate", "plan", "matrix", "verify", "drift", "preview"])
    parser.add_argument("--config", default="configs/surimi.yaml")
    parser.add_argument("--output")
    parser.add_argument("--allow-api", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate locked configuration on test (default: deferred)")
    parser.add_argument("--suite", choices=["t1-t2", "t3", "unseen"], default="t1-t2")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--support-seeds", type=int, default=1)
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.command == "audit":
        result = audit_data(cfg, args.output or "artifacts/surimi_audit_v1")
    elif args.command == "create-splits":
        meta, _, _, _ = load_surimi(cfg)
        create_splits(meta, args.output or cfg["splits"], cfg.get("repeats", 5), cfg.get("split_seed", 20261012))
        result = {"status": "splits created"}
    elif args.command == "run":
        result = run(cfg, args.allow_api, args.evaluate_test)
    elif args.command == "aggregate":
        result = aggregate(args.output or cfg["output"])
    elif args.command == "matrix":
        from .workflows import matrix
        result = matrix(cfg, args.suite, args.repeats or cfg["repeats"], args.support_seeds,
                        args.allow_api, args.evaluate_test)
    elif args.command == "verify":
        from .workflows import verify_run
        if not args.run_dir:
            parser.error("verify requires --run-dir")
        result = verify_run(args.run_dir)
    elif args.command == "drift":
        from .workflows import drift
        result = drift(cfg, args.output or "artifacts/surimi9_drift_v1")
    elif args.command == "preview":
        from .workflows import preview
        result = preview(cfg, args.output or "artifacts/surimi9_prompt_preview_v1")
    else:
        from .models import REGISTRY
        result = {"registered_models": {m: "selected" if m in cfg["models"] else "not_selected_in_this_config" for m in REGISTRY},
                  "conditions_per_full_t1_t2": 6*4*5*5,
                  "queries_per_llm_condition": len(cfg["data"].get("levels", range(10)))*16,
                  "full_t1_t2_requests_per_llm": 6*4*5*5*len(cfg["data"].get("levels", range(10)))*16,
                  "pricing": "unknown until exact model IDs and tariffs are configured",
                  "default_api_requests": 0}
        if args.output:
            write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
