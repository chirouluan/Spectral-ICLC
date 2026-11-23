"""Preview or execute the fixed local comparison matched to web JSON inputs."""
import argparse
from spectral_icl.web_baselines import plan_matched, run_matched


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/web_matched_baselines.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--packages", nargs="+")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg, plan = plan_matched(args.config, args.packages, args.smoke)
    print(f"Selected {plan.directory.nunique()} packages, {len(plan)} model runs; API calls=0", flush=True)
    if args.execute:
        print(run_matched(args.config, args.packages, args.smoke, args.output).resolve(), flush=True)
    else:
        print(plan.groupby(["batch", "model_id"]).size().to_string())
