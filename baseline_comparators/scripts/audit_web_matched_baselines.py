"""Check all matched local predictions against the unchanged web packages."""
import argparse
import json
from spectral_icl.web_baseline_audit import audit_matched_run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    report = audit_matched_run(args.results)
    print(json.dumps({k:v for k,v in report.items() if k != "warnings"},ensure_ascii=False,indent=2))
