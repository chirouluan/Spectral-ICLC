"""Build a fresh four-batch prompt-ready data package from the raw spectra."""
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys

import yaml


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT.parents[1] / "Spectral-ICLC"
BATCH_MAP = {
    "T1": "batch_01",
    "T2": "batch_02",
    "UNSEEN": "batch_03",
    "PUBLIC": "batch_04",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT,
                        help="原始 Spectral-ICLC 仓库，需包含 datasets/、data/metadata/ 和 splits/")
    parser.add_argument("--output", type=Path,
                        help="新数据包输出路径；默认创建 data_json/generated/four_batches_<时间戳>")
    parser.add_argument("--batches", nargs="+", choices=tuple(BATCH_MAP), default=tuple(BATCH_MAP),
                        help="选择要构建的批次，默认构建全部")
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    if not source_root.is_dir():
        parser.error(f"source root does not exist: {source_root}")
    selected = set(args.batches)
    required = []
    if selected & {"T1", "T2", "UNSEEN"}:
        required.extend(
            source_root / "datasets" / "Surimi adulteration" / f"day{day} Surimi adulteration.xlsx"
            for day in (0, 7, 14)
        )
        required.extend([
            source_root / "data" / "metadata" / "physical_sample_mapping.csv",
            source_root / "splits" / "surimi9_v1",
        ])
    if "PUBLIC" in selected:
        public_cfg = yaml.safe_load((HERE / "configs" / "public_datasets.yaml").read_text(encoding="utf-8"))
        raw_root = source_root / public_cfg["raw_root"]
        batch_cfg = yaml.safe_load((HERE / "configs" / "web_batches.yaml").read_text(encoding="utf-8"))
        for name in batch_cfg["batch_04"].get("datasets", public_cfg["datasets"]):
            spec = public_cfg["datasets"][name]
            if "extends" in spec:
                spec = {**public_cfg["datasets"][spec["extends"]], **spec}
            folder = raw_root / spec["folder"]
            required.extend([folder / spec["file"], folder / spec["paper"]])
            if spec.get("test_file"):
                required.append(folder / spec["test_file"])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("required raw inputs are missing:\n  " + "\n  ".join(missing))

    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        output = PROJECT_ROOT / "data_json" / "generated" / f"four_batches_{stamp}"
    elif not output.is_absolute():
        output = PROJECT_ROOT / output
    output = output.resolve()
    if output.exists():
        parser.error(f"output already exists; choose a new path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(HERE / "src"))
    from spectral_icl.web_batches import export_four_batches

    config = HERE / "configs" / "web_batches.yaml"
    os.chdir(source_root)
    result = export_four_batches(
        config_path=str(config),
        output=str(output),
        batches=[BATCH_MAP[name] for name in args.batches],
    )
    print(f"Built data package: {result}")
    print("No LLM calls or predictions were made during data-package generation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
