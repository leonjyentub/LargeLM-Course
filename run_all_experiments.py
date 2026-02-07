from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def load_config(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError("YAML config requires pyyaml. Install with: pip install pyyaml") from exc

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    raise ValueError(f"Unsupported config type: {path}")


def run_cmd(cmd, dry_run: bool = False):
    print("$", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def bool_flag(value: bool, flag: str):
    return [flag] if value else []


def build_train_cmd(method_cfg: Dict[str, Any], global_cfg: Dict[str, Any], data_dir: str):
    merged = dict(global_cfg)
    merged.update(method_cfg.get("overrides", {}))

    cmd = [
        sys.executable,
        method_cfg["script"],
        "--data-dir",
        data_dir,
        "--out-dir",
        method_cfg["out_dir"],
        "--epochs",
        str(merged["epochs"]),
        "--batch-size",
        str(merged["batch_size"]),
        "--lr",
        str(merged["lr"]),
        "--seed",
        str(merged["seed"]),
    ]
    cmd.extend(bool_flag(bool(merged.get("regen_data", False)), "--regen-data"))
    return cmd


def main():
    parser = argparse.ArgumentParser(description="One-click runner for all 7 NLP addition experiments.")
    parser.add_argument("--config", type=str, default="configs/experiment_config.json")
    parser.add_argument("--dry-run", action="store_true", help="Only print commands without executing.")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)

    data_cfg = cfg["data"]
    global_cfg = cfg["global_train"]
    methods = cfg["methods"]
    comparison_cfg = cfg.get("comparison", {})

    if data_cfg.get("generate", True):
        data_cmd = [
            sys.executable,
            data_cfg.get("script", "generate_data.py"),
            "--data-dir",
            data_cfg["data_dir"],
            "--n-train",
            str(data_cfg["n_train"]),
            "--n-val",
            str(data_cfg["n_val"]),
            "--n-test",
            str(data_cfg["n_test"]),
            "--seed",
            str(data_cfg.get("seed", global_cfg.get("seed", 42))),
        ]
        run_cmd(data_cmd, dry_run=args.dry_run)

    for method in methods:
        if not method.get("enabled", True):
            continue
        print(f"\n### Running method: {method['name']}")
        train_cmd = build_train_cmd(method, global_cfg, data_cfg["data_dir"])
        run_cmd(train_cmd, dry_run=args.dry_run)

    if comparison_cfg.get("enabled", True):
        print("\n### Running comparison")
        cmp_cmd = [
            sys.executable,
            comparison_cfg.get("script", "compare_all_methods.py"),
            "--out-dir",
            comparison_cfg.get("out_dir", "outputs/comparison"),
        ]
        run_cmd(cmp_cmd, dry_run=args.dry_run)

    print("\nFinished all configured experiment steps.")


if __name__ == "__main__":
    main()
