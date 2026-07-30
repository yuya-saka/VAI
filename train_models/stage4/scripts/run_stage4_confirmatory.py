"""Run paired Weak-only/Mixed Stage4 seeds with resumable logging."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_models.stage4.src.data_utils import load_config  # noqa: E402

ARM_CONFIGS = {
    "weak_only": ROOT / "train_models/stage4/config/stage4_weak_only.yaml",
    "mixed": ROOT / "train_models/stage4/config/stage4_mixed.yaml",
}


def _flatten(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in config.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            values.update(_flatten(value, f"{path}."))
        else:
            values[path] = value
    return values


def validate_confirmatory_config_difference() -> None:
    """Require arm configs to differ only by name and region-loss scale."""
    weak = _flatten(load_config(ARM_CONFIGS["weak_only"]))
    mixed = _flatten(load_config(ARM_CONFIGS["mixed"]))
    allowed = {"experiment.name", "training.lambda_region_scale"}
    differences = {
        key for key in set(weak) | set(mixed) if weak.get(key) != mixed.get(key)
    }
    if differences != allowed:
        raise ValueError(
            f"confirmatory configs have unexpected differences: {sorted(differences)}"
        )


def _output_dir(arm: str, seed: int) -> Path:
    experiment_name = load_config(ARM_CONFIGS[arm])["experiment"]["name"]
    return ROOT / "train_models/stage4/outputs" / str(experiment_name) / f"seed{seed}"


def _is_complete(arm: str, seed: int) -> bool:
    output_dir = _output_dir(arm, seed)
    return (output_dir / "oof_predictions.csv").exists() and all(
        (output_dir / f"fold{fold}" / "final_model.pt").exists() for fold in range(5)
    )


def run_arm_seed(
    arm: str,
    seed: int,
    log_dir: Path,
) -> None:
    """Run or resume one five-fold arm/seed job."""
    if _is_complete(arm, seed):
        print(f"[SKIP] {arm} seed={seed} already complete", flush=True)
        return
    output_dir = _output_dir(arm, seed)
    command = [
        sys.executable,
        str(ROOT / "train_models/stage4/train.py"),
        "--config",
        str(ARM_CONFIGS[arm]),
        "--seed",
        str(seed),
    ]
    if (output_dir / "config.yaml").exists():
        command.append("--resume")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{arm}_seed{seed}.log"
    print(f"[RUN] {' '.join(command)}", flush=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_evaluation(seeds: list[int], log_dir: Path) -> None:
    """Build the confirmatory report after every requested run completes."""
    command = [
        sys.executable,
        str(ROOT / "train_models/stage4/scripts/stage4_evaluate.py"),
        "--seeds",
        *(str(seed) for seed in seeds),
    ]
    log_path = log_dir / "evaluation.log"
    print(f"[EVALUATE] {' '.join(command)}", flush=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
    )
    parser.add_argument(
        "--arms",
        choices=tuple(ARM_CONFIGS),
        nargs="+",
        default=["weak_only", "mixed"],
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "train_models/stage4/outputs/confirmatory_logs_v2",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    validate_confirmatory_config_difference()
    status_path = arguments.log_dir / "status.json"
    for seed in arguments.seeds:
        for arm in arguments.arms:
            run_arm_seed(arm, seed, arguments.log_dir)
            status = {
                f"{current_arm}_seed{current_seed}": _is_complete(
                    current_arm,
                    current_seed,
                )
                for current_seed in arguments.seeds
                for current_arm in arguments.arms
            }
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status_path.write_text(
                json.dumps(status, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    if set(arguments.arms) == set(ARM_CONFIGS) and all(status.values()):
        run_evaluation(arguments.seeds, arguments.log_dir)


if __name__ == "__main__":
    main()
