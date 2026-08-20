"""6構成とartifactを正式実験manifestへ凍結するCLI。"""

from __future__ import annotations

import argparse
from pathlib import Path

from fracture_detection.config.schema import load_config
from fracture_detection.core.artifacts import create_frozen_manifest

ARM_ARGUMENTS = (
    "baseline0",
    "control_b",
    "baseline1_b",
    "proposed_b",
    "proposed_max",
    "proposed_max_beta0",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="骨折検出正式実験manifestの凍結")
    for arm in ARM_ARGUMENTS:
        parser.add_argument(
            f"--{arm.replace('_', '-')}-config", type=Path, required=True
        )
    parser.add_argument("--loss-weights", type=Path, required=True)
    parser.add_argument("--resource-profile", type=Path, action="append", required=True)
    parser.add_argument("--fold-plan", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = {
        arm: load_config(getattr(args, f"{arm}_config")) for arm in ARM_ARGUMENTS
    }
    create_frozen_manifest(
        args.output,
        configs=configs,
        loss_weights_path=args.loss_weights,
        resource_profiles=args.resource_profile,
        fold_execution_plans=args.fold_plan,
    )
    print(f"正式実験manifestを凍結しました: {args.output}")


if __name__ == "__main__":
    main()
