"""Proposed projectから共有coreを起動するentry point。"""

from __future__ import annotations

from pathlib import Path

from fracture_detection.cli.project_entry import ProjectCli, run_project_cli

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
PROJECT = ProjectCli(
    project="proposed",
    arm_configs={
        "proposed_b": CONFIG_DIR / "proposed_b.yaml",
        "proposed_max": CONFIG_DIR / "proposed_max.yaml",
        "proposed_max_beta0": CONFIG_DIR / "proposed_max_beta0.yaml",
    },
    calibration_kinds=("beta",),
)


def main() -> None:
    """CLI entry point。"""
    run_project_cli(PROJECT)


if __name__ == "__main__":
    main()
