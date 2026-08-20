"""Baseline 0 projectから共有coreを起動するentry point。"""

from __future__ import annotations

from pathlib import Path

from fracture_detection.cli.project_entry import ProjectCli, run_project_cli

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
PROJECT = ProjectCli(
    project="baseline0",
    arm_configs={"baseline0": CONFIG_DIR / "shared_core.yaml"},
)


def main() -> None:
    """CLI entry point。"""
    run_project_cli(PROJECT)


if __name__ == "__main__":
    main()
