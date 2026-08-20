"""MTL projectから共有coreを起動するentry point。"""

from __future__ import annotations

from pathlib import Path

from fracture_detection.cli.project_entry import ProjectCli, run_project_cli

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
PROJECT = ProjectCli(
    project="mtl",
    arm_configs={
        "control_b": CONFIG_DIR / "control_b.yaml",
        "baseline1_b": CONFIG_DIR / "baseline1_b.yaml",
    },
    calibration_kinds=("lambda",),
)


def main() -> None:
    """CLI entry point。"""
    run_project_cli(PROJECT)


if __name__ == "__main__":
    main()
