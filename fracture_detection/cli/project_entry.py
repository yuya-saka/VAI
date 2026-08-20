"""各projectのcliパッケージが共有するproject単位entry point。"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from fracture_detection.cli.calibrate import REFERENCE_ARMS, run_calibration
from fracture_detection.cli.resource_profile import run_resource_profile
from fracture_detection.cli.sync_wandb import sync_experiment
from fracture_detection.cli.train import run_cli
from fracture_detection.config.schema import load_config
from fracture_detection.core.experiment import resolve_experiment_root

DEFAULT_PROFILE_STEPS = 20
DEFAULT_PROFILE_WARMUP_STEPS = 10
DEFAULT_EXPECTED_BATCHES = 64


@dataclass(frozen=True)
class ProjectCli:
    """1 projectが公開するarmと、そのprojectで実行できる校正kind。"""

    project: str
    arm_configs: Mapping[str, Path]
    calibration_kinds: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.arm_configs:
            raise ValueError("arm_configsは非空が必要です")
        unknown = set(self.calibration_kinds) - set(REFERENCE_ARMS)
        if unknown:
            raise ValueError(f"未知の校正kindです: {sorted(unknown)}")

    @property
    def arms(self) -> list[str]:
        """このprojectのarm名を昇順で返す。"""
        return sorted(self.arm_configs)


def run_project_cli(project: ProjectCli, argv: Sequence[str] | None = None) -> None:
    """projectのsubcommandを解釈し、共通CLI実装へ委譲する。"""
    args = _parse_args(project, argv)
    config_path = project.arm_configs[args.arm]
    if args.command == "train":
        run_cli(
            config_path,
            outer_fold=args.outer_fold,
            gpu_id=args.gpu_id,
            resume=args.resume,
            smoke_steps=args.smoke_steps,
        )
        return
    if args.command == "profile":
        run_resource_profile(
            config_path,
            output=args.output,
            gpu_id=args.gpu_id,
            outer_fold=args.outer_fold,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
        )
        return
    if args.command == "calibrate":
        _verify_reference_arm(args.kind, args.arm)
        run_calibration(
            config_path,
            kind=args.kind,
            output=args.output,
            gpu_id=args.gpu_id,
            expected_batches=args.expected_batches,
        )
        return
    experiment_dir = resolve_experiment_root(load_config(config_path))
    synced = sync_experiment(experiment_dir)
    print(f"total: {synced} epochs synced", flush=True)


def _verify_reference_arm(kind: str, arm: str) -> None:
    """校正artifactの参照armと実行armの不一致を拒否する。"""
    expected = REFERENCE_ARMS[kind]
    if arm != expected:
        raise ValueError(f"{kind}校正の参照armは{expected}です: {arm}が指定されました")


def _parse_args(project: ProjectCli, argv: Sequence[str] | None) -> argparse.Namespace:
    """projectのsubcommand定義を組み立てて引数を解釈する。"""
    parser = argparse.ArgumentParser(
        prog=f"python -m fracture_detection.{project.project}.cli",
        description=f"{project.project} projectの共有core実行",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="nested fold学習を実行する")
    _add_arm_argument(train, project)
    train.add_argument("--outer-fold", type=int, default=None)
    train.add_argument("--gpu-id", type=int, default=None)
    train.add_argument("--resume", action="store_true")
    train.add_argument(
        "--smoke-steps",
        type=int,
        default=None,
        help="凍結前の構造検証として各epochのstep数を制限する",
    )

    profile = subparsers.add_parser("profile", help="resource profileを測定する")
    _add_arm_argument(profile, project)
    profile.add_argument("--output", type=Path, required=True)
    profile.add_argument("--gpu-id", type=int, default=0)
    profile.add_argument("--outer-fold", type=int, default=0)
    profile.add_argument("--steps", type=int, default=DEFAULT_PROFILE_STEPS)
    profile.add_argument(
        "--warmup-steps", type=int, default=DEFAULT_PROFILE_WARMUP_STEPS
    )

    if project.calibration_kinds:
        calibrate = subparsers.add_parser("calibrate", help="λまたはβを校正する")
        _add_arm_argument(calibrate, project)
        _add_kind_argument(calibrate, project)
        calibrate.add_argument("--output", type=Path, default=None)
        calibrate.add_argument("--gpu-id", type=int, default=0)
        calibrate.add_argument(
            "--expected-batches", type=int, default=DEFAULT_EXPECTED_BATCHES
        )

    sync = subparsers.add_parser("sync-wandb", help="既存履歴をW&Bへ同期する")
    _add_arm_argument(sync, project)

    return parser.parse_args(argv)


def _add_arm_argument(parser: argparse.ArgumentParser, project: ProjectCli) -> None:
    """単一armのprojectでは--armを省略可能にする。"""
    arms = project.arms
    default = arms[0] if len(arms) == 1 else None
    parser.add_argument(
        "--arm",
        choices=arms,
        default=default,
        required=default is None,
        help="このprojectが公開するarm",
    )


def _add_kind_argument(parser: argparse.ArgumentParser, project: ProjectCli) -> None:
    """校正kindが1種類のprojectでは--kindを省略可能にする。"""
    kinds = project.calibration_kinds
    default = kinds[0] if len(kinds) == 1 else None
    parser.add_argument(
        "--kind",
        choices=kinds,
        default=default,
        required=default is None,
        help="校正する損失係数",
    )
