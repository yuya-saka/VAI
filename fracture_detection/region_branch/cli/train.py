"""region_branchのnested 5-fold学習CLI。"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fracture_detection.baseline0.data.dataset import augment_from_config, load_manifest
from fracture_detection.baseline0.data.splits import split_nested_manifest
from fracture_detection.baseline0.training.trainer import set_seed
from fracture_detection.region_branch.cli.runtime import configure_local_temp_dir
from fracture_detection.region_branch.config.schema import (
    apply_cli_overrides,
    load_config,
)
from fracture_detection.region_branch.data_pipeline.constants import (
    DATASET_DIR,
    DEFAULT_PSEUDO_LABEL_DIR,
)
from fracture_detection.region_branch.data_pipeline.loaders import (
    build_eval_loader,
    build_outer_fold_loaders,
)
from fracture_detection.region_branch.data_pipeline.pseudo_labels import (
    attach_teacher_scores,
    load_pseudo_scores,
)
from fracture_detection.region_branch.modeling.initialization import (
    build_initialized_model,
    save_initialization_report,
)
from fracture_detection.region_branch.training.calibration import (
    load_calibration,
    validate_calibration_compatibility,
)
from fracture_detection.region_branch.training.experiment import (
    resolve_calibration_path,
    resolve_fold_dir,
    save_effective_config,
    save_fold_effective_config,
)
from fracture_detection.region_branch.training.monitoring import (
    select_diagnostic_subset,
)
from fracture_detection.region_branch.training.trainer import train_fold


def parse_args() -> argparse.Namespace:
    """CLI引数を解釈する。"""
    parser = argparse.ArgumentParser(description="region_branchのnested 5-fold学習")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("fracture_detection/region_branch/config/region_branch_all.yaml"),
    )
    parser.add_argument("--start-outer-fold", type=int, default=None)
    parser.add_argument("--end-outer-fold", type=int, default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve_device(gpu_id: int) -> torch.device:
    """指定GPUが利用可能ならCUDAを、そうでなければCPUを返す。"""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


@contextmanager
def _timed_phase(label: str) -> Iterator[None]:
    """時間のかかる起動処理の開始・完了・失敗を標準出力へ記録する。"""
    started_at = time.monotonic()
    print(f"{label} 開始", flush=True)
    try:
        yield
    except BaseException:
        elapsed = time.monotonic() - started_at
        print(f"{label} 失敗 ({elapsed:.1f}秒)", flush=True)
        raise
    elapsed = time.monotonic() - started_at
    print(f"{label} 完了 ({elapsed:.1f}秒)", flush=True)


def run_training(config: dict[str, Any], resume: bool) -> None:
    """設定されたouter fold範囲をnested選択で順に学習する。"""
    data = config["data"]
    start_outer_fold = int(data["start_outer_fold"])
    end_outer_fold = int(data["end_outer_fold"])
    device = resolve_device(int(config["training"]["gpu_id"]))
    print(
        f"学習対象outer fold: {start_outer_fold}〜{end_outer_fold}, device={device}",
        flush=True,
    )
    with _timed_phase("manifest読込"):
        manifest = load_manifest()
    print(f"manifest件数: {len(manifest):,}", flush=True)
    for outer_fold in range(start_outer_fold, end_outer_fold + 1):
        with _timed_phase(f"[outer {outer_fold}] fold準備"):
            fold_config = apply_cli_overrides(config, outer_fold=outer_fold)
            set_seed(int(data["random_seed"]), outer_fold)
            fold_dir = resolve_fold_dir(fold_config, outer_fold)
        if any(fold_dir.iterdir()) and not resume:
            raise FileExistsError(
                f"既存のouter成果物があります: {fold_dir}。"
                "再開する場合は--resumeを指定してください"
            )
        outer_prediction_path = fold_dir / "outer_predictions.csv"
        if resume and outer_prediction_path.is_file():
            print(f"[outer {outer_fold}] outer推論済みのためskipします", flush=True)
            continue
        save_fold_effective_config(fold_config, fold_dir)

        calibration_path = resolve_calibration_path(fold_config, outer_fold)
        print(
            f"[outer {outer_fold}] 校正結果を読み込みます: {calibration_path}",
            flush=True,
        )
        with _timed_phase(f"[outer {outer_fold}] 校正結果の読込・検証"):
            calibration = load_calibration(calibration_path)
            validate_calibration_compatibility(calibration, fold_config)

        with _timed_phase(f"[outer {outer_fold}] 全DataLoader構築"):
            loaders, inner_loader, outer_loader, diagnostic_loader = (
                _build_fold_loaders(manifest, fold_config, outer_fold, device)
            )
        set_seed(int(data["random_seed"]), outer_fold)
        with _timed_phase(f"[outer {outer_fold}] model初期化"):
            model, initialization = build_initialized_model(fold_config)
            initialization_path = fold_dir / "initialization.json"
            save_initialization_report(initialization, initialization_path)
        print(
            f"[outer {outer_fold}] Baseline 0から初期化しました: "
            f"loaded={initialization.loaded_key_count}, "
            f"random={initialization.random_key_count}, "
            f"report={initialization_path}",
            flush=True,
        )
        print(f"[outer {outer_fold}] train_foldを開始します", flush=True)
        result = train_fold(
            model,
            loaders,
            inner_loader,
            outer_loader,
            diagnostic_loader,
            fold_config,
            calibration,
            outer_fold,
            fold_dir,
            device,
            resume=resume,
        )
        print(
            f"outer={outer_fold} completed: best_epoch={result.best_epoch} "
            f"val_total={result.best_val_metrics['total']:.6f} "
            f"outer_rows={len(result.outer_predictions):,}"
        )


def _build_fold_loaders(
    manifest: Any, fold_config: dict[str, Any], outer_fold: int, device: torch.device
) -> tuple[Any, Any, Any, Any]:
    """1 outer foldの学習・検証・診断に必要な全DataLoaderを構築する。"""
    data = fold_config["data"]
    region = fold_config["region"]
    training = fold_config["training"]
    dataset_dir = Path(data.get("dataset_dir") or DATASET_DIR)
    pseudo_label_dir = Path(region.get("pseudo_label_dir") or DEFAULT_PSEUDO_LABEL_DIR)
    natural_batch_size = int(training["natural_batch_size"])
    num_workers = int(data["num_workers"])
    stream_seed = int(data["random_seed"]) + outer_fold

    prefix = f"[outer {outer_fold}]"
    with _timed_phase(f"{prefix} 学習用DataLoader構築"):
        loaders = build_outer_fold_loaders(
            manifest,
            outer_fold,
            dataset_dir,
            pseudo_label_dir,
            natural_batch_size,
            int(region["human_bags_per_batch"]),
            int(region["negative_bags_per_batch"]),
            int(region["pseudo_bags_per_batch"]),
            num_workers=num_workers,
            seed=stream_seed,
            device=device,
            train_transform=augment_from_config(fold_config["augmentation"]),
        )
    with _timed_phase(f"{prefix} nested manifest分割"):
        _, inner_manifest, outer_manifest = split_nested_manifest(manifest, outer_fold)
    with _timed_phase(f"{prefix} inner評価DataLoader構築"):
        inner_loader = build_eval_loader(
            inner_manifest,
            dataset_dir,
            natural_batch_size,
            num_workers,
            stream_seed + 10_000,
            device,
        )
    with _timed_phase(f"{prefix} outer評価DataLoader構築"):
        outer_loader = build_eval_loader(
            outer_manifest,
            dataset_dir,
            natural_batch_size,
            num_workers,
            stream_seed + 20_000,
            device,
        )

    with _timed_phase(f"{prefix} 診断DataLoader構築"):
        diagnostic_manifest = select_diagnostic_subset(
            loaders.pools,
            int(region["diagnostic_subset_size"]),
            seed=int(data["random_seed"]),
        )
        diagnostic_manifest = attach_teacher_scores(
            diagnostic_manifest, load_pseudo_scores(pseudo_label_dir, outer_fold)
        )
        diagnostic_loader = build_eval_loader(
            diagnostic_manifest,
            dataset_dir,
            natural_batch_size,
            num_workers,
            stream_seed + 30_000,
            device,
        )
    return loaders, inner_loader, outer_loader, diagnostic_loader


def main() -> None:
    """CLIのエントリポイント。"""
    args = parse_args()
    local_temp_dir = configure_local_temp_dir()
    print(f"multiprocessing一時領域: {local_temp_dir}", flush=True)
    with _timed_phase(f"config読込: {args.config}"):
        config = apply_cli_overrides(
            load_config(args.config),
            gpu_id=args.gpu_id,
            start_outer_fold=args.start_outer_fold,
            end_outer_fold=args.end_outer_fold,
        )
        config_path = save_effective_config(config)
    print(f"実効configを保存しました: {config_path}", flush=True)
    run_training(config, args.resume)


if __name__ == "__main__":
    main()
