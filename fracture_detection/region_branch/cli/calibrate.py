"""alpha_k / lambda_k を一度だけ校正するCLI。

統合4領域referenceモデル（`active_regions`が4領域全てのconfig）でのみ実行する。
単一領域モデルはこのCLIを実行せず、`training.experiment.resolve_calibration_path`が
指す固定位置（`experiment.phase`/`name`に依存しない）から`calibration.json`を
読むだけにする。
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fracture_detection.baseline0.data.dataset import load_manifest
from fracture_detection.baseline0.training.trainer import set_seed
from fracture_detection.region_branch.cli.runtime import configure_local_temp_dir
from fracture_detection.region_branch.config.schema import (
    apply_cli_overrides,
    load_config,
)
from fracture_detection.region_branch.data_pipeline.constants import (
    DATASET_DIR,
    DEFAULT_PSEUDO_LABEL_DIR,
    N_REGIONS,
)
from fracture_detection.region_branch.data_pipeline.loaders import (
    build_outer_fold_loaders,
)
from fracture_detection.region_branch.modeling.initialization import (
    build_initialized_model,
    save_initialization_report,
)
from fracture_detection.region_branch.training.calibration import (
    attach_calibration_metadata,
    calibrate,
    load_calibration,
    save_calibration,
    validate_calibration_compatibility,
)
from fracture_detection.region_branch.training.experiment import (
    resolve_calibration_path,
)


def parse_args() -> argparse.Namespace:
    """CLI引数を解釈する。"""
    parser = argparse.ArgumentParser(description="alpha_k / lambda_kの一度きり校正")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("fracture_detection/region_branch/config/region_branch_all.yaml"),
    )
    parser.add_argument(
        "--outer-fold",
        type=int,
        default=None,
        help="省略時はconfig.data.start_outer_fold〜end_outer_foldを順に校正する",
    )
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--n-batches", type=int, default=None)
    return parser.parse_args()


def resolve_device(gpu_id: int) -> torch.device:
    """指定GPUが利用可能ならCUDAを、そうでなければCPUを返す。"""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def run_calibration(
    config: dict[str, Any], outer_fold: int, n_batches: int | None
) -> Path:
    """1 outer foldの校正を実行し、`calibration.json`を保存する。"""
    active_regions = config["region"]["active_regions"]
    if len(active_regions) != N_REGIONS:
        raise ValueError(
            "calibrate.pyはactive_regionsが4領域全てのconfig（統合referenceモデル）"
            f"でのみ実行できます: {active_regions}"
        )
    fold_config = apply_cli_overrides(config, outer_fold=outer_fold)
    output_path = resolve_calibration_path(fold_config, outer_fold)
    if output_path.is_file():
        existing = load_calibration(output_path)
        validate_calibration_compatibility(existing, fold_config)
        print(
            f"[outer {outer_fold}] compatibleな校正結果が存在するためskipします: "
            f"{output_path}",
            flush=True,
        )
        return output_path
    device = resolve_device(int(fold_config["training"]["gpu_id"]))
    data = fold_config["data"]
    region = fold_config["region"]
    dataset_dir = Path(data.get("dataset_dir") or DATASET_DIR)
    pseudo_label_dir = Path(region.get("pseudo_label_dir") or DEFAULT_PSEUDO_LABEL_DIR)

    manifest = load_manifest()
    stream_seed = int(data["random_seed"]) + outer_fold
    loaders = build_outer_fold_loaders(
        manifest,
        outer_fold,
        dataset_dir,
        pseudo_label_dir,
        int(fold_config["training"]["natural_batch_size"]),
        int(region["human_bags_per_batch"]),
        int(region["negative_bags_per_batch"]),
        int(region["pseudo_bags_per_batch"]),
        num_workers=int(data["num_workers"]),
        seed=stream_seed,
        device=device,
        train_transform=None,  # 校正は決定的である必要があるため、augmentationは無効
    )
    set_seed(int(data["random_seed"]), outer_fold)
    model, initialization = build_initialized_model(fold_config)
    kwargs: dict[str, Any] = {}
    if n_batches is not None:
        kwargs["n_batches"] = n_batches
    result = attach_calibration_metadata(
        calibrate(
            model,
            loaders,
            outer_fold,
            pos_weight=float(fold_config["training"]["pos_weight"]),
            seed=stream_seed,
            device=device,
            **kwargs,
        ),
        fold_config,
    )
    save_calibration(result, output_path)
    save_initialization_report(
        initialization, output_path.with_name("initialization.json")
    )
    print(
        f"[outer {outer_fold}] alpha={result.alpha:.6f} (clipped={result.alpha_clipped}), "
        f"lambda={result.lambda_:.6f} (clipped={result.lambda_clipped}) -> {output_path}",
        flush=True,
    )
    return output_path


def resolve_outer_folds(config: dict[str, Any], outer_fold: int | None) -> list[int]:
    """CLIで単一fold指定がなければconfig.data.start/end_outer_foldの範囲を返す。"""
    if outer_fold is not None:
        return [outer_fold]
    data = config["data"]
    start_outer_fold = int(data["start_outer_fold"])
    end_outer_fold = int(data["end_outer_fold"])
    return list(range(start_outer_fold, end_outer_fold + 1))


def main() -> None:
    """CLIのエントリポイント。"""
    args = parse_args()
    local_temp_dir = configure_local_temp_dir()
    print(f"multiprocessing一時領域: {local_temp_dir}", flush=True)
    config = apply_cli_overrides(load_config(args.config), gpu_id=args.gpu_id)
    for outer_fold in resolve_outer_folds(config, args.outer_fold):
        run_calibration(config, outer_fold, args.n_batches)


if __name__ == "__main__":
    main()
