"""Baseline 0のnested 5-fold学習CLI。"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fracture_detection.baseline0.config.schema import apply_cli_overrides, load_config
from fracture_detection.baseline0.data.dataset import (
    Baseline0Dataset,
    augment_from_config,
    load_manifest,
)
from fracture_detection.baseline0.data.staging import manifest_sha256, stage_dataset
from fracture_detection.baseline0.modeling.model import Baseline0Model
from fracture_detection.baseline0.training.experiment import (
    resolve_fold_dir,
    save_effective_config,
    save_fold_effective_config,
)
from fracture_detection.baseline0.training.trainer import (
    create_data_loader,
    set_seed,
    train_fold,
)
from fracture_detection.common.constants import DATASET_DIR, INPUT_MANIFEST_CSV
from fracture_detection.common.sampling import EpochShuffleSampler
from fracture_detection.common.splits import split_nested_manifest


def configure_local_temp_dir(base_dir: Path = Path("/tmp")) -> Path:
    """multiprocessing一時ファイルをNFS外のローカル領域へ配置する。"""
    local_temp_dir = base_dir / f"vai-baseline0-{os.getuid()}"
    local_temp_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    for variable in ("TMPDIR", "TEMP", "TMP"):
        os.environ[variable] = str(local_temp_dir)
    tempfile.tempdir = str(local_temp_dir)
    return local_temp_dir


def parse_args() -> argparse.Namespace:
    """CLI引数を解釈する。"""
    parser = argparse.ArgumentParser(description="Baseline 0のnested 5-fold学習")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("fracture_detection/baseline0/config/baseline0.yaml"),
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


def build_model(config: dict[str, Any]) -> Baseline0Model:
    """YAMLのモデルsectionからBaseline0Modelを作る。"""
    model = config["model"]
    return Baseline0Model(
        backbone_name=str(model["backbone"]),
        pretrained=bool(model["pretrained"]),
        drop_rate=float(model["drop_rate"]),
        drop_path_rate=float(model["drop_path_rate"]),
        head_dropout=float(model["head_dropout"]),
        lstm_hidden=int(model["lstm_hidden"]),
        lstm_layers=int(model["lstm_layers"]),
        n_planes=int(model["n_planes"]),
    )


def run_training(config: dict[str, Any], resume: bool) -> None:
    """設定されたouter fold範囲をnested選択で順に学習する。"""
    local_temp_dir = configure_local_temp_dir()
    print(f"multiprocessing一時領域: {local_temp_dir}", flush=True)
    data = config["data"]
    start_outer_fold = int(data["start_outer_fold"])
    end_outer_fold = int(data["end_outer_fold"])
    print("凍結fullマニフェストを読み込んでいます", flush=True)
    manifest = load_manifest()
    print(f"マニフェストを読み込みました: {len(manifest):,} bag", flush=True)
    source_dir = Path(data.get("dataset_dir") or DATASET_DIR)
    dataset_dir = source_dir
    if data["stage_to_local"]:
        print(
            f"共有ローカルキャッシュを準備しています: {data['stage_root']}", flush=True
        )
        dataset_dir = stage_dataset(
            manifest,
            manifest_sha256(INPUT_MANIFEST_CSV),
            source_dir=source_dir,
            stage_root=Path(data["stage_root"]),
            copy_workers=int(data["stage_copy_workers"]),
        )
        print(f"共有ローカルキャッシュを使用します: {dataset_dir}", flush=True)

    device = resolve_device(int(config["training"]["gpu_id"]))
    print(
        f"学習対象outer fold: {start_outer_fold}〜{end_outer_fold}, device={device}",
        flush=True,
    )
    for outer_fold in range(start_outer_fold, end_outer_fold + 1):
        fold_config = apply_cli_overrides(config, outer_fold=outer_fold)
        inner_fold = int(fold_config["runtime"]["inner_fold"])
        train_folds = tuple(
            int(value) for value in fold_config["runtime"]["train_folds"]
        )
        print(
            f"[outer {outer_fold}] train={train_folds}, val={inner_fold}を準備します",
            flush=True,
        )
        fold_dir = resolve_fold_dir(fold_config, outer_fold)
        if any(fold_dir.iterdir()) and not resume:
            raise FileExistsError(
                f"既存のouter成果物があります: {fold_dir}。"
                "再開する場合は--resumeを指定してください"
            )
        completed_outer_paths = (
            fold_dir / "outer_predictions.csv",
            fold_dir / "outer_predictions_prauc_checkpoint.csv",
        )
        if resume and all(path.is_file() for path in completed_outer_paths):
            print(
                f"[outer {outer_fold}] 両checkpointのouter推論済みのためskipします",
                flush=True,
            )
            continue
        if resume and any(path.is_file() for path in completed_outer_paths):
            raise RuntimeError(
                f"[outer {outer_fold}] outer成果物が片方だけ存在します。"
                "正式評価の再推論を避けるため自動再開できません"
            )
        save_fold_effective_config(fold_config, fold_dir)
        set_seed(int(data["random_seed"]), outer_fold)
        train_manifest, inner_manifest, outer_manifest = split_nested_manifest(
            manifest, outer_fold
        )
        print(
            f"[outer {outer_fold}] 分割完了: train={len(train_manifest):,}, "
            f"val={len(inner_manifest):,}, outer={len(outer_manifest):,}",
            flush=True,
        )

        train_dataset = Baseline0Dataset(
            train_manifest,
            dataset_dir=dataset_dir,
            transform=augment_from_config(fold_config["augmentation"]),
        )
        inner_dataset = Baseline0Dataset(inner_manifest, dataset_dir=dataset_dir)
        outer_dataset = Baseline0Dataset(outer_manifest, dataset_dir=dataset_dir)
        batch_size = int(fold_config["training"]["natural_batch_size"])
        workers = int(data["num_workers"])
        stream_seed = int(data["random_seed"]) + outer_fold
        natural_sampler = EpochShuffleSampler(train_dataset, seed=stream_seed)
        train_loader = create_data_loader(
            train_dataset,
            batch_size,
            num_workers=workers,
            seed=stream_seed,
            device=device,
            sampler=natural_sampler,
        )
        inner_loader = create_data_loader(
            inner_dataset,
            batch_size,
            num_workers=workers,
            seed=stream_seed + 10_000,
            device=device,
        )
        outer_loader = create_data_loader(
            outer_dataset,
            batch_size,
            num_workers=workers,
            seed=stream_seed + 20_000,
            device=device,
        )
        print(
            f"[outer {outer_fold}] DataLoader作成完了: train={len(train_loader)}, "
            f"val={len(inner_loader)}, outer={len(outer_loader)}, workers={workers}",
            flush=True,
        )
        model = build_model(fold_config)
        result = train_fold(
            model,
            train_loader,
            inner_loader,
            outer_loader,
            fold_config,
            outer_fold,
            fold_dir,
            device,
            resume=resume,
        )
        print(
            f"outer={outer_fold} completed: best_epoch={result.best_epoch} "
            f"val_auroc={result.best_metrics['auroc']:.6f} "
            f"best_prauc_epoch={result.best_prauc_epoch} "
            f"val_prauc={result.best_prauc_metrics['average_precision']:.6f} "
            f"outer_rows={len(result.outer_predictions):,}"
        )


def main() -> None:
    """CLIのエントリポイント。"""
    args = parse_args()
    config = apply_cli_overrides(
        load_config(args.config),
        gpu_id=args.gpu_id,
        start_outer_fold=args.start_outer_fold,
        end_outer_fold=args.end_outer_fold,
    )
    config_path = save_effective_config(config)
    print(f"実効configを保存しました: {config_path}")
    run_training(config, args.resume)


if __name__ == "__main__":
    main()
