"""Baseline 1の固定5分割交差検証学習CLI。"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fracture_detection.baseline1.config import apply_cli_overrides, load_config
from fracture_detection.baseline1.dataset import (
    Baseline1Dataset,
    augment_from_config,
    load_mode_manifest,
    split_fold_manifest,
)
from fracture_detection.baseline1.experiment import (
    resolve_fold_dir,
    save_effective_config,
    save_fold_effective_config,
)
from fracture_detection.baseline1.model import Baseline1Model
from fracture_detection.baseline1.staging import manifest_sha256, stage_dataset
from fracture_detection.baseline1.trainer import (
    create_data_loader,
    set_seed,
    train_fold,
)
from fracture_detection.cohorts.constants import INPUT_MANIFEST_CSV
from fracture_detection.common.constants import DATASET_DIR


def configure_local_temp_dir(base_dir: Path = Path("/tmp")) -> Path:
    """multiprocessing一時ファイルをNFS外のローカル領域へ配置する。"""
    local_temp_dir = base_dir / f"vai-baseline1-{os.getuid()}"
    local_temp_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    for variable in ("TMPDIR", "TEMP", "TMP"):
        os.environ[variable] = str(local_temp_dir)
    tempfile.tempdir = str(local_temp_dir)
    return local_temp_dir


def parse_args() -> argparse.Namespace:
    """CLI引数を解釈する。"""
    parser = argparse.ArgumentParser(description="Baseline 1の凍結5-fold学習")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("fracture_detection/baseline1/config/matched_b0.yaml"),
    )
    parser.add_argument("--start-fold", type=int, default=None)
    parser.add_argument("--end-fold", type=int, default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve_device(gpu_id: int) -> torch.device:
    """指定GPUが利用可能ならCUDAを、そうでなければCPUを返す。"""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def build_model(config: dict[str, Any]) -> Baseline1Model:
    """YAMLのモデルセクションからBaseline1Modelを作る。"""
    model = config["model"]
    return Baseline1Model(
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
    """設定済みのfold範囲を順に学習する。"""
    local_temp_dir = configure_local_temp_dir()
    print(f"multiprocessing一時領域: {local_temp_dir}", flush=True)
    data = config["data"]
    start_fold = int(data["start_fold"])
    end_fold = int(data["end_fold"])
    mode = data["mode"]
    print(f"データマニフェストを読み込んでいます: mode={mode}", flush=True)
    manifest = load_mode_manifest(mode)
    print(f"データマニフェストを読み込みました: {len(manifest):,} bag", flush=True)
    source_dir = Path(data.get("dataset_dir") or DATASET_DIR)
    dataset_dir = source_dir
    if data["stage_to_local"]:
        print(f"ローカルキャッシュを準備しています: {data['stage_root']}", flush=True)
        dataset_dir = stage_dataset(
            manifest,
            manifest_sha256(INPUT_MANIFEST_CSV),
            source_dir=source_dir,
            stage_root=Path(data["stage_root"]),
        )
        print(f"ローカルキャッシュを使用します: {dataset_dir}", flush=True)

    device = resolve_device(int(config["training"]["gpu_id"]))
    print(
        f"学習対象を開始します: fold={start_fold}〜{end_fold}, device={device}",
        flush=True,
    )
    for fold in range(start_fold, end_fold + 1):
        print(f"[fold {fold}] データと出力先を準備しています", flush=True)
        fold_config = apply_cli_overrides(config, fold=fold)
        fold_dir = resolve_fold_dir(fold_config, fold)
        if any(fold_dir.iterdir()) and not resume:
            raise FileExistsError(
                f"既存のfold成果物があります: {fold_dir}。再開する場合は--resumeを指定してください"
            )
        save_fold_effective_config(fold_config, fold_dir)
        set_seed(int(data["random_seed"]), fold)
        train_manifest, validation_manifest = split_fold_manifest(manifest, fold)
        print(
            f"[fold {fold}] 分割完了: train={len(train_manifest):,} bag, "
            f"validation={len(validation_manifest):,} bag",
            flush=True,
        )
        train_dataset = Baseline1Dataset(
            train_manifest,
            dataset_dir=dataset_dir,
            transform=augment_from_config(mode, fold_config["augmentation"]),
        )
        validation_dataset = Baseline1Dataset(
            validation_manifest, dataset_dir=dataset_dir
        )
        batch_size = int(fold_config["training"]["batch_size"])
        workers = int(data["num_workers"])
        train_loader = create_data_loader(
            train_dataset,
            batch_size,
            shuffle=True,
            num_workers=workers,
            seed=int(data["random_seed"]) + fold,
            device=device,
        )
        validation_loader = create_data_loader(
            validation_dataset,
            batch_size,
            shuffle=False,
            num_workers=workers,
            seed=int(data["random_seed"]) + fold,
            device=device,
        )
        print(
            f"[fold {fold}] DataLoader作成完了: train={len(train_loader)} batch, "
            f"validation={len(validation_loader)} batch, workers={workers}",
            flush=True,
        )
        print(
            f"[fold {fold}] モデルを初期化しています: "
            f"{fold_config['model']['backbone']}",
            flush=True,
        )
        model = build_model(fold_config)
        print(f"[fold {fold}] モデル初期化完了。学習ループへ移ります", flush=True)
        result = train_fold(
            model,
            train_loader,
            validation_loader,
            fold_config,
            fold,
            fold_dir,
            device,
            resume=resume,
        )
        print(
            f"fold={fold} completed: best_epoch={result.best_epoch} "
            f"val_auroc={result.best_metrics['auroc']:.6f}"
        )


def main() -> None:
    """CLIのエントリポイント。"""
    args = parse_args()
    config = apply_cli_overrides(
        load_config(args.config),
        gpu_id=args.gpu_id,
        start_fold=args.start_fold,
        end_fold=args.end_fold,
    )
    config_path = save_effective_config(config)
    print(f"実効configを保存しました: {config_path}")
    run_training(config, args.resume)


if __name__ == "__main__":
    main()
