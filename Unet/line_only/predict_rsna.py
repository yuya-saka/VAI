"""RSNAデータセットの最大面積スライスに対する直線推論スクリプト。

fracture_dataset/{study_uid}/{vertebra}/ の ct.npy と vertebra_mask.npy から
max_area_forced プレーンを抽出し、line_only モデルで4本線を予測する。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# train.py と同じパターン: Unet/ を sys.path に追加
_here = Path(__file__).resolve().parent  # line_only/
_unet = _here.parent                     # Unet/
if str(_unet) not in sys.path:
    sys.path.insert(0, str(_unet))

from line_only.src.model import VERTEBRA_TO_IDX, TinyUNet  # noqa: E402
from line_only.utils.detection import detect_line_moments, line_extent, moments_to_phi_rho  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RSNA_DATA_DIR = PROJECT_ROOT / "data" / "rsna_data"
FRACTURE_DATASET_DIR = RSNA_DATA_DIR / "fracture_dataset"
METADATA_DIR = RSNA_DATA_DIR / "processing_metadata"

DEFAULT_CHECKPOINT_DIR = (
    PROJECT_ROOT / "Unet" / "outputs" / "line_20260616" / "sig4.0_ALL(CC適用)" / "checkpoints"
)
DEFAULT_TRAINING_DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
VERTEBRA_LEVELS = [f"C{i}" for i in range(1, 8)]
LINE_KEYS = tuple(f"line_{i}" for i in range(1, 5))
CT_CENTER_CHANNEL = 2
IMAGE_SIZE = 224
FALLBACK_LINE_LENGTH_PX = 80.0


def find_max_area_plane_index(metadata: dict, vertebra: str) -> int | None:
    """メタデータからmax_area_forcedプレーンのインデックスを返す。"""
    planes = metadata["vertebrae"][vertebra]["classifier_planes"]["planes"]
    for plane in planes:
        if plane.get("max_area_forced", False):
            return int(plane["sequence_index"])
    return None


def load_metadata(study_uid: str) -> dict | None:
    """processing_metadataからスタディのメタデータを読み込む。"""
    meta_path = METADATA_DIR / f"{study_uid}.json"
    if not meta_path.exists():
        return None
    with meta_path.open(encoding="utf-8") as f:
        return json.load(f)


def extract_max_area_input(
    study_dir: Path,
    vertebra: str,
    plane_index: int,
) -> np.ndarray | None:
    """max_area プレーンから (2, 224, 224) float32 入力を抽出する。

    ch0: CT中心チャネル [0,1], ch1: vertebra_mask [0,1]
    """
    ct_path = study_dir / vertebra / "ct.npy"
    mask_path = study_dir / vertebra / "vertebra_mask.npy"
    if not ct_path.exists() or not mask_path.exists():
        return None

    ct = np.load(ct_path, allow_pickle=False)
    mask = np.load(mask_path, allow_pickle=False)

    ct_slice = ct[plane_index, CT_CENTER_CHANNEL].astype(np.float32) / 255.0
    mask_slice = mask[plane_index].astype(np.float32)

    return np.stack([ct_slice, mask_slice], axis=0)


def load_ensemble_models(
    checkpoint_dir: Path,
    n_folds: int,
    device: torch.device,
) -> list[TinyUNet]:
    """全foldのモデルをロードしてリストで返す。"""
    models = []
    for fold in range(n_folds):
        ckpt_path = checkpoint_dir / f"best_fold{fold}.pt"
        if not ckpt_path.exists():
            print(f"[WARN] チェックポイントが見つかりません: {ckpt_path}")
            continue

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        cfg = checkpoint.get("cfg", {})
        model_cfg = cfg.get("model", {})

        model = TinyUNet(
            in_ch=int(model_cfg.get("in_channels", 2)),
            out_ch=int(model_cfg.get("out_channels", 4)),
            feats=tuple(model_cfg.get("features", [16, 32, 64, 128])),
            dropout=0.0,
            num_vertebra=int(model_cfg.get("num_vertebra", 7))
            if model_cfg.get("use_vertebra_conditioning", False)
            else 0,
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models.append(model)
        print(f"[INFO] fold{fold} モデル読み込み完了: {ckpt_path.name}")

    return models


def compute_average_training_line_lengths(dataset_dir: Path) -> dict[str, float]:
    """学習データの lines.json から line別平均長を計算する。"""
    sums = {key: 0.0 for key in LINE_KEYS}
    counts = {key: 0 for key in LINE_KEYS}

    for lines_path in sorted(dataset_dir.glob("sample*/C*/lines.json")):
        try:
            data = json.loads(lines_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        for slice_data in data.values():
            if not isinstance(slice_data, dict):
                continue
            for key in LINE_KEYS:
                length = line_extent(slice_data.get(key))
                if length <= 1e-6:
                    continue
                sums[key] += length
                counts[key] += 1

    averages: dict[str, float] = {}
    for key in LINE_KEYS:
        if counts[key] == 0:
            print(f"[WARN] {key} の学習線長を計算できません。fallback={FALLBACK_LINE_LENGTH_PX}px")
            averages[key] = FALLBACK_LINE_LENGTH_PX
            continue
        averages[key] = sums[key] / counts[key]
    return averages


@torch.no_grad()
def predict_lines(
    models: list[TinyUNet],
    image: np.ndarray,
    vertebra: str,
    device: torch.device,
    average_line_lengths: dict[str, float],
    heatmap_threshold: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """アンサンブル推論で4本線を予測する。"""
    if heatmap_threshold is None:
        heatmap_threshold = {"mode": "adaptive", "min": 0.10, "peak_ratio": 0.4}

    x = torch.from_numpy(image).unsqueeze(0).to(device)
    vertebra_idx = torch.tensor(
        [VERTEBRA_TO_IDX.get(vertebra, 0)], device=device, dtype=torch.long,
    )

    heatmap_sum = None
    for model in models:
        pred = torch.sigmoid(model(x, vertebra_idx))
        if heatmap_sum is None:
            heatmap_sum = pred
        else:
            heatmap_sum = heatmap_sum + pred
    heatmap_avg = (heatmap_sum / len(models)).cpu().numpy()[0]

    lines: dict[str, Any] = {}
    for ch in range(4):
        line_name = f"line_{ch + 1}"
        result = detect_line_moments(
            heatmap_avg[ch],
            length_px=average_line_lengths.get(line_name, FALLBACK_LINE_LENGTH_PX),
            extend_ratio=1.0,
            clip=False,
            threshold=heatmap_threshold,
        )
        if result is None:
            lines[line_name] = None
            continue

        phi, rho = moments_to_phi_rho(result, IMAGE_SIZE)
        lines[line_name] = {
            "endpoints": result["endpoints"],
            "centroid": result["centroid"],
            "angle_deg": result["angle_deg"],
            "phi_rad": phi,
            "rho_normalized": rho,
            "line_length_px": average_line_lengths.get(line_name, FALLBACK_LINE_LENGTH_PX),
        }

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RSNAデータセットのmax_areaスライスに対する直線推論",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="モデルチェックポイントのディレクトリ",
    )
    parser.add_argument(
        "--fracture-dataset-dir",
        type=Path,
        default=FRACTURE_DATASET_DIR,
        help="fracture_datasetのパス",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=METADATA_DIR,
        help="processing_metadataのパス",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Unet" / "outputs" / "rsna_line_predictions",
        help="出力ディレクトリ",
    )
    parser.add_argument(
        "--training-dataset-dir",
        type=Path,
        default=DEFAULT_TRAINING_DATASET_DIR,
        help="学習に使った lines.json を含む dataset ディレクトリ",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--study-ids",
        nargs="*",
        default=None,
        help="特定のstudy IDのみ処理（省略時は全件）",
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    models = load_ensemble_models(args.checkpoint_dir, args.n_folds, device)
    if not models:
        print("[ERROR] モデルが1つもロードできませんでした")
        return
    print(f"[INFO] {len(models)} fold アンサンブルで推論します")

    average_line_lengths = compute_average_training_line_lengths(args.training_dataset_dir)
    print("[INFO] 学習データ平均線長(px): " + ", ".join(
        f"{key}={value:.1f}" for key, value in average_line_lengths.items()
    ))

    if args.study_ids:
        study_uids = args.study_ids
    else:
        study_uids = sorted(
            d.name
            for d in args.fracture_dataset_dir.iterdir()
            if d.is_dir()
        )
    print(f"[INFO] 対象スタディ数: {len(study_uids)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    skipped_no_meta = 0
    skipped_no_plane = 0
    skipped_no_data = 0
    success = 0

    for study_idx, study_uid in enumerate(study_uids):
        metadata = load_metadata(study_uid)
        if metadata is None:
            skipped_no_meta += 1
            continue

        study_dir = args.fracture_dataset_dir / study_uid
        study_result: dict[str, Any] = {"study_uid": study_uid, "vertebrae": {}}

        for vertebra in VERTEBRA_LEVELS:
            total += 1
            plane_index = find_max_area_plane_index(metadata, vertebra)
            if plane_index is None:
                skipped_no_plane += 1
                continue

            image = extract_max_area_input(study_dir, vertebra, plane_index)
            if image is None:
                skipped_no_data += 1
                continue

            lines = predict_lines(models, image, vertebra, device, average_line_lengths)
            study_result["vertebrae"][vertebra] = {
                "max_area_plane_index": plane_index,
                "predicted_lines": lines,
            }
            success += 1

        out_path = args.output_dir / f"{study_uid}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(study_result, f, indent=2, ensure_ascii=False)

        if (study_idx + 1) % 100 == 0:
            print(
                f"[PROGRESS] {study_idx + 1}/{len(study_uids)} studies "
                f"({success} predictions saved)"
            )

    print(f"\n[DONE] 推論完了")
    print(f"  対象椎体数: {total}")
    print(f"  成功: {success}")
    print(f"  メタデータなし: {skipped_no_meta}")
    print(f"  max_areaプレーンなし: {skipped_no_plane}")
    print(f"  データなし: {skipped_no_data}")
    print(f"  出力先: {args.output_dir}")


if __name__ == "__main__":
    main()
