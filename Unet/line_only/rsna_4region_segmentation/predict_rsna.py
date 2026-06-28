"""RSNA データセットの最大面積スライスに対する直線推論スクリプト。

fracture_dataset/{study_uid}/{vertebra}/ の ct.npy と vertebra_mask.npy から
max_area_forced プレーンを抽出し、line_only モデルで 4 本線を予測する。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .constants import (
    CENTER_CHANNEL,
    DEFAULT_CKPT_DIR,
    FRACTURE_DATASET_DIR,
    PROJECT_ROOT,
    TRAINING_DATASET_DIR,
    VERTEBRA_LEVELS,
)
from .inference import predict_single_slice
from .metadata import find_max_area_plane_index, load_metadata
from .model_io import compute_avg_line_lengths, load_models


def extract_max_area_input(
    study_dir: Path,
    vertebra: str,
    plane_index: int,
) -> np.ndarray | None:
    """max_area プレーンから (2, 224, 224) float32 入力を抽出する。"""
    ct_path = study_dir / vertebra / "ct.npy"
    mask_path = study_dir / vertebra / "vertebra_mask.npy"
    if not ct_path.exists() or not mask_path.exists():
        return None

    ct = np.load(ct_path, allow_pickle=False)
    mask = np.load(mask_path, allow_pickle=False)

    ct_slice = ct[plane_index, CENTER_CHANNEL].astype(np.float32) / 255.0
    mask_slice = mask[plane_index].astype(np.float32)
    return np.stack([ct_slice, mask_slice], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RSNA データセットの max_area スライスに対する直線推論",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument(
        "--fracture-dataset-dir", type=Path, default=FRACTURE_DATASET_DIR
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Unet" / "outputs" / "rsna_line_predictions",
    )
    parser.add_argument(
        "--training-dataset-dir", type=Path, default=TRAINING_DATASET_DIR
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--study-ids", nargs="*", default=None)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    models = load_models(args.checkpoint_dir, args.n_folds, device)
    if not models:
        print("[ERROR] モデルが1つもロードできませんでした")
        return
    print(f"[INFO] {len(models)} fold アンサンブルで推論します")

    avg_lengths = compute_avg_line_lengths(args.training_dataset_dir)
    print(
        "[INFO] 学習データ平均線長(px): "
        + ", ".join(f"{k}={v:.1f}" for k, v in avg_lengths.items())
    )

    if args.study_ids:
        study_uids = args.study_ids
    else:
        study_uids = sorted(
            d.name for d in args.fracture_dataset_dir.iterdir() if d.is_dir()
        )
    print(f"[INFO] 対象スタディ数: {len(study_uids)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total = skipped_no_meta = skipped_no_plane = skipped_no_data = success = 0

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

            _hm, lines = predict_single_slice(
                models,
                image[0],
                image[1],
                vertebra,
                device,
                avg_lengths,
            )
            lines_dict: dict[str, Any] = {}
            for k, v in lines.items():
                if v is None:
                    lines_dict[k] = None
                else:
                    lines_dict[k] = {
                        "endpoints": v.endpoints,
                        "centroid": v.centroid,
                        "angle_deg": v.angle_deg,
                        "phi_rad": v.phi,
                        "rho_normalized": v.rho,
                        "line_length_px": v.line_length_px,
                    }

            study_result["vertebrae"][vertebra] = {
                "max_area_plane_index": plane_index,
                "predicted_lines": lines_dict,
            }
            success += 1

        out_path = args.output_dir / f"{study_uid}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(study_result, f, indent=2, ensure_ascii=False)

        if (study_idx + 1) % 100 == 0:
            print(
                f"[PROGRESS] {study_idx + 1}/{len(study_uids)} studies ({success} saved)"
            )

    print("\n[DONE] 推論完了")
    print(f"  対象椎体数: {total}")
    print(f"  成功: {success}")
    print(f"  メタデータなし: {skipped_no_meta}")
    print(f"  max_areaプレーンなし: {skipped_no_plane}")
    print(f"  データなし: {skipped_no_data}")
    print(f"  出力先: {args.output_dir}")


if __name__ == "__main__":
    main()
