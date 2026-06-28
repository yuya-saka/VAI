"""SDF 境界面補間を用いた 4 領域分割の全椎体・全 15 プレーンへの適用。

処理フロー:
  1. UNet モデル (5-fold アンサンブル) を 5 枚の seg_ct.npy で推論
  2. 予測線 (phi, rho) から SDFBoundaryInterpolator を構築
  3. 分類器用 15 プレーンの z オフセットをメタデータから読み込み
  4. 各プレーンで境界線を補間 → generate_region_mask
  5. region_4class.npy (15, 224, 224) を保存

出力:
  fracture_dataset/{study_id}/{level}/region_4class.npy
    - dtype: uint8
    - values: 0=背景, 1=椎体, 2=右椎間孔, 3=左椎間孔, 4=後方要素
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data_preprocessing.rsna_pipeline.sdf_boundary_interpolation import (
    SDFBoundaryInterpolator,
)
from data_preprocessing.segmentation_dataset.generate_region_mask import (
    generate_region_mask,
)

from .constants import (
    CENTER_CHANNEL,
    DEFAULT_CKPT_DIR,
    FRACTURE_DATASET_DIR,
    IMAGE_SIZE,
    LINE_KEYS,
    PROCESSING_METADATA_DIR,
    SEG_INDEX_OFFSETS,
    TRAINING_DATASET_DIR,
    VERTEBRA_LEVELS,
)
from .exclusions import load_excluded_levels, load_excluded_studies
from .inference import PredictedLine, predict_5planes
from .metadata import load_classifier_plane_z_offsets
from .model_io import compute_avg_line_lengths, load_models


def process_one_level(
    study_id: str,
    vertebra: str,
    level_dir: Path,
    models: list,
    device: torch.device,
    avg_lengths: dict[str, float],
    metadata_dir: Path,
) -> dict[str, Any]:
    """1 椎体レベルの 4 領域マスクを生成して保存する。"""
    seg_ct = np.load(level_dir / "seg_ct.npy")
    seg_mask = np.load(level_dir / "seg_vertebra_mask.npy")
    vert_mask = np.load(level_dir / "vertebra_mask.npy")

    z_cls_offsets, dz, _max_area_idx = load_classifier_plane_z_offsets(
        study_id,
        vertebra,
        metadata_dir,
    )
    z_seg_offsets = [idx * dz for idx in SEG_INDEX_OFFSETS]

    plane_preds = predict_5planes(
        models,
        seg_ct,
        seg_mask,
        vertebra,
        device,
        avg_lengths,
    )

    phi_rho_anchors: dict[str, list[tuple[float | None, float | None]]] = {}
    centroid_anchors: dict[str, list[tuple[float, float] | None]] = {}
    for line_key in LINE_KEYS:
        line_params: list[tuple[float | None, float | None]] = []
        line_centroids: list[tuple[float, float] | None] = []
        for plane_prediction in plane_preds:
            pred: PredictedLine | None = plane_prediction.get(line_key)
            if pred is None:
                line_params.append((None, None))
                line_centroids.append(None)
            else:
                line_params.append((pred.phi, pred.rho))
                line_centroids.append(pred.centroid)
        phi_rho_anchors[line_key] = line_params
        centroid_anchors[line_key] = line_centroids

    interp = SDFBoundaryInterpolator(
        phi_rho_anchors=phi_rho_anchors,
        z_offsets=z_seg_offsets,
        centre_idx=CENTER_CHANNEL,
        image_size=IMAGE_SIZE,
        centroid_anchors=centroid_anchors,
        line_lengths_px=avg_lengths,
    )

    if len(interp.available_lines) < 4:
        return {
            "status": "failed",
            "reason": f"insufficient anchors: {interp.available_lines}",
        }

    region_4class = np.zeros((15, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    plane_stats: list[dict[str, Any]] = []

    for plane_idx, z_target in enumerate(z_cls_offsets):
        lines = interp.get_lines(z_target)
        mask_plane = vert_mask[plane_idx].astype(np.uint8)

        if lines is None or mask_plane.sum() == 0:
            plane_stats.append({"plane": plane_idx, "success": False})
            continue

        try:
            seg, _ = generate_region_mask(
                line_1=lines["line_1"],
                line_2=lines["line_2"],
                line_3=lines["line_3"],
                line_4=lines["line_4"],
                vertebra_mask=mask_plane,
            )
            region_4class[plane_idx] = np.argmax(seg, axis=0).astype(np.uint8)
            plane_stats.append({"plane": plane_idx, "success": True})
        except Exception as e:
            plane_stats.append({"plane": plane_idx, "success": False, "reason": str(e)})

    np.save(level_dir / "region_4class.npy", region_4class)

    n_ok = sum(1 for s in plane_stats if s["success"])
    return {
        "status": "complete",
        "planes_ok": n_ok,
        "planes_total": 15,
        "available_lines": interp.available_lines,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SDF 境界面補間で 15 プレーンの 4 領域マスクを生成する",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument(
        "--fracture-dataset-dir", type=Path, default=FRACTURE_DATASET_DIR
    )
    parser.add_argument("--metadata-dir", type=Path, default=PROCESSING_METADATA_DIR)
    parser.add_argument(
        "--training-dataset-dir", type=Path, default=TRAINING_DATASET_DIR
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--study-id", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    models = load_models(args.checkpoint_dir, args.n_folds, device)
    if not models:
        raise RuntimeError(f"モデルが見つかりません: {args.checkpoint_dir}")
    print(f"[INFO] {len(models)} fold モデル読み込み完了")

    avg_lengths = compute_avg_line_lengths(args.training_dataset_dir)

    excluded_studies = load_excluded_studies()
    excluded_levels = load_excluded_levels()

    if args.study_id:
        study_ids = [args.study_id]
    else:
        study_ids = sorted(
            d.name
            for d in args.fracture_dataset_dir.iterdir()
            if d.is_dir() and d.name not in excluded_studies
        )
    print(f"[INFO] 対象 study: {len(study_ids)} 件")

    total = ok = failed = skipped = 0
    started = time.monotonic()

    for si, study_id in enumerate(study_ids):
        excl_levels = excluded_levels.get(study_id, set())
        for vertebra in VERTEBRA_LEVELS:
            if vertebra in excl_levels:
                continue
            level_dir = args.fracture_dataset_dir / study_id / vertebra
            if not (level_dir / "seg_ct.npy").exists():
                continue

            out_path = level_dir / "region_4class.npy"
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue

            total += 1
            try:
                result = process_one_level(
                    study_id,
                    vertebra,
                    level_dir,
                    models,
                    device,
                    avg_lengths,
                    args.metadata_dir,
                )
                if result["status"] == "complete":
                    ok += 1
                else:
                    failed += 1
                    print(f"  [WARN] {study_id}/{vertebra}: {result.get('reason')}")
            except Exception:
                failed += 1
                print(f"  [ERROR] {study_id}/{vertebra}:\n{traceback.format_exc()}")

        if (si + 1) % 100 == 0:
            elapsed = time.monotonic() - started
            rate = total / elapsed if elapsed > 0 else 0
            print(
                f"[{si + 1}/{len(study_ids)}] ok={ok} failed={failed} skipped={skipped} "
                f"({rate:.1f} levels/s)"
            )

    elapsed = time.monotonic() - started
    print(f"\n=== 完了 ({elapsed:.0f}s) ===")
    print(f"  OK: {ok}, 失敗: {failed}, スキップ: {skipped}, 合計: {total}")
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
