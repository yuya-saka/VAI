"""bbox中心15断面へSDF補間した4領域maskを生成する。"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data_preprocessing.rsna_pipeline.segmentation_plane_sampling import (
    write_npy_atomic,
)

from .apply_sdf_segmentation import (
    build_boundary_interpolator,
    generate_region_masks,
)
from .constants import (
    DEFAULT_CKPT_DIR,
    FRACTURE_DATASET_DIR,
    PROCESSING_METADATA_DIR,
    TRAINING_DATASET_DIR,
)
from .inference import predict_5planes
from .model_io import compute_avg_line_lengths, load_models

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BBOX_CENTERED_DATASET_DIR = (
    PROJECT_ROOT / "data" / "rsna_data" / "bbox_centered_dataset"
)


def process_level_runs(
    study_id: str,
    level: str,
    run_directories: list[Path],
    source_level_directory: Path,
    models: list,
    device: torch.device,
    avg_lengths: dict[str, float],
    processing_metadata: dict[str, Any],
    *,
    overwrite: bool,
) -> list[dict[str, Any]]:
    """1椎骨の線推論を共有して全bbox runの4領域maskを生成する。"""
    seg_ct = np.load(source_level_directory / "seg_ct.npy", allow_pickle=False)
    seg_mask = np.load(
        source_level_directory / "seg_vertebra_mask.npy",
        allow_pickle=False,
    )
    slice_spacing_mm = float(
        processing_metadata["dicom_geometry"]["median_slice_spacing_mm"]
    )
    plane_predictions = predict_5planes(
        models,
        seg_ct,
        seg_mask,
        level,
        device,
        avg_lengths,
    )
    interpolator = build_boundary_interpolator(
        plane_predictions,
        avg_lengths,
        slice_spacing_mm,
    )
    if len(interpolator.available_lines) < 4:
        return [
            {
                "run": str(run_directory),
                "status": "failed",
                "reason": f"insufficient anchors: {interpolator.available_lines}",
            }
            for run_directory in run_directories
        ]

    max_area_position = float(
        processing_metadata["vertebrae"][level]["orientation"]["max_area_position_mm"]
    )
    results: list[dict[str, Any]] = []
    for run_directory in run_directories:
        output_path = run_directory / "region_4class.npy"
        if output_path.exists() and not overwrite:
            results.append({"run": str(run_directory), "status": "skipped"})
            continue
        run_metadata = json.loads(
            (run_directory / "metadata.json").read_text(encoding="utf-8")
        )
        z_targets = [
            float(plane["position_mm"]) - max_area_position
            for plane in run_metadata["planes"]
        ]
        vertebra_mask = np.load(
            run_directory / "vertebra_mask.npy",
            allow_pickle=False,
        )
        region_mask, plane_stats = generate_region_masks(
            interpolator,
            z_targets,
            vertebra_mask,
        )
        write_npy_atomic(output_path, region_mask)
        report = {
            "status": "complete",
            "available_lines": interpolator.available_lines,
            "planes_ok": sum(bool(item["success"]) for item in plane_stats),
            "planes_total": len(plane_stats),
            "plane_stats": plane_stats,
        }
        (run_directory / "region_generation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        results.append({"run": str(run_directory), **report})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="bbox中心15断面へ4領域maskを生成する",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument(
        "--bbox-centered-dataset-dir",
        type=Path,
        default=BBOX_CENTERED_DATASET_DIR,
    )
    parser.add_argument(
        "--fracture-dataset-dir",
        type=Path,
        default=FRACTURE_DATASET_DIR,
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=PROCESSING_METADATA_DIR,
    )
    parser.add_argument(
        "--training-dataset-dir",
        type=Path,
        default=TRAINING_DATASET_DIR,
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--study-id", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()

    device = torch.device(
        f"cuda:{arguments.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    models = load_models(arguments.checkpoint_dir, arguments.n_folds, device)
    if not models:
        raise RuntimeError(f"モデルが見つかりません: {arguments.checkpoint_dir}")
    avg_lengths = compute_avg_line_lengths(arguments.training_dataset_dir)
    studies = sorted(
        path
        for path in arguments.bbox_centered_dataset_dir.iterdir()
        if path.is_dir()
        and (arguments.study_id is None or path.name == arguments.study_id)
    )
    total = complete = failed = skipped = 0
    started = time.monotonic()
    for study_directory in studies:
        study_id = study_directory.name
        processing_metadata = json.loads(
            (arguments.metadata_dir / f"{study_id}.json").read_text(encoding="utf-8")
        )
        for level_directory in sorted(study_directory.glob("C[1-7]")):
            level = level_directory.name
            run_directories = sorted(level_directory.glob("run_*"))
            source_level_directory = arguments.fracture_dataset_dir / study_id / level
            try:
                results = process_level_runs(
                    study_id,
                    level,
                    run_directories,
                    source_level_directory,
                    models,
                    device,
                    avg_lengths,
                    processing_metadata,
                    overwrite=arguments.overwrite,
                )
            except Exception:
                failed += len(run_directories)
                total += len(run_directories)
                print(f"[ERROR] {study_id}/{level}:\n{traceback.format_exc()}")
                continue
            for result in results:
                total += 1
                if result["status"] == "complete":
                    complete += 1
                elif result["status"] == "skipped":
                    skipped += 1
                else:
                    failed += 1
                    print(f"[WARN] {result}")
        print(
            f"{study_id}: total={total} complete={complete} "
            f"failed={failed} skipped={skipped}"
        )
    print(f"elapsed_seconds={time.monotonic() - started:.1f}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
