"""3Dネイティブ4領域ソフトラベルのプロトタイプ検証。

既存の`region_4class.npy`は、15枚の補正後平面それぞれで線境界を推論し、
argmaxで焼いた2Dラベルを積んだものである。本スクリプトはそれを使わず、
以下3つの3D部品を直接fine z-grid上で評価し、椎体ごとに一度だけ3D積分してαを出す。

  - bbox 3D envelope (`bbox_geometry.render_bbox_run`): 単一plane交差ではなく
    任意平面群への断面化を再利用する。
  - NIfTI 3D椎体seg (`vertebrae_{level}.nii.gz`): per-plane vertebra_maskより
    完全な3D支持領域。
  - 線境界の連続SDF場 (`SDFBoundaryInterpolator`): z(mm)方向にcos/sin/rhoを
    線形補間する解析的な場であり、任意zで評価できる。

15枚の等間隔平面には一切依存しない。fine z-gridはbbox支持範囲と椎体の
robust_range_mmの和集合とし、椎体3Dsegが存在しないzは自動的に無効として
積分から除外する（`region_4class.npy`のgenerate_region_maskはTLS+局所補正を
含むためplane単位で評価し、結果を3D積分でまとめる）。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from data_preprocessing.rsna_pipeline.bbox_geometry import (
    BboxRun,
    load_metadata,
    load_study_bbox_runs,
    render_bbox_run,
)
from data_preprocessing.rsna_pipeline.mask_processing import (
    load_and_process_vertebra_mask,
)
from data_preprocessing.rsna_pipeline.plane_sampling import (
    DEFAULT_OUTPUT_SIZE,
    DEFAULT_PIXEL_SPACING_MM,
    PhysicalPlane,
    sample_nifti_physical_planes,
)
from data_preprocessing.segmentation_dataset.generate_region_mask import (
    generate_region_mask,
)

from .apply_sdf_segmentation import build_boundary_interpolator
from .constants import (
    DEFAULT_CKPT_DIR,
    FRACTURE_DATASET_DIR,
    PROCESSING_METADATA_DIR,
    REGION_NAMES,
    TRAINING_DATASET_DIR,
)
from .inference import predict_5planes
from .model_io import compute_avg_line_lengths, load_models

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "rsna_data"
TRAIN_IMAGES_DIR = DATA_DIR / "train_images"
SEGMENTATION_DIR = DATA_DIR / "segmentations"
BBOX_CSV = DATA_DIR / "train_bounding_boxes.csv"
BBOX_CENTERED_DATASET_DIR = DATA_DIR / "bbox_centered_dataset"

FINE_STEP_MM_DEFAULT = 0.5
GRID_MARGIN_MM = 5.0
GAMMA_SWEEP = (0.0, 0.25, 0.5, 0.75, 1.0)

# (study_id, level, run_id) — posterior優勢/body優勢/foramen優勢/低-高inside_fractionを
# 網羅するように既存527 runの実測から選んだ多様な20例。
EXAMPLES: tuple[tuple[str, str, int], ...] = (
    ("1.2.826.0.1.3680043.1981", "C6", 0),
    ("1.2.826.0.1.3680043.31681", "C7", 0),
    ("1.2.826.0.1.3680043.18141", "C7", 0),
    ("1.2.826.0.1.3680043.6620", "C2", 0),
    ("1.2.826.0.1.3680043.4338", "C7", 0),
    ("1.2.826.0.1.3680043.20120", "C1", 1),
    ("1.2.826.0.1.3680043.5783", "C2", 1),
    ("1.2.826.0.1.3680043.11227", "C1", 1),
    ("1.2.826.0.1.3680043.4338", "C6", 1),
    ("1.2.826.0.1.3680043.4338", "C6", 0),
    ("1.2.826.0.1.3680043.25772", "C4", 0),
    ("1.2.826.0.1.3680043.3759", "C2", 0),
    ("1.2.826.0.1.3680043.26034", "C1", 0),
    ("1.2.826.0.1.3680043.22678", "C1", 0),
    ("1.2.826.0.1.3680043.8330", "C6", 0),
    ("1.2.826.0.1.3680043.24307", "C2", 0),
    ("1.2.826.0.1.3680043.20773", "C6", 0),
    ("1.2.826.0.1.3680043.30524", "C7", 0),
    ("1.2.826.0.1.3680043.30177", "C3", 0),
    ("1.2.826.0.1.3680043.11901", "C2", 0),
)


@dataclass(frozen=True)
class GeometryAlignmentCheck:
    """既存パイプラインが保存した平面定義と、本スクリプトの再構築を比較する。"""

    position_mm: float
    center_diff_mm: float
    row_basis_diff: float
    column_basis_diff: float
    vertebra_mask_iou: float
    bbox_occupancy_relative_diff: float


@dataclass(frozen=True)
class Soft3DLabelResult:
    study_id: str
    level: str
    run_id: int
    n_planes_total: int
    n_planes_valid: int
    inside_fraction_3d: float
    total_occ_mass: float
    mass: tuple[float, float, float, float]
    volume3d: tuple[float, float, float, float]
    alignment: GeometryAlignmentCheck | None


def _reference_plane_and_position(
    vertebra_metadata: dict,
) -> tuple[PhysicalPlane, float]:
    """`build_bbox_centered_dataset._reference_plane`と同一のロジック。"""
    planes = vertebra_metadata["classifier_planes"]["planes"]
    reference = next(
        (plane for plane in planes if bool(plane.get("max_area_forced"))),
        planes[len(planes) // 2],
    )
    plane = PhysicalPlane(
        center=tuple(float(v) for v in reference["center_lps_mm"]),
        row_basis=tuple(float(v) for v in reference["row_basis_lps"]),
        column_basis=tuple(float(v) for v in reference["column_basis_lps"]),
    )
    return plane, float(reference["position_mm"])


def _shift_along_normal(plane: PhysicalPlane, offset_mm: float) -> PhysicalPlane:
    """`build_bbox_centered_dataset._shifted_plane`と同一のロジック。"""
    row_basis = np.asarray(plane.row_basis, dtype=np.float64)
    column_basis = np.asarray(plane.column_basis, dtype=np.float64)
    normal = np.cross(row_basis, column_basis)
    normal = normal / np.linalg.norm(normal)
    center = np.asarray(plane.center, dtype=np.float64) + offset_mm * normal
    return PhysicalPlane(
        center=tuple(float(v) for v in center),
        row_basis=plane.row_basis,
        column_basis=plane.column_basis,
    )


def build_fine_grid_positions(
    robust_range_mm: tuple[float, float],
    bbox_support_range_mm: tuple[float, float],
    step_mm: float,
) -> np.ndarray:
    """椎体robust_rangeとbbox支持範囲の和集合をfine z-gridでカバーする。"""
    low = min(robust_range_mm[0], bbox_support_range_mm[0]) - GRID_MARGIN_MM
    high = max(robust_range_mm[1], bbox_support_range_mm[1]) + GRID_MARGIN_MM
    if low >= high:
        raise ValueError("Invalid fine-grid range")
    return np.arange(low, high + step_mm * 0.5, step_mm, dtype=np.float64)


def check_geometry_alignment(
    reference_plane: PhysicalPlane,
    reference_position_mm: float,
    run_metadata: dict,
    plane_index: int,
    run_directory: Path,
    vertebra_data: np.ndarray,
    affine_ras: np.ndarray,
    run: BboxRun,
) -> GeometryAlignmentCheck:
    """既存保存済み平面定義との一致を確認する（実装の座標整合QC）。"""
    stored_plane = run_metadata["planes"][plane_index]
    stored_position = float(stored_plane["position_mm"])
    rebuilt = _shift_along_normal(
        reference_plane, stored_position - reference_position_mm
    )

    center_diff = float(
        np.linalg.norm(
            np.asarray(rebuilt.center) - np.asarray(stored_plane["center_lps_mm"])
        )
    )
    row_diff = float(
        np.linalg.norm(
            np.asarray(rebuilt.row_basis) - np.asarray(stored_plane["row_basis_lps"])
        )
    )
    column_diff = float(
        np.linalg.norm(
            np.asarray(rebuilt.column_basis)
            - np.asarray(stored_plane["column_basis_lps"])
        )
    )

    sampled_mask = sample_nifti_physical_planes(
        vertebra_data,
        affine_ras,
        (rebuilt,),
        output_size=DEFAULT_OUTPUT_SIZE,
        pixel_spacing_mm=DEFAULT_PIXEL_SPACING_MM,
        interpolation_order=0,
        cval=0.0,
    )[0]
    stored_mask = np.load(run_directory / "vertebra_mask.npy")[plane_index]
    intersection = int(np.logical_and(sampled_mask > 0, stored_mask > 0).sum())
    union = int(np.logical_or(sampled_mask > 0, stored_mask > 0).sum())
    iou = intersection / union if union > 0 else 1.0

    sampled_occ = render_bbox_run(run, (rebuilt,))[0][0].astype(np.float64)
    stored_occ = np.load(run_directory / "bbox_corrected_occupancy.npy")[
        plane_index
    ].astype(np.float64)
    denom = max(stored_occ.sum(), sampled_occ.sum(), 1.0)
    relative_diff = float(np.abs(sampled_occ - stored_occ).sum() / denom)

    return GeometryAlignmentCheck(
        position_mm=stored_position,
        center_diff_mm=center_diff,
        row_basis_diff=row_diff,
        column_basis_diff=column_diff,
        vertebra_mask_iou=iou,
        bbox_occupancy_relative_diff=relative_diff,
    )


def compute_3d_soft_label(
    study_id: str,
    level: str,
    run_id: int,
    run: BboxRun,
    processing_metadata: dict,
    interpolator,
    run_directory: Path,
    *,
    step_mm: float = FINE_STEP_MM_DEFAULT,
    check_alignment: bool = True,
) -> Soft3DLabelResult:
    vertebra_metadata = processing_metadata["vertebrae"][level]
    reference_plane, reference_position = _reference_plane_and_position(
        vertebra_metadata
    )
    robust_range_mm = tuple(
        float(v) for v in vertebra_metadata["classifier_planes"]["robust_range_mm"]
    )
    max_area_position_mm = float(
        vertebra_metadata["orientation"]["max_area_position_mm"]
    )

    run_metadata = json.loads((run_directory / "metadata.json").read_text())
    bbox_support_range_mm = tuple(
        float(v) for v in run_metadata["bbox_support_range_mm"]
    )

    positions_mm = build_fine_grid_positions(
        robust_range_mm, bbox_support_range_mm, step_mm
    )
    fine_planes = tuple(
        _shift_along_normal(reference_plane, float(position - reference_position))
        for position in positions_mm
    )

    mask_path = SEGMENTATION_DIR / study_id / f"vertebrae_{level}.nii.gz"
    processed = load_and_process_vertebra_mask(mask_path, dicom_geometry=None)

    vertebra_stack = sample_nifti_physical_planes(
        processed.mask,
        processed.affine_ras,
        fine_planes,
        output_size=DEFAULT_OUTPUT_SIZE,
        pixel_spacing_mm=DEFAULT_PIXEL_SPACING_MM,
        interpolation_order=0,
        cval=0.0,
    )
    occupancy_stack = render_bbox_run(run, fine_planes)[0]
    occupancy_fraction = occupancy_stack.astype(np.float64) / 255.0

    z_targets = positions_mm - max_area_position_mm

    mass = np.zeros(4, dtype=np.float64)
    volume3d = np.zeros(4, dtype=np.float64)
    inside_mass = 0.0
    n_valid = 0
    for index in range(len(positions_mm)):
        mask_plane = vertebra_stack[index]
        if mask_plane.sum() == 0:
            continue
        lines = interpolator.get_lines(float(z_targets[index]))
        if lines is None:
            continue
        seg, _debug = generate_region_mask(
            line_1=lines["line_1"],
            line_2=lines["line_2"],
            line_3=lines["line_3"],
            line_4=lines["line_4"],
            vertebra_mask=mask_plane,
        )
        occ_plane = occupancy_fraction[index]
        for region_channel in range(4):
            region_area = seg[region_channel + 1]
            mass[region_channel] += float((occ_plane * region_area).sum())
            volume3d[region_channel] += float(region_area.sum())
        inside_mass += float((occ_plane * (seg[1:].sum(axis=0) > 0)).sum())
        n_valid += 1

    total_occ_mass = float(occupancy_fraction.sum())
    inside_fraction_3d = inside_mass / total_occ_mass if total_occ_mass > 0 else 0.0

    alignment: GeometryAlignmentCheck | None = None
    if check_alignment:
        center_plane_index = len(run_metadata["planes"]) // 2
        alignment = check_geometry_alignment(
            reference_plane,
            reference_position,
            run_metadata,
            center_plane_index,
            run_directory,
            processed.mask,
            processed.affine_ras,
            run,
        )

    return Soft3DLabelResult(
        study_id=study_id,
        level=level,
        run_id=run_id,
        n_planes_total=len(positions_mm),
        n_planes_valid=n_valid,
        inside_fraction_3d=inside_fraction_3d,
        total_occ_mass=total_occ_mass,
        mass=tuple(mass.tolist()),
        volume3d=tuple(volume3d.tolist()),
        alignment=alignment,
    )


def alpha_raw(mass: tuple[float, float, float, float]) -> np.ndarray:
    m = np.asarray(mass, dtype=np.float64)
    total = m.sum()
    return m / total if total > 0 else np.full(4, 0.25)


def alpha_enrichment(
    mass: tuple[float, float, float, float],
    volume3d: tuple[float, float, float, float],
    gamma: float,
) -> np.ndarray:
    m = np.asarray(mass, dtype=np.float64)
    v = np.asarray(volume3d, dtype=np.float64)
    v_floor = np.maximum(v, 1.0)
    density = m / (v_floor**gamma)
    total = density.sum()
    return density / total if total > 0 else np.full(4, 0.25)


def _print_result(result: Soft3DLabelResult) -> None:
    raw = alpha_raw(result.mass)
    print(
        f"\n{result.study_id}/{result.level}/run_{result.run_id:02d} "
        f"(planes valid={result.n_planes_valid}/{result.n_planes_total}, "
        f"inside_fraction_3d={result.inside_fraction_3d:.2f})"
    )
    print(f"  volume3d(voxel): {[f'{v:.0f}' for v in result.volume3d]}")
    print(f"  alpha_raw       : {np.round(raw, 3)}  top1={REGION_NAMES[1 + raw.argmax()]}")
    for gamma in GAMMA_SWEEP:
        enrich = alpha_enrichment(result.mass, result.volume3d, gamma)
        print(
            f"  alpha_gamma={gamma:.2f}   : {np.round(enrich, 3)}  "
            f"top1={REGION_NAMES[1 + enrich.argmax()]}"
        )
    if result.alignment is not None:
        a = result.alignment
        print(
            f"  [alignment QC] center_diff_mm={a.center_diff_mm:.4f} "
            f"row_diff={a.row_basis_diff:.2e} col_diff={a.column_basis_diff:.2e} "
            f"vertebra_mask_iou={a.vertebra_mask_iou:.4f} "
            f"bbox_occ_relative_diff={a.bbox_occupancy_relative_diff:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3Dネイティブ4領域ソフトラベルのプロトタイプ検証",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--step-mm", type=float, default=FINE_STEP_MM_DEFAULT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    arguments = parser.parse_args()

    device = torch.device(
        f"cuda:{arguments.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    models = load_models(arguments.checkpoint_dir, arguments.n_folds, device)
    if not models:
        raise RuntimeError(f"モデルが見つかりません: {arguments.checkpoint_dir}")
    avg_lengths = compute_avg_line_lengths(TRAINING_DATASET_DIR)

    examples = EXAMPLES[: arguments.limit] if arguments.limit else EXAMPLES
    by_study_level: dict[tuple[str, str], list[int]] = {}
    for study_id, level, run_id in examples:
        by_study_level.setdefault((study_id, level), []).append(run_id)

    all_results: list[Soft3DLabelResult] = []
    for (study_id, level), run_ids in by_study_level.items():
        metadata_path = PROCESSING_METADATA_DIR / f"{study_id}.json"
        processing_metadata = load_metadata(metadata_path)
        runs_by_level = load_study_bbox_runs(
            study_id,
            BBOX_CSV,
            TRAIN_IMAGES_DIR / study_id,
            processing_metadata,
        )
        runs = runs_by_level.get(level)
        if runs is None:
            print(f"[SKIP] {study_id}/{level}: no bbox runs found")
            continue

        seg_ct = np.load(
            FRACTURE_DATASET_DIR / study_id / level / "seg_ct.npy",
            allow_pickle=False,
        )
        seg_mask = np.load(
            FRACTURE_DATASET_DIR / study_id / level / "seg_vertebra_mask.npy",
            allow_pickle=False,
        )
        slice_spacing_mm = float(
            processing_metadata["dicom_geometry"]["median_slice_spacing_mm"]
        )
        plane_predictions = predict_5planes(
            models, seg_ct, seg_mask, level, device, avg_lengths
        )
        interpolator = build_boundary_interpolator(
            plane_predictions, avg_lengths, slice_spacing_mm
        )
        if len(interpolator.available_lines) < 4:
            print(
                f"[SKIP] {study_id}/{level}: insufficient anchors "
                f"{interpolator.available_lines}"
            )
            continue

        for run_id in run_ids:
            if run_id >= len(runs):
                print(f"[SKIP] {study_id}/{level}/run_{run_id:02d}: run not found")
                continue
            run_directory = (
                BBOX_CENTERED_DATASET_DIR / study_id / level / f"run_{run_id:02d}"
            )
            if not run_directory.is_dir():
                print(f"[SKIP] {study_id}/{level}/run_{run_id:02d}: directory missing")
                continue
            try:
                result = compute_3d_soft_label(
                    study_id,
                    level,
                    run_id,
                    runs[run_id],
                    processing_metadata,
                    interpolator,
                    run_directory,
                    step_mm=arguments.step_mm,
                )
            except Exception as error:  # noqa: BLE001
                print(f"[ERROR] {study_id}/{level}/run_{run_id:02d}: {error}")
                continue
            _print_result(result)
            all_results.append(result)

    _print_summary(all_results)

    if arguments.output_json is not None:
        payload = [
            {
                "study_id": r.study_id,
                "level": r.level,
                "run_id": r.run_id,
                "n_planes_total": r.n_planes_total,
                "n_planes_valid": r.n_planes_valid,
                "inside_fraction_3d": r.inside_fraction_3d,
                "mass": r.mass,
                "volume3d": r.volume3d,
                "alpha_raw": alpha_raw(r.mass).tolist(),
                "alpha_by_gamma": {
                    str(g): alpha_enrichment(r.mass, r.volume3d, g).tolist()
                    for g in GAMMA_SWEEP
                },
                "alignment": (
                    {
                        "center_diff_mm": r.alignment.center_diff_mm,
                        "row_basis_diff": r.alignment.row_basis_diff,
                        "column_basis_diff": r.alignment.column_basis_diff,
                        "vertebra_mask_iou": r.alignment.vertebra_mask_iou,
                        "bbox_occupancy_relative_diff": (
                            r.alignment.bbox_occupancy_relative_diff
                        ),
                    }
                    if r.alignment is not None
                    else None
                ),
            }
            for r in all_results
        ]
        arguments.output_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nSaved: {arguments.output_json}")


def _print_summary(results: list[Soft3DLabelResult]) -> None:
    if not results:
        print("\nNo results.")
        return
    print(f"\n=== Summary over {len(results)} examples ===")
    raw_top1 = np.zeros(4, dtype=int)
    gamma_top1 = {g: np.zeros(4, dtype=int) for g in GAMMA_SWEEP}
    flips_vs_raw = {g: 0 for g in GAMMA_SWEEP}
    inside_fractions = []
    align_ious = []
    align_center_diffs = []
    for r in results:
        raw = alpha_raw(r.mass)
        raw_top1[raw.argmax()] += 1
        for g in GAMMA_SWEEP:
            enrich = alpha_enrichment(r.mass, r.volume3d, g)
            gamma_top1[g][enrich.argmax()] += 1
            if enrich.argmax() != raw.argmax():
                flips_vs_raw[g] += 1
        inside_fractions.append(r.inside_fraction_3d)
        if r.alignment is not None:
            align_ious.append(r.alignment.vertebra_mask_iou)
            align_center_diffs.append(r.alignment.center_diff_mm)

    print("top-1 region distribution:")
    print(f"  raw          : {dict(zip(REGION_NAMES[1:], raw_top1.tolist()))}")
    for g in GAMMA_SWEEP:
        print(
            f"  gamma={g:.2f}    : {dict(zip(REGION_NAMES[1:], gamma_top1[g].tolist()))}"
            f"  (flips vs raw: {flips_vs_raw[g]}/{len(results)})"
        )
    print(
        f"\ninside_fraction_3d: mean={np.mean(inside_fractions):.3f} "
        f"median={np.median(inside_fractions):.3f}"
    )
    if align_ious:
        print(
            f"alignment QC: vertebra_mask_iou mean={np.mean(align_ious):.4f} "
            f"min={np.min(align_ious):.4f}; "
            f"center_diff_mm max={np.max(align_center_diffs):.6f}"
        )


if __name__ == "__main__":
    main()
