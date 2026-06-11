from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from Unet.preprocessing.generate_region_mask import (
    generate_region_mask,
    validate_region_mask,
)
from Unet.preprocessing.preprocess_all import (
    LINE_KEYS,
    REGION_COLORS_BGR,
    build_overlay_image,
    build_slice_filename,
    collect_valid_slices,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = ROOT_DIR / "dataset"
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "region_mask_evaluation" / "annotated_local_correction"
CLASS_NAMES = ("body", "right_foramen", "left_foramen", "posterior")
REVIEW_LIMIT = 100


def _dice_score(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.sum(first & second))
    denominator = int(np.sum(first)) + int(np.sum(second))
    if denominator == 0:
        return 1.0
    return 2.0 * intersection / denominator


def _comparison_image(
    image: np.ndarray,
    old_label: np.ndarray | None,
    new_label: np.ndarray,
) -> np.ndarray:
    new_overlay = build_overlay_image(image, new_label)
    if old_label is None:
        old_overlay = np.zeros_like(new_overlay)
        diff_overlay = np.zeros_like(new_overlay)
    else:
        old_overlay = build_overlay_image(image, old_label)
        diff_overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        changed = old_label != new_label
        diff_overlay[changed] = (0, 0, 255)
    return np.concatenate(
        [
            cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
            old_overlay,
            new_overlay,
            diff_overlay,
        ],
        axis=1,
    )


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for record in records for key in record})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def rebuild_annotated_masks(
    source_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    candidates = collect_valid_slices(source_root, list(f"C{i}" for i in range(1, 8)))
    records: list[dict[str, Any]] = []
    successful_reviews: list[tuple[float, Path, np.ndarray]] = []
    failures: list[dict[str, Any]] = []
    status_counter: Counter[str] = Counter()
    vertebra_stats: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for candidate in candidates:
        slice_filename = build_slice_filename(candidate.slice_idx)
        relative_dir = Path(candidate.sample) / candidate.vertebra
        mask = cv2.imread(str(candidate.mask_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(str(candidate.image_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or image is None:
            failure = {
                "sample": candidate.sample,
                "vertebra": candidate.vertebra,
                "slice": candidate.slice_idx,
                "status": "load_failure",
            }
            failures.append(failure)
            records.append(failure)
            status_counter["load_failure"] += 1
            continue

        try:
            seg_mask, debug_info = generate_region_mask(
                **{key: candidate.lines[key] for key in LINE_KEYS},
                vertebra_mask=mask,
            )
            validation = validate_region_mask(seg_mask, mask)
            if bool(validation.get("hard_fail", False)):
                raise ValueError("; ".join(validation.get("warnings", [])))
        except Exception as error:
            failure = {
                "sample": candidate.sample,
                "vertebra": candidate.vertebra,
                "slice": candidate.slice_idx,
                "status": "generation_failure",
                "reason": str(error),
            }
            failures.append(failure)
            records.append(failure)
            status_counter["generation_failure"] += 1
            continue

        new_label = np.argmax(seg_mask, axis=0).astype(np.uint8)
        mask_output = output_root / relative_dir / "gt_masks" / slice_filename
        overlay_output = output_root / relative_dir / "gt_overlays" / slice_filename
        mask_output.parent.mkdir(parents=True, exist_ok=True)
        overlay_output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_output), new_label)
        cv2.imwrite(str(overlay_output), build_overlay_image(image, new_label))

        old_path = source_root / relative_dir / "gt_masks" / slice_filename
        old_label = cv2.imread(str(old_path), cv2.IMREAD_GRAYSCALE)
        record: dict[str, Any] = {
            "sample": candidate.sample,
            "vertebra": candidate.vertebra,
            "slice": candidate.slice_idx,
            "status": "success",
            "component_count": int(debug_info["component_count"]),
            "corrected_pixels": int(debug_info.get("corrected_pixels", 0)),
            "correction_status": str(debug_info.get("correction_status", "")),
            "changed_pixels": "",
            "changed_fraction": "",
        }
        if old_label is not None:
            inside = mask > 0
            changed_pixels = int(np.sum((old_label != new_label) & inside))
            inside_count = max(int(np.sum(inside)), 1)
            record["changed_pixels"] = changed_pixels
            record["changed_fraction"] = changed_pixels / inside_count
            for label_value, class_name in enumerate(CLASS_NAMES, start=1):
                record[f"dice_{class_name}"] = _dice_score(
                    old_label == label_value,
                    new_label == label_value,
                )
                record[f"old_area_{class_name}"] = int(np.sum(old_label == label_value))
                record[f"new_area_{class_name}"] = int(np.sum(new_label == label_value))
            comparison = _comparison_image(image, old_label, new_label)
            successful_reviews.append(
                (
                    float(record["changed_fraction"]),
                    relative_dir / slice_filename,
                    comparison,
                )
            )

        records.append(record)
        vertebra_stats[candidate.vertebra].append(record)
        status_counter["success"] += 1

    review_root = output_root / "review_largest_changes_current"
    for rank, (_, relative_path, comparison) in enumerate(
        sorted(successful_reviews, key=lambda item: item[0], reverse=True)[
            :REVIEW_LIMIT
        ],
        start=1,
    ):
        review_path = review_root / (
            f"{rank:03d}_{relative_path.parent.parent.name}_"
            f"{relative_path.parent.name}_{relative_path.name}"
        )
        review_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(review_path), comparison)

    successful_records = [
        record for record in records if record.get("status") == "success"
    ]
    dice_summary = {}
    for class_name in CLASS_NAMES:
        values = [
            float(record[f"dice_{class_name}"])
            for record in successful_records
            if f"dice_{class_name}" in record
        ]
        dice_summary[class_name] = {
            "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "minimum": float(np.min(values)) if values else None,
        }

    summary = {
        "method": "half_plane_with_local_posterior_correction",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "candidate_count": len(candidates),
        "status_counts": dict(status_counter),
        "success_rate": status_counter["success"] / max(len(candidates), 1),
        "old_new_dice": dice_summary,
        "failure_count": len(failures),
        "review_count": min(len(successful_reviews), REVIEW_LIMIT),
        "review_directory": str(review_root),
        "review_columns": ["image", "old_overlay", "new_overlay", "changed_red"],
        "colors_bgr": {
            str(index): list(color) for index, color in enumerate(REGION_COLORS_BGR)
        },
    }
    _write_csv(output_root / "per_slice_metrics.csv", records)
    _save_json(output_root / "summary.json", summary)
    _save_json(output_root / "failures.json", failures)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="手動アノテーション画像の中央posterior侵入を局所補正する",
    )
    parser.add_argument("--source_root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    summary = rebuild_annotated_masks(args.source_root, args.output_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
