"""全高4線から領域形成率とz平滑性を評価する。"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_preprocessing.segmentation_dataset.generate_region_mask import (  # noqa: E402
    generate_region_mask,
)

from .metrics import smoothness_metrics  # noqa: E402
from .visualization import save_reformat_visualization  # noqa: E402

REGION_NAMES = ("body", "right_foramen", "left_foramen", "posterior")
LINE_KEYS = ("line_1", "line_2", "line_3", "line_4")


@dataclass
class MissingCounter:
    """領域欠損数を集計する。"""

    total: int = 0
    any_missing: int = 0
    missing_by_region: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in REGION_NAMES}
    )

    def update(self, areas: dict[str, int]) -> None:
        """1スライス分の領域面積を加える。"""
        self.total += 1
        missing = [name for name in REGION_NAMES if int(areas.get(name, 0)) == 0]
        if missing:
            self.any_missing += 1
        for name in missing:
            self.missing_by_region[name] += 1

    def summary(self) -> dict[str, Any]:
        """件数と率をJSON化可能なdictへ変換する。"""
        denominator = max(1, self.total)
        return {
            "slice_count": self.total,
            "any_missing_count": self.any_missing,
            "any_missing_rate": self.any_missing / denominator,
            "missing_by_region": {
                name: {
                    "count": count,
                    "rate": count / denominator,
                }
                for name, count in self.missing_by_region.items()
            },
        }


def _read_json(path: Path) -> Any:
    """JSONを読み込む。"""
    return json.loads(path.read_text(encoding="utf-8"))


def _manual_anchor_indices(
    annotation_root: Path,
    sample: str,
    vertebra: str,
) -> list[int]:
    """手動lines.jsonで4線が揃うz indexを返す。"""
    path = annotation_root / sample / vertebra / "lines.json"
    if not path.exists():
        return []
    data = _read_json(path)
    return sorted(
        int(index)
        for index, lines in data.items()
        if isinstance(lines, dict)
        and all(
            isinstance(lines.get(key), list) and len(lines[key]) >= 2
            for key in LINE_KEYS
        )
    )


def _distance_scope(
    slice_index: int,
    anchors: list[int],
    spacing_mm: float,
    bin_width_mm: float,
) -> tuple[str, str]:
    """帯内外とアンカー帯からの距離binを返す。"""
    if not anchors:
        return "unknown", "unknown"
    lower, upper = min(anchors), max(anchors)
    if lower <= slice_index <= upper:
        return "inside", "inside"
    distance_slices = (
        lower - slice_index if slice_index < lower else slice_index - upper
    )
    distance_mm = distance_slices * spacing_mm
    bin_lower = math.floor(distance_mm / bin_width_mm) * bin_width_mm
    bin_upper = bin_lower + bin_width_mm - spacing_mm
    return "outside", f"{bin_lower:.1f}-{bin_upper:.1f}mm"


def _load_mask(path: Path, image_size: int) -> np.ndarray:
    """椎体maskを二値配列として読む。"""
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if mask.shape != (image_size, image_size):
        mask = np.asarray(
            Image.fromarray(mask).resize(
                (image_size, image_size),
                resample=Image.Resampling.NEAREST,
            )
        )
    return (mask > 0).astype(np.uint8)


def _surface_smoothness(surface: dict[str, Any]) -> dict[str, Any]:
    """椎体ごと・線ごとのz平滑性を計算する。"""
    output: dict[str, Any] = {}
    ordered_slices = sorted(int(index) for index in surface)
    for line_name in LINE_KEYS:
        centroids = []
        angles = []
        for slice_index in ordered_slices:
            line = surface[str(slice_index)][line_name]
            centroids.append(line["centroid_math"])
            angles.append(line["normal_angle_deg"])
        output[line_name] = smoothness_metrics(
            np.asarray(centroids, dtype=np.float64),
            np.asarray(angles, dtype=np.float64),
        )
    return output


def _save_reformat(
    dense_root: Path,
    sample: str,
    vertebra: str,
    lines: dict[str, Any],
    output_path: Path,
) -> None:
    """密CT stackと予測線からリフォーマット図を保存する。"""
    image_paths = sorted(
        (dense_root / sample / vertebra / "images").glob("slice_*.png")
    )
    available = {
        int(path.stem.split("_")[-1]): path
        for path in image_paths
        if str(int(path.stem.split("_")[-1])) in lines
    }
    slice_indices = sorted(available)
    if not slice_indices:
        return
    ct_stack = np.stack(
        [
            np.asarray(Image.open(available[index]).convert("L"), dtype=np.float32)
            / 255.0
            for index in slice_indices
        ],
        axis=0,
    )
    save_reformat_visualization(
        ct_stack,
        slice_indices,
        lines,
        output_path,
    )


def evaluate_prediction_tree(
    prediction_root: Path,
    dense_root: Path,
    annotation_root: Path,
    output_root: Path,
    image_size: int,
    spacing_mm: float,
    bin_width_mm: float,
) -> dict[str, Any]:
    """予測tree全体の領域欠損率と平滑性を集計する。"""
    counters: dict[str, MissingCounter] = defaultdict(MissingCounter)
    distance_counters: dict[str, MissingCounter] = defaultdict(MissingCounter)
    vertebra_results: list[dict[str, Any]] = []
    for lines_path in sorted(prediction_root.glob("sample*/C*/lines.json")):
        sample = lines_path.parents[1].name
        vertebra = lines_path.parent.name
        lines = _read_json(lines_path)
        surface_path = lines_path.with_name("surface.json")
        surface = _read_json(surface_path)
        anchors = _manual_anchor_indices(annotation_root, sample, vertebra)
        failures = 0
        processed = 0
        for slice_key, slice_lines in sorted(
            lines.items(),
            key=lambda item: int(item[0]),
        ):
            slice_index = int(slice_key)
            mask_path = (
                dense_root
                / sample
                / vertebra
                / "masks"
                / f"slice_{slice_index:03d}.png"
            )
            if not mask_path.exists():
                continue
            try:
                _, debug = generate_region_mask(
                    line_1=slice_lines["line_1"],
                    line_2=slice_lines["line_2"],
                    line_3=slice_lines["line_3"],
                    line_4=slice_lines["line_4"],
                    vertebra_mask=_load_mask(mask_path, image_size),
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                failures += 1
                continue
            scope, distance_bin = _distance_scope(
                slice_index,
                anchors,
                spacing_mm,
                bin_width_mm,
            )
            areas = {name: int(debug["region_areas"][name]) for name in REGION_NAMES}
            counters["all"].update(areas)
            counters[scope].update(areas)
            distance_counters[distance_bin].update(areas)
            processed += 1
        smoothness = _surface_smoothness(surface)
        vertebra_results.append(
            {
                "sample": sample,
                "vertebra": vertebra,
                "processed_slices": processed,
                "generation_failures": failures,
                "smoothness": smoothness,
            }
        )
        _save_reformat(
            dense_root,
            sample,
            vertebra,
            lines,
            output_root / "reformats" / f"{sample}_{vertebra}.png",
        )
    summary = {
        "baseline_reference": {
            "outside_any_missing_rate": 0.144,
            "distance_9.6_12.4mm_missing_rate": 0.288,
        },
        "scopes": {
            scope: counter.summary() for scope, counter in sorted(counters.items())
        },
        "distance_bins": {
            name: counter.summary()
            for name, counter in sorted(distance_counters.items())
        },
        "vertebrae": vertebra_results,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "region_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
