"""保存済みGTマスクを可視化するスクリプト。"""

from __future__ import annotations

import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".tmp" / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib_fontja  # noqa: F401
import matplotlib.pyplot as plt

DATASET_DIR = ROOT_DIR / "dataset"
OUTPUT_DIR = ROOT_DIR / "Unet" / "preprocessing" / "output" / "gt_mask_viz"

TARGET_VERTEBRAE = ("C1", "C2", "C3", "C4", "C5", "C6", "C7")
LINE_KEYS = ("line_1", "line_2", "line_3", "line_4")

LINE_COLORS: dict[str, str] = {
    "line_1": "red",
    "line_2": "orange",
    "line_3": "blue",
    "line_4": "cyan",
}

REGION_OVERLAY_SPECS: tuple[tuple[int, tuple[float, float, float]], ...] = (
    (1, (0.0, 1.0, 0.0)),
    (2, (1.0, 0.0, 0.0)),
    (3, (0.0, 0.0, 1.0)),
    (4, (1.0, 1.0, 0.0)),
)

OVERLAY_ALPHA = 0.4
TARGET_SLICE_COUNT = 20
RANDOM_SEED = 42
GRID_COLUMNS = 4


@dataclass(frozen=True)
class SliceCandidate:
    """可視化候補スライスのパス情報を保持する。"""

    sample_name: str
    vertebra: str
    slice_idx: int
    lines_path: Path
    image_path: Path
    gt_mask_path: Path


@dataclass
class ProcessedSlice:
    """描画に必要な読み込み済みデータを保持する。"""

    candidate: SliceCandidate
    ct_image: np.ndarray
    seg_mask: np.ndarray
    lines: dict[str, list[list[float]]]


def sample_sort_key(sample_name: str) -> tuple[float, str]:
    """sample名を数値順で比較するためのキーを返す。"""
    match = re.fullmatch(r"sample([0-9]+(?:\.[0-9]+)?)", sample_name)
    if match is None:
        return (math.inf, sample_name)
    return (float(match.group(1)), sample_name)


def vertebra_sort_key(vertebra_name: str) -> int:
    """椎骨名(C1-C7)の並び順キーを返す。"""
    suffix = vertebra_name[1:]
    if suffix.isdigit():
        return int(suffix)
    return 999


def candidate_sort_key(item: SliceCandidate) -> tuple[float, str, int, int]:
    """候補スライスの安定ソートキーを返す。"""
    return (
        sample_sort_key(item.sample_name)[0],
        item.sample_name,
        vertebra_sort_key(item.vertebra),
        item.slice_idx,
    )


def is_valid_point(point: Any) -> bool:
    """点データが [x, y] 形式であるかを検証する。"""
    if not isinstance(point, list | tuple):
        return False
    if len(point) != 2:
        return False
    x, y = point
    return isinstance(x, int | float) and isinstance(y, int | float)


def is_valid_polyline(points: Any) -> bool:
    """折れ線が要件(2点以上)を満たすか判定する。"""
    if not isinstance(points, list):
        return False
    if len(points) < 2:
        return False
    return all(is_valid_point(point) for point in points)


def load_lines_json(lines_path: Path) -> dict[str, Any]:
    """lines.json を辞書として読み込む。"""
    with lines_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_slice_index(key: str) -> int | None:
    """lines.jsonのキーをスライス番号へ変換する。"""
    try:
        return int(key)
    except (TypeError, ValueError):
        return None


def slice_key_sort_key(key: str) -> tuple[float, str]:
    """lines.jsonキーの安全なソートキーを返す。"""
    parsed = parse_slice_index(key)
    if parsed is None:
        return (math.inf, key)
    return (float(parsed), key)


def load_gt_mask_as_seg(gt_mask_path: Path) -> np.ndarray:
    """GTラベルPNG(0-4)を5ch one-hot形式へ変換して返す。"""
    label = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
    if label is None:
        raise FileNotFoundError(f"GTマスクを読めません: {gt_mask_path}")

    if label.ndim != 2:
        raise ValueError(f"GTマスクの次元が不正です: {label.shape}")
    if np.any(label < 0) or np.any(label > 4):
        raise ValueError(f"GTマスクの値域が不正です(0-4想定): {gt_mask_path}")

    seg_mask = np.eye(5, dtype=np.uint8)[label].transpose(2, 0, 1)
    return seg_mask


def collect_slice_candidates(
    dataset_dir: Path,
) -> dict[tuple[str, str], list[SliceCandidate]]:
    """データセットから有効な可視化候補を収集する。"""
    pair_map: dict[tuple[str, str], list[SliceCandidate]] = {}

    sample_dirs = sorted(
        [
            directory
            for directory in dataset_dir.iterdir()
            if directory.is_dir() and directory.name.startswith("sample")
        ],
        key=lambda directory: sample_sort_key(directory.name),
    )

    for sample_dir in sample_dirs:
        for vertebra in TARGET_VERTEBRAE:
            vertebra_dir = sample_dir / vertebra
            lines_path = vertebra_dir / "lines.json"
            if not vertebra_dir.is_dir() or not lines_path.exists():
                continue

            lines_data = load_lines_json(lines_path)
            candidates: list[SliceCandidate] = []

            for key in sorted(lines_data.keys(), key=slice_key_sort_key):
                slice_idx = parse_slice_index(key)
                if slice_idx is None:
                    continue

                annotation = lines_data.get(key)
                if not isinstance(annotation, dict):
                    continue

                if any(not is_valid_polyline(annotation.get(line_key)) for line_key in LINE_KEYS):
                    continue

                image_path = vertebra_dir / "images" / f"slice_{slice_idx:03d}.png"
                gt_mask_path = vertebra_dir / "gt_masks" / f"slice_{slice_idx:03d}.png"
                if not image_path.exists() or not gt_mask_path.exists():
                    continue

                candidates.append(
                    SliceCandidate(
                        sample_name=sample_dir.name,
                        vertebra=vertebra,
                        slice_idx=slice_idx,
                        lines_path=lines_path,
                        image_path=image_path,
                        gt_mask_path=gt_mask_path,
                    )
                )

            if candidates:
                pair_map[(sample_dir.name, vertebra)] = sorted(
                    candidates,
                    key=lambda item: item.slice_idx,
                )

    return pair_map


def select_random_slices(
    pair_map: dict[tuple[str, str], list[SliceCandidate]],
    target_count: int,
    random_seed: int,
) -> list[SliceCandidate]:
    """各(sample, vertebra)で乱択しつつ目標枚数までスライスを選ぶ。"""
    if target_count <= 0:
        return []

    random.seed(random_seed)

    pair_keys = sorted(
        pair_map.keys(),
        key=lambda item: (sample_sort_key(item[0]), vertebra_sort_key(item[1])),
    )

    first_pass: list[SliceCandidate] = []
    first_pass_ids: set[tuple[str, str, int]] = set()

    for pair_key in pair_keys:
        pair_items = pair_map[pair_key]
        if not pair_items:
            continue
        chosen = random.choice(pair_items)
        first_pass.append(chosen)
        first_pass_ids.add((chosen.sample_name, chosen.vertebra, chosen.slice_idx))

    if len(first_pass) >= target_count:
        sampled = random.sample(first_pass, target_count)
        return sorted(sampled, key=candidate_sort_key)

    remaining_pool: list[SliceCandidate] = []
    for pair_items in pair_map.values():
        for item in pair_items:
            item_id = (item.sample_name, item.vertebra, item.slice_idx)
            if item_id in first_pass_ids:
                continue
            remaining_pool.append(item)

    random.shuffle(remaining_pool)

    selected = list(first_pass)
    for item in remaining_pool:
        if len(selected) >= target_count:
            break
        selected.append(item)

    return sorted(selected, key=candidate_sort_key)


def extract_slice_lines(lines_data: dict[str, Any], slice_idx: int) -> dict[str, list[list[float]]]:
    """対象スライスの4本折れ線を lines.json から取り出す。"""
    key_candidates = (
        str(slice_idx),
        f"{slice_idx:03d}",
    )

    annotation: Any = None
    for key in key_candidates:
        if key in lines_data:
            annotation = lines_data[key]
            break

    if not isinstance(annotation, dict):
        raise KeyError(f"lines.jsonにslice={slice_idx}が見つかりません")

    lines: dict[str, list[list[float]]] = {}
    for line_key in LINE_KEYS:
        points = annotation.get(line_key)
        if not is_valid_polyline(points):
            raise ValueError(f"{line_key} が不正です: slice={slice_idx}")
        lines[line_key] = [[float(point[0]), float(point[1])] for point in points]

    return lines


def load_gt_slice(item: SliceCandidate) -> ProcessedSlice:
    """CT/GTマスク/線情報を読み込み、描画用データを返す。"""
    ct_img = cv2.imread(str(item.image_path), cv2.IMREAD_GRAYSCALE)
    if ct_img is None:
        raise FileNotFoundError(f"CT画像を読めません: {item.image_path}")

    seg_mask = load_gt_mask_as_seg(item.gt_mask_path)
    if seg_mask.ndim != 3 or seg_mask.shape[0] != 5:
        raise ValueError(f"seg_maskの形状が不正です: {seg_mask.shape}")

    lines_data = load_lines_json(item.lines_path)
    lines = extract_slice_lines(lines_data, item.slice_idx)

    return ProcessedSlice(
        candidate=item,
        ct_image=ct_img,
        seg_mask=seg_mask.astype(np.uint8),
        lines=lines,
    )


def build_region_overlay(seg_mask: np.ndarray, alpha: float = OVERLAY_ALPHA) -> np.ndarray:
    """4領域をRGBAで重ねるためのオーバーレイ配列を作る。"""
    height, width = seg_mask.shape[1], seg_mask.shape[2]
    overlay = np.zeros((height, width, 4), dtype=np.float32)

    for channel, rgb in REGION_OVERLAY_SPECS:
        region = seg_mask[channel] > 0
        if not np.any(region):
            continue
        overlay[region, 0] = rgb[0]
        overlay[region, 1] = rgb[1]
        overlay[region, 2] = rgb[2]
        overlay[region, 3] = alpha

    return overlay


def draw_polylines(ax: plt.Axes, lines: dict[str, list[list[float]]]) -> None:
    """4本の折れ線を指定色で描画する。"""
    for line_key in LINE_KEYS:
        points = np.asarray(lines[line_key], dtype=np.float32)
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=LINE_COLORS[line_key],
            linewidth=1.5,
        )


def render_slice(ax: plt.Axes, processed: ProcessedSlice) -> None:
    """1スライス分のCT背景+領域+線を描画する。"""
    ax.imshow(processed.ct_image, cmap="gray", interpolation="nearest")
    ax.imshow(build_region_overlay(processed.seg_mask, alpha=OVERLAY_ALPHA), interpolation="nearest")
    draw_polylines(ax=ax, lines=processed.lines)

    ax.set_title(
        (
            f"{processed.candidate.sample_name} "
            f"{processed.candidate.vertebra} "
            f"slice{processed.candidate.slice_idx}"
        ),
        fontsize=9,
    )
    ax.set_axis_off()


def save_single_figure(processed: ProcessedSlice, output_dir: Path) -> Path:
    """1スライスの可視化画像を保存してパスを返す。"""
    figure, axis = plt.subplots(figsize=(5.6, 5.6), dpi=140)
    render_slice(ax=axis, processed=processed)
    figure.tight_layout()

    output_path = (
        output_dir
        / f"{processed.candidate.sample_name}_{processed.candidate.vertebra}_slice{processed.candidate.slice_idx}.png"
    )
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_combined_grid(
    processed_items: list[ProcessedSlice],
    output_dir: Path,
    columns: int = GRID_COLUMNS,
) -> Path | None:
    """全スライス可視化を4列グリッドで保存する。"""
    if not processed_items:
        return None

    rows = int(math.ceil(len(processed_items) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 4.8, rows * 4.8), dpi=140)
    axes_array = np.atleast_1d(axes).ravel()

    for axis, processed in zip(axes_array, processed_items, strict=False):
        render_slice(ax=axis, processed=processed)

    for axis in axes_array[len(processed_items) :]:
        axis.set_axis_off()

    figure.tight_layout()

    grid_path = output_dir / "gt_mask_grid.png"
    figure.savefig(grid_path, bbox_inches="tight")
    plt.close(figure)
    return grid_path


def main() -> None:
    """GTマスク可視化を実行し、個別画像とグリッド画像を保存する。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pair_map = collect_slice_candidates(DATASET_DIR)
    if not pair_map:
        print("[INFO] 有効なC1-C7スライスが見つかりませんでした。")
        return

    selected_items = select_random_slices(
        pair_map=pair_map,
        target_count=TARGET_SLICE_COUNT,
        random_seed=RANDOM_SEED,
    )
    if not selected_items:
        print("[INFO] 可視化対象スライスが選択されませんでした。")
        return

    print(
        f"[INFO] 対象スライスを選択: {len(selected_items)}件 "
        f"(target={TARGET_SLICE_COUNT}, seed={RANDOM_SEED})"
    )

    processed_items: list[ProcessedSlice] = []

    for index, item in enumerate(selected_items, start=1):
        try:
            processed = load_gt_slice(item)
            output_path = save_single_figure(processed, OUTPUT_DIR)
            processed_items.append(processed)
            print(
                f"[INFO] ({index}/{len(selected_items)}) 保存完了: {output_path.name}"
            )
        except Exception as exc:
            print(
                "[WARN] "
                f"({index}/{len(selected_items)}) "
                f"{item.sample_name} {item.vertebra} slice{item.slice_idx} の処理に失敗: {exc}"
            )
            continue

    if not processed_items:
        print("[INFO] すべてのスライス処理に失敗しました。")
        return

    grid_path = save_combined_grid(
        processed_items=processed_items,
        output_dir=OUTPUT_DIR,
        columns=GRID_COLUMNS,
    )

    print(f"[INFO] 処理完了スライス数: {len(processed_items)}")
    print(f"[INFO] 個別画像保存先: {OUTPUT_DIR}")
    if grid_path is not None:
        print(f"[INFO] グリッド画像保存先: {grid_path}")


if __name__ == "__main__":
    main()
