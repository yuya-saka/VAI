"""頚椎データセットのポリライン可視化を生成するスクリプト。"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib_fontja  # noqa: F401
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / "dataset"
OUTPUT_DIR = ROOT_DIR / "Unet" / "preprocessing" / "output" / "phase1_polyline_viz"
MPLCONFIG_DIR = ROOT_DIR / ".tmp" / "matplotlib"

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
matplotlib.use("Agg")

TARGET_VERTEBRAE = ("C3", "C4", "C5", "C6", "C7")
LINE_KEYS = ("line_1", "line_2", "line_3", "line_4")

LINE_COLORS: dict[str, str] = {
    "line_1": "red",
    "line_2": "orange",
    "line_3": "blue",
    "line_4": "cyan",
}

LINE_LABELS: dict[str, str] = {
    "line_1": "line_1: right vertical boundary",
    "line_2": "line_2: right horizontal boundary",
    "line_3": "line_3: left vertical boundary",
    "line_4": "line_4: left horizontal boundary",
}

MASK_ALPHA = 0.3
MAX_PAIRS = 5
MAX_SLICES_PER_PAIR = 2


@dataclass(frozen=True)
class SliceVisualizationItem:
    """可視化対象スライスのメタ情報を保持する。"""

    sample_name: str
    vertebra: str
    slice_idx: int
    image_path: Path
    mask_path: Path
    line_map: dict[str, list[list[float]]]


@dataclass
class PreparedSlice:
    """描画に必要なデータを事前読込したスライス情報。"""

    item: SliceVisualizationItem
    ct_image: np.ndarray
    mask_image: np.ndarray


def sample_sort_key(sample_name: str) -> tuple[float, str]:
    """sample名を数値順で比較するためのキーを返す。"""
    match = re.fullmatch(r"sample([0-9]+(?:\.[0-9]+)?)", sample_name)
    if match is None:
        return (math.inf, sample_name)
    return (float(match.group(1)), sample_name)


def vertebra_sort_key(vertebra_name: str) -> int:
    """椎骨名(C3-C7)の並び順キーを返す。"""
    suffix = vertebra_name[1:]
    if suffix.isdigit():
        return int(suffix)
    return 999


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
    """lines.json を読み込んで辞書として返す。"""
    with lines_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def collect_valid_slices_for_vertebra(
    sample_name: str,
    vertebra_dir: Path,
) -> list[SliceVisualizationItem]:
    """1椎骨ディレクトリ内の有効スライス一覧を抽出する。"""
    lines_path = vertebra_dir / "lines.json"
    if not lines_path.exists():
        return []

    lines_data = load_lines_json(lines_path)
    valid_items: list[SliceVisualizationItem] = []

    sorted_keys = sorted(lines_data.keys(), key=lambda key: int(key))
    for key in sorted_keys:
        slice_annotation = lines_data.get(key)
        if not isinstance(slice_annotation, dict):
            continue

        parsed_line_map: dict[str, list[list[float]]] = {}
        is_valid_slice = True

        for line_key in LINE_KEYS:
            points = slice_annotation.get(line_key)
            if not is_valid_polyline(points):
                is_valid_slice = False
                break
            parsed_line_map[line_key] = [
                [float(point[0]), float(point[1])] for point in points
            ]

        if not is_valid_slice:
            continue

        slice_idx = int(key)
        image_path = vertebra_dir / "images" / f"slice_{slice_idx:03d}.png"
        mask_path = vertebra_dir / "masks" / f"slice_{slice_idx:03d}.png"

        if not image_path.exists() or not mask_path.exists():
            continue

        valid_items.append(
            SliceVisualizationItem(
                sample_name=sample_name,
                vertebra=vertebra_dir.name,
                slice_idx=slice_idx,
                image_path=image_path,
                mask_path=mask_path,
                line_map=parsed_line_map,
            )
        )

    return valid_items


def collect_all_valid_pairs(
    dataset_dir: Path,
) -> dict[tuple[str, str], list[SliceVisualizationItem]]:
    """データセット全体から (sample, vertebra) ごとの有効スライスを収集する。"""
    pair_map: dict[tuple[str, str], list[SliceVisualizationItem]] = {}

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
            if not vertebra_dir.is_dir():
                continue

            valid_slices = collect_valid_slices_for_vertebra(
                sample_name=sample_dir.name,
                vertebra_dir=vertebra_dir,
            )
            if not valid_slices:
                continue

            pair_map[(sample_dir.name, vertebra)] = valid_slices

    return pair_map


def select_pair_keys(
    pair_map: dict[tuple[str, str], list[SliceVisualizationItem]],
    max_pairs: int,
) -> list[tuple[str, str]]:
    """可視化対象の (sample, vertebra) ペアを最大数まで選ぶ。"""
    if max_pairs <= 0:
        return []

    all_pair_keys = sorted(
        pair_map.keys(),
        key=lambda item: (sample_sort_key(item[0]), vertebra_sort_key(item[1])),
    )

    selected_keys: list[tuple[str, str]] = []
    used_keys: set[tuple[str, str]] = set()

    # まずサンプルの多様性を優先して1サンプル1ペアずつ選ぶ
    seen_samples: set[str] = set()
    for pair_key in all_pair_keys:
        sample_name, _ = pair_key
        if sample_name in seen_samples:
            continue
        selected_keys.append(pair_key)
        used_keys.add(pair_key)
        seen_samples.add(sample_name)
        if len(selected_keys) >= max_pairs:
            return selected_keys

    # まだ枠がある場合は残りを順に補充する
    for pair_key in all_pair_keys:
        if pair_key in used_keys:
            continue
        selected_keys.append(pair_key)
        if len(selected_keys) >= max_pairs:
            break

    return selected_keys


def pick_slice_indices(total_count: int, max_count: int) -> list[int]:
    """1ペア内で可視化するスライス位置を 1〜2 件選択する。"""
    if total_count <= 0 or max_count <= 0:
        return []

    if total_count == 1 or max_count == 1:
        return [0]

    # 分布が見えるように先頭と末尾を採用
    return [0, total_count - 1][:max_count]


def select_visualization_items(
    pair_map: dict[tuple[str, str], list[SliceVisualizationItem]],
    max_pairs: int = MAX_PAIRS,
    max_slices_per_pair: int = MAX_SLICES_PER_PAIR,
) -> list[SliceVisualizationItem]:
    """可視化対象のスライス一覧を構築する。"""
    selected_items: list[SliceVisualizationItem] = []

    selected_pair_keys = select_pair_keys(pair_map=pair_map, max_pairs=max_pairs)
    for pair_key in selected_pair_keys:
        pair_items = pair_map[pair_key]
        item_indices = pick_slice_indices(
            total_count=len(pair_items),
            max_count=max_slices_per_pair,
        )

        for index in item_indices:
            selected_items.append(pair_items[index])

    return selected_items


def read_grayscale_image(image_path: Path) -> np.ndarray | None:
    """グレースケールPNGを読み込む。"""
    return cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)


def build_mask_overlay(mask_image: np.ndarray, alpha: float) -> np.ndarray:
    """二値マスクを赤色のRGBAオーバーレイに変換する。"""
    mask_binary = (mask_image > 0).astype(np.float32)
    overlay = np.zeros((mask_image.shape[0], mask_image.shape[1], 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 3] = mask_binary * alpha
    return overlay


def create_legend_handles() -> list[Line2D]:
    """凡例表示用のハンドルを生成する。"""
    return [
        Line2D(
            [0],
            [0],
            color=LINE_COLORS[line_key],
            linewidth=2.0,
            marker="o",
            markersize=5,
            label=LINE_LABELS[line_key],
        )
        for line_key in LINE_KEYS
    ]


def draw_polylines(ax: plt.Axes, line_map: dict[str, list[list[float]]]) -> None:
    """4本の折れ線と端点マーカーを描画する。"""
    for line_key in LINE_KEYS:
        points = np.asarray(line_map[line_key], dtype=np.float32)
        color = LINE_COLORS[line_key]

        ax.plot(points[:, 0], points[:, 1], color=color, linewidth=2.0)
        ax.scatter(
            [points[0, 0], points[-1, 0]],
            [points[0, 1], points[-1, 1]],
            c=color,
            s=30,
            edgecolors="white",
            linewidths=0.5,
        )


def render_slice(
    ax: plt.Axes,
    prepared: PreparedSlice,
    show_legend: bool,
    legend_handles: list[Line2D],
) -> None:
    """1スライス分の可視化を1つのaxesに描画する。"""
    ax.imshow(prepared.ct_image, cmap="gray", interpolation="nearest")
    ax.imshow(
        build_mask_overlay(prepared.mask_image, alpha=MASK_ALPHA),
        interpolation="nearest",
    )
    draw_polylines(ax=ax, line_map=prepared.item.line_map)

    if show_legend:
        ax.legend(handles=legend_handles, loc="lower right", fontsize=7, framealpha=0.9)

    ax.set_title(
        f"{prepared.item.sample_name} {prepared.item.vertebra} slice {prepared.item.slice_idx}",
        fontsize=10,
    )
    ax.set_axis_off()


def prepare_slices(items: list[SliceVisualizationItem]) -> list[PreparedSlice]:
    """描画前にCT画像とマスクを読み込んで検証する。"""
    prepared_slices: list[PreparedSlice] = []

    for item in items:
        ct_image = read_grayscale_image(item.image_path)
        mask_image = read_grayscale_image(item.mask_path)

        if ct_image is None:
            print(f"[WARN] CT画像を読めません: {item.image_path}")
            continue
        if mask_image is None:
            print(f"[WARN] マスク画像を読めません: {item.mask_path}")
            continue

        prepared_slices.append(
            PreparedSlice(
                item=item,
                ct_image=ct_image,
                mask_image=mask_image,
            )
        )

    return prepared_slices


def save_single_figures(
    prepared_slices: list[PreparedSlice],
    output_dir: Path,
    legend_handles: list[Line2D],
) -> None:
    """各スライスの個別可視化画像を保存する。"""
    for prepared in prepared_slices:
        figure, axis = plt.subplots(figsize=(5.5, 5.5), dpi=140)
        render_slice(
            ax=axis,
            prepared=prepared,
            show_legend=True,
            legend_handles=legend_handles,
        )
        figure.tight_layout()

        output_path = (
            output_dir
            / f"{prepared.item.sample_name}_{prepared.item.vertebra}_slice{prepared.item.slice_idx}.png"
        )
        figure.savefig(output_path, bbox_inches="tight")
        plt.close(figure)


def save_combined_grid(
    prepared_slices: list[PreparedSlice],
    output_dir: Path,
    legend_handles: list[Line2D],
) -> Path | None:
    """全可視化対象を1枚のグリッド画像として保存する。"""
    if not prepared_slices:
        return None

    total = len(prepared_slices)
    cols = min(3, total)
    rows = int(math.ceil(total / cols))

    figure, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 5.2), dpi=140)
    axes_array = np.atleast_1d(axes).ravel()

    for axis, prepared in zip(axes_array, prepared_slices, strict=False):
        render_slice(
            ax=axis,
            prepared=prepared,
            show_legend=False,
            legend_handles=legend_handles,
        )

    for axis in axes_array[len(prepared_slices) :]:
        axis.set_axis_off()

    figure.legend(handles=legend_handles, loc="lower center", ncol=2, framealpha=0.95)
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))

    grid_path = output_dir / "combined_grid.png"
    figure.savefig(grid_path, bbox_inches="tight")
    plt.close(figure)
    return grid_path


def main() -> None:
    """可視化処理全体を実行する。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pair_map = collect_all_valid_pairs(DATASET_DIR)
    selected_items = select_visualization_items(
        pair_map=pair_map,
        max_pairs=MAX_PAIRS,
        max_slices_per_pair=MAX_SLICES_PER_PAIR,
    )

    if not selected_items:
        print("[INFO] 有効なスライスが見つかりませんでした。")
        return

    prepared_slices = prepare_slices(selected_items)
    if not prepared_slices:
        print("[INFO] 読み込み可能なスライスがありませんでした。")
        return

    legend_handles = create_legend_handles()

    save_single_figures(
        prepared_slices=prepared_slices,
        output_dir=OUTPUT_DIR,
        legend_handles=legend_handles,
    )
    grid_path = save_combined_grid(
        prepared_slices=prepared_slices,
        output_dir=OUTPUT_DIR,
        legend_handles=legend_handles,
    )

    selected_pair_set = {
        (prepared.item.sample_name, prepared.item.vertebra) for prepared in prepared_slices
    }

    print(f"[INFO] ペア数: {len(selected_pair_set)}")
    print(f"[INFO] スライス数: {len(prepared_slices)}")
    print(f"[INFO] 個別画像保存先: {OUTPUT_DIR}")
    if grid_path is not None:
        print(f"[INFO] グリッド画像: {grid_path}")


if __name__ == "__main__":
    main()
