"""
line detection 評価結果を集計・可視化するスクリプト

使用例:
    uv run python Unet/debug/eval_error_viz.py \
        --exp-dir Unet/outputs/regularization/sig3.5 \
        --top-n 20 \
        --metric angle \
        --output-dir Unet/outputs/regularization/sig3.5/vis/error_viz
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MPL_CONFIG_DIR = Path(__file__).resolve().parents[2] / ".tmp" / "matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

# isort: off
import matplotlib_fontja  # noqa: E402, F401
import matplotlib.pyplot as plt  # noqa: E402
# isort: on


LINE_ORDER = ["line_1", "line_2", "line_3", "line_4"]
VERTEBRA_ORDER = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
PERCENTILES = [25, 50, 75, 90, 95]
PERCENTILE_COLORS = {
    25: "#6baed6",
    50: "#08519c",
    75: "#31a354",
    90: "#e6550d",
    95: "#a50f15",
}
METRIC_META = {
    "angle": {
        "column": "angle_error_deg",
        "label": "Angle Error [deg]",
        "title": "角度誤差",
    },
    "rho": {
        "column": "rho_error_px",
        "label": "Rho Error [px]",
        "title": "rho 誤差",
    },
    "perp": {
        "column": "perpendicular_dist_px",
        "label": "Perpendicular Distance [px]",
        "title": "垂線距離誤差",
    },
}
FILE_RE = re.compile(
    r"(?P<sample>.+?)_(?P<vertebra>C[1-7])_slice(?P<slice>\d+)_PRED_lines\.json$"
)


def parse_args() -> argparse.Namespace:
    """CLI 引数を取得する"""
    parser = argparse.ArgumentParser(
        description="fold 横断で line detection の誤差分布と worst sample を可視化する"
    )
    parser.add_argument(
        "--exp-dir",
        required=True,
        type=Path,
        help="outputs/{phase}/{name} ディレクトリ",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="worst sample として保存する slice 数",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_META.keys()),
        nargs="+",
        default=["angle", "rho"],
        help="worst sample の順位付けに使うメトリクス（複数指定可）",
    )
    parser.add_argument(
        "--sort-aggregate",
        choices=["max", "mean"],
        default="max",
        help="slice 単位の順位付け集約方法",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="可視化結果の保存先。省略時は exp-dir/vis/error_viz",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="rho のピクセル換算に使う画像サイズ",
    )
    return parser.parse_args()


def resolve_output_dir(exp_dir: Path, output_dir: Path | None) -> Path:
    """出力先ディレクトリを決定する"""
    if output_dir is not None:
        return output_dir
    return exp_dir / "vis" / "error_viz"


def discover_test_line_dirs(exp_dir: Path) -> list[tuple[str, Path]]:
    """test_lines ディレクトリを fold 単位で探索する"""
    candidates: list[tuple[str, Path]] = []

    vis_dir = exp_dir / "vis"
    if vis_dir.exists():
        for fold_dir in sorted(vis_dir.glob("fold*/test_lines")):
            candidates.append((fold_dir.parent.name, fold_dir))

    if not candidates and exp_dir.name == "vis":
        for fold_dir in sorted(exp_dir.glob("fold*/test_lines")):
            candidates.append((fold_dir.parent.name, fold_dir))

    if not candidates and exp_dir.name == "test_lines":
        candidates.append((exp_dir.parent.name, exp_dir))

    if not candidates:
        raise FileNotFoundError(
            f"test_lines が見つかりません: {exp_dir} "
            "(期待パス: exp-dir/vis/fold*/test_lines)"
        )

    return candidates


def safe_float(value: Any) -> float | None:
    """JSON 内の数値を float に変換する"""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def parse_json_name(json_path: Path) -> dict[str, Any]:
    """ファイル名から sample / vertebra / slice を取得する"""
    match = FILE_RE.match(json_path.name)
    if match is None:
        raise ValueError(f"ファイル名を解釈できません: {json_path.name}")
    return {
        "sample": match.group("sample"),
        "vertebra": match.group("vertebra"),
        "slice_idx": int(match.group("slice")),
    }


def load_line_records(exp_dir: Path) -> list[dict[str, Any]]:
    """全 fold の `_PRED_lines.json` を line 単位レコードに展開する"""
    records: list[dict[str, Any]] = []

    for fold_name, test_lines_dir in discover_test_line_dirs(exp_dir):
        test_dir = test_lines_dir.parent / "test"
        for json_path in sorted(test_lines_dir.glob("*_PRED_lines.json")):
            meta = parse_json_name(json_path)
            comparison_path = json_path.with_name(
                json_path.name.replace("_PRED_lines.json", "_comparison.png")
            )
            # test_lines/ のヒートマップ+ライン画像（3パネル: heatmap | pred | GT）
            heatmap_lines_path = json_path.with_name(
                json_path.name.replace("_PRED_lines.json", "_heatmap_lines.png")
            )

            with json_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

            metrics = payload.get("metrics", {})
            pred_lines = payload.get("pred_lines", {})

            for line_name in LINE_ORDER:
                metric_info = metrics.get(line_name) or {}
                pred_info = pred_lines.get(line_name) or {}

                records.append(
                    {
                        "fold": fold_name,
                        "sample": meta["sample"],
                        "vertebra": meta["vertebra"],
                        "slice_idx": meta["slice_idx"],
                        "line_name": line_name,
                        "json_path": str(json_path),
                        "comparison_path": str(comparison_path),
                        "has_comparison": comparison_path.exists(),
                        "heatmap_lines_path": str(heatmap_lines_path),
                        "angle_error_deg": safe_float(
                            metric_info.get("angle_error_deg")
                        ),
                        "rho_error_px": safe_float(metric_info.get("rho_error_px")),
                        "perpendicular_dist_px": safe_float(
                            metric_info.get("perpendicular_dist_px")
                        ),
                        "gt_phi": safe_float(metric_info.get("gt_phi")),
                        "gt_rho": safe_float(metric_info.get("gt_rho")),
                        "pred_phi": safe_float(metric_info.get("pred_phi")),
                        "pred_rho": safe_float(metric_info.get("pred_rho")),
                        "pred_angle_deg": safe_float(pred_info.get("angle_deg")),
                        "pred_length": safe_float(pred_info.get("length")),
                    }
                )

    if not records:
        raise FileNotFoundError(f"_PRED_lines.json が見つかりません: {exp_dir}")

    return records


def group_valid_values(
    records: list[dict[str, Any]], metric_column: str, group_field: str
) -> list[tuple[str, list[float]]]:
    """group ごとに有効値を集約する"""
    grouped: dict[str, list[float]] = defaultdict(list)

    for record in records:
        value = record.get(metric_column)
        if value is None:
            continue
        grouped[str(record[group_field])].append(float(value))

    if group_field == "vertebra":
        order = VERTEBRA_ORDER
    elif group_field == "line_name":
        order = LINE_ORDER
    else:
        order = sorted(grouped)

    result: list[tuple[str, list[float]]] = []
    for key in order:
        values = grouped.get(key, [])
        if values:
            result.append((key, values))
    return result


def compute_percentile_row(values: list[float]) -> dict[str, float]:
    """percentile 要約を返す"""
    arr = np.asarray(values, dtype=np.float64)
    stats = {
        "count": float(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
    for percentile in PERCENTILES:
        stats[f"p{percentile}"] = float(np.percentile(arr, percentile))
    return stats


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """辞書 list を CSV に保存する"""
    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_percentile_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """metric / group 単位の percentile 要約テーブルを作る"""
    rows: list[dict[str, Any]] = []

    for metric_name, metric_meta in METRIC_META.items():
        metric_column = metric_meta["column"]

        overall_values = [
            float(record[metric_column])
            for record in records
            if record.get(metric_column) is not None
        ]
        if overall_values:
            rows.append(
                {
                    "metric": metric_name,
                    "group_type": "overall",
                    "group_name": "all",
                    **compute_percentile_row(overall_values),
                }
            )

        for group_field, group_type in (
            ("vertebra", "vertebra"),
            ("line_name", "line"),
        ):
            for group_name, values in group_valid_values(
                records, metric_column, group_field
            ):
                rows.append(
                    {
                        "metric": metric_name,
                        "group_type": group_type,
                        "group_name": group_name,
                        **compute_percentile_row(values),
                    }
                )

    return rows


def format_stats_block(values: list[float]) -> str:
    """plot 右側に表示する要約文字列を作る"""
    stats = compute_percentile_row(values)
    return "\n".join(
        [
            f"n={int(stats['count'])}",
            f"mean={stats['mean']:.2f}",
            f"p25={stats['p25']:.2f}",
            f"p50={stats['p50']:.2f}",
            f"p75={stats['p75']:.2f}",
            f"p90={stats['p90']:.2f}",
            f"p95={stats['p95']:.2f}",
        ]
    )


def plot_overall_histogram(
    records: list[dict[str, Any]], metric_name: str, output_path: Path
) -> None:
    """全体分布ヒストグラムを保存する"""
    metric_meta = METRIC_META[metric_name]
    metric_column = metric_meta["column"]
    values = [
        float(record[metric_column])
        for record in records
        if record.get(metric_column) is not None
    ]
    if not values:
        return

    fig = plt.figure(figsize=(10, 5.5))
    ax = fig.add_subplot(111)

    bins = min(40, max(10, int(np.sqrt(len(values)))))
    ax.hist(values, bins=bins, color="#4c78a8", alpha=0.82, edgecolor="white")
    for percentile in PERCENTILES:
        x = float(np.percentile(values, percentile))
        ax.axvline(
            x,
            color=PERCENTILE_COLORS[percentile],
            linestyle="--",
            linewidth=1.5,
            label=f"p{percentile}={x:.2f}",
        )

    ax.set_title(f"{metric_meta['title']}の全体分布")
    ax.set_xlabel(metric_meta["label"])
    ax.set_ylabel("Count")
    ax.grid(alpha=0.18, axis="y")
    ax.legend(loc="upper right", fontsize=9)
    ax.text(
        1.02,
        0.98,
        format_stats_block(values),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_group_distribution(
    records: list[dict[str, Any]],
    metric_name: str,
    group_field: str,
    title: str,
    output_path: Path,
) -> None:
    """group 別の violin + box plot を保存する"""
    metric_meta = METRIC_META[metric_name]
    metric_column = metric_meta["column"]
    grouped = group_valid_values(records, metric_column, group_field)
    if not grouped:
        return

    labels = [label for label, _ in grouped]
    values_list = [values for _, values in grouped]
    positions = np.arange(1, len(labels) + 1)

    fig = plt.figure(figsize=(max(9, len(labels) * 1.3), 6.0))
    ax = fig.add_subplot(111)

    violin = ax.violinplot(
        values_list,
        positions=positions,
        widths=0.8,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in violin["bodies"]:
        body.set_facecolor("#9ecae1")
        body.set_alpha(0.6)
        body.set_edgecolor("#3182bd")

    ax.boxplot(
        values_list,
        positions=positions,
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "white", "alpha": 0.9},
        medianprops={"color": "#111111", "linewidth": 2},
    )

    for pos, values in zip(positions, values_list, strict=True):
        for percentile in PERCENTILES:
            y = float(np.percentile(values, percentile))
            marker = "D" if percentile in (90, 95) else "o"
            ax.scatter(
                pos,
                y,
                color=PERCENTILE_COLORS[percentile],
                s=28,
                marker=marker,
                zorder=4,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric_meta["label"])
    ax.set_title(title)
    ax.grid(alpha=0.18, axis="y")

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o" if p not in (90, 95) else "D",
            color="w",
            label=f"p{p}",
            markerfacecolor=PERCENTILE_COLORS[p],
            markersize=7,
        )
        for p in PERCENTILES
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=9,
        title="Percentile",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_slice_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """line 単位レコードを slice 単位に集約する"""
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        key = (
            str(record["fold"]),
            str(record["sample"]),
            str(record["vertebra"]),
            int(record["slice_idx"]),
        )
        grouped[key].append(record)

    slice_records: list[dict[str, Any]] = []
    for (fold, sample, vertebra, slice_idx), items in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "fold": fold,
            "sample": sample,
            "vertebra": vertebra,
            "slice_idx": slice_idx,
            "comparison_path": items[0]["comparison_path"],
            "has_comparison": items[0]["has_comparison"],
            "heatmap_lines_path": items[0]["heatmap_lines_path"],
            "n_valid_lines": 0,
        }

        for metric_name, metric_meta in METRIC_META.items():
            metric_column = metric_meta["column"]
            values = [
                float(item[metric_column])
                for item in items
                if item.get(metric_column) is not None
            ]
            summary[f"{metric_name}_max"] = float(np.max(values)) if values else None
            summary[f"{metric_name}_mean"] = (
                float(np.mean(values)) if values else None
            )

        per_line_parts: list[str] = []
        valid_line_count = 0
        for line_name in LINE_ORDER:
            item = next((row for row in items if row["line_name"] == line_name), None)
            if item is None:
                continue
            if item["angle_error_deg"] is not None:
                valid_line_count += 1
            line_metrics = []
            for metric_name in METRIC_META:
                column = METRIC_META[metric_name]["column"]
                value = item.get(column)
                if value is not None:
                    line_metrics.append(f"{metric_name}={value:.2f}")
            per_line_parts.append(f"{line_name}: " + ", ".join(line_metrics))

        summary["n_valid_lines"] = valid_line_count
        summary["line_summary"] = " | ".join(per_line_parts)
        slice_records.append(summary)

    return slice_records


def select_worst_slices(
    slice_records: list[dict[str, Any]],
    metric_name: str,
    aggregate: str,
    top_n: int,
) -> list[dict[str, Any]]:
    """指定 metric で worst slice を抽出する"""
    score_key = f"{metric_name}_{aggregate}"
    ranked = [
        {**record, "ranking_score": record[score_key]}
        for record in slice_records
        if record.get(score_key) is not None
    ]
    ranked.sort(key=lambda row: float(row["ranking_score"]), reverse=True)
    return ranked[:top_n]


def add_label_bar(image: np.ndarray, text: str) -> np.ndarray:
    """画像上部にラベルバーを追加する"""
    height, width = image.shape[:2]
    bar = np.zeros((32, width, 3), dtype=np.uint8)
    cv2.putText(
        bar,
        text[: min(len(text), 120)],
        (6, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 245, 210),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([bar, image])


def copy_worst_images(
    worst_slices: list[dict[str, Any]], metric_name: str, aggregate: str, output_dir: Path
) -> list[Path]:
    """worst sample のタイル画像（予測ヒートマップ + ライン比較）を保存する"""
    PANEL_W = 224
    PANEL_H = 224
    out_dir = output_dir / "worst_samples" / f"{metric_name}_{aggregate}"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for rank, row in enumerate(worst_slices, start=1):
        tile = _make_sample_tile(row, rank, metric_name, PANEL_W, PANEL_H)
        if tile is None:
            continue

        score = float(row["ranking_score"])
        stem = (
            f"rank{rank:02d}_{row['fold']}_{row['sample']}_"
            f"{row['vertebra']}_slice{int(row['slice_idx']):03d}_{metric_name}{score:.2f}"
        )
        destination = out_dir / f"{stem}.png"
        cv2.imwrite(str(destination), tile)
        saved_paths.append(destination)

    return saved_paths


def _load_resize(path: Path, w: int, h: int) -> np.ndarray | None:
    """画像を読み込んでリサイズする。失敗時は None を返す"""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _make_sample_tile(
    row: dict[str, Any],
    rank: int,
    metric_name: str,
    panel_w: int,
    panel_h: int,
) -> np.ndarray | None:
    """
    1サンプル分のタイルを作る

    上段: heatmap_lines.png（ヒートマップ | 予測線 | GT線 の3パネル）
    下段: comparison.png（GT線 | 予測線 の2パネル）を中央配置
    """
    tile_w = panel_w * 3  # 3パネル幅に統一

    # 上段: heatmap | pred lines | GT lines（3パネル）
    hm_lines = _load_resize(Path(str(row["heatmap_lines_path"])), tile_w, panel_h)
    if hm_lines is None:
        hm_lines = np.zeros((panel_h, tile_w, 3), dtype=np.uint8)
        cv2.putText(hm_lines, "heatmap_lines N/A", (8, panel_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)

    # 下段: GT lines | PRED lines（2パネル）→ 左右に黒帯を追加して3パネル幅に揃える
    comp = _load_resize(Path(str(row["comparison_path"])), panel_w * 2, panel_h)
    if comp is None:
        comp = np.zeros((panel_h, panel_w * 2, 3), dtype=np.uint8)
        cv2.putText(comp, "comparison N/A", (8, panel_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
    pad = panel_w // 2
    bottom_row = np.concatenate(
        [np.zeros((panel_h, pad, 3), dtype=np.uint8),
         comp,
         np.zeros((panel_h, tile_w - panel_w * 2 - pad, 3), dtype=np.uint8)],
        axis=1,
    )

    tile = np.concatenate([hm_lines, bottom_row], axis=0)  # (2H, 3W, 3)

    score = float(row["ranking_score"])
    label = (
        f"#{rank:02d} {row['fold']} {row['sample']} {row['vertebra']} "
        f"slice{int(row['slice_idx']):03d} {metric_name}={score:.2f}"
    )
    return add_label_bar(tile, label)


def build_worst_summary_grid(
    worst_slices: list[dict[str, Any]], metric_name: str, aggregate: str, output_dir: Path
) -> Path | None:
    """worst sample の GT/PRED ヒートマップ + ライン比較から summary grid を作る"""
    PANEL_W = 224
    PANEL_H = 224
    tiles: list[np.ndarray] = []

    for rank, row in enumerate(worst_slices, start=1):
        tile = _make_sample_tile(row, rank, metric_name, PANEL_W, PANEL_H)
        if tile is not None:
            tiles.append(tile)

    if not tiles:
        return None

    n_cols = min(2, max(1, math.ceil(math.sqrt(len(tiles)))))
    n_rows = math.ceil(len(tiles) / n_cols)
    tile_h, tile_w = tiles[0].shape[:2]
    canvas = np.zeros((n_rows * tile_h, n_cols * tile_w, 3), dtype=np.uint8)

    for idx, tile in enumerate(tiles):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        y0 = row_idx * tile_h
        x0 = col_idx * tile_w
        canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile

    grid_dir = output_dir / "worst_samples" / f"{metric_name}_{aggregate}"
    grid_dir.mkdir(parents=True, exist_ok=True)
    grid_path = grid_dir / "worst_summary_grid.png"
    cv2.imwrite(str(grid_path), canvas)
    return grid_path


def write_summary_text(
    records: list[dict[str, Any]],
    slice_records: list[dict[str, Any]],
    percentile_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """集計サマリーをテキストで保存する"""
    overall_lines = [
        f"line_records={len(records)}",
        f"slice_records={len(slice_records)}",
        f"folds={', '.join(sorted({str(record['fold']) for record in records}))}",
        "",
    ]

    for metric_name in METRIC_META:
        row = next(
            (
                r
                for r in percentile_rows
                if r["metric"] == metric_name
                and r["group_type"] == "overall"
                and r["group_name"] == "all"
            ),
            None,
        )
        if row is None:
            continue
        overall_lines.extend(
            [
                f"[{metric_name}]",
                f"  mean={row['mean']:.3f}",
                f"  p50={row['p50']:.3f}",
                f"  p90={row['p90']:.3f}",
                f"  p95={row['p95']:.3f}",
                "",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(overall_lines), encoding="utf-8")


def plot_bland_altman_angle(
    records: list[dict[str, Any]], output_path: Path
) -> None:
    """
    角度のBland-Altmanプロット

    X軸: (GT + 予測) / 2 [deg]
    Y軸: 予測 - GT [deg]（180°周期で符号付き差分）
    """
    means: list[float] = []
    diffs: list[float] = []

    for record in records:
        gt_phi = record.get("gt_phi")
        pred_phi = record.get("pred_phi")
        if gt_phi is None or pred_phi is None:
            continue
        gt_deg = math.degrees(gt_phi)
        pred_deg = math.degrees(pred_phi)
        # 180°周期の符号付き差分: [-90, 90) に正規化
        diff = ((pred_deg - gt_deg) + 90.0) % 180.0 - 90.0
        mean = (gt_deg + pred_deg) / 2.0
        means.append(mean)
        diffs.append(diff)

    if not means:
        return

    diffs_arr = np.array(diffs)
    bias = float(np.mean(diffs_arr))
    sd = float(np.std(diffs_arr))
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(means, diffs, alpha=0.35, s=12, color="#4c78a8")
    ax.axhline(bias, color="#e6550d", linewidth=1.8, label=f"平均差分 {bias:.2f}°")
    ax.axhline(loa_upper, color="#31a354", linewidth=1.4, linestyle="--",
               label=f"+1.96SD {loa_upper:.2f}°")
    ax.axhline(loa_lower, color="#31a354", linewidth=1.4, linestyle="--",
               label=f"-1.96SD {loa_lower:.2f}°")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")

    ax.set_xlabel("(GT + 予測) / 2 [deg]")
    ax.set_ylabel("予測 - GT [deg]")
    ax.set_title("Bland-Altman プロット（角度）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.18)

    stats_text = f"n={len(diffs)}\nbias={bias:.2f}\nSD={sd:.2f}\nLoA=[{loa_lower:.2f}, {loa_upper:.2f}]"
    ax.text(1.02, 0.98, stats_text, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9})

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_bland_altman_rho(
    records: list[dict[str, Any]], image_size: int, output_path: Path
) -> None:
    """
    rho のBland-Altmanプロット

    X軸: (GT + 予測) / 2 [px]
    Y軸: 予測 - GT [px]
    """
    D = math.sqrt(image_size**2 + image_size**2)
    means: list[float] = []
    diffs: list[float] = []

    for record in records:
        gt_rho = record.get("gt_rho")
        pred_rho = record.get("pred_rho")
        if gt_rho is None or pred_rho is None:
            continue
        gt_px = gt_rho * D
        pred_px = pred_rho * D
        means.append((gt_px + pred_px) / 2.0)
        diffs.append(pred_px - gt_px)

    if not means:
        return

    diffs_arr = np.array(diffs)
    bias = float(np.mean(diffs_arr))
    sd = float(np.std(diffs_arr))
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(means, diffs, alpha=0.35, s=12, color="#9ecae1")
    ax.axhline(bias, color="#e6550d", linewidth=1.8, label=f"平均差分 {bias:.2f} px")
    ax.axhline(loa_upper, color="#31a354", linewidth=1.4, linestyle="--",
               label=f"+1.96SD {loa_upper:.2f} px")
    ax.axhline(loa_lower, color="#31a354", linewidth=1.4, linestyle="--",
               label=f"-1.96SD {loa_lower:.2f} px")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")

    ax.set_xlabel("(GT + 予測) / 2 [px]")
    ax.set_ylabel("予測 - GT [px]")
    ax.set_title("Bland-Altman プロット（ρ）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.18)

    stats_text = f"n={len(diffs)}\nbias={bias:.2f}\nSD={sd:.2f}\nLoA=[{loa_lower:.2f}, {loa_upper:.2f}]"
    ax.text(1.02, 0.98, stats_text, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9})

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_plots(records: list[dict[str, Any]], output_dir: Path, image_size: int = 224) -> None:
    """全 metric の分布プロットを保存する"""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for metric_name, metric_meta in METRIC_META.items():
        plot_overall_histogram(
            records,
            metric_name,
            plots_dir / f"{metric_name}_overall_hist.png",
        )
        plot_group_distribution(
            records,
            metric_name,
            "vertebra",
            f"{metric_meta['title']}の椎体別分布",
            plots_dir / f"{metric_name}_by_vertebra_violin.png",
        )
        plot_group_distribution(
            records,
            metric_name,
            "line_name",
            f"{metric_meta['title']}の line 別分布",
            plots_dir / f"{metric_name}_by_line_violin.png",
        )

    plot_bland_altman_angle(records, plots_dir / "bland_altman_angle.png")
    plot_bland_altman_rho(records, image_size, plots_dir / "bland_altman_rho.png")


def main() -> None:
    """エントリーポイント"""
    args = parse_args()
    exp_dir = args.exp_dir.resolve()
    output_dir = resolve_output_dir(exp_dir, args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_line_records(exp_dir)
    slice_records = build_slice_records(records)
    percentile_rows = build_percentile_rows(records)

    write_csv(records, output_dir / "line_records.csv")
    write_csv(slice_records, output_dir / "slice_records.csv")
    write_csv(percentile_rows, output_dir / "percentiles.csv")
    make_plots(records, output_dir, image_size=args.image_size)
    write_summary_text(
        records,
        slice_records,
        percentile_rows,
        output_dir / "summary.txt",
    )

    print("[INFO] eval_error_viz 完了")
    print(f"[INFO] exp_dir: {exp_dir}")
    print(f"[INFO] output_dir: {output_dir}")
    print(f"[INFO] line_records: {len(records)}")
    print(f"[INFO] slice_records: {len(slice_records)}")

    for metric_name in args.metric:
        worst_slices = select_worst_slices(
            slice_records,
            metric_name=metric_name,
            aggregate=args.sort_aggregate,
            top_n=args.top_n,
        )
        copied = copy_worst_images(
            worst_slices,
            metric_name=metric_name,
            aggregate=args.sort_aggregate,
            output_dir=output_dir,
        )
        grid_path = build_worst_summary_grid(
            worst_slices,
            metric_name=metric_name,
            aggregate=args.sort_aggregate,
            output_dir=output_dir,
        )
        print(
            f"[INFO] worst [{metric_name}]: aggregate={args.sort_aggregate}"
            f" top_n={args.top_n} copied={len(copied)}"
        )
        if grid_path is not None:
            print(f"[INFO] worst_grid [{metric_name}]: {grid_path}")


if __name__ == "__main__":
    main()
