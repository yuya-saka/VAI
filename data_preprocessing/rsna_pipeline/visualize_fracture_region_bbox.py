"""骨折bboxを224×224プレーン空間に投影し、region_4classと重ねて可視化する。

各(study, level)のプレーンごとにct.npy中央チャンネル上に
region_4classカラーオーバーレイ + bbox矩形を描画し、
各領域のshare_r値を表示する。

使い方:
    uv run python data_preprocessing/rsna_pipeline/visualize_fracture_region_bbox.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa: F401
import numpy as np
import pandas as pd

# --- パス定義 ---
DATA_DIR = Path("data/rsna_data")
FRACTURE_DATASET_DIR = DATA_DIR / "fracture_dataset"
META_DIR = DATA_DIR / "processing_metadata"
BBOX_CSV = DATA_DIR / "train_bounding_boxes.csv"
TRAIN_CSV = DATA_DIR / "train.csv"
OUTPUT_DIR = Path("Unet/outputs/fracture_region_bbox_vis")

SAMPLING_PIXEL_SPACING_MM = 0.4
OUTPUT_SIZE = 224
HALF_PX = (OUTPUT_SIZE - 1) / 2.0  # 111.5

REGION_COLORS_BGR = {
    1: (255, 80, 80),    # 青系
    2: (80, 255, 80),    # 緑系
    3: (80, 80, 255),    # 赤系
    4: (255, 255, 80),   # シアン系
}
REGION_COLORS_PLOT = {
    1: "dodgerblue",
    2: "limegreen",
    3: "tomato",
    4: "gold",
}


def dicom_bbox_to_plane_px(
    x: float,
    y: float,
    w: float,
    h: float,
    image_position_lps: list[float],
    row_direction_lps: list[float],
    column_direction_lps: list[float],
    pixel_spacing_row_mm: float,
    pixel_spacing_col_mm: float,
    plane_center_lps: list[float],
    plane_row_basis_lps: list[float],
    plane_column_basis_lps: list[float],
) -> tuple[float, float, float, float]:
    """DICOMのbbox (x,y,w,h) を224×224プレーン座標に変換する。

    戻り値: (row_min, col_min, row_max, col_max) in 224px空間
    """
    image_pos = np.array(image_position_lps, dtype=np.float64)
    row_dir = np.array(row_direction_lps, dtype=np.float64)
    col_dir = np.array(column_direction_lps, dtype=np.float64)
    center = np.array(plane_center_lps, dtype=np.float64)
    row_basis = np.array(plane_row_basis_lps, dtype=np.float64)
    col_basis = np.array(plane_column_basis_lps, dtype=np.float64)

    # bboxの4角 (col_px_dicom, row_px_dicom)
    # train_bounding_boxes: x=列方向, y=行方向
    corners_dicom = [
        (x, y),
        (x + w, y),
        (x + w, y + h),
        (x, y + h),
    ]

    rows_224, cols_224 = [], []
    for cx, cy in corners_dicom:
        # DICOM voxel → LPS
        # voxel_to_patient: lps = origin + col_idx * col_spacing * row_dir + row_idx * row_spacing * col_dir
        lps = (
            image_pos
            + cx * pixel_spacing_col_mm * row_dir
            + cy * pixel_spacing_row_mm * col_dir
        )
        # LPS → 224px (_patient_coordinates の逆変換)
        # lps = center + col_offset * row_basis + row_offset * col_basis
        # row_offset = dot(lps - center, col_basis)
        # col_offset = dot(lps - center, row_basis)
        delta = lps - center
        row_px = float(np.dot(delta, col_basis)) / SAMPLING_PIXEL_SPACING_MM + HALF_PX
        col_px = float(np.dot(delta, row_basis)) / SAMPLING_PIXEL_SPACING_MM + HALF_PX
        rows_224.append(row_px)
        cols_224.append(col_px)

    return min(rows_224), min(cols_224), max(rows_224), max(cols_224)


def compute_share_r(
    region_mask: np.ndarray,
    vertebra_mask: np.ndarray,
    bbox_mask: np.ndarray,
) -> dict[int, float]:
    """bbox∩vertebra内の各領域の占有率を計算する（Gaussian重みなし版）。"""
    # bbox∩vertebra の有効領域
    valid = (bbox_mask > 0) & (vertebra_mask > 0)
    total_valid = int(valid.sum())
    if total_valid == 0:
        return {r: 0.0 for r in range(1, 5)}

    shares = {}
    for r in range(1, 5):
        overlap = int(((region_mask == r) & valid).sum())
        shares[r] = overlap / total_valid
    return shares


def make_region_overlay(
    ct_plane: np.ndarray,
    region_4class: np.ndarray,
    vertebra_mask: np.ndarray,
    alpha: float = 0.35,
) -> np.ndarray:
    """ct_plane (224,224 uint8) にregion_4classをカラーオーバーレイした BGR画像を返す。"""
    bgr = cv2.cvtColor(ct_plane, cv2.COLOR_GRAY2BGR)
    overlay = bgr.copy()
    for r, color in REGION_COLORS_BGR.items():
        mask = (region_4class == r) & (vertebra_mask > 0)
        overlay[mask] = color
    return cv2.addWeighted(bgr, 1 - alpha, overlay, alpha, 0)


def visualize_study_level(
    study_id: str,
    level: str,
    bbox_df: pd.DataFrame,
    meta: dict,
    output_dir: Path,
    n_planes_to_show: int = 5,
) -> None:
    """1つの(study, level)について複数プレーンの可視化を生成する。"""
    level_dir = FRACTURE_DATASET_DIR / study_id / level
    ct = np.load(level_dir / "ct.npy")               # (15, 5, 224, 224)
    vmask = np.load(level_dir / "vertebra_mask.npy")  # (15, 224, 224)
    region = np.load(level_dir / "region_4class.npy") # (15, 224, 224)

    slices_meta = meta["dicom_geometry"]["slices"]
    row_dir = meta["dicom_geometry"]["row_direction_lps"]
    col_dir = meta["dicom_geometry"]["column_direction_lps"]
    ps_row, ps_col = meta["dicom_geometry"]["pixel_spacing_row_column_mm"]

    # slice_number（ファイルstem数値優先、instance_number補助）→ slices_metaインデックス
    # classifier_planes.py の _slice_index_by_number と同じロジック
    slice_num_to_idx: dict[int, int] = {}
    for idx, s in enumerate(slices_meta):
        stem = Path(s["source_file"]).stem
        try:
            slice_num_to_idx[int(stem)] = idx
        except ValueError:
            pass
        if s["instance_number"] is not None:
            slice_num_to_idx.setdefault(s["instance_number"], idx)

    def get_image_pos(sn: int) -> list[float] | None:
        idx = slice_num_to_idx.get(int(sn))
        return slices_meta[idx]["image_position_lps_mm"] if idx is not None else None

    def get_slice_pos(sn: int) -> float | None:
        idx = slice_num_to_idx.get(int(sn))
        return slices_meta[idx]["slice_position_mm"] if idx is not None else None

    planes = meta["vertebrae"][level]["classifier_planes"]["planes"]
    full_range = meta["vertebrae"][level]["classifier_planes"]["full_range_mm"]
    z_min, z_max = min(full_range), max(full_range)

    # このlevelに属するbbox
    level_bboxes = bbox_df[
        bbox_df["slice_number"].apply(
            lambda s: (sp := get_slice_pos(int(s))) is not None and z_min <= sp <= z_max
        )
    ]

    if level_bboxes.empty:
        return

    # 各bboxスライスを最近傍プレーンに割り当て
    plane_positions = np.array([pl["position_mm"] for pl in planes])

    # プレーンインデックスごとのbboxリスト
    plane_bboxes: dict[int, list[dict]] = {i: [] for i in range(len(planes))}
    for _, row in level_bboxes.iterrows():
        sn = int(row["slice_number"])
        slice_pos = get_slice_pos(sn)
        if slice_pos is None:
            continue
        nearest_plane_idx = int(np.argmin(np.abs(plane_positions - slice_pos)))
        img_pos = get_image_pos(sn)
        if img_pos is None:
            continue
        plane_bboxes[nearest_plane_idx].append(
            {
                "x": row["x"],
                "y": row["y"],
                "w": row["width"],
                "h": row["height"],
                "image_position_lps": img_pos,
            }
        )

    # bboxがあるプレーンを選んで可視化
    planes_with_bbox = [i for i, blist in plane_bboxes.items() if blist]
    if not planes_with_bbox:
        return

    # 等間隔にn_planes_to_show個選ぶ
    step = max(1, len(planes_with_bbox) // n_planes_to_show)
    selected = planes_with_bbox[::step][:n_planes_to_show]

    fig, axes = plt.subplots(
        1, len(selected), figsize=(4 * len(selected), 5)
    )
    if len(selected) == 1:
        axes = [axes]

    fig.suptitle(f"{study_id.split('.')[-1]} / {level}", fontsize=11)

    for ax, plane_idx in zip(axes, selected):
        plane = planes[plane_idx]
        ct_plane = ct[plane_idx, 2]  # 中央チャンネル
        vis = make_region_overlay(ct_plane, region[plane_idx], vmask[plane_idx])
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

        ax.imshow(vis_rgb)
        ax.set_title(f"plane {plane_idx}", fontsize=8)
        ax.axis("off")

        all_shares: dict[int, float] = {r: 0.0 for r in range(1, 5)}
        bbox_mask = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.uint8)

        for b in plane_bboxes[plane_idx]:
            r_min, c_min, r_max, c_max = dicom_bbox_to_plane_px(
                b["x"], b["y"], b["w"], b["h"],
                b["image_position_lps"],
                row_dir, col_dir, ps_row, ps_col,
                plane["center_lps_mm"],
                plane["row_basis_lps"],
                plane["column_basis_lps"],
            )
            r0 = max(0, int(np.floor(r_min)))
            c0 = max(0, int(np.floor(c_min)))
            r1 = min(OUTPUT_SIZE, int(np.ceil(r_max)))
            c1 = min(OUTPUT_SIZE, int(np.ceil(c_max)))
            bbox_mask[r0:r1, c0:c1] = 1

            # bbox矩形を描画
            rect = plt.Rectangle(
                (c0, r0), c1 - c0, r1 - r0,
                linewidth=1.5, edgecolor="white", facecolor="none",
            )
            ax.add_patch(rect)

        shares = compute_share_r(region[plane_idx], vmask[plane_idx], bbox_mask)
        for r, s in shares.items():
            all_shares[r] = max(all_shares[r], s)

        # share_r をテキスト表示
        share_text = " ".join(
            f"R{r}:{s:.2f}" for r, s in all_shares.items() if s > 0
        )
        ax.set_xlabel(share_text, fontsize=7)

    plt.tight_layout()
    out_path = output_dir / f"{study_id.split('.')[-1]}_{level}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bbox_df = pd.read_csv(BBOX_CSV)
    train_df = pd.read_csv(TRAIN_CSV)

    level_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    studies_with_bbox = set(bbox_df["StudyInstanceUID"].unique())

    # 骨折あり・bboxあり・region_4class存在の(study, level)を収集
    targets: list[tuple[str, str]] = []
    for _, row in train_df.iterrows():
        sid = row["StudyInstanceUID"]
        if sid not in studies_with_bbox:
            continue
        for level in level_cols:
            if row[level] != 1:
                continue
            level_dir = FRACTURE_DATASET_DIR / sid / level
            if not (level_dir / "region_4class.npy").exists():
                continue
            targets.append((sid, level))

    print(f"対象 (study, level): {len(targets)} 件")

    # seed固定でランダムサンプリング
    rng = np.random.default_rng(42)
    n_samples = min(30, len(targets))
    indices = rng.choice(len(targets), size=n_samples, replace=False)
    sampled = [targets[i] for i in sorted(indices)]

    for study_id, level in sampled:
        meta_path = META_DIR / f"{study_id}.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        study_bboxes = bbox_df[bbox_df["StudyInstanceUID"] == study_id]
        try:
            visualize_study_level(study_id, level, study_bboxes, meta, OUTPUT_DIR)
            print(f"  保存: {study_id.split('.')[-1]} / {level}")
        except Exception as e:
            print(f"  エラー: {study_id.split('.')[-1]} / {level}: {e}")

    print(f"\n出力先: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
