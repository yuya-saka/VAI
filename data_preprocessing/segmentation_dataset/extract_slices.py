"""
predata_simple/ → 代表スライス PNG 抽出スクリプト。

アプローチ:
  1. auto_tilt と同じアルゴリズムで最適傾き (rx, ry) を探索
  2. 3D ボリュームを回転させず、傾いた平面を直接サンプリング（斜断面サンプリング）
     → 回転補間によるぼけが出ない

使い方:
    uv run python data_preprocessing/segmentation_dataset/extract_slices.py
    uv run python data_preprocessing/segmentation_dataset/extract_slices.py --sample sample10 --n_slices 7
    uv run python data_preprocessing/segmentation_dataset/extract_slices.py --no_tilt  # 傾き補正なし
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import nibabel.processing
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial.transform import Rotation
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]

WINDOW_LEVEL = 400
WINDOW_WIDTH = 2000
TARGET_SIZE = 224
DEFAULT_N_SLICES = 7
MIN_MASK_AREA = 50       # 有効スライスとみなす最小マスク面積（ピクセル数）
COARSE_RANGE = 30.0
COARSE_STEP = 5.0
FINE_STEP = 1.0


def apply_window(
    ct_slice: np.ndarray,
    level: int = WINDOW_LEVEL,
    width: int = WINDOW_WIDTH,
) -> np.ndarray:
    """CT ウィンドウ処理を適用して [0, 1] に正規化する。"""
    lo = level - width / 2
    hi = level + width / 2
    return (np.clip(ct_slice, lo, hi) - lo) / (hi - lo)


def _max_area_axis2(mask: np.ndarray, rx: float, ry: float) -> float:
    """rx, ry 回転後の軸2方向スライス最大面積を返す（探索用）。"""
    center = np.array(mask.shape, dtype=float) / 2.0
    R = Rotation.from_euler("XY", [rx, ry], degrees=True).as_matrix()
    offset = center - R.T @ center
    rotated = ndimage.affine_transform(
        mask, R.T, offset=offset, order=0, mode="constant", cval=0
    )
    return float(rotated.sum(axis=(0, 1)).max())


def find_best_tilt(mask: np.ndarray) -> tuple[float, float]:
    """coarse→fine 2段階探索で最適 (rx, ry) を返す。"""
    # 速度のため 0.5 倍ダウンサンプルで coarse search
    small = ndimage.zoom(mask.astype(np.float32), 0.5, order=0)
    angles = np.arange(-COARSE_RANGE, COARSE_RANGE + COARSE_STEP / 2, COARSE_STEP)

    best_area = -1.0
    best_rx = best_ry = 0.0
    for rx in angles:
        for ry in angles:
            area = _max_area_axis2(small, rx, ry)
            if area > best_area:
                best_area, best_rx, best_ry = area, rx, ry

    # best 付近を元解像度で fine search
    for rx in np.arange(best_rx - COARSE_STEP, best_rx + COARSE_STEP + FINE_STEP / 2, FINE_STEP):
        for ry in np.arange(best_ry - COARSE_STEP, best_ry + COARSE_STEP + FINE_STEP / 2, FINE_STEP):
            area = _max_area_axis2(mask.astype(np.float32), rx, ry)
            if area > best_area:
                best_area, best_rx, best_ry = area, rx, ry

    return float(best_rx), float(best_ry)


PEAK_MARGIN = 5  # 最大面積スライス前後の取得範囲（スライス数）


def get_valid_k_range(mask: np.ndarray, rx: float, ry: float) -> list[int]:
    """
    傾き (rx, ry) 方向で最もマスク面積が大きいスライスの
    前後 PEAK_MARGIN スライス以内の範囲を返す。
    """
    center = np.array(mask.shape, dtype=float) / 2.0
    R = Rotation.from_euler("XY", [rx, ry], degrees=True).as_matrix()
    offset = center - R.T @ center
    rotated = ndimage.affine_transform(
        mask.astype(np.float32), R.T, offset=offset, order=0, mode="constant", cval=0
    )
    areas = rotated.sum(axis=(0, 1))
    if areas.max() < MIN_MASK_AREA:
        return []
    peak_k = int(areas.argmax())
    k_min = max(0, peak_k - PEAK_MARGIN)
    k_max = min(len(areas) - 1, peak_k + PEAK_MARGIN)
    return list(range(k_min, k_max + 1))


def extract_oblique_slice(
    volume: np.ndarray,
    rx: float,
    ry: float,
    k: int,
    size: int = TARGET_SIZE,
    order: int = 1,
) -> np.ndarray:
    """
    傾き (rx, ry) の回転後における z=k の断面を、
    元ボリュームからの直接サンプリングで抽出する（ボリュームを回転しない）。

    3D ボリュームを回転させて z スライスを取るのと数学的に等価だが、
    一度のサンプリングで済むためぼけが少ない。
    """
    center = np.array(volume.shape, dtype=float) / 2.0
    R = Rotation.from_euler("XY", [rx, ry], degrees=True).as_matrix()

    # 出力画像上の (i, j) は回転後空間の (i_c + di, j_c + dj, k) に対応
    # 元空間座標: src = R^T @ (dst - center) + center
    half = size // 2
    i_vals = np.arange(center[0] - half, center[0] + half, dtype=np.float64)
    j_vals = np.arange(center[1] - half, center[1] + half, dtype=np.float64)
    ii, jj = np.meshgrid(i_vals, j_vals, indexing="ij")
    kk = np.full_like(ii, float(k))

    pts = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=0)  # (3, N)
    src_coords = R.T @ (pts - center[:, None]) + center[:, None]   # (3, N)

    sampled = ndimage.map_coordinates(
        volume, src_coords, order=order, mode="constant", cval=0
    )
    # auto_tilt の rotated[:,:,z].T と同じ軸順に合わせる
    return sampled.reshape(size, size).T


def select_representative_k(valid_k: list[int], n: int) -> list[int]:
    """有効スライス位置から n 枚を等間隔で選ぶ。"""
    if not valid_k:
        return []
    k_min, k_max = valid_k[0], valid_k[-1]
    count = min(n, k_max - k_min + 1)
    if count == 1:
        return [(k_min + k_max) // 2]
    indices = np.linspace(k_min, k_max, count)
    return sorted({int(round(i)) for i in indices})


def extract_vertebra_slices(
    vertebra_dir: Path,
    output_dir: Path,
    n_slices: int = DEFAULT_N_SLICES,
    overwrite: bool = False,
    apply_tilt: bool = True,
) -> int:
    """
    1 椎体から代表スライスを抽出して output_dir に PNG 保存する。

    戻り値:
        -1: lines.json 済みのためスキップ
         0: NIfTI が見つからない、またはマスクが空
        N>0: 抽出したスライス数
    """
    if not overwrite and (output_dir / "lines.json").exists():
        return -1

    ct_path = vertebra_dir / "ct.nii.gz"
    mask_path = vertebra_dir / "mask.nii.gz"
    if not ct_path.exists() or not mask_path.exists():
        return 0

    ct_nii = nib.load(ct_path)
    mask_nii = nib.load(mask_path)

    if not np.allclose(ct_nii.affine, mask_nii.affine) or ct_nii.shape != mask_nii.shape:
        mask_nii = nibabel.processing.resample_from_to(mask_nii, ct_nii, order=0)

    ct_data = ct_nii.get_fdata()
    mask_data = mask_nii.get_fdata()

    if apply_tilt:
        rx, ry = find_best_tilt(mask_data)
    else:
        rx, ry = 0.0, 0.0

    valid_k = get_valid_k_range(mask_data, rx, ry)
    if not valid_k:
        return 0

    selected_k = select_representative_k(valid_k, n_slices)
    if not selected_k:
        return 0

    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)

    # 回転パラメータを保存（逆変換や記録用）
    np.save(output_dir / "tilt_deg.npy", np.array([rx, ry]))

    for k in selected_k:
        ct_slice = extract_oblique_slice(ct_data, rx, ry, k, order=1)
        mask_slice = extract_oblique_slice(mask_data, rx, ry, k, order=0)

        ct_windowed = apply_window(ct_slice)

        # 上下反転（convert_to_png.py と同じ）
        ct_final = np.flipud(ct_windowed)
        mask_final = np.flipud(mask_slice)

        Image.fromarray((ct_final * 255).astype(np.uint8)).save(
            output_dir / "images" / f"slice_{k:03d}.png"
        )
        Image.fromarray((mask_final > 0).astype(np.uint8) * 255).save(
            output_dir / "masks" / f"slice_{k:03d}.png"
        )

    return len(selected_k)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="predata_simple/ から傾き補正済み代表スライス PNG を dataset/ に抽出する"
    )
    parser.add_argument(
        "--input_dir",
        default=str(ROOT_DIR / "data" / "predata_simple"),
        help="入力ディレクトリ（デフォルト: predata_simple）",
    )
    parser.add_argument(
        "--output_dir",
        default=str(ROOT_DIR / "data" / "dataset"),
        help="出力ディレクトリ（デフォルト: dataset）",
    )
    parser.add_argument(
        "--n_slices",
        type=int,
        default=DEFAULT_N_SLICES,
        help=f"抽出するスライス数（デフォルト: {DEFAULT_N_SLICES}）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="lines.json 済みの椎体も上書きする",
    )
    parser.add_argument(
        "--no_tilt",
        action="store_true",
        help="傾き補正をスキップする",
    )
    parser.add_argument(
        "--sample",
        default=None,
        help="特定のサンプルのみ処理（例: sample10）",
    )
    parser.add_argument(
        "--vertebra",
        default=None,
        help="特定の椎体のみ処理（例: C3）",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: {input_dir} が見つかりません", file=sys.stderr)
        sys.exit(1)

    sample_dirs = sorted(
        d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("sample")
    )
    if args.sample:
        sample_dirs = [d for d in sample_dirs if d.name == args.sample]

    counts = {"extracted": 0, "skipped": 0, "failed": 0}

    for sample_dir in tqdm(sample_dirs, desc="サンプル処理中"):
        vertebra_dirs = sorted(v for v in sample_dir.iterdir() if v.is_dir())
        if args.vertebra:
            vertebra_dirs = [v for v in vertebra_dirs if v.name == args.vertebra]

        for vertebra_dir in vertebra_dirs:
            out_dir = output_dir / sample_dir.name / vertebra_dir.name
            result = extract_vertebra_slices(
                vertebra_dir=vertebra_dir,
                output_dir=out_dir,
                n_slices=args.n_slices,
                overwrite=args.overwrite,
                apply_tilt=not args.no_tilt,
            )

            if result == -1:
                counts["skipped"] += 1
            elif result > 0:
                counts["extracted"] += 1
                print(f"  {sample_dir.name}/{vertebra_dir.name}: {result} スライス (tilt={not args.no_tilt})")
            else:
                counts["failed"] += 1

    print(
        f"\n完了: 抽出={counts['extracted']} / "
        f"スキップ(既存)={counts['skipped']} / "
        f"失敗={counts['failed']}"
    )


if __name__ == "__main__":
    main()
