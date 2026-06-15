"""
fracture_labels/ を annotation_data/ の椎骨ボリューム座標系に変換するスクリプト。

処理フロー:
  1. fracture_labels/{sample}.nii.gz (元CT座標系)
  2. segmentations/ の重心から generate_pre.py と同じクロップ変換を適用
  3. annotation_data の回転 (auto_tilt: rotation_deg.npy / 手動: CT affine) を適用
  4. annotation_data/{sample}/{V}/fracture.nii.gz として保存

z方向のスライスラベルは:
  fracture = nib.load('fracture.nii.gz').get_fdata()
  slice_labels = (fracture.sum(axis=(0, 1)) > 0).astype(int)

使い方:
    uv run python data_preprocessing/volume_prep/add_fracture_labels.py
    uv run python data_preprocessing/volume_prep/add_fracture_labels.py --sample sample1
    uv run python data_preprocessing/volume_prep/add_fracture_labels.py --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.linalg import svd
from scipy.ndimage import label as nd_label
from scipy.spatial.transform import Rotation
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]

ANNOT_DIR = ROOT_DIR / "data" / "annotation_data"
FRACTURE_DIR = ROOT_DIR / "data" / "fracture_labels"
PREDATA_DIR = ROOT_DIR / "data" / "predata_simple"
SEG_DIR = ROOT_DIR / "data" / "segmentations"

# generate_pre.py と同じパラメータ
OUTPUT_RESOLUTION = 0.4   # mm/voxel
TEMP_Z_SIZE = 500
Z_MARGIN_VOXELS = 5

VERTEBRAE = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

# 椎体ごとの骨折情報（記載なしのサンプルは空リスト）
FRACTURE_VERTEBRAE: dict[str, list[str]] = {
    "sample1":    ["C2"],
    "sample2":    ["C1", "C2"],
    "sample3":    ["C2", "C3", "C4"],
    "sample4":    ["C2"],
    "sample5":    ["C2"],
    "sample6":    [],
    "sample7":    ["C2", "C7"],
    "sample9":    ["C2"],
    "sample10":   ["C6", "C7"],
    "sample12":   ["C7"],
    "sample13":   ["C2"],
    "sample15":   ["C2"],
    "sample15.2": ["C6"],
    "sample17":   ["C2"],
    "sample19":   ["C2"],
    "sample21":   ["C6"],
    "sample22":   ["C2"],
    "sample23":   ["C2"],
    "sample24":   ["C1"],
    "sample25":   ["C7"],
    "sample27":   ["C6", "C7"],
    "sample28":   ["C1", "C2"],
    "sample29":   ["C6", "C7"],
    "sample31":   ["C2"],
    "sample32":   ["C2", "C3", "C4"],
    "sample33":   ["C2", "C5", "C7"],
    "sample34":   ["C5", "C6"],
    "sample35":   ["C6"],
    "sample36":   ["C2", "C7"],
    "sample37":   ["C7"],
    "sample38":   ["C7"],
    "sample41":   ["C6"],
    "sample42":   ["C7"],
    "sample43":   ["C2"],
    "sample44":   ["C2"],
    "sample47":   ["C2", "C3", "C4", "C5"],
    "sample48":   ["C2", "C7"],
    "sample50":   ["C7"],
    "sample51":   ["C2"],
    "sample52":   ["C1"],
    "sample53":   ["C2"],
    "sample54":   ["C3", "C4"],
    "sample55":   ["C5", "C6"],
    "sample56":   ["C5", "C6"],
    "sample57":   ["C2"],
    "sample59":   ["C1", "C2"],
    "sample60":   ["C1", "C2", "C5", "C6"],
    "sample61":   ["C3", "C4"],
    "sample66":   ["C1", "C2"],
    "sample67":   ["C1"],
    "sample68":   ["C2"],
}


# ---------------------------------------------------------------------------
# クロップ変換（generate_pre.py の _resample_fixed_crop と同じ）
# ---------------------------------------------------------------------------

def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, n = nd_label(mask > 0)
    if n <= 1:
        return mask
    sizes = [(labeled == i).sum() for i in range(1, n + 1)]
    return (labeled == np.argmax(sizes) + 1).astype(mask.dtype)


def _centroid_mm(mask: np.ndarray, affine: np.ndarray) -> np.ndarray | None:
    """マスク重心を mm 座標で返す。"""
    idx = np.argwhere(mask > 0)
    if len(idx) < 10:
        return None
    homo = np.column_stack([idx, np.ones(len(idx))])
    return np.mean((homo @ affine.T)[:, :3], axis=0)


def _resample_crop(
    data: np.ndarray,
    affine: np.ndarray,
    center_mm: np.ndarray,
    output_shape: tuple[int, int, int],
    order: int = 0,
    cval: float = 0.0,
) -> np.ndarray:
    """generate_pre.py の _resample_fixed_crop と同一の変換式。"""
    out_center = np.array(output_shape) / 2.0
    T_center = np.eye(4); T_center[:3, 3] = -out_center
    S = np.eye(4); S[0, 0] = S[1, 1] = S[2, 2] = OUTPUT_RESOLUTION
    T_loc = np.eye(4); T_loc[:3, 3] = center_mm
    M = np.linalg.inv(affine) @ T_loc @ S @ T_center
    return ndimage.affine_transform(
        data, M[:3, :3], offset=M[:3, 3],
        output_shape=output_shape, order=order, cval=cval,
    )


def _derive_crop_params(
    seg_data: np.ndarray,
    seg_affine: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int]] | tuple[None, None]:
    """
    generate_pre.py の 2 パスロジックで adjusted_center と final_shape を再現する。
    """
    seg_data = _largest_component(seg_data)
    center = _centroid_mm(seg_data, seg_affine)
    if center is None:
        return None, None

    # 1 パス目: Z 範囲確認
    temp_mask = _resample_crop(seg_data, seg_affine, center, (256, 256, TEMP_Z_SIZE), order=0)
    z_idx = np.where(temp_mask > 0)[2]
    if len(z_idx) == 0:
        return None, None

    z_min = max(0, int(z_idx.min()) - Z_MARGIN_VOXELS)
    z_max = min(TEMP_Z_SIZE, int(z_idx.max()) + Z_MARGIN_VOXELS + 1)
    z_shift = (z_min + z_max) / 2.0 - TEMP_Z_SIZE / 2.0
    adjusted_center = center + np.array([0.0, 0.0, 1.0]) * z_shift * OUTPUT_RESOLUTION
    return adjusted_center, (256, 256, z_max - z_min)


# ---------------------------------------------------------------------------
# 回転行列取得
# ---------------------------------------------------------------------------

def _get_rotation_matrix(
    annot_vdir: Path,
) -> tuple[np.ndarray, np.ndarray, bool] | tuple[None, None, None]:
    """
    annotation_data 椎骨ディレクトリから回転行列 R_apply と出力 affine を取得する。

    戻り値:
        (R_apply, out_affine, is_manual)
        R_apply   : apply_rotation と同義の回転行列
        out_affine: fracture.nii.gz に使う affine
        is_manual : True = 手動クロップ（Z 中心補正が必要）
        (None, None, None): 情報が取れなかった場合
    """
    # --- auto_tilt 処理済み ---
    rot_path = annot_vdir / "rotation_deg.npy"
    ct_auto = annot_vdir / "ct.nii.gz"
    if rot_path.exists() and ct_auto.exists():
        rx, ry = np.load(rot_path).tolist()
        R_apply = Rotation.from_euler("XY", [rx, ry], degrees=True).as_matrix()
        out_affine = nib.load(ct_auto).affine
        return R_apply, out_affine, False

    # --- 手動クロップ（3D Slicer） ---
    ct_patterns = (
        "ct cropped.nii.gz",
        "ct_*.nii.gz",
        "ct_*.nii",
        "ct_*.nrrd",
    )
    ct_files = sorted({path for pattern in ct_patterns for path in annot_vdir.glob(pattern)})
    if ct_files:
        ct_path = ct_files[0]

        if ct_path.name.endswith(".nrrd"):
            try:
                ct_img = nib.load(ct_path)
            except Exception:
                return None, None, None
        else:
            ct_img = nib.load(ct_path)

        A = ct_img.affine[:3, :3]
        spacing = float(ct_img.affine[0, 0])   # X 軸は回転なし → spacing 代表値

        # SVD で純回転行列を抽出（数値誤差除去）
        U, _, Vt = svd(A / spacing)
        R_tilt = U @ Vt                         # 椎骨の傾きを表す回転
        R_apply = R_tilt.T                      # 傾きを補正する回転
        return R_apply, ct_img.affine, True

    return None, None, None


def _correct_z_center_for_manual(
    annot_vdir: Path,
    R_apply: np.ndarray,
    adjusted_center: np.ndarray,
    fracture_data: np.ndarray,
    seg_affine: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """
    手動クロップサンプル用: annotation_data マスクの Z 中心に合わせて
    adjusted_center と final_shape を補正する。

    3D Slicer のクロップ中心は generate_pre.py の重心基準と異なるため、
    annotation_data マスクを逆回転して Z 重心を求め、ずれ分を補正する。
    """
    # annotation_data マスクを読み込み
    mask_files = sorted(annot_vdir.glob("mask*.nii.gz"))
    ct_files = sorted(annot_vdir.glob("ct_*.nii.gz"))
    if not mask_files or not ct_files:
        # マスクが取れない場合はそのまま返す
        ct = nib.load(ct_files[0]) if ct_files else None
        z_size = ct.shape[2] if ct else fracture_data.shape[2]
        return adjusted_center, (256, 256, z_size)

    ct_annot = nib.load(ct_files[0])
    mask_annot = nib.load(mask_files[0]).get_fdata()
    annot_z_size = ct_annot.shape[2]

    # annotation_data マスクを逆回転（R_tilt = R_apply.T）
    R_tilt = R_apply.T
    center = np.array(mask_annot.shape, dtype=float) / 2.0
    offset = center - R_tilt.T @ center
    mask_unrot = ndimage.affine_transform(mask_annot, R_tilt.T, offset=offset, order=0)

    # annotation_data の Z 重心（逆回転後）
    areas_annot = mask_unrot.sum(axis=(0, 1))
    if areas_annot.sum() == 0:
        return adjusted_center, (256, 256, annot_z_size)
    z_idx = np.arange(len(areas_annot), dtype=float)
    z_center_annot = float((areas_annot * z_idx).sum() / areas_annot.sum())

    # predata_simple の Z 重心（segmentation 重心基準クロップのマスク）
    from scipy.ndimage import label as nd_label2
    seg_files = list((annot_vdir.parent.parent / "segmentations" /
                      annot_vdir.parent.name / f"vertebrae_{annot_vdir.name}.nii.gz").parent.glob(
                      f"vertebrae_{annot_vdir.name}.nii.gz"))
    # 既に外側で seg_affine を渡してあるので再計算で代用
    dummy_shape = (256, 256, annot_z_size)
    temp = _resample_crop(
        nib.load(
            SEG_DIR / annot_vdir.parent.name / f"vertebrae_{annot_vdir.name}.nii.gz"
        ).get_fdata(),
        seg_affine, adjusted_center, dummy_shape, order=0,
    )
    areas_pre = temp.sum(axis=(0, 1))
    if areas_pre.sum() == 0:
        return adjusted_center, (256, 256, annot_z_size)
    z_center_pre = float((areas_pre * np.arange(dummy_shape[2], dtype=float)).sum() / areas_pre.sum())

    # Z ずれを adjusted_center に反映（0.4mm/voxel）
    dz_mm = (z_center_annot - z_center_pre) * OUTPUT_RESOLUTION
    adjusted_center_corrected = adjusted_center + np.array([0.0, 0.0, 1.0]) * dz_mm

    return adjusted_center_corrected, (256, 256, annot_z_size)


# ---------------------------------------------------------------------------
# 回転適用（auto_tilt.py の apply_rotation と同じ操作）
# ---------------------------------------------------------------------------

def _apply_rotation(data: np.ndarray, R_apply: np.ndarray, order: int = 0) -> np.ndarray:
    """data に R_apply 回転を適用する（auto_tilt.apply_rotation と等価）。"""
    center = np.array(data.shape, dtype=float) / 2.0
    offset = center - R_apply.T @ center
    return ndimage.affine_transform(
        data, R_apply.T, offset=offset, order=order, mode="constant", cval=0,
    )


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def process_vertebra(
    sample_id: str,
    vertebra: str,
    fracture_data: np.ndarray,
    seg_affine: np.ndarray,
    is_fractured: bool,
    overwrite: bool = False,
) -> str:
    """
    1 椎骨分の fracture.nii.gz を生成する。

    is_fractured が False の場合は全ゼロマスクを保存する。
    is_fractured が True の場合は骨折ラベルを椎体セグメンテーションで
    マスクした上で保存する（隣接椎体へのはみ出しを防ぐ）。

    戻り値: "done" / "skip" / "no_annot" / "no_seg" / "empty" / "no_rot"
    """
    annot_vdir = ANNOT_DIR / sample_id / vertebra
    if not annot_vdir.exists():
        return "no_annot"

    out_path = annot_vdir / "fracture.nii.gz"
    if not overwrite and out_path.exists():
        return "skip"

    # segmentation 読み込み
    seg_path = SEG_DIR / sample_id / f"vertebrae_{vertebra}.nii.gz"
    if not seg_path.exists():
        return "no_seg"
    seg_data = nib.load(seg_path).get_fdata()

    # クロップパラメータ再現
    adjusted_center, final_shape = _derive_crop_params(seg_data, seg_affine)
    if adjusted_center is None:
        return "empty"

    # 回転取得（affine 取得のため骨折有無に関わらず必要）
    R_apply, out_affine, is_manual = _get_rotation_matrix(annot_vdir)
    if R_apply is None:
        return "no_rot"

    # 骨折なし椎体: 全ゼロマスクを保存して終了
    if not is_fractured:
        zero_mask = np.zeros(final_shape, dtype=np.uint8)
        nib.save(nib.Nifti1Image(zero_mask, out_affine), out_path)
        return "done"

    # fracture_labels を predata_simple 座標系にクロップ
    fracture_crop = _resample_crop(
        fracture_data, seg_affine, adjusted_center, final_shape, order=0
    )
    # 椎体セグメンテーションも同座標系にクロップ（はみ出し防止マスク）
    seg_crop = _resample_crop(
        seg_data, seg_affine, adjusted_center, final_shape, order=0
    )

    # 手動クロップサンプル: annotation_data の Z 中心に合わせて補正
    if is_manual:
        adjusted_center, final_shape = _correct_z_center_for_manual(
            annot_vdir, R_apply, adjusted_center, fracture_data, seg_affine
        )
        fracture_crop = _resample_crop(
            fracture_data, seg_affine, adjusted_center, final_shape, order=0
        )
        seg_crop = _resample_crop(
            seg_data, seg_affine, adjusted_center, final_shape, order=0
        )

    # 回転適用
    fracture_rotated = _apply_rotation(fracture_crop, R_apply, order=0)
    seg_rotated = _apply_rotation(seg_crop, R_apply, order=0)

    # 骨折ラベルを椎体セグメンテーションで AND マスク
    fracture_masked = (fracture_rotated > 0) & (seg_rotated > 0)

    nib.save(
        nib.Nifti1Image(fracture_masked.astype(np.uint8), out_affine),
        out_path,
    )
    return "done"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="fracture_labels/ を annotation_data/ 座標系に変換して fracture.nii.gz を生成する"
    )
    parser.add_argument("--sample", default=None, help="特定サンプルのみ処理（例: sample1）")
    parser.add_argument("--overwrite", action="store_true", help="既存 fracture.nii.gz を上書き")
    args = parser.parse_args()

    if args.sample:
        sample_dirs = [ANNOT_DIR / args.sample]
    else:
        sample_dirs = sorted(
            d for d in ANNOT_DIR.iterdir()
            if d.is_dir() and d.name.startswith("sample")
        )

    counts: dict[str, int] = {
        "done": 0, "skip": 0, "no_annot": 0,
        "no_seg": 0, "empty": 0, "no_rot": 0,
    }

    for sample_dir in tqdm(sample_dirs, desc="サンプル処理中"):
        sample_id = sample_dir.name

        fracture_path = FRACTURE_DIR / f"{sample_id}.nii.gz"
        if not fracture_path.exists():
            print(f"  {sample_id}: fracture_labels なし → スキップ")
            counts["no_seg"] += len(VERTEBRAE)
            continue

        # seg affine（fracture_labels と同じ座標系）
        seg_affine = None
        for v in VERTEBRAE:
            seg_path = SEG_DIR / sample_id / f"vertebrae_{v}.nii.gz"
            if seg_path.exists():
                seg_affine = nib.load(seg_path).affine
                break
        if seg_affine is None:
            print(f"  {sample_id}: segmentation なし → スキップ")
            counts["no_seg"] += len(VERTEBRAE)
            continue

        fracture_data = nib.load(fracture_path).get_fdata()
        # テーブルに記載のないサンプルは全椎体を非骨折として扱う
        fractured_vertebrae = set(FRACTURE_VERTEBRAE.get(sample_id, []))

        for vertebra in VERTEBRAE:
            is_fractured = vertebra in fractured_vertebrae
            result = process_vertebra(
                sample_id, vertebra, fracture_data, seg_affine,
                is_fractured=is_fractured,
                overwrite=args.overwrite,
            )
            counts[result] += 1
            if result == "done":
                print(f"  {sample_id}/{vertebra}: 生成完了")

    print("\n===== 結果 =====")
    print(f"  生成: {counts['done']}")
    print(f"  スキップ（既存）: {counts['skip']}")
    print(f"  annotation_data なし: {counts['no_annot']}")
    print(f"  segmentation なし: {counts['no_seg']}")
    print(f"  マスク空: {counts['empty']}")
    print(f"  回転情報なし: {counts['no_rot']}")


if __name__ == "__main__":
    main()
