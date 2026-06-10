"""
椎体の最適なaxial断面傾きを自動検出し、補正済みボリュームを保存する。

アプローチ:
  axial断面にした時、椎体マスクが途切れず最大面積になる向きを探索する。
  傾きがある場合は斜めカットになり面積が下がるので、最大面積＝最適傾きとなる。

入力: predata_simple/sampleX/CY/{ct.nii.gz, mask.nii.gz}
出力: annotation_data/sampleX/CY/{ct.nii.gz, mask.nii.gz, rotation_deg.npy}
      ※ .mrk.json が既存の椎体はスキップ（アノテーション済みとみなす）
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation


def max_axial_mask_area(mask: np.ndarray, rx: float, ry: float) -> float:
    """rx (x軸周り)・ry (y軸周り) 度回転後の最大axial断面マスク面積"""
    center = np.array(mask.shape, dtype=float) / 2.0
    R = Rotation.from_euler("XY", [rx, ry], degrees=True).as_matrix()
    # affine_transform: output[x] = input[R^T @ (x - center) + center]
    offset = center - R.T @ center
    rotated = ndimage.affine_transform(
        mask, R.T, offset=offset, order=0, mode="constant", cval=0
    )
    # z軸 (dim=2) の各スライス面積の最大値
    return float(rotated.sum(axis=(0, 1)).max())


def find_best_tilt(
    mask: np.ndarray,
    coarse_range: float = 30.0,
    coarse_step: float = 5.0,
    fine_step: float = 1.0,
) -> tuple[float, float]:
    """coarse→fineの2段階で最適 (rx, ry) を探索"""
    # 速度のため0.5倍にダウンサンプルしてcoarse search
    small = ndimage.zoom(mask.astype(np.float32), 0.5, order=0)

    best_area = -1.0
    best_rx = best_ry = 0.0
    angles = np.arange(-coarse_range, coarse_range + coarse_step / 2, coarse_step)

    total = len(angles) ** 2
    done = 0
    for rx in angles:
        for ry in angles:
            area = max_axial_mask_area(small, rx, ry)
            if area > best_area:
                best_area = area
                best_rx, best_ry = rx, ry
            done += 1
            if done % 20 == 0:
                print(f"  coarse: {done}/{total}", end="\r", flush=True)
    print()

    # best付近をfine search (元解像度)
    fine_x = np.arange(best_rx - coarse_step, best_rx + coarse_step + fine_step / 2, fine_step)
    fine_y = np.arange(best_ry - coarse_step, best_ry + coarse_step + fine_step / 2, fine_step)
    for rx in fine_x:
        for ry in fine_y:
            area = max_axial_mask_area(mask.astype(np.float32), rx, ry)
            if area > best_area:
                best_area = area
                best_rx, best_ry = rx, ry

    return float(best_rx), float(best_ry)


def apply_rotation(
    data: np.ndarray, rx: float, ry: float, order: int = 1
) -> np.ndarray:
    """3DボリュームにXY回転を適用して返す"""
    center = np.array(data.shape, dtype=float) / 2.0
    R = Rotation.from_euler("XY", [rx, ry], degrees=True).as_matrix()
    offset = center - R.T @ center
    return ndimage.affine_transform(
        data, R.T, offset=offset, order=order, mode="constant", cval=0
    )


def is_annotated(output_dir: Path) -> bool:
    """出力ディレクトリに .mrk.json があればアノテーション済みと判断"""
    return output_dir.exists() and any(output_dir.glob("*.mrk.json"))


def process_vertebra(
    sample_dir: Path,
    output_base: Path,
    coarse_range: float = 30.0,
    dry_run: bool = False,
) -> None:
    """1椎体分を処理"""
    ct_path = sample_dir / "ct.nii.gz"
    mask_path = sample_dir / "mask.nii.gz"

    if not ct_path.exists() or not mask_path.exists():
        print(f"  スキップ (ファイルなし): {sample_dir}")
        return

    out_dir = output_base / sample_dir.parent.name / sample_dir.name
    if is_annotated(out_dir):
        print(f"  スキップ (アノテーション済み): {sample_dir.parent.name}/{sample_dir.name}")
        return

    ct_img = nib.load(ct_path)
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()

    print(f"  傾き探索中 (shape={mask_data.shape})...")
    rx, ry = find_best_tilt(mask_data, coarse_range=coarse_range)
    print(f"  最適tilt: rx={rx:.1f}°, ry={ry:.1f}°")

    if dry_run:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # CT: 補間あり (order=1)
    ct_data = ct_img.get_fdata()
    ct_rotated = apply_rotation(ct_data, rx, ry, order=1)
    nib.save(nib.Nifti1Image(ct_rotated, ct_img.affine, ct_img.header), out_dir / "ct.nii.gz")

    # マスク: nearest neighbor (order=0)
    mask_rotated = apply_rotation(mask_data, rx, ry, order=0)
    nib.save(nib.Nifti1Image(mask_rotated, mask_img.affine, mask_img.header), out_dir / "mask.nii.gz")

    # 回転パラメータ保存 (逆変換用)
    np.save(out_dir / "rotation_deg.npy", np.array([rx, ry]))

    print(f"  保存: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="椎体axial断面の自動tilt補正")
    parser.add_argument("input_dir", type=Path, help="predata_simpleのパス")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="出力先 (デフォルト: input_dirと同階層のannotation_data)",
    )
    parser.add_argument("--sample", type=str, default=None, help="特定サンプルのみ処理 (例: sample1)")
    parser.add_argument("--vertebra", type=str, default=None, help="特定椎体のみ (例: C3)")
    parser.add_argument("--range", type=float, default=30.0, help="探索範囲±degree (デフォルト: 30)")
    parser.add_argument("--dry_run", action="store_true", help="保存せず角度だけ確認")
    args = parser.parse_args()

    input_dir: Path = args.input_dir.resolve()
    output_dir: Path = (
        args.output_dir.resolve()
        if args.output_dir
        else input_dir.parent / "annotation_data"
    )

    samples = sorted(input_dir.iterdir())
    if args.sample:
        samples = [s for s in samples if s.name == args.sample]

    for sample in samples:
        if not sample.is_dir():
            continue
        vertebrae = sorted(sample.iterdir())
        if args.vertebra:
            vertebrae = [v for v in vertebrae if v.name == args.vertebra]

        for vert_dir in vertebrae:
            if not vert_dir.is_dir():
                continue
            print(f"\n[{sample.name}/{vert_dir.name}]")
            process_vertebra(vert_dir, output_dir, coarse_range=args.range, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
