"""
NIfTI 3D → PNG 2D データセット変換スクリプト

annotation_dataのNIfTI形式データを、PNG画像ベースのデータセットに変換します。
- CT画像: 224x224 PNG（ウィンドウ処理済み、0-255正規化、上下反転）
- マスク画像: 224x224 PNG（バイナリ、上下反転）
- オーバーレイ画像: 224x224 PNG（確認用、上下反転）
- 線分座標: JSON（crop後、反転後の座標）
"""

import os
import sys
import json
import argparse
import numpy as np
import nibabel as nib
import nibabel.processing
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm


def load_vertebra_data(vertebra_dir):
    """
    椎骨データを読み込む

    Args:
        vertebra_dir: 椎骨データのディレクトリ

    Returns:
        ct_nii, mask_nii, annotations
    """
    vertebra_dir = Path(vertebra_dir)

    # CTファイルを検索
    ct_file = None
    for pattern in ['ct_*.nii.gz', '*_bone.nii.gz', 'ct_* cropped.nii.gz', 'ct cropped.nii.gz']:
        ct_files = list(vertebra_dir.glob(pattern))
        if ct_files:
            ct_file = ct_files[0]
            break

    if ct_file is None:
        for f in vertebra_dir.glob('*.nii.gz'):
            name = f.name.lower()
            if 'mask' in name or 'label' in name or 'segment' in name:
                continue
            if 'ct' in name or 'cropped' in name or 'bone' in name:
                ct_file = f
                break

    if ct_file is None:
        raise FileNotFoundError(f"CT file not found in {vertebra_dir}")

    # マスクファイルを検索
    mask_file = None
    for pattern in ['mask*.nii.gz', '*_mask.nii.gz', 'mask.nii.gz*']:
        mask_files = list(vertebra_dir.glob(pattern))
        if mask_files:
            mask_file = mask_files[0]
            break

    if mask_file is None:
        raise FileNotFoundError(f"Mask file not found in {vertebra_dir}")

    # NIfTI読み込み
    ct_nii = nib.load(ct_file)
    mask_nii = nib.load(mask_file)

    # JSONアノテーションを読み込み
    json_files = sorted(list(vertebra_dir.glob('*.mrk.json')) + list(vertebra_dir.glob('[0-9].json')))
    annotations = {}

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        if 'markups' in data and len(data['markups']) > 0:
            markup = data['markups'][0]
            if 'controlPoints' in markup:
                control_points = [cp['position'] for cp in markup['controlPoints']]
                annotations[json_file.name] = control_points

    return ct_nii, mask_nii, annotations


def world_to_voxel(world_coords, affine):
    """
    ワールド座標（LPS）をボクセル座標に変換

    Args:
        world_coords: [[x, y, z], ...] LPS座標系
        affine: nibabelのアフィン行列 (RAS座標系)

    Returns:
        voxel_coords: [[i, j, k], ...] ボクセル座標
    """
    world_coords = np.array(world_coords)

    # LPS → RAS 変換
    world_coords_ras = world_coords.copy()
    world_coords_ras[:, 0] = -world_coords_ras[:, 0]
    world_coords_ras[:, 1] = -world_coords_ras[:, 1]

    # アフィン変換の逆行列
    inv_affine = np.linalg.inv(affine)

    # 同次座標に変換
    homogeneous = np.hstack([world_coords_ras, np.ones((world_coords_ras.shape[0], 1))])

    # ボクセル座標に変換
    voxel_coords = (inv_affine @ homogeneous.T).T[:, :3]

    return voxel_coords


def apply_window(ct_slice, window_level=400, window_width=2000):
    """
    CTウィンドウ処理を適用

    Args:
        ct_slice: CT画像スライス
        window_level: ウィンドウレベル（中心値）
        window_width: ウィンドウ幅

    Returns:
        windowed: ウィンドウ処理後の画像 [0, 1]
    """
    lower = window_level - window_width / 2
    upper = window_level + window_width / 2

    windowed = np.clip(ct_slice, lower, upper)
    windowed = (windowed - lower) / (upper - lower)

    return windowed


def center_crop(image, target_size=224):
    """
    画像を中心からcrop

    Args:
        image: 入力画像 (H, W) または (H, W, C)
        target_size: 目標サイズ

    Returns:
        cropped: crop後の画像
        offset: cropのオフセット (top, left)
    """
    h, w = image.shape[:2]
    c_h, c_w = h // 2, w // 2
    half_size = target_size // 2

    top = max(0, c_h - half_size)
    left = max(0, c_w - half_size)
    bottom = min(h, c_h + half_size)
    right = min(w, c_w + half_size)

    if image.ndim == 2:
        cropped = image[top:bottom, left:right]
    else:
        cropped = image[top:bottom, left:right, :]

    # パディング（必要な場合）
    if cropped.shape[0] < target_size or cropped.shape[1] < target_size:
        pad_h = max(0, target_size - cropped.shape[0])
        pad_w = max(0, target_size - cropped.shape[1])

        if image.ndim == 2:
            cropped = np.pad(cropped,
                           ((pad_h // 2, pad_h - pad_h // 2),
                            (pad_w // 2, pad_w - pad_w // 2)),
                           mode='constant', constant_values=0)
        else:
            cropped = np.pad(cropped,
                           ((pad_h // 2, pad_h - pad_h // 2),
                            (pad_w // 2, pad_w - pad_w // 2),
                            (0, 0)),
                           mode='constant', constant_values=0)

    return cropped, (top, left)


def draw_line_on_slice(overlay, voxel_coords, slice_idx, offset, target_size, color, threshold=1.5, flip_vertical=True):
    """
    スライスに線分を描画し、crop後の座標を返す

    Args:
        overlay: オーバーレイ画像
        voxel_coords: ボクセル座標 (N, 3)
        slice_idx: スライスインデックス
        offset: cropオフセット (top, left)
        target_size: 画像サイズ
        color: 線の色 (B, G, R)
        threshold: スライス付近の閾値
        flip_vertical: 上下反転するか

    Returns:
        line_points: crop後の座標リスト [[x, y], ...]
    """
    line_points = []

    # スライス付近の点を抽出
    z_coords = voxel_coords[:, 2]
    close_mask = np.abs(z_coords - slice_idx) <= threshold

    if not np.any(close_mask):
        return line_points

    close_points = voxel_coords[close_mask]

    # crop後の座標に変換
    for point in close_points:
        x_crop = point[0] - offset[1]
        y_crop = point[1] - offset[0]

        # 上下反転（オプション）
        if flip_vertical:
            y_crop = (target_size - 1) - y_crop

        # 画像範囲内かチェック
        if 0 <= x_crop < target_size and 0 <= y_crop < target_size:
            line_points.append([float(x_crop), float(y_crop)])

    # 点を接続して線を描画
    for i in range(len(line_points) - 1):
        pt1 = (int(line_points[i][0]), int(line_points[i][1]))
        pt2 = (int(line_points[i+1][0]), int(line_points[i+1][1]))
        cv2.line(overlay, pt1, pt2, color, 2)

    # 点をマーカーで描画
    for point in line_points:
        pt = (int(point[0]), int(point[1]))
        cv2.circle(overlay, pt, 3, color, -1)

    return line_points


def get_valid_slices(ct_nii, annotations):
    """
    4本の線が全て存在するスライスのリストを取得

    Args:
        ct_nii: CT画像
        annotations: アノテーションデータ

    Returns:
        valid_slices: 有効なスライスインデックスのリスト
    """
    affine = ct_nii.affine
    ct_data = ct_nii.get_fdata()
    num_slices = ct_data.shape[2]

    # 各線のZ座標範囲を取得
    z_ranges = []
    for json_file in sorted(annotations.keys()):
        control_points = annotations[json_file]
        if len(control_points) == 0:
            return []

        voxel_coords = world_to_voxel(control_points, affine)
        z_coords = voxel_coords[:, 2]
        z_ranges.append((np.min(z_coords), np.max(z_coords)))

    # 全ての線が存在するスライス範囲を計算
    common_z_min = max([z_range[0] for z_range in z_ranges])
    common_z_max = min([z_range[1] for z_range in z_ranges])

    if common_z_min >= common_z_max:
        return []

    # スライスインデックスのリストを生成
    valid_slices = []
    for slice_idx in range(int(np.ceil(common_z_min)), int(np.floor(common_z_max)) + 1):
        if 0 <= slice_idx < num_slices:
            valid_slices.append(slice_idx)

    return valid_slices


def convert_vertebra_to_png(vertebra_dir, output_dir, target_size=224, window_level=400, window_width=2000, flip_vertical=True):
    """
    1つの椎骨データをPNG形式に変換

    Args:
        vertebra_dir: 椎骨データのディレクトリ
        output_dir: 出力ディレクトリ
        target_size: 画像サイズ
        window_level: ウィンドウレベル
        window_width: ウィンドウ幅
        flip_vertical: 上下反転するか

    Returns:
        num_slices: 変換されたスライス数（エラー時は0）
    """
    vertebra_dir = Path(vertebra_dir)
    output_dir = Path(output_dir)

    # 出力ディレクトリを作成
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'masks').mkdir(parents=True, exist_ok=True)
    (output_dir / 'overlays').mkdir(parents=True, exist_ok=True)

    try:
        # データ読み込み
        ct_nii, mask_nii, annotations = load_vertebra_data(vertebra_dir)

        if not np.allclose(ct_nii.affine, mask_nii.affine) or ct_nii.shape != mask_nii.shape:
            print('  警告: CTとマスクのボクセルグリッドが異なります。マスクをCT空間にリサンプリングします。')
            mask_nii = nibabel.processing.resample_from_to(mask_nii, ct_nii, order=0)

        ct_data = ct_nii.get_fdata()
        mask_data = mask_nii.get_fdata()
        affine = ct_nii.affine

        if len(annotations) != 4:
            print(f"  Warning: Expected 4 annotations, got {len(annotations)}")
            return 0

        # 有効なスライスを取得
        valid_slices = get_valid_slices(ct_nii, annotations)
        if len(valid_slices) == 0:
            print(f"  Warning: No valid slices found")
            return 0

        # 各線のボクセル座標を計算
        lines_voxel = {}
        for json_file in sorted(annotations.keys()):
            control_points = annotations[json_file]
            voxel_coords = world_to_voxel(control_points, affine)
            lines_voxel[json_file] = voxel_coords

        # 線分座標を保存するためのJSON
        lines_json = {}

        # 各スライスを処理
        for slice_idx in valid_slices:
            # CT画像を取得
            ct_slice = ct_data[:, :, slice_idx].T  # (H, W)
            mask_slice = mask_data[:, :, slice_idx].T  # (H, W)

            # ウィンドウ処理
            ct_windowed = apply_window(ct_slice, window_level, window_width)

            # 中心crop
            ct_cropped, offset = center_crop(ct_windowed, target_size)
            mask_cropped, _ = center_crop(mask_slice, target_size)

            # 上下反転
            if flip_vertical:
                ct_cropped = np.flipud(ct_cropped)
                mask_cropped = np.flipud(mask_cropped)

            # CT画像を0-255に変換してPNG保存
            ct_uint8 = (ct_cropped * 255).astype(np.uint8)
            Image.fromarray(ct_uint8).save(output_dir / 'images' / f'slice_{slice_idx:03d}.png')

            # マスク画像をバイナリ化してPNG保存
            mask_binary = (mask_cropped > 0).astype(np.uint8) * 255
            Image.fromarray(mask_binary).save(output_dir / 'masks' / f'slice_{slice_idx:03d}.png')

            # オーバーレイ画像を作成
            overlay = cv2.cvtColor(ct_uint8, cv2.COLOR_GRAY2BGR)

            # マスクをオーバーレイ
            mask_colored = np.zeros_like(overlay)
            mask_colored[:, :, 0] = mask_binary  # 赤チャンネル
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)

            # 線分をオーバーレイ & 座標を記録
            colors = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 255, 0)]  # cyan, magenta, green, yellow
            lines_json[str(slice_idx)] = {}

            for idx, (json_file, voxel_coords) in enumerate(sorted(lines_voxel.items())):
                line_key = f"line_{idx + 1}"
                line_points = draw_line_on_slice(
                    overlay, voxel_coords, slice_idx, offset, target_size, colors[idx],
                    flip_vertical=flip_vertical
                )
                lines_json[str(slice_idx)][line_key] = line_points

            # オーバーレイ画像を保存
            cv2.imwrite(str(output_dir / 'overlays' / f'slice_{slice_idx:03d}.png'), overlay)

        # 線分座標をJSON保存
        with open(output_dir / 'lines.json', 'w') as f:
            json.dump(lines_json, f, indent=2)

        return len(valid_slices)

    except Exception as e:
        print(f"  Error: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description='NIfTI 3D → PNG 2D データセット変換')
    parser.add_argument('--input_dir', type=str, default='../annotation_data',
                        help='入力ディレクトリ（annotation_data）')
    parser.add_argument('--output_dir', type=str, default='../dataset',
                        help='出力ディレクトリ（dataset）')
    parser.add_argument('--target_size', type=int, default=224,
                        help='画像サイズ（デフォルト: 224）')
    parser.add_argument('--window_level', type=int, default=400,
                        help='CTウィンドウレベル（デフォルト: 400）')
    parser.add_argument('--window_width', type=int, default=2000,
                        help='CTウィンドウ幅（デフォルト: 2000）')
    parser.add_argument('--no_flip', action='store_true',
                        help='上下反転を無効化（デフォルト: 反転する）')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    # サンプルディレクトリを取得
    sample_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith('sample')])

    if len(sample_dirs) == 0:
        print(f"Error: No sample directories found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(sample_dirs)} samples")
    print(f"Output directory: {output_dir}")
    print(f"Settings: size={args.target_size}, level={args.window_level}, width={args.window_width}, flip={not args.no_flip}")
    print()

    # 統計
    total_samples = 0
    total_vertebrae = 0
    total_slices = 0
    failed_vertebrae = []

    # 各サンプルを処理
    for sample_dir in tqdm(sample_dirs, desc="Processing samples"):
        sample_name = sample_dir.name

        # 椎骨ディレクトリを取得
        vertebra_names = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

        for vertebra_name in vertebra_names:
            vertebra_dir = sample_dir / vertebra_name

            if not vertebra_dir.exists():
                continue

            output_vertebra_dir = output_dir / sample_name / vertebra_name

            # 変換実行
            num_slices = convert_vertebra_to_png(
                vertebra_dir,
                output_vertebra_dir,
                target_size=args.target_size,
                window_level=args.window_level,
                window_width=args.window_width,
                flip_vertical=not args.no_flip
            )

            if num_slices > 0:
                total_vertebrae += 1
                total_slices += num_slices
            else:
                failed_vertebrae.append(f"{sample_name}/{vertebra_name}")

        total_samples += 1

    # 結果サマリー
    print()
    print("=" * 80)
    print("変換完了")
    print("=" * 80)
    print(f"処理サンプル数: {total_samples}")
    print(f"成功した椎骨数: {total_vertebrae}")
    print(f"総スライス数: {total_slices}")

    if len(failed_vertebrae) > 0:
        print(f"\n失敗した椎骨 ({len(failed_vertebrae)}):")
        for v in failed_vertebrae[:10]:  # 最初の10件のみ表示
            print(f"  - {v}")
        if len(failed_vertebrae) > 10:
            print(f"  ... and {len(failed_vertebrae) - 10} more")

    print()
    print(f"出力先: {output_dir}")


if __name__ == "__main__":
    main()
