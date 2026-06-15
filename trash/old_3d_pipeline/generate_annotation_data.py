import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import affine_transform, label as nd_label, center_of_mass
from scipy.interpolate import CubicSpline
import traceback
import argparse

# プロジェクトルート設定
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# パス設定
CSV_FILE = os.path.join(PROJECT_ROOT, "nifti_list.csv")
NIFTI_DIR = os.path.join(PROJECT_ROOT, "nifti_output")
SEG_DIR = os.path.join(PROJECT_ROOT, "segmentations")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "predata_2")

VERTEBRAE = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]


class SpineCenterlineGenerator:
    """Global Spine Centerline基準のアノテーション用データ生成クラス
    修正: スプライン微分を廃止し、重心ベクトル + 線形補間(order=1)を採用
    """

    # 固定パラメータ
    OUTPUT_RESOLUTION = 0.4  # mm
    GROUP_SETTINGS = {
        "C1": {"xy_size": 256},       # 102.4 mm
        "C2": {"xy_size": 192},       # 76.8 mm
        "C3_C7": {"xy_size": 224},    # 89.6 mm (C3, C4, C5, C6, C7)
    }
    Z_MARGIN_VOXELS = 5  # Z軸のマージン
    BACKGROUND_VALUE = -1024  # 背景のHU値

    def __init__(self, csv_file, nifti_dir, seg_dir, output_dir):
        self.csv_file = csv_file
        self.nifti_dir = nifti_dir
        self.seg_dir = seg_dir
        self.output_dir = output_dir
        self.csv_df = pd.read_csv(csv_file)
        os.makedirs(output_dir, exist_ok=True)

    def _load_ct_data(self, sample_id):
        ct_path = os.path.join(self.nifti_dir, f"{sample_id}.nii.gz")
        if not os.path.exists(ct_path):
            raise FileNotFoundError(f"CT画像が見つかりません: {ct_path}")
        ct_img = nib.load(ct_path)
        ct_data = ct_img.get_fdata().astype(np.float32)
        return ct_img, ct_data

    def _load_all_masks(self, sample_id):
        masks = {}
        mask_img = None
        for vertebra in VERTEBRAE:
            mask_path = os.path.join(self.seg_dir, sample_id, f"vertebrae_{vertebra}.nii.gz")
            if os.path.exists(mask_path):
                img = nib.load(mask_path)
                mask_data = img.get_fdata().astype(np.float32)
                mask_data = self._get_largest_component(mask_data)
                masks[vertebra] = mask_data
                if mask_img is None:
                    mask_img = img
        return masks, mask_img

    def _get_largest_component(self, mask_data):
        labeled, num_components = nd_label(mask_data > 0)
        if num_components <= 1:
            return mask_data
        component_sizes = [(labeled == i).sum() for i in range(1, num_components + 1)]
        largest_component = np.argmax(component_sizes) + 1
        return (labeled == largest_component).astype(mask_data.dtype)

    def _compute_centroids(self, masks, affine):
        centroids = {}
        for vertebra, mask_data in masks.items():
            if np.sum(mask_data) < 10:
                continue
            idx_centroid = center_of_mass(mask_data)
            if np.any(np.isnan(idx_centroid)):
                continue
            idx_homo = np.array([*idx_centroid, 1.0])
            centroid_mm = affine @ idx_homo
            centroids[vertebra] = centroid_mm[:3]
        return centroids

    def _get_group(self, vertebra):
        if vertebra == "C1":
            return "C1"
        elif vertebra == "C2":
            return "C2"
        else:
            return "C3_C7"

    def _get_stable_local_coordinate_system(self, current_center, prev_center, next_center):
        """
        ★修正ポイント2: 重心ベクトルを使う★
        スプライン微分ではなく、前後の重心位置から安定したZ軸(体軸)を決定する。
        古いコードの `_compute_spine_vector` と同等のロジック。
        """
        # 1. Z軸（体軸方向）の決定
        # 前後の椎骨がある場合は、その間のベクトルを使用（平均的な向き）
        if prev_center is not None and next_center is not None:
            tangent = next_center - prev_center
        elif prev_center is None and next_center is not None:
            tangent = next_center - current_center
        elif prev_center is not None and next_center is None:
            tangent = current_center - prev_center
        else:
            # 孤立している場合はCTのZ軸(0,0,1)とする
            tangent = np.array([0.0, 0.0, 1.0])
            
        # 正規化（長さを1にする）
        norm = np.linalg.norm(tangent)
        if norm > 1e-6:
            tangent = tangent / norm
        else:
            tangent = np.array([0.0, 0.0, 1.0])

        # 2. X軸（左右方向）の決定
        # CT座標系のY軸（Posterior-Anterior）と、求めたZ軸の外積からX軸を作る
        # これにより、体がねじれるような不自然な回転を防止する
        ct_y_axis = np.array([0.0, 1.0, 0.0])
        
        # Z軸とCT-Y軸の外積 ＝ 仮のX軸（左右）
        binormal = np.cross(ct_y_axis, tangent)
        
        # 万が一平行に近い場合の保護
        if np.linalg.norm(binormal) < 1e-2:
            binormal = np.array([1.0, 0.0, 0.0])
        
        binormal = binormal / np.linalg.norm(binormal)

        # 3. Y軸（前後方向）の決定
        # Z軸とX軸から直交するY軸を再計算
        normal = np.cross(tangent, binormal)
        normal = normal / np.linalg.norm(normal)

        # 回転行列の構築 (列ベクトル: X, Y, Z)
        # これで「画像中心は重心」かつ「向きは背骨の並び」に固定される
        rotation_matrix = np.column_stack([binormal, normal, tangent])

        return rotation_matrix, current_center

    def _resample_along_centerline(self, ct_data, affine, rotation_matrix, center, output_shape, order=1):
        """Centerline基準でリサンプリング + 切り出し"""
        out_shape = output_shape
        out_res = self.OUTPUT_RESOLUTION

        # 出力空間の中心（ボクセル単位）
        out_center = np.array(out_shape) / 2.0

        # 変換行列の構築
        # 1. 出力空間の中心を原点に移動
        T_center = np.eye(4)
        T_center[:3, 3] = -out_center

        # 2. スケーリング: 出力ボクセル -> mm
        S = np.eye(4)
        S[0, 0] = out_res
        S[1, 1] = out_res
        S[2, 2] = out_res

        # 3. 回転の逆変換（局所座標系 -> 元の物理空間）
        R_inv = np.eye(4)
        R_inv[:3, :3] = rotation_matrix.T

        # 4. 中心への移動
        T_loc = np.eye(4)
        T_loc[:3, 3] = center

        # 5. 元画像座標への変換
        affine_inv = np.linalg.inv(affine)

        # 総合変換: 出力ボクセル -> 入力ボクセル
        M_total = affine_inv @ T_loc @ R_inv @ S @ T_center

        matrix = M_total[:3, :3]
        offset = M_total[:3, 3]

        # リサンプリング実行
        resampled_data = affine_transform(
            ct_data,
            matrix,
            offset=offset,
            output_shape=out_shape,
            order=order,  # ★指定されたorderを使用★
            cval=self.BACKGROUND_VALUE
        )

        return resampled_data

    def process_sample(self, sample_id):
        """1サンプル（C1〜C7全体）を処理"""
        try:
            print(f"\n処理中: {sample_id}")

            # 1. CT画像読み込み
            ct_img, ct_data = self._load_ct_data(sample_id)

            # 2. 全椎骨マスク読み込み
            masks, mask_img = self._load_all_masks(sample_id)
            if len(masks) < 3:
                print(f"  エラー: 椎骨マスクが不足しています ({len(masks)}個)")
                return 0

            # 3. 重心計算
            centroids = self._compute_centroids(masks, mask_img.affine)

            # 4. 各椎骨を処理
            success_count = 0
            
            # 各椎骨についてループ（VERTEBRAEの順番通り）
            for i, vertebra in enumerate(VERTEBRAE):
                if vertebra not in masks or vertebra not in centroids:
                    continue

                try:
                    # 前後の椎骨の重心を取得（重心ベクトル計算用）
                    prev_v = VERTEBRAE[i - 1] if i > 0 else None
                    next_v = VERTEBRAE[i + 1] if i < len(VERTEBRAE) - 1 else None
                    
                    prev_c = centroids.get(prev_v)
                    next_c = centroids.get(next_v)
                    curr_c = centroids[vertebra]

                    # ★修正ポイント2の適用★
                    # スプラインではなく、重心ベクトルから座標系を取得
                    rotation_matrix, center = self._get_stable_local_coordinate_system(
                        curr_c, prev_c, next_c
                    )

                    # グループを取得してXYサイズを決定
                    group = self._get_group(vertebra)
                    xy_size = self.GROUP_SETTINGS[group]["xy_size"]

                    # ステップ1: マスクのリサンプリング (order=0)
                    temp_shape = (xy_size, xy_size, 300) 
                    temp_mask = self._resample_along_centerline(
                        masks[vertebra], ct_img.affine, rotation_matrix, center, temp_shape, order=0
                    )

                    # ステップ2: マスクのZ軸範囲を検出
                    z_indices = np.where(temp_mask > 0.5)[2]
                    if len(z_indices) == 0:
                        print(f"  {vertebra}: エラー - マスクが空です")
                        continue

                    z_min = max(0, z_indices.min() - self.Z_MARGIN_VOXELS)
                    z_max = min(temp_shape[2], z_indices.max() + self.Z_MARGIN_VOXELS + 1)
                    z_size = z_max - z_min

                    # ステップ3: 最終的な出力形状を決定
                    final_shape = (xy_size, xy_size, z_size)

                    # Z軸オフセットの調整
                    z_center_offset = (z_min + z_max) / 2.0 - temp_shape[2] / 2.0
                    adjusted_center = center + rotation_matrix[:, 2] * z_center_offset * self.OUTPUT_RESOLUTION

                    # ステップ4: リサンプリング実行
                    
                    # ★修正ポイント1の適用: CT画像は order=1 (線形補間) を使用★
                    # これにより、骨が薄くなったり消えたりするのを防ぐ
                    resampled_ct = self._resample_along_centerline(
                        ct_data, ct_img.affine, rotation_matrix, adjusted_center, final_shape, order=1
                    )
                    
                    # マスクは order=0 (最近傍補間)
                    resampled_mask = self._resample_along_centerline(
                        masks[vertebra], ct_img.affine, rotation_matrix, adjusted_center, final_shape, order=0
                    )

                    # 保存処理
                    vertebra_output_dir = os.path.join(self.output_dir, sample_id, vertebra)
                    os.makedirs(vertebra_output_dir, exist_ok=True)

                    output_affine = np.eye(4)
                    output_affine[0, 0] = self.OUTPUT_RESOLUTION
                    output_affine[1, 1] = self.OUTPUT_RESOLUTION
                    output_affine[2, 2] = self.OUTPUT_RESOLUTION

                    ct_output_path = os.path.join(vertebra_output_dir, "ct.nii.gz")
                    ct_img_out = nib.Nifti1Image(resampled_ct.astype(np.float32), output_affine)
                    nib.save(ct_img_out, ct_output_path)

                    mask_output_path = os.path.join(vertebra_output_dir, "mask.nii.gz")
                    mask_img_out = nib.Nifti1Image(resampled_mask.astype(np.float32), output_affine)
                    nib.save(mask_img_out, mask_output_path)

                    print(f"  {vertebra}: 成功 -> {final_shape}")
                    success_count += 1

                except Exception as e:
                    print(f"  {vertebra}: エラー - {e}")

            return success_count

        except Exception as e:
            print(f"  サンプル全体のエラー: {e}")
            traceback.print_exc()
            return 0

    def generate_all(self, sample_ids=None, resume_from=None):
        base_df = self.csv_df[self.csv_df['Exclude'] != True].copy()
        base_df['ID'] = base_df['ID'].astype(str)
        all_target_ids = base_df['ID'].tolist()

        target_ids = []
        if sample_ids is not None:
            target_ids = [sid for sid in sample_ids if sid in all_target_ids]
        elif resume_from is not None:
            resume_from = str(resume_from)
            if resume_from in all_target_ids:
                start_index = all_target_ids.index(resume_from)
                target_ids = all_target_ids[start_index:]
        else:
            target_ids = all_target_ids

        print(f"処理開始: {len(target_ids)}件")
        
        total_success = 0
        for i, sample_id in enumerate(target_ids):
            print(f"[{i+1}/{len(target_ids)}] ", end="")
            success_count = self.process_sample(str(sample_id))
            total_success += success_count

        print("処理完了")

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--sample-id', nargs='+')
    group.add_argument('--resume-from', type=str)
    args = parser.parse_args()

    generator = SpineCenterlineGenerator(
        csv_file=CSV_FILE,
        nifti_dir=NIFTI_DIR,
        seg_dir=SEG_DIR,
        output_dir=OUTPUT_DIR
    )

    generator.generate_all(
        sample_ids=args.sample_id,
        resume_from=args.resume_from
    )

if __name__ == "__main__":
    main()


'''
使用例:

  # 全サンプルを処理

  python generate_annotation_data.py



  # 特定のサンプルのみを処理 (指定リストのみ)

  python generate_annotation_data.py --sample-id sample1 sample2



  # 指定したサンプルから再開し、最後まで処理 (途中から再開)

  python generate_annotation_data.py --resume-from sample15

'''