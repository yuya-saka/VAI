import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import affine_transform, label as nd_label
import traceback
import argparse

# ==========================================
# 1. 設定・定数定義
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

CSV_FILE = os.path.join(PROJECT_ROOT, "nifti_list.csv")
NIFTI_DIR = os.path.join(PROJECT_ROOT, "nifti_output")
SEG_DIR = os.path.join(PROJECT_ROOT, "segmentations")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "predata_simple")  # 出力先

VERTEBRAE = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

class SimpleSpineCropper:
    """
    軸探索を行わず、マスク重心を中心として固定サイズ(256x256)で切り出すクラス
    """

    OUTPUT_RESOLUTION = 0.4  # mm/voxel (解像度を統一)
    XY_SIZE = 256            # pixel (切り出しサイズを統一)
    Z_MARGIN_VOXELS = 5     # 上下の余白（スライス数）
    BACKGROUND_VALUE = -1024

    def __init__(self, csv_file, nifti_dir, seg_dir, output_dir):
        self.csv_file = csv_file
        self.nifti_dir = nifti_dir
        self.seg_dir = seg_dir
        self.output_dir = output_dir
        
        if os.path.exists(csv_file):
            self.csv_df = pd.read_csv(csv_file)
        else:
            self.csv_df = None
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
                # 最大連結成分のみ抽出（ノイズ除去）
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

    def _get_center_of_mass(self, mask_data, affine):
        """
        マスクの重心座標(mm)を計算する
        """
        indices = np.argwhere(mask_data > 0)
        if len(indices) < 10:
            return None
        
        # ボクセル座標 -> 物理座標(mm)
        coords_vox = np.column_stack([indices, np.ones(len(indices))])
        coords_mm = np.dot(coords_vox, affine.T)[:, :3]
        
        # 単純平均（重心）
        center = np.mean(coords_mm, axis=0)
        return center

    # -------------------------------------------------------------------------
    # リサンプリング処理 (回転なし・固定サイズ)
    # -------------------------------------------------------------------------
    def _resample_fixed_crop(self, ct_data, affine, center, output_shape, order=1, cval=None):
        """
        回転行列を使わず(単位行列)、重心を中心としてリサンプリングする
        """
        if cval is None: cval = self.BACKGROUND_VALUE
        
        # 設定された解像度
        out_res = self.OUTPUT_RESOLUTION
        out_center = np.array(output_shape) / 2.0

        # アフィン変換行列の構築 (Global軸に平行)
        # 1. 出力画像の中心を原点へ
        T_center = np.eye(4); T_center[:3, 3] = -out_center
        # 2. スケーリング (0.4mm/pix)
        S = np.eye(4); S[0,0]=S[1,1]=S[2,2] = out_res
        # 3. 回転はなし (単位行列)
        R_inv = np.eye(4) 
        # 4. 指定された重心位置へ移動
        T_loc = np.eye(4); T_loc[:3, 3] = center
        
        # 元画像のAffineの逆行列
        affine_inv = np.linalg.inv(affine)
        
        # 全体の変換行列合成: (Pixel -> Physical) -> (Center Offset) -> ... -> (Input Index)
        # Input_Index = Affine_inv @ T_loc @ R_inv @ S @ T_center @ Output_Index
        M_total = affine_inv @ T_loc @ R_inv @ S @ T_center

        return affine_transform(
            ct_data, M_total[:3, :3], offset=M_total[:3, 3],
            output_shape=output_shape, order=order, cval=cval
        )

    def process_sample(self, sample_id):
        try:
            print(f"\n--- Processing: {sample_id} ---")
            try:
                ct_img, ct_data = self._load_ct_data(sample_id)
                masks, mask_img = self._load_all_masks(sample_id)
            except Exception as e:
                print(f"  Load Error: {e}"); return 0
            
            if len(masks) == 0: return 0
            success_count = 0
            
            for vertebra in VERTEBRAE:
                if vertebra not in masks: continue

                try:
                    # 1. 重心計算
                    center = self._get_center_of_mass(masks[vertebra], ct_img.affine)
                    if center is None: continue

                    # 2. Z軸範囲（スライス数）の決定
                    # まず仮にZ方向に十分長い範囲で切り出して、マスクがある範囲を探す
                    # XYは256固定
                    xy_size = self.XY_SIZE
                    temp_z_size = 500 # 十分大きく
                    temp_shape = (xy_size, xy_size, temp_z_size)
                    
                    # マスクを仮切り出し (Nearest Neighbor)
                    temp_mask = self._resample_fixed_crop(
                        masks[vertebra], ct_img.affine, center, temp_shape, 
                        order=0, cval=0
                    )
                    
                    # マスクが存在するZインデックスを取得
                    z_indices = np.where(temp_mask > 0)[2]
                    
                    if len(z_indices) == 0:
                        print(f"  {vertebra}: Skipped (Empty in crop)"); continue

                    # 範囲決定
                    z_min = max(0, z_indices.min() - self.Z_MARGIN_VOXELS)
                    z_max = min(temp_z_size, z_indices.max() + self.Z_MARGIN_VOXELS + 1)
                    
                    real_z_size = z_max - z_min
                    final_shape = (xy_size, xy_size, real_z_size)

                    # Z軸の中心位置を補正 (仮切り出し空間での中心ズレをPhysical Centerに反映)
                    # 仮空間での中心は temp_z_size / 2.0
                    # 今回の切り出し中心オフセット
                    z_center_shift_pix = (z_min + z_max) / 2.0 - (temp_z_size / 2.0)
                    
                    # 補正後の中心座標 (Global Z軸方向にずらす)
                    # 回転していないので、CTのAffineのZ軸方向ではなく、単純にZ座標(Physical)に加算ではない
                    # ここでは「Global座標系(単位行列ベース)」でリサンプリングしているので、
                    # Z方向のベクトルは (0, 0, 1) である。
                    adjusted_center = center + np.array([0, 0, 1]) * z_center_shift_pix * self.OUTPUT_RESOLUTION

                    # 3. 本番切り出し
                    resampled_ct = self._resample_fixed_crop(
                        ct_data, ct_img.affine, adjusted_center, final_shape, 
                        order=1, cval=self.BACKGROUND_VALUE
                    )
                    resampled_mask = self._resample_fixed_crop(
                        masks[vertebra], ct_img.affine, adjusted_center, final_shape, 
                        order=0, cval=0
                    ) # マスクは0/1維持のためorder=0推奨

                    # 4. 保存
                    vertebra_dir = os.path.join(self.output_dir, sample_id, vertebra)
                    os.makedirs(vertebra_dir, exist_ok=True)
                    
                    # 出力画像のAffine (単位行列 * 解像度)
                    out_aff = np.eye(4)
                    np.fill_diagonal(out_aff, self.OUTPUT_RESOLUTION)
                    out_aff[3,3] = 1

                    nib.save(nib.Nifti1Image(resampled_ct, out_aff), os.path.join(vertebra_dir, "ct.nii.gz"))
                    nib.save(nib.Nifti1Image(resampled_mask, out_aff), os.path.join(vertebra_dir, "mask.nii.gz"))

                    print(f"  {vertebra}: Saved -> {final_shape}")
                    success_count += 1

                except Exception as e:
                    print(f"  {vertebra}: Error - {e}"); traceback.print_exc()

            return success_count

        except Exception as e:
            print(f"  Global Error: {e}"); traceback.print_exc(); return 0

    def generate_all(self, sample_ids=None):
        if self.csv_df is not None:
            if 'Exclude' in self.csv_df.columns:
                base_df = self.csv_df[self.csv_df['Exclude'] != True]
            else:
                base_df = self.csv_df
            all_ids = [str(x) for x in base_df['ID'].tolist()]
        else:
            all_ids = []

        target_ids = []
        if sample_ids: target_ids = [str(sid) for sid in sample_ids]
        else: target_ids = all_ids

        print(f"Targets: {len(target_ids)}, Output: {self.output_dir}")
        for i, sid in enumerate(target_ids):
            print(f"[{i+1}/{len(target_ids)}] ", end="")
            self.process_sample(str(sid))
        print("Completed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids', nargs='+')
    args = parser.parse_args()

    cropper = SimpleSpineCropper(CSV_FILE, NIFTI_DIR, SEG_DIR, OUTPUT_DIR)
    cropper.generate_all(sample_ids=args.ids)

if __name__ == "__main__":
    main()