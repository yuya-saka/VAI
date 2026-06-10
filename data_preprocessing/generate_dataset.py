import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import center_of_mass, affine_transform, label as nd_label
from scipy.spatial.transform import Rotation
import traceback

# --- 設定 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# パス設定
CSV_FILE = os.path.join(PROJECT_ROOT, "nifti_list.csv")
NIFTI_DIR = os.path.join(PROJECT_ROOT, "nifti_output")
SEG_DIR = os.path.join(PROJECT_ROOT, "segmentations")
LABEL_DIR = os.path.join(PROJECT_ROOT, "fracture_labels")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "spine_data")

# 出力ボリュームサイズ (固定)
OUTPUT_SHAPE = (128, 128, 64)

VERTEBRAE = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

# Windowing設定
BONE_WL, BONE_WW = 500, 2000
SOFT_WL, SOFT_WW = 40, 400

# 骨折判定の閾値
FRACTURE_OVERLAP_THRESHOLD = 0.10

# ★マージン設定 (Bounding Boxに対する倍率)★
MARGIN_XY_RATIO = 1.2   # XY方向：BBoxの1.2倍
MARGIN_Z_RATIO = 1.0    # Z方向：BBoxの1.5倍（椎間板を少し含む）


class SpineDatasetGenerator:
    """頸椎データセット生成クラス (Bounding Box基準版)"""

    def __init__(self, csv_file, nifti_dir, seg_dir, label_dir, output_dir):
        self.csv_file = csv_file
        self.nifti_dir = nifti_dir
        self.seg_dir = seg_dir
        self.label_dir = label_dir
        self.output_dir = output_dir

        self.csv_df = pd.read_csv(csv_file)
        os.makedirs(output_dir, exist_ok=True)

    def _load_nifti_data(self, sample_id):
        """CT画像と骨折ラベルの読み込み"""
        ct_path = os.path.join(self.nifti_dir, f"{sample_id}.nii.gz")
        if not os.path.exists(ct_path):
            raise FileNotFoundError(f"CT画像なし: {ct_path}")
        ct_img = nib.load(ct_path)
        ct_data = ct_img.get_fdata()

        fracture_path = os.path.join(self.label_dir, f"{sample_id}.nii.gz")
        if os.path.exists(fracture_path):
            fracture_img = nib.load(fracture_path)
            fracture_data = fracture_img.get_fdata()
        else:
            fracture_data = np.zeros_like(ct_data)

        return ct_img, ct_data, fracture_data

    def _load_vertebra_mask(self, sample_id, vertebra_id):
        """椎骨マスクの読み込み"""
        mask_path = os.path.join(self.seg_dir, sample_id, f"vertebrae_{vertebra_id}.nii.gz")
        if not os.path.exists(mask_path):
            return None, None
        
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        return mask_img, mask_data

    def _get_largest_component(self, mask_data):
        """マスクの最大連結成分のみを抽出"""
        labeled, num_components = nd_label(mask_data > 0)
        if num_components <= 1:
            return mask_data
        
        component_sizes = [(labeled == i).sum() for i in range(1, num_components + 1)]
        largest_component = np.argmax(component_sizes) + 1
        return (labeled == largest_component).astype(mask_data.dtype)

    def _compute_physical_centroid(self, mask_img, mask_data):
        """物理座標空間(mm)での重心を計算"""
        idx_centroid = center_of_mass(mask_data)
        if np.any(np.isnan(idx_centroid)):
            raise ValueError("重心計算エラー")
        idx_homogeneous = np.array([*idx_centroid, 1.0])
        centroid_mm = mask_img.affine @ idx_homogeneous
        return centroid_mm[:3]

    def _compute_all_centroids(self, sample_id):
        """全椎骨の物理重心を計算"""
        centroids_mm = {}
        for vid in VERTEBRAE:
            try:
                mask_img, mask_data = self._load_vertebra_mask(sample_id, vid)
                if mask_img is None:
                    centroids_mm[vid] = None
                    continue
                if np.sum(mask_data) < 10:
                    centroids_mm[vid] = None
                    continue
                mask_data = self._get_largest_component(mask_data)
                centroids_mm[vid] = self._compute_physical_centroid(mask_img, mask_data)
            except Exception:
                centroids_mm[vid] = None
        return centroids_mm

    def _compute_spine_vector(self, vertebra_id, centroids):
        """脊椎の方向ベクトル（下から上）を計算"""
        idx = int(vertebra_id[1])
        if idx == 7:
            upper, lower = "C6", "C7"
        else:
            upper, lower = f"C{idx}", f"C{idx+1}"

        p_upper, p_lower = centroids.get(upper), centroids.get(lower)
        if p_upper is None or p_lower is None:
            return None

        vec = p_upper - p_lower
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-6 else None

    def _compute_rotation_matrix(self, vertebra_id, centroids):
        """回転行列計算 (Z軸を脊椎方向に合わせる)"""
        if vertebra_id in ["C1", "C2"]:
            if centroids.get("C3") is not None:
                return self._compute_rotation_matrix("C3", centroids)
        
        spine_vec = self._compute_spine_vector(vertebra_id, centroids)
        if spine_vec is None:
            return None
            
        target_vec = np.array([0, 0, 1])
        rotation, _ = Rotation.align_vectors([target_vec], [spine_vec])
        return rotation.as_matrix()

    def _compute_bounding_box_physical(self, mask_img, mask_data, R_phys):
        """
        ★新しい関数★
        回転後の座標系でマスクのBounding Boxを計算（物理座標mm）
        
        Returns:
            bbox_min: (x_min, y_min, z_min) in mm (rotated space)
            bbox_max: (x_max, y_max, z_max) in mm (rotated space)
            centroid_rotated: 回転後空間での重心
        """
        indices = np.argwhere(mask_data > 0)
        if len(indices) == 0:
            return None, None, None

        # 画像座標 -> 物理座標(mm)
        N = len(indices)
        indices_homo = np.hstack([indices, np.ones((N, 1))])
        coords_phys = (mask_img.affine @ indices_homo.T).T[:, :3]

        # 重心
        centroid_phys = np.mean(coords_phys, axis=0)

        # 重心移動 & 回転 (Spine Aligned)
        coords_centered = coords_phys - centroid_phys
        coords_rotated = coords_centered @ R_phys

        # Bounding Box (回転後空間)
        bbox_min = np.min(coords_rotated, axis=0)
        bbox_max = np.max(coords_rotated, axis=0)

        return bbox_min, bbox_max, centroid_phys

    def _apply_windowing(self, data, wl, ww):
        lower, upper = wl - ww/2, wl + ww/2
        return np.clip((data - lower) / (upper - lower), 0, 1).astype(np.float32)

    def _extract_volume_bbox(self, ct_data, mask_data, fracture_data, ct_affine, 
                              centroid_phys, R_phys, bbox_min, bbox_max):
        """
        Bounding Box基準でボリュームを切り出し、固定サイズにリサイズ
        ★重心を中心として、FOVサイズのみBBoxから決定★
        """
        # 1. BBoxサイズ計算
        bbox_size = bbox_max - bbox_min
        
        # 2. マージン追加（倍率方式）
        # 各軸のサイズに倍率を適用してFOVを決定
        roi_size_mm = np.array([
            bbox_size[0] * MARGIN_XY_RATIO,
            bbox_size[1] * MARGIN_XY_RATIO,
            bbox_size[2] * MARGIN_Z_RATIO
        ])

        # 3. 元画像のボクセルサイズを取得（参考情報）
        voxel_size = np.abs(np.diag(ct_affine)[:3])

        # 5. 変換行列の構築
        # ★シンプル化：重心を中心として切り出し★
        
        # 出力空間の中心 -> 原点（ボクセル単位）
        out_center = np.array(OUTPUT_SHAPE) / 2.0
        T_center = np.eye(4)
        T_center[:3, 3] = -out_center  # ボクセル単位

        # スケーリング: 出力ボクセル -> mm（FOVサイズ / 出力ピクセル数）
        S = np.eye(4)
        S[0, 0] = roi_size_mm[0] / OUTPUT_SHAPE[0]
        S[1, 1] = roi_size_mm[1] / OUTPUT_SHAPE[1]
        S[2, 2] = roi_size_mm[2] / OUTPUT_SHAPE[2]

        # 回転の逆行列（回転後空間 -> 物理空間）
        R_inv = np.eye(4)
        R_inv[:3, :3] = R_phys.T

        # 重心への移動（物理空間）
        T_loc = np.eye(4)
        T_loc[:3, 3] = centroid_phys

        # 元画像座標への変換
        ct_affine_inv = np.linalg.inv(ct_affine)

        # 総合変換行列: 出力ボクセル -> 入力ボクセル
        # 出力中心を原点に -> スケール -> 回転戻し -> 重心へ移動 -> 画像座標へ
        M_total = ct_affine_inv @ T_loc @ R_inv @ S @ T_center

        matrix = M_total[:3, :3]
        offset = M_total[:3, 3]

        # 6. 直接OUTPUT_SHAPEに切り出し（リサイズ不要）
        ct_resized = affine_transform(ct_data, matrix, offset=offset, 
                                      output_shape=OUTPUT_SHAPE, order=1, cval=-1024)
        mask_resized = affine_transform(mask_data, matrix, offset=offset,
                                        output_shape=OUTPUT_SHAPE, order=0, cval=0)
        frac_resized = affine_transform(fracture_data, matrix, offset=offset,
                                        output_shape=OUTPUT_SHAPE, order=0, cval=0)

        # 8. 3チャンネル出力の構築
        volume = np.zeros((3, *OUTPUT_SHAPE), dtype=np.float32)
        
        # Ch 0: Bone window
        volume[0] = self._apply_windowing(ct_resized, BONE_WL, BONE_WW)
        
        # Ch 1: Soft tissue window
        volume[1] = self._apply_windowing(ct_resized, SOFT_WL, SOFT_WW)
        
        # Ch 2: Vertebra mask
        volume[2] = (mask_resized > 0.5).astype(np.float32)

        # 解像度情報 (参考用)
        final_resolution_mm = roi_size_mm / np.array(OUTPUT_SHAPE)

        return volume, mask_resized, frac_resized, final_resolution_mm

    def _determine_fracture_label(self, mask_resized, frac_resized):
        """骨折判定"""
        vertebra_voxels = np.sum(mask_resized > 0.5)
        if vertebra_voxels == 0:
            return 0, 0.0

        overlap = np.logical_and(mask_resized > 0.5, frac_resized > 0.5)
        overlap_voxels = np.sum(overlap)
        overlap_ratio = overlap_voxels / vertebra_voxels
        
        is_fracture = 1 if overlap_ratio > FRACTURE_OVERLAP_THRESHOLD else 0
        return is_fracture, overlap_ratio

    def _save_debug_nifti(self, save_dir, vertebra_id, volume):
        """確認用保存"""
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(volume[0], affine), 
                 os.path.join(save_dir, f"{vertebra_id}_bone.nii.gz"))
        nib.save(nib.Nifti1Image(volume[2], affine), 
                 os.path.join(save_dir, f"{vertebra_id}_mask.nii.gz"))

    def process_sample(self, sample_id):
        print(f"Processing ID: {sample_id} ...")
        metadata_list = []

        try:
            ct_img, ct_data, fracture_data = self._load_nifti_data(sample_id)
            centroids_mm = self._compute_all_centroids(sample_id)
            
            for vid in VERTEBRAE:
                if centroids_mm.get(vid) is None:
                    continue

                # 1. 回転行列
                R_phys = self._compute_rotation_matrix(vid, centroids_mm)
                if R_phys is None:
                    continue

                # 2. マスク読み込み
                mask_img, mask_data_full = self._load_vertebra_mask(sample_id, vid)
                if mask_img is None:
                    continue
                mask_data = self._get_largest_component(mask_data_full)

                # 3. Bounding Box計算 (回転後空間)
                bbox_min, bbox_max, centroid_phys = self._compute_bounding_box_physical(
                    mask_img, mask_data, R_phys
                )
                if bbox_min is None:
                    continue

                # 4. ボリューム抽出 & リサイズ
                volume, mask_resized, frac_resized, resolution = self._extract_volume_bbox(
                    ct_data, mask_data, fracture_data, ct_img.affine,
                    centroid_phys, R_phys, bbox_min, bbox_max
                )

                # 5. ラベル判定
                is_frac, overlap_ratio = self._determine_fracture_label(mask_resized, frac_resized)

                # 6. 保存
                save_dir = os.path.join(self.output_dir, sample_id)
                os.makedirs(save_dir, exist_ok=True)

                # PyTorch用 (C, Z, Y, X)
                volume_zyx = np.transpose(volume, (0, 3, 2, 1))
                np.save(os.path.join(save_dir, f"{vid}.npy"), volume_zyx)

                # 確認用NIfTI
                self._save_debug_nifti(save_dir, vid, volume)

                # BBoxサイズ (参考情報)
                bbox_size = bbox_max - bbox_min

                metadata_list.append({
                    'sample_id': sample_id,
                    'vertebra': vid,
                    'fracture_label': int(is_frac),
                    'overlap': float(overlap_ratio),
                    'bbox_x_mm': float(bbox_size[0]),
                    'bbox_y_mm': float(bbox_size[1]),
                    'bbox_z_mm': float(bbox_size[2]),
                    'res_x_mm': float(resolution[0]),
                    'res_y_mm': float(resolution[1]),
                    'res_z_mm': float(resolution[2]),
                    'file_path': f"{sample_id}/{vid}.npy"
                })
                
                status = "FRACTURE" if is_frac else "Normal"
                print(f"  {vid}: {status} | BBox: {bbox_size[0]:.1f}x{bbox_size[1]:.1f}x{bbox_size[2]:.1f}mm | Res: {resolution[0]:.2f}x{resolution[1]:.2f}x{resolution[2]:.2f}mm/vox")

        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            traceback.print_exc()

        return metadata_list

    def generate_all(self):
        target_ids = self.csv_df[self.csv_df['Exclude'] != True]['ID'].tolist()
        all_metadata = []
        print(f"Total samples: {len(target_ids)}")
        print(f"Output shape: {OUTPUT_SHAPE}")
        print(f"Margin XY: {MARGIN_XY_RATIO}x, Margin Z: {MARGIN_Z_RATIO}x")
        print("-" * 60)
        
        for sample_id in target_ids:
            meta = self.process_sample(str(sample_id))
            all_metadata.extend(meta)

        df = pd.DataFrame(all_metadata)
        df.to_csv(os.path.join(self.output_dir, "dataset_metadata.csv"), index=False)
        
        print("\n" + "=" * 60)
        print("Completed.")
        print(f"Total vertebrae: {len(df)}")
        if 'fracture_label' in df.columns:
            print(f"Fractures: {df['fracture_label'].sum()}")
        print("=" * 60)


if __name__ == "__main__":
    generator = SpineDatasetGenerator(CSV_FILE, NIFTI_DIR, SEG_DIR, LABEL_DIR, OUTPUT_DIR)
    generator.generate_all()