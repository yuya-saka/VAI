import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import center_of_mass, affine_transform, label as nd_label
from scipy.spatial.transform import Rotation
import traceback
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# =================================================================
# 設定
# =================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

CSV_FILE = os.path.join(PROJECT_ROOT, "nifti_list.csv")
SEG_DIR = os.path.join(PROJECT_ROOT, "segmentations")
LABEL_DIR = os.path.join(PROJECT_ROOT, "fracture_labels")
GENERATED_DATA_DIR = os.path.join(PROJECT_ROOT, "spine_data")
OUTPUT_CSV_NAME = "slice_annotations.csv"

NUM_WORKERS = None

# 出力形状（generate_datasetと同じ）
OUTPUT_SHAPE = (128, 128, 64)

# マージン設定（generate_datasetと同じ・倍率）
MARGIN_XY_RATIO = 1.2
MARGIN_Z_RATIO = 1.0

VERTEBRAE = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
FRACTURE_OVERLAP_THRESHOLD = 0.10
# =================================================================


class SliceLabelGeneratorFast:
    def __init__(self, seg_dir, label_dir, data_dir):
        self.seg_dir = seg_dir
        self.label_dir = label_dir
        self.data_dir = data_dir

    def _get_largest_component(self, mask_data):
        labeled, n = nd_label(mask_data > 0)
        if n <= 1:
            return mask_data
        sizes = [(labeled == i).sum() for i in range(1, n + 1)]
        largest = np.argmax(sizes) + 1
        return (labeled == largest).astype(mask_data.dtype)

    def _compute_physical_centroid(self, affine, mask_data):
        c = center_of_mass(mask_data)
        if np.any(np.isnan(c)):
            return None
        return (affine @ np.array([*c, 1.0]))[:3]

    def _compute_rotation_matrix(self, vertebra_id, centroids):
        if vertebra_id in ["C1", "C2"] and centroids.get("C3") is not None:
            return self._compute_rotation_matrix("C3", centroids)
        idx = int(vertebra_id[1])
        upper, lower = ("C6", "C7") if idx == 7 else (f"C{idx}", f"C{idx + 1}")
        if centroids.get(upper) is None or centroids.get(lower) is None:
            return None
        vec = centroids[upper] - centroids[lower]
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return None
        R, _ = Rotation.align_vectors([np.array([0, 0, 1])], [vec / norm])
        return R.as_matrix()

    def _compute_bounding_box_physical(self, mask_img, mask_data, R_phys):
        """回転後空間でのBounding Box計算"""
        indices = np.argwhere(mask_data > 0)
        if len(indices) == 0:
            return None, None, None

        N = len(indices)
        indices_homo = np.hstack([indices, np.ones((N, 1))])
        coords_phys = (mask_img.affine @ indices_homo.T).T[:, :3]

        centroid_phys = np.mean(coords_phys, axis=0)
        coords_centered = coords_phys - centroid_phys
        coords_rotated = coords_centered @ R_phys

        bbox_min = np.min(coords_rotated, axis=0)
        bbox_max = np.max(coords_rotated, axis=0)

        return bbox_min, bbox_max, centroid_phys

    def _extract_and_resize_mask(self, data, affine, centroid_phys, R_phys, 
                                  bbox_min, bbox_max, order=0):
        """Bounding Box基準で切り出し、固定サイズに直接変換"""
        # BBoxサイズ計算
        bbox_size = bbox_max - bbox_min
        
        # マージン追加（倍率方式）
        roi_size_mm = np.array([
            bbox_size[0] * MARGIN_XY_RATIO,
            bbox_size[1] * MARGIN_XY_RATIO,
            bbox_size[2] * MARGIN_Z_RATIO
        ])

        # 変換行列構築（重心を中心として切り出し）
        # 出力空間の中心 -> 原点（ボクセル単位）
        out_center = np.array(OUTPUT_SHAPE) / 2.0
        T_center = np.eye(4)
        T_center[:3, 3] = -out_center

        # スケーリング: 出力ボクセル -> mm
        S = np.eye(4)
        S[0, 0] = roi_size_mm[0] / OUTPUT_SHAPE[0]
        S[1, 1] = roi_size_mm[1] / OUTPUT_SHAPE[1]
        S[2, 2] = roi_size_mm[2] / OUTPUT_SHAPE[2]

        # 回転の逆行列
        R_inv = np.eye(4)
        R_inv[:3, :3] = R_phys.T

        # 重心への移動
        T_loc = np.eye(4)
        T_loc[:3, 3] = centroid_phys

        # 元画像座標への変換
        affine_inv = np.linalg.inv(affine)

        # 総合変換行列
        M_total = affine_inv @ T_loc @ R_inv @ S @ T_center

        matrix = M_total[:3, :3]
        offset = M_total[:3, 3]

        # 直接OUTPUT_SHAPEに切り出し
        data_resized = affine_transform(data, matrix, offset=offset,
                                        output_shape=OUTPUT_SHAPE, order=order, cval=0)

        return data_resized

    def process_single_patient(self, sample_id):
        """1人の患者データを処理"""
        slice_records = []

        try:
            # 骨折ラベル読み込み
            fracture_path = os.path.join(self.label_dir, f"{sample_id}.nii.gz")
            if os.path.exists(fracture_path):
                f_img = nib.load(fracture_path)
                f_full_data = f_img.get_fdata()
                affine_ref = f_img.affine
            else:
                first_seg = os.path.join(self.seg_dir, sample_id, "vertebrae_C1.nii.gz")
                if os.path.exists(first_seg):
                    f_img = nib.load(first_seg)
                    f_full_data = np.zeros(f_img.shape)
                    affine_ref = f_img.affine
                else:
                    return []

            # 全椎骨の重心計算
            centroids = {}
            masks_cache = {}
            
            for vid in VERTEBRAE:
                v_path = os.path.join(self.seg_dir, sample_id, f"vertebrae_{vid}.nii.gz")
                if not os.path.exists(v_path):
                    centroids[vid] = None
                    continue
                v_img = nib.load(v_path)
                v_data = v_img.get_fdata()
                if np.sum(v_data) < 10:
                    centroids[vid] = None
                    continue

                v_data = self._get_largest_component(v_data)
                centroids[vid] = self._compute_physical_centroid(v_img.affine, v_data)
                masks_cache[vid] = (v_img, v_data)

            # 各椎骨の処理
            for vid in VERTEBRAE:
                npy_rel_path = f"{sample_id}/{vid}.npy"
                npy_full_path = os.path.join(self.data_dir, npy_rel_path)
                if not os.path.exists(npy_full_path):
                    continue

                if centroids.get(vid) is None:
                    continue
                if vid not in masks_cache:
                    continue

                R = self._compute_rotation_matrix(vid, centroids)
                if R is None:
                    continue

                v_img, v_data = masks_cache[vid]

                # Bounding Box計算
                bbox_min, bbox_max, centroid_phys = self._compute_bounding_box_physical(
                    v_img, v_data, R
                )
                if bbox_min is None:
                    continue

                # マスクと骨折ラベルを切り出し＆リサイズ
                v_resized = self._extract_and_resize_mask(
                    v_data, v_img.affine, centroid_phys, R, bbox_min, bbox_max, order=0
                )
                f_resized = self._extract_and_resize_mask(
                    f_full_data, affine_ref, centroid_phys, R, bbox_min, bbox_max, order=0
                )

                # スライスごとのラベル判定
                for z in range(OUTPUT_SHAPE[2]):
                    v_slice = v_resized[:, :, z] > 0.5

                    label = 0
                    if np.sum(v_slice) >= 10:
                        f_slice = f_resized[:, :, z] > 0.5
                        overlap = np.logical_and(v_slice, f_slice)
                        ratio = np.sum(overlap) / np.sum(v_slice)
                        if ratio > FRACTURE_OVERLAP_THRESHOLD:
                            label = 1

                    slice_records.append({
                        "sample_id": sample_id,
                        "vertebra": vid,
                        "npy_path": npy_full_path,
                        "slice_index": z,
                        "label": label
                    })

        except Exception as e:
            print(f"\nError in {sample_id}: {e}")
            traceback.print_exc()

        return slice_records


def process_wrapper(args):
    """並列処理用ラッパー"""
    generator, sample_id = args
    return generator.process_single_patient(sample_id)


def main():
    if not os.path.exists(CSV_FILE):
        print("CSVファイルがありません")
        return
    df = pd.read_csv(CSV_FILE)
    target_ids = df[df['Exclude'] != True]['ID'].tolist()

    if not os.path.exists(GENERATED_DATA_DIR):
        print(f"Data directory not found: {GENERATED_DATA_DIR}")
        return

    print(f"Target Samples: {len(target_ids)}")
    print(f"Output shape: {OUTPUT_SHAPE}")
    print(f"Margin XY: {MARGIN_XY_RATIO}x, Margin Z: {MARGIN_Z_RATIO}x")
    print(f"Processing with {NUM_WORKERS if NUM_WORKERS else 'ALL'} CPU cores...")

    gen = SliceLabelGeneratorFast(SEG_DIR, LABEL_DIR, GENERATED_DATA_DIR)

    all_records = []
    args_list = [(gen, str(pid)) for pid in target_ids]

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(executor.map(process_wrapper, args_list), total=len(target_ids)))

    for res in results:
        all_records.extend(res)

    out_df = pd.DataFrame(all_records)
    out_path = os.path.join(GENERATED_DATA_DIR, OUTPUT_CSV_NAME)
    out_df.to_csv(out_path, index=False)

    print("\n" + "=" * 50)
    print(f"Completed! Labels saved to: {out_path}")
    print(f"Total slices: {len(out_df)}")
    print(f"Fracture slices: {out_df['label'].sum()}")
    print("=" * 50)


if __name__ == "__main__":
    main()