import os
import glob
import subprocess
import pandas as pd
import pydicom
import nibabel as nib
import numpy as np
import sys

# --- 設定 ---
INPUT_CSV = "input_list.csv"        # 作成したCSVファイル名
OUTPUT_NIFTI_DIR = "./nifti_out"    # 生成されるNIfTIの保存先
OUTPUT_CSV = "fracture_list.csv"    # GPUサーバーへ送る用の完成CSV

def get_dicom_z_coords(dicom_dir):
    """
    DICOMディレクトリ内のファイルを読み込み、(InstanceNumber, Z座標) のリストを返す。
    InstanceNumber順（医師が見ている順序）にソートする。
    Enhanced Multi-frame DICOMと従来型DICOMの両方に対応。
    """
    dcm_data = []
    files = glob.glob(os.path.join(dicom_dir, "*"))

    for f in files:
        # ディレクトリなら無視
        if os.path.isdir(f):
            continue

        try:
            # ヘッダのみ読み込み（高速化）
            dcm = pydicom.dcmread(f, stop_before_pixels=True)

            # Enhanced Multi-frame DICOMの場合
            if "PerFrameFunctionalGroupsSequence" in dcm:
                per_frame_seq = dcm.PerFrameFunctionalGroupsSequence
                for frame_idx, frame in enumerate(per_frame_seq):
                    # フレーム番号は1から始まる（医師が見る順序）
                    inst_no = frame_idx + 1

                    # PlanePositionSequenceからZ座標を取得
                    if hasattr(frame, 'PlanePositionSequence') and len(frame.PlanePositionSequence) > 0:
                        z_coord = float(frame.PlanePositionSequence[0].ImagePositionPatient[2])
                        dcm_data.append((inst_no, z_coord))

            # 従来型DICOMの場合
            elif "InstanceNumber" in dcm and "ImagePositionPatient" in dcm:
                inst_no = int(dcm.InstanceNumber)
                z_coord = float(dcm.ImagePositionPatient[2])
                dcm_data.append((inst_no, z_coord))
        except Exception:
            # DICOMでないファイルは無視
            continue

    # Instance Number順にソート (1, 2, 3...)
    dcm_data.sort(key=lambda x: x[0])
    return dcm_data

def map_dicom_range_to_nifti(dcm_data, nifti_path, start_dcm, end_dcm):
    """
    DICOMの開始・終了番号に対応するZ座標を探し、NIfTIのZインデックスに変換する
    """
    if not dcm_data:
        return None, None

    # 指定範囲内のZ座標を抽出
    target_z = [z for i, z in dcm_data if start_dcm <= i <= end_dcm]
    
    if not target_z:
        print(f"    [Warning] Range {start_dcm}-{end_dcm} not found in DICOM instances.")
        return None, None

    # NIfTIファイルを読み込む
    try:
        nii = nib.load(nifti_path)
    except FileNotFoundError:
        print(f"    [Error] NIfTI file not found: {nifti_path}")
        return None, None

    # アフィン変換行列の逆行列を計算 (World -> Voxel)
    inv_affine = np.linalg.inv(nii.affine)
    
    nifti_indices = []
    for z in target_z:
        # Z座標以外は0（中心）として計算
        point_world = np.array([0, 0, z, 1.0])
        point_voxel = inv_affine.dot(point_world)
        z_idx = int(round(point_voxel[2]))
        nifti_indices.append(z_idx)
        
    if not nifti_indices:
        return None, None

    # NIfTI上での最小・最大を返す（データの向きによってstart/endが逆転するためmin/maxをとる）
    # 画像範囲外にはみ出さないようクリップ
    max_slice = nii.shape[2] - 1
    final_start = max(0, min(nifti_indices))
    final_end = min(max_slice, max(nifti_indices))
    
    return final_start, final_end

def main():
    # CSV読み込み
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    os.makedirs(OUTPUT_NIFTI_DIR, exist_ok=True)
    
    processed_results = []

    print(f"Start processing {len(df)} rows...")
    print("-" * 50)

    for idx, row in df.iterrows():
        sample_id = str(row['ID'])
        dicom_path = str(row['DicomPath'])
        exclude_flag = row['Exclude']
        
        # 1. 除外フラグのチェック
        if exclude_flag == True or str(exclude_flag).lower() == 'true':
            print(f"Skip: {sample_id} (Exclude=True)")
            continue

        # 2. 数値チェック (Start/Endが数値でない、またはNaNの場合はスキップ)
        try:
            # 文字列が含まれている場合のエラー回避
            start_val = pd.to_numeric(row['FractureStart'], errors='coerce')
            end_val = pd.to_numeric(row['FractureEnd'], errors='coerce')
            
            if pd.isna(start_val) or pd.isna(end_val):
                print(f"Skip: {sample_id} (Invalid slice numbers)")
                continue
                
            start_slice = int(start_val)
            end_slice = int(end_val)
        except ValueError:
            print(f"Skip: {sample_id} (ValueError in slice numbers)")
            continue

        print(f"Processing: {sample_id} ...")

        # 3. dcm2niix による変換
        # 出力ファイル名の期待値
        expected_nifti_filename = f"{sample_id}.nii.gz"
        expected_nifti_path = os.path.join(OUTPUT_NIFTI_DIR, expected_nifti_filename)
        
        # dcm2niix実行 (既にファイルがあっても上書き更新する場合は変換を実行)
        # -z y: Gzip圧縮, -f: ファイル名指定, -o: 出力先
        # 注意: dcm2niixはファイル名に自動でサフィックスをつけることがあるため、生成後に確認が必要
        cmd = [
            "dcm2niix", 
            "-z", "y", 
            "-f", sample_id, 
            "-o", OUTPUT_NIFTI_DIR, 
            dicom_path
        ]
        
        # ログを抑制して実行
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # dcm2niixは "sample1_Eq_1.nii.gz" のように名前を変えることがあるため検索する
        generated_files = glob.glob(os.path.join(OUTPUT_NIFTI_DIR, f"{sample_id}*.nii.gz"))
        
        if not generated_files:
            print(f"    [Error] dcm2niix failed to create file for {sample_id}")
            continue
            
        # 最も適切なファイルを選択（通常は一番短い名前のもの、あるいはJSONがあるもの）
        # ここではシンプルに見つかった最初のファイルを使用
        actual_nifti_path = generated_files[0]
        actual_nifti_filename = os.path.basename(actual_nifti_path)

        # 4. 座標マッピング計算
        dcm_data = get_dicom_z_coords(dicom_path)
        if not dcm_data:
            print("    [Error] No valid DICOM files found.")
            continue
            
        n_start, n_end = map_dicom_range_to_nifti(dcm_data, actual_nifti_path, start_slice, end_slice)
        
        if n_start is not None:
            print(f"    Mapped: DICOM[{start_slice}-{end_slice}] -> NIfTI[{n_start}-{n_end}]")
            
            processed_results.append({
                "ID": sample_id,
                "NiftiFileName": actual_nifti_filename,
                "Original_Start": start_slice,
                "Original_End": end_slice,
                "Target_Start_Z": n_start,  # これが正解ラベルの開始
                "Target_End_Z": n_end,      # これが正解ラベルの終了
                "Note": row['Note']
            })
        else:
            print("    [Error] Mapping failed (slice not found in Z-range).")

    # 結果をCSVに保存
    if processed_results:
        result_df = pd.DataFrame(processed_results)
        result_df.to_csv(OUTPUT_CSV, index=False)
        print("-" * 50)
        print(f"Conversion Complete! Saved list to: {OUTPUT_CSV}")
        print(f"Please transfer '{OUTPUT_NIFTI_DIR}' and '{OUTPUT_CSV}' to your GPU server.")
    else:
        print("No data processed.")

if __name__ == "__main__":
    main()