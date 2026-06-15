import pandas as pd
import numpy as np

# --- 設定 ---
INPUT_CSV = "input_list.csv"       # 入力ファイル
OUTPUT_CSV = "nifti_list.csv"     # 出力ファイル

def fix_coordinates_from_csv():
    # 1. CSV読み込み
    df = pd.read_csv(INPUT_CSV)
    
    # 列の確認 (TotalSlicesがないと計算できません)
    if 'TotalSlices' not in df.columns:
        print("エラー: CSVに 'TotalSlices' (総スライス数) の列がありません。")
        print("現在の列名:", df.columns)
        return

    # 2. 数値型への変換 (エラー回避)
    cols_to_fix = ['FractureStart', 'FractureEnd', 'TotalSlices']
    for col in cols_to_fix:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"全 {len(df)} 件の座標変換を行います...")

    # 3. 計算処理
    results = []
    
    for index, row in df.iterrows():
        # 値の取得
        start = row['FractureStart']
        end = row['FractureEnd']
        total = row['TotalSlices']
        note = str(row['Note'])
        
        # 値が空ならスキップ
        if pd.isna(start) or pd.isna(end) or pd.isna(total):
            results.append({
                'Z_Start': start, 'Z_End': end, 'Is_Flipped': False
            })
            continue

        # --- 判定 ---
        if "Top to Bottom" in note:
            # 反転計算: (N-1) - x
            new_start = (total - 1) - start
            new_end = (total - 1) - end
            
            # StartとEndの大小関係を整理 (小さい方をStartに)
            final_start = min(new_start, new_end)
            final_end = max(new_start, new_end)
            is_flipped = True
        else:
            # そのまま
            final_start = start
            final_end = end
            is_flipped = False
            
        results.append({
            'Z_Start': final_start,
            'Z_End': final_end,
            'Is_Flipped': is_flipped
        })

    # 4. 結果を元のデータフレームに結合して保存
    result_df = pd.DataFrame(results)
    final_df = pd.concat([df, result_df], axis=1)
    
    # 見やすいように列を並べ替え（任意）
    cols = ['ID', 'Z_Start', 'Z_End', 'Is_Flipped', 'TotalSlices', 'Note', 'FractureStart', 'FractureEnd']
    # 存在する列だけ選ぶ
    cols = [c for c in cols if c in final_df.columns]
    
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"完了しました: {OUTPUT_CSV}")
    print(final_df[cols].head())

if __name__ == "__main__":
    fix_coordinates_from_csv()