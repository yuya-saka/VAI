"""
V字型ポリラインと角度誤差の相関分析

V字度の定義:
  ポリライン上の全点をPCA主軸に射影し、射影値の「単調でなさ」を測る。
  - 単調増加/減少なら直線 → V字度 = 0
  - 折り返しがあれば → V字度 > 0（折り返し量の最大値をV字度とする）
"""

import glob
import json
import re
from pathlib import Path

import matplotlib_fontja
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

# --- パス設定 ---
REPO_ROOT = Path("/mnt/nfs1/home/yamamoto-hiroto/research/VAI")
PRED_PATTERN = str(
    REPO_ROOT
    / "Unet/outputs/multitask_v4(aug修正)/sig3.5-alpha0.07/vis/fold*/test_lines/*_PRED_lines.json"
)
DATASET_ROOT = REPO_ROOT / "data" / "dataset"
OUTPUT_DIR = REPO_ROOT / "Unet/debug"
OUTPUT_PNG = OUTPUT_DIR / "vshape_correlation.png"
OUTPUT_JSON = OUTPUT_DIR / "vshape_correlation.json"


def compute_vshape_score(pts_xy: list) -> float:
    """
    ポリライン点列のV字度を計算

    手順:
    1. PCAで主軸方向を求める（np.linalg.eigh）
    2. 全点を主軸に射影
    3. 射影値の列で「折り返し量」= max(0, -(前進量の最小値)) を計算
       具体的には diff(projections) の累積和の最小値を使って
       「最大どれだけ後退したか」を測る

    戻り値: float（0以上、単位はピクセル）
    """
    pts = np.array(pts_xy, dtype=np.float64)
    if len(pts) < 3:
        return 0.0

    c = pts.mean(axis=0)
    xc = pts - c
    cov = xc.T @ xc
    evals, evecs = np.linalg.eigh(cov)
    principal = evecs[:, np.argmax(evals)]

    projections = xc @ principal  # 主軸への射影
    diffs = np.diff(projections)

    # 累積後退量: 前進方向と逆に動いた最大量
    cumsum = np.cumsum(diffs)
    # 正方向の場合: 最大値からの最大下落
    backtrack_pos = float(np.maximum(0, cumsum.max() - cumsum).max()) if cumsum.max() > 0 else 0.0
    # 負方向の場合: 最小値からの最大上昇
    backtrack_neg = float(np.maximum(0, cumsum - cumsum.min()).max()) if cumsum.min() < 0 else 0.0

    return max(backtrack_pos, backtrack_neg)


def parse_filename(fname: str) -> tuple[str, str, int] | None:
    """
    ファイル名から (sample, vertebra, slice_idx) をパース
    例: sample1_C3_slice060_PRED_lines.json -> ("sample1", "C3", 60)
    """
    pattern = r"^(sample\d+)_([A-Z]\d+)_slice(\d+)_PRED_lines\.json$"
    m = re.match(pattern, fname)
    if m is None:
        return None
    sample = m.group(1)
    vertebra = m.group(2)
    slice_idx = int(m.group(3))
    return sample, vertebra, slice_idx


def load_gt_polylines(sample: str, vertebra: str) -> dict[str, dict[str, list]]:
    """
    GTポリラインを読み込む
    戻り値: {slice_idx_str: {line_key: [[x,y],...]}}
    """
    lines_path = DATASET_ROOT / sample / vertebra / "lines.json"
    if not lines_path.exists():
        return {}
    with open(lines_path) as f:
        return json.load(f)


def main() -> None:
    # --- 全PRED JSONを収集 ---
    pred_files = sorted(glob.glob(PRED_PATTERN))
    print(f"Found {len(pred_files)} PRED JSON files")

    # GTキャッシュ: (sample, vertebra) -> gt_data
    gt_cache: dict[tuple[str, str], dict] = {}

    records: list[dict] = []
    skipped = 0

    for pred_path in pred_files:
        fname = Path(pred_path).name
        parsed = parse_filename(fname)
        if parsed is None:
            skipped += 1
            continue
        sample, vertebra, slice_idx = parsed

        # PRED JSON読み込み
        with open(pred_path) as f:
            pred_data = json.load(f)
        metrics = pred_data.get("metrics", {})

        # GT読み込み（キャッシュ利用）
        key = (sample, vertebra)
        if key not in gt_cache:
            gt_cache[key] = load_gt_polylines(sample, vertebra)
        gt_data = gt_cache[key]

        slice_key = str(slice_idx)
        gt_slice = gt_data.get(slice_key, {})

        # 各ラインについてV字度と角度誤差を収集
        for line_key in ["line_1", "line_2", "line_3", "line_4"]:
            line_metrics = metrics.get(line_key)
            if line_metrics is None:
                continue
            angle_error = line_metrics.get("angle_error_deg")
            if angle_error is None:
                continue

            gt_pts = gt_slice.get(line_key)
            if gt_pts is None or len(gt_pts) < 2:
                continue

            vshape = compute_vshape_score(gt_pts)
            records.append(
                {
                    "sample": sample,
                    "vertebra": vertebra,
                    "slice": slice_idx,
                    "line": line_key,
                    "vshape_score": vshape,
                    "angle_error_deg": angle_error,
                }
            )

    print(f"Collected {len(records)} (sample, line) pairs (skipped {skipped} files)")

    if len(records) == 0:
        print("No data collected. Exiting.")
        return

    vscores = np.array([r["vshape_score"] for r in records])
    aerrors = np.array([r["angle_error_deg"] for r in records])

    # --- 相関係数 ---
    pearson_r, pearson_p = pearsonr(vscores, aerrors)
    spearman_r, spearman_p = spearmanr(vscores, aerrors)
    print(f"Pearson  r={pearson_r:.4f}  p={pearson_p:.4e}")
    print(f"Spearman r={spearman_r:.4f}  p={spearman_p:.4e}")

    # --- 高V字度 vs 低V字度の比較 ---
    threshold = 5.0
    high_mask = vscores > threshold
    low_mask = ~high_mask
    n_high = int(high_mask.sum())
    n_low = int(low_mask.sum())
    mean_err_high = float(aerrors[high_mask].mean()) if n_high > 0 else float("nan")
    mean_err_low = float(aerrors[low_mask].mean()) if n_low > 0 else float("nan")
    print(f"V字度>{threshold}px: n={n_high}, mean_angle_error={mean_err_high:.3f}deg")
    print(f"V字度<={threshold}px: n={n_low}, mean_angle_error={mean_err_low:.3f}deg")

    # --- 上位20件 ---
    sorted_records = sorted(records, key=lambda r: r["vshape_score"], reverse=True)
    top20 = sorted_records[:20]
    print("\nTop 20 highest V字度:")
    for i, r in enumerate(top20):
        print(
            f"  {i+1:2d}. {r['sample']} {r['vertebra']} slice{r['slice']:03d} {r['line']}"
            f"  vshape={r['vshape_score']:.2f}px  angle_error={r['angle_error_deg']:.2f}deg"
        )

    # --- 散布図 ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(vscores, aerrors, alpha=0.3, s=10, color="steelblue", label="各ライン")
    ax.set_xlabel("V字度 (px)")
    ax.set_ylabel("角度誤差 (deg)")
    ax.set_title(
        f"V字度 vs 角度誤差\n"
        f"Pearson r={pearson_r:.3f}, Spearman r={spearman_r:.3f}  (n={len(records)})"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150)
    plt.close(fig)
    print(f"\n散布図を保存: {OUTPUT_PNG}")

    # --- 数値結果をJSON保存 ---
    result = {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "n_samples": len(records),
        "n_high_vshape": n_high,
        "mean_angle_error_high_vshape": mean_err_high,
        "mean_angle_error_low_vshape": mean_err_low,
        "vshape_threshold_px": threshold,
        "top20_worst": [
            {
                "sample": r["sample"],
                "vertebra": r["vertebra"],
                "slice": r["slice"],
                "line": r["line"],
                "vshape_score": r["vshape_score"],
                "angle_error_deg": r["angle_error_deg"],
            }
            for r in top20
        ],
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"数値結果を保存: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
