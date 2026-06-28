"""椎体レベル別・線別の角度統計を収集するスクリプト。

seg_ct.npy (5,224,224) の中央スライス (slice 2) を使い、
line_1〜line_4 の angle_deg を全データで収集する。
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from .constants import (
    CENTER_CHANNEL,
    DEFAULT_CKPT_DIR,
    FRACTURE_DATASET_DIR,
    LINE_KEYS,
    PROJECT_ROOT,
    TRAINING_DATASET_DIR,
    VERTEBRA_LEVELS,
)
from .exclusions import load_excluded_levels, load_excluded_studies
from .inference import predict_single_slice
from .model_io import compute_avg_line_lengths, load_models


def angle_stats(values: list[float]) -> dict[str, float]:
    """角度リストの統計量を計算する。"""
    if not values:
        return {}
    arr = np.array(values)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    outlier_mask = (arr < q1 - 1.5 * iqr) | (arr > q3 + 1.5 * iqr)
    return {
        "n": len(arr),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(q1),
        "median": float(np.median(arr)),
        "p75": float(q3),
        "p95": float(np.percentile(arr, 95)),
        "iqr": float(iqr),
        "outlier_n": int(outlier_mask.sum()),
        "outlier_pct": float(outlier_mask.mean() * 100),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="椎体レベル別の線角度統計収集")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument(
        "--fracture-dataset-dir", type=Path, default=FRACTURE_DATASET_DIR
    )
    parser.add_argument(
        "--training-dataset-dir", type=Path, default=TRAINING_DATASET_DIR
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Unet" / "outputs" / "angle_stats",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}", flush=True)

    models = load_models(args.checkpoint_dir, args.n_folds, device)
    print(f"[INFO] {len(models)} fold モデル読み込み完了", flush=True)
    avg_lengths = compute_avg_line_lengths(args.training_dataset_dir)

    excluded_studies = load_excluded_studies()
    excluded_levels = load_excluded_levels()

    all_study_ids = sorted(
        d.name
        for d in args.fracture_dataset_dir.iterdir()
        if d.is_dir() and d.name not in excluded_studies
    )
    print(f"[INFO] 対象スタディ: {len(all_study_ids)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv_path = args.output_dir / "angles_raw.csv"

    angles_by_level_line: dict[str, dict[str, list[float]]] = {
        level: {key: [] for key in LINE_KEYS} for level in VERTEBRA_LEVELS
    }
    none_counts: dict[str, dict[str, int]] = {
        level: {key: 0 for key in LINE_KEYS} for level in VERTEBRA_LEVELS
    }

    started = time.monotonic()
    total = 0

    with raw_csv_path.open("w", encoding="utf-8", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["study_uid", "vertebra", *LINE_KEYS])

        for idx, study_id in enumerate(all_study_ids):
            excl = excluded_levels.get(study_id, set())
            for level in VERTEBRA_LEVELS:
                if level in excl:
                    continue
                level_dir = args.fracture_dataset_dir / study_id / level
                seg_ct_path = level_dir / "seg_ct.npy"
                seg_mask_path = level_dir / "seg_vertebra_mask.npy"
                if not seg_ct_path.exists() or not seg_mask_path.exists():
                    continue

                seg_ct = np.load(seg_ct_path)
                seg_mask = np.load(seg_mask_path)
                ct_f = seg_ct[CENTER_CHANNEL].astype(np.float32) / 255.0
                mask_f = seg_mask[CENTER_CHANNEL].astype(np.float32)

                _, lines = predict_single_slice(
                    models,
                    ct_f,
                    mask_f,
                    level,
                    device,
                    avg_lengths,
                )

                row: list[str] = [study_id, level]
                for key in LINE_KEYS:
                    pred = lines.get(key)
                    if pred is not None:
                        row.append(f"{pred.angle_deg:.4f}")
                        angles_by_level_line[level][key].append(pred.angle_deg)
                    else:
                        row.append("")
                        none_counts[level][key] += 1
                writer.writerow(row)
                total += 1

            if (idx + 1) % 200 == 0:
                elapsed = time.monotonic() - started
                rate = total / elapsed
                print(
                    f"  [{idx + 1}/{len(all_study_ids)}] {total} levels, "
                    f"{rate:.1f} lv/s",
                    flush=True,
                )

    elapsed_total = time.monotonic() - started
    print(f"\n[INFO] 全処理完了: {total} levels, {elapsed_total:.0f}s", flush=True)

    # ---- レポート生成 ----
    report = ["# 椎体レベル別 線角度統計レポート\n"]
    report.append(f"処理時間: {elapsed_total:.0f}s, 処理レベル数: {total}\n")

    for line_key in LINE_KEYS:
        report.append(f"\n## {line_key}\n")
        report.append(
            "| Level | N | None | Mean | Std | P5 | P25 | Median | P75 | P95 | "
            "Outlier N | Outlier % |"
        )
        report.append(
            "|-------|---|------|------|-----|----|-----|--------|-----|-----|"
            "-----------|-----------|"
        )
        for level in VERTEBRA_LEVELS:
            vals = angles_by_level_line[level][line_key]
            none_n = none_counts[level][line_key]
            s = angle_stats(vals)
            if not s:
                report.append(
                    f"| {level} | 0 | {none_n} | — | — | — | — | — | — | — | — | — |"
                )
                continue
            report.append(
                f"| {level} | {s['n']} | {none_n} | "
                f"{s['mean']:.1f} | {s['std']:.1f} | "
                f"{s['p5']:.1f} | {s['p25']:.1f} | {s['median']:.1f} | "
                f"{s['p75']:.1f} | {s['p95']:.1f} | "
                f"{s['outlier_n']} | {s['outlier_pct']:.1f}% |"
            )

    report_text = "\n".join(report) + "\n"
    report_path = args.output_dir / "angle_stats_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text, flush=True)
    print(f"\n[SAVED] {report_path.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"[SAVED] {raw_csv_path.relative_to(PROJECT_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
