"""椎体レベル別・線別の角度統計を収集するスクリプト。

seg_ct.npy (5,224,224) の中央スライス(slice 2)を使い、
line_1〜line_4 の angle_deg を全データで収集する。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_here = Path(__file__).resolve().parent
_unet = _here.parent
_root = _unet.parent
if str(_unet) not in sys.path:
    sys.path.insert(0, str(_unet))
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from line_only.src.model import VERTEBRA_TO_IDX, TinyUNet  # noqa: E402
from line_only.utils.detection import (  # noqa: E402
    detect_line_moments,
    line_extent,
    moments_to_phi_rho,
)

PROJECT_ROOT = _root
RSNA_DATA_DIR = PROJECT_ROOT / "data" / "rsna_data"
FRACTURE_DATASET_DIR = RSNA_DATA_DIR / "fracture_dataset"
DEFAULT_CKPT_DIR = (
    PROJECT_ROOT / "Unet" / "outputs" / "line_20260616"
    / "sig4.0_ALL(CC適用)" / "checkpoints"
)
DEFAULT_TRAINING_DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
VERTEBRA_LEVELS = [f"C{i}" for i in range(1, 8)]
LINE_KEYS = tuple(f"line_{i}" for i in range(1, 5))
IMAGE_SIZE = 224
FALLBACK_LINE_LENGTH_PX = 80.0
CENTER_SLICE = 2


def load_models(ckpt_dir: Path, n_folds: int, device: torch.device) -> list[TinyUNet]:
    models: list[TinyUNet] = []
    for fold in range(n_folds):
        p = ckpt_dir / f"best_fold{fold}.pt"
        if not p.exists():
            continue
        ckpt = torch.load(p, map_location=device, weights_only=True)
        cfg = ckpt.get("cfg", {})
        mc = cfg.get("model", {})
        m = TinyUNet(
            in_ch=int(mc.get("in_channels", 2)),
            out_ch=int(mc.get("out_channels", 4)),
            feats=tuple(mc.get("features", [16, 32, 64, 128])),
            dropout=0.0,
            num_vertebra=int(mc.get("num_vertebra", 7))
            if mc.get("use_vertebra_conditioning", False) else 0,
        ).to(device)
        m.load_state_dict(ckpt["model"])
        m.eval()
        models.append(m)
    return models


def compute_average_training_line_lengths(dataset_dir: Path) -> dict[str, float]:
    sums = {key: 0.0 for key in LINE_KEYS}
    counts = {key: 0 for key in LINE_KEYS}
    for lines_path in sorted(dataset_dir.glob("sample*/C*/lines.json")):
        try:
            data = json.loads(lines_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for slice_data in data.values():
            if not isinstance(slice_data, dict):
                continue
            for key in LINE_KEYS:
                length = line_extent(slice_data.get(key))
                if length <= 1e-6:
                    continue
                sums[key] += length
                counts[key] += 1
    return {
        key: sums[key] / counts[key] if counts[key] > 0 else FALLBACK_LINE_LENGTH_PX
        for key in LINE_KEYS
    }


@torch.no_grad()
def predict_angles(
    models: list[TinyUNet],
    ct_slice: np.ndarray,
    mask_slice: np.ndarray,
    vertebra: str,
    device: torch.device,
    avg_lengths: dict[str, float],
) -> dict[str, float | None]:
    """中央スライスの4本線角度(deg)を返す。検出失敗はNone。"""
    x = torch.from_numpy(
        np.stack([ct_slice, mask_slice], axis=0)
    ).unsqueeze(0).to(device)
    vidx = torch.tensor(
        [VERTEBRA_TO_IDX.get(vertebra, 0)], device=device, dtype=torch.long,
    )
    hm_sum: torch.Tensor | None = None
    for model in models:
        out = torch.sigmoid(model(x, vidx))
        hm_sum = out if hm_sum is None else hm_sum + out
    hm = (hm_sum / len(models)).cpu().numpy()[0]

    threshold = {"mode": "adaptive", "min": 0.10, "peak_ratio": 0.4}
    angles: dict[str, float | None] = {}
    for ch in range(4):
        line_name = f"line_{ch + 1}"
        result = detect_line_moments(
            hm[ch],
            length_px=avg_lengths.get(line_name, FALLBACK_LINE_LENGTH_PX),
            extend_ratio=1.0,
            clip=False,
            threshold=threshold,
        )
        angles[line_name] = float(result["angle_deg"]) if result is not None else None
    return angles


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
    parser.add_argument("--fracture-dataset-dir", type=Path, default=FRACTURE_DATASET_DIR)
    parser.add_argument(
        "--training-dataset-dir", type=Path, default=DEFAULT_TRAINING_DATASET_DIR,
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "Unet" / "outputs" / "angle_stats",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}", flush=True)

    models = load_models(args.checkpoint_dir, args.n_folds, device)
    print(f"[INFO] {len(models)} fold モデル読み込み完了", flush=True)
    avg_lengths = compute_average_training_line_lengths(args.training_dataset_dir)

    excluded_studies_csv = RSNA_DATA_DIR / "excluded_studies.csv"
    excluded_levels_csv = RSNA_DATA_DIR / "excluded_levels.csv"
    excluded_studies: set[str] = set()
    excluded_levels_by_study: dict[str, set[str]] = defaultdict(set)
    if excluded_studies_csv.exists():
        with excluded_studies_csv.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                excluded_studies.add(row["study_uid"])
    if excluded_levels_csv.exists():
        with excluded_levels_csv.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                excluded_levels_by_study[row["study_uid"]].add(row["vertebra"])

    all_study_ids = sorted(
        d.name for d in args.fracture_dataset_dir.iterdir()
        if d.is_dir() and d.name not in excluded_studies
    )
    print(f"[INFO] 対象スタディ: {len(all_study_ids)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv_path = args.output_dir / "angles_raw.csv"

    # レベル別・線別の角度リスト
    angles_by_level_line: dict[str, dict[str, list[float]]] = {
        level: {key: [] for key in LINE_KEYS}
        for level in VERTEBRA_LEVELS
    }
    none_counts: dict[str, dict[str, int]] = {
        level: {key: 0 for key in LINE_KEYS}
        for level in VERTEBRA_LEVELS
    }

    started = time.monotonic()
    total = 0

    with raw_csv_path.open("w", encoding="utf-8", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["study_uid", "vertebra", "line_1", "line_2", "line_3", "line_4"])

        for idx, study_id in enumerate(all_study_ids):
            excl = excluded_levels_by_study.get(study_id, set())
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
                ct_f = seg_ct[CENTER_SLICE].astype(np.float32) / 255.0
                mask_f = seg_mask[CENTER_SLICE].astype(np.float32)

                angles = predict_angles(models, ct_f, mask_f, level, device, avg_lengths)

                row = [study_id, level]
                for key in LINE_KEYS:
                    val = angles.get(key)
                    row.append(f"{val:.4f}" if val is not None else "")
                    if val is not None:
                        angles_by_level_line[level][key].append(val)
                    else:
                        none_counts[level][key] += 1
                writer.writerow(row)
                total += 1

            if (idx + 1) % 200 == 0:
                elapsed = time.monotonic() - started
                rate = total / elapsed
                remaining = (len(all_study_ids) - idx - 1) * 7
                eta = remaining / rate if rate > 0 else 0
                print(
                    f"  [{idx + 1}/{len(all_study_ids)}] {total} levels, "
                    f"{rate:.1f} lv/s, ETA {eta/60:.1f}min",
                    flush=True,
                )

    elapsed_total = time.monotonic() - started
    print(f"\n[INFO] 全処理完了: {total} levels, {elapsed_total:.0f}s", flush=True)

    # ---- レポート生成 ----
    report = ["# 椎体レベル別 線角度統計レポート\n"]
    report.append(f"処理時間: {elapsed_total:.0f}s, 処理レベル数: {total}\n")
    report.append("外れ値基準: IQR法（Q1 - 1.5×IQR または Q3 + 1.5×IQR）\n")
    report.append("対象スライス: 中央スライス（slice 2 = 最大面積断面）\n")

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
                report.append(f"| {level} | 0 | {none_n} | — | — | — | — | — | — | — | — | — |")
                continue
            report.append(
                f"| {level} | {s['n']} | {none_n} | "
                f"{s['mean']:.1f} | {s['std']:.1f} | "
                f"{s['p5']:.1f} | {s['p25']:.1f} | {s['median']:.1f} | "
                f"{s['p75']:.1f} | {s['p95']:.1f} | "
                f"{s['outlier_n']} | {s['outlier_pct']:.1f}% |"
            )

    # 全レベル統合の要約
    report.append("\n## 全レベル統合\n")
    report.append(
        "| Line | N | None% | Mean | Std | Median | IQR | Outlier% |"
    )
    report.append("|------|---|-------|------|-----|--------|-----|----------|")
    for line_key in LINE_KEYS:
        all_vals: list[float] = []
        all_none = 0
        all_n = 0
        for level in VERTEBRA_LEVELS:
            all_vals.extend(angles_by_level_line[level][line_key])
            all_none += none_counts[level][line_key]
            all_n += len(angles_by_level_line[level][line_key]) + none_counts[level][line_key]
        s = angle_stats(all_vals)
        none_pct = all_none / all_n * 100 if all_n > 0 else 0
        report.append(
            f"| {line_key} | {s.get('n', 0)} | {none_pct:.1f}% | "
            f"{s.get('mean', 0):.1f} | {s.get('std', 0):.1f} | "
            f"{s.get('median', 0):.1f} | {s.get('iqr', 0):.1f} | "
            f"{s.get('outlier_pct', 0):.1f}% |"
        )

    # 簡易ヒストグラム（10°刻み）
    report.append("\n## 角度分布ヒストグラム（10°刻み、全レベル統合）\n")
    for line_key in LINE_KEYS:
        all_vals = []
        for level in VERTEBRA_LEVELS:
            all_vals.extend(angles_by_level_line[level][line_key])
        if not all_vals:
            continue
        arr = np.array(all_vals)
        bins = np.arange(-90, 100, 10)
        counts_hist, edges = np.histogram(arr, bins=bins)
        report.append(f"\n### {line_key}")
        report.append("```")
        max_count = counts_hist.max()
        bar_width = 40
        for i, (lo, count) in enumerate(zip(edges[:-1], counts_hist)):
            bar = "█" * int(count / max_count * bar_width)
            pct = count / len(arr) * 100
            report.append(f"{lo:+4.0f}〜{edges[i+1]:+4.0f}° | {bar:<{bar_width}} {count:5d} ({pct:.1f}%)")
        report.append("```")

    report_text = "\n".join(report) + "\n"
    report_path = args.output_dir / "angle_stats_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text, flush=True)
    print(f"\n[SAVED] {report_path.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"[SAVED] {raw_csv_path.relative_to(PROJECT_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
