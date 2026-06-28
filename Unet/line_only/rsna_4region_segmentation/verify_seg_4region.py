"""seg_ct.npy (5,224,224) を使った 4 領域分割の実現可能性検証。

各椎体レベル(C1-C7)ごとに 5 件サンプリングして可視化 +
全データで 4 領域生成の成功率・面積統計を出力する。
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from data_preprocessing.segmentation_dataset.generate_region_mask import (
    generate_region_mask,
)

from .constants import (
    CENTER_CHANNEL,
    DEFAULT_CKPT_DIR,
    FRACTURE_DATASET_DIR,
    IMAGE_SIZE,
    N_SEG_PLANES,
    PROJECT_ROOT,
    TRAINING_DATASET_DIR,
    VERTEBRA_LEVELS,
)
from .exclusions import load_excluded_levels, load_excluded_studies
from .inference import predict_single_slice
from .model_io import compute_avg_line_lengths, load_models
from .visualization import (
    add_title_bar,
    concat_with_separator,
    ct_to_bgr,
    draw_lines_on_image,
    lines_to_polylines,
    make_region_overlay,
)


def try_generate_region_mask(
    lines: dict[str, Any],
    mask_slice: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """4 領域マスクを安全に生成する。失敗時は None を返す。"""
    polylines = lines_to_polylines(lines)
    if polylines is None:
        return None, {"error": "missing lines"}
    vertebra_mask = (mask_slice > 0.5).astype(np.uint8)
    try:
        seg, debug = generate_region_mask(
            line_1=polylines["line_1"],
            line_2=polylines["line_2"],
            line_3=polylines["line_3"],
            line_4=polylines["line_4"],
            vertebra_mask=vertebra_mask,
        )
    except Exception as e:
        return None, {"error": str(e)}
    return seg, debug


def evaluate_region_quality(
    seg: np.ndarray | None,
    debug: dict[str, Any],
) -> dict[str, Any]:
    """4 領域分割の品質を評価する。"""
    if seg is None:
        return {
            "success": False,
            "n_regions": 0,
            "areas": {},
            "error": debug.get("error", "unknown"),
        }
    areas = debug.get("region_areas", {})
    n_regions = sum(1 for v in areas.values() if v > 0)
    total_fg = sum(areas.values())
    ratios = {k: v / total_fg if total_fg > 0 else 0.0 for k, v in areas.items()}
    return {
        "success": n_regions == 4,
        "n_regions": n_regions,
        "areas": areas,
        "ratios": ratios,
        "swap_detected": debug.get("swap_detected", False),
    }


# ---- 可視化 ----


def draw_single_panel(
    ct_slice: np.ndarray,
    mask_slice: np.ndarray,
    seg: np.ndarray | None,
    lines: dict[str, Any],
    slice_idx: int,
    alpha: float = 0.45,
) -> np.ndarray:
    """1 スライスの可視化パネル (H, 2W, 3)。左:CT+線、右:4 領域。"""
    left = ct_to_bgr(ct_slice)
    draw_lines_on_image(left, lines)
    cv2.putText(
        left,
        f"slice {slice_idx}",
        (4, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if seg is not None:
        label = np.argmax(seg, axis=0).astype(np.uint8)
        ct_u8 = (np.clip(ct_slice, 0, 1) * 255).astype(np.uint8)
        right = make_region_overlay(ct_u8, label, alpha)
    else:
        right = ct_to_bgr(ct_slice)
        cv2.putText(
            right,
            "FAILED",
            (IMAGE_SIZE // 2 - 40, IMAGE_SIZE // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return concat_with_separator([left, right], axis=1)


def visualize_5slice(
    seg_ct: np.ndarray,
    seg_mask: np.ndarray,
    all_segs: list[np.ndarray | None],
    all_lines: list[dict[str, Any]],
    all_quality: list[dict[str, Any]],
    study_uid: str,
    vertebra: str,
    save_path: Path,
) -> None:
    """5 スライス全てを縦に並べて 1 枚の PNG に保存する。"""
    panels = []
    for i in range(N_SEG_PLANES):
        ct_f = seg_ct[i].astype(np.float32) / 255.0
        mask_f = seg_mask[i].astype(np.float32)
        panels.append(draw_single_panel(ct_f, mask_f, all_segs[i], all_lines[i], i))

    canvas = concat_with_separator(panels, axis=0)
    n_ok = sum(1 for q in all_quality if q["success"])
    short_uid = study_uid.split(".")[-1]
    final = add_title_bar(
        canvas,
        f"{short_uid}  {vertebra}  ({n_ok}/5 slices OK)",
        height=30,
        font_scale=0.6,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), final)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="4 領域分割の実現可能性検証（seg_ct.npy 使用）"
    )
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
        default=PROJECT_ROOT / "Unet" / "outputs" / "seg_4region_verification",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--n-vis-per-level", type=int, default=5)
    parser.add_argument("--vis-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    models = load_models(args.checkpoint_dir, args.n_folds, device)
    print(f"[INFO] {len(models)} fold モデル読み込み完了")
    avg_lengths = compute_avg_line_lengths(args.training_dataset_dir)

    excluded_studies = load_excluded_studies()
    excluded_levels = load_excluded_levels()

    all_study_ids = sorted(
        d.name
        for d in args.fracture_dataset_dir.iterdir()
        if d.is_dir() and d.name not in excluded_studies
    )
    print(f"[INFO] 対象スタディ: {len(all_study_ids)}")

    # 椎体レベル別の候補 study 列挙
    level_candidates: dict[str, list[str]] = {level: [] for level in VERTEBRA_LEVELS}
    for study_id in all_study_ids:
        excl = excluded_levels.get(study_id, set())
        for level in VERTEBRA_LEVELS:
            if level in excl:
                continue
            if (args.fracture_dataset_dir / study_id / level / "seg_ct.npy").exists():
                level_candidates[level].append(study_id)

    rng = np.random.RandomState(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: 各レベル N 件の可視化 ----
    print("\n=== Phase 1: 各椎体レベルの可視化サンプル ===")
    level_vis_samples: dict[str, list[tuple[str, np.ndarray]]] = {
        lv: [] for lv in VERTEBRA_LEVELS
    }

    for level in VERTEBRA_LEVELS:
        candidates = level_candidates[level]
        n = min(args.n_vis_per_level, len(candidates))
        chosen_ids = [
            candidates[i]
            for i in sorted(rng.choice(len(candidates), size=n, replace=False))
        ]
        print(f"  {level}: {len(candidates)} candidates, visualizing {n}")

        for study_id in chosen_ids:
            level_dir = args.fracture_dataset_dir / study_id / level
            seg_ct = np.load(level_dir / "seg_ct.npy")
            seg_mask = np.load(level_dir / "seg_vertebra_mask.npy")

            all_segs, all_lines_list, all_quality = [], [], []
            for i in range(N_SEG_PLANES):
                ct_f = seg_ct[i].astype(np.float32) / 255.0
                mask_f = seg_mask[i].astype(np.float32)
                _, lines = predict_single_slice(
                    models, ct_f, mask_f, level, device, avg_lengths
                )
                seg, debug = try_generate_region_mask(lines, mask_f)
                all_segs.append(seg)
                all_lines_list.append(lines)
                all_quality.append(evaluate_region_quality(seg, debug))

            save_path = (
                args.output_dir / "per_level" / level / f"{study_id.split('.')[-1]}.png"
            )
            visualize_5slice(
                seg_ct,
                seg_mask,
                all_segs,
                all_lines_list,
                all_quality,
                study_id,
                level,
                save_path,
            )

            img = cv2.imread(str(save_path))
            if img is not None:
                level_vis_samples[level].append((study_id, img))

    # レベルごとのグリッド画像
    for level, samples in level_vis_samples.items():
        if not samples:
            continue
        cols = [s[1] for s in samples]
        canvas = concat_with_separator(cols, axis=1, sep_width=4, sep_value=40)
        grid = add_title_bar(
            canvas, f"{level}: {len(samples)} samples", height=35, font_scale=0.8
        )
        grid_path = args.output_dir / f"level_grid_{level}.png"
        cv2.imwrite(str(grid_path), grid)
        print(f"  [SAVED] {grid_path.relative_to(PROJECT_ROOT)}")

    if args.vis_only:
        print("\n[DONE] 可視化のみモード完了")
        return

    # ---- Phase 2: 全データ統計 ----
    print("\n=== Phase 2: 全データ統計 ===")
    stats_by_level: dict[str, dict[str, Any]] = {}
    all_records: list[dict[str, Any]] = []
    started = time.monotonic()
    total_processed = 0

    for level in VERTEBRA_LEVELS:
        level_stats: dict[str, Any] = {
            "total": 0,
            "success_all5": 0,
            "success_center": 0,
            "n_regions_hist": defaultdict(int),
            "swap_count": 0,
            "area_ratios": defaultdict(list),
        }

        for idx, study_id in enumerate(level_candidates[level]):
            level_dir = args.fracture_dataset_dir / study_id / level
            try:
                seg_ct = np.load(level_dir / "seg_ct.npy")
                seg_mask = np.load(level_dir / "seg_vertebra_mask.npy")
            except Exception:
                continue

            slice_results = []
            for i in range(N_SEG_PLANES):
                ct_f = seg_ct[i].astype(np.float32) / 255.0
                mask_f = seg_mask[i].astype(np.float32)
                _, lines = predict_single_slice(
                    models, ct_f, mask_f, level, device, avg_lengths
                )
                seg, debug = try_generate_region_mask(lines, mask_f)
                quality = evaluate_region_quality(seg, debug)
                quality["slice_idx"] = i
                slice_results.append(quality)

            level_stats["total"] += 1
            total_processed += 1

            center_ok = slice_results[CENTER_CHANNEL]["success"]
            all_ok = all(r["success"] for r in slice_results)
            if center_ok:
                level_stats["success_center"] += 1
            if all_ok:
                level_stats["success_all5"] += 1

            for r in slice_results:
                level_stats["n_regions_hist"][r["n_regions"]] += 1
                if r.get("swap_detected"):
                    level_stats["swap_count"] += 1
                if r["success"] and r.get("ratios"):
                    for region, ratio in r["ratios"].items():
                        level_stats["area_ratios"][region].append(ratio)

            all_records.append(
                {
                    "study_uid": study_id,
                    "vertebra": level,
                    "center_ok": center_ok,
                    "all5_ok": all_ok,
                    "n_regions": [r["n_regions"] for r in slice_results],
                }
            )

            if (idx + 1) % 200 == 0:
                elapsed = time.monotonic() - started
                print(
                    f"  {level} [{idx + 1}/{len(level_candidates[level])}] ({total_processed / elapsed:.1f} levels/s)"
                )

        stats_by_level[level] = level_stats
        t = level_stats["total"]
        if t > 0:
            print(
                f"  {level}: {t} total, "
                f"center_ok={level_stats['success_center']}/{t} ({level_stats['success_center'] / t * 100:.1f}%), "
                f"all5_ok={level_stats['success_all5']}/{t} ({level_stats['success_all5'] / t * 100:.1f}%)"
            )

    # ---- レポート出力 ----
    elapsed_total = time.monotonic() - started
    print(f"\n=== 総合結果 ({elapsed_total:.0f}s) ===")

    report_lines = [
        "# 4 領域分割 実現可能性検証レポート\n",
        f"処理時間: {elapsed_total:.0f}s\n",
    ]
    report_lines.append("## レベル別成功率\n")
    report_lines.append(
        "| Level | Total | Center OK | Center % | All5 OK | All5 % | Swap |"
    )
    report_lines.append(
        "|-------|-------|-----------|----------|---------|--------|------|"
    )

    grand_total = grand_center = grand_all5 = 0
    for level in VERTEBRA_LEVELS:
        s = stats_by_level[level]
        t = s["total"]
        grand_total += t
        grand_center += s["success_center"]
        grand_all5 += s["success_all5"]
        if t > 0:
            report_lines.append(
                f"| {level} | {t} | {s['success_center']} | {s['success_center'] / t * 100:.1f}% | "
                f"{s['success_all5']} | {s['success_all5'] / t * 100:.1f}% | {s['swap_count']} |"
            )
    if grand_total > 0:
        report_lines.append(
            f"| **Total** | {grand_total} | {grand_center} | {grand_center / grand_total * 100:.1f}% | "
            f"{grand_all5} | {grand_all5 / grand_total * 100:.1f}% | — |"
        )

    report_text = "\n".join(report_lines) + "\n"
    report_path = args.output_dir / "verification_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)

    csv_path = args.output_dir / "verification_details.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "study_uid",
                "vertebra",
                "center_ok",
                "all5_ok",
                *[f"n_regions_s{i}" for i in range(5)],
            ]
        )
        for r in all_records:
            writer.writerow(
                [
                    r["study_uid"],
                    r["vertebra"],
                    int(r["center_ok"]),
                    int(r["all5_ok"]),
                    *r["n_regions"],
                ]
            )
    print(f"[SAVED] {csv_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
