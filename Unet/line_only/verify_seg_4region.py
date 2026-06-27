"""seg_ct.npy (5,224,224) を使った4領域分割の実現可能性検証。

各椎体レベル(C1-C7)ごとに5件サンプリングして可視化 +
全データで4領域生成の成功率・面積統計を出力する。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
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
from data_preprocessing.segmentation_dataset.generate_region_mask import (  # noqa: E402
    generate_region_mask,
)

PROJECT_ROOT = _root
RSNA_DATA_DIR = PROJECT_ROOT / "data" / "rsna_data"
FRACTURE_DATASET_DIR = RSNA_DATA_DIR / "fracture_dataset"
DEFAULT_CKPT_DIR = (
    PROJECT_ROOT / "Unet" / "outputs" / "line_20260616"
    / "sig4.0_ALL(CC適用)" / "checkpoints"
)
DEFAULT_TRAINING_DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
IMAGE_SIZE = 224
LINE_KEYS = tuple(f"line_{i}" for i in range(1, 5))
FALLBACK_LINE_LENGTH_PX = 80.0
VERTEBRA_LEVELS = [f"C{i}" for i in range(1, 8)]
N_SEG_SLICES = 5
CENTER_SLICE = 2

LINE_COLORS = {
    "line_1": (0, 220, 0),
    "line_2": (0, 60, 255),
    "line_3": (255, 100, 0),
    "line_4": (0, 220, 220),
}
REGION_COLORS_BGR = (
    (0, 0, 0),
    (0, 200, 0),
    (0, 0, 200),
    (200, 0, 0),
    (0, 200, 200),
)
REGION_NAMES = ("background", "body", "right_foramen", "left_foramen", "posterior")


def load_models(ckpt_dir: Path, n_folds: int, device: torch.device) -> list[TinyUNet]:
    """全foldのTinyUNetをロードする。"""
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
    """学習データの lines.json から line別平均長を計算する。"""
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
def predict_lines_single(
    models: list[TinyUNet],
    ct_slice: np.ndarray,
    mask_slice: np.ndarray,
    vertebra: str,
    device: torch.device,
    avg_lengths: dict[str, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    """1枚のCT+マスクから4本線を予測する。ヒートマップ(4,H,W)も返す。"""
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
    lines: dict[str, Any] = {}
    for ch in range(4):
        line_name = f"line_{ch + 1}"
        result = detect_line_moments(
            hm[ch],
            length_px=avg_lengths.get(line_name, FALLBACK_LINE_LENGTH_PX),
            extend_ratio=1.0,
            clip=False,
            threshold=threshold,
        )
        if result is None:
            lines[line_name] = None
            continue
        phi, rho = moments_to_phi_rho(result, IMAGE_SIZE)
        lines[line_name] = {
            "endpoints": result["endpoints"],
            "centroid": result["centroid"],
            "angle_deg": result["angle_deg"],
            "phi_rad": phi,
            "rho_normalized": rho,
        }
    return hm, lines


def try_generate_region_mask(
    lines: dict[str, Any],
    mask_slice: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """4領域マスクを安全に生成する。失敗時はNoneを返す。"""
    polylines: dict[str, list] = {}
    for key in LINE_KEYS:
        info = lines.get(key)
        if info is None or info.get("endpoints") is None:
            return None, {"error": f"missing {key}"}
        polylines[key] = [
            list(info["endpoints"][0]),
            list(info["endpoints"][1]),
        ]
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
    """4領域分割の品質を評価する。"""
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
    """1スライスの可視化パネル(H, 2W, 3)を生成。左:CT+線、右:4領域。"""
    H, W = IMAGE_SIZE, IMAGE_SIZE
    ct_u8 = (np.clip(ct_slice, 0, 1) * 255).astype(np.uint8)
    base = cv2.cvtColor(ct_u8, cv2.COLOR_GRAY2BGR)

    # 左パネル: CT + 線
    left = base.copy()
    for key in LINE_KEYS:
        info = lines.get(key)
        if info is None or info.get("endpoints") is None:
            continue
        color = LINE_COLORS[key]
        (x1, y1), (x2, y2) = info["endpoints"]
        cv2.line(left, (int(round(x1)), int(round(y1))),
                 (int(round(x2)), int(round(y2))), color, 2)
    cv2.putText(left, f"slice {slice_idx}", (4, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # 右パネル: 4領域
    if seg is not None:
        label = np.argmax(seg, axis=0).astype(np.uint8)
        ct_f = base.astype(np.float32)
        color_layer = np.zeros_like(ct_f)
        for lbl, bgr in enumerate(REGION_COLORS_BGR):
            color_layer[label == lbl] = bgr
        fg = label > 0
        right = ct_f.copy()
        right[fg] = ct_f[fg] * (1 - alpha) + color_layer[fg] * alpha
        right = np.clip(right, 0, 255).astype(np.uint8)
    else:
        right = base.copy()
        cv2.putText(right, "FAILED", (W // 2 - 40, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    sep = np.full((H, 2, 3), 80, dtype=np.uint8)
    return np.concatenate([left, sep, right], axis=1)


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
    """5スライス全てを縦に並べて1枚のPNGに保存する。"""
    panels = []
    for i in range(N_SEG_SLICES):
        ct_f = seg_ct[i].astype(np.float32) / 255.0
        mask_f = seg_mask[i].astype(np.float32)
        panel = draw_single_panel(ct_f, mask_f, all_segs[i], all_lines[i], i)
        panels.append(panel)

    sep_h = np.full((2, panels[0].shape[1], 3), 80, dtype=np.uint8)
    rows = [panels[0]]
    for p in panels[1:]:
        rows.append(sep_h)
        rows.append(p)
    canvas = np.concatenate(rows, axis=0)

    n_ok = sum(1 for q in all_quality if q["success"])
    title_h = 30
    title_bar = np.zeros((title_h, canvas.shape[1], 3), dtype=np.uint8)
    short_uid = study_uid.split(".")[-1]
    title = f"{short_uid}  {vertebra}  ({n_ok}/5 slices OK)"
    cv2.putText(title_bar, title, (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    final = np.concatenate([title_bar, canvas], axis=0)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), final)


def visualize_level_grid(
    samples: list[tuple[str, np.ndarray]],
    level: str,
    save_path: Path,
) -> None:
    """1レベルの5件サンプルを横に並べて保存する（各サンプルは5スライス縦並び）。"""
    if not samples:
        return
    sep_w = np.full((samples[0][1].shape[0], 4, 3), 40, dtype=np.uint8)
    cols = [samples[0][1]]
    for _, img in samples[1:]:
        cols.append(sep_w)
        cols.append(img)
    canvas = np.concatenate(cols, axis=1)

    title_h = 35
    title_bar = np.zeros((title_h, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, f"{level}: 5 samples", (6, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv2.LINE_AA)
    final = np.concatenate([title_bar, canvas], axis=0)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), final)


# ---- 全データ統計 ----

def process_one_level(
    seg_ct: np.ndarray,
    seg_mask: np.ndarray,
    vertebra: str,
    models: list[TinyUNet],
    device: torch.device,
    avg_lengths: dict[str, float],
) -> list[dict[str, Any]]:
    """5スライスそれぞれで4領域分割を試みて品質結果を返す。"""
    results = []
    for i in range(N_SEG_SLICES):
        ct_f = seg_ct[i].astype(np.float32) / 255.0
        mask_f = seg_mask[i].astype(np.float32)
        _, lines = predict_lines_single(models, ct_f, mask_f, vertebra, device, avg_lengths)
        seg, debug = try_generate_region_mask(lines, mask_f)
        quality = evaluate_region_quality(seg, debug)
        quality["slice_idx"] = i
        results.append(quality)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="4領域分割の実現可能性検証（seg_ct.npy使用）",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--fracture-dataset-dir", type=Path, default=FRACTURE_DATASET_DIR)
    parser.add_argument(
        "--training-dataset-dir", type=Path, default=DEFAULT_TRAINING_DATASET_DIR,
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "Unet" / "outputs" / "seg_4region_verification",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--n-vis-per-level", type=int, default=5,
        help="各椎体レベルの可視化サンプル数",
    )
    parser.add_argument(
        "--vis-only", action="store_true",
        help="可視化のみ（全データ統計をスキップ）",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    models = load_models(args.checkpoint_dir, args.n_folds, device)
    print(f"[INFO] {len(models)} fold モデル読み込み完了")
    avg_lengths = compute_average_training_line_lengths(args.training_dataset_dir)
    print("[INFO] 学習データ平均線長(px): " + ", ".join(
        f"{k}={v:.1f}" for k, v in avg_lengths.items()
    ))

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
    print(f"[INFO] 対象スタディ: {len(all_study_ids)}")

    rng = np.random.RandomState(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: 各レベル5件の可視化 ----
    print("\n=== Phase 1: 各椎体レベルの可視化サンプル ===")
    level_vis_samples: dict[str, list[tuple[str, np.ndarray]]] = {
        level: [] for level in VERTEBRA_LEVELS
    }
    level_candidates: dict[str, list[str]] = {level: [] for level in VERTEBRA_LEVELS}
    for study_id in all_study_ids:
        excl = excluded_levels_by_study.get(study_id, set())
        for level in VERTEBRA_LEVELS:
            if level in excl:
                continue
            seg_ct_path = args.fracture_dataset_dir / study_id / level / "seg_ct.npy"
            if seg_ct_path.exists():
                level_candidates[level].append(study_id)

    for level in VERTEBRA_LEVELS:
        candidates = level_candidates[level]
        n = min(args.n_vis_per_level, len(candidates))
        chosen = rng.choice(len(candidates), size=n, replace=False)
        chosen_ids = [candidates[i] for i in sorted(chosen)]
        print(f"  {level}: {len(candidates)} candidates, visualizing {n}")

        for study_id in chosen_ids:
            level_dir = args.fracture_dataset_dir / study_id / level
            seg_ct = np.load(level_dir / "seg_ct.npy")
            seg_mask = np.load(level_dir / "seg_vertebra_mask.npy")

            all_segs = []
            all_lines_list = []
            all_quality = []
            for i in range(N_SEG_SLICES):
                ct_f = seg_ct[i].astype(np.float32) / 255.0
                mask_f = seg_mask[i].astype(np.float32)
                _, lines = predict_lines_single(
                    models, ct_f, mask_f, level, device, avg_lengths,
                )
                seg, debug = try_generate_region_mask(lines, mask_f)
                quality = evaluate_region_quality(seg, debug)
                all_segs.append(seg)
                all_lines_list.append(lines)
                all_quality.append(quality)

            save_path = args.output_dir / "per_level" / level / f"{study_id.split('.')[-1]}.png"
            visualize_5slice(
                seg_ct, seg_mask, all_segs, all_lines_list, all_quality,
                study_id, level, save_path,
            )

            # 個別画像を返してグリッド化用
            img = cv2.imread(str(save_path))
            if img is not None:
                level_vis_samples[level].append((study_id, img))

    # レベルごとのサマリー画像
    for level, samples in level_vis_samples.items():
        if samples:
            grid_path = args.output_dir / f"level_grid_{level}.png"
            visualize_level_grid(samples, level, grid_path)
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
        level_stats = {
            "total": 0,
            "success_all5": 0,
            "success_center": 0,
            "n_regions_hist": defaultdict(int),
            "swap_count": 0,
            "area_ratios": defaultdict(list),
        }

        candidates = level_candidates[level]
        for idx, study_id in enumerate(candidates):
            level_dir = args.fracture_dataset_dir / study_id / level
            try:
                seg_ct = np.load(level_dir / "seg_ct.npy")
                seg_mask = np.load(level_dir / "seg_vertebra_mask.npy")
            except Exception:
                continue

            slice_results = process_one_level(
                seg_ct, seg_mask, level, models, device, avg_lengths,
            )
            level_stats["total"] += 1
            total_processed += 1

            center_ok = slice_results[CENTER_SLICE]["success"]
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

            record = {
                "study_uid": study_id,
                "vertebra": level,
                "center_ok": center_ok,
                "all5_ok": all_ok,
                "n_regions": [r["n_regions"] for r in slice_results],
            }
            all_records.append(record)

            if (idx + 1) % 200 == 0:
                elapsed = time.monotonic() - started
                rate = total_processed / elapsed
                print(
                    f"  {level} [{idx + 1}/{len(candidates)}] "
                    f"({rate:.1f} levels/s)"
                )

        stats_by_level[level] = level_stats
        t = level_stats["total"]
        if t > 0:
            print(
                f"  {level}: {t} total, "
                f"center_ok={level_stats['success_center']}/{t} "
                f"({level_stats['success_center']/t*100:.1f}%), "
                f"all5_ok={level_stats['success_all5']}/{t} "
                f"({level_stats['success_all5']/t*100:.1f}%)"
            )

    # ---- レポート出力 ----
    elapsed_total = time.monotonic() - started
    print(f"\n=== 総合結果 ({elapsed_total:.0f}s) ===")

    report_lines = ["# 4領域分割 実現可能性検証レポート\n"]
    report_lines.append(f"処理時間: {elapsed_total:.0f}s\n")
    report_lines.append("## レベル別成功率\n")
    report_lines.append("| Level | Total | Center OK | Center % | All5 OK | All5 % | Swap |")
    report_lines.append("|-------|-------|-----------|----------|---------|--------|------|")

    grand_total = 0
    grand_center_ok = 0
    grand_all5_ok = 0
    for level in VERTEBRA_LEVELS:
        s = stats_by_level[level]
        t = s["total"]
        grand_total += t
        grand_center_ok += s["success_center"]
        grand_all5_ok += s["success_all5"]
        if t == 0:
            continue
        report_lines.append(
            f"| {level} | {t} | {s['success_center']} | "
            f"{s['success_center']/t*100:.1f}% | {s['success_all5']} | "
            f"{s['success_all5']/t*100:.1f}% | {s['swap_count']} |"
        )
    report_lines.append(
        f"| **Total** | {grand_total} | {grand_center_ok} | "
        f"{grand_center_ok/grand_total*100:.1f}% | {grand_all5_ok} | "
        f"{grand_all5_ok/grand_total*100:.1f}% | — |"
    )

    report_lines.append("\n## 領域数分布（全スライス）\n")
    report_lines.append("| Level | 0 | 1 | 2 | 3 | 4 |")
    report_lines.append("|-------|---|---|---|---|---|")
    for level in VERTEBRA_LEVELS:
        s = stats_by_level[level]
        h = s["n_regions_hist"]
        total_slices = sum(h.values())
        if total_slices == 0:
            continue
        row = f"| {level} |"
        for n in range(5):
            count = h.get(n, 0)
            pct = count / total_slices * 100
            row += f" {count} ({pct:.1f}%) |"
        report_lines.append(row)

    report_lines.append("\n## 領域面積比の統計（4領域成功スライスのみ）\n")
    report_lines.append("| Level | Region | Mean | Std | Min | Max |")
    report_lines.append("|-------|--------|------|-----|-----|-----|")
    for level in VERTEBRA_LEVELS:
        s = stats_by_level[level]
        for region in ("body", "right_foramen", "left_foramen", "posterior"):
            vals = s["area_ratios"].get(region, [])
            if not vals:
                continue
            arr = np.array(vals)
            report_lines.append(
                f"| {level} | {region} | {arr.mean():.3f} | {arr.std():.3f} | "
                f"{arr.min():.3f} | {arr.max():.3f} |"
            )

    report_text = "\n".join(report_lines) + "\n"
    report_path = args.output_dir / "verification_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"[SAVED] {report_path.relative_to(PROJECT_ROOT)}")

    # CSV出力
    csv_path = args.output_dir / "verification_details.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "study_uid", "vertebra", "center_ok", "all5_ok",
            "n_regions_s0", "n_regions_s1", "n_regions_s2",
            "n_regions_s3", "n_regions_s4",
        ])
        for r in all_records:
            writer.writerow([
                r["study_uid"], r["vertebra"],
                int(r["center_ok"]), int(r["all5_ok"]),
                *r["n_regions"],
            ])
    print(f"[SAVED] {csv_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
