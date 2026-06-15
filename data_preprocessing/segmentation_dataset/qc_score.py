"""椎骨マスクのQCスコアリングスクリプト。

各スライスを keep / downweight / exclude に分類し、
dataset/{sample}/{vertebra}/qc_scores.json として保存する。

指標:
  - solidity       : 凸包面積 / 実面積（歪み検出）
  - region_valid   : 4領域が全て非ゼロか
  - area_continuity: 同椎骨内の中央値面積との比率（孤立異常検出）
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from Unet.preprocessing.generate_region_mask import generate_region_mask
from Unet.preprocessing.pilot_region_mask import (
    collect_valid_slices,
    SliceCandidate,
)

DATASET_ROOT = Path("/mnt/nfs1/home/yamamoto-hiroto/research/VAI/dataset")
VERTEBRAE = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

# 判定閾値
SOLIDITY_EXCLUDE = 0.50        # 真に異常な歪み（全体の~0.3%）
SOLIDITY_DOWNWEIGHT = 0.60     # 境界ケース（~5%）
AREA_CONTINUITY_EXCLUDE = 3.0  # 中央値の3倍超 or 1/3未満でexclude
AREA_CONTINUITY_DOWNWEIGHT = 2.0


@dataclass
class SliceQC:
    """1スライスのQC結果を保持する。"""

    sample: str
    vertebra: str
    slice_idx: int
    solidity: float
    region_valid: bool
    area: int
    area_ratio: float  # 同椎骨中央値との比率（後で計算）
    label: str         # keep / downweight / exclude
    reason: str


def compute_solidity(mask: np.ndarray) -> float:
    """マスクのsolidity（凸包面積/実面積）を返す。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area == 0:
        return 0.0
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0.0
    return float(area / hull_area)


def check_region_valid(
    lines: dict[str, list[list[float]]],
    mask: np.ndarray,
) -> bool:
    """4領域（body/right_foramen/left_foramen/posterior）が全て非ゼロか判定する。"""
    try:
        seg_mask, _ = generate_region_mask(
            line_1=lines["line_1"],
            line_2=lines["line_2"],
            line_3=lines["line_3"],
            line_4=lines["line_4"],
            vertebra_mask=mask,
        )
        return all(int(seg_mask[ch].sum()) > 0 for ch in [1, 2, 3, 4])
    except Exception:
        return False


def classify_label(
    solidity: float,
    region_valid: bool,
    area_ratio: float,
) -> tuple[str, str]:
    """指標からlabelとreasonを返す。"""
    if solidity < SOLIDITY_EXCLUDE:
        return "exclude", f"solidity={solidity:.3f}"
    if not region_valid:
        return "exclude", "region_invalid"
    if area_ratio > AREA_CONTINUITY_EXCLUDE or area_ratio < 1.0 / AREA_CONTINUITY_EXCLUDE:
        return "exclude", f"area_ratio={area_ratio:.2f}"

    if solidity < SOLIDITY_DOWNWEIGHT:
        return "downweight", f"solidity={solidity:.3f}"
    if area_ratio > AREA_CONTINUITY_DOWNWEIGHT or area_ratio < 1.0 / AREA_CONTINUITY_DOWNWEIGHT:
        return "downweight", f"area_ratio={area_ratio:.2f}"

    return "keep", ""


def score_vertebra_slices(
    candidates: list[SliceCandidate],
) -> list[SliceQC]:
    """同一椎骨のスライスリストをスコアリングする。"""
    results: list[SliceQC] = []

    # 先にarea計算（中央値算出用）
    areas: list[int] = []
    raw: list[tuple[SliceCandidate, float, bool, int]] = []

    for c in candidates:
        mask = cv2.imread(str(c.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raw.append((c, 0.0, False, 0))
            areas.append(0)
            continue
        binary = (mask > 0).astype(np.uint8)
        area = int(binary.sum())
        solidity = compute_solidity(binary)
        region_valid = check_region_valid(c.lines, binary)
        raw.append((c, solidity, region_valid, area))
        areas.append(area)

    median_area = float(np.median(areas)) if areas else 1.0

    for c, solidity, region_valid, area in raw:
        area_ratio = float(area / median_area) if median_area > 0 else 0.0
        label, reason = classify_label(solidity, region_valid, area_ratio)
        results.append(SliceQC(
            sample=c.sample,
            vertebra=c.vertebra,
            slice_idx=c.slice_idx,
            solidity=round(solidity, 4),
            region_valid=region_valid,
            area=area,
            area_ratio=round(area_ratio, 4),
            label=label,
            reason=reason,
        ))

    return results


def save_qc_scores(
    results: list[SliceQC],
    vertebra_dir: Path,
) -> None:
    """QCスコアをJSONに保存する。"""
    payload: dict[str, Any] = {}
    for r in results:
        payload[str(r.slice_idx)] = {
            k: v for k, v in asdict(r).items()
            if k not in ("sample", "vertebra", "slice_idx")
        }
    out_path = vertebra_dir / "qc_scores.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """全サンプル・全椎骨のQCスコアを計算・保存する。"""
    all_candidates = collect_valid_slices(DATASET_ROOT, VERTEBRAE)
    print(f"対象スライス数: {len(all_candidates)}")

    # (sample, vertebra) 単位でグループ化
    groups: dict[tuple[str, str], list[SliceCandidate]] = {}
    for c in all_candidates:
        key = (c.sample, c.vertebra)
        groups.setdefault(key, []).append(c)

    total = keep = downweight = exclude = 0

    for (sample, vertebra), candidates in sorted(groups.items()):
        vertebra_dir = DATASET_ROOT / sample / vertebra
        results = score_vertebra_slices(candidates)
        save_qc_scores(results, vertebra_dir)

        g_keep = sum(1 for r in results if r.label == "keep")
        g_dw = sum(1 for r in results if r.label == "downweight")
        g_ex = sum(1 for r in results if r.label == "exclude")
        total += len(results)
        keep += g_keep
        downweight += g_dw
        exclude += g_ex
        print(f"{sample}/{vertebra}: keep={g_keep} downweight={g_dw} exclude={g_ex}")

    print(f"\nSummary: {total}件 | keep={keep} downweight={downweight} exclude={exclude}")


if __name__ == "__main__":
    main()
