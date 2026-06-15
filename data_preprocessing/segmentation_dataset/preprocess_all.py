from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from Unet.preprocessing.generate_region_mask import (
    generate_region_mask,
    validate_region_mask,
)

VERTEBRAE = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
DATASET_ROOT = Path("/mnt/nfs1/home/yamamoto-hiroto/research/VAI/dataset")
BAD_SLICES_OUTPUT = DATASET_ROOT / "bad_slices_all.json"

LINE_KEYS = ("line_1", "line_2", "line_3", "line_4")

PROGRESS_INTERVAL = 100
MIN_POLYLINE_POINTS = 2
POINT_DIMENSIONS = 2
SLICE_INDEX_DIGITS = 3
SEG_MASK_NDIM = 3
SEG_MASK_CHANNELS = 5
GT_MASK_DIRNAME = "gt_masks"
GT_OVERLAY_DIRNAME = "gt_overlays"
LABEL_IMAGE_DTYPE = np.uint8

# 領域カラーマップ (BGR): 0=bg, 1=body, 2=right, 3=left, 4=posterior
REGION_COLORS_BGR: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),        # 0: 背景 (黒)
    (0, 200, 0),      # 1: body (緑)
    (0, 0, 200),      # 2: right foramen (赤)
    (200, 0, 0),      # 3: left foramen (青)
    (0, 200, 200),    # 4: posterior (黄)
)
OVERLAY_ALPHA = 0.45


@dataclass(frozen=True)
class SliceCandidate:
    """処理対象スライスの情報を保持する。"""

    sample: str
    vertebra: str
    slice_idx: int
    lines: dict[str, list[list[float]]]
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class SliceProcessResult:
    """1スライス処理結果を保持する。"""

    sample: str
    vertebra: str
    slice_idx: int
    status: str
    reason: str
    stage: str
    fallback_type: str

    @property
    def is_success(self) -> bool:
        """成功判定を返す。"""
        return self.status == "success"

    def to_failure_record(self) -> dict[str, Any]:
        """失敗レコード形式へ変換する。"""
        return {
            "sample": self.sample,
            "vertebra": self.vertebra,
            "slice": self.slice_idx,
            "status": self.status,
            "reason": self.reason,
            "stage": self.stage,
            "fallback_type": self.fallback_type,
        }


def is_valid_polyline(points: Any) -> bool:
    """有効なポリライン点列かを判定する。"""
    if not isinstance(points, list) or len(points) < MIN_POLYLINE_POINTS:
        return False

    for point in points:
        if not isinstance(point, (list, tuple)):
            return False
        if len(point) != POINT_DIMENSIONS:
            return False

    return True


def parse_slice_index(slice_key: str) -> int | None:
    """スライスキー文字列を整数へ変換する。"""
    try:
        return int(slice_key)
    except (TypeError, ValueError):
        return None


def build_slice_filename(slice_idx: int) -> str:
    """スライス番号から標準ファイル名を生成する。"""
    return f"slice_{slice_idx:0{SLICE_INDEX_DIGITS}d}.png"


def extract_valid_lines(entry: Any) -> dict[str, list[list[float]]] | None:
    """line_1-4 がすべて有効なら辞書を返す。"""
    if not isinstance(entry, dict):
        return None

    lines: dict[str, list[list[float]]] = {}
    for key in LINE_KEYS:
        points = entry.get(key)
        if not is_valid_polyline(points):
            return None
        lines[key] = [[float(p[0]), float(p[1])] for p in points]
    return lines


def collect_valid_slices(
    dataset_root: Path,
    vertebrae: list[str],
) -> list[SliceCandidate]:
    """データセット全体から有効スライス候補を収集する。"""
    candidates: list[SliceCandidate] = []

    sample_dirs = sorted(
        path
        for path in dataset_root.glob("sample*")
        if path.is_dir()
    )

    for sample_dir in sample_dirs:
        sample_name = sample_dir.name
        for vertebra in vertebrae:
            vertebra_dir = sample_dir / vertebra
            lines_path = vertebra_dir / "lines.json"
            image_dir = vertebra_dir / "images"
            mask_dir = vertebra_dir / "masks"

            if not lines_path.exists():
                continue

            try:
                lines_data = json.loads(lines_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue

            if not isinstance(lines_data, dict):
                continue

            for slice_key, entry in lines_data.items():
                slice_idx = parse_slice_index(str(slice_key))
                if slice_idx is None:
                    continue

                lines = extract_valid_lines(entry)
                if lines is None:
                    continue

                slice_filename = build_slice_filename(slice_idx)
                image_path = image_dir / slice_filename
                mask_path = mask_dir / slice_filename

                if not image_path.exists() or not mask_path.exists():
                    continue

                candidates.append(
                    SliceCandidate(
                        sample=sample_name,
                        vertebra=vertebra,
                        slice_idx=slice_idx,
                        lines=lines,
                        image_path=image_path,
                        mask_path=mask_path,
                    )
                )

    return candidates


def classify_exception_stage(error: Exception) -> str:
    """例外内容から失敗ステージを推定する。"""
    message = str(error).lower()
    if "connectedcomponents" in message or "connected components" in message:
        return "connectedComponents"
    return "generate"


def build_overlay_image(ct_gray: np.ndarray, label_img: np.ndarray) -> np.ndarray:
    """CT画像とラベル画像からカラーオーバーレイBGR画像を生成する。"""
    ct_bgr = cv2.cvtColor(ct_gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    color_layer = np.zeros_like(ct_bgr)

    for label_val, bgr in enumerate(REGION_COLORS_BGR):
        if label_val == 0:
            continue
        mask = label_img == label_val
        color_layer[mask] = bgr

    fg_mask = (label_img > 0).astype(np.float32)[..., np.newaxis]
    overlay = ct_bgr * (1.0 - fg_mask * OVERLAY_ALPHA) + color_layer * OVERLAY_ALPHA
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_failures(failures: list[SliceProcessResult], output_path: Path) -> None:
    """失敗スライスをJSONとして保存する。"""
    payload = [result.to_failure_record() for result in failures]
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_summary(results: list[SliceProcessResult]) -> None:
    """実行結果サマリーを標準出力へ表示する。"""
    total = len(results)
    success_count = sum(1 for r in results if r.is_success)
    failure_count = total - success_count
    failure_rate = (failure_count / total * 100.0) if total > 0 else 0.0

    fallback_counter = Counter(
        r.fallback_type for r in results if r.fallback_type.strip()
    )
    failures = [r for r in results if not r.is_success]

    print(f"Total processed: {total}")
    print(f"Success count: {success_count}")
    print(f"Failure count: {failure_count}")
    print(f"Failure rate: {failure_rate:.2f}%")
    print("")

    print("Fallback type distribution:")
    if fallback_counter:
        for fallback_type, count in sorted(fallback_counter.items()):
            print(f"  {fallback_type}: {count}")
    else:
        print("  (none)")
    print("")

    print("Failed slices:")
    if failures:
        for result in failures:
            print(
                f"  {result.sample}/{result.vertebra}/"
                f"{build_slice_filename(result.slice_idx)}: "
                f"{result.reason}"
            )
    else:
        print("  (none)")


def process_and_save_slice(
    candidate: SliceCandidate,
    dataset_root: Path,
) -> SliceProcessResult:
    """1スライスの領域マスクを生成し、バイナリPNGで保存する。"""
    fallback_type = ""

    try:
        vertebra_mask = cv2.imread(
            str(candidate.mask_path),
            cv2.IMREAD_GRAYSCALE,
        )
        if vertebra_mask is None:
            return SliceProcessResult(
                sample=candidate.sample,
                vertebra=candidate.vertebra,
                slice_idx=candidate.slice_idx,
                status="hard_fail",
                reason=f"マスク読込失敗: {candidate.mask_path}",
                stage="load",
                fallback_type=fallback_type,
            )

        seg_mask, debug_info = generate_region_mask(
            line_1=candidate.lines["line_1"],
            line_2=candidate.lines["line_2"],
            line_3=candidate.lines["line_3"],
            line_4=candidate.lines["line_4"],
            vertebra_mask=vertebra_mask,
        )
        fallback_type = str(debug_info.get("fallback_type", "")) if isinstance(
            debug_info, dict
        ) else ""

        if seg_mask.ndim != SEG_MASK_NDIM:
            return SliceProcessResult(
                sample=candidate.sample,
                vertebra=candidate.vertebra,
                slice_idx=candidate.slice_idx,
                status="hard_fail",
                reason=f"seg_mask次元不正: {seg_mask.shape}",
                stage="generate",
                fallback_type=fallback_type,
            )
        if seg_mask.shape[0] != SEG_MASK_CHANNELS:
            return SliceProcessResult(
                sample=candidate.sample,
                vertebra=candidate.vertebra,
                slice_idx=candidate.slice_idx,
                status="hard_fail",
                reason=f"seg_maskチャンネル不正: {seg_mask.shape}",
                stage="generate",
                fallback_type=fallback_type,
            )

        validation = validate_region_mask(seg_mask, vertebra_mask)
        hard_fail = bool(validation.get("hard_fail", False))
        raw_warnings = validation.get("warnings", [])
        warning_list = (
            [str(warning) for warning in raw_warnings]
            if isinstance(raw_warnings, list)
            else []
        )

        if hard_fail:
            reason = str(warning_list[0]) if warning_list else "unknown"
            return SliceProcessResult(
                sample=candidate.sample,
                vertebra=candidate.vertebra,
                slice_idx=candidate.slice_idx,
                status="hard_fail",
                reason=reason,
                stage="validate",
                fallback_type=fallback_type,
            )

        label_img = np.argmax(seg_mask, axis=0).astype(LABEL_IMAGE_DTYPE)
        slice_filename = build_slice_filename(candidate.slice_idx)

        out_dir = dataset_root / candidate.sample / candidate.vertebra / GT_MASK_DIRNAME
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / slice_filename
        if not cv2.imwrite(str(out_path), label_img):
            return SliceProcessResult(
                sample=candidate.sample,
                vertebra=candidate.vertebra,
                slice_idx=candidate.slice_idx,
                status="hard_fail",
                reason=f"保存失敗: {out_path}",
                stage="save",
                fallback_type=fallback_type,
            )

        ct_gray = cv2.imread(str(candidate.image_path), cv2.IMREAD_GRAYSCALE)
        if ct_gray is not None:
            overlay_dir = (
                dataset_root / candidate.sample / candidate.vertebra / GT_OVERLAY_DIRNAME
            )
            overlay_dir.mkdir(parents=True, exist_ok=True)
            overlay_img = build_overlay_image(ct_gray, label_img)
            cv2.imwrite(str(overlay_dir / slice_filename), overlay_img)

        return SliceProcessResult(
            sample=candidate.sample,
            vertebra=candidate.vertebra,
            slice_idx=candidate.slice_idx,
            status="success",
            reason="",
            stage="",
            fallback_type=fallback_type,
        )
    except Exception as error:
        return SliceProcessResult(
            sample=candidate.sample,
            vertebra=candidate.vertebra,
            slice_idx=candidate.slice_idx,
            status="exception",
            reason=str(error),
            stage=classify_exception_stage(error),
            fallback_type=fallback_type,
        )


def main() -> None:
    """全候補スライスのGT領域マスクを生成して保存する。"""
    candidates = collect_valid_slices(DATASET_ROOT, VERTEBRAE)
    print(f"Total candidates: {len(candidates)}")

    total = len(candidates)
    results: list[SliceProcessResult] = []

    for i, candidate in enumerate(candidates, start=1):
        result = process_and_save_slice(candidate, DATASET_ROOT)
        results.append(result)

        if i % PROGRESS_INTERVAL == 0:
            print(f"Progress: {i}/{total}")

    failures = [result for result in results if not result.is_success]

    save_failures(failures, BAD_SLICES_OUTPUT)
    print_summary(results)
    print(f"Saved failures: {BAD_SLICES_OUTPUT}")


if __name__ == "__main__":
    main()
