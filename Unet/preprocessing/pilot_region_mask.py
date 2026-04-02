from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from Unet.preprocessing.generate_region_mask import (
    generate_region_mask,
    validate_region_mask,
)

DATASET_ROOT = Path("/mnt/nfs1/home/yamamoto-hiroto/research/VAI/dataset")
VERTEBRAE = ["C3", "C4", "C5", "C6", "C7"]
SAMPLE_SIZE = 100
RANDOM_SEED = 42

BAD_SLICES_OUTPUT = DATASET_ROOT / "bad_slices_pilot.json"
LINE_KEYS = ("line_1", "line_2", "line_3", "line_4")


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
    if not isinstance(points, list) or len(points) < 2:
        return False
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return False
    return True


def parse_slice_index(slice_key: str) -> int | None:
    """スライスキー文字列を整数へ変換する。"""
    try:
        return int(slice_key)
    except (TypeError, ValueError):
        return None


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

                image_path = image_dir / f"slice_{slice_idx:03d}.png"
                mask_path = mask_dir / f"slice_{slice_idx:03d}.png"

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


def sample_candidates(
    candidates: list[SliceCandidate],
    sample_size: int,
    seed: int,
) -> list[SliceCandidate]:
    """固定シードで候補をランダム抽出する。"""
    if not candidates:
        return []
    take = min(sample_size, len(candidates))
    rng = random.Random(seed)
    return rng.sample(candidates, k=take)


def classify_exception_stage(error: Exception) -> str:
    """例外内容から失敗ステージを推定する。"""
    message = str(error).lower()
    if "connectedcomponents" in message or "connected components" in message:
        return "connectedComponents"
    return "generate"


def process_one_slice(candidate: SliceCandidate) -> SliceProcessResult:
    """1スライスに対して生成と検証を実行する。"""
    fallback_type = ""

    try:
        vertebra_mask = cv2.imread(
            str(candidate.mask_path),
            cv2.IMREAD_GRAYSCALE,
        )
        if vertebra_mask is None:
            raise ValueError(f"マスク読込失敗: {candidate.mask_path}")

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

        validation = validate_region_mask(seg_mask, vertebra_mask)
        hard_fail = bool(validation.get("hard_fail", False))
        warnings = validation.get("warnings", [])
        warning_list = warnings if isinstance(warnings, list) else []

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
                f"  {result.sample}/{result.vertebra}/slice_{result.slice_idx:03d}: "
                f"{result.reason}"
            )
    else:
        print("  (none)")


def main() -> None:
    """パイロット用にランダム100スライスを検証する。"""
    candidates = collect_valid_slices(DATASET_ROOT, VERTEBRAE)
    sampled = sample_candidates(candidates, SAMPLE_SIZE, RANDOM_SEED)

    if len(sampled) < SAMPLE_SIZE:
        print(
            f"[WARN] 有効スライス数が不足: requested={SAMPLE_SIZE}, "
            f"available={len(candidates)}, sampled={len(sampled)}"
        )

    results = [process_one_slice(candidate) for candidate in sampled]
    failures = [result for result in results if not result.is_success]

    save_failures(failures, BAD_SLICES_OUTPUT)
    print_summary(results)
    print("")
    print(f"Saved failures: {BAD_SLICES_OUTPUT}")


if __name__ == "__main__":
    main()
