"""良好な可視化症例を参照分布とする SDF 4 領域 QC。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .constants import (
    FRACTURE_DATASET_DIR,
    N_CLASSIFIER_PLANES,
    PROJECT_ROOT,
    VERTEBRA_LEVELS,
)
from .visualization import concat_with_separator, make_region_overlay

REFERENCE_VIS_DIR = PROJECT_ROOT / "Unet" / "outputs" / "sdf_segmentation_vis_multi_new"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Unet" / "outputs" / "sdf_qc_reference_based"
REGION_LABELS = (1, 2, 3, 4)
MIN_PASS_THRESHOLD = 0.08
THRESHOLD_MARGIN = 1.5


@dataclass(frozen=True)
class LevelFeatures:
    """15 プレーン分の領域分布・位置・形状特徴を保持する。"""

    ratios: np.ndarray
    present: np.ndarray
    centroids: np.ndarray
    compactness: np.ndarray
    hard_failures: tuple[str, ...]


@dataclass(frozen=True)
class ReferenceModel:
    """椎体別参照特徴と受入閾値を保持する。"""

    features: dict[str, list[LevelFeatures]]
    thresholds: dict[str, float]


def _normalised_region_features(
    label_map: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """1 プレーンの面積比・出現・重心・形状を計算する。"""
    foreground = mask > 0
    ratios = np.zeros(4, dtype=np.float64)
    present = np.zeros(4, dtype=np.float64)
    centroids = np.full((4, 2), np.nan, dtype=np.float64)
    compactness = np.zeros(4, dtype=np.float64)
    total_area = int(foreground.sum())
    if total_area == 0:
        return ratios, present, centroids, compactness

    mask_y, mask_x = np.where(foreground)
    x_min, x_max = int(mask_x.min()), int(mask_x.max())
    y_min, y_max = int(mask_y.min()), int(mask_y.max())
    width = max(x_max - x_min, 1)
    height = max(y_max - y_min, 1)

    for fi, label in enumerate(REGION_LABELS):
        region = (label_map == label) & foreground
        area = int(region.sum())
        ratios[fi] = area / total_area
        if area == 0:
            continue
        present[fi] = 1.0
        ry, rx = np.where(region)
        centroids[fi] = (
            (float(rx.mean()) - x_min) / width,
            (float(ry.mean()) - y_min) / height,
        )
        bbox_area = (int(rx.max()) - int(rx.min()) + 1) * (
            int(ry.max()) - int(ry.min()) + 1
        )
        compactness[fi] = area / max(bbox_area, 1)
    return ratios, present, centroids, compactness


def extract_level_features(region: np.ndarray, mask: np.ndarray) -> LevelFeatures:
    """1 椎体の region 配列から参照比較用特徴を抽出する。"""
    hard_failures: list[str] = []
    if region.shape != (N_CLASSIFIER_PLANES, 224, 224):
        hard_failures.append(f"region_shape={region.shape}")
    if mask.shape != (N_CLASSIFIER_PLANES, 224, 224):
        hard_failures.append(f"mask_shape={mask.shape}")
    if hard_failures:
        zeros = np.zeros((N_CLASSIFIER_PLANES, 4), dtype=np.float64)
        return LevelFeatures(
            ratios=zeros,
            present=zeros,
            centroids=np.full((N_CLASSIFIER_PLANES, 4, 2), np.nan),
            compactness=zeros,
            hard_failures=tuple(hard_failures),
        )

    invalid_labels = np.setdiff1d(np.unique(region), np.arange(5))
    if invalid_labels.size:
        hard_failures.append(f"invalid_labels={invalid_labels.tolist()}")
    if bool(np.any(region[mask == 0] != 0)):
        hard_failures.append("labels_outside_vertebra_mask")
    if bool(np.any(region[mask > 0] == 0)):
        hard_failures.append("unlabelled_inside_vertebra_mask")

    ratios = np.zeros((N_CLASSIFIER_PLANES, 4), dtype=np.float64)
    present = np.zeros((N_CLASSIFIER_PLANES, 4), dtype=np.float64)
    centroids = np.full((N_CLASSIFIER_PLANES, 4, 2), np.nan, dtype=np.float64)
    compactness = np.zeros((N_CLASSIFIER_PLANES, 4), dtype=np.float64)
    for pi in range(N_CLASSIFIER_PLANES):
        pf = _normalised_region_features(region[pi], mask[pi])
        ratios[pi], present[pi], centroids[pi], compactness[pi] = pf

    return LevelFeatures(
        ratios=ratios,
        present=present,
        centroids=centroids,
        compactness=compactness,
        hard_failures=tuple(hard_failures),
    )


def feature_distance(left: LevelFeatures, right: LevelFeatures) -> float:
    """2 症例間の領域傾向・分布・形状距離を返す。"""
    ratio_d = float(np.mean(np.abs(left.ratios - right.ratios)))
    presence_d = float(np.mean(np.abs(left.present - right.present)))
    compactness_d = float(np.mean(np.abs(left.compactness - right.compactness)))
    both = (left.present > 0) & (right.present > 0)
    centroid_mask = np.repeat(both[..., None], 2, axis=-1)
    centroid_d = (
        float(
            np.mean(
                np.abs(left.centroids[centroid_mask] - right.centroids[centroid_mask])
            )
        )
        if bool(np.any(centroid_mask))
        else 1.0
    )
    return 0.50 * ratio_d + 0.20 * presence_d + 0.20 * centroid_d + 0.10 * compactness_d


def reference_study_ids(reference_vis_dir: Path) -> list[str]:
    """可視化 PNG 名から参照 study UID を復元する。"""
    short_ids = sorted(
        {p.stem.rsplit("_", maxsplit=1)[0] for p in reference_vis_dir.glob("*_C?.png")}
    )
    full_ids = {
        d.name.split(".")[-1]: d.name
        for d in FRACTURE_DATASET_DIR.iterdir()
        if d.is_dir()
    }
    missing = [sid for sid in short_ids if sid not in full_ids]
    if missing:
        raise ValueError(f"参照 study が見つかりません: {missing}")
    return [full_ids[sid] for sid in short_ids]


def _load_level_features(
    dataset_dir: Path, study_id: str, vertebra: str
) -> LevelFeatures:
    level_dir = dataset_dir / study_id / vertebra
    return extract_level_features(
        np.load(level_dir / "region_4class.npy"),
        np.load(level_dir / "vertebra_mask.npy"),
    )


def build_reference_model(dataset_dir: Path, study_ids: list[str]) -> ReferenceModel:
    """参照症例の最近傍距離から椎体別受入閾値を決める。"""
    features = {
        v: [_load_level_features(dataset_dir, sid, v) for sid in study_ids]
        for v in VERTEBRA_LEVELS
    }
    thresholds: dict[str, float] = {}
    for v, refs in features.items():
        nearest = []
        for i, ref in enumerate(refs):
            dists = [feature_distance(ref, c) for j, c in enumerate(refs) if j != i]
            nearest.append(min(dists))
        thresholds[v] = max(MIN_PASS_THRESHOLD, max(nearest) * THRESHOLD_MARGIN)
    return ReferenceModel(features=features, thresholds=thresholds)


def score_level(
    features: LevelFeatures, references: list[LevelFeatures]
) -> tuple[float, int]:
    """最も近い参照症例との距離と index を返す。"""
    dists = [feature_distance(features, r) for r in references]
    idx = int(np.argmin(dists))
    return dists[idx], idx


def save_level_visualization(
    dataset_dir: Path,
    study_id: str,
    vertebra: str,
    score: float,
    threshold: float,
    output_dir: Path,
) -> None:
    """QC 不合格椎体の 15 プレーンを保存する。"""
    level_dir = dataset_dir / study_id / vertebra
    ct = np.load(level_dir / "ct.npy")
    region = np.load(level_dir / "region_4class.npy")
    panels = [
        make_region_overlay(ct[i, 2], region[i]) for i in range(N_CLASSIFIER_PLANES)
    ]
    row = concat_with_separator(panels, axis=1)
    title = np.zeros((26, row.shape[1], 3), dtype=np.uint8)
    text = f"{study_id.split('.')[-1]} {vertebra} score={score:.3f} threshold={threshold:.3f}"
    cv2.putText(
        title,
        text,
        (4, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(output_dir / f"{study_id.split('.')[-1]}_{vertebra}.png"),
        np.concatenate([title, row], axis=0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="参照症例分布に基づく SDF 4 領域 QC")
    parser.add_argument(
        "--fracture-dataset-dir", type=Path, default=FRACTURE_DATASET_DIR
    )
    parser.add_argument("--reference-vis-dir", type=Path, default=REFERENCE_VIS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-studies", type=int, default=0)
    parser.add_argument("--n-failure-vis", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ref_ids = reference_study_ids(args.reference_vis_dir)
    ref_model = build_reference_model(args.fracture_dataset_dir, ref_ids)
    study_ids = sorted(
        d.name
        for d in args.fracture_dataset_dir.iterdir()
        if d.is_dir()
        and all((d / v / "region_4class.npy").exists() for v in VERTEBRA_LEVELS)
    )
    if 0 < args.n_studies < len(study_ids):
        rng = np.random.RandomState(args.seed)
        idx = sorted(rng.choice(len(study_ids), args.n_studies, replace=False))
        study_ids = [study_ids[i] for i in idx]

    rows: list[dict[str, object]] = []
    failures: list[tuple[float, str, str, float]] = []
    for si, study_id in enumerate(study_ids, start=1):
        for vertebra in VERTEBRA_LEVELS:
            feats = _load_level_features(args.fracture_dataset_dir, study_id, vertebra)
            score, nearest_idx = score_level(feats, ref_model.features[vertebra])
            threshold = ref_model.thresholds[vertebra]
            passed = not feats.hard_failures and score <= threshold
            rows.append(
                {
                    "study_id": study_id,
                    "vertebra": vertebra,
                    "score": score,
                    "threshold": threshold,
                    "passed": int(passed),
                    "nearest_reference": ref_ids[nearest_idx],
                    "hard_failures": ";".join(feats.hard_failures),
                }
            )
            if not passed:
                failures.append((score / threshold, study_id, vertebra, score))
        if si % 200 == 0:
            print(f"[{si}/{len(study_ids)}] QC 完了", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "level_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report_lines = [
        "# 参照症例ベース SDF 4 領域 QC",
        "",
        f"- 対象 study: {len(study_ids)}",
        f"- 参照 study: {', '.join(sid.split('.')[-1] for sid in ref_ids)}",
        "",
        "## 椎体別結果",
        "",
        "| Level | Pass | Total | Pass rate | Median score | P95 score | Threshold |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for v in VERTEBRA_LEVELS:
        lr = [r for r in rows if r["vertebra"] == v]
        scores = np.asarray([float(r["score"]) for r in lr])
        pc = sum(int(r["passed"]) for r in lr)
        report_lines.append(
            f"| {v} | {pc} | {len(lr)} | {pc / len(lr):.1%} | {np.median(scores):.3f} | "
            f"{np.quantile(scores, 0.95):.3f} | {ref_model.thresholds[v]:.3f} |"
        )

    report_path = args.output_dir / "qc_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    fail_dir = args.output_dir / "failure_samples"
    for _ratio, study_id, vertebra, score in sorted(failures, reverse=True)[
        : args.n_failure_vis
    ]:
        save_level_visualization(
            args.fracture_dataset_dir,
            study_id,
            vertebra,
            score,
            ref_model.thresholds[vertebra],
            fail_dir,
        )

    print(f"[DONE] pass={sum(int(r['passed']) for r in rows)}/{len(rows)}")
    print(f"[SAVED] {report_path}")
    print(f"[SAVED] {csv_path}")


if __name__ == "__main__":
    main()
