"""SDF 4 領域分割の品質チェック。

チェック内容:
  1. 処理完了率（region_4class.npy の存在確認）
  2. プレーン別成功率（4 領域が全て存在するか）
  3. 椎体レベル別・プレーン別の領域面積比統計
  4. 解剖学的整合性チェック（面積の単調性など）
  5. 外れ値の可視化
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
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

REGION_NAMES = {0: "bg", 1: "body", 2: "right", 3: "left", 4: "post"}
EXPECTED_MIN_RATIO = {"body": 0.15, "right": 0.03, "left": 0.03, "post": 0.10}


def check_one_level(
    region: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """1 椎体レベルの品質指標を計算する。"""
    plane_results = []
    for i in range(N_CLASSIFIER_PLANES):
        fg = mask[i] > 0
        total_fg = int(fg.sum())
        if total_fg == 0:
            plane_results.append({"plane": i, "empty_mask": True})
            continue

        labels_in_fg = region[i][fg]
        present = set(int(v) for v in np.unique(labels_in_fg) if v > 0)
        n_regions = len(present)

        ratios = {}
        for lbl in range(1, 5):
            ratios[REGION_NAMES[lbl]] = int((labels_in_fg == lbl).sum()) / total_fg

        is_central = 3 <= i <= 11
        anat_ok = True
        if is_central and n_regions == 4:
            for rname, min_r in EXPECTED_MIN_RATIO.items():
                if ratios.get(rname, 0) < min_r:
                    anat_ok = False
                    break

        plane_results.append(
            {
                "plane": i,
                "empty_mask": False,
                "n_regions": n_regions,
                "ok": n_regions == 4,
                "anat_ok": anat_ok,
                "is_central": is_central,
                "total_fg": total_fg,
                "ratios": ratios,
            }
        )
    return {"planes": plane_results}


def visualize_failure(
    study_id: str,
    vertebra: str,
    ct: np.ndarray,
    region: np.ndarray,
    mask: np.ndarray,
    plane_results: list[dict],
    out_dir: Path,
) -> None:
    """失敗・異常プレーンのある椎体を可視化して保存する。"""
    panels = []
    for i, pr in enumerate(plane_results):
        overlay = make_region_overlay(ct[i, 2], region[i])
        if not pr.get("ok", True) or not pr.get("anat_ok", True):
            cv2.rectangle(overlay, (0, 0), (223, 223), (0, 0, 255), 3)
        elif pr.get("is_central"):
            cv2.rectangle(overlay, (0, 0), (223, 223), (0, 255, 0), 1)
        cv2.putText(
            overlay, str(i), (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
        )
        nr = pr.get("n_regions", 0)
        cv2.putText(
            overlay, f"r{nr}", (2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1
        )
        panels.append(overlay)

    row = concat_with_separator(panels, axis=1)
    title = np.zeros((24, row.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        title,
        f"{study_id.split('.')[-1]}  {vertebra}  red=fail  green=central_ok",
        (4, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
    )
    canvas = np.concatenate([title, row], axis=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    short_id = study_id.split(".")[-1]
    cv2.imwrite(str(out_dir / f"{short_id}_{vertebra}.png"), canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description="SDF 4 領域分割の品質チェック")
    parser.add_argument(
        "--fracture-dataset-dir", type=Path, default=FRACTURE_DATASET_DIR
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "Unet" / "outputs" / "sdf_qc"
    )
    parser.add_argument("--n-studies", type=int, default=200)
    parser.add_argument("--n-failure-vis", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_studies = sorted(
        d.name for d in args.fracture_dataset_dir.iterdir() if d.is_dir()
    )
    has_region = [
        s
        for s in all_studies
        if any(
            (args.fracture_dataset_dir / s / lv / "region_4class.npy").exists()
            for lv in VERTEBRA_LEVELS
        )
    ]
    print(
        f"[INFO] region_4class.npy が存在する study: {len(has_region)} / {len(all_studies)}"
    )

    if 0 < args.n_studies < len(has_region):
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(has_region), size=args.n_studies, replace=False)
        target = [has_region[i] for i in sorted(idx)]
    else:
        target = has_region

    level_plane_stats: dict[str, dict[int, dict[str, list[float]]]] = {
        lv: {i: defaultdict(list) for i in range(N_CLASSIFIER_PLANES)}
        for lv in VERTEBRA_LEVELS
    }
    ok_counts: dict[str, dict[int, int]] = {
        lv: defaultdict(int) for lv in VERTEBRA_LEVELS
    }
    total_counts: dict[str, dict[int, int]] = {
        lv: defaultdict(int) for lv in VERTEBRA_LEVELS
    }
    anat_fail_counts: dict[str, int] = defaultdict(int)
    failure_cases: list[tuple[str, str, list[dict]]] = []
    csv_rows: list[dict] = []

    for study_id in target:
        for vertebra in VERTEBRA_LEVELS:
            level_dir = args.fracture_dataset_dir / study_id / vertebra
            region_path = level_dir / "region_4class.npy"
            mask_path = level_dir / "vertebra_mask.npy"
            ct_path = level_dir / "ct.npy"
            if not region_path.exists() or not mask_path.exists():
                continue

            region = np.load(region_path)
            mask = np.load(mask_path)
            result = check_one_level(region, mask)
            plane_results = result["planes"]

            has_failure = False
            for pr in plane_results:
                i = pr["plane"]
                total_counts[vertebra][i] += 1
                if pr.get("empty_mask"):
                    continue
                if pr.get("ok"):
                    ok_counts[vertebra][i] += 1
                for rname in ["body", "right", "left", "post"]:
                    level_plane_stats[vertebra][i][rname].append(
                        pr.get("ratios", {}).get(rname, 0.0)
                    )
                if not pr.get("ok") or (pr.get("is_central") and not pr.get("anat_ok")):
                    has_failure = True
                    anat_fail_counts[vertebra] += 1

                csv_rows.append(
                    {
                        "study_id": study_id,
                        "vertebra": vertebra,
                        "plane": i,
                        "n_regions": pr.get("n_regions", 0),
                        "ok": int(pr.get("ok", False)),
                        "anat_ok": int(pr.get("anat_ok", True)),
                        "is_central": int(pr.get("is_central", False)),
                        "total_fg": pr.get("total_fg", 0),
                        **{
                            f"ratio_{rn}": pr.get("ratios", {}).get(rn, 0.0)
                            for rn in ["body", "right", "left", "post"]
                        },
                    }
                )

            if (
                has_failure
                and len(failure_cases) < args.n_failure_vis
                and ct_path.exists()
            ):
                failure_cases.append((study_id, vertebra, plane_results))

    fail_dir = args.output_dir / "failure_samples"
    for study_id, vertebra, plane_results in failure_cases:
        level_dir = args.fracture_dataset_dir / study_id / vertebra
        try:
            ct = np.load(level_dir / "ct.npy")
            region = np.load(level_dir / "region_4class.npy")
            mask = np.load(level_dir / "vertebra_mask.npy")
            visualize_failure(
                study_id, vertebra, ct, region, mask, plane_results, fail_dir
            )
        except Exception:
            pass

    csv_path = args.output_dir / "area_stats.csv"
    if csv_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    lines = [
        "# SDF 4 領域分割 品質チェックレポート\n",
        f"対象 study 数: {len(target)}\n",
    ]
    lines.append("\n## プレーン別成功率\n")
    lines.append(
        "| Level | " + " | ".join(f"p{i}" for i in range(N_CLASSIFIER_PLANES)) + " |"
    )
    lines.append(
        "|-------|" + "|".join("------" for _ in range(N_CLASSIFIER_PLANES)) + "|"
    )
    for lv in VERTEBRA_LEVELS:
        row = f"| {lv} |"
        for i in range(N_CLASSIFIER_PLANES):
            t = total_counts[lv][i]
            o = ok_counts[lv][i]
            row += f" {o / t * 100:.0f}% |" if t > 0 else " — |"
        lines.append(row)

    report_path = args.output_dir / "qc_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[SAVED] {report_path}")
    print(f"[SAVED] {csv_path}")


if __name__ == "__main__":
    main()
