"""outer fold 0限定のno_pseudo/cam_soft/cam_soft_shuffled 3アーム比較。

`.claude/docs/REGION_MODEL_DESIGN_JA.md` §5.4の最初の受入試験に対応する。3アームは
アーキテクチャ・augmentation・optimizer・schedule・seed・batch size・lambdaを共有し、
`pseudo_arm`（とそれに紐づくpseudo target artifactの有無・対応関係）だけが異なる。

`gate_passed`はcam_softのmacro AP差分の点推定がno_pseudo・cam_soft_shuffledの両方を
上回るかだけを見る単純な符号判定であり、信頼区間がゼロを跨がないことまでは要求しない。
統計的に厳密な採否判定ではなく、残り4 outer foldへ進むかどうかの一次スクリーニングとして
使う。最終判断は保存されたreportを人が読んで行う。
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from fracture_detection.baseline0.evaluation.metrics import (
    safe_auroc,
    safe_average_precision,
)
from fracture_detection.region_branch.cli.compare_arms import (
    paired_cluster_bootstrap_difference,
)
from fracture_detection.region_branch.config.schema import load_config
from fracture_detection.region_branch.data_pipeline.constants import REGION_COLUMNS
from fracture_detection.region_branch.training.experiment import resolve_experiment_root

DEFAULT_NO_PSEUDO_CONFIG = Path(
    "fracture_detection/region_branch/config/region_branch_outer0_no_pseudo.yaml"
)
DEFAULT_CAM_SOFT_CONFIG = Path(
    "fracture_detection/region_branch/config/region_branch_outer0_cam_soft.yaml"
)
DEFAULT_CAM_SOFT_SHUFFLED_CONFIG = Path(
    "fracture_detection/region_branch/config/region_branch_outer0_cam_soft_shuffled.yaml"
)
KEY_COLUMNS = ("study_id", "level")


def load_arm_outer0_predictions(
    config_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """1アームconfigのouter fold 0予測を読み、outer-0限定であることを検証する。"""
    config = load_config(config_path)
    data = config["data"]
    if int(data["start_outer_fold"]) != 0 or int(data["end_outer_fold"]) != 0:
        raise ValueError(
            f"{config_path}はouter fold 0限定の3アーム比較用configである必要があります: "
            f"start={data['start_outer_fold']}, end={data['end_outer_fold']}"
        )
    prediction_path = (
        resolve_experiment_root(config) / "outer0" / "outer_predictions.csv"
    )
    if not prediction_path.is_file():
        raise FileNotFoundError(f"outer0の予測がありません: {prediction_path}")
    predictions = pd.read_csv(prediction_path, dtype={"study_id": str, "level": str})
    if not predictions["fold"].eq(0).all():
        raise ValueError(f"{prediction_path}にfold0以外の行が含まれています")
    return predictions, config


def _assert_identical_population(
    no_pseudo: pd.DataFrame, cam_soft: pd.DataFrame, cam_soft_shuffled: pd.DataFrame
) -> None:
    """3アームの評価母集団（bag集合とhard target）が完全一致することを検証する。"""
    invariant_columns = [
        "vertebra_target",
        *[f"{column}_target" for column in REGION_COLUMNS],
        *[f"{column}_target_valid" for column in REGION_COLUMNS],
    ]
    reference = no_pseudo.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    for name, frame in (
        ("cam_soft", cam_soft),
        ("cam_soft_shuffled", cam_soft_shuffled),
    ):
        other = frame.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
        if len(reference) != len(other) or not reference[list(KEY_COLUMNS)].equals(
            other[list(KEY_COLUMNS)]
        ):
            raise ValueError(f"no_pseudoと{name}のbag集合が一致しません")
        for column in invariant_columns:
            if not reference[column].equals(other[column]):
                raise ValueError(f"no_pseudoと{name}の{column}が一致しません")


def compare_pseudo_arms(
    no_pseudo_config_path: Path,
    cam_soft_config_path: Path,
    cam_soft_shuffled_config_path: Path,
    n_bootstrap: int,
) -> dict[str, Any]:
    """outer fold 0だけでcam_softがno_pseudo/cam_soft_shuffledを上回るかを比較する。"""
    no_pseudo, _no_pseudo_config = load_arm_outer0_predictions(no_pseudo_config_path)
    cam_soft, cam_soft_config = load_arm_outer0_predictions(cam_soft_config_path)
    cam_soft_shuffled, _shuffled_config = load_arm_outer0_predictions(
        cam_soft_shuffled_config_path
    )
    _assert_identical_population(no_pseudo, cam_soft, cam_soft_shuffled)

    active_regions = tuple(int(v) for v in cam_soft_config["region"]["active_regions"])
    key_columns = list(KEY_COLUMNS)
    no_pseudo = no_pseudo.sort_values(key_columns).reset_index(drop=True)
    cam_soft = cam_soft.sort_values(key_columns).reset_index(drop=True)
    cam_soft_shuffled = cam_soft_shuffled.sort_values(key_columns).reset_index(
        drop=True
    )
    groups = cam_soft["study_id"].to_numpy()

    region_results: list[dict[str, Any]] = []
    ap_diffs_vs_no_pseudo: list[float] = []
    ap_diffs_vs_shuffled: list[float] = []
    for region_index in active_regions:
        column = REGION_COLUMNS[region_index]
        valid = cam_soft[f"{column}_target_valid"].to_numpy().astype(bool)
        if not valid.any():
            raise ValueError(f"{column}に有効セルがありません")
        targets = cam_soft.loc[valid, f"{column}_target"].to_numpy()
        cam_soft_scores = cam_soft.loc[valid, f"{column}_score"].to_numpy()
        no_pseudo_scores = no_pseudo.loc[valid, f"{column}_score"].to_numpy()
        shuffled_scores = cam_soft_shuffled.loc[valid, f"{column}_score"].to_numpy()
        region_groups = groups[valid]

        cam_soft_ap = safe_average_precision(targets, cam_soft_scores)
        no_pseudo_ap = safe_average_precision(targets, no_pseudo_scores)
        shuffled_ap = safe_average_precision(targets, shuffled_scores)

        ap_ci_vs_no_pseudo = paired_cluster_bootstrap_difference(
            targets,
            cam_soft_scores,
            no_pseudo_scores,
            region_groups,
            safe_average_precision,
            n_bootstrap,
        )
        ap_ci_vs_shuffled = paired_cluster_bootstrap_difference(
            targets,
            cam_soft_scores,
            shuffled_scores,
            region_groups,
            safe_average_precision,
            n_bootstrap,
        )
        auroc_ci_vs_no_pseudo = paired_cluster_bootstrap_difference(
            targets,
            cam_soft_scores,
            no_pseudo_scores,
            region_groups,
            safe_auroc,
            n_bootstrap,
        )
        auroc_ci_vs_shuffled = paired_cluster_bootstrap_difference(
            targets,
            cam_soft_scores,
            shuffled_scores,
            region_groups,
            safe_auroc,
            n_bootstrap,
        )
        ap_diffs_vs_no_pseudo.append(cam_soft_ap - no_pseudo_ap)
        ap_diffs_vs_shuffled.append(cam_soft_ap - shuffled_ap)
        region_results.append(
            {
                "region": column,
                "n": int(valid.sum()),
                "positives": int(targets.sum()),
                "cam_soft_ap": cam_soft_ap,
                "no_pseudo_ap": no_pseudo_ap,
                "cam_soft_shuffled_ap": shuffled_ap,
                "ap_difference_vs_no_pseudo_ci": ap_ci_vs_no_pseudo,
                "ap_difference_vs_cam_soft_shuffled_ci": ap_ci_vs_shuffled,
                "cam_soft_auroc": safe_auroc(targets, cam_soft_scores),
                "no_pseudo_auroc": safe_auroc(targets, no_pseudo_scores),
                "cam_soft_shuffled_auroc": safe_auroc(targets, shuffled_scores),
                "auroc_difference_vs_no_pseudo_ci": auroc_ci_vs_no_pseudo,
                "auroc_difference_vs_cam_soft_shuffled_ci": auroc_ci_vs_shuffled,
            }
        )

    macro_ap_diff_vs_no_pseudo = float(np.mean(ap_diffs_vs_no_pseudo))
    macro_ap_diff_vs_shuffled = float(np.mean(ap_diffs_vs_shuffled))
    whole_summary = {
        arm: {
            "auroc": safe_auroc(
                frame["vertebra_target"].to_numpy(), frame["vertebra_score"].to_numpy()
            ),
            "ap": safe_average_precision(
                frame["vertebra_target"].to_numpy(), frame["vertebra_score"].to_numpy()
            ),
        }
        for arm, frame in (
            ("no_pseudo", no_pseudo),
            ("cam_soft", cam_soft),
            ("cam_soft_shuffled", cam_soft_shuffled),
        )
    }
    gate_passed = macro_ap_diff_vs_no_pseudo > 0.0 and macro_ap_diff_vs_shuffled > 0.0
    return {
        "regions": region_results,
        "macro_ap_difference_vs_no_pseudo": macro_ap_diff_vs_no_pseudo,
        "macro_ap_difference_vs_cam_soft_shuffled": macro_ap_diff_vs_shuffled,
        "gate_passed": gate_passed,
        "whole": whole_summary,
        "n_bootstrap": n_bootstrap,
    }


def parse_args() -> argparse.Namespace:
    """CLI引数を解釈する。"""
    parser = argparse.ArgumentParser(
        description="outer fold 0限定のno_pseudo/cam_soft/cam_soft_shuffled比較"
    )
    parser.add_argument(
        "--no-pseudo-config", type=Path, default=DEFAULT_NO_PSEUDO_CONFIG
    )
    parser.add_argument("--cam-soft-config", type=Path, default=DEFAULT_CAM_SOFT_CONFIG)
    parser.add_argument(
        "--cam-soft-shuffled-config",
        type=Path,
        default=DEFAULT_CAM_SOFT_SHUFFLED_CONFIG,
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    """CLIのエントリポイント。"""
    args = parse_args()
    results = compare_pseudo_arms(
        args.no_pseudo_config,
        args.cam_soft_config,
        args.cam_soft_shuffled_config,
        args.n_bootstrap,
    )
    cam_soft_config = load_config(args.cam_soft_config)
    output_root = resolve_experiment_root(cam_soft_config)
    output_path = output_root / "compare_pseudo_arms.json"
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for row in results["regions"]:
        print(
            f"{row['region']}: cam_soft_AP={row['cam_soft_ap']:.6f} "
            f"no_pseudo_AP={row['no_pseudo_ap']:.6f} "
            f"cam_soft_shuffled_AP={row['cam_soft_shuffled_ap']:.6f} "
            f"diff_vs_no_pseudo_CI={row['ap_difference_vs_no_pseudo_ci']} "
            f"diff_vs_shuffled_CI={row['ap_difference_vs_cam_soft_shuffled_ci']}",
            flush=True,
        )
    print(
        f"macro AP diff vs no_pseudo={results['macro_ap_difference_vs_no_pseudo']:.6f}, "
        f"vs cam_soft_shuffled={results['macro_ap_difference_vs_cam_soft_shuffled']:.6f}, "
        f"gate_passed={results['gate_passed']}",
        flush=True,
    )
    print(f"比較結果を保存しました: {output_path}", flush=True)


if __name__ == "__main__":
    main()
