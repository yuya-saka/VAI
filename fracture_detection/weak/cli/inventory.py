"""Read-only pre-training audit: N/A/U counts and per-bag region mask coverage.

Run once before training, per ``fracture_detection/REGION_MIL_DESIGN.md``
section 4's "全領域が観測可能であることを前提とする...症例は無効として件数
を報告する" requirement. This does NOT re-litigate whether the selected 15
planes clinically cover every fracture finding (the user confirmed that
separately) -- it only checks the structural fact the model depends on: does
every bag's region label map actually contain each of the 4 regions on at
least one plane. A bag failing this check would make that (bag, region) pair
globally unobserved for the model, which the design requires to be reported,
not silently treated as q=0.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fracture_detection.baseline0.data.constants import DATASET_DIR
from fracture_detection.baseline0.data.dataset import load_manifest
from fracture_detection.weak.data_pipeline.constants import (
    N_REGIONS,
    REGION_MASK_FILENAME,
)
from fracture_detection.weak.data_pipeline.groups import (
    build_group_inventory,
    resolve_bag_groups,
)


def scan_region_mask_coverage(
    manifest: Any, dataset_dir: Path
) -> tuple[int, list[dict[str, Any]]]:
    """Return (bags scanned, list of bags missing >=1 region on every plane)."""
    resolved = resolve_bag_groups(manifest)
    missing_regions: list[dict[str, Any]] = []
    for _, row in resolved.iterrows():
        mask_path = (
            dataset_dir
            / str(row["study_id"])
            / str(row["level"])
            / REGION_MASK_FILENAME
        )
        region_mask = np.load(mask_path, allow_pickle=False)
        observed_anywhere = [
            bool(np.any(region_mask == region)) for region in range(1, N_REGIONS + 1)
        ]
        if not all(observed_anywhere):
            missing_regions.append(
                {
                    "study_id": row["study_id"],
                    "level": row["level"],
                    "bag_group": row["bag_group"],
                    "observed_regions": observed_anywhere,
                }
            )
    return len(resolved), missing_regions


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="weak region-MILの学習前inventory監査")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--skip-mask-scan",
        action="store_true",
        help="件数集計のみ行い、13,432 bagの全数mask読込みをskipする",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    manifest = load_manifest()
    inventory = build_group_inventory(manifest)
    print(
        f"全fold: N={inventory.n_negative:,} A={inventory.n_annotated:,} "
        f"U={inventory.n_weak:,} annotated_cells={inventory.n_annotated_cells:,} "
        f"positive_cells={inventory.n_annotated_positive_cells:,}",
        flush=True,
    )

    report: dict[str, Any] = {"inventory": inventory.as_dict()}
    if not args.skip_mask_scan:
        print(
            "4領域maskの全数被覆スキャンを実行しています（時間がかかります）",
            flush=True,
        )
        scanned, missing = scan_region_mask_coverage(manifest, args.dataset_dir)
        print(
            f"scan完了: {scanned:,} bag中、1領域以上が全面で観測不能なbag: {len(missing):,}",
            flush=True,
        )
        report["mask_coverage"] = {"scanned": scanned, "missing_region_bags": missing}

    if args.output is not None:
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"reportを保存しました: {args.output}", flush=True)


if __name__ == "__main__":
    main()
