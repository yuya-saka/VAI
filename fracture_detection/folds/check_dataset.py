"""統合済み fracture_dataset のフェーズ0データ契約を検証する。

モデル実装前に次の項目を検証する。
1. 領域アノテーション済みの全268 bagを想定した形状・データ型で読み込み、
   SHA256フィンガープリントを付与する（マスク・CTのバージョン固定）。
2. 全bagについてファイルの有無とサイズを棚卸しする（配列は読み込まない）。
3. train.csvと照合し、画像データのないstudyと、実際に学習へ使用できる
   椎体ラベルの母集団を確認する。

出力先（fracture_detection/folds/outputs/）:
- annotated_bag_manifest.csv: 268行のラベル、形状、SHA256フィンガープリント
- bag_inventory.csv: 各(study, level)ディレクトリにつき1行
- check_report.json: 作業ログ用の集計値

実行方法: uv run python fracture_detection/folds/check_dataset.py
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from load_labels import LEVELS, load_region_labels, load_vertebra_labels

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "data/rsna_data/fracture_dataset"
TRAIN_CSV = REPO_ROOT / "data/rsna_data/train.csv"
REGION_CSV = REPO_ROOT / "data/rsna_data/fracture_region_labels_dicom.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

BAG_FILES = ["ct.npy", "vertebra_mask.npy", "region_4class.npy"]
EXPECTED_CT_SHAPE = (15, 5, 224, 224)
EXPECTED_MASK_SHAPE = (15, 224, 224)
EXPECTED_CT_DTYPE = np.uint8
REGION_MASK_ALLOWED_VALUES = {0, 1, 2, 3, 4}


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_dataset() -> pd.DataFrame:
    """全(study, level)ディレクトリのファイル有無とサイズを一覧化する。"""
    rows: list[dict[str, object]] = []
    for study_dir in sorted(DATASET_DIR.iterdir()):
        if not study_dir.is_dir():
            continue
        for level in LEVELS:
            bag_dir = study_dir / level
            if not bag_dir.is_dir():
                continue
            row: dict[str, object] = {"study_id": study_dir.name, "level": level}
            for file_name in BAG_FILES:
                file_path = bag_dir / file_name
                key = file_name.removesuffix(".npy")
                row[f"{key}_bytes"] = (
                    file_path.stat().st_size if file_path.exists() else -1
                )
            rows.append(row)
    return pd.DataFrame(rows)


def check_annotated_bag(bag_dir: Path) -> dict[str, object]:
    """アノテーション済みbagを読み込み、形状検査結果と指紋を返す。"""
    result: dict[str, object] = {}

    ct = np.load(bag_dir / "ct.npy")
    vertebra_mask = np.load(bag_dir / "vertebra_mask.npy")
    region_mask = np.load(bag_dir / "region_4class.npy")

    result["ct_shape_ok"] = ct.shape == EXPECTED_CT_SHAPE
    result["ct_dtype_ok"] = ct.dtype == EXPECTED_CT_DTYPE
    result["vertebra_mask_shape_ok"] = vertebra_mask.shape == EXPECTED_MASK_SHAPE
    result["region_mask_shape_ok"] = region_mask.shape == EXPECTED_MASK_SHAPE
    result["vertebra_mask_nonzero"] = bool(vertebra_mask.any())
    result["region_values_ok"] = set(np.unique(region_mask)).issubset(
        REGION_MASK_ALLOWED_VALUES
    )
    # このbagに存在する領域クラスを記録する。横突孔は一部の面、まれにbag全体に
    # 存在しないため、検査条件として強制はしない。
    for region_value in (1, 2, 3, 4):
        result[f"region_{region_value}_present"] = bool(
            (region_mask == region_value).any()
        )

    for file_name in BAG_FILES:
        key = file_name.removesuffix(".npy")
        result[f"{key}_sha256"] = sha256_of_file(bag_dir / file_name)
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    region_df = load_region_labels(REGION_CSV)
    vertebra_df = load_vertebra_labels(TRAIN_CSV)

    print("== データセットの棚卸し ==")
    inventory = inventory_dataset()
    inventory.to_csv(OUTPUT_DIR / "bag_inventory.csv", index=False)
    complete = inventory[
        (inventory[[f"{f.removesuffix('.npy')}_bytes" for f in BAG_FILES]] > 0).all(
            axis=1
        )
    ]
    print(f"検出したbag数: {len(inventory)}（完備: {len(complete)}）")
    print(f"データが存在するstudy数: {inventory['study_id'].nunique()}")

    print("== アノテーション済みbagの全読み込み検査（268 bag） ==")
    manifest_rows: list[dict[str, object]] = []
    failures: list[str] = []
    for row in region_df.itertuples(index=False):
        bag_dir = DATASET_DIR / row.study_id / row.level
        entry: dict[str, object] = {
            "study_id": row.study_id,
            "level": row.level,
            "n_runs": row.n_runs,
            "region_1": row.region_1,
            "region_2": row.region_2,
            "region_3": row.region_3,
            "region_4": row.region_4,
        }
        if not bag_dir.is_dir():
            failures.append(f"bagディレクトリがありません: {bag_dir}")
            manifest_rows.append(entry)
            continue
        checks = check_annotated_bag(bag_dir)
        entry.update(checks)
        failed_checks = [k for k, v in checks.items() if k.endswith("_ok") and not v]
        if failed_checks:
            failures.append(f"{row.study_id}/{row.level}: {failed_checks}")
        if not checks["vertebra_mask_nonzero"]:
            failures.append(f"{row.study_id}/{row.level}: 椎体マスクが空です")
        manifest_rows.append(entry)
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUTPUT_DIR / "annotated_bag_manifest.csv", index=False)

    print("== train.csvとの照合 ==")
    studies_with_data = set(inventory["study_id"])
    studies_in_train = set(vertebra_df["study_id"])
    missing_studies = sorted(studies_in_train - studies_with_data)
    extra_studies = sorted(studies_with_data - studies_in_train)

    bag_keys = set(zip(inventory["study_id"], inventory["level"], strict=True))
    available = vertebra_df[
        [
            key in bag_keys
            for key in zip(vertebra_df["study_id"], vertebra_df["level"], strict=True)
        ]
    ]
    annotated_in_train = region_df.merge(
        vertebra_df, on=["study_id", "level"], how="left"
    )
    annotated_not_fractured = annotated_in_train[annotated_in_train["fractured"] != 1]
    if len(annotated_not_fractured):
        failures.append(
            f"椎体陽性ラベルのないアノテーション済みbag: {len(annotated_not_fractured)}"
        )

    # 横突孔マスクの有無とラベルを照合する。R2/R3が陽性でも領域マスクに
    # 該当クラスがなければ、そのbagから対象領域を学習できない。
    for region_value, column in ((2, "region_2"), (3, "region_3")):
        labeled = manifest[manifest[column] == 1]
        mask_absent = labeled[~labeled[f"region_{region_value}_present"].fillna(False)]
        if len(mask_absent):
            failures.append(
                f"{column}は陽性ですがマスククラスがありません: "
                f"{len(mask_absent)} bag: "
                + ", ".join(f"{r.study_id}/{r.level}" for r in mask_absent.itertuples())
            )

    report = {
        "bags_total": int(len(inventory)),
        "bags_complete": int(len(complete)),
        "studies_with_data": int(inventory["study_id"].nunique()),
        "studies_in_train_csv": int(len(studies_in_train)),
        "studies_missing_data": missing_studies,
        "studies_not_in_train_csv": extra_studies,
        "vertebra_labels_available": int(len(available)),
        "vertebra_positives_available": int(available["fractured"].sum()),
        "annotated_bags_checked": int(len(manifest)),
        "failures": failures,
    }
    (OUTPUT_DIR / "check_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))
    if failures:
        print(f"検査失敗（{len(failures)}件）:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("すべての検査に合格しました")


if __name__ == "__main__":
    main()
