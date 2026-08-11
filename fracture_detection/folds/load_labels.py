"""fracture_detection研究用のラベル読み込み機能。

ラベルは次の2箇所から読み込む。
- data/rsna_data/train.csv: 椎体単位の骨折ラベル（C1〜C7列の横持ち形式）
- data/rsna_data/fracture_region_labels_dicom.csv: アノテーションrunごとに1行の4領域ラベル

`run`は同じ対象への再アノテーションではない。アノテーションツール
（Unet/dicom_bbox_annotation_tool）は、各椎体の骨折bboxを連続するDICOMスライスの
runにまとめる。そのため、1椎体に2つのrunがある場合、それらは互いに離れた骨折部位であり、
個別に表示・判定されている（run間は5〜50スライス空く）。アノテータは2026-08-07に、
各runのラベルが記録どおり正しいことを確認済みである。

したがって、椎体単位の領域ラベルには、その椎体に属する全runの論理和を用いる。
いずれかの骨折部位が領域rに達していれば、その椎体の領域rを陽性とする。
1つのrunだけを採用すると、実在する骨折部位が暗黙に失われる。

この集約による件数は268 bag、160 study、R1=78、R2=59、R3=72、R4=158である。
以前の文書にあるR1=77、R3=71、R4=155は、骨折部位を落とす最新run採用規則による
旧集計値であり、現在はこの集計値で置き換える。
"""

from pathlib import Path

import pandas as pd

REGION_COLUMNS = ["region_1", "region_2", "region_3", "region_4"]
LEVELS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

EXPECTED_ANNOTATED_BAGS = 268
EXPECTED_ANNOTATED_STUDIES = 160
EXPECTED_REGION_POSITIVES = {
    "region_1": 78,  # 椎体
    "region_2": 59,  # 右横突孔
    "region_3": 72,  # 左横突孔
    "region_4": 158,  # 後方要素
}


def load_region_labels(csv_path: Path) -> pd.DataFrame:
    """全runの論理和を取り、椎体ごとに1行の領域ラベルを読み込む。

    study_id、level、n_runs、region_1〜region_4列を返す。
    件数が確認済みの値と異なる場合はValueErrorを送出する。この不一致はCSVが変更され、
    後続処理で使うすべての数値を再導出する必要があることを意味する。
    """
    raw = pd.read_csv(csv_path)
    df = raw.groupby(["study_id", "level"], as_index=False).agg(
        n_runs=("run_id", "nunique"),
        region_1=("region_1", "max"),
        region_2=("region_2", "max"),
        region_3=("region_3", "max"),
        region_4=("region_4", "max"),
    )
    df = df.sort_values(["study_id", "level"]).reset_index(drop=True)

    if len(df) != EXPECTED_ANNOTATED_BAGS:
        raise ValueError(
            f"想定bag数は{EXPECTED_ANNOTATED_BAGS}ですが、実際は{len(df)}です"
        )
    if df["study_id"].nunique() != EXPECTED_ANNOTATED_STUDIES:
        raise ValueError(
            f"想定study数は{EXPECTED_ANNOTATED_STUDIES}ですが、"
            f"実際は{df['study_id'].nunique()}です"
        )
    for column, expected in EXPECTED_REGION_POSITIVES.items():
        actual = int(df[column].sum())
        if actual != expected:
            raise ValueError(
                f"{column}: 想定陽性数は{expected}ですが、実際は{actual}です"
            )
    return df


def load_vertebra_labels(csv_path: Path) -> pd.DataFrame:
    """train.csvをstudy_id、level、fractured（0/1）の縦持ち形式で読み込む。"""
    wide = pd.read_csv(csv_path)
    long = wide.melt(
        id_vars=["StudyInstanceUID"],
        value_vars=LEVELS,
        var_name="level",
        value_name="fractured",
    )
    long = long.rename(columns={"StudyInstanceUID": "study_id"})
    return long.sort_values(["study_id", "level"]).reset_index(drop=True)
