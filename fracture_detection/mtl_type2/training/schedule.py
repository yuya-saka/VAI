"""1 epoch内でregion更新を行うnatural step位置を決める。

RSNA修正方針の§4「Partial-label学習の方法を変える」に対応する。従来は
natural stepの**毎回**annotated batchを混ぜていたが、詳細症例（約160 bag）が
少ないため同じ症例を何周も学習してしまっていた。ここでは1 epochでannotated
datasetをちょうど1周させ、その`annotated_steps`回のupdateをnatural
step列へ等間隔・決定論的に配置する。
"""

from __future__ import annotations


def region_step_schedule(natural_steps: int, annotated_steps: int) -> list[int]:
    """region更新を行うnatural step index（0始まり）を昇順で返す。

    `annotated_steps`個の位置を`natural_steps`区間へ均等配置する
    （各区間の中点、`round`ではなく`int`切り捨てで決定論的に選ぶ）。
    """
    if natural_steps < 1:
        raise ValueError("natural_stepsは1以上である必要があります")
    if annotated_steps < 1:
        raise ValueError("annotated_stepsは1以上である必要があります")
    if annotated_steps > natural_steps:
        raise ValueError("annotated_stepsがnatural_stepsを超えています")
    positions = sorted(
        {
            min(
                int((index + 0.5) * natural_steps / annotated_steps),
                natural_steps - 1,
            )
            for index in range(annotated_steps)
        }
    )
    if len(positions) != annotated_steps:
        raise ValueError("region step scheduleに重複が生じました")
    return positions
