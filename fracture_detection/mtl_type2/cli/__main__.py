"""`python -m fracture_detection.mtl_type2.cli --arm <arm> --outer-fold <n> --gpu-id <n>`のentry point。

正式パイプラインの`project_entry.py`と違い`train`/`profile`/`calibrate`の
subcommandは持たない。この探索projectには学習しかないため直接
`cli.train.main`へ委譲する。
"""

from __future__ import annotations

from fracture_detection.mtl_type2.cli.train import main as train_main


def main() -> None:
    """`cli.train.main`（`--arm`必須）へ委譲する。"""
    train_main()


if __name__ == "__main__":
    main()
