"""raw λ/β artifactから正式loss weightsを一度だけ生成するCLI。"""

from __future__ import annotations

import argparse
from pathlib import Path

from fracture_detection.core.artifacts import combine_loss_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="λ/β校正artifactの結合")
    parser.add_argument("--lambda-artifact", type=Path, required=True)
    parser.add_argument("--beta-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combine_loss_weights(args.lambda_artifact, args.beta_artifact, args.output)
    print(f"loss weightsを凍結しました: {args.output}")


if __name__ == "__main__":
    main()
