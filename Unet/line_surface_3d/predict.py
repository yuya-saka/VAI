#!/usr/bin/env python
"""best checkpointで全高推論と領域形成評価を行うCLI。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parent
UNET_DIR = PROJECT_DIR.parent
if str(UNET_DIR) not in sys.path:
    sys.path.insert(0, str(UNET_DIR))

from line_surface_3d.src.data_utils import (  # noqa: E402
    create_inference_loader,
    discover_samples,
    load_config,
    prepare_splits,
)
from line_surface_3d.src.experiment import fold_paths  # noqa: E402
from line_surface_3d.src.inference import predict_loader  # noqa: E402
from line_surface_3d.src.trainer import (  # noqa: E402
    CHECKPOINT_PROTOCOL,
    build_model,
)
from line_surface_3d.utils.region_eval import (  # noqa: E402
    evaluate_prediction_tree,
)


def parse_args() -> argparse.Namespace:
    """CLI引数を解析する。"""
    parser = argparse.ArgumentParser(description="3D line surface全高推論")
    parser.add_argument(
        "--config",
        default="Unet/line_surface_3d/config/baseline.yaml",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--split",
        choices=("test", "all"),
        default="test",
    )
    return parser.parse_args()


def main() -> None:
    """checkpointを検証して推論・領域評価を行う。"""
    args = parse_args()
    config = load_config(args.config)
    config["data"]["test_fold"] = args.fold
    paths = fold_paths(config, args.fold)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else paths["checkpoint"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpointがありません: {checkpoint_path}")
    gpu_id = int(config["training"].get("gpu_id", 0))
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if checkpoint.get("protocol") != CHECKPOINT_PROTOCOL:
        raise ValueError("checkpoint protocolが一致しません")
    if int(checkpoint.get("slab_size", -1)) != int(config["data"]["slab_size"]):
        raise ValueError("checkpointとconfigのslab_sizeが一致しません")
    model = build_model(config, device)
    model.load_state_dict(checkpoint["model"])
    if args.split == "test":
        _, _, sample_names = prepare_splits(config)
    else:
        sample_names = discover_samples(
            Path(config["data"]["dense_root"]),
            Path(config["data"]["annotation_root"]),
        )
    loader, inference_manifest_hash = create_inference_loader(
        config,
        sample_names,
    )
    prediction_root = paths["prediction"] / args.split
    inference_summary = predict_loader(
        model,
        loader,
        device,
        config,
        prediction_root,
    )
    evaluation_config = config.get("evaluation", {})
    region_summary = evaluate_prediction_tree(
        prediction_root=prediction_root,
        dense_root=Path(config["data"]["dense_root"]),
        annotation_root=Path(config["data"]["annotation_root"]),
        output_root=paths["visualization"] / args.split,
        image_size=int(config["data"]["image_size"]),
        spacing_mm=float(evaluation_config.get("voxel_spacing_mm", 0.4)),
        bin_width_mm=float(evaluation_config.get("distance_bin_mm", 3.2)),
    )
    print(
        "[DONE] "
        f"windows={inference_summary['window_count']} "
        f"manifest={inference_manifest_hash[:12]} "
        f"outside_missing="
        f"{region_summary['scopes'].get('outside', {}).get('any_missing_rate')}"
    )


if __name__ == "__main__":
    main()
