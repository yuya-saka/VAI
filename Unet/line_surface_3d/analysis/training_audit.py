"""baseline-v1の学習・評価設計の欠陥を再現検証する。

`.claude/docs/research/line-surface-3d-training-audit.md` の各所見に対応する。

検証項目:

- plane-fit: 現行 `fit_plane` が未アノテーションスライスに引きずられず傾きを回復すること
  （旧 `fit_ribbon` は `valid` を無視して16倍減衰していた。回帰防止用）
- peak-dist: `peak_dist` がリッジ形状の教師に対して退化していること
- multiplicity: 評価が同一スライスを重複カウントすること
- rho-sign: 符号不変rho誤差が原点反対側の失敗を隠しうること
- loss-gradient: sigmoid+MSEが正例で勾配消失すること
- selection: angle単独のcheckpoint選択がrhoへ与える影響
- loss-balance: 幾何3項とheatmap項の勾配比。plane.yaml の重みはこれで決める
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..src.dataset import build_slab_records
from ..utils.plane import extract_gt_line_params, fit_plane

LINE_KEYS = ("line_1", "line_2", "line_3", "line_4")
IMAGE_SIZE = 224
SIGMA = 4.0
SLAB_SIZE = 15
ANNOTATED_RANGE = (4, 10)
KNOWN_SLOPE_PX_PER_SLICE = -1.0


def check_plane_fit() -> dict[str, Any]:
    """`fit_plane` が既知の傾きを回復することを確認する。

    旧 `fit_ribbon` はここで16倍減衰していた。回帰防止のための検査。
    """
    height = width = 64
    heatmaps = torch.zeros(1, SLAB_SIZE, 4, height, width)
    y_grid, _ = torch.meshgrid(
        torch.arange(height).float(),
        torch.arange(width).float(),
        indexing="ij",
    )
    centre = (ANNOTATED_RANGE[0] + ANNOTATED_RANGE[1] - 1) / 2.0
    for slab_index in range(*ANNOTATED_RANGE):
        line_y = 32.0 - KNOWN_SLOPE_PX_PER_SLICE * (slab_index - centre)
        heatmaps[0, slab_index, 0] = torch.exp(
            -((y_grid - line_y) ** 2) / (2.0 * SIGMA**2)
        )

    # 水平線なので共有法線は (0, 1)。slope はそのまま数学座標での drho/dz
    fitted = fit_plane(heatmaps)
    recovered = float(fitted.slope[0, 0])
    return {
        "known_slope_px_per_slice": KNOWN_SLOPE_PX_PER_SLICE,
        "recovered_slope_px_per_slice": recovered,
        "relative_error": abs(recovered - KNOWN_SLOPE_PX_PER_SLICE)
        / abs(KNOWN_SLOPE_PX_PER_SLICE),
    }


def _polyline_heatmap(points: np.ndarray) -> np.ndarray:
    """折れ線からGaussianリッジのheatmapを描く。"""
    y_grid, x_grid = np.meshgrid(
        np.arange(IMAGE_SIZE),
        np.arange(IMAGE_SIZE),
        indexing="ij",
    )
    distance = np.full((IMAGE_SIZE, IMAGE_SIZE), np.inf)
    for start, end in zip(points[:-1], points[1:], strict=False):
        segment = end - start
        length = float(np.hypot(*segment))
        if length < 1e-6:
            continue
        projection = np.clip(
            ((x_grid - start[0]) * segment[0] + (y_grid - start[1]) * segment[1])
            / length**2,
            0.0,
            1.0,
        )
        distance = np.minimum(
            distance,
            np.hypot(
                x_grid - (start[0] + projection * segment[0]),
                y_grid - (start[1] + projection * segment[1]),
            ),
        )
    return np.exp(-(distance**2) / (2.0 * SIGMA**2))


def check_peak_dist(annotation_root: Path) -> dict[str, Any]:
    """リッジ教師に対しargmaxが任意であることを示す。"""
    lines_path = next(iter(sorted(annotation_root.glob("sample*/C*/lines.json"))))
    payload = json.loads(lines_path.read_text())
    key = sorted(payload)[len(payload) // 2]
    heatmap = _polyline_heatmap(np.asarray(payload[key]["line_1"], dtype=np.float64))

    peak = float(heatmap.max())
    at_peak = np.argwhere(heatmap >= peak - 1e-6)
    spread = (
        float(np.hypot(*(at_peak.max(axis=0) - at_peak.min(axis=0))))
        if len(at_peak) > 1
        else 0.0
    )
    return {
        "pixels_at_exact_max": int(len(at_peak)),
        "spread_of_max_pixels_px": spread,
        "pixels_within_0.001": int((heatmap >= peak - 0.001).sum()),
        "pixels_within_0.05": int((heatmap >= peak - 0.05).sum()),
    }


def check_multiplicity(
    config: dict[str, Any], sample_names: list[str]
) -> dict[str, Any]:
    """評価が同一スライスを何回数えるかを測る。"""
    data_config = config["data"]
    records = build_slab_records(
        dense_root=Path(data_config["dense_root"]),
        annotation_root=Path(data_config["annotation_root"]),
        sample_names=sample_names,
        group=str(data_config.get("group", "ALL")),
        slab_size=int(data_config["slab_size"]),
        stride=int(data_config["train_stride"]),
        min_labeled_slices=int(data_config["min_labeled_slices"]),
        require_labels=True,
    )
    per_slice: Counter[tuple[str, str, int]] = Counter()
    for record in records:
        for slice_index in record.slice_indices:
            if slice_index in record.labels:
                per_slice[(record.sample, record.vertebra, slice_index)] += 1

    counts = np.asarray(list(per_slice.values()))
    per_vertebra: Counter[tuple[str, str]] = Counter()
    for (sample, vertebra, _), count in per_slice.items():
        per_vertebra[(sample, vertebra)] += count
    vertebra_counts = np.asarray(sorted(per_vertebra.values(), reverse=True))
    return {
        "windows": len(records),
        "unique_annotated_slices": int(len(per_slice)),
        "counted_observations": int(counts.sum()),
        "multiplicity_mean": float(counts.mean()),
        "multiplicity_min": int(counts.min()),
        "multiplicity_max": int(counts.max()),
        "vertebrae": int(len(per_vertebra)),
        "vertebra_contribution_ratio": float(
            vertebra_counts[0] / max(vertebra_counts[-1], 1)
        ),
    }


def check_rho_sign(annotation_root: Path, max_files: int = 120) -> dict[str, Any]:
    """符号不変rho誤差が隠しうる誤差量を測る。"""
    diagonal = float(np.hypot(IMAGE_SIZE, IMAGE_SIZE))
    magnitudes: list[float] = []
    paths = sorted(annotation_root.glob("sample*/C*/lines.json"))[:max_files]
    for lines_path in paths:
        payload = json.loads(lines_path.read_text())
        for slice_lines in payload.values():
            for line_key in LINE_KEYS:
                _, rho = extract_gt_line_params(slice_lines.get(line_key), IMAGE_SIZE)
                if not np.isnan(rho):
                    magnitudes.append(abs(rho) * diagonal)
    values = np.asarray(magnitudes)
    return {
        "observations": int(values.size),
        "abs_rho_median_px": float(np.median(values)),
        "abs_rho_p90_px": float(np.percentile(values, 90)),
        "hidden_error_median_px": float(2.0 * np.median(values)),
        "hidden_error_p90_px": float(2.0 * np.percentile(values, 90)),
    }


def check_loss_gradient() -> dict[str, Any]:
    """正例画素でのsigmoid+MSEとBCEの勾配を比較する。"""
    target = torch.tensor([1.0])
    rows: list[dict[str, float]] = []
    for logit_value in (-8.0, -6.0, -4.0, -2.0, 0.0):
        mse_logit = torch.tensor([logit_value], requires_grad=True)
        mse_loss = (torch.sigmoid(mse_logit) - target).square().mean()
        bce_logit = torch.tensor([logit_value], requires_grad=True)
        bce_loss = F.binary_cross_entropy_with_logits(bce_logit, target)
        mse_gradient = abs(float(torch.autograd.grad(mse_loss, mse_logit)[0].item()))
        bce_gradient = abs(float(torch.autograd.grad(bce_loss, bce_logit)[0].item()))
        rows.append(
            {
                "logit": logit_value,
                "probability": float(torch.sigmoid(torch.tensor(logit_value))),
                "mse_gradient": mse_gradient,
                "bce_gradient": bce_gradient,
                "ratio": bce_gradient / max(mse_gradient, 1e-30),
            }
        )
    return {"rows": rows}


def check_selection(metrics_dir: Path) -> dict[str, Any]:
    """angle単独選択がrhoへ与える影響と無駄epoch数を測る。"""
    folds: list[dict[str, Any]] = []
    for metrics_path in sorted(metrics_dir.glob("fold*.jsonl")):
        rows = [
            json.loads(line) for line in metrics_path.read_text().splitlines() if line
        ]
        if not rows:
            continue
        angles = np.asarray([row["angle_error_deg"] for row in rows])
        rhos = np.asarray([row["rho_error_px"] for row in rows])
        selected = int(np.argmin(angles))
        best_rho = int(np.argmin(rhos))
        folds.append(
            {
                "fold": metrics_path.stem,
                "selected_epoch": int(rows[selected]["epoch"]),
                "rho_at_selected_px": float(rhos[selected]),
                "best_rho_px": float(rhos[best_rho]),
                "rho_cost_px": float(rhos[selected] - rhos[best_rho]),
                "epochs_run": len(rows),
                "wasted_epochs": len(rows) - selected - 1,
            }
        )
    return {
        "folds": folds,
        "mean_rho_cost_px": float(np.mean([f["rho_cost_px"] for f in folds]))
        if folds
        else float("nan"),
    }


def check_loss_balance(
    config: dict[str, Any],
    sample_names: list[str],
    warmup_epochs: int = 25,
    max_records: int = 40,
) -> dict[str, Any]:
    """幾何3項とheatmap項の勾配ノルムを比較し、推奨重みを返す。

    値ではなく勾配で比べる。3項ともpx単位へ揃えても、heatmapまで遡る
    感度は項ごとに桁違いになるため。
    heatmap項だけで少し学習させてから測る（warmup終了時に相当）。
    """
    from torch.utils.data import DataLoader

    from ..src.dataset import SlabLineDataset
    from ..src.trainer import build_model
    from ..utils.losses import compute_plane_loss, masked_heatmap_mse
    from ..utils.plane import centered_positions

    torch.manual_seed(0)
    data_config = config["data"]
    image_size = int(data_config["image_size"])
    slab_size = int(data_config["slab_size"])
    records = build_slab_records(
        dense_root=Path(data_config["dense_root"]),
        annotation_root=Path(data_config["annotation_root"]),
        sample_names=sample_names,
        group=str(data_config.get("group", "ALL")),
        slab_size=slab_size,
        stride=int(data_config["train_stride"]),
        min_labeled_slices=int(data_config["min_labeled_slices"]),
        require_labels=True,
        image_size=image_size,
    )[:max_records]
    dataset = SlabLineDataset(records, image_size, float(data_config["sigma"]))
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    device = torch.device("cpu")
    model = build_model(config, device)
    positions = centered_positions(slab_size, device, torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for _ in range(warmup_epochs):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch["image"].float(),
                torch.zeros(batch["image"].shape[0], dtype=torch.long),
            )
            prediction = torch.sigmoid(
                logits.reshape(-1, slab_size, 4, image_size, image_size)
            )
            loss = masked_heatmap_mse(
                prediction, batch["heatmaps"].float(), batch["label_mask"].bool()
            )
            trainable = [p for p in model.parameters() if p.requires_grad]
            for parameter, parameter_grad in zip(
                trainable,
                torch.autograd.grad(loss, trainable, allow_unused=True),
                strict=True,
            ):
                parameter.grad = parameter_grad
            optimizer.step()

    def gradient_norm(
        batch: dict[str, Any], term: str, **weights: float
    ) -> tuple[float, float]:
        model.zero_grad(set_to_none=True)
        logits = model(
            batch["image"].float(),
            torch.zeros(batch["image"].shape[0], dtype=torch.long),
        )
        prediction = torch.sigmoid(
            logits.reshape(-1, slab_size, 4, image_size, image_size)
        ).float()
        plane_config: dict[str, Any] = {
            "enabled": True,
            "angle_weight": 0.0,
            "rho_weight": 0.0,
            "tilt_weight": 0.0,
            "fallback_weight": 0.25,
        }
        plane_config.update(weights)
        output = compute_plane_loss(
            prediction,
            batch["heatmaps"].float(),
            batch["label_mask"].bool(),
            batch["line_params_gt"].float(),
            batch["plane_slope_gt"].float(),
            batch["plane_reliable"].bool(),
            positions,
            image_size,
            plane_config,
            geometry_weight=1.0,
        )
        component = getattr(output, term)
        value = float(component.detach())
        parameters = [p for p in model.parameters() if p.requires_grad]
        gradients = torch.autograd.grad(component, parameters, allow_unused=True)
        total = sum(
            float(gradient.norm() ** 2)
            for gradient in gradients
            if gradient is not None
        )
        return total**0.5, value

    # reliable面は疎なので、tilt項の勾配は1バッチだと大きくばらつく。複数で平均する
    settings = {
        "heatmap": {},
        "angle": {"angle_weight": 1.0},
        "rho": {"rho_weight": 1.0},
        "tilt": {"tilt_weight": 1.0},
    }
    gradients: dict[str, list[float]] = {name: [] for name in settings}
    values: dict[str, list[float]] = {name: [] for name in settings}
    for batch in loader:
        for name, weights in settings.items():
            gradient, value = gradient_norm(batch, name, **weights)
            gradients[name].append(gradient)
            values[name].append(value)

    mean_gradients = {
        name: float(np.mean(series)) for name, series in gradients.items()
    }
    heatmap_gradient = mean_gradients["heatmap"]
    return {
        "batches": len(gradients["heatmap"]),
        "warmup_heatmap_loss": float(np.mean(values["heatmap"])),
        "gradient_norms": mean_gradients,
        "gradient_norm_p90": {
            name: float(np.percentile(series, 90)) for name, series in gradients.items()
        },
        "values_at_weight_one": {
            name: float(np.mean(series)) for name, series in values.items()
        },
        "suggested_weights_for_half_of_heatmap": {
            name: 0.5 * heatmap_gradient / (3.0 * max(mean_gradients[name], 1e-12))
            for name in ("angle", "rho", "tilt")
        },
    }


def main() -> None:
    """指定された検証を実行し、結果をJSONで出力する。"""
    from ..src.data_utils import load_config, prepare_splits

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("Unet/line_surface_3d/config/baseline.yaml"),
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("Unet/outputs/line_surface_3d/baseline-v1/metrics"),
    )
    parser.add_argument(
        "--check",
        choices=(
            "all",
            "plane-fit",
            "peak-dist",
            "multiplicity",
            "rho-sign",
            "loss-gradient",
            "selection",
            "loss-balance",
        ),
        default="all",
    )
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    annotation_root = Path(config["data"]["annotation_root"])
    report: dict[str, Any] = {}

    if arguments.check in ("all", "plane-fit"):
        report["plane_fit"] = check_plane_fit()
    if arguments.check in ("all", "peak-dist"):
        report["peak_dist"] = check_peak_dist(annotation_root)
    if arguments.check in ("all", "multiplicity"):
        _, _, test_samples = prepare_splits(config)
        report["multiplicity"] = check_multiplicity(config, test_samples)
    if arguments.check in ("all", "rho-sign"):
        report["rho_sign"] = check_rho_sign(annotation_root)
    if arguments.check in ("all", "loss-gradient"):
        report["loss_gradient"] = check_loss_gradient()
    if arguments.check == "loss-balance":
        _, _, test_samples = prepare_splits(config)
        report["loss_balance"] = check_loss_balance(config, test_samples)
    if arguments.check in ("all", "selection") and arguments.metrics_dir.exists():
        report["selection"] = check_selection(arguments.metrics_dir)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
