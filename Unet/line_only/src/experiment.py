"""実験出力パス、wandb、結果表示の管理。"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_epoch_log(
    epoch: int,
    epochs: int,
    learning_rate: float,
    train_stats: dict[str, float],
    val_metrics: dict[str, Any],
    use_line_loss: bool,
    warmup_weight: float,
    elapsed_seconds: float,
) -> str:
    """コンソール用の epoch ログ文字列を構築する。"""
    parts = [
        f"[EPOCH {epoch:03d}/{epochs}]",
        f"lr={learning_rate:.2e}",
        f"train_mse={train_stats['loss']:.6f}",
        f"val_mse={val_metrics['val_loss_mse']:.6f}",
        f"peak={val_metrics['peak_dist_mean']:.2f}px",
    ]
    if use_line_loss:
        parts.extend(
            [
                f"L_ang={train_stats['angle']:.4f}",
                f"L_rho={train_stats['rho']:.4f}",
                f"gate={train_stats['gate']:.2f}",
                f"w={warmup_weight:.2f}",
            ]
        )
    if "angle_error_deg" in val_metrics:
        parts.extend(
            [
                f"angle={val_metrics['angle_error_deg']:.2f}°",
                f"rho={val_metrics['rho_error_px']:.2f}px",
            ]
        )
    parts.append(f"time={elapsed_seconds:.1f}s")
    return "  ".join(parts)


def log_wandb_epoch(
    wandb_module: Any,
    epoch: int,
    learning_rate: float,
    train_stats: dict[str, float],
    val_metrics: dict[str, Any],
    use_line_loss: bool,
    warmup_weight: float,
) -> None:
    """1 epoch 分の指標を wandb へ送信する。"""
    log_values = {
        "epoch": epoch,
        "lr": learning_rate,
        "train_mse": train_stats["loss"],
        "val_mse": val_metrics["val_loss_mse"],
        "peak_dist": val_metrics["peak_dist_mean"],
        "blob_iou": val_metrics["blob_iou"],
        "warmup_weight": warmup_weight,
    }
    if use_line_loss:
        log_values.update(
            {
                "train_L_ang": train_stats["angle"],
                "train_L_rho": train_stats["rho"],
                "train_gate_ratio": train_stats["gate"],
            }
        )
    if "angle_error_deg" in val_metrics:
        log_values.update(
            {
                "angle_error_deg": val_metrics["angle_error_deg"],
                "rho_error_px": val_metrics["rho_error_px"],
            }
        )
    if "val_outlier_angle_rate" in val_metrics:
        log_values.update(
            {
                "val_outlier_angle_rate": val_metrics["val_outlier_angle_rate"],
                "val_outlier_rho_rate": val_metrics["val_outlier_rho_rate"],
            }
        )
    wandb_module.log(log_values, step=epoch)


def update_best_summary(
    wandb_module: Any,
    epoch: int,
    best_value: float,
    val_metrics: dict[str, Any],
) -> None:
    """ベスト epoch の指標を wandb summary へ保存する。"""
    wandb_module.run.summary["best_epoch"] = epoch
    wandb_module.run.summary["best_angle_error_deg"] = best_value
    wandb_module.run.summary["best_val_mse"] = val_metrics["val_loss_mse"]
    wandb_module.run.summary["best_peak_dist"] = val_metrics["peak_dist_mean"]
    if "rho_error_px" in val_metrics:
        wandb_module.run.summary["best_rho_error_px"] = val_metrics["rho_error_px"]


def _get_wandb() -> Any | None:
    """wandb を遅延インポートする。"""
    try:
        import wandb

        return wandb
    except ImportError:
        return None


def _resolve_output_base(
    cfg: dict[str, Any],
    script_dir: Path,
) -> Path | None:
    """experiment 設定があれば実験出力ディレクトリを返す。"""
    experiment_config = cfg.get("experiment")
    if not experiment_config:
        return None
    phase = experiment_config.get("phase")
    name = experiment_config.get("name")
    if not phase or not name:
        return None
    return script_dir / "outputs" / str(phase) / str(name)


def initialize_wandb(
    cfg: dict[str, Any],
    test_fold: int,
) -> tuple[bool, Any | None]:
    """設定に基づいて wandb run を初期化する。"""
    wandb_config = cfg.get("wandb", {})
    if not wandb_config.get("enabled", False):
        return False, None

    wandb_module = _get_wandb()
    if wandb_module is None:
        print(
            "[WARNING] wandb.enabled=true だが wandb が"
            "インストールされていません。ログをスキップします。"
        )
        return False, None

    experiment_config = cfg.get("experiment", {})
    phase = experiment_config.get("phase")
    name = experiment_config.get("name")
    default_project = f"unet-{phase}-{name}" if phase and name else "vai-unet-line"
    wandb_module.init(
        project=wandb_config.get("project") or default_project,
        name=wandb_config.get("run_name") or f"fold{test_fold}",
        config=cfg,
        reinit=True,
    )
    return True, wandb_module


def resolve_fold_paths(
    cfg: dict[str, Any],
    test_fold: int,
) -> tuple[Path, Path, Path]:
    """チェックポイント・可視化・直線出力パスを解決する。"""
    script_dir = Path(__file__).resolve().parent.parent.parent
    output_base = _resolve_output_base(cfg, script_dir)
    training_config = cfg.get("training", {})
    evaluation_config = cfg.get("evaluation", {})

    checkpoint_dir = (
        output_base / "checkpoints"
        if output_base is not None
        else script_dir / training_config.get("checkpoint_dir", "checkpoints")
    )
    visualization_dir = (
        output_base / "vis"
        if output_base is not None
        else script_dir / evaluation_config.get("visualization_dir", "vis")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / f"best_fold{test_fold}.pt"
    line_output_dir = visualization_dir / f"fold{test_fold}" / "test_lines"
    return best_path, visualization_dir, line_output_dir


def print_line_summary(line_summary: dict[str, Any]) -> None:
    """直線幾何評価結果をコンソールへ表示する。"""
    print("\n" + "=" * 60)
    print("[LINE GEOMETRY EVALUATION]")
    print("=" * 60)
    print(
        f"  Perpendicular Distance: {line_summary['perpendicular_dist_px_mean']:.2f} px"
    )
    print(f"  Angle Error:           {line_summary['angle_error_deg_mean']:.2f} deg")
    print(f"  Rho Error:             {line_summary['rho_error_px_mean']:.2f} px")
    print(f"  Outlier Angle Rate:    {line_summary['outlier_angle_rate']:.1%}")
    print(f"  Outlier Rho Rate:      {line_summary['outlier_rho_rate']:.1%}")
    print("\n[Per-Channel Breakdown]")
    for line_name, metrics in line_summary["per_channel"].items():
        print(
            f"  {line_name}: "
            f"perp={metrics['perpendicular_dist_px_mean']:.2f}px  "
            f"angle={metrics['angle_error_deg_mean']:.2f}deg  "
            f"rho={metrics['rho_error_px_mean']:.2f}px  "
            f"(n={metrics['n']})"
        )
    print(f"\n[Output] {line_summary['out_dir']}")
    print("=" * 60)


def finish_wandb(
    wandb_module: Any,
    test_metrics: dict[str, Any],
    line_summary: dict[str, Any],
) -> None:
    """テスト指標を wandb summary へ保存して run を終了する。"""
    wandb_module.run.summary["test_mse"] = test_metrics["val_loss_mse"]
    wandb_module.run.summary["test_peak_dist"] = test_metrics["peak_dist_mean"]
    wandb_module.run.summary["line_perp_dist"] = line_summary[
        "perpendicular_dist_px_mean"
    ]
    wandb_module.run.summary["line_angle_error"] = line_summary["angle_error_deg_mean"]
    wandb_module.run.summary["line_rho_error"] = line_summary["rho_error_px_mean"]
    wandb_module.run.summary["test_outlier_angle_rate"] = line_summary["outlier_angle_rate"]
    wandb_module.run.summary["test_outlier_rho_rate"] = line_summary["outlier_rho_rate"]
    wandb_module.finish()
