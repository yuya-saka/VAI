"""
実験出力パス、wandb、ログの管理。

Unet/line_only/src/experiment.py のパターンに準拠。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


# -------------------------
# 出力パス解決
# -------------------------
def resolve_output_base(cfg: dict, root: Path) -> Path:
    """
    実験出力ベースディレクトリを解決する。

    experiment.phase と experiment.name が設定されている場合:
        {root}/train_models/stage1/outputs/{phase}/{name}/
    設定がない場合:
        {root}/train_models/stage1/outputs/default/
    """
    exp_cfg = cfg.get("experiment", {})
    phase = exp_cfg.get("phase")
    name = exp_cfg.get("name")
    base = root / "train_models" / "stage1" / "outputs"
    if phase and name:
        return base / str(phase) / str(name)
    return base / "default"


def resolve_fold_paths(
    cfg: dict, fold: int, root: Path
) -> tuple[Path, Path]:
    """
    fold固有のチェックポイントパスとfoldディレクトリを解決する。

    Returns:
        (best_model_path, fold_dir)
    """
    output_base = resolve_output_base(cfg, root)
    fold_dir = output_base / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_path = fold_dir / "best_model.pt"
    return best_path, fold_dir


# -------------------------
# ログ管理
# -------------------------
_LOG_HEADER = (
    f"{'fold':>4} {'epoch':>5} {'train_loss':>10} "
    f"{'val_loss':>9} {'auroc':>7} {'auprc':>7} "
    f"{'f1':>6} {'lr':>10} {'sec':>6} {'best':>5}"
)


def write_log_header(log_path: Path, fold: int) -> None:
    """training.log のヘッダーを書く。"""
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# Fold {fold} training log\n")
        f.write(_LOG_HEADER + "\n")
        f.write("-" * len(_LOG_HEADER) + "\n")


def append_log(
    log_path: Path,
    fold: int,
    epoch: int,
    train_loss: float,
    val_loss: float,
    auroc: float,
    auprc: float,
    f1: float,
    lr: float,
    elapsed: float,
    is_best: bool,
) -> None:
    """1 epoch 分のログ行を追記する。"""
    line = (
        f"{fold:>4} {epoch:>5} {train_loss:>10.4f} "
        f"{val_loss:>9.4f} {auroc:>7.4f} {auprc:>7.4f} "
        f"{f1:>6.4f} {lr:>10.2e} {elapsed:>6.0f} "
        f"{'*' if is_best else '':>5}"
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_epoch_log(
    epoch: int,
    epochs: int,
    lr: float,
    train_loss: float,
    val_loss: float,
    metrics: dict,
    elapsed: float,
) -> str:
    """コンソール用の epoch ログ文字列を構築する。"""
    auroc = metrics.get("auroc", float("nan"))
    auprc = metrics.get("auprc", float("nan"))
    f1 = metrics.get("at_05", {}).get("f1", float("nan"))
    return (
        f"[EPOCH {epoch:03d}/{epochs}] "
        f"lr={lr:.2e}  "
        f"train={train_loss:.4f}  val={val_loss:.4f}  "
        f"AUROC={auroc:.4f}  AUPRC={auprc:.4f}  F1={f1:.4f}  "
        f"time={elapsed:.1f}s"
    )


# -------------------------
# wandb
# -------------------------
def _get_wandb() -> Any | None:
    """wandb を遅延インポートする。"""
    try:
        import wandb
        return wandb
    except ImportError:
        return None


def initialize_wandb(cfg: dict, fold: int) -> tuple[bool, Any | None]:
    """設定に基づいて wandb run を初期化する。"""
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return False, None

    w = _get_wandb()
    if w is None:
        print("[WARNING] wandb.enabled=true だが wandb がインストールされていません。")
        return False, None

    exp_cfg = cfg.get("experiment", {})
    phase = exp_cfg.get("phase", "stage1")
    name = exp_cfg.get("name", "run")
    default_project = f"{phase}-{name}"

    w.init(
        project=wandb_cfg.get("project") or default_project,
        name=wandb_cfg.get("run_name") or f"fold{fold}",
        config=cfg,
        reinit=True,
    )
    return True, w


def log_wandb_epoch(
    w: Any,
    epoch: int,
    lr: float,
    train_loss: float,
    val_loss: float,
    metrics: dict,
) -> None:
    """1 epoch 分の指標を wandb へ送信する。"""
    w.log(
        {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "auroc": metrics.get("auroc", float("nan")),
            "auprc": metrics.get("auprc", float("nan")),
            "f1_05": metrics.get("at_05", {}).get("f1", float("nan")),
            "precision_05": metrics.get("at_05", {}).get("precision", float("nan")),
            "recall_05": metrics.get("at_05", {}).get("recall", float("nan")),
        },
        step=epoch,
    )


def finish_wandb(w: Any, fold_metrics: dict) -> None:
    """fold最終指標を wandb summary へ保存して run を終了する。"""
    w.run.summary["val_auroc"] = fold_metrics.get("auroc", float("nan"))
    w.run.summary["val_auprc"] = fold_metrics.get("auprc", float("nan"))
    w.run.summary["val_f1"] = fold_metrics.get("at_05", {}).get("f1", float("nan"))
    w.finish()
