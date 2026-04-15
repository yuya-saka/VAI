"""訓練ループ・評価・テスト評価パイプライン（Seg+SDF版）"""

import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..utils.losses import compute_sdf_seg_loss, extract_gt_line_params
from ..utils.metrics import (
    compute_perpendicular_distance,
    compute_sdf_boundary_iou,
    compute_sdf_mae,
    compute_seg_metrics,
)
from ..utils.detection import LinesJsonCache, detect_line_from_sdf, line_extent
from ..utils.visualization import save_sdf_grid, save_seg_overlay
from .data_utils import (
    create_data_loaders,
    create_model_optimizer_scheduler,
    prepare_datasets_and_splits,
)
from .model import VERTEBRA_TO_IDX

tempfile.tempdir = "/tmp"


def _resolve_output_base(cfg: dict[str, Any], script_dir: Path) -> Path | None:
    """experiment セクションが存在すればベースパスを返す。なければ None。

    outputs/{phase}/{name}/ をベースとして使用する。
    """
    exp = cfg.get("experiment")
    if exp and exp.get("phase") and exp.get("name"):
        return script_dir / "outputs" / exp["phase"] / exp["name"]
    return None


def _get_wandb():
    """wandb を遅延インポート（無効時にインストール不要にするため）"""
    try:
        import wandb

        return wandb
    except ImportError:
        return None


def _extract_seg_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """セグ教師を batch から取得する（互換キーを許容）。"""
    if "gt_mask" in batch:
        gt_mask = batch["gt_mask"].to(device).long()
    else:
        gt_mask = batch["gt_region_mask"].to(device).long()

    if "has_seg_label" in batch:
        has_seg_label = batch["has_seg_label"].to(device).bool()
    else:
        has_seg_label = batch["has_gt_region_mask"].to(device).bool()

    return gt_mask, has_seg_label


def _sdf_result_to_phi_rho(result: dict, image_size: int) -> tuple[float, float]:
    """SDF検出結果の辞書から (phi, rho) パラメータに変換する。

    angle_rad（数学座標系の方向角）から直線の法線方向を求め、
    重心の画像座標から rho（正規化済み）を計算する。

    引数:
        result: detect_line_from_sdf の戻り値辞書
        image_size: 画像サイズ（正方形を仮定）

    戻り値:
        (phi_rad, rho_normalized)
    """
    theta = result["angle_rad"]
    vx = math.cos(theta)
    vy = math.sin(theta)
    # 方向ベクトルを90度回転して法線ベクトルを得る
    nx, ny = -vy, vx
    # phi を [0, pi) に制限
    if ny < 0 or (ny == 0 and nx < 0):
        nx, ny = -nx, -ny
    phi = math.atan2(ny, nx)
    # 重心を画像座標系 → 数学座標系に変換
    cx_img, cy_img = result["centroid"]
    center = image_size / 2.0
    xbar = cx_img - center
    ybar = -(cy_img - center)
    rho = nx * xbar + ny * ybar
    D = math.sqrt(image_size**2 + image_size**2)
    return float(phi), float(rho / D)


# -------------------------
# 評価関数
# -------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    cfg: dict[str, Any],
    image_size: int = 224,
) -> dict[str, Any]:
    """モデル評価を実行し、SDF/seg の主要メトリクスを返す。"""
    model.eval()

    lambda_sdf = float(cfg.get("loss", {}).get("lambda_sdf", 3.0))

    total_loss_sum = 0.0
    sdf_loss_sum = 0.0
    seg_loss_sum = 0.0
    wsdf_loss_sum = 0.0
    sdf_mae_sum = 0.0
    sdf_biou_sum = 0.0
    n = 0

    seg_miou_sum = 0.0
    seg_dice_sum = 0.0
    seg_fg_miou_sum = 0.0
    seg_fg_mdice_sum = 0.0
    seg_miou_count = 0
    per_class_dice: dict[str, list[float]] = {}
    per_class_iou_acc: dict[str, list[float]] = {}

    for batch in loader:
        x = batch["image"].to(device).float()
        gt_sdf = batch["sdf_field"].to(device).float()
        gt_mask, has_seg_label = _extract_seg_batch(batch, device)

        v_idx = torch.as_tensor(
            [VERTEBRA_TO_IDX.get(v, 0) for v in batch["vertebra"]],
            device=device,
            dtype=torch.long,
        )
        out = model(x, v_idx)

        loss_dict = compute_sdf_seg_loss(
            seg_logits=out["seg_logits"],
            sdf_field=out["sdf_field"],
            gt_mask=gt_mask,
            gt_sdf=gt_sdf,
            has_seg_label=has_seg_label,
            lambda_sdf=lambda_sdf,
        )

        total_loss_sum += loss_dict["total"].item()
        sdf_loss_sum += loss_dict["raw_sdf_loss"].item()
        seg_loss_sum += loss_dict["raw_seg_loss"].item()
        wsdf_loss_sum += loss_dict["weighted_sdf_loss"].item()

        sdf_mae_sum += compute_sdf_mae(out["sdf_field"], gt_sdf)
        sdf_biou_sum += compute_sdf_boundary_iou(out["sdf_field"], gt_sdf)
        n += 1

        if has_seg_label.any():
            seg_m = compute_seg_metrics(
                out["seg_logits"][has_seg_label],
                gt_mask[has_seg_label],
            )
            lc = int(has_seg_label.sum().item())
            seg_miou_sum += seg_m["miou"] * lc
            seg_dice_sum += seg_m["dice"] * lc
            seg_fg_miou_sum += seg_m["fg_miou"] * lc
            seg_fg_mdice_sum += seg_m["fg_mdice"] * lc
            seg_miou_count += lc
            for cls_name, vals in seg_m["per_class"].items():
                per_class_dice.setdefault(cls_name, []).append(vals["dice"])
                per_class_iou_acc.setdefault(cls_name, []).append(vals["iou"])

    return {
        "val_loss": total_loss_sum / max(1, n),
        "val_sdf_loss": sdf_loss_sum / max(1, n),
        "val_seg_loss": seg_loss_sum / max(1, n),
        "val_weighted_sdf_loss": wsdf_loss_sum / max(1, n),
        "sdf_mae": sdf_mae_sum / max(1, n),
        "sdf_boundary_iou": sdf_biou_sum / max(1, n),
        "seg_miou": (
            float(seg_miou_sum / seg_miou_count)
            if seg_miou_count > 0
            else float("nan")
        ),
        "seg_dice": (
            float(seg_dice_sum / seg_miou_count)
            if seg_miou_count > 0
            else float("nan")
        ),
        "seg_fg_miou": (
            float(seg_fg_miou_sum / seg_miou_count)
            if seg_miou_count > 0
            else float("nan")
        ),
        "seg_fg_mdice": (
            float(seg_fg_mdice_sum / seg_miou_count)
            if seg_miou_count > 0
            else float("nan")
        ),
        "per_class": (
            {
                cls_name: {
                    "dice": float(np.mean(dices)),
                    "iou": float(np.mean(per_class_iou_acc[cls_name])),
                }
                for cls_name, dices in per_class_dice.items()
            }
            if per_class_dice
            else {}
        ),
    }


# -------------------------
# 訓練ループ
# -------------------------
def run_training_loop(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    scheduler,
    train_loader,
    val_loader,
    device: torch.device,
    cfg: dict[str, Any],
    best_path: Path,
    wandb_enabled: bool = False,
    _wandb=None,
) -> None:
    """訓練ループを実行（早期停止あり）。"""
    tr_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})
    loss_cfg = cfg.get("loss", {})

    epochs = int(tr_cfg.get("epochs", 20))
    es_pat = int(tr_cfg.get("early_stopping_patience", 20))
    grad_clip = float(tr_cfg.get("grad_clip", 1.0))
    image_size = int(cfg.get("data", {}).get("image_size", 224))
    lambda_sdf = float(loss_cfg.get("lambda_sdf", 3.0))

    mfreq = int(eval_cfg.get("metrics_frequency", 1))
    if mfreq <= 0:
        mfreq = 1

    best_val = float("inf")
    no_improve = 0

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()

        train_loss_sum = 0.0
        train_sdf_sum = 0.0
        train_seg_sum = 0.0
        train_wsdf_sum = 0.0
        steps = 0

        for batch in train_loader:
            x = batch["image"].to(device).float()
            gt_sdf = batch["sdf_field"].to(device).float()
            gt_mask, has_seg_label = _extract_seg_batch(batch, device)

            v_idx = torch.as_tensor(
                [VERTEBRA_TO_IDX.get(v, 0) for v in batch["vertebra"]],
                device=device,
                dtype=torch.long,
            )
            out = model(x, v_idx)
            loss_dict = compute_sdf_seg_loss(
                seg_logits=out["seg_logits"],
                sdf_field=out["sdf_field"],
                gt_mask=gt_mask,
                gt_sdf=gt_sdf,
                has_seg_label=has_seg_label,
                lambda_sdf=lambda_sdf,
            )
            loss = loss_dict["total"]

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            train_loss_sum += loss.item()
            train_sdf_sum += loss_dict["raw_sdf_loss"].item()
            train_seg_sum += loss_dict["raw_seg_loss"].item()
            train_wsdf_sum += loss_dict["weighted_sdf_loss"].item()
            steps += 1

        train_loss = train_loss_sum / max(1, steps)
        train_sdf = train_sdf_sum / max(1, steps)
        train_seg = train_seg_sum / max(1, steps)
        train_wsdf = train_wsdf_sum / max(1, steps)

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            cfg,
            image_size=image_size,
        )

        scheduler.step(val_metrics["val_loss"])

        cur_lr = opt.param_groups[0]["lr"]
        elapsed = int(round(time.time() - t0))
        print(
            f"[EPOCH {ep:03d}/{epochs}] "
            f"lr={cur_lr:.2e} "
            f"train={train_loss:.6f} "
            f"val={val_metrics['val_loss']:.6f} "
            f"sdf={val_metrics['val_sdf_loss']:.6f} "
            f"seg={val_metrics['val_seg_loss']:.6f} "
            f"sdf_mae={val_metrics['sdf_mae']:.4f} "
            f"sdf_biou={val_metrics['sdf_boundary_iou']:.4f} "
            f"seg_miou={val_metrics['seg_miou']:.4f} "
            f"time={elapsed}s"
        )

        if wandb_enabled and _wandb is not None:
            _wandb.log(
                {
                    "epoch": ep,
                    "lr": cur_lr,
                    "train_loss": train_loss,
                    "train_raw_sdf_loss": train_sdf,
                    "train_raw_seg_loss": train_seg,
                    "train_weighted_sdf_loss": train_wsdf,
                    "val_loss": val_metrics["val_loss"],
                    "val_sdf_loss": val_metrics["val_sdf_loss"],
                    "val_seg_loss": val_metrics["val_seg_loss"],
                    "val_weighted_sdf_loss": val_metrics["val_weighted_sdf_loss"],
                    "sdf_mae": val_metrics["sdf_mae"],
                    "sdf_boundary_iou": val_metrics["sdf_boundary_iou"],
                    "seg_miou": val_metrics["seg_miou"],
                    "seg_dice": val_metrics["seg_dice"],
                },
                step=ep,
            )

        if val_metrics["val_loss"] < best_val - 1e-8:
            best_val = val_metrics["val_loss"]
            no_improve = 0
            torch.save(
                {"model": model.state_dict(), "cfg": cfg, "val": val_metrics},
                best_path,
            )
            print(f"  [SAVE] best -> {best_path} (val_loss={best_val:.6f})")

            if wandb_enabled and _wandb is not None:
                _wandb.run.summary["best_val_loss"] = best_val
                _wandb.run.summary["best_epoch"] = ep
                _wandb.run.summary["best_val_sdf_loss"] = val_metrics["val_sdf_loss"]
                _wandb.run.summary["best_val_seg_loss"] = val_metrics["val_seg_loss"]
                _wandb.run.summary["best_sdf_mae"] = val_metrics["sdf_mae"]
                _wandb.run.summary["best_sdf_boundary_iou"] = val_metrics["sdf_boundary_iou"]
                _wandb.run.summary["best_seg_miou"] = val_metrics["seg_miou"]
                _wandb.run.summary["best_seg_dice"] = val_metrics["seg_dice"]
        else:
            no_improve += 1
            if no_improve >= es_pat:
                print(
                    f"[EARLY STOP] no improvement for {es_pat} epochs. "
                    f"best_val={best_val:.6f}"
                )
                break


# -------------------------
# テストデータに対する直線予測と評価
# -------------------------
@torch.no_grad()
def predict_lines_and_eval_test(
    cfg: dict[str, Any],
    model: nn.Module,
    test_loader,
    device: torch.device,
    dataset_root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """テストデータに対する直線検出と統一評価を実行する。"""
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = float(cfg.get("evaluation", {}).get("line_extend_ratio", 1.10))
    zero_thr = float(cfg.get("evaluation", {}).get("sdf_zero_threshold", 0.1))
    image_size = int(cfg.get("data", {}).get("image_size", 224))

    cache = LinesJsonCache(Path(dataset_root))

    angle_errors: list[float] = []
    rho_errors: list[float] = []
    perp_dists: list[float] = []

    per_ch = {
        1: {"angle": [], "rho": [], "perp": []},
        2: {"angle": [], "rho": [], "perp": []},
        3: {"angle": [], "rho": [], "perp": []},
        4: {"angle": [], "rho": [], "perp": []},
    }

    per_vertebra = {
        v: {"angle": [], "rho": [], "perp": []}
        for v in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    }
    saved = 0

    for batch in test_loader:
        x = batch["image"].to(device).float()

        v_idx = torch.as_tensor(
            [VERTEBRA_TO_IDX.get(v, 0) for v in batch["vertebra"]],
            device=device,
            dtype=torch.long,
        )
        out = model(x, v_idx)
        # sdf_field は線形出力（sigmoid 不要）
        pred_sdf = out["sdf_field"]

        x_np = x.cpu().numpy()
        pred_sdf_np = pred_sdf.cpu().numpy()

        batch_size = pred_sdf_np.shape[0]
        for i in range(batch_size):
            sample = batch["sample"][i]
            vertebra = batch["vertebra"][i]
            slice_idx = int(batch["slice_idx"][i])
            ct01 = x_np[i, 0]
            name = f"{sample}_{vertebra}_slice{slice_idx:03d}"
            gt_lines = cache.get_lines_for_slice(sample, vertebra, slice_idx) or {}

            pred_lines_out: dict[str, Any] = {}
            metrics_out: dict[str, Any] = {}

            for c in range(4):
                k = f"line_{c + 1}"
                gt_pts = gt_lines.get(k, None)

                gt_phi, gt_rho = extract_gt_line_params(gt_pts, image_size)

                # SDF ゼロ交差ベースで直線検出
                result = detect_line_from_sdf(
                    pred_sdf_np[i, c],
                    image_size=image_size,
                    zero_threshold=zero_thr,
                    extend_ratio=ext,
                )

                if result is not None:
                    pred_phi, pred_rho = _sdf_result_to_phi_rho(result, image_size)
                else:
                    pred_phi, pred_rho = float("nan"), float("nan")

                pred_lines_out[k] = result

                if not np.isnan(gt_phi) and result is not None:
                    # 角度誤差（度数法、180度対称）
                    cos_diff = abs(math.cos(pred_phi - gt_phi))
                    cos_diff = min(cos_diff, 1.0)
                    angle_err = math.degrees(math.acos(cos_diff))

                    # rho 誤差（ピクセル単位、符号不変）
                    D = math.sqrt(image_size**2 + image_size**2)
                    err1 = abs(pred_rho - gt_rho) * D
                    err2 = abs(pred_rho + gt_rho) * D
                    rho_err = min(err1, err2)

                    # 垂直距離
                    perp_dist = compute_perpendicular_distance(
                        gt_pts, pred_phi, pred_rho, image_size
                    )

                    angle_errors.append(angle_err)
                    rho_errors.append(rho_err)
                    perp_dists.append(perp_dist)

                    per_ch[c + 1]["angle"].append(angle_err)
                    per_ch[c + 1]["rho"].append(rho_err)
                    per_ch[c + 1]["perp"].append(perp_dist)

                    if vertebra in per_vertebra:
                        per_vertebra[vertebra]["angle"].append(angle_err)
                        per_vertebra[vertebra]["rho"].append(rho_err)
                        per_vertebra[vertebra]["perp"].append(perp_dist)

                    metrics_out[k] = {
                        "angle_error_deg": float(angle_err),
                        "rho_error_px": float(rho_err),
                        "perpendicular_dist_px": float(perp_dist),
                        "gt_phi": float(gt_phi),
                        "gt_rho": float(gt_rho),
                        "pred_phi": float(pred_phi),
                        "pred_rho": float(pred_rho),
                    }
                else:
                    metrics_out[k] = {
                        "angle_error_deg": None,
                        "rho_error_px": None,
                        "perpendicular_dist_px": None,
                    }

            # SDF グリッドを保存
            save_sdf_grid(ct01, pred_sdf_np[i], out_dir / f"{name}_PRED_sdf_grid.png")

            with open(out_dir / f"{name}_PRED_lines.json", "w") as f:
                json.dump(
                    {
                        "pred_lines": {
                            k: v if v is None else {
                                key: val
                                for key, val in v.items()
                                if not isinstance(val, np.ndarray)
                            }
                            for k, v in pred_lines_out.items()
                        },
                        "metrics": metrics_out,
                        "sdf_zero_threshold": zero_thr,
                    },
                    f,
                    indent=2,
                )

            saved += 1

    def _mean(vals: list[float | None]) -> float:
        valid_vals = [
            x
            for x in vals
            if x is not None and not (isinstance(x, float) and np.isnan(x))
        ]
        return float(np.mean(valid_vals)) if valid_vals else float("nan")

    summary = {
        "n_samples": int(saved),
        "angle_error_deg_mean": _mean(angle_errors),
        "rho_error_px_mean": _mean(rho_errors),
        "perpendicular_dist_px_mean": _mean(perp_dists),
        "per_channel": {
            f"line_{k}": {
                "angle_error_deg_mean": _mean(v["angle"]),
                "rho_error_px_mean": _mean(v["rho"]),
                "perpendicular_dist_px_mean": _mean(v["perp"]),
                "n": int(len(v["angle"])),
            }
            for k, v in per_ch.items()
        },
        "per_vertebra": {
            v: {
                "angle_error_deg_mean": _mean(vals["angle"]) if vals["angle"] else None,
                "rho_error_px_mean": _mean(vals["rho"]) if vals["rho"] else None,
                "perpendicular_dist_px_mean": (
                    _mean(vals["perp"]) if vals["perp"] else None
                ),
                "n": int(len(vals["angle"])),
            }
            for v, vals in per_vertebra.items()
        },
        "sdf_zero_threshold": float(zero_thr),
        "line_extend_ratio": float(ext),
        "out_dir": str(out_dir),
    }
    return summary


# -------------------------
# サンプル画像保存
# -------------------------
@torch.no_grad()
def save_examples(
    model: nn.Module,
    loader,
    device: torch.device,
    out_dir: Path,
    n_save: int = 12,
    tag: str = "VAL",
) -> None:
    """サンプル画像（SDF グリッド）を保存する。"""
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for batch in loader:
        x = batch["image"].to(device).float()
        gt_sdf = batch["sdf_field"].to(device).float()

        v_idx = torch.as_tensor(
            [VERTEBRA_TO_IDX.get(v, 0) for v in batch["vertebra"]],
            device=device,
            dtype=torch.long,
        )
        out = model(x, v_idx)

        x_np = x.cpu().numpy()
        gt_np = gt_sdf.cpu().numpy()
        pred_np = out["sdf_field"].cpu().numpy()

        batch_size = x_np.shape[0]
        for i in range(batch_size):
            ct01 = x_np[i, 0]
            name = (
                f"{batch['sample'][i]}_{batch['vertebra'][i]}_"
                f"slice{int(batch['slice_idx'][i]):03d}"
            )

            save_sdf_grid(ct01, gt_np[i], out_dir / f"{tag}_{name}_GT_sdf_grid.png")
            save_sdf_grid(ct01, pred_np[i], out_dir / f"{tag}_{name}_PRED_sdf_grid.png")

            saved += 1
            if saved >= n_save:
                return


def save_seg_examples(
    model: nn.Module,
    loader,
    device: torch.device,
    out_dir: Path,
    tag: str = "TEST",
) -> None:
    """C1~C7を網羅したセグメンテーション可視化を専用フォルダに保存する。"""
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device).float()
            gt_mask, has_seg_label = _extract_seg_batch(batch, device)

            v_idx = torch.as_tensor(
                [VERTEBRA_TO_IDX.get(v, 0) for v in batch["vertebra"]],
                device=device,
                dtype=torch.long,
            )
            out = model(x, v_idx)
            pred_mask = out["seg_logits"].argmax(dim=1)

            x_np = x.cpu().numpy()
            pred_mask_np = pred_mask.cpu().numpy()

            batch_size = x_np.shape[0]
            for i in range(batch_size):
                if not bool(has_seg_label[i].item()):
                    continue
                ct01 = x_np[i, 0]
                name = (
                    f"{batch['sample'][i]}_{batch['vertebra'][i]}_"
                    f"slice{int(batch['slice_idx'][i]):03d}"
                )
                save_seg_overlay(
                    ct=ct01,
                    seg_pred=pred_mask_np[i].astype(np.int32),
                    save_path=out_dir / f"{tag}_{name}_seg.png",
                )


# -------------------------
# 1 Fold 訓練メイン関数
# -------------------------
def train_one_fold(cfg: dict[str, Any]) -> dict[str, Any]:
    """1つの fold に対して学習・評価・可視化を実行する。"""
    data_cfg = cfg.get("data", {})
    tr_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})

    test_fold = int(data_cfg.get("test_fold", 0))

    train_s, val_s, test_s, root_dir, group, image_size, sdf_tau, seed = (
        prepare_datasets_and_splits(cfg)
    )

    train_loader, val_loader, test_loader = create_data_loaders(
        train_s,
        val_s,
        test_s,
        root_dir,
        group,
        image_size,
        sdf_tau,
        seed,
        cfg,
    )

    gpu_id = int(tr_cfg.get("gpu_id", 0))
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    wandb_cfg = cfg.get("wandb", {})
    wandb_enabled = bool(wandb_cfg.get("enabled", False))
    _wandb = None
    if wandb_enabled:
        _wandb = _get_wandb()
        if _wandb is None:
            print(
                "[WARNING] wandb.enabled=true だが wandb がインストールされていません。"
                "ログをスキップします。"
            )
            wandb_enabled = False
        else:
            run_name = wandb_cfg.get("run_name") or f"fold{test_fold}"
            exp = cfg.get("experiment", {})
            if exp.get("phase") and exp.get("name"):
                default_project = f"unet-{exp['phase']}-{exp['name']}"
            else:
                default_project = "vai-unet-seg-sdf"
            _wandb.init(
                project=wandb_cfg.get("project") or default_project,
                name=run_name,
                config=cfg,
                reinit=True,
            )

    model, opt, scheduler = create_model_optimizer_scheduler(cfg, device)

    # script_dir は seg_sdf/ ディレクトリ
    script_dir = Path(__file__).resolve().parent.parent
    output_base = _resolve_output_base(cfg, script_dir)
    if output_base is not None:
        ckpt_dir = output_base / "checkpoints"
    else:
        ckpt_dir = script_dir / tr_cfg.get("checkpoint_dir", "checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"best_fold{test_fold}.pt"

    run_training_loop(
        model,
        opt,
        scheduler,
        train_loader,
        val_loader,
        device,
        cfg,
        best_path,
        wandb_enabled=wandb_enabled,
        _wandb=_wandb,
    )

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
    else:
        print(
            "[WARNING] No best checkpoint saved (no improvement during training). "
            "Using current model state."
        )

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        cfg,
        image_size=image_size,
    )
    print(
        f"[TEST] fold={test_fold} "
        f"loss={test_metrics['val_loss']:.6f} "
        f"sdf={test_metrics['val_sdf_loss']:.6f} "
        f"seg={test_metrics['val_seg_loss']:.6f} "
        f"seg_miou={test_metrics['seg_miou']:.4f} "
        f"sdf_mae={test_metrics['sdf_mae']:.4f} "
        f"sdf_biou={test_metrics['sdf_boundary_iou']:.4f}"
    )

    if output_base is not None:
        vis_base = output_base / "vis"
    else:
        vis_base = script_dir / cfg.get("evaluation", {}).get("visualization_dir", "vis")

    out_dir = vis_base / f"fold{test_fold}" / "test_lines"
    line_summary = predict_lines_and_eval_test(
        cfg=cfg,
        model=model,
        test_loader=test_loader,
        device=device,
        dataset_root=root_dir,
        out_dir=out_dir,
    )
    seg_vis_dir = vis_base / f"fold{test_fold}" / "test_seg"
    save_seg_examples(model, test_loader, device, seg_vis_dir, tag="TEST")
    print(f"[INFO] seg overlays saved to {seg_vis_dir}/")

    print("\n" + "=" * 60)
    print("[SEG+SDF EVALUATION]")
    print("=" * 60)
    print(f"  Seg mIoU:              {test_metrics['seg_miou']:.4f}")
    print(f"  Seg Dice:              {test_metrics['seg_dice']:.4f}")
    print(f"  SDF MAE:               {test_metrics['sdf_mae']:.4f}")
    print(f"  SDF Boundary IoU:      {test_metrics['sdf_boundary_iou']:.4f}")
    print(f"  Perpendicular Distance: {line_summary['perpendicular_dist_px_mean']:.2f} px")
    print(f"  Angle Error:           {line_summary['angle_error_deg_mean']:.2f} deg")
    print(f"  Rho Error:             {line_summary['rho_error_px_mean']:.2f} px")
    print("\n[Per-Channel Breakdown]")
    for k, v in line_summary["per_channel"].items():
        print(
            f"  {k}: perp={v['perpendicular_dist_px_mean']:.2f}px "
            f"angle={v['angle_error_deg_mean']:.2f}deg "
            f"rho={v['rho_error_px_mean']:.2f}px (n={v['n']})"
        )
    print(f"\n[Output] {line_summary['out_dir']}")
    print("=" * 60)

    print("[INFO] saving example overlays ...")
    save_examples(
        model,
        val_loader,
        device,
        vis_base / f"fold{test_fold}" / "val",
        n_save=16,
        tag="VAL",
    )
    save_examples(
        model,
        test_loader,
        device,
        vis_base / f"fold{test_fold}" / "test",
        n_save=16,
        tag="TEST",
    )
    print(f"[INFO] saved to {vis_base}/")

    if wandb_enabled and _wandb is not None:
        _wandb.run.summary["test_loss"] = test_metrics["val_loss"]
        _wandb.run.summary["test_sdf_loss"] = test_metrics["val_sdf_loss"]
        _wandb.run.summary["test_seg_loss"] = test_metrics["val_seg_loss"]
        _wandb.run.summary["test_seg_miou"] = test_metrics["seg_miou"]
        _wandb.run.summary["test_seg_dice"] = test_metrics["seg_dice"]
        _wandb.run.summary["test_sdf_mae"] = test_metrics["sdf_mae"]
        _wandb.run.summary["test_sdf_boundary_iou"] = test_metrics["sdf_boundary_iou"]
        _wandb.run.summary["line_perp_dist"] = line_summary["perpendicular_dist_px_mean"]
        _wandb.run.summary["line_angle_error"] = line_summary["angle_error_deg_mean"]
        _wandb.run.summary["line_rho_error"] = line_summary["rho_error_px_mean"]
        _wandb.finish()

    return {
        "test_loss": test_metrics["val_loss"],
        "test_sdf_loss": test_metrics["val_sdf_loss"],
        "test_seg_loss": test_metrics["val_seg_loss"],
        "test_seg_miou": test_metrics["seg_miou"],
        "test_seg_dice": test_metrics["seg_dice"],
        "test_seg_fg_miou": test_metrics["seg_fg_miou"],
        "test_seg_fg_mdice": test_metrics["seg_fg_mdice"],
        "seg_miou": test_metrics["seg_miou"],
        "test_sdf_mae": test_metrics["sdf_mae"],
        "test_sdf_boundary_iou": test_metrics["sdf_boundary_iou"],
        "line_perpendicular_dist_px_mean": line_summary["perpendicular_dist_px_mean"],
        "line_angle_error_deg_mean": line_summary["angle_error_deg_mean"],
        "line_rho_error_px_mean": line_summary["rho_error_px_mean"],
        "per_vertebra": line_summary.get("per_vertebra", {}),
        "per_class": test_metrics.get("per_class", {}),
    }
