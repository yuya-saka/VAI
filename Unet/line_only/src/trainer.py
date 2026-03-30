"""訓練ループ・評価・テスト評価パイプライン"""

import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import losses as line_losses
from ..utils import metrics as line_metrics
from ..utils.detection import LinesJsonCache, detect_line_moments, line_extent
from ..utils.visualization import (
    draw_heatmap_with_lines,
    draw_line_comparison,
    save_heatmap_grid,
    save_heatmap_overlay,
)
from .data_utils import (
    create_data_loaders,
    create_model_optimizer_scheduler,
    prepare_datasets_and_splits,
)
from .model import VERTEBRA_TO_IDX

tempfile.tempdir = "/tmp"


def _get_wandb():
    """wandb を遅延インポート（無効時にインストール不要にするため）"""
    try:
        import wandb

        return wandb
    except ImportError:
        return None


# -------------------------
# 評価指標
# -------------------------
def peak_dist(pred, gt):
    """ヒートマップピーク間距離（ヒートマップ品質のデバッグ用）"""
    gy, gx = np.unravel_index(np.argmax(gt), gt.shape)
    py, px = np.unravel_index(np.argmax(pred), pred.shape)
    return math.sqrt((px - gx) ** 2 + (py - gy) ** 2)


@torch.no_grad()
def evaluate(model, loader, device, image_size=224, heatmap_threshold=0.2):
    """
    モデルの評価を実行

    引数:
        model: 評価するモデル
        loader: データローダー
        device: 計算デバイス
        image_size: 画像サイズ
        heatmap_threshold: 評価時のヒートマップ閾値（デフォルト0.2）
    """
    model.eval()
    mse_sum = 0.0
    n = 0

    peak_dists = []

    # Line metrics
    angle_errors = []
    rho_errors = []

    # Per-vertebra statistics
    vertebrae = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    per_vertebra = {v: {"peak_dists": []} for v in vertebrae}

    for batch in loader:
        x = batch["image"].to(device).float()
        v_idx = torch.as_tensor(
            [VERTEBRA_TO_IDX.get(v, 0) for v in batch["vertebra"]],
            device=device,
            dtype=torch.long,
        )
        gt = batch["heatmaps"].to(device).float()
        gt_params = batch.get("line_params_gt")

        pred = torch.sigmoid(model(x, v_idx))

        mse_sum += F.mse_loss(pred, gt, reduction="mean").item()
        n += 1

        pr = pred.cpu().numpy()
        g = gt.cpu().numpy()

        B = pr.shape[0]
        for i in range(B):
            v_name = batch["vertebra"][i]  # 椎体名を取得
            for c in range(4):
                pd = peak_dist(pr[i, c], g[i, c])
                peak_dists.append(pd)

                # Per-vertebra statistics
                if v_name in per_vertebra:
                    per_vertebra[v_name]["peak_dists"].append(pd)

        # Compute line metrics
        if gt_params is not None:
            gt_params = gt_params.to(device).float()
            pred_params, confidence = line_losses.extract_pred_line_params_batch(
                pred, image_size, threshold=heatmap_threshold
            )

            gt_valid = ~torch.isnan(gt_params).any(dim=-1)
            pred_valid = confidence > 0
            valid_mask = gt_valid & pred_valid

            angle_err = line_metrics.compute_angle_error(
                pred_params, gt_params, valid_mask
            )
            rho_err = line_metrics.compute_rho_error(
                pred_params, gt_params, image_size, valid_mask
            )

            angle_errors.append(angle_err)
            rho_errors.append(rho_err)

    # Per-vertebra statistics
    per_vert_stats = {}
    for v, vals in per_vertebra.items():
        if len(vals["peak_dists"]) > 0:
            per_vert_stats[v] = {
                "peak_dist_mean": float(np.nanmean(vals["peak_dists"])),
                "n_samples": len(vals["peak_dists"]) // 4,  # 4 channels
            }

    metrics = {
        "val_loss_mse": mse_sum / max(1, n),
        "peak_dist_mean": float(np.nanmean(peak_dists)),
        "per_vertebra": per_vert_stats,
    }

    # Add line metrics if available
    if angle_errors:
        metrics["angle_error_deg"] = float(np.nanmean(angle_errors))
        metrics["rho_error_px"] = float(np.nanmean(rho_errors))

    return metrics


# -------------------------
# 訓練ループ
# -------------------------
def run_training_loop(
    model, opt, scheduler, train_loader, val_loader, device, cfg, best_path,
    wandb_enabled=False, _wandb=None,
):
    """
    訓練ループを実行（早期停止機能付き）

    引数:
        model: PyTorchモデル
        opt: オプティマイザー
        scheduler: 学習率スケジューラー
        train_loader: 訓練用DataLoader
        val_loader: 検証用DataLoader
        device: PyTorchデバイス
        cfg: 設定辞書
        best_path: ベストモデル保存パス
        wandb_enabled: wandb ログを有効にするか
        _wandb: wandb モジュール（遅延インポート済み）
    """
    tr_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})
    loss_cfg = cfg.get("loss", {})

    epochs = int(tr_cfg.get("epochs", 20))
    es_pat = int(tr_cfg.get("early_stopping_patience", 20))
    grad_clip = float(tr_cfg.get("grad_clip", 1.0))
    image_size = int(cfg.get("data", {}).get("image_size", 224))
    heatmap_threshold = float(eval_cfg.get("heatmap_threshold", 0.2))

    # Line loss configuration（旧キーへのフォールバック付き）
    use_line_loss = loss_cfg.get("use_line_loss", False)
    lambda_angle = float(loss_cfg.get("lambda_angle", loss_cfg.get("lambda_theta", 1.0)))
    lambda_rho = float(loss_cfg.get("lambda_rho", 1.0))
    warmup_epochs = int(loss_cfg.get("warmup_epochs", 10))
    warmup_start_epoch = int(loss_cfg.get("warmup_start_epoch", 0))
    warmup_mode = loss_cfg.get("warmup_mode", "linear")
    conf_gate_low = float(loss_cfg.get("confidence_gate_low", 0.3))
    conf_gate_high = float(loss_cfg.get("confidence_gate_high", 0.6))

    best_val = float("inf")
    no_improve = 0

    # 評価指標の計算頻度（オプション）
    mfreq = int(eval_cfg.get("metrics_frequency", 1))  # 1=毎エポック、0=毎エポック
    if mfreq <= 0:
        mfreq = 1

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        steps = 0

        # Compute warmup weight
        warmup_weight = line_losses.get_warmup_weight(
            ep, warmup_epochs, warmup_mode, warmup_start_epoch
        )

        ang_sum = 0.0
        rho_sum = 0.0
        gate_sum = 0.0

        for batch in train_loader:
            x = batch["image"].to(device).float()
            v_idx = torch.as_tensor(
                [VERTEBRA_TO_IDX.get(v, 0) for v in batch["vertebra"]],
                device=device,
                dtype=torch.long,
            )
            gt = batch["heatmaps"].to(device).float()
            gt_params = batch.get("line_params_gt")

            pred = torch.sigmoid(model(x, v_idx))

            # MSE loss
            loss_mse = F.mse_loss(pred, gt, reduction="mean")

            # Line losses
            if use_line_loss and gt_params is not None:
                gt_params = gt_params.to(device).float()
                line_loss_dict = line_losses.compute_line_loss(
                    pred,
                    gt_params,
                    image_size,
                    lambda_angle=lambda_angle,
                    lambda_rho=lambda_rho,
                    use_line_loss=True,
                    confidence_gate_low=conf_gate_low,
                    confidence_gate_high=conf_gate_high,
                )
                # MSE + w * L_line（MSE 重みは常に 1.0）
                loss = loss_mse + warmup_weight * line_loss_dict["total"]
                ang_sum += line_loss_dict["angle"].item()
                rho_sum += line_loss_dict["rho"].item()
                gate_sum += line_loss_dict["gate_ratio"].item()
            else:
                # MSE only
                loss = loss_mse

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            loss_sum += loss.item()
            steps += 1

        train_loss = loss_sum / max(1, steps)
        train_ang = ang_sum / max(1, steps)
        train_rho = rho_sum / max(1, steps)
        train_gate = gate_sum / max(1, steps)

        # 評価
        if ep % mfreq == 0:
            val_metrics = evaluate(model, val_loader, device, image_size, heatmap_threshold)
        else:
            val_metrics = evaluate(model, val_loader, device, image_size, heatmap_threshold)

        # validation lossに基づいてスケジューラを更新
        scheduler.step(val_metrics["val_loss_mse"])

        cur_lr = opt.param_groups[0]["lr"]
        log_str = (
            f"[EPOCH {ep:03d}/{epochs}] lr={cur_lr:.2e} "
            f"train_mse={train_loss:.6f}  "
            f"val_mse={val_metrics['val_loss_mse']:.6f}  "
            f"peak={val_metrics['peak_dist_mean']:.2f}px  "
        )

        if use_line_loss:
            log_str += (
                f"L_ang={train_ang:.4f}  "
                f"L_rho={train_rho:.4f}  "
                f"gate={train_gate:.2f}  "
                f"w={warmup_weight:.2f}  "
            )

        if "angle_error_deg" in val_metrics:
            log_str += (
                f"angle={val_metrics['angle_error_deg']:.2f}°  "
                f"rho={val_metrics['rho_error_px']:.2f}px  "
            )

        log_str += f"time={time.time() - t0:.1f}s"
        print(log_str)

        # wandb にメトリクスを記録
        if wandb_enabled and _wandb is not None:
            log_dict = {
                "epoch": ep,
                "lr": cur_lr,
                "train_mse": train_loss,
                "val_mse": val_metrics["val_loss_mse"],
                "peak_dist": val_metrics["peak_dist_mean"],
                "warmup_weight": warmup_weight,
            }
            if use_line_loss:
                log_dict["train_L_ang"] = train_ang
                log_dict["train_L_rho"] = train_rho
                log_dict["train_gate_ratio"] = train_gate
            if "angle_error_deg" in val_metrics:
                log_dict["angle_error_deg"] = val_metrics["angle_error_deg"]
                log_dict["rho_error_px"] = val_metrics["rho_error_px"]
            _wandb.log(log_dict, step=ep)

        # val_mseによる早期停止
        if val_metrics["val_loss_mse"] < best_val - 1e-8:
            best_val = val_metrics["val_loss_mse"]
            no_improve = 0
            torch.save(
                {"model": model.state_dict(), "cfg": cfg, "val": val_metrics}, best_path
            )
            print(f"  [SAVE] best -> {best_path} (val_mse={best_val:.6f})")
            if wandb_enabled and _wandb is not None:
                _wandb.run.summary["best_val_mse"] = best_val
                _wandb.run.summary["best_epoch"] = ep
                _wandb.run.summary["best_peak_dist"] = val_metrics["peak_dist_mean"]
                if "angle_error_deg" in val_metrics:
                    _wandb.run.summary["best_angle_error_deg"] = val_metrics["angle_error_deg"]
                    _wandb.run.summary["best_rho_error_px"] = val_metrics["rho_error_px"]
        else:
            no_improve += 1
            if no_improve >= es_pat:
                print(
                    f"[EARLY STOP] no improvement for {es_pat} epochs. best_val={best_val:.6f}"
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
    """
    テストデータに対する直線検出と評価（バリデーションと統一）

    処理内容:
        - GT: polylineから直接(φ, ρ)を抽出
        - 予測: ヒートマップから(φ, ρ)を抽出
        - 評価: line_metricsの関数を使用（バリデーションと同じ）
        - 描画: moments法で端点を計算（可視化用）
        - 保存: オーバーレイ画像(PNG) + 検出結果(JSON)

    戻り値:
        サマリー辞書（平均誤差、チャンネル別統計など）
    """
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = float(cfg.get("evaluation", {}).get("line_extend_ratio", 1.10))
    hm_thr = float(
        cfg.get("evaluation", {}).get("heatmap_threshold", 0.15)
    )  # 参照用（強制閾値ではない）
    image_size = int(cfg.get("data", {}).get("image_size", 224))

    cache = LinesJsonCache(Path(dataset_root))

    # Line params-based metrics (統一評価)
    angle_errors = []
    rho_errors = []
    perp_dists = []

    per_ch = {
        1: {"angle": [], "rho": [], "perp": []},
        2: {"angle": [], "rho": [], "perp": []},
        3: {"angle": [], "rho": [], "perp": []},
        4: {"angle": [], "rho": [], "perp": []},
    }

    # 椎体ごとの統計用辞書
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
        pred = torch.sigmoid(model(x, v_idx))

        # Extract pred line params for entire batch
        pred_params, confidence = line_losses.extract_pred_line_params_batch(
            pred, image_size, threshold=hm_thr
        )

        x_np = x.cpu().numpy()
        pr_np = pred.cpu().numpy()
        pred_params_np = pred_params.cpu().numpy()
        conf_np = confidence.cpu().numpy()

        B = pr_np.shape[0]
        for i in range(B):
            sample = batch["sample"][i]
            vertebra = batch["vertebra"][i]
            slice_idx = int(batch["slice_idx"][i])
            ct01 = x_np[i, 0]

            name = f"{sample}_{vertebra}_slice{slice_idx:03d}"
            gt_lines = cache.get_lines_for_slice(sample, vertebra, slice_idx) or {}

            pred_lines_out = {}
            metrics_out = {}

            for c in range(4):
                k = f"line_{c + 1}"

                # GT折れ線
                gt_pts = gt_lines.get(k, None)

                # GT line params from polyline (統一評価)
                gt_phi, gt_rho = line_losses.extract_gt_line_params(gt_pts, image_size)

                # Pred line params
                pred_phi = pred_params_np[i, c, 0]
                pred_rho = pred_params_np[i, c, 1]
                pred_conf = conf_np[i, c]

                # GT線長（描画長さの参照）：最遠点間距離でV字2倍カウントを回避
                Lgt = line_extent(gt_pts)
                if Lgt <= 1e-6:
                    Lgt = None

                # ヒートマップモーメントから予測（描画用）
                pred_info = detect_line_moments(
                    pr_np[i, c], length_px=Lgt, extend_ratio=ext
                )
                pred_lines_out[k] = pred_info

                # 統一評価メトリクス（confidence > 0 で有効判定）
                if not np.isnan(gt_phi) and pred_conf > 0:
                    # Angle error
                    gt_params_single = torch.tensor([[gt_phi, gt_rho]], dtype=torch.float32)
                    pred_params_single = torch.tensor([[pred_phi, pred_rho]], dtype=torch.float32)
                    valid_mask = torch.tensor([True])

                    angle_err = line_metrics.compute_angle_error(
                        pred_params_single, gt_params_single, valid_mask
                    )
                    rho_err = line_metrics.compute_rho_error(
                        pred_params_single, gt_params_single, image_size, valid_mask
                    )
                    perp_dist = line_metrics.compute_perpendicular_distance(
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

            # GT/予測の比較画像を保存（2パネル版）
            draw_line_comparison(
                ct01, pred_lines_out, gt_lines, out_dir / f"{name}_comparison.png"
            )

            # ヒートマップ・予測線・GT線の3パネル版を保存
            draw_heatmap_with_lines(
                ct01,
                pr_np[i],  # (4,H,W) ヒートマップ
                pred_lines_out,
                gt_lines,
                out_dir / f"{name}_heatmap_lines.png",
            )

            with open(out_dir / f"{name}_PRED_lines.json", "w") as f:
                json.dump(
                    {
                        "pred_lines": pred_lines_out,
                        "metrics": metrics_out,
                        "heatmap_threshold_ref": hm_thr,
                    },
                    f,
                    indent=2,
                )

            saved += 1

    def _mean(vals):
        v = [
            x
            for x in vals
            if x is not None and not (isinstance(x, float) and np.isnan(x))
        ]
        return float(np.mean(v)) if len(v) else float("nan")

    summary = {
        "n_samples": int(saved),
        # 統一評価メトリクス
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
                "perpendicular_dist_px_mean": _mean(vals["perp"]) if vals["perp"] else None,
                "n": int(len(vals["angle"])),
            }
            for v, vals in per_vertebra.items()
        },
        "line_extend_ratio": float(ext),
        "heatmap_threshold_ref": float(hm_thr),
        "out_dir": str(out_dir),
    }
    return summary


# -------------------------
# サンプル画像保存
# -------------------------
@torch.no_grad()
def save_examples(model, loader, device, out_dir: Path, n_save=12, tag="VAL"):
    """サンプル画像（ヒートマップグリッド・オーバーレイ）を保存"""
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for batch in loader:
        x = batch["image"].to(device).float()
        v_idx = torch.as_tensor(
            [VERTEBRA_TO_IDX.get(v, 0) for v in batch["vertebra"]],
            device=device,
            dtype=torch.long,
        )
        gt = batch["heatmaps"].to(device).float()
        pred = torch.sigmoid(model(x, v_idx))

        x_np = x.cpu().numpy()
        gt_np = gt.cpu().numpy()
        pr_np = pred.cpu().numpy()

        B = x_np.shape[0]
        for i in range(B):
            ct01 = x_np[i, 0]  # CT
            name = f"{batch['sample'][i]}_{batch['vertebra'][i]}_slice{int(batch['slice_idx'][i]):03d}"

            save_heatmap_grid(ct01, gt_np[i], out_dir / f"{tag}_{name}_GT_grid.png")
            save_heatmap_grid(
                ct01, pr_np[i], out_dir / f"{tag}_{name}_PRED_grid.png"
            )
            save_heatmap_overlay(
                ct01, gt_np[i], out_dir / f"{tag}_{name}_GT_merged.png"
            )
            save_heatmap_overlay(
                ct01, pr_np[i], out_dir / f"{tag}_{name}_PRED_merged.png"
            )

            saved += 1
            if saved >= n_save:
                return


# -------------------------
# 1 Fold 訓練メイン関数
# -------------------------
def train_one_fold(cfg):
    """
    1つのfoldに対する訓練を実行

    戻り値:
        dict: テスト結果の辞書
    """
    # 設定の取得
    data_cfg = cfg.get("data", {})
    tr_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})
    test_fold = int(data_cfg.get("test_fold", 0))
    heatmap_threshold = float(eval_cfg.get("heatmap_threshold", 0.2))

    # データセット準備と分割
    train_s, val_s, test_s, root_dir, group, image_size, sigma, seed = (
        prepare_datasets_and_splits(cfg)
    )

    # データローダー作成
    train_loader, val_loader, test_loader = create_data_loaders(
        train_s, val_s, test_s, root_dir, group, image_size, sigma, seed, cfg
    )

    # デバイス設定
    gpu_id = int(tr_cfg.get("gpu_id", 0))
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    # wandb 初期化
    wandb_cfg = cfg.get("wandb", {})
    wandb_enabled = wandb_cfg.get("enabled", False)
    _wandb = None
    if wandb_enabled:
        _wandb = _get_wandb()
        if _wandb is None:
            print("[WARNING] wandb.enabled=true だが wandb がインストールされていません。ログをスキップします。")
            wandb_enabled = False
        else:
            run_name = wandb_cfg.get("run_name") or f"fold{test_fold}"
            _wandb.init(
                project=wandb_cfg.get("project", "vai-unet-line"),
                name=run_name,
                config=cfg,
                reinit=True,
            )

    # モデル、最適化器、スケジューラー作成
    model, opt, scheduler = create_model_optimizer_scheduler(cfg, device)

    # チェックポイントディレクトリ作成（Unetディレクトリを基準に）
    script_dir = Path(__file__).resolve().parent.parent.parent  # Unet/ directory
    ckpt_dir = script_dir / tr_cfg.get("checkpoint_dir", "checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"best_fold{test_fold}.pt"

    # 訓練ループ実行
    run_training_loop(
        model, opt, scheduler, train_loader, val_loader, device, cfg, best_path,
        wandb_enabled=wandb_enabled, _wandb=_wandb,
    )

    # ベストモデルを読み込んでテスト
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
    else:
        print(f"[WARNING] No best checkpoint saved (no improvement during training). Using current model state.")
    test_metrics = evaluate(model, test_loader, device, image_size, heatmap_threshold)
    print(
        f"[TEST] fold={test_fold}  "
        f"mse={test_metrics['val_loss_mse']:.6f}  "
        f"peak={test_metrics['peak_dist_mean']:.2f}px"
    )

    # テストデータに対する直線検出
    out_dir = (
        script_dir
        / cfg.get("evaluation", {}).get("visualization_dir", "vis")
        / f"fold{test_fold}"
        / "test_lines"
    )
    line_summary = predict_lines_and_eval_test(
        cfg=cfg,
        model=model,
        test_loader=test_loader,
        device=device,
        dataset_root=root_dir,
        out_dir=out_dir,
    )

    print("\n" + "=" * 60)
    print("[LINE GEOMETRY EVALUATION]")
    print("=" * 60)
    print(f"  Perpendicular Distance: {line_summary['perpendicular_dist_px_mean']:.2f} px")
    print(f"  Angle Error:           {line_summary['angle_error_deg_mean']:.2f} deg")
    print(f"  Rho Error:             {line_summary['rho_error_px_mean']:.2f} px")
    print("\n[Per-Channel Breakdown]")
    for k, v in line_summary["per_channel"].items():
        print(
            f"  {k}: perp={v['perpendicular_dist_px_mean']:.2f}px  "
            f"angle={v['angle_error_deg_mean']:.2f}deg  "
            f"rho={v['rho_error_px_mean']:.2f}px  (n={v['n']})"
        )
    print(f"\n[Output] {line_summary['out_dir']}")
    print("=" * 60)

    # サンプル画像を保存
    vis_root = script_dir / eval_cfg.get("visualization_dir", "vis_2")
    print("[INFO] saving example overlays ...")
    save_examples(
        model,
        val_loader,
        device,
        vis_root / f"fold{test_fold}" / "val",
        n_save=16,
        tag="VAL",
    )
    save_examples(
        model,
        test_loader,
        device,
        vis_root / f"fold{test_fold}" / "test",
        n_save=16,
        tag="TEST",
    )
    print(f"[INFO] saved to {vis_root}/")

    # wandb 終了
    if wandb_enabled and _wandb is not None:
        _wandb.run.summary["test_mse"] = test_metrics["val_loss_mse"]
        _wandb.run.summary["test_peak_dist"] = test_metrics["peak_dist_mean"]
        _wandb.run.summary["line_perp_dist"] = line_summary["perpendicular_dist_px_mean"]
        _wandb.run.summary["line_angle_error"] = line_summary["angle_error_deg_mean"]
        _wandb.run.summary["line_rho_error"] = line_summary["rho_error_px_mean"]
        _wandb.finish()

    # Return results for train.py
    return {
        "test_mse": test_metrics["val_loss_mse"],
        "test_peak_dist_mean": test_metrics["peak_dist_mean"],
        "line_perpendicular_dist_px_mean": line_summary["perpendicular_dist_px_mean"],
        "line_angle_error_deg_mean": line_summary["angle_error_deg_mean"],
        "line_rho_error_px_mean": line_summary["rho_error_px_mean"],
        "per_vertebra": test_metrics.get("per_vertebra", {}),
    }
