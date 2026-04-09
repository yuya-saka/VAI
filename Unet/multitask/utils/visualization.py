"""可視化関数: ヒートマップオーバーレイ・直線比較・グリッド表示"""

from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


# -------------------------
# ヒートマップ可視化（train_heat.py から抽出）
# -------------------------
def save_heatmap_overlay(ct01, hm4, save_path, alpha=0.55, vmax=1.0):
    """CT画像にヒートマップ（4ch最大値）を重ねて保存"""
    H, W = ct01.shape
    ct_u8 = (np.clip(ct01, 0, 1) * 255).astype(np.uint8)
    base = cv2.cvtColor(ct_u8, cv2.COLOR_GRAY2BGR)

    hm = np.max(hm4, axis=0)
    hm = np.clip(hm / max(vmax, 1e-6), 0, 1)
    hm_u8 = (hm * 255).astype(np.uint8)

    heat_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    out = cv2.addWeighted(base, 1 - alpha, heat_color, alpha, 0)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), out)


def save_heatmap_grid(ct01, hm4, save_path, alpha=0.55):
    """4チャンネルヒートマップを2x2グリッドで保存"""
    H, W = ct01.shape
    ct_u8 = (np.clip(ct01, 0, 1) * 255).astype(np.uint8)
    base = cv2.cvtColor(ct_u8, cv2.COLOR_GRAY2BGR)

    tiles = []
    for c in range(4):
        hm = np.clip(hm4[c], 0, 1)
        hm_u8 = (hm * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        out = cv2.addWeighted(base, 1 - alpha, heat_color, alpha, 0)
        cv2.putText(
            out,
            f"CH{c + 1}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(out)

    top = np.concatenate([tiles[0], tiles[1]], axis=1)
    bot = np.concatenate([tiles[2], tiles[3]], axis=1)
    grid = np.concatenate([top, bot], axis=0)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), grid)


# -------------------------
# 直線比較可視化（line_detection.py から抽出）
# -------------------------
# 直線色の定義（全関数で共通）
LINE_COLORS = {
    "line_1": (0, 255, 0),    # 緑
    "line_2": (0, 0, 255),    # 赤
    "line_3": (255, 0, 0),    # 青
    "line_4": (0, 255, 255),  # 黄
}


def draw_line_overlay(ct01: np.ndarray, lines4: dict[str, Any], save_path: Path):
    """
    直線検出結果をCT画像に重ねて描画

    引数:
        ct01: (H,W) float [0,1] CT画像
        lines4: {"line_1": {...}, ...} 直線検出結果
    """
    ct_u8 = (np.clip(ct01, 0, 1) * 255).astype(np.uint8)
    img = cv2.cvtColor(ct_u8, cv2.COLOR_GRAY2BGR)

    for k, v in lines4.items():
        if v is None:
            continue
        ep = v.get("endpoints")
        if ep is None or len(ep) != 2:
            continue
        (x1, y1), (x2, y2) = ep
        c = LINE_COLORS.get(k, (255, 255, 255))
        cv2.line(
            img,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            c,
            2,
        )
        cx, cy = v.get("centroid", [None, None])
        if cx is not None:
            cv2.circle(img, (int(round(cx)), int(round(cy))), 2, c, -1)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img)


def draw_line_comparison(
    ct01: np.ndarray,
    pred_lines: dict[str, Any],
    gt_lines: dict[str, Any],
    save_path: Path,
):
    """
    GT線と予測線を横並びで比較表示

    引数:
        ct01: (H,W) float [0,1] CT画像
        pred_lines: {"line_1": {"endpoints": [[x1,y1],[x2,y2]], ...}, ...} 予測結果
        gt_lines: {"line_1": [[x,y], ...], ...} GT（lines.jsonの形式）
    """
    ct_u8 = (np.clip(ct01, 0, 1) * 255).astype(np.uint8)

    # GT画像（左）
    img_gt = cv2.cvtColor(ct_u8.copy(), cv2.COLOR_GRAY2BGR)
    # 予測画像（右）
    img_pred = cv2.cvtColor(ct_u8.copy(), cv2.COLOR_GRAY2BGR)

    # GT線を描画（折れ線）
    for k, pts in gt_lines.items():
        if pts is None or len(pts) < 2:
            continue
        c = LINE_COLORS.get(k, (255, 255, 255))
        pts_i32 = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img_gt, [pts_i32], isClosed=False, color=c, thickness=2)

    # 予測線を描画（直線）
    for k, v in pred_lines.items():
        if v is None:
            continue
        ep = v.get("endpoints")
        if ep is None or len(ep) != 2:
            continue
        c = LINE_COLORS.get(k, (255, 255, 255))
        (x1, y1), (x2, y2) = ep
        cv2.line(
            img_pred,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            c,
            2,
        )
        # centroidにマーカー
        cx, cy = v.get("centroid", [None, None])
        if cx is not None:
            cv2.circle(img_pred, (int(round(cx)), int(round(cy))), 3, c, -1)

    # ラベル追加
    cv2.putText(
        img_gt, "GT", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )
    cv2.putText(
        img_pred, "Pred", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )

    # 横に結合
    combined = np.concatenate([img_gt, img_pred], axis=1)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), combined)


def draw_heatmap_with_lines(
    ct01: np.ndarray,
    heatmaps: np.ndarray,
    pred_lines: dict[str, Any],
    gt_lines: dict[str, Any],
    save_path: Path,
    alpha: float = 0.6,
):
    """
    ヒートマップ・予測線・GT線の3つを横並びで表示

    引数:
        ct01: (H,W) float [0,1] CT画像
        heatmaps: (4,H,W) float [0,1] 予測ヒートマップ（4チャンネル）
        pred_lines: {"line_1": {"endpoints": [[x1,y1],[x2,y2]], ...}, ...} 予測線
        gt_lines: {"line_1": [[x,y], ...], ...} GT線（lines.jsonの形式）
        save_path: 保存先パス
        alpha: ヒートマップのブレンド係数
    """
    ct_u8 = (np.clip(ct01, 0, 1) * 255).astype(np.uint8)
    base = cv2.cvtColor(ct_u8, cv2.COLOR_GRAY2BGR)

    # 1. ヒートマップ画像（左）
    hm_merged = np.max(heatmaps, axis=0)  # 4チャンネルを最大値で統合
    hm_merged = np.clip(hm_merged, 0, 1)
    hm_u8 = (hm_merged * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    img_heatmap = cv2.addWeighted(base.copy(), 1 - alpha, heat_color, alpha, 0)
    cv2.putText(
        img_heatmap,
        "Heatmap",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    # 2. 予測線画像（中央）
    img_pred = base.copy()
    for k, v in pred_lines.items():
        if v is None:
            continue
        ep = v.get("endpoints")
        if ep is None or len(ep) != 2:
            continue
        c = LINE_COLORS.get(k, (255, 255, 255))
        (x1, y1), (x2, y2) = ep
        cv2.line(
            img_pred,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            c,
            2,
        )
        # centroidにマーカー
        cx, cy = v.get("centroid", [None, None])
        if cx is not None:
            cv2.circle(img_pred, (int(round(cx)), int(round(cy))), 3, c, -1)

    cv2.putText(
        img_pred,
        "Pred Lines",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    # 3. GT線画像（右）
    img_gt = base.copy()
    for k, pts in gt_lines.items():
        if pts is None or len(pts) < 2:
            continue
        c = LINE_COLORS.get(k, (255, 255, 255))
        pts_i32 = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img_gt, [pts_i32], isClosed=False, color=c, thickness=2)

    cv2.putText(
        img_gt, "GT Lines", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )

    # 3つを横に結合
    combined = np.concatenate([img_heatmap, img_pred, img_gt], axis=1)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), combined)


def save_seg_overlay(ct: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray, out_path: Path) -> None:
    '''セグメンテーション予測とGTのオーバーレイ画像を保存する

    引数:
        ct: CT画像配列 (H, W) float [0,1]
        pred_mask: 予測マスク (H, W) int [0,4]
        gt_mask: GTマスク (H, W) int [0,4]
        out_path: 保存先パス
    '''
    # 5クラスカラーマップ: 背景=黒, 左横突孔=赤, 椎体中心=緑, 右横突孔=青, 後方要素=黄
    colors = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]], dtype=np.uint8)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(ct, cmap='gray')
    axes[0].set_title('CT')
    axes[0].axis('off')
    pred_rgb = colors[pred_mask.clip(0, 4)]
    axes[1].imshow(ct, cmap='gray')
    axes[1].imshow(pred_rgb, alpha=0.5)
    axes[1].set_title('Pred Seg')
    axes[1].axis('off')
    gt_rgb = colors[gt_mask.clip(0, 4)]
    axes[2].imshow(ct, cmap='gray')
    axes[2].imshow(gt_rgb, alpha=0.5)
    axes[2].set_title('GT Seg')
    axes[2].axis('off')
    error_map = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    non_background = (pred_mask != 0) | (gt_mask != 0)
    ok_pixels = non_background & (pred_mask == gt_mask)
    ng_pixels = non_background & (pred_mask != gt_mask)
    error_map[ok_pixels] = [0, 255, 0]
    error_map[ng_pixels] = [255, 0, 0]
    axes[3].imshow(error_map)
    axes[3].set_title('Error (G=OK, R=NG)')
    axes[3].axis('off')
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=80, bbox_inches='tight')
    plt.close()
