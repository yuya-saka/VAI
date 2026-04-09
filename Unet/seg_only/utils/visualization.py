"""セグメンテーション可視化関数"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_seg_overlay(ct: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray, out_path: Path) -> None:
    """セグメンテーション予測とGTのオーバーレイ画像を保存する

    引数:
        ct: CT画像配列 (H, W) float [0,1]
        pred_mask: 予測マスク (H, W) int [0,4]
        gt_mask: GTマスク (H, W) int [0,4]
        out_path: 保存先パス
    """
    # 5クラスカラーマップ: 背景=黒, 椎体=赤, 右横突孔=緑, 左横突孔=青, 後方要素=黄
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
