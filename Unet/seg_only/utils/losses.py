"""セグメンテーション専用損失関数 - Weighted CE + fg Dice"""

import torch
import torch.nn.functional as F


def compute_seg_only_loss(
    seg_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    gamma_dice: float = 0.4,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """セグメンテーション専用損失を計算する

    L = L_CE(weighted) + gamma_dice * L_Dice_fg

    L_CE: クラス重み付きクロスエントロピー（全クラス）
    L_Dice_fg: 前景クラス（1-4）の平均Dice損失（background除く）

    引数:
        seg_logits: セグメンテーションロジット (B, 5, H, W)
        gt_mask: GTセグメンテーションマスク (B, H, W) int64
        class_weights: クラス重み (5,) float（None時は均等重み）
        gamma_dice: Dice損失の重み（デフォルト 0.4）
        eps: ゼロ除算防止

    戻り値:
        {
            'total': 総損失,
            'ce_loss': CE損失（detach済み）,
            'dice_fg_loss': gamma_dice * Dice_fg損失（detach済み）,
            'raw_dice_fg_loss': Dice_fg損失（detach済み）,
        }
    """
    # クラス重み付きCE
    ce_loss = F.cross_entropy(seg_logits, gt_mask, weight=class_weights, reduction='mean')

    # 前景クラス（1-4）の Dice 損失
    probs = torch.softmax(seg_logits, dim=1)  # (B, 5, H, W)
    fg_dice_sum = torch.tensor(0.0, device=seg_logits.device)
    for c in range(1, 5):
        pred_c = probs[:, c]  # (B, H, W)
        gt_c = (gt_mask == c).float()
        intersection = (pred_c * gt_c).sum()
        denom = pred_c.sum() + gt_c.sum()
        dice_c = (2.0 * intersection + eps) / (denom + eps)
        fg_dice_sum = fg_dice_sum + (1.0 - dice_c)
    raw_dice_fg = fg_dice_sum / 4.0

    total = ce_loss + gamma_dice * raw_dice_fg

    return {
        'total': total,
        'ce_loss': ce_loss.detach(),
        'dice_fg_loss': (gamma_dice * raw_dice_fg).detach(),
        'raw_dice_fg_loss': raw_dice_fg.detach(),
    }


def build_class_weights(
    background_weight: float = 0.3,
    device: torch.device | None = None,
) -> torch.Tensor:
    """クラス重みテンソルを作成する

    背景（class 0）の重みを抑制し、前景クラス（1-4）を均等に重み付けする。

    引数:
        background_weight: 背景クラスの重み（デフォルト 0.3）
        device: テンソルのデバイス

    戻り値:
        (5,) float テンソル
    """
    weights = torch.tensor(
        [background_weight, 1.0, 1.0, 1.0, 1.0],
        dtype=torch.float32,
        device=device,
    )
    return weights
