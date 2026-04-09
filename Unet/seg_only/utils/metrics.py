"""セグメンテーション評価指標 - fg-mDice を primary metric とする"""

import torch


CLASS_NAMES = ['bg', 'body', 'right', 'left', 'posterior']


def compute_seg_fg_metrics(
    seg_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    num_classes: int = 5,
    eps: float = 1e-6,
) -> dict:
    """セグメンテーション評価指標を計算する（fg-mDice を primary metric とする）

    引数:
        seg_logits: セグメンテーションロジット (B, C, H, W)
        gt_mask: GTマスク (B, H, W) int64
        num_classes: クラス数（デフォルト5）
        eps: ゼロ除算防止

    戻り値:
        {
            'fg_mdice': float,       # primary metric（bg除く前景4クラス平均）
            'fg_miou': float,        # 前景4クラス平均IoU
            'miou': float,           # 全クラス平均IoU（参考値）
            'dice': float,           # 全クラス平均Dice（参考値）
            'per_class': {           # クラス別 IoU/Dice
                'bg': {'iou': ..., 'dice': ...},
                'body': {'iou': ..., 'dice': ...},
                'right': {'iou': ..., 'dice': ...},
                'left': {'iou': ..., 'dice': ...},
                'posterior': {'iou': ..., 'dice': ...},
            }
        }
    """
    pred_class = seg_logits.argmax(dim=1)  # (B, H, W)

    per_class = {}
    all_ious = []
    all_dices = []
    fg_ious = []
    fg_dices = []

    for c in range(num_classes):
        pred_c = pred_class == c
        gt_c = gt_mask == c
        intersection = (pred_c & gt_c).sum().float()
        union = (pred_c | gt_c).sum().float()
        iou = (intersection + eps) / (union + eps)
        dice_denom = pred_c.sum().float() + gt_c.sum().float()
        dice = (2.0 * intersection + eps) / (dice_denom + eps)

        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f'class_{c}'
        per_class[name] = {'iou': iou.item(), 'dice': dice.item()}
        all_ious.append(iou.item())
        all_dices.append(dice.item())

        if c > 0:  # 前景クラスのみ
            fg_ious.append(iou.item())
            fg_dices.append(dice.item())

    return {
        'fg_mdice': float(sum(fg_dices) / len(fg_dices)),   # primary
        'fg_miou': float(sum(fg_ious) / len(fg_ious)),
        'miou': float(sum(all_ious) / len(all_ious)),        # 参考値（bg込み）
        'dice': float(sum(all_dices) / len(all_dices)),      # 参考値（bg込み）
        'per_class': per_class,
    }
