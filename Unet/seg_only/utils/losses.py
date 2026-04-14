"""セグメンテーション専用損失関数 - Weighted CE + fg Dice"""

import torch
import torch.nn.functional as F


def make_internal_boundary_band(gt_mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """前景クラス間の内部境界バンドを作成する

    前景（>0）同士でラベルが変化する位置のみを内部境界とみなし、
    境界の両側ピクセルを1にした後、max_pool2dで半径`radius`だけ膨張する。

    引数:
        gt_mask: GTセグメンテーションマスク (B, H, W) int64
        radius: 境界バンドの膨張半径

    戻り値:
        内部境界バンド (B, H, W) float
    """
    fg = gt_mask > 0
    boundary = torch.zeros_like(fg, dtype=torch.bool)

    v = (gt_mask[:, 1:, :] != gt_mask[:, :-1, :]) & fg[:, 1:, :] & fg[:, :-1, :]
    h = (gt_mask[:, :, 1:] != gt_mask[:, :, :-1]) & fg[:, :, 1:] & fg[:, :, :-1]

    # 境界の両側ピクセルを境界バンドとしてマーキング
    boundary[:, 1:, :] |= v
    boundary[:, :-1, :] |= v
    boundary[:, :, 1:] |= h
    boundary[:, :, :-1] |= h

    band = F.max_pool2d(
        boundary.float().unsqueeze(1),
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )
    return band.squeeze(1)


def boundary_band_dice_loss(
    seg_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    band: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """内部境界バンド領域に限定した前景Dice損失を計算する

    各サンプルについて前景クラス（1-4）のDiceを計算し、
    「そのサンプルで対象クラスGTが存在しない」または
    「そのサンプルでバンドが存在しない」場合はスキップする。
    有効なクラスが1つも無い場合は0.0を返す。

    引数:
        seg_logits: セグメンテーションロジット (B, 5, H, W)
        gt_mask: GTセグメンテーションマスク (B, H, W) int64
        band: 内部境界バンド (B, H, W) float
        eps: ゼロ除算防止

    戻り値:
        バンド限定Dice損失（平均）
    """
    probs = torch.softmax(seg_logits, dim=1)
    losses: list[torch.Tensor] = []

    for b in range(seg_logits.shape[0]):
        band_b = band[b]
        if band_b.sum().item() <= 0:
            continue

        for c in range(1, 5):
            gt_c = (gt_mask[b] == c).float()
            if gt_c.sum().item() <= 0:
                continue

            pred_in_band = probs[b, c] * band_b
            gt_in_band = gt_c * band_b
            intersection = (pred_in_band * gt_in_band).sum()
            denom = pred_in_band.sum() + gt_in_band.sum()
            dice_c = (2.0 * intersection + eps) / (denom + eps)
            losses.append(1.0 - dice_c)

    if not losses:
        return seg_logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def compute_seg_only_loss(
    seg_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    gamma_dice: float = 0.4,
    alpha_boundary: float = 0.0,
    lambda_bd: float = 0.0,
    boundary_radius: int = 1,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """セグメンテーション専用損失を計算する

    L = L_CE(weighted) + gamma_dice * L_Dice_fg + lambda_bd * L_Dice_boundary

    L_CE: クラス重み付きクロスエントロピー（全クラス）
        alpha_boundary > 0 のとき、内部境界バンドで重み付け
    L_Dice_fg: 前景クラス（1-4）の平均Dice損失（background除く）
    L_Dice_boundary: 内部境界バンド領域に限定した前景Dice損失

    引数:
        seg_logits: セグメンテーションロジット (B, 5, H, W)
        gt_mask: GTセグメンテーションマスク (B, H, W) int64
        class_weights: クラス重み (5,) float（None時は均等重み）
        gamma_dice: Dice損失の重み（デフォルト 0.4）
        alpha_boundary: CEの内部境界重み係数（0で無効）
        lambda_bd: 境界バンドDice損失の重み（0で無効）
        boundary_radius: 内部境界バンドの膨張半径
        eps: ゼロ除算防止

    戻り値:
        {
            'total': 総損失,
            'ce_loss': CE損失（detach済み）,
            'dice_fg_loss': gamma_dice * Dice_fg損失（detach済み）,
            'raw_dice_fg_loss': Dice_fg損失（detach済み）,
            'boundary_dice_loss': lambda_bd * 境界バンドDice損失（detach済み）,
        }
    """
    band: torch.Tensor | None = None
    if alpha_boundary > 0.0 or lambda_bd > 0.0:
        band = make_internal_boundary_band(gt_mask, radius=boundary_radius)

    if alpha_boundary > 0.0:
        # 境界重み付きCE（サンプル内平均を1に正規化して全体スケールを維持）
        ce_map = F.cross_entropy(seg_logits, gt_mask, weight=class_weights, reduction='none')
        weight_map = 1.0 + alpha_boundary * band
        weight_map = weight_map / weight_map.mean(dim=(1, 2), keepdim=True)
        ce_loss = (ce_map * weight_map).mean()
    else:
        # 後方互換のため、既存実装と同一のCE計算を維持
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

    bd_dice = seg_logits.new_tensor(0.0)
    if lambda_bd > 0.0:
        assert band is not None
        bd_dice = boundary_band_dice_loss(seg_logits, gt_mask, band=band, eps=eps)

    boundary_dice_term = lambda_bd * bd_dice
    total = ce_loss + gamma_dice * raw_dice_fg + boundary_dice_term

    return {
        'total': total,
        'ce_loss': ce_loss.detach(),
        'dice_fg_loss': (gamma_dice * raw_dice_fg).detach(),
        'raw_dice_fg_loss': raw_dice_fg.detach(),
        'boundary_dice_loss': boundary_dice_term.detach(),
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
