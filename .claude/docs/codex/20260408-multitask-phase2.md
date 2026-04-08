# Codex Implementation: Multitask Phase 2

Date: 2026-04-08

## Task
Phase 2 implementation for Unet/multitask/: losses.py, metrics.py, model.py

## Status: SUCCESS (with one fix applied)

## Files Created
- `Unet/multitask/utils/losses.py` - line loss helpers + compute_multitask_loss
- `Unet/multitask/utils/metrics.py` - line metrics + compute_seg_metrics
- `Unet/multitask/src/model.py` - ResUNet (Shared Encoder + Dual Decoder)

## Fix Applied
Codex created the files but GroupNorm failed when `in_ch=2 < norm_groups=8` at Encoder stage1.

Fix in `ResBlock.__init__`:
```python
# クランプ追加: num_channels が norm_groups より小さい場合に対応
g1 = min(norm_groups, in_ch)
g2 = min(norm_groups, out_ch)
self.norm1 = nn.GroupNorm(g1, in_ch)
self.norm2 = nn.GroupNorm(g2, out_ch)
```

## Verification Results

### Model Forward Pass
```
seg_logits:   torch.Size([2, 5, 224, 224])  ✓
line_heatmaps: torch.Size([2, 4, 224, 224]) ✓
params: 1,581,781
```

### Loss Function (compute_multitask_loss)
- Keys: ['total', 'raw_line_loss', 'raw_seg_loss', 'weighted_seg_loss'] ✓
- Partial supervision (has_seg_label mask) works correctly ✓

### Metrics (compute_seg_metrics)
- Returns: miou, per_class_iou, dice ✓

## Architecture Summary
- ResUNet: Shared Encoder (4-stage ResBlock + MaxPool) + 2 independent Decoders
- Encoder features: (24, 48, 96, 192)
- Loss formula: L = L_line + 0.03 * L_seg (partial supervision for seg)
