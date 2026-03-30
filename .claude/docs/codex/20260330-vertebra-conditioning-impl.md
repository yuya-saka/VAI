# Codex Implementation: Vertebra Label Conditioning for TinyUNet
Date: 2026-03-30

## Task
Implement vertebra label conditioning for TinyUNet in `/mnt/nfs1/home/yamamoto-hiroto/research/VAI`.

## Files Modified

### 1. Unet/line_only/src/model.py
- Added `import torch.nn.functional as F`
- Added module-level `VERTEBRA_TO_IDX = {"C1": 0, ..., "C7": 6}`
- Added `num_vertebra: int = 0` parameter to `TinyUNet.__init__`
- Added `self.cond_proj = nn.Conv2d(f4 + num_vertebra, f4, kernel_size=1, bias=True)` when `num_vertebra > 0`
- Added `_init_cond_proj_identity(f4)` method with identity weight initialization
- Added `_onehot_map()` helper using `F.one_hot`
- Updated `forward()` signature to `forward(self, x, vertebra_idx=None)`
- Added conditioning logic after bottleneck (`x4`)

### 2. Unet/line_only/src/data_utils.py
- In `create_model_optimizer_scheduler`: reads `num_vertebra = int(model_cfg.get("num_vertebra", 0))`
- Passes `num_vertebra=num_vertebra` to `TinyUNet` constructor

### 3. Unet/line_only/src/trainer.py
- Added `from .model import VERTEBRA_TO_IDX` import
- In 4 locations (`evaluate`, `run_training_loop`, `predict_lines_and_eval_test`, `save_examples`):
  - Extracts `v_idx` via `torch.as_tensor([VERTEBRA_TO_IDX.get(v, 0) for v in batch["vertebra"]], ...)`
  - Changed `model(x)` to `model(x, v_idx)`

### 4. Unet/config/config.yaml
- Added `num_vertebra: 7` under `model:` section

## Verification
- `python -m py_compile` passed on all 3 Python files
- Backward compatible: `vertebra_idx=None` path works when `num_vertebra=0`

## Result
Implementation succeeded. All 4 files modified as specified.
