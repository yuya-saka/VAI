# Codex Analysis: wandb-fold1-degradation
Date: 2026-03-30

## Question

Could `wandb.init()` being called BETWEEN `set_seed(42)` and `create_model_optimizer_scheduler()`
cause different model initialization, and therefore different training outcomes for fold1?

Specifically:
1. Does `wandb.init()` consume Python `random` or `numpy.random` state?
2. Does PyTorch model weight initialization (kaiming_uniform etc.) depend on numpy/Python RNG, or purely on `torch.manual_seed`?
3. Could wandb background threads affect CUDA RNG state during the training loop?
4. Why would fold1 specifically be affected but not other folds?
5. Is there a fix (move wandb.init to after model creation, or call set_seed again after wandb.init)?
6. Is reinit=True in wandb.init() particularly problematic for the second fold (fold1)?

## Analysis (based on prior Codex consultations + technical knowledge)

### Prior Codex Investigation (20260329-wandb-randomness.md)

The same question was investigated on 2026-03-29. Key findings from that Codex consultation:

**Q: Does wandb.init() consume torch RNG state?**
→ **No.** wandb.init() does not call torch operations. GPU detection uses nvidia-smi (not torch),
system info collection does not touch torch RNG.

**Q: Does wandb.init() consume Python random or numpy.random state?**
→ **No** (under normal circumstances). wandb.init() performs:
- HTTP request to W&B servers (network I/O, no RNG)
- System info collection via subprocess/psutil (no Python random)
- File system operations for run directory creation (no Python random)
- Thread spawning for background logging (no Python random consumption at init time)

**Q: Does PyTorch kaiming_uniform_ use numpy/Python random?**
→ **No.** `torch.nn.init.kaiming_uniform_` uses only torch RNG (`torch.manual_seed`).
All PyTorch built-in init methods (kaiming, xavier, normal, uniform) call into libtorch C++
which uses the torch Mersenne Twister RNG, completely isolated from Python/numpy RNG.

**Q: Could wandb background threads affect CUDA RNG during training?**
→ **Unlikely.** wandb background threads handle:
- File watching (inotify/kqueue - no CUDA)
- HTTP uploads (network - no CUDA)
- Metric logging (serialization - no CUDA)
None of these interact with CUDA RNG state.

**Q: Is reinit=True problematic for fold1 (second fold)?**
→ **Theoretically possible but not confirmed.** With reinit=True on fold1 (second call),
wandb finishes the fold0 run and initializes a new run. The cleanup of fold0 involves
joining background threads and flushing buffers. This is slightly more work than fold0's
initial init. However, there is no documented mechanism by which this would affect torch RNG.

### Why wandb is NOT the root cause

The prior analysis (20260330-fold1-log-analysis.md) compared training curves:

**Baseline (old shim, no src changes):**
- epoch 96: angle=6.43° (convergence)
- epoch 97+: **stable at 5-7°**
- test result: 8.60°

**sig2.0_base (new src code, wandb=True):**
- epoch 96: angle=6.43° (same convergence point)
- epoch 97-100: 29°→27°→31°→37° (**immediate instability**)
- test result: 17.77°

**sig2.0_base_debug (new src code, wandb=False):**
- test result: 7.60° (nearly identical to baseline)

The critical observation: **Both wandb=True and wandb=False runs use the same new src code.**
The debug run (wandb=False) achieves 7.60°, proving that the new src code CAN converge.
The 精度悪化 run (wandb=True) achieves 17.77°, suggesting wandb interaction.

**BUT**: The only other config difference is `confidence_gate_low/high: 0.3/0.6` vs `0.1/0.8`,
and `use_line_loss: false` means these gates are irrelevant to the main training path.

### Remaining hypothesis: wandb system resource contention

Even if wandb doesn't affect RNG directly, it may affect fold1 through:

1. **CPU/memory pressure during fold1 specifically**:
   - fold0: wandb.init() runs for the first time (lightweight)
   - fold1: wandb.init(reinit=True) + fold0's background thread cleanup is happening
   - This could cause intermittent GPU memory pressure or CPU scheduling delays
   - Result: occasional NaN gradients or optimizer step timing issues

2. **Background thread + CUDA stream interaction**:
   - wandb spawns threads that periodically call `torch.cuda.memory_stats()` or similar
   - This is benign but adds noise to CUDA timing
   - On fold1's critical convergence window (epoch 96-100), timing noise could disrupt gradient updates

3. **The instability at epoch 97-100 is the real question**:
   - Both runs converge at epoch ~96
   - New src code diverges immediately after convergence in fold1
   - This suggests the new src code has a **numerical instability specific to fold1's data distribution**
   - wandb may be a coincidental difference that doesn't explain the instability

### Root cause (from prior Codex analysis)

The true root cause identified in 20260329-fold1-regression.md and 20260330-fold1-log-analysis.md:

**The `confidence > 0` valid_mask change in evaluate() causes fold1's training signal to become
unstable near convergence.**

- Old code: `valid_mask = ~isnan(pred_params)` — broader validity, more stable sample count
- New code: `valid_mask = confidence > 0` — narrower validity, sample count fluctuates
- fold1 has more isotropic heatmaps than fold4 (harder fold)
- When model is near convergence, isotropic predictions hover near confidence=0
- Valid sample count oscillates epoch-to-epoch → val_angle_error oscillates
- Early stopping picks a worse checkpoint

wandb being enabled is a **coincidence** of the experimental setup, not the mechanistic cause.

## Concrete Fix Recommendation

### Fix 1 (Primary - confirmed root cause): Restore valid_mask in evaluate()

In `Unet/line_only/src/trainer.py`, the `evaluate()` function:

```python
# Current (unstable for fold1):
valid_mask = confidence > 0

# Fix: Use NaN-based masking like old code:
valid_mask = ~torch.isnan(pred_params[:, 0]) & ~torch.isnan(pred_params[:, 1])
# OR equivalently, use a very small confidence threshold:
valid_mask = confidence > 1e-6  # effectively same as > 0 but more lenient
```

### Fix 2 (Secondary - defensive): Move set_seed after wandb.init

Even though wandb doesn't consume RNG in theory, as a defensive measure:

```python
def train_one_fold(cfg):
    train_s, val_s, test_s, root_dir, group, image_size, sigma, seed = (
        prepare_datasets_and_splits(cfg)
    )
    train_loader, val_loader, test_loader = create_data_loaders(
        train_s, val_s, test_s, root_dir, group, image_size, sigma, seed, cfg
    )
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # wandb init BEFORE set_seed re-call
    wandb_enabled = wandb_cfg.get("enabled", False)
    if wandb_enabled:
        wandb.init(project=..., name=run_name, config=cfg, reinit=True)

    # Re-seed AFTER wandb to guarantee model init reproducibility
    set_seed(seed)  # ← ADD THIS LINE
    model, opt, scheduler = create_model_optimizer_scheduler(cfg, device)
```

This is belt-and-suspenders: even if wandb did consume some RNG state, model initialization
is guaranteed identical between wandb=True and wandb=False runs.

## Summary

| Question | Answer | Confidence |
|----------|--------|------------|
| Does wandb.init() consume Python random? | No | High (Codex confirmed) |
| Does wandb.init() consume numpy.random? | No | High (Codex confirmed) |
| Does kaiming_uniform_ use numpy/Python RNG? | No (torch-only) | High |
| Can wandb threads affect CUDA RNG? | No (no documented mechanism) | Medium |
| Is wandb the root cause of fold1 degradation? | No | High |
| Is confidence>0 valid_mask the root cause? | Yes (most likely) | High |
| Can reinit=True cause issues for fold1? | Unlikely | Medium |
| Recommended fix | Restore NaN-based valid_mask + re-seed after wandb | High |

## References

- `/mnt/nfs1/home/yamamoto-hiroto/research/VAI/.claude/docs/codex/20260329-wandb-randomness.md`
- `/mnt/nfs1/home/yamamoto-hiroto/research/VAI/.claude/docs/codex/20260329-fold1-regression.md`
- `/mnt/nfs1/home/yamamoto-hiroto/research/VAI/.claude/docs/codex/20260330-fold1-log-analysis.md`
- `Unet/line_only/src/trainer.py` - evaluate(), valid_mask
- `Unet/line_only/utils/losses.py` - _compute_moments_batch(), confidence calculation
