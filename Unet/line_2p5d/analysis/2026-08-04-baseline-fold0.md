# line_2p5d baseline fold0 result

## Experiment

- Experiment: `line_2p5d_20260804/test-v1`
- Config: `Unet/line_2p5d/config/baseline.yaml`
- Loss: center-slice mean MSE only (`loss.geometry.enabled: false`)
- Context: `[z-2, z-1, z, z+1, z+2]`
- Split: train 940 / validation 378 / test 293 center images
- Completed epoch: 118 (early stopping)
- Best combined-error checkpoint: epoch 61
- Best validation heatmap MSE: epoch 103, `0.00213244`

## Test result

| Metric | Value |
|---|---:|
| Heatmap MSE | 0.00135792 |
| Heatmap Dice | 0.890752 |
| Angle error | 4.6016 deg |
| Rho error | 3.2705 px |
| Combined error | 7.7680 px |
| Angle outlier rate (>10 deg) | 9.556% |
| Rho outlier rate (>8 px) | 7.850% |
| Heatmap collapse rate | 0.000% |

## Matched comparison with line_only

Comparison target:
`Unet/outputs/line_20260616/sig4.0_ALL(CC適用)/vis/fold0/test_lines_reeval`

The comparison uses the 293 common test images (1,172 lines). The one old-only QC
slice, `sample22_C1_slice038`, is excluded from both sides.

| Metric | line_only | line_2p5d | Delta |
|---|---:|---:|---:|
| Angle error | 4.6145 deg | 4.6016 deg | -0.0129 deg |
| Rho error | 2.9228 px | 3.2705 px | +0.3476 px |
| Combined error | 7.4330 px | 7.7680 px | +0.3350 px |
| Angle outlier rate | 9.471% | 9.556% | +0.085 pt |
| Rho outlier rate | 4.693% | 7.850% | +3.157 pt |
| Collapse rate | 0.000% | 0.000% | 0.000 pt |

Per-line rho error (`line_only -> line_2p5d`):

- `line_1`: `2.6425 -> 2.8501 px`
- `line_2`: `2.9377 -> 3.3378 px`
- `line_3`: `2.8194 -> 2.7556 px`
- `line_4`: `3.2918 -> 4.1383 px`

## Conclusion

- Five-slice context alone does not reduce angle outliers.
- Rho error and rho outliers regress, driven mainly by `line_4`.
- Both models avoid heatmap collapse on the matched test set.
- The next controlled comparison is fold0 with delayed local geometry enabled,
  keeping the split and heatmap MSE unchanged.

## Artifacts

- Metrics: `Unet/outputs/line_2p5d_20260804/test-v1/fold_0/test_metrics.json`
- Checkpoint: `Unet/outputs/line_2p5d_20260804/test-v1/fold_0/best.pt`
- Visualizations: `Unet/outputs/line_2p5d_20260804/test-v1/vis/fold0/test_lines`
