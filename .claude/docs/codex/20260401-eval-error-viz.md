# Codex Analysis: eval-error-viz script design
Date: 2026-04-01

## Note
Codex CLI timed out (120s). Analysis performed by subagent based on direct codebase inspection.
The script `Unet/debug/eval_error_viz.py` already exists with a complete implementation.

## Question
Design a Python visualization script for analyzing model training results in a medical CT image line detection project.

The project predicts 4 lines (line_1..4) per CT slice, using (phi, rho) parameterization in Hough space.

Data format:
- Output files: outputs/{phase}/{name}/vis/fold{N}/test_lines/{sample}_{vertebra}_{slice}_PRED_lines.json
- JSON structure: pred_lines + metrics per line
- Existing comparison PNGs: {sample}_{vertebra}_{slice}_comparison.png

Requirements:
1. Error Distribution Plots (violin+box, per-vertebra, per-line, percentiles)
2. Worst Sample Visualization (top-N, summary grid)
3. CLI: --exp-dir, --top-n, --metric, --output-dir
4. matplotlib-fontja, multi-fold aggregation

## Analysis: Existing Implementation

The script is already fully implemented at `Unet/debug/eval_error_viz.py` (760 lines).

### Architecture

**Key Functions:**

1. `load_line_records(exp_dir)` → `list[dict]`
   - Discovers test_lines dirs via `discover_test_line_dirs()`
   - Parses filenames with regex: `(?P<sample>.+?)_(?P<vertebra>C[1-7])_slice(?P<slice>\d+)_PRED_lines.json`
   - Returns flat list of line-level records with all metrics

2. `build_slice_records(records)` → `list[dict]`
   - Aggregates line-level records to slice-level
   - Computes per-metric max and mean across lines
   - Builds line_summary string for display

3. `plot_overall_histogram(records, metric_name, output_path)`
   - Histogram with percentile vertical lines
   - Stats block (n, mean, p25/50/75/90/95) in text box

4. `plot_group_distribution(records, metric_name, group_field, title, output_path)`
   - Violin + box plot overlay
   - Percentile markers (circle for p25/50/75, diamond for p90/95)
   - Supports both vertebra (C1-C7) and line_name grouping

5. `select_worst_slices(slice_records, metric_name, aggregate, top_n)`
   - Ranks by `{metric}_{aggregate}` key (max or mean across lines)
   - Returns top-N worst

6. `build_worst_summary_grid(worst_slices, ...)`
   - Loads existing comparison PNGs with cv2
   - Resizes to 420x210, adds label bar
   - Creates grid (ceil(sqrt(N)) columns, max 4 cols)

7. `copy_worst_images(worst_slices, ...)`
   - Copies PNGs with rank-prefixed filenames

### CLI Design
```
--exp-dir     : path to outputs/{phase}/{name}
--top-n       : default 20
--metric      : angle / rho / perp (default: angle)
--sort-aggregate : max / mean (default: max)
--output-dir  : default exp-dir/vis/error_viz
```

### Output Structure
```
{output_dir}/
├── line_records.csv         # raw line-level data
├── slice_records.csv        # slice-level aggregated data
├── percentiles.csv          # p25/50/75/90/95 summary
├── summary.txt              # human-readable overall stats
├── plots/
│   ├── {metric}_overall_hist.png
│   ├── {metric}_by_vertebra_violin.png
│   └── {metric}_by_line_violin.png
└── worst_samples/{metric}_{aggregate}/
    ├── worst_summary_grid.png
    └── rank{N}_{fold}_{sample}_{vertebra}_slice{X}_{metric}{score}.png
```

### Design Decisions

1. **Violin + Box overlay**: Best for medical imaging — shows full distribution shape (violin) while highlighting quartiles (box)
2. **Sort by max line error** (default): Identifies catastrophic failures better than mean; a single bad line matters clinically
3. **Multi-fold aggregation**: Single pass over all fold dirs, fold name stored in records
4. **matplotlib-fontja**: Imported at top for Japanese labels (椎体別分布, etc.)
5. **Percentile colors**: Color-coded lines for quick visual hierarchy (p25=blue, p50=dark blue, p75=green, p90=orange, p95=red)

### Example Usage
```bash
uv run python Unet/debug/eval_error_viz.py \
    --exp-dir Unet/outputs/regularization/sig3.5 \
    --top-n 20 \
    --metric angle \
    --output-dir Unet/outputs/regularization/sig3.5/vis/error_viz
```

## Key Recommendations (for future improvements)

1. **Add fold-level breakdown**: Currently aggregates all folds together; a per-fold comparison plot could reveal cross-validation variance
2. **Scatter plot**: phi vs rho error scatter with vertebra color coding to detect systematic biases
3. **Error correlation**: Check if angle_error and rho_error are correlated (outlier detection)
4. **Missing comparison PNGs**: Script silently skips; could add a warning count
