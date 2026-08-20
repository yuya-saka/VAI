# Proposed：Mask-Guided Branch

10ch early inputをEfficientNetV2-Sの`blocks[4]`まで共有し、14×14特徴から4領域へ分岐します。
各領域branchは独立したMask-Guided Attentionと`blocks[5] -> conv_head -> bn2 -> GAP -> BiLSTM -> head`
を持ちます。attentionのspatial map`s`をarea縮小した対応領域maskへRMSE回帰します。

| 構成 | whole出力 | attention重み |
|---|---|---|
| Proposed–B | 独立whole branch | 校正β |
| Proposed–max | 面ごとの4 region logit最大値 | 校正β |
| Proposed–max β=0 | Proposed–maxと同一経路 | 0 |

β=0でもattentionと全branchを計算し、β>0とのparameter・VRAM・時間経路を一致させます。

## 実行

このproject専用のentry pointから、`--arm`で3構成を選びます。

```bash
uv run python -m fracture_detection.proposed.cli train --arm proposed_b \
  --outer-fold 0 --gpu-id 0 --smoke-steps 1
uv run python -m fracture_detection.proposed.cli train --arm proposed_max
uv run python -m fracture_detection.proposed.cli train --arm proposed_max_beta0
```

`--outer-fold`を省くと`parallel.gpu_ids`へ5 foldを並列割当します。β校正とresource profileも
同じentry pointから実行します。β校正の参照armは`proposed_b`固定です。

```bash
uv run python -m fracture_detection.proposed.cli calibrate --arm proposed_b --gpu-id 1
uv run python -m fracture_detection.proposed.cli profile --arm proposed_max \
  --output fracture_detection/profiling/outputs/<phase>/proposed_max.json
```

実装は共通の`fracture_detection/core`と`fracture_detection/cli`にあり、entry pointは委譲のみを
行います。共通CLIを直接叩く旧来の形式も同じ実装へ到達します。

```bash
uv run python -m fracture_detection.cli.train \
  --config fracture_detection/proposed/config/proposed_b.yaml \
  --outer-fold 0 --gpu-id 0 --smoke-steps 1
```

正式run前に3構成の実データsmokeと、同一GPU上の20-step resource profileを完了します。49 GB gateを
超えた場合は自動で構造を変更せず停止します。smokeはouter推論を行いません。
