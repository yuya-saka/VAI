# Control–B / Baseline 1–B

同じhard-sharing MTLモデルで、入力channelだけを切り替える比較アームです。

| 構成 | 入力 | whole出力 | region出力 |
|---|---|---|---|
| Control–B | CT 5 + whole mask = 6ch | 独立head `[B,15]` | `[B,15,4]` |
| Baseline 1–B | CT 5 + whole + R1..R4 = 10ch | 独立head `[B,15]` | `[B,15,4]` |

各optimizer stepはnatural batchの`L_whole`をbackwardした後、annotated 1 bagを別forwardして
`λL_region`をbackwardし、optimizerを1回だけ更新します。annotated forward中はBatchNorm moduleだけをevalにし、
Dropoutは有効なままです。annotated用torch RNGはnatural/mixupから分離し、checkpointへ保存します。

## 実行

このproject専用のentry pointから、`--arm`でControl–B / Baseline 1–Bを選びます。

```bash
uv run python -m fracture_detection.mtl.cli train --arm baseline1_b \
  --outer-fold 0 --gpu-id 0 --smoke-steps 1
uv run python -m fracture_detection.mtl.cli train --arm control_b
```

`--outer-fold`を省くと`parallel.gpu_ids`へ5 foldを並列割当します。λ校正はこのprojectの
`calibrate` subcommandで行います。参照armは`baseline1_b`固定で、他armを指定すると拒否します。

```bash
uv run python -m fracture_detection.mtl.cli calibrate --arm baseline1_b --gpu-id 0
```

実装は共通の`fracture_detection/core`と`fracture_detection/cli`にあり、entry pointは委譲のみを
行います。共通CLIを直接叩く旧来の形式も同じ実装へ到達します。

```bash
uv run python -m fracture_detection.cli.train \
  --config fracture_detection/mtl/config/baseline1_b.yaml \
  --outer-fold 0 --gpu-id 0 --smoke-steps 1
```

smokeはouter推論を行いません。正式runは校正済み`loss_weights.json`とfrozen manifestが必須です。
