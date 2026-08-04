# 2026-08-01 作業ログ（Line Surface 3D 実装・baseline完了）

`Unet/line_only/` だけを既存実装の参照元として、独立プロジェクト
`Unet/line_surface_3d/` を新規実装した。連続15スライスを入力し、各スライスの
4本線ヒートマップを推定したうえで、z方向に重心とdoubled-angleを1次fitして
リボン状の面として扱う。

**現在地**: heatmap-only baselineの5-fold学習・テストが完了した。
線単体は `line_only` と同水準の約5°だが、面指標はfold間差が大きく、特にfold 4が課題。

---

## 1. 実装済み

- 密画像 `dataset_zprop/images` と疎な手動教師 `dataset/lines.json` から連続スラブを構築
- `dataset_zprop/lines.json` の擬似ラベルは不使用
- 入力 `(B, 2N, H, W)`、出力 `(B, N, 4, H, W)` の独立TinyUNet
- CTとmaskに同一幾何変換を適用するスラブ単位augmentation
- ヒートマップmomentから重心・主軸法線を抽出
- 180°周期を扱うdoubled-angleと重心をz方向に閉形式1次fit
- baseline（heatmap lossのみ）とribbon loss版をconfig分離
- sliding-window推論、重複窓統合、不一致スコア、領域欠損率、z平滑性、直交断面可視化
- checkpoint、metrics JSONL、test JSON、5-fold summary、W&B記録

主要パス：

- 実装: `Unet/line_surface_3d/`
- baseline config: `Unet/line_surface_3d/config/baseline.yaml`
- ribbon config: `Unet/line_surface_3d/config/ribbon.yaml`
- 成果物: `Unet/outputs/line_surface_3d/baseline-v1/`
- 集約結果: `Unet/outputs/line_surface_3d/baseline-v1/summary.json`

---

## 2. 学習条件

| 項目 | 設定 |
|---|---:|
| スラブ長 `N` | 15 |
| batch size | 16 |
| DataLoader workers | 8 |
| prefetch factor | 1 |
| persistent workers | true |
| optimizer初期LR | 5e-4 |
| checkpoint選択 | `angle_error_deg` 最小 |
| scheduler / early stopping | `val_loss_mse` |

8 workers化により、初期の約281秒/epochから約37秒/epochまで高速化した。
実際の5-fold実行ではvalidationを含めて概ね80〜90秒/epochだった。

実行コマンド：

```bash
UV_CACHE_DIR=/tmp/vai-uv-cache uv run python \
  Unet/line_surface_3d/train.py \
  --config Unet/line_surface_3d/config/baseline.yaml \
  --start_fold 0 --end_fold 4
```

---

## 3. 学習中に修正した問題

### 3.1 早期終了条件の不整合

初期実装ではsurface fitted angleをpatience判定に使っていたため、MSEが改善中でも
fold 0がepoch 17付近で停止した。

修正後：

- checkpointは`line_only`共通の `angle_error_deg`
- schedulerとearly stoppingは`line_only`共通の `val_loss_mse`
- surface固有指標は `surface_raw_*` / `surface_fitted_*` として別名で記録

### 3.2 評価指標の互換性

`line_only` と同じ評価契約へ揃えた。

- GT: 手動ポリライン由来の `(phi, rho)`
- prediction: 適応閾値 + peak connected-component filter後のmoment
- 共通指標: MSE、peak距離、Blob IoU、角度、rho、outlier率、椎体別集計
- 追加指標: 面のraw/fitted角度・重心誤差、検出率、loss成分

checkpoint protocolは `line_surface_3d_v2`。旧v1 checkpointは推論時に拒否する。

### 3.3 実行安定性

- 10 stepごとの進捗表示を追加
- fresh run時に古いcheckpoint / metrics / test出力を削除
- W&Bのdeprecatedなbool `reinit` を `finish_previous` へ変更
- 大きなslab batchによる先読み負荷を抑えるためprefetchを1に制限

---

## 4. baseline 5-foldテスト結果

### 4.1 平均

| 指標 | 5-fold平均 |
|---|---:|
| 線角度誤差 | **4.963°** |
| rho誤差 | **3.116 px** |
| Peak距離 | **19.976 px** |
| Blob IoU | **0.685** |
| 面 fitted角度誤差 | **17.527°** |
| 面 fitted角度誤差 median | **14.850°** |
| 面 fitted角度誤差 P90 | **35.657°** |
| 面 fitted重心誤差 | **9.495 px** |
| 面 fitted重心誤差 median | **9.187 px** |
| 面検出率 | **100%** |

ここで「面角度誤差」は厳密な3D平面法線同士の角度ではない。
各スライス・各線のヒートマップから得た方向をz方向に1次fitし、fit後の2D線方向と
GT方向の差を全有効線で平均した値。

### 4.2 fold別

| fold | 線角度 | 面角度 | 面重心 | Blob IoU |
|---:|---:|---:|---:|---:|
| 0 | 4.602° | 15.814° | 5.975 px | 0.743 |
| 1 | 4.801° | 16.992° | 8.457 px | 0.684 |
| 2 | 4.716° | 10.088° | 4.240 px | 0.702 |
| 3 | 5.462° | 16.886° | 7.699 px | 0.697 |
| 4 | 5.235° | **27.854°** | **21.106 px** | 0.600 |

線単体は全foldで4.6〜5.5°と安定。一方、面評価はfold 4が明確に悪く、
5-fold平均を押し上げている。面fit前後の差は全体として小さく、現baselineでは
z方向fitそのものによる精度改善は限定的。

---

## 5. 学習挙動

- 多くのfoldで20〜35 epoch付近まで角度誤差が40°前後に留まり、その後5°前後へ急落
- fold 0ではepoch 32: 13.28°、epoch 33: 6.71°、epoch 34: 4.97°
- その後、線角度は概ね5°前後で安定
- 各foldはearly stoppingにより67〜90 epochで終了

---

## 6. 検証

- `Unet/line_surface_3d/test`: 17 passed
- `Unet/line_only` 回帰テスト: 30 passed
- Ruff: passed
- targeted mypy: passed
- compileall: passed

学習終了後、Python multiprocessingの一時ディレクトリ `.tmp/pymp-*` 削除時に
`OSError: [Errno 16] Device or resource busy` が出た。ただし全foldのtest JSONと
`summary.json`生成後のfinalizer警告であり、学習成果物への影響はない。

---

## 7. 次の候補

1. fold 4の症例構成・椎体別誤差・予測可視化を調査
2. `ribbon.yaml` で明示的な角度・重心・リボン損失ありの5-fold実験を実施
3. baselineとribbonを線指標・面指標・外挿領域欠損率で比較
4. best checkpointによる全高推論と冠状断・矢状断可視化を生成
5. multiprocessing一時ディレクトリのNFS cleanup警告を抑制
