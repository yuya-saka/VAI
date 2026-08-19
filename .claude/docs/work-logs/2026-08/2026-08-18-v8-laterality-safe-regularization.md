# 2026-08-18: Baseline 0 v8 — horizontal flip復活とStage1 parity修正

## 背景

`08_18/v4`（protocol v7）の outer 0 学習中に、参照実装 `train_models/stage1` との学習挙動の乖離を確認した。

| | Stage1 v1_parity（5 folds） | Baseline 0 v4 outer 0 |
|---|---|---|
| 到達epoch | 74, 75, 75, 75, 75（全fold完走） | 38で早期終了 |
| early stopping発火 | **一度もなし** | patience 15で発火 |
| best val AUROCのepoch | 59, 61, 61, 73, 74 | 17 |
| 最終 train/val BCE gap | +0.021, -0.027, +0.031, -0.035, +0.003 | **+0.105** |
| best val AUROC | 0.909〜0.931 | 0.898 |

## 原因の特定

同一epochで直接比較した。

| epoch | Stage1 fold0 train / val (gap) | Baseline 0 v4 outer0 train / val (gap) |
|---|---|---|
| 17 | 0.3767 / 0.3569 (**-0.020**) | 0.3066 / 0.2734 (-0.033) |
| 24 | 0.3550 / 0.3265 (**-0.029**) | 0.2765 / 0.2941 (**+0.018**) |
| 38 | 0.3151 / 0.2682 (**-0.047**) | 0.2007 / 0.3057 (**+0.105**) |

前提条件はほぼ揃っている。

- **同一データセット**: Stage1 = train_val 10,730 + test 2,703 = 13,433 bag、Baseline 0 = 13,432 bag
- **学習量もほぼ同じ**: Stage1 fold0 train 8,586 bag（陽性855、9.96%）vs Baseline 0 約8,080 bag（505 batch×16）。差は6%
- **augmentationの粒度も同じ**: Stage1も `_augment_volume` で15面をchannel方向へstackし1 bagにつき1回だけ呼び出す。v7の実装はStage1に正しく一致していた
- **設定差はflip 3種だけ**: affine / brightness / blur-noise / distortion / cutout / mixup は全て数値一致

最も強い証拠は **train loss** である。同じepochでBaseline 0の方が一貫して深く下がっており（ep38で0.2007 vs 0.3151）、train lossはval splitの違いの影響を受けないため、「学習タスク自体が易しくなっている」ことを示す。AUROCの推移も整合的で、Baseline 0はep17で0.898に到達（Stage1は同epochで0.804）した後に頭打ち、Stage1はep59で0.909まで伸び続ける。弱い正則化の典型パターン。

⚠️ **これは実験による証明ではなく推論**である。val splitが異なる（Stage1はtest 20%分離後のval 2,144 bag、Baseline 0はval 2,686 bag）ためAUROCの絶対値は厳密比較できず、flip以外の未知の差が残る可能性も排除できていない。

### 波及した問題

`max_epochs=75` 前提の cosine を epoch 38 で切っているため、停止時LRは 1.29e-4（eta_min へのアニール進捗 **49%**）。Stage1が best checkpoint を出している低LR収束フェーズを一度も通過していなかった。

### 副次的に判明した非parity

| 項目 | Stage1 | v7 |
|---|---|---|
| `gradient_clip_norm` | `null`（trainerで`inf`扱い、実質無効） | 5.0（0〜2.4%のstepで発火） |
| weight decay適用範囲 | 全パラメータ（`model.parameters()`単一group） | bias・1次元パラメータを除外 |

## 決定（protocol `baseline0-nested-v8`）

### dropoutによる代替は却下

当初 Codex（`gpt-5.6-sol`）へ相談し `drop_path_rate=0.10` / `drop_rate=0.10` の追加を推奨されたが、**採用しなかった**。Stage1は `drop_rate=0.0 / drop_path_rate=0.0` で過学習していない実績があり、参照実装が使っていない別機構（ネットワークへのノイズ注入）で埋めるのは正当化が弱い。失われたもの（augmentationの多様性）と同じ機構で戻すべきと判断した。

Codexの分析全文は `.claude/docs/codex/20260818-1400-laterality-safe-regularization.md`。augmentation強化の各案を却下した理由（RandomResizedCropはregion evidenceを切り落とす、ElasticTransformは局所Jacobianの正値が保証されずorientation-safeと証明できない、等）と、後続アームでのattention干渉回避策5点は引き続き有効。

### horizontal flipだけを復活させる

`common/constants.py` の領域定義を確認した。

```
R1 = vertebral_body              （正中）
R2 = right_transverse_foramen
R3 = left_transverse_foramen
R4 = posterior_elements          （正中）
```

**R2とR3は左右対称の同種構造**なので、水平反転と同時にラベルとマスク値を入れ替えれば意味論が完全に保存される。R1/R4は正中構造なので水平反転の影響を受けない。入れ替えはR2/R3のペアに対して行うため、どちらを名目上「左」と呼ぶかの規約にも依存しない。

vertical flip と transpose は R1 と R4 を入れ替えることになるが、椎体と後方要素は鏡像関係にない**別種の構造**であり、対応する入れ替えが存在しない。**恒久的に禁止**とし、`schema.py` で設定キーごと拒否する。

| 項目 | v7 | v8 | 根拠 |
|---|---|---|---|
| `horizontal_flip_probability` | なし（禁止） | **0.5** | R2↔R3スワップで意味論が保存される。発火率0%→50% |
| vertical flip / transpose | 禁止 | **禁止のまま** | R1/R4に正しい入れ替えが存在しない |
| weight decay適用範囲 | bias/norm除外 | **全パラメータ**（値は1e-4据え置き） | Stage1 parity |
| `gradient_clip_norm` | 5.0 | **null** | Proposedでattention loss追加時にclip頻度がアーム依存で変わるのを防ぐ |
| `early_stopping_patience` | 15 | **20** | ユーザー判断 |
| `drop_rate` / `drop_path_rate` | 0.0 | **0.0据え置き** | 上記のとおり却下 |
| `max_epochs` / `T_max` | 75 | **75据え置き** | Stage1のbestが59〜74に集中しており40への圧縮に根拠がない |

hflip単独では発火率50%で、Stage1の87.5%（3種のいずれか）には届かない。**取り返せるのは一部**である。

### やってはいけないこと

- **`min_epoch` を上げない**。`trainer.py:228` の `eligible = epoch >= min_epoch` が early stopping と checkpoint 保存の両方を制御しており、上げると epoch 17 のような早期 best が保存対象から外れる
- vertical flip / transpose を復活させない
- 領域ラベルを持つアームで `A.HorizontalFlip` を直接使わない（R2/R3スワップが漏れる）。必ず `common.dataset.flip_horizontal` を使う
- アーム別に augmentation を変えない

## 実装

| ファイル | 変更 |
|---|---|
| `common/dataset.py` | `flip_horizontal()` と `LR_SWAPPED_REGION_ORDER` を追加。CT・椎体マスク・領域マスク・領域ラベルを反転し、R2/R3のマスク値（lookup tableで2↔3）とラベル（index 1↔2）を同時に入れ替える |
| `baseline0/config/schema.py` | `PROTOCOL_VERSION` v7→v8、`FROZEN_AUGMENTATION` に `horizontal_flip_probability: 0.5`、禁止キーを `vertical_flip` / `transpose` 系のみへ変更（理由をコメントで明記）、`FROZEN_TRAINING` の gradient_clip_norm / early_stopping_patience |
| `baseline0/data/dataset.py` | `build_train_transform` 先頭に `A.HorizontalFlip`（Stage1と同じ順序）、`default_augmentation` にキー追加 |
| `baseline0/training/optimization.py` | `_is_no_decay` を削除し、backbone/head の2グループ全てに weight decay を適用 |
| `baseline0/training/trainer.py` | `gradient_clip_norm: null` を Stage1 と同じく `float("inf")` へ変換 |
| `baseline0/config/baseline0.yaml` | 上記の値、`experiment.name` v4→v5、`gpu_id` 1→0 |
| `common/tests/test_dataset.py` | スワップの正当性テスト3件（R2/R3のみ入れ替わる・2回で恒等・マスクチャンネルとラベルの整合） |
| `baseline0/tests/test_config.py` | hflip許可 / vflip・transpose拒否のテストへ置換 |
| `baseline0/tests/test_dataset.py` | transform順序の期待値を更新 |
| `baseline0/tests/test_optimization.py` | 全パラメータが decay 対象であることを検証 |
| `baseline0/tests/test_trainer.py` | fixture を v8 / `gradient_clip_norm: None` へ |

## 検証

- common + Baseline 0: **70 tests passed**
- Ruff check / format: passed
- mypy（`--exclude tests`）: 検出21件は全て既存の stub 不足（pandas/tqdm）と `folds/` の既存 import エラー。今回の変更由来はゼロ
- 実機確認:
  - hflip発火率 **0.5038**（4000試行、95%CI [0.4883, 0.5192]、検出漏れ0件）
  - CTと椎体マスクが同期して反転
  - transform順序 `[HorizontalFlip, RandomBrightnessContrast, Affine, OneOf, OneOf, CoarseDropout]`（Stage1と同じくflipが先頭）
  - R2(値2)@x=20 → 反転先x=203で値3、R3(値3)@x=200 → 反転先x=23で値2、dtype保持
  - `region_targets [1,1,0,0] → [1,0,1,0]`
  - optimizer 2グループ、weight_decay は `{1e-4}` のみ、**472/472 パラメータ**を網羅、うち1次元（bias/BatchNorm）**292個**も同じ decay 対象
  - clip 閾値 inf で grad norm 8.944 → 8.944 と無変更

## 登録すべき limitation

この改訂は inner-val の挙動を見た後の protocol amendment である。候補グリッド探索は行わず事前指定値1点を freeze したが、選択バイアスは消えない。

> The regularization protocol was amended after an exploratory diagnostic run on one predefined inner-validation split and before outer-fold inference. No candidate grid was evaluated, and the amended configuration was subsequently frozen across all arms and folds.

さらに、**flipが過学習の原因であることは実験で確かめていない**。同一epochでのtrain loss差という観察的証拠に基づく推論である。決着させるにはStage1の設定からflip 3種だけをoffにしたablation（fold0を約40 epoch、GPU 1枚で約3時間）が必要。

## 後続アーム（Proposed）実装時の制約

Codexが提示した attention supervision との干渉回避策。

1. `L_att` は spatial attention map に直接かけ、LSTM/head dropout より**前**で計算する
2. attention logits/maps そのものへ追加 dropout を置かない
3. branch別に異なる dropout 値を使わない
4. `beta>0` と `beta=0` で dropout 位置・乱数処理を同一にする
5. mixup では CT・whole/region mask 入力・whole target・`L_att` の mask target を**同じ permutation と同じ λ** で混合する。annotated stream の `L_region` に mixup を適用しない現在の two-stream 設計は問題なし

加えて **hflip適用時は必ず `common.dataset.flip_horizontal` を使い、R2/R3のラベルとマスク値を同時に入れ替える**こと。Albumentationsの `A.HorizontalFlip` を直接使うとラベルが静かに壊れる。

## 状態

- `08_18/v4`（v7、GPU 1）は診断記録として継続中
- `08_18/v5`（v8、GPU 0）は**未起動**
- 6構成・λ/β・code/config hash の凍結前の outer 推論は引き続き禁止
