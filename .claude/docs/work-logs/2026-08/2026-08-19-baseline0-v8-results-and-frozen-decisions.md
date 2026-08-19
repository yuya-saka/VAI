# 2026-08-19: Baseline 0 v8 実測結果と設定の確定・却下事項

## このセッションでやったこと

`08_18/v4`（protocol v7）と `08_18/v5`（protocol v8）を並列で回し、v8改修の効果を実測した。
あわせてバッチサイズ・学習率・mixupについて検討し、**却下事項を確定**した。

前セッションのログ: `2026-08-18-v8-laterality-safe-regularization.md`（v8改修の設計と実装）

---

## 1. 実測結果

### v4（protocol v7、旧設定、GPU1）— 5fold完走

| outer | stop | ckpt | AUROC | AP | Precision | Recall | F1 | 閾値 |
|---|---|---|---|---|---|---|---|---|
| 0 | 38 | **17** | 0.8852 | 0.6983 | 0.7256 | 0.5954 | 0.6541 | 0.299 |
| 1 | 54 | **35** | 0.9059 | 0.6938 | 0.7737 | 0.5485 | 0.6419 | 0.746 |
| 2 | 48 | **37** | 0.9163 | 0.7564 | 0.7991 | 0.6506 | 0.7172 | 0.334 |
| 3 | 37 | **37** | 0.8828 | 0.7155 | 0.8488 | 0.5551 | 0.6713 | 0.617 |
| 4 | 45 | **30** | 0.8841 | 0.6770 | 0.7308 | 0.5630 | 0.6360 | 0.582 |

平均: AUROC **0.8949 ± 0.0153** / AP 0.7082 ± 0.0302 / Precision 0.7756 ± 0.0511 / Recall 0.5825 ± 0.0421 / F1 0.6641 ± 0.0326

pooled（5fold合算、F1最適閾値）: TP=776 FP=228 FN=556 TN=11872 → precision 0.7729 / recall 0.5826 / F1 0.6644

### v5（protocol v8、新設定、GPU0）— outer0〜3完了、outer4学習中

v8の変更点: hflip 0.5復活（R2↔R3スワップ付き）/ weight decay全パラメータ一律 / `gradient_clip_norm: null` / patience 15→20

| outer | stop | ckpt | AUROC | AP | Precision | Recall | F1 | 閾値 |
|---|---|---|---|---|---|---|---|---|
| 0 | 56 | **53** | 0.9109 | 0.7387 | 0.8594 | 0.6298 | 0.7269 | 0.656 |
| 1 | 63 | **49** | 0.9155 | 0.7298 | 0.7200 | 0.6716 | 0.6950 | 0.535 |
| 2 | 52 | **51** | 0.9153 | 0.7652 | 0.8705 | 0.6245 | 0.7273 | 0.342 |
| 3 | 59 | **39** | 0.8819 | 0.7131 | 0.6762 | 0.6274 | 0.6509 | 0.441 |

平均: AUROC **0.9059 ± 0.0161** / AP 0.7367 ± 0.0218 / Precision 0.7815 ± 0.0981 / Recall 0.6383 ± 0.0223 / F1 0.7000 ± 0.0361

### 評価

| | v4 | v5 | 判定 |
|---|---|---|---|
| AUROC | 0.8949 ± 0.0153 | **0.9059** ± 0.0161 | 精度向上、安定性は同等 |
| AP | 0.7082 ± 0.0302 | **0.7367** ± **0.0218** | 向上かつ安定化 |
| Recall | 0.5825 ± 0.0421 | **0.6383** ± **0.0223** | 大きく向上かつ安定化 |
| Precision | 0.7756 ± 0.0511 | 0.7815 ± **0.0981** | 同水準だが**ばらつき悪化** |
| F1 | 0.6641 ± 0.0326 | **0.7000** ± 0.0361 | 向上 |
| ckpt epoch | 31.2 ± 8.44 | 48.0 ± **6.22** | 後方へ移動・安定化 |

**v8は精度を全面的に改善し、AUROCの安定性は維持した。** 特にRecallが+0.056（0.583→0.638）で、Precisionをほぼ維持したまま見逃しが減っている。

checkpoint epochがv4の17〜37からv5の39〜53へ後退したのが設計意図どおりの結果。cosineの低LR収束フェーズを使えるようになった（Stage1のbestはepoch 59〜74）。

⚠️ **Precisionのfold間stdだけ悪化**（0.0511→0.0981）。原因はF1最適閾値の選択ブレ（v5で0.342〜0.656）。val陽性が268件しかなくF1曲線が頂点付近で平坦なためargmaxが不安定。**ただしprimary endpointはAUROCなので主要な結論には影響しない**。

---

## 2. 確定した却下事項（今後蒸し返さない）

| 案 | 判断 | 理由 |
|---|---|---|
| **mixup 0.2 → 0.5** | **恒久却下** | 下記§3 |
| dropout `drop_rate`/`drop_path_rate` 0.10 | 却下 | Stage1は0.0で過学習していない。参照実装にない機構を持ち込む正当化がない |
| LR変更（4.6e-4等） | 却下 | 下記§4 |
| batch 16 → 32 | 却下 | 下記§5 |
| vertical flip / transpose | **恒久禁止** | R1=vertebral_body と R4=posterior_elements は鏡像関係にない別種の構造で、正しいラベル入れ替えが存在しない |

---

## 3. mixup増量を却下した理由（2026-08-19ユーザー決定）

RSNA原典 `stage2-type1.ipynb` は `p_mixup=0.5`、自前の `stage2/parity.yaml` 以降は 0.2。
v5に残る train/val gap（ep56で **+0.087**、Stage1は **-0.016**）への対処として原典復帰を検討したが**不採用**。

理由は `L_att` との相互作用。設計上 `L_att` は **spatial attention map を領域maskへRMSE回帰させる密な空間損失**である（PMGAN方式、残差形式 `f̂ = (1+m) ⊗ f`）。

1. **教師が意味を失う**: mixupすると attention の教師が `λ·M_a + (1-λ)·M_b` になる。M_a と M_b は位置も大きさも向きも違う別患者の解剖なので、これは実在しない重ね合わせ。分類ラベルは意味論的なので混ざっても解釈できるが、**空間座標は重ね合わせても意味を持たない**。「ここが右横突孔」という空間的対応を教える `L_att` の役目を壊す
2. **H2の検出力を下げる**: `H2: AUROC(Proposed–max β>0) > AUROC(Proposed–max β=0)` は「attention回帰教師の新規性」を検証する仮説。mixupを0.5にすると**全stepの5割で attention 教師が曖昧になり**（現状2割）、検出したい効果そのものを薄める
3. **損失バランスとλ/β校正がずれる**: mixupは natural stream（`L_whole`/`L_att`）にはかかるが annotated stream（`L_region`）にはかからない。増量すると3損失項の相対バランスが変わり、全アーム共通で凍結するλ/βの前提が崩れる

**結論: mixup は `p=0.2` で確定。**

---

## 4. 学習率を2.3e-4で据え置く理由

### 参考コードの系譜（実測で確認）

| 段階 | batch | 実効batch | LR | eta_min |
|---|---|---|---|---|
| RSNA原典 `stage2-type1.ipynb` | 8（単一プロセス） | 8 | `init_lr = 23e-5` = 2.3e-4 | 2.3e-5 |
| 自前 `stage2/config/parity.yaml` | 8 × `n_gpu: 2` | **16** | 2.3e-4 | 2.3e-5 |
| 自前 `stage1/config/config.yaml` | 16（単一GPU） | **16** | 2.3e-4 | 2.3e-5 |
| Baseline 0 | 16 | **16** | 2.3e-4 | 2.3e-5 |

**バッチ8→16の倍増は Baseline 0 ではなく stage2 の時点（2GPU DDP）で行われ、そのときLRは据え置かれていた。** DDPは勾配をプロセス間で平均するので 8×2GPU は単一プロセスのbatch16と等価。つまりBaseline 0は原典から逸脱しておらず、検証済みの実効構成を引き継いでいる。

### 据え置きの根拠

1. **実測検証済み**: Stage1（batch16 + 2.3e-4）が5foldで AUROC 0.909〜0.931、75 epoch完走、過学習なし
2. **実効バッチは240**: `model.py:66` で `inputs.reshape(batch_size * plane_count, ...)` とbagを面へ展開するため、encoder/BNが見るのは 16×15=**240サンプル**（原典は8×15=120）。この規模では bag単位2倍のLR感度は小さい
3. **線形スケーリング則の適用範囲外**: Goyal et al. はImageNet+SGDでバッチ256→8kの経験則。AdamWかつこの規模では根拠が弱い

### v5のval loss挙動はLR問題を示していない

| fold | val_bce最良epoch | そのときのLR | 最終epoch |
|---|---|---|---|
| outer0 | 36 | 1.37e-4 | 56 |
| outer1 | 43 | 1.07e-4 | 63 |
| outer2 | 32 | 1.54e-4 | 52 |
| outer3 | 39 | 1.24e-4 | 59 |

- **高すぎる兆候なし**: val_bceのepoch間変動はLRが高い前半（0.015〜0.030）より低い後半（0.011〜0.016）の方が小さい。発散も振動もない
- **低すぎる兆候もなし**: 低すぎるなら最後まで改善し続けるはずだが、実際は全foldがep32〜43で底を打ち以降は悪化

**残る課題はLRではなく正則化不足。** Stage1との同一epoch比較:

| epoch | Stage1 train/val (gap) | v5 train/val (gap) |
|---|---|---|
| 30 | 0.3356 / 0.3019 (**-0.034**) | 0.2690 / 0.2602 (-0.009) |
| 40 | 0.3086 / 0.2667 (**-0.042**) | 0.2285 / 0.2567 (+0.028) |
| 50 | 0.2801 / 0.2558 (**-0.024**) | 0.2102 / 0.2969 (+0.087) |
| 56 | 0.2695 / 0.2533 (**-0.016**) | 0.1960 / 0.2833 (**+0.087**) |

ep56でv5のtrain lossは0.196、Stage1は0.270。**v5はStage1が最後まで到達しなかった深さまで訓練データにフィットしている。** hflipだけでは発火率50%でStage1の87.5%に届かない分が残っている。

---

## 5. batch 32を却下した理由

1. **速くならない**: GPU utilization が既に **100%**（compute 221s / data_wait 3.5〜4.0s per epoch）。1 epochの総演算量はバッチサイズに依らず不変（8,074 bag × 15面）。入力律速でもない
2. **勾配更新回数が半減**: steps/epoch 505→253、75 epochでの総更新回数 **37,875→18,975**。同じLR・同じepoch数のまま最適化予算が半分になる。補償にLRを上げると2変数を同時に変えることになる
3. メモリは足りるが余裕が減る（現状20.5GB/49GB → 推定38〜40GB）
4. parityが壊れる

**高速化が目的なら、`start_outer_fold`/`end_outer_fold` でfoldを複数GPUへ分割する方が無害**（結果に一切影響しない）。

---

## 6. 並列実行の検証結果

v4（GPU1）とv5（GPU0）を同時実行しても問題なかった。

- CPU: 32論理CPU（AMD EPYC 7313P 16C/32T）に対し1 runあたり約312%（メイン101% + workers 8×約25.7%）。2 run合計でも約625% = 6.3コア相当
- 相互の速度低下なし: data_wait 3.5〜4.0s / compute 221s とGPU律速で、データパイプラインにスラックが大きい
- メモリ: **同一manifest SHA（`9bc0b8b9...`）なので `/dev/shm` キャッシュを再利用**。`[staging] 既存cacheを再利用` を確認、66GBの再コピーは発生しない
- 安全性確認済み: `_remove_stale_temporary_paths` は `.{sha}.tmp-*` のみ対象で最終キャッシュ `{sha}` は消えない。終了時にキャッシュを削除するコードも存在しない
- W&Bプロジェクトは `fracture-08_18-v4` / `fracture-08_18-v5` に自動分離

---

## 7. 現在の状態

- `08_18/v4`（v7）: **5fold完走・正常終了**
- `08_18/v5`（v8）: outer0〜3完了、**outer4がepoch31付近で学習中**（PID 1497764、GPU0、経過18時間半）
- `config/baseline0.yaml`: ユーザーが `08_19/v6` / `patience 30` へ書き換え済み（**未起動**）
- 未commit（指示があるまでcommitしない方針）

---

## 8. 未決事項

1. **`early_stopping_patience` / `max_epochs`** — 現在v6用configは30。ただし全foldがep32〜43で底を打つため、30は過学習領域を30 epoch走るだけで得るものが乏しく、**argmax対象epochが増えてwinner's curseで選択のばらつきが増える**懸念がある。v5のpatience 20で十分捕捉できている
2. **残る train/val gap（+0.087）への対処** — mixup・dropoutとも却下したため、現状は打ち手が残っていない。受け入れるか別の手を探すか
3. **ベースライン設定の最終確定** — **後続アーム（Control–B / Baseline 1–B / Proposed 3構成）の挙動を見てから判断する方針**（2026-08-19ユーザー決定）

---

## 9. 次のセッションでやること

1. v5のouter4完走を確認し、**5fold揃った最終数値**を出す（outer3の0.882が外れ値かどうかも判断できる）
2. **Control–B / Baseline 1–B を共通sampler・nested契約へ接続して実装**
3. Proposed（PMGAN式mask-guided branch）の3構成を実装
4. 5 outer分のλ・βをreference modelで校正し、全アーム共通値として凍結
5. **凍結してからouter推論**

### 後続アーム実装時に必ず守ること

- **領域ラベルを持つアームでhflipを使うときは必ず `common.dataset.flip_horizontal` を通す。** `A.HorizontalFlip` を直接使うとR2/R3が静かに壊れる（Baseline 0はwholeラベルのみなので `A.HorizontalFlip` を直接使っている）
- `L_att` は spatial attention map に直接かけ、**LSTM/head dropoutより前**で計算する
- attention logits/maps そのものへ追加dropoutを置かない
- branch別に異なるdropout値を使わない。`β>0` と `β=0` でdropout位置・乱数処理を同一にする
- mixupでは CT・whole/region mask入力・whole target・`L_att`のmask target を**同じpermutationと同じλ**で混合する。annotated streamの `L_region` にmixupを適用しない現在のtwo-stream設計は問題なし
- **`min_epoch` を上げない**。`trainer.py:228` の `eligible = epoch >= min_epoch` が early stopping と checkpoint保存の両方を制御しており、上げると早期bestが保存対象から外れる
- `folds/outputs/folds.csv` を再生成しない
- outer foldをcheckpoint選択・構成選択・ハイパラ調整に使わない
- Codex CLI に `--full-auto` を付けない

---

## 10. 参照

- 前セッション（v8設計・実装）: `.claude/docs/work-logs/2026-08/2026-08-18-v8-laterality-safe-regularization.md`
- Codex分析（正則化・attention干渉）: `.claude/docs/codex/20260818-1400-laterality-safe-regularization.md`
- 実装ハンドオフ: `.claude/docs/work-logs/2026-08/2026-08-18-implementation-handoff.md`
- 設計本体: `memo/計画書/提案手法.md`
- 進捗台帳: `fracture_detection/PROGRESS.md`
