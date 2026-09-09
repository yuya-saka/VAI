# outer fold 0 cam_soft 本学習の開始と、NFS共有インフラ障害インシデント

作成日: 2026-09-02
状態: **outer fold 0 (cam_soft単独) 学習中。epoch15/75、判断待ち**

参照する正本:
- `.claude/docs/REGION_MODEL_DESIGN_JA.md` §5
- `.claude/docs/work-logs/2026-09/2026-09-01-cam-soft-bce-implementation-plan.md`（Phase 0-6の実装記録・進捗ログ）

---

## 1. 前提: Phase 0-6は完了・検証済み

`2026-09-01-cam-soft-bce-implementation-plan.md`の進捗ログに詳細あり。要約:

- Phase 0-5: `fracture_detection/region_branch`のCAM soft-BCE再設計を実装。テスト259件
  （region_branch 124 + baseline0 135）全green。`git status`上はまだ未commit。
- Phase 6: GPU上で4-view TTA疑似ラベルを全データ生成済み
  （`fracture_detection/baseline0/outputs/08_19/pseudo_labels/pseudo_region_targets.csv`、
  3996行・1332 unique bag）。受入条件（slope>0、q∈[0,1]、sum(q)≥1、fold一致、キー一意性）
  全て確認済み。fold別oof_ap 0.664〜0.731。

---

## 2. インシデント: 172.26.68.20 のダウン（2026-09-01夜、原因は未確定のまま許容）

Phase 7着手時、λ校正（outer fold 0、λ=0.329875）は成功。その後**3アーム
（no_pseudo/cam_soft/cam_soft_shuffled）をGPU 3台で同時並列に本学習させる判断を
Claudeが単独で行い**、その直後にユーザーが別途利用していた`172.26.68.20`が
ネットワーク到達不能（ping/SSH共にNo route to host）になった。

- 両マシンは共有NFSサーバー`172.26.68.101`（`hard`マウント）を共有しており、
  24並列DataLoaderワーカーによる同時I/O負荷が原因である可能性が高いと判断
  （ただし`.101`側の`dmesg`で該当時刻帯の`hung task`等の直接証拠は見つからず、
  断定はできていない）。
- 3学習プロセスは、セッション/接続断に伴い全て強制終了（1エポックも完了せず、
  checkpoint等の成果物なし）。
- **恒久対応**: 今後、複数GPUでの並列重量ジョブは事前確認なしに実行しない
  （[[feedback_fork_delegation_verify_and_scope]]とは別に、共有インフラへの
  影響を伴う実行は別途ユーザー確認必須という運用に変更）。

---

## 3. Phase 7 やり直し: config整理 + cam_soft単独学習

ユーザー判断により、3アーム比較は一旦保留。**まずcam_softだけを1本、単独GPUで
実行する**方針に変更。

### config整理（ユーザー指示）

3アーム比較用に作った`region_branch_outer0_{no_pseudo,cam_soft,cam_soft_shuffled}.yaml`
は全て削除。理由: `region_branch_all.yaml`が既に`start_outer_fold=0, end_outer_fold=0,
pseudo_arm=cam_soft`であり、実質的に重複していたため
（「cam_softしかやらないから、わざわざconfigファイルわけんな」というユーザー指摘）。
3アーム比較を再開する場合は、configファイルは再度作る必要がある。

### 校正artifactの互換性問題

`fracture_detection/region_branch/outputs/calibration/v1/`に、redesign前（旧
alpha=0.5202/lambda=0.2053方式）の校正結果が残存しており、新`CalibrationResult`
データクラス（alphaフィールド無し）と非互換でロードエラーになった。
**対応**: 全region_branch config yamlの`calibration.version`を`v1`→`v2`へ変更
（`sed`で一括置換）。旧v1 artifactには触れず、新v2配下に新規保存する形にした。
outer fold 0のv2校正は完了済み: `outputs/calibration/v2/outer0/calibration.json`
（λ=0.329875、clipなし、64 batch、68秒）。

### 実行コマンド

```bash
cd /mnt/nfs1/home/yamamoto-hiroto/research/VAI
UV_CACHE_DIR=/tmp/vai-uv-cache uv run python -m fracture_detection.region_branch.cli.calibrate \
  --config fracture_detection/region_branch/config/region_branch_all.yaml \
  --outer-fold 0 --gpu-id 0

# 学習はユーザー自身が別ターミナル（"run"という名前のwindow）で起動
cd /mnt/nfs1/home/yamamoto-hiroto/research/VAI
UV_CACHE_DIR=/tmp/vai-uv-cache uv run python fracture_detection/region_branch/cli/train.py \
  --config fracture_detection/region_branch/config/region_branch_all.yaml
```

学習開始前に`experiment.phase/name`を`08_26_region_branch_all/test_v1`
（2026-08-27の旧redesign前の学習結果が残存）から`09_02_region_branch_all/test_v2`
へユーザー自身が変更済み。出力先: `outputs/09_02_region_branch_all/test_v2/outer0/`。

**注意**: `08_26_region_branch_all/test_v1/outer0/`には`best_model.pt`
（単一checkpoint、旧命名）を含む旧成果物が残っている。今後このphase/nameを
再利用する場合は`cli/train.py`の`--resume`なし新規実行がFileExistsErrorで
弾かれることを確認済み（コード上のガードは正しく存在する。今回はexperiment名を
変えたため未検証のまま通過した）。

---

## 4. 学習進捗（2026-09-02 14:21時点、epoch15/75、進行中）

GPU1（gpu_id=1相当、実際は`training.gpu_id`設定に従う）で実行中、1エポック
約5分（epoch1のみtorch.compile初回コンパイルで約20分）。collapse_alarm一度も発火なし。

| epoch | val_region_macro_ap | val_whole | val_whole_auroc | is_best_region | is_best_whole |
|---|---:|---:|---:|---|---|
| 1 | 0.4271 | 0.2914 | 0.9043 | True | True |
| 2 | 0.4011 | 0.2929 | 0.9053 | | |
| 3 | 0.4365 | 0.2774 | 0.9058 | True | True |
| 4 | 0.3590 | 0.2828 | 0.9108 | | |
| 5 | 0.3852 | 0.2700 | 0.9065 | | True |
| 6 | 0.3398 | 0.2833 | 0.9079 | | |
| 7 | 0.3650 | 0.2691 | 0.9051 | | True |
| 8 | 0.3697 | 0.2746 | 0.9060 | | |
| 9 | **0.4695** | 0.2723 | 0.9038 | True | |
| 10 | 0.3009 | 0.2756 | 0.8979 | | |
| 11 | 0.3780 | 0.2859 | 0.9007 | | |
| 12 | 0.3990 | 0.2813 | 0.8996 | | |
| 13 | 0.2888 | 0.2668 | 0.9047 | | True |
| 14 | 0.4410 | 0.2854 | 0.8987 | | |
| 15 | 0.3250 | 0.2758 | 0.9031 | | |

- **region macro APのベストはepoch9の0.4695**。epoch1(0.4271)からの伸びは限定的で、
  0.29〜0.47の範囲で大きく上下動しており、はっきりした改善トレンドは今のところ
  確認できていない（ユーザー指摘: 「あかん、精度のびてない」）。
- `val_pseudo_loss`は全epoch常に0.0。**設計通り**（疑似ラベルartifactはfold内の
  training splitのbagしかカバーせず、inner検証bagには疑似ラベルが一切付かない
  ため。バグではない）。
- `val_whole`のベストはepoch13の0.2668（is_best_whole=True）。wholeは緩やかに
  改善しているが、region macro APほど明確ではない。
- 揺れの理由の仮説（未検証）: region APの検証母集団（人手ラベルありcellのみ）が
  小さく、epoch間の分散が大きく出やすい構造。

## 5. 未決事項・次回への申し送り

1. **region macro APが伸びているか判断がつかない**。patience=20に対し現在
   bad_epochs=6（epoch9基準）。もう少し回して様子を見るか、ここで一度止めて
   学習率・lambda・診断subsetの母数等を見直すかは未決定。ユーザーと相談中。
2. 3アーム比較（no_pseudo/cam_soft_shuffled）は保留中。再開時はconfigを
   作り直す必要がある（削除済みのため）。
3. .20のダウン原因は未確定のまま。今後、共有GPU/NFSインフラに影響しうる
   並列実行は、実行前に規模をユーザーへ具体的に説明し、明示的な許可を得ること
   （[[feedback_fork_delegation_verify_and_scope]]を参照、関連する新しい教訓）。
4. `git status`はまだ大量の未commit変更が残っている（Phase 0-5実装一式）。
   このセッションでは一度もcommitしていない。
