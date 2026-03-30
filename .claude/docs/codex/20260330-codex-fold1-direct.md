# Codex Analysis: fold1-degradation (wandb + non-determinism)
Date: 2026-03-30

## Question

PyTorch 5-fold CV で fold1 だけ精度劣化する問題。
- wandb=True: fold1=17.77°（他は5-6°）
- wandb=False: fold1=7.60°（他は5-6°）

## Codex Response (直接実行結果)

結論: wandb自体が主因というより「同一プロセス内のfold連続実行 + fold1の不安定さ」が原因の可能性が高い。
最も確実な修正は「1 fold = 1 process」にして wandb.reinit=True をやめること。

Q1: wandb.init() は set_seed(42) 後の random/numpy を消費するか？
→ No。run id生成は secrets.choice を使いグローバル random 依存なし。
   実測（mode='disabled'）でも py/np/torch RNG state は変化しなかった（5回 reinit でも不変）。

Q2: kaiming_uniform_ は numpy/Python RNG を使うか？
→ No。torch RNGのみ。

Q3: reinit=True の2回目（fold1）でGPU状態に影響するか？
→ 直接影響はないが、同一プロセス再初期化は副作用源になりやすい（multiprocessing併用時）。

Q4: なぜ fold1 だけか？
→ fold1は元々難しいfoldで微小な非決定性が最も増幅されやすい。
   かつ fold1 は「最初の finish->reinit 境界」を跨ぐ最初のfold。
   (wandb.initはモデル生成前/trainer.py:638、seed再設定はfold先頭のみ/train.py:120)

Q5: 決定的な修正:
→ 推奨本命: foldごとに別プロセス実行（--start_fold f --end_fold f）で reinit=True 連鎖をやめる。
→ 追加策1: wandb.init() 直後に set_seed(seed) を再実行
→ 追加策2: torch.use_deterministic_algorithms(True) を有効化
→ 追加策3: num_workers=0 で再実験（DataLoader並列要因の除去）
