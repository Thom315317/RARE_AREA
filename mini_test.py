#!/usr/bin/env python3
"""Mini integration test: 2 epochs, 1 task, all 4 experiments."""
import sys, os
sys.path.insert(0, ".")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import rare_jepa as rj
import random
import numpy as np
from torch.utils.data import DataLoader

rj.seed_everything()

# Tiny dataset
rng = random.Random(42)
tr = rj._gen_task1(64, rng)
te = rj._gen_task1(16, rng)
vocab = rj.build_vocab([tr, te])
print(f"Vocab: {len(vocab)}")

tr_ds = rj.BabiDataset(tr, vocab)
te_ds = rj.BabiDataset(te, vocab)
tr_ld = DataLoader(tr_ds, 16, shuffle=True, collate_fn=rj.collate_fn, num_workers=0)
te_ld = DataLoader(te_ds, 16, shuffle=False, collate_fn=rj.collate_fn, num_workers=0)

# Override epochs for speed
orig_total = rj.TOTAL_EPOCHS
rj.TOTAL_EPOCHS = 2
rj.PHASE1_END = 1
rj.PHASE2_END = 1

for name, cfg in rj.EXPERIMENTS.items():
    print(f"\n{'='*40} {name}")
    hist = rj.run_experiment(name, cfg, 1, tr_ld, te_ld, len(vocab))
    print(f"Keys: {sorted(hist.keys())}")
    print(f"te_acc: {hist.get('te_acc', [])}")
    print(f"te_heatmap shape: {np.array(hist.get('te_heatmap', [[]])[0]).shape if hist.get('te_heatmap') else 'N/A'}")

rj.TOTAL_EPOCHS = orig_total
rj.PHASE1_END = 20
rj.PHASE2_END = 30
print("\n=== MINI TEST PASSED ===")
