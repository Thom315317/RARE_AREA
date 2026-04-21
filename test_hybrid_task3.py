#!/usr/bin/env python3
"""
Test the hybrid JEPA routing on Task 3 only.

Compares: baseline | gumbel | jepa | jepa_hybrid
All on Task 3, 35 epochs, seed 42. ~30 min total.

The jepa_hybrid uses Gumbel-Softmax on JEPA value_scores with
step-dependent temperature:
  tau = 2.0 at t=0 (JEPA unreliable → more exploration)
  tau = 1.0 at t=1
  tau = 0.5 at t>=2 (JEPA reliable → exploitation)
"""
import sys, os, time, random
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import rare_jepa as rj

print("=" * 62)
print("  HYBRID JEPA TEST — Task 3 only")
print("=" * 62)

# Experiments to compare on Task 3
TEST_EXPS = ["baseline", "gumbel", "jepa", "jepa_hybrid"]

rj.seed_everything()
rng = random.Random(rj.SEED)

# Task 3 data
samples = rj._gen_task3(rj.N_TRAIN + rj.N_TEST, rng)
tr_data = samples[:rj.N_TRAIN]
te_data = samples[rj.N_TRAIN:]
vocab = rj.build_vocab([tr_data, te_data])
print(f"Vocab: {len(vocab)}")

tr_ds = rj.BabiDataset(tr_data, vocab)
te_ds = rj.BabiDataset(te_data, vocab)
tr_ld = DataLoader(tr_ds, rj.BATCH_SIZE, shuffle=True,
                   collate_fn=rj.collate_fn, num_workers=0, pin_memory=True)
te_ld = DataLoader(te_ds, rj.BATCH_SIZE, shuffle=False,
                   collate_fn=rj.collate_fn, num_workers=0, pin_memory=True)

results = {}
t0_all = time.time()

for i, name in enumerate(TEST_EXPS):
    cfg = rj.EXPERIMENTS[name]
    hist = rj.run_experiment(
        name, cfg, 3, tr_ld, te_ld, len(vocab),
        run_idx=i, total_runs=len(TEST_EXPS), global_t0=t0_all
    )
    results[name] = hist

# ═══════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  FINAL TEST ACC — Task 3")
print("=" * 62)
print(f"{'Experiment':<15} {'Test %':>10} {'L_jepa final':>14} {'Halt avg':>10}")
print("-" * 52)
for name in TEST_EXPS:
    h = results[name]
    acc = h["te_acc"][-1]
    ljepa = h.get("tr_L_jepa", [0])[-1]
    halt = h.get("te_avg_halt", [0])[-1]
    mark = ""
    if name == "jepa_hybrid":
        gumbel_acc = results.get("gumbel", {}).get("te_acc", [0])[-1]
        jepa_acc = results.get("jepa", {}).get("te_acc", [0])[-1]
        if acc > gumbel_acc and acc > jepa_acc:
            mark = " ★ BEATS BOTH"
        elif acc > jepa_acc:
            mark = " > jepa"
    print(f"{name:<15} {acc:>9.1f}% {ljepa:>14.4f} {halt:>9.1f}{mark}")

# Plot
fig, ax = plt.subplots(figsize=(9, 5))
colors = {"baseline": "#888", "gumbel": "#e74c3c",
          "jepa": "#2ecc71", "jepa_hybrid": "#8e44ad"}
for name in TEST_EXPS:
    ax.plot(results[name]["te_acc"], label=name,
            color=colors.get(name, "#333"), linewidth=2)
ax.axvline(rj.PHASE1_END, color="gray", linestyle="--", alpha=0.4, label="Phase 2")
ax.axvline(rj.PHASE2_END, color="gray", linestyle=":",  alpha=0.4, label="Phase 3")
ax.set_xlabel("Epoch")
ax.set_ylabel("Test Accuracy %")
ax.set_title("Hybrid JEPA vs baselines — Task 3")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = os.path.join(rj.PLOT_DIR, "hybrid_task3.png")
plt.savefig(out, dpi=150)
print(f"\nPlot saved: {out}")

total_min = (time.time() - t0_all) / 60
print(f"\nTotal time: {total_min:.1f} min")

fn = rj.save_run("hybrid_task3", {
    "results": results,
    "total_min": total_min,
})
print(f"Run saved: {fn}")
