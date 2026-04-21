#!/usr/bin/env python3
"""
H1 Diagnostic: Does JEPA prediction error accumulate over steps on Task 3?

Runs JEPA on Task 3 only, tracks per-step prediction error at each epoch.
Plot: jepa_error vs step t, for final epoch.
If error grows monotonically with t → H1 confirmed.
"""
import sys, os, time, random
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import rare_jepa as rj

print("=" * 62)
print("  H1 DIAGNOSTIC — JEPA Task 3 only (per-step error)")
print("=" * 62)

rj.seed_everything()
rng = random.Random(rj.SEED)

# Generate Task 3 data only
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

cfg = rj.EXPERIMENTS["jepa"]
t0 = time.time()
hist = rj.run_experiment("jepa", cfg, 3, tr_ld, te_ld, len(vocab),
                         run_idx=0, total_runs=1, global_t0=t0)

# ═══════════════════════════════════════════════════════
# Analysis: per-step error at final epoch
# ═══════════════════════════════════════════════════════
final_per_step = hist["te_jerr_per_step"][-1]  # dict {t: err}

print("\n" + "=" * 62)
print("  PER-STEP JEPA ERROR (final epoch)")
print("=" * 62)
print(f"{'Step t':>8} | {'jepa_err':>12}")
print("-" * 24)
steps_sorted = sorted(final_per_step.keys())
errs = [final_per_step[t] for t in steps_sorted]
for t, e in zip(steps_sorted, errs):
    bar = "█" * int(e * 50)
    print(f"{t:>8} | {e:>12.4f} {bar}")

# Verdict
if len(errs) >= 3:
    early = np.mean(errs[:len(errs)//2])
    late  = np.mean(errs[len(errs)//2:])
    growth = (late - early) / (early + 1e-6) * 100
    print(f"\nEarly steps avg: {early:.4f}")
    print(f"Late  steps avg: {late:.4f}")
    print(f"Growth: {growth:+.1f}%")
    if growth > 20:
        print("\n✓ H1 CONFIRMED: JEPA error accumulates with steps.")
    elif growth < -20:
        print("\n✗ H1 REJECTED: JEPA error DECREASES with steps.")
    else:
        print("\n? H1 INCONCLUSIVE: JEPA error roughly constant.")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: per-step error
ax = axes[0]
ax.plot(steps_sorted, errs, "o-", linewidth=2, markersize=8, color="#e74c3c")
ax.set_xlabel("Step t")
ax.set_ylabel("JEPA prediction error (MSE)")
ax.set_title("H1: JEPA error per step (Task 3, final epoch)")
ax.grid(True, alpha=0.3)

# Plot 2: evolution across epochs (error at each step over training)
ax = axes[1]
all_epochs = hist["te_jerr_per_step"]
max_t = max(max(d.keys(), default=0) for d in all_epochs) + 1
err_matrix = np.zeros((len(all_epochs), max_t))
for ep, d in enumerate(all_epochs):
    for t, e in d.items():
        err_matrix[ep, t] = e
im = ax.imshow(err_matrix.T, aspect="auto", cmap="YlOrRd", origin="lower")
ax.set_xlabel("Epoch")
ax.set_ylabel("Step t")
ax.set_title("JEPA error heatmap: epoch × step")
plt.colorbar(im, ax=ax)

plt.tight_layout()
out = os.path.join(rj.PLOT_DIR, "h1_diagnostic.png")
plt.savefig(out, dpi=150)
print(f"\nPlot saved: {out}")

total_min = (time.time() - t0) / 60
print(f"\nTotal time: {total_min:.1f} min")
print(f"Final test acc: {hist['te_acc'][-1]:.1f}%")
print(f"Final avg halt: {hist['te_avg_halt'][-1]:.1f}")

# Save diagnostic run
fn = rj.save_run("diagnostic_h1_task3", {
    "hist": hist,
    "per_step_final": final_per_step,
    "steps_sorted": steps_sorted,
    "errs": errs,
    "early_avg": float(early) if len(errs) >= 3 else None,
    "late_avg":  float(late)  if len(errs) >= 3 else None,
    "growth_pct": float(growth) if len(errs) >= 3 else None,
    "total_min": total_min,
})
print(f"Run saved: {fn}")
