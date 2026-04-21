#!/usr/bin/env python3
"""
Barycentric Routing Test — 6 experiments × 3 tasks
===================================================
Tests the barycentric routing (CRISTAL-inspired) against all baselines.

Experiments:
  baseline         : sequential 1→6 (reference)
  gumbel           : blind Gumbel-ST router
  jepa             : JEPA + ValueHead (existing)
  jepa_hybrid      : JEPA + Gumbel noise on value_scores
  jepa_bary        : JEPA + barycentric geometry routing (NEW)
  jepa_bary_nogru  : bary ablation (m=0)

Routing logic for jepa_bary:
  1. JEPA predicts z_pred_k for all 6 experts
  2. bary = mean(z_preds)
  3. CCV = mean pairwise cosine distance between predictions
  4. If CCV < threshold: stability mode → pick most central expert
     Else:               exploration mode → pick most divergent
  5. Halt when classifier(bary) confidence > 0.95

Saves to runs/barycentric/
"""
import sys, os, time, random, argparse
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import rare_jepa as rj

# ── CLI options ──────────────────────────────────────────
DEFAULT_EXPS = [
    "baseline", "gumbel", "jepa", "jepa_hybrid",
    "jepa_bary", "jepa_bary_nogru",
]
parser = argparse.ArgumentParser(description="Barycentric routing test")
parser.add_argument("--exps", type=str, default=",".join(DEFAULT_EXPS),
                    help="Comma-separated experiments to run. "
                         "Available: " + ", ".join(rj.EXPERIMENTS.keys()) +
                         f". Default: all 6 ({','.join(DEFAULT_EXPS)}). "
                         "Use --exps jepa_bary,jepa_bary_nogru to only run new bary.")
parser.add_argument("--tasks", type=str, default="1,2,3",
                    help="Comma-separated task IDs (1,2,3). Default: all.")
parser.add_argument("--tag", type=str, default="barycentric_suite",
                    help="Name tag for saved run file.")
args = parser.parse_args()

TEST_EXPS = [e.strip() for e in args.exps.split(",") if e.strip()]
TASKS     = [int(t) for t in args.tasks.split(",") if t.strip()]

# Validate
unknown = [e for e in TEST_EXPS if e not in rj.EXPERIMENTS]
if unknown:
    print(f"ERROR: unknown experiments: {unknown}")
    print(f"Available: {list(rj.EXPERIMENTS.keys())}")
    sys.exit(1)

print("=" * 62)
print(f"  BARYCENTRIC ROUTING TEST — {len(TEST_EXPS)} exps × {len(TASKS)} tasks")
print(f"  Experiments: {TEST_EXPS}")
print(f"  Tasks: {TASKS}")
print("=" * 62)

rj.seed_everything()
rng = random.Random(rj.SEED)

# Generate data for selected tasks (also keep reference of all for consistent vocab)
tr_data, te_data = {}, {}
for tid in [1, 2, 3]:
    samples = rj.TASK_GEN[tid](rj.N_TRAIN + rj.N_TEST, rng)
    tr_data[tid] = samples[:rj.N_TRAIN]
    te_data[tid] = samples[rj.N_TRAIN:]

vocab = rj.build_vocab(list(tr_data.values()) + list(te_data.values()))
print(f"Vocab: {len(vocab)} tokens")

results = {}
t0_all = time.time()
total_runs = len(TEST_EXPS) * len(TASKS)
run_idx = 0

for tid in TASKS:
    tr_ds = rj.BabiDataset(tr_data[tid], vocab)
    te_ds = rj.BabiDataset(te_data[tid], vocab)
    tr_ld = DataLoader(tr_ds, rj.BATCH_SIZE, shuffle=True,
                       collate_fn=rj.collate_fn, num_workers=0, pin_memory=True)
    te_ld = DataLoader(te_ds, rj.BATCH_SIZE, shuffle=False,
                       collate_fn=rj.collate_fn, num_workers=0, pin_memory=True)

    for name in TEST_EXPS:
        cfg = rj.EXPERIMENTS[name]
        hist = rj.run_experiment(
            name, cfg, tid, tr_ld, te_ld, len(vocab),
            run_idx=run_idx, total_runs=total_runs, global_t0=t0_all
        )
        results[f"{name}_t{tid}"] = hist
        run_idx += 1

# ═══════════════════════════════════════════════════════
# Summary Table
# ═══════════════════════════════════════════════════════
print(f"\n{'=' * 72}")
print("FINAL TEST ACCURACY %")
print(f"{'=' * 72}")
hdr = f"{'Experiment':<18} " + " ".join(f"{'Task'+str(t):>8}" for t in TASKS) + f" {'Avg':>8}"
print(hdr)
print("-" * len(hdr))
for name in TEST_EXPS:
    accs = []
    for t in TASKS:
        a = results.get(f"{name}_t{t}", {}).get("te_acc", [0])[-1]
        accs.append(a)
    avg = np.mean(accs)
    row = f"{name:<18} " + " ".join(f"{a:>7.1f}%" for a in accs) + f" {avg:>7.1f}%"
    print(row)

# ═══════════════════════════════════════════════════════
# Kill Criteria Check (only if relevant experiments present)
# ═══════════════════════════════════════════════════════
if "jepa_bary" in TEST_EXPS:
    print(f"\n{'=' * 62}")
    print("BARYCENTRIC KILL CRITERIA")
    print(f"{'=' * 62}")
    for t in TASKS:
        bary_a = results.get(f"jepa_bary_t{t}", {}).get("te_acc", [0])[-1]
        print(f"\nTask {t}:  bary = {bary_a:.1f}%")
        for comp in ["gumbel", "jepa", "jepa_bary_nogru"]:
            if comp in TEST_EXPS:
                comp_a = results.get(f"{comp}_t{t}", {}).get("te_acc", [0])[-1]
                ok = "YES" if bary_a > comp_a else "NO"
                print(f"  bary > {comp:<18}? {ok}  ({bary_a:.1f}% vs {comp_a:.1f}%)")

# ═══════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════
colors = {
    "baseline": "#888", "gumbel": "#e74c3c", "jepa": "#2ecc71",
    "jepa_hybrid": "#8e44ad",
    "jepa_bary": "#f39c12", "jepa_bary_nogru": "#3498db",
}

# Plot a: Accuracy per task vs epochs
nT = len(TASKS)
fig, axes = plt.subplots(1, nT, figsize=(5 * nT, 4), sharey=True, squeeze=False)
axes = axes[0]
fig.suptitle("(a) Test Accuracy per Task", fontweight="bold")
for i, t in enumerate(TASKS):
    ax = axes[i]
    for name in TEST_EXPS:
        k = f"{name}_t{t}"
        if k in results:
            ax.plot(results[k]["te_acc"], label=name,
                    color=colors.get(name, "#333"), linewidth=1.5)
    ax.set_title(f"Task {t}"); ax.set_xlabel("Epoch")
    if i == 0: ax.set_ylabel("Test Acc %")
    ax.axvline(rj.PHASE1_END, color="gray", ls="--", alpha=.4)
    ax.axvline(rj.PHASE2_END, color="gray", ls=":",  alpha=.4)
    ax.legend(fontsize=7); ax.grid(True, alpha=.3)
plt.tight_layout()
out = os.path.join(rj.PLOT_DIR, "bary_a_accuracy.png")
plt.savefig(out, dpi=150)
plt.close()

# Plot b: CCV evolution (jepa_bary only, per task)
if "jepa_bary" in TEST_EXPS:
    fig, axes = plt.subplots(1, nT, figsize=(5 * nT, 4), squeeze=False)
    axes = axes[0]
    fig.suptitle("(b) CCV mean vs Epochs (jepa_bary)", fontweight="bold")
    for i, t in enumerate(TASKS):
        ax = axes[i]
        k = f"jepa_bary_t{t}"
        if k in results:
            hist = results[k]
            if "tr_L_ccv" in hist:
                ax.plot(hist["tr_L_ccv"], color=colors["jepa_bary"], linewidth=1.5)
        ax.set_title(f"Task {t}"); ax.set_xlabel("Epoch"); ax.set_ylabel("L_ccv")
        ax.grid(True, alpha=.3)
    plt.tight_layout()
    out = os.path.join(rj.PLOT_DIR, "bary_b_ccv.png")
    plt.savefig(out, dpi=150)
    plt.close()

# Plot d: Final accuracy comparison bar
fig, ax = plt.subplots(figsize=(max(8, 2 * len(TEST_EXPS)), 5))
x = np.arange(len(TEST_EXPS))
width = 0.8 / max(nT, 1)
for i, t in enumerate(TASKS):
    accs = [results.get(f"{n}_t{t}", {}).get("te_acc", [0])[-1] for n in TEST_EXPS]
    offset = (i - (nT - 1) / 2) * width
    ax.bar(x + offset, accs, width, label=f"Task {t}", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(TEST_EXPS, rotation=20, ha="right")
ax.set_ylabel("Test Accuracy %")
ax.set_title("Final Test Accuracy")
ax.legend()
ax.grid(True, alpha=.3, axis="y")
plt.tight_layout()
out = os.path.join(rj.PLOT_DIR, "bary_d_final.png")
plt.savefig(out, dpi=150)
plt.close()

total_min = (time.time() - t0_all) / 60
print(f"\nTotal time: {total_min:.1f} min")
print(f"Plots saved: {rj.PLOT_DIR}/bary_*.png")

# Save to barycentric/ subfolder
fn = rj.save_run(args.tag, {
    "results": results,
    "total_min": total_min,
    "experiments": TEST_EXPS,
    "tasks": TASKS,
}, subfolder="barycentric")
print(f"Run saved: {fn}")
