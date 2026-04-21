#!/usr/bin/env python3
"""
Benchmark wall-clock cost vs accuracy for different routing methods.

Measures per-epoch time + test accuracy for:
  baseline      : sequential 1→6 (no routing)
  gumbel        : GumbelRouter MLP + 1 expert forward per step
  unified_jepa  : UnifiedRouter + batched JEPA (6 preds) + 1 expert forward

Same task, same config, same seed. Reports:
  - Time per epoch (Phase 1 / Phase 2)
  - Total time
  - Final + max test accuracy
  - Epochs to reach 95% threshold
  - Time-to-convergence

Saves to runs/benchmark/
"""
import sys, os, time, random, argparse
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from collections import defaultdict
from torch.utils.data import DataLoader

import rare_jepa as rj

parser = argparse.ArgumentParser(description="Cost/accuracy benchmark")
parser.add_argument("--task", type=int, default=1)
parser.add_argument("--total-ep", type=int, default=50)
parser.add_argument("--p1-end", type=int, default=20)
parser.add_argument("--tag", type=str, default="benchmark")
parser.add_argument("--threshold", type=float, default=95.0,
                    help="Acc %% threshold to measure convergence time")
args = parser.parse_args()

TASK = args.task
TOTAL_EP = args.total_ep
P1_END = args.p1_end

METHODS = [
    {"name": "baseline",     "routing": "sequential", "submode": None,
     "desc": "Fixed 1→6, no routing"},
    {"name": "gumbel",       "routing": "gumbel",     "submode": None,
     "desc": "GumbelRouter MLP + 1 expert/step"},
    {"name": "unified_jepa", "routing": "unified",    "submode": "jepa",
     "desc": "UnifiedRouter + batched JEPA(6) + 1 expert/step"},
]

print("=" * 70)
print(f"  COST BENCHMARK — Task {TASK}")
print(f"  Phase 1 (0-{P1_END-1}) : forced sequential")
print(f"  Phase 2 ({P1_END}-{TOTAL_EP-1}) : routing active")
print("=" * 70)

rj.seed_everything()
rng = random.Random(rj.SEED)
samples = rj.TASK_GEN[TASK](rj.N_TRAIN + rj.N_TEST, rng)
tr_data = samples[:rj.N_TRAIN]
te_data = samples[rj.N_TRAIN:]
vocab = rj.build_vocab([tr_data, te_data])
print(f"Vocab: {len(vocab)} tokens\n")

tr_ds = rj.BabiDataset(tr_data, vocab)
te_ds = rj.BabiDataset(te_data, vocab)
tr_ld = DataLoader(tr_ds, rj.BATCH_SIZE, shuffle=True,
                   collate_fn=rj.collate_fn, num_workers=0, pin_memory=True)
te_ld = DataLoader(te_ds, rj.BATCH_SIZE, shuffle=False,
                   collate_fn=rj.collate_fn, num_workers=0, pin_memory=True)

bench = {}
t0_all = time.time()

for m_idx, cfg in enumerate(METHODS):
    print(f"\n{'─' * 70}\n▶ {cfg['name']}: {cfg['desc']}\n{'─' * 70}")
    rj.seed_everything()

    model = rj.RAREJEPA(len(vocab), routing=cfg["routing"], use_gru=True).to(rj.DEVICE)
    if cfg["submode"] is not None:
        model.unified_submode = cfg["submode"]
        model.action_mask_mode = "all"
    opt   = torch.optim.Adam(model.parameters(), lr=rj.LR, weight_decay=rj.WEIGHT_DECAY)
    sclr  = rj._amp_scaler(rj.USE_AMP)

    epoch_times = []
    test_accs = []
    p1_times, p2_times = [], []
    underused = []

    for ep in range(TOTAL_EP):
        if ep < P1_END:
            phase, eps = 1, 0.0
        else:
            progress = (ep - P1_END) / max(TOTAL_EP - P1_END - 1, 1)
            eps = 0.3 - progress * (0.3 - 0.05)
            phase = 2

        if cfg["submode"] is not None:
            model.unified_submode = cfg["submode"]

        torch.cuda.synchronize() if rj.USE_AMP else None
        t0 = time.time()
        trm = rj.train_epoch(model, tr_ld, opt, sclr, phase, eps, underused)
        tem = rj.evaluate(model, te_ld, phase)
        torch.cuda.synchronize() if rj.USE_AMP else None
        dt = time.time() - t0
        epoch_times.append(dt)
        test_accs.append(tem["acc"])
        (p1_times if phase == 1 else p2_times).append(dt)
        underused = trm.get("underused", [])

        print(f"  E{ep:02d} p={phase} | test {tem['acc']:5.1f}% | {dt:5.1f}s")

    # Compute metrics
    max_acc = max(test_accs)
    final_acc = test_accs[-1]
    # Epochs to reach threshold
    ep_to_thr = next((i for i, a in enumerate(test_accs) if a >= args.threshold), None)
    time_to_thr = sum(epoch_times[:ep_to_thr + 1]) if ep_to_thr is not None else None

    bench[cfg["name"]] = {
        "epoch_times": epoch_times,
        "test_accs": test_accs,
        "p1_avg": np.mean(p1_times),
        "p2_avg": np.mean(p2_times) if p2_times else 0,
        "total_time": sum(epoch_times),
        "final_acc": final_acc,
        "max_acc": max_acc,
        "ep_to_thr": ep_to_thr,
        "time_to_thr": time_to_thr,
    }

# ═══════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print(f"  COST / ACCURACY SUMMARY — Task {TASK}, {TOTAL_EP} epochs")
print(f"{'=' * 80}")
hdr = (f"{'Method':<14} {'P1 s/ep':>8} {'P2 s/ep':>8} {'Total':>8} "
       f"{'Final':>8} {'Max':>7} {f'→{args.threshold:.0f}%':>8} {'Time→':>10}")
print(hdr)
print("-" * len(hdr))
for name in bench:
    b = bench[name]
    ep_str = str(b["ep_to_thr"]) if b["ep_to_thr"] is not None else "—"
    time_str = f"{b['time_to_thr']:.0f}s" if b["time_to_thr"] is not None else "—"
    print(f"{name:<14} {b['p1_avg']:>7.1f}s {b['p2_avg']:>7.1f}s "
          f"{b['total_time']:>7.0f}s "
          f"{b['final_acc']:>7.1f}% {b['max_acc']:>6.1f}% "
          f"{ep_str:>8} {time_str:>10}")

# Relative comparison
print(f"\n{'─' * 80}")
print("RATIOS (vs baseline)")
print(f"{'─' * 80}")
base = bench["baseline"]
for name in bench:
    b = bench[name]
    time_ratio = b["total_time"] / base["total_time"]
    acc_delta = b["max_acc"] - base["max_acc"]
    cost_per_point = (b["total_time"] - base["total_time"]) / max(acc_delta, 0.01) if acc_delta > 0 else float("nan")
    print(f"{name:<14} time ×{time_ratio:.2f}  max_acc {acc_delta:+.1f}pp  "
          f"cost/gain: {cost_per_point:.0f}s per +1pp" if acc_delta > 0 else
          f"{name:<14} time ×{time_ratio:.2f}  max_acc {acc_delta:+.1f}pp  (no gain)")

# ═══════════════════════════════════════════════════════
# Plot: accuracy vs time (Pareto-like)
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: test acc vs epoch
ax = axes[0]
colors = {"baseline": "#888", "gumbel": "#e74c3c", "unified_jepa": "#2ecc71"}
for name, b in bench.items():
    ax.plot(b["test_accs"], label=name, color=colors.get(name, "#333"), linewidth=1.8)
ax.axhline(args.threshold, color="gray", ls=":", alpha=0.5, label=f"{args.threshold:.0f}% threshold")
ax.axvline(P1_END, color="gray", ls="--", alpha=0.3)
ax.set_xlabel("Epoch"); ax.set_ylabel("Test Acc %")
ax.set_title(f"Task {TASK} — accuracy vs epoch")
ax.legend(); ax.grid(True, alpha=0.3)

# Right: test acc vs wall-clock time
ax = axes[1]
for name, b in bench.items():
    cum_time = np.cumsum(b["epoch_times"])
    ax.plot(cum_time, b["test_accs"], label=name, color=colors.get(name, "#333"), linewidth=1.8)
ax.axhline(args.threshold, color="gray", ls=":", alpha=0.5)
ax.set_xlabel("Wall-clock time (s)"); ax.set_ylabel("Test Acc %")
ax.set_title(f"Task {TASK} — accuracy vs wall-clock time (Pareto)")
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(rj.PLOT_DIR, f"benchmark_cost_t{TASK}.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"\nPlot saved: {out}")

total_min = (time.time() - t0_all) / 60
print(f"Total benchmark time: {total_min:.1f} min")

fn = rj.save_run(args.tag, {
    "task": TASK,
    "total_ep": TOTAL_EP,
    "p1_end": P1_END,
    "threshold": args.threshold,
    "bench": bench,
    "total_min": total_min,
}, subfolder="benchmark")
print(f"Run saved: {fn}")
