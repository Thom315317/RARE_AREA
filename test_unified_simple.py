#!/usr/bin/env python3
"""
Unified Simple — Pure JEPA routing, no pilot, 50 epochs
========================================================
Phase 1 (0-19)  : forced sequential (trains experts + JEPA predictor)
Phase 2 (20-49) : unified+jepa submode, ε decay 0.3 → 0.05

No gumbel pilot, no bary pilot. Just give JEPA more time to learn.
Saves to runs/unified_simple/
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

# ── CLI ──────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Unified-simple: pure JEPA, no pilot")
parser.add_argument("--tasks", type=str, default="1,2,3")
parser.add_argument("--tag", type=str, default="unified_simple")
parser.add_argument("--total-ep", type=int, default=50)
parser.add_argument("--p1-end", type=int, default=20)
parser.add_argument("--jepa-colearn", action="store_true",
                    help="Enable JEPA co-learning: L_jepa gradient flows into experts "
                         "(target = live h_real, no detach). JEPA_COEFF reduced to 0.05.")
args = parser.parse_args()

TASKS = [int(t) for t in args.tasks.split(",") if t.strip()]
TOTAL_EP = args.total_ep
P1_END = args.p1_end
P2_LEN = TOTAL_EP - P1_END

print("=" * 62)
print(f"  UNIFIED SIMPLE — JEPA only, no pilot")
print(f"  Phase 1 (0-{P1_END-1}) : forced sequential")
print(f"  Phase 2 ({P1_END}-{TOTAL_EP-1}) : unified+jepa, ε 0.3 → 0.05")
if args.jepa_colearn:
    print(f"  ⚡ JEPA CO-LEARNING enabled (L_jepa grad → experts, coef=0.05)")
else:
    print(f"  JEPA standard (EMA target, no grad to experts, coef=0.1)")
print(f"  Tasks: {TASKS}")
print("=" * 62)


def run_one(task_id, tr_ld, te_ld, vocab_size, t0_all, run_idx, total_runs):
    print(f"\n{'=' * 62}")
    print(f"  Task {task_id}  [{run_idx+1}/{total_runs}]")
    print(f"{'=' * 62}")

    rj.seed_everything()
    model = rj.RAREJEPA(vocab_size, routing="unified", use_gru=True,
                        jepa_colearn=args.jepa_colearn).to(rj.DEVICE)
    model.unified_submode = "jepa"
    model.action_mask_mode = "all"

    opt  = torch.optim.Adam(model.parameters(), lr=rj.LR, weight_decay=rj.WEIGHT_DECAY)
    sclr = rj._amp_scaler(rj.USE_AMP)

    hist = defaultdict(list)
    underused = []
    epoch_times = []
    action_counts_per_epoch = []

    for ep in range(TOTAL_EP):
        if ep < P1_END:
            phase, eps = 1, 0.0
        else:
            progress = (ep - P1_END) / max(P2_LEN - 1, 1)
            eps = 0.3 - progress * (0.3 - 0.05)
            phase = 2

        # Ensure submode stays jepa
        model.unified_submode = "jepa"

        t0 = time.time()
        trm = rj.train_epoch(model, tr_ld, opt, sclr, phase, eps, underused)
        tem = rj.evaluate(model, te_ld, phase)
        dt  = time.time() - t0
        epoch_times.append(dt)
        underused = trm.get("underused", [])

        for k, v in trm.items():
            if isinstance(v, (int, float)):
                hist[f"tr_{k}"].append(v)
        for k, v in tem.items():
            if isinstance(v, (int, float)):
                hist[f"te_{k}"].append(v)

        # ETA
        avg_ep = np.mean(epoch_times)
        remain = (TOTAL_EP - ep - 1) * avg_ep + (total_runs - run_idx - 1) * TOTAL_EP * avg_ep
        line = (f"E{ep:02d} p={phase} ε={eps:.2f} | "
                f"loss {trm['loss']:.3f} cls {trm['L_cls']:.3f} jepa {trm.get('L_jepa', 0):.3f} val {trm.get('L_value', 0):.3f} | "
                f"acc {trm['acc']:.1f}/{tem['acc']:.1f}% | halt {trm.get('avg_halt', 0):.1f} | "
                f"{dt:.1f}s | ETA {int(remain/60)}m{int(remain%60):02d}s")
        print(line)

    return dict(hist)


# ═══════════════════════════════════════════════════════
rj.seed_everything()
rng = random.Random(rj.SEED)
tr_data, te_data = {}, {}
for tid in [1, 2, 3]:
    samples = rj.TASK_GEN[tid](rj.N_TRAIN + rj.N_TEST, rng)
    tr_data[tid] = samples[:rj.N_TRAIN]
    te_data[tid] = samples[rj.N_TRAIN:]
vocab = rj.build_vocab(list(tr_data.values()) + list(te_data.values()))
print(f"Vocab: {len(vocab)} tokens")

results = {}
t0_all = time.time()
for i, tid in enumerate(TASKS):
    tr_ds = rj.BabiDataset(tr_data[tid], vocab)
    te_ds = rj.BabiDataset(te_data[tid], vocab)
    tr_ld = DataLoader(tr_ds, rj.BATCH_SIZE, shuffle=True,
                       collate_fn=rj.collate_fn, num_workers=0, pin_memory=True)
    te_ld = DataLoader(te_ds, rj.BATCH_SIZE, shuffle=False,
                       collate_fn=rj.collate_fn, num_workers=0, pin_memory=True)
    results[f"t{tid}"] = run_one(tid, tr_ld, te_ld, len(vocab),
                                   t0_all, i, len(TASKS))

# ═══════════════════════════════════════════════════════
print(f"\n{'=' * 62}")
print("UNIFIED SIMPLE RESULTS")
print(f"{'=' * 62}")
hdr = f"{'Task':>6} {'Final test':>12} {'Max test':>12} {'@ epoch':>10}"
print(hdr)
print("-" * len(hdr))
for tid in TASKS:
    te = results[f"t{tid}"]["te_acc"]
    final = te[-1]
    mx = max(te)
    mx_ep = te.index(mx)
    print(f"{tid:>6} {final:>11.1f}% {mx:>11.1f}% {mx_ep:>10}")

# Plot
nT = len(TASKS)
fig, axes = plt.subplots(1, nT, figsize=(5 * nT, 4), squeeze=False)
axes = axes[0]
fig.suptitle("Unified Simple: JEPA only, no pilot", fontweight="bold")
for i, tid in enumerate(TASKS):
    ax = axes[i]
    te = results[f"t{tid}"]["te_acc"]
    ax.plot(te, "-", linewidth=1.5, color="#2ecc71")
    ax.axvline(P1_END, color="gray", ls="--", alpha=.4, label="Phase 2 start")
    ax.set_title(f"Task {tid} — final {te[-1]:.1f}% / max {max(te):.1f}%")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Test Acc %")
    ax.grid(True, alpha=.3); ax.legend(fontsize=8)
plt.tight_layout()
out = os.path.join(rj.PLOT_DIR, "unified_simple.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"\nPlot saved: {out}")

total_min = (time.time() - t0_all) / 60
print(f"Total time: {total_min:.1f} min")

fn = rj.save_run(args.tag, {
    "results": results,
    "tasks": TASKS,
    "total_ep": TOTAL_EP,
    "p1_end": P1_END,
    "jepa_colearn": args.jepa_colearn,
    "total_min": total_min,
}, subfolder="unified_simple")
print(f"Run saved: {fn}")
