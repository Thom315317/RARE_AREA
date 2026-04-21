#!/usr/bin/env python3
"""
Meta Pilot Test — per-task adaptive routing selection
=====================================================
Architecture: single model with ALL routing components (gumbel_router,
jepa predictor + value_head, ccv_threshold_param). During training,
switches active routing to pilot each mode, then picks the winner.

Schedule (35 epochs):
  Phase 1     (0-19): forced sequential, trains experts + JEPA predictor
  Pilot       (20-28): 3 epochs each in gumbel, jepa, bary
  Winner      (29-34): 6 more epochs in the best-performing pilot mode

Saves to runs/meta/
"""
import sys, os, time, random, argparse, copy
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import DataLoader

import rare_jepa as rj

# ── CLI ──────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Meta pilot routing test")
parser.add_argument("--tasks", type=str, default="1,2,3",
                    help="Comma-separated task IDs (default: 1,2,3)")
parser.add_argument("--tag", type=str, default="meta_pilot",
                    help="Name tag for saved run file")
parser.add_argument("--pilot-ep", type=int, default=3,
                    help="Epochs per mode during pilot phase (default: 3)")
args = parser.parse_args()

TASKS = [int(t) for t in args.tasks.split(",") if t.strip()]
PILOT_EP = args.pilot_ep
PILOT_MODES = ["gumbel", "jepa", "bary"]
# Schedule
P1_END       = rj.PHASE1_END                               # 20
PILOT_START  = P1_END                                      # 20
PILOT_END    = P1_END + len(PILOT_MODES) * PILOT_EP        # 29
WINNER_END   = rj.TOTAL_EPOCHS                             # 35
WINNER_EP    = WINNER_END - PILOT_END                      # 6

assert WINNER_EP > 0, f"TOTAL_EPOCHS too small for PILOT_EP={PILOT_EP}"

print("=" * 62)
print(f"  META PILOT TEST — Tasks: {TASKS}")
print(f"  Phase 1    : ep 0-{P1_END-1} (forced sequential, trains JEPA)")
for i, m in enumerate(PILOT_MODES):
    s = P1_END + i * PILOT_EP
    print(f"  Pilot {m:<6}: ep {s}-{s + PILOT_EP - 1}")
print(f"  Winner     : ep {PILOT_END}-{WINNER_END-1} (best mode, {WINNER_EP} epochs)")
print("=" * 62)


def run_meta_experiment(task_id, tr_ld, te_ld, vocab_size, global_t0, run_idx, total_runs):
    """Run meta pilot for one task."""
    print(f"\n{'=' * 62}")
    print(f"  META — Task {task_id}  [{run_idx+1}/{total_runs}]")
    print(f"{'=' * 62}")

    rj.seed_everything()
    model = rj.RAREJEPA(vocab_size, routing="meta", use_gru=True).to(rj.DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=rj.LR, weight_decay=rj.WEIGHT_DECAY)
    sclr  = rj._amp_scaler(rj.USE_AMP)

    hist = defaultdict(list)
    underused = []
    epoch_times = []
    pilot_acc = {m: [] for m in PILOT_MODES}  # track pilot acc per mode
    winner = None
    mode_per_epoch = []

    for ep in range(rj.TOTAL_EPOCHS):
        # ── Determine phase and active mode ──────────────
        if ep < P1_END:
            phase, eps = 1, 0.0
            # During Phase 1, we want JEPA observations to train the predictor.
            # Bary is compatible too, but JEPA is simpler.
            model.routing = "jepa"
            current_mode = "p1"
        elif ep < PILOT_END:
            # Which pilot mode are we in?
            pilot_idx = (ep - P1_END) // PILOT_EP
            mode = PILOT_MODES[pilot_idx]
            model.routing = mode
            phase, eps = 2, 0.3
            current_mode = f"pilot_{mode}"
        else:
            # Winner phase
            if winner is None:
                # Decide winner from pilot results
                pilot_scores = {m: np.mean(pilot_acc[m]) if pilot_acc[m] else 0.0
                                for m in PILOT_MODES}
                winner = max(pilot_scores, key=pilot_scores.get)
                print(f"\n{'*' * 62}")
                print(f"  WINNER: {winner}  (avg acc during pilot: "
                      f"{pilot_scores[winner]:.1f}%)")
                for m, s in pilot_scores.items():
                    mark = " ★" if m == winner else ""
                    print(f"    {m:<10} {s:.1f}%{mark}")
                print(f"{'*' * 62}\n")
            model.routing = winner
            # Decaying epsilon in winner phase
            progress = (ep - PILOT_END) / max(WINNER_EP - 1, 1)
            eps = 0.3 - progress * (0.3 - 0.05)
            phase = 3
            current_mode = f"winner_{winner}"

        mode_per_epoch.append(current_mode)

        # ── Train and evaluate ───────────────────────────
        t0 = time.time()
        trm = rj.train_epoch(model, tr_ld, opt, sclr, phase, eps, underused)
        tem = rj.evaluate(model, te_ld, phase)
        dt  = time.time() - t0
        epoch_times.append(dt)
        underused = trm.get("underused", [])

        # Track pilot accuracy
        if ep >= P1_END and ep < PILOT_END:
            pilot_idx = (ep - P1_END) // PILOT_EP
            pilot_mode = PILOT_MODES[pilot_idx]
            pilot_acc[pilot_mode].append(tem["acc"])

        # Record history
        for k, v in trm.items():
            if isinstance(v, (int, float)):
                hist[f"tr_{k}"].append(v)
        for k, v in tem.items():
            if isinstance(v, (int, float)):
                hist[f"te_{k}"].append(v)
        hist["mode"].append(current_mode)

        # ETA
        avg_ep = np.mean(epoch_times)
        remain_this = (rj.TOTAL_EPOCHS - ep - 1) * avg_ep
        remain_runs = (total_runs - run_idx - 1) * rj.TOTAL_EPOCHS * avg_ep
        eta = remain_this + remain_runs
        elapsed = time.time() - global_t0
        print(f"E{ep:02d} [{current_mode:<14}] phase={phase} eps={eps:.2f} | "
              f"loss {trm['loss']:.4f} | acc {trm['acc']:.1f}/{tem['acc']:.1f}% | "
              f"{dt:.1f}s | ETA {int(eta/60)}m{int(eta%60):02d}s")

    return {
        "hist": dict(hist),
        "pilot_acc": pilot_acc,
        "winner": winner,
        "mode_per_epoch": mode_per_epoch,
    }


# ═══════════════════════════════════════════════════════
# Run all tasks
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

    results[f"t{tid}"] = run_meta_experiment(
        tid, tr_ld, te_ld, len(vocab),
        global_t0=t0_all, run_idx=i, total_runs=len(TASKS)
    )

# ═══════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════
print(f"\n{'=' * 72}")
print("META PILOT RESULTS")
print(f"{'=' * 72}")
hdr = (f"{'Task':>5} {'Winner':>10} " +
       " ".join(f"{'pilot_'+m:>14}" for m in PILOT_MODES) +
       f" {'final_test':>12}")
print(hdr)
print("-" * len(hdr))
for tid in TASKS:
    r = results[f"t{tid}"]
    winner = r["winner"]
    pilot_scores = {m: np.mean(r["pilot_acc"][m]) if r["pilot_acc"][m] else 0.0
                    for m in PILOT_MODES}
    final = r["hist"]["te_acc"][-1] if r["hist"]["te_acc"] else 0.0
    row = (f"{tid:>5} {winner:>10} " +
           " ".join(f"{pilot_scores[m]:>13.1f}%" for m in PILOT_MODES) +
           f" {final:>11.1f}%")
    print(row)

# Plot: test acc per task with mode annotations
nT = len(TASKS)
fig, axes = plt.subplots(1, nT, figsize=(5 * nT, 4), squeeze=False)
axes = axes[0]
fig.suptitle("Meta Pilot: test acc + mode schedule", fontweight="bold")
for i, tid in enumerate(TASKS):
    ax = axes[i]
    r = results[f"t{tid}"]
    te = r["hist"]["te_acc"]
    ax.plot(te, "-", linewidth=1.5, color="#333")
    # Annotate mode regions
    for ep, mode in enumerate(r["mode_per_epoch"]):
        if "pilot_gumbel" in mode:
            ax.axvspan(ep, ep + 1, alpha=0.15, color="#e74c3c")
        elif "pilot_jepa" in mode:
            ax.axvspan(ep, ep + 1, alpha=0.15, color="#2ecc71")
        elif "pilot_bary" in mode:
            ax.axvspan(ep, ep + 1, alpha=0.15, color="#f39c12")
        elif "winner" in mode:
            ax.axvspan(ep, ep + 1, alpha=0.2, color="#8e44ad")
    ax.set_title(f"Task {tid} — winner: {r['winner']}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Test Acc %")
    ax.grid(True, alpha=.3)
plt.tight_layout()
out = os.path.join(rj.PLOT_DIR, "meta_pilot.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"\nPlot saved: {out}")

total_min = (time.time() - t0_all) / 60
print(f"Total time: {total_min:.1f} min")

fn = rj.save_run(args.tag, {
    "results": results,
    "tasks": TASKS,
    "pilot_modes": PILOT_MODES,
    "pilot_ep": PILOT_EP,
    "total_min": total_min,
}, subfolder="meta")
print(f"Run saved: {fn}")
