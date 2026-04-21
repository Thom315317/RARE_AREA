#!/usr/bin/env python3
"""
RARE-UNIFIED Test
=================
Fusion of RARE-MoE (sequential: advance/revise) and Meta Pilot (parallel: jump).

Experiments in this run:
  baseline    : forward fixe 1→6 (reference)
  rare_moe    : unified with only advance + revise actions (sequential MoE)
  meta_pilot  : unified with only jump actions (parallel, mode pilot-selected)
  unified     : all 8 actions (advance + revise + jump×6, mode pilot-selected)

Schedule (35 epochs total):
  Phase 1 (0-19) : forced sequential (trains JEPA predictor)
  Phase 2 (20-28): pilot each submode 3 ep (gumbel / jepa / bary)
  Phase 3 (29-34): winner takes over (6 ep with ε decay)

What we want to see:
  Task 1 (1 hop): unified picks advance → ~baseline performance
  Task 3 (3 hop): unified picks jump(bary) → cracks multi-hop reasoning

Saves to runs/unified/
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
parser = argparse.ArgumentParser(description="RARE-UNIFIED test")
parser.add_argument("--tasks", type=str, default="1,2,3",
                    help="Comma-separated task IDs")
parser.add_argument("--exps", type=str,
                    default="baseline,rare_moe,meta_pilot,unified",
                    help="Comma-separated experiments")
parser.add_argument("--tag", type=str, default="unified_run",
                    help="Name tag for saved run file")
parser.add_argument("--pilot-ep", type=int, default=3,
                    help="Epochs per submode during pilot (default: 3)")
args = parser.parse_args()

TASKS = [int(t) for t in args.tasks.split(",") if t.strip()]
EXPS  = [e.strip() for e in args.exps.split(",") if e.strip()]
PILOT_EP = args.pilot_ep
PILOT_MODES = ["gumbel", "jepa", "bary"]

P1_END = rj.PHASE1_END                                      # 20
PILOT_END = P1_END + len(PILOT_MODES) * PILOT_EP            # 29
WINNER_END = rj.TOTAL_EPOCHS                                # 35
WINNER_EP  = WINNER_END - PILOT_END                         # 6

print("=" * 62)
print(f"  RARE-UNIFIED — Tasks: {TASKS}  Exps: {EXPS}")
print(f"  Phase 1    : ep 0-{P1_END-1}")
print(f"  Pilot      : ep {P1_END}-{PILOT_END-1}  ({PILOT_EP} ep × 3 modes)")
print(f"  Winner     : ep {PILOT_END}-{WINNER_END-1}  ({WINNER_EP} ep)")
print("=" * 62)


def run_one(name, task_id, tr_ld, te_ld, vocab_size, t0_all, run_idx, total_runs):
    """Run one experiment on one task."""
    print(f"\n{'=' * 62}")
    print(f"  {name.upper()} — Task {task_id}  [{run_idx+1}/{total_runs}]")
    print(f"{'=' * 62}")

    rj.seed_everything()

    # ── Build model per experiment ───────────────────────
    if name == "baseline":
        model = rj.RAREJEPA(vocab_size, routing="sequential", use_gru=True).to(rj.DEVICE)
        uses_pilot = False
        action_mask_mode = None
    elif name == "rare_moe":
        # unified with adv_rev only
        model = rj.RAREJEPA(vocab_size, routing="unified", use_gru=True).to(rj.DEVICE)
        model.action_mask_mode = "adv_rev"
        uses_pilot = True
        action_mask_mode = "adv_rev"
    elif name == "meta_pilot":
        # unified with jumps only (equivalent to original meta pilot)
        model = rj.RAREJEPA(vocab_size, routing="unified", use_gru=True).to(rj.DEVICE)
        model.action_mask_mode = "jump"
        uses_pilot = True
        action_mask_mode = "jump"
    elif name == "unified":
        model = rj.RAREJEPA(vocab_size, routing="unified", use_gru=True).to(rj.DEVICE)
        model.action_mask_mode = "all"
        uses_pilot = True
        action_mask_mode = "all"
    else:
        raise ValueError(f"Unknown experiment: {name}")

    opt   = torch.optim.Adam(model.parameters(), lr=rj.LR, weight_decay=rj.WEIGHT_DECAY)
    sclr  = rj._amp_scaler(rj.USE_AMP)

    hist = defaultdict(list)
    underused = []
    epoch_times = []
    pilot_acc = {m: [] for m in PILOT_MODES}
    winner = None
    mode_per_epoch = []
    action_log = []  # list of action_counts per epoch (for unified experiments)

    for ep in range(rj.TOTAL_EPOCHS):
        # ── Schedule ─────────────────────────────────────
        if not uses_pilot:
            # Baseline: phase 1 always (sequential)
            phase, eps = 1, 0.0
            current_mode = "seq"
        elif ep < P1_END:
            phase, eps = 1, 0.0
            model.unified_submode = "jepa"  # train JEPA during Phase 1
            current_mode = "p1"
        elif ep < PILOT_END:
            pilot_idx = (ep - P1_END) // PILOT_EP
            sm = PILOT_MODES[pilot_idx]
            model.unified_submode = sm
            phase, eps = 2, 0.3
            current_mode = f"pilot_{sm}"
        else:
            if winner is None:
                pilot_scores = {m: np.mean(pilot_acc[m]) if pilot_acc[m] else 0.0
                                for m in PILOT_MODES}
                winner = max(pilot_scores, key=pilot_scores.get)
                print(f"\n{'*' * 62}")
                print(f"  WINNER: {winner}  ({pilot_scores[winner]:.1f}%)")
                for m, s in pilot_scores.items():
                    mark = " ★" if m == winner else ""
                    print(f"    {m:<8} {s:.1f}%{mark}")
                print(f"{'*' * 62}\n")
            model.unified_submode = winner
            progress = (ep - PILOT_END) / max(WINNER_EP - 1, 1)
            eps = 0.3 - progress * (0.3 - 0.05)
            phase = 3
            current_mode = f"winner_{winner}"

        mode_per_epoch.append(current_mode)

        # ── Train + eval ─────────────────────────────────
        t0 = time.time()
        trm = rj.train_epoch(model, tr_ld, opt, sclr, phase, eps, underused)
        tem = rj.evaluate(model, te_ld, phase)
        dt  = time.time() - t0
        epoch_times.append(dt)
        underused = trm.get("underused", [])

        if uses_pilot and ep >= P1_END and ep < PILOT_END:
            pilot_idx = (ep - P1_END) // PILOT_EP
            pilot_acc[PILOT_MODES[pilot_idx]].append(tem["acc"])

        for k, v in trm.items():
            if isinstance(v, (int, float)):
                hist[f"tr_{k}"].append(v)
        for k, v in tem.items():
            if isinstance(v, (int, float)):
                hist[f"te_{k}"].append(v)
        hist["mode"].append(current_mode)

        # ETA
        avg_ep = np.mean(epoch_times)
        remain = (rj.TOTAL_EPOCHS - ep - 1) * avg_ep + (total_runs - run_idx - 1) * rj.TOTAL_EPOCHS * avg_ep
        elapsed = time.time() - t0_all

        line = (f"E{ep:02d} [{current_mode:<14}] p={phase} ε={eps:.2f} | "
                f"loss {trm['loss']:.3f} | acc {trm['acc']:.1f}/{tem['acc']:.1f}% | "
                f"{dt:.1f}s | ETA {int(remain/60)}m{int(remain%60):02d}s")
        print(line)

    return {
        "hist": dict(hist),
        "pilot_acc": pilot_acc,
        "winner": winner,
        "mode_per_epoch": mode_per_epoch,
    }


# ═══════════════════════════════════════════════════════
# Run all experiments × tasks
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
total_runs = len(EXPS) * len(TASKS)
run_idx = 0

for tid in TASKS:
    tr_ds = rj.BabiDataset(tr_data[tid], vocab)
    te_ds = rj.BabiDataset(te_data[tid], vocab)
    tr_ld = DataLoader(tr_ds, rj.BATCH_SIZE, shuffle=True,
                       collate_fn=rj.collate_fn, num_workers=0, pin_memory=True)
    te_ld = DataLoader(te_ds, rj.BATCH_SIZE, shuffle=False,
                       collate_fn=rj.collate_fn, num_workers=0, pin_memory=True)
    for name in EXPS:
        results[f"{name}_t{tid}"] = run_one(
            name, tid, tr_ld, te_ld, len(vocab),
            t0_all, run_idx, total_runs
        )
        run_idx += 1

# ═══════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("RARE-UNIFIED RESULTS")
print(f"{'=' * 80}")
hdr = f"{'Exp':<14} " + " ".join(f"{'T'+str(t):>8}" for t in TASKS) + f" {'Avg':>8} {'Winner':>10}"
print(hdr)
print("-" * len(hdr))
for name in EXPS:
    accs, winners = [], []
    for t in TASKS:
        r = results.get(f"{name}_t{t}", {})
        acc = r.get("hist", {}).get("te_acc", [0])[-1]
        accs.append(acc)
        winners.append(r.get("winner") or "—")
    avg = np.mean(accs)
    win_str = ",".join(winners) if any(w != "—" for w in winners) else "—"
    print(f"{name:<14} " + " ".join(f"{a:>7.1f}%" for a in accs) +
          f" {avg:>7.1f}% {win_str:>10}")

# Plot
nT = len(TASKS)
fig, axes = plt.subplots(1, nT, figsize=(5 * nT, 4), squeeze=False)
axes = axes[0]
fig.suptitle("RARE-UNIFIED: Test Accuracy per Task", fontweight="bold")
colors = {"baseline": "#888", "rare_moe": "#e74c3c",
          "meta_pilot": "#2ecc71", "unified": "#8e44ad"}
for i, t in enumerate(TASKS):
    ax = axes[i]
    for name in EXPS:
        k = f"{name}_t{t}"
        if k in results:
            ax.plot(results[k]["hist"]["te_acc"], label=name,
                    color=colors.get(name, "#333"), linewidth=1.5)
    ax.set_title(f"Task {t}"); ax.set_xlabel("Epoch"); ax.set_ylabel("Test Acc %")
    ax.axvline(P1_END, color="gray", ls="--", alpha=.4)
    ax.axvline(PILOT_END, color="gray", ls=":", alpha=.4)
    ax.legend(fontsize=8); ax.grid(True, alpha=.3)
plt.tight_layout()
out = os.path.join(rj.PLOT_DIR, "unified.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"\nPlot saved: {out}")

total_min = (time.time() - t0_all) / 60
print(f"Total time: {total_min:.1f} min")

fn = rj.save_run(args.tag, {
    "results": results,
    "tasks": TASKS,
    "experiments": EXPS,
    "total_min": total_min,
}, subfolder="unified")
print(f"Run saved: {fn}")
