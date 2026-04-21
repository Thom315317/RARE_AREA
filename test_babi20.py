#!/usr/bin/env python3
"""
RARE-JEPA on bAbI 1-20 benchmark.

Mode: unified simple JEPA (standard, detach target). Pure JEPA routing, no pilot.
  Phase 1 (0-19)  : forced sequential (trains experts + JEPA predictor)
  Phase 2 (20-69) : unified+jepa submode, ε decay 0.3 → 0.05

Data: HuggingFace Muennighoff/babi (bAbI en-1k, 900 train / 1000 test per task).

Logs breakthrough_epoch = first epoch where test_acc >= threshold (default 95%).

Saves to runs/babi20/
"""
import sys, os, time, argparse
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset

import rare_jepa as rj
import babi_loader

# ── CLI ──────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="RARE-JEPA on bAbI 1-20")
parser.add_argument("--tasks", type=str, default="1-20",
                    help="Range or list, e.g. '1-20', '1,3,5', '7-12'")
parser.add_argument("--tag", type=str, default="babi20")
parser.add_argument("--total-ep", type=int, default=70)
parser.add_argument("--p1-end", type=int, default=20)
parser.add_argument("--threshold", type=float, default=95.0,
                    help="Breakthrough threshold (test %%)")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--routing", type=str, default="unified",
                    choices=["unified", "sequential", "gumbel"],
                    help="Baseline diagnostic: --routing sequential")
parser.add_argument("--lr", type=float, default=None,
                    help=f"Override learning rate (default: {rj.LR})")
parser.add_argument("--hidden-dim", type=int, default=None,
                    help=f"Override hidden_dim (default: {rj.HIDDEN_DIM})")
parser.add_argument("--ffn-dim", type=int, default=None,
                    help=f"Override ffn_dim (default: {rj.FFN_DIM})")
args = parser.parse_args()

# Apply overrides BEFORE building any model
if args.lr is not None:
    rj.LR = args.lr
if args.hidden_dim is not None:
    rj.HIDDEN_DIM = args.hidden_dim
if args.ffn_dim is not None:
    rj.FFN_DIM = args.ffn_dim
print(f"  Overrides: LR={rj.LR}  HIDDEN_DIM={rj.HIDDEN_DIM}  FFN_DIM={rj.FFN_DIM}")

def parse_tasks(s):
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out

TASKS = parse_tasks(args.tasks)
TOTAL_EP = args.total_ep
P1_END = args.p1_end
P2_LEN = TOTAL_EP - P1_END
BATCH = args.batch_size

print("=" * 70)
print(f"  RARE-JEPA bAbI 1-20 benchmark")
print(f"  Tasks: {TASKS}")
print(f"  Routing: {args.routing}")
print(f"  Phase 1 (0-{P1_END-1}) : forced sequential")
if args.routing == "sequential":
    print(f"  Phase 2 ({P1_END}-{TOTAL_EP-1}) : continued sequential (baseline)")
elif args.routing == "unified":
    print(f"  Phase 2 ({P1_END}-{TOTAL_EP-1}) : unified+jepa, ε 0.3 → 0.05")
else:
    print(f"  Phase 2 ({P1_END}-{TOTAL_EP-1}) : {args.routing}")
print(f"  Breakthrough threshold: {args.threshold}% test")
print("=" * 70)


# ── Per-task vocab: include atomic answer strings ────────
class Vocab:
    def __init__(self):
        self.w2i = {"<pad>": 0, "<unk>": 1}
        self.i2w = {0: "<pad>", 1: "<unk>"}

    def add(self, word):
        if word not in self.w2i:
            i = len(self.w2i)
            self.w2i[word] = i
            self.i2w[i] = word

    def encode(self, text):
        return [self.w2i.get(w, 1) for w in text.split()]

    def __len__(self):
        return len(self.w2i)


def build_vocab(samples_list):
    v = Vocab()
    for samples in samples_list:
        for s, q, a in samples:
            for w in (s + " " + q).split():
                v.add(w)
            # Answer as ATOMIC (handles multi-word like "south east")
            v.add(a)
    return v


class BabiDataset(Dataset):
    def __init__(self, samples, vocab, max_len=512):
        self.data = []
        for s, q, a in samples:
            toks = vocab.encode(s + " " + q)[:max_len]
            tgt = vocab.w2i.get(a, 1)
            self.data.append((toks, tgt, len(toks)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    toks, tgts, lens = zip(*batch)
    ml = max(lens)
    import torch
    padded = torch.zeros(len(batch), ml, dtype=torch.long)
    mask   = torch.zeros(len(batch), ml, dtype=torch.bool)
    for i, (t, _, l) in enumerate(zip(toks, tgts, lens)):
        padded[i, :l] = torch.tensor(t, dtype=torch.long)
        mask[i, :l] = True
    return padded, mask, torch.tensor(tgts, dtype=torch.long), torch.tensor(lens, dtype=torch.long)


def run_one(task_id, tr_ds, te_ds, vocab_size, t0_all, run_idx, total_runs):
    name = babi_loader.task_name(task_id)
    print(f"\n{'=' * 70}")
    print(f"  Task {task_id} — {name}  [{run_idx+1}/{total_runs}]")
    print(f"  Train: {len(tr_ds)}  Test: {len(te_ds)}  Vocab: {vocab_size}")
    print(f"{'=' * 70}")

    tr_ld = DataLoader(tr_ds, BATCH, shuffle=True, collate_fn=collate_fn,
                       num_workers=0, pin_memory=True)
    te_ld = DataLoader(te_ds, BATCH, shuffle=False, collate_fn=collate_fn,
                       num_workers=0, pin_memory=True)

    rj.seed_everything()
    model = rj.RAREJEPA(vocab_size, routing=args.routing, use_gru=True,
                        jepa_colearn=False).to(rj.DEVICE)
    if args.routing == "unified":
        model.unified_submode = "jepa"
        model.action_mask_mode = "all"
    opt  = torch.optim.Adam(model.parameters(), lr=rj.LR, weight_decay=rj.WEIGHT_DECAY)
    sclr = rj._amp_scaler(rj.USE_AMP)

    hist = defaultdict(list)
    underused = []
    epoch_times = []
    breakthrough_ep = None

    for ep in range(TOTAL_EP):
        if ep < P1_END:
            phase, eps = 1, 0.0
        else:
            progress = (ep - P1_END) / max(P2_LEN - 1, 1)
            eps = 0.3 - progress * (0.3 - 0.05)
            phase = 2
        if args.routing == "unified":
            model.unified_submode = "jepa"

        t0 = time.time()
        trm = rj.train_epoch(model, tr_ld, opt, sclr, phase, eps, underused)
        tem = rj.evaluate(model, te_ld, phase)
        dt = time.time() - t0
        epoch_times.append(dt)
        underused = trm.get("underused", [])

        for k, v in trm.items():
            if isinstance(v, (int, float)):
                hist[f"tr_{k}"].append(v)
        for k, v in tem.items():
            if isinstance(v, (int, float)):
                hist[f"te_{k}"].append(v)

        if breakthrough_ep is None and tem["acc"] >= args.threshold:
            breakthrough_ep = ep

        avg_ep = np.mean(epoch_times)
        remain = (TOTAL_EP - ep - 1) * avg_ep + (total_runs - run_idx - 1) * TOTAL_EP * avg_ep
        mark = "★" if breakthrough_ep == ep else " "
        print(f"  E{ep:02d} {mark} p={phase} ε={eps:.2f} | "
              f"loss {trm['loss']:.3f} cls {trm['L_cls']:.3f} | "
              f"acc {trm['acc']:.1f}/{tem['acc']:.1f}% | "
              f"{dt:.1f}s | ETA {int(remain/60)}m{int(remain%60):02d}s")

    return {
        "hist": dict(hist),
        "epoch_times": epoch_times,
        "breakthrough_ep": breakthrough_ep,
        "final_acc": hist["te_acc"][-1],
        "max_acc": max(hist["te_acc"]),
        "task_name": name,
    }


# ═══════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════
rj.seed_everything()
results = {}
t0_all = time.time()

for i, tid in enumerate(TASKS):
    tr, te = babi_loader.load_task(tid)
    vocab = build_vocab([tr, te])
    tr_ds = BabiDataset(tr, vocab)
    te_ds = BabiDataset(te, vocab)
    results[f"t{tid}"] = run_one(tid, tr_ds, te_ds, len(vocab),
                                   t0_all, i, len(TASKS))

# ═══════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("BABI 1-20 RESULTS — unified simple JEPA (standard)")
print(f"{'=' * 80}")
hdr = f"{'Task':>5} {'Name':<25} {'Final':>8} {'Max':>7} {f'→{args.threshold:.0f}%':>10} {'Time':>8}"
print(hdr)
print("-" * len(hdr))
for tid in TASKS:
    r = results[f"t{tid}"]
    bt = str(r["breakthrough_ep"]) if r["breakthrough_ep"] is not None else "—"
    tt = sum(r["epoch_times"])
    print(f"{tid:>5} {r['task_name'][:25]:<25} {r['final_acc']:>7.1f}% {r['max_acc']:>6.1f}% "
          f"{bt:>10} {tt:>7.0f}s")

# Aggregate stats
max_accs = [results[f"t{t}"]["max_acc"] for t in TASKS]
solved = sum(1 for a in max_accs if a >= args.threshold)
print(f"\nSolved (max ≥ {args.threshold}%): {solved} / {len(TASKS)}")
print(f"Mean max acc: {np.mean(max_accs):.1f}%")
print(f"Median max acc: {np.median(max_accs):.1f}%")

# Plot: breakthrough epoch per task
bt_eps = [results[f"t{t}"]["breakthrough_ep"] or TOTAL_EP for t in TASKS]
fig, ax = plt.subplots(figsize=(max(8, len(TASKS) * 0.5), 5))
bars = ax.bar(range(len(TASKS)), bt_eps, color=["#2ecc71" if results[f"t{t}"]["breakthrough_ep"] is not None
                                                  else "#e74c3c" for t in TASKS])
ax.axhline(P1_END, color="gray", ls="--", alpha=0.4, label="Phase 2 start")
ax.set_xticks(range(len(TASKS)))
ax.set_xticklabels([str(t) for t in TASKS])
ax.set_xlabel("Task ID")
ax.set_ylabel(f"Breakthrough epoch (≥{args.threshold}%)")
ax.set_title(f"bAbI 1-20: epochs to reach {args.threshold}% test accuracy\n(red = not reached in {TOTAL_EP} ep)")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
out = os.path.join(rj.PLOT_DIR, f"babi20_breakthrough.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"\nPlot saved: {out}")

total_min = (time.time() - t0_all) / 60
print(f"Total time: {total_min:.1f} min")

fn = rj.save_run(args.tag, {
    "tasks": TASKS,
    "total_ep": TOTAL_EP,
    "p1_end": P1_END,
    "threshold": args.threshold,
    "results": results,
    "total_min": total_min,
}, subfolder="babi20")
print(f"Run saved: {fn}")
