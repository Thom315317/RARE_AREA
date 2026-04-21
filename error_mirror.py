#!/usr/bin/env python3
"""
Phase 2 — Détection automatique de lacunes via surprise JEPA.

Pipeline :
  Step 1: Entraîner B0 sur COGS (ou charger un checkpoint existant)
  Step 2: Ajouter un MLP prédicteur léger sur les représentations encoder
          Entraîner le prédicteur à prédire enc(token_{t+1}) depuis enc(token_t)
  Step 3: Passer le gen set → mesurer L_surprise par exemple
  Step 4: Clusterer les exemples haute surprise → comparer aux catégories gen

Usage:
  # Step 1: Entraîner B0 (si pas déjà fait)
  python3 cogs_compositional.py --variant B0 --seed 42 --epochs 60 --runs-dir runs_master

  # Step 2-4: Analyse surprise
  python3 error_mirror.py --checkpoint runs_master/B0_s42_.../checkpoint.pt
  python3 error_mirror.py --checkpoint runs_master/B0_s42_.../checkpoint.pt --dataset slog
"""
import os, sys, json, argparse, math
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cogs_compositional import (
    TransformerSeq2Seq, COGSDataset, collate, parse_cogs_tsv,
    build_vocabs, PAD, BOS, EOS, MAX_IN, MAX_OUT, MAX_DECODE,
)

HERE = os.path.dirname(os.path.abspath(__file__))


# ══════════════════════════════════════════════════════════════
# JEPA predictor — lightweight MLP on encoder representations
# ══════════════════════════════════════════════════════════════
class JEPAPredictor(nn.Module):
    """Predicts encoder representation at position t+1 from position t.
    Lightweight MLP: d_model → d_model → d_model."""

    def __init__(self, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, enc_out, mask):
        """enc_out: (B, L, D), mask: (B, L) bool.
        Returns predictions for positions 1..L-1 from positions 0..L-2."""
        pred = self.net(enc_out[:, :-1])    # (B, L-1, D)
        target = enc_out[:, 1:].detach()    # (B, L-1, D) — stop gradient
        return pred, target


def train_jepa_predictor(model, predictor, train_loader, device, epochs=10, lr=1e-3):
    """Train the JEPA predictor on encoder representations (frozen encoder)."""
    model.eval()
    predictor.train()
    opt = torch.optim.Adam(predictor.parameters(), lr=lr)

    for ep in range(epochs):
        total_loss = 0; n = 0
        for src, src_mask, tgt, _, cats in train_loader:
            src = src.to(device); src_mask = src_mask.to(device)
            with torch.no_grad():
                enc_out = model.encode(src, src_mask)
            pred, target = predictor(enc_out, src_mask)
            # Mask: only compute loss on real tokens (not padding)
            mask = src_mask[:, 1:].unsqueeze(-1).float()   # (B, L-1, 1)
            loss = ((pred - target) ** 2 * mask).sum() / mask.sum() / pred.size(-1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item(); n += 1
        print(f"  JEPA predictor E{ep}: loss={total_loss/max(n,1):.6f}")
    return predictor


# ══════════════════════════════════════════════════════════════
# Surprise measurement
# ══════════════════════════════════════════════════════════════
def measure_surprise(model, predictor, loader, device):
    """Compute per-example surprise = mean L2 prediction error on encoder repr.
    Returns list of (surprise, inp_str, category)."""
    model.eval()
    predictor.eval()
    results = []
    with torch.no_grad():
        for src, src_mask, tgt, _, cats in loader:
            src = src.to(device); src_mask = src_mask.to(device)
            enc_out = model.encode(src, src_mask)
            pred, target = predictor(enc_out, src_mask)
            mask = src_mask[:, 1:]   # (B, L-1)
            err = ((pred - target) ** 2).mean(-1)  # (B, L-1) — per-position error
            for i in range(src.size(0)):
                m = mask[i]
                if m.sum() > 0:
                    surprise = err[i][m].mean().item()
                else:
                    surprise = 0.0
                results.append((surprise, cats[i]))
    return results


def measure_decoder_entropy(model, loader, device):
    """Measure per-example mean decoder entropy (teacher-forced).
    High entropy = decoder hesitates = lexical/generation gap."""
    model.eval()
    results = []
    with torch.no_grad():
        for src, src_mask, tgt, _, cats in loader:
            src = src.to(device); src_mask = src_mask.to(device); tgt = tgt.to(device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits, _ = model(src, src_mask, tgt_in)
            # Entropy per position: -sum(p * log(p))
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(-1)          # (B, T)
            mask = (tgt_out != 0)                            # (B, T)
            for i in range(src.size(0)):
                m = mask[i]
                if m.sum() > 0:
                    ent = entropy[i][m].mean().item()
                else:
                    ent = 0.0
                results.append((ent, cats[i]))
    return results


def analyze_surprise(results, top_k=500):
    """Cluster high-surprise examples by category, compare to ground truth."""
    # Sort by surprise descending
    results.sort(key=lambda x: -x[0])

    # Overall stats by category
    by_cat = defaultdict(list)
    for surprise, cat in results:
        by_cat[cat].append(surprise)

    print(f"\n{'='*70}")
    print(f"  SURPRISE ANALYSIS — {len(results)} examples")
    print(f"{'='*70}")

    # Mean surprise per category
    cat_stats = {}
    for cat, vals in sorted(by_cat.items()):
        cat_stats[cat] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n": len(vals),
        }
    print(f"\n{'Category':<35s} {'Mean surpr':>10s} {'Std':>8s} {'N':>5s}")
    print("-" * 62)
    for cat, s in sorted(cat_stats.items(), key=lambda x: -x[1]["mean"]):
        print(f"  {cat:<33s} {s['mean']:9.6f} {s['std']:7.5f} {s['n']:>5d}")

    # Top-K analysis: which categories dominate the highest surprise?
    top = results[:top_k]
    top_cats = defaultdict(int)
    for _, cat in top:
        top_cats[cat] += 1
    print(f"\n  Top-{top_k} highest surprise — category distribution:")
    for cat, count in sorted(top_cats.items(), key=lambda x: -x[1]):
        pct = count / top_k * 100
        print(f"    {cat:<33s} {count:>4d} ({pct:5.1f}%)")

    return cat_stats


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="Phase 2 — JEPA surprise detector")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to B0 checkpoint.pt")
    p.add_argument("--dataset", type=str, default="cogs",
                   choices=["cogs", "slog"])
    p.add_argument("--jepa-epochs", type=int, default=10,
                   help="Epochs to train JEPA predictor")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out-dir", type=str, default="runs_master")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    in_w2i = ckpt["in_w2i"]
    out_w2i = ckpt["out_w2i"]
    d_model = ckpt["d_model"]
    max_in = ckpt.get("max_in", MAX_IN)
    max_out = ckpt.get("max_out", MAX_OUT)

    # Rebuild model
    model = TransformerSeq2Seq(len(in_w2i), len(out_w2i), d_model=d_model,
                               max_in=max_in, max_out=max_out).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    # Load data
    if args.dataset == "cogs":
        data_dir = os.path.join(HERE, "data", "cogs")
        train_pairs = parse_cogs_tsv(os.path.join(data_dir, "train.tsv"))
        gen_pairs = parse_cogs_tsv(os.path.join(data_dir, "gen.tsv"))
    else:
        data_dir = os.path.join(HERE, "data", "slog")
        train_pairs = parse_cogs_tsv(os.path.join(data_dir, "train.tsv"))
        gen_pairs = parse_cogs_tsv(os.path.join(data_dir,
                                   "generalization_sets", "gen_cogsLF.tsv"))

    # Use the SAME vocabs as the checkpoint (critical!)
    train_ds = COGSDataset(train_pairs, in_w2i, out_w2i)
    gen_ds = COGSDataset(gen_pairs, in_w2i, out_w2i)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0)
    gen_loader = DataLoader(gen_ds, args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=0)

    # Step 2: Train JEPA predictor
    print(f"\nTraining JEPA predictor ({args.jepa_epochs} epochs)...")
    predictor = JEPAPredictor(d_model).to(device)
    train_jepa_predictor(model, predictor, train_loader, device,
                         epochs=args.jepa_epochs)

    # Step 3: Measure surprise on gen
    print("\nMeasuring surprise on gen set...")
    gen_results = measure_surprise(model, predictor, gen_loader, device)

    # Step 4: Analyze encoder surprise
    cat_stats = analyze_surprise(gen_results, top_k=500)

    # Step 5: Measure decoder entropy
    print("\nMeasuring decoder entropy on gen set (teacher-forced)...")
    gen_entropy = measure_decoder_entropy(model, gen_loader, device)
    print("\nMeasuring decoder entropy on train set (baseline)...")
    train_entropy = measure_decoder_entropy(model, train_loader, device)

    # Cross-analysis: surprise × entropy per category
    # Build per-example joined data
    surprise_by_cat = defaultdict(list)
    for surprise, cat in gen_results:
        surprise_by_cat[cat].append(surprise)
    entropy_by_cat = defaultdict(list)
    for ent, cat in gen_entropy:
        entropy_by_cat[cat].append(ent)

    train_ent_mean = float(np.mean([e for e, _ in train_entropy]))
    train_surp_mean = float(np.mean([s for s, _ in gen_results[:len(train_entropy)]]))

    print(f"\n{'='*70}")
    print(f"  CROSS-ANALYSIS — Encoder Surprise × Decoder Entropy")
    print(f"{'='*70}")
    print(f"  Train baseline: surprise={float(np.mean([s for s,_ in measure_surprise(model, predictor, train_loader, device)])):.4f}  entropy={train_ent_mean:.4f}")
    print(f"\n{'Category':<40s} {'Enc surpr':>9s} {'Dec ent':>8s} {'Gap type':>15s}")
    print("-" * 75)
    all_cats = sorted(set(c for _, c in gen_results))
    for cat in sorted(all_cats, key=lambda c: -np.mean(surprise_by_cat[c])):
        s = float(np.mean(surprise_by_cat[cat]))
        e = float(np.mean(entropy_by_cat[cat]))
        # Classify gap type
        high_s = s > 0.06   # above median
        high_e = e > train_ent_mean * 1.5
        if high_s and high_e:
            gap = "STRUCTURAL"
        elif not high_s and high_e:
            gap = "DECODER-ONLY"
        elif high_s and not high_e:
            gap = "ENCODER-ONLY"
        else:
            gap = "low"
        print(f"  {cat:<38s} {s:8.4f} {e:7.4f}   {gap}")

    # Also measure on train for baseline comparison
    print("\nMeasuring surprise on train set (baseline)...")
    train_results = measure_surprise(model, predictor, train_loader, device)
    train_surprises = [s for s, _ in train_results]
    gen_surprises = [s for s, _ in gen_results]
    print(f"\n  Train surprise: mean={np.mean(train_surprises):.6f} std={np.std(train_surprises):.6f}")
    print(f"  Gen surprise:   mean={np.mean(gen_surprises):.6f} std={np.std(gen_surprises):.6f}")
    print(f"  Ratio gen/train: {np.mean(gen_surprises)/max(np.mean(train_surprises), 1e-10):.2f}x")
    print(f"  Train entropy:  mean={train_ent_mean:.6f}")
    print(f"  Gen entropy:    mean={float(np.mean([e for e,_ in gen_entropy])):.6f}")

    # Save results
    out_path = os.path.join(args.out_dir, f"surprise_{args.dataset}.json")
    cat_full = {}
    for cat in all_cats:
        cat_full[cat] = {
            "surprise_mean": float(np.mean(surprise_by_cat[cat])),
            "surprise_std": float(np.std(surprise_by_cat[cat])),
            "entropy_mean": float(np.mean(entropy_by_cat[cat])),
            "entropy_std": float(np.std(entropy_by_cat[cat])),
            "n": len(surprise_by_cat[cat]),
        }
    out_data = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "jepa_epochs": args.jepa_epochs,
        "train_surprise_mean": float(np.mean(train_surprises)),
        "train_entropy_mean": train_ent_mean,
        "gen_surprise_mean": float(np.mean(gen_surprises)),
        "gen_entropy_mean": float(np.mean([e for e, _ in gen_entropy])),
        "by_category": cat_full,
        "top500_categories": dict(sorted(
            {cat: sum(1 for s, c in sorted(gen_results, key=lambda x: -x[0])[:500] if c == cat)
             for cat in set(c for _, c in gen_results)}.items(),
            key=lambda x: -x[1])),
    }
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
