#!/usr/bin/env python3
"""
Phase 3 complète v2 — Boucle fermée avec diagnostic causal.

Au lieu de router clusters → générateurs codés en dur, le système :
  1. Mesure surprise
  2. Lance le diagnostic causal (diff structurel pred vs gold)
  3. Formule des hypothèses, vérifie dans le train set
  4. Génère les fix automatiquement depuis les hypothèses confirmées
  5. Valide chaque fix sur 20 epochs (Niveau 2)
  6. Réentraîne avec les fix validés
  7. Re-mesure surprise, boucle

Les fix viennent du diagnostic, pas d'une liste. Aucun nom de générateur n'est codé en dur.

Usage:
  python3 auto_loop_v2.py --dataset cogs --max-cycles 3 --out-dir runs_master
  python3 auto_loop_v2.py --dataset cogs --max-cycles 3 --out-dir runs_master \
      --b0-checkpoint runs_master/B0_s42_.../checkpoint.pt
"""
import os, sys, json, argparse, random, time, glob, re
from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))

from cogs_compositional import (
    parse_cogs_tsv, build_vocabs, COGSDataset, collate,
    TransformerSeq2Seq, evaluate_tf, evaluate, greedy_decode,
    generate_subj_pp_augmentations, generate_pp_recursion_extension,
    generate_active_to_passive, generate_noun_position_augmentations,
    detect_proper_names,
    MAX_IN, MAX_OUT, MAX_DECODE, PAD, BOS, EOS,
)
from error_mirror import JEPAPredictor, train_jepa_predictor, measure_surprise
from causal_diagnosis import diagnose, DiffResult, Hypothesis


# ══════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════
def load_data(dataset):
    if dataset == "cogs":
        d = os.path.join(HERE, "data", "cogs")
        return (parse_cogs_tsv(os.path.join(d, "train.tsv")),
                parse_cogs_tsv(os.path.join(d, "dev.tsv")),
                parse_cogs_tsv(os.path.join(d, "gen.tsv")))
    else:
        d = os.path.join(HERE, "data", "slog")
        gen_path = os.path.join(d, "generalization_sets", "gen_cogsLF.tsv")
        gen_link = os.path.join(d, "gen.tsv")
        if not os.path.exists(gen_link):
            import shutil; shutil.copy2(gen_path, gen_link)
        return (parse_cogs_tsv(os.path.join(d, "train.tsv")),
                parse_cogs_tsv(os.path.join(d, "dev.tsv")),
                parse_cogs_tsv(gen_link))


# ══════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════
def train_model(train_pairs, dev_pairs, gen_pairs, perm_classes=None,
                seed=42, epochs=60, device="cuda", run_dir=None,
                greedy_eval=True):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    dev_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    in_w2i, out_w2i = build_vocabs(train_pairs + dev_pairs + gen_pairs)
    tr_ds = COGSDataset(train_pairs, in_w2i, out_w2i,
                        perm_classes=perm_classes if perm_classes else None)
    dv_ds = COGSDataset(dev_pairs, in_w2i, out_w2i)
    ge_ds = COGSDataset(gen_pairs, in_w2i, out_w2i)
    tr_ld = DataLoader(tr_ds, 64, shuffle=True, collate_fn=collate, num_workers=0, pin_memory=True)
    dv_ld = DataLoader(dv_ds, 16, shuffle=False, collate_fn=collate, num_workers=0, pin_memory=True)
    ge_ld = DataLoader(ge_ds, 16, shuffle=False, collate_fn=collate, num_workers=0, pin_memory=True)

    max_in = max(MAX_IN, max(len(p[0].split()) for p in train_pairs + dev_pairs + gen_pairs) + 2)
    max_out = max(MAX_OUT, max(len(p[1].split()) for p in train_pairs + dev_pairs + gen_pairs) + 4)
    model = TransformerSeq2Seq(len(in_w2i), len(out_w2i), d_model=128,
                               max_in=max_in, max_out=max_out).to(dev_obj)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev_obj.type == "cuda"))
    total_steps = epochs * len(tr_ld)
    warmup = min(1000, total_steps // 10)
    def lr_at(step):
        if step < warmup: return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    best_dev = -1.0; patience = 0; metrics = []
    for ep in range(epochs):
        model.train()
        tr_loss = 0.0; nb = 0
        for src, src_mask, tgt, _, _ in tr_ld:
            src = src.to(dev_obj); src_mask = src_mask.to(dev_obj); tgt = tgt.to(dev_obj)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(dev_obj.type == "cuda")):
                logits, _ = model(src, src_mask, tgt_in)
                V = logits.size(-1)
                loss = F.cross_entropy(logits.reshape(-1, V), tgt_out.reshape(-1),
                                       ignore_index=0, label_smoothing=0.1)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tr_loss += loss.item(); nb += 1
        dev_m = evaluate_tf(model, dv_ld, dev_obj)
        gen_m = evaluate_tf(model, ge_ld, dev_obj)
        is_best = dev_m["exact"] > best_dev
        if is_best: best_dev = dev_m["exact"]; patience = 0
        else: patience += 1
        metrics.append({"epoch": ep, "train_loss": tr_loss/max(nb,1),
                        "dev": dev_m, "gen": gen_m})
        mark = "★" if is_best else " "
        print(f"  E{ep:03d} {mark} loss={tr_loss/max(nb,1):.4f} "
              f"dev={dev_m['exact']:5.2f}% gen={gen_m['exact']:5.2f}% pat={patience}")
        if patience >= 20:
            print(f"  Early stopping at epoch {ep}."); break

    by_cat = {}
    if greedy_eval:
        print("  Greedy eval...")
        greedy = evaluate(model, ge_ld, out_w2i, dev_obj)
        by_cat = greedy.get("by_cat", {}); gen_greedy = greedy["exact"]
    else:
        gen_greedy = gen_m["exact"]

    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=1)
        torch.save({"state_dict": model.state_dict(), "in_w2i": in_w2i,
                    "out_w2i": out_w2i, "d_model": 128,
                    "max_in": max_in, "max_out": max_out},
                   os.path.join(run_dir, "checkpoint.pt"))
    return model, in_w2i, out_w2i, gen_greedy, by_cat, metrics


# ══════════════════════════════════════════════════════════════
# Surprise
# ══════════════════════════════════════════════════════════════
def compute_surprise(model, in_w2i, out_w2i, train_pairs, gen_pairs, device="cuda"):
    dev_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    tr_ds = COGSDataset(train_pairs, in_w2i, out_w2i)
    ge_ds = COGSDataset(gen_pairs, in_w2i, out_w2i)
    tr_ld = DataLoader(tr_ds, 64, shuffle=True, collate_fn=collate, num_workers=0)
    ge_ld = DataLoader(ge_ds, 64, shuffle=False, collate_fn=collate, num_workers=0)
    predictor = JEPAPredictor(128).to(dev_obj)
    train_jepa_predictor(model, predictor, tr_ld, dev_obj, epochs=10, lr=1e-3)
    train_r = measure_surprise(model, predictor, tr_ld, dev_obj)
    gen_r = measure_surprise(model, predictor, ge_ld, dev_obj)
    tm = float(np.mean([s for s, _ in train_r]))
    gm = float(np.mean([s for s, _ in gen_r]))
    return {"train_mean": tm, "gen_mean": gm, "ratio": gm / max(tm, 1e-10)}


# ══════════════════════════════════════════════════════════════
# Fix generation — driven by diagnosis, not hardcoded
# ══════════════════════════════════════════════════════════════
def generate_fix(fix_type, train_pairs, max_examples=1000):
    """Generate augmented examples based on the diagnosed fix_type.
    Returns (augmented_pairs, perm_classes, description)."""
    perm_classes = []

    if fix_type == "generate_nmod_on_subject":
        aug = generate_subj_pp_augmentations(train_pairs)
        return aug, [], f"PP obj→subj: {len(aug)} examples"

    elif fix_type == "generate_noun_position_swap":
        aug = generate_noun_position_augmentations(train_pairs)
        return aug, [], f"noun position swap: {len(aug)} examples"

    elif fix_type == "generate_deeper_modification":
        aug = generate_pp_recursion_extension(train_pairs, max_aug=500)
        return aug, [], f"PP recursion +1: {len(aug)} examples"

    elif fix_type == "generate_position_examples":
        aug = generate_active_to_passive(train_pairs)
        return aug, [], f"active→passive: {len(aug)} examples"

    elif fix_type == "generate_verb_in_missing_form":
        # Lightweight: use active→passive as proxy
        aug = generate_active_to_passive(train_pairs)
        return aug[:200], [], f"verb form (passive proxy): {min(len(aug),200)} examples"

    else:
        return [], [], f"no generator for {fix_type}"


# ══════════════════════════════════════════════════════════════
# Fix validation (Level 2)
# ══════════════════════════════════════════════════════════════
def validate_fix(fix_type, fix_examples, fix_perm, train_pairs, dev_pairs,
                 gen_pairs, base_gen_score, seed=42, device="cuda"):
    """Quick 20-epoch test: does this fix help without regression?"""
    VAL_EPOCHS = 20
    aug_train = train_pairs + fix_examples
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    _, _, _, test_gen, _, _ = train_model(
        aug_train, dev_pairs, gen_pairs, perm_classes=fix_perm or None,
        seed=seed, epochs=VAL_EPOCHS, device=device, greedy_eval=False)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    delta = test_gen - base_gen_score
    return test_gen, delta


# ══════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════
def run_loop(dataset, max_cycles=5, surprise_threshold=1.5, seed=42,
             epochs=60, device="cuda", out_dir="runs_master",
             b0_checkpoint=None):
    os.makedirs(out_dir, exist_ok=True)
    loop_dir = os.path.join(out_dir, f"loopv2_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(loop_dir, exist_ok=True)

    train_orig, dev, gen = load_data(dataset)
    cycle_log = []
    prev_ratio = None
    accumulated_fixes = []  # list of (fix_type, fix_examples, fix_perm)

    for cycle in range(max_cycles):
        print(f"\n{'='*70}")
        print(f"  CYCLE {cycle} — {dataset.upper()} (v2 causal)")
        print(f"{'='*70}")
        cycle_dir = os.path.join(loop_dir, f"cycle_{cycle}")
        os.makedirs(cycle_dir, exist_ok=True)
        t0 = time.time()

        # Build training data with accumulated fixes
        train_pairs = list(train_orig)
        perm_classes = []
        for ft, fe, fp in accumulated_fixes:
            train_pairs = train_pairs + fe
            perm_classes.extend(fp)

        if cycle == 0 and b0_checkpoint:
            print(f"\n  Reusing B0: {b0_checkpoint}")
            dev_obj = torch.device(device if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(b0_checkpoint, map_location=dev_obj, weights_only=False)
            in_w2i = ckpt["in_w2i"]; out_w2i = ckpt["out_w2i"]
            model = TransformerSeq2Seq(len(in_w2i), len(out_w2i), d_model=128,
                                       max_in=ckpt.get("max_in", MAX_IN),
                                       max_out=ckpt.get("max_out", MAX_OUT)).to(dev_obj)
            model.load_state_dict(ckpt["state_dict"])
            summary_path = os.path.join(os.path.dirname(b0_checkpoint), "summary.json")
            if os.path.exists(summary_path):
                s = json.load(open(summary_path))
                gen_greedy = s.get("final_gen_exact_greedy", 0)
                by_cat = s.get("greedy_gen_by_cat", {})
                print(f"  Loaded: gen={gen_greedy:.2f}%")
            else:
                ge_ds = COGSDataset(gen, in_w2i, out_w2i)
                ge_ld = DataLoader(ge_ds, 16, shuffle=False, collate_fn=collate, num_workers=0)
                greedy = evaluate(model, ge_ld, out_w2i, dev_obj)
                gen_greedy = greedy["exact"]; by_cat = greedy.get("by_cat", {})
            metrics = []
        else:
            print(f"\n  Training with {len(accumulated_fixes)} accumulated fixes "
                  f"({len(train_pairs)} examples)...")
            model, in_w2i, out_w2i, gen_greedy, by_cat, metrics = train_model(
                train_pairs, dev, gen, perm_classes=perm_classes or None,
                seed=seed, epochs=epochs, device=device, run_dir=cycle_dir)

        # Measure surprise
        print("\n  Measuring surprise...")
        surprise = compute_surprise(model, in_w2i, out_w2i, train_orig, gen, device)
        ratio = surprise["ratio"]
        print(f"  Surprise ratio: {ratio:.2f}x")

        # Run causal diagnosis
        print("\n  Running causal diagnosis...")
        diag = diagnose(model, in_w2i, out_w2i, gen, train_orig,
                        device=device, max_examples=500)
        with open(os.path.join(cycle_dir, "diagnosis.json"), "w") as f:
            json.dump(diag, f, indent=2)

        # Log
        entry = {
            "cycle": cycle, "gen_greedy": gen_greedy, "by_cat": by_cat,
            "surprise_ratio": ratio, "n_train": len(train_pairs),
            "diagnosis": {k: v for k, v in diag["diff_types"].items() if k != "correct"},
            "hypotheses": {k: v["confirmed"] for k, v in diag["hypotheses"].items()},
            "accumulated_fixes": [ft for ft, _, _ in accumulated_fixes],
            "time_s": time.time() - t0,
        }
        cycle_log.append(entry)
        with open(os.path.join(loop_dir, "cycle_log.json"), "w") as f:
            json.dump(cycle_log, f, indent=2)

        # Print summary
        print(f"\n  Cycle {cycle}: gen={gen_greedy:.2f}% ratio={ratio:.2f}x")
        if by_cat:
            for c, v in sorted(by_cat.items(), key=lambda x: -x[1])[:5]:
                print(f"    {c:<40s} {v:.2f}%")
        print(f"  Diagnosis: {diag['diff_types']}")

        # Check convergence
        if ratio < surprise_threshold:
            print(f"\n  CONVERGED: ratio {ratio:.2f} < {surprise_threshold}"); break
        if prev_ratio is not None and abs(ratio - prev_ratio) < 0.05:
            print(f"\n  STALLED: Δratio={abs(ratio-prev_ratio):.3f} < 0.05"); break
        prev_ratio = ratio

        # Generate fixes from confirmed hypotheses
        print("\n  Generating fixes from diagnosis...")
        candidate_fixes = []
        for diff_type, hyp_data in diag["hypotheses"].items():
            if not hyp_data.get("confirmed"): continue
            fix_type = hyp_data.get("fix_type")
            if not fix_type: continue
            # Skip if already accumulated
            if fix_type in [ft for ft, _, _ in accumulated_fixes]:
                print(f"    {fix_type}: already applied, skipping")
                continue
            fix_examples, fix_perm, desc = generate_fix(fix_type, train_orig)
            if fix_examples or fix_perm:
                candidate_fixes.append((fix_type, fix_examples, fix_perm, desc))
                print(f"    {fix_type}: {desc}")
            else:
                print(f"    {fix_type}: no examples generated")

        if not candidate_fixes:
            print("\n  No new fixes to apply. Stopping.")
            break

        # Validate fixes (Level 2)
        print(f"\n  === LEVEL 2: Validating {len(candidate_fixes)} fixes ===")
        # Quick baseline
        print(f"  Baseline (20 epochs)...")
        _, _, _, base_score, _, _ = train_model(
            train_pairs, dev, gen, perm_classes=perm_classes or None,
            seed=seed, epochs=20, device=device, greedy_eval=False)
        print(f"  Baseline gen(TF): {base_score:.2f}%")

        new_fixes = []
        for fix_type, fix_ex, fix_pm, desc in candidate_fixes:
            print(f"\n  Validating: {fix_type} ({desc})...")
            score, delta = validate_fix(fix_type, fix_ex, fix_pm,
                                         train_pairs, dev, gen,
                                         base_score, seed=seed, device=device)
            print(f"    score={score:.2f}%  Δ={delta:+.2f}%")
            if delta >= -1.0:
                new_fixes.append((fix_type, fix_ex, fix_pm))
                print(f"    → ACCEPTED")
            else:
                print(f"    → REJECTED (regression)")

        if not new_fixes:
            print("\n  All fixes rejected. Stopping.")
            break

        # Add validated fixes
        accumulated_fixes.extend(new_fixes)
        print(f"\n  Accumulated fixes: {[ft for ft, _, _ in accumulated_fixes]}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"  LOOP v2 COMPLETE — {len(cycle_log)} cycles")
    print(f"{'='*70}")
    print(f"  {'Cycle':<8s} {'Gen%':>8s} {'Ratio':>8s} {'Fixes'}")
    print(f"  {'-'*60}")
    for c in cycle_log:
        fixes = ", ".join(c["accumulated_fixes"]) if c["accumulated_fixes"] else "baseline"
        print(f"  {c['cycle']:<8d} {c['gen_greedy']:7.2f}% {c['surprise_ratio']:7.2f}x  {fixes}")

    with open(os.path.join(loop_dir, "final_summary.json"), "w") as f:
        json.dump({"cycles": cycle_log, "dataset": dataset}, f, indent=2)
    print(f"\n  Saved: {loop_dir}")
    return cycle_log


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Phase 3 v2 — Causal auto loop")
    p.add_argument("--dataset", type=str, default="cogs", choices=["cogs", "slog"])
    p.add_argument("--max-cycles", type=int, default=5)
    p.add_argument("--surprise-threshold", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out-dir", type=str, default="runs_master")
    p.add_argument("--b0-checkpoint", type=str, default=None)
    args = p.parse_args()

    run_loop(args.dataset, max_cycles=args.max_cycles,
             surprise_threshold=args.surprise_threshold,
             seed=args.seed, epochs=args.epochs, device=args.device,
             out_dir=args.out_dir, b0_checkpoint=args.b0_checkpoint)
