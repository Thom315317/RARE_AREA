#!/usr/bin/env python3
"""
Phase 3b — Automatic augmentation from cluster analysis.

Reads cluster results, identifies the gap type, selects the right generator,
augments the training set, retrains, and re-measures surprise.

The system discovers what to augment WITHOUT human intervention:
  1. Load clusters → find the cluster with nmod_on_subj > 0.5
  2. Select generator: nmod_on_subj → generate_subj_pp_augmentations
  3. Augment train set
  4. Retrain
  5. Re-measure surprise on gen

Usage:
  python3 auto_augment.py --clusters runs_master/clusters_slog.json \
      --checkpoint runs_master/B0_s42_.../checkpoint.pt \
      --dataset slog --out-dir runs_master
"""
import os, sys, json, argparse, random, time
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))


def identify_gaps(clusters):
    """From cluster analysis, identify which generators to activate."""
    generators = []
    for cname, cdata in clusters.items():
        fm = cdata.get("feature_means", {})
        n = cdata.get("n", 0)
        purity = cdata.get("purity", 0)

        # Rule 1: nmod_on_subj high → PP on subject augmentation
        if fm.get("nmod_on_subj", 0) > 0.5:
            generators.append({
                "type": "pp_on_subject",
                "cluster": cname,
                "reason": f"nmod_on_subj={fm['nmod_on_subj']:.2f}, n={n}",
            })

        # Rule 2: n_ccomp very high → CP recursion (skip — known to hurt Q categories)
        if fm.get("n_ccomp", 0) > 5:
            generators.append({
                "type": "cp_recursion_SKIP",
                "cluster": cname,
                "reason": f"n_ccomp={fm['n_ccomp']:.1f} — skipped (hurts Q categories)",
            })

        # Rule 3: n_nmod very high → PP recursion
        # DISABLED: causes Q_iobj_ditransV regression (79%→9%)
        if fm.get("n_nmod", 0) > 5:
            generators.append({
                "type": "pp_recursion_SKIP",
                "cluster": cname,
                "reason": f"n_nmod={fm['n_nmod']:.1f} — skipped (hurts Q_iobj_ditransV)",
            })

        # Rule 4: has_passive high → passive augmentation
        if fm.get("has_passive", 0) > 0.5:
            generators.append({
                "type": "passive",
                "cluster": cname,
                "reason": f"has_passive={fm['has_passive']:.2f}, n={n}",
            })

        # Rule 5: lexical gap — large cluster, low structural features,
        # moderate surprise. This is the "fourre-tout" with many categories
        # at ~5% each. Proper name permutation helps here.
        cats = cdata.get("categories", {})
        has_proper_cats = any("proper" in c.lower() for c in cats)
        if (n > 100 and has_proper_cats and
            fm.get("n_nmod", 0) < 2 and fm.get("n_ccomp", 0) < 2 and
            fm.get("nmod_on_subj", 0) < 0.3 and fm.get("has_passive", 0) < 0.3):
            generators.append({
                "type": "proper_names",
                "cluster": cname,
                "reason": f"lexical cluster with proper-name categories, n={n}",
            })

    return generators


def run_augmented_training(dataset, generators, checkpoint_path, out_dir,
                           seed=42, epochs=60, device="cuda"):
    """Retrain with automatically selected augmentations."""
    from cogs_compositional import (
        parse_cogs_tsv, build_vocabs, COGSDataset, collate,
        TransformerSeq2Seq, evaluate_tf, evaluate,
        generate_subj_pp_augmentations, generate_pp_recursion_extension,
        generate_active_to_passive, detect_proper_names,
        MAX_IN, MAX_OUT, MAX_DECODE, PAD, BOS, EOS,
    )
    from torch.utils.data import DataLoader
    import torch.nn as nn
    import torch.nn.functional as F
    import math

    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load data
    if dataset == "cogs":
        data_dir = os.path.join(HERE, "data", "cogs")
        train_pairs = parse_cogs_tsv(os.path.join(data_dir, "train.tsv"))
        dev_pairs = parse_cogs_tsv(os.path.join(data_dir, "dev.tsv"))
        gen_pairs = parse_cogs_tsv(os.path.join(data_dir, "gen.tsv"))
    else:
        data_dir = os.path.join(HERE, "data", "slog")
        train_pairs = parse_cogs_tsv(os.path.join(data_dir, "train.tsv"))
        dev_pairs = parse_cogs_tsv(os.path.join(data_dir, "dev.tsv"))
        gen_path = os.path.join(data_dir, "generalization_sets", "gen_cogsLF.tsv")
        # Ensure gen.tsv symlink exists
        gen_link = os.path.join(data_dir, "gen.tsv")
        if not os.path.exists(gen_link):
            import shutil; shutil.copy2(gen_path, gen_link)
        gen_pairs = parse_cogs_tsv(gen_link)

    print(f"Train: {len(train_pairs)}  Dev: {len(dev_pairs)}  Gen: {len(gen_pairs)}")

    # Apply augmentations based on identified gaps
    active_types = [g["type"] for g in generators if "SKIP" not in g["type"]]
    print(f"\nActive generators: {active_types}")

    perm_classes = []
    for g in generators:
        gtype = g["type"]
        print(f"  [{gtype}] {g['reason']}")

        if gtype == "pp_on_subject":
            aug = generate_subj_pp_augmentations(train_pairs)
            print(f"    → {len(aug)} PP-on-subject examples")
            train_pairs = train_pairs + aug

        elif gtype == "pp_recursion":
            aug = generate_pp_recursion_extension(train_pairs, max_aug=500)
            print(f"    → {len(aug)} PP recursion +1 examples")
            train_pairs = train_pairs + aug

        elif gtype == "passive":
            aug = generate_active_to_passive(train_pairs)
            print(f"    → {len(aug)} active→passive examples")
            train_pairs = train_pairs + aug

        elif "SKIP" in gtype:
            print(f"    → skipped")

    # Proper name permutation ONLY if a cluster requests it
    # (no cluster → no permutation; avoids Q_iobj regression on SLOG)
    if "proper_names" in active_types:
        in_w2i_tmp, out_w2i_tmp = build_vocabs(train_pairs + dev_pairs + gen_pairs)
        pn = detect_proper_names(train_pairs, in_w2i_tmp, out_w2i_tmp)
        if pn:
            perm_classes = [[(p[0], p[1]) for p in pn]]
            print(f"  [proper_names] {len(pn)} names for permutation")
    else:
        print(f"  [proper_names] not activated (no lexical cluster detected)")

    print(f"\nAugmented train: {len(train_pairs)}")

    # Build vocabs and datasets
    in_w2i, out_w2i = build_vocabs(train_pairs + dev_pairs + gen_pairs)
    tr_ds = COGSDataset(train_pairs, in_w2i, out_w2i,
                        perm_classes=perm_classes if perm_classes else None)
    dv_ds = COGSDataset(dev_pairs, in_w2i, out_w2i)
    ge_ds = COGSDataset(gen_pairs, in_w2i, out_w2i)

    tr_ld = DataLoader(tr_ds, 64, shuffle=True, collate_fn=collate, num_workers=0, pin_memory=True)
    dv_ld = DataLoader(dv_ds, 16, shuffle=False, collate_fn=collate, num_workers=0, pin_memory=True)
    ge_ld = DataLoader(ge_ds, 16, shuffle=False, collate_fn=collate, num_workers=0, pin_memory=True)

    # Build model
    max_in = max(MAX_IN, max(len(p[0].split()) for p in train_pairs + dev_pairs + gen_pairs) + 2)
    max_out = max(MAX_OUT, max(len(p[1].split()) for p in train_pairs + dev_pairs + gen_pairs) + 4)
    model = TransformerSeq2Seq(len(in_w2i), len(out_w2i), d_model=128,
                               max_in=max_in, max_out=max_out).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M params")

    # Train
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    total_steps = epochs * len(tr_ld)
    warmup = min(1000, total_steps // 10)
    def lr_at(step):
        if step < warmup: return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"auto_{dataset}_s{seed}_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    best_dev = -1.0
    patience = 0
    metrics_log = []

    for ep in range(epochs):
        t0 = time.time()
        model.train()
        tr_loss = 0.0; nb = 0
        for src, src_mask, tgt, _, _ in tr_ld:
            src = src.to(device); src_mask = src_mask.to(device); tgt = tgt.to(device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits, _ = model(src, src_mask, tgt_in)
                V = logits.size(-1)
                loss = F.cross_entropy(logits.reshape(-1, V), tgt_out.reshape(-1),
                                       ignore_index=0, label_smoothing=0.1)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tr_loss += loss.item(); nb += 1

        dev_m = evaluate_tf(model, dv_ld, device)
        gen_m = evaluate_tf(model, ge_ld, device)

        is_best = dev_m["exact"] > best_dev
        if is_best:
            best_dev = dev_m["exact"]
            patience = 0
        else:
            patience += 1

        entry = {"epoch": ep, "train_loss": tr_loss/max(nb,1),
                 "dev": dev_m, "gen": gen_m, "lr": sched.get_last_lr()[0]}
        metrics_log.append(entry)

        dt = time.time() - t0
        mark = "★" if is_best else " "
        print(f"E{ep:03d} {mark} loss={tr_loss/max(nb,1):.4f} "
              f"dev={dev_m['exact']:5.2f}% gen={gen_m['exact']:5.2f}% "
              f"pat={patience} | {dt:.1f}s")

        if patience >= 20:
            print(f"Early stopping at epoch {ep}.")
            break

    # Final greedy eval
    print("\nFinal greedy eval...")
    greedy_gen = evaluate(model, ge_ld, out_w2i, device)
    print(f"  Greedy gen_ex={greedy_gen['exact']:.2f}%")
    if "by_cat" in greedy_gen:
        print("  Per-category:")
        for cat, pct in sorted(greedy_gen["by_cat"].items(), key=lambda x: -x[1]):
            if pct > 0:
                print(f"    {cat:<40s} {pct:6.2f}%")

    # Save
    summary = {
        "type": "auto_augment",
        "dataset": dataset,
        "seed": seed,
        "generators": generators,
        "active_types": active_types,
        "epochs_run": ep + 1,
        "final_gen_greedy": greedy_gen["exact"],
        "greedy_by_cat": greedy_gen.get("by_cat", {}),
        "n_params": n_params,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics_log, f, indent=1)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    torch.save({"state_dict": model.state_dict(), "in_w2i": in_w2i,
                "out_w2i": out_w2i, "d_model": 128,
                "max_in": max_in, "max_out": max_out},
               os.path.join(run_dir, "checkpoint.pt"))
    print(f"\nSaved: {run_dir}")
    return summary


def main():
    p = argparse.ArgumentParser(description="Phase 3b — automatic augmentation")
    p.add_argument("--clusters", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="B0 checkpoint (for reference, not loaded)")
    p.add_argument("--dataset", type=str, default="slog", choices=["cogs", "slog"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out-dir", type=str, default="runs_master")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Step 1: Load clusters and identify gaps
    print(f"Loading clusters: {args.clusters}")
    clusters = json.load(open(args.clusters))
    generators = identify_gaps(clusters)

    print(f"\nIdentified {len(generators)} potential augmentations:")
    for g in generators:
        skip = " [SKIP]" if "SKIP" in g["type"] else ""
        print(f"  {g['type']}{skip}: {g['reason']}")

    # Step 2-5: Generate, augment, retrain
    summary = run_augmented_training(
        args.dataset, generators, args.checkpoint, args.out_dir,
        seed=args.seed, epochs=args.epochs, device=args.device)

    # Compare to B0
    print(f"\n{'='*70}")
    print(f"  AUTO-AUGMENT RESULT vs B0")
    print(f"{'='*70}")
    print(f"  B0 gen (from checkpoint): see original run")
    print(f"  Auto gen: {summary['final_gen_greedy']:.2f}%")
    print(f"  Generators used: {summary['active_types']}")


if __name__ == "__main__":
    main()
