#!/usr/bin/env python3
"""
Phase 3 complète — Boucle fermée itérative avec validation des générateurs.

Niveau 1 : Entraîne → mesure surprise → clustere → augmente → réentraîne → re-mesure
Niveau 2 : Valide chaque générateur sur 10 epochs avant de l'engager pour 60

Usage:
  python3 auto_loop.py --dataset cogs --max-cycles 5 --out-dir runs_master
  python3 auto_loop.py --dataset slog --max-cycles 3 --out-dir runs_master
  python3 auto_loop.py --dataset cogs --recap  # juste afficher les résultats des cycles
"""
import os, sys, json, argparse, random, time, copy, glob
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional

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
    TransformerSeq2Seq, evaluate_tf, evaluate,
    generate_subj_pp_augmentations, generate_pp_recursion_extension,
    generate_active_to_passive, detect_proper_names,
    MAX_IN, MAX_OUT, MAX_DECODE, PAD, BOS, EOS,
)
from error_mirror import JEPAPredictor, train_jepa_predictor, measure_surprise, measure_decoder_entropy
from cluster_gaps import extract_features, FEATURE_NAMES
from auto_augment import identify_gaps


# ══════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════
def load_data(dataset):
    if dataset == "cogs":
        data_dir = os.path.join(HERE, "data", "cogs")
        train = parse_cogs_tsv(os.path.join(data_dir, "train.tsv"))
        dev = parse_cogs_tsv(os.path.join(data_dir, "dev.tsv"))
        gen = parse_cogs_tsv(os.path.join(data_dir, "gen.tsv"))
    else:
        data_dir = os.path.join(HERE, "data", "slog")
        train = parse_cogs_tsv(os.path.join(data_dir, "train.tsv"))
        dev = parse_cogs_tsv(os.path.join(data_dir, "dev.tsv"))
        gen_path = os.path.join(data_dir, "generalization_sets", "gen_cogsLF.tsv")
        gen_link = os.path.join(data_dir, "gen.tsv")
        if not os.path.exists(gen_link):
            import shutil; shutil.copy2(gen_path, gen_link)
        gen = parse_cogs_tsv(gen_link)
    return train, dev, gen


# ══════════════════════════════════════════════════════════════
# Training (shared by full runs and quick validation)
# ══════════════════════════════════════════════════════════════
def train_model(train_pairs, dev_pairs, gen_pairs, perm_classes=None,
                seed=42, epochs=60, device="cuda", run_dir=None,
                greedy_eval=True):
    """Train a model, return (model, in_w2i, out_w2i, metrics, by_cat)."""
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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
            scaler.step(opt)
            scaler.update()
            sched.step()
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
            print(f"  Early stopping at epoch {ep}.")
            break

    # Greedy eval
    by_cat = {}
    if greedy_eval:
        print("  Greedy eval...")
        greedy = evaluate(model, ge_ld, out_w2i, dev_obj)
        by_cat = greedy.get("by_cat", {})
        gen_greedy = greedy["exact"]
    else:
        gen_greedy = gen_m["exact"]

    # Save
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
# Surprise measurement (wraps error_mirror)
# ══════════════════════════════════════════════════════════════
def compute_surprise(model, in_w2i, out_w2i, train_pairs, gen_pairs, device="cuda"):
    """Train JEPA predictor, measure surprise on train and gen, return stats."""
    dev_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    tr_ds = COGSDataset(train_pairs, in_w2i, out_w2i)
    ge_ds = COGSDataset(gen_pairs, in_w2i, out_w2i)
    tr_ld = DataLoader(tr_ds, 64, shuffle=True, collate_fn=collate, num_workers=0)
    ge_ld = DataLoader(ge_ds, 64, shuffle=False, collate_fn=collate, num_workers=0)

    predictor = JEPAPredictor(128).to(dev_obj)
    train_jepa_predictor(model, predictor, tr_ld, dev_obj, epochs=10, lr=1e-3)

    train_results = measure_surprise(model, predictor, tr_ld, dev_obj)
    gen_results = measure_surprise(model, predictor, ge_ld, dev_obj)

    train_mean = float(np.mean([s for s, _ in train_results]))
    gen_mean = float(np.mean([s for s, _ in gen_results]))
    ratio = gen_mean / max(train_mean, 1e-10)

    by_cat = defaultdict(list)
    for s, cat in gen_results:
        by_cat[cat].append(s)
    cat_stats = {cat: {"surprise_mean": float(np.mean(vals)), "n": len(vals)}
                 for cat, vals in by_cat.items()}

    return {"train_mean": train_mean, "gen_mean": gen_mean, "ratio": ratio,
            "by_category": cat_stats}


# ══════════════════════════════════════════════════════════════
# Clustering (wraps cluster_gaps)
# ══════════════════════════════════════════════════════════════
def compute_clusters(surprise_data, gen_pairs, k=4):
    """Run KMeans clustering on gen examples, stratified by surprise."""
    from sklearn.cluster import KMeans

    cat_surprise = {cat: info["surprise_mean"] for cat, info in surprise_data["by_category"].items()}
    sorted_cats = sorted(cat_surprise, key=lambda c: -cat_surprise[c])

    cat_examples = defaultdict(list)
    for inp, lf, c in gen_pairs:
        cat_examples[c].append((inp, lf, c))

    per_cat = max(20, 500 // len(sorted_cats))
    top_examples = []
    for cat in sorted_cats:
        top_examples.extend(cat_examples[cat][:per_cat])

    X = []
    cats = []
    for inp, lf, cat in top_examples:
        f = extract_features(inp, lf)
        X.append([f[name] for name in FEATURE_NAMES])
        cats.append(cat)
    X = np.array(X, dtype=float)
    mu = X.mean(0); std = X.std(0) + 1e-8
    X_norm = (X - mu) / std

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_norm)

    clusters = {}
    from collections import Counter
    for ci in range(k):
        mask = labels == ci
        if mask.sum() == 0: continue
        cluster_cats = [cats[i] for i in range(len(cats)) if mask[i]]
        cat_counts = Counter(cluster_cats)
        clusters[f"cluster_{ci}"] = {
            "n": int(mask.sum()),
            "categories": dict(cat_counts.most_common()),
            "feature_means": {name: float(X[mask].mean(0)[j])
                              for j, name in enumerate(FEATURE_NAMES)},
            "purity": float(cat_counts.most_common(1)[0][1] / mask.sum() * 100),
        }
    return clusters


# ══════════════════════════════════════════════════════════════
# Level 2 — Generator validation (quick 10-epoch test)
# ══════════════════════════════════════════════════════════════
def apply_single_generator(train_pairs, gen_type, in_w2i=None, out_w2i=None):
    """Apply a single generator to the training data, return augmented pairs + perm_classes."""
    perm_classes = []
    aug_pairs = list(train_pairs)

    if gen_type == "pp_on_subject":
        aug = generate_subj_pp_augmentations(train_pairs)
        aug_pairs = train_pairs + aug

    elif gen_type == "pp_recursion":
        aug = generate_pp_recursion_extension(train_pairs, max_aug=500)
        aug_pairs = train_pairs + aug

    elif gen_type == "passive":
        aug = generate_active_to_passive(train_pairs)
        aug_pairs = train_pairs + aug

    elif gen_type == "proper_names":
        if in_w2i and out_w2i:
            pn = detect_proper_names(train_pairs, in_w2i, out_w2i)
            if pn:
                perm_classes = [[(p[0], p[1]) for p in pn]]

    return aug_pairs, perm_classes


def validate_generators(generators, train_pairs, dev_pairs, gen_pairs,
                        device="cuda", seed=42):
    """Level 2 + 2b: Validate generators individually, then test pairs.

    Pass 1: test each generator solo (20 epochs). Keep if Δgen >= -1%.
    Pass 2: test pairs of validated generators. Keep best combination.
    If a pair regresses but both components are good solo, keep the one
    with the best individual score."""
    VAL_EPOCHS = 20

    print(f"\n  === LEVEL 2: Generator Validation ({VAL_EPOCHS} epochs each) ===")

    # Baseline: quick eval without any augmentation
    print(f"  Training baseline ({VAL_EPOCHS} epochs)...")
    _, in_w2i, out_w2i, base_gen_ex, _, base_metrics = train_model(
        train_pairs, dev_pairs, gen_pairs, seed=seed, epochs=VAL_EPOCHS,
        device=device, greedy_eval=False)
    print(f"  Baseline gen(TF): {base_gen_ex:.2f}%")

    # ── Pass 1: individual validation ──
    print(f"\n  --- Pass 1: Individual generators ---")
    validated = []
    rejected = []

    for g in generators:
        if "SKIP" in g["type"]:
            rejected.append({**g, "reject_reason": "pre-skipped", "delta": 0})
            continue

        print(f"\n  Testing: {g['type']}...")
        aug_train, perm_cls = apply_single_generator(
            train_pairs, g["type"], in_w2i, out_w2i)
        _, _, _, test_gen, _, _ = train_model(
            aug_train, dev_pairs, gen_pairs, perm_classes=perm_cls,
            seed=seed, epochs=VAL_EPOCHS, device=device, greedy_eval=False)

        delta = test_gen - base_gen_ex
        print(f"    gen(TF): {test_gen:.2f}%  Δ={delta:+.2f}%")

        if delta >= -1.0:
            validated.append({**g, "delta": delta, "gen_score": test_gen})
            print(f"    → ACCEPTED")
        else:
            rejected.append({**g, "reject_reason": f"regression Δ={delta:.2f}%", "delta": delta})
            print(f"    → REJECTED")

    # ── Pass 2 (Level 2b): pair validation ──
    if len(validated) >= 2:
        print(f"\n  --- Pass 2 (Level 2b): Pair validation ---")
        # Sort by individual score, test top pairs
        validated.sort(key=lambda g: -g["gen_score"])
        pairs_to_test = []
        for i in range(len(validated)):
            for j in range(i + 1, len(validated)):
                pairs_to_test.append((validated[i], validated[j]))
        # Cap at 6 pairs
        pairs_to_test = pairs_to_test[:6]

        best_combo = None
        best_combo_score = base_gen_ex

        for ga, gb in pairs_to_test:
            pair_name = f"{ga['type']}+{gb['type']}"
            print(f"\n  Testing pair: {pair_name}...")

            # Apply both generators
            aug_train = list(train_pairs)
            perm_cls_all = []
            for g in [ga, gb]:
                aug_train, pc = apply_single_generator(aug_train, g["type"],
                                                        in_w2i, out_w2i)
                perm_cls_all.extend(pc)

            _, _, _, pair_gen, _, _ = train_model(
                aug_train, dev_pairs, gen_pairs,
                perm_classes=perm_cls_all or None,
                seed=seed, epochs=VAL_EPOCHS, device=device, greedy_eval=False)

            pair_delta = pair_gen - base_gen_ex
            # Compare to best individual
            best_solo = max(ga["gen_score"], gb["gen_score"])
            pair_vs_solo = pair_gen - best_solo
            print(f"    pair gen(TF): {pair_gen:.2f}%  Δ_base={pair_delta:+.2f}%  "
                  f"Δ_solo={pair_vs_solo:+.2f}%")

            if pair_delta >= -1.0 and pair_gen > best_combo_score:
                best_combo = [ga, gb]
                best_combo_score = pair_gen
                print(f"    → BEST COMBO so far")
            elif pair_delta < -1.0:
                print(f"    → PAIR REJECTED (regression)")
            else:
                print(f"    → not better than current best")

        # Decide: use best combo or best individual?
        best_solo_gen = validated[0]  # highest individual score
        if best_combo and best_combo_score > best_solo_gen["gen_score"]:
            print(f"\n  → Using pair: {[g['type'] for g in best_combo]} "
                  f"(score={best_combo_score:.2f}% > solo={best_solo_gen['gen_score']:.2f}%)")
            return best_combo, rejected
        else:
            print(f"\n  → Using best solo: {best_solo_gen['type']} "
                  f"(score={best_solo_gen['gen_score']:.2f}%)")
            return [best_solo_gen], rejected

    return validated, rejected


# ══════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════
def run_loop(dataset, max_cycles=5, surprise_threshold=1.5, seed=42,
             epochs=60, device="cuda", out_dir="runs_master", validate=True,
             b0_checkpoint=None):
    """Full autonomous loop: train → surprise → cluster → augment → retrain.

    If b0_checkpoint is provided, skip cycle 0 training and use existing B0."""
    os.makedirs(out_dir, exist_ok=True)
    loop_dir = os.path.join(out_dir, f"loop_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(loop_dir, exist_ok=True)

    train_orig, dev, gen = load_data(dataset)
    cycle_log = []
    prev_ratio = None

    for cycle in range(max_cycles):
        print(f"\n{'='*70}")
        print(f"  CYCLE {cycle} — {dataset.upper()}")
        print(f"{'='*70}")
        cycle_dir = os.path.join(loop_dir, f"cycle_{cycle}")
        t0 = time.time()

        if cycle == 0 and b0_checkpoint:
            # Reuse existing B0 checkpoint + summary
            print(f"\n  Reusing B0 checkpoint: {b0_checkpoint}")
            os.makedirs(cycle_dir, exist_ok=True)
            dev_obj = torch.device(device if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(b0_checkpoint, map_location=dev_obj, weights_only=False)
            in_w2i = ckpt["in_w2i"]; out_w2i = ckpt["out_w2i"]
            max_in = ckpt.get("max_in", MAX_IN)
            max_out = ckpt.get("max_out", MAX_OUT)
            model = TransformerSeq2Seq(len(in_w2i), len(out_w2i), d_model=128,
                                       max_in=max_in, max_out=max_out).to(dev_obj)
            model.load_state_dict(ckpt["state_dict"])

            # Try to load greedy results from summary.json (skip re-eval)
            summary_path = os.path.join(os.path.dirname(b0_checkpoint), "summary.json")
            if os.path.exists(summary_path):
                b0_summary = json.load(open(summary_path))
                gen_greedy = b0_summary.get("final_gen_exact_greedy", 0)
                by_cat = b0_summary.get("greedy_gen_by_cat", {})
                print(f"  Loaded from summary: gen={gen_greedy:.2f}% ({len(by_cat)} categories)")
            else:
                ge_ds = COGSDataset(gen, in_w2i, out_w2i)
                ge_ld = DataLoader(ge_ds, 16, shuffle=False, collate_fn=collate, num_workers=0)
                print("  Greedy eval on existing B0...")
                greedy = evaluate(model, ge_ld, out_w2i, dev_obj)
                gen_greedy = greedy["exact"]
                by_cat = greedy.get("by_cat", {})

            metrics = []
            train_pairs = list(train_orig)
            perm_classes = []
        elif cycle == 0:
            # Cycle 0: train baseline B0
            print("\n  Training baseline B0...")
            train_pairs = list(train_orig)
            perm_classes = []
        else:
            # Apply accumulated augmentations
            print(f"\n  Applying {len(active_generators)} generators...")
            train_pairs = list(train_orig)
            perm_classes = []
            in_w2i_tmp, out_w2i_tmp = build_vocabs(train_pairs + dev + gen)
            for g in active_generators:
                aug, pc = apply_single_generator(train_pairs, g["type"],
                                                 in_w2i_tmp, out_w2i_tmp)
                train_pairs = aug
                perm_classes.extend(pc)
                print(f"    {g['type']}: {len(train_pairs)} train examples")

        if not (cycle == 0 and b0_checkpoint):
            model, in_w2i, out_w2i, gen_greedy, by_cat, metrics = train_model(
                train_pairs, dev, gen, perm_classes=perm_classes or None,
                seed=seed, epochs=epochs, device=device, run_dir=cycle_dir)

        # Measure surprise
        print("\n  Measuring surprise...")
        surprise = compute_surprise(model, in_w2i, out_w2i, train_orig, gen, device)
        ratio = surprise["ratio"]
        print(f"  Surprise ratio: {ratio:.2f}x (train={surprise['train_mean']:.4f} gen={surprise['gen_mean']:.4f})")

        # Log cycle
        entry = {
            "cycle": cycle,
            "gen_greedy": gen_greedy,
            "by_cat": by_cat,
            "surprise_ratio": ratio,
            "surprise_train": surprise["train_mean"],
            "surprise_gen": surprise["gen_mean"],
            "n_train": len(train_pairs),
            "generators": [g["type"] for g in active_generators] if cycle > 0 else ["baseline"],
            "time_s": time.time() - t0,
        }
        cycle_log.append(entry)
        with open(os.path.join(loop_dir, "cycle_log.json"), "w") as f:
            json.dump(cycle_log, f, indent=2)
        with open(os.path.join(cycle_dir, "surprise.json"), "w") as f:
            json.dump(surprise, f, indent=2)

        print(f"\n  Cycle {cycle} result: gen={gen_greedy:.2f}% ratio={ratio:.2f}x")
        if by_cat:
            nonzero = {c: v for c, v in sorted(by_cat.items(), key=lambda x: -x[1]) if v > 0}
            for c, v in list(nonzero.items())[:5]:
                print(f"    {c:<40s} {v:.2f}%")

        # Check convergence
        if ratio < surprise_threshold:
            print(f"\n  CONVERGED: ratio {ratio:.2f} < {surprise_threshold}")
            break
        if prev_ratio is not None and abs(ratio - prev_ratio) < 0.05:
            print(f"\n  STALLED: ratio change {abs(ratio-prev_ratio):.3f} < 0.05")
            break
        prev_ratio = ratio

        # Cluster and identify generators for next cycle
        print("\n  Clustering gaps...")
        clusters = compute_clusters(surprise, gen, k=4)
        with open(os.path.join(cycle_dir, "clusters.json"), "w") as f:
            json.dump(clusters, f, indent=2)

        generators = identify_gaps(clusters)
        print(f"  Identified {len(generators)} potential generators:")
        for g in generators:
            skip = " [SKIP]" if "SKIP" in g["type"] else ""
            print(f"    {g['type']}{skip}: {g['reason']}")

        # Level 2: Validate generators
        if validate and cycle > 0:
            active_generators, rejected = validate_generators(
                generators, train_orig, dev, gen, device=device, seed=seed)
            print(f"\n  Validated: {[g['type'] for g in active_generators]}")
            print(f"  Rejected: {[g['type'] for g in rejected]}")
        else:
            active_generators = [g for g in generators if "SKIP" not in g["type"]]

    # Final summary
    print(f"\n{'='*70}")
    print(f"  LOOP COMPLETE — {len(cycle_log)} cycles")
    print(f"{'='*70}")
    print(f"  {'Cycle':<8s} {'Gen%':>8s} {'Ratio':>8s} {'Generators'}")
    print(f"  {'-'*60}")
    for c in cycle_log:
        gens = ", ".join(c["generators"])
        print(f"  {c['cycle']:<8d} {c['gen_greedy']:7.2f}% {c['surprise_ratio']:7.2f}x  {gens}")

    with open(os.path.join(loop_dir, "final_summary.json"), "w") as f:
        json.dump({"cycles": cycle_log, "dataset": dataset}, f, indent=2)
    print(f"\n  Saved: {loop_dir}")
    return cycle_log


def recap(out_dir):
    """Display results from previous loop runs."""
    for d in sorted(glob.glob(os.path.join(out_dir, "loop_*"))):
        log_path = os.path.join(d, "cycle_log.json")
        if not os.path.exists(log_path): continue
        log = json.load(open(log_path))
        print(f"\n{'='*70}")
        print(f"  {os.path.basename(d)}")
        print(f"{'='*70}")
        for c in log:
            gens = ", ".join(c["generators"])
            print(f"  Cycle {c['cycle']}: gen={c['gen_greedy']:.2f}%  "
                  f"ratio={c['surprise_ratio']:.2f}x  [{gens}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Phase 3 — Auto loop")
    p.add_argument("--dataset", type=str, default="cogs", choices=["cogs", "slog"])
    p.add_argument("--max-cycles", type=int, default=5)
    p.add_argument("--surprise-threshold", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out-dir", type=str, default="runs_master")
    p.add_argument("--no-validate", action="store_true",
                   help="Skip Level 2 generator validation")
    p.add_argument("--b0-checkpoint", type=str, default=None,
                   help="Reuse existing B0 checkpoint for cycle 0 (skip training)")
    p.add_argument("--recap", action="store_true")
    args = p.parse_args()

    if args.recap:
        recap(args.out_dir)
    else:
        run_loop(args.dataset, max_cycles=args.max_cycles,
                 surprise_threshold=args.surprise_threshold,
                 seed=args.seed, epochs=args.epochs, device=args.device,
                 out_dir=args.out_dir, validate=not args.no_validate,
                 b0_checkpoint=args.b0_checkpoint)
