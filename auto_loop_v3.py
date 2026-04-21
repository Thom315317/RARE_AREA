#!/usr/bin/env python3
"""
Phase 3 v3 — Fusion détection lexicale (v1) + structurelle (v2).

À chaque cycle :
  1. Mesure surprise
  2. Détection lexicale : cluster des erreurs, détecte les clusters "fourre-tout"
     (uniformes) avec des catégories *_proper → active permutation
  3. Détection structurelle : diagnostic causal → fixes transformationnels
  4. Valide chaque fix sur 20 epochs
  5. Applique, réentraîne, boucle

Combine les deux approches pour couvrir gaps lexicaux ET structurels.

Usage:
  python3 auto_loop_v3.py --dataset cogs --max-cycles 3 --out-dir runs_master
  python3 auto_loop_v3.py --dataset cogs --max-cycles 3 --out-dir runs_master \
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
    TransformerSeq2Seq, TransformerSeq2SeqWithCopy, build_in_to_out_map,
    evaluate_tf, evaluate,
    generate_subj_pp_augmentations, generate_pp_recursion_extension,
    generate_active_to_passive, generate_noun_position_augmentations,
    detect_proper_names,
    MAX_IN, MAX_OUT, MAX_DECODE, PAD, BOS, EOS,
)
from error_mirror import JEPAPredictor, train_jepa_predictor, measure_surprise
from cluster_gaps import extract_features, FEATURE_NAMES
from causal_diagnosis import diagnose


# ══════════════════════════════════════════════════════════════
# Data + training (same as v1/v2)
# ══════════════════════════════════════════════════════════════
def load_data(dataset):
    if dataset == "cogs":
        d = os.path.join(HERE, "data", "cogs")
        return (parse_cogs_tsv(os.path.join(d, "train.tsv")),
                parse_cogs_tsv(os.path.join(d, "dev.tsv")),
                parse_cogs_tsv(os.path.join(d, "gen.tsv")))
    else:
        d = os.path.join(HERE, "data", "slog")
        gen_link = os.path.join(d, "gen.tsv")
        if not os.path.exists(gen_link):
            import shutil
            shutil.copy2(os.path.join(d, "generalization_sets", "gen_cogsLF.tsv"), gen_link)
        return (parse_cogs_tsv(os.path.join(d, "train.tsv")),
                parse_cogs_tsv(os.path.join(d, "dev.tsv")),
                parse_cogs_tsv(gen_link))


def train_model(train_pairs, dev_pairs, gen_pairs, perm_classes=None,
                seed=42, epochs=60, device="cuda", run_dir=None, greedy_eval=True,
                use_copy=False):
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
    if use_copy:
        in_to_out = build_in_to_out_map(in_w2i, out_w2i)
        model = TransformerSeq2SeqWithCopy(len(in_w2i), len(out_w2i), d_model=128,
                                           max_in=max_in, max_out=max_out,
                                           in_to_out_map=in_to_out).to(dev_obj)
        print(f"  [COPY] {(in_to_out >= 0).sum().item()}/{len(in_w2i)} tokens mapped")
    else:
        model = TransformerSeq2Seq(len(in_w2i), len(out_w2i), d_model=128,
                                   max_in=max_in, max_out=max_out).to(dev_obj)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev_obj.type == "cuda"))
    total_steps = epochs * len(tr_ld); warmup = min(1000, total_steps // 10)
    def lr_at(step):
        if step < warmup: return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    best_dev = -1.0; patience = 0; metrics = []
    for ep in range(epochs):
        model.train(); tr_loss = 0.0; nb = 0
        # Scheduled sampling for copy: linear 1.0 → 0.5 from ep=20 to end
        tfr = 1.0
        if use_copy and ep >= 20:
            progress = min(max((ep - 20) / max(epochs - 20, 1), 0.0), 1.0)
            tfr = 1.0 - 0.5 * progress
        for src, src_mask, tgt, _, _ in tr_ld:
            src = src.to(dev_obj); src_mask = src_mask.to(dev_obj); tgt = tgt.to(dev_obj)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            opt.zero_grad(set_to_none=True)
            if tfr < 1.0:
                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=(dev_obj.type == "cuda")):
                        logits_1, _ = model(src, src_mask, tgt_in)
                    preds = logits_1.argmax(-1)
                B, T = tgt_in.shape
                rm = torch.rand(B, T, device=dev_obj) > tfr
                rm[:, 0] = False
                ps = torch.cat([tgt_in[:, :1], preds[:, :-1]], dim=1)
                tgt_in_u = torch.where(rm, ps, tgt_in)
            else:
                tgt_in_u = tgt_in
            with torch.amp.autocast("cuda", enabled=(dev_obj.type == "cuda")):
                logits, info = model(src, src_mask, tgt_in_u)
                V = logits.size(-1)
                if info.get("uses_copy"):
                    loss = F.nll_loss(logits.reshape(-1, V), tgt_out.reshape(-1),
                                      ignore_index=0)
                else:
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
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return model, in_w2i, out_w2i, gen_greedy, by_cat, metrics


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
    by_cat = defaultdict(list)
    for s, cat in gen_r:
        by_cat[cat].append(s)
    cat_stats = {cat: {"surprise_mean": float(np.mean(vals)), "n": len(vals)}
                 for cat, vals in by_cat.items()}
    return {"train_mean": tm, "gen_mean": gm, "ratio": gm / max(tm, 1e-10),
            "by_category": cat_stats}


# ══════════════════════════════════════════════════════════════
# Lexical detection (v1 style)
# ══════════════════════════════════════════════════════════════
def detect_lexical_gaps(surprise_data, gen_pairs, train_pairs, in_w2i, out_w2i):
    """Cluster high-surprise examples, find the 'uniform' cluster with proper-name
    categories → proper_names permutation."""
    from sklearn.cluster import KMeans
    cat_surprise = {cat: info["surprise_mean"]
                    for cat, info in surprise_data["by_category"].items()}
    sorted_cats = sorted(cat_surprise, key=lambda c: -cat_surprise[c])

    cat_examples = defaultdict(list)
    for inp, lf, c in gen_pairs:
        cat_examples[c].append((inp, lf, c))
    per_cat = max(20, 500 // len(sorted_cats))
    top_examples = []
    for cat in sorted_cats:
        top_examples.extend(cat_examples[cat][:per_cat])

    X = []; cats = []
    for inp, lf, cat in top_examples:
        f = extract_features(inp, lf)
        X.append([f[name] for name in FEATURE_NAMES])
        cats.append(cat)
    X = np.array(X, dtype=float)
    mu = X.mean(0); std = X.std(0) + 1e-8
    X_norm = (X - mu) / std

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(X_norm)

    lexical_gaps = []
    for ci in range(4):
        mask = labels == ci
        if mask.sum() == 0: continue
        cluster_cats = [cats[i] for i in range(len(cats)) if mask[i]]
        cat_counts = Counter(cluster_cats)
        n = int(mask.sum())
        feat_means = X[mask].mean(0)
        fm = {name: float(feat_means[j]) for j, name in enumerate(FEATURE_NAMES)}

        # Detect: large cluster + proper-name categories + no structural signature
        has_proper = any("proper" in c.lower() for c in cluster_cats)
        is_uniform = (fm.get("n_nmod", 0) < 2 and fm.get("n_ccomp", 0) < 2
                      and fm.get("nmod_on_subj", 0) < 0.3
                      and fm.get("has_passive", 0) < 0.3)
        if n > 100 and has_proper and is_uniform:
            lexical_gaps.append({
                "type": "proper_names",
                "reason": f"uniform cluster with proper-name categories, n={n}",
                "n": n,
            })
    return lexical_gaps


# ══════════════════════════════════════════════════════════════
# Fix generation
# ══════════════════════════════════════════════════════════════
def generate_fix(fix_type, train_pairs, in_w2i=None, out_w2i=None):
    """Returns (aug_examples, perm_classes, description)."""
    if fix_type == "generate_nmod_on_subject":
        aug = generate_subj_pp_augmentations(train_pairs)
        return aug, [], f"PP obj→subj: {len(aug)}"
    elif fix_type == "generate_noun_position_swap":
        aug = generate_noun_position_augmentations(train_pairs)
        return aug, [], f"noun position swap: {len(aug)}"
    elif fix_type == "generate_deeper_modification":
        aug = generate_pp_recursion_extension(train_pairs, max_aug=500)
        return aug, [], f"PP recursion +1: {len(aug)}"
    elif fix_type == "generate_position_examples":
        aug = generate_active_to_passive(train_pairs)
        return aug, [], f"active→passive: {len(aug)}"
    elif fix_type == "generate_verb_in_missing_form":
        aug = generate_active_to_passive(train_pairs)
        return aug[:200], [], f"verb form: {min(len(aug),200)}"
    elif fix_type == "proper_names":
        if in_w2i and out_w2i:
            pn = detect_proper_names(train_pairs, in_w2i, out_w2i)
            if pn:
                pc = [[(p[0], p[1]) for p in pn]]
                return [], pc, f"proper names permutation: {len(pn)} names"
        return [], [], "no proper names detected"
    else:
        return [], [], f"unknown fix {fix_type}"


# ══════════════════════════════════════════════════════════════
# Fix validation (Level 2)
# ══════════════════════════════════════════════════════════════
def validate_fix(fix_type, fix_aug, fix_perm, train_pairs, dev_pairs, gen_pairs,
                 base_score, seed=42, device="cuda"):
    aug_train = train_pairs + fix_aug
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    _, _, _, test_gen, _, _ = train_model(
        aug_train, dev_pairs, gen_pairs, perm_classes=fix_perm or None,
        seed=seed, epochs=20, device=device, greedy_eval=False)
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return test_gen, test_gen - base_score


# ══════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════
def run_loop(dataset, max_cycles=5, surprise_threshold=1.5, seed=42,
             epochs=60, device="cuda", out_dir="runs_master",
             b0_checkpoint=None):
    os.makedirs(out_dir, exist_ok=True)
    loop_dir = os.path.join(out_dir, f"loopv3_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(loop_dir, exist_ok=True)

    train_orig, dev, gen = load_data(dataset)
    cycle_log = []
    prev_ratio = None
    accumulated_fixes = []  # list of (fix_type, fix_aug, fix_perm)
    use_copy = False  # switched ON if TF/greedy ratio > 5

    for cycle in range(max_cycles):
        print(f"\n{'='*70}")
        print(f"  CYCLE {cycle} — {dataset.upper()} (v3 combined)")
        print(f"{'='*70}")
        cycle_dir = os.path.join(loop_dir, f"cycle_{cycle}")
        os.makedirs(cycle_dir, exist_ok=True)
        t0 = time.time()

        # Build training data with accumulated fixes
        train_pairs = list(train_orig)
        perm_classes = []
        for ft, fa, fp in accumulated_fixes:
            train_pairs = train_pairs + fa
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
            else:
                ge_ds = COGSDataset(gen, in_w2i, out_w2i)
                ge_ld = DataLoader(ge_ds, 16, shuffle=False, collate_fn=collate, num_workers=0)
                greedy = evaluate(model, ge_ld, out_w2i, dev_obj)
                gen_greedy = greedy["exact"]; by_cat = greedy.get("by_cat", {})
            metrics = []
        else:
            copy_str = " (COPY+schedSampling)" if use_copy else ""
            print(f"\n  Training with {len(accumulated_fixes)} fixes "
                  f"({len(train_pairs)} examples){copy_str}...")
            model, in_w2i, out_w2i, gen_greedy, by_cat, metrics = train_model(
                train_pairs, dev, gen, perm_classes=perm_classes or None,
                seed=seed, epochs=epochs, device=device, run_dir=cycle_dir,
                use_copy=use_copy)

        # Measure surprise
        print("\n  Measuring surprise...")
        surprise = compute_surprise(model, in_w2i, out_w2i, train_orig, gen, device)
        ratio = surprise["ratio"]
        print(f"  Surprise ratio: {ratio:.2f}x")

        # ── DETECTION: Lexical + Structural ──
        print("\n  [LEXICAL] Clustering for lexical gaps...")
        lexical = detect_lexical_gaps(surprise, gen, train_orig, in_w2i, out_w2i)
        print(f"  [LEXICAL] Detected: {[g['type'] for g in lexical]}")
        for g in lexical:
            print(f"    {g['type']}: {g['reason']}")

        print("\n  [STRUCTURAL] Causal diagnosis...")
        diag = diagnose(model, in_w2i, out_w2i, gen, train_orig,
                        device=device, max_examples=500)
        with open(os.path.join(cycle_dir, "diagnosis.json"), "w") as f:
            json.dump(diag, f, indent=2)

        # Collect all candidate fixes
        candidate_fixes = []
        # Lexical
        for g in lexical:
            if g["type"] not in [ft for ft, _, _ in accumulated_fixes]:
                fa, fp, desc = generate_fix(g["type"], train_orig, in_w2i, out_w2i)
                if fa or fp:
                    candidate_fixes.append({"fix_type": g["type"], "aug": fa,
                                            "perm": fp, "desc": desc,
                                            "source": "lexical"})
        # Structural
        for diff_type, hyp_data in diag["hypotheses"].items():
            if not hyp_data.get("confirmed"): continue
            ft = hyp_data.get("fix_type")
            if not ft: continue
            if ft in [x[0] for x in accumulated_fixes]: continue
            # Skip verb_form if generate_position already in candidates (redundant)
            if ft == "generate_verb_in_missing_form" and any(
                    c["fix_type"] == "generate_position_examples" for c in candidate_fixes):
                continue
            fa, fp, desc = generate_fix(ft, train_orig, in_w2i, out_w2i)
            if fa or fp:
                candidate_fixes.append({"fix_type": ft, "aug": fa,
                                        "perm": fp, "desc": desc,
                                        "source": "structural"})

        print(f"\n  Total candidates: {len(candidate_fixes)}")
        for c in candidate_fixes:
            print(f"    [{c['source']}] {c['fix_type']}: {c['desc']}")

        # ── TF vs Greedy gap detection ──
        # If the model's TF gen is much higher than greedy gen, the issue
        # is autoregressive inference, not coverage. Activate copy+sched sampling.
        gen_tf = metrics[-1]["gen"]["exact"] if metrics else 0.0
        tf_greedy_ratio = gen_tf / max(gen_greedy, 1.0) if gen_greedy > 0 else (gen_tf if gen_tf > 0 else 0)
        inference_gap = gen_tf > 5 * max(gen_greedy, 1.0) and gen_tf > 10
        if inference_gap and not use_copy:
            print(f"\n  [INFERENCE GAP] TF={gen_tf:.2f}% >> Greedy={gen_greedy:.2f}% "
                  f"(ratio={tf_greedy_ratio:.1f}x)")
            print(f"  → Activating COPY mechanism + scheduled sampling for next cycle")
            print(f"  → Skipping augmentation-based fixes this cycle")
            use_copy = True
            candidate_fixes = []  # override — focus on copy mechanism

        # Log
        entry = {
            "cycle": cycle, "gen_greedy": gen_greedy, "by_cat": by_cat,
            "gen_tf": gen_tf, "tf_greedy_ratio": tf_greedy_ratio,
            "surprise_ratio": ratio, "n_train": len(train_pairs),
            "lexical_gaps": [g["type"] for g in lexical],
            "structural_diagnosis": {k: v for k, v in diag["diff_types"].items() if k != "correct"},
            "candidate_fixes": [c["fix_type"] for c in candidate_fixes],
            "accumulated_fixes": [ft for ft, _, _ in accumulated_fixes],
            "use_copy": use_copy,
            "time_s": time.time() - t0,
        }
        cycle_log.append(entry)
        with open(os.path.join(loop_dir, "cycle_log.json"), "w") as f:
            json.dump(cycle_log, f, indent=2)

        print(f"\n  Cycle {cycle}: gen={gen_greedy:.2f}% ratio={ratio:.2f}x")

        # Convergence
        if ratio < surprise_threshold:
            print(f"\n  CONVERGED: ratio {ratio:.2f} < {surprise_threshold}"); break
        if prev_ratio is not None and abs(ratio - prev_ratio) < 0.05 and len(accumulated_fixes) > 0:
            print(f"\n  STALLED: Δratio={abs(ratio-prev_ratio):.3f} < 0.05"); break
        prev_ratio = ratio
        if not candidate_fixes:
            if use_copy and (not accumulated_fixes or
                              accumulated_fixes[-1][0] != "__copy_mechanism__"):
                # Copy just activated — continue with an empty fix placeholder
                print("\n  No new augmentations, but copy activated → continue with copy only")
                accumulated_fixes.append(("__copy_mechanism__", [], []))
                continue
            print("\n  No new fixes. Stopping."); break

        # ── VALIDATION (Level 2) ──
        print(f"\n  === LEVEL 2: Validating {len(candidate_fixes)} candidates ===")
        print(f"  Baseline (20 epochs)...")
        _, _, _, base_score, _, _ = train_model(
            train_pairs, dev, gen, perm_classes=perm_classes or None,
            seed=seed, epochs=20, device=device, greedy_eval=False)
        print(f"  Baseline gen(TF): {base_score:.2f}%")

        new_fixes = []
        for c in candidate_fixes:
            print(f"\n  Testing [{c['source']}] {c['fix_type']} ({c['desc']})...")
            score, delta = validate_fix(c["fix_type"], c["aug"], c["perm"],
                                         train_pairs, dev, gen, base_score,
                                         seed=seed, device=device)
            print(f"    score={score:.2f}%  Δ={delta:+.2f}%")
            if delta >= -1.0:
                new_fixes.append((c["fix_type"], c["aug"], c["perm"]))
                print(f"    → ACCEPTED")
            else:
                print(f"    → REJECTED")

        if not new_fixes:
            print("\n  All fixes rejected. Stopping."); break
        accumulated_fixes.extend(new_fixes)
        print(f"\n  Accumulated: {[ft for ft, _, _ in accumulated_fixes]}")

    # Final
    print(f"\n{'='*70}")
    print(f"  LOOP v3 COMPLETE — {len(cycle_log)} cycles")
    print(f"{'='*70}")
    for c in cycle_log:
        fixes = ", ".join(c["accumulated_fixes"]) if c["accumulated_fixes"] else "baseline"
        print(f"  Cycle {c['cycle']}: gen={c['gen_greedy']:.2f}% "
              f"ratio={c['surprise_ratio']:.2f}x  [{fixes}]")
    with open(os.path.join(loop_dir, "final_summary.json"), "w") as f:
        json.dump({"cycles": cycle_log, "dataset": dataset}, f, indent=2)
    print(f"\n  Saved: {loop_dir}")
    return cycle_log


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Phase 3 v3 — Combined auto loop")
    p.add_argument("--dataset", type=str, default="cogs", choices=["cogs", "slog"])
    p.add_argument("--max-cycles", type=int, default=3)
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
