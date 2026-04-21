#!/usr/bin/env python3
"""
Phase 3a — Clustering des gaps structurels sans labels.

Prend les top-K exemples haute surprise, extrait des features structurelles,
cluster avec KMeans, puis vérifie si les clusters correspondent aux catégories.

Usage:
  python3 cluster_gaps.py --surprise runs_master/surprise_cogs.json --dataset cogs
"""
import os, sys, json, argparse, re
from collections import defaultdict, Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cogs_compositional import parse_cogs_tsv

HERE = os.path.dirname(os.path.abspath(__file__))


def extract_features(inp, lf):
    """Extract structural features from input + LF (no category info)."""
    tokens = inp.split()
    lf_tokens = lf.split()
    n_nmod = len(re.findall(r'nmod', lf))
    n_ccomp = len(re.findall(r'ccomp', lf))
    n_and = lf.count(" AND ")
    has_passive = 1.0 if ("was" in tokens and "by" in tokens) else 0.0
    has_pp_dative = 0.0
    if "to" in tokens:
        to_idx = tokens.index("to")
        if to_idx + 1 < len(tokens) and tokens[to_idx + 1][0].isupper():
            has_pp_dative = 1.0
    # nmod position: does the nmod attach to x_1 (subject) or x_3+ (object)?
    # LF format: "noun . nmod . PREP ( x _ N , x _ M )" or "noun . nmod ( x _ N , x _ M )"
    nmod_on_subj = 0.0
    nmod_matches = re.findall(r'nmod[^(]*\( (x _ \d+)', lf)
    for m in nmod_matches:
        idx = int(m.split()[-1])
        if idx <= 1:
            nmod_on_subj = 1.0
            break

    return {
        "input_len": len(tokens),
        "lf_len": len(lf_tokens),
        "n_nmod": n_nmod,
        "n_ccomp": n_ccomp,
        "has_passive": has_passive,
        "has_pp_dative": has_pp_dative,
        "n_and": n_and,
        "lf_input_ratio": len(lf_tokens) / max(len(tokens), 1),
        "nmod_on_subj": nmod_on_subj,
    }

FEATURE_NAMES = ["input_len", "lf_len", "n_nmod", "n_ccomp",
                 "has_passive", "has_pp_dative", "n_and", "lf_input_ratio",
                 "nmod_on_subj"]


def main():
    p = argparse.ArgumentParser(description="Phase 3a — cluster gaps")
    p.add_argument("--surprise", type=str, required=True,
                   help="Path to surprise_cogs.json")
    p.add_argument("--dataset", type=str, default="cogs", choices=["cogs", "slog"])
    p.add_argument("--top-k", type=int, default=500)
    p.add_argument("--out-dir", type=str, default="runs_master")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load surprise data to get per-category ranking
    surprise_data = json.load(open(args.surprise))
    by_cat = surprise_data["by_category"]

    # Load gen data
    if args.dataset == "cogs":
        gen_pairs = parse_cogs_tsv(os.path.join(HERE, "data", "cogs", "gen.tsv"))
    else:
        gen_pairs = parse_cogs_tsv(os.path.join(HERE, "data", "slog",
                                   "generalization_sets", "gen_cogsLF.tsv"))

    # Compute surprise per example (re-run from saved category means as proxy,
    # but ideally we'd save per-example. Approximate: sort by category mean surprise)
    # Actually, we need per-example surprise. Let's re-derive from the JSON.
    # The JSON has category-level stats. For proper clustering, we need to pick
    # the top-K examples. Since we don't have per-example surprise saved,
    # we'll use category surprise as proxy: pick all examples from the
    # highest-surprise categories until we have top-K.
    cat_surprise = {cat: info["surprise_mean"] for cat, info in by_cat.items()}
    sorted_cats = sorted(cat_surprise, key=lambda c: -cat_surprise[c])

    # Stratified sampling: take top examples from each category,
    # more from higher-surprise categories
    # Split budget proportionally to surprise rank
    n_cats = len(sorted_cats)
    per_cat = max(20, args.top_k // n_cats)  # at least 20 per category
    top_examples = []
    cat_examples = defaultdict(list)
    for inp, lf, c in gen_pairs:
        cat_examples[c].append((inp, lf, c, cat_surprise.get(c, 0)))
    for cat in sorted_cats:
        exs = cat_examples[cat][:per_cat]
        top_examples.extend(exs)
    # Cap to top_k, prioritizing higher-surprise categories
    if len(top_examples) > args.top_k:
        top_examples = top_examples[:args.top_k]
    print(f"Stratified {len(top_examples)} examples from {len(set(c for _,_,c,_ in top_examples))} categories")

    # Extract features
    X = []
    cats = []
    examples = []
    for inp, lf, cat, surp in top_examples:
        f = extract_features(inp, lf)
        X.append([f[name] for name in FEATURE_NAMES])
        cats.append(cat)
        examples.append((inp, lf))
    X = np.array(X, dtype=float)

    # Normalize features
    mu = X.mean(0)
    std = X.std(0) + 1e-8
    X_norm = (X - mu) / std

    # KMeans for k=3,4,5
    from sklearn.cluster import KMeans

    for k in [3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_norm)

        print(f"\n{'='*70}")
        print(f"  KMeans k={k}")
        print(f"{'='*70}")

        for ci in range(k):
            mask = labels == ci
            n_in_cluster = mask.sum()
            if n_in_cluster == 0:
                continue

            # Category distribution
            cluster_cats = [cats[i] for i in range(len(cats)) if mask[i]]
            cat_counts = Counter(cluster_cats)
            total = len(cluster_cats)
            top_cat = cat_counts.most_common(1)[0]
            purity = top_cat[1] / total * 100

            # Feature means
            feat_means = X[mask].mean(0)

            print(f"\n  Cluster {ci} — {n_in_cluster} examples (purity: {purity:.0f}% {top_cat[0]})")
            print(f"    Features: ", end="")
            for name, val in zip(FEATURE_NAMES, feat_means):
                print(f"{name}={val:.1f} ", end="")
            print()

            # Category breakdown
            print(f"    Categories:")
            for cat, count in cat_counts.most_common(5):
                pct = count / total * 100
                print(f"      {cat:<40s} {count:>4d} ({pct:5.1f}%)")

            # 3 example inputs
            cluster_indices = [i for i in range(len(cats)) if mask[i]]
            print(f"    Examples:")
            for idx in cluster_indices[:3]:
                print(f"      {examples[idx][0][:80]}")

    # Save clustering results for best k
    best_k = 4
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X_norm)
    cluster_data = {}
    for ci in range(best_k):
        mask = labels == ci
        cluster_cats = [cats[i] for i in range(len(cats)) if mask[i]]
        cat_counts = Counter(cluster_cats)
        cluster_data[f"cluster_{ci}"] = {
            "n": int(mask.sum()),
            "categories": dict(cat_counts.most_common()),
            "feature_means": {name: float(X[mask].mean(0)[j])
                              for j, name in enumerate(FEATURE_NAMES)},
            "purity": float(cat_counts.most_common(1)[0][1] / mask.sum() * 100),
        }
    out_path = os.path.join(args.out_dir, f"clusters_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump(cluster_data, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
