#!/usr/bin/env python3
"""SLOG — identifie les "catégories problématiques" pour le meta-modèle.

Critère : catégories où le modèle de base se trompe MAIS surprise/entropy sont
faibles (= erreurs silencieuses, ce que le meta-modèle scalaire rate). Ce sont
les zones où l'encodeur structural sera testé.

Score par catégorie :
  silence_score = error_rate × (1 - surprise_z(errors)/surprise_z(corrects))
                = élevé si erreurs nombreuses ET surprise des erreurs
                  similaire à celle des corrects.

Usage :
  python3 slog_identify_problem_cats.py \\
      --dataset meta_data/slog_meta_dataset.jsonl \\
      --top 4 \\
      --output meta_data/slog_problem_cats.txt
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    by_cat = defaultdict(list)
    with open(args.dataset, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_cat[r["category"]].append(r)
    print(f"Loaded {sum(len(v) for v in by_cat.values())} records across "
          f"{len(by_cat)} categories")

    # Per-category stats
    stats = []
    for cat, recs in by_cat.items():
        n = len(recs)
        n_err = sum(1 for r in recs if not r["exact_match"])
        if n_err == 0 or n_err == n or n < 30:
            silence_score = 0.0
        else:
            err_surp = [r["features_inference"]["surprise_mean"]
                        for r in recs if not r["exact_match"]]
            cor_surp = [r["features_inference"]["surprise_mean"]
                        for r in recs if r["exact_match"]]
            ratio = (np.mean(err_surp) / max(np.mean(cor_surp), 1e-9)
                     if cor_surp else 0.0)
            err_rate = n_err / n
            # silence : faible si surprise(err)/surprise(cor) >> 1, élevé si ≈ 1
            silence = 1.0 / max(ratio, 1.0)
            # combiné : la cat doit avoir des erreurs mesurables
            silence_score = float(err_rate * silence)
            err_mean = float(np.mean(err_surp))
            cor_mean = float(np.mean(cor_surp))
        stats.append({
            "cat": cat, "n": n, "n_errors": n_err,
            "error_rate": n_err / n if n else 0.0,
            "silence_score": silence_score,
        })

    stats.sort(key=lambda x: -x["silence_score"])

    print()
    print(f"{'category':40s}  n  err  err%  silence_score")
    print("-" * 75)
    for s in stats:
        print(f"{s['cat']:40s}  {s['n']:4d}  {s['n_errors']:4d}  "
              f"{100*s['error_rate']:5.1f}%  {s['silence_score']:.4f}")

    # Top problematic
    top = [s["cat"] for s in stats[:args.top]]
    print()
    print(f"Top {args.top} problem cats : {top}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(",".join(top))
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
