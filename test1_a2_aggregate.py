#!/usr/bin/env python3
"""TEST 1 — Reproduction A.2 cycle 1 sur 3 seeds.

Lit cycle 0 et cycle 1 pour chaque seed, calcule deltas par catégorie.
Critères pré-engagés (Δ_pria = mean(cycle1.prim_to_inf_arg) - mean(cycle0)).

CONVENTION : les valeurs `final_gen_exact_greedy` et `greedy_gen_by_cat` dans
summary.json sont en pourcentage (0-100), NOT en fraction. Donc Δ se calcule
par soustraction directe (résultat en points de pourcentage, pp).

Usage :
  python3 test1_a2_aggregate.py \\
      --cycle0-glob 'runs/B4_jepa_s{seed}/B4_s{seed}_*/summary.json' \\
      --cycle1-glob 'runs/B4_a2_cycle1_s{seed}/B4_s{seed}_*/summary.json' \\
      --seeds 42,123,456 \\
      --output-dir tests/test1_a2_repro
"""
import argparse
import glob
import json
import math
import os

import numpy as np


def _load(pattern, seed):
    p = pattern.replace("{seed}", str(seed))
    matches = sorted(glob.glob(p))
    if not matches:
        return None
    with open(matches[0], "r") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle0-glob", required=True,
                    help="Glob avec {seed} comme placeholder")
    ap.add_argument("--cycle1-glob", required=True)
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",")]
    per_seed = {}
    missing = []
    for s in seeds:
        c0 = _load(args.cycle0_glob, s)
        c1 = _load(args.cycle1_glob, s)
        if c0 is None or c1 is None:
            missing.append({"seed": s,
                            "cycle0_found": c0 is not None,
                            "cycle1_found": c1 is not None})
            continue
        per_seed[s] = {"cycle0": c0, "cycle1": c1}

    if missing:
        print(f"WARNING: missing summaries for {len(missing)} seed(s):")
        for m in missing:
            print(f"  {m}")
    if not per_seed:
        print("ERROR: no seeds found, aborting")
        raise SystemExit(1)

    # Toutes les valeurs ci-dessous sont en POURCENTAGE (0-100), pas en fraction.
    cat_deltas = {}    # cat → list[Δ_pp par seed]
    global_deltas = []  # Δ gen_greedy par seed
    pria_deltas = []   # Δ prim_to_inf_arg par seed (cible critère)
    for s, d in per_seed.items():
        c0_gg = d["cycle0"].get("final_gen_exact_greedy")
        c1_gg = d["cycle1"].get("final_gen_exact_greedy")
        if c0_gg is None or c1_gg is None:
            continue
        global_deltas.append(c1_gg - c0_gg)  # déjà en pp

        c0_by = d["cycle0"].get("greedy_gen_by_cat", {})
        c1_by = d["cycle1"].get("greedy_gen_by_cat", {})
        all_cats = set(c0_by) | set(c1_by)
        for cat in all_cats:
            v0 = c0_by.get(cat); v1 = c1_by.get(cat)
            if v0 is None or v1 is None:
                continue
            cat_deltas.setdefault(cat, []).append(v1 - v0)
        if "prim_to_inf_arg" in c0_by and "prim_to_inf_arg" in c1_by:
            pria_deltas.append(c1_by["prim_to_inf_arg"] - c0_by["prim_to_inf_arg"])

    # ── Stats agrégées
    def _stats(arr):
        if not arr:
            return {"mean": float("nan"), "std": float("nan"),
                    "ci95_lo": float("nan"), "ci95_hi": float("nan")}
        a = np.array(arr, dtype=float)
        m = float(a.mean())
        sd = float(a.std(ddof=0)) if len(a) > 1 else 0.0
        # IC95 normal approx (n=3 trop petit pour bootstrap, on reste honnête)
        return {"mean": m, "std": sd,
                "ci95_lo": m - 1.96 * sd / max(math.sqrt(len(a)), 1),
                "ci95_hi": m + 1.96 * sd / max(math.sqrt(len(a)), 1),
                "values": [float(x) for x in a]}

    pria_stats = _stats(pria_deltas)
    global_stats = _stats(global_deltas)
    cat_stats = {cat: _stats(deltas) for cat, deltas in cat_deltas.items()}

    # ── Verdict pré-engagé
    pria_mean = pria_stats["mean"]
    big_losers = [cat for cat, st in cat_stats.items()
                  if not math.isnan(st["mean"]) and st["mean"] <= -10.0]
    has_5pt_loser = any(st["mean"] <= -5.0 for st in cat_stats.values()
                         if not math.isnan(st["mean"]))

    if math.isnan(pria_mean):
        verdict = "REPORTÉ"
        reason = "Δ_pria indisponible — données manquantes."
    elif pria_mean > -2.0 and not has_5pt_loser:
        verdict = "GO A.2"
        reason = (f"Δ_pria = {pria_mean:+.2f}pp > -2pp et aucune catégorie "
                  "ne perd > 5pp. Le résultat original était bruit.")
    elif pria_mean >= -8.0:
        verdict = "TEST"
        reason = (f"Δ_pria = {pria_mean:+.2f}pp ∈ [-8, -2]. Régression "
                  "modérée, investiguer la cause spécifique avant scaling.")
    elif pria_mean < -8.0 or len(big_losers) > 1:
        verdict = "NO-GO A.2"
        reason = (f"Δ_pria = {pria_mean:+.2f}pp < -8pp"
                  + (f" OU {len(big_losers)} catégories perdent > 10pp"
                     if big_losers else "")
                  + ". Catastrophic forgetting structurel confirmé.")
    else:
        verdict = "AMBIGU"
        reason = "Critères ambigus — voir détails."

    # ── Sortie
    summary = {
        "seeds": list(per_seed.keys()),
        "missing_seeds": missing,
        "global_gen_greedy_delta_pp": global_stats,
        "prim_to_inf_arg_delta_pp": pria_stats,
        "categories_losing_more_than_10pp": big_losers,
        "verdict": verdict,
        "reason": reason,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.output_dir, "per_category_deltas.json"), "w") as f:
        json.dump(cat_stats, f, indent=2)

    # ── verdict.txt
    lines = [f"# TEST 1 — Reproduction A.2 cycle 1 sur {len(per_seed)} seeds", ""]
    lines.append(f"## Verdict : {verdict}")
    lines.append("")
    lines.append(reason)
    lines.append("")
    lines.append(f"- Seeds analysés : {list(per_seed.keys())}")
    if missing:
        lines.append(f"- ⚠️ Seeds manquants : {[m['seed'] for m in missing]}")
    lines.append("")
    lines.append("## Statistiques globales (en pp)")
    lines.append("")
    lines.append("| Métrique | mean | std | IC95 |")
    lines.append("|---|---:|---:|---|")
    lines.append(f"| Δ gen greedy global | {global_stats['mean']:+.2f}pp | "
                 f"{global_stats['std']:.2f}pp | "
                 f"[{global_stats['ci95_lo']:+.2f}, {global_stats['ci95_hi']:+.2f}] |")
    lines.append(f"| Δ prim_to_inf_arg | {pria_stats['mean']:+.2f}pp | "
                 f"{pria_stats['std']:.2f}pp | "
                 f"[{pria_stats['ci95_lo']:+.2f}, {pria_stats['ci95_hi']:+.2f}] |")
    lines.append("")
    lines.append(f"Valeurs Δ_pria par seed : {pria_stats.get('values', [])}")
    lines.append(f"Valeurs Δ global par seed : {global_stats.get('values', [])}")
    lines.append("")
    lines.append("## Régressions par catégorie (mean Δ ≤ -3pp)")
    lines.append("")
    lines.append("| Catégorie | Δ mean | std |")
    lines.append("|---|---:|---:|")
    big_regs = sorted([(cat, st) for cat, st in cat_stats.items()
                        if not math.isnan(st["mean"]) and st["mean"] <= -3.0],
                       key=lambda x: x[1]["mean"])
    if not big_regs:
        lines.append("| _aucune_ | | |")
    for cat, st in big_regs:
        lines.append(f"| {cat} | {st['mean']:+.2f}pp | {st['std']:.2f}pp |")
    lines.append("")
    lines.append("Détails complets : `summary.json`, `per_category_deltas.json`.")

    with open(os.path.join(args.output_dir, "verdict.txt"), "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {args.output_dir}/{{summary.json, per_category_deltas.json, verdict.txt}}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
