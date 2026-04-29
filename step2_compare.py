#!/usr/bin/env python3
"""ÉTAPE 2 — comparaison B4 sans JEPA vs B4+JEPA historiques.

CONVENTION : `final_gen_exact_greedy` et `greedy_gen_by_cat` sont des valeurs
en pourcentage (0-100), pas en fraction. Δ se calcule par soustraction
directe et est en points de pourcentage (pp).

Usage :
  python3 step2_compare.py \\
      --new-cogs-glob 'pipeline_clean/step2_base_model/runs/cogs_s{seed}/B4_s{seed}_*/summary.json' \\
      --new-slog-glob 'pipeline_clean/step2_base_model/runs/slog_s{seed}/B4_s{seed}_*/summary.json' \\
      --hist-cogs-glob 'runs/B4_jepa_s{seed}/B4_s{seed}_*/summary.json' \\
      --hist-slog-glob 'runs/B4_slog_jepa/B4_s{seed}_*/summary.json' \\
      --seeds 42,123,456 \\
      --output-dir pipeline_clean/step2_base_model
"""
import argparse
import glob
import json
import math
import os

import numpy as np


def _load_for_seeds(pattern, seeds):
    out = {}
    for s in seeds:
        p = pattern.replace("{seed}", str(s))
        matches = sorted(glob.glob(p))
        if matches:
            with open(matches[0], "r") as f:
                out[s] = json.load(f)
    return out


def _stats_per_metric(summaries, key):
    """summaries: {seed: summary_dict}. key in summary_dict (en pp 0-100)."""
    vals = [d.get(key) for d in summaries.values() if d.get(key) is not None]
    vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def _by_cat_means(summaries):
    """{seed: summary} → {cat: list[values across seeds] in pp}."""
    cat_to_vals = {}
    for d in summaries.values():
        by = d.get("greedy_gen_by_cat", {})
        for cat, v in by.items():
            cat_to_vals.setdefault(cat, []).append(v)
    return cat_to_vals


def _fmt_pp(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.2f}%"


def _fmt_delta(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:+.2f}pp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-cogs-glob", required=True)
    ap.add_argument("--new-slog-glob", required=True)
    ap.add_argument("--hist-cogs-glob", required=True)
    ap.add_argument("--hist-slog-glob", required=True)
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",")]
    new_c = _load_for_seeds(args.new_cogs_glob, seeds)
    new_s = _load_for_seeds(args.new_slog_glob, seeds)
    hist_c = _load_for_seeds(args.hist_cogs_glob, seeds)
    hist_s = _load_for_seeds(args.hist_slog_glob, seeds)
    print(f"Loaded — new COGS: {sorted(new_c)}, new SLOG: {sorted(new_s)}")
    print(f"        hist COGS: {sorted(hist_c)}, hist SLOG: {sorted(hist_s)}")

    def _bench_compare(new, hist, bench_name):
        new_m, new_sd, new_n = _stats_per_metric(new, "final_gen_exact_greedy")
        hist_m, hist_sd, hist_n = _stats_per_metric(hist, "final_gen_exact_greedy")
        delta = new_m - hist_m if not (math.isnan(new_m) or math.isnan(hist_m)) else float("nan")

        # By category : delta moyen par cat
        new_by = _by_cat_means(new); hist_by = _by_cat_means(hist)
        cats_common = sorted(set(new_by) & set(hist_by))
        cat_deltas = {}
        for cat in cats_common:
            n_v = [v for v in new_by[cat] if not math.isnan(v)]
            h_v = [v for v in hist_by[cat] if not math.isnan(v)]
            if n_v and h_v:
                cat_deltas[cat] = float(np.mean(n_v)) - float(np.mean(h_v))

        big_losers = [c for c, d in cat_deltas.items() if d <= -5.0]

        return {
            "new_mean": new_m, "new_std": new_sd, "new_n_seeds": new_n,
            "hist_mean": hist_m, "hist_std": hist_sd, "hist_n_seeds": hist_n,
            "delta_pp": delta,
            "cat_deltas_pp": cat_deltas,
            "big_losers_5pp": big_losers,
        }

    cogs_res = _bench_compare(new_c, hist_c, "COGS")
    slog_res = _bench_compare(new_s, hist_s, "SLOG")
    print(f"\n[COGS] new={_fmt_pp(cogs_res['new_mean'])} hist={_fmt_pp(cogs_res['hist_mean'])} "
          f"Δ={_fmt_delta(cogs_res['delta_pp'])}")
    print(f"[SLOG] new={_fmt_pp(slog_res['new_mean'])} hist={_fmt_pp(slog_res['hist_mean'])} "
          f"Δ={_fmt_delta(slog_res['delta_pp'])}")

    # ── Verdict pré-engagé
    cogs_d = cogs_res["delta_pp"]; slog_d = slog_res["delta_pp"]
    big_losers_total = cogs_res["big_losers_5pp"] + slog_res["big_losers_5pp"]
    if math.isnan(cogs_d) or math.isnan(slog_d):
        verdict = "REPORTÉ"
        reason = "Données manquantes pour comparer."
    elif cogs_d < -2.0:
        verdict = "INATTENDU (COGS dégradé)"
        reason = (f"COGS Δ = {cogs_d:+.2f}pp < -2pp. JEPA contribuait "
                  "positivement à COGS d'une manière qu'on n'avait pas vue. STOP.")
    elif slog_d < +1.0:
        verdict = "PARTIELLEMENT INATTENDU (SLOG insuffisant)"
        reason = (f"SLOG Δ = {slog_d:+.2f}pp < +1pp. La dégradation SLOG vient "
                  "peu du JEPA. STOP. Investiguer ce qui dégrade vraiment SLOG.")
    elif (-1.0 <= cogs_d <= 1.0) and (2.0 <= slog_d <= 5.0):
        verdict = "ATTENDU"
        reason = "Retrait JEPA validé. On continue."
    else:
        verdict = "MIXTE"
        reason = f"COGS Δ={cogs_d:+.2f}pp, SLOG Δ={slog_d:+.2f}pp. Ni STOP ni ATTENDU clean."

    alerts = ([f"⚠️ {cat} régresse de {d:+.2f}pp" for cat, d in cogs_res["cat_deltas_pp"].items() if d <= -5]
              + [f"⚠️ SLOG/{cat} régresse de {d:+.2f}pp" for cat, d in slog_res["cat_deltas_pp"].items() if d <= -5])

    # ── Markdown
    lines = ["# ÉTAPE 2 — Re-mesure modèle de base sans JEPA", ""]
    lines.append("Toutes les valeurs `gen_greedy` sont en pourcentage (0-100). "
                 "Δ par soustraction directe = points de pourcentage (pp).")
    lines.append("")
    lines.append(f"## Verdict : {verdict}")
    lines.append("")
    lines.append(reason)
    lines.append("")
    lines.append("## COGS")
    lines.append("")
    lines.append("| Métrique | B4 sans JEPA (nouveau) | B4+JEPA (historique) | Δ |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| gen_greedy mean | {_fmt_pp(cogs_res['new_mean'])} "
                 f"({cogs_res['new_n_seeds']} seeds) | "
                 f"{_fmt_pp(cogs_res['hist_mean'])} ({cogs_res['hist_n_seeds']} seeds) | "
                 f"{_fmt_delta(cogs_res['delta_pp'])} |")
    lines.append(f"| gen_greedy std  | {cogs_res['new_std']:.2f}pp | "
                 f"{cogs_res['hist_std']:.2f}pp | — |")
    lines.append("")
    lines.append("## SLOG")
    lines.append("")
    lines.append("| Métrique | B4 sans JEPA (nouveau) | B4+JEPA (historique) | Δ |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| gen_greedy mean | {_fmt_pp(slog_res['new_mean'])} "
                 f"({slog_res['new_n_seeds']} seeds) | "
                 f"{_fmt_pp(slog_res['hist_mean'])} ({slog_res['hist_n_seeds']} seeds) | "
                 f"{_fmt_delta(slog_res['delta_pp'])} |")
    lines.append(f"| gen_greedy std  | {slog_res['new_std']:.2f}pp | "
                 f"{slog_res['hist_std']:.2f}pp | — |")
    lines.append("")
    lines.append("## Δ par catégorie (les plus grandes variations)")
    lines.append("")
    lines.append("### COGS")
    lines.append("")
    lines.append("| Catégorie | Δ pp |")
    lines.append("|---|---:|")
    sorted_c = sorted(cogs_res["cat_deltas_pp"].items(), key=lambda kv: kv[1])
    for cat, d in sorted_c[:5] + sorted_c[-5:]:
        lines.append(f"| {cat} | {_fmt_delta(d)} |")
    lines.append("")
    lines.append("### SLOG")
    lines.append("")
    lines.append("| Catégorie | Δ pp |")
    lines.append("|---|---:|")
    sorted_s = sorted(slog_res["cat_deltas_pp"].items(), key=lambda kv: kv[1])
    for cat, d in sorted_s[:5] + sorted_s[-5:]:
        lines.append(f"| {cat} | {_fmt_delta(d)} |")
    lines.append("")

    if alerts:
        lines.append("## Alertes (régression cat ≥ 5pp)")
        lines.append("")
        for a in alerts:
            lines.append(f"- {a}")
        lines.append("")
    else:
        lines.append("Aucune régression > 5pp par catégorie.")
        lines.append("")

    with open(os.path.join(args.output_dir, "comparison_with_jepa.md"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(args.output_dir, "verdict.txt"), "w") as f:
        f.write("\n".join(lines))

    # JSON dump
    dump = {
        "cogs": cogs_res,
        "slog": slog_res,
        "verdict": verdict,
        "reason": reason,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(dump, f, indent=2)
    print(f"\nWrote {args.output_dir}/{{verdict.txt, comparison_with_jepa.md, results.json}}")


if __name__ == "__main__":
    main()
