#!/usr/bin/env python3
"""Phase A.2 — Étape 5 : compare 2 cycles (current vs previous) à partir de
leurs summary.json (sortis par cogs_compositional.py).

Détecte gain global et régressions par catégorie ≥ 3pt.

Usage :
  python3 meta_a2_evaluate.py \\
      --current  runs/B4_a2_cycle1/B4_s42_*/summary.json \\
      --previous runs/B4_jepa_s42/B4_s42_*/summary.json \\
      --output   runs_meta/etape_a2/cycle_1/delta_vs_cycle0.md
"""
import argparse
import glob
import json
import math
import os


def _load_summary(pattern):
    matches = glob.glob(pattern)
    if not matches:
        return None
    with open(sorted(matches)[0], "r") as f:
        return json.load(f)


def _fmt(v):
    """Les valeurs dans summary.json sont déjà en pourcentage (0-100)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", required=True)
    ap.add_argument("--previous", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cycle-name", default="N")
    args = ap.parse_args()

    cur = _load_summary(args.current)
    prev = _load_summary(args.previous)
    if not cur or not prev:
        print(f"ERROR: could not load summaries: cur={cur is not None} prev={prev is not None}")
        raise SystemExit(1)

    cur_gg = cur.get("final_gen_exact_greedy")
    prev_gg = prev.get("final_gen_exact_greedy")
    if cur_gg is None or prev_gg is None:
        print(f"ERROR: missing final_gen_exact_greedy")
        raise SystemExit(1)

    delta_global = cur_gg - prev_gg  # déjà en points (valeurs 0-100)

    cur_by_cat = cur.get("greedy_gen_by_cat", {})
    prev_by_cat = prev.get("greedy_gen_by_cat", {})
    all_cats = sorted(set(cur_by_cat.keys()) | set(prev_by_cat.keys()))

    cat_deltas = []
    regressions = []
    gains = []
    for cat in all_cats:
        c = cur_by_cat.get(cat); p = prev_by_cat.get(cat)
        if c is None or p is None:
            cat_deltas.append((cat, p, c, None))
            continue
        d = c - p  # déjà en points
        cat_deltas.append((cat, p, c, d))
        if d <= -3.0:
            regressions.append((cat, p, c, d))
        elif d >= 1.0:
            gains.append((cat, p, c, d))

    lines = [f"# A.2 cycle {args.cycle_name} — delta vs précédent", ""]
    lines.append(f"- Précédent (cycle - 1) : {args.previous}")
    lines.append(f"- Courant (cycle {args.cycle_name}) : {args.current}")
    lines.append("")
    lines.append("## Global")
    lines.append("")
    lines.append("| Métrique | Précédent | Courant | Delta |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Gen greedy | {_fmt(prev_gg)} | {_fmt(cur_gg)} | "
                 f"{delta_global:+.2f}pt |")
    lines.append(f"| Best dev | {_fmt(prev.get('best_dev_exact'))} | "
                 f"{_fmt(cur.get('best_dev_exact'))} | "
                 f"{cur.get('best_dev_exact', 0) - prev.get('best_dev_exact', 0):+.2f}pt |")
    lines.append(f"| Best gen TF | {_fmt(prev.get('best_gen_exact_tf'))} | "
                 f"{_fmt(cur.get('best_gen_exact_tf'))} | "
                 f"{cur.get('best_gen_exact_tf', 0) - prev.get('best_gen_exact_tf', 0):+.2f}pt |")
    lines.append("")
    lines.append("## Par catégorie")
    lines.append("")
    lines.append("| Catégorie | Précédent | Courant | Delta |")
    lines.append("|---|---:|---:|---:|")
    cat_deltas.sort(key=lambda x: (x[3] if x[3] is not None else 0))
    for cat, p, c, d in cat_deltas:
        d_str = f"{d:+.2f}pt" if d is not None else "N/A"
        lines.append(f"| {cat} | {_fmt(p)} | {_fmt(c)} | {d_str} |")
    lines.append("")

    lines.append("## Régressions ≥ 3pt")
    lines.append("")
    if not regressions:
        lines.append("_Aucune._")
    else:
        for cat, p, c, d in regressions:
            lines.append(f"- ⚠️ **{cat}** : {_fmt(p)} → {_fmt(c)} ({d:+.2f}pt)")
    lines.append("")

    lines.append("## Gains ≥ 1pt")
    lines.append("")
    if not gains:
        lines.append("_Aucun._")
    else:
        for cat, p, c, d in gains:
            lines.append(f"- ✓ {cat} : {_fmt(p)} → {_fmt(c)} ({d:+.2f}pt)")
    lines.append("")

    # Verdict
    lines.append("## Verdict cycle")
    lines.append("")
    if delta_global >= 1.0 and not regressions:
        lines.append(f"**GO cycle suivant** — gain global {delta_global:+.2f}pt, "
                     "aucune régression > 3pt.")
    elif delta_global >= 1.0 and regressions:
        lines.append(f"**ALERTE** — gain global {delta_global:+.2f}pt mais "
                     f"{len(regressions)} catégorie(s) régressent > 3pt. "
                     "Retour Julien obligatoire avant cycle suivant.")
    elif delta_global < 1.0:
        lines.append(f"**STOP boucle** — gain global {delta_global:+.2f}pt < 1pt. "
                     "Le concept ne progresse pas (ou plus). À documenter.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {args.output}")
    print()
    print("\n".join(lines[:30]))


if __name__ == "__main__":
    main()
