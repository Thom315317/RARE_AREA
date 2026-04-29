#!/usr/bin/env python3
"""Diagnostic SLOG — agrège P2 + D1 + D2 et écrit le verdict.

Lit les baseline_sanity.json des 3 runs, les compare au baseline non-JEPA
(30.14%) et applique la grille du spec §6.

Usage :
  python3 diag_slog_summary.py \\
      --p2-run 'runs/B4_slog_jepa_p2_epochs90/B4_s42_*' \\
      --d1-run 'runs/B4_slog_jepa_diag_H1_smaller_aug/B4_s42_*' \\
      --d2-run 'runs/B4_slog_jepa_diag_H2_no_tfr/B4_s42_*' \\
      --output runs/diag_slog_summary.md
"""
import argparse
import glob
import json
import os


# Toutes les valeurs gen_greedy / best_gen_tf dans baseline_sanity.json sont
# déjà en POURCENTAGE (ex: 26.86 pour 26.86%), pas en fraction.
REF_NON_JEPA = 30.14  # référence SLOG sans JEPA, gen greedy (%)
THR_OK = 29.0         # critère de validation H1/H2 (%)


def _load_sanity(pattern):
    matches = glob.glob(os.path.join(pattern, "baseline_sanity.json"))
    if not matches:
        return None
    with open(matches[0], "r") as f:
        return json.load(f)


def _fmt_pct(v):
    if v is None:
        return "N/A"
    return f"{v:.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2-run", required=True)
    ap.add_argument("--d1-run", required=True)
    ap.add_argument("--d2-run", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    p2 = _load_sanity(args.p2_run)
    d1 = _load_sanity(args.d1_run)
    d2 = _load_sanity(args.d2_run)

    def _gg(d):
        return d.get("gen_greedy") if d else None
    def _bgt(d):
        return d.get("best_gen_tf") if d else None
    def _jl(d):
        return d.get("jepa_loss_final") if d else None
    def _delta(d):
        gg = _gg(d)
        if gg is None or _gg(p2) is None:
            return None
        return gg - _gg(p2)

    lines = ["# Diagnostic dégradation SLOG B4+JEPA", "",
             "| Run | Modification | gen greedy | best_gen_tf | "
             "jepa_loss_final | Δ vs P2 |",
             "|---|---|---:|---:|---:|---:|",
             f"| Référence non-JEPA | (pas de JEPA) | "
             f"{_fmt_pct(REF_NON_JEPA)} | ~31% | n/a | référence |"]

    p2_gg = _gg(p2)
    p2_str = "P2 (baseline JEPA)"
    if p2:
        p2_jl = _jl(p2)
        p2_jl_str = f"{p2_jl:.4f}" if p2_jl is not None else "N/A"
        delta_p2 = (p2_gg - REF_NON_JEPA) if p2_gg is not None else None
        delta_str = f"{delta_p2:+.2f}pt" if delta_p2 is not None else "N/A"
        lines.append(f"| {p2_str} | aucune | {_fmt_pct(p2_gg)} | "
                     f"{_fmt_pct(_bgt(p2))} | {p2_jl_str} | {delta_str} |")
    else:
        lines.append(f"| {p2_str} | aucune | N/A | N/A | N/A | N/A |")

    for tag, d in [("D1", d1), ("D2", d2)]:
        if d is None:
            lines.append(f"| {tag} | (run absent) | N/A | N/A | N/A | N/A |")
            continue
        mod = ("augmentations réduites" if tag == "D1"
               else "sans scheduled sampling")
        delta = _delta(d)
        delta_str = f"{delta:+.2f}pt" if delta is not None else "N/A"
        jl = _jl(d)
        jl_str = f"{jl:.4f}" if jl is not None else "N/A"
        lines.append(f"| {tag} | {mod} | {_fmt_pct(_gg(d))} | "
                     f"{_fmt_pct(_bgt(d))} | {jl_str} | {delta_str} |")

    lines.append("")
    lines.append("## Verdict")
    lines.append("")

    d1_ok = (_gg(d1) is not None and _gg(d1) >= THR_OK)
    d2_ok = (_gg(d2) is not None and _gg(d2) >= THR_OK)
    d1_gg = _gg(d1); d2_gg = _gg(d2)

    if d1_ok and d2_ok:
        lines.append("**H1 ET H2 contribuent.** Les deux interventions "
                     "(augmentations réduites, sans scheduled sampling) "
                     "ramènent gen greedy ≥ 29%. Le baseline propre serait "
                     "la combinaison des deux. Décision Julien.")
    elif d1_ok and not d2_ok:
        lines.append(f"**H1 validée** (D1 = {_fmt_pct(d1_gg)} ≥ 29%, "
                     f"D2 = {_fmt_pct(d2_gg)}). Les augmentations massives "
                     "perturbent le JEPA. Le scheduled sampling n'est pas "
                     "(seul) en cause.")
    elif d2_ok and not d1_ok:
        lines.append(f"**H2 validée** (D2 = {_fmt_pct(d2_gg)} ≥ 29%, "
                     f"D1 = {_fmt_pct(d1_gg)}). Le scheduled sampling "
                     "perturbe le JEPA. Les augmentations massives ne sont "
                     "pas (seules) en cause.")
    else:
        lines.append(f"**Ni H1 ni H2 explication principale** "
                     f"(D1 = {_fmt_pct(d1_gg)}, D2 = {_fmt_pct(d2_gg)}, "
                     "tous deux < 29%). H3 (conflit task/JEPA intrinsèque sur "
                     "SLOG) probable. À documenter comme limite intrinsèque.")
    lines.append("")
    lines.append("**Détail :** P2 baseline = "
                 f"{_fmt_pct(p2_gg)} ; référence non-JEPA = "
                 f"{_fmt_pct(REF_NON_JEPA)} ; cible diagnostic = ≥29%.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {args.output}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
