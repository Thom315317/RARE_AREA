#!/usr/bin/env python3
"""COGS transfer test analysis — Julien avril 2026.

Reads the 2 summary.json files produced by the transfer test
(runs_transfer_cogs/A_baseline/s42 and runs_transfer_cogs/B_pool_filtered/s42)
and writes:
  results.md   aggregated + per-category table (sorted by delta desc)
  verdict.md   5-10 line synthesis per the spec criteria

Usage:
  python3 analyze_transfer.py
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSFER_DIR = os.path.join(HERE, "runs_transfer_cogs")

CONFIG_BASE = {
    "A": os.path.join(TRANSFER_DIR, "A_baseline", "s42"),
    "B": os.path.join(TRANSFER_DIR, "B_pool_filtered", "s42"),
}


def _find_summary(config_dir):
    matches = sorted(glob.glob(os.path.join(config_dir, "*", "summary.json")))
    return matches[-1] if matches else None


def _load(config_dir):
    s = _find_summary(config_dir)
    if s is None:
        return None
    with open(s, "r") as f:
        return json.load(f)


def _load_pool_used(config_dir):
    matches = sorted(glob.glob(os.path.join(config_dir, "*", "pool_used.txt")))
    if not matches:
        return None
    with open(matches[-1], "r") as f:
        return f.read()


def main():
    if not os.path.isdir(TRANSFER_DIR):
        print(f"No transfer dir at {TRANSFER_DIR}. Run the 2 COGS configs first.",
              file=sys.stderr)
        sys.exit(1)

    a = _load(CONFIG_BASE["A"])
    b = _load(CONFIG_BASE["B"])
    if a is None or b is None:
        print("Missing summary.json for A or B. Runs incomplete.", file=sys.stderr)
        print(f"  A found: {a is not None}  B found: {b is not None}")
        sys.exit(1)

    a_gen = a.get("final_gen_exact_greedy")
    b_gen = b.get("final_gen_exact_greedy")
    a_dev = a.get("final_dev_exact_greedy")
    b_dev = b.get("final_dev_exact_greedy")
    delta_gen = (b_gen - a_gen) if (a_gen is not None and b_gen is not None) else None

    a_cats = a.get("greedy_gen_by_cat", {}) or {}
    b_cats = b.get("greedy_gen_by_cat", {}) or {}
    all_cats = sorted(set(a_cats.keys()) | set(b_cats.keys()))

    pool_used = _load_pool_used(CONFIG_BASE["B"]) or ""
    # Extract removed list roughly
    removed_line = [l for l in pool_used.splitlines() if "removed from pool" in l]
    removed_str = removed_line[0].split(":", 1)[1].strip() if removed_line else "—"

    # Results.md
    lines = [
        "# Transfert pool filtré SLOG → COGS (seed 42)",
        "",
        "## Agrégé",
        "",
        "| Config | Gen greedy | Dev greedy | Verbes retirés |",
        "|---|---|---|---|",
        f"| A — baseline | {a_gen:.2f}% | {a_dev:.2f}% | — |",
        f"| B — pool filtré | {b_gen:.2f}% | {b_dev:.2f}% | {removed_str} |",
        "",
        f"**Delta B − A :** {delta_gen:+.2f} points",
        "",
        "## Par catégorie (triée par delta décroissant)",
        "",
        "| Catégorie | A (%) | B (%) | Delta |",
        "|---|---|---|---|",
    ]
    rows = []
    for cat in all_cats:
        av = a_cats.get(cat)
        bv = b_cats.get(cat)
        if av is None or bv is None:
            continue
        rows.append((cat, av, bv, bv - av))
    rows.sort(key=lambda r: -r[3])
    for cat, av, bv, dv in rows:
        lines.append(f"| {cat} | {av:.2f} | {bv:.2f} | {dv:+.2f} |")
    if pool_used:
        lines.extend(["", "## pool_used.txt (config B)", "", "```",
                      pool_used.rstrip(), "```"])
    out_path = os.path.join(TRANSFER_DIR, "results.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {out_path}")

    # Verdict
    v_lines = ["# Verdict transfert COGS (seed 42)", ""]
    if delta_gen is None:
        v_lines.append("Données manquantes — un des deux runs n'a pas de summary.json.")
    else:
        if delta_gen > 1.5:
            v_lines.append(f"**Signal fort.** Delta = {delta_gen:+.2f} pts. "
                           "Le finding SLOG transfère sans adaptation.")
            v_lines.append("")
            v_lines.append("**Action recommandée :** sweep 3-seed COGS pour confirmer.")
        elif delta_gen > 0.5:
            v_lines.append(f"**Signal plausible.** Delta = {delta_gen:+.2f} pts, "
                           "cohérent avec l'amplitude SLOG (+1.6 pts).")
            v_lines.append("")
            v_lines.append("**Action recommandée :** sweep 3-seed COGS pour lever "
                           "l'ambiguïté due au seed unique.")
        elif delta_gen > -0.5:
            v_lines.append(f"**Pas de transfert détectable.** Delta = "
                           f"{delta_gen:+.2f} pts, dans la zone bruit.")
            v_lines.append("")
            v_lines.append("**Action recommandée :** documenter le résultat "
                           "négatif. Le finding reste SLOG-spécifique. "
                           "Ne pas engager plus de compute.")
        else:
            v_lines.append(f"**Transfert inversé.** Delta = {delta_gen:+.2f} pts "
                           "— le filtrage du pool DÉGRADE COGS.")
            v_lines.append("")
            v_lines.append("**Action recommandée :** investiguer. "
                           "Probable que le pool unaccusatif porte sur COGS des "
                           "informations utiles à d'autres catégories (notamment "
                           "unacc_to_transitive qui a été débloqué par shatter).")
    v_lines.append("")
    v_lines.append(f"Seed 42 — 1 run = bruit possible. Pas de revendication sans "
                   f"réplication 3-seed.")
    out_path = os.path.join(TRANSFER_DIR, "verdict.md")
    with open(out_path, "w") as f:
        f.write("\n".join(v_lines))
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
