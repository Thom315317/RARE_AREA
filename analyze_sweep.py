#!/usr/bin/env python3
"""SLOG sweep analysis — Julien avril 2026.

Reads the 6 summary.json files produced by sweep_slog.py and writes:
  - results.md      aggregated table + per-category table with paired t-test
  - curves.png      gen_ex per epoch overlays (2 colours × 3 seeds)
  - red_flags.md    automated checklist
  - verdict.md      5-10-line synthesis driven by the criteria in the spec

Usage: python3 analyze_sweep.py
"""
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(HERE, "runs_sweep_slog")

CONFIG_BASE = {
    "A": "A_baseline_pool130",
    "B": "B_pool120_filtered",
}
SEEDS = [42, 123, 456]

# t_{2, 0.025} for 95% CI with n=3
T_CRIT_N3 = 4.303


def _find_summary(config, seed):
    """summary.json lives in runs_sweep_slog/{base}/s{seed}/{variant}_s{seed}_TS/summary.json"""
    base = CONFIG_BASE[config]
    pattern = os.path.join(SWEEP_DIR, base, f"s{seed}", "*", "summary.json")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def _load_metrics(config, seed):
    """metrics.json sits next to summary.json — used for epoch curves."""
    s = _find_summary(config, seed)
    if s is None:
        return None
    m_path = os.path.join(os.path.dirname(s), "metrics.json")
    if not os.path.exists(m_path):
        return None
    with open(m_path, "r") as f:
        return json.load(f)


def _load_summary(config, seed):
    s = _find_summary(config, seed)
    if s is None:
        return None
    with open(s, "r") as f:
        return json.load(f)


def _mean_std(xs):
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def _ci95_n3(xs):
    m, sd = _mean_std(xs)
    if m is None:
        return None, None
    return m, T_CRIT_N3 * sd / math.sqrt(len(xs))


def _paired_t(xs_a, xs_b):
    """Paired t-test on equal-length lists. Returns (t, p_two_sided)."""
    if len(xs_a) != len(xs_b) or len(xs_a) < 2:
        return None, None
    diffs = [a - b for a, b in zip(xs_a, xs_b)]
    m, sd = _mean_std(diffs)
    if sd == 0 or sd is None:
        return float("inf") if (m or 0) != 0 else 0.0, 1.0
    t = m / (sd / math.sqrt(len(diffs)))
    # Two-sided p via the absolute value. We do NOT import scipy to keep deps
    # light — an approximation from Abramowitz & Stegun (7.1.26) for the
    # Student-t CDF with df=n-1 would be overkill; instead we use a simple
    # small-df heuristic: with n=3 (df=2), report the t value and let the
    # reader compare to the critical values (2.920 for p=0.1, 4.303 for
    # p=0.05, 9.925 for p=0.01).
    if abs(t) >= 9.925:
        p = 0.01
    elif abs(t) >= 4.303:
        p = 0.05
    elif abs(t) >= 2.920:
        p = 0.1
    else:
        p = 0.2
    return t, p


def _sig_marker(p):
    if p is None:
        return ""
    if p <= 0.01:
        return "**"
    if p <= 0.05:
        return "*"
    return ""


def aggregate():
    """Gather gen/dev per seed per config."""
    data = {c: {"gen": [], "dev": [], "by_cat": {}, "seeds": []}
            for c in ("A", "B")}
    for c in ("A", "B"):
        for s in SEEDS:
            summ = _load_summary(c, s)
            if summ is None:
                continue
            data[c]["seeds"].append(s)
            data[c]["gen"].append(summ.get("final_gen_exact_greedy", 0.0))
            data[c]["dev"].append(summ.get("final_dev_exact_greedy", 0.0))
            cats = summ.get("greedy_gen_by_cat", {}) or {}
            for cat, val in cats.items():
                data[c]["by_cat"].setdefault(cat, []).append((s, val))
    return data


def write_results(data):
    a_gen_mean, a_gen_ci = _ci95_n3(data["A"]["gen"])
    b_gen_mean, b_gen_ci = _ci95_n3(data["B"]["gen"])
    a_dev_mean, a_dev_ci = _ci95_n3(data["A"]["dev"])
    b_dev_mean, b_dev_ci = _ci95_n3(data["B"]["dev"])
    t_gen, p_gen = _paired_t(data["B"]["gen"], data["A"]["gen"])
    delta_gen = ((b_gen_mean - a_gen_mean)
                 if (a_gen_mean is not None and b_gen_mean is not None) else None)

    lines = [
        "# SLOG sweep results (3 seeds)",
        "",
        f"Sweep dir: `{SWEEP_DIR}`",
        "",
        "## Aggregated gen / dev",
        "",
        "| Config | Gen greedy (mean ± IC95) | Dev greedy (mean ± IC95) | Seeds |",
        "|---|---|---|---|",
    ]
    for c in ("A", "B"):
        label = "A — pool 130 (baseline)" if c == "A" else "B — pool 120 (filtered)"
        gm, gci = _ci95_n3(data[c]["gen"])
        dm, dci = _ci95_n3(data[c]["dev"])
        gs = f"{gm:.2f} ± {gci:.2f}" if gm is not None else "—"
        ds = f"{dm:.2f} ± {dci:.2f}" if dm is not None else "—"
        seeds = ", ".join(str(s) for s in data[c]["seeds"])
        lines.append(f"| {label} | {gs} | {ds} | {seeds} |")
    lines.append("")

    if delta_gen is not None and t_gen is not None:
        lines.append(f"**Delta B − A :** {delta_gen:+.2f} points "
                     f"(t = {t_gen:+.2f}, p ≈ {p_gen}, paired t-test over 3 seeds)")
    lines.append("")

    # Per-category table. Only categories present in BOTH configs with same
    # number of seeds are compared.
    all_cats = sorted(
        set(data["A"]["by_cat"].keys()) | set(data["B"]["by_cat"].keys()))
    lines.extend([
        "## Per-category (paired, over 3 seeds)",
        "",
        "| Catégorie | A mean ± IC | B mean ± IC | Delta | p (paired) |",
        "|---|---|---|---|---|",
    ])
    for cat in all_cats:
        a_vals = [v for (_s, v) in data["A"]["by_cat"].get(cat, [])]
        b_vals = [v for (_s, v) in data["B"]["by_cat"].get(cat, [])]
        am, aci = _ci95_n3(a_vals)
        bm, bci = _ci95_n3(b_vals)
        # Pair B/A only on shared seeds
        a_by_seed = dict(data["A"]["by_cat"].get(cat, []))
        b_by_seed = dict(data["B"]["by_cat"].get(cat, []))
        shared = sorted(set(a_by_seed) & set(b_by_seed))
        paired_a = [a_by_seed[s] for s in shared]
        paired_b = [b_by_seed[s] for s in shared]
        delta = (bm - am) if (am is not None and bm is not None) else None
        t, p = _paired_t(paired_b, paired_a)
        a_s = f"{am:.2f} ± {aci:.2f}" if am is not None else "—"
        b_s = f"{bm:.2f} ± {bci:.2f}" if bm is not None else "—"
        d_s = f"{delta:+.2f}" if delta is not None else "—"
        p_s = (f"≈{p} {_sig_marker(p)}" if p is not None else "—")
        lines.append(f"| {cat} | {a_s} | {b_s} | {d_s} | {p_s} |")
    lines.extend([
        "",
        "Note : p est estimé par bucket (0.01 / 0.05 / 0.1 / 0.2) via les valeurs "
        "critiques de t_{df=2}. Pas de correction Bonferroni — ce n'est pas un "
        "test de découverte multiple mais une grille de lecture.",
    ])
    out = os.path.join(SWEEP_DIR, "results.md")
    os.makedirs(SWEEP_DIR, exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {out}")
    return (a_gen_mean, b_gen_mean, delta_gen, t_gen, p_gen)


def write_curves(data):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  SKIP curves.png (matplotlib missing: {e})")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    opacities = [1.0, 0.7, 0.5]
    colors = {"A": "tab:red", "B": "tab:blue"}
    for c in ("A", "B"):
        for i, s in enumerate(SEEDS):
            m = _load_metrics(c, s)
            if m is None:
                continue
            xs = [e.get("epoch", idx) for idx, e in enumerate(m)]
            ys = [e.get("gen", {}).get("exact", 0.0) for e in m]
            ax.plot(xs, ys, color=colors[c], alpha=opacities[i],
                    label=f"{c}/s{s}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("gen_ex (%)")
    ax.set_title("SLOG sweep — gen_ex per epoch, 2 configs × 3 seeds")
    ax.legend(loc="lower right", ncols=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    out = os.path.join(SWEEP_DIR, "curves.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


def write_red_flags(data):
    gen_a = data["A"]["gen"]
    gen_b = data["B"]["gen"]
    dev_a = data["A"]["dev"]
    dev_b = data["B"]["dev"]
    _, sd_a = _mean_std(gen_a)
    _, sd_b = _mean_std(gen_b)
    var_a = (sd_a or 0) > 3
    var_b = (sd_b or 0) > 3

    non_monotone = False
    if len(gen_a) == len(gen_b) and len(gen_a) >= 2:
        signs = [(b - a) for a, b in zip(gen_a, gen_b)]
        non_monotone = (min(signs) < 0 < max(signs))

    dev_low = any(d < 94.0 for d in dev_a + dev_b)

    # Per-category regression (>3 pts drop B vs A on a shared category)
    regression_cats = []
    shared_cats = set(data["A"]["by_cat"]) & set(data["B"]["by_cat"])
    for cat in sorted(shared_cats):
        a_by = dict(data["A"]["by_cat"][cat])
        b_by = dict(data["B"]["by_cat"][cat])
        common = sorted(set(a_by) & set(b_by))
        if not common:
            continue
        a_m = sum(a_by[s] for s in common) / len(common)
        b_m = sum(b_by[s] for s in common) / len(common)
        if b_m - a_m < -3.0:
            regression_cats.append((cat, b_m - a_m))

    def _chk(ok):
        return "[x]" if ok else "[ ]"

    lines = [
        "# SLOG sweep red flags",
        "",
        "Auto-cochée à partir des 6 runs :",
        "",
        f"- {_chk(var_a)} Variance A > 3 points sur gen greedy "
        f"(sd = {sd_a:.2f} pts)" if sd_a is not None
        else "- [ ] Variance A — non calculée (manque de seeds)",
        f"- {_chk(var_b)} Variance B > 3 points sur gen greedy "
        f"(sd = {sd_b:.2f} pts)" if sd_b is not None
        else "- [ ] Variance B — non calculée (manque de seeds)",
        f"- {_chk(non_monotone)} Corrélation non-monotone "
        "(par exemple B42 > A42 mais B123 < A123)",
        f"- {_chk(dev_low)} Dev greedy < 94 % sur ≥ 1 run (sous-apprentissage)",
        f"- {_chk(bool(regression_cats))} Régression > 3 pts sur ≥ 1 catégorie "
        f"non ciblée :",
    ]
    if regression_cats:
        for cat, delta in regression_cats:
            lines.append(f"    - `{cat}` : {delta:+.2f}")
    out = os.path.join(SWEEP_DIR, "red_flags.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {out}")


def write_verdict(gen_stats):
    a_mean, b_mean, delta, t, p = gen_stats
    lines = ["# Verdict", ""]
    if delta is None or t is None:
        lines.extend([
            "Données incomplètes — au moins un des 6 runs n'a pas de summary.json.",
            "Relancer le sweep avec `--resume` pour combler les trous.",
        ])
    else:
        abs_p = p if p is not None else 1.0
        if delta > 1.0 and abs_p <= 0.05:
            lines.append(f"**Effet confirmé.** B − A = {delta:+.2f} pts, "
                         f"p ≈ {abs_p}. Le pool filtré améliore SLOG gen "
                         "greedy de manière consistente sur 3 seeds.")
            lines.append("")
            lines.append("**Action recommandée :** tester le pool filtré "
                         "sur COGS (transfert du mécanisme).")
        elif delta > 0.5 and abs_p <= 0.1:
            lines.append(f"**Effet marginal.** B − A = {delta:+.2f} pts, "
                         f"p ≈ {abs_p}. Dans la zone de gain modeste.")
            lines.append("")
            lines.append("**Action recommandée :** ajouter les seeds 789 et "
                         "1337 pour lever l'ambiguïté.")
        elif abs(delta) < 0.5:
            lines.append(f"**Effet nul.** B − A = {delta:+.2f} pts, "
                         f"p ≈ {abs_p}. Pas d'effet détectable au seuil "
                         "que les 3 seeds permettent.")
            lines.append("")
            lines.append("**Action recommandée :** abandonner le filtrage "
                         "du pool, revenir à la config pool 130.")
        elif delta < 0:
            lines.append(f"**Régression.** B − A = {delta:+.2f} pts, "
                         f"p ≈ {abs_p}. Le filtrage du pool dégrade.")
            lines.append("")
            lines.append("**Action recommandée :** investiguer avant toute "
                         "autre décision.")
        else:
            lines.append(f"**Zone grise.** B − A = {delta:+.2f} pts, "
                         f"p ≈ {abs_p}.")
            lines.append("")
            lines.append("**Action recommandée :** ajouter les seeds 789 et "
                         "1337 et relire results.md.")
    out = os.path.join(SWEEP_DIR, "verdict.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {out}")


def main():
    if not os.path.isdir(SWEEP_DIR):
        print(f"No sweep directory found at {SWEEP_DIR}. Run sweep_slog.py first.",
              file=sys.stderr)
        sys.exit(1)
    data = aggregate()
    n_a = len(data["A"]["gen"])
    n_b = len(data["B"]["gen"])
    print(f"Loaded A:{n_a} runs, B:{n_b} runs.")
    gen_stats = write_results(data)
    write_curves(data)
    write_red_flags(data)
    write_verdict(gen_stats)
    print("Done.")


if __name__ == "__main__":
    main()
