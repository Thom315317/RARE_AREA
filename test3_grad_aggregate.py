#!/usr/bin/env python3
"""TEST 3 — Aggrégateur des snapshots de gradient COGS vs SLOG.

Lit les JSON de snapshots, calcule P_neg = % steps avec cos_sim < -0.3,
applique les critères pré-engagés.

CONVENTION : cos_sim ∈ [-1, +1]. P_neg est en pourcentage (0-100).

Usage :
  python3 test3_grad_aggregate.py \\
      --cogs-glob 'tests/test3_grad_conflict/cogs_s{seed}_grad.json' \\
      --slog-glob 'tests/test3_grad_conflict/slog_s{seed}_grad.json' \\
      --seeds 42,123,456 \\
      --output-dir tests/test3_grad_conflict
"""
import argparse
import glob
import json
import math
import os

import numpy as np


def _load_seeds(pattern, seeds):
    out = {}
    for s in seeds:
        p = pattern.replace("{seed}", str(s))
        matches = sorted(glob.glob(p))
        if not matches:
            continue
        with open(matches[0], "r") as f:
            out[s] = json.load(f)
    return out


def _p_neg(snapshots, threshold=-0.3):
    """% snapshots avec cos_sim < threshold (en pourcentage 0-100)."""
    if not snapshots:
        return float("nan")
    n_neg = sum(1 for s in snapshots if s.get("cos_sim", 0) < threshold)
    return 100.0 * n_neg / len(snapshots)


def _stats_across_seeds(per_seed_pneg):
    vals = [v for v in per_seed_pneg.values() if not math.isnan(v)]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cogs-glob", required=True)
    ap.add_argument("--slog-glob", required=True)
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",")]
    cogs_seeds = _load_seeds(args.cogs_glob, seeds)
    slog_seeds = _load_seeds(args.slog_glob, seeds)
    print(f"Loaded COGS: {list(cogs_seeds.keys())}, "
          f"SLOG: {list(slog_seeds.keys())}")

    pneg_cogs = {s: _p_neg(snaps) for s, snaps in cogs_seeds.items()}
    pneg_slog = {s: _p_neg(snaps) for s, snaps in slog_seeds.items()}
    cogs_mean, cogs_std = _stats_across_seeds(pneg_cogs)
    slog_mean, slog_std = _stats_across_seeds(pneg_slog)

    # P_neg(SLOG) - P_neg(COGS) en points de pourcentage
    diff_pp = slog_mean - cogs_mean if not (math.isnan(slog_mean) or math.isnan(cogs_mean)) else float("nan")

    # Cos_sim distribution stats per seed
    def _distrib(snaps):
        if not snaps:
            return {}
        cos = [s.get("cos_sim", 0) for s in snaps]
        return {
            "mean": float(np.mean(cos)),
            "median": float(np.median(cos)),
            "min": float(np.min(cos)),
            "max": float(np.max(cos)),
            "n_snapshots": len(cos),
        }

    cogs_distrib = {s: _distrib(snaps) for s, snaps in cogs_seeds.items()}
    slog_distrib = {s: _distrib(snaps) for s, snaps in slog_seeds.items()}

    # Verdict
    if math.isnan(diff_pp):
        verdict = "REPORTÉ"
        reason = "Données manquantes."
    elif diff_pp > 15.0:
        verdict = "Conflit confirmé"
        reason = (f"P_neg(SLOG) − P_neg(COGS) = {diff_pp:+.2f}pp > 15pp. "
                  "Refonte JEPA = post-hoc gelé OBLIGATOIRE pour SLOG.")
    elif diff_pp > 5.0:
        verdict = "Conflit modéré"
        reason = (f"P_neg(SLOG) − P_neg(COGS) = {diff_pp:+.2f}pp ∈ ]5, 15]pp. "
                  "Post-hoc préférable mais intégré + scheduler λ + grad clipping séparé reste viable.")
    else:
        verdict = "Pas de conflit gradient"
        reason = (f"P_neg(SLOG) − P_neg(COGS) = {diff_pp:+.2f}pp ≤ 5pp. "
                  "La dégradation SLOG vient d'ailleurs (probable : capacité "
                  "predictor JEPA insuffisante pour les patterns SLOG).")

    summary = {
        "seeds_analyzed": seeds,
        "p_neg_cogs_per_seed": pneg_cogs,
        "p_neg_slog_per_seed": pneg_slog,
        "p_neg_cogs_mean": cogs_mean,
        "p_neg_cogs_std": cogs_std,
        "p_neg_slog_mean": slog_mean,
        "p_neg_slog_std": slog_std,
        "diff_slog_minus_cogs_pp": diff_pp,
        "cos_sim_distribution_cogs": cogs_distrib,
        "cos_sim_distribution_slog": slog_distrib,
        "verdict": verdict,
        "reason": reason,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── Histograms PNG
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, label, seeds_data in [(axes[0], "COGS", cogs_seeds),
                                        (axes[1], "SLOG", slog_seeds)]:
            all_cos = []
            for s, snaps in seeds_data.items():
                all_cos.extend([sn.get("cos_sim", 0) for sn in snaps])
            if all_cos:
                ax.hist(all_cos, bins=40, alpha=0.7, edgecolor="black")
                ax.axvline(-0.3, color="red", linestyle="--",
                           label="seuil -0.3 (antagonisme fort)")
                ax.axvline(0.0, color="gray", linestyle=":", alpha=0.5)
            ax.set_xlabel("cos_sim(grad_task, grad_jepa)")
            ax.set_ylabel("Nombre de snapshots")
            ax.set_title(f"{label} — distribution cos_sim")
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "histograms.png"), dpi=120)
        plt.close(fig)
        print(f"Wrote histograms.png")
    except Exception as e:
        print(f"histograms.png skipped: {e}")

    # ── verdict.txt
    lines = ["# TEST 3 — Diagnostic conflit gradient task/JEPA", ""]
    lines.append(f"## Verdict : {verdict}")
    lines.append("")
    lines.append(reason)
    lines.append("")
    lines.append("## P_neg = % steps avec cos_sim < -0.3 (antagonisme fort)")
    lines.append("")
    lines.append("| Benchmark | seed 42 | seed 123 | seed 456 | mean | std |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    def _row(name, per_seed, mean, std):
        cells = []
        for s in seeds:
            v = per_seed.get(s)
            cells.append(f"{v:.2f}%" if v is not None and not math.isnan(v) else "N/A")
        return f"| {name} | " + " | ".join(cells) + f" | {mean:.2f}% | {std:.2f}% |"
    lines.append(_row("COGS", pneg_cogs, cogs_mean, cogs_std))
    lines.append(_row("SLOG", pneg_slog, slog_mean, slog_std))
    lines.append("")
    lines.append(f"**P_neg(SLOG) − P_neg(COGS) = {diff_pp:+.2f}pp**")
    lines.append("")
    lines.append("## Distribution cos_sim moyenne (par benchmark)")
    lines.append("")
    if cogs_distrib:
        means = [d["mean"] for d in cogs_distrib.values() if "mean" in d]
        if means:
            lines.append(f"- COGS cos_sim moyen sur tous snapshots : {np.mean(means):+.4f}")
    if slog_distrib:
        means = [d["mean"] for d in slog_distrib.values() if "mean" in d]
        if means:
            lines.append(f"- SLOG cos_sim moyen sur tous snapshots : {np.mean(means):+.4f}")
    lines.append("")
    lines.append("Détails complets : `summary.json`, `histograms.png`.")

    with open(os.path.join(args.output_dir, "verdict.txt"), "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {args.output_dir}/{{summary.json, verdict.txt, histograms.png}}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
