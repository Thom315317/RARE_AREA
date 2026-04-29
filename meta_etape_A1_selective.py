#!/usr/bin/env python3
"""Phase A.1 — Selective prediction.

Pour chaque exemple meta-test, le meta-modèle produit P(error). Si > threshold,
on s'abstient ; sinon on prédit avec le modèle de base.
On mesure (coverage, risk) sur 10 thresholds, on calcule l'AURC et on compare
à la baseline triviale (toujours prédire).

Réutilise scores.json sortis par meta_train_etape3.py (single seed).
3 seeds : agréger les courbes.

Critère GO §A.2 :
  - À coverage 80%, risk ≤ 5% (vs 19% baseline)
  - AURC ≤ 0.05

Usage :
  python3 meta_etape_A1_selective.py \\
      --runs runs_meta/etape3_s42,runs_meta/etape3_s123,runs_meta/etape3_s456 \\
      --output-dir runs_meta/etapeA1
"""
import argparse
import json
import math
import os

import numpy as np


def _load_scores(d):
    with open(os.path.join(d, "scores.json"), "r") as f:
        sd = json.load(f)
    return {
        "yte": np.array(sd["yte"]),
        "scores": np.array(sd["s_enc_te"]),
        "cats": sd["cats_test"],
    }


def risk_coverage_curve(y_true, scores, n_points=99):
    """Pour chaque seuil, mesurer (coverage, risk).
    coverage = fraction d'exemples sur lesquels on prédit (score < threshold)
    risk     = taux d'erreur sur ces exemples = mean(y_true | score < threshold)
    """
    thresholds = np.linspace(0.01, 0.99, n_points)
    cov_list = []; risk_list = []
    for t in thresholds:
        keep = (scores < t)
        n_keep = int(keep.sum())
        coverage = n_keep / max(len(y_true), 1)
        if n_keep > 0:
            risk = float(y_true[keep].mean())
        else:
            risk = float("nan")
        cov_list.append(coverage)
        risk_list.append(risk)
    return thresholds, np.array(cov_list), np.array(risk_list)


def aurc(coverages, risks):
    """Area under risk-coverage. On trie par coverage croissant et on intègre."""
    # Filter out NaN risks
    valid = ~np.isnan(risks)
    cov = coverages[valid]
    rk = risks[valid]
    # Sort by coverage
    order = np.argsort(cov)
    cov_s = cov[order]; rk_s = rk[order]
    # Trapezoidal
    if len(cov_s) < 2:
        return float("nan")
    trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if trap is None:
        # Manual trapezoidal
        return float(0.5 * np.sum((cov_s[1:] - cov_s[:-1]) * (rk_s[1:] + rk_s[:-1])))
    return float(trap(rk_s, cov_s))


def risk_at_coverage(coverages, risks, target=0.8):
    """Interpolate risk at a target coverage."""
    valid = ~np.isnan(risks)
    cov = coverages[valid]
    rk = risks[valid]
    order = np.argsort(cov)
    cov_s = cov[order]; rk_s = rk[order]
    if not len(cov_s):
        return float("nan")
    return float(np.interp(target, cov_s, rk_s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="Comma-separated dirs containing scores.json")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    run_dirs = [d.strip() for d in args.runs.split(",")]
    seeds_data = [_load_scores(d) for d in run_dirs]
    print(f"Loaded {len(seeds_data)} seeds")

    # Per-seed risk-coverage curves
    seed_results = []
    for i, sd in enumerate(seeds_data):
        thr, cov, rk = risk_coverage_curve(sd["yte"], sd["scores"])
        a = aurc(cov, rk)
        risk_80 = risk_at_coverage(cov, rk, 0.8)
        baseline_risk = float(sd["yte"].mean())  # taux d'erreur global
        seed_results.append({
            "seed": run_dirs[i],
            "thresholds": thr.tolist(),
            "coverages": cov.tolist(),
            "risks": rk.tolist(),
            "aurc": a,
            "risk_at_coverage_80": risk_80,
            "baseline_risk": baseline_risk,
        })
        print(f"  {run_dirs[i]} : AURC={a:.4f}  risk@cov=0.80={risk_80:.4f}  "
              f"baseline_risk={baseline_risk:.4f}")

    # Aggregate
    aurc_mean = float(np.mean([r["aurc"] for r in seed_results]))
    aurc_std = float(np.std([r["aurc"] for r in seed_results]))
    risk80_mean = float(np.mean([r["risk_at_coverage_80"] for r in seed_results]))
    risk80_std = float(np.std([r["risk_at_coverage_80"] for r in seed_results]))
    baseline_mean = float(np.mean([r["baseline_risk"] for r in seed_results]))

    # ── Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        for r in seed_results:
            cov = np.array(r["coverages"]); rk = np.array(r["risks"])
            valid = ~np.isnan(rk)
            order = np.argsort(cov[valid])
            ax.plot(cov[valid][order], rk[valid][order],
                    alpha=0.7, label=os.path.basename(r["seed"]))
        ax.axhline(baseline_mean, color="red", linestyle="--",
                   label=f"baseline (toujours prédire) = {baseline_mean:.3f}")
        ax.axvline(0.8, color="gray", linestyle=":", alpha=0.5,
                   label="coverage = 0.8")
        ax.axhline(0.05, color="green", linestyle=":", alpha=0.5,
                   label="cible risk = 0.05")
        ax.set_xlabel("Coverage (fraction d'exemples prédits)")
        ax.set_ylabel("Risk (taux d'erreur conditionnel)")
        ax.set_title("Phase A.1 — Risk-Coverage (selective prediction)")
        ax.set_xlim(0, 1); ax.set_ylim(0, max(0.30, baseline_mean * 1.2))
        ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "selective_prediction_curve.png"),
                    dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"plot skipped: {e}")

    aurc_path = os.path.join(args.output_dir, "aurc.json")
    with open(aurc_path, "w") as f:
        json.dump({
            "aurc_mean": aurc_mean, "aurc_std": aurc_std,
            "risk_at_coverage_80_mean": risk80_mean,
            "risk_at_coverage_80_std": risk80_std,
            "baseline_risk_mean": baseline_mean,
            "per_seed": seed_results,
        }, f, indent=2)
    print(f"Wrote {aurc_path}")

    # ── Verdict
    vlines = ["# Verdict Phase A.1 — Selective prediction", ""]
    vlines.append(f"- Risk baseline (toujours prédire) : **{baseline_mean:.4f}**")
    vlines.append(f"- AURC (mean ± std)               : **{aurc_mean:.4f} ± {aurc_std:.4f}**")
    vlines.append(f"- Risk @ coverage = 0.80          : **{risk80_mean:.4f} ± {risk80_std:.4f}**")
    vlines.append("")
    vlines.append("## Critères GO")
    vlines.append("")
    vlines.append(f"- Cible risk @ cov 0.80 ≤ 0.05 : "
                  f"{'✓' if risk80_mean <= 0.05 else '✗'} "
                  f"(obtenu {risk80_mean:.4f})")
    vlines.append(f"- Cible AURC ≤ 0.05 : "
                  f"{'✓' if aurc_mean <= 0.05 else '✗'} "
                  f"(obtenu {aurc_mean:.4f})")
    vlines.append("")
    if risk80_mean <= 0.05 and aurc_mean <= 0.05:
        vlines.append("**GO Phase A.2** — selective prediction efficace. "
                      "On peut atteindre 80% de coverage avec ≤5% de risk "
                      f"(vs {100*baseline_mean:.0f}% baseline).")
    elif risk80_mean <= 0.10:
        vlines.append("**PARTIEL** — gain net mais sous la cible. Documenter, "
                      "à valider avec Julien si on enchaîne A.2.")
    else:
        vlines.append("**NO-GO Phase A.1** — le meta-modèle ne discrimine pas assez "
                      "pour une selective prediction utile. Investigation requise.")
    vlines.append("")
    vlines.append("Détails dans `aurc.json` et courbe dans `selective_prediction_curve.png`.")
    vp = os.path.join(args.output_dir, "verdict_A1.md")
    with open(vp, "w") as f:
        f.write("\n".join(vlines))
    print(f"Wrote {vp}")
    print()
    print("\n".join(vlines))


if __name__ == "__main__":
    main()
