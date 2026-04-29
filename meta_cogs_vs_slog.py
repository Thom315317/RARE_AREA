#!/usr/bin/env python3
"""Réplication COGS → SLOG : tableau comparatif final.

Lit les sorties existantes (Phase B robust + A.1) sur COGS et SLOG, produit
`runs_meta/cogs_vs_slog.md` avec verdict de transférabilité.

Usage :
  python3 meta_cogs_vs_slog.py \\
      --cogs-robust runs_meta/etapeB_robust \\
      --cogs-a1     runs_meta/etapeA1_robust \\
      --slog-robust runs_meta/etape_slog_robust \\
      --slog-a1     runs_meta/etape_slog_A1 \\
      --output      runs_meta/cogs_vs_slog.md
"""
import argparse
import json
import math
import os

import numpy as np


def _load_seed_metrics(robust_dir):
    """Lit les seed_*/metrics.json d'un Phase B robust dir."""
    out = []
    for sub in sorted(os.listdir(robust_dir)):
        if not sub.startswith("seed_"):
            continue
        path = os.path.join(robust_dir, sub, "metrics.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                out.append(json.load(f))
    return out


def _load_a1(a1_dir):
    path = os.path.join(a1_dir, "aurc.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _agg_robust(seed_metrics):
    """Renvoie dict {auc_orig, auc_pert, auc_pert_avg_problem}."""
    if not seed_metrics:
        return {}
    a_o = [m.get("auc_orig_global", float("nan")) for m in seed_metrics]
    a_p = [m.get("auc_pert_global", float("nan")) for m in seed_metrics]
    a_l = [m.get("auc_loco_global", float("nan")) for m in seed_metrics]
    pert_avg = [m.get("auc_pert_avg_problem", float("nan")) for m in seed_metrics]
    def _m(arr):
        valid = [v for v in arr if not math.isnan(v)]
        return float(np.mean(valid)) if valid else float("nan")
    return {
        "auc_orig_global": _m(a_o),
        "auc_pert_global": _m(a_p),
        "auc_loco_global": _m(a_l),
        "auc_pert_avg_problem": _m(pert_avg),
        "n_seeds": len(seed_metrics),
    }


def _fmt(v, p=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.{p}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cogs-robust", required=True)
    ap.add_argument("--cogs-a1", required=True)
    ap.add_argument("--slog-robust", required=True)
    ap.add_argument("--slog-a1", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cogs_b = _agg_robust(_load_seed_metrics(args.cogs_robust))
    slog_b = _agg_robust(_load_seed_metrics(args.slog_robust))
    cogs_a1 = _load_a1(args.cogs_a1)
    slog_a1 = _load_a1(args.slog_a1)

    lines = ["# Comparaison réplication COGS vs SLOG", ""]
    lines.append("## Tableau global")
    lines.append("")
    lines.append("| Métrique | COGS | SLOG |")
    lines.append("|---|---:|---:|")
    lines.append(f"| AUC global meta (encoder robust)  | "
                 f"{_fmt(cogs_b.get('auc_orig_global'), 4)} | "
                 f"{_fmt(slog_b.get('auc_orig_global'), 4)} |")
    lines.append(f"| AUC perturbé global               | "
                 f"{_fmt(cogs_b.get('auc_pert_global'), 4)} | "
                 f"{_fmt(slog_b.get('auc_pert_global'), 4)} |")
    lines.append(f"| AUC perturbé sur cats prob (mean) | "
                 f"{_fmt(cogs_b.get('auc_pert_avg_problem'), 4)} | "
                 f"{_fmt(slog_b.get('auc_pert_avg_problem'), 4)} |")
    lines.append(f"| AUC LOCO global                   | "
                 f"{_fmt(cogs_b.get('auc_loco_global'), 4)} | "
                 f"{_fmt(slog_b.get('auc_loco_global'), 4)} |")
    if cogs_a1 and slog_a1:
        lines.append(f"| AURC selective prediction         | "
                     f"{_fmt(cogs_a1.get('aurc_mean'), 4)} | "
                     f"{_fmt(slog_a1.get('aurc_mean'), 4)} |")
        lines.append(f"| Risk @ cov 0.80                   | "
                     f"{_fmt(cogs_a1.get('risk_at_coverage_80_mean'), 4)} | "
                     f"{_fmt(slog_a1.get('risk_at_coverage_80_mean'), 4)} |")
        lines.append(f"| Baseline risk (toujours prédire)  | "
                     f"{_fmt(cogs_a1.get('baseline_risk_mean'), 4)} | "
                     f"{_fmt(slog_a1.get('baseline_risk_mean'), 4)} |")

    lines.append("")
    lines.append("## Verdict transférabilité")
    lines.append("")
    auc_global = slog_b.get("auc_orig_global", float("nan"))
    aurc = slog_a1.get("aurc_mean", float("nan")) if slog_a1 else float("nan")
    risk80 = slog_a1.get("risk_at_coverage_80_mean", float("nan")) if slog_a1 else float("nan")
    crit_auc = (not math.isnan(auc_global)) and auc_global >= 0.90
    crit_aurc = (not math.isnan(aurc)) and aurc <= 0.05
    crit_risk = (not math.isnan(risk80)) and risk80 <= 0.10
    lines.append(f"- AUC global meta SLOG ≥ 0.90 : "
                 f"{'✓' if crit_auc else '✗'} ({_fmt(auc_global, 3)})")
    lines.append(f"- AURC SLOG ≤ 0.05            : "
                 f"{'✓' if crit_aurc else '✗'} ({_fmt(aurc, 4)})")
    lines.append(f"- Risk @ cov 0.80 SLOG ≤ 0.10 : "
                 f"{'✓' if crit_risk else '✗'} ({_fmt(risk80, 4)})")
    lines.append("")
    if crit_auc and crit_aurc and crit_risk:
        verdict = ("**Transférabilité confirmée** — le pipeline COGS s'étend "
                   "sans réajustement majeur à SLOG. Le claim porte sur 2 "
                   "benchmarks.")
    elif (not math.isnan(auc_global)) and 0.85 <= auc_global < 0.90:
        verdict = ("**Transférabilité partielle** — dégradation notable. À "
                   "comprendre avant de revendiquer la généralité du résultat.")
    elif (not math.isnan(auc_global)) and auc_global < 0.85:
        verdict = ("**Non-transférabilité** — le pattern paraît COGS-spécifique. "
                   "Investiguer (features mal adaptées ? JEPA SLOG-spécifique ?).")
    else:
        verdict = "Verdict reporté — données insuffisantes."
    lines.append(verdict)
    lines.append("")

    with open(args.output, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {args.output}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
