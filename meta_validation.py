#!/usr/bin/env python3
"""Phase C — module de validation automatique.

Deux modes :

  Mode A (sanity-only)        : prend un metrics.json + previous_metrics.json
                                optionnel et flagge les anomalies
                                (AUC=1.000 sur cat ≥5 erreurs, std<0.0005,
                                sauts d'AUC > 0.30/0.50).

  Mode B (robustness)         : appelle meta_etape3_verification.py-style tests
                                (LOCO + perturbation) sur un dataset + scores
                                existants.

Sortie : `auto_alerts.md` listant chaque alerte avec son niveau
(CRITIQUE / AVERTISSEMENT / INFO).

Si has_critical_alerts(), on quitte avec exit code 2 — le script appelant peut
choisir de bloquer le pipeline.

Tests retrospectifs :
  python3 meta_validation.py \\
      --metrics runs_meta/etape3_s42/metrics.json \\
      --previous-metrics runs_meta/etape_bplus/metrics.json \\
      --output runs_meta/etape3_s42/auto_alerts.md
"""
import argparse
import json
import math
import os
import sys


CRITICAL = "CRITIQUE"
WARNING = "AVERTISSEMENT"
INFO = "INFO"


# ══════════════════════════════════════════════════════════════════════════
# Sanity report
# ══════════════════════════════════════════════════════════════════════════
class SanityReport:
    def __init__(self):
        self.alerts = []  # list of (level, code, message)

    def add(self, level, code, message):
        self.alerts.append({"level": level, "code": code, "message": message})

    def has_critical_alerts(self):
        return any(a["level"] == CRITICAL for a in self.alerts)

    def to_markdown(self, header="# auto_alerts"):
        lines = [header, ""]
        if not self.alerts:
            lines.append("_Aucune alerte._")
            return "\n".join(lines)
        # Group by level
        by_level = {CRITICAL: [], WARNING: [], INFO: []}
        for a in self.alerts:
            by_level[a["level"]].append(a)
        for level in (CRITICAL, WARNING, INFO):
            if not by_level[level]:
                continue
            marker = {"CRITIQUE": "🔴", "AVERTISSEMENT": "🟡", "INFO": "🔵"}[level]
            lines.append(f"## {marker} {level} ({len(by_level[level])})")
            lines.append("")
            for a in by_level[level]:
                lines.append(f"- **[{a['code']}]** {a['message']}")
            lines.append("")
        n_crit = len(by_level[CRITICAL])
        n_warn = len(by_level[WARNING])
        lines.append(f"**Total** : {n_crit} CRITIQUE, {n_warn} AVERTISSEMENT, "
                     f"{len(by_level[INFO])} INFO.")
        return "\n".join(lines)

    def save(self, path):
        with open(path, "w") as f:
            f.write(self.to_markdown())


# ══════════════════════════════════════════════════════════════════════════
# Helpers — extraire AUC par catégorie depuis un metrics dict
# ══════════════════════════════════════════════════════════════════════════
def _per_cat_aucs(metrics):
    """Renvoie {model_name: {cat: {auc, n_errors}}} depuis un metrics.json
    de meta_train_etape3.py ou meta_compare_bplus.py."""
    out = {}
    # Format meta_train_etape3 : metrics["per_category"][model][cat] = {auc, n, n_errors}
    if "per_category" in metrics and isinstance(metrics["per_category"], dict):
        for model, pc in metrics["per_category"].items():
            if not isinstance(pc, dict):
                continue
            out[model] = {}
            for cat, info in pc.items():
                if isinstance(info, dict):
                    out[model][cat] = {
                        "auc": info.get("auc", float("nan")),
                        "n_errors": info.get("n_errors", 0),
                        "n": info.get("n", 0),
                    }
    # Format meta_compare_bplus : metrics["bplus"]["per_category"][cat] = {auc_mean, auc_std, ...}
    for k in ("etape2", "bplus"):
        if k in metrics and "per_category" in metrics[k]:
            out[k] = {}
            for cat, info in metrics[k]["per_category"].items():
                if isinstance(info, dict):
                    out[k][cat] = {
                        "auc": info.get("auc_mean", float("nan")),
                        "auc_std": info.get("auc_std", 0.0),
                        "n_errors": int(info.get("n_errors_mean", 0)),
                        "n": info.get("n", 0),
                    }
    return out


def _global_aucs(metrics):
    """Renvoie {model_name: auc} depuis un metrics.json. Plusieurs schémas
    supportés."""
    out = {}
    # meta_train_etape3 : models = [{name, auc, ...}]
    if "models" in metrics and isinstance(metrics["models"], list):
        for m in metrics["models"]:
            if isinstance(m, dict) and "name" in m and "auc" in m:
                out[m["name"]] = m["auc"]
    # meta_compare_bplus
    if "etape2" in metrics and "auc_global_mean" in metrics["etape2"]:
        out["metamlp_etape2_global"] = metrics["etape2"]["auc_global_mean"]
    if "bplus" in metrics and "auc_global_mean" in metrics["bplus"]:
        out["metamlp_bplus_global"] = metrics["bplus"]["auc_global_mean"]
    return out


# ══════════════════════════════════════════════════════════════════════════
# Sanity checks
# ══════════════════════════════════════════════════════════════════════════
def run_sanity_checks(metrics, previous_metrics=None,
                      label="current"):
    """Vérifie les anomalies suspectes dans `metrics` (et compare à
    `previous_metrics` si fourni). Renvoie un SanityReport."""
    report = SanityReport()
    pc = _per_cat_aucs(metrics)
    g = _global_aucs(metrics)

    # ── 1. AUC = 1.000 sur catégorie avec n_errors >= 5 → CRITIQUE
    for model, cats in pc.items():
        for cat, info in cats.items():
            auc = info.get("auc", float("nan"))
            ne = info.get("n_errors", 0)
            if isinstance(auc, float) and not math.isnan(auc):
                if abs(auc - 1.0) < 1e-6 and ne >= 5:
                    report.add(CRITICAL, "AUC_1.000_with_errors",
                               f"{model}/{cat} : AUC = 1.000 exact avec "
                               f"n_errors = {ne} (≥5). Probable mémorisation.")
                elif abs(auc - 1.0) < 1e-6 and ne < 5:
                    report.add(WARNING, "AUC_1.000_few_errors",
                               f"{model}/{cat} : AUC = 1.000 exact mais "
                               f"n_errors = {ne} (<5). Faible interprétabilité.")

    # ── 2. AUC std < 0.0005 sur 3 seeds → CRITIQUE (variance suspecte)
    for model, cats in pc.items():
        for cat, info in cats.items():
            std = info.get("auc_std", None)
            if std is not None and std < 0.0005:
                # Mais seulement si l'AUC est élevée (pour éviter de flagger les NaN)
                auc = info.get("auc", float("nan"))
                if not math.isnan(auc) and auc > 0.5:
                    report.add(CRITICAL, "AUC_std_quasi_zero",
                               f"{model}/{cat} : AUC std = {std:.5f} (<0.0005). "
                               "Variance quasi-nulle entre seeds — suspect.")

    # ── 3. Saut > 0.30 sur AUC global entre étapes
    if previous_metrics is not None:
        g_prev = _global_aucs(previous_metrics)
        for model in g.keys() & g_prev.keys():
            d = g[model] - g_prev[model]
            if abs(d) > 0.30:
                report.add(CRITICAL, "AUC_global_jump",
                           f"{model} : saut AUC global = {d:+.3f} entre "
                           f"étapes (>0.30 absolu). Progression anormale.")
            elif abs(d) > 0.10:
                report.add(WARNING, "AUC_global_change_large",
                           f"{model} : changement AUC global = {d:+.3f} entre "
                           "étapes (>0.10).")

    # ── 4. Saut > 0.50 sur AUC par catégorie entre étapes
    if previous_metrics is not None:
        pc_prev = _per_cat_aucs(previous_metrics)
        # Match models that exist in both (or any model in prev compared to any in current)
        all_prev_models = list(pc_prev.keys())
        for model_cur, cats_cur in pc.items():
            for model_prev in all_prev_models:
                cats_prev = pc_prev[model_prev]
                for cat in set(cats_cur) & set(cats_prev):
                    auc_c = cats_cur[cat].get("auc", float("nan"))
                    auc_p = cats_prev[cat].get("auc", float("nan"))
                    if math.isnan(auc_c) or math.isnan(auc_p):
                        continue
                    d = auc_c - auc_p
                    if abs(d) > 0.50:
                        report.add(CRITICAL, "AUC_cat_jump",
                                   f"{cat} : saut AUC = {d:+.3f} entre "
                                   f"{model_prev} ({auc_p:.3f}) et "
                                   f"{model_cur} ({auc_c:.3f}). Saut > 0.50 absolu.")

    # ── 5. AUC global > 0.99 → AVERTISSEMENT (proche du parfait)
    for model, auc in g.items():
        if auc > 0.99:
            report.add(WARNING, "AUC_near_perfect",
                       f"{model} : AUC global = {auc:.4f} > 0.99. "
                       "À vérifier (mémorisation possible).")

    # ── 6. Catégories avec n_errors < 10 → INFO (peu fiable)
    for model, cats in pc.items():
        for cat, info in cats.items():
            ne = info.get("n_errors", 0)
            if 0 < ne < 10:
                auc = info.get("auc", float("nan"))
                if not math.isnan(auc):
                    report.add(INFO, "low_n_errors",
                               f"{model}/{cat} : n_errors = {ne} (<10). "
                               f"AUC = {auc:.3f} peu fiable.")

    return report


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True,
                    help="Path to current metrics.json")
    ap.add_argument("--previous-metrics", default=None,
                    help="Path to previous metrics.json for jump detection")
    ap.add_argument("--output", required=True,
                    help="Path to write auto_alerts.md")
    ap.add_argument("--exit-code-on-critical", action="store_true",
                    help="Exit with code 2 if any CRITIQUE alert (for pipeline blocking)")
    args = ap.parse_args()

    with open(args.metrics, "r") as f:
        metrics = json.load(f)
    previous = None
    if args.previous_metrics:
        with open(args.previous_metrics, "r") as f:
            previous = json.load(f)

    report = run_sanity_checks(metrics, previous_metrics=previous)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    report.save(args.output)
    print(f"Wrote {args.output}")
    print()
    print(report.to_markdown())

    if report.has_critical_alerts() and args.exit_code_on_critical:
        print("\n⚠️ ALERTES CRITIQUES détectées. Voir auto_alerts.md.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
