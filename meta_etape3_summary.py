#!/usr/bin/env python3
"""Étape 3 — agrégation 3 seeds + comparaison + verdict.

Lit les métriques de N runs (chaque dossier `runs_meta/etape3_s{seed}/`)
et écrit :
  - etape3_summary.md       : tableau global + per-catégorie comparé à étape 2/B+
  - etape3_qualitative.md   : 10 exemples (5 vrais positifs + 5 faux négatifs sur cats problématiques)
  - etape3_verdict.md       : décision per spec §7

Usage :
  python3 meta_etape3_summary.py \\
      --runs runs_meta/etape3_s42,runs_meta/etape3_s123,runs_meta/etape3_s456 \\
      --output-dir runs_meta
"""
import argparse
import json
import math
import os

import numpy as np


PROBLEM_CATS = ["subj_to_obj_common", "passive_to_active",
                "unacc_to_transitive", "obj_omitted_transitive_to_transitive"]
NEAR_PERFECT = ["active_to_passive", "obj_to_subj_proper",
                "only_seen_as_transitive_subj_as_unacc_subj",
                "prim_to_subj_proper"]


def _fmt(v, p=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.{p}f}"


def _aggregate_per_cat(per_seed_pc):
    """per_seed_pc : list of {cat: {auc, n, n_errors}}.
    Returns {cat: {auc_mean, auc_std, n, n_errors_mean, n_seeds_auc}}."""
    cats = set()
    for r in per_seed_pc:
        cats.update(r.keys() if r else [])
    out = {}
    for c in cats:
        aucs = [r.get(c, {}).get("auc", float("nan")) for r in per_seed_pc]
        valid = [a for a in aucs if not math.isnan(a)]
        ns = [r.get(c, {}).get("n", 0) for r in per_seed_pc]
        errs = [r.get(c, {}).get("n_errors", 0) for r in per_seed_pc]
        out[c] = {
            "auc_mean": float(np.mean(valid)) if valid else float("nan"),
            "auc_std": float(np.std(valid)) if len(valid) > 1 else 0.0,
            "n_seeds_auc": len(valid),
            "n": int(np.mean(ns)) if ns else 0,
            "n_errors_mean": float(np.mean(errs)) if errs else 0.0,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="Comma-separated list of per-seed run directories")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    run_dirs = [s.strip() for s in args.runs.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    # Load metrics from each run
    metrics_list = []
    scores_list = []
    for d in run_dirs:
        with open(os.path.join(d, "metrics.json"), "r") as f:
            metrics_list.append(json.load(f))
        with open(os.path.join(d, "scores.json"), "r") as f:
            scores_list.append(json.load(f))
    seeds = [m["seed"] for m in metrics_list]
    print(f"Loaded {len(metrics_list)} runs (seeds={seeds})")

    # ── Aggregate global AUC per model
    model_aucs = {}
    for m in metrics_list:
        for mod in m["models"]:
            model_aucs.setdefault(mod["name"], []).append(mod["auc"])
    global_summary = {}
    for name, aucs in model_aucs.items():
        global_summary[name] = {
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)) if len(aucs) > 1 else 0.0,
            "n_seeds": len(aucs),
        }

    # ── Aggregate per-category for each model
    pc_mlp4 = _aggregate_per_cat([m["per_category"]["metamlp_etape2"] for m in metrics_list])
    pc_mlpB = _aggregate_per_cat([m["per_category"].get("metamlp_bplus", {}) for m in metrics_list])
    pc_enc = _aggregate_per_cat([m["per_category"]["metaencoder_etape3"] for m in metrics_list])

    # Sort categories by error rate desc
    cats_sorted = sorted(pc_enc.keys(),
        key=lambda c: (-(pc_enc[c]["n_errors_mean"] / max(pc_enc[c]["n"], 1)), c))

    # ── Aggregate decorrelation
    decorrs = [m.get("decorrelation_auc_on_gbdt_surprise_errors") for m in metrics_list]
    decorrs = [d for d in decorrs if d is not None]
    decorr_mean = float(np.mean(decorrs)) if decorrs else None

    # ── Aggregate bootstrap CIs (mean of mean across seeds)
    bs_keys = set()
    for m in metrics_list:
        bs_keys.update(m.get("bootstrap_ci", {}).keys())
    bs_agg = {}
    for k in bs_keys:
        ms = [m["bootstrap_ci"][k]["mean"] for m in metrics_list
              if k in m.get("bootstrap_ci", {})]
        los = [m["bootstrap_ci"][k]["lo"] for m in metrics_list
              if k in m.get("bootstrap_ci", {})]
        his = [m["bootstrap_ci"][k]["hi"] for m in metrics_list
              if k in m.get("bootstrap_ci", {})]
        if ms:
            bs_agg[k] = {"mean": float(np.mean(ms)),
                         "lo_mean": float(np.mean(los)),
                         "hi_mean": float(np.mean(his)),
                         "n_seeds": len(ms)}

    # ── Encoder params (from any run)
    enc_params = metrics_list[0].get("encoder_params", -1)
    vocab_size = metrics_list[0].get("vocab_size", -1)

    # ──────────────────────────────────────────────
    # Write summary.md
    # ──────────────────────────────────────────────
    lines = []
    lines.append("# Étape 3 — Encodeur léger sur input (3 seeds)")
    lines.append("")
    lines.append(f"- Seeds : `{','.join(str(s) for s in seeds)}`")
    lines.append(f"- Encoder params : {enc_params/1000:.1f}k")
    lines.append(f"- Vocab size : {vocab_size}")
    lines.append("")

    # Global table
    lines.append("## AUC global (3 seeds)")
    lines.append("")
    lines.append("| Modèle | AUC mean | AUC std |")
    lines.append("|---|---:|---:|")
    order = ["raw_surprise", "logreg_surprise", "gbdt_1d_surprise",
             "raw_entropy", "gbdt_1d_entropy",
             "metamlp_etape2", "metamlp_bplus", "metaencoder_etape3"]
    for name in order:
        if name in global_summary:
            s = global_summary[name]
            lines.append(f"| {name} | {_fmt(s['auc_mean'], 4)} | {_fmt(s['auc_std'], 4)} |")
    lines.append("")
    if decorr_mean is not None:
        lines.append(f"- **Décorrélation MetaEncoder | erreurs GBDT-surprise** : "
                     f"{decorr_mean:.3f} (mean sur {len(decorrs)} seeds)")
    lines.append("")
    if bs_agg:
        lines.append("## Bootstrap CI (mean sur seeds)")
        lines.append("")
        lines.append("| Comparaison | Δ moyen | IC95 lo | IC95 hi |")
        lines.append("|---|---:|---:|---:|")
        for k, v in bs_agg.items():
            lines.append(f"| {k} | {_fmt(v['mean'], 4)} | {_fmt(v['lo_mean'], 4)} | "
                         f"{_fmt(v['hi_mean'], 4)} |")
        lines.append("")

    # Per-category table
    lines.append("## Par catégorie — AUC mean (3 seeds)")
    lines.append("")
    lines.append("| Catégorie | err rate | étape 2 | B+ | étape 3 | Δ vs B+ |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for c in cats_sorted:
        e_enc = pc_enc[c]
        n = e_enc["n"]
        err = e_enc["n_errors_mean"]
        rate = (err / n) if n else 0.0
        a4 = pc_mlp4.get(c, {}).get("auc_mean", float("nan"))
        aB = pc_mlpB.get(c, {}).get("auc_mean", float("nan"))
        ae = e_enc["auc_mean"]
        d = ae - aB if not (math.isnan(ae) or math.isnan(aB)) else float("nan")
        bold = "**" if c in PROBLEM_CATS else ""
        lines.append(f"| {bold}{c}{bold} | {100*rate:.1f}% | "
                     f"{_fmt(a4)} | {_fmt(aB)} | {_fmt(ae)} | {_fmt(d, 3)} |")
    lines.append("")

    # Focus problem cats
    lines.append("## Focus catégories problématiques")
    lines.append("")
    lines.append("| Catégorie | étape 2 | B+ | étape 3 | Δ vs B+ |")
    lines.append("|---|---:|---:|---:|---:|")
    aucs_problem_e2 = []; aucs_problem_bp = []; aucs_problem_e3 = []
    for c in PROBLEM_CATS:
        a4 = pc_mlp4.get(c, {}).get("auc_mean", float("nan"))
        aB = pc_mlpB.get(c, {}).get("auc_mean", float("nan"))
        ae = pc_enc.get(c, {}).get("auc_mean", float("nan"))
        if not math.isnan(a4): aucs_problem_e2.append(a4)
        if not math.isnan(aB): aucs_problem_bp.append(aB)
        if not math.isnan(ae): aucs_problem_e3.append(ae)
        d = ae - aB if not (math.isnan(ae) or math.isnan(aB)) else float("nan")
        lines.append(f"| {c} | {_fmt(a4)} | {_fmt(aB)} | {_fmt(ae)} | {_fmt(d, 3)} |")
    avg_e2 = float(np.mean(aucs_problem_e2)) if aucs_problem_e2 else float("nan")
    avg_bp = float(np.mean(aucs_problem_bp)) if aucs_problem_bp else float("nan")
    avg_e3 = float(np.mean(aucs_problem_e3)) if aucs_problem_e3 else float("nan")
    lines.append("")
    lines.append(f"- AUC moyenne sur 4 problématiques : "
                 f"étape 2 = {_fmt(avg_e2)}  →  B+ = {_fmt(avg_bp)}  →  "
                 f"**étape 3 = {_fmt(avg_e3)}**")
    lines.append("")

    # Near-perfect FP check (use threshold-based predictions from scores.json)
    lines.append("## Catégories quasi-parfaites — faux positifs cumulés (3 seeds)")
    lines.append("")
    lines.append("| Catégorie | n total | FP encoder (somme) |")
    lines.append("|---|---:|---:|")
    fp_sum_total = 0
    for cat in NEAR_PERFECT:
        n_total = 0
        fp_sum = 0
        for sd in scores_list:
            yte = np.array(sd["yte"])
            cats = sd["cats_test"]
            sco = np.array(sd["s_enc_te"])
            mask = np.array([c == cat for c in cats])
            if mask.sum() == 0:
                continue
            n_total += int(mask.sum())
            # Threshold = best F1 on val (we don't have that here; use 0.5)
            pred = (sco[mask] >= 0.5).astype(int)
            fp_sum += int(((pred == 1) & (yte[mask] == 0)).sum())
        fp_sum_total += fp_sum
        lines.append(f"| {cat} | {n_total} | {fp_sum} |")
    lines.append("")
    lines.append(f"- FP totaux sur catégories quasi-parfaites : **{fp_sum_total}**")
    lines.append("")

    summary_path = os.path.join(args.output_dir, "etape3_summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {summary_path}")

    # ──────────────────────────────────────────────
    # Qualitative
    # ──────────────────────────────────────────────
    qual = ["# Étape 3 — Inspection qualitative", ""]
    qual.append("Sur seed 42 (1er run), 5 vrais positifs + 5 faux négatifs sur "
                "les catégories problématiques.")
    qual.append("")
    sd = scores_list[0]
    yte = np.array(sd["yte"]); sco = np.array(sd["s_enc_te"])
    cats = sd["cats_test"]; toks = sd["input_tokens_test"]
    # Threshold = median val score (proxy without val info here)
    thr = 0.5
    tp_examples = []; fn_examples = []
    for i, c in enumerate(cats):
        if c not in PROBLEM_CATS:
            continue
        is_err = bool(yte[i])
        pred_err = bool(sco[i] >= thr)
        if is_err and pred_err and len(tp_examples) < 5:
            tp_examples.append((i, c, sco[i], toks[i]))
        elif is_err and not pred_err and len(fn_examples) < 5:
            fn_examples.append((i, c, sco[i], toks[i]))
        if len(tp_examples) >= 5 and len(fn_examples) >= 5:
            break

    qual.append("## Vrais positifs (erreurs détectées par l'encodeur)")
    qual.append("")
    if not tp_examples:
        qual.append("_Aucun vrai positif aux catégories problématiques avec threshold=0.5._")
    for i, c, s, t in tp_examples:
        qual.append(f"- **{c}**  (idx {i}, score {s:.3f})")
        qual.append(f"  - input : `{' '.join(t)}`")
    qual.append("")

    qual.append("## Faux négatifs (erreurs ratées par l'encodeur)")
    qual.append("")
    if not fn_examples:
        qual.append("_Aucun faux négatif aux catégories problématiques avec threshold=0.5._")
    for i, c, s, t in fn_examples:
        qual.append(f"- **{c}**  (idx {i}, score {s:.3f})")
        qual.append(f"  - input : `{' '.join(t)}`")
    qual.append("")

    qual_path = os.path.join(args.output_dir, "etape3_qualitative.md")
    with open(qual_path, "w") as f:
        f.write("\n".join(qual))
    print(f"Wrote {qual_path}")

    # ──────────────────────────────────────────────
    # Verdict per spec §7
    # ──────────────────────────────────────────────
    auc_global_e3 = global_summary.get("metaencoder_etape3", {}).get("auc_mean", float("nan"))

    vlines = ["# Étape 3 — Verdict", ""]
    vlines.append(f"- AUC global étape 3 : {_fmt(auc_global_e3, 4)} (cible ≥ 0.97)")
    vlines.append(f"- AUC moyenne 4 problématiques : étape 2 = {_fmt(avg_e2)}, "
                  f"B+ = {_fmt(avg_bp)}, **étape 3 = {_fmt(avg_e3)}**")
    vlines.append(f"- FP cumulés sur catégories quasi-parfaites : {fp_sum_total} "
                  f"(cible ≤ 5 ; alerte > 10)")
    vlines.append("")

    # Critère secondaire (no regression)
    secondary_ok = (not math.isnan(auc_global_e3) and auc_global_e3 >= 0.97)
    fp_ok = fp_sum_total <= 5
    fp_alert = fp_sum_total > 10

    if not secondary_ok and not math.isnan(auc_global_e3) and auc_global_e3 < 0.95:
        vlines.append("**PROBLÈME critère secondaire** : AUC globale < 0.95 — "
                      "l'encodeur a introduit du bruit. Investiguer (sur-paramétrisation, "
                      "tokenisation) avant de conclure.")
    elif fp_alert:
        vlines.append(f"**PROBLÈME** : {fp_sum_total} faux positifs sur les catégories "
                      "quasi-parfaites (> 10). L'encodeur génère des alarmes à tort.")

    # Critère principal
    if not math.isnan(avg_e3):
        if avg_e3 >= 0.75:
            vlines.append("**GO étape 4** : AUC moyenne sur 4 problématiques ≥ 0.75. "
                          "Signal sub-distributionnel capturé par l'encodeur.")
            vlines.append("→ Continuer avec boucle de correction (selective prediction "
                          "et/ou augmentation ciblée automatique).")
        elif avg_e3 >= 0.65:
            vlines.append("**GO partiel** : AUC moyenne sur 4 problématiques ∈ [0.65, 0.75[. "
                          "Amélioration claire vs B+ (~0.49). Décider avec Julien si on "
                          "continue ou si on raffine d'abord (étape 3bis).")
        elif avg_e3 >= 0.55:
            vlines.append("**PARTIEL** : AUC moyenne sur 4 problématiques ∈ [0.55, 0.65[. "
                          "L'encodeur aide mais ne résout pas le problème. "
                          "Possible étape 3bis avec activations internes du modèle de base.")
        else:
            vlines.append("**NO-GO** : AUC moyenne sur 4 problématiques < 0.55. "
                          "L'encodeur n'apporte pas le signal manquant. "
                          "Hypothèse 'structure d'input suffit' falsifiée. Pivoter.")
    vlines.append("")
    vlines.append("**Toujours revenir à Julien avec ces chiffres avant l'étape suivante.**")

    vp = os.path.join(args.output_dir, "etape3_verdict.md")
    with open(vp, "w") as f:
        f.write("\n".join(vlines))
    print(f"Wrote {vp}")
    print()
    print("\n".join(vlines))


if __name__ == "__main__":
    main()
