#!/usr/bin/env python3
"""Baselines formelles pour défendre le claim MetaEncoder.

Compare le MetaEncoder robust (étape B) à 4 baselines stupides :
  1. Category-only      : logreg sur one-hot des catégories
  2. Structure-only     : GBDT sur input_length + nesting_depth + rare_token_count
  3. Surprise-only      : raw surprise_mean
  4. Entropy-only       : raw entropy_mean_greedy

Métriques :
  - AUC global sur meta-test
  - AUC intra-catégorie macro (sur cats mixtes : n_err ≥ 20 ET n_correct ≥ 20)
  - Bootstrap CI 1000 resamples sur les deltas vs MetaEncoder

CONVENTION : tous les AUC sont des fractions ∈ [0, 1], formatés avec :.4f.
JAMAIS multipliés par 100. JAMAIS exprimés en pourcentage.

Usage :
  python3 meta_baselines_formelles.py \\
      --dataset meta_data/cogs_meta_dataset_bplus.jsonl \\
      --splits  meta_data/cogs_meta_splits.json \\
      --meta-scores runs_meta/etapeB_robust/seed_42/scores.json \\
      --output-dir runs_meta/etape_baselines \\
      --seed 42
"""
import argparse
import json
import math
import os

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def _load_records(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _features(recs, names):
    return np.array(
        [[r["features_inference"][f] for f in names] for r in recs],
        dtype=np.float32,
    )


def _y(recs):
    return np.array([0 if r["exact_match"] else 1 for r in recs], dtype=np.int32)


def _cats(recs):
    return [r["category"] for r in recs]


def _safe_auc(y, s):
    """AUC ∈ [0, 1]. NaN si une seule classe ou erreur."""
    try:
        if len(set(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def macro_intra_cat_auc(scores, y, cats, mixed_cats):
    """Renvoie (auc_macro, dict cat → auc). Tous AUC ∈ [0, 1]."""
    per_cat = {}
    aucs = []
    for cat in mixed_cats:
        mask = np.array([c == cat for c in cats])
        if mask.sum() < 5:
            per_cat[cat] = float("nan")
            continue
        y_c = y[mask]; s_c = scores[mask]
        a = _safe_auc(y_c, s_c)
        per_cat[cat] = a
        if not math.isnan(a):
            aucs.append(a)
    macro = float(np.mean(aucs)) if aucs else float("nan")
    return macro, per_cat


def bootstrap_delta_global(sa, sb, y, n_boot=1000, seed=42):
    """IC95 sur (auc(sa) - auc(sb)) global. Tout en fractions [0, 1]."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a = _safe_auc(y[idx], sa[idx])
        b = _safe_auc(y[idx], sb[idx])
        if not (math.isnan(a) or math.isnan(b)):
            deltas.append(a - b)
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(deltas)
    return (float(arr.mean()),
            float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 97.5)))


def bootstrap_delta_macro(sa, sb, y, cats, mixed_cats, n_boot=1000, seed=42):
    """IC95 sur (macro_intra_cat(sa) - macro_intra_cat(sb))."""
    rng = np.random.default_rng(seed)
    n = len(y)
    cats_arr = np.array(cats)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]; sa_b = sa[idx]; sb_b = sb[idx]
        cats_b = cats_arr[idx]
        aucs_a = []; aucs_b = []
        for cat in mixed_cats:
            m = (cats_b == cat)
            if m.sum() < 5:
                continue
            y_c = y_b[m]
            if len(set(y_c)) < 2:
                continue
            a = _safe_auc(y_c, sa_b[m])
            b = _safe_auc(y_c, sb_b[m])
            if not (math.isnan(a) or math.isnan(b)):
                aucs_a.append(a); aucs_b.append(b)
        if aucs_a and aucs_b:
            deltas.append(float(np.mean(aucs_a)) - float(np.mean(aucs_b)))
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(deltas)
    return (float(arr.mean()),
            float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--meta-scores", required=True,
                    help="Path to scores.json from a MetaEncoder robust run")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    records = _load_records(args.dataset)
    with open(args.splits, "r") as f:
        splits = json.load(f)
    train_recs = [records[i] for i in splits["train"]]
    val_recs = [records[i] for i in splits["val"]]
    test_recs = [records[i] for i in splits["test"]]
    print(f"Loaded {len(records)} records "
          f"(train={len(train_recs)}, val={len(val_recs)}, test={len(test_recs)})")

    y_train = _y(train_recs)
    y_test = _y(test_recs)
    cats_train = _cats(train_recs)
    cats_test = _cats(test_recs)

    print(f"Class balance test: {int(y_test.sum())}/{len(y_test)} "
          f"errors ({y_test.mean():.4f} fraction)")

    # ── Identifier les catégories mixtes (pré-engagé : n_err ≥ 20 ET n_cor ≥ 20)
    all_cats = sorted(set(cats_train) | set(cats_test))
    mixed_cats = []
    cat_stats = {}
    for cat in all_cats:
        m = np.array([c == cat for c in cats_test])
        if m.sum() == 0:
            continue
        n_err = int(y_test[m].sum())
        n_cor = int(m.sum() - n_err)
        cat_stats[cat] = {"n": int(m.sum()), "n_err": n_err, "n_cor": n_cor}
        if n_err >= 20 and n_cor >= 20:
            mixed_cats.append(cat)
    print(f"\nCatégories mixtes (n_err ≥ 20 ET n_cor ≥ 20) : {len(mixed_cats)}")
    for cat in mixed_cats:
        s = cat_stats[cat]
        print(f"  - {cat:50s}  n={s['n']}  err={s['n_err']}  cor={s['n_cor']}")

    if not mixed_cats:
        print("AUCUNE catégorie mixte — verdict reporté.")
        raise SystemExit(1)

    # ── Baseline 1 : Category-only
    print("\n── 1. Category-only (logreg one-hot) ──")
    cat_to_idx = {c: i for i, c in enumerate(all_cats)}
    def _one_hot(cs):
        X = np.zeros((len(cs), len(all_cats)), dtype=np.float32)
        for i, c in enumerate(cs):
            X[i, cat_to_idx[c]] = 1.0
        return X
    Xc_tr = _one_hot(cats_train)
    Xc_te = _one_hot(cats_test)
    lr_cat = LogisticRegression(max_iter=1000, random_state=args.seed)
    lr_cat.fit(Xc_tr, y_train)
    proba_cat = lr_cat.predict_proba(Xc_te)[:, 1]
    auc_cat = _safe_auc(y_test, proba_cat)
    print(f"  AUC global = {auc_cat:.4f}")

    # ── Baseline 2 : Structure-only
    print("\n── 2. Structure-only (GBDT sur length, depth, rare_count) ──")
    feat_struct = ["input_length", "nesting_depth", "rare_token_count"]
    Xs_tr = _features(train_recs, feat_struct)
    Xs_te = _features(test_recs, feat_struct)
    scaler = StandardScaler()
    Xs_tr_s = scaler.fit_transform(Xs_tr)
    Xs_te_s = scaler.transform(Xs_te)
    gbdt = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                       random_state=args.seed)
    gbdt.fit(Xs_tr_s, y_train)
    proba_struct = gbdt.predict_proba(Xs_te_s)[:, 1]
    auc_struct = _safe_auc(y_test, proba_struct)
    print(f"  AUC global = {auc_struct:.4f}")

    # ── Baseline 3 : Surprise-only
    print("\n── 3. Surprise-only ──")
    sur_te = _features(test_recs, ["surprise_mean"])[:, 0]
    auc_sur = _safe_auc(y_test, sur_te)
    print(f"  AUC global = {auc_sur:.4f}")

    # ── Baseline 4 : Entropy-only
    print("\n── 4. Entropy-only ──")
    ent_te = _features(test_recs, ["entropy_mean_greedy"])[:, 0]
    auc_ent = _safe_auc(y_test, ent_te)
    print(f"  AUC global = {auc_ent:.4f}")

    # ── MetaEncoder (référence)
    print("\n── 5. MetaEncoder (robust, scores existants) ──")
    with open(args.meta_scores, "r") as f:
        sd = json.load(f)
    meta_scores = np.array(sd["s_enc_te"], dtype=np.float32)
    if len(meta_scores) != len(y_test):
        raise SystemExit(f"Length mismatch: meta_scores={len(meta_scores)} "
                         f"vs y_test={len(y_test)}")
    auc_meta = _safe_auc(y_test, meta_scores)
    print(f"  AUC global = {auc_meta:.4f}")

    # ── Macro intra-cat AUC pour chaque modèle
    print("\n── Macro intra-cat AUC (sur cats mixtes) ──")
    macro_cat, perc_cat = macro_intra_cat_auc(proba_cat, y_test, cats_test, mixed_cats)
    macro_struct, perc_struct = macro_intra_cat_auc(proba_struct, y_test, cats_test, mixed_cats)
    macro_sur, perc_sur = macro_intra_cat_auc(sur_te, y_test, cats_test, mixed_cats)
    macro_ent, perc_ent = macro_intra_cat_auc(ent_te, y_test, cats_test, mixed_cats)
    macro_meta, perc_meta = macro_intra_cat_auc(meta_scores, y_test, cats_test, mixed_cats)
    print(f"  Category-only  : {macro_cat:.4f} (théorique 0.5000)")
    print(f"  Structure-only : {macro_struct:.4f}")
    print(f"  Surprise-only  : {macro_sur:.4f}")
    print(f"  Entropy-only   : {macro_ent:.4f}")
    print(f"  MetaEncoder    : {macro_meta:.4f}")

    # ── Bootstrap CI sur les deltas clés
    print("\n── Bootstrap CI 1000 resamples ──")
    bs = {}
    bs["meta_minus_cat_global"] = bootstrap_delta_global(
        meta_scores, proba_cat, y_test, n_boot=args.n_boot, seed=args.seed)
    bs["meta_minus_struct_global"] = bootstrap_delta_global(
        meta_scores, proba_struct, y_test, n_boot=args.n_boot, seed=args.seed)
    bs["meta_minus_sur_global"] = bootstrap_delta_global(
        meta_scores, sur_te, y_test, n_boot=args.n_boot, seed=args.seed)
    bs["meta_minus_struct_macro"] = bootstrap_delta_macro(
        meta_scores, proba_struct, y_test, cats_test, mixed_cats,
        n_boot=args.n_boot, seed=args.seed)
    for k, (m, lo, hi) in bs.items():
        print(f"  {k:30s}  Δ={m:+.4f}  IC95 [{lo:+.4f}, {hi:+.4f}]")

    # ── Dump metrics.json (tout en fractions [0, 1])
    metrics = {
        "n_test": int(len(y_test)),
        "error_rate_test": float(y_test.mean()),
        "mixed_categories": mixed_cats,
        "global_auc": {
            "category_only": auc_cat,
            "structure_only": auc_struct,
            "surprise_only": auc_sur,
            "entropy_only": auc_ent,
            "metaencoder": auc_meta,
        },
        "macro_intra_cat_auc": {
            "category_only": macro_cat,
            "structure_only": macro_struct,
            "surprise_only": macro_sur,
            "entropy_only": macro_ent,
            "metaencoder": macro_meta,
        },
        "per_cat_auc": {
            "category_only": perc_cat,
            "structure_only": perc_struct,
            "surprise_only": perc_sur,
            "entropy_only": perc_ent,
            "metaencoder": perc_meta,
        },
        "bootstrap_ci": {
            k: {"mean": v[0], "lo": v[1], "hi": v[2]}
            for k, v in bs.items()
        },
    }
    out = os.path.join(args.output_dir, "metrics.json")
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nWrote {out}")

    # ── Table 5.1 : globale + macro
    lines_g = ["# Table baselines — comparaison globale et macro intra-cat", ""]
    lines_g.append("Tous les AUC sont des fractions ∈ [0, 1]. "
                   "Cats mixtes : n_err ≥ 20 ET n_cor ≥ 20 dans le meta-test.")
    lines_g.append("")
    lines_g.append("| Modèle | AUC global | AUC intra-cat macro | n_features |")
    lines_g.append("|---|---:|---:|---:|")
    lines_g.append(f"| Category-only        | {auc_cat:.4f} | {macro_cat:.4f}* | "
                   f"{len(all_cats)} (one-hot) |")
    lines_g.append(f"| Structure-only       | {auc_struct:.4f} | {macro_struct:.4f} | 3 |")
    lines_g.append(f"| Surprise-only        | {auc_sur:.4f} | {macro_sur:.4f} | 1 |")
    lines_g.append(f"| Entropy-only         | {auc_ent:.4f} | {macro_ent:.4f} | 1 |")
    lines_g.append(f"| **MetaEncoder (robust)** | **{auc_meta:.4f}** | "
                   f"**{macro_meta:.4f}** | tokens + 4 |")
    lines_g.append("")
    lines_g.append("*par construction : prédiction constante par catégorie → "
                   "AUC intra-cat ≡ 0.5")
    lines_g.append("")
    lines_g.append("## Bootstrap CI 1000 resamples")
    lines_g.append("")
    lines_g.append("| Comparaison | Δ moyen | IC95 lo | IC95 hi |")
    lines_g.append("|---|---:|---:|---:|")
    for k, (m, lo, hi) in bs.items():
        lines_g.append(f"| {k} | {m:+.4f} | {lo:+.4f} | {hi:+.4f} |")
    with open(os.path.join(args.output_dir, "table_global.md"), "w") as f:
        f.write("\n".join(lines_g))
    print(f"Wrote {os.path.join(args.output_dir, 'table_global.md')}")

    # ── Table 5.2 : par catégorie mixte
    lines_pc = ["# Table baselines — par catégorie mixte", ""]
    lines_pc.append("AUC ∈ [0, 1] sur chaque catégorie mixte (toutes en gras = "
                    "n_err ≥ 20 ET n_cor ≥ 20).")
    lines_pc.append("")
    lines_pc.append("| Catégorie | n | err rate | Cat-only | Struct-only | "
                    "Surprise | Entropy | MetaEncoder |")
    lines_pc.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for cat in mixed_cats:
        s = cat_stats[cat]
        err_rate = s["n_err"] / s["n"]
        lines_pc.append(
            f"| {cat} | {s['n']} | {err_rate:.4f} | "
            f"{perc_cat[cat]:.4f} | {perc_struct[cat]:.4f} | "
            f"{perc_sur[cat]:.4f} | {perc_ent[cat]:.4f} | "
            f"{perc_meta[cat]:.4f} |"
        )
    with open(os.path.join(args.output_dir, "table_per_category.md"), "w") as f:
        f.write("\n".join(lines_pc))
    print(f"Wrote {os.path.join(args.output_dir, 'table_per_category.md')}")

    # ── Verdict §6
    delta_macro_mean, delta_macro_lo, delta_macro_hi = bs["meta_minus_struct_macro"]
    vlines = ["# Verdict baselines formelles", ""]
    vlines.append(f"- AUC global MetaEncoder : **{auc_meta:.4f}**")
    vlines.append(f"- AUC global Structure-only : {auc_struct:.4f}  "
                  f"(Δ = {auc_meta - auc_struct:+.4f})")
    vlines.append(f"- Macro intra-cat MetaEncoder : **{macro_meta:.4f}**")
    vlines.append(f"- Macro intra-cat Structure-only : {macro_struct:.4f}")
    vlines.append(f"- **Δ MetaEncoder − Structure-only (macro) : "
                  f"{delta_macro_mean:+.4f}** "
                  f"(IC95 [{delta_macro_lo:+.4f}, {delta_macro_hi:+.4f}])")
    vlines.append("")

    ic_excludes_zero = (delta_macro_lo > 0)
    if math.isnan(delta_macro_mean):
        vlines.append("**Verdict reporté** — bootstrap macro indisponible.")
    elif delta_macro_mean < 0:
        vlines.append(f"**BUG** : Δ < 0 ({delta_macro_mean:+.4f}). "
                      "Investiguer avant toute conclusion. "
                      "Possible bug d'implémentation.")
    elif delta_macro_mean >= 0.10 and ic_excludes_zero:
        vlines.append(f"**GO claim défendable** : Δ macro = {delta_macro_mean:+.4f} "
                      f"≥ 0.10 avec IC95 excluant 0. "
                      "Le MetaEncoder bat la baseline structurelle dans "
                      "les zones difficiles. Le claim 'introspection au-delà "
                      "de la difficulté de catégorie' tient.")
    elif delta_macro_mean >= 0.05 and ic_excludes_zero:
        vlines.append(f"**PARTIEL** : Δ macro = {delta_macro_mean:+.4f} ∈ "
                      "[0.05, 0.10] avec IC95 excluant 0. "
                      "Signal réel mais modeste. Documenter honnêtement.")
    else:
        vlines.append(f"**NO-GO claim actuel** : Δ macro = {delta_macro_mean:+.4f} "
                      "< 0.05 ou IC95 inclut 0. Le MetaEncoder n'apporte pas "
                      "significativement plus qu'une baseline structurelle. "
                      "Repenser le claim.")
    vlines.append("")
    vlines.append("Détails : `metrics.json`, `table_global.md`, `table_per_category.md`.")
    vp = os.path.join(args.output_dir, "verdict_baselines.md")
    with open(vp, "w") as f:
        f.write("\n".join(vlines))
    print(f"Wrote {vp}")
    print()
    print("\n".join(vlines))


if __name__ == "__main__":
    main()
