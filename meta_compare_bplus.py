#!/usr/bin/env python3
"""Étape B+ — Compare meta-modèle 4 features (étape 2) vs 6 features (B+ distributionnel).

Sur le meta-dataset augmenté par add_distributional_features.py :
  - étape 2  : surprise_mean, entropy_mean_greedy, rare_token_count, nesting_depth
  - B+       : étape 2 + verb_construction_count_train + verb_construction_seen_binary

Pour chaque config × 3 seeds : entraîne MetaMLP, calcule AUC par catégorie sur
meta-test. Agrège (mean ± std) et écrit comparison_with_etape2.md + verdict.md.

Critère §B+ :
  - Sur 4 catégories problématiques (subj_to_obj_common, passive_to_active,
    unacc_to_transitive, obj_omitted_transitive_to_transitive),
    AUC moyenne B+ > 0.6 (vs ~0.32 étape 2) → GO
  - AUC moyenne reste autour de 0.32 → NO-GO
  - AUC globale chute < 0.96 → PROBLÈME

Usage :
  python3 meta_compare_bplus.py \\
      --dataset meta_data/cogs_meta_dataset_bplus.jsonl \\
      --splits  meta_data/cogs_meta_splits.json \\
      --output-dir runs_meta/etape_bplus
"""
import argparse
import json
import math
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FEAT_ETAPE2 = ["surprise_mean", "entropy_mean_greedy",
               "rare_token_count", "nesting_depth"]
FEAT_BPLUS = FEAT_ETAPE2 + ["verb_construction_count_train",
                            "verb_construction_seen_binary"]
PROBLEM_CATS = {"subj_to_obj_common", "passive_to_active",
                "unacc_to_transitive", "obj_omitted_transitive_to_transitive"}


# ══════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════
def _load_records(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _features_matrix(records, names):
    return np.array(
        [[r["features_inference"][f] for f in names] for r in records],
        dtype=np.float32,
    )


def _labels(records):
    return np.array([0 if r["exact_match"] else 1 for r in records],
                    dtype=np.float32)


def _categories(records):
    return [r["category"] for r in records]


def _standardize(X_train, X_other):
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (X_train - mu) / sd, (X_other - mu) / sd, (mu, sd)


# ══════════════════════════════════════════════════════════════════════════
# MetaMLP
# ══════════════════════════════════════════════════════════════════════════
class MetaMLP(nn.Module):
    def __init__(self, n_features, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _train_mlp(X_train, y_train, X_val, y_val, n_features, seed,
               lr=1e-3, batch_size=64, max_epochs=30, patience=5):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetaMLP(n_features).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.from_numpy(X_train).to(device)
    yt = torch.from_numpy(y_train).to(device)
    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val).to(device)
    best_auc, best_state, pat = -1.0, None, 0
    for ep in range(max_epochs):
        model.train()
        idx = torch.randperm(Xt.size(0), device=device)
        for start in range(0, Xt.size(0), batch_size):
            sl = idx[start:start + batch_size]
            logits = model(Xt[sl])
            loss = F.binary_cross_entropy_with_logits(logits, yt[sl])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            sv = torch.sigmoid(model(Xv)).cpu().numpy()
        auc = _roc_auc(yv.cpu().numpy(), sv)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience:
                break
    model.load_state_dict(best_state)
    return model


# ══════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════
def _roc_auc(y_true, scores):
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float("nan")


# ══════════════════════════════════════════════════════════════════════════
# Per-config × seed run
# ══════════════════════════════════════════════════════════════════════════
def _run_one(seed, train_recs, val_recs, test_recs, feat_names):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    Xtr = _features_matrix(train_recs, feat_names)
    Xv = _features_matrix(val_recs, feat_names)
    Xte = _features_matrix(test_recs, feat_names)
    ytr = _labels(train_recs); yv = _labels(val_recs); yte = _labels(test_recs)

    Xtr_s, _, (mu, sd) = _standardize(Xtr, Xv)
    Xv_s = (Xv - mu) / sd
    Xte_s = (Xte - mu) / sd

    mlp = _train_mlp(Xtr_s, ytr, Xv_s, yv, n_features=Xtr_s.shape[1], seed=seed)
    mlp.eval()
    device = next(mlp.parameters()).device
    with torch.no_grad():
        s_te = torch.sigmoid(mlp(torch.from_numpy(Xte_s).to(device))).cpu().numpy()

    cats = _categories(test_recs)
    auc_global = _roc_auc(yte, s_te)
    cat_to_idx = defaultdict(list)
    for i, c in enumerate(cats):
        cat_to_idx[c].append(i)
    cat_aucs = {}
    for cat, idxs in cat_to_idx.items():
        idxs = np.array(idxs)
        y_c = yte[idxs]; s_c = s_te[idxs]
        if len(idxs) >= 20 and len(set(y_c)) == 2:
            cat_aucs[cat] = _roc_auc(y_c, s_c)
        else:
            cat_aucs[cat] = float("nan")
    return {"auc_global": auc_global, "cat_aucs": cat_aucs,
            "n_per_cat": {c: len(i) for c, i in cat_to_idx.items()},
            "errors_per_cat": {c: int(yte[np.array(i)].sum())
                               for c, i in cat_to_idx.items()}}


# ══════════════════════════════════════════════════════════════════════════
# Aggregate over seeds
# ══════════════════════════════════════════════════════════════════════════
def _agg_seeds(seed_results):
    cats = set()
    aucs_global = []
    for r in seed_results:
        cats.update(r["cat_aucs"].keys())
        aucs_global.append(r["auc_global"])
    out_cats = {}
    for c in cats:
        vals = [r["cat_aucs"].get(c, float("nan")) for r in seed_results]
        valid = [v for v in vals if not math.isnan(v)]
        out_cats[c] = {
            "auc_mean": float(np.mean(valid)) if valid else float("nan"),
            "auc_std": float(np.std(valid)) if len(valid) > 1 else 0.0,
            "n_seeds_auc": len(valid),
            "n": int(np.mean([r["n_per_cat"].get(c, 0) for r in seed_results])),
            "n_errors_mean": float(np.mean(
                [r["errors_per_cat"].get(c, 0) for r in seed_results])),
        }
    return {
        "auc_global_mean": float(np.mean(aucs_global)),
        "auc_global_std": float(np.std(aucs_global))
            if len(aucs_global) > 1 else 0.0,
        "per_category": out_cats,
    }


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="JSONL augmenté (sortie de add_distributional_features.py)")
    ap.add_argument("--splits", required=True)
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    records = _load_records(args.dataset)
    with open(args.splits, "r") as f:
        splits = json.load(f)
    train_recs = [records[i] for i in splits["train"]]
    val_recs = [records[i] for i in splits["val"]]
    test_recs = [records[i] for i in splits["test"]]
    print(f"Loaded {len(records)} records, train={len(train_recs)}, "
          f"val={len(val_recs)}, test={len(test_recs)}")
    print(f"étape 2 features : {FEAT_ETAPE2}")
    print(f"B+ features      : {FEAT_BPLUS}")
    print(f"Seeds            : {seeds}")
    print()

    # Run both configs × all seeds
    res_e2 = []
    res_bp = []
    for s in seeds:
        print(f"── Seed {s} ──")
        r2 = _run_one(s, train_recs, val_recs, test_recs, FEAT_ETAPE2)
        rb = _run_one(s, train_recs, val_recs, test_recs, FEAT_BPLUS)
        print(f"   étape2 AUC = {r2['auc_global']:.4f}   |   "
              f"B+ AUC = {rb['auc_global']:.4f}")
        res_e2.append(r2)
        res_bp.append(rb)

    agg_e2 = _agg_seeds(res_e2)
    agg_bp = _agg_seeds(res_bp)

    # ── Build comparison MD
    # Sort categories by error_rate desc (most-erroring first), then by name
    all_cats = set(agg_e2["per_category"].keys()) | set(agg_bp["per_category"].keys())
    def _err_rate(c):
        e = agg_e2["per_category"].get(c, {})
        if "n" in e and e["n"]:
            return e["n_errors_mean"] / e["n"]
        return 0.0
    cats_sorted = sorted(all_cats, key=lambda c: (-_err_rate(c), c))

    def _fmt(v, p=3):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "N/A"
        return f"{v:.{p}f}"

    lines = ["# Comparison étape 2 (4 feat) vs B+ (6 feat distributionnelles)"]
    lines.append("")
    lines.append(f"- Seeds : `{','.join(str(s) for s in seeds)}`")
    lines.append(f"- AUC global étape 2 : **{_fmt(agg_e2['auc_global_mean'], 4)} "
                 f"± {_fmt(agg_e2['auc_global_std'], 4)}**")
    lines.append(f"- AUC global B+      : **{_fmt(agg_bp['auc_global_mean'], 4)} "
                 f"± {_fmt(agg_bp['auc_global_std'], 4)}**")
    delta_global = agg_bp["auc_global_mean"] - agg_e2["auc_global_mean"]
    lines.append(f"- Δ global B+ − étape 2 : **{delta_global:+.4f}**")
    lines.append("")

    lines.append("## Par catégorie (AUC mean sur 3 seeds)")
    lines.append("")
    lines.append("| Catégorie | err rate | étape 2 AUC | B+ AUC | Delta |")
    lines.append("|---|---:|---:|---:|---:|")
    for cat in cats_sorted:
        e2 = agg_e2["per_category"].get(cat, {})
        bp = agg_bp["per_category"].get(cat, {})
        n = e2.get("n", bp.get("n", 0))
        n_err = e2.get("n_errors_mean", 0)
        err_rate = (n_err / n) if n else 0.0
        a2 = e2.get("auc_mean", float("nan"))
        ab = bp.get("auc_mean", float("nan"))
        d = ab - a2 if not (math.isnan(a2) or math.isnan(ab)) else float("nan")
        bold = "**" if cat in PROBLEM_CATS else ""
        lines.append(f"| {bold}{cat}{bold} | {100*err_rate:.1f}% | "
                     f"{_fmt(a2)} | {_fmt(ab)} | {_fmt(d, 3)} |")
    lines.append("")

    # Specific block for problem cats
    lines.append("## Catégories problématiques (focus B+)")
    lines.append("")
    lines.append("| Catégorie | étape 2 AUC | B+ AUC | Delta |")
    lines.append("|---|---:|---:|---:|")
    deltas_problem = []
    for cat in sorted(PROBLEM_CATS):
        e2 = agg_e2["per_category"].get(cat, {})
        bp = agg_bp["per_category"].get(cat, {})
        a2 = e2.get("auc_mean", float("nan"))
        ab = bp.get("auc_mean", float("nan"))
        d = ab - a2 if not (math.isnan(a2) or math.isnan(ab)) else float("nan")
        if not math.isnan(d):
            deltas_problem.append(d)
        lines.append(f"| {cat} | {_fmt(a2)} | {_fmt(ab)} | {_fmt(d, 3)} |")
    avg_problem_bplus = float(np.mean(
        [agg_bp["per_category"].get(c, {}).get("auc_mean", float("nan"))
         for c in PROBLEM_CATS
         if not math.isnan(agg_bp["per_category"].get(c, {}).get("auc_mean", float("nan")))]
    )) if any(not math.isnan(agg_bp["per_category"].get(c, {}).get("auc_mean", float("nan")))
              for c in PROBLEM_CATS) else float("nan")
    avg_problem_e2 = float(np.mean(
        [agg_e2["per_category"].get(c, {}).get("auc_mean", float("nan"))
         for c in PROBLEM_CATS
         if not math.isnan(agg_e2["per_category"].get(c, {}).get("auc_mean", float("nan")))]
    )) if any(not math.isnan(agg_e2["per_category"].get(c, {}).get("auc_mean", float("nan")))
              for c in PROBLEM_CATS) else float("nan")
    lines.append("")
    lines.append(f"- AUC moyenne sur les 4 problématiques (étape 2) : **{_fmt(avg_problem_e2)}**")
    lines.append(f"- AUC moyenne sur les 4 problématiques (B+)      : **{_fmt(avg_problem_bplus)}**")
    lines.append("")

    comp_path = os.path.join(args.output_dir, "comparison_with_etape2.md")
    with open(comp_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {comp_path}")

    # ── Verdict
    vlines = ["# Verdict B+", ""]
    vlines.append(f"- AUC global : étape 2 = {_fmt(agg_e2['auc_global_mean'], 4)} → "
                  f"B+ = {_fmt(agg_bp['auc_global_mean'], 4)} (Δ {delta_global:+.4f})")
    vlines.append(f"- AUC moyenne sur les 4 problématiques : "
                  f"étape 2 = {_fmt(avg_problem_e2)} → B+ = {_fmt(avg_problem_bplus)}")
    vlines.append("")

    if not math.isnan(agg_bp["auc_global_mean"]) and agg_bp["auc_global_mean"] < 0.96:
        vlines.append(f"**PROBLÈME** : AUC globale B+ chute sous 0.96. "
                      "Les nouvelles features bruitent le modèle. Investiguer.")
    elif not math.isnan(avg_problem_bplus) and avg_problem_bplus > 0.6:
        vlines.append(f"**GO** : AUC moyenne sur les catégories problématiques "
                      f"= {avg_problem_bplus:.3f} > 0.6 (vs ~{avg_problem_e2:.3f} sans B+). "
                      "Le signal distributionnel répare les erreurs silencieuses.")
        vlines.append("→ Étape 3 : encodeur léger + features scalaires + features distributionnelles "
                      "(architecture hybride).")
    elif not math.isnan(avg_problem_bplus):
        vlines.append(f"**NO-GO** : AUC moyenne sur les catégories problématiques "
                      f"reste à {avg_problem_bplus:.3f} (≤ 0.6). Le signal n'est pas "
                      "(seulement) distributionnel.")
        vlines.append("→ Étape 3 : explorer une représentation plus riche (activations du "
                      "modèle de base à un layer choisi).")
    else:
        vlines.append("Verdict reporté — AUC sur catégories problématiques indisponible.")

    verdict_path = os.path.join(args.output_dir, "verdict.md")
    with open(verdict_path, "w") as f:
        f.write("\n".join(vlines))
    print(f"Wrote {verdict_path}")

    # ── Dump JSON for downstream
    json_dump = {
        "etape2": {"auc_global_mean": agg_e2["auc_global_mean"],
                   "auc_global_std": agg_e2["auc_global_std"],
                   "per_category": agg_e2["per_category"]},
        "bplus": {"auc_global_mean": agg_bp["auc_global_mean"],
                  "auc_global_std": agg_bp["auc_global_std"],
                  "per_category": agg_bp["per_category"]},
        "delta_global": delta_global,
        "avg_problem_etape2": avg_problem_e2,
        "avg_problem_bplus": avg_problem_bplus,
        "feat_etape2": FEAT_ETAPE2,
        "feat_bplus": FEAT_BPLUS,
        "seeds": seeds,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(json_dump, f, indent=2)
    print(f"Wrote {os.path.join(args.output_dir, 'metrics.json')}")
    print()
    print("\n".join(vlines))


if __name__ == "__main__":
    main()
