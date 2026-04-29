#!/usr/bin/env python3
"""Analyse fine du meta-modèle par catégorie COGS.

Re-entraîne le MetaMLP sur 3 seeds avec les features étape 2 (configurables)
puis, sur le meta-test, calcule pour chaque catégorie :
  - AUC (si ≥20 exemples ET les 2 classes présentes)
  - matrice de confusion au seuil val-optimal
  - tagging spécifique pour cp_recursion / pp_recursion (impossibles)
    et active_to_passive / passive_to_active / unacc_to_transitive (faciles)

Agrégation 3 seeds : mean ± std AUC, somme TP/FN/TN/FP.

Pas de critère GO/NO-GO — exploration informative.

Usage:
  python3 meta_analyse_par_categorie.py \\
      --dataset meta_data/cogs_meta_dataset.jsonl \\
      --splits  meta_data/cogs_meta_splits.json \\
      --features surprise_mean,entropy_mean_greedy,rare_token_count,nesting_depth \\
      --seeds 42,123,456 \\
      --output runs_meta/etape2_analyse_par_categorie.md
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


# ══════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════
def _load_records(jsonl_path):
    with open(jsonl_path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _features_matrix(records, feature_names):
    return np.array(
        [[r["features_inference"][f] for f in feature_names] for r in records],
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
# MetaMLP (identique étape 1/2)
# ══════════════════════════════════════════════════════════════════════════
class MetaMLP(nn.Module):
    def __init__(self, n_features, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
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
    best_auc = -1.0
    best_state = None
    pat = 0
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
            scores_v = torch.sigmoid(model(Xv)).cpu().numpy()
        auc = _roc_auc(yv.cpu().numpy(), scores_v)
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


def _f1_at_threshold(y_true, scores, threshold):
    pred = (scores >= threshold).astype(np.float32)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1}


def _best_threshold_on_val(y_val, scores_val):
    if len(set(y_val)) < 2:
        return 0.5
    cands = np.linspace(0.01, 0.99, 99)
    best_f1, best_t = -1, 0.5
    for t in cands:
        m = _f1_at_threshold(y_val, scores_val, t)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]; best_t = t
    return float(best_t)


# ══════════════════════════════════════════════════════════════════════════
# Per-seed analysis
# ══════════════════════════════════════════════════════════════════════════
def _run_one_seed(seed, train_recs, val_recs, test_recs, feat_names):
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
        s_v = torch.sigmoid(mlp(torch.from_numpy(Xv_s).to(device))).cpu().numpy()

    thr = _best_threshold_on_val(yv, s_v)
    auc_global = _roc_auc(yte, s_te)
    return {
        "seed": seed,
        "scores_test": s_te,
        "labels_test": yte,
        "categories_test": _categories(test_recs),
        "threshold": thr,
        "auc_global": auc_global,
    }


# ══════════════════════════════════════════════════════════════════════════
# Aggregation
# ══════════════════════════════════════════════════════════════════════════
def _per_category_for_seed(seed_result):
    """Returns {category: {auc, tp, fp, fn, tn, n, n_errors, error_rate, f1}}."""
    yte = seed_result["labels_test"]
    sco = seed_result["scores_test"]
    cats = seed_result["categories_test"]
    thr = seed_result["threshold"]
    cat_to_idx = defaultdict(list)
    for i, c in enumerate(cats):
        cat_to_idx[c].append(i)
    out = {}
    for cat, idxs in cat_to_idx.items():
        idxs = np.array(idxs)
        y_c = yte[idxs]; s_c = sco[idxs]
        n = len(idxs); n_err = int(y_c.sum())
        if n >= 20 and len(set(y_c)) == 2:
            auc = _roc_auc(y_c, s_c)
        else:
            auc = float("nan")
        cm = _f1_at_threshold(y_c, s_c, thr)
        out[cat] = {
            "n": n, "n_errors": n_err,
            "error_rate": n_err / n if n else float("nan"),
            "auc": auc,
            "tp": cm["tp"], "fp": cm["fp"], "fn": cm["fn"], "tn": cm["tn"],
            "f1": cm["f1"], "precision": cm["precision"], "recall": cm["recall"],
        }
    return out


def _aggregate(per_seed):
    """per_seed: list of {cat: stats}.
    Returns {cat: {auc_mean, auc_std, tp_sum, ..., n_seed_with_auc}}."""
    cats = set()
    for r in per_seed:
        cats.update(r.keys())
    out = {}
    for cat in cats:
        aucs = [r[cat]["auc"] for r in per_seed
                if cat in r and not math.isnan(r[cat]["auc"])]
        tps = [r[cat]["tp"] for r in per_seed if cat in r]
        fps = [r[cat]["fp"] for r in per_seed if cat in r]
        fns = [r[cat]["fn"] for r in per_seed if cat in r]
        tns = [r[cat]["tn"] for r in per_seed if cat in r]
        ns = [r[cat]["n"] for r in per_seed if cat in r]
        errs = [r[cat]["n_errors"] for r in per_seed if cat in r]
        out[cat] = {
            "n": int(np.mean(ns)) if ns else 0,
            "n_errors_mean": float(np.mean(errs)) if errs else 0.0,
            "error_rate": float(np.mean(errs) / np.mean(ns)) if ns else float("nan"),
            "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
            "auc_std": float(np.std(aucs)) if len(aucs) > 1 else 0.0,
            "n_seeds_auc": len(aucs),
            "tp_sum": int(sum(tps)), "fp_sum": int(sum(fps)),
            "fn_sum": int(sum(fns)), "tn_sum": int(sum(tns)),
            "fp_rate_per_seed_mean": float(np.mean(
                [fp / max(n - e, 1) for fp, n, e in zip(fps, ns, errs)])
                if ns else float("nan")),
            "miss_rate_per_seed_mean": float(np.mean(
                [fn / max(e, 1) for fn, e in zip(fns, errs)])
                if errs else float("nan")),
            "flag_rate_per_seed_mean": float(np.mean(
                [(tp + fp) / max(n, 1) for tp, fp, n in zip(tps, fps, ns)])
                if ns else float("nan")),
        }
    return out


# ══════════════════════════════════════════════════════════════════════════
# Markdown report
# ══════════════════════════════════════════════════════════════════════════
IMPOSSIBLE = {"cp_recursion", "pp_recursion"}
NEAR_PERFECT = {"active_to_passive", "passive_to_active", "unacc_to_transitive"}


def _write_report(out_path, agg, feat_names, seeds, auc_global_mean, auc_global_std):
    cats_sorted = sorted(agg.keys(),
                         key=lambda c: (-agg[c]["error_rate"], c))

    def _fmt(v, p=4):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "N/A"
        return f"{v:.{p}f}"

    lines = []
    lines.append("# Analyse meta-modèle par catégorie")
    lines.append("")
    lines.append(f"- Features : `{','.join(feat_names)}`")
    lines.append(f"- Seeds : `{','.join(str(s) for s in seeds)}`")
    lines.append(f"- AUC global meta-test : **{_fmt(auc_global_mean)} ± {_fmt(auc_global_std)}**")
    lines.append(f"- Catégories : {len(cats_sorted)}")
    lines.append("")

    # Tableau global
    lines.append("## Tableau global (3 seeds, agrégat)")
    lines.append("")
    lines.append("| Catégorie | n | err rate | AUC mean | AUC std | n_seeds | TP | FN | TN | FP |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cat in cats_sorted:
        s = agg[cat]
        lines.append(
            f"| {cat} | {s['n']} | {100*s['error_rate']:.1f}% | "
            f"{_fmt(s['auc_mean'], 3)} | {_fmt(s['auc_std'], 3)} | "
            f"{s['n_seeds_auc']}/{len(seeds)} | "
            f"{s['tp_sum']} | {s['fn_sum']} | {s['tn_sum']} | {s['fp_sum']} |"
        )
    lines.append("")

    # Catégories problématiques
    problems = [c for c in cats_sorted
                if not math.isnan(agg[c]["auc_mean"])
                and (agg[c]["auc_mean"] < 0.7 or agg[c]["auc_std"] > 0.05)]
    lines.append("## Catégories problématiques (AUC < 0.7 ou std > 0.05)")
    lines.append("")
    if not problems:
        lines.append("_Aucune. Le meta-modèle est uniforme et stable sur toutes les "
                     "catégories où l'AUC est calculable._")
    else:
        for cat in problems:
            s = agg[cat]
            reason = []
            if s["auc_mean"] < 0.7:
                reason.append(f"AUC mean {s['auc_mean']:.3f} < 0.7")
            if s["auc_std"] > 0.05:
                reason.append(f"AUC std {s['auc_std']:.3f} > 0.05")
            lines.append(f"- **{cat}** ({s['n']} ex, err {100*s['error_rate']:.1f}%) : "
                         f"{' ; '.join(reason)}")
    lines.append("")

    # Catégories impossibles
    lines.append("## Catégories impossibles (cp_recursion, pp_recursion)")
    lines.append("")
    for cat in IMPOSSIBLE:
        if cat not in agg:
            continue
        s = agg[cat]
        flag_pct = 100 * s["flag_rate_per_seed_mean"]
        miss_pct = 100 * s["miss_rate_per_seed_mean"]
        lines.append(f"### {cat}  (n={s['n']}, taux d'erreur cible {100*s['error_rate']:.1f}%)")
        lines.append("")
        lines.append(f"- Le meta-modèle flagge **{flag_pct:.1f}%** des inputs comme à risque "
                     f"(au seuil val-optimal)")
        lines.append(f"- Erreurs ratées (faux négatifs sur les vraies erreurs) : **{miss_pct:.1f}%**")
        lines.append(f"- AUC = {_fmt(s['auc_mean'], 3)} ± {_fmt(s['auc_std'], 3)}")
        if s["error_rate"] > 0.7 and flag_pct < 70:
            lines.append(f"- ⚠️ Cible erronée >{100*s['error_rate']:.0f}% mais flag <70% → "
                         "le meta-modèle ne capte pas l'irréductibilité architecturale.")
        elif s["error_rate"] > 0.7 and flag_pct >= 70:
            lines.append(f"- ✓ Flag conforme à la cible ({100*s['error_rate']:.0f}% d'erreurs).")
        lines.append("")

    # Catégories quasi-parfaites
    lines.append("## Catégories quasi-parfaites (active_to_passive, etc.)")
    lines.append("")
    for cat in sorted(NEAR_PERFECT):
        if cat not in agg:
            continue
        s = agg[cat]
        fp_pct = 100 * s["fp_rate_per_seed_mean"]
        lines.append(f"### {cat}  (n={s['n']}, taux d'erreur cible {100*s['error_rate']:.1f}%)")
        lines.append("")
        lines.append(f"- Faux positifs : **{s['fp_sum']}** sur les 3 seeds combinés "
                     f"({fp_pct:.2f}% des négatifs en moyenne)")
        lines.append(f"- AUC = {_fmt(s['auc_mean'], 3)} (n_seeds_auc={s['n_seeds_auc']}/{len(seeds)})")
        if fp_pct > 5:
            lines.append(f"- ⚠️ Taux de faux positifs >5% sur une catégorie facile.")
        elif s["n_errors_mean"] < 5 and math.isnan(s["auc_mean"]):
            lines.append("- AUC indéfini : trop peu d'erreurs pour discriminer "
                         "(comportement attendu).")
        else:
            lines.append("- ✓ Faux positifs acceptables.")
        lines.append("")

    # Synthèse
    lines.append("## Synthèse (5-10 lignes)")
    lines.append("")
    n_uniform = sum(1 for c in cats_sorted
                    if not math.isnan(agg[c]["auc_mean"])
                    and agg[c]["auc_mean"] >= 0.85
                    and agg[c]["auc_std"] <= 0.05)
    n_total_with_auc = sum(1 for c in cats_sorted
                           if not math.isnan(agg[c]["auc_mean"]))
    lines.append(f"- **{n_uniform}/{n_total_with_auc}** catégories ont AUC ≥ 0.85 et std ≤ 0.05 "
                 "(zones où le meta-modèle est uniforme et stable)")
    lines.append(f"- **{len(problems)}** catégories problématiques (AUC < 0.7 ou std > 0.05)")
    impossible_present = [c for c in IMPOSSIBLE if c in agg]
    if impossible_present:
        flag_avg = np.mean([agg[c]["flag_rate_per_seed_mean"]
                            for c in impossible_present])
        err_avg = np.mean([agg[c]["error_rate"] for c in impossible_present])
        lines.append(f"- Catégories architecturalement impossibles "
                     f"({', '.join(impossible_present)}) : flag moyen "
                     f"**{100*flag_avg:.1f}%** vs cible {100*err_avg:.1f}%")
    near_present = [c for c in NEAR_PERFECT if c in agg]
    if near_present:
        fp_total = sum(agg[c]["fp_sum"] for c in near_present)
        n_total = sum(agg[c]["n"] * len(seeds) for c in near_present)
        lines.append(f"- Catégories quasi-parfaites : "
                     f"**{fp_total}/{n_total}** faux positifs sur 3 seeds combinés "
                     f"({100*fp_total/max(n_total,1):.2f}%)")
    lines.append("")
    lines.append("_Pas de critère GO/NO-GO. Lecture purement informative pour orienter l'étape 3._")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--features",
                    default="surprise_mean,entropy_mean_greedy,rare_token_count,nesting_depth")
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    feat_names = [s.strip() for s in args.features.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]

    records = _load_records(args.dataset)
    with open(args.splits, "r") as f:
        splits = json.load(f)
    train_recs = [records[i] for i in splits["train"]]
    val_recs = [records[i] for i in splits["val"]]
    test_recs = [records[i] for i in splits["test"]]
    print(f"Loaded {len(records)} records (train={len(train_recs)}, "
          f"val={len(val_recs)}, test={len(test_recs)})")
    print(f"Features : {feat_names}")
    print(f"Seeds    : {seeds}")
    print(f"Output   : {args.output}")
    print()

    per_seed_results = []
    auc_globals = []
    for s in seeds:
        print(f"── Seed {s} ──")
        r = _run_one_seed(s, train_recs, val_recs, test_recs, feat_names)
        print(f"   AUC global = {r['auc_global']:.4f}, threshold = {r['threshold']:.3f}")
        per_seed_results.append(_per_category_for_seed(r))
        auc_globals.append(r["auc_global"])

    agg = _aggregate(per_seed_results)
    auc_mean = float(np.mean(auc_globals))
    auc_std = float(np.std(auc_globals)) if len(auc_globals) > 1 else 0.0

    _write_report(args.output, agg, feat_names, seeds, auc_mean, auc_std)
    print(f"\nWritten {args.output}")
    print(f"AUC global meta-test (3 seeds) = {auc_mean:.4f} ± {auc_std:.4f}")


if __name__ == "__main__":
    main()
