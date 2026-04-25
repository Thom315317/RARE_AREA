#!/usr/bin/env python3
"""Phase C — étape 1 du meta-modèle.

Entraîne un MLP simple pour prédire `exact_match=False` à partir des features
inférence-time-valides. Compare à 3 baselines :
  1. Surprise brute (AUC direct, pas de seuil)
  2. Logreg(surprise_mean uniquement)
  3. GBDT 1D sur surprise_mean (baseline forte à battre)

Toutes les features sont standardisées sur meta-train, appliquées à val/test.

Cible : surprise_mean + input_length minimum (correctif review hostile).

Usage:
  python3 meta_train_etape1.py \\
      --dataset meta_data/cogs_meta_dataset.jsonl \\
      --splits meta_data/cogs_meta_splits.json \\
      --output runs_meta/etape1_s42 \\
      --seed 42
"""
import argparse
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════
def _load_records(jsonl_path):
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _features_matrix(records, feature_names):
    rows = []
    for r in records:
        rows.append([r["features_inference"][f] for f in feature_names])
    return np.array(rows, dtype=np.float32)


def _labels(records):
    return np.array([0 if r["exact_match"] else 1 for r in records],
                    dtype=np.float32)  # 1 = error


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
        # Eval on val
        model.eval()
        with torch.no_grad():
            scores_v = torch.sigmoid(model(Xv)).cpu().numpy()
        auc = _roc_auc(yv.cpu().numpy(), scores_v)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience:
                break
    model.load_state_dict(best_state)
    return model


# ══════════════════════════════════════════════════════════════════════════
# Metrics (no scipy/scikit dependency for AUC; numpy-only fallback)
# ══════════════════════════════════════════════════════════════════════════
def _roc_auc(y_true, scores):
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, scores))
    except Exception:
        # Manual rank-sum AUC
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(scores) + 1)
        pos = (y_true == 1)
        n_pos = pos.sum(); n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        rank_sum_pos = ranks[pos].sum()
        return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _brier(y_true, probs):
    return float(np.mean((probs - y_true) ** 2))


def _f1_at_threshold(y_true, scores, threshold):
    pred = (scores >= threshold).astype(np.float32)
    tp = float(((pred == 1) & (y_true == 1)).sum())
    fp = float(((pred == 1) & (y_true == 0)).sum())
    fn = float(((pred == 0) & (y_true == 1)).sum())
    prec = tp / max(tp + fp, 1.0)
    rec = tp / max(tp + fn, 1.0)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    acc = float((pred == y_true).mean())
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1,
            "threshold": float(threshold)}


def _best_threshold_on_val(y_val, scores_val):
    """Threshold that maximises F1 on val (binary search-ish over 200 candidates)."""
    if len(set(y_val)) < 2:
        return 0.5
    cands = np.linspace(0.01, 0.99, 99)
    best_f1, best_t = -1, 0.5
    for t in cands:
        m = _f1_at_threshold(y_val, scores_val, t)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]; best_t = t
    return float(best_t)


def _bootstrap_ci(y_true, scores_a, scores_b, n_resample=1000, seed=42):
    """Returns (mean_diff_AUC, lo, hi) with 95% CI from resampling."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_resample):
        idx = rng.integers(0, n, size=n)
        a = _roc_auc(y_true[idx], scores_a[idx])
        b = _roc_auc(y_true[idx], scores_b[idx])
        if not (np.isnan(a) or np.isnan(b)):
            diffs.append(a - b)
    if not diffs:
        return None, None, None
    diffs = np.array(diffs)
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(args.output, exist_ok=True)
    records = _load_records(args.dataset)
    with open(args.splits, "r") as f:
        splits = json.load(f)
    train_recs = [records[i] for i in splits["train"]]
    val_recs = [records[i] for i in splits["val"]]
    test_recs = [records[i] for i in splits["test"]]

    # Features for the etape-1 cible: surprise_mean + input_length (≥2 features)
    feat_names = ["surprise_mean", "input_length"]
    Xtr = _features_matrix(train_recs, feat_names)
    Xv = _features_matrix(val_recs, feat_names)
    Xte = _features_matrix(test_recs, feat_names)
    ytr = _labels(train_recs)
    yv = _labels(val_recs)
    yte = _labels(test_recs)

    # Standardise
    Xtr_s, _Xv_s, (mu, sd) = _standardize(Xtr, Xv)
    Xv_s = (Xv - mu) / sd
    Xte_s = (Xte - mu) / sd

    # Surprise-only arrays (1D)
    sur_idx = feat_names.index("surprise_mean")
    sur_tr = Xtr[:, sur_idx:sur_idx + 1]
    sur_v = Xv[:, sur_idx:sur_idx + 1]
    sur_te = Xte[:, sur_idx:sur_idx + 1]

    print(f"Loaded {len(records)} records (train={len(train_recs)}, "
          f"val={len(val_recs)}, test={len(test_recs)})")
    print(f"Class balance test: errors={int(yte.sum())}/{len(yte)} "
          f"({100*yte.mean():.1f}%)")

    # ── Baseline 1: raw surprise as score (AUC direct)
    s_raw_test = sur_te.flatten()
    auc_raw = _roc_auc(yte, s_raw_test)

    # ── Baseline 2: Logreg(surprise)
    try:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=200)
        lr.fit(sur_tr, ytr)
        s_lr_test = lr.predict_proba(sur_te)[:, 1]
        auc_lr = _roc_auc(yte, s_lr_test)
    except Exception as e:
        print(f"Logreg unavailable: {e}")
        s_lr_test = None
        auc_lr = float("nan")

    # ── Baseline 3 (forte): GBDT 1D sur surprise
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        gbdt = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                           random_state=args.seed)
        gbdt.fit(sur_tr, ytr)
        s_gbdt_test = gbdt.predict_proba(sur_te)[:, 1]
        auc_gbdt = _roc_auc(yte, s_gbdt_test)
    except Exception as e:
        print(f"GBDT unavailable: {e}")
        s_gbdt_test = None
        auc_gbdt = float("nan")

    # ── Cible: MetaMLP avec 2 features
    mlp = _train_mlp(Xtr_s, ytr, Xv_s, yv, n_features=Xtr_s.shape[1],
                      seed=args.seed)
    mlp.eval()
    device = next(mlp.parameters()).device
    with torch.no_grad():
        s_mlp_test = torch.sigmoid(
            mlp(torch.from_numpy(Xte_s).to(device))).cpu().numpy()
    auc_mlp = _roc_auc(yte, s_mlp_test)

    # Best threshold from val for MLP
    with torch.no_grad():
        s_mlp_val = torch.sigmoid(
            mlp(torch.from_numpy(Xv_s).to(device))).cpu().numpy()
    thr = _best_threshold_on_val(yv, s_mlp_val)
    mlp_at_thr = _f1_at_threshold(yte, s_mlp_test, thr)
    brier_mlp = _brier(yte, s_mlp_test)

    # Bootstrap CI on the differences
    bs_results = {}
    if s_gbdt_test is not None:
        d, lo, hi = _bootstrap_ci(yte, s_mlp_test, s_gbdt_test, seed=args.seed)
        bs_results["mlp_minus_gbdt_1d"] = {"mean": d, "lo": lo, "hi": hi}
    if s_lr_test is not None:
        d, lo, hi = _bootstrap_ci(yte, s_mlp_test, s_lr_test, seed=args.seed)
        bs_results["mlp_minus_logreg_1d"] = {"mean": d, "lo": lo, "hi": hi}
    d, lo, hi = _bootstrap_ci(yte, s_mlp_test, s_raw_test, seed=args.seed)
    bs_results["mlp_minus_raw_surprise"] = {"mean": d, "lo": lo, "hi": hi}

    # Decorrelation test: subset where GBDT_1D errs at threshold 0.5 → MLP AUC
    decorr = None
    if s_gbdt_test is not None:
        gbdt_pred = (s_gbdt_test >= 0.5).astype(np.float32)
        wrong_mask = (gbdt_pred != yte)
        if wrong_mask.sum() >= 30:
            decorr = _roc_auc(yte[wrong_mask], s_mlp_test[wrong_mask])

    metrics = {
        "n_test": int(len(yte)),
        "feature_names": feat_names,
        "auc_raw_surprise": auc_raw,
        "auc_logreg_1d": auc_lr,
        "auc_gbdt_1d": auc_gbdt,
        "auc_mlp": auc_mlp,
        "mlp_at_threshold": mlp_at_thr,
        "brier_mlp": brier_mlp,
        "bootstrap_ci": bs_results,
        "decorrelation_auc_on_gbdt_errors": decorr,
    }
    with open(os.path.join(args.output, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    with open(os.path.join(args.output, "bootstrap_ci.json"), "w") as f:
        json.dump(bs_results, f, indent=2)
    if decorr is not None:
        with open(os.path.join(args.output, "decorrelation_test.json"), "w") as f:
            json.dump({"auc_on_gbdt_errors": decorr,
                       "n_gbdt_errors": int(wrong_mask.sum())}, f, indent=2)

    # ROC curves png
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
        fig, ax = plt.subplots(figsize=(6, 6))
        for label, scores in [
            ("raw surprise", s_raw_test),
            ("logreg(1D)", s_lr_test),
            ("GBDT(1D)", s_gbdt_test),
            ("MetaMLP(2D)", s_mlp_test),
        ]:
            if scores is None:
                continue
            fpr, tpr, _ = roc_curve(yte, scores)
            ax.plot(fpr, tpr, label=f"{label}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("Etape 1 — ROC curves")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.output, "roc_curves.png"), dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"  roc_curves.png skipped: {e}")

    # Verdict per spec C.8
    delta_gbdt = (auc_mlp - auc_gbdt) if not math.isnan(auc_gbdt) else None
    bs_gbdt = bs_results.get("mlp_minus_gbdt_1d", {})
    lines = ["# Verdict étape 1", ""]
    if delta_gbdt is None:
        lines.append("GBDT inaccessible — verdict reporté.")
    else:
        if (delta_gbdt >= 0.03 and bs_gbdt.get("lo", -1) > 0
                and decorr is not None and decorr > 0.55):
            lines.append(f"**GO étape 2.** AUC(MLP) − AUC(GBDT_1D) = "
                         f"{delta_gbdt:+.3f} (IC95 [{bs_gbdt['lo']:.3f}, "
                         f"{bs_gbdt['hi']:.3f}]), décorrélation AUC = {decorr:.3f}.")
            lines.append("Le MetaMLP exploite des features structurelles "
                         "au-delà de la surprise. Continuer en ajoutant "
                         "entropie + nesting_depth.")
        elif (delta_gbdt < 0):
            lines.append(f"**BUG.** AUC(MLP) = {auc_mlp:.3f} < AUC(GBDT_1D) = "
                         f"{auc_gbdt:.3f}. Investiguer l'implémentation.")
        elif (delta_gbdt < 0.01):
            lines.append(f"**NO-GO concept actuel.** Delta = {delta_gbdt:+.3f}. "
                         "Pas de signal au-delà de la non-linéarité de la surprise.")
        else:
            lines.append(f"**MARGINAL.** Delta = {delta_gbdt:+.3f}, IC95 "
                         f"[{bs_gbdt.get('lo', float('nan')):.3f}, "
                         f"{bs_gbdt.get('hi', float('nan')):.3f}]. "
                         "Re-run sur seeds 123, 456 avant verdict final.")
    with open(os.path.join(args.output, "verdict.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
