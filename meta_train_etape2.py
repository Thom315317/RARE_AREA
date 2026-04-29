#!/usr/bin/env python3
"""Phase C — étape 2 du meta-modèle.

Tester si `entropy_mean_greedy` apporte un signal orthogonal à `surprise_mean`,
là où `input_length` (étape 1, corr 0.92) avait échoué (decorrelation 0.247).

Hypothèse : surprise (input) et entropie (génération) capturent des phénomènes
différents et seront moins collinéaires.

6 modèles comparés sur meta-test :
  1. raw_surprise (AUC direct)
  2. logreg(surprise_mean)
  3. GBDT_1D(surprise_mean)            ← baseline forte étape 1
  4. raw_entropy (AUC direct)          ← NOUVEAU
  5. GBDT_1D(entropy_mean_greedy)      ← NOUVEAU
  6. MetaMLP(surprise_mean + entropy_mean_greedy)

Critère principal (§5 du prompt) : decorrelation_auc(MLP | GBDT_1D-surprise) > 0.55.
Critère secondaire : AUC(MLP) − AUC(GBDT_1D-surprise) ≥ 0.03 IC>0.

Usage:
  python3 meta_train_etape2.py \\
      --dataset meta_data/cogs_meta_dataset.jsonl \\
      --splits  meta_data/cogs_meta_splits.json \\
      --output  runs_meta/etape2_s42 \\
      --seed 42
"""
import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURES_ETAPE2 = ["surprise_mean", "entropy_mean_greedy"]


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
                    dtype=np.float32)


def _standardize(X_train, X_other):
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (X_train - mu) / sd, (X_other - mu) / sd, (mu, sd)


# ══════════════════════════════════════════════════════════════════════════
# MetaMLP (identique étape 1)
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
    return (float(diffs.mean()),
            float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)))


def _gbdt_1d(x_train, y_train, x_test, seed):
    """Returns (scores_test, model) or (None, None) if sklearn unavailable."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        gbdt = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                          random_state=seed)
        gbdt.fit(x_train, y_train)
        return gbdt.predict_proba(x_test)[:, 1], gbdt
    except Exception as e:
        print(f"GBDT unavailable: {e}")
        return None, None


def _full_metrics(name, y_test, y_val, scores_test, scores_val):
    """AUC + accuracy/precision/recall/F1 at val-best threshold + Brier."""
    auc = _roc_auc(y_test, scores_test)
    thr = _best_threshold_on_val(y_val, scores_val) if scores_val is not None else 0.5
    at_thr = _f1_at_threshold(y_test, scores_test, thr)
    brier = _brier(y_test, scores_test)
    return {
        "name": name,
        "auc": auc,
        "at_threshold": at_thr,
        "brier": brier,
    }


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

    feat_names = FEATURES_ETAPE2
    Xtr = _features_matrix(train_recs, feat_names)
    Xv = _features_matrix(val_recs, feat_names)
    Xte = _features_matrix(test_recs, feat_names)
    ytr = _labels(train_recs)
    yv = _labels(val_recs)
    yte = _labels(test_recs)

    Xtr_s, _, (mu, sd) = _standardize(Xtr, Xv)
    Xv_s = (Xv - mu) / sd
    Xte_s = (Xte - mu) / sd

    sur_idx = feat_names.index("surprise_mean")
    ent_idx = feat_names.index("entropy_mean_greedy")
    sur_tr = Xtr[:, sur_idx:sur_idx + 1]
    sur_v = Xv[:, sur_idx:sur_idx + 1]
    sur_te = Xte[:, sur_idx:sur_idx + 1]
    ent_tr = Xtr[:, ent_idx:ent_idx + 1]
    ent_v = Xv[:, ent_idx:ent_idx + 1]
    ent_te = Xte[:, ent_idx:ent_idx + 1]

    # ── §2.1 corrélation features sur meta-train
    corr_train = float(np.corrcoef(Xtr[:, sur_idx], Xtr[:, ent_idx])[0, 1])
    corr_alert = abs(corr_train) > 0.85
    print(f"\nCorrélation surprise_mean × entropy_mean_greedy "
          f"(meta-train) : {corr_train:.3f}"
          + ("   ⚠️  > 0.85" if corr_alert else "   (< 0.85, signal probablement orthogonal)"))
    with open(os.path.join(args.output, "feature_correlation.txt"), "w") as f:
        f.write(f"corr(surprise_mean, entropy_mean_greedy) = {corr_train:.6f}\n")
        f.write(f"alert (>0.85): {corr_alert}\n")

    print(f"\nLoaded {len(records)} records (train={len(train_recs)}, "
          f"val={len(val_recs)}, test={len(test_recs)})")
    print(f"Class balance test: errors={int(yte.sum())}/{len(yte)} "
          f"({100*yte.mean():.1f}%)")

    # ── 1. raw_surprise
    s_raw_sur_te = sur_te.flatten()
    s_raw_sur_v = sur_v.flatten()
    m_raw_sur = _full_metrics("raw_surprise", yte, yv, s_raw_sur_te, s_raw_sur_v)

    # ── 2. logreg(surprise)
    try:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=200)
        lr.fit(sur_tr, ytr)
        s_lr_te = lr.predict_proba(sur_te)[:, 1]
        s_lr_v = lr.predict_proba(sur_v)[:, 1]
        m_lr = _full_metrics("logreg_surprise", yte, yv, s_lr_te, s_lr_v)
    except Exception as e:
        print(f"Logreg unavailable: {e}")
        s_lr_te = None
        m_lr = None

    # ── 3. GBDT_1D(surprise)
    s_gbdt_sur_te, _ = _gbdt_1d(sur_tr, ytr, sur_te, args.seed)
    s_gbdt_sur_v, _ = _gbdt_1d(sur_tr, ytr, sur_v, args.seed)
    m_gbdt_sur = _full_metrics("gbdt_1d_surprise", yte, yv,
                               s_gbdt_sur_te, s_gbdt_sur_v) if s_gbdt_sur_te is not None else None

    # ── 4. raw_entropy (NOUVEAU)
    s_raw_ent_te = ent_te.flatten()
    s_raw_ent_v = ent_v.flatten()
    m_raw_ent = _full_metrics("raw_entropy", yte, yv, s_raw_ent_te, s_raw_ent_v)

    # ── 5. GBDT_1D(entropy) (NOUVEAU)
    s_gbdt_ent_te, _ = _gbdt_1d(ent_tr, ytr, ent_te, args.seed)
    s_gbdt_ent_v, _ = _gbdt_1d(ent_tr, ytr, ent_v, args.seed)
    m_gbdt_ent = _full_metrics("gbdt_1d_entropy", yte, yv,
                               s_gbdt_ent_te, s_gbdt_ent_v) if s_gbdt_ent_te is not None else None

    # ── 6. MetaMLP(surprise + entropy)
    mlp = _train_mlp(Xtr_s, ytr, Xv_s, yv, n_features=Xtr_s.shape[1],
                     seed=args.seed)
    mlp.eval()
    device = next(mlp.parameters()).device
    with torch.no_grad():
        s_mlp_te = torch.sigmoid(
            mlp(torch.from_numpy(Xte_s).to(device))).cpu().numpy()
        s_mlp_v = torch.sigmoid(
            mlp(torch.from_numpy(Xv_s).to(device))).cpu().numpy()
    m_mlp = _full_metrics("mlp_2feat", yte, yv, s_mlp_te, s_mlp_v)

    # ── Bootstrap CI sur les deltas (vs GBDT-surprise et GBDT-entropy)
    bs_results = {}
    if s_gbdt_sur_te is not None:
        d, lo, hi = _bootstrap_ci(yte, s_mlp_te, s_gbdt_sur_te, seed=args.seed)
        bs_results["mlp_minus_gbdt_surprise"] = {"mean": d, "lo": lo, "hi": hi}
    if s_gbdt_ent_te is not None:
        d, lo, hi = _bootstrap_ci(yte, s_mlp_te, s_gbdt_ent_te, seed=args.seed)
        bs_results["mlp_minus_gbdt_entropy"] = {"mean": d, "lo": lo, "hi": hi}
    d, lo, hi = _bootstrap_ci(yte, s_mlp_te, s_raw_sur_te, seed=args.seed)
    bs_results["mlp_minus_raw_surprise"] = {"mean": d, "lo": lo, "hi": hi}
    d, lo, hi = _bootstrap_ci(yte, s_mlp_te, s_raw_ent_te, seed=args.seed)
    bs_results["mlp_minus_raw_entropy"] = {"mean": d, "lo": lo, "hi": hi}

    # ── Decorrelation tests (§4)
    decorr = {}
    for label, scores in [("gbdt_surprise", s_gbdt_sur_te),
                          ("gbdt_entropy", s_gbdt_ent_te)]:
        if scores is None:
            continue
        gbdt_pred = (scores >= 0.5).astype(np.float32)
        wrong_mask = (gbdt_pred != yte)
        if wrong_mask.sum() >= 30:
            decorr[label] = {
                "n_errors": int(wrong_mask.sum()),
                "auc_mlp_on_errors": _roc_auc(yte[wrong_mask],
                                              s_mlp_te[wrong_mask]),
            }

    # ── Aggregate metrics dump
    all_models = [m_raw_sur, m_lr, m_gbdt_sur, m_raw_ent, m_gbdt_ent, m_mlp]
    metrics = {
        "n_test": int(len(yte)),
        "feature_names": feat_names,
        "feature_correlation_train": corr_train,
        "feature_correlation_alert": corr_alert,
        "models": [m for m in all_models if m is not None],
        "bootstrap_ci": bs_results,
        "decorrelation_tests": decorr,
    }
    with open(os.path.join(args.output, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    with open(os.path.join(args.output, "bootstrap_ci.json"), "w") as f:
        json.dump(bs_results, f, indent=2)
    if decorr:
        with open(os.path.join(args.output, "decorrelation_test.json"), "w") as f:
            json.dump(decorr, f, indent=2)

    # ── ROC curves png (6 modèles)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
        fig, ax = plt.subplots(figsize=(7, 7))
        for label, scores in [
            ("raw_surprise", s_raw_sur_te),
            ("logreg(surprise)", s_lr_te),
            ("GBDT_1D(surprise)", s_gbdt_sur_te),
            ("raw_entropy", s_raw_ent_te),
            ("GBDT_1D(entropy)", s_gbdt_ent_te),
            ("MetaMLP(2D)", s_mlp_te),
        ]:
            if scores is None:
                continue
            fpr, tpr, _ = roc_curve(yte, scores)
            this_auc = _roc_auc(yte, scores)
            ax.plot(fpr, tpr, label=f"{label}  (AUC={this_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("Étape 2 — ROC (surprise + entropy)")
        ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.output, "roc_curves.png"), dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"  roc_curves.png skipped: {e}")

    # ── Verdict §5
    auc_mlp = m_mlp["auc"]
    auc_baselines_1d = [m["auc"] for m in
                        [m_raw_sur, m_lr, m_gbdt_sur, m_raw_ent, m_gbdt_ent]
                        if m is not None and not math.isnan(m["auc"])]
    max_baseline = max(auc_baselines_1d) if auc_baselines_1d else float("nan")

    decorr_sur = decorr.get("gbdt_surprise", {}).get("auc_mlp_on_errors")
    bs_vs_gbdt_sur = bs_results.get("mlp_minus_gbdt_surprise", {})
    delta_vs_gbdt_sur = (auc_mlp - m_gbdt_sur["auc"]) if m_gbdt_sur else None
    auc_gbdt_ent = m_gbdt_ent["auc"] if m_gbdt_ent else float("nan")

    lines = ["# Verdict étape 2", ""]
    lines.append(f"- Features : {feat_names}")
    lines.append(f"- Corr(surprise, entropy) sur meta-train : **{corr_train:.3f}**"
                 + ("  ⚠️ >0.85 (alerte)" if corr_alert else ""))
    lines.append(f"- AUC MLP = **{auc_mlp:.4f}**, max baseline 1D = {max_baseline:.4f}")
    if decorr_sur is not None:
        lines.append(f"- Décorrélation MLP | GBDT_1D(surprise) erreurs : **{decorr_sur:.3f}** "
                     f"(critère principal : > 0.55)")
    if delta_vs_gbdt_sur is not None and bs_vs_gbdt_sur:
        lines.append(f"- Δ MLP − GBDT_1D(surprise) = {delta_vs_gbdt_sur:+.4f} "
                     f"IC95 [{bs_vs_gbdt_sur.get('lo', float('nan')):.4f}, "
                     f"{bs_vs_gbdt_sur.get('hi', float('nan')):.4f}]")
    lines.append("")

    if not math.isnan(auc_mlp) and not math.isnan(auc_gbdt_ent) and auc_mlp < auc_gbdt_ent:
        lines.append(f"**BUG** : AUC(MLP) = {auc_mlp:.4f} < AUC(GBDT_1D-entropy) "
                     f"= {auc_gbdt_ent:.4f}. Investiguer.")
    elif (decorr_sur is not None and decorr_sur > 0.55
          and not math.isnan(max_baseline)
          and (auc_mlp - max_baseline) >= 0.03
          and bs_vs_gbdt_sur.get("lo", -1) > 0):
        lines.append("**GO étape 3** — signal orthogonal confirmé : decorrélation > 0.55 "
                     "et MLP > max(baselines 1D) + 0.03 avec IC>0.")
        lines.append("Continuer avec encodeur léger sur input.")
    elif decorr_sur is not None and decorr_sur > 0.55:
        lines.append("**PARTIEL** : décorrélation > 0.55 mais AUC(MLP) ≈ max baseline 1D. "
                     "L'entropie seule capture l'essentiel ; le MLP combine peu. "
                     "À documenter avec Julien avant étape 3.")
    elif decorr_sur is not None:
        lines.append(f"**NO-GO** : décorrélation = {decorr_sur:.3f} ≤ 0.55. "
                     "Même résultat qu'étape 1, le concept 'features scalaires combinées' "
                     "ne tient pas. Repenser ou passer à étape 3 (encodeur sur input).")
    else:
        lines.append("Test de décorrélation indisponible — verdict reporté.")

    if (m_raw_ent is not None and not math.isnan(m_raw_ent["auc"])
            and m_raw_ent["auc"] > 0.95):
        lines.append("")
        lines.append(f"⚠️ raw_entropy AUC = {m_raw_ent['auc']:.3f} (>0.95) : "
                     "l'entropie greedy seule porte presque tout le signal. "
                     "Probablement parce qu'elle reflète directement l'incertitude "
                     "générative au moment de l'erreur. À discuter avec Julien "
                     "avant de continuer.")

    with open(os.path.join(args.output, "verdict.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
