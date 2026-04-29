#!/usr/bin/env python3
"""ÉTAPE 3 — MetaEncoder C5 (tokens + entropy + surprise, SANS structure).

Mêmes hyperparams que meta_train_etape3_robust.py, mais features C5 = 2
scalaires uniquement. 3 seeds × 2 benchmarks. Génère la courbe risk-coverage
complète pour diagnostiquer l'incohérence 0.103 vs 0.4150 (résolue : c'était
la borne théorique avec err_rate=0.532, np.interp boundary value sur scores
bimodaux).

CONVENTION : AUC ∈ [0, 1] formaté :.4f. Risk ∈ [0, 1] formaté :.4f.
Coverage ∈ [0, 1] formaté :.4f. JAMAIS ×100.

Usage :
  python3 meta_step3_clean.py \\
      --cogs-dataset meta_data/cogs_meta_dataset_bplus.jsonl \\
      --cogs-splits  meta_data/cogs_meta_splits.json \\
      --slog-dataset meta_data/slog_meta_dataset.jsonl \\
      --slog-splits  meta_data/slog_meta_splits.json \\
      --output-dir   pipeline_clean/step3_meta_encoder \\
      --seeds 42,123,456
"""
import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# C5 = tokens + entropy + surprise (sans structure : pas de length, depth, rare_count)
SCALAR_FEATURES_C5 = ["entropy_mean_greedy", "surprise_mean"]
PAD_TOK = "<pad>"; UNK_TOK = "<unk>"; MAX_LEN = 64
DETERMINERS = {"A", "An", "The", "a", "an", "the"}
FUNCTION_WORDS = {"that", "was", "were", "by", "to", "in", "on", "."}
COVERAGE_TARGETS = [0.30, 0.50, 0.70, 0.80, 0.90, 0.95]


# ══════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════
def _load(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]

def _y(recs):
    return np.array([0 if r["exact_match"] else 1 for r in recs], dtype=np.float32)

def _cats(recs):
    return [r["category"] for r in recs]

def _features(recs, names):
    if not names:
        return np.zeros((len(recs), 0), dtype=np.float32)
    return np.array([[r["features_inference"][f] for f in names] for r in recs],
                    dtype=np.float32)

def _stand(X_train, X_other):
    if X_train.shape[1] == 0:
        return X_train, X_other, (np.zeros(0), np.ones(0))
    mu = X_train.mean(axis=0); sd = X_train.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (X_train - mu) / sd, (X_other - mu) / sd, (mu, sd)

def build_vocab(recs):
    counter = Counter()
    for r in recs:
        for t in r.get("input_tokens", []):
            counter[t] += 1
    tok2id = {PAD_TOK: 0, UNK_TOK: 1}
    for t, _ in counter.most_common():
        if t not in tok2id:
            tok2id[t] = len(tok2id)
    return tok2id

def tokenize(token_lists, tok2id, max_len=MAX_LEN):
    N = len(token_lists)
    ids = np.zeros((N, max_len), dtype=np.int64)
    mask = np.zeros((N, max_len), dtype=np.float32)
    for i, toks in enumerate(token_lists):
        for j, t in enumerate(toks[:max_len]):
            ids[i, j] = tok2id.get(t, tok2id[UNK_TOK])
            mask[i, j] = 1.0
    return ids, mask

def build_lexical_pools(recs):
    proper, common = set(), Counter()
    for r in recs:
        toks = r.get("input_tokens", [])
        for i, t in enumerate(toks):
            if not t or not t[0].isalpha(): continue
            if t in DETERMINERS or t in FUNCTION_WORDS: continue
            if t[0].isupper() and len(t) > 1 and t.isalpha():
                proper.add(t)
            elif t.islower() and t.isalpha() and len(t) > 1:
                if i > 0 and toks[i - 1] in DETERMINERS:
                    common[t] += 1
    return sorted(proper), sorted([t for t, _ in common.most_common(30)])

def perturb_tokens(tokens, proper_pool, common_pool, rng, p_perturb=0.5, force=False):
    if not force and rng.random() > p_perturb:
        return list(tokens)
    proper_used = sorted({t for t in tokens if t in proper_pool})
    common_used = sorted({t for i, t in enumerate(tokens)
                          if t in common_pool and i > 0 and tokens[i-1] in DETERMINERS})
    pmap, cmap = {}, {}
    if proper_used:
        cands = [p for p in proper_pool if p not in proper_used]
        if cands:
            picks = rng.choice(cands, size=len(proper_used),
                               replace=len(cands) < len(proper_used))
            pmap = {old: str(p) for old, p in zip(proper_used, picks)}
    if common_used:
        cands = [c for c in common_pool if c not in common_used]
        if cands:
            picks = rng.choice(cands, size=len(common_used),
                               replace=len(cands) < len(common_used))
            cmap = {old: str(c) for old, c in zip(common_used, picks)}
    out = []
    for i, t in enumerate(tokens):
        if t in pmap: out.append(pmap[t])
        elif (t in cmap and i > 0 and tokens[i-1] in DETERMINERS): out.append(cmap[t])
        else: out.append(t)
    return out


# ══════════════════════════════════════════════════════════════════════════
# MetaEncoder + train (mirror étape 3 robust)
# ══════════════════════════════════════════════════════════════════════════
class MetaEncoder(nn.Module):
    def __init__(self, vocab_size, n_scalar_features, d_model=64, n_heads=4,
                 n_layers=2, dropout=0.1, max_len=MAX_LEN):
        super().__init__()
        self.n_scalar_features = n_scalar_features
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=128,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        hidden = 128
        self.head = nn.Sequential(
            nn.Linear(d_model + n_scalar_features, hidden),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, input_ids, input_mask, scalar_features):
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        x = self.encoder(x, src_key_padding_mask=~input_mask.bool())
        m = input_mask.unsqueeze(-1).float()
        pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        if self.n_scalar_features == 0:
            combined = pooled
        else:
            combined = torch.cat([pooled, scalar_features], dim=-1)
        return self.head(combined).squeeze(-1)


def _set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one(train_recs, val_recs, tok2id, feat_names, seed,
              proper_pool, common_pool, p_perturb=0.5,
              lr=3e-4, batch_size=64, max_epochs=50, patience=8,
              log_prefix=""):
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr = _features(train_recs, feat_names)
    Xv = _features(val_recs, feat_names)
    Xtr_s, Xv_s, _ = _stand(Xtr, Xv)
    ytr = _y(train_recs); yv = _y(val_recs)
    ids_v, mask_v = tokenize([r["input_tokens"] for r in val_recs], tok2id)

    n_scalar = Xtr.shape[1]
    model = MetaEncoder(len(tok2id), n_scalar).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    Iv = torch.from_numpy(ids_v).to(device)
    Mv = torch.from_numpy(mask_v).to(device)
    Sv = torch.from_numpy(Xv_s).to(device)
    Str = torch.from_numpy(Xtr_s).to(device)
    Ytr = torch.from_numpy(ytr).to(device)

    rng = np.random.default_rng(seed)
    train_tokens = [r["input_tokens"] for r in train_recs]
    best_auc, best_state, pat = -1.0, None, 0
    N = len(train_recs)
    for ep in range(max_epochs):
        t0 = time.time()
        perturbed = [perturb_tokens(t, proper_pool, common_pool, rng,
                                      p_perturb=p_perturb) for t in train_tokens]
        ids_tr, mask_tr = tokenize(perturbed, tok2id)
        Itr = torch.from_numpy(ids_tr).to(device)
        Mtr = torch.from_numpy(mask_tr).to(device)
        model.train()
        idx = torch.randperm(N, device=device)
        loss_sum = 0.0; nb = 0
        for st in range(0, N, batch_size):
            sl = idx[st:st + batch_size]
            logits = model(Itr[sl], Mtr[sl], Str[sl])
            loss = F.binary_cross_entropy_with_logits(logits, Ytr[sl])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item()); nb += 1
        model.eval()
        with torch.no_grad():
            sv = torch.sigmoid(model(Iv, Mv, Sv)).cpu().numpy()
        auc = _auc(yv, sv)
        marker = "★" if auc > best_auc else " "
        print(f"      {log_prefix}E{ep:02d} {marker} loss={loss_sum/max(nb,1):.4f}  "
              f"val_auc={auc:.4f}  pat={pat}  ({time.time()-t0:.1f}s)", flush=True)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience:
                print(f"      {log_prefix}early stop ep {ep}", flush=True)
                break
    model.load_state_dict(best_state)
    return model


def _score(model, ids, mask, sc):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        return torch.sigmoid(model(
            torch.from_numpy(ids).to(device),
            torch.from_numpy(mask).to(device),
            torch.from_numpy(sc).to(device))).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════
# Metrics (AUC ∈ [0,1], risk ∈ [0,1], coverage ∈ [0,1])
# ══════════════════════════════════════════════════════════════════════════
def _auc(y, s):
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(y)) < 2: return float("nan")
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def find_mixed_cats(y, cats, n_min=20):
    cat_to_idx = defaultdict(list)
    for i, c in enumerate(cats):
        cat_to_idx[c].append(i)
    mixed = []
    for cat, idxs in cat_to_idx.items():
        idxs = np.array(idxs)
        n_err = int(y[idxs].sum())
        n_cor = int(len(idxs) - n_err)
        if n_err >= n_min and n_cor >= n_min:
            mixed.append(cat)
    return mixed


def _macro_intra(scores, y, cats, mixed):
    aucs = []
    for cat in mixed:
        m = np.array([c == cat for c in cats])
        if m.sum() < 5 or len(set(y[m])) < 2: continue
        a = _auc(y[m], scores[m])
        if not math.isnan(a): aucs.append(a)
    return float(np.mean(aucs)) if aucs else float("nan")


def risk_coverage_curve(y, scores, n_points=100):
    """Sort by score asc, compute (coverage, risk) for n_points cumulative
    keep-fractions. coverage et risk ∈ [0, 1]."""
    n = len(y)
    order = np.argsort(scores)
    y_sorted = y[order]
    cumulative_errors = np.cumsum(y_sorted)
    coverages = np.arange(1, n + 1) / n
    risks = cumulative_errors / np.arange(1, n + 1)
    # Sample n_points
    if n > n_points:
        step = max(1, n // n_points)
        idx = np.arange(0, n, step)
    else:
        idx = np.arange(n)
    return coverages[idx], risks[idx]


def risk_at_coverage_exact(y, scores, target):
    """Méthode TEST 2 : keep int(target * N) examples with lowest scores.
    Pas d'interpolation. Renvoie risk ∈ [0, 1]."""
    n = len(y)
    n_target = int(target * n)
    if n_target == 0:
        return float("nan")
    order = np.argsort(scores)
    kept = order[:n_target]
    return float(y[kept].mean())


def risk_lower_bound(err_rate, coverage):
    """Borne théorique avec perfect AUC : (err_rate - (1 - coverage)) / coverage."""
    if coverage <= 0: return float("nan")
    return max(0.0, err_rate - (1.0 - coverage)) / coverage


def _aurc(y, s):
    cov, risk = risk_coverage_curve(y, s, n_points=200)
    if len(cov) < 2:
        return float("nan")
    return float(0.5 * np.sum((cov[1:] - cov[:-1]) * (risk[1:] + risk[:-1])))


# ══════════════════════════════════════════════════════════════════════════
# Single (config, seed, benchmark) run
# ══════════════════════════════════════════════════════════════════════════
def run_config(records, splits, bench_name, seed, do_loco):
    train_recs = [records[i] for i in splits["train"]]
    val_recs = [records[i] for i in splits["val"]]
    test_recs = [records[i] for i in splits["test"]]
    tok2id = build_vocab(train_recs)
    proper_pool, common_pool = build_lexical_pools(train_recs)

    Xtr = _features(train_recs, SCALAR_FEATURES_C5)
    Xv = _features(val_recs, SCALAR_FEATURES_C5)
    Xte = _features(test_recs, SCALAR_FEATURES_C5)
    Xtr_s, Xte_s, (mu, sd) = _stand(Xtr, Xte)
    Xv_s = (Xv - mu) / sd
    yte = _y(test_recs); cats_te = _cats(test_recs)

    print(f"\n[{bench_name}/s{seed}/C5] training MetaEncoder (features={SCALAR_FEATURES_C5})...")
    model = train_one(train_recs, val_recs, tok2id, SCALAR_FEATURES_C5, seed,
                       proper_pool, common_pool,
                       log_prefix=f"[{bench_name}/s{seed}/orig] ")

    ids_te, mask_te = tokenize([r["input_tokens"] for r in test_recs], tok2id)
    s_orig = _score(model, ids_te, mask_te, Xte_s)
    auc_orig = _auc(yte, s_orig)

    rng_eval = np.random.default_rng(seed)
    pert = [perturb_tokens(r["input_tokens"], proper_pool, common_pool,
                            rng_eval, force=True) for r in test_recs]
    ids_pe, mask_pe = tokenize(pert, tok2id)
    s_pert = _score(model, ids_pe, mask_pe, Xte_s)
    auc_pert = _auc(yte, s_pert)

    mixed = find_mixed_cats(yte, cats_te, n_min=20)
    macro_orig = _macro_intra(s_orig, yte, cats_te, mixed)
    macro_pert = _macro_intra(s_pert, yte, cats_te, mixed)

    # Risk-coverage à plusieurs targets (méthode TEST 2 : exact, sans interp)
    risk_at = {f"{t:.2f}": risk_at_coverage_exact(yte, s_orig, t)
               for t in COVERAGE_TARGETS}
    err_rate = float(yte.mean())
    risk_lb = {f"{t:.2f}": risk_lower_bound(err_rate, t) for t in COVERAGE_TARGETS}

    # Courbe complète pour le PNG (200 points)
    cov_curve, risk_curve = risk_coverage_curve(yte, s_orig, n_points=200)

    aurc = _aurc(yte, s_orig)

    out = {
        "config": "C5_tokens_ent_sur",
        "bench": bench_name,
        "seed": seed,
        "feature_names": SCALAR_FEATURES_C5,
        "auc_orig": auc_orig,
        "auc_pert": auc_pert,
        "macro_intra_orig": macro_orig,
        "macro_intra_pert": macro_pert,
        "aurc": aurc,
        "err_rate_test": err_rate,
        "risk_at_coverage": risk_at,
        "risk_lower_bound": risk_lb,
        "n_test": int(len(yte)),
        "n_errors_test": int(yte.sum()),
        "mixed_cats": mixed,
        "n_mixed_cats": len(mixed),
        "cov_curve": cov_curve.tolist(),
        "risk_curve": risk_curve.tolist(),
    }

    if do_loco and mixed:
        train_loco = [r for r in train_recs if r["category"] not in mixed]
        if len(train_loco) > 100:
            print(f"\n[{bench_name}/s{seed}/C5/loco] training without mixed cats...")
            model_loco = train_one(train_loco, val_recs, tok2id, SCALAR_FEATURES_C5,
                                     seed, proper_pool, common_pool,
                                     log_prefix=f"[{bench_name}/s{seed}/loco] ")
            s_loco = _score(model_loco, ids_te, mask_te, Xte_s)
            out["auc_loco_global"] = _auc(yte, s_loco)
        else:
            out["auc_loco_global"] = float("nan")
    else:
        out["auc_loco_global"] = None
    return out


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cogs-dataset", required=True)
    ap.add_argument("--cogs-splits", required=True)
    ap.add_argument("--slog-dataset", required=True)
    ap.add_argument("--slog-splits", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--skip-loco", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    do_loco = not args.skip_loco

    bench_specs = [
        ("cogs", args.cogs_dataset, args.cogs_splits),
        ("slog", args.slog_dataset, args.slog_splits),
    ]

    all_results = {}
    for bench_name, dpath, spath in bench_specs:
        print(f"\n{'═'*60}\n  BENCHMARK : {bench_name}\n{'═'*60}")
        records = _load(dpath)
        with open(spath, "r") as f:
            splits = json.load(f)
        bench_results = []
        for seed in seeds:
            try:
                res = run_config(records, splits, bench_name, seed, do_loco)
                print(f"[{bench_name}/s{seed}] AUC={res['auc_orig']:.4f}  "
                      f"AUC_pert={res['auc_pert']:.4f}  "
                      f"macro={res['macro_intra_orig']:.4f}  "
                      f"AURC={res['aurc']:.4f}", flush=True)
                bench_results.append(res)
            except Exception as e:
                print(f"FAILED [{bench_name}/s{seed}]: {e}", flush=True)
                bench_results.append({"bench": bench_name, "seed": seed, "error": str(e)})
        all_results[bench_name] = bench_results
        # Strip cov_curve and risk_curve from saved JSON to keep size reasonable
        light = []
        for r in bench_results:
            r2 = {k: v for k, v in r.items() if k not in ("cov_curve", "risk_curve")}
            light.append(r2)
        with open(os.path.join(args.output_dir, f"results_{bench_name}.json"), "w") as f:
            json.dump(light, f, indent=2)

    # ── Plot risk-coverage curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, bench in zip(axes, ("cogs", "slog")):
            for r in all_results[bench]:
                if "cov_curve" not in r: continue
                cov = np.array(r["cov_curve"]); rk = np.array(r["risk_curve"])
                ax.plot(cov, rk, alpha=0.6, label=f"seed {r['seed']} (AUC={r['auc_orig']:.3f})")
            err_rate = all_results[bench][0].get("err_rate_test", 0)
            cov_ref = np.linspace(0.01, 1.0, 100)
            lb = np.array([risk_lower_bound(err_rate, c) for c in cov_ref])
            ax.plot(cov_ref, lb, "k--", alpha=0.5, label=f"borne LB (perfect AUC, err={err_rate:.3f})")
            for t in COVERAGE_TARGETS:
                ax.axvline(t, color="gray", linestyle=":", alpha=0.3)
            ax.set_xlabel("coverage"); ax.set_ylabel("risk")
            ax.set_title(f"{bench.upper()} — risk-coverage (C5)")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.legend(loc="upper left", fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "risk_coverage_curves.png"), dpi=120)
        plt.close(fig)
        print(f"Wrote risk_coverage_curves.png")
    except Exception as e:
        print(f"plot skipped: {e}")

    # ── Diagnostic SLOG (cohérence 0.103 vs 0.4150)
    slog_diag = ["# SLOG risk@coverage diagnosis", ""]
    slog_diag.append("CONVENTION : risk et coverage ∈ [0, 1]. err_rate test = mean(yte).")
    slog_diag.append("")
    slog_diag.append(f"## Borne théorique (perfect AUC)")
    slog_diag.append("")
    slog_diag.append("`risk_lb(coverage) = max(0, err_rate - (1 - coverage)) / coverage`")
    slog_diag.append("")
    if all_results.get("slog"):
        slog_first = all_results["slog"][0]
        if "err_rate_test" in slog_first:
            err = slog_first["err_rate_test"]
            slog_diag.append(f"err_rate SLOG = **{err:.4f}**")
            slog_diag.append("")
            slog_diag.append("| coverage | risk_lb théorique | risk observé (seed 42) | seed 123 | seed 456 |")
            slog_diag.append("|---|---:|---:|---:|---:|")
            for t in COVERAGE_TARGETS:
                lb = risk_lower_bound(err, t)
                vals = []
                for r in all_results["slog"]:
                    rs = r.get("risk_at_coverage", {}).get(f"{t:.2f}")
                    vals.append(f"{rs:.4f}" if rs is not None else "N/A")
                slog_diag.append(f"| {t:.2f} | {lb:.4f} | {' | '.join(vals)} |")
            slog_diag.append("")
            slog_diag.append("## Lecture")
            slog_diag.append("")
            slog_diag.append("Le 0.4150 figé sur 6 configs en TEST 2 est **la borne théorique** "
                             "(plafond mathématique avec perfect AUC) à coverage=0.80, pas un bug. "
                             "À coverage 0.30-0.50 le risk varie réellement selon l'AUC.")
    with open(os.path.join(args.output_dir, "slog_risk_diagnosis.md"), "w") as f:
        f.write("\n".join(slog_diag))

    # ── Comparison table
    write_comparison(all_results, args.output_dir)
    write_verdict(all_results, args.output_dir)
    print(f"\nAll outputs in {args.output_dir}/")


def write_comparison(all_results, out_dir):
    lines = ["# ÉTAPE 3 — MetaEncoder C5 (sans structure) — comparaison", ""]
    lines.append("AUC ∈ [0, 1]. mean ± std sur 3 seeds.")
    lines.append("")
    for bench, results in all_results.items():
        ok = [r for r in results if "auc_orig" in r]
        if not ok:
            continue
        lines.append(f"## {bench.upper()}")
        lines.append("")
        lines.append("| Métrique | mean | std |")
        lines.append("|---|---:|---:|")
        def _ms(key):
            vals = [r[key] for r in ok if r.get(key) is not None
                    and not (isinstance(r[key], float) and math.isnan(r[key]))]
            if not vals:
                return float("nan"), float("nan")
            return float(np.mean(vals)), (float(np.std(vals)) if len(vals) > 1 else 0.0)
        for k in ("auc_orig", "auc_pert", "macro_intra_orig", "macro_intra_pert",
                  "aurc", "auc_loco_global"):
            m, s = _ms(k)
            if math.isnan(m): continue
            lines.append(f"| {k} | {m:.4f} | {s:.4f} |")
        lines.append("")
        lines.append("### Risk @ coverage (mean across seeds)")
        lines.append("")
        lines.append("| coverage | risk mean | risk_lb théorique |")
        lines.append("|---|---:|---:|")
        if ok:
            err = float(np.mean([r.get("err_rate_test", 0) for r in ok]))
            for t in COVERAGE_TARGETS:
                vals = [r.get("risk_at_coverage", {}).get(f"{t:.2f}") for r in ok]
                vals = [v for v in vals if v is not None]
                m = float(np.mean(vals)) if vals else float("nan")
                lb = risk_lower_bound(err, t)
                lines.append(f"| {t:.2f} | {m:.4f} | {lb:.4f} |")
        lines.append("")
    with open(os.path.join(out_dir, "comparison_with_full.md"), "w") as f:
        f.write("\n".join(lines))


def write_verdict(all_results, out_dir):
    """Compare C5 (this run) vs C6 full (TEST 2 results) en macro intra-cat
    si TEST 2 a tourné."""
    lines = ["# ÉTAPE 3 — Verdict (cohérence vs TEST 2 C6 full)", ""]
    # Try loading TEST 2 results
    test2_path = "tests/test2_ablation"
    deltas = {}
    for bench in ("cogs", "slog"):
        new_macros = [r["macro_intra_orig"] for r in all_results.get(bench, [])
                      if "macro_intra_orig" in r and not math.isnan(r["macro_intra_orig"])]
        if not new_macros:
            continue
        new_mean = float(np.mean(new_macros))
        # TEST 2 C6
        t2_path = os.path.join(test2_path, f"results_{bench}.json")
        if os.path.exists(t2_path):
            with open(t2_path, "r") as f:
                t2 = json.load(f)
            c6_macros = [r["macro_intra_orig"] for r in t2
                         if r.get("config") == "C6_full"
                         and "macro_intra_orig" in r
                         and not math.isnan(r["macro_intra_orig"])]
            if c6_macros:
                c6_mean = float(np.mean(c6_macros))
                deltas[bench] = {"c5_mean": new_mean, "c6_mean": c6_mean,
                                  "delta": new_mean - c6_mean}

    lines.append("Comparaison macro intra-cat AUC : C5 (sans structure) vs C6 (full).")
    lines.append("")
    lines.append("| Bench | C5 macro | C6 macro (TEST 2) | Δ |")
    lines.append("|---|---:|---:|---:|")
    for b, d in deltas.items():
        lines.append(f"| {b.upper()} | {d['c5_mean']:.4f} | {d['c6_mean']:.4f} | "
                     f"{d['delta']:+.4f} |")
    lines.append("")

    if not deltas:
        lines.append("**Verdict reporté** — TEST 2 results non disponibles pour comparaison.")
    else:
        cogs_d = deltas.get("cogs", {}).get("delta", float("nan"))
        slog_d = deltas.get("slog", {}).get("delta", float("nan"))
        if (not math.isnan(cogs_d) and not math.isnan(slog_d)
                and -0.005 <= cogs_d <= 0.005 and -0.005 <= slog_d <= 0.005):
            verdict = "COHÉRENT"
            reason = "Δ macro ∈ [-0.005, +0.005] sur les 2 benchmarks. Pipeline propre validé."
        elif (cogs_d < -0.01 or slog_d < -0.01):
            verdict = "DÉGRADATION"
            reason = (f"COGS Δ={cogs_d:+.4f}, SLOG Δ={slog_d:+.4f}. "
                      "La feature structure contribuait via mécanisme non-vu en TEST 2.")
        elif (cogs_d > 0.01 or slog_d > 0.01):
            verdict = "AMÉLIORATION INATTENDUE"
            reason = (f"COGS Δ={cogs_d:+.4f}, SLOG Δ={slog_d:+.4f}. "
                      "Logger. Possible interaction négative entre structure et autres.")
        else:
            verdict = "MIXTE"
            reason = f"COGS Δ={cogs_d:+.4f}, SLOG Δ={slog_d:+.4f}."
        lines.append(f"## Verdict : {verdict}")
        lines.append("")
        lines.append(reason)

    with open(os.path.join(out_dir, "verdict.txt"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
