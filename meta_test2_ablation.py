#!/usr/bin/env python3
"""TEST 2 — Ablation factorielle MetaEncoder.

6 configs × 3 seeds × 2 benchmarks (COGS, SLOG). Architecture MetaEncoder
identique pour les 6 configs ; seuls les features scalaires concaténés
changent. Si une config a 0 feature scalaire, n_scalar_features = 0
(MetaEncoder se contente de pooled_tokens → head).

Configs :
  C1 — tokens-only          : []
  C2 — tokens+structure     : [input_length, nesting_depth, rare_token_count]
  C3 — tokens+entropy       : [entropy_mean_greedy]
  C4 — tokens+surprise      : [surprise_mean]
  C5 — tokens+ent+surprise  : [entropy_mean_greedy, surprise_mean]
  C6 — full                 : [surprise_mean, entropy_mean_greedy,
                               rare_token_count, nesting_depth]

CONVENTION : tous les AUC ∈ [0, 1] formatés :.4f. JAMAIS ×100. JAMAIS %.

Usage :
  python3 meta_test2_ablation.py \\
      --cogs-dataset meta_data/cogs_meta_dataset_bplus.jsonl \\
      --cogs-splits  meta_data/cogs_meta_splits.json \\
      --slog-dataset meta_data/slog_meta_dataset.jsonl \\
      --slog-splits  meta_data/slog_meta_splits.json \\
      --output-dir tests/test2_ablation \\
      --seeds 42,123,456
"""
import argparse
import json
import math
import os
import random
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PAD_TOK = "<pad>"
UNK_TOK = "<unk>"
MAX_LEN = 64
DETERMINERS = {"A", "An", "The", "a", "an", "the"}
FUNCTION_WORDS = {"that", "was", "were", "by", "to", "in", "on", "."}

CONFIGS = {
    "C1_tokens_only":      [],
    "C2_tokens_struct":    ["input_length", "nesting_depth", "rare_token_count"],
    "C3_tokens_entropy":   ["entropy_mean_greedy"],
    "C4_tokens_surprise":  ["surprise_mean"],
    "C5_tokens_ent_sur":   ["entropy_mean_greedy", "surprise_mean"],
    "C6_full":             ["surprise_mean", "entropy_mean_greedy",
                            "rare_token_count", "nesting_depth"],
}


# ══════════════════════════════════════════════════════════════════════════
# Data + tokenization
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
    return np.array(
        [[r["features_inference"][f] for f in names] for r in recs],
        dtype=np.float32,
    )


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
            if not t or not t[0].isalpha():
                continue
            if t in DETERMINERS or t in FUNCTION_WORDS:
                continue
            if t[0].isupper() and len(t) > 1 and t.isalpha():
                proper.add(t)
            elif t.islower() and t.isalpha() and len(t) > 1:
                if i > 0 and toks[i - 1] in DETERMINERS:
                    common[t] += 1
    return sorted(proper), sorted([t for t, _ in common.most_common(30)])


def perturb_tokens(tokens, proper_pool, common_pool, rng,
                    p_perturb=0.5, force=False):
    if not force and rng.random() > p_perturb:
        return list(tokens)
    proper_used = sorted({t for t in tokens if t in proper_pool})
    common_used = sorted({t for i, t in enumerate(tokens)
                          if t in common_pool and i > 0
                          and tokens[i - 1] in DETERMINERS})
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
        if t in pmap:
            out.append(pmap[t])
        elif (t in cmap and i > 0 and tokens[i - 1] in DETERMINERS):
            out.append(cmap[t])
        else:
            out.append(t)
    return out


# ══════════════════════════════════════════════════════════════════════════
# MetaEncoder (with optional empty scalar features)
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
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        hidden = 128
        head_input_dim = d_model + n_scalar_features
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, hidden),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

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


def train_one(train_recs, val_recs, tok2id, feature_names, seed,
              proper_pool, common_pool, p_perturb=0.5,
              lr=3e-4, batch_size=64, max_epochs=50, patience=8,
              log_prefix=""):
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr = _features(train_recs, feature_names)
    Xv = _features(val_recs, feature_names)
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
        # Re-tokenize with perturbation
        perturbed = [perturb_tokens(t, proper_pool, common_pool, rng,
                                      p_perturb=p_perturb)
                      for t in train_tokens]
        ids_tr, mask_tr = tokenize(perturbed, tok2id)
        Itr = torch.from_numpy(ids_tr).to(device)
        Mtr = torch.from_numpy(mask_tr).to(device)

        model.train()
        idx = torch.randperm(N, device=device)
        loss_sum = 0.0; nb = 0
        for s in range(0, N, batch_size):
            sl = idx[s:s + batch_size]
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
        elapsed = time.time() - t0
        print(f"      {log_prefix}E{ep:02d} {marker} "
              f"loss={loss_sum/max(nb,1):.4f}  val_auc={auc:.4f}  "
              f"pat={pat}  ({elapsed:.1f}s)", flush=True)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience:
                print(f"      {log_prefix}early stop at ep {ep}", flush=True)
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
# Metrics
# ══════════════════════════════════════════════════════════════════════════
def _auc(y, s):
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def _macro_intra(scores, y, cats, mixed):
    aucs = []
    for cat in mixed:
        m = np.array([c == cat for c in cats])
        if m.sum() < 5 or len(set(y[m])) < 2:
            continue
        aucs.append(_auc(y[m], scores[m]))
    aucs = [a for a in aucs if not math.isnan(a)]
    return float(np.mean(aucs)) if aucs else float("nan")


def _risk_at_coverage(y, s, target=0.80):
    """Coverage = fraction kept (s < threshold) ; risk = mean(y) sur kept."""
    n = len(y)
    n_target = int(target * n)
    order = np.argsort(s)  # ascending : low score = safe
    kept = order[:n_target]
    if len(kept) == 0:
        return float("nan")
    return float(y[kept].mean())


def _aurc(y, s, n_points=99):
    thresholds = np.linspace(0.01, 0.99, n_points)
    cov, risk = [], []
    for t in thresholds:
        m = (s < t)
        n_keep = int(m.sum())
        if n_keep == 0:
            continue
        cov.append(n_keep / max(len(y), 1))
        risk.append(float(y[m].mean()))
    if len(cov) < 2:
        return float("nan")
    cov = np.array(cov); risk = np.array(risk)
    order = np.argsort(cov)
    cov, risk = cov[order], risk[order]
    return float(0.5 * np.sum((cov[1:] - cov[:-1]) * (risk[1:] + risk[:-1])))


def find_mixed_cats(y, cats, n_min=20):
    """Cats où n_err >= n_min ET n_correct >= n_min sur les exemples donnés."""
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


def cluster_bootstrap_macro(scores_a, scores_b, y, cats, mixed,
                             n_boot=1000, seed=42):
    """Bootstrap clusterisé par catégorie : resampler les cats avec remise,
    calculer macro_intra(sa) - macro_intra(sb)."""
    rng = np.random.default_rng(seed)
    cat_to_idx = defaultdict(list)
    for i, c in enumerate(cats):
        if c in mixed:
            cat_to_idx[c].append(i)
    mixed_list = list(cat_to_idx.keys())
    if not mixed_list:
        return float("nan"), float("nan"), float("nan")
    deltas = []
    for _ in range(n_boot):
        sampled = rng.choice(mixed_list, size=len(mixed_list), replace=True)
        aucs_a, aucs_b = [], []
        for cat in sampled:
            idxs = np.array(cat_to_idx[cat])
            y_c = y[idxs]
            if len(set(y_c)) < 2 or len(idxs) < 5:
                continue
            a = _auc(y_c, scores_a[idxs])
            b = _auc(y_c, scores_b[idxs])
            if math.isnan(a) or math.isnan(b):
                continue
            aucs_a.append(a); aucs_b.append(b)
        if aucs_a and aucs_b:
            deltas.append(float(np.mean(aucs_a)) - float(np.mean(aucs_b)))
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(deltas)
    return (float(arr.mean()),
            float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 97.5)))


# ══════════════════════════════════════════════════════════════════════════
# Single (config, seed, benchmark) run
# ══════════════════════════════════════════════════════════════════════════
def run_config(records, splits, config_name, feature_names, seed, do_loco):
    train_recs = [records[i] for i in splits["train"]]
    val_recs = [records[i] for i in splits["val"]]
    test_recs = [records[i] for i in splits["test"]]
    tok2id = build_vocab(train_recs)
    proper_pool, common_pool = build_lexical_pools(train_recs)

    # ── Train + score original
    t0 = time.time()
    model = train_one(train_recs, val_recs, tok2id, feature_names, seed,
                       proper_pool, common_pool,
                       log_prefix=f"[{config_name}/s{seed}/orig] ")
    t_train = time.time() - t0

    Xte = _features(test_recs, feature_names)
    Xtr = _features(train_recs, feature_names)
    Xv = _features(val_recs, feature_names)
    _, Xte_s, (mu, sd) = _stand(Xtr, Xte)
    Xv_s = (Xv - mu) / sd if Xv.shape[1] > 0 else Xv

    yte = _y(test_recs); cats_te = _cats(test_recs)
    ids_te, mask_te = tokenize([r["input_tokens"] for r in test_recs], tok2id)
    s_orig = _score(model, ids_te, mask_te, Xte_s)
    auc_orig = _auc(yte, s_orig)

    # ── Perturbé
    rng_eval = np.random.default_rng(seed)
    pert = [perturb_tokens(r["input_tokens"], proper_pool, common_pool,
                            rng_eval, force=True) for r in test_recs]
    ids_pe, mask_pe = tokenize(pert, tok2id)
    s_pert = _score(model, ids_pe, mask_pe, Xte_s)
    auc_pert = _auc(yte, s_pert)

    # ── Macro intra-cat
    mixed = find_mixed_cats(yte, cats_te, n_min=20)
    macro_orig = _macro_intra(s_orig, yte, cats_te, mixed)
    macro_pert = _macro_intra(s_pert, yte, cats_te, mixed)

    # ── Risk@0.80, AURC
    risk_at_80 = _risk_at_coverage(yte, s_orig, target=0.80)
    aurc = _aurc(yte, s_orig)

    out = {
        "config": config_name,
        "seed": seed,
        "feature_names": feature_names,
        "n_scalar_features": len(feature_names),
        "auc_orig": auc_orig,
        "auc_pert": auc_pert,
        "macro_intra_orig": macro_orig,
        "macro_intra_pert": macro_pert,
        "risk_at_coverage_80": risk_at_80,
        "aurc": aurc,
        "mixed_cats": mixed,
        "train_time_sec": t_train,
        "scores_test_orig": s_orig.tolist(),
    }

    # ── LOCO (optionnel : retrain sans cats mixtes — coûteux)
    if do_loco and mixed:
        train_loco = [r for r in train_recs if r["category"] not in mixed]
        if len(train_loco) > 100:
            model_loco = train_one(train_loco, val_recs, tok2id, feature_names,
                                     seed, proper_pool, common_pool,
                                     log_prefix=f"[{config_name}/s{seed}/loco] ")
            s_loco = _score(model_loco, ids_te, mask_te, Xte_s)
            out["auc_loco_global"] = _auc(yte, s_loco)
            out["macro_intra_loco"] = _macro_intra(s_loco, yte, cats_te, mixed)
        else:
            out["auc_loco_global"] = float("nan")
            out["macro_intra_loco"] = float("nan")
    else:
        out["auc_loco_global"] = None  # skipped
        out["macro_intra_loco"] = None
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
    ap.add_argument("--skip-loco", action="store_true",
                    help="Skip LOCO retraining to save compute")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    do_loco = not args.skip_loco

    bench_specs = [
        ("cogs", args.cogs_dataset, args.cogs_splits),
        ("slog", args.slog_dataset, args.slog_splits),
    ]

    all_results = {}
    for bench_name, dataset_path, splits_path in bench_specs:
        print(f"\n{'═'*60}")
        print(f"  BENCHMARK : {bench_name}")
        print(f"{'═'*60}")
        records = _load(dataset_path)
        with open(splits_path, "r") as f:
            splits = json.load(f)
        bench_results = []
        for cfg_name, feat_names in CONFIGS.items():
            for seed in seeds:
                print(f"\n[{bench_name}] {cfg_name} seed={seed} feat={feat_names}")
                try:
                    res = run_config(records, splits, cfg_name, feat_names,
                                      seed, do_loco)
                    print(f"  AUC orig={res['auc_orig']:.4f}  "
                          f"AUC pert={res['auc_pert']:.4f}  "
                          f"macro={res['macro_intra_orig']:.4f}  "
                          f"AURC={res['aurc']:.4f}", flush=True)
                    bench_results.append(res)
                except Exception as e:
                    print(f"  FAILED: {e}", flush=True)
                    bench_results.append({"config": cfg_name, "seed": seed,
                                          "error": str(e)})
        all_results[bench_name] = bench_results
        with open(os.path.join(args.output_dir, f"results_{bench_name}.json"), "w") as f:
            json.dump(bench_results, f, indent=2)

    # ── Bootstrap clusterisé : full vs max(C3, C4, C5) en macro intra-cat
    print("\n── Bootstrap clusterisé par catégorie ──")
    bootstrap_results = {}
    for bench_name, dataset_path, splits_path in bench_specs:
        records = _load(dataset_path)
        with open(splits_path, "r") as f:
            splits = json.load(f)
        test_recs = [records[i] for i in splits["test"]]
        yte = _y(test_recs); cats_te = _cats(test_recs)
        mixed = find_mixed_cats(yte, cats_te, n_min=20)

        bench_res = all_results[bench_name]
        # Pick scores per (config, seed=42 for bootstrap)
        scores_by_cfg = {}
        for r in bench_res:
            if r.get("seed") == 42 and "scores_test_orig" in r:
                scores_by_cfg[r["config"]] = np.array(r["scores_test_orig"])

        if "C6_full" not in scores_by_cfg:
            print(f"  {bench_name}: C6_full missing, skipping bootstrap")
            continue

        # Pick the best macro among C3, C4, C5 (single-feature alternatives)
        single_feat_cfgs = ["C3_tokens_entropy", "C4_tokens_surprise", "C5_tokens_ent_sur"]
        single_macros = []
        for cfg in single_feat_cfgs:
            for r in bench_res:
                if r["config"] == cfg and r.get("seed") == 42 and "macro_intra_orig" in r:
                    single_macros.append((cfg, r["macro_intra_orig"]))
        if not single_macros:
            continue
        best_single = max(single_macros, key=lambda x: x[1])
        print(f"  {bench_name}: full vs best_single={best_single[0]} "
              f"(macro={best_single[1]:.4f})")

        bootstrap_results[bench_name] = {}
        # Bootstrap full vs best_single
        if best_single[0] in scores_by_cfg:
            d_mean, d_lo, d_hi = cluster_bootstrap_macro(
                scores_by_cfg["C6_full"], scores_by_cfg[best_single[0]],
                yte, cats_te, mixed, n_boot=1000, seed=42)
            bootstrap_results[bench_name][f"full_minus_{best_single[0]}"] = {
                "delta_mean": d_mean, "ci95_lo": d_lo, "ci95_hi": d_hi,
                "n_mixed_cats": len(mixed),
            }
            print(f"    Δ macro = {d_mean:+.4f}  IC95 [{d_lo:+.4f}, {d_hi:+.4f}]")
        # Also bootstrap full vs C1 (tokens-only)
        if "C1_tokens_only" in scores_by_cfg:
            d_mean, d_lo, d_hi = cluster_bootstrap_macro(
                scores_by_cfg["C6_full"], scores_by_cfg["C1_tokens_only"],
                yte, cats_te, mixed, n_boot=1000, seed=42)
            bootstrap_results[bench_name]["full_minus_C1"] = {
                "delta_mean": d_mean, "ci95_lo": d_lo, "ci95_hi": d_hi,
            }

    with open(os.path.join(args.output_dir, "bootstrap_deltas.json"), "w") as f:
        json.dump(bootstrap_results, f, indent=2)

    # ── Comparison table
    write_comparison_table(all_results, args.output_dir)

    # ── Verdict
    write_verdict(all_results, bootstrap_results, args.output_dir)
    print(f"\nAll outputs written to {args.output_dir}/")


def write_comparison_table(all_results, out_dir):
    lines = ["# TEST 2 — Tableau comparatif ablation factorielle", ""]
    lines.append("AUC ∈ [0, 1]. mean ± std sur 3 seeds.")
    lines.append("")
    for bench_name, results in all_results.items():
        lines.append(f"## {bench_name.upper()}")
        lines.append("")
        lines.append("| Config | n_feat | AUC orig | AUC pert | "
                     "macro intra-cat orig | macro pert | risk@0.80 | AURC |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for cfg in CONFIGS.keys():
            seeds_res = [r for r in results if r.get("config") == cfg
                         and "auc_orig" in r]
            if not seeds_res:
                continue
            def _ms(key):
                vals = [r[key] for r in seeds_res
                        if not (math.isnan(r[key]) if r[key] is not None else True)]
                if not vals:
                    return float("nan"), float("nan")
                return float(np.mean(vals)), (float(np.std(vals))
                                              if len(vals) > 1 else 0.0)
            n_feat = seeds_res[0]["n_scalar_features"]
            o_m, o_s = _ms("auc_orig")
            p_m, p_s = _ms("auc_pert")
            mo_m, mo_s = _ms("macro_intra_orig")
            mp_m, mp_s = _ms("macro_intra_pert")
            r_m, r_s = _ms("risk_at_coverage_80")
            a_m, a_s = _ms("aurc")
            def _f(m, s):
                return f"{m:.4f}±{s:.4f}" if not math.isnan(m) else "N/A"
            lines.append(f"| {cfg} | {n_feat} | {_f(o_m, o_s)} | "
                         f"{_f(p_m, p_s)} | {_f(mo_m, mo_s)} | "
                         f"{_f(mp_m, mp_s)} | {_f(r_m, r_s)} | "
                         f"{_f(a_m, a_s)} |")
        lines.append("")
    with open(os.path.join(out_dir, "comparison_table.md"), "w") as f:
        f.write("\n".join(lines))


def write_verdict(all_results, bootstrap, out_dir):
    lines = ["# TEST 2 — Verdict ablation factorielle", ""]
    # Critère : full − max(C3, C4, C5) en macro intra-cat sur COGS ET SLOG
    verdicts = {}
    for bench in ["cogs", "slog"]:
        bs = bootstrap.get(bench, {})
        # Find the entry full_minus_*
        single_keys = [k for k in bs if k.startswith("full_minus_C") and
                       not k.endswith("_C1")]
        if not single_keys:
            verdicts[bench] = ("REPORTÉ", "Pas de bootstrap macro disponible.")
            continue
        d = bs[single_keys[0]]
        delta = d["delta_mean"]; lo = d["ci95_lo"]; hi = d["ci95_hi"]
        ic_excludes_zero = lo > 0
        if math.isnan(delta):
            verdicts[bench] = ("REPORTÉ", "Δ indisponible")
        elif delta < 0:
            verdicts[bench] = ("BUG", f"Δ={delta:+.4f} < 0")
        elif delta >= 0.05 and ic_excludes_zero:
            verdicts[bench] = ("GO",
                f"Δ={delta:+.4f} ≥ 0.05, IC95 [{lo:+.4f}, {hi:+.4f}] exclut 0")
        elif delta >= 0.02 and ic_excludes_zero:
            verdicts[bench] = ("TEST",
                f"Δ={delta:+.4f} ∈ [0.02, 0.05[, IC95 [{lo:+.4f}, {hi:+.4f}] exclut 0")
        else:
            verdicts[bench] = ("NO-GO",
                f"Δ={delta:+.4f} < 0.02 ou IC95 [{lo:+.4f}, {hi:+.4f}] inclut 0")

    lines.append("## Verdicts par benchmark")
    lines.append("")
    lines.append("| Benchmark | Verdict | Raison |")
    lines.append("|---|---|---|")
    for bench, (v, r) in verdicts.items():
        lines.append(f"| {bench.upper()} | **{v}** | {r} |")
    lines.append("")

    # Verdict combiné per spec : critère sur COGS ET SLOG
    cogs_v = verdicts.get("cogs", ("REPORTÉ", ""))[0]
    slog_v = verdicts.get("slog", ("REPORTÉ", ""))[0]
    lines.append("## Verdict combiné")
    lines.append("")
    if cogs_v == "GO" and slog_v == "GO":
        lines.append("**GO JEPA causalement utile** sur les 2 benchmarks. "
                     "Refonte JEPA justifiée.")
    elif cogs_v == "TEST" or slog_v == "TEST":
        lines.append("**TEST conditionnel** — JEPA marginalement utile sur au "
                     "moins un benchmark. Refonte possible mais alternatives "
                     "à envisager (MC dropout, ensembles).")
    elif cogs_v == "NO-GO" and slog_v == "NO-GO":
        lines.append("**NO-GO JEPA actuel** sur les 2 benchmarks. "
                     "Le JEPA est décoratif — pivot vers claim 'lightweight "
                     "failure predictor' sans JEPA, ou alternatives.")
    elif cogs_v == "BUG" or slog_v == "BUG":
        lines.append("**BUG** détecté — investiguer avant toute conclusion.")
    else:
        lines.append("**Verdict mixte** — voir détails par benchmark.")

    # Cas particulier : C1 atteint déjà macro > 0.95 sur COGS
    cogs_results = all_results.get("cogs", [])
    c1_macros = [r["macro_intra_orig"] for r in cogs_results
                 if r.get("config") == "C1_tokens_only"
                 and "macro_intra_orig" in r
                 and not math.isnan(r["macro_intra_orig"])]
    if c1_macros and float(np.mean(c1_macros)) > 0.95:
        lines.append("")
        lines.append("⚠️ **ALERTE ROUGE** : C1 (tokens-only) atteint macro "
                     f"intra-cat = {np.mean(c1_macros):.4f} > 0.95 sur COGS. "
                     "Le mini-Transformer apprend l'input lui-même. "
                     "Le claim 'introspection' devient très fragile.")

    with open(os.path.join(out_dir, "verdict.txt"), "w") as f:
        f.write("\n".join(lines))
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
