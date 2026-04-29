#!/usr/bin/env python3
"""Étape 3 — Encodeur léger sur input + features scalaires.

Entraîne un MetaEncoder (Transformer 2 layers d=64) qui regarde directement
les tokens de l'input et concatène un pooling moyen avec les 4 features
scalaires de l'étape 2 (surprise_mean, entropy_mean_greedy, rare_token_count,
nesting_depth). Compare à 7 baselines.

Single-seed. Lance 3 fois et utilise meta_etape3_summary.py pour agréger.

Usage :
  python3 meta_train_etape3.py \\
      --dataset meta_data/cogs_meta_dataset_bplus.jsonl \\
      --splits  meta_data/cogs_meta_splits.json \\
      --output  runs_meta/etape3_s42 \\
      --seed 42
"""
import argparse
import json
import math
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCALAR_FEATURES = ["surprise_mean", "entropy_mean_greedy",
                   "rare_token_count", "nesting_depth"]
BPLUS_FEATURES = SCALAR_FEATURES + ["verb_construction_count_train",
                                    "verb_construction_seen_binary"]
PAD_TOK = "<pad>"
UNK_TOK = "<unk>"
MAX_LEN = 64


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
    mu = X_train.mean(axis=0); sd = X_train.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (X_train - mu) / sd, (X_other - mu) / sd, (mu, sd)


def build_vocab(train_recs):
    """Construit un vocab depuis les input_tokens de train. Retourne
    (tok2id, id2tok)."""
    counter = Counter()
    for r in train_recs:
        for t in r.get("input_tokens", []):
            counter[t] += 1
    tok2id = {PAD_TOK: 0, UNK_TOK: 1}
    for t, _ in counter.most_common():
        if t not in tok2id:
            tok2id[t] = len(tok2id)
    id2tok = {i: t for t, i in tok2id.items()}
    return tok2id, id2tok


def tokenize_records(records, tok2id, max_len=MAX_LEN):
    """Returns (input_ids [N, max_len], input_mask [N, max_len])."""
    N = len(records)
    ids = np.zeros((N, max_len), dtype=np.int64)
    mask = np.zeros((N, max_len), dtype=np.float32)
    truncated = 0
    for i, r in enumerate(records):
        toks = r.get("input_tokens", [])
        if len(toks) > max_len:
            toks = toks[:max_len]
            truncated += 1
        for j, t in enumerate(toks):
            ids[i, j] = tok2id.get(t, tok2id[UNK_TOK])
            mask[i, j] = 1.0
    if truncated > 0:
        print(f"  ⚠️  {truncated}/{N} examples truncated to max_len={max_len}")
    return ids, mask


# ══════════════════════════════════════════════════════════════════════════
# MetaEncoder
# ══════════════════════════════════════════════════════════════════════════
class MetaEncoder(nn.Module):
    def __init__(self, vocab_size, n_scalar_features, d_model=64, n_heads=4,
                 n_layers=2, dropout=0.1, max_len=MAX_LEN):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=128,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        hidden = 128
        self.head = nn.Sequential(
            nn.Linear(d_model + n_scalar_features, hidden),
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
        combined = torch.cat([pooled, scalar_features], dim=-1)
        return self.head(combined).squeeze(-1)


def _train_encoder(model, ids_tr, mask_tr, sc_tr, ytr,
                   ids_v, mask_v, sc_v, yv, seed,
                   lr=3e-4, batch_size=64, max_epochs=50, patience=8,
                   weight_decay=1e-4):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    Idtr = torch.from_numpy(ids_tr).to(device)
    Mtr = torch.from_numpy(mask_tr).to(device)
    Str = torch.from_numpy(sc_tr).to(device)
    Ytr = torch.from_numpy(ytr).to(device)
    Idv = torch.from_numpy(ids_v).to(device)
    Mv = torch.from_numpy(mask_v).to(device)
    Sv = torch.from_numpy(sc_v).to(device)
    Yv = torch.from_numpy(yv).to(device)

    best_auc, best_state, pat = -1.0, None, 0
    train_losses, val_aucs = [], []
    N = Idtr.size(0)
    for ep in range(max_epochs):
        model.train()
        idx = torch.randperm(N, device=device)
        loss_sum = 0.0; nb = 0
        for start in range(0, N, batch_size):
            sl = idx[start:start + batch_size]
            logits = model(Idtr[sl], Mtr[sl], Str[sl])
            loss = F.binary_cross_entropy_with_logits(logits, Ytr[sl])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item()); nb += 1
        model.eval()
        with torch.no_grad():
            sv = torch.sigmoid(model(Idv, Mv, Sv)).cpu().numpy()
        auc = _roc_auc(yv, sv)
        train_losses.append(loss_sum / max(nb, 1))
        val_aucs.append(auc)
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
    return model, train_losses, val_aucs


# ══════════════════════════════════════════════════════════════════════════
# MetaMLP (re-utilisé pour étape 2 et B+ baselines)
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
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
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


def _full_metrics(name, y_test, y_val, s_test, s_val):
    auc = _roc_auc(y_test, s_test)
    thr = _best_threshold_on_val(y_val, s_val) if s_val is not None else 0.5
    at_thr = _f1_at_threshold(y_test, s_test, thr)
    brier = _brier(y_test, s_test)
    return {"name": name, "auc": auc, "at_threshold": at_thr, "brier": brier}


def _bootstrap_ci(y_true, sa, sb, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    n_test = len(y_true)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, n_test, size=n_test)
        a = _roc_auc(y_true[idx], sa[idx])
        b = _roc_auc(y_true[idx], sb[idx])
        if not (np.isnan(a) or np.isnan(b)):
            diffs.append(a - b)
    if not diffs:
        return None, None, None
    diffs = np.array(diffs)
    return (float(diffs.mean()),
            float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)))


def _gbdt_1d(x_train, y_train, x_test, x_val, seed):
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        gbdt = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                          random_state=seed)
        gbdt.fit(x_train, y_train)
        return gbdt.predict_proba(x_test)[:, 1], gbdt.predict_proba(x_val)[:, 1]
    except Exception as e:
        print(f"GBDT unavailable: {e}")
        return None, None


# ══════════════════════════════════════════════════════════════════════════
# Per-category breakdown
# ══════════════════════════════════════════════════════════════════════════
def _per_category(yte, scores, cats):
    from collections import defaultdict
    out = {}
    cat_to_idx = defaultdict(list)
    for i, c in enumerate(cats):
        cat_to_idx[c].append(i)
    for cat, idxs in cat_to_idx.items():
        idxs = np.array(idxs)
        y_c = yte[idxs]; s_c = scores[idxs]
        n = len(idxs); n_err = int(y_c.sum())
        if n >= 20 and len(set(y_c)) == 2:
            auc = _roc_auc(y_c, s_c)
        else:
            auc = float("nan")
        out[cat] = {"n": n, "n_errors": n_err, "auc": auc}
    return out


# ══════════════════════════════════════════════════════════════════════════
# Sanity checks
# ══════════════════════════════════════════════════════════════════════════
def _sanity_checks(records, tok2id, id2tok, max_len=MAX_LEN):
    print("── Sanity checks ──")
    # 1. token mapping
    for i in range(min(5, len(records))):
        toks = records[i].get("input_tokens", [])
        ids = [tok2id.get(t, tok2id[UNK_TOK]) for t in toks[:max_len]]
        recon = [id2tok[ix] for ix in ids]
        ok = (recon == toks[:max_len])
        print(f"  ex{i}: ok={ok}  len={len(toks)}  first6={toks[:6]}")
        if not ok:
            print(f"    EXPECTED {toks[:max_len]}")
            print(f"    GOT      {recon}")
    # 2. length distribution
    lens = [len(r.get("input_tokens", [])) for r in records]
    print(f"  Length distribution: min={min(lens)}, max={max(lens)}, "
          f"mean={np.mean(lens):.1f}, p99={int(np.percentile(lens, 99))}")
    if max(lens) > max_len:
        print(f"  ⚠️  {sum(1 for l in lens if l > max_len)} examples > max_len")
    # 3. vocab size
    print(f"  Vocab size : {len(tok2id)} (incl. <pad>, <unk>)")
    n_unk = 0; n_tok = 0
    for r in records:
        for t in r.get("input_tokens", []):
            n_tok += 1
            if t not in tok2id:
                n_unk += 1
    print(f"  UNK rate on full dataset : {n_unk}/{n_tok} ({100*n_unk/max(n_tok,1):.2f}%)")
    print()


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
    print(f"Loaded {len(records)} records "
          f"(train={len(train_recs)}, val={len(val_recs)}, test={len(test_recs)})")

    has_bplus = all(f in records[0]["features_inference"] for f in BPLUS_FEATURES)
    print(f"B+ features in dataset: {has_bplus}")
    print()

    # ── Vocab + tokenization (sanity checks)
    tok2id, id2tok = build_vocab(train_recs)
    _sanity_checks(records, tok2id, id2tok)

    ids_tr, mask_tr = tokenize_records(train_recs, tok2id)
    ids_v, mask_v = tokenize_records(val_recs, tok2id)
    ids_te, mask_te = tokenize_records(test_recs, tok2id)

    # ── Scalar features
    Xtr4 = _features_matrix(train_recs, SCALAR_FEATURES)
    Xv4 = _features_matrix(val_recs, SCALAR_FEATURES)
    Xte4 = _features_matrix(test_recs, SCALAR_FEATURES)
    Xtr4_s, _, (mu4, sd4) = _standardize(Xtr4, Xv4)
    Xv4_s = (Xv4 - mu4) / sd4
    Xte4_s = (Xte4 - mu4) / sd4

    if has_bplus:
        Xtr6 = _features_matrix(train_recs, BPLUS_FEATURES)
        Xv6 = _features_matrix(val_recs, BPLUS_FEATURES)
        Xte6 = _features_matrix(test_recs, BPLUS_FEATURES)
        Xtr6_s, _, (mu6, sd6) = _standardize(Xtr6, Xv6)
        Xv6_s = (Xv6 - mu6) / sd6
        Xte6_s = (Xte6 - mu6) / sd6

    ytr = _labels(train_recs); yv = _labels(val_recs); yte = _labels(test_recs)
    cats_test = _categories(test_recs)
    print(f"Class balance test: {int(yte.sum())}/{len(yte)} errors "
          f"({100*yte.mean():.1f}%)")
    print()

    # ── 1. raw_surprise
    sur_idx = SCALAR_FEATURES.index("surprise_mean")
    s_raw_sur_te = Xte4[:, sur_idx]; s_raw_sur_v = Xv4[:, sur_idx]
    m_raw_sur = _full_metrics("raw_surprise", yte, yv, s_raw_sur_te, s_raw_sur_v)

    # ── 2. raw_entropy
    ent_idx = SCALAR_FEATURES.index("entropy_mean_greedy")
    s_raw_ent_te = Xte4[:, ent_idx]; s_raw_ent_v = Xv4[:, ent_idx]
    m_raw_ent = _full_metrics("raw_entropy", yte, yv, s_raw_ent_te, s_raw_ent_v)

    # ── 3. logreg(surprise)
    try:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=200)
        lr.fit(Xtr4[:, sur_idx:sur_idx+1], ytr)
        s_lr_te = lr.predict_proba(Xte4[:, sur_idx:sur_idx+1])[:, 1]
        s_lr_v = lr.predict_proba(Xv4[:, sur_idx:sur_idx+1])[:, 1]
        m_lr = _full_metrics("logreg_surprise", yte, yv, s_lr_te, s_lr_v)
    except Exception as e:
        print(f"Logreg unavailable: {e}")
        s_lr_te = s_lr_v = None
        m_lr = None

    # ── 4. GBDT_1D(surprise)
    s_gbdt_sur_te, s_gbdt_sur_v = _gbdt_1d(Xtr4[:, sur_idx:sur_idx+1], ytr,
                                            Xte4[:, sur_idx:sur_idx+1],
                                            Xv4[:, sur_idx:sur_idx+1], args.seed)
    m_gbdt_sur = _full_metrics("gbdt_1d_surprise", yte, yv,
                               s_gbdt_sur_te, s_gbdt_sur_v) if s_gbdt_sur_te is not None else None

    # ── 5. GBDT_1D(entropy)
    s_gbdt_ent_te, s_gbdt_ent_v = _gbdt_1d(Xtr4[:, ent_idx:ent_idx+1], ytr,
                                            Xte4[:, ent_idx:ent_idx+1],
                                            Xv4[:, ent_idx:ent_idx+1], args.seed)
    m_gbdt_ent = _full_metrics("gbdt_1d_entropy", yte, yv,
                               s_gbdt_ent_te, s_gbdt_ent_v) if s_gbdt_ent_te is not None else None

    # ── 6. MetaMLP étape 2 (4 features)
    print("Training MetaMLP étape 2 (4 feat)…")
    mlp4 = _train_mlp(Xtr4_s, ytr, Xv4_s, yv, n_features=4, seed=args.seed)
    mlp4.eval()
    device = next(mlp4.parameters()).device
    with torch.no_grad():
        s_mlp4_te = torch.sigmoid(mlp4(torch.from_numpy(Xte4_s).to(device))).cpu().numpy()
        s_mlp4_v = torch.sigmoid(mlp4(torch.from_numpy(Xv4_s).to(device))).cpu().numpy()
    m_mlp4 = _full_metrics("metamlp_etape2", yte, yv, s_mlp4_te, s_mlp4_v)

    # ── 7. MetaMLP B+ (6 features)
    s_mlpB_te = s_mlpB_v = None
    m_mlpB = None
    if has_bplus:
        print("Training MetaMLP B+ (6 feat)…")
        mlpB = _train_mlp(Xtr6_s, ytr, Xv6_s, yv, n_features=6, seed=args.seed)
        mlpB.eval()
        with torch.no_grad():
            s_mlpB_te = torch.sigmoid(mlpB(torch.from_numpy(Xte6_s).to(device))).cpu().numpy()
            s_mlpB_v = torch.sigmoid(mlpB(torch.from_numpy(Xv6_s).to(device))).cpu().numpy()
        m_mlpB = _full_metrics("metamlp_bplus", yte, yv, s_mlpB_te, s_mlpB_v)

    # ── 8. MetaEncoder (cible)
    print("Training MetaEncoder (cible étape 3)…")
    encoder = MetaEncoder(vocab_size=len(tok2id),
                          n_scalar_features=4, d_model=64,
                          n_heads=4, n_layers=2, dropout=0.1)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Params: {n_params/1000:.1f}k")
    encoder, train_losses, val_aucs = _train_encoder(
        encoder, ids_tr, mask_tr, Xtr4_s, ytr,
        ids_v, mask_v, Xv4_s, yv, seed=args.seed)
    encoder.eval()
    device = next(encoder.parameters()).device
    with torch.no_grad():
        s_enc_te = torch.sigmoid(encoder(
            torch.from_numpy(ids_te).to(device),
            torch.from_numpy(mask_te).to(device),
            torch.from_numpy(Xte4_s).to(device))).cpu().numpy()
        s_enc_v = torch.sigmoid(encoder(
            torch.from_numpy(ids_v).to(device),
            torch.from_numpy(mask_v).to(device),
            torch.from_numpy(Xv4_s).to(device))).cpu().numpy()
    m_enc = _full_metrics("metaencoder_etape3", yte, yv, s_enc_te, s_enc_v)
    print(f"  Final AUC test = {m_enc['auc']:.4f}")

    # ── Bootstrap CI
    bs = {}
    if s_mlp4_te is not None:
        d, lo, hi = _bootstrap_ci(yte, s_enc_te, s_mlp4_te, seed=args.seed)
        bs["enc_minus_mlp_etape2"] = {"mean": d, "lo": lo, "hi": hi}
    if s_mlpB_te is not None:
        d, lo, hi = _bootstrap_ci(yte, s_enc_te, s_mlpB_te, seed=args.seed)
        bs["enc_minus_mlp_bplus"] = {"mean": d, "lo": lo, "hi": hi}
    if s_gbdt_ent_te is not None:
        d, lo, hi = _bootstrap_ci(yte, s_enc_te, s_gbdt_ent_te, seed=args.seed)
        bs["enc_minus_gbdt_entropy"] = {"mean": d, "lo": lo, "hi": hi}

    # ── Decorrelation
    decorr = None
    if s_gbdt_sur_te is not None:
        gp = (s_gbdt_sur_te >= 0.5).astype(np.float32)
        wm = (gp != yte)
        if wm.sum() >= 30:
            decorr = _roc_auc(yte[wm], s_enc_te[wm])

    # ── Per-category breakdowns
    pc_mlp4 = _per_category(yte, s_mlp4_te, cats_test)
    pc_enc = _per_category(yte, s_enc_te, cats_test)
    pc_mlpB = _per_category(yte, s_mlpB_te, cats_test) if s_mlpB_te is not None else {}

    # ── Save scores for downstream summary
    scores_dump = {
        "yte": yte.tolist(),
        "cats_test": cats_test,
        "s_enc_te": s_enc_te.tolist(),
        "s_mlp4_te": s_mlp4_te.tolist(),
        "s_mlpB_te": s_mlpB_te.tolist() if s_mlpB_te is not None else None,
        "input_tokens_test": [r.get("input_tokens", []) for r in test_recs],
        "categories_test": cats_test,
    }
    with open(os.path.join(args.output, "scores.json"), "w") as f:
        json.dump(scores_dump, f)

    metrics = {
        "seed": args.seed,
        "n_test": int(len(yte)),
        "vocab_size": len(tok2id),
        "encoder_params": int(n_params),
        "models": [m for m in [m_raw_sur, m_lr, m_gbdt_sur,
                               m_raw_ent, m_gbdt_ent,
                               m_mlp4, m_mlpB, m_enc]
                   if m is not None],
        "bootstrap_ci": bs,
        "decorrelation_auc_on_gbdt_surprise_errors": decorr,
        "per_category": {
            "metamlp_etape2": pc_mlp4,
            "metamlp_bplus": pc_mlpB,
            "metaencoder_etape3": pc_enc,
        },
        "encoder_train_losses": train_losses,
        "encoder_val_aucs": val_aucs,
    }
    with open(os.path.join(args.output, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.output, "bootstrap_ci.json"), "w") as f:
        json.dump(bs, f, indent=2)

    # ── learning curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(train_losses, label="train_loss")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("BCE loss")
        ax1.set_title(f"Encoder training (seed {args.seed})")
        ax1.grid(True, alpha=0.3); ax1.legend()
        ax2.plot(val_aucs, label="val AUC", color="green")
        ax2.set_xlabel("epoch"); ax2.set_ylabel("AUC")
        ax2.set_title(f"Encoder val AUC (seed {args.seed})")
        ax2.grid(True, alpha=0.3); ax2.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.output, "learning_curves.png"), dpi=110)
        plt.close(fig)
    except Exception as e:
        print(f"learning_curves.png skipped: {e}")

    # ── Print summary
    print()
    print("─" * 60)
    print(f"Seed {args.seed} — AUC test summary")
    print("─" * 60)
    for m in metrics["models"]:
        print(f"  {m['name']:24s} AUC={m['auc']:.4f}  "
              f"F1={m['at_threshold']['f1']:.3f}  "
              f"prec={m['at_threshold']['precision']:.3f}  "
              f"rec={m['at_threshold']['recall']:.3f}")
    if decorr is not None:
        print(f"  decorrelation (encoder | gbdt_sur errors) = {decorr:.3f}")
    print()
    PROBLEM_CATS = ["subj_to_obj_common", "passive_to_active",
                    "unacc_to_transitive", "obj_omitted_transitive_to_transitive"]
    print("Catégories problématiques (cible étape 3):")
    print(f"{'cat':45s}  mlp4    mlpB+   enc")
    for c in PROBLEM_CATS:
        a4 = pc_mlp4.get(c, {}).get("auc", float("nan"))
        aB = pc_mlpB.get(c, {}).get("auc", float("nan"))
        ae = pc_enc.get(c, {}).get("auc", float("nan"))
        f4 = "N/A" if math.isnan(a4) else f"{a4:.3f}"
        fB = "N/A" if math.isnan(aB) else f"{aB:.3f}"
        fe = "N/A" if math.isnan(ae) else f"{ae:.3f}"
        print(f"  {c:45s}  {f4}  {fB}  {fe}")


if __name__ == "__main__":
    main()
