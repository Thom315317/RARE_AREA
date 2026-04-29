#!/usr/bin/env python3
"""Phase B — MetaEncoder + perturbation lexicale en data augmentation.

Entraîne le MetaEncoder avec perturbation des proper nouns et common nouns
top-30 en input (probabilité 0.5 par exemple), pour rendre le signal
indépendant du vocabulaire concret. Les features scalaires (surprise,
entropy, etc.) restent celles de l'input ORIGINAL — on enseigne au modèle
que la structure compte, pas les mots.

Évalue sur 3 conditions :
  - Test original (non perturbé)
  - Test perturbé (perturbation déterministe seed-fixée)
  - LOCO : ré-entraîne sans les 4 cats prob, test sur ces 4 cats

Sur 3 seeds (42, 123, 456) avec agrégation. Verdict per spec §B.5.

Usage :
  python3 meta_train_etape3_robust.py \\
      --dataset meta_data/cogs_meta_dataset_bplus.jsonl \\
      --splits  meta_data/cogs_meta_splits.json \\
      --output-dir runs_meta/etapeB_robust \\
      --seeds 42,123,456 \\
      --p-perturb 0.5
"""
import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCALAR_FEATURES = ["surprise_mean", "entropy_mean_greedy",
                   "rare_token_count", "nesting_depth"]
# Default = COGS problem cats. Override via --problem-cats for SLOG.
PROBLEM_CATS = ["subj_to_obj_common", "passive_to_active",
                "unacc_to_transitive", "obj_omitted_transitive_to_transitive"]
DETERMINERS = {"A", "An", "The", "a", "an", "the"}
FUNCTION_WORDS = {"that", "was", "were", "by", "to", "in", "on", "."}
PAD_TOK = "<pad>"
UNK_TOK = "<unk>"
MAX_LEN = 64


# ══════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════
def _load_jsonl(p):
    with open(p, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _features_matrix(recs, names):
    return np.array([[r["features_inference"][f] for f in names] for r in recs],
                    dtype=np.float32)


def _labels(recs):
    return np.array([0 if r["exact_match"] else 1 for r in recs], dtype=np.float32)


def _categories(recs):
    return [r["category"] for r in recs]


def _standardize(X_train, X_other):
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


def tokenize_records(recs, tok2id, max_len=MAX_LEN):
    N = len(recs)
    ids = np.zeros((N, max_len), dtype=np.int64)
    mask = np.zeros((N, max_len), dtype=np.float32)
    for i, r in enumerate(recs):
        toks = r.get("input_tokens", [])[:max_len]
        for j, t in enumerate(toks):
            ids[i, j] = tok2id.get(t, tok2id[UNK_TOK])
            mask[i, j] = 1.0
    return ids, mask


def tokenize_token_lists(token_lists, tok2id, max_len=MAX_LEN):
    N = len(token_lists)
    ids = np.zeros((N, max_len), dtype=np.int64)
    mask = np.zeros((N, max_len), dtype=np.float32)
    for i, toks in enumerate(token_lists):
        for j, t in enumerate(toks[:max_len]):
            ids[i, j] = tok2id.get(t, tok2id[UNK_TOK])
            mask[i, j] = 1.0
    return ids, mask


def _roc_auc(y, s):
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


# ══════════════════════════════════════════════════════════════════════════
# Lexical pools
# ══════════════════════════════════════════════════════════════════════════
def build_lexical_pools(recs):
    proper_set = set()
    common_freq = Counter()
    for r in recs:
        toks = r.get("input_tokens", [])
        for i, t in enumerate(toks):
            if not t or not t[0].isalpha():
                continue
            if t in DETERMINERS or t in FUNCTION_WORDS:
                continue
            if t[0].isupper() and len(t) > 1 and t.isalpha():
                proper_set.add(t)
            elif t.islower() and t.isalpha() and len(t) > 1:
                if i > 0 and toks[i - 1] in DETERMINERS:
                    common_freq[t] += 1
    common_top = set(t for t, _ in common_freq.most_common(30))
    return sorted(proper_set), sorted(common_top)


def perturb_input_tokens(tokens, proper_pool, common_pool, rng,
                         p_perturb=0.5, force=False):
    """Perturb proper nouns and common nouns (post-déterminant) consistently
    per sentence. With prob (1 - p_perturb), no perturbation."""
    if not force and rng.random() > p_perturb:
        return list(tokens)
    proper_used = sorted({t for t in tokens if t in proper_pool})
    common_used = sorted({t for i, t in enumerate(tokens)
                          if t in common_pool and i > 0
                          and tokens[i - 1] in DETERMINERS})
    proper_map = {}
    if proper_used:
        cands = [p for p in proper_pool if p not in proper_used]
        if cands:
            picks = rng.choice(cands, size=len(proper_used),
                               replace=len(cands) < len(proper_used))
            for old, new in zip(proper_used, picks):
                proper_map[old] = str(new)
    common_map = {}
    if common_used:
        cands = [c for c in common_pool if c not in common_used]
        if cands:
            picks = rng.choice(cands, size=len(common_used),
                               replace=len(cands) < len(common_used))
            for old, new in zip(common_used, picks):
                common_map[old] = str(new)
    out = []
    for i, t in enumerate(tokens):
        if t in proper_map:
            out.append(proper_map[t])
        elif (t in common_map and i > 0 and tokens[i - 1] in DETERMINERS):
            out.append(common_map[t])
        else:
            out.append(t)
    return out


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


def _set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _train_encoder_with_perturbation(
        model, train_recs, val_recs, tok2id,
        Xtr_s, Xv_s, ytr, yv,
        proper_pool, common_pool, p_perturb, seed,
        lr=3e-4, batch_size=64, max_epochs=50, patience=8,
        weight_decay=1e-4, log_prefix=""):
    """Train encoder with online perturbation augmentation on input tokens.
    Scalar features (Xtr_s) are NOT perturbed."""
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Pre-extract original train tokens (for fresh perturbation each epoch)
    train_tokens = [r.get("input_tokens", []) for r in train_recs]
    N = len(train_tokens)

    # Validation : tokenize once (no perturbation on val)
    ids_v, mask_v = tokenize_records(val_recs, tok2id)
    Idv = torch.from_numpy(ids_v).to(device)
    Mv = torch.from_numpy(mask_v).to(device)
    Sv = torch.from_numpy(Xv_s).to(device)

    Str = torch.from_numpy(Xtr_s).to(device)
    Ytr = torch.from_numpy(ytr).to(device)

    import time
    rng = np.random.default_rng(seed)
    best_auc, best_state, pat = -1.0, None, 0
    for ep in range(max_epochs):
        t0 = time.time()
        # ─ Re-tokenize train with fresh perturbation per epoch
        perturbed = [perturb_input_tokens(t, proper_pool, common_pool, rng,
                                           p_perturb=p_perturb)
                     for t in train_tokens]
        ids_tr, mask_tr = tokenize_token_lists(perturbed, tok2id)
        Idtr = torch.from_numpy(ids_tr).to(device)
        Mtr = torch.from_numpy(mask_tr).to(device)

        model.train()
        idx = torch.randperm(N, device=device)
        loss_sum = 0.0; n_batch = 0
        for start in range(0, N, batch_size):
            sl = idx[start:start + batch_size]
            logits = model(Idtr[sl], Mtr[sl], Str[sl])
            loss = F.binary_cross_entropy_with_logits(logits, Ytr[sl])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item()); n_batch += 1
        model.eval()
        with torch.no_grad():
            sv = torch.sigmoid(model(Idv, Mv, Sv)).cpu().numpy()
        auc = _roc_auc(yv, sv)
        elapsed = time.time() - t0
        marker = "★" if auc > best_auc else " "
        print(f"      {log_prefix}E{ep:02d} {marker} loss={loss_sum/max(n_batch,1):.4f}  "
              f"val_auc={auc:.4f}  pat={pat}  "
              f"({elapsed:.1f}s)", flush=True)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience:
                print(f"      {log_prefix}early stop at epoch {ep}", flush=True)
                break
    model.load_state_dict(best_state)
    return model


def _encoder_score(model, ids, mask, sc):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        return torch.sigmoid(model(
            torch.from_numpy(ids).to(device),
            torch.from_numpy(mask).to(device),
            torch.from_numpy(sc).to(device))).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════
# Eval helpers
# ══════════════════════════════════════════════════════════════════════════
def _per_category_auc(yte, sco, cats):
    cat_to_idx = defaultdict(list)
    for i, c in enumerate(cats):
        cat_to_idx[c].append(i)
    out = {}
    for c, idxs in cat_to_idx.items():
        idxs = np.array(idxs)
        y_c = yte[idxs]; s_c = sco[idxs]
        n_err = int(y_c.sum())
        auc = _roc_auc(y_c, s_c) if (len(idxs) >= 20 and len(set(y_c)) == 2) else float("nan")
        out[c] = {"n": int(len(idxs)), "n_errors": n_err, "auc": auc}
    return out


def _avg_problem(per_cat):
    vs = [per_cat.get(c, {}).get("auc", float("nan")) for c in PROBLEM_CATS]
    valid = [v for v in vs if not math.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


# ══════════════════════════════════════════════════════════════════════════
# Single seed run
# ══════════════════════════════════════════════════════════════════════════
def run_one_seed(seed, train_recs, val_recs, test_recs,
                 proper_pool, common_pool, p_perturb, out_dir,
                 skip_loco=False):
    print(f"\n{'═'*60}\nSEED {seed}\n{'═'*60}")
    tok2id = build_vocab(train_recs)
    print(f"  vocab={len(tok2id)}  proper_pool={len(proper_pool)}  "
          f"common_pool={len(common_pool)}")

    Xtr = _features_matrix(train_recs, SCALAR_FEATURES)
    Xv = _features_matrix(val_recs, SCALAR_FEATURES)
    Xte = _features_matrix(test_recs, SCALAR_FEATURES)
    Xtr_s, _, (mu, sd) = _standardize(Xtr, Xv)
    Xv_s = (Xv - mu) / sd
    Xte_s = (Xte - mu) / sd

    ytr = _labels(train_recs); yv = _labels(val_recs); yte = _labels(test_recs)
    cats_te = _categories(test_recs)

    # ── Train robust encoder
    print("  Training robust encoder (perturbation p_perturb={:.2f})…".format(p_perturb))
    enc = MetaEncoder(vocab_size=len(tok2id), n_scalar_features=4)
    enc = _train_encoder_with_perturbation(
        enc, train_recs, val_recs, tok2id,
        Xtr_s, Xv_s, ytr, yv,
        proper_pool, common_pool, p_perturb, seed,
        log_prefix="[ROBUST] ")

    # ── Condition 1 : test original
    ids_te, mask_te = tokenize_records(test_recs, tok2id)
    s_orig = _encoder_score(enc, ids_te, mask_te, Xte_s)
    auc_orig = _roc_auc(yte, s_orig)
    pc_orig = _per_category_auc(yte, s_orig, cats_te)

    # ── Condition 2 : test perturbé (seed fixé pour repro)
    rng_eval = np.random.default_rng(seed)
    pert_tokens = [perturb_input_tokens(r.get("input_tokens", []),
                                         proper_pool, common_pool, rng_eval,
                                         force=True)
                   for r in test_recs]
    ids_pe, mask_pe = tokenize_token_lists(pert_tokens, tok2id)
    s_pert = _encoder_score(enc, ids_pe, mask_pe, Xte_s)
    auc_pert = _roc_auc(yte, s_pert)
    pc_pert = _per_category_auc(yte, s_pert, cats_te)

    # ── Condition 3 : LOCO retraining (optionnel)
    if skip_loco:
        s_loco = np.full_like(s_orig, np.nan)
        auc_loco = float("nan")
        pc_loco = {}
    else:
        print("  Training LOCO encoder (sans 4 cats prob)…")
        train_loco = [r for r in train_recs if r["category"] not in PROBLEM_CATS]
        Xtr_l = _features_matrix(train_loco, SCALAR_FEATURES)
        Xtr_l_s = (Xtr_l - mu) / sd  # use same standardization
        ytr_l = _labels(train_loco)
        enc_loco = MetaEncoder(vocab_size=len(tok2id), n_scalar_features=4)
        enc_loco = _train_encoder_with_perturbation(
            enc_loco, train_loco, val_recs, tok2id,
            Xtr_l_s, Xv_s, ytr_l, yv,
            proper_pool, common_pool, p_perturb, seed,
            log_prefix="[LOCO]   ")
        s_loco = _encoder_score(enc_loco, ids_te, mask_te, Xte_s)
        auc_loco = _roc_auc(yte, s_loco)
        pc_loco = _per_category_auc(yte, s_loco, cats_te)

    avg_orig = _avg_problem(pc_orig)
    avg_pert = _avg_problem(pc_pert)
    avg_loco = _avg_problem(pc_loco) if pc_loco else float("nan")

    print(f"  AUC global  : orig {auc_orig:.4f}  pert {auc_pert:.4f}  "
          f"loco {auc_loco:.4f}")
    print(f"  Avg 4 prob  : orig {avg_orig:.4f}  pert {avg_pert:.4f}  "
          f"loco {avg_loco:.4f}")

    seed_out = os.path.join(out_dir, f"seed_{seed}")
    os.makedirs(seed_out, exist_ok=True)
    res = {
        "seed": seed,
        "auc_orig_global": auc_orig,
        "auc_pert_global": auc_pert,
        "auc_loco_global": auc_loco,
        "auc_orig_avg_problem": avg_orig,
        "auc_pert_avg_problem": avg_pert,
        "auc_loco_avg_problem": avg_loco,
        "per_category_orig": pc_orig,
        "per_category_pert": pc_pert,
        "per_category_loco": pc_loco,
    }
    with open(os.path.join(seed_out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)

    # ── scores.json (compat A.1 selective prediction)
    scores_dump = {
        "yte": yte.tolist(),
        "cats_test": cats_te,
        "s_enc_te": s_orig.tolist(),       # robust encoder, original test inputs
        "s_enc_te_pert": s_pert.tolist(),  # robust encoder, perturbed test inputs
        "s_enc_te_loco": (s_loco.tolist() if not np.all(np.isnan(s_loco))
                          else None),
        "input_tokens_test": [r.get("input_tokens", []) for r in test_recs],
    }
    with open(os.path.join(seed_out, "scores.json"), "w") as f:
        json.dump(scores_dump, f)
    return res


# ══════════════════════════════════════════════════════════════════════════
# Aggregate + verdict
# ══════════════════════════════════════════════════════════════════════════
def _agg(rs, key):
    vs = [r[key] for r in rs if not math.isnan(r.get(key, float("nan")))]
    if not vs:
        return float("nan"), float("nan")
    return float(np.mean(vs)), float(np.std(vs))


def aggregate_and_verdict(seed_results, out_dir, p_perturb):
    auc_orig_mean, auc_orig_std = _agg(seed_results, "auc_orig_global")
    auc_pert_mean, auc_pert_std = _agg(seed_results, "auc_pert_global")
    auc_loco_mean, auc_loco_std = _agg(seed_results, "auc_loco_global")
    avg_orig_mean, _ = _agg(seed_results, "auc_orig_avg_problem")
    avg_pert_mean, avg_pert_std = _agg(seed_results, "auc_pert_avg_problem")
    avg_loco_mean, avg_loco_std = _agg(seed_results, "auc_loco_avg_problem")

    seeds = [r["seed"] for r in seed_results]

    # ── comparison_with_etape3.md
    lines = ["# Phase B — Robustesse lexicale vs étape 3", ""]
    lines.append(f"- Seeds : {seeds}")
    lines.append(f"- p_perturb : {p_perturb}")
    lines.append("")
    lines.append("## AUC global (3 conditions × 3 seeds)")
    lines.append("")
    lines.append("| Condition | AUC mean | AUC std |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Original  | {auc_orig_mean:.4f} | {auc_orig_std:.4f} |")
    lines.append(f"| Perturbé  | {auc_pert_mean:.4f} | {auc_pert_std:.4f} |")
    lines.append(f"| LOCO      | {auc_loco_mean:.4f} | {auc_loco_std:.4f} |")
    lines.append("")
    lines.append("## AUC moyenne 4 cats prob")
    lines.append("")
    lines.append("| Condition | étape 3 (mémo) | étape B (mean) | std |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Original  | 1.000 | {avg_orig_mean:.4f} | {_agg(seed_results, 'auc_orig_avg_problem')[1]:.4f} |")
    lines.append(f"| Perturbé  | 0.318 | {avg_pert_mean:.4f} | {avg_pert_std:.4f} |")
    lines.append(f"| LOCO      | n/a   | {avg_loco_mean:.4f} | {avg_loco_std:.4f} |")
    lines.append("")
    lines.append("## Per catégorie problématique (mean across seeds)")
    lines.append("")
    lines.append("| Cat | orig | pert | loco |")
    lines.append("|---|---:|---:|---:|")
    for c in PROBLEM_CATS:
        os_ = [r["per_category_orig"].get(c, {}).get("auc", float("nan")) for r in seed_results]
        ps_ = [r["per_category_pert"].get(c, {}).get("auc", float("nan")) for r in seed_results]
        ls_ = [r["per_category_loco"].get(c, {}).get("auc", float("nan")) for r in seed_results]
        def _m(arr):
            v = [a for a in arr if not math.isnan(a)]
            return float(np.mean(v)) if v else float("nan")
        a_o = _m(os_); a_p = _m(ps_); a_l = _m(ls_)
        f_o = "N/A" if math.isnan(a_o) else f"{a_o:.3f}"
        f_p = "N/A" if math.isnan(a_p) else f"{a_p:.3f}"
        f_l = "N/A" if math.isnan(a_l) else f"{a_l:.3f}"
        lines.append(f"| {c} | {f_o} | {f_p} | {f_l} |")
    lines.append("")
    cmp_path = os.path.join(out_dir, "comparison_with_etape3.md")
    with open(cmp_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {cmp_path}")

    # ── verdict per spec §B.5
    vlines = ["# Verdict Phase B (robustesse lexicale)", ""]
    vlines.append(f"- AUC original (mean 3 seeds) : **{auc_orig_mean:.4f}**")
    vlines.append(f"- AUC perturbé sur 4 cats prob : **{avg_pert_mean:.4f}** "
                  f"(étape 3 sans perturbation entraînement : 0.318)")
    vlines.append(f"- AUC LOCO sur 4 cats prob    : **{avg_loco_mean:.4f}**")
    vlines.append("")
    if auc_orig_mean < 0.95:
        vlines.append(f"**PROBLÈME** : AUC original = {auc_orig_mean:.3f} < 0.95. "
                      "La perturbation a dégradé l'apprentissage de base. "
                      f"Réduire p_perturb (actuel = {p_perturb}) à 0.3 ou 0.2.")
    elif (auc_orig_mean >= 0.97 and avg_pert_mean >= 0.80
          and avg_loco_mean >= 0.85):
        vlines.append("**GO Phase C** — robustesse atteinte.")
        vlines.append(f"- AUC orig ≥ 0.97 : ✓ ({auc_orig_mean:.3f})")
        vlines.append(f"- AUC pert sur 4 prob ≥ 0.80 : ✓ ({avg_pert_mean:.3f})")
        vlines.append(f"- AUC LOCO sur 4 prob ≥ 0.85 : ✓ ({avg_loco_mean:.3f})")
    elif (auc_orig_mean >= 0.97 and 0.60 <= avg_pert_mean < 0.80):
        vlines.append("**PARTIEL** — amélioration mais signal lexical résiduel. "
                      "Continuer Phase C avec note. Documenter la limite.")
    elif avg_pert_mean < 0.60:
        vlines.append("**NO-GO** — le signal est intrinsèquement lexical pour ces "
                      "catégories. Repenser (embeddings pré-entraînés ? scope COGS ?).")
    else:
        vlines.append("**Verdict ambigu** — voir détails ci-dessus, retour Julien.")
    vlines.append("")
    vlines.append("Détails dans `comparison_with_etape3.md` et `seed_*/metrics.json`.")
    vp = os.path.join(out_dir, "verdict_B.md")
    with open(vp, "w") as f:
        f.write("\n".join(vlines))
    print(f"Wrote {vp}")
    print()
    print("\n".join(vlines))


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--p-perturb", type=float, default=0.5)
    ap.add_argument("--skip-loco", action="store_true",
                    help="Skip LOCO retraining (utile pour ré-générer scores.json rapidement)")
    ap.add_argument("--problem-cats", default=None,
                    help="Comma-separated category names treated as 'problem cats' "
                         "for LOCO + reporting. Default = COGS hardcoded list.")
    args = ap.parse_args()
    if args.problem_cats:
        global PROBLEM_CATS
        PROBLEM_CATS = [c.strip() for c in args.problem_cats.split(",") if c.strip()]
        print(f"PROBLEM_CATS overridden : {PROBLEM_CATS}")

    os.makedirs(args.output_dir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    records = _load_jsonl(args.dataset)
    with open(args.splits, "r") as f:
        splits = json.load(f)
    train_recs = [records[i] for i in splits["train"]]
    val_recs = [records[i] for i in splits["val"]]
    test_recs = [records[i] for i in splits["test"]]
    print(f"Loaded {len(records)} records "
          f"(train={len(train_recs)}, val={len(val_recs)}, test={len(test_recs)})")

    proper_pool, common_pool = build_lexical_pools(train_recs)
    print(f"Pools : {len(proper_pool)} proper nouns, {len(common_pool)} common top-30")

    seed_results = []
    for s in seeds:
        seed_results.append(run_one_seed(
            s, train_recs, val_recs, test_recs,
            proper_pool, common_pool, args.p_perturb, args.output_dir,
            skip_loco=args.skip_loco))

    if not args.skip_loco:
        aggregate_and_verdict(seed_results, args.output_dir, args.p_perturb)
    else:
        print("\n[--skip-loco activé : pas d'agrégation finale, scores.json sauvés]")


if __name__ == "__main__":
    main()
