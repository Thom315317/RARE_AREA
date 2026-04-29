#!/usr/bin/env python3
"""Étape 3 — Tests de robustesse pour trancher : signal réel ou mémorisation.

Implémente les 3 tests de PROMPT_META_ETAPE3_VERIFICATION.md :

  Test 1 : voisinage Jaccard sur les exemples d'erreur des 4 cats prob
           - critère : si plus proche voisin train est une erreur dans > 50% des
             cas, fuite probable.
  Test 2 : leave-one-category-out (LOCO)
           - train SANS les 4 cats prob, test sur ces 4 cats.
           - critère : AUC ≥ 0.85 → généralise, ≤ 0.6 → mémorisation.
  Test 3 : perturbation lexicale (proper nouns + common nouns)
           - réutilise l'encodeur seed 42 (re-train déterministe), évalue sur
             test perturbé. Critère : AUC perturbé ≥ 0.95 vs ~1.00 original.

Verdict combiné per §6.

Usage :
  python3 meta_etape3_verification.py \\
      --dataset meta_data/cogs_meta_dataset_bplus.jsonl \\
      --splits  meta_data/cogs_meta_splits.json \\
      --output-dir runs_meta \\
      --seed 42
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
PROBLEM_CATS = ["subj_to_obj_common", "passive_to_active",
                "unacc_to_transitive", "obj_omitted_transitive_to_transitive"]
PAD_TOK = "<pad>"
UNK_TOK = "<unk>"
MAX_LEN = 64

DETERMINERS = {"A", "An", "The", "a", "an", "the"}
FUNCTION_WORDS = {"that", "was", "were", "by", "to", "in", "on", "."}


# ══════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════
def _load_jsonl(path):
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


def build_vocab(records):
    counter = Counter()
    for r in records:
        for t in r.get("input_tokens", []):
            counter[t] += 1
    tok2id = {PAD_TOK: 0, UNK_TOK: 1}
    for t, _ in counter.most_common():
        if t not in tok2id:
            tok2id[t] = len(tok2id)
    return tok2id


def tokenize_records(records, tok2id, max_len=MAX_LEN):
    N = len(records)
    ids = np.zeros((N, max_len), dtype=np.int64)
    mask = np.zeros((N, max_len), dtype=np.float32)
    for i, r in enumerate(records):
        toks = r.get("input_tokens", [])[:max_len]
        for j, t in enumerate(toks):
            ids[i, j] = tok2id.get(t, tok2id[UNK_TOK])
            mask[i, j] = 1.0
    return ids, mask


def tokenize_token_lists(token_lists, tok2id, max_len=MAX_LEN):
    """Comme tokenize_records mais sur des listes de tokens directement."""
    N = len(token_lists)
    ids = np.zeros((N, max_len), dtype=np.int64)
    mask = np.zeros((N, max_len), dtype=np.float32)
    for i, toks in enumerate(token_lists):
        for j, t in enumerate(toks[:max_len]):
            ids[i, j] = tok2id.get(t, tok2id[UNK_TOK])
            mask[i, j] = 1.0
    return ids, mask


def _roc_auc(y_true, scores):
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float("nan")


# ══════════════════════════════════════════════════════════════════════════
# MetaEncoder + train (identique étape 3)
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


def _set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    best_auc, best_state, pat = -1.0, None, 0
    N = Idtr.size(0)
    for ep in range(max_epochs):
        model.train()
        idx = torch.randperm(N, device=device)
        for start in range(0, N, batch_size):
            sl = idx[start:start + batch_size]
            logits = model(Idtr[sl], Mtr[sl], Str[sl])
            loss = F.binary_cross_entropy_with_logits(logits, Ytr[sl])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            sv = torch.sigmoid(model(Idv, Mv, Sv)).cpu().numpy()
        auc = _roc_auc(yv, sv)
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


def _encoder_score(model, ids, mask, sc):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        s = torch.sigmoid(model(
            torch.from_numpy(ids).to(device),
            torch.from_numpy(mask).to(device),
            torch.from_numpy(sc).to(device))).cpu().numpy()
    return s


# ══════════════════════════════════════════════════════════════════════════
# TEST 1 — Voisinage Jaccard
# ══════════════════════════════════════════════════════════════════════════
def jaccard(a, b):
    sa = set(a); sb = set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def run_test1(train_recs, test_recs, out_dir):
    print("\n══ TEST 1 — Voisinage Jaccard ══")
    train_token_sets = [set(r.get("input_tokens", [])) for r in train_recs]
    train_labels = [int(not r["exact_match"]) for r in train_recs]
    train_inputs = [" ".join(r.get("input_tokens", [])) for r in train_recs]

    lines = ["# Test 1 — Voisinage Jaccard sur exemples d'erreur (4 cats prob)",
             "",
             "Pour chaque erreur du meta-test sur les 4 catégories problématiques, "
             "on cherche les 5 plus proches voisins du meta-train (similarité Jaccard "
             "sur les tokens d'input).",
             "",
             "**Critère** : si le top-1 train est `error` (label=False) dans > 50% des "
             "cas → fuite probable. ≤ 30% → pas de fuite directe.",
             ""]

    n_top1_error = 0
    n_total = 0
    error_test_indices = [i for i, r in enumerate(test_recs)
                          if r["category"] in PROBLEM_CATS and not r["exact_match"]]
    print(f"  {len(error_test_indices)} erreurs sur les 4 cats prob")

    for ti in error_test_indices:
        r = test_recs[ti]
        inp = r.get("input_tokens", [])
        s_inp = set(inp)
        sims = []
        for j, ts in enumerate(train_token_sets):
            if not s_inp and not ts:
                sim = 0.0
            else:
                sim = len(s_inp & ts) / len(s_inp | ts)
            sims.append((sim, j))
        sims.sort(key=lambda x: -x[0])
        top5 = sims[:5]

        lines.append(f"### {r['category']} — meta-test idx {ti}")
        lines.append("")
        lines.append(f"- INPUT TEST (erreur) : `{' '.join(inp)}`")
        lines.append("- 5 plus proches voisins TRAIN (sim, label, input) :")
        lines.append("")
        for sim, j in top5:
            lab = "**ERROR**" if train_labels[j] == 1 else "correct"
            lines.append(f"  - {sim:.3f} {lab} `{train_inputs[j]}`")
        # Top-1 label
        top1_lab = train_labels[top5[0][1]]
        if top1_lab == 1:
            n_top1_error += 1
        n_total += 1
        lines.append("")

    pct = (100 * n_top1_error / n_total) if n_total else 0.0
    lines.append(f"## Résumé")
    lines.append("")
    lines.append(f"- Erreurs test analysées : **{n_total}**")
    lines.append(f"- Top-1 voisin train est aussi une erreur : "
                 f"**{n_top1_error}/{n_total} ({pct:.0f}%)**")
    lines.append("")
    if pct > 50:
        verdict = "KO"
        lines.append(f"**Verdict Test 1 : KO ({pct:.0f}% > 50%)** — fuite probable. "
                     "Le voisinage train de chaque erreur test contient typiquement une "
                     "erreur similaire.")
    elif pct <= 30:
        verdict = "OK"
        lines.append(f"**Verdict Test 1 : OK ({pct:.0f}% ≤ 30%)** — pas de fuite directe "
                     "par voisinage Jaccard.")
    else:
        verdict = "NUANCED"
        lines.append(f"**Verdict Test 1 : zone grise ({pct:.0f}%)** — voir tests 2 et 3.")

    with open(os.path.join(out_dir, "etape3_test1_voisinage.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"  → top-1 voisin = erreur dans {pct:.0f}% des cas  [{verdict}]")
    return {"verdict": verdict, "pct_top1_error": pct, "n": n_total}


# ══════════════════════════════════════════════════════════════════════════
# TEST 2 — LOCO
# ══════════════════════════════════════════════════════════════════════════
def run_test2(train_recs, val_recs, test_recs, seed, out_dir, original_per_cat):
    print("\n══ TEST 2 — Leave-one-category-out ══")
    # Filter train
    train_filt = [r for r in train_recs if r["category"] not in PROBLEM_CATS]
    val_filt = val_recs  # spec: meta-val ORIGINAL (inchangé)
    print(f"  train filtré : {len(train_filt)} (vs {len(train_recs)})")
    print(f"  val original : {len(val_filt)}")
    print(f"  test original : {len(test_recs)}")

    tok2id = build_vocab(train_filt)
    print(f"  vocab (train filt) : {len(tok2id)}")

    ids_tr, mask_tr = tokenize_records(train_filt, tok2id)
    ids_v, mask_v = tokenize_records(val_filt, tok2id)
    ids_te, mask_te = tokenize_records(test_recs, tok2id)

    Xtr = _features_matrix(train_filt, SCALAR_FEATURES)
    Xv = _features_matrix(val_filt, SCALAR_FEATURES)
    Xte = _features_matrix(test_recs, SCALAR_FEATURES)
    Xtr_s, _, (mu, sd) = _standardize(Xtr, Xv)
    Xv_s = (Xv - mu) / sd
    Xte_s = (Xte - mu) / sd

    ytr = _labels(train_filt); yv = _labels(val_filt); yte = _labels(test_recs)
    cats_test = _categories(test_recs)

    _set_seed(seed)
    enc = MetaEncoder(vocab_size=len(tok2id), n_scalar_features=4)
    enc = _train_encoder(enc, ids_tr, mask_tr, Xtr_s, ytr,
                         ids_v, mask_v, Xv_s, yv, seed=seed)
    s_te = _encoder_score(enc, ids_te, mask_te, Xte_s)

    # Per-cat
    cat_to_idx = defaultdict(list)
    for i, c in enumerate(cats_test):
        cat_to_idx[c].append(i)
    auc_per_cat = {}
    for c, idxs in cat_to_idx.items():
        idxs = np.array(idxs)
        y_c = yte[idxs]; s_c = s_te[idxs]
        if len(set(y_c)) == 2 and len(idxs) >= 20:
            auc_per_cat[c] = _roc_auc(y_c, s_c)
        else:
            auc_per_cat[c] = float("nan")

    auc_problem_global = _roc_auc(
        yte[np.array([i for i, c in enumerate(cats_test) if c in PROBLEM_CATS])],
        s_te[np.array([i for i, c in enumerate(cats_test) if c in PROBLEM_CATS])])
    auc_other_global = _roc_auc(
        yte[np.array([i for i, c in enumerate(cats_test) if c not in PROBLEM_CATS])],
        s_te[np.array([i for i, c in enumerate(cats_test) if c not in PROBLEM_CATS])])
    auc_global = _roc_auc(yte, s_te)

    avg_problem = float(np.mean([auc_per_cat[c] for c in PROBLEM_CATS
                                  if not math.isnan(auc_per_cat.get(c, float("nan")))]))

    lines = ["# Test 2 — Leave-one-category-out (LOCO)",
             "",
             f"- Train filtré : meta-train original moins toutes les exemples des 4 "
             f"cats problématiques (n={len(train_filt)})",
             f"- Val : meta-val ORIGINAL (n={len(val_filt)})",
             f"- Test : meta-test ORIGINAL (n={len(test_recs)})",
             f"- Seed : {seed}",
             "",
             "## AUC par catégorie problématique",
             "",
             "| Cat | AUC LOCO | AUC original (étape 3) |",
             "|---|---:|---:|"]
    for c in PROBLEM_CATS:
        loco = auc_per_cat.get(c, float("nan"))
        orig = original_per_cat.get(c, float("nan"))
        loco_s = "N/A" if math.isnan(loco) else f"{loco:.3f}"
        orig_s = "N/A" if math.isnan(orig) else f"{orig:.3f}"
        lines.append(f"| {c} | {loco_s} | {orig_s} |")
    lines.append("")
    lines.append(f"- AUC global meta-test (LOCO)      : {auc_global:.4f}")
    lines.append(f"- AUC sur 4 cats prob (LOCO)       : **{auc_problem_global:.4f}**")
    lines.append(f"- AUC sur autres cats (LOCO)       : {auc_other_global:.4f}")
    lines.append(f"- AUC moyenne par cat prob (LOCO)  : **{avg_problem:.4f}**")
    lines.append("")
    if avg_problem >= 0.85:
        verdict = "OK"
        lines.append(f"**Verdict Test 2 : OK** ({avg_problem:.3f} ≥ 0.85). "
                     "Le signal généralise ; pas de mémorisation par catégorie.")
    elif avg_problem <= 0.6:
        verdict = "KO"
        lines.append(f"**Verdict Test 2 : KO** ({avg_problem:.3f} ≤ 0.6). "
                     "C'était essentiellement de la mémorisation par catégorie. "
                     "NO-GO sur l'étape 3 actuelle.")
    else:
        verdict = "NUANCED"
        lines.append(f"**Verdict Test 2 : nuancé** ({avg_problem:.3f} ∈ ]0.6, 0.85[). "
                     "Signal partiel ; mémorisation partielle.")

    with open(os.path.join(out_dir, "etape3_test2_leaveout.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"  → AUC LOCO sur 4 cats prob = {avg_problem:.4f}  [{verdict}]")
    return {"verdict": verdict,
            "auc_problem_avg": avg_problem,
            "auc_problem_global": auc_problem_global,
            "auc_other_global": auc_other_global,
            "auc_global": auc_global,
            "auc_per_cat": auc_per_cat}


# ══════════════════════════════════════════════════════════════════════════
# TEST 3 — Perturbation lexicale
# ══════════════════════════════════════════════════════════════════════════
def _classify_token(tok):
    """Renvoie 'proper', 'common', ou 'other' selon des heuristiques simples."""
    if not tok or not tok[0].isalpha():
        return "other"
    if tok in DETERMINERS or tok in FUNCTION_WORDS:
        return "other"
    if tok[0].isupper() and len(tok) > 1 and tok.isalpha():
        return "proper"
    if tok.islower() and tok.isalpha() and len(tok) > 1:
        return "common_or_verb"
    return "other"


def build_lexical_pools(train_recs):
    """Retourne (proper_nouns_set, common_nouns_top30_set)."""
    proper_set = set()
    common_freq = Counter()
    for r in train_recs:
        toks = r.get("input_tokens", [])
        for i, t in enumerate(toks):
            cls = _classify_token(t)
            if cls == "proper":
                # Skip sentence-initial capitalized non-proper-nouns: i==0 not a problem
                # since COGS proper nouns appear sentence-initial too; we accept.
                proper_set.add(t)
            elif cls == "common_or_verb":
                # Heuristique : noms communs apparaissent souvent après un déterminant.
                if i > 0 and toks[i - 1] in DETERMINERS:
                    common_freq[t] += 1
    common_top = set(t for t, _ in common_freq.most_common(30))
    return proper_set, common_top


def perturb_input(input_tokens, proper_pool_list, common_pool_list, rng):
    """Remplace chaque proper noun par un autre proper noun aléatoire (consistant
    par sentence : le même nom est remplacé par le même), idem pour common noun.
    Préserve déterminants, function words, verbes."""
    proper_used = sorted({t for t in input_tokens if t in proper_pool_list})
    common_used = sorted({t for t, idx in zip(input_tokens, range(len(input_tokens)))
                          if t in common_pool_list and idx > 0
                          and input_tokens[idx - 1] in DETERMINERS})
    proper_map = {}
    if proper_used:
        candidates = [p for p in proper_pool_list if p not in proper_used]
        if candidates:
            picks = rng.choice(candidates, size=len(proper_used),
                               replace=len(candidates) < len(proper_used))
            for old, new in zip(proper_used, picks):
                proper_map[old] = new
    common_map = {}
    if common_used:
        candidates = [c for c in common_pool_list if c not in common_used]
        if candidates:
            picks = rng.choice(candidates, size=len(common_used),
                               replace=len(candidates) < len(common_used))
            for old, new in zip(common_used, picks):
                common_map[old] = new
    out = []
    for i, t in enumerate(input_tokens):
        if t in proper_map:
            out.append(str(proper_map[t]))
        elif (t in common_map and i > 0 and input_tokens[i - 1] in DETERMINERS):
            out.append(str(common_map[t]))
        else:
            out.append(t)
    return out


def run_test3(train_recs, val_recs, test_recs, seed, out_dir):
    print("\n══ TEST 3 — Perturbation lexicale ══")
    # Re-train encoder seed-42 sur splits originaux (déterministe)
    tok2id = build_vocab(train_recs)
    print(f"  vocab original : {len(tok2id)}")

    ids_tr, mask_tr = tokenize_records(train_recs, tok2id)
    ids_v, mask_v = tokenize_records(val_recs, tok2id)
    ids_te, mask_te = tokenize_records(test_recs, tok2id)

    Xtr = _features_matrix(train_recs, SCALAR_FEATURES)
    Xv = _features_matrix(val_recs, SCALAR_FEATURES)
    Xte = _features_matrix(test_recs, SCALAR_FEATURES)
    Xtr_s, _, (mu, sd) = _standardize(Xtr, Xv)
    Xv_s = (Xv - mu) / sd
    Xte_s = (Xte - mu) / sd

    ytr = _labels(train_recs); yv = _labels(val_recs); yte = _labels(test_recs)
    cats_test = _categories(test_recs)

    _set_seed(seed)
    enc = MetaEncoder(vocab_size=len(tok2id), n_scalar_features=4)
    enc = _train_encoder(enc, ids_tr, mask_tr, Xtr_s, ytr,
                         ids_v, mask_v, Xv_s, yv, seed=seed)

    # Score original
    s_te_orig = _encoder_score(enc, ids_te, mask_te, Xte_s)
    auc_orig = _roc_auc(yte, s_te_orig)

    # Build pools
    proper_pool, common_pool = build_lexical_pools(train_recs)
    proper_pool_list = sorted(proper_pool)
    common_pool_list = sorted(common_pool)
    print(f"  proper nouns pool : {len(proper_pool_list)}")
    print(f"  common nouns pool : {len(common_pool_list)} (top-30)")

    # Perturb test
    rng = np.random.default_rng(seed)
    perturbed_tokens = []
    for r in test_recs:
        new = perturb_input(r.get("input_tokens", []),
                            proper_pool_list, common_pool_list, rng)
        perturbed_tokens.append(new)

    # Sanity samples
    sample_lines = []
    for i in range(5):
        sample_lines.append(f"  ex{i}: ORIG `{' '.join(test_recs[i].get('input_tokens', []))}`")
        sample_lines.append(f"        PERT `{' '.join(perturbed_tokens[i])}`")

    ids_pe, mask_pe = tokenize_token_lists(perturbed_tokens, tok2id)
    s_te_pert = _encoder_score(enc, ids_pe, mask_pe, Xte_s)
    auc_pert = _roc_auc(yte, s_te_pert)

    # Per-cat AUC perturbé pour les 4 cats prob
    cat_to_idx = defaultdict(list)
    for i, c in enumerate(cats_test):
        cat_to_idx[c].append(i)
    auc_pert_per_cat = {}
    for c, idxs in cat_to_idx.items():
        idxs = np.array(idxs)
        y_c = yte[idxs]; s_c = s_te_pert[idxs]
        if len(set(y_c)) == 2 and len(idxs) >= 20:
            auc_pert_per_cat[c] = _roc_auc(y_c, s_c)
        else:
            auc_pert_per_cat[c] = float("nan")
    auc_orig_per_cat = {}
    for c, idxs in cat_to_idx.items():
        idxs = np.array(idxs)
        y_c = yte[idxs]; s_c = s_te_orig[idxs]
        if len(set(y_c)) == 2 and len(idxs) >= 20:
            auc_orig_per_cat[c] = _roc_auc(y_c, s_c)
        else:
            auc_orig_per_cat[c] = float("nan")

    avg_problem_orig = float(np.mean(
        [auc_orig_per_cat[c] for c in PROBLEM_CATS
         if not math.isnan(auc_orig_per_cat.get(c, float("nan")))]))
    avg_problem_pert = float(np.mean(
        [auc_pert_per_cat[c] for c in PROBLEM_CATS
         if not math.isnan(auc_pert_per_cat.get(c, float("nan")))]))

    lines = ["# Test 3 — Perturbation lexicale (proper + common nouns)",
             "",
             f"- Pool proper nouns (extrait du train) : {len(proper_pool_list)} mots",
             f"- Pool common nouns top-30 : {len(common_pool_list)} mots",
             f"- Perturbation : chaque proper noun et chaque common noun (post-déterminant) "
             "est remplacé par un mot aléatoire du même pool, consistant par sentence.",
             f"- Verbes / déterminants / function words préservés.",
             "- Labels `exact_match` du test conservés (limite : approximation, voir §5.4 spec).",
             "",
             "## Échantillons (5 premiers test examples)",
             ""]
    lines.extend(sample_lines)
    lines.append("")
    lines.append("## AUC global")
    lines.append("")
    lines.append("| Variant | AUC global |")
    lines.append("|---|---:|")
    lines.append(f"| Original   | {auc_orig:.4f} |")
    lines.append(f"| Perturbé   | {auc_pert:.4f} |")
    lines.append(f"| Δ (orig − pert) | {auc_orig - auc_pert:+.4f} |")
    lines.append("")
    lines.append("## AUC par catégorie problématique")
    lines.append("")
    lines.append("| Cat | AUC orig | AUC pert | Δ |")
    lines.append("|---|---:|---:|---:|")
    for c in PROBLEM_CATS:
        ao = auc_orig_per_cat.get(c, float("nan"))
        ap = auc_pert_per_cat.get(c, float("nan"))
        d = ao - ap if not (math.isnan(ao) or math.isnan(ap)) else float("nan")
        ao_s = "N/A" if math.isnan(ao) else f"{ao:.3f}"
        ap_s = "N/A" if math.isnan(ap) else f"{ap:.3f}"
        d_s = "N/A" if math.isnan(d) else f"{d:+.3f}"
        lines.append(f"| {c} | {ao_s} | {ap_s} | {d_s} |")
    lines.append("")
    lines.append(f"- AUC moyenne 4 cats prob (orig) : **{avg_problem_orig:.3f}**")
    lines.append(f"- AUC moyenne 4 cats prob (pert) : **{avg_problem_pert:.3f}**")
    lines.append("")
    if auc_pert >= 0.95:
        verdict = "OK"
        lines.append(f"**Verdict Test 3 : OK** (AUC pert {auc_pert:.4f} ≥ 0.95). "
                     "Pattern robuste ; le signal n'est pas purement lexical.")
    elif auc_pert <= 0.85:
        verdict = "KO"
        lines.append(f"**Verdict Test 3 : KO** (AUC pert {auc_pert:.4f} ≤ 0.85). "
                     "Le signal est essentiellement lexical ; perturber les noms casse "
                     "la prédiction.")
    else:
        verdict = "NUANCED"
        lines.append(f"**Verdict Test 3 : nuancé** (AUC pert ∈ ]0.85, 0.95[). "
                     "Robuste mais sensible.")

    with open(os.path.join(out_dir, "etape3_test3_perturbation.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"  → AUC orig {auc_orig:.4f} / AUC pert {auc_pert:.4f}  [{verdict}]")
    return {"verdict": verdict, "auc_orig": auc_orig, "auc_pert": auc_pert,
            "auc_problem_orig": avg_problem_orig,
            "auc_problem_pert": avg_problem_pert,
            "auc_orig_per_cat": auc_orig_per_cat,
            "auc_pert_per_cat": auc_pert_per_cat}


# ══════════════════════════════════════════════════════════════════════════
# Verdict combiné
# ══════════════════════════════════════════════════════════════════════════
def write_final(t1, t2, t3, out_dir):
    v1 = t1.get("verdict") if t1 else None
    v2 = t2.get("verdict") if t2 else None
    v3 = t3.get("verdict") if t3 else None

    lines = ["# Étape 3 — Verdict de vérification (synthèse)", ""]
    lines.append("| Test | Verdict | Métrique |")
    lines.append("|---|---|---|")
    if t1: lines.append(f"| 1 — Voisinage Jaccard | {v1} | "
                        f"top-1 voisin = erreur dans {t1.get('pct_top1_error', 0):.0f}% |")
    if t2: lines.append(f"| 2 — Leave-one-category-out | {v2} | "
                        f"AUC moyenne 4 cats prob = {t2.get('auc_problem_avg', float('nan')):.3f} |")
    if t3: lines.append(f"| 3 — Perturbation lexicale | {v3} | "
                        f"AUC orig {t3.get('auc_orig', float('nan')):.3f} → "
                        f"pert {t3.get('auc_pert', float('nan')):.3f} |")
    lines.append("")

    # Per spec §6 grid
    if v1 == "KO":
        verdict_final = ("**NO-GO** — fuite directe par voisinage. Le dataset meta a un "
                         "problème structurel.")
    elif v2 == "KO":
        verdict_final = ("**NO-GO** — mémorisation par catégorie confirmée. Repenser le "
                         "splitting ou l'architecture.")
    elif v1 == "OK" and v2 == "OK" and v3 == "OK":
        verdict_final = ("**Signal réel confirmé**. GO étape 4.")
    elif v1 == "OK" and v2 == "OK" and v3 == "KO":
        verdict_final = ("Signal réel mais sensible aux tokens. **GO étape 4 nuancé** : "
                         "documenter, prévoir variation lexicale plus grande au train du "
                         "meta-modèle.")
    else:
        verdict_final = ("**Verdict mixte**. À discuter avec Julien — voir détails par test.")
    lines.append("## Verdict combiné")
    lines.append("")
    lines.append(verdict_final)
    lines.append("")
    lines.append("Détails dans :")
    lines.append("- `etape3_test1_voisinage.md`")
    lines.append("- `etape3_test2_leaveout.md`")
    lines.append("- `etape3_test3_perturbation.md`")

    with open(os.path.join(out_dir, "etape3_verification_finale.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"\n{verdict_final}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tests", default="1,2,3",
                    help="Comma-separated test ids to run (1,2,3)")
    ap.add_argument("--original-scores",
                    default="runs_meta/etape3_s42/scores.json",
                    help="Pour Test 2, comparer avec AUC original par cat")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    records = _load_jsonl(args.dataset)
    with open(args.splits, "r") as f:
        splits = json.load(f)
    train_recs = [records[i] for i in splits["train"]]
    val_recs = [records[i] for i in splits["val"]]
    test_recs = [records[i] for i in splits["test"]]
    print(f"Loaded {len(records)} records "
          f"(train={len(train_recs)}, val={len(val_recs)}, test={len(test_recs)})")

    # Load original per-cat AUC for Test 2 comparison
    original_per_cat = {}
    if os.path.exists(args.original_scores):
        with open(args.original_scores, "r") as f:
            sd0 = json.load(f)
        cats0 = sd0["cats_test"]; sco0 = np.array(sd0["s_enc_te"])
        yte0 = np.array(sd0["yte"])
        for c in PROBLEM_CATS:
            mask = np.array([cc == c for cc in cats0])
            if mask.sum() == 0:
                continue
            y_c = yte0[mask]; s_c = sco0[mask]
            if len(set(y_c)) == 2:
                original_per_cat[c] = _roc_auc(y_c, s_c)

    tests_to_run = [int(s) for s in args.tests.split(",")]
    t1 = t2 = t3 = None
    if 1 in tests_to_run:
        t1 = run_test1(train_recs, test_recs, args.output_dir)
    if 2 in tests_to_run:
        t2 = run_test2(train_recs, val_recs, test_recs, args.seed,
                       args.output_dir, original_per_cat)
    if 3 in tests_to_run:
        t3 = run_test3(train_recs, val_recs, test_recs, args.seed,
                       args.output_dir)

    write_final(t1, t2, t3, args.output_dir)


if __name__ == "__main__":
    main()
