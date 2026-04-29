#!/usr/bin/env python3
"""Vérifications de robustesse étape 3.

Tests :
  1. Toutes les erreurs des 4 cats problématiques avec scores 3 seeds
     (consistance cross-seed)
  2. Top tokens discriminants entre erreurs et corrects (le pattern "on the TV"
     est-il vraiment un leak, ou bien c'est juste que les erreurs co-occurrent
     avec des modifs PP ?)
  3. Leave-one-category-out : ré-entraîne l'encodeur sur train SANS les 4 cats
     problématiques, test SUR les 4 cats. Si AUC chute drastiquement → le
     signal est category-specific (préoccupant). Si AUC reste élevé → le
     signal est structural-général (rassurant).

Usage :
  python3 verify_etape3.py \\
      --dataset meta_data/cogs_meta_dataset_bplus.jsonl \\
      --splits  meta_data/cogs_meta_splits.json \\
      --runs runs_meta/etape3_s42,runs_meta/etape3_s123,runs_meta/etape3_s456 \\
      --output runs_meta/etape3_verification \\
      --do-loco
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


# ══════════════════════════════════════════════════════════════════════════
# Helpers (re-imported from meta_train_etape3 for self-containment)
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


def _roc_auc(y_true, scores):
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float("nan")


# ══════════════════════════════════════════════════════════════════════════
# MetaEncoder (identique étape 3)
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


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--runs", required=True,
                    help="Comma-separated dirs containing scores.json")
    ap.add_argument("--output", required=True)
    ap.add_argument("--do-loco", action="store_true",
                    help="Also run leave-one-category-out retraining (1 seed)")
    ap.add_argument("--loco-seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load all 3 seeds' scores
    run_dirs = [d.strip() for d in args.runs.split(",")]
    scores_per_seed = []
    for d in run_dirs:
        with open(os.path.join(d, "scores.json"), "r") as f:
            scores_per_seed.append(json.load(f))
    print(f"Loaded scores from {len(scores_per_seed)} seeds")

    # ──────────────────────────────────────────────────────────────────
    # TEST 1 : Toutes les erreurs des 4 cats problématiques, scores cross-seed
    # ──────────────────────────────────────────────────────────────────
    print("\n── TEST 1 : erreurs sur cats problématiques (cross-seed) ──")
    sd0 = scores_per_seed[0]
    yte = np.array(sd0["yte"])
    cats = sd0["cats_test"]
    toks = sd0["input_tokens_test"]
    error_indices = [i for i, c in enumerate(cats)
                     if c in PROBLEM_CATS and yte[i] == 1]
    print(f"  {len(error_indices)} erreurs total sur 4 cats × 150 ex")
    print()

    test1_lines = ["# Test 1 — Erreurs cats problématiques, scores cross-seed",
                   "",
                   "| idx | cat | score_s42 | score_s123 | score_s456 | input |",
                   "|---|---|---:|---:|---:|---|"]
    for i in error_indices:
        scores = []
        for sd in scores_per_seed:
            scores.append(sd["s_enc_te"][i])
        scores_str = " | ".join(f"{s:.3f}" for s in scores)
        inp = " ".join(toks[i])
        cat = cats[i]
        # Mark agreement
        all_high = all(s >= 0.5 for s in scores)
        marker = "✓" if all_high else "✗"
        test1_lines.append(f"| {i} | {cat} | {scores[0]:.3f} | {scores[1]:.3f} | "
                           f"{scores[2]:.3f} | {marker} `{inp}` |")
    n_consistent = sum(1 for i in error_indices
                       if all(sd["s_enc_te"][i] >= 0.5 for sd in scores_per_seed))
    test1_lines.append("")
    test1_lines.append(f"- **{n_consistent}/{len(error_indices)}** erreurs "
                       "détectées (score ≥ 0.5) par les 3 seeds simultanément")
    print(f"  {n_consistent}/{len(error_indices)} erreurs détectées par 3 seeds")

    # ──────────────────────────────────────────────────────────────────
    # TEST 2 : Tokens discriminants
    # ──────────────────────────────────────────────────────────────────
    print("\n── TEST 2 : tokens discriminants entre erreurs et corrects ──")
    err_tok_count = Counter()
    cor_tok_count = Counter()
    n_err = 0; n_cor = 0
    for i in range(len(yte)):
        if yte[i] == 1:
            n_err += 1
            for t in set(toks[i]):
                err_tok_count[t] += 1
        else:
            n_cor += 1
            for t in set(toks[i]):
                cor_tok_count[t] += 1
    # Lift = P(token | error) / P(token | correct)
    lifts = []
    for t in set(err_tok_count) | set(cor_tok_count):
        p_err = err_tok_count[t] / max(n_err, 1)
        p_cor = cor_tok_count[t] / max(n_cor, 1)
        if p_err >= 0.05 and p_cor >= 0.001:  # filtrer le bruit
            lifts.append((t, p_err, p_cor, p_err / max(p_cor, 1e-6)))
    lifts.sort(key=lambda x: -x[3])

    test2_lines = ["# Test 2 — Tokens discriminants (P(t|err) / P(t|cor))",
                   "",
                   f"- {n_err} erreurs vs {n_cor} corrects sur meta-test",
                   "",
                   "## Top 20 tokens enrichis dans les erreurs",
                   "",
                   "| token | P(t|err) | P(t|cor) | lift |",
                   "|---|---:|---:|---:|"]
    for t, pe, pc, lift in lifts[:20]:
        test2_lines.append(f"| `{t}` | {100*pe:.1f}% | {100*pc:.1f}% | {lift:.2f} |")

    test2_lines.append("")
    test2_lines.append("## Restriction aux 4 cats problématiques")
    test2_lines.append("")
    err_tok_p = Counter(); cor_tok_p = Counter()
    np_err = 0; np_cor = 0
    for i in range(len(yte)):
        if cats[i] not in PROBLEM_CATS:
            continue
        if yte[i] == 1:
            np_err += 1
            for t in set(toks[i]):
                err_tok_p[t] += 1
        else:
            np_cor += 1
            for t in set(toks[i]):
                cor_tok_p[t] += 1
    lifts_p = []
    for t in set(err_tok_p) | set(cor_tok_p):
        pe = err_tok_p[t] / max(np_err, 1)
        pc = cor_tok_p[t] / max(np_cor, 1)
        if pe >= 0.20 and pc >= 0.001:
            lifts_p.append((t, pe, pc, pe / max(pc, 1e-6)))
    lifts_p.sort(key=lambda x: -x[3])
    test2_lines.append(f"- {np_err} erreurs vs {np_cor} corrects (cats prob)")
    test2_lines.append("")
    test2_lines.append("| token | P(t|err) | P(t|cor) | lift |")
    test2_lines.append("|---|---:|---:|---:|")
    for t, pe, pc, lift in lifts_p[:15]:
        test2_lines.append(f"| `{t}` | {100*pe:.1f}% | {100*pc:.1f}% | {lift:.2f} |")
    print(f"  top tokens err: {[t for t, *_ in lifts[:5]]}")
    print(f"  top tokens err (prob cats): {[t for t, *_ in lifts_p[:5]]}")

    # ──────────────────────────────────────────────────────────────────
    # TEST 3 : Leave-one-category-out (4 problem cats)
    # ──────────────────────────────────────────────────────────────────
    test3_lines = ["# Test 3 — Leave-one-category-out (LOCO)", ""]
    if not args.do_loco:
        test3_lines.append("_Skip (--do-loco non passé)._")
    else:
        print("\n── TEST 3 : LOCO (re-train sans les 4 cats prob) ──")
        records = _load_jsonl(args.dataset)
        with open(args.splits, "r") as f:
            splits = json.load(f)
        train_recs = [records[i] for i in splits["train"]]
        val_recs = [records[i] for i in splits["val"]]
        test_recs = [records[i] for i in splits["test"]]

        # Filter train + val to exclude problem cats
        train_filt = [r for r in train_recs if r["category"] not in PROBLEM_CATS]
        val_filt = [r for r in val_recs if r["category"] not in PROBLEM_CATS]
        # Test = original (we want to evaluate on problem cats specifically)
        test_problem = [r for r in test_recs if r["category"] in PROBLEM_CATS]
        test_other = [r for r in test_recs if r["category"] not in PROBLEM_CATS]

        print(f"  train filtered : {len(train_filt)} (vs {len(train_recs)})")
        print(f"  val filtered   : {len(val_filt)}")
        print(f"  test problem   : {len(test_problem)}")
        print(f"  test other     : {len(test_other)}")

        # Build vocab from filtered train
        tok2id = build_vocab(train_filt)
        print(f"  vocab (train filt) : {len(tok2id)}")

        ids_tr, mask_tr = tokenize_records(train_filt, tok2id)
        ids_v, mask_v = tokenize_records(val_filt, tok2id)
        ids_p, mask_p = tokenize_records(test_problem, tok2id)
        ids_o, mask_o = tokenize_records(test_other, tok2id)

        Xtr = _features_matrix(train_filt, SCALAR_FEATURES)
        Xv = _features_matrix(val_filt, SCALAR_FEATURES)
        Xp = _features_matrix(test_problem, SCALAR_FEATURES)
        Xo = _features_matrix(test_other, SCALAR_FEATURES)
        Xtr_s, _, (mu, sd) = _standardize(Xtr, Xv)
        Xv_s = (Xv - mu) / sd
        Xp_s = (Xp - mu) / sd
        Xo_s = (Xo - mu) / sd

        ytr = _labels(train_filt); yv = _labels(val_filt)
        yp = _labels(test_problem); yo = _labels(test_other)

        # ── Train encoder LOCO
        random.seed(args.loco_seed); np.random.seed(args.loco_seed)
        torch.manual_seed(args.loco_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.loco_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        encoder_loco = MetaEncoder(vocab_size=len(tok2id), n_scalar_features=4)
        encoder_loco = _train_encoder(encoder_loco, ids_tr, mask_tr, Xtr_s, ytr,
                                       ids_v, mask_v, Xv_s, yv,
                                       seed=args.loco_seed)
        encoder_loco.eval()
        device = next(encoder_loco.parameters()).device
        with torch.no_grad():
            s_p = torch.sigmoid(encoder_loco(
                torch.from_numpy(ids_p).to(device),
                torch.from_numpy(mask_p).to(device),
                torch.from_numpy(Xp_s).to(device))).cpu().numpy()
            s_o = torch.sigmoid(encoder_loco(
                torch.from_numpy(ids_o).to(device),
                torch.from_numpy(mask_o).to(device),
                torch.from_numpy(Xo_s).to(device))).cpu().numpy()

        # Per-category AUC on problem cats
        cats_p = _categories(test_problem)
        cat_to_idx_p = defaultdict(list)
        for i, c in enumerate(cats_p):
            cat_to_idx_p[c].append(i)
        auc_p_per_cat = {}
        for c, idxs in cat_to_idx_p.items():
            idxs = np.array(idxs)
            y_c = yp[idxs]; s_c = s_p[idxs]
            if len(set(y_c)) == 2 and len(idxs) >= 20:
                auc_p_per_cat[c] = _roc_auc(y_c, s_c)
            else:
                auc_p_per_cat[c] = float("nan")
        auc_p_global = _roc_auc(yp, s_p) if len(set(yp)) == 2 else float("nan")
        auc_o_global = _roc_auc(yo, s_o) if len(set(yo)) == 2 else float("nan")

        test3_lines.append(f"- LOCO seed : {args.loco_seed}")
        test3_lines.append(f"- Train sans les 4 cats prob (taille filtrée) ; "
                           "test sur ces 4 cats")
        test3_lines.append("")
        test3_lines.append("## AUC par catégorie problématique")
        test3_lines.append("")
        test3_lines.append("| Cat | AUC LOCO | AUC original (étape 3 s42) |")
        test3_lines.append("|---|---:|---:|")
        sd0_pc = scores_per_seed[0]
        # Get original per-cat AUC from sd0
        cats0 = sd0_pc["cats_test"]
        sco0 = np.array(sd0_pc["s_enc_te"])
        yte0 = np.array(sd0_pc["yte"])
        for c in PROBLEM_CATS:
            mask_c = np.array([cc == c for cc in cats0])
            y_c = yte0[mask_c]; s_c = sco0[mask_c]
            orig_auc = _roc_auc(y_c, s_c) if len(set(y_c)) == 2 else float("nan")
            loco_auc = auc_p_per_cat.get(c, float("nan"))
            test3_lines.append(f"| {c} | "
                               f"{'N/A' if math.isnan(loco_auc) else f'{loco_auc:.3f}'} | "
                               f"{'N/A' if math.isnan(orig_auc) else f'{orig_auc:.3f}'} |")
        avg_loco = np.mean([v for v in auc_p_per_cat.values() if not math.isnan(v)])
        test3_lines.append("")
        test3_lines.append(f"- AUC global sur 4 cats prob (LOCO) : **{auc_p_global:.4f}**")
        test3_lines.append(f"- AUC global sur autres cats (LOCO) : {auc_o_global:.4f}")
        test3_lines.append(f"- AUC moyenne par cat prob (LOCO)   : **{avg_loco:.4f}**")
        test3_lines.append("")
        test3_lines.append("## Lecture")
        test3_lines.append("")
        if not math.isnan(avg_loco):
            if avg_loco >= 0.85:
                test3_lines.append(f"- ✓ AUC LOCO ≥ 0.85 sur cats prob → "
                                   "le signal est **structural-général**, pas dépendant "
                                   "de la mémorisation par catégorie.")
            elif avg_loco >= 0.7:
                test3_lines.append(f"- ⚠️ AUC LOCO entre 0.7 et 0.85 → signal partiellement "
                                   "transférable, mais une partie venait de la "
                                   "mémorisation per-catégorie.")
            else:
                test3_lines.append(f"- ❌ AUC LOCO < 0.7 → le signal de l'étape 3 vient "
                                   "principalement de patterns lexicaux par catégorie. "
                                   "Le résultat global est moins solide qu'il en a l'air.")
        print(f"  AUC LOCO global cats prob : {auc_p_global:.4f}")
        print(f"  AUC LOCO moyenne per-cat  : {avg_loco:.4f}")

    # ──────────────────────────────────────────────────────────────────
    # Save all
    # ──────────────────────────────────────────────────────────────────
    with open(os.path.join(args.output, "test1_cross_seed.md"), "w") as f:
        f.write("\n".join(test1_lines))
    with open(os.path.join(args.output, "test2_token_lift.md"), "w") as f:
        f.write("\n".join(test2_lines))
    with open(os.path.join(args.output, "test3_loco.md"), "w") as f:
        f.write("\n".join(test3_lines))
    print(f"\nWrote outputs to {args.output}/")


if __name__ == "__main__":
    main()
