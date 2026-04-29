#!/usr/bin/env python3
"""Phase A.2 — Étape 1 : score chaque exemple TRAIN avec le meta-encoder
robust et identifie les exemples flagged à risque.

Pipeline :
  1. Charger checkpoint B4+JEPA → forward sur le train (surprise depuis JEPA,
     entropy via greedy decode sans gold)
  2. Calculer features inférence-time (input_length, rare_token_count,
     nesting_depth) sur le même schéma que meta_dataset_builder.py
  3. Charger / re-entraîner meta-encoder seed 42 robust (déterministe) sur le
     meta-dataset existant
  4. Score chaque exemple train, écrit flagged.json + sanity stats

Usage :
  python3 meta_a2_identify_risk.py \\
      --checkpoint runs/B4_jepa_s42/B4_s42_*/checkpoint.pt \\
      --meta-dataset meta_data/cogs_meta_dataset_bplus.jsonl \\
      --meta-splits  meta_data/cogs_meta_splits.json \\
      --train-file   data/cogs/train.tsv \\
      --token-freq   meta_data/token_freq.json \\
      --output-dir   runs_meta/etape_a2/cycle_1 \\
      --threshold    0.5 \\
      --seed 42
"""
import argparse
import glob
import json
import math
import os
import random
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cogs_compositional import (
    TransformerSeq2Seq, TransformerSeq2SeqWithCopy, TransformerSeq2SeqCopyTags,
    build_in_to_out_map, build_token_categories,
    parse_cogs_tsv,
    PAD, BOS, EOS,
)


# ══════════════════════════════════════════════════════════════════════════
# Constantes meta-encoder (alignées sur meta_train_etape3_robust.py)
# ══════════════════════════════════════════════════════════════════════════
SCALAR_FEATURES = ["surprise_mean", "entropy_mean_greedy",
                   "rare_token_count", "nesting_depth"]
PAD_TOK = "<pad>"
UNK_TOK = "<unk>"
MAX_LEN = 64


def _load_records(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _features_matrix(recs, names):
    return np.array(
        [[r["features_inference"][f] for f in names] for r in recs],
        dtype=np.float32,
    )


def _labels(recs):
    return np.array([0 if r["exact_match"] else 1 for r in recs], dtype=np.float32)


def _standardize_train_only(X):
    mu = X.mean(axis=0); sd = X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd


def build_meta_vocab(meta_train_recs):
    counter = Counter()
    for r in meta_train_recs:
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


# ══════════════════════════════════════════════════════════════════════════
# MetaEncoder + perturbation training (mirror of meta_train_etape3_robust)
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


def _train_meta_encoder(meta_train_recs, meta_val_recs, tok2id, seed,
                         lr=3e-4, batch_size=64, max_epochs=50, patience=8):
    """Train robust meta-encoder (with perturbation) on the meta dataset.
    Returns trained model + standardisation params."""
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr = _features_matrix(meta_train_recs, SCALAR_FEATURES)
    Xv = _features_matrix(meta_val_recs, SCALAR_FEATURES)
    mu, sd = _standardize_train_only(Xtr)
    Xtr_s = (Xtr - mu) / sd
    Xv_s = (Xv - mu) / sd
    ytr = _labels(meta_train_recs)
    yv = _labels(meta_val_recs)

    ids_tr, mask_tr = tokenize([r["input_tokens"] for r in meta_train_recs], tok2id)
    ids_v, mask_v = tokenize([r["input_tokens"] for r in meta_val_recs], tok2id)

    model = MetaEncoder(vocab_size=len(tok2id), n_scalar_features=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    Itr = torch.from_numpy(ids_tr).to(device)
    Mtr = torch.from_numpy(mask_tr).to(device)
    Str = torch.from_numpy(Xtr_s).to(device)
    Ytr = torch.from_numpy(ytr).to(device)
    Iv = torch.from_numpy(ids_v).to(device)
    Mv = torch.from_numpy(mask_v).to(device)
    Sv = torch.from_numpy(Xv_s).to(device)
    Yv = torch.from_numpy(yv).to(device)

    best_auc, best_state, pat = -1.0, None, 0
    N = Itr.size(0)
    for ep in range(max_epochs):
        model.train()
        idx = torch.randperm(N, device=device)
        for start in range(0, N, batch_size):
            sl = idx[start:start + batch_size]
            logits = model(Itr[sl], Mtr[sl], Str[sl])
            loss = F.binary_cross_entropy_with_logits(logits, Ytr[sl])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            sv = torch.sigmoid(model(Iv, Mv, Sv)).cpu().numpy()
        # cheap AUC via roc_auc_score
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(yv, sv))
        except Exception:
            auc = -1.0
        print(f"    [meta-enc] ep{ep:02d}  val_auc={auc:.4f}  pat={pat}", flush=True)
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
    return model, mu, sd


# ══════════════════════════════════════════════════════════════════════════
# Forward sur train COGS pour extraire features
# ══════════════════════════════════════════════════════════════════════════
def _load_base_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    in_w2i, out_w2i = ckpt["in_w2i"], ckpt["out_w2i"]
    n_layers = ckpt.get("n_layers", 2)
    n_heads = ckpt.get("n_heads", 4)
    ffn = ckpt.get("ffn", 256)
    use_jepa = bool(ckpt.get("use_jepa", False))
    use_jepa_ema = bool(ckpt.get("use_jepa_ema", False))
    jepa_ema_decay = float(ckpt.get("jepa_ema_decay", 0.99))
    if not use_jepa:
        raise RuntimeError("Checkpoint not trained with --jepa")
    if ckpt.get("use_tags"):
        in_to_out = build_in_to_out_map(in_w2i, out_w2i)
        token_cats = build_token_categories(in_w2i)
        model = TransformerSeq2SeqCopyTags(
            len(in_w2i), len(out_w2i), d_model=ckpt["d_model"],
            n_heads=n_heads, n_layers=n_layers, ffn=ffn,
            max_in=ckpt["max_in"], max_out=ckpt["max_out"],
            in_to_out_map=in_to_out, token_categories=token_cats,
            use_jepa=True, use_jepa_ema=use_jepa_ema,
            jepa_ema_decay=jepa_ema_decay).to(device)
    elif ckpt.get("use_copy"):
        in_to_out = build_in_to_out_map(in_w2i, out_w2i)
        model = TransformerSeq2SeqWithCopy(
            len(in_w2i), len(out_w2i), d_model=ckpt["d_model"],
            n_heads=n_heads, n_layers=n_layers, ffn=ffn,
            max_in=ckpt["max_in"], max_out=ckpt["max_out"],
            in_to_out_map=in_to_out, use_jepa=True,
            use_jepa_ema=use_jepa_ema, jepa_ema_decay=jepa_ema_decay).to(device)
    else:
        model = TransformerSeq2Seq(
            len(in_w2i), len(out_w2i), d_model=ckpt["d_model"],
            n_heads=n_heads, n_layers=n_layers, ffn=ffn,
            max_in=ckpt["max_in"], max_out=ckpt["max_out"],
            use_jepa=True, use_jepa_ema=use_jepa_ema,
            jepa_ema_decay=jepa_ema_decay).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def _surprise_for_input(model, src, src_mask):
    enc = model.encode(src, src_mask)
    pred_j, target_j = model.jepa_predictor(enc)
    sq = (pred_j - target_j) ** 2
    per_pos = sq.mean(dim=-1)
    mask = src_mask[:, 1:].float()
    n = mask.sum(dim=1).clamp(min=1.0)
    mean = ((per_pos * mask).sum(dim=1) / n).cpu().tolist()
    return mean, enc


def _greedy_with_entropy(model, src, src_mask, out_w2i, max_len):
    B = src.size(0)
    device = src.device
    bos, eos = out_w2i[BOS], out_w2i[EOS]
    enc = model.encode(src, src_mask)
    mem_kpm = ~src_mask
    uses_tags = isinstance(model, TransformerSeq2SeqCopyTags)
    uses_copy = isinstance(model, TransformerSeq2SeqWithCopy)
    tgt = torch.full((B, 1), bos, device=device, dtype=torch.long)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    entropy_sums = torch.zeros(B, device=device)
    entropy_cnt = torch.zeros(B, device=device)
    for _ in range(max_len):
        if uses_tags:
            logits, _ = model.decode_with_copy_tags(src, enc, mem_kpm, tgt, roles_gt=None)
            log_probs = logits[:, -1]
        elif uses_copy:
            log_probs = model.decode_with_copy(src, enc, mem_kpm, tgt)[:, -1]
        else:
            log_probs = torch.log_softmax(model.decode(enc, mem_kpm, tgt)[:, -1], dim=-1)
        probs = log_probs.exp()
        ent = -(probs * log_probs).sum(dim=-1)
        active = (~done).float()
        entropy_sums = entropy_sums + ent * active
        entropy_cnt = entropy_cnt + active
        nxt = log_probs.argmax(-1)
        nxt = torch.where(done, torch.zeros_like(nxt), nxt)
        tgt = torch.cat([tgt, nxt.unsqueeze(1)], dim=1)
        done = done | (nxt == eos)
        if done.all():
            break
    entropy_mean = (entropy_sums / entropy_cnt.clamp(min=1.0)).cpu().tolist()
    return entropy_mean


def compute_train_features(base_model, ckpt, train_pairs, token_freq,
                            rare_thresh, batch_size=16, device=None):
    """Renvoie une liste de dicts par exemple train avec input_tokens + features."""
    in_w2i = ckpt["in_w2i"]; out_w2i = ckpt["out_w2i"]
    max_out = ckpt["max_out"]
    out = []
    print(f"  Forwarding {len(train_pairs)} train examples...", flush=True)
    with torch.no_grad():
        for i in range(0, len(train_pairs), batch_size):
            batch = train_pairs[i:i + batch_size]
            src_lists = [[in_w2i.get(t, 0) for t in inp.split()] for inp, _, _ in batch]
            S = max(len(s) for s in src_lists)
            src_pad = torch.zeros(len(batch), S, dtype=torch.long)
            for j, sl in enumerate(src_lists):
                src_pad[j, :len(sl)] = torch.tensor(sl, dtype=torch.long)
            src = src_pad.to(device)
            src_mask = (src != 0)
            mean_l, _enc = _surprise_for_input(base_model, src, src_mask)
            ent_mean_l = _greedy_with_entropy(base_model, src, src_mask, out_w2i, max_out)
            for j, (inp, lf, cat) in enumerate(batch):
                inp_tokens = inp.split()
                rare = sum(1 for t in inp_tokens if token_freq.get(t, 0) <= rare_thresh)
                nest = sum(1 for t in inp_tokens if t == "that")
                out.append({
                    "train_idx": i + j,
                    "input_tokens": inp_tokens,
                    "category": cat,
                    "input": inp,
                    "lf": lf,
                    "features_inference": {
                        "surprise_mean": float(mean_l[j]),
                        "entropy_mean_greedy": float(ent_mean_l[j]),
                        "input_length": len(inp_tokens),
                        "rare_token_count": int(rare),
                        "nesting_depth": int(nest),
                    },
                })
            if (i // batch_size) % 50 == 0:
                print(f"    {i + len(batch)}/{len(train_pairs)} processed", flush=True)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="Glob ou path vers checkpoint B4+JEPA")
    ap.add_argument("--meta-dataset", required=True)
    ap.add_argument("--meta-splits", required=True)
    ap.add_argument("--train-file", required=True,
                    help="data/cogs/train.tsv (ou similaire)")
    ap.add_argument("--token-freq", required=True,
                    help="JSON dump from meta_dataset_builder.py")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Resolve checkpoint glob
    matches = sorted(glob.glob(args.checkpoint))
    if not matches:
        raise SystemExit(f"No checkpoint matching {args.checkpoint}")
    ckpt_path = matches[0]
    print(f"Using checkpoint: {ckpt_path}")

    # ── Load base model
    base_model, ckpt = _load_base_model(ckpt_path, device)
    print(f"  Base model loaded ({sum(p.numel() for p in base_model.parameters())/1e6:.2f}M params)")

    # ── Load token freq
    with open(args.token_freq, "r") as f:
        token_freq = json.load(f)
    counts_sorted = sorted(token_freq.values())
    rare_thresh = counts_sorted[max(0, int(0.2 * len(counts_sorted)) - 1)]
    print(f"  Rare token threshold: count <= {rare_thresh}")

    # ── Compute features on train
    train_pairs = parse_cogs_tsv(args.train_file)
    train_features = compute_train_features(base_model, ckpt, train_pairs,
                                              token_freq, rare_thresh,
                                              batch_size=16, device=device)
    del base_model

    # ── Build / train meta-encoder
    meta_records = _load_records(args.meta_dataset)
    with open(args.meta_splits, "r") as f:
        meta_splits = json.load(f)
    meta_train_recs = [meta_records[i] for i in meta_splits["train"]]
    meta_val_recs = [meta_records[i] for i in meta_splits["val"]]
    tok2id = build_meta_vocab(meta_train_recs)
    print(f"\nTraining meta-encoder seed {args.seed} (vocab={len(tok2id)})...")
    meta_model, mu, sd = _train_meta_encoder(meta_train_recs, meta_val_recs,
                                              tok2id, seed=args.seed)

    # ── Score train
    print(f"\nScoring {len(train_features)} train examples with meta-encoder...")
    Xtr = np.array([[r["features_inference"][f] for f in SCALAR_FEATURES]
                    for r in train_features], dtype=np.float32)
    Xtr_s = (Xtr - mu) / sd
    ids_tr, mask_tr = tokenize([r["input_tokens"] for r in train_features], tok2id)
    meta_model.eval()
    device_meta = next(meta_model.parameters()).device
    with torch.no_grad():
        scores = []
        bs = 512
        for i in range(0, len(train_features), bs):
            end = min(i + bs, len(train_features))
            s = torch.sigmoid(meta_model(
                torch.from_numpy(ids_tr[i:end]).to(device_meta),
                torch.from_numpy(mask_tr[i:end]).to(device_meta),
                torch.from_numpy(Xtr_s[i:end]).to(device_meta))).cpu().numpy()
            scores.append(s)
    scores = np.concatenate(scores)

    # ── Sanity stats
    flagged_mask = scores > args.threshold
    n_flagged = int(flagged_mask.sum())
    print(f"\nFlagged {n_flagged}/{len(scores)} ({100*n_flagged/len(scores):.1f}%) "
          f"at threshold {args.threshold}")
    pct_distribution = {
        "score>0.3": int((scores > 0.3).sum()),
        "score>0.5": int((scores > 0.5).sum()),
        "score>0.7": int((scores > 0.7).sum()),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
    }
    print(f"Score distribution: {pct_distribution}")

    # ── Write flagged.json
    flagged = []
    for i, (rec, score) in enumerate(zip(train_features, scores)):
        if score > args.threshold:
            flagged.append({
                "train_idx": rec["train_idx"],
                "score": float(score),
                "input": rec["input"],
                "lf": rec["lf"],
                "category": rec["category"],
                "features": rec["features_inference"],
            })
    flagged_path = os.path.join(args.output_dir, "flagged.json")
    with open(flagged_path, "w") as f:
        json.dump({
            "n_train": len(scores),
            "n_flagged": n_flagged,
            "threshold": args.threshold,
            "score_distribution": pct_distribution,
            "examples": flagged,
        }, f, indent=2)
    print(f"\nWrote {flagged_path}")

    # ── Sanity log: 10 flagged + 10 non-flagged
    sanity_lines = ["# A.2 Étape 1 — sanity check (flagged vs non-flagged)", ""]
    sanity_lines.append(f"- Train total : {len(scores)}")
    sanity_lines.append(f"- Flagged (score > {args.threshold}) : {n_flagged} "
                        f"({100*n_flagged/len(scores):.1f}%)")
    sanity_lines.append(f"- Score mean : {pct_distribution['mean']:.3f}, "
                        f"std : {pct_distribution['std']:.3f}")
    sanity_lines.append("")
    sanity_lines.append("## 10 exemples flagged (score le plus élevé)")
    sanity_lines.append("")
    sorted_idx = np.argsort(-scores)
    for k in sorted_idx[:10]:
        rec = train_features[k]
        sanity_lines.append(f"- score {scores[k]:.3f}  cat={rec['category']}  "
                            f"`{rec['input'][:80]}`")
    sanity_lines.append("")
    sanity_lines.append("## 10 exemples non-flagged (score le plus bas)")
    sanity_lines.append("")
    for k in sorted_idx[-10:]:
        rec = train_features[k]
        sanity_lines.append(f"- score {scores[k]:.3f}  cat={rec['category']}  "
                            f"`{rec['input'][:80]}`")
    with open(os.path.join(args.output_dir, "sanity.md"), "w") as f:
        f.write("\n".join(sanity_lines))


if __name__ == "__main__":
    main()
