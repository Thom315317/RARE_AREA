#!/usr/bin/env python3
"""Phase B — meta-dataset builder for the JEPA + meta-modèle pipeline.

For every (input, gold) pair from dev + gen of the chosen benchmark, runs:
  1. forward(encoder + JEPA) → surprise_mean / surprise_max on input tokens
  2. greedy_decode → predicted token sequence
  3. entropy of decoder distribution at each greedy step (no gold) → mean / max
  4. structural input-only features:
       input_length, rare_token_count, nesting_depth (number of `that`)
  5. exact_match (gold == pred) recorded as label, NOT a feature

Outputs JSONL where each line is one example with `features_inference` (all
inference-time-valid) and `metadata` (informative only). Also produces stats.md,
feature_distributions.png, token_freq.json, and the stratified split file.

Usage:
  python3 meta_dataset_builder.py \\
      --checkpoint runs_*/B4_jepa_s42_*/checkpoint.pt \\
      --benchmark cogs \\
      --output meta_data/cogs_meta_dataset.jsonl
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cogs_compositional import (
    TransformerSeq2Seq, TransformerSeq2SeqWithCopy, TransformerSeq2SeqCopyTags,
    JEPAPredictor, build_in_to_out_map, build_token_categories,
    parse_cogs_tsv, greedy_decode,
    PAD, BOS, EOS,
)


HERE = os.path.dirname(os.path.abspath(__file__))


def _load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    in_w2i, out_w2i = ckpt["in_w2i"], ckpt["out_w2i"]
    n_layers = ckpt.get("n_layers", 2)
    n_heads = ckpt.get("n_heads", 4)
    ffn = ckpt.get("ffn", 256)
    use_jepa = bool(ckpt.get("use_jepa", False))
    if not use_jepa:
        raise RuntimeError("Checkpoint was not trained with --jepa. "
                           "Re-run training with --jepa to produce a "
                           "JEPA-augmented checkpoint.")
    if ckpt.get("use_tags"):
        in_to_out = build_in_to_out_map(in_w2i, out_w2i)
        token_cats = build_token_categories(in_w2i)
        model = TransformerSeq2SeqCopyTags(
            len(in_w2i), len(out_w2i), d_model=ckpt["d_model"],
            n_heads=n_heads, n_layers=n_layers, ffn=ffn,
            max_in=ckpt["max_in"], max_out=ckpt["max_out"],
            in_to_out_map=in_to_out, token_categories=token_cats,
            use_jepa=True).to(device)
    elif ckpt.get("use_copy"):
        in_to_out = build_in_to_out_map(in_w2i, out_w2i)
        model = TransformerSeq2SeqWithCopy(
            len(in_w2i), len(out_w2i), d_model=ckpt["d_model"],
            n_heads=n_heads, n_layers=n_layers, ffn=ffn,
            max_in=ckpt["max_in"], max_out=ckpt["max_out"],
            in_to_out_map=in_to_out, use_jepa=True).to(device)
    else:
        model = TransformerSeq2Seq(
            len(in_w2i), len(out_w2i), d_model=ckpt["d_model"],
            n_heads=n_heads, n_layers=n_layers, ffn=ffn,
            max_in=ckpt["max_in"], max_out=ckpt["max_out"],
            use_jepa=True).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def _resolve_data_paths(benchmark):
    if benchmark == "cogs":
        return {
            "train": os.path.join(HERE, "data", "cogs", "train.tsv"),
            "dev":   os.path.join(HERE, "data", "cogs", "dev.tsv"),
            "gen":   os.path.join(HERE, "data", "cogs", "gen.tsv"),
        }
    if benchmark == "slog":
        slog = os.path.join(HERE, "data", "slog")
        return {
            "train": os.path.join(slog, "train.tsv"),
            "dev":   os.path.join(slog, "dev.tsv"),
            "gen":   os.path.join(slog, "generalization_sets", "gen_cogsLF.tsv"),
        }
    raise ValueError(benchmark)


def _build_token_freq(train_pairs, cache_path):
    """Count token frequencies on train inputs. Cache to JSON."""
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)
    counter = Counter()
    for inp, _lf, _cat in train_pairs:
        counter.update(inp.split())
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(dict(counter), f)
    return dict(counter)


def _rare_threshold(freq_dict):
    """Bottom 20% frequency threshold over the train vocabulary."""
    if not freq_dict:
        return 1
    counts = sorted(freq_dict.values())
    cutoff = counts[max(0, int(0.2 * len(counts)) - 1)]
    return cutoff


def _ids_to_tokens(ids, vocab_i2w):
    out = []
    for i in ids:
        i = int(i)
        if i == 0:
            break
        w = vocab_i2w.get(i, f"<{i}>")
        if w == EOS:
            break
        if w == BOS:
            continue
        out.append(w)
    return out


def _surprise_for_input(model, src, src_mask):
    enc = model.encode(src, src_mask)
    pred_j, target_j = model.jepa_predictor(enc)
    sq = (pred_j - target_j) ** 2  # (B, T-1, d_model)
    per_pos = sq.mean(dim=-1)        # (B, T-1)
    mask = src_mask[:, 1:].float()   # (B, T-1)
    per_pos_masked = per_pos * mask
    n = mask.sum(dim=1).clamp(min=1.0)
    mean = (per_pos_masked.sum(dim=1) / n).cpu().tolist()
    # Max with masked positions = 0 (already). Take max:
    very_neg = torch.full_like(per_pos, -1e9)
    masked_for_max = torch.where(mask.bool(), per_pos, very_neg)
    mx = masked_for_max.max(dim=1).values.cpu().tolist()
    return mean, mx, enc


def _greedy_with_entropy(model, src, src_mask, out_w2i, max_len):
    """Greedy decode + per-step entropy (using the production decoder).
    Returns (pred_tokens_per_example, entropy_mean, entropy_max).
    Entropy is computed on the produced step distribution — no gold involved.
    """
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
    entropy_max = torch.full((B,), -1.0, device=device)
    entropy_cnt = torch.zeros(B, device=device)
    for _ in range(max_len):
        if uses_tags:
            logits, _ = model.decode_with_copy_tags(src, enc, mem_kpm, tgt,
                                                    roles_gt=None)
            log_probs = logits[:, -1]
        elif uses_copy:
            log_probs = model.decode_with_copy(src, enc, mem_kpm, tgt)[:, -1]
        else:
            log_probs = torch.log_softmax(
                model.decode(enc, mem_kpm, tgt)[:, -1], dim=-1)
        probs = log_probs.exp()
        ent = -(probs * log_probs).sum(dim=-1)  # (B,)
        # Only count for non-done examples
        active = (~done).float()
        entropy_sums = entropy_sums + ent * active
        entropy_cnt = entropy_cnt + active
        entropy_max = torch.maximum(entropy_max,
                                    torch.where(done, entropy_max, ent))
        nxt = log_probs.argmax(-1)
        nxt = torch.where(done, torch.zeros_like(nxt), nxt)
        tgt = torch.cat([tgt, nxt.unsqueeze(1)], dim=1)
        done = done | (nxt == eos)
        if done.all():
            break
    entropy_mean = (entropy_sums / entropy_cnt.clamp(min=1.0)).cpu().tolist()
    entropy_max_l = entropy_max.cpu().tolist()
    return tgt[:, 1:], entropy_mean, entropy_max_l


def _nesting_depth(tokens):
    """Surface nesting heuristic: count of 'that' tokens (CP nesting in COGS).
    For SLOG this approximates the visible depth too — RC nesting in SLOG is
    typically signalled by repeated 'that'."""
    return sum(1 for t in tokens if t == "that")


def _first_divergence(pred_tokens, gold_tokens):
    n = min(len(pred_tokens), len(gold_tokens))
    for i in range(n):
        if pred_tokens[i] != gold_tokens[i]:
            return i
    if len(pred_tokens) != len(gold_tokens):
        return n
    return -1


def _stratified_split(records, seed, ratios=(0.7, 0.15, 0.15)):
    """Stratify by (category, exact_match). Preserve list ordering inside groups."""
    rng = random.Random(seed)
    by_key = {}
    for i, rec in enumerate(records):
        key = (rec["category"], bool(rec["exact_match"]))
        by_key.setdefault(key, []).append(i)
    train_idx, val_idx, test_idx = [], [], []
    for key, idxs in by_key.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train + n_val])
        test_idx.extend(idxs[n_train + n_val:])
    return sorted(train_idx), sorted(val_idx), sorted(test_idx)


def _stats_md(records, out_path):
    n = len(records)
    err = sum(1 for r in records if not r["exact_match"])
    by_cat_err = Counter()
    by_cat_total = Counter()
    for r in records:
        by_cat_total[r["category"]] += 1
        if not r["exact_match"]:
            by_cat_err[r["category"]] += 1

    def _sep(field):
        ok = [r["features_inference"][field] for r in records if r["exact_match"]]
        ko = [r["features_inference"][field] for r in records if not r["exact_match"]]
        if not ok or not ko:
            return None, None
        return float(np.mean(ok)), float(np.mean(ko))

    surprise_ok, surprise_ko = _sep("surprise_mean")
    entropy_ok, entropy_ko = _sep("entropy_mean_greedy")

    lines = [
        "# Meta-dataset stats",
        "",
        f"Total: {n}",
        f"Errors: {err} ({100*err/max(n,1):.1f}%)",
        "",
        "## Distribution exact_match per category",
        "",
        "| Category | total | errors | err rate |",
        "|---|---:|---:|---:|",
    ]
    for cat in sorted(by_cat_total):
        tot = by_cat_total[cat]
        e = by_cat_err[cat]
        lines.append(f"| {cat} | {tot} | {e} | {100*e/max(tot,1):.1f}% |")

    lines.extend([
        "",
        "## Sanity checks",
        "",
        f"surprise_mean | exact (mean): {surprise_ok}",
        f"surprise_mean | error (mean): {surprise_ko}",
    ])
    if surprise_ok is not None and surprise_ko is not None:
        if surprise_ko > surprise_ok:
            lines.append("→ ✓ surprise(error) > surprise(exact) — JEPA separates")
        else:
            lines.append("→ ✗ surprise(error) ≤ surprise(exact) — STOP, JEPA does not separate")
    lines.append("")
    lines.append(f"entropy_mean_greedy | exact (mean): {entropy_ok}")
    lines.append(f"entropy_mean_greedy | error (mean): {entropy_ko}")
    if entropy_ok is not None and entropy_ko is not None:
        if entropy_ko > entropy_ok:
            lines.append("→ ✓ entropy(error) > entropy(exact)")
        else:
            lines.append("→ ✗ entropy(error) ≤ entropy(exact)")

    # Pearson correlations
    feats = ["surprise_mean", "surprise_max", "entropy_mean_greedy",
             "entropy_max_greedy", "input_length", "rare_token_count",
             "nesting_depth"]
    feature_arrays = {f: np.array([r["features_inference"][f] for r in records],
                                  dtype=float) for f in feats}
    lines.extend(["", "## Pearson correlations between features", ""])
    lines.append("| | " + " | ".join(feats) + " |")
    lines.append("|---|" + "|".join(["---"] * len(feats)) + "|")
    for fa in feats:
        row = [fa]
        for fb in feats:
            a = feature_arrays[fa]; b = feature_arrays[fb]
            if a.std() == 0 or b.std() == 0:
                row.append("—")
            else:
                r = float(np.corrcoef(a, b)[0, 1])
                row.append(f"{r:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def _feature_histograms(records, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  feature_distributions.png skipped: {e}")
        return
    feats = ["surprise_mean", "surprise_max", "entropy_mean_greedy",
             "entropy_max_greedy", "input_length", "rare_token_count",
             "nesting_depth"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.flatten()
    for ax, f in zip(axes, feats):
        ok = [r["features_inference"][f] for r in records if r["exact_match"]]
        ko = [r["features_inference"][f] for r in records if not r["exact_match"]]
        ax.hist(ok, bins=30, alpha=0.5, color="tab:green", label="exact", density=True)
        ax.hist(ko, bins=30, alpha=0.5, color="tab:red", label="error", density=True)
        ax.set_title(f)
        ax.legend(fontsize=8)
    for ax in axes[len(feats):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--benchmark", choices=["cogs", "slog"], default="cogs")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--splits", type=str, default=None,
                    help="Path to write the stratified splits JSON. Default: "
                         "<output_dir>/<benchmark>_meta_splits.json.")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = _resolve_data_paths(args.benchmark)
    train = parse_cogs_tsv(paths["train"])
    dev = parse_cogs_tsv(paths["dev"])
    gen = parse_cogs_tsv(paths["gen"])

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    token_freq_path = os.path.join(out_dir, "token_freq.json")
    token_freq = _build_token_freq(train, token_freq_path)
    rare_thresh = _rare_threshold(token_freq)

    model, ckpt = _load_model(args.checkpoint, device)
    in_w2i, out_w2i = ckpt["in_w2i"], ckpt["out_w2i"]
    out_i2w = {v: k for k, v in out_w2i.items()}
    max_out = ckpt["max_out"]

    eval_examples = []
    for split_name, pairs in (("dev", dev), ("gen", gen)):
        for inp, lf, cat in pairs:
            eval_examples.append((split_name, inp, lf, cat))

    # Process in batches
    records = []
    bs = args.batch_size
    print(f"Processing {len(eval_examples)} examples in batches of {bs} ...")
    with torch.no_grad():
        for batch_start in range(0, len(eval_examples), bs):
            batch = eval_examples[batch_start:batch_start + bs]
            # Pad to max length within batch
            src_lists = [[in_w2i.get(t, 0) for t in inp.split()] for _s, inp, _l, _c in batch]
            S = max(len(s) for s in src_lists)
            src_pad = torch.zeros(len(batch), S, dtype=torch.long)
            for i, sl in enumerate(src_lists):
                src_pad[i, :len(sl)] = torch.tensor(sl, dtype=torch.long)
            src = src_pad.to(device)
            src_mask = (src != 0)
            mean_l, max_l, _enc = _surprise_for_input(model, src, src_mask)
            pred_ids, ent_mean_l, ent_max_l = _greedy_with_entropy(
                model, src, src_mask, out_w2i, max_len=max_out)
            for i, (split_name, inp, lf, cat) in enumerate(batch):
                inp_tokens = inp.split()
                gold_tokens = lf.split()
                pred_tokens = _ids_to_tokens(pred_ids[i].tolist(), out_i2w)
                exact = pred_tokens == gold_tokens
                fdiv = _first_divergence(pred_tokens, gold_tokens)
                rare = sum(1 for t in inp_tokens
                           if token_freq.get(t, 0) <= rare_thresh)
                records.append({
                    "split": split_name,
                    "category": cat,
                    "input_tokens": inp_tokens,
                    "gold_tokens": gold_tokens,
                    "pred_tokens": pred_tokens,
                    "exact_match": bool(exact),
                    "first_divergence": fdiv,
                    "features_inference": {
                        "surprise_mean": float(mean_l[i]),
                        "surprise_max":  float(max_l[i]),
                        "entropy_mean_greedy": float(ent_mean_l[i]),
                        "entropy_max_greedy":  float(ent_max_l[i]),
                        "input_length": len(inp_tokens),
                        "rare_token_count": int(rare),
                        "nesting_depth": _nesting_depth(inp_tokens),
                    },
                    "metadata": {
                        "output_length_pred": len(pred_tokens),
                    },
                })
            if (batch_start // bs) % 50 == 0:
                print(f"  {batch_start + len(batch)}/{len(eval_examples)} processed")

    # Write JSONL
    with open(args.output, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(records)} records → {args.output}")

    # Stratified split
    splits_path = (args.splits if args.splits
                   else os.path.join(out_dir, f"{args.benchmark}_meta_splits.json"))
    train_idx, val_idx, test_idx = _stratified_split(records, seed=args.seed)
    with open(splits_path, "w") as f:
        json.dump({"train": train_idx, "val": val_idx, "test": test_idx}, f)
    print(f"Wrote splits → {splits_path}")
    print(f"  train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # Stats + histograms
    stats_path = os.path.join(out_dir, "stats.md")
    _stats_md(records, stats_path)
    print(f"Wrote {stats_path}")
    hist_path = os.path.join(out_dir, "feature_distributions.png")
    _feature_histograms(records, hist_path)
    if os.path.exists(hist_path):
        print(f"Wrote {hist_path}")


if __name__ == "__main__":
    main()
