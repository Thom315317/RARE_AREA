#!/usr/bin/env python3
"""Diagnostic: load a checkpoint, greedy-decode N passive_to_active examples from gen,
print input + gold + model prediction.

Usage:
  python3 diagnose_p2a_predictions.py --checkpoint PATH [--n 5] [--category passive_to_active]
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cogs_compositional import (
    TransformerSeq2Seq, TransformerSeq2SeqWithCopy, TransformerSeq2SeqCopyTags,
    build_in_to_out_map, build_token_categories,
    parse_cogs_tsv, greedy_decode,
    PAD, BOS, EOS,
)

HERE = os.path.dirname(os.path.abspath(__file__))

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--n", type=int, default=5)
p.add_argument("--category", type=str, default="passive_to_active")
p.add_argument("--dataset", type=str, default="cogs")
args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
in_w2i, out_w2i = ckpt["in_w2i"], ckpt["out_w2i"]
in_i2w = {v: k for k, v in in_w2i.items()}
out_i2w = {v: k for k, v in out_w2i.items()}

n_layers = ckpt.get("n_layers", 2)
n_heads = ckpt.get("n_heads", 4)
ffn = ckpt.get("ffn", 256)

if ckpt.get("use_tags"):
    in_to_out = build_in_to_out_map(in_w2i, out_w2i)
    token_cats = build_token_categories(in_w2i)
    model = TransformerSeq2SeqCopyTags(
        len(in_w2i), len(out_w2i), d_model=ckpt["d_model"],
        n_heads=n_heads, n_layers=n_layers, ffn=ffn,
        max_in=ckpt["max_in"], max_out=ckpt["max_out"],
        in_to_out_map=in_to_out, token_categories=token_cats).to(device)
elif ckpt.get("use_copy"):
    in_to_out = build_in_to_out_map(in_w2i, out_w2i)
    model = TransformerSeq2SeqWithCopy(
        len(in_w2i), len(out_w2i), d_model=ckpt["d_model"],
        n_heads=n_heads, n_layers=n_layers, ffn=ffn,
        max_in=ckpt["max_in"], max_out=ckpt["max_out"],
        in_to_out_map=in_to_out).to(device)
else:
    model = TransformerSeq2Seq(
        len(in_w2i), len(out_w2i), d_model=ckpt["d_model"],
        n_heads=n_heads, n_layers=n_layers, ffn=ffn,
        max_in=ckpt["max_in"], max_out=ckpt["max_out"]).to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()

# Load gen
gen_path = (os.path.join(HERE, "data", "cogs", "gen.tsv")
            if args.dataset == "cogs"
            else os.path.join(HERE, "data", "slog", "generalization_sets", "gen_cogsLF.tsv"))
gen = parse_cogs_tsv(gen_path)
examples = [(inp, lf) for inp, lf, cat in gen if cat == args.category]
print(f"Found {len(examples)} examples of category '{args.category}'")
if not examples:
    sys.exit(1)

pad_id, bos_id, eos_id = out_w2i[PAD], out_w2i[BOS], out_w2i[EOS]


def ids_to_tokens(ids, vocab_i2w):
    toks = []
    for i in ids:
        i = int(i)
        if i == 0:
            break
        w = vocab_i2w.get(i, f"<{i}>")
        if w == EOS:
            break
        if w == BOS:
            continue
        toks.append(w)
    return " ".join(toks)


# Take first N examples
samples = examples[:args.n]
print(f"\nDecoding {len(samples)} examples greedy...\n")

with torch.no_grad():
    for idx, (inp, gold_lf) in enumerate(samples):
        src_ids = [in_w2i.get(t, 0) for t in inp.split()]
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = (src != 0)
        pred_ids = greedy_decode(model, src, src_mask, out_w2i, max_len=ckpt["max_out"])
        pred_str = ids_to_tokens(pred_ids[0].tolist(), out_i2w)
        match = "✓" if pred_str.strip() == gold_lf.strip() else "✗"
        print(f"━━━━ Example {idx + 1} [{match}] ━━━━")
        print(f"INPUT:  {inp}")
        print(f"GOLD:   {gold_lf}")
        print(f"PRED:   {pred_str}")
        # Diff summary
        gold_toks = gold_lf.split()
        pred_toks = pred_str.split()
        if pred_toks != gold_toks:
            # Find first divergence
            j = 0
            while j < min(len(gold_toks), len(pred_toks)) and gold_toks[j] == pred_toks[j]:
                j += 1
            print(f"DIVERGE at token {j}:")
            print(f"   gold[{j}:{j+5}] = {' '.join(gold_toks[j:j+5])}")
            print(f"   pred[{j}:{j+5}] = {' '.join(pred_toks[j:j+5])}")
        print()
