#!/usr/bin/env python3
"""Re-run greedy eval on a copy-mechanism checkpoint."""
import os, sys, json, argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cogs_compositional import (
    TransformerSeq2Seq, TransformerSeq2SeqWithCopy, build_in_to_out_map,
    COGSDataset, collate, parse_cogs_tsv, evaluate, MAX_IN, MAX_OUT, MAX_DECODE,
)

HERE = os.path.dirname(os.path.abspath(__file__))

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--dataset", default="cogs", choices=["cogs", "slog"])
args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
in_w2i, out_w2i = ckpt["in_w2i"], ckpt["out_w2i"]

# Rebuild model
if ckpt.get("use_copy"):
    in_to_out = build_in_to_out_map(in_w2i, out_w2i)
    model = TransformerSeq2SeqWithCopy(
        len(in_w2i), len(out_w2i), d_model=ckpt["d_model"],
        max_in=ckpt["max_in"], max_out=ckpt["max_out"],
        in_to_out_map=in_to_out).to(device)
    print("Copy mechanism: ON")
else:
    model = TransformerSeq2Seq(
        len(in_w2i), len(out_w2i), d_model=ckpt["d_model"],
        max_in=ckpt["max_in"], max_out=ckpt["max_out"]).to(device)
    print("Copy mechanism: OFF")
model.load_state_dict(ckpt["state_dict"])
model.eval()

# Load gen
if args.dataset == "cogs":
    gen = parse_cogs_tsv(os.path.join(HERE, "data", "cogs", "gen.tsv"))
else:
    gen = parse_cogs_tsv(os.path.join(HERE, "data", "slog", "generalization_sets", "gen_cogsLF.tsv"))
ge_ds = COGSDataset(gen, in_w2i, out_w2i)
ge_ld = DataLoader(ge_ds, 16, shuffle=False, collate_fn=collate, num_workers=0)

print("Greedy eval...")
greedy = evaluate(model, ge_ld, out_w2i, device)
print(f"\nGreedy gen_ex = {greedy['exact']:.2f}%")
if "by_cat" in greedy:
    print("\nPer-category:")
    for c, v in sorted(greedy["by_cat"].items(), key=lambda x: -x[1]):
        if v > 0: print(f"  {c:<40s} {v:6.2f}%")

# Save
out = {"final_gen_exact_greedy": greedy["exact"],
       "greedy_gen_by_cat": greedy.get("by_cat", {})}
out_path = os.path.join(os.path.dirname(args.checkpoint), "greedy_recomputed.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {out_path}")
