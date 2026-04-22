#!/usr/bin/env python3
"""Run constrained-beam eval on an existing checkpoint.

Loads the model, extracts constraints from the train set, runs beam search with
re-ranking, and prints per-category results + violation counts + gold-rank diagnostic.
"""
import os, sys, json, argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cogs_compositional import (
    TransformerSeq2Seq, TransformerSeq2SeqWithCopy, TransformerSeq2SeqCopyTags,
    build_in_to_out_map, build_token_categories,
    COGSDataset, collate, parse_cogs_tsv, evaluate,
    extract_train_constraints,
    MAX_IN, MAX_OUT, MAX_DECODE,
)

HERE = os.path.dirname(os.path.abspath(__file__))

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--dataset", default="cogs", choices=["cogs", "slog"])
p.add_argument("--beam-size", type=int, default=10)
p.add_argument("--max-examples", type=int, default=0,
               help="Cap on number of gen examples (0 = all)")
p.add_argument("--fast", action="store_true",
               help="Quick diagnostic: 500 examples, frequent progress")
args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
in_w2i, out_w2i = ckpt["in_w2i"], ckpt["out_w2i"]

# Rebuild model
if ckpt.get("use_tags"):
    in_to_out = build_in_to_out_map(in_w2i, out_w2i)
    token_cats = build_token_categories(in_w2i)
    model = TransformerSeq2SeqCopyTags(
        len(in_w2i), len(out_w2i), d_model=ckpt["d_model"],
        max_in=ckpt["max_in"], max_out=ckpt["max_out"],
        in_to_out_map=in_to_out, token_categories=token_cats).to(device)
    print("Copy+Tags: ON")
elif ckpt.get("use_copy"):
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

# Load train (needed for constraints extraction) + gen
if args.dataset == "cogs":
    train = parse_cogs_tsv(os.path.join(HERE, "data", "cogs", "train.tsv"))
    gen = parse_cogs_tsv(os.path.join(HERE, "data", "cogs", "gen.tsv"))
else:
    train = parse_cogs_tsv(os.path.join(HERE, "data", "slog", "generalization_sets", "train_cogsLF.tsv"))
    gen = parse_cogs_tsv(os.path.join(HERE, "data", "slog", "generalization_sets", "gen_cogsLF.tsv"))

print(f"Extracting constraints from {len(train)} train examples...")
constraints = extract_train_constraints(train)
print(f"  always_agent: {len(constraints['always_agent'])}  "
      f"always_theme: {len(constraints['always_theme'])}  "
      f"flexible: {len(constraints['flexible'])}")
print(f"  proper_names: {len(constraints['proper_names'])}  "
      f"known_verbs: {len(constraints['known_verbs'])}")
print(f"  nmod_heads: {len(constraints['nmod_heads'])}  "
      f"ccomp_heads: {len(constraints['ccomp_heads'])}")

ge_ds = COGSDataset(gen, in_w2i, out_w2i)
ge_ld = DataLoader(ge_ds, 16, shuffle=False, collate_fn=collate, num_workers=0)

in_i2w = {v: k for k, v in in_w2i.items()}
out_i2w = {v: k for k, v in out_w2i.items()}
diag_cats = {"active_to_passive", "passive_to_active",
             "do_dative_to_pp_dative", "pp_dative_to_do_dative",
             "unacc_to_transitive", "obj_omitted_transitive_to_transitive",
             "cp_recursion"}

beam_max = args.max_examples
prog_every = 500
if args.fast:
    if beam_max == 0:
        beam_max = 500
    prog_every = 50
    print(f"FAST MODE: evaluating {beam_max} examples, progress every {prog_every}")

print(f"\nRunning constrained beam search (k={args.beam_size})...")
result = evaluate(model, ge_ld, out_w2i, device,
                  constraints=constraints, beam_size=args.beam_size,
                  in_i2w=in_i2w, out_i2w=out_i2w, diag_cats=diag_cats,
                  max_examples=beam_max, progress_every=prog_every)

print(f"\nConstrained-beam gen_ex = {result['exact']:.2f}%")
print("\nPer-category:")
for c, v in sorted(result["by_cat"].items(), key=lambda x: -x[1]):
    print(f"  {c:<50s} {v:6.2f}%")

if result.get("violations"):
    print("\nViolation counts (over all beams scored):")
    for k, v in sorted(result["violations"].items(), key=lambda x: -x[1]):
        print(f"  {k:20s} {v}")

if result.get("beam_rank_log"):
    from collections import defaultdict
    by_c = defaultdict(list)
    for cat, rank, nb in result["beam_rank_log"]:
        by_c[cat].append(rank)
    print("\nGold rank diagnostic (structural categories):")
    for cat in sorted(by_c):
        ranks = by_c[cat]
        found = [r for r in ranks if r >= 0]
        if found:
            med = sorted(found)[len(found)//2]
            print(f"  {cat:<50s} gold in beams: {len(found)}/{len(ranks)} "
                  f"(median rank {med})")
        else:
            print(f"  {cat:<50s} gold NEVER in top-{args.beam_size} beams "
                  f"({len(ranks)} examples)")

# Save
out_path = os.path.join(os.path.dirname(args.checkpoint), "constrained_beam.json")
with open(out_path, "w") as f:
    json.dump({
        "final_gen_exact_constrained": result["exact"],
        "by_cat": result.get("by_cat", {}),
        "violations": result.get("violations", {}),
        "beam_size": args.beam_size,
        "n_examples": result["n"],
    }, f, indent=2)
print(f"\nSaved: {out_path}")
