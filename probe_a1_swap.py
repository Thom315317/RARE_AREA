#!/usr/bin/env python3
"""
Probe A1 : swap encoder representations of "jump" with "walk" at inference time.

Hypothèse testée :
  Si l'encoder A1 a appris une représentation factorisée par rôle, alors
  remplacer l'embedding encoder de "jump" par celui de "walk" (dans un
  contexte équivalent du train) devrait permettre au décodeur de produire
  la séquence attendue (soit pour walk, soit pour jump, selon ce qui domine
  dans la représentation).

Procédure :
  1. Charger A1 checkpoint
  2. Pour chaque test example avec "jump" (tous sur addprim_jump) :
     - Encoder normal : enc_jump
     - Construire la commande équivalente en remplaçant "jump" par "walk"
     - Encoder la version "walk" : enc_walk
     - Swap : enc_jump[positions où input==jump] = enc_walk[mêmes positions]
     - Decoder greedy
  3. Comparer la sortie à :
     - Target jump original (attend I_JUMP)
     - Target walk équivalent (attend I_WALK)
  4. Si exact match walk > 10% → l'encoder est effectivement swappable par rôle

Usage :
  python3 probe_a1_swap.py --a1-checkpoint runs/scan_compositional/A1_s42_XXX/checkpoint.pt
"""
import os, sys, re, json, argparse, urllib.request
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# On réutilise les définitions de scan_compositional
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_compositional import (
    A1_RoleSuper, SCANDataset, collate, build_vocabs,
    parse_scan_file, split_train_val,
    BOS, EOS, PAD, DATA_DIR, TRAIN_URL, TEST_URL, _download,
)


def swap_encoder_decode(model, src, src_mask, src_swap, src_swap_mask,
                        jump_id, out_w2i, device, max_len=50):
    """Encode src, encode src_swap, swap at jump positions, decode.
    src: (B, L_test) test command tokens (contient jump)
    src_swap: (B, L_walk) walk-equivalent command tokens (jump → walk)
    """
    model.eval()
    with torch.no_grad():
        enc_main = model.encode(src, src_mask)                  # (B, L, D)
        enc_walk = model.encode(src_swap, src_swap_mask)        # (B, L', D)

        # Positions dans enc_main où input == jump
        jump_positions = (src == jump_id)                       # (B, L)
        walk_id = out_w2i.get("I_WALK")  # placeholder, not used
        # Find walk position in src_swap (one per row)
        walk_token = src_swap_mask.new_zeros(src_swap.shape)    # placeholder
        # We assume the walk token replaces jump at the SAME position index
        # (since we only substitute the word, positions align)

        # Swap: at each jump position, take enc_walk[b, same_pos]
        enc_swapped = enc_main.clone()
        for b in range(src.size(0)):
            for l in range(src.size(1)):
                if jump_positions[b, l] and l < enc_walk.size(1):
                    enc_swapped[b, l] = enc_walk[b, l]

        # Decode greedy
        bos = out_w2i[BOS]
        eos = out_w2i[EOS]
        B = src.size(0)
        tgt = torch.full((B, 1), bos, device=device, dtype=torch.long)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        mem_kpm = ~src_mask
        for _ in range(max_len):
            logits = model.decode(enc_swapped, enc_swapped, mem_kpm, tgt)
            nxt = logits[:, -1].argmax(-1)
            nxt = torch.where(done, torch.zeros_like(nxt), nxt)
            tgt = torch.cat([tgt, nxt.unsqueeze(1)], dim=1)
            done = done | (nxt == eos)
            if done.all(): break
        return tgt[:, 1:]


def make_walk_equivalent(cmd_tokens, jump_id, walk_id):
    """Return list of tokens with every jump_id replaced by walk_id."""
    return [walk_id if t == jump_id else t for t in cmd_tokens]


def action_swap_jump_to_walk(act_tokens, i_jump, i_walk):
    """Replace I_JUMP → I_WALK in action sequence (for 'walk target' comparison)."""
    return [i_walk if t == i_jump else t for t in act_tokens]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a1-checkpoint", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-examples", type=int, default=None,
                   help="Limit test examples (for quick check)")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading A1 checkpoint from {args.a1_checkpoint}")
    ckpt = torch.load(args.a1_checkpoint, map_location=device, weights_only=False)
    in_w2i = ckpt["in_w2i"]
    out_w2i = ckpt["out_w2i"]

    # Load SCAN addprim_jump
    train_path = os.path.join(DATA_DIR, "train.txt")
    test_path  = os.path.join(DATA_DIR, "test.txt")
    _download(TRAIN_URL, train_path)
    _download(TEST_URL, test_path)
    train_pairs = parse_scan_file(train_path)
    test_pairs  = parse_scan_file(test_path)

    # Rebuild A1
    model = A1_RoleSuper(len(in_w2i), len(out_w2i), d_model=args.d_model).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Relevant token IDs
    jump_id = in_w2i.get("jump")
    walk_id = in_w2i.get("walk")
    if jump_id is None or walk_id is None:
        raise RuntimeError("jump or walk not found in input vocab")
    i_jump = out_w2i.get("I_JUMP")
    i_walk = out_w2i.get("I_WALK")
    print(f"jump_id={jump_id}  walk_id={walk_id}  I_JUMP={i_jump}  I_WALK={i_walk}")

    # Filter test pairs containing "jump"
    probe_pairs = []
    for cmd, act in test_pairs:
        if "jump" in cmd.split():
            probe_pairs.append((cmd, act))
    if args.max_examples:
        probe_pairs = probe_pairs[:args.max_examples]
    print(f"Probe examples: {len(probe_pairs)}")

    # Build batches manually
    n_correct_jump = 0         # normal decode matches jump target
    n_correct_swap_jump = 0    # after swap, still matches jump target
    n_correct_swap_walk = 0    # after swap, matches walk-equivalent target
    n_match_structure = 0      # turns/connectors match (ignoring jump/walk action diffs)
    n_total = len(probe_pairs)

    for start in range(0, len(probe_pairs), args.batch_size):
        batch = probe_pairs[start:start + args.batch_size]
        B = len(batch)

        # Tokenize
        src_toks_list  = []
        swap_toks_list = []
        tgt_jump_list  = []
        tgt_walk_list  = []
        for cmd, act in batch:
            t = [in_w2i.get(w, 0) for w in cmd.split()]
            s = make_walk_equivalent(t, jump_id, walk_id)
            src_toks_list.append(t)
            swap_toks_list.append(s)
            at = [out_w2i[w] for w in act.split()]
            tgt_jump_list.append(at)
            tgt_walk_list.append(action_swap_jump_to_walk(at, i_jump, i_walk))

        Ls = max(len(x) for x in src_toks_list)
        Lw = max(len(x) for x in swap_toks_list)
        src = torch.zeros(B, Ls, dtype=torch.long, device=device)
        src_mask = torch.zeros(B, Ls, dtype=torch.bool, device=device)
        swap_src = torch.zeros(B, Lw, dtype=torch.long, device=device)
        swap_mask = torch.zeros(B, Lw, dtype=torch.bool, device=device)
        for i, (ts, ss) in enumerate(zip(src_toks_list, swap_toks_list)):
            src[i, :len(ts)] = torch.tensor(ts, device=device)
            src_mask[i, :len(ts)] = True
            swap_src[i, :len(ss)] = torch.tensor(ss, device=device)
            swap_mask[i, :len(ss)] = True

        # 1) Normal decode (A1 baseline on test)
        with torch.no_grad():
            enc = model.encode(src, src_mask)
            bos = out_w2i[BOS]; eos = out_w2i[EOS]
            tgt = torch.full((B, 1), bos, device=device, dtype=torch.long)
            done = torch.zeros(B, dtype=torch.bool, device=device)
            for _ in range(50):
                logits = model.decode(enc, enc, ~src_mask, tgt)
                nxt = logits[:, -1].argmax(-1)
                nxt = torch.where(done, torch.zeros_like(nxt), nxt)
                tgt = torch.cat([tgt, nxt.unsqueeze(1)], dim=1)
                done = done | (nxt == eos)
                if done.all(): break
            pred_normal = tgt[:, 1:].cpu().tolist()

        # 2) Swap decode
        pred_swap = swap_encoder_decode(
            model, src, src_mask, swap_src, swap_mask,
            jump_id, out_w2i, device
        ).cpu().tolist()

        # Compare
        for i in range(B):
            p_n = [x for x in pred_normal[i] if x != 0 and x != eos]
            p_s = [x for x in pred_swap[i] if x != 0 and x != eos]
            tj = tgt_jump_list[i]
            tw = tgt_walk_list[i]
            if p_n == tj:
                n_correct_jump += 1
            if p_s == tj:
                n_correct_swap_jump += 1
            if p_s == tw:
                n_correct_swap_walk += 1
            # Structural match: replace both I_JUMP and I_WALK by a common symbol
            def canon(seq):
                return [i_walk if (x == i_jump or x == i_walk) else x for x in seq]
            if canon(p_s) == canon(tj):
                n_match_structure += 1

    print(f"\n{'='*70}")
    print(f"  PROBE A1 — swap encoder(jump) ← encoder(walk) at jump positions")
    print(f"{'='*70}")
    print(f"Test examples containing 'jump' : {n_total}")
    print()
    print(f"Baseline (no swap) exact match jump target : "
          f"{n_correct_jump}/{n_total} = {n_correct_jump/n_total*100:.2f}%")
    print(f"After swap, exact match jump target (I_JUMP) : "
          f"{n_correct_swap_jump}/{n_total} = {n_correct_swap_jump/n_total*100:.2f}%")
    print(f"After swap, exact match walk-equivalent (I_WALK) : "
          f"{n_correct_swap_walk}/{n_total} = {n_correct_swap_walk/n_total*100:.2f}%")
    print(f"After swap, structural match (ignore jump/walk) : "
          f"{n_match_structure}/{n_total} = {n_match_structure/n_total*100:.2f}%")
    print()
    print("Interpretation:")
    print("  - baseline ~0%       → A1 échoue normalement sur addprim_jump (attendu)")
    print("  - swap→walk high     → l'encoder distingue jump et walk → identité plus")
    print("                         structure dans la représentation")
    print("  - swap→walk low,")
    print("    structural high    → l'encoder partage le rôle ACTION mais la IDENTITE")
    print("                         ne se propage pas (probe montre factorisation rôle)")
    print("  - structural ~0%     → l'encoder n'a RIEN appris sur les compositions")

    out_dir = os.path.dirname(args.a1_checkpoint)
    with open(os.path.join(out_dir, "probe_swap_results.json"), "w") as f:
        json.dump({
            "n_total": n_total,
            "baseline_exact_jump": n_correct_jump,
            "swap_exact_jump": n_correct_swap_jump,
            "swap_exact_walk": n_correct_swap_walk,
            "swap_structural_match": n_match_structure,
            "baseline_exact_jump_pct": n_correct_jump/n_total*100,
            "swap_exact_jump_pct": n_correct_swap_jump/n_total*100,
            "swap_exact_walk_pct": n_correct_swap_walk/n_total*100,
            "swap_structural_match_pct": n_match_structure/n_total*100,
        }, f, indent=2)
    print(f"\nResults saved: {os.path.join(out_dir, 'probe_swap_results.json')}")


if __name__ == "__main__":
    main()
