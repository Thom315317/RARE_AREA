#!/usr/bin/env python3
"""COGS compositional generalization — B0 baseline + B1 permutation.

Standalone file. Same Transformer seq2seq as A0/A4 in scan_compositional.py,
adapted to COGS vocab + logical form output.

Variants:
  B0 : baseline Transformer seq2seq
  B1 : B0 + per-example permutation of proper-name tokens in TRAIN only,
       consistently applied in input AND output (same mapping within example).

B1 automatically detects proper names : capitalized input tokens whose
lowercased form appears in the output vocabulary (emma/Emma, liam/Liam, ...).
The list of detected names is logged at launch.

Usage:
  python3 cogs_compositional.py --variant B0 --seed 42 --epochs 100
  python3 cogs_compositional.py --variant B1 --seed 42 --epochs 100
"""
import os, sys, json, time, math, random, re, argparse, urllib.request
from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ══════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "cogs")
RUNS_DIR = os.path.join(HERE, "runs", "cogs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

COGS_BASE = "https://raw.githubusercontent.com/najoungkim/COGS/master/data"
COGS_SPLITS = {
    "train": f"{COGS_BASE}/train.tsv",
    "dev":   f"{COGS_BASE}/dev.tsv",
    "gen":   f"{COGS_BASE}/gen.tsv",
}

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
MAX_IN, MAX_OUT = 30, 220      # COGS logical forms can be long
MAX_DECODE = 120                # p95 of gen; longer outputs are rare + won't match early


# ══════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════
def _download(url: str, dest: str):
    if not os.path.exists(dest):
        print(f"Downloading {url}")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            raise RuntimeError(f"Failed to download {url}: {e}")


def parse_cogs_tsv(path: str) -> List[Tuple[str, str, str]]:
    """Each line: input<TAB>logical_form<TAB>category"""
    pairs = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2: continue
            inp = parts[0].strip()
            lf  = parts[1].strip()
            cat = parts[2].strip() if len(parts) >= 3 else ""
            if not inp or not lf: continue
            pairs.append((inp, lf, cat))
    return pairs


def build_vocabs(pairs):
    in_w2i  = {PAD: 0}
    out_w2i = {PAD: 0, BOS: 1, EOS: 2}
    for inp, lf, _ in pairs:
        for t in inp.split():
            if t not in in_w2i: in_w2i[t] = len(in_w2i)
        for t in lf.split():
            if t not in out_w2i: out_w2i[t] = len(out_w2i)
    return in_w2i, out_w2i


COGS_DETERMINERS = {"A", "The", "An", "Every", "No", "Some"}

def detect_proper_names(train_pairs, in_w2i, out_w2i):
    """Return list of (in_id, out_id, in_tok, out_tok) for proper-name tokens.

    In COGS, proper names keep the SAME casing in input and output
    (e.g. Emma → Emma). We detect: capitalized alphabetic, len > 1,
    not a determiner, appears in both input and output vocab with same form."""
    in_cap = set()
    for inp, _, _ in train_pairs:
        for t in inp.split():
            if t.isalpha() and t[0].isupper() and len(t) > 1 and t not in COGS_DETERMINERS:
                in_cap.add(t)
    pairs_io = []
    for t in sorted(in_cap):
        if t in in_w2i and t in out_w2i:
            pairs_io.append((in_w2i[t], out_w2i[t], t, t))
    return pairs_io


def generate_pp_recursion_extension(train_pairs, max_aug=500):
    """Extend PP chains by 1 level. Depth 2→3, depth 4→5.

    Finds the last PP in the input (PREP DET NOUN before '.'), appends
    a new PREP DET NOUN, and extends the nmod chain in the LF."""
    PREPS = ["beside", "in", "on"]
    DETS = {"a", "A", "the", "The"}
    # Collect common nouns from train for the new PP
    from collections import Counter
    noun_pool = []
    for inp, _, _ in train_pairs:
        tokens = inp.split()
        for i, t in enumerate(tokens):
            if t in DETS and i + 1 < len(tokens) and tokens[i+1][0].islower() and tokens[i+1].isalpha():
                noun_pool.append(tokens[i+1])
    noun_freq = Counter(noun_pool)
    common_nouns = [n for n, c in noun_freq.most_common(50)]

    aug = []
    for inp, lf, _ in train_pairs:
        if ". nmod ." not in lf:
            continue
        if ". recipient" in lf:
            continue  # skip datives — "to NAME" at end corrupts PP extension
        tokens = inp.split()
        if tokens[-1] != ".":
            continue
        if "to" in tokens or "by" in tokens:
            continue  # skip datives ("to") and passives ("by") — end tokens corrupt extension
        # Find last PP: scan backwards from period
        # Pattern: ... PREP DET NOUN .
        n = len(tokens)
        if n < 6:
            continue
        last_prep_i = None
        for i in range(n - 4, 0, -1):
            if tokens[i] in PREPS and tokens[i+1] in DETS:
                last_prep_i = i
                break
        if last_prep_i is None:
            continue
        last_noun = tokens[last_prep_i + 2]
        last_noun_pos = last_prep_i + 2
        last_noun_var = f"x _ {last_noun_pos}"

        # Choose random prep and noun for extension
        new_prep = random.choice(PREPS)
        new_det = random.choice(["a", "the"])
        new_noun = random.choice(common_nouns)
        while new_noun == last_noun:
            new_noun = random.choice(common_nouns)

        # Build new input: insert "PREP DET NOUN" before "."
        new_tokens = tokens[:-1] + [new_prep, new_det, new_noun, "."]
        new_inp = " ".join(new_tokens)

        # New noun position
        new_noun_pos = len(new_tokens) - 2  # position of new_noun (before ".")
        new_noun_var = f"x _ {new_noun_pos}"

        # Extend LF: add nmod link and noun predicate
        nmod_pred = f"{last_noun} . nmod . {new_prep} ( {last_noun_var} , {new_noun_var} )"
        if new_det == "a":
            noun_pred = f"{new_noun} ( {new_noun_var} )"
            lf_ext = f" AND {nmod_pred} AND {noun_pred}"
        else:
            # Definite: add presupposition
            noun_pred = f"* {new_noun} ( {new_noun_var} )"
            # Insert presupposition before body
            if " ; " in lf:
                parts = lf.split(" ; ")
                presups = parts[:-1]
                body = parts[-1]
                presups.append(noun_pred)
                new_lf = " ; ".join(presups) + " ; " + body + f" AND {nmod_pred}"
            else:
                new_lf = f"{noun_pred} ; {lf} AND {nmod_pred}"
            aug.append((new_inp, new_lf, "aug_pp_recursion"))
            continue

        new_lf = lf + lf_ext
        aug.append((new_inp, new_lf, "aug_pp_recursion"))

    # Deduplicate and cap
    seen = set()
    deduped = []
    for item in aug:
        if item[0] not in seen:
            seen.add(item[0])
            deduped.append(item)
    if len(deduped) > max_aug:
        random.shuffle(deduped)
        deduped = deduped[:max_aug]
    return deduped


def generate_cp_recursion_extension(train_pairs, max_aug=500):
    """Extend CP embedding by 1 level. Wraps sentence in 'NAME VERBed that ...'.

    All x_N in original LF shift by +3 (prepending 3 tokens)."""
    # Collect proper names and ccomp verbs from train
    DETS = {"a", "A", "the", "The"}
    names = set()
    ccomp_verbs = {}  # lemma → past_tense
    for inp, lf, _ in train_pairs:
        tokens = inp.split()
        for t in tokens:
            if t[0].isupper() and t not in DETS and len(t) > 1 and t.isalpha():
                names.add(t)
        if ". ccomp" in lf:
            m = re.search(r'(\w+) \. ccomp', lf)
            if m and tokens[0][0].isupper() and tokens[0] not in DETS:
                ccomp_verbs[m.group(1)] = tokens[1]
    names = sorted(names)
    if not ccomp_verbs:
        return []

    aug = []
    for inp, lf, _ in train_pairs:
        if ". ccomp" not in lf:
            continue
        tokens = inp.split()
        if tokens[-1] != ".":
            continue

        # Choose wrapper
        wrapper_name = random.choice(names)
        wrapper_lemma, wrapper_past = random.choice(list(ccomp_verbs.items()))

        # Build new input: "NAME VERBed that original_lowered ."
        # Lowercase the first token of original if it was capitalized
        orig_first = tokens[0]
        if orig_first in DETS:
            orig_first_low = orig_first.lower()
        else:
            orig_first_low = orig_first  # proper name stays capitalized
        inner_tokens = [orig_first_low] + tokens[1:-1]
        new_tokens = [wrapper_name, wrapper_past, "that"] + inner_tokens + ["."]
        new_inp = " ".join(new_tokens)

        # Shift all x_N in LF by +3
        def shift_var(m):
            n = int(m.group(1))
            return f"x _ {n + 3}"
        shifted_lf = re.sub(r'x _ (\d+)', shift_var, lf)

        # Add wrapper predicates
        wrapper_event_var = "x _ 1"
        # Find the top-level event in shifted LF (the first agent/theme/ccomp)
        m = re.search(r'(\w+) \. (?:agent|theme|ccomp) \( (x _ \d+)', shifted_lf)
        if not m:
            continue
        inner_top_var = m.group(2)

        wrapper_preds = (f"{wrapper_lemma} . agent ( {wrapper_event_var} , {wrapper_name} ) "
                        f"AND {wrapper_lemma} . ccomp ( {wrapper_event_var} , {inner_top_var} )")

        # Combine: wrapper presups (none) + shifted presups + wrapper body + shifted body
        if " ; " in shifted_lf:
            parts = shifted_lf.split(" ; ")
            presups = " ; ".join(parts[:-1])
            body = parts[-1]
            new_lf = f"{presups} ; {wrapper_preds} AND {body}"
        else:
            new_lf = f"{wrapper_preds} AND {shifted_lf}"

        aug.append((new_inp, new_lf, "aug_cp_recursion"))

    seen = set()
    deduped = []
    for item in aug:
        if item[0] not in seen:
            seen.add(item[0])
            deduped.append(item)
    if len(deduped) > max_aug:
        random.shuffle(deduped)
        deduped = deduped[:max_aug]
    return deduped


def _build_participle_map(train_pairs):
    """Build past_tense → participle mapping from passive+active examples."""
    DETS = {"a", "A", "the", "The"}
    lemma_to_pp = {}   # lemma → participle (from passives)
    lemma_to_past = {} # lemma → past tense (from actives)
    for inp, lf, _ in train_pairs:
        tokens = inp.split()
        if "was" in tokens:
            was_i = tokens.index("was")
            if was_i + 1 < len(tokens):
                m = re.search(r'(\w+) \. theme', lf)
                if m:
                    lemma_to_pp[m.group(1)] = tokens[was_i + 1]
        elif ". agent" in lf and "was" not in tokens:
            m = re.search(r'(\w+) \. agent', lf)
            if m:
                lemma = m.group(1)
                if tokens[0] in DETS:
                    lemma_to_past[lemma] = tokens[2]
                elif tokens[0][0].isupper() and tokens[0] not in DETS:
                    lemma_to_past[lemma] = tokens[1]
    past_to_pp = {}
    for lemma, past in lemma_to_past.items():
        if lemma in lemma_to_pp:
            past_to_pp[past] = lemma_to_pp[lemma]
        elif past.endswith("ed"):
            past_to_pp[past] = past
    return past_to_pp


def generate_active_to_passive(train_pairs):
    """Transform simple active transitives → passive.

    Pattern A: 'DET NOUN VERB DET NOUN .' (6 tok) → 'DET NOUN was PP by DET NOUN .' (8 tok)
    Pattern B: 'NAME VERB DET NOUN .' (5 tok) → 'DET NOUN was PP by NAME .' (7 tok)
    """
    DETS = {"a", "A", "the", "The"}
    past_to_pp = _build_participle_map(train_pairs)
    aug = []
    for inp, lf, _ in train_pairs:
        if ". nmod ." in lf or ". recipient" in lf or ". xcomp" in lf:
            continue
        if ". agent" not in lf or ". theme" not in lf:
            continue
        tokens = inp.split()
        if tokens[-1] != ".":
            continue

        m = re.search(r'(\w+) \. agent', lf)
        if not m:
            continue
        verb_lemma = m.group(1)

        # Pattern A: DET(0) NOUN(1) VERB(2) DET(3) NOUN(4) .(5) — 6 tokens
        if (len(tokens) == 6 and tokens[0] in DETS and tokens[3] in DETS):
            subj_det, subj_noun, verb_past = tokens[0], tokens[1], tokens[2]
            obj_det, obj_noun = tokens[3], tokens[4]
            pp = past_to_pp.get(verb_past)
            if not pp:
                continue
            # Fix case: sentence-initial capitalize, mid-sentence lowercase
            obj_det_cap = obj_det.capitalize()
            subj_det_low = subj_det.lower()
            # → OBJ_DET(0) OBJ(1) was(2) PP(3) by(4) SUBJ_DET(5) SUBJ(6) .(7)
            new_inp = f"{obj_det_cap} {obj_noun} was {pp} by {subj_det_low} {subj_noun} ."
            ov, ev, sv = "x _ 1", "x _ 3", "x _ 6"
            presups = []
            if obj_det.lower() == "the": presups.append(f"* {obj_noun} ( {ov} )")
            if subj_det.lower() == "the": presups.append(f"* {subj_noun} ( {sv} )")
            body = []
            if obj_det.lower() == "a": body.append(f"{obj_noun} ( {ov} )")
            body.append(f"{verb_lemma} . theme ( {ev} , {ov} )")
            body.append(f"{verb_lemma} . agent ( {ev} , {sv} )")
            if subj_det.lower() == "a": body.append(f"{subj_noun} ( {sv} )")

        # Pattern B: NAME(0) VERB(1) DET(2) NOUN(3) .(4) — 5 tokens
        elif (len(tokens) == 5 and tokens[0][0].isupper() and
              tokens[0] not in DETS and tokens[2] in DETS):
            name, verb_past = tokens[0], tokens[1]
            obj_det, obj_noun = tokens[2], tokens[3]
            pp = past_to_pp.get(verb_past)
            if not pp:
                continue
            obj_det_cap = obj_det.capitalize()
            # → OBJ_DET(0) OBJ(1) was(2) PP(3) by(4) NAME(5) .(6)
            new_inp = f"{obj_det_cap} {obj_noun} was {pp} by {name} ."
            ov, ev = "x _ 1", "x _ 3"
            presups = []
            if obj_det.lower() == "the": presups.append(f"* {obj_noun} ( {ov} )")
            body = []
            if obj_det.lower() == "a": body.append(f"{obj_noun} ( {ov} )")
            body.append(f"{verb_lemma} . theme ( {ev} , {ov} )")
            body.append(f"{verb_lemma} . agent ( {ev} , {name} )")
        else:
            continue

        new_lf = (" ; ".join(presups) + " ; " + " AND ".join(body)) if presups else " AND ".join(body)
        aug.append((new_inp, new_lf, "aug_active_to_passive"))
    return aug


def generate_dative_swap(train_pairs):
    """Transform DO-dative ↔ PP-dative.

    DO→PP patterns:
      'NAME VERB NAME DET OBJ .'         (6t) → 'NAME VERB DET OBJ to NAME .'       (7t)
      'DET NOUN VERB NAME DET OBJ .'     (7t) → 'DET NOUN VERB DET OBJ to NAME .'   (8t)
      'DET NOUN VERB DET N DET OBJ .'    (8t) → 'DET NOUN VERB DET OBJ to DET N .'  (9t)
    PP→DO patterns (reverse of above).
    """
    DETS = {"a", "A", "the", "The"}
    aug = []
    for inp, lf, _ in train_pairs:
        if ". recipient" not in lf:
            continue
        if ". nmod ." in lf or ". xcomp" in lf:
            continue
        m = re.search(r'(\w+) \. agent', lf)
        if not m:
            continue
        verb_lemma = m.group(1)
        tokens = inp.split()
        if tokens[-1] != ".":
            continue

        def _var(n): return f"x _ {n}"
        def _is_proper(t): return t[0].isupper() and t not in DETS and len(t) > 1

        if "to" not in tokens:
            # ========= DO-dative → PP-dative =========

            # DO-A: NAME(0) VERB(1) NAME(2) DET(3) OBJ(4) .(5) — 6 tokens
            if (len(tokens) == 6 and _is_proper(tokens[0]) and
                _is_proper(tokens[2]) and tokens[3] in DETS):
                ag_name, vf, rec_name = tokens[0], tokens[1], tokens[2]
                od, on = tokens[3], tokens[4]
                # → NAME(0) VERB(1) DET(2) OBJ(3) to(4) NAME(5) .(6)
                new_inp = f"{ag_name} {vf} {od} {on} to {rec_name} ."
                ev = _var(1); tv = _var(3)
                presups = [f"* {on} ( {tv} )"] if od.lower() == "the" else []
                body = [f"{verb_lemma} . agent ( {ev} , {ag_name} )",
                        f"{verb_lemma} . theme ( {ev} , {tv} )",
                        f"{verb_lemma} . recipient ( {ev} , {rec_name} )"]
                if od.lower() == "a": body.append(f"{on} ( {tv} )")

            # DO-B: DET(0) NOUN(1) VERB(2) NAME(3) DET(4) OBJ(5) .(6) — 7 tokens
            elif (len(tokens) == 7 and tokens[0] in DETS and
                  _is_proper(tokens[3]) and tokens[4] in DETS):
                sd, sn, vf, rec_name = tokens[0], tokens[1], tokens[2], tokens[3]
                od, on = tokens[4], tokens[5]
                # → DET(0) NOUN(1) VERB(2) DET(3) OBJ(4) to(5) NAME(6) .(7)
                new_inp = f"{sd} {sn} {vf} {od} {on} to {rec_name} ."
                sv = _var(1); ev = _var(2); tv = _var(4)
                presups = []
                if sd.lower() == "the": presups.append(f"* {sn} ( {sv} )")
                if od.lower() == "the": presups.append(f"* {on} ( {tv} )")
                body = []
                if sd.lower() == "a": body.append(f"{sn} ( {sv} )")
                body += [f"{verb_lemma} . agent ( {ev} , {sv} )",
                         f"{verb_lemma} . theme ( {ev} , {tv} )",
                         f"{verb_lemma} . recipient ( {ev} , {rec_name} )"]
                if od.lower() == "a": body.append(f"{on} ( {tv} )")

            # DO-C: DET(0) N(1) VERB(2) DET(3) REC(4) DET(5) OBJ(6) .(7) — 8 tok
            elif (len(tokens) == 8 and tokens[0] in DETS and
                  tokens[3] in DETS and tokens[5] in DETS):
                sd, sn, vf = tokens[0], tokens[1], tokens[2]
                rd, rn = tokens[3], tokens[4]
                od, on = tokens[5], tokens[6]
                # → DET(0) N(1) VERB(2) DET(3) OBJ(4) to(5) DET(6) REC(7) .(8)
                new_inp = f"{sd} {sn} {vf} {od} {on} to {rd} {rn} ."
                sv = _var(1); ev = _var(2); tv = _var(4); rv = _var(7)
                presups = []
                if sd.lower() == "the": presups.append(f"* {sn} ( {sv} )")
                if od.lower() == "the": presups.append(f"* {on} ( {tv} )")
                if rd.lower() == "the": presups.append(f"* {rn} ( {rv} )")
                body = []
                if sd.lower() == "a": body.append(f"{sn} ( {sv} )")
                body += [f"{verb_lemma} . agent ( {ev} , {sv} )",
                         f"{verb_lemma} . theme ( {ev} , {tv} )",
                         f"{verb_lemma} . recipient ( {ev} , {rv} )"]
                if od.lower() == "a": body.append(f"{on} ( {tv} )")
                if rd.lower() == "a": body.append(f"{rn} ( {rv} )")
            else:
                continue

        else:
            # ========= PP-dative → DO-dative =========
            to_idx = tokens.index("to")

            # PP-A: NAME(0) VERB(1) DET(2) OBJ(3) to(4) NAME(5) .(6) — 7 tok
            if (len(tokens) == 7 and _is_proper(tokens[0]) and
                tokens[2] in DETS and _is_proper(tokens[5])):
                ag_name, vf = tokens[0], tokens[1]
                od, on = tokens[2], tokens[3]
                rec_name = tokens[5]
                # → NAME(0) VERB(1) NAME(2) DET(3) OBJ(4) .(5)
                new_inp = f"{ag_name} {vf} {rec_name} {od} {on} ."
                ev = _var(1); tv = _var(4)
                presups = [f"* {on} ( {tv} )"] if od.lower() == "the" else []
                body = []
                body += [f"{verb_lemma} . agent ( {ev} , {ag_name} )",
                         f"{verb_lemma} . recipient ( {ev} , {rec_name} )",
                         f"{verb_lemma} . theme ( {ev} , {tv} )"]
                if od.lower() == "a": body.append(f"{on} ( {tv} )")

            # PP-B: DET(0) N(1) VERB(2) DET(3) OBJ(4) to(5) NAME(6) .(7) — 8 tok
            elif (len(tokens) == 8 and tokens[0] in DETS and
                  tokens[3] in DETS and _is_proper(tokens[6])):
                sd, sn, vf = tokens[0], tokens[1], tokens[2]
                od, on = tokens[3], tokens[4]
                rec_name = tokens[6]
                # → DET(0) N(1) VERB(2) NAME(3) DET(4) OBJ(5) .(6)
                new_inp = f"{sd} {sn} {vf} {rec_name} {od} {on} ."
                sv = _var(1); ev = _var(2); tv = _var(5)
                presups = []
                if sd.lower() == "the": presups.append(f"* {sn} ( {sv} )")
                if od.lower() == "the": presups.append(f"* {on} ( {tv} )")
                body = []
                if sd.lower() == "a": body.append(f"{sn} ( {sv} )")
                body += [f"{verb_lemma} . agent ( {ev} , {sv} )",
                         f"{verb_lemma} . recipient ( {ev} , {rec_name} )",
                         f"{verb_lemma} . theme ( {ev} , {tv} )"]
                if od.lower() == "a": body.append(f"{on} ( {tv} )")

            # PP-C: NAME(0) VERB(1) DET(2) OBJ(3) to(4) DET(5) REC(6) .(7) — 8 tok
            elif (len(tokens) == 8 and _is_proper(tokens[0]) and
                  tokens[2] in DETS and tokens[5] in DETS):
                ag_name, vf = tokens[0], tokens[1]
                od, on = tokens[2], tokens[3]
                rd, rn = tokens[5], tokens[6]
                # → NAME(0) VERB(1) DET(2) REC(3) DET(4) OBJ(5) .(6)
                new_inp = f"{ag_name} {vf} {rd} {rn} {od} {on} ."
                ev = _var(1); rv = _var(3); tv = _var(5)
                presups = []
                if rd.lower() == "the": presups.append(f"* {rn} ( {rv} )")
                if od.lower() == "the": presups.append(f"* {on} ( {tv} )")
                body = []
                body += [f"{verb_lemma} . agent ( {ev} , {ag_name} )",
                         f"{verb_lemma} . recipient ( {ev} , {rv} )",
                         f"{verb_lemma} . theme ( {ev} , {tv} )"]
                if rd.lower() == "a": body.append(f"{rn} ( {rv} )")
                if od.lower() == "a": body.append(f"{on} ( {tv} )")
            else:
                continue

        new_lf = (" ; ".join(presups) + " ; " + " AND ".join(body)) if presups else " AND ".join(body)
        aug.append((new_inp, new_lf, "aug_dative_swap"))
    return aug


def generate_noun_position_augmentations(train_pairs):
    """For nouns seen only as object (or only as subject), generate examples
    in the missing position by swapping subj↔obj in existing train examples.

    Handles simple transitives: DET NOUN VERB DET NOUN . (6 tokens)"""
    DETS = {"a", "A", "the", "The"}
    # Count subject/object occurrences per common noun
    from collections import defaultdict
    subj_count = defaultdict(int)
    obj_count  = defaultdict(int)
    for inp, lf, _ in train_pairs:
        tokens = inp.split()
        if len(tokens) != 6 or tokens[-1] != "." or tokens[0] not in DETS or tokens[3] not in DETS:
            continue
        sn, on = tokens[1], tokens[4]
        if sn[0].islower(): subj_count[sn] += 1
        if on[0].islower(): obj_count[on] += 1

    all_nouns = set(subj_count.keys()) | set(obj_count.keys())
    needs_subj = {n for n in all_nouns if subj_count[n] == 0 and obj_count[n] > 0}
    needs_obj  = {n for n in all_nouns if obj_count[n] == 0 and subj_count[n] > 0}
    # Also add nouns with very few occurrences in one position
    for n in all_nouns:
        if subj_count[n] <= 1 and obj_count[n] >= 5:
            needs_subj.add(n)
        if obj_count[n] <= 1 and subj_count[n] >= 5:
            needs_obj.add(n)

    aug = []
    for inp, lf, _ in train_pairs:
        tokens = inp.split()
        if len(tokens) != 6 or tokens[-1] != "." or tokens[0] not in DETS or tokens[3] not in DETS:
            continue
        if ". nmod ." in lf or ". recipient" in lf or ". xcomp" in lf:
            continue

        m = re.search(r'(\w+) \. agent', lf)
        if not m:
            continue
        verb_lemma = m.group(1)

        sd, sn, vf = tokens[0], tokens[1], tokens[2]
        od, on = tokens[3], tokens[4]
        if not (sn[0].islower() and on[0].islower()):
            continue

        # If object noun needs subject position → swap subj↔obj
        if on in needs_subj:
            new_inp = f"{od.capitalize()} {on} {vf} {sd.lower()} {sn} ."
            sv, ev, ov = "x _ 1", "x _ 2", "x _ 4"
            presups = []
            if od.lower() == "the": presups.append(f"* {on} ( {sv} )")
            if sd.lower() == "the": presups.append(f"* {sn} ( {ov} )")
            body = []
            if od.lower() == "a": body.append(f"{on} ( {sv} )")
            body.append(f"{verb_lemma} . agent ( {ev} , {sv} )")
            body.append(f"{verb_lemma} . theme ( {ev} , {ov} )")
            if sd.lower() == "a": body.append(f"{sn} ( {ov} )")
            new_lf = (" ; ".join(presups) + " ; " + " AND ".join(body)) if presups else " AND ".join(body)
            aug.append((new_inp, new_lf, "aug_noun_pos"))

        # If subject noun needs object position → swap subj↔obj
        if sn in needs_obj:
            new_inp = f"{od.capitalize()} {on} {vf} {sd.lower()} {sn} ."
            sv, ev, ov = "x _ 1", "x _ 2", "x _ 4"
            presups = []
            if od.lower() == "the": presups.append(f"* {on} ( {sv} )")
            if sd.lower() == "the": presups.append(f"* {sn} ( {ov} )")
            body = []
            if od.lower() == "a": body.append(f"{on} ( {sv} )")
            body.append(f"{verb_lemma} . agent ( {ev} , {sv} )")
            body.append(f"{verb_lemma} . theme ( {ev} , {ov} )")
            if sd.lower() == "a": body.append(f"{sn} ( {ov} )")
            new_lf = (" ; ".join(presups) + " ; " + " AND ".join(body)) if presups else " AND ".join(body)
            aug.append((new_inp, new_lf, "aug_noun_pos"))

    # Deduplicate and cap at 1000
    seen = set()
    deduped = []
    for item in aug:
        if item[0] not in seen:
            seen.add(item[0])
            deduped.append(item)
    if len(deduped) > 1000:
        import random as _r
        _r.shuffle(deduped)
        deduped = deduped[:1000]
    return deduped


def generate_subj_pp_augmentations(train_pairs):
    """From training examples with PP on object + common-noun subject,
    generate variants with PP moved to subject. Returns [(inp, lf, cat)].

    Only handles simple transitive: DET NOUN VERB DET NOUN PREP DET NOUN ."""
    PREPS = {"beside", "in", "on"}
    DETS = {"a", "A", "the", "The"}
    aug = []
    for inp, lf, cat in train_pairs:
        if ". nmod ." not in lf:
            continue
        tokens = inp.split()
        if len(tokens) != 9 or tokens[-1] != ".":
            continue
        if tokens[0] not in DETS or tokens[3] not in DETS:
            continue
        if tokens[5] not in PREPS or tokens[6] not in DETS:
            continue

        subj_det, subj_noun = tokens[0], tokens[1]
        verb_form = tokens[2]
        obj_det, obj_noun = tokens[3], tokens[4]
        prep, pp_det, pp_noun = tokens[5], tokens[6], tokens[7]

        m = re.search(r'(\w+) \. agent', lf)
        if not m:
            continue
        verb_lemma = m.group(1)

        new_inp = (f"{subj_det} {subj_noun} {prep} {pp_det} {pp_noun} "
                   f"{verb_form} {obj_det} {obj_noun} .")

        # x_N = 0-indexed word position
        # DET(0) SUBJ(1) PREP(2) DET(3) PP(4) VERB(5) DET(6) OBJ(7) .(8)
        sv, pv, ev, ov = "x _ 1", "x _ 4", "x _ 5", "x _ 7"

        presups = []
        if subj_det.lower() == "the":
            presups.append(f"* {subj_noun} ( {sv} )")
        if pp_det.lower() == "the":
            presups.append(f"* {pp_noun} ( {pv} )")
        if obj_det.lower() == "the":
            presups.append(f"* {obj_noun} ( {ov} )")

        body = []
        if subj_det.lower() == "a":
            body.append(f"{subj_noun} ( {sv} )")
        body.append(f"{subj_noun} . nmod . {prep} ( {sv} , {pv} )")
        if pp_det.lower() == "a":
            body.append(f"{pp_noun} ( {pv} )")
        body.append(f"{verb_lemma} . agent ( {ev} , {sv} )")
        body.append(f"{verb_lemma} . theme ( {ev} , {ov} )")
        if obj_det.lower() == "a":
            body.append(f"{obj_noun} ( {ov} )")

        if presups:
            new_lf = " ; ".join(presups) + " ; " + " AND ".join(body)
        else:
            new_lf = " AND ".join(body)
        aug.append((new_inp, new_lf, "aug_struct_auto"))
    return aug


def _is_verb_cluster(tokens, train_pairs):
    """Heuristic: a cluster is verbs if it's all lowercase AND >50% of its
    members appear right after 'to' (infinitive position).
    Capitalized clusters (proper names) are never verbs."""
    if all(t[0].isupper() for t in tokens if t):
        return False
    after_to = set()
    for inp, _, _ in train_pairs:
        ws = inp.split()
        for i, w in enumerate(ws):
            if w == "to" and i + 1 < len(ws):
                after_to.add(ws[i + 1])
    return sum(1 for t in tokens if t in after_to) > len(tokens) * 0.5


def build_extended_perm_classes(train_pairs, in_w2i, out_w2i, exclude_verbs=False):
    """Build permutation classes from detect_permutable clusters.

    Keeps only classes where every token exists identically in both
    in_w2i and out_w2i (safe to permute in/out).
    If exclude_verbs=True, drop clusters that look like verb classes."""
    from detect_permutable import detect_permutable
    pairs_flat = [(inp, lf) for inp, lf, _ in train_pairs]
    clusters = detect_permutable(pairs_flat, window=2, sim_thr=0.8, min_freq=5,
                                 exclude={".", ",", ";"})
    classes = []
    skipped = []
    for cluster in clusters:
        safe_tokens = [t for t in cluster if t in in_w2i and t in out_w2i and len(t) > 1]
        if len(safe_tokens) < 2:
            continue
        if exclude_verbs and _is_verb_cluster(safe_tokens, train_pairs):
            skipped.append(safe_tokens)
            continue
        safe = [(in_w2i[t], out_w2i[t]) for t in safe_tokens]
        classes.append(safe)
    return classes, clusters, skipped


# ══════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════
class COGSDataset(Dataset):
    """COGS (input, logical_form, category). Supports multi-class permutation:
    perm_classes is a list of classes, each class is a list of (in_id, out_id).
    Each __getitem__ draws an independent random permutation per class."""

    def __init__(self, pairs, in_w2i, out_w2i, max_out=MAX_OUT,
                 perm_classes=None):
        self.data = []
        self.cats = []
        for inp, lf, cat in pairs:
            src = [in_w2i.get(t, 0) for t in inp.split()]
            tgt = [out_w2i[BOS]] + [out_w2i.get(t, 0) for t in lf.split()] + [out_w2i[EOS]]
            tgt = tgt[:max_out]
            self.data.append((src, tgt, len(inp.split())))
            self.cats.append(cat)
        self.perm_classes = [c for c in (perm_classes or []) if len(c) >= 2]
        self.permute = len(self.perm_classes) > 0

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        src, tgt, n = self.data[i]
        cat = self.cats[i]
        if self.permute:
            m_in, m_out = {}, {}
            for cls in self.perm_classes:
                K = len(cls)
                order = list(range(K))
                random.shuffle(order)
                for k in range(K):
                    m_in[cls[k][0]]  = cls[order[k]][0]
                    m_out[cls[k][1]] = cls[order[k]][1]
            src = [m_in.get(t, t)  for t in src]
            tgt = [m_out.get(t, t) for t in tgt]
        return (src, tgt, n, cat)


def collate(batch):
    src, tgt, in_lens, cats = zip(*batch)
    S = max(len(x) for x in src)
    T = max(len(x) for x in tgt)
    B = len(batch)
    s  = torch.zeros(B, S, dtype=torch.long)
    t_ = torch.zeros(B, T, dtype=torch.long)
    s_mask = torch.zeros(B, S, dtype=torch.bool)
    for i, (a, b, _, _) in enumerate(batch):
        s[i, :len(a)]  = torch.tensor(a)
        t_[i, :len(b)] = torch.tensor(b)
        s_mask[i, :len(a)] = True
    return s, s_mask, t_, torch.tensor(in_lens), list(cats)


# ══════════════════════════════════════════════════════════════
# Model (standalone, mirrors BaseSeq2Seq in scan_compositional.py)
# ══════════════════════════════════════════════════════════════
class CrossAttnDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn, dropout=0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.n3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ffn, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, mem_kpm, causal):
        h = self.n1(x)
        h, _ = self.self_attn(h, h, h, attn_mask=causal, need_weights=False)
        x = x + self.drop(h)
        h = self.n2(x)
        h, _ = self.cross_attn(h, memory, memory, key_padding_mask=mem_kpm, need_weights=False)
        x = x + self.drop(h)
        h = self.n3(x)
        x = x + self.drop(self.ffn(h))
        return x


class TransformerSeq2Seq(nn.Module):
    """Encoder 2 layers + Decoder 2 layers, d=128, 4 heads. Pre-LN."""

    def __init__(self, in_vocab, out_vocab, d_model=128, n_heads=4,
                 n_layers=2, ffn=256, dropout=0.1,
                 max_in=MAX_IN, max_out=MAX_OUT):
        super().__init__()
        self.d_model = d_model
        self.in_emb  = nn.Embedding(in_vocab,  d_model, padding_idx=0)
        self.out_emb = nn.Embedding(out_vocab, d_model, padding_idx=0)
        self.in_pe   = nn.Embedding(max_in,  d_model)
        self.out_pe  = nn.Embedding(max_out, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, ffn, dropout, activation=F.gelu,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.dec_layers = nn.ModuleList([
            CrossAttnDecoderLayer(d_model, n_heads, ffn, dropout)
            for _ in range(n_layers)
        ])
        self.dec_norm = nn.LayerNorm(d_model)
        self.out_head = nn.Linear(d_model, out_vocab)
        self.max_in, self.max_out = max_in, max_out

    def encode(self, src, src_mask):
        L = src.size(1)
        if L > self.max_in:
            raise RuntimeError(f"src length {L} > max_in {self.max_in}")
        pos = torch.arange(L, device=src.device).unsqueeze(0)
        x = self.in_emb(src) + self.in_pe(pos)
        return self.encoder(x, src_key_padding_mask=~src_mask)

    def decode(self, memory, mem_kpm, tgt):
        B, T = tgt.shape
        if T > self.max_out:
            raise RuntimeError(f"tgt length {T} > max_out {self.max_out}")
        pos = torch.arange(T, device=tgt.device).unsqueeze(0)
        x = self.out_emb(tgt) + self.out_pe(pos)
        causal = torch.triu(torch.ones(T, T, device=tgt.device, dtype=torch.bool), 1)
        for layer in self.dec_layers:
            x = layer(x, memory, mem_kpm, causal)
        x = self.dec_norm(x)
        return self.out_head(x)

    def forward(self, src, src_mask, tgt_in):
        enc = self.encode(src, src_mask)
        logits = self.decode(enc, ~src_mask, tgt_in)
        return logits, {"enc_out": enc}


class TransformerSeq2SeqWithCopy(TransformerSeq2Seq):
    """Same architecture + pointer network / copy mechanism.

    Each output position mixes:
      - gen_probs: standard vocab softmax
      - copy_probs: attention over input, scattered onto output vocab
        via in_to_out_map (mapping input token ids to matching output ids)
    gate = sigmoid(Linear(d_model → 1)) mixes the two.

    Forward returns log-probs (not logits) — training uses nll_loss.
    """

    def __init__(self, in_vocab, out_vocab, d_model=128, n_heads=4, n_layers=2,
                 ffn=256, dropout=0.1, max_in=MAX_IN, max_out=MAX_OUT,
                 in_to_out_map=None):
        super().__init__(in_vocab, out_vocab, d_model, n_heads, n_layers,
                         ffn, dropout, max_in, max_out)
        self.copy_gate = nn.Linear(d_model, 1)
        if in_to_out_map is None:
            in_to_out_map = torch.full((in_vocab,), -1, dtype=torch.long)
        self.register_buffer("in_to_out_map", in_to_out_map)

    def decode_with_copy(self, src_ids, memory, mem_kpm, tgt):
        """Run decoder + compute mixed log-probs using copy mechanism."""
        B, T = tgt.shape
        pos = torch.arange(T, device=tgt.device).unsqueeze(0)
        x = self.out_emb(tgt) + self.out_pe(pos)
        causal = torch.triu(torch.ones(T, T, device=tgt.device, dtype=torch.bool), 1)
        for layer in self.dec_layers:
            x = layer(x, memory, mem_kpm, causal)
        x = self.dec_norm(x)

        # Generation distribution
        gen_logits = self.out_head(x)                           # (B, T, V)
        gen_log_probs = F.log_softmax(gen_logits, dim=-1)
        gen_probs = gen_log_probs.exp()

        # Copy attention: dot product between decoder state and encoder memory
        copy_scores = torch.bmm(x, memory.transpose(1, 2))      # (B, T, T_src)
        copy_scores = copy_scores.masked_fill(mem_kpm.unsqueeze(1), -6e4)
        copy_attn = F.softmax(copy_scores.float(), dim=-1)      # (B, T, T_src)

        # Map input ids to output ids (or -1 if not in out vocab)
        src_out_ids = self.in_to_out_map[src_ids]               # (B, T_src)
        valid = (src_out_ids >= 0).float()                      # (B, T_src)
        # Also mask padding positions
        valid = valid * (src_ids != 0).float()
        copy_attn_masked = copy_attn * valid.unsqueeze(1)       # (B, T, T_src)
        src_out_ids_safe = src_out_ids.clamp(min=0)             # negative → 0 (will be zeroed)

        # Scatter onto vocab
        copy_probs = torch.zeros_like(gen_probs)                # (B, T, V)
        idx = src_out_ids_safe.unsqueeze(1).expand(-1, T, -1)   # (B, T, T_src)
        copy_probs.scatter_add_(2, idx, copy_attn_masked.to(copy_probs.dtype))

        # Gate
        gate = torch.sigmoid(self.copy_gate(x))                 # (B, T, 1)

        # Mix
        mixed = gate * copy_probs + (1.0 - gate) * gen_probs
        log_probs = torch.log(mixed.clamp(min=1e-10))
        return log_probs

    def forward(self, src, src_mask, tgt_in):
        enc = self.encode(src, src_mask)
        log_probs = self.decode_with_copy(src, enc, ~src_mask, tgt_in)
        return log_probs, {"enc_out": enc, "uses_copy": True}


def build_in_to_out_map(in_w2i, out_w2i):
    """For each input token, find its equivalent in output vocab (same string).
    Returns a tensor of shape (len(in_w2i),) with out_ids or -1."""
    mapping = torch.full((len(in_w2i),), -1, dtype=torch.long)
    for tok, iid in in_w2i.items():
        if tok in out_w2i:
            mapping[iid] = out_w2i[tok]
    return mapping


# ══════════════════════════════════════════════════════════════
# Structural tagger (role prediction on encoder)
# ══════════════════════════════════════════════════════════════
ROLE_OTHER, ROLE_SUBJ, ROLE_VERB, ROLE_OBJ, ROLE_PREP, ROLE_MOD, ROLE_BY_AGENT = 0, 1, 2, 3, 4, 5, 6
N_ROLES = 7
ROLE_NAMES_STR = ["OTHER", "SUBJ", "VERB", "OBJ", "PREP", "MOD", "BY_AGENT"]


def build_token_categories(in_w2i):
    """Return sets of token IDs for rule-based role derivation."""
    def ids(toks): return {in_w2i[t] for t in toks if t in in_w2i}
    return {
        "det":   ids(["a", "A", "the", "The"]),
        "was":   ids(["was"]),
        "by":    ids(["by"]),
        "to":    ids(["to"]),
        "that":  ids(["that"]),
        "prep":  ids(["beside", "in", "on"]),
        "period": ids(["."]),
    }


def derive_roles_tensor(src_ids, cats):
    """src_ids: (B, T) LongTensor. Returns (B, T) role labels.
    Uses rule-based parsing: subj → verb → (by agent | to recip | PP | obj)."""
    B, T = src_ids.shape
    device = src_ids.device
    roles = torch.zeros((B, T), dtype=torch.long, device=device)
    det, was, by_, to_, that_, prep_s, period = (cats["det"], cats["was"],
        cats["by"], cats["to"], cats["that"], cats["prep"], cats["period"])
    # Process each row — vectorization possible but complex; Python loop is fine
    for b in range(B):
        tokens = src_ids[b].tolist()
        n = T
        # Skip padding from end
        while n > 0 and tokens[n-1] == 0:
            n -= 1
        i = 0
        # initial det?
        if i < n and tokens[i] in det:
            i += 1
        # subject noun (or proper name at position 0)
        if i < n and tokens[i] not in det and tokens[i] != 0:
            roles[b, i] = ROLE_SUBJ
            i += 1
        # auxiliary "was"?
        if i < n and tokens[i] in was:
            i += 1
        # main verb
        if i < n and tokens[i] != 0:
            roles[b, i] = ROLE_VERB
            i += 1
        # rest of sentence
        while i < n:
            t = tokens[i]
            if t == 0:
                i += 1; continue
            if t in det:
                i += 1; continue
            if t in by_:
                roles[b, i] = ROLE_PREP
                i += 1
                if i < n and tokens[i] in det: i += 1
                if i < n and tokens[i] != 0:
                    roles[b, i] = ROLE_BY_AGENT
                    i += 1
            elif t in to_ or t in prep_s:
                roles[b, i] = ROLE_PREP
                i += 1
                if i < n and tokens[i] in det: i += 1
                if i < n and tokens[i] != 0:
                    roles[b, i] = ROLE_MOD
                    i += 1
            elif t in that_:
                # embedded clause — reset parsing
                i += 1
                if i < n and tokens[i] in det: i += 1
                if i < n and tokens[i] != 0:
                    roles[b, i] = ROLE_SUBJ
                    i += 1
                if i < n and tokens[i] in was:
                    i += 1
                if i < n and tokens[i] != 0:
                    roles[b, i] = ROLE_VERB
                    i += 1
            elif t in period:
                i += 1  # OTHER, no increment of roles
            else:
                # direct object noun (default)
                roles[b, i] = ROLE_OBJ
                i += 1
    return roles


class CrossAttnDecoderLayerSplit(nn.Module):
    """Same as CrossAttnDecoderLayer but takes separate K and V in cross-attn."""
    def __init__(self, d_model, n_heads, ffn, dropout=0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.n3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ffn, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory_k, memory_v, mem_kpm, causal):
        h = self.n1(x)
        h, _ = self.self_attn(h, h, h, attn_mask=causal, need_weights=False)
        x = x + self.drop(h)
        h = self.n2(x)
        h, _ = self.cross_attn(h, memory_k, memory_v, key_padding_mask=mem_kpm, need_weights=False)
        x = x + self.drop(h)
        h = self.n3(x)
        x = x + self.drop(self.ffn(h))
        return x


class TransformerSeq2SeqCopyTags(TransformerSeq2SeqWithCopy):
    """Copy mechanism + light structural tagger.

    Encoder output → tagger head (N_ROLES classes).
    Role embedding (16d) concatenated with enc_out (128d) → projection to 128d = K.
    V = enc_out unchanged.
    Aux loss: CE on role predictions vs rule-based ground truth (λ=0.3).
    """
    def __init__(self, in_vocab, out_vocab, d_model=128, n_heads=4, n_layers=2,
                 ffn=256, dropout=0.1, max_in=MAX_IN, max_out=MAX_OUT,
                 in_to_out_map=None, role_dim=16, token_categories=None):
        super().__init__(in_vocab, out_vocab, d_model, n_heads, n_layers, ffn,
                         dropout, max_in, max_out, in_to_out_map=in_to_out_map)
        self.tagger = nn.Linear(d_model, N_ROLES)
        self.role_emb = nn.Embedding(N_ROLES, role_dim)
        self.k_proj = nn.Linear(d_model + role_dim, d_model)
        # Replace decoder layers with split-KV versions
        self.dec_layers = nn.ModuleList([
            CrossAttnDecoderLayerSplit(d_model, n_heads, ffn, dropout)
            for _ in range(n_layers)
        ])
        # Token categories for rule-based role derivation
        self.token_categories = token_categories or {}

    def decode_with_copy_tags(self, src_ids, memory, mem_kpm, tgt, roles_gt=None):
        """Run decoder with role-enriched K; return (log_probs, role_logits)."""
        # Tagger predictions
        role_logits = self.tagger(memory)            # (B, T_src, N_ROLES)

        # Use ground-truth roles for cross-attention K at training,
        # predicted roles at inference
        if roles_gt is not None:
            roles_used = roles_gt
        else:
            roles_used = role_logits.argmax(-1)

        role_vecs = self.role_emb(roles_used)        # (B, T_src, role_dim)
        enriched_k = torch.cat([memory, role_vecs], dim=-1)  # (B, T_src, 144)
        K = self.k_proj(enriched_k)                  # (B, T_src, 128)

        # Decoder with split K/V
        B, T = tgt.shape
        pos = torch.arange(T, device=tgt.device).unsqueeze(0)
        x = self.out_emb(tgt) + self.out_pe(pos)
        causal = torch.triu(torch.ones(T, T, device=tgt.device, dtype=torch.bool), 1)
        for layer in self.dec_layers:
            x = layer(x, K, memory, mem_kpm, causal)
        x = self.dec_norm(x)

        # Generation
        gen_logits = self.out_head(x)
        gen_log_probs = F.log_softmax(gen_logits, dim=-1)
        gen_probs = gen_log_probs.exp()

        # Copy (use ORIGINAL memory, not K, for attention scores is OK but
        # conceptually we want the copy to respect structure too — use K)
        copy_scores = torch.bmm(x, K.transpose(1, 2))
        copy_scores = copy_scores.masked_fill(mem_kpm.unsqueeze(1), -6e4)
        copy_attn = F.softmax(copy_scores.float(), dim=-1)

        src_out_ids = self.in_to_out_map[src_ids]
        valid = (src_out_ids >= 0).float() * (src_ids != 0).float()
        copy_attn_masked = copy_attn * valid.unsqueeze(1)
        src_out_ids_safe = src_out_ids.clamp(min=0)

        copy_probs = torch.zeros_like(gen_probs)
        idx = src_out_ids_safe.unsqueeze(1).expand(-1, T, -1)
        copy_probs.scatter_add_(2, idx, copy_attn_masked.to(copy_probs.dtype))

        gate = torch.sigmoid(self.copy_gate(x))
        mixed = gate * copy_probs + (1.0 - gate) * gen_probs
        log_probs = torch.log(mixed.clamp(min=1e-10))

        return log_probs, role_logits

    def forward(self, src, src_mask, tgt_in):
        enc = self.encode(src, src_mask)
        # Derive ground-truth roles from input
        roles_gt = derive_roles_tensor(src, self.token_categories) if self.training else None
        log_probs, role_logits = self.decode_with_copy_tags(
            src, enc, ~src_mask, tgt_in, roles_gt=roles_gt)
        info = {"enc_out": enc, "uses_copy": True, "uses_tags": True,
                "role_logits": role_logits, "roles_gt": roles_gt}
        return log_probs, info


# ══════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_tf(model, loader, device):
    """Teacher-forced eval — one forward pass per batch, runs in seconds.
    Returns overall metrics + per-category breakdown."""
    model.eval()
    n, exact, tok_c, tok_t = 0, 0, 0, 0
    by_cat: Dict[str, Tuple[int, int]] = {}   # cat → (total, correct)
    for src, src_mask, tgt, _, cats in loader:
        src = src.to(device); src_mask = src_mask.to(device); tgt = tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits, _ = model(src, src_mask, tgt_in)
        pred = logits.argmax(-1)
        mask = (tgt_out != 0)
        match = ((pred == tgt_out) | ~mask).all(dim=1)
        exact += match.sum().item()
        tok_c += ((pred == tgt_out) & mask).sum().item()
        tok_t += mask.sum().item()
        n += src.size(0)
        for i, cat in enumerate(cats):
            t_, c_ = by_cat.get(cat, (0, 0))
            by_cat[cat] = (t_ + 1, c_ + (1 if match[i].item() else 0))
    by_cat_pct = {c: round(cor / tot * 100, 2) for c, (tot, cor) in sorted(by_cat.items())}
    return {"exact": exact / n * 100, "tok_acc": tok_c / max(tok_t, 1) * 100,
            "n": n, "by_cat": by_cat_pct}


@torch.no_grad()
def greedy_decode(model, src, src_mask, out_w2i, max_len=MAX_DECODE):
    B = src.size(0)
    device = src.device
    bos, eos = out_w2i[BOS], out_w2i[EOS]
    enc = model.encode(src, src_mask)
    mem_kpm = ~src_mask
    tgt = torch.full((B, 1), bos, device=device, dtype=torch.long)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    uses_tags = isinstance(model, TransformerSeq2SeqCopyTags)
    uses_copy = isinstance(model, TransformerSeq2SeqWithCopy)
    for _ in range(max_len):
        if uses_tags:
            logits, _ = model.decode_with_copy_tags(src, enc, mem_kpm, tgt, roles_gt=None)
        elif uses_copy:
            logits = model.decode_with_copy(src, enc, mem_kpm, tgt)
        else:
            logits = model.decode(enc, mem_kpm, tgt)
        nxt = logits[:, -1].argmax(-1)
        nxt = torch.where(done, torch.zeros_like(nxt), nxt)
        tgt = torch.cat([tgt, nxt.unsqueeze(1)], dim=1)
        done = done | (nxt == eos)
        if done.all(): break
    return tgt[:, 1:]


@torch.no_grad()
def evaluate(model, loader, out_w2i, device, max_len=MAX_DECODE):
    model.eval()
    n, exact, tok_c, tok_t = 0, 0, 0, 0
    by_cat: Dict[str, Tuple[int, int]] = {}
    for src, src_mask, tgt, _, cats in loader:
        src = src.to(device); src_mask = src_mask.to(device); tgt = tgt.to(device)
        pred = greedy_decode(model, src, src_mask, out_w2i, max_len=max_len)
        tgt_cmp = tgt[:, 1:]
        Lp, Lt = pred.size(1), tgt_cmp.size(1)
        if Lp < Lt:
            pad = torch.zeros(pred.size(0), Lt - Lp, dtype=pred.dtype, device=device)
            pred = torch.cat([pred, pad], dim=1)
        else:
            pred = pred[:, :Lt]
        mask = (tgt_cmp != 0)
        match = ((pred == tgt_cmp) | ~mask).all(dim=1)
        exact += match.sum().item()
        tok_c += ((pred == tgt_cmp) & mask).sum().item()
        tok_t += mask.sum().item()
        n += src.size(0)
        for i, cat in enumerate(cats):
            t_, c_ = by_cat.get(cat, (0, 0))
            by_cat[cat] = (t_ + 1, c_ + (1 if match[i].item() else 0))
    by_cat_pct = {c: round(cor / tot * 100, 2) for c, (tot, cor) in sorted(by_cat.items())}
    return {"exact": exact / n * 100, "tok_acc": tok_c / max(tok_t, 1) * 100,
            "n": n, "by_cat": by_cat_pct}


def train_one_run(variant: str, seed: int, args):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = args.runs_dir if args.runs_dir else RUNS_DIR
    os.makedirs(base_dir, exist_ok=True)
    run_dir = os.path.join(base_dir, f"{variant}_s{seed}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n{'='*70}")
    bench_name = getattr(args, '_bench_name', 'COGS')
    print(f"  {bench_name}  |  {variant}  |  seed {seed}  |  {run_dir}")
    print(f"{'='*70}")

    # Data
    for split, url in COGS_SPLITS.items():
        _download(url, os.path.join(DATA_DIR, f"{split}.tsv"))
    train_pairs = parse_cogs_tsv(os.path.join(DATA_DIR, "train.tsv"))
    dev_pairs   = parse_cogs_tsv(os.path.join(DATA_DIR, "dev.tsv"))
    gen_pairs   = parse_cogs_tsv(os.path.join(DATA_DIR, "gen.tsv"))

    # B3/B3_auto/B4/B4_turbo/B5/B6: augment training set with structural examples
    if variant in ("B3", "B3_auto", "B4", "B4_turbo", "B5", "B5_lite", "B6", "B7a", "B7c"):
        # Manual PP examples (B3 only; B3_auto/B4/B5/B5_lite use auto-generated)
        if variant == "B3":
            aug_path = os.path.join(DATA_DIR, "aug_struct.tsv")
            if not os.path.exists(aug_path):
                raise FileNotFoundError(f"B3 requires {aug_path}")
            aug_pairs = parse_cogs_tsv(aug_path)
            n_copies = max(1, int(len(train_pairs) * 0.01 / len(aug_pairs)))
            aug_expanded = aug_pairs * n_copies
            print(f"  B3 manual: {len(aug_pairs)} × {n_copies} = {len(aug_expanded)} PP examples")
            train_pairs = train_pairs + aug_expanded

        # Auto PP-on-subject
        if variant in ("B3_auto", "B4", "B4_turbo", "B5", "B5_lite", "B6", "B7a", "B7c"):
            pp_aug = generate_subj_pp_augmentations(train_pairs)
            print(f"  PP-on-subject auto: {len(pp_aug)} examples")
            for i, (a, l, _) in enumerate(pp_aug[:3]):
                print(f"    pp{i}: {a}  →  {l[:80]}...")
            # B4_turbo: oversample PP examples ×2
            if variant == "B4_turbo":
                pp_aug = pp_aug * 2
                print(f"  PP oversampled ×2: {len(pp_aug)} examples")
            train_pairs = train_pairs + pp_aug

        # Active → Passive (B5/B5_lite)
        if variant in ("B5", "B5_lite"):
            pass_aug = generate_active_to_passive(train_pairs)
            print(f"  Active→Passive: {len(pass_aug)} examples")
            for i, (a, l, _) in enumerate(pass_aug[:3]):
                print(f"    pas{i}: {a}  →  {l[:80]}...")
            train_pairs = train_pairs + pass_aug

        # Dative swap — disabled: 40/41 verbs already in both forms in train,
        # gen only tests 2 novel verbs (teleport, ship). 1177 dative examples
        # are pure noise that dilutes the PP signal.
        # Kept for B5_dat if needed for ablation.
        if variant == "B5_dat":
            dat_aug = generate_dative_swap(train_pairs)
            print(f"  Dative swap: {len(dat_aug)} examples (ablation only)")
            train_pairs = train_pairs + dat_aug

    # B4_turbo: add 10 PP depth-3 + 5 RC_modif_subj manual examples
    if variant == "B4_turbo":
        turbo_path = os.path.join(DATA_DIR, "aug_turbo.tsv")
        if os.path.exists(turbo_path):
            turbo = parse_cogs_tsv(turbo_path)
            pp3 = [x for x in turbo if x[2] == "aug_pp3"]
            rc = [x for x in turbo if x[2] == "aug_rc_subj"]
            print(f"  Turbo: {len(pp3)} PP-depth3 + {len(rc)} RC-modif-subj")
            for i, (a, l, _) in enumerate(turbo[:2]):
                print(f"    t{i}: {a}")
            train_pairs = train_pairs + turbo
        else:
            print(f"  WARNING: {turbo_path} not found, skipping turbo examples")

    # Noun positional augmentation (B6): swap subj↔obj for gap nouns
    if variant == "B6":
        np_aug = generate_noun_position_augmentations(train_pairs)
        print(f"  Noun position: {len(np_aug)} examples")
        for i, (a, l, _) in enumerate(np_aug[:3]):
            print(f"    np{i}: {a}  →  {l[:80]}...")
        train_pairs = train_pairs + np_aug

    # B7a: PP recursion extension only (+1 depth level)
    # CP recursion disabled: it corrupts Q_subj_active/passive categories
    if variant in ("B7a", "B7c"):
        pp_rec = generate_pp_recursion_extension(train_pairs, max_aug=500)
        print(f"  PP recursion +1: {len(pp_rec)} examples")
        for i, (a, l, _) in enumerate(pp_rec[:2]):
            print(f"    ppr{i}: {a}")
            print(f"           {l[:100]}...")
        train_pairs = train_pairs + pp_rec

    print(f"train: {len(train_pairs)}  dev: {len(dev_pairs)}  gen: {len(gen_pairs)}")

    in_w2i, out_w2i = build_vocabs(train_pairs + dev_pairs + gen_pairs)
    print(f"Input vocab: {len(in_w2i)}  Output vocab: {len(out_w2i)}")

    # Build permutation classes depending on variant
    perm_classes = []
    perm_log = {"variant": variant}
    n_proper_names = 0

    if variant in ("B1", "B4", "B4_turbo", "B5", "B5_lite", "B6", "B7a", "B7c"):
        perm_pairs_io = detect_proper_names(train_pairs, in_w2i, out_w2i)
        perm_ids = [(p[0], p[1]) for p in perm_pairs_io]
        perm_classes = [perm_ids]
        n_proper_names = len(perm_pairs_io)
        print(f"  Proper name permutation: {n_proper_names} names")
        perm_log["n_classes"] = 1
        perm_log["classes"] = [{"name": "proper_names", "n": len(perm_ids),
                                "tokens": [p[2] for p in perm_pairs_io]}]

    if variant in ("B5", "B5_lite"):
        # Add common noun permutation (within detect_permutable sub-classes)
        ext_classes, _, skipped = build_extended_perm_classes(
            train_pairs, in_w2i, out_w2i, exclude_verbs=True)
        i2in = {v: k for k, v in in_w2i.items()}
        noun_classes = [c for c in ext_classes
                        if i2in.get(c[0][0], "X")[0].islower() and len(c) > 5]
        for cls in noun_classes:
            perm_classes.append(cls)
            names = [i2in[p[0]] for p in cls]
            print(f"  Common noun class ({len(cls)}): {names[:8]}...")
        perm_log["n_classes"] = len(perm_classes)
        if skipped:
            print(f"  Excluded verb classes: {len(skipped)}")

    elif variant in ("B2", "B2b", "B2c"):
        no_verbs = (variant in ("B2b", "B2c"))
        ext_classes, raw_clusters, skipped = build_extended_perm_classes(
            train_pairs, in_w2i, out_w2i, exclude_verbs=no_verbs)
        if skipped:
            print(f"  Excluded {len(skipped)} verb class(es): "
                  f"{[c[:5] for c in skipped]}")
        if variant == "B2c":
            # Only keep common noun classes (lowercase), drop proper names + prepositions
            i2in = {v: k for k, v in in_w2i.items()}
            kept = []
            for cls in ext_classes:
                first_tok = i2in.get(cls[0][0], "")
                if first_tok and first_tok[0].islower() and len(cls) > 5:
                    kept.append(cls)
            print(f"  B2c: keeping {len(kept)} common noun classes, "
                  f"dropped {len(ext_classes) - len(kept)}")
            ext_classes = kept
        perm_classes = ext_classes
        i2in = {v: k for k, v in in_w2i.items()}
        print(f"B2: {len(ext_classes)} classes from detect_permutable")
        cls_log = []
        for ci, cls in enumerate(ext_classes):
            names = [i2in.get(pair[0], "?") for pair in cls]
            preview = names[:15]
            if len(names) > 15: preview.append(f"...+{len(names)-15}")
            print(f"  class {ci+1} ({len(cls)}): {preview}")
            cls_log.append({"n": len(cls), "tokens": names})
        perm_log["n_classes"] = len(ext_classes)
        perm_log["classes"] = cls_log
    else:
        print("B0: no permutation")
        perm_log["n_classes"] = 0

    with open(os.path.join(run_dir, "permutation_classes.json"), "w") as f:
        json.dump(perm_log, f, indent=2)

    tr_ds = COGSDataset(train_pairs, in_w2i, out_w2i,
                        perm_classes=perm_classes if variant in ("B1", "B2", "B2b", "B2c", "B4", "B4_turbo", "B5", "B5_lite", "B6", "B7a", "B7c") else None)
    dv_ds = COGSDataset(dev_pairs,   in_w2i, out_w2i)
    ge_ds = COGSDataset(gen_pairs,   in_w2i, out_w2i)

    eval_bs = args.eval_batch_size
    tr_ld = DataLoader(tr_ds, args.batch_size, shuffle=True,  collate_fn=collate, num_workers=0, pin_memory=True)
    dv_ld = DataLoader(dv_ds, eval_bs, shuffle=False, collate_fn=collate, num_workers=0, pin_memory=True)
    ge_ld = DataLoader(ge_ds, eval_bs, shuffle=False, collate_fn=collate, num_workers=0, pin_memory=True)

    # Clip very long targets from vocab-side max_out (safety)
    mx_in = max(MAX_IN, max(len(p[0].split()) for p in train_pairs + dev_pairs + gen_pairs) + 2)
    mx_out = max(MAX_OUT, max(len(p[1].split()) for p in train_pairs + dev_pairs + gen_pairs) + 4)
    use_copy = getattr(args, "copy", False)
    use_tags = getattr(args, "tags", False)
    if use_tags:
        in_to_out = build_in_to_out_map(in_w2i, out_w2i)
        n_shared = (in_to_out >= 0).sum().item()
        token_cats = build_token_categories(in_w2i)
        print(f"Copy+Tags mechanism ON: {n_shared}/{len(in_w2i)} input tokens mapped")
        model = TransformerSeq2SeqCopyTags(len(in_w2i), len(out_w2i),
                                           d_model=args.d_model,
                                           max_in=mx_in, max_out=mx_out,
                                           in_to_out_map=in_to_out,
                                           token_categories=token_cats).to(device)
        use_copy = True     # tags implies copy
    elif use_copy:
        in_to_out = build_in_to_out_map(in_w2i, out_w2i)
        n_shared = (in_to_out >= 0).sum().item()
        print(f"Copy mechanism ON: {n_shared}/{len(in_w2i)} input tokens mapped to output vocab")
        model = TransformerSeq2SeqWithCopy(len(in_w2i), len(out_w2i),
                                           d_model=args.d_model,
                                           max_in=mx_in, max_out=mx_out,
                                           in_to_out_map=in_to_out).to(device)
    else:
        model = TransformerSeq2Seq(len(in_w2i), len(out_w2i),
                                   d_model=args.d_model,
                                   max_in=mx_in, max_out=mx_out).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params/1e6:.2f}M  max_in={model.max_in}  max_out={model.max_out}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    total_steps = args.epochs * len(tr_ld)
    warmup = min(1000, total_steps // 10)
    def lr_at(step):
        if step < warmup: return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    metrics_log = []
    best_dev_exact = -1.0
    patience = 0
    epoch_times = []
    last_gen_m = {"exact": 0.0, "tok_acc": 0.0, "n": 0}

    for ep in range(args.epochs):
        t_ep = time.time()
        model.train()
        tr_loss = 0.0; n_batches = 0

        # Scheduled sampling (copy only): TFR = 1.0 for ep<20, then linear 1.0→0.5 up to ep=60
        if use_copy and ep >= 20:
            progress = min(max((ep - 20) / max(args.epochs - 20, 1), 0.0), 1.0)
            tfr = 1.0 - 0.5 * progress
        else:
            tfr = 1.0

        for src, src_mask, tgt, _, _ in tr_ld:
            src = src.to(device); src_mask = src_mask.to(device); tgt = tgt.to(device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            opt.zero_grad(set_to_none=True)

            # Scheduled sampling: replace some input positions with previous predictions
            if tfr < 1.0:
                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                        logits_1, _ = model(src, src_mask, tgt_in)
                    preds = logits_1.argmax(-1)                  # (B, T)
                B, T = tgt_in.shape
                repl_mask = torch.rand(B, T, device=device) > tfr
                repl_mask[:, 0] = False                          # never replace BOS
                preds_shift = torch.cat([tgt_in[:, :1], preds[:, :-1]], dim=1)
                tgt_in_used = torch.where(repl_mask, preds_shift, tgt_in)
            else:
                tgt_in_used = tgt_in

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits, info = model(src, src_mask, tgt_in_used)
                V = logits.size(-1)
                if info.get("uses_copy"):
                    loss = F.nll_loss(logits.reshape(-1, V), tgt_out.reshape(-1),
                                      ignore_index=0)
                else:
                    loss = F.cross_entropy(logits.reshape(-1, V), tgt_out.reshape(-1),
                                           ignore_index=0, label_smoothing=0.1)
                # Auxiliary role tagger loss (Copy+Tags only)
                if info.get("uses_tags"):
                    rl = info["role_logits"]                       # (B, T_src, N)
                    rg = info["roles_gt"]                          # (B, T_src)
                    role_mask = src_mask.reshape(-1)
                    role_ce = F.cross_entropy(rl.reshape(-1, N_ROLES),
                                              rg.reshape(-1),
                                              reduction="none")
                    role_loss = (role_ce * role_mask.float()).sum() / role_mask.float().sum().clamp(min=1)
                    loss = loss + 0.3 * role_loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tr_loss += loss.item(); n_batches += 1

        dev_m = evaluate_tf(model, dv_ld, device)
        gen_m = evaluate_tf(model, ge_ld, device)

        is_best = dev_m["exact"] > best_dev_exact
        gen_log = {"exact": gen_m["exact"], "tok_acc": gen_m["tok_acc"], "n": gen_m["n"]}
        if "by_cat" in gen_m:
            gen_log["by_cat"] = gen_m["by_cat"]
        entry = {
            "epoch": ep,
            "train_loss": tr_loss / max(n_batches, 1),
            "dev": {"exact": dev_m["exact"], "tok_acc": dev_m["tok_acc"], "n": dev_m["n"]},
            "gen": gen_log,
            "lr": sched.get_last_lr()[0],
        }
        metrics_log.append(entry)
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics_log, f, indent=1)

        if is_best:
            best_dev_exact = dev_m["exact"]
            patience = 0
        else:
            patience += 1

        dt = time.time() - t_ep
        epoch_times.append(dt)
        avg_ep = sum(epoch_times) / len(epoch_times)
        remaining = min(args.epochs - ep - 1, max(args.patience - patience, 0))
        eta_s = remaining * avg_ep
        eta_m, eta_sec = int(eta_s // 60), int(eta_s % 60)
        mark = "★" if is_best else " "
        tfr_str = f" tfr={tfr:.2f}" if use_copy and tfr < 1.0 else ""
        print(f"E{ep:03d} {mark} loss={tr_loss/max(n_batches,1):.4f} "
              f"dev_ex={dev_m['exact']:5.2f}% gen_ex={gen_m['exact']:5.2f}% "
              f"tok_g={gen_m['tok_acc']:5.2f}% lr={sched.get_last_lr()[0]:.2e}{tfr_str} "
              f"pat={patience} | {dt:.1f}s ETA {eta_m}m{eta_sec:02d}s")

        if patience >= args.patience:
            print(f"Early stopping at epoch {ep} (patience {args.patience}).")
            break

    # Final greedy eval (autoregressive, slower but accurate)
    print("Running final greedy eval on dev + gen...")
    greedy_dev = evaluate(model, dv_ld, out_w2i, device)
    greedy_gen = evaluate(model, ge_ld, out_w2i, device)
    print(f"  Greedy dev_ex={greedy_dev['exact']:.2f}%  gen_ex={greedy_gen['exact']:.2f}%")
    if "by_cat" in greedy_gen:
        print("  Per-category gen (greedy):")
        for cat, pct in sorted(greedy_gen["by_cat"].items(), key=lambda x: -x[1]):
            print(f"    {cat:50s} {pct:6.2f}%")

    summary = {
        "variant": variant, "seed": seed, "args": vars(args),
        "n_params": n_params, "epochs_run": ep + 1,
        "final_train_loss": metrics_log[-1]["train_loss"],
        "final_dev_exact_tf":  metrics_log[-1]["dev"]["exact"],
        "final_gen_exact_tf":  metrics_log[-1]["gen"]["exact"],
        "final_dev_exact_greedy": greedy_dev["exact"],
        "final_gen_exact_greedy": greedy_gen["exact"],
        "greedy_gen_by_cat": greedy_gen.get("by_cat", {}),
        "best_dev_exact":   best_dev_exact,
        "best_gen_exact_tf": max(m["gen"]["exact"] for m in metrics_log),
        "n_proper_names":   n_proper_names,
        "run_dir": run_dir,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    torch.save({"state_dict": model.state_dict(), "in_w2i": in_w2i,
                "out_w2i": out_w2i, "variant": variant,
                "d_model": args.d_model,
                "max_in": model.max_in, "max_out": model.max_out,
                "use_copy": use_copy, "use_tags": use_tags,
               }, os.path.join(run_dir, "checkpoint.pt"))
    print(f"\nFinished {bench_name} {variant}/s{seed}: "
          f"best_gen_tf={summary['best_gen_exact_tf']:.2f}%  "
          f"final_gen_greedy={summary['final_gen_exact_greedy']:.2f}%")
    return summary


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="COGS compositional generalization")
    p.add_argument("--variant", type=str,
                   choices=["B0", "B1", "B2", "B2b", "B2c", "B3", "B3_auto", "B4",
                            "B4_turbo", "B5", "B5_lite", "B6", "B7a", "B7c"],
                   required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=16,
                   help="Batch size for eval (smaller to avoid OOM on long outputs)")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--runs-dir", type=str, default=None,
                   help="Override output directory for runs")
    p.add_argument("--copy", action="store_true",
                   help="Enable copy mechanism (pointer network) in decoder")
    p.add_argument("--tags", action="store_true",
                   help="Enable structural role tagger (implies --copy)")
    args = p.parse_args()
    train_one_run(args.variant, args.seed, args)
