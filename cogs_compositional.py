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


def build_unified_vocab(pairs):
    """Merge input and output tokens into a single vocabulary.
    Needed for bidirectional training — same embedding space for both directions."""
    w2i = {PAD: 0, BOS: 1, EOS: 2}
    for inp, lf, _ in pairs:
        for t in inp.split():
            if t not in w2i: w2i[t] = len(w2i)
        for t in lf.split():
            if t not in w2i: w2i[t] = len(w2i)
    return w2i


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


def generate_wh_passive_examples(train_pairs, n=None, seed=42):
    """Convert simple declarative passives with by-phrase into wh-questions:
      'The X was V-ed by Y .'  →  'What was V-ed by Y ?'
      with LF:   V.theme(x_2, ?) AND V.agent(x_2, <agent_filler>)

    Also handles the wh-on-agent variant:
      'A X was V-ed by Y .'  →  'Who V-ed a X ?'  (skipped here for simplicity
                                 since the direction Who-V is already covered
                                 by Q_subj_active in SLOG train)

    The wh-trace is the token `?` in COGS/SLOG LF syntax. Positions in the
    produced input:  What(0) was(1) V-ed(2) by(3) <agent_np>(4..)  ?(last).
    """
    rng = random.Random(seed)
    results = []
    for inp, lf, _ in train_pairs:
        toks = inp.split()
        if not toks or toks[-1] != ".":
            continue
        if "was" not in toks or "by" not in toks:
            continue
        # Exclude datives / ccomp / nmod for a clean conversion
        if ". recipient" in lf or ". ccomp" in lf or ". nmod" in lf \
                or ". xcomp" in lf:
            continue

        try:
            was_idx = toks.index("was")
            by_idx = toks.index("by")
        except ValueError:
            continue
        if by_idx != was_idx + 2:
            continue  # skip "was Xed by" with extra tokens (to Y, in Z, …)

        pp_word = toks[was_idx + 1]
        agent_np = toks[by_idx + 1:-1]   # between "by" and "."
        if not agent_np:
            continue

        # Identify verb lemma from LF
        lf_toks = lf.split()
        verb_lemma = None
        for j in range(len(lf_toks) - 2):
            if lf_toks[j + 1] == "." and lf_toks[j + 2] in ("agent", "theme"):
                verb_lemma = lf_toks[j]
                break
        if verb_lemma is None:
            continue

        # Classify agent NP (proper, indef common, def common)
        if len(agent_np) == 1 and agent_np[0] and agent_np[0][0].isupper() \
                and agent_np[0] not in ("The", "A") and agent_np[0].isalpha():
            agent_kind = "proper"
            agent_noun_word = agent_np[0]
        elif len(agent_np) == 2 and agent_np[0] in ("a", "A", "the", "The") \
                and agent_np[1].isalpha() and agent_np[1][0].islower():
            agent_kind = "the" if agent_np[0].lower() == "the" else "a"
            agent_noun_word = agent_np[1]
        else:
            continue

        # Build new tokens: What / was / pp_word / by / <agent_np> / ?
        new_tokens = ["What", "was", pp_word, "by"] + list(agent_np) + ["?"]
        new_inp = " ".join(new_tokens)

        # Positions in new_tokens:
        #   What=0, was=1, pp_word=2, by=3, agent_np starts at 4
        verb_event_pos = 2
        # Agent filler position (for common nouns): determiner at 4, noun at 5.
        if agent_kind == "proper":
            agent_filler = agent_np[0]            # proper name constant
            # Build LF directly
            new_lf = (f"{verb_lemma} . theme ( x _ {verb_event_pos} , ? ) "
                      f"AND {verb_lemma} . agent ( x _ {verb_event_pos} , {agent_filler} )")
        else:
            agent_noun_pos = 5                     # noun token position
            if agent_kind == "the":
                # Definite → presupposition at the start
                new_lf = (f"* {agent_noun_word} ( x _ {agent_noun_pos} ) ; "
                          f"{verb_lemma} . theme ( x _ {verb_event_pos} , ? ) "
                          f"AND {verb_lemma} . agent ( x _ {verb_event_pos} , "
                          f"x _ {agent_noun_pos} )")
            else:
                # Indefinite → noun predicate in body
                new_lf = (f"{verb_lemma} . theme ( x _ {verb_event_pos} , ? ) "
                          f"AND {verb_lemma} . agent ( x _ {verb_event_pos} , "
                          f"x _ {agent_noun_pos} ) "
                          f"AND {agent_noun_word} ( x _ {agent_noun_pos} )")
        results.append((new_inp, new_lf, "aug_wh_passive"))

    # Dedupe
    seen = set()
    unique = []
    for item in results:
        if item[0] not in seen:
            seen.add(item[0])
            unique.append(item)
    rng.shuffle(unique)
    if n is not None and n > 0 and len(unique) > n:
        unique = unique[:n]
    return unique


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


# ══════════════════════════════════════════════════════════════
# Cross-voice augmentation (Innovation: --cross-voice)
# ══════════════════════════════════════════════════════════════
def _classify_np(np_toks):
    """Return (type, noun_idx_in_np, is_definite). type in {proper, common, unknown}."""
    DETS_LOW = {"a", "the"}
    DETS_CAP = {"A", "The"}
    if len(np_toks) == 1:
        t = np_toks[0]
        if t and t[0].isupper() and t not in DETS_CAP:
            return ("proper", 0, False)
        return ("unknown", -1, False)
    if len(np_toks) == 2:
        d, n = np_toks[0], np_toks[1]
        if d in DETS_LOW or d in DETS_CAP:
            is_def = d.lower() == "the"
            # The noun must be lowercase common noun
            if n and n[0].islower() and n.isalpha():
                return ("common", 1, is_def)
    return ("unknown", -1, False)


def build_verb_form_maps(train_pairs):
    """Extract verb_lemma → past_tense (from actives) and verb_lemma → past_participle
    (from passives) by scanning train. Returns (pt_map, pp_map)."""
    pt_map, pp_map = {}, {}
    for inp, lf, _ in train_pairs:
        toks = inp.split()
        lf_toks = lf.split()
        # Find verb lemma + event position from LF (head of .agent)
        verb_lemma, verb_pos = None, None
        for j in range(len(lf_toks) - 6):
            if lf_toks[j + 1] == "." and lf_toks[j + 2] == "agent" \
                    and lf_toks[j + 3] == "(" and lf_toks[j + 4] == "x" \
                    and lf_toks[j + 5] == "_" and lf_toks[j + 6].isdigit():
                verb_lemma = lf_toks[j]
                verb_pos = int(lf_toks[j + 6])
                break
        if verb_lemma is None or verb_pos is None or verb_pos >= len(toks):
            continue
        verb_word = toks[verb_pos]
        if "was" in toks and "by" in toks:
            pp_map.setdefault(verb_lemma, verb_word)
        else:
            pt_map.setdefault(verb_lemma, verb_word)
    return pt_map, pp_map


def _simple_active_transitive(inp, lf):
    """Return (verb_lemma, verb_pos, subj_toks, obj_toks) for simple active transitive,
    or None. Rejects examples with nmod/ccomp/recipient."""
    toks = inp.split()
    if toks[-1] != ".":
        return None
    if "was" in toks or "by" in toks:
        return None
    if ". nmod" in lf or ". ccomp" in lf or ". xcomp" in lf \
            or ". recipient" in lf:
        return None
    # Find verb lemma + pos
    lf_toks = lf.split()
    verb_lemma, verb_pos = None, None
    for j in range(len(lf_toks) - 6):
        if lf_toks[j + 1] == "." and lf_toks[j + 2] == "agent" \
                and lf_toks[j + 3] == "(" and lf_toks[j + 4] == "x" \
                and lf_toks[j + 5] == "_" and lf_toks[j + 6].isdigit():
            verb_lemma = lf_toks[j]
            verb_pos = int(lf_toks[j + 6])
            break
    if verb_pos is None or verb_pos >= len(toks) - 1:
        return None
    # Must have both .agent and .theme (transitive)
    if ". theme (" not in lf:
        return None
    subj_toks = toks[0:verb_pos]
    obj_toks = toks[verb_pos + 1:-1]  # exclude final "."
    if not subj_toks or not obj_toks:
        return None
    return verb_lemma, verb_pos, subj_toks, obj_toks


def _simple_passive(inp, lf):
    """Return (verb_lemma, subj_toks_passive, agent_toks_after_by) for simple passive,
    or None. Rejects recipients/nmod/ccomp."""
    toks = inp.split()
    if toks[-1] != ".":
        return None
    if "was" not in toks or "by" not in toks:
        return None
    if ". nmod" in lf or ". ccomp" in lf or ". xcomp" in lf \
            or ". recipient" in lf:
        return None
    try:
        was_idx = toks.index("was")
        by_idx = toks.index("by")
    except ValueError:
        return None
    if by_idx <= was_idx + 1:
        return None
    # Verb lemma from LF (head of .agent)
    lf_toks = lf.split()
    verb_lemma = None
    for j in range(len(lf_toks) - 2):
        if lf_toks[j + 1] == "." and lf_toks[j + 2] == "agent":
            verb_lemma = lf_toks[j]
            break
    if verb_lemma is None:
        return None
    subj_toks = toks[0:was_idx]              # passive subject (theme)
    agent_toks = toks[by_idx + 1:-1]         # by-phrase NP (agent)
    if not subj_toks or not agent_toks:
        return None
    return verb_lemma, subj_toks, agent_toks


def _build_lf_for_transitive(verb_lemma, verb_pos, subj_toks, subj_noun_pos,
                              obj_toks, obj_noun_pos, active=True):
    """Build an LF for a simple transitive sentence.

    Order for active: [subj_noun if common indef] AND v.agent AND v.theme [AND obj_noun if common indef]
    Order for passive: [theme_noun if common indef] AND v.theme AND v.agent [AND agent_noun if common indef]
    Definite NPs become `* noun ( x _ N )` presuppositions prepended with `;`.
    """
    subj_type, subj_noun_idx, subj_def = _classify_np(subj_toks)
    obj_type, obj_noun_idx, obj_def = _classify_np(obj_toks)
    if subj_type == "unknown" or obj_type == "unknown":
        return None

    subj_term = (subj_toks[subj_noun_idx] if subj_type == "proper"
                 else f"x _ {subj_noun_pos}")
    obj_term = (obj_toks[obj_noun_idx] if obj_type == "proper"
                else f"x _ {obj_noun_pos}")

    presup, body = [], []
    def _pred(noun_low, pos, is_def):
        if is_def:
            presup.append(f"* {noun_low} ( x _ {pos} )")
        else:
            body.append(f"{noun_low} ( x _ {pos} )")

    # ORDER depends on active vs passive
    if active:
        # subj_noun (if common) — v.agent — v.theme — obj_noun (if common)
        if subj_type == "common":
            _pred(subj_toks[subj_noun_idx].lower(), subj_noun_pos, subj_def)
        body.append(f"{verb_lemma} . agent ( x _ {verb_pos} , {subj_term} )")
        body.append(f"{verb_lemma} . theme ( x _ {verb_pos} , {obj_term} )")
        if obj_type == "common":
            _pred(obj_toks[obj_noun_idx].lower(), obj_noun_pos, obj_def)
    else:
        # passive: theme_noun (obj here acts as theme=subject of passive) — v.theme — v.agent — agent_noun
        if obj_type == "common":
            _pred(obj_toks[obj_noun_idx].lower(), obj_noun_pos, obj_def)
        body.append(f"{verb_lemma} . theme ( x _ {verb_pos} , {obj_term} )")
        body.append(f"{verb_lemma} . agent ( x _ {verb_pos} , {subj_term} )")
        if subj_type == "common":
            _pred(subj_toks[subj_noun_idx].lower(), subj_noun_pos, subj_def)

    body_str = " AND ".join(body)
    if presup:
        return " ; ".join(presup) + " ; " + body_str
    return body_str


def _capitalize_first(toks):
    """Capitalize first word of NP if it's a lowercase determiner (for sentence start)."""
    out = list(toks)
    if out and out[0] in ("a", "the"):
        out[0] = out[0].capitalize()
    return out


def _lowercase_det(toks):
    """Lowercase a leading capitalized determiner when NP moves mid-sentence."""
    out = list(toks)
    if out and out[0] in ("A", "The"):
        out[0] = out[0].lower()
    return out


def generate_passive_from_active(inp, lf, pp_map):
    """Active → passive conversion. Returns (new_inp, new_lf) or None."""
    res = _simple_active_transitive(inp, lf)
    if res is None:
        return None
    verb_lemma, _, subj_toks, obj_toks = res
    if verb_lemma not in pp_map:
        return None
    pp_word = pp_map[verb_lemma]

    # Build passive: obj (as passive subject) + "was <pp>" + "by" + subj + "."
    new_obj_toks = _capitalize_first(obj_toks)   # used as passive subject at sentence start
    new_subj_toks = _lowercase_det(subj_toks)    # used after "by"
    passive_tokens = new_obj_toks + ["was", pp_word, "by"] + new_subj_toks + ["."]
    new_inp = " ".join(passive_tokens)

    # Compute new positions
    obj_len = len(new_obj_toks)
    new_theme_noun_pos = 0 + 1  # for "<det> <noun>": noun at idx 1; for proper: idx 0
    obj_type, obj_noun_idx, _ = _classify_np(new_obj_toks)
    if obj_type == "unknown":
        return None
    new_theme_noun_pos = obj_noun_idx  # 0 or 1
    new_verb_pos = obj_len + 1          # after "was"
    new_agent_noun_pos = obj_len + 3    # position after "was pp by", then + noun_idx
    subj_type, subj_noun_idx, _ = _classify_np(new_subj_toks)
    if subj_type == "unknown":
        return None
    new_agent_noun_pos = obj_len + 3 + subj_noun_idx

    new_lf = _build_lf_for_transitive(
        verb_lemma, new_verb_pos,
        new_subj_toks, new_agent_noun_pos,
        new_obj_toks, new_theme_noun_pos,
        active=False)
    if new_lf is None:
        return None
    return new_inp, new_lf


def generate_active_from_passive(inp, lf, pt_map):
    """Passive → active conversion. Returns (new_inp, new_lf) or None."""
    res = _simple_passive(inp, lf)
    if res is None:
        return None
    verb_lemma, passive_subj_toks, passive_agent_toks = res
    if verb_lemma not in pt_map:
        return None
    past_word = pt_map[verb_lemma]

    # Active: agent-NP (passive's by-phrase) + past + theme-NP (passive's subject) + "."
    new_subj_toks = _capitalize_first(passive_agent_toks)
    new_obj_toks = _lowercase_det(passive_subj_toks)
    active_tokens = new_subj_toks + [past_word] + new_obj_toks + ["."]
    new_inp = " ".join(active_tokens)

    # Positions
    subj_type, subj_noun_idx, _ = _classify_np(new_subj_toks)
    obj_type, obj_noun_idx, _ = _classify_np(new_obj_toks)
    if subj_type == "unknown" or obj_type == "unknown":
        return None
    new_subj_noun_pos = subj_noun_idx
    new_verb_pos = len(new_subj_toks)
    new_obj_noun_pos = len(new_subj_toks) + 1 + obj_noun_idx

    new_lf = _build_lf_for_transitive(
        verb_lemma, new_verb_pos,
        new_subj_toks, new_subj_noun_pos,
        new_obj_toks, new_obj_noun_pos,
        active=True)
    if new_lf is None:
        return None
    return new_inp, new_lf


def generate_cp_recursion_extension(train_pairs, max_aug=500):
    """Extend CP chains by 1 level. Wrap depth-N CP examples in an outer
    '<NAME> <V_matrix> that …' clause to produce depth-(N+1) examples.

    Builds a pool of (matrix_verb_lemma, past_tense) by scanning train examples
    that contain `. ccomp `. Uses only proper names as new outer subjects (3 new
    tokens prepended: `NAME V that`). All `x _ N` indices in the old LF shift
    by +3. Two new predicates are prepended: `v.agent(x_1, NAME)` and
    `v.ccomp(x_1, x_<old_matrix_pos+3>)`.

    Returns a list of (inp, lf, "aug_cp_recursion") tuples, deduplicated.
    """
    cp_verbs = {}  # lemma -> past_tense_word
    proper_subjects = set()
    for inp, lf, _ in train_pairs:
        if ". ccomp " not in lf:
            continue
        toks = inp.split()
        lf_toks = lf.split()
        matrix_lemma = None
        matrix_pos = None
        for j in range(len(lf_toks) - 6):
            if lf_toks[j + 1] == "." and lf_toks[j + 2] == "ccomp" \
                    and lf_toks[j + 3] == "(" and lf_toks[j + 4] == "x" \
                    and lf_toks[j + 5] == "_" and lf_toks[j + 6].isdigit():
                matrix_lemma = lf_toks[j]
                matrix_pos = int(lf_toks[j + 6])
                break
        if matrix_lemma is None or matrix_pos is None or matrix_pos >= len(toks):
            continue
        cp_verbs.setdefault(matrix_lemma, toks[matrix_pos])
        if toks and toks[0] and toks[0][0].isupper() and toks[0].isalpha() \
                and toks[0] not in ("The", "A"):
            proper_subjects.add(toks[0])

    if not cp_verbs or not proper_subjects:
        return []
    cp_list = list(cp_verbs.items())
    proper_list = list(proper_subjects)

    aug = []
    for inp, lf, _ in train_pairs:
        if ". ccomp " not in lf:
            continue
        toks = inp.split()
        if not toks or toks[-1] != ".":
            continue
        inner_tokens = toks[:-1]
        new_name = random.choice(proper_list)
        new_lemma, new_past = random.choice(cp_list)
        if inner_tokens and inner_tokens[0] == new_name:
            continue  # avoid "NAME V that NAME …"

        # Lowercase leading determiner in inner clause (no longer at sentence start)
        adj_inner = list(inner_tokens)
        if adj_inner and adj_inner[0] in ("The", "A"):
            adj_inner[0] = adj_inner[0].lower()
        new_tokens = [new_name, new_past, "that"] + adj_inner + ["."]
        new_inp = " ".join(new_tokens)

        # Shift all `x _ N` in old LF by +3
        lf_toks = lf.split()
        shifted_toks = []
        j = 0
        while j < len(lf_toks):
            if j + 2 < len(lf_toks) and lf_toks[j] == "x" and lf_toks[j + 1] == "_" \
                    and lf_toks[j + 2].isdigit():
                shifted_toks.extend(["x", "_", str(int(lf_toks[j + 2]) + 3)])
                j += 3
            else:
                shifted_toks.append(lf_toks[j])
                j += 1

        # Locate OLD matrix event position for ccomp link
        old_matrix_pos = None
        for jj in range(len(lf_toks) - 6):
            if lf_toks[jj + 1] == "." and lf_toks[jj + 2] == "ccomp" \
                    and lf_toks[jj + 3] == "(" and lf_toks[jj + 4] == "x" \
                    and lf_toks[jj + 5] == "_" and lf_toks[jj + 6].isdigit():
                old_matrix_pos = int(lf_toks[jj + 6])
                break
        if old_matrix_pos is None:
            continue
        shifted_matrix_pos = old_matrix_pos + 3

        prefix = (f"{new_lemma} . agent ( x _ 1 , {new_name} ) "
                  f"AND {new_lemma} . ccomp ( x _ 1 , x _ {shifted_matrix_pos} )")
        shifted_lf = " ".join(shifted_toks)
        if " ; " in shifted_lf:
            parts = shifted_lf.split(" ; ")
            presups = parts[:-1]
            body = parts[-1]
            new_lf = " ; ".join(presups) + " ; " + prefix + " AND " + body
        else:
            new_lf = prefix + " AND " + shifted_lf
        aug.append((new_inp, new_lf, "aug_cp_recursion"))

    seen = set()
    deduped = []
    for item in aug:
        if item[0] not in seen:
            seen.add(item[0])
            deduped.append(item)
    random.shuffle(deduped)
    return deduped[:max_aug]


# Verbs that need to be present in the permutation pool with both roles
# (agent AND theme) even though the raw extractor may miss them:
#   - either they appear in train only as .theme-only (no .agent extractable)
#   - or they appear in train only transitively, but the gen test uses them
#     as unaccusatives (wh-questions "Who shortened ?" → shorten.theme(x_1, ?))
# Force-added with regular -ed morphology. A no-op if already in both maps
# after normal extraction + fallback. Extend from diagnose_a2p_coverage.py.
# COGS baseline:
#   - squeeze: passives without by-phrase in train
#   - shatter: unaccusative only in train
# SLOG adds verbs tested as unaccusatives in Q_subj_active even though
# transitive in train (shorten, float), plus other alternating-unaccusatives
# seen in the gen diagnostic:
FORCED_REGULAR_VERBS_FOR_POOL = [
    "squeeze", "shatter",                    # COGS originals
    "shorten", "float",                      # SLOG Q_subj_active unaccusatives
    "freeze", "melt", "break", "collapse",   # common alternators (no-op if already covered)
    "improve", "bounce", "roll",             # optional safety net
]

# SLOG-derived list of unaccusative-class verbs to OPTIONALLY remove from the
# permutation pool (transfer test, see PROMPT_TRANSFER_COGS.md). The 10 verbs
# below are exactly those that the SLOG sweep filtered out and that improved
# SLOG gen greedy by ≈+1.6 points. The COGS run does an extra vocab check
# at runtime: a verb is only removed if its lemma is in the OUTPUT vocab AND
# it is currently in the freshly-built permutation pool — otherwise it's
# silently skipped and reported in pool_used.txt.
SLOG_UNACCUSATIVE_FILTERED = [
    "break", "burn", "collapse", "disintegrate", "float",
    "freeze", "grow", "roll", "shatter", "shorten",
]


def _regular_past_form(lemma):
    """Regular past/past-participle from lemma. e-final → +d, else +ed."""
    if lemma.endswith("e"):
        return lemma + "d"
    return lemma + "ed"


def build_verb_perm_pool(train_pairs, in_w2i, out_w2i, morphology_fallback=True,
                          filter_unaccusative=False):
    """Return a list of (past_in_id, pp_in_id, lemma_out_id) triples for every
    transitive verb that has both a past-tense and a past-participle form available
    (with optional morphological fallback for regular verbs).

    This pool is passed to COGSDataset for A4-style verb permutation: each batch,
    a random bijection on the pool is applied to both src (inflected forms) and
    tgt (lemmas). Analog of SCAN's A4_permute on {walk, run, look, jump}.

    If `filter_unaccusative=True`, also remove the SLOG-derived list of
    unaccusative verbs from the pool (see SLOG_UNACCUSATIVE_FILTERED). Returns
    extra info under stats["unaccusative_*"] for pool_used.txt.
    """
    pt_map, pp_map = build_verb_form_maps(train_pairs)
    pt_map = dict(pt_map)
    pp_map = dict(pp_map)
    if morphology_fallback:
        for lemma, past in pt_map.items():
            if lemma not in pp_map and past.endswith("ed"):
                pp_map[lemma] = past

    # Force-add verbs that appear in train only as passives without by-phrase.
    # For these, build_verb_form_maps leaves pt_map AND pp_map empty of them.
    n_forced = 0
    for lemma in FORCED_REGULAR_VERBS_FOR_POOL:
        if lemma in pt_map and lemma in pp_map:
            continue  # already covered by normal extraction
        regular = _regular_past_form(lemma)
        pt_map[lemma] = regular
        pp_map[lemma] = regular
        n_forced += 1

    pool = []
    missing_inv = 0
    missing_outv = 0
    skipped_missing_form = 0
    for lemma, past in pt_map.items():
        if lemma not in pp_map:
            skipped_missing_form += 1
            continue
        pp = pp_map[lemma]
        past_id = in_w2i.get(past)
        pp_id = in_w2i.get(pp)
        lemma_id = out_w2i.get(lemma)
        if past_id is None or pp_id is None:
            missing_inv += 1
            continue
        if lemma_id is None:
            missing_outv += 1
            continue
        pool.append((past_id, pp_id, lemma_id))

    # Optional: filter out SLOG-derived unaccusative verbs (transfer test).
    # Keep track of what was checked, kept, and skipped for pool_used.txt.
    unacc_checked = []
    unacc_removed = []
    unacc_skipped = []
    if filter_unaccusative:
        removed_ids = set()
        for lemma in SLOG_UNACCUSATIVE_FILTERED:
            unacc_checked.append(lemma)
            lemma_id = out_w2i.get(lemma)
            if lemma_id is None:
                unacc_skipped.append((lemma, "lemma not in output vocab"))
                continue
            present = any(l == lemma_id for (_p, _pp, l) in pool)
            if not present:
                unacc_skipped.append((lemma, "not currently in permutation pool"))
                continue
            removed_ids.add(lemma_id)
            unacc_removed.append(lemma)
        if removed_ids:
            before = len(pool)
            pool = [t for t in pool if t[2] not in removed_ids]
            after = len(pool)
            print("POOL FILTERING ON (COGS):")
            print(f"  candidates checked: {unacc_checked}")
            print(f"  effectively removed from pool: {unacc_removed}")
            if unacc_skipped:
                print(f"  skipped ({len(unacc_skipped)}):")
                for lemma, reason in unacc_skipped:
                    print(f"    {lemma}: {reason}")
            print(f"  new pool size: {after} (was {before})")
            if len(unacc_removed) < 5:
                print(f"  WARNING: only {len(unacc_removed)} verbs removed (<5). "
                      "Transfer test is underpowered — consider stopping.")

    stats = {"pool_size": len(pool),
             "skipped_missing_form": skipped_missing_form,
             "missing_input_vocab": missing_inv,
             "missing_output_vocab": missing_outv,
             "pt_map_size": len(pt_map), "pp_map_size": len(pp_map),
             "n_forced_regular_added": n_forced,
             "unaccusative_checked": unacc_checked,
             "unaccusative_removed": unacc_removed,
             "unaccusative_skipped": unacc_skipped,
             "filter_unaccusative_active": bool(filter_unaccusative)}
    return pool, stats


def _extract_verb_lemma_from_lf(lf):
    toks = lf.split()
    for j in range(len(toks) - 2):
        if toks[j + 1] == "." and toks[j + 2] in ("agent", "theme"):
            return toks[j]
    return None


# ══════════════════════════════════════════════════════════════
# Causal anonymization (Innovation: --causal-curriculum)
# ══════════════════════════════════════════════════════════════
ROLE_TOKENS = ("VERB", "AGENT", "THEME", "RECIPIENT")


def anonymize_example(inp, lf):
    """Replace verb, agent, theme, recipient with role tokens (VERB/AGENT/...).
    Preserves structural tokens (., AND, ;, (, ), ,, x, _, numbers, determiners, was, by, to, that).
    Returns (anon_input, anon_lf) or None if the example can't be cleanly anonymized.
    """
    lf_toks = lf.split()
    inp_toks = inp.split()

    # Step 1: build variable -> noun map from single-arg noun predicates
    noun_of_var = {}
    j = 0
    while j < len(lf_toks):
        if j + 5 < len(lf_toks) and lf_toks[j + 1] == "(" and lf_toks[j + 2] == "x" \
                and lf_toks[j + 3] == "_" and lf_toks[j + 4].isdigit() \
                and lf_toks[j + 5] == ")":
            noun = lf_toks[j]
            if noun not in ("*", "AND", ";"):
                noun_of_var[f"x_{lf_toks[j + 4]}"] = noun
            j += 6
        else:
            j += 1

    # Step 2: collect verb lemmas + role fillers + verb event position
    verb_lemmas = set()
    role_fillers = {"agent": set(), "theme": set(), "recipient": set()}
    verb_event_pos = None
    j = 0
    while j < len(lf_toks) - 6:
        if lf_toks[j + 1] == "." and lf_toks[j + 2] in ("agent", "theme", "recipient") \
                and lf_toks[j + 3] == "(":
            rel = lf_toks[j + 2]
            verb = lf_toks[j]
            verb_lemmas.add(verb)
            # Event position = first arg (x_N after "(")
            if lf_toks[j + 4] == "x" and lf_toks[j + 5] == "_" and lf_toks[j + 6].isdigit():
                if verb_event_pos is None and rel == "agent":
                    verb_event_pos = int(lf_toks[j + 6])
            # Second arg = filler (after comma)
            k = j + 4
            while k < len(lf_toks) and lf_toks[k] != ",":
                k += 1
            if k + 1 < len(lf_toks):
                if k + 3 < len(lf_toks) and lf_toks[k + 1] == "x" \
                        and lf_toks[k + 2] == "_" and lf_toks[k + 3].isdigit():
                    filler_var = f"x_{lf_toks[k + 3]}"
                    fn = noun_of_var.get(filler_var)
                    if fn is not None:
                        role_fillers[rel].add(fn)
                else:
                    candidate = lf_toks[k + 1]
                    if candidate and candidate[0].isupper() and candidate.isalpha():
                        role_fillers[rel].add(candidate)
            j += 3
        else:
            j += 1

    if not verb_lemmas or verb_event_pos is None:
        return None

    # Role priority: AGENT > THEME > RECIPIENT (if same noun fills multiple, take first)
    noun_to_role = {}
    for n in role_fillers["agent"]:
        noun_to_role[n] = "AGENT"
    for n in role_fillers["theme"]:
        noun_to_role.setdefault(n, "THEME")
    for n in role_fillers["recipient"]:
        noun_to_role.setdefault(n, "RECIPIENT")

    if not noun_to_role:
        return None

    # Step 3: rewrite LF tokens
    anon_lf_toks = []
    for tok in lf_toks:
        if tok in verb_lemmas:
            anon_lf_toks.append("VERB")
        elif tok in noun_to_role:
            anon_lf_toks.append(noun_to_role[tok])
        else:
            anon_lf_toks.append(tok)
    anon_lf = " ".join(anon_lf_toks)

    # Step 4: rewrite input
    # Verb position known; replace that token.
    # For other positions, replace tokens that match noun_to_role keys.
    anon_inp_toks = []
    for i, tok in enumerate(inp_toks):
        if i == verb_event_pos:
            anon_inp_toks.append("VERB")
        elif tok in noun_to_role:
            anon_inp_toks.append(noun_to_role[tok])
        elif tok.lower() in noun_to_role:
            anon_inp_toks.append(noun_to_role[tok.lower()])
        else:
            anon_inp_toks.append(tok)
    anon_inp = " ".join(anon_inp_toks)
    return anon_inp, anon_lf


def generate_anon_pairs(source_pairs, n=None, seed=42):
    """Anonymize examples from source (train + cv). Returns list of
    (anon_input, anon_lf, 'anon') suitable for appending to train pool."""
    rng = random.Random(seed)
    results = []
    for inp, lf, _ in source_pairs:
        r = anonymize_example(inp, lf)
        if r is None:
            continue
        results.append((r[0], r[1], "anon"))
    # Dedupe by anon_input (the same active→passive yields same anon)
    seen = set()
    unique = []
    for c in results:
        if c[0] not in seen:
            seen.add(c[0])
            unique.append(c)
    rng.shuffle(unique)
    if n is not None and n > 0 and len(unique) > n:
        unique = unique[:n]
    return unique


def generate_targeted_voice_pairs(train_pairs, target_verb, target_voice,
                                   n=50, seed=42, morphology_fallback=True):
    """Generate N examples of `target_verb` in `target_voice` by transforming
    train examples with that verb in the opposite voice.

    target_voice in {'passive', 'active'} — the voice we WANT to produce.
    Uses morphology fallback by default so active-only verbs like 'bless'
    can get a passive form (blessed).
    """
    rng = random.Random(seed)
    pt_map, pp_map = build_verb_form_maps(train_pairs)
    if morphology_fallback:
        for lemma, past in pt_map.items():
            if lemma not in pp_map and past.endswith("ed"):
                pp_map[lemma] = past
        for lemma, ppw in pp_map.items():
            if lemma not in pt_map and ppw.endswith("ed"):
                pt_map[lemma] = ppw

    results = []
    for inp, lf, _ in train_pairs:
        verb = _extract_verb_lemma_from_lf(lf)
        if verb != target_verb:
            continue
        if target_voice == "passive":
            res = generate_passive_from_active(inp, lf, pp_map)
            tag = f"cv_boost_{target_verb}_passive"
        elif target_voice == "active":
            res = generate_active_from_passive(inp, lf, pt_map)
            tag = f"cv_boost_{target_verb}_active"
        else:
            continue
        if res is not None:
            results.append((res[0], res[1], tag))

    # Dedupe by input
    seen = set()
    unique = []
    for c in results:
        if c[0] not in seen:
            seen.add(c[0])
            unique.append(c)
    rng.shuffle(unique)
    return unique[:n]


def generate_cross_voice_pairs(train_pairs, n_pairs=None, seed=42,
                                morphology_fallback=False):
    """Generate cross-voice (active↔passive) augmentation pairs from train.

    Only handles simple transitive examples (no nmod / ccomp / recipient).
    Returns a list of (input, lf, source_tag) suitable for appending to train.

    morphology_fallback: if True, for each verb in pt_map but missing in pp_map,
    assume regular morphology (past_tense ends in 'ed' → past_participle = past_tense).
    Enables coverage of verbs like 'bless → blessed' that appear only in active
    in train but whose regular form lets us produce the passive.
    """
    rng = random.Random(seed)
    pt_map, pp_map = build_verb_form_maps(train_pairs)
    n_morph_added = 0
    if morphology_fallback:
        for lemma, past in pt_map.items():
            if lemma in pp_map:
                continue
            # Regular verb heuristic: past form ends in "ed"
            if past.endswith("ed"):
                pp_map[lemma] = past
                n_morph_added += 1
    # Candidate pool: all (inp, lf) in train, try both directions
    candidates = []
    n_ok_a2p, n_ok_p2a, n_skip = 0, 0, 0
    for inp, lf, _ in train_pairs:
        res_a2p = generate_passive_from_active(inp, lf, pp_map)
        if res_a2p is not None:
            candidates.append((res_a2p[0], res_a2p[1], "cross_voice_a2p"))
            n_ok_a2p += 1
            continue
        res_p2a = generate_active_from_passive(inp, lf, pt_map)
        if res_p2a is not None:
            candidates.append((res_p2a[0], res_p2a[1], "cross_voice_p2a"))
            n_ok_p2a += 1
            continue
        n_skip += 1
    # Dedupe by input string
    seen = set()
    unique = []
    for c in candidates:
        if c[0] not in seen:
            seen.add(c[0])
            unique.append(c)
    rng.shuffle(unique)
    if n_pairs is not None and n_pairs > 0 and len(unique) > n_pairs:
        unique = unique[:n_pairs]
    return unique, {"n_a2p_generated": n_ok_a2p, "n_p2a_generated": n_ok_p2a,
                    "n_skipped": n_skip, "pp_map_size": len(pp_map),
                    "pt_map_size": len(pt_map), "n_unique_final": len(unique),
                    "n_morph_fallback_added": n_morph_added}


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

    If bidirectional=True, with prob 0.5 per __getitem__ the (src, tgt) pair
    is reversed: LF becomes the source (stripped of BOS/EOS), phrase becomes
    the target (wrapped with BOS/EOS). Requires a unified vocabulary."""

    def __init__(self, pairs, in_w2i, out_w2i, max_out=MAX_OUT,
                 perm_classes=None, bidirectional=False, verb_perm_pool=None,
                 verb_perm_prob=0.5):
        self.data = []
        self.cats = []
        bos = out_w2i[BOS]
        eos = out_w2i[EOS]
        for inp, lf, cat in pairs:
            src = [in_w2i.get(t, 0) for t in inp.split()]
            tgt = [bos] + [out_w2i.get(t, 0) for t in lf.split()] + [eos]
            tgt = tgt[:max_out]
            self.data.append((src, tgt, len(inp.split())))
            self.cats.append(cat)
        self.perm_classes = [c for c in (perm_classes or []) if len(c) >= 2]
        self.permute = len(self.perm_classes) > 0
        self.bidirectional = bidirectional
        self.bos = bos
        self.eos = eos
        # Verb permutation: pool of (past_in_id, pp_in_id, lemma_out_id) tuples.
        # With probability verb_perm_prob, a random bijection on the pool is drawn
        # and applied consistently to src (past + pp forms) and tgt (lemma).
        self.verb_perm_pool = verb_perm_pool if verb_perm_pool and len(verb_perm_pool) >= 2 else None
        self.verb_perm_prob = verb_perm_prob

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
        # Verb permutation: bijection on the verb pool at verb_perm_prob.
        # Analogous to A4_permute on SCAN: shuffles verb identities while keeping structure.
        if self.verb_perm_pool is not None and random.random() < self.verb_perm_prob:
            pool = self.verb_perm_pool
            K = len(pool)
            order = list(range(K))
            random.shuffle(order)
            m_past = {pool[k][0]: pool[order[k]][0] for k in range(K)}
            m_pp   = {pool[k][1]: pool[order[k]][1] for k in range(K)}
            m_lem  = {pool[k][2]: pool[order[k]][2] for k in range(K)}
            src = [m_past.get(t, m_pp.get(t, t)) for t in src]
            tgt = [m_lem.get(t, t) for t in tgt]
        # Bidirectional swap: 50% chance
        if self.bidirectional and random.random() < 0.5:
            # new_src = old tgt without BOS/EOS
            new_src = [t for t in tgt if t != self.bos and t != self.eos]
            # new_tgt = [BOS] + old_src + [EOS]
            new_tgt = [self.bos] + src + [self.eos]
            return (new_src, new_tgt, len(new_src), cat)
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


class JEPAPredictor(nn.Module):
    """Predicts encoder representation at position t+1 from position t.
    Lightweight MLP: d_model → d_model → d_model. Stop-gradient on the target
    so the predictor must learn the prediction without collapsing both sides.

    When `target_enc_out` is provided (typically the EMA encoder output, already
    no_grad), it is used as target instead of `enc_out[:, 1:].detach()`.
    """
    def __init__(self, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, enc_out, target_enc_out=None):
        # enc_out: (B, T, d_model). Predict t+1 from t.
        pred = self.net(enc_out[:, :-1])
        if target_enc_out is None:
            target = enc_out[:, 1:].detach()  # stop-gradient
        else:
            target = target_enc_out[:, 1:]    # already no_grad (EMA encoder)
        return pred, target


def _jepa_loss(pred, target, src_mask):
    """Mean squared error between (predicted, target) representations,
    masked by valid input positions and normalised by the d_model dimension.
    src_mask : (B, T) boolean True where token is real."""
    # pred / target shape: (B, T-1, d_model). Mask is on positions [1:].
    mask = src_mask[:, 1:].unsqueeze(-1).float()
    sq = (pred - target) ** 2
    denom = mask.sum().clamp(min=1.0) * pred.size(-1)
    return (sq * mask).sum() / denom


def _grad_snapshot(model, batch, device, use_copy):
    """TEST 3 helper — calcule cos_sim entre gradient task et gradient JEPA
    sur encoder.parameters() sur un batch fixe.
    Retourne dict {cos_sim, norm_task, norm_jepa, loss_task, loss_jepa}.

    `batch` est la sortie de `collate(...)` :
        (src, src_mask, tgt, in_lens, cats)
    """
    model.train()
    src = batch[0].to(device)
    src_mask = batch[1].to(device).bool()
    tgt = batch[2].to(device)
    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]

    # Task forward
    enc_out = model.encode(src, src_mask)
    if use_copy:
        log_probs = model.decode_with_copy(src, enc_out, ~src_mask, tgt_in)
        V = log_probs.size(-1)
        task_loss = F.nll_loss(log_probs.reshape(-1, V),
                                tgt_out.reshape(-1), ignore_index=0)
    else:
        logits = model.decode(enc_out, ~src_mask, tgt_in)
        V = logits.size(-1)
        task_loss = F.cross_entropy(logits.reshape(-1, V),
                                     tgt_out.reshape(-1), ignore_index=0)

    # JEPA loss (re-use same enc_out — identique au training)
    if getattr(model, "use_jepa_ema", False):
        target_enc = model.encode_ema(src, src_mask)
        pred_j, target_j = model.jepa_predictor(enc_out, target_enc)
    else:
        pred_j, target_j = model.jepa_predictor(enc_out)
    jepa_loss = _jepa_loss(pred_j, target_j, src_mask)

    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    if not enc_params:
        return None
    grads_task = torch.autograd.grad(task_loss, enc_params,
                                      retain_graph=True, allow_unused=True)
    grads_jepa = torch.autograd.grad(jepa_loss, enc_params,
                                      retain_graph=False, allow_unused=True)
    flat_t = torch.cat([g.flatten() for g in grads_task if g is not None])
    flat_j = torch.cat([g.flatten() for g in grads_jepa if g is not None])
    if flat_t.numel() == 0 or flat_j.numel() == 0:
        return None
    cs = F.cosine_similarity(flat_t.unsqueeze(0), flat_j.unsqueeze(0)).item()
    return {
        "cos_sim": float(cs),
        "norm_task": float(flat_t.norm().item()),
        "norm_jepa": float(flat_j.norm().item()),
        "loss_task": float(task_loss.item()),
        "loss_jepa": float(jepa_loss.item()),
    }


class TransformerSeq2Seq(nn.Module):
    """Encoder 2 layers + Decoder 2 layers, d=128, 4 heads. Pre-LN."""

    def __init__(self, in_vocab, out_vocab, d_model=128, n_heads=4,
                 n_layers=2, ffn=256, dropout=0.1,
                 max_in=MAX_IN, max_out=MAX_OUT, use_jepa=False,
                 use_jepa_ema=False, jepa_ema_decay=0.99):
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
        # Optional JEPA predictor (Phase A integration)
        self.use_jepa = use_jepa
        self.use_jepa_ema = bool(use_jepa and use_jepa_ema)
        self.jepa_ema_decay = float(jepa_ema_decay)
        if use_jepa:
            self.jepa_predictor = JEPAPredictor(d_model)
        if self.use_jepa_ema:
            import copy as _copy
            self.encoder_ema = _copy.deepcopy(self.encoder)
            for p in self.encoder_ema.parameters():
                p.requires_grad = False

    def encode(self, src, src_mask):
        L = src.size(1)
        if L > self.max_in:
            raise RuntimeError(f"src length {L} > max_in {self.max_in}")
        pos = torch.arange(L, device=src.device).unsqueeze(0)
        x = self.in_emb(src) + self.in_pe(pos)
        return self.encoder(x, src_key_padding_mask=~src_mask)

    def encode_ema(self, src, src_mask):
        """Forward through EMA encoder (no_grad). Only valid when use_jepa_ema."""
        L = src.size(1)
        pos = torch.arange(L, device=src.device).unsqueeze(0)
        with torch.no_grad():
            x = self.in_emb(src) + self.in_pe(pos)
            return self.encoder_ema(x, src_key_padding_mask=~src_mask)

    @torch.no_grad()
    def update_ema(self):
        """EMA update of encoder_ema params from live encoder. Call AFTER opt.step()."""
        if not self.use_jepa_ema:
            return
        d = self.jepa_ema_decay
        for p_ema, p in zip(self.encoder_ema.parameters(),
                            self.encoder.parameters()):
            p_ema.data.mul_(d).add_(p.data, alpha=1.0 - d)

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
                 in_to_out_map=None, use_jepa=False,
                 use_jepa_ema=False, jepa_ema_decay=0.99):
        super().__init__(in_vocab, out_vocab, d_model, n_heads, n_layers,
                         ffn, dropout, max_in, max_out, use_jepa=use_jepa,
                         use_jepa_ema=use_jepa_ema, jepa_ema_decay=jepa_ema_decay)
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
                 in_to_out_map=None, role_dim=16, token_categories=None,
                 use_jepa=False, use_jepa_ema=False, jepa_ema_decay=0.99):
        super().__init__(in_vocab, out_vocab, d_model, n_heads, n_layers, ffn,
                         dropout, max_in, max_out, in_to_out_map=in_to_out_map,
                         use_jepa=use_jepa, use_jepa_ema=use_jepa_ema,
                         jepa_ema_decay=jepa_ema_decay)
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


def extract_train_constraints(train_pairs):
    """Extract soft constraints from the train set to re-rank beams at inference.

    Returns a dict with:
      always_agent : set of nouns only seen as agent (never theme)
      always_theme : set of nouns only seen as theme (never agent)
      proper_names : set of capitalized tokens (excluding determiners)
      nmod_heads   : set of tokens preceded by .nmod (i.e., nouns)
      ccomp_heads  : set of tokens preceded by .ccomp (i.e., verbs)
      known_verbs  : set of verb stems seen in LFs (prefix before .agent/.theme/.recipient)
    """
    from collections import defaultdict
    agent_nouns = set()
    theme_nouns = set()
    nmod_heads = set()
    ccomp_heads = set()
    known_verbs = set()
    proper_names = set()
    DETS = {"The", "A", "the", "a"}

    for inp, lf, _ in train_pairs:
        # Proper names from the input
        for tok in inp.split():
            if tok and tok[0].isupper() and tok not in DETS and tok.isalpha():
                proper_names.add(tok)

        # Scan LF tokens. LFs use tokens like "eat . agent", "cake", "x _ 1"
        # We iterate over substrings delimited by spaces and look for the patterns
        # "<HEAD> . agent ( ... , <VAR_OR_NOUN> )" etc.
        toks = lf.split()
        # Build a simple linear scan: whenever we see "<head> . <rel>", note it
        i = 0
        while i < len(toks) - 2:
            if toks[i + 1] == "." and toks[i + 2] in ("agent", "theme", "recipient",
                                                       "nmod", "ccomp", "xcomp"):
                head = toks[i]
                rel = toks[i + 2]
                if rel in ("agent", "theme", "recipient"):
                    known_verbs.add(head)
                if rel == "nmod":
                    nmod_heads.add(head)
                elif rel == "ccomp":
                    ccomp_heads.add(head)
                i += 3
            else:
                i += 1

        # Crude animacy detection: collect (variable, noun-used) pairs per example
        # For COGS, each "noun ( x _ N )" or "* noun ( x _ N )" declares a noun.
        # And "verb . agent ( x_i , x_j )" assigns the verb's agent to x_j.
        # Simple heuristic: find all noun predicates, then verb.role predicates,
        # and note which nouns end up as agent vs theme.
        noun_of_var = {}
        j = 0
        while j < len(toks):
            # pattern: [noun] ( x _ N )
            if j + 4 < len(toks) and toks[j + 1] == "(" and toks[j + 2] == "x" \
                    and toks[j + 3] == "_" and toks[j + 4].isdigit():
                noun = toks[j]
                if noun not in ("*", "AND", ";"):
                    var = f"{toks[j+2]}_{toks[j+4]}"
                    noun_of_var[var] = noun
                j += 6
            else:
                j += 1
        # Now find role assignments
        j = 0
        while j < len(toks) - 6:
            if toks[j + 1] == "." and toks[j + 2] in ("agent", "theme") \
                    and toks[j + 3] == "(":
                rel = toks[j + 2]
                # The second argument is at toks[j+7..] — pattern: ( x _ i , x _ j )
                # Find comma, then take the variable after it
                k = j + 4
                # Skip until comma
                while k < len(toks) and toks[k] != ",":
                    k += 1
                if k + 3 < len(toks) and toks[k + 1] == "x" and toks[k + 2] == "_":
                    var = f"x_{toks[k+3]}"
                    if var in noun_of_var:
                        noun = noun_of_var[var]
                        if rel == "agent":
                            agent_nouns.add(noun)
                        elif rel == "theme":
                            theme_nouns.add(noun)
                j += 4
            else:
                j += 1

    return {
        "always_agent": agent_nouns - theme_nouns,
        "always_theme": theme_nouns - agent_nouns,
        "flexible":     agent_nouns & theme_nouns,
        "proper_names": proper_names,
        "nmod_heads":   nmod_heads,
        "ccomp_heads":  ccomp_heads,
        "known_verbs":  known_verbs,
    }


def score_hypothesis(lf_tokens, input_tokens, constraints, log_prob):
    """Score an LF hypothesis against train-set constraints.

    Returns (adjusted_score, violations_list). Higher score = better.
    """
    score = float(log_prob)
    violations = []
    input_words = list(input_tokens)
    input_lower = [w.lower() for w in input_words]

    toks = lf_tokens
    # Build variable -> noun map for this LF
    noun_of_var = {}
    j = 0
    while j < len(toks):
        if j + 4 < len(toks) and toks[j + 1] == "(" and toks[j + 2] == "x" \
                and toks[j + 3] == "_" and toks[j + 4].isdigit():
            noun = toks[j]
            if noun not in ("*", "AND", ";"):
                var = f"x_{toks[j+4]}"
                noun_of_var[var] = noun
            j += 6
        else:
            j += 1

    # Check each head.rel(args) predicate
    j = 0
    while j < len(toks) - 3:
        if toks[j + 1] == "." and toks[j + 2] in ("agent", "theme", "recipient",
                                                   "nmod", "ccomp", "xcomp"):
            head = toks[j]
            rel = toks[j + 2]

            # Constraint: verb head must be a known verb lemma from train
            # (skipping direct input match — COGS LFs lemmatize verbs, e.g. "eat" vs "eaten")
            if rel in ("agent", "theme", "recipient"):
                if head not in constraints["known_verbs"]:
                    score -= 5.0
                    violations.append(f"UNK_VERB: '{head}' not in known_verbs")

            # Constraint: nmod should attach to a noun, not a verb
            if rel == "nmod":
                if head in constraints["ccomp_heads"] and head not in constraints["nmod_heads"]:
                    score -= 3.0
                    violations.append(f"ATTACH: nmod on verb '{head}'")

            # Constraint: ccomp should attach to a verb, not a noun
            if rel == "ccomp":
                if head in constraints["nmod_heads"] and head not in constraints["ccomp_heads"]:
                    score -= 3.0
                    violations.append(f"ATTACH: ccomp on noun '{head}'")

            # Constraint: animacy on agent/theme
            if rel in ("agent", "theme"):
                # Walk forward to find "( _ , x _ N )"
                k = j + 4
                while k < len(toks) and toks[k] != ",":
                    k += 1
                if k + 3 < len(toks) and toks[k + 1] == "x" and toks[k + 2] == "_":
                    var = f"x_{toks[k+3]}"
                    filler_noun = noun_of_var.get(var, None)
                    if filler_noun is None:
                        # Maybe a proper name directly
                        if k + 1 < len(toks) and toks[k + 1] and toks[k + 1][0].isupper():
                            filler_noun = toks[k + 1]
                    if filler_noun is not None:
                        if rel == "agent" and filler_noun in constraints["always_theme"]:
                            score -= 5.0
                            violations.append(f"ANIMACY: '{filler_noun}' as agent")
                        if rel == "theme" and filler_noun in constraints["proper_names"] \
                                and filler_noun in constraints["always_agent"]:
                            score -= 5.0
                            violations.append(f"ANIMACY: proper '{filler_noun}' as theme")

            # Constraint: passive voice — if input has "was" + "by X", agent should be X
            if rel == "agent" and "was" in input_words and "by" in input_words:
                try:
                    by_idx = input_words.index("by")
                    if by_idx + 1 < len(input_words):
                        expected = input_words[by_idx + 1]
                        k = j + 4
                        while k < len(toks) and toks[k] != ",":
                            k += 1
                        # The agent filler should be expected
                        ag_filler = None
                        if k + 3 < len(toks) and toks[k + 1] == "x" and toks[k + 2] == "_":
                            var = f"x_{toks[k+3]}"
                            ag_filler = noun_of_var.get(var, None)
                        elif k + 1 < len(toks):
                            ag_filler = toks[k + 1]
                        if ag_filler is not None and ag_filler.lower() != expected.lower():
                            score -= 3.0
                            violations.append(f"PASSIVE: agent should be '{expected}', got '{ag_filler}'")
                except ValueError:
                    pass

            j += 3
        else:
            j += 1

    return score, violations


@torch.no_grad()
def beam_decode(model, src, src_mask, out_w2i, beam_size=10, max_len=MAX_DECODE):
    """Batched beam search. For each example in the batch of B inputs, expands a
    beam of size K in parallel. Returns list[B] of list[K] of (tokens, log_prob).

    Implementation: we replicate each encoder output K times so all K beams of a
    single example are processed in one forward pass.
    """
    B = src.size(0)
    K = beam_size
    device = src.device
    bos, eos = out_w2i[BOS], out_w2i[EOS]
    enc = model.encode(src, src_mask)
    mem_kpm = ~src_mask
    uses_tags = isinstance(model, TransformerSeq2SeqCopyTags)
    uses_copy = isinstance(model, TransformerSeq2SeqWithCopy)

    out_beams = []
    for b in range(B):
        # Replicate encoder output K times for parallel beam expansion
        enc_b = enc[b:b+1].expand(K, -1, -1).contiguous()
        mem_kpm_b = mem_kpm[b:b+1].expand(K, -1).contiguous()
        src_b = src[b:b+1].expand(K, -1).contiguous()

        # Initialize beams: all K slots start with BOS.
        # At step 0, only the first slot is active so we don't get K identical beams.
        tokens = torch.full((K, 1), bos, device=device, dtype=torch.long)
        log_probs = torch.full((K,), -1e9, device=device, dtype=torch.float32)
        log_probs[0] = 0.0
        done = torch.zeros(K, dtype=torch.bool, device=device)

        for step in range(max_len):
            if uses_tags:
                logits, _ = model.decode_with_copy_tags(src_b, enc_b, mem_kpm_b,
                                                       tokens, roles_gt=None)
            elif uses_copy:
                logits = model.decode_with_copy(src_b, enc_b, mem_kpm_b, tokens)
            else:
                logits = model.decode(enc_b, mem_kpm_b, tokens)
            # Next-token log-probs: (K, V)
            if uses_copy:
                step_lp = logits[:, -1, :]
            else:
                step_lp = F.log_softmax(logits[:, -1, :], dim=-1)
            V = step_lp.size(-1)
            # For finished beams, lock in with EOS log_prob=0 (keep their tokens)
            # We achieve this by: expanded_lp = log_probs[:, None] + step_lp (for active)
            # For done beams, force them to "repeat EOS" with 0 marginal contribution.
            expanded = log_probs.unsqueeze(1) + step_lp   # (K, V)
            if done.any():
                # For done rows, set all candidates to -inf except PAD/EOS=itself with 0
                expanded[done] = -1e9
                expanded[done, eos] = log_probs[done]     # keep EOS as non-scoring continuation
            # Top-K across the flattened K*V space
            flat = expanded.reshape(-1)
            top_lp, top_idx = flat.topk(K)
            new_beam_src = top_idx // V
            new_token = top_idx % V
            tokens = torch.cat([tokens[new_beam_src], new_token.unsqueeze(1)], dim=1)
            log_probs = top_lp
            done = done[new_beam_src] | (new_token == eos)
            if done.all():
                break

        # Strip BOS (first column), return as list of (tokens_tensor, log_prob)
        finals = []
        for k in range(K):
            finals.append((tokens[k, 1:].clone(), float(log_probs[k].item())))
        out_beams.append(finals)
    return out_beams


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
def evaluate(model, loader, out_w2i, device, max_len=MAX_DECODE,
             constraints=None, beam_size=10, in_i2w=None, out_i2w=None,
             diag_cats=None, max_examples=0, progress_every=500):
    """Greedy eval (constraints=None) or constrained-beam eval (constraints=dict).

    If constraints is provided, runs beam_decode(k=beam_size), scores each hypothesis
    against constraints, and picks the highest adjusted score.

    diag_cats (optional): set of category names for which to log beam ranks of gold.
    max_examples (optional, >0): stop after this many examples (for quick tests).
    progress_every: print a progress line every N examples (only when constraints used).
    """
    import time as _time
    model.eval()
    n, exact, tok_c, tok_t = 0, 0, 0, 0
    by_cat: Dict[str, Tuple[int, int]] = {}
    viol_counts = {}
    beam_rank_log = []  # (cat, gold_rank_or_minus_one, n_beams)
    t_start = _time.time()
    for src, src_mask, tgt, _, cats in loader:
        if max_examples and n >= max_examples:
            break
        src = src.to(device); src_mask = src_mask.to(device); tgt = tgt.to(device)

        if constraints is not None:
            beams_per_ex = beam_decode(model, src, src_mask, out_w2i,
                                       beam_size=beam_size, max_len=max_len)
            # For each example, pick best-scoring hypothesis against constraints
            B = src.size(0)
            # Build pred batch
            pred_list = []
            for bi in range(B):
                beams = beams_per_ex[bi]
                # Compute input words (without padding)
                src_ids_b = src[bi].tolist()
                input_words = []
                if in_i2w is not None:
                    for tid in src_ids_b:
                        if tid == 0:
                            break
                        w = in_i2w.get(tid, "")
                        if w:
                            input_words.append(w)
                # Score beams
                best = None
                best_score = -1e18
                for tokens_t, lp in beams:
                    lf_toks_ids = tokens_t.tolist()
                    # Strip trailing EOS / pads
                    eos = out_w2i[EOS]
                    if eos in lf_toks_ids:
                        lf_toks_ids = lf_toks_ids[:lf_toks_ids.index(eos)]
                    lf_toks_strs = []
                    if out_i2w is not None:
                        for tid in lf_toks_ids:
                            if tid == 0:
                                break
                            w = out_i2w.get(tid, "")
                            if w:
                                lf_toks_strs.append(w)
                    adj, viols = score_hypothesis(lf_toks_strs, input_words,
                                                   constraints, lp)
                    for v in viols:
                        key = v.split(":")[0]
                        viol_counts[key] = viol_counts.get(key, 0) + 1
                    if adj > best_score:
                        best_score = adj
                        best = tokens_t
                # Diagnostic: find where the gold ranks
                if diag_cats is not None and cats[bi] in diag_cats:
                    gold_ids = [t for t in tgt[bi, 1:].tolist() if t != 0]
                    # Strip trailing EOS
                    eos = out_w2i[EOS]
                    if eos in gold_ids:
                        gold_ids = gold_ids[:gold_ids.index(eos)]
                    rank = -1
                    for bi_r, (tk, _lp) in enumerate(beams):
                        tk_ids = [t for t in tk.tolist() if t != 0]
                        if eos in tk_ids:
                            tk_ids = tk_ids[:tk_ids.index(eos)]
                        if tk_ids == gold_ids:
                            rank = bi_r
                            break
                    beam_rank_log.append((cats[bi], rank, len(beams)))
                pred_list.append(best if best is not None
                                 else torch.zeros(1, dtype=torch.long, device=device))
            # Pad to same length for batch comparison
            L_max = max(p.size(0) for p in pred_list)
            pred = torch.zeros(len(pred_list), L_max, dtype=torch.long, device=device)
            for i_, p_ in enumerate(pred_list):
                pred[i_, :p_.size(0)] = p_
        else:
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
        if constraints is not None and n > 0 and (n // progress_every) != ((n - src.size(0)) // progress_every):
            elapsed = _time.time() - t_start
            rate = n / max(elapsed, 1e-9)
            cur_acc = exact / n * 100
            total = max_examples if max_examples else len(loader.dataset)
            eta = (total - n) / max(rate, 1e-9)
            print(f"    [beam] {n}/{total}  acc={cur_acc:.2f}%  "
                  f"rate={rate:.1f}ex/s  ETA={eta/60:.1f}min", flush=True)
    by_cat_pct = {c: round(cor / tot * 100, 2) for c, (tot, cor) in sorted(by_cat.items())}
    out = {"exact": exact / n * 100, "tok_acc": tok_c / max(tok_t, 1) * 100,
           "n": n, "by_cat": by_cat_pct}
    if constraints is not None:
        out["violations"] = viol_counts
        if beam_rank_log:
            out["beam_rank_log"] = beam_rank_log
    return out


def train_one_run(variant: str, seed: int, args):
    # Strict seeding for reproducibility across sweeps (spec: 7 calls).
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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

    # Phase A.2 : extra-train file (augmentation pilotée par meta-modèle)
    extra_train_file = getattr(args, "extra_train_file", None)
    if extra_train_file and os.path.exists(extra_train_file):
        extra_pairs = parse_cogs_tsv(extra_train_file)
        print(f"EXTRA TRAIN ON: +{len(extra_pairs)} examples from {extra_train_file}")
        for pp in extra_pairs[:3]:
            print(f"  ex: {pp[0][:80]} ...")
        train_pairs = train_pairs + extra_pairs

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

    # Innovation K — Peel-and-stack (minimal viable): iterate PP recursion extension
    # to cover depths 3, 4, 5 (or more with --peel-stack-depth N).
    if getattr(args, "peel_stack", False):
        ps_depth = getattr(args, "peel_stack_depth", 3)
        print(f"PEEL-AND-STACK augmentation ON (depth={ps_depth}):")
        cur_pairs = list(train_pairs)
        total_k = 0
        for depth_level in range(ps_depth):
            pp_next = generate_pp_recursion_extension(cur_pairs, max_aug=700)
            if not pp_next:
                break
            print(f"  depth_level {depth_level+1}: +{len(pp_next)} examples")
            cur_pairs = cur_pairs + pp_next
            train_pairs = train_pairs + pp_next
            total_k += len(pp_next)
        print(f"  Total peel-stack augmentation: {total_k}")

    # Wh-passive augmentation (generate "What was V-ed by Y ?" from declarative passives)
    if getattr(args, "wh_passive_aug", False):
        wh_n = getattr(args, "wh_passive_n", 0)
        wh_examples = generate_wh_passive_examples(
            train_pairs, n=(wh_n if wh_n > 0 else None), seed=seed)
        print(f"WH-PASSIVE augmentation ON: +{len(wh_examples)} examples "
              f"(cap={wh_n if wh_n > 0 else 'none'})")
        for i, (a, l, _t) in enumerate(wh_examples[:3]):
            print(f"  whp{i}: {a}")
            print(f"         {l[:130]}...")
        train_pairs = train_pairs + wh_examples

    # CP recursion peel (same cascade logic as PP peel, but for clausal complements)
    if getattr(args, "peel_cp", False):
        cp_depth = getattr(args, "peel_cp_depth", 3)
        print(f"PEEL-CP augmentation ON (depth={cp_depth}):")
        cur_pairs = list(train_pairs)
        total_cp = 0
        for cp_level in range(cp_depth):
            cp_next = generate_cp_recursion_extension(cur_pairs, max_aug=700)
            if not cp_next:
                break
            print(f"  cp_level {cp_level+1}: +{len(cp_next)} examples")
            for i, (a, l, _t) in enumerate(cp_next[:2] if cp_level == 0 else []):
                print(f"    cp{i}: {a}")
                print(f"          {l[:130]}...")
            cur_pairs = cur_pairs + cp_next
            train_pairs = train_pairs + cp_next
            total_cp += len(cp_next)
        print(f"  Total CP peel augmentation: {total_cp}")

    # Innovation: Cross-voice augmentation
    if getattr(args, "cross_voice", False):
        cv_n = getattr(args, "cross_voice_n", 0)
        cv_oversample = max(1, getattr(args, "cross_voice_oversample", 1))
        cv_morph = getattr(args, "cross_voice_morphology", False)
        cv_pairs, cv_stats = generate_cross_voice_pairs(
            train_pairs, n_pairs=(cv_n if cv_n > 0 else None), seed=seed,
            morphology_fallback=cv_morph)
        cv_stats["oversample"] = cv_oversample
        cv_stats["morphology_fallback"] = cv_morph
        print(f"CROSS-VOICE augmentation ON:")
        print(f"  past-tense map: {cv_stats['pt_map_size']} verbs  "
              f"past-participle map: {cv_stats['pp_map_size']} verbs"
              + (f"  (+{cv_stats['n_morph_fallback_added']} regular-ed fallbacks)"
                 if cv_morph and cv_stats.get('n_morph_fallback_added') else ""))
        print(f"  candidates: a2p={cv_stats['n_a2p_generated']}  "
              f"p2a={cv_stats['n_p2a_generated']}  skipped={cv_stats['n_skipped']}")
        n_unique = len(cv_pairs)
        if cv_oversample > 1:
            cv_pairs_final = cv_pairs * cv_oversample
            print(f"  final: {n_unique} unique × {cv_oversample} oversample = "
                  f"{len(cv_pairs_final)} pairs added to train")
        else:
            cv_pairs_final = cv_pairs
            print(f"  final (after dedupe + cap): {n_unique} pairs added to train")
        for i, (a, l, tag) in enumerate(cv_pairs[:3]):
            print(f"    cv{i} [{tag}]: {a}")
            print(f"             {l[:100]}...")
        train_pairs = train_pairs + cv_pairs_final
        with open(os.path.join(run_dir, "cross_voice_stats.json"), "w") as f:
            json.dump(cv_stats, f, indent=2)

    # Innovation: Targeted cross-voice boost for bless (passive) + squeeze (active)
    if getattr(args, "cv_boost_bless_squeeze", False):
        cv_boost_n = getattr(args, "cv_boost_n", 50)
        bless_passives = generate_targeted_voice_pairs(
            train_pairs, target_verb="bless", target_voice="passive",
            n=cv_boost_n, seed=seed, morphology_fallback=True)
        squeeze_actives = generate_targeted_voice_pairs(
            train_pairs, target_verb="squeeze", target_voice="active",
            n=cv_boost_n, seed=seed + 1, morphology_fallback=True)
        print(f"CV-BOOST (targeted) ON: bless passives={len(bless_passives)}  "
              f"squeeze actives={len(squeeze_actives)}  (n_per_verb={cv_boost_n})")
        for i, (a, l, tag) in enumerate(bless_passives[:2]):
            print(f"  boost_b{i}: {a}")
            print(f"            {l[:110]}...")
        for i, (a, l, tag) in enumerate(squeeze_actives[:2]):
            print(f"  boost_s{i}: {a}")
            print(f"            {l[:110]}...")
        train_pairs = train_pairs + bless_passives + squeeze_actives

    # Granular targeted boosts: --cv-boost-squeeze N and --cv-boost-bless N
    n_squeeze = getattr(args, "cv_boost_squeeze", 0)
    if n_squeeze > 0:
        squeeze_pairs = generate_targeted_voice_pairs(
            train_pairs, target_verb="squeeze", target_voice="active",
            n=n_squeeze, seed=seed, morphology_fallback=True)
        print(f"CV-BOOST-SQUEEZE ON: {len(squeeze_pairs)} active-voice pairs for squeeze")
        for i, (a, l, _t) in enumerate(squeeze_pairs[:2]):
            print(f"  sq{i}: {a}")
            print(f"       {l[:120]}...")
        train_pairs = train_pairs + squeeze_pairs

    n_bless = getattr(args, "cv_boost_bless", 0)
    if n_bless > 0:
        bless_pairs = generate_targeted_voice_pairs(
            train_pairs, target_verb="bless", target_voice="passive",
            n=n_bless, seed=seed + 1, morphology_fallback=True)
        print(f"CV-BOOST-BLESS ON: {len(bless_pairs)} passive-voice pairs for bless")
        for i, (a, l, _t) in enumerate(bless_pairs[:2]):
            print(f"  bl{i}: {a}")
            print(f"       {l[:120]}...")
        train_pairs = train_pairs + bless_pairs

    # Innovation: Causal curriculum — generate anonymized pool from train + cv pairs
    causal_curriculum = getattr(args, "causal_curriculum", False)
    anon_pairs = []
    if causal_curriculum:
        # Source = everything that looks like a real training example with role predicates.
        # The anonymizer keeps only those where it can identify at least one role filler.
        source_for_anon = list(train_pairs)  # includes peel-stack + cross-voice if enabled
        cap_anon = getattr(args, "causal_n_anon", 0)
        anon_pairs = generate_anon_pairs(source_for_anon,
                                         n=(cap_anon if cap_anon > 0 else None),
                                         seed=seed)
        print(f"CAUSAL CURRICULUM (anonymization) ON:")
        print(f"  source pool: {len(source_for_anon)} examples  "
              f"anon generated: {len(anon_pairs)} unique anonymized")
        for i, (a, l, _t) in enumerate(anon_pairs[:3]):
            print(f"  anon{i}: {a}")
            print(f"         {l[:130]}...")
        # Add anon pairs to train_pairs so role tokens land in vocab
        train_pairs = train_pairs + anon_pairs
        if len(anon_pairs) == 0:
            print("WARNING: 0 anonymized examples generated — causal curriculum will be disabled at batch time.")
            causal_curriculum = False

    print(f"train: {len(train_pairs)}  dev: {len(dev_pairs)}  gen: {len(gen_pairs)}")

    bidirectional = getattr(args, "bidirectional", False)
    if bidirectional:
        unified = build_unified_vocab(train_pairs + dev_pairs + gen_pairs)
        in_w2i, out_w2i = unified, unified
        print(f"BIDIRECTIONAL: unified vocab = {len(unified)}")
    else:
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

    # Build verb permutation pool (A4-style for COGS)
    verb_perm_pool = None
    verb_perm_prob = getattr(args, "permute_verbs_prob", 0.5)
    if getattr(args, "permute_verbs", False):
        filter_unacc = getattr(args, "filter_unaccusative_from_pool", False)
        verb_perm_pool, vpp_stats = build_verb_perm_pool(
            train_pairs, in_w2i, out_w2i, morphology_fallback=True,
            filter_unaccusative=filter_unacc)
        # Write pool_used.txt for transfer-test bookkeeping (spec §7)
        if filter_unacc:
            pool_used_path = os.path.join(run_dir, "pool_used.txt")
            with open(pool_used_path, "w") as f:
                f.write("=== Unaccusative filter (COGS transfer test) ===\n")
                f.write(f"checked: {vpp_stats['unaccusative_checked']}\n")
                f.write(f"removed from pool: {vpp_stats['unaccusative_removed']}\n")
                f.write("skipped:\n")
                for lemma, reason in vpp_stats['unaccusative_skipped']:
                    f.write(f"  {lemma}: {reason}\n")
                f.write(f"\nPool size after filtering: {vpp_stats['pool_size']}\n")
                f.write(f"n_forced_regular_added: {vpp_stats['n_forced_regular_added']}\n")
            print(f"  wrote {pool_used_path}")
        print(f"VERB PERMUTATION ON: pool size={vpp_stats['pool_size']} verbs  "
              f"(pt_map={vpp_stats['pt_map_size']}, pp_map={vpp_stats['pp_map_size']}, "
              f"forced={vpp_stats.get('n_forced_regular_added', 0)}, "
              f"skipped={vpp_stats['skipped_missing_form']}) prob={verb_perm_prob}")
        if vpp_stats.get('n_forced_regular_added', 0) > 0:
            print(f"  Forced-added to pool via regular morphology: "
                  f"{FORCED_REGULAR_VERBS_FOR_POOL[:vpp_stats['n_forced_regular_added']]}")
        if vpp_stats["missing_input_vocab"] or vpp_stats["missing_output_vocab"]:
            print(f"  WARNING: {vpp_stats['missing_input_vocab']} verbs skipped for missing input ids, "
                  f"{vpp_stats['missing_output_vocab']} for missing output lemma ids")
        if len(verb_perm_pool) < 2:
            print("  Pool too small for permutation — disabling verb permutation.")
            verb_perm_pool = None

    tr_ds = COGSDataset(train_pairs, in_w2i, out_w2i,
                        perm_classes=perm_classes if variant in ("B1", "B2", "B2b", "B2c", "B4", "B4_turbo", "B5", "B5_lite", "B6", "B7a", "B7c") else None,
                        bidirectional=bidirectional,
                        verb_perm_pool=verb_perm_pool,
                        verb_perm_prob=verb_perm_prob)
    # dev and gen are ALWAYS in forward direction (we evaluate phrase → LF)
    dv_ds = COGSDataset(dev_pairs,   in_w2i, out_w2i)
    ge_ds = COGSDataset(gen_pairs,   in_w2i, out_w2i)

    eval_bs = args.eval_batch_size
    tr_ld = DataLoader(tr_ds, args.batch_size, shuffle=True,  collate_fn=collate, num_workers=0, pin_memory=True)
    dv_ld = DataLoader(dv_ds, eval_bs, shuffle=False, collate_fn=collate, num_workers=0, pin_memory=True)
    ge_ld = DataLoader(ge_ds, eval_bs, shuffle=False, collate_fn=collate, num_workers=0, pin_memory=True)

    # Causal curriculum: 3-pool triple batch iterator
    tr_ld_causal = None
    if causal_curriculum and len(anon_pairs) > 0:
        causal_ratio = getattr(args, "causal_ratio", 0.33)
        n_anon_batch = max(1, int(round(args.batch_size * causal_ratio)))
        remaining = args.batch_size - n_anon_batch
        n_orig_batch = remaining // 2
        n_cv_batch = remaining - n_orig_batch

        orig_pool_causal = [p for p in train_pairs
                            if not (len(p) >= 3 and isinstance(p[2], str)
                                    and (p[2].startswith("cross_voice") or p[2] == "anon"
                                         or p[2].startswith("cv_boost")))]
        cv_pool_causal = [p for p in train_pairs
                          if len(p) >= 3 and isinstance(p[2], str)
                          and p[2].startswith("cross_voice")]
        anon_pool_causal = anon_pairs

        # Fall back gracefully if cv pool is empty
        if len(cv_pool_causal) == 0:
            print("WARNING: causal curriculum has no cross-voice pool; collapsing to anon + orig.")
            n_orig_batch = remaining
            n_cv_batch = 0

        print(f"CAUSAL CURRICULUM batches: {n_orig_batch} orig + {n_cv_batch} cv + {n_anon_batch} anon "
              f"(batch_size={args.batch_size})")
        print(f"  Pools: orig={len(orig_pool_causal)} cv={len(cv_pool_causal)} anon={len(anon_pool_causal)}")

        orig_ds_causal = COGSDataset(
            orig_pool_causal, in_w2i, out_w2i,
            perm_classes=perm_classes if variant in ("B1", "B2", "B2b", "B2c", "B4", "B4_turbo", "B5", "B5_lite", "B6", "B7a", "B7c") else None,
            bidirectional=bidirectional)
        cv_ds_causal = COGSDataset(cv_pool_causal, in_w2i, out_w2i,
                                    perm_classes=None, bidirectional=bidirectional) if cv_pool_causal else None
        anon_ds_causal = COGSDataset(anon_pool_causal, in_w2i, out_w2i,
                                      perm_classes=None, bidirectional=bidirectional)

        class CausalTripleBatchIterator:
            def __init__(self, orig_ds, cv_ds, anon_ds, n_orig, n_cv, n_anon):
                self.orig_ds = orig_ds
                self.cv_ds = cv_ds
                self.anon_ds = anon_ds
                self.n_orig = n_orig
                self.n_cv = n_cv
                self.n_anon = n_anon
            def __len__(self):
                return len(self.orig_ds) // max(self.n_orig, 1)
            def __iter__(self):
                orig_idx = list(range(len(self.orig_ds)))
                random.shuffle(orig_idx)
                n_batches = len(orig_idx) // max(self.n_orig, 1)
                cv_pool = list(range(len(self.cv_ds))) if self.cv_ds is not None else []
                anon_pool = list(range(len(self.anon_ds)))
                for b in range(n_batches):
                    batch_orig_ids = orig_idx[b * self.n_orig:(b + 1) * self.n_orig]
                    items = [self.orig_ds[i] for i in batch_orig_ids]
                    if self.n_cv > 0 and cv_pool:
                        items += [self.cv_ds[random.choice(cv_pool)] for _ in range(self.n_cv)]
                    items += [self.anon_ds[random.choice(anon_pool)] for _ in range(self.n_anon)]
                    random.shuffle(items)
                    yield collate(items)

        tr_ld_causal = CausalTripleBatchIterator(
            orig_ds_causal, cv_ds_causal, anon_ds_causal,
            n_orig_batch, n_cv_batch, n_anon_batch)

    # Balanced curriculum: each batch has a fixed cv/orig ratio
    curriculum_balanced = getattr(args, "curriculum_balanced", False)
    cv_ratio = getattr(args, "cv_ratio", 0.3)
    tr_ld_balanced = None
    if curriculum_balanced:
        orig_pairs = [p for p in train_pairs
                      if not (len(p) >= 3 and isinstance(p[2], str) and p[2].startswith("cross_voice"))]
        cv_pairs = [p for p in train_pairs
                    if len(p) >= 3 and isinstance(p[2], str) and p[2].startswith("cross_voice")]
        if len(cv_pairs) == 0:
            print("WARNING: --curriculum-balanced specified but no cross_voice pairs in train. Disabling.")
            curriculum_balanced = False
        else:
            n_cv_batch = max(1, int(round(args.batch_size * cv_ratio)))
            n_orig_batch = args.batch_size - n_cv_batch
            print(f"BALANCED CURRICULUM: cv_ratio={cv_ratio} → each batch = "
                  f"{n_orig_batch} orig + {n_cv_batch} cv (batch_size={args.batch_size})")
            print(f"  Pools: {len(orig_pairs)} orig, {len(cv_pairs)} cv")
            orig_ds = COGSDataset(
                orig_pairs, in_w2i, out_w2i,
                perm_classes=perm_classes if variant in ("B1", "B2", "B2b", "B2c", "B4", "B4_turbo", "B5", "B5_lite", "B6", "B7a", "B7c") else None,
                bidirectional=bidirectional)
            cv_ds = COGSDataset(cv_pairs, in_w2i, out_w2i,
                                perm_classes=None, bidirectional=bidirectional)

            class BalancedBatchIterator:
                def __init__(self, orig_ds, cv_ds, n_orig, n_cv):
                    self.orig_ds = orig_ds; self.cv_ds = cv_ds
                    self.n_orig = n_orig; self.n_cv = n_cv
                def __len__(self):
                    return len(self.orig_ds) // self.n_orig
                def __iter__(self):
                    orig_idx = list(range(len(self.orig_ds)))
                    random.shuffle(orig_idx)
                    n_batches = len(orig_idx) // self.n_orig
                    cv_pool = list(range(len(self.cv_ds)))
                    for b in range(n_batches):
                        batch_orig_ids = orig_idx[b * self.n_orig:(b + 1) * self.n_orig]
                        batch_cv_ids = [random.choice(cv_pool) for _ in range(self.n_cv)]
                        items = ([self.orig_ds[i] for i in batch_orig_ids] +
                                 [self.cv_ds[i] for i in batch_cv_ids])
                        random.shuffle(items)
                        yield collate(items)

            tr_ld_balanced = BalancedBatchIterator(orig_ds, cv_ds, n_orig_batch, n_cv_batch)

    # Reversed curriculum: Phase 1 = cross-voice examples only, then Phase 2 = full train
    curriculum_passive = getattr(args, "curriculum_passive_first", False)
    curriculum_switch = getattr(args, "curriculum_switch_epoch", 10)
    tr_ld_phase1 = None
    if curriculum_passive:
        cv_only_pairs = [p for p in train_pairs
                         if len(p) >= 3 and isinstance(p[2], str) and p[2].startswith("cross_voice")]
        if len(cv_only_pairs) == 0:
            print("WARNING: --curriculum-passive-first specified but no cross_voice pairs in train. Disabling curriculum.")
            curriculum_passive = False
        else:
            print(f"REVERSED CURRICULUM: Phase 1 (epochs 0-{curriculum_switch - 1}) = "
                  f"{len(cv_only_pairs)} cross-voice pairs ONLY; "
                  f"Phase 2 (ep {curriculum_switch}+) = full train ({len(train_pairs)} pairs).")
            tr_ds_phase1 = COGSDataset(cv_only_pairs, in_w2i, out_w2i,
                                       perm_classes=None, bidirectional=bidirectional)
            tr_ld_phase1 = DataLoader(tr_ds_phase1, args.batch_size, shuffle=True,
                                       collate_fn=collate, num_workers=0, pin_memory=True)

    # Clip very long targets from vocab-side max_out (safety)
    mx_in = max(MAX_IN, max(len(p[0].split()) for p in train_pairs + dev_pairs + gen_pairs) + 2)
    mx_out = max(MAX_OUT, max(len(p[1].split()) for p in train_pairs + dev_pairs + gen_pairs) + 4)
    if bidirectional:
        # Symmetrize: either direction might see either length
        mx = max(mx_in, mx_out)
        mx_in = mx_out = mx
    use_copy = getattr(args, "copy", False)
    use_tags = getattr(args, "tags", False)
    use_jepa = bool(getattr(args, "jepa", False))
    use_jepa_ema = bool(getattr(args, "jepa_ema", False))
    jepa_ema_decay = float(getattr(args, "jepa_ema_decay", 0.99))
    if use_tags:
        in_to_out = build_in_to_out_map(in_w2i, out_w2i)
        n_shared = (in_to_out >= 0).sum().item()
        token_cats = build_token_categories(in_w2i)
        print(f"Copy+Tags mechanism ON: {n_shared}/{len(in_w2i)} input tokens mapped")
        model = TransformerSeq2SeqCopyTags(len(in_w2i), len(out_w2i),
                                           d_model=args.d_model,
                                           n_heads=args.n_heads,
                                           n_layers=args.n_layers,
                                           ffn=args.ffn,
                                           max_in=mx_in, max_out=mx_out,
                                           in_to_out_map=in_to_out,
                                           token_categories=token_cats,
                                           use_jepa=use_jepa,
                                           use_jepa_ema=use_jepa_ema,
                                           jepa_ema_decay=jepa_ema_decay).to(device)
        use_copy = True     # tags implies copy
    elif use_copy:
        in_to_out = build_in_to_out_map(in_w2i, out_w2i)
        n_shared = (in_to_out >= 0).sum().item()
        print(f"Copy mechanism ON: {n_shared}/{len(in_w2i)} input tokens mapped to output vocab")
        model = TransformerSeq2SeqWithCopy(len(in_w2i), len(out_w2i),
                                           d_model=args.d_model,
                                           n_heads=args.n_heads,
                                           n_layers=args.n_layers,
                                           ffn=args.ffn,
                                           max_in=mx_in, max_out=mx_out,
                                           in_to_out_map=in_to_out,
                                           use_jepa=use_jepa,
                                           use_jepa_ema=use_jepa_ema,
                                           jepa_ema_decay=jepa_ema_decay).to(device)
    else:
        model = TransformerSeq2Seq(len(in_w2i), len(out_w2i),
                                   d_model=args.d_model,
                                   n_heads=args.n_heads,
                                   n_layers=args.n_layers,
                                   ffn=args.ffn,
                                   max_in=mx_in, max_out=mx_out,
                                   use_jepa=use_jepa,
                                   use_jepa_ema=use_jepa_ema,
                                   jepa_ema_decay=jepa_ema_decay).to(device)
    if use_jepa:
        jl = float(getattr(args, "jepa_lambda", 0.1))
        ema_str = f" + EMA target (decay={jepa_ema_decay})" if use_jepa_ema else ""
        print(f"JEPA predictor ON: lambda={jl}{ema_str} (loss = task + lambda * jepa)")
    # Bidirectional: tie input/output embeddings (single embedding for unified vocab)
    if bidirectional:
        model.out_emb.weight = model.in_emb.weight
        # Tie positional embeddings if shapes match
        if model.in_pe.weight.shape == model.out_pe.weight.shape:
            model.out_pe.weight = model.in_pe.weight
        # For copy: in_to_out is identity on unified vocab
        if hasattr(model, "in_to_out_map"):
            model.in_to_out_map = torch.arange(len(in_w2i), dtype=torch.long).to(device)
        print("  BIDIRECTIONAL: embeddings tied (in_emb = out_emb)")

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

    # Innovation B — Selective forgetting setup
    sel_forget = getattr(args, "selective_forgetting", False)
    forget_mode = getattr(args, "forgetting_mode", "permute")
    forget_every = getattr(args, "forgetting_every", 20)
    forget_duration = getattr(args, "forgetting_duration", 1)
    forget_start = getattr(args, "forgetting_start", 0)
    if sel_forget:
        print(f"SELECTIVE FORGETTING ON: mode={forget_mode} start@ep{forget_start} "
              f"cycle={forget_every}ep ({forget_duration}ep perturbed + {forget_every - forget_duration}ep clean)")

    # Phase A.5 — JEPA trajectory sampling. Pick fixed subsets once, sample
    # surprise every 5 epochs. Stratified by category on gen for the per-cat
    # decomposition. Only active when use_jepa is True.
    jepa_trajectory = []
    jepa_train_sample_idx = None
    jepa_gen_sample_idx_by_cat = None
    # TEST 3 — gradient conflict snapshots (per-epoch on fixed val batch)
    grad_snapshots = []
    grad_snapshots_out = getattr(args, "grad_snapshots_out", None)
    grad_fixed_batch = None
    if grad_snapshots_out and use_jepa:
        # Take a fixed batch from val set for reproducible grad measurements
        try:
            grad_fixed_batch = next(iter(dv_ld))
            print(f"GRAD SNAPSHOTS ON: writing to {grad_snapshots_out} after each epoch")
        except StopIteration:
            grad_fixed_batch = None
    if use_jepa:
        rng_sample = random.Random(seed + 1)
        jepa_train_sample_idx = rng_sample.sample(
            range(len(tr_ds)), min(200, len(tr_ds)))
        # Stratify gen by category: at most 200 examples total, evenly split
        cat_to_idx = {}
        for i in range(len(ge_ds)):
            cat_to_idx.setdefault(ge_ds.cats[i], []).append(i)
        n_cat = len(cat_to_idx)
        per_cat = max(1, 200 // max(n_cat, 1))
        jepa_gen_sample_idx_by_cat = {}
        for cat, idxs in cat_to_idx.items():
            sample_idxs = rng_sample.sample(idxs, min(per_cat, len(idxs)))
            jepa_gen_sample_idx_by_cat[cat] = sample_idxs

    for ep in range(args.epochs):
        t_ep = time.time()
        model.train()
        tr_loss = 0.0; n_batches = 0
        tr_jepa_loss = 0.0  # Phase A.5: per-epoch JEPA loss (averaged at log time)

        # Innovation B — warmup until forget_start, then cycles of `forget_duration` perturbed
        # epochs followed by `forget_every - forget_duration` clean epochs.
        # Mode=gentle is single-shot: fires only in [forget_start, forget_start + forget_duration),
        # then never again regardless of forget_every (free reconstruction phase).
        if sel_forget and ep >= forget_start:
            if forget_mode == "gentle":
                forget_active = ep < forget_start + forget_duration
            else:
                cycle_pos = (ep - forget_start) % forget_every
                forget_active = cycle_pos < forget_duration
        else:
            forget_active = False
        # Gentle-mode one-time marker when the single shot window ends
        if sel_forget and forget_mode == "gentle" and ep == forget_start + forget_duration:
            print(f"  [forget:gentle] ep={ep} — perturbation over, free reconstruction begins")
        if forget_active:
            with torch.no_grad():
                if forget_mode in ("permute", "gentle"):
                    # Permute positional embeddings to disrupt position→role binding.
                    # 'gentle' uses the same perturbation but fires only once (single shot).
                    L_in = model.in_pe.weight.size(0)
                    perm_in = torch.randperm(L_in, device=model.in_pe.weight.device)
                    model.in_pe.weight.data = model.in_pe.weight.data[perm_in]
                    if not bidirectional:
                        L_out = model.out_pe.weight.size(0)
                        perm_out = torch.randperm(L_out, device=model.out_pe.weight.device)
                        model.out_pe.weight.data = model.out_pe.weight.data[perm_out]
                    print(f"  [forget:{forget_mode}] ep={ep} — positional embeddings permuted")
                elif forget_mode == "dropout":
                    # Heavy stochastic zero-out of pos embeddings (80%)
                    mask = (torch.rand_like(model.in_pe.weight) > 0.8).float()
                    model.in_pe.weight.data.mul_(mask)
                    if not bidirectional:
                        mask_o = (torch.rand_like(model.out_pe.weight) > 0.8).float()
                        model.out_pe.weight.data.mul_(mask_o)
                    print(f"  [forget:dropout] ep={ep} — 80% of pos embeddings zeroed")
                elif forget_mode == "reset":
                    # Reinit first encoder self-attention (where position→role binding is formed)
                    layer0 = model.encoder.layers[0]
                    if hasattr(layer0.self_attn, "in_proj_weight") and layer0.self_attn.in_proj_weight is not None:
                        nn.init.xavier_uniform_(layer0.self_attn.in_proj_weight)
                    if hasattr(layer0.self_attn, "out_proj"):
                        nn.init.xavier_uniform_(layer0.self_attn.out_proj.weight)
                    print(f"  [forget:reset] ep={ep} — first encoder self-attn reinitialized")

        # Scheduled sampling (copy only): TFR = 1.0 until ep=tfr_start, then linear 1.0→0.5 up to end
        tfr_start = getattr(args, "tfr_start", 20)
        if use_copy and ep >= tfr_start:
            progress = min(max((ep - tfr_start) / max(args.epochs - tfr_start, 1), 0.0), 1.0)
            tfr = 1.0 - 0.5 * progress
        else:
            tfr = 1.0

        # Curriculum selection priority: causal > balanced > passive-first > default
        if causal_curriculum and tr_ld_causal is not None:
            current_tr_ld = tr_ld_causal
            if ep == 0:
                print(f"  [curriculum] causal 3-track batches for {args.epochs} epochs")
        elif curriculum_balanced and tr_ld_balanced is not None:
            current_tr_ld = tr_ld_balanced
            if ep == 0:
                print(f"  [curriculum] balanced batches for {args.epochs} epochs")
        elif curriculum_passive and tr_ld_phase1 is not None and ep < curriculum_switch:
            current_tr_ld = tr_ld_phase1
            if ep == 0:
                print(f"  [curriculum] Phase 1 begins — cv-only training for {curriculum_switch} epochs")
        else:
            current_tr_ld = tr_ld
            if curriculum_passive and ep == curriculum_switch:
                print(f"  [curriculum] Phase 2 begins ep={ep} — switching to full train")

        for src, src_mask, tgt, _, _ in current_tr_ld:
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
                # Innovation C — Contrastive errors (per-token margin loss)
                # Push gold log-prob above the top-1 non-gold competitor by `margin`.
                # No extra forward, uses logits already computed. Masks PAD positions.
                if getattr(args, "contrastive_errors", False):
                    c_margin = 1.0
                    c_lambda = 0.2
                    if info.get("uses_copy"):
                        log_probs_all = logits  # already log-probs under copy
                    else:
                        log_probs_all = F.log_softmax(logits, dim=-1)
                    gold_lp = log_probs_all.gather(-1, tgt_out.unsqueeze(-1)).squeeze(-1)  # (B, T)
                    top2_lp, top2_idx = log_probs_all.topk(2, dim=-1)                     # (B, T, 2)
                    is_gold_top1 = (top2_idx[..., 0] == tgt_out)
                    competitor_lp = torch.where(is_gold_top1, top2_lp[..., 1], top2_lp[..., 0])
                    c_per_tok = F.relu(competitor_lp - gold_lp + c_margin)
                    c_mask = (tgt_out != 0).float()
                    c_loss = (c_per_tok * c_mask).sum() / c_mask.sum().clamp(min=1.0)
                    loss = loss + c_lambda * c_loss
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

                # JEPA auxiliary loss (Phase A integration). Predictor sits on
                # encoder output; loss is masked MSE between net(enc[:-1]) and
                # detached enc[1:] (or EMA encoder output if use_jepa_ema).
                # Adds args.jepa_lambda × jepa_loss to the main loss.
                jepa_loss_value = 0.0
                if getattr(model, "use_jepa", False):
                    enc_out = info.get("enc_out")
                    if enc_out is not None:
                        if getattr(model, "use_jepa_ema", False):
                            target_enc = model.encode_ema(src, src_mask)
                            pred_j, target_j = model.jepa_predictor(enc_out, target_enc)
                        else:
                            pred_j, target_j = model.jepa_predictor(enc_out)
                        jepa_loss = _jepa_loss(pred_j, target_j, src_mask)
                        loss = loss + float(getattr(args, "jepa_lambda", 0.1)) * jepa_loss
                        jepa_loss_value = float(jepa_loss.detach().item())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            # EMA update of encoder_ema after the optimizer step
            if getattr(model, "use_jepa_ema", False):
                model.update_ema()
            tr_loss += loss.item(); n_batches += 1
            tr_jepa_loss += jepa_loss_value

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
        if use_jepa:
            entry["jepa_loss_train_avg"] = tr_jepa_loss / max(n_batches, 1)
        metrics_log.append(entry)

        # TEST 3 — gradient conflict snapshot (per-epoch on fixed val batch)
        if grad_snapshots_out and use_jepa and grad_fixed_batch is not None:
            try:
                snap = _grad_snapshot(model, grad_fixed_batch, device, use_copy)
                if snap is not None:
                    snap["epoch"] = ep
                    grad_snapshots.append(snap)
            except Exception as e:
                print(f"  grad_snapshot ep{ep} skipped: {e}")

        # Phase A.5 — sample JEPA surprise on fixed train + stratified gen subsets
        # at ep 0, 5, 10, … and at the last epoch.
        if use_jepa and (ep % 5 == 0 or ep == args.epochs - 1):
            model.eval()
            with torch.no_grad():
                def _surprise_for_ds(ds, idxs):
                    if not idxs:
                        return None
                    items = [ds[i] for i in idxs]
                    src_b, src_m_b, tgt_b, _, _ = collate(items)
                    src_b = src_b.to(device); src_m_b = src_m_b.to(device)
                    enc = model.encode(src_b, src_m_b)
                    pred_j, target_j = model.jepa_predictor(enc)
                    sq = (pred_j - target_j) ** 2
                    mask = src_m_b[:, 1:].unsqueeze(-1).float()
                    denom = mask.sum().clamp(min=1.0) * pred_j.size(-1)
                    return float(((sq * mask).sum() / denom).item())
                tr_s = _surprise_for_ds(tr_ds, jepa_train_sample_idx)
                gen_by_cat = {}
                gen_all_pairs = []
                for cat, idxs in jepa_gen_sample_idx_by_cat.items():
                    s = _surprise_for_ds(ge_ds, idxs)
                    if s is not None:
                        gen_by_cat[cat] = s
                        gen_all_pairs.append((cat, len(idxs), s))
                # weighted mean across categories
                if gen_all_pairs:
                    total_n = sum(n for _c, n, _s in gen_all_pairs)
                    gen_s = sum(s * n for _c, n, s in gen_all_pairs) / max(total_n, 1)
                else:
                    gen_s = None
                jepa_trajectory.append({
                    "epoch": ep,
                    "train_surprise_sample_mean": tr_s,
                    "gen_surprise_sample_mean": gen_s,
                    "gen_surprise_by_category": gen_by_cat,
                })
                with open(os.path.join(run_dir, "jepa_trajectory.json"), "w") as f:
                    json.dump(jepa_trajectory, f, indent=1)
            model.train()
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
        jepa_str = f" jepa={tr_jepa_loss/max(n_batches,1):.4f}" if use_jepa else ""
        print(f"E{ep:03d} {mark} loss={tr_loss/max(n_batches,1):.4f}{jepa_str} "
              f"dev_ex={dev_m['exact']:5.2f}% gen_ex={gen_m['exact']:5.2f}% "
              f"tok_g={gen_m['tok_acc']:5.2f}% lr={sched.get_last_lr()[0]:.2e}{tfr_str} "
              f"pat={patience} | {dt:.1f}s ETA {eta_m}m{eta_sec:02d}s",
              flush=True)

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

    # Optional: constrained beam search eval (Innovation: --constrained-beam)
    constrained_gen = None
    if getattr(args, "constrained_beam", False):
        beam_size = getattr(args, "beam_size", 10)
        print(f"Running constrained beam search eval (k={beam_size}) on gen...")
        constraints = extract_train_constraints(train_pairs)
        in_i2w = {v: k for k, v in in_w2i.items()}
        out_i2w = {v: k for k, v in out_w2i.items()}
        diag_cats = {"active_to_passive", "passive_to_active",
                     "do_dative_to_pp_dative", "pp_dative_to_do_dative",
                     "unacc_to_transitive", "obj_omitted_transitive_to_transitive",
                     "cp_recursion"}
        print(f"  Constraints: {len(constraints['always_agent'])} always_agent, "
              f"{len(constraints['always_theme'])} always_theme, "
              f"{len(constraints['known_verbs'])} verbs, "
              f"{len(constraints['proper_names'])} proper_names")
        fast_mode = getattr(args, "fast", False)
        beam_max = getattr(args, "beam_max_examples", 0)
        prog_every = 500
        if fast_mode:
            if beam_max == 0:
                beam_max = 500
            prog_every = 50
            print(f"  FAST MODE: evaluating {beam_max} examples only, progress every {prog_every}")
        constrained_gen = evaluate(model, ge_ld, out_w2i, device,
                                   constraints=constraints, beam_size=beam_size,
                                   in_i2w=in_i2w, out_i2w=out_i2w, diag_cats=diag_cats,
                                   max_examples=beam_max, progress_every=prog_every)
        print(f"  Constrained-beam gen_ex={constrained_gen['exact']:.2f}% "
              f"(greedy was {greedy_gen['exact']:.2f}%)")
        if "by_cat" in constrained_gen:
            print("  Per-category gen (constrained beam):")
            for cat, pct in sorted(constrained_gen["by_cat"].items(), key=lambda x: -x[1]):
                print(f"    {cat:50s} {pct:6.2f}%")
        if constrained_gen.get("violations"):
            print("  Violations counted over all beams considered:")
            for k, v in sorted(constrained_gen["violations"].items(), key=lambda x: -x[1]):
                print(f"    {k:20s} {v}")
        # Beam rank diagnostic on structural categories
        if constrained_gen.get("beam_rank_log"):
            brl = constrained_gen["beam_rank_log"]
            from collections import defaultdict
            by_c = defaultdict(list)
            for cat, rank, nb in brl:
                by_c[cat].append(rank)
            print("  Gold rank in beams (structural categories):")
            for cat in sorted(by_c):
                ranks = by_c[cat]
                found = [r for r in ranks if r >= 0]
                notfound = len(ranks) - len(found)
                if found:
                    print(f"    {cat:50s} gold found in beams: {len(found)}/{len(ranks)} "
                          f"(median rank {sorted(found)[len(found)//2]})")
                else:
                    print(f"    {cat:50s} gold NEVER in top-{beam_size} beams ({notfound} examples)")

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
    if constrained_gen is not None:
        summary["final_gen_exact_constrained_beam"] = constrained_gen["exact"]
        summary["constrained_beam_by_cat"] = constrained_gen.get("by_cat", {})
        summary["constrained_beam_violations"] = constrained_gen.get("violations", {})
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    ckpt_dict = {"state_dict": model.state_dict(), "in_w2i": in_w2i,
                 "out_w2i": out_w2i, "variant": variant,
                 "d_model": args.d_model,
                 "n_layers": args.n_layers,
                 "n_heads": args.n_heads,
                 "ffn": args.ffn,
                 "max_in": model.max_in, "max_out": model.max_out,
                 "use_copy": use_copy, "use_tags": use_tags,
                 "use_jepa": use_jepa,
                 "use_jepa_ema": bool(getattr(model, "use_jepa_ema", False)),
                 "jepa_ema_decay": float(getattr(model, "jepa_ema_decay", 0.99))}
    if use_jepa:
        ckpt_dict["jepa_state_dict"] = model.jepa_predictor.state_dict()
        ckpt_dict["jepa_lambda"] = float(getattr(args, "jepa_lambda", 0.1))
    torch.save(ckpt_dict, os.path.join(run_dir, "checkpoint.pt"))

    # Phase A.5 — write jepa_curves.png if any trajectory was recorded
    if use_jepa and jepa_trajectory:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            xs = [e["epoch"] for e in jepa_trajectory]
            ys_train = [e.get("train_surprise_sample_mean") for e in jepa_trajectory]
            ys_gen = [e.get("gen_surprise_sample_mean") for e in jepa_trajectory]
            fig, ax = plt.subplots(figsize=(8, 4.5))
            if any(y is not None for y in ys_train):
                ax.plot(xs, [y if y is not None else float("nan") for y in ys_train],
                        label="train (200 sampled)", color="tab:blue")
            if any(y is not None for y in ys_gen):
                ax.plot(xs, [y if y is not None else float("nan") for y in ys_gen],
                        label="gen (stratified 200)", color="tab:red")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("JEPA surprise (mean MSE / d_model)")
            ax.set_title("JEPA surprise trajectory")
            ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(run_dir, "jepa_curves.png"), dpi=120)
            plt.close(fig)
            print(f"  wrote {os.path.join(run_dir, 'jepa_curves.png')}")
        except Exception as e:
            print(f"  jepa_curves.png skipped: {e}")

    # TEST 3 — dump grad_snapshots
    if grad_snapshots_out and grad_snapshots:
        os.makedirs(os.path.dirname(grad_snapshots_out) or ".", exist_ok=True)
        with open(grad_snapshots_out, "w") as f:
            json.dump(grad_snapshots, f, indent=2)
        print(f"  wrote {grad_snapshots_out} ({len(grad_snapshots)} epochs)")

    print(f"\nFinished {bench_name} {variant}/s{seed}: "
          f"best_gen_tf={summary['best_gen_exact_tf']:.2f}%  "
          f"final_gen_greedy={summary['final_gen_exact_greedy']:.2f}%")
    return summary


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def build_arg_parser(description="COGS compositional generalization"):
    """Build the full argparse parser with every mechanism flag. Shared by
    cogs_compositional.py (__main__) and slog_compositional.py so that SLOG
    benefits from every augmentation added to COGS automatically."""
    p = argparse.ArgumentParser(description=description)
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
    p.add_argument("--n-layers", type=int, default=2,
                   help="Number of encoder AND decoder layers (symmetric, default 2+2)")
    p.add_argument("--n-heads", type=int, default=4,
                   help="Number of attention heads (default 4). Must divide d-model.")
    p.add_argument("--ffn", type=int, default=256,
                   help="Feed-forward dimension (default 256). Rule of thumb: 4 × d-model.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--runs-dir", type=str, default=None,
                   help="Override output directory for runs")
    p.add_argument("--copy", action="store_true",
                   help="Enable copy mechanism (pointer network) in decoder")
    p.add_argument("--tags", action="store_true",
                   help="Enable structural role tagger (implies --copy)")
    p.add_argument("--bidirectional", action="store_true",
                   help="Train both phrase→LF and LF→phrase (50/50). Uses unified vocab.")
    p.add_argument("--selective-forgetting", action="store_true",
                   help="Innovation B: periodically perturb positional bindings to force re-learning position→role")
    p.add_argument("--forgetting-mode", type=str, default="permute",
                   choices=["dropout", "permute", "reset", "gentle"],
                   help="Perturbation type for --selective-forgetting. 'gentle' = single-shot permute at forgetting_start for forgetting_duration epochs, then free reconstruction (no restore).")
    p.add_argument("--forgetting-every", type=int, default=20,
                   help="Cycle length (in epochs) for selective forgetting")
    p.add_argument("--forgetting-duration", type=int, default=1,
                   help="Number of consecutive epochs within each cycle to apply perturbation")
    p.add_argument("--forgetting-start", type=int, default=0,
                   help="Epoch at which selective-forgetting cycles begin (warmup before)")
    p.add_argument("--peel-stack", action="store_true",
                   help="Innovation K: train-time decomposition of PP-containing examples into sub-problems")
    p.add_argument("--peel-stack-depth", type=int, default=3,
                   help="Number of cascade iterations for peel-stack (each adds +1 PP depth). Default 3 → depths 3,4,5.")
    p.add_argument("--peel-cp", action="store_true",
                   help="CP recursion peel: wrap depth-N ccomp examples in an extra 'NAME V that …' layer. Cascade count controlled by --peel-cp-depth.")
    p.add_argument("--peel-cp-depth", type=int, default=3,
                   help="Number of cascade iterations for --peel-cp (default 3 → clause depths up to 3+original).")
    p.add_argument("--wh-passive-aug", action="store_true",
                   help="Augment train with wh-questions on passives: 'The X was V-ed by Y .' → 'What was V-ed by Y ?' with ? in theme position.")
    p.add_argument("--wh-passive-n", type=int, default=0,
                   help="Cap number of wh-passive examples (0 = all derivable)")
    p.add_argument("--contrastive-errors", action="store_true",
                   help="Innovation C: per-token margin loss pushing gold above top-1 competitor (lambda=0.2, margin=1.0)")
    p.add_argument("--constrained-beam", action="store_true",
                   help="Inference-only: beam search + re-ranking with train-set constraints")
    p.add_argument("--beam-size", type=int, default=10,
                   help="Beam size for --constrained-beam (default 10)")
    p.add_argument("--beam-max-examples", type=int, default=0,
                   help="Cap number of gen examples evaluated under --constrained-beam (0 = all)")
    p.add_argument("--fast", action="store_true",
                   help="Quick diagnostic mode for --constrained-beam: caps at 500 examples and prints progress every 50")
    p.add_argument("--cross-voice", action="store_true",
                   help="Innovation: add active↔passive augmentation pairs to train")
    p.add_argument("--cross-voice-n", type=int, default=0,
                   help="Cap on number of cross-voice pairs added (0 = all available)")
    p.add_argument("--cross-voice-oversample", type=int, default=1,
                   help="Repeat cross-voice pairs N times in train (to up-weight them)")
    p.add_argument("--cross-voice-morphology", action="store_true",
                   help="For verbs in pt_map but missing in pp_map, use regular -ed heuristic to derive pp (e.g. bless → blessed). Enables passives for active-only verbs.")
    p.add_argument("--cv-boost-bless-squeeze", action="store_true",
                   help="Add N targeted cross-voice pairs: bless in passive + squeeze in active. N controlled by --cv-boost-n.")
    p.add_argument("--cv-boost-n", type=int, default=50,
                   help="Number of targeted examples per boost verb (default 50)")
    p.add_argument("--cv-boost-squeeze", type=int, default=0,
                   help="Add N targeted cross-voice pairs: squeeze in ACTIVE (derived from train passives). 0 = disabled.")
    p.add_argument("--cv-boost-bless", type=int, default=0,
                   help="Add N targeted cross-voice pairs: bless in PASSIVE (derived from train actives, needs morphology fallback). 0 = disabled.")
    p.add_argument("--causal-curriculum", action="store_true",
                   help="3-track causal curriculum: every batch contains anonymized (structure) + orig + cross-voice examples. Requires --cross-voice.")
    p.add_argument("--causal-ratio", type=float, default=0.33,
                   help="Fraction of anonymized examples per batch (default 0.33 = 1/3). Remaining split 50/50 between orig and cv.")
    p.add_argument("--causal-n-anon", type=int, default=0,
                   help="Cap the size of the anon pool (0 = use all anonymizable examples from train+cv)")
    p.add_argument("--permute-verbs", action="store_true",
                   help="A4-style verb permutation: random bijection on transitive verbs applied per __getitem__ (prob 0.5). Shuffles verb identity while keeping structure.")
    p.add_argument("--permute-verbs-prob", type=float, default=0.5,
                   help="Probability per example of applying the verb permutation (default 0.5)")
    p.add_argument("--filter-unaccusative-from-pool", action="store_true",
                   help="Transfer test (COGS): remove the SLOG-derived list of unaccusative verbs "
                        "(break, burn, collapse, disintegrate, float, freeze, grow, roll, shatter, shorten) "
                        "from the permutation pool, with vocab check. Writes pool_used.txt to the run dir.")
    p.add_argument("--jepa", action="store_true",
                   help="Activate JEPA predictor during training (Phase A integration)")
    p.add_argument("--jepa-lambda", type=float, default=0.1,
                   help="Coefficient of the JEPA loss (default 0.1)")
    p.add_argument("--jepa-ema", action="store_true",
                   help="Use EMA encoder copy as JEPA target (more stable). "
                        "Decay set by --jepa-ema-decay.")
    p.add_argument("--jepa-ema-decay", type=float, default=0.99,
                   help="EMA decay rate for the JEPA target encoder (default 0.99)")
    p.add_argument("--extra-train-file", default=None,
                   help="Path to a TSV with extra train pairs (Phase A.2). "
                        "Same format as train.tsv (input\\tlf\\tcategory). "
                        "Concatenated to train_pairs after the base load.")
    p.add_argument("--grad-snapshots-out", default=None,
                   help="Path to JSON output for per-epoch gradient cos_sim "
                        "snapshots (TEST 3 diagnostic). Requires --jepa.")
    p.add_argument("--tfr-start", type=int, default=20,
                   help="Epoch at which scheduled sampling begins ramping TFR from 1.0→0.5 (copy only)")
    p.add_argument("--curriculum-passive-first", action="store_true",
                   help="Reversed curriculum: train on cross-voice pairs ONLY for --curriculum-switch-epoch epochs, then switch to full train (requires --cross-voice)")
    p.add_argument("--curriculum-switch-epoch", type=int, default=10,
                   help="Epoch at which phase 1 (cv-only) transitions to phase 2 (full train)")
    p.add_argument("--curriculum-balanced", action="store_true",
                   help="Every batch has a fixed cv/orig ratio (requires --cross-voice). See --cv-ratio.")
    p.add_argument("--cv-ratio", type=float, default=0.3,
                   help="Fraction of cross-voice examples per batch when --curriculum-balanced is on")
    return p


if __name__ == "__main__":
    p = build_arg_parser("COGS compositional generalization")
    args = p.parse_args()
    train_one_run(args.variant, args.seed, args)
