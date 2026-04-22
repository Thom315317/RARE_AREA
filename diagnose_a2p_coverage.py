#!/usr/bin/env python3
"""Diagnostic: for active_to_passive gen examples, which verbs are covered
by passive examples in train?

Usage:
  python3 diagnose_a2p_coverage.py
"""
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "data", "cogs", "train.tsv")
GEN = os.path.join(HERE, "data", "cogs", "gen.tsv")


def extract_verb_from_lf(lf):
    """Return first verb lemma found before '.agent' or '.theme' in LF."""
    toks = lf.split()
    for j in range(len(toks) - 2):
        if toks[j + 1] == "." and toks[j + 2] in ("agent", "theme"):
            return toks[j]
    return None


def load_tsv(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                rows.append((parts[0], parts[1], parts[2]))
    return rows


# Load train
train = load_tsv(TRAIN)
gen = load_tsv(GEN)

# Verbs seen in passive in train (input contains 'was' + 'by')
train_passive_verbs = set()
train_active_verbs = set()
for inp, lf, _ in train:
    verb = extract_verb_from_lf(lf)
    if verb is None:
        continue
    toks = inp.split()
    if "was" in toks and "by" in toks:
        train_passive_verbs.add(verb)
    else:
        train_active_verbs.add(verb)

# Gen: active_to_passive and passive_to_active
gen_a2p = [(inp, lf) for inp, lf, cat in gen if cat == "active_to_passive"]
gen_p2a = [(inp, lf) for inp, lf, cat in gen if cat == "passive_to_active"]

gen_a2p_verbs = Counter()
gen_p2a_verbs = Counter()
for inp, lf in gen_a2p:
    v = extract_verb_from_lf(lf)
    if v:
        gen_a2p_verbs[v] += 1
for inp, lf in gen_p2a:
    v = extract_verb_from_lf(lf)
    if v:
        gen_p2a_verbs[v] += 1


def classify(gen_verbs, train_passive_verbs, train_active_verbs, label, opposite_direction_msg):
    unique_verbs = set(gen_verbs.keys())
    seen_in_passive = unique_verbs & train_passive_verbs
    not_seen_in_passive = unique_verbs - train_passive_verbs
    seen_only_in_active = unique_verbs & train_active_verbs - train_passive_verbs
    not_seen_at_all = unique_verbs - train_passive_verbs - train_active_verbs

    total_examples = sum(gen_verbs.values())
    covered_examples = sum(gen_verbs[v] for v in seen_in_passive)
    uncovered_examples = total_examples - covered_examples

    print(f"\n=== {label} ===")
    print(f"Total gen examples: {total_examples}")
    print(f"Unique verbs in gen: {len(unique_verbs)}")
    print(f"  Seen in passive in train: {len(seen_in_passive)}")
    print(f"  NOT seen in passive in train: {len(not_seen_in_passive)}")
    print(f"    └─ seen only in active: {len(seen_only_in_active)}")
    print(f"    └─ not seen at all: {len(not_seen_at_all)}")
    print(f"\nExample-level coverage:")
    print(f"  Covered (verb has passive form in train): {covered_examples}/{total_examples} "
          f"({100*covered_examples/max(total_examples,1):.1f}%)")
    print(f"  UNcovered ({opposite_direction_msg}): {uncovered_examples}/{total_examples} "
          f"({100*uncovered_examples/max(total_examples,1):.1f}%)")

    print(f"\nTop 15 uncovered verbs (by frequency in gen):")
    uncovered_verb_list = [(v, c) for v, c in gen_verbs.most_common() if v in not_seen_in_passive]
    for v, c in uncovered_verb_list[:15]:
        status = "active-only" if v in seen_only_in_active else "unseen"
        print(f"  {v:20s} {c:4d} ({status})")

    print(f"\nTop 15 covered verbs (by frequency in gen):")
    for v, c in gen_verbs.most_common():
        if v in seen_in_passive:
            print(f"  {v:20s} {c:4d}")


classify(gen_a2p_verbs, train_passive_verbs, train_active_verbs,
         "active_to_passive (tested as passives in gen)",
         "verb never in passive form in train")

# For p2a, the test sentences are actives; we need the opposite check:
# which verbs are covered by active examples in train?
classify(gen_p2a_verbs, train_active_verbs, train_passive_verbs,
         "passive_to_active (tested as actives in gen)",
         "verb never in active form in train")

print(f"\n=== Summary ===")
print(f"train_passive_verbs (usable for a2p via cross-voice): {len(train_passive_verbs)}")
print(f"train_active_verbs (usable for p2a via cross-voice): {len(train_active_verbs)}")
print(f"Intersection (both voices seen): {len(train_passive_verbs & train_active_verbs)}")
