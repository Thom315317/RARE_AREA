#!/usr/bin/env python3
"""Diagnostic: for each structural generalization category in the gen set,
compute verb coverage against the permute-verbs pool (plus custom valency
analysis for active_to_passive / passive_to_active / unacc_to_transitive).

Works on either COGS or SLOG via --dataset.

Usage:
  python3 diagnose_a2p_coverage.py --dataset cogs
  python3 diagnose_a2p_coverage.py --dataset slog
  python3 diagnose_a2p_coverage.py --dataset slog --all-structural  # print coverage for every structural category found
"""
import os
import argparse
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))

# Categories that behave like COGS voice/valency alternations and that we
# specifically analyse with the passive/active/transitive decomposition.
VOICE_CATEGORIES = {
    "active_to_passive",
    "passive_to_active",
    "unacc_to_transitive",
    "obj_omitted_transitive_to_transitive",
    "do_dative_to_pp_dative",
    "pp_dative_to_do_dative",
}

# Heuristic: any gen category whose name contains one of these keywords is
# treated as "structural" and gets a pool-coverage summary when
# --all-structural is passed.
STRUCTURAL_KEYWORDS = (
    "passive", "active", "dative", "transitive", "unacc", "recursion", "cp_",
    "pp_", "center", "question", "relative",
)


def extract_verb_from_lf(lf):
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


def dataset_paths(dataset):
    if dataset == "cogs":
        return {
            "train": os.path.join(HERE, "data", "cogs", "train.tsv"),
            "gen":   os.path.join(HERE, "data", "cogs", "gen.tsv"),
        }
    if dataset == "slog":
        slog_root = os.path.join(HERE, "data", "slog")
        # SLOG's gen file lives under generalization_sets/gen_cogsLF.tsv.
        gen_candidates = [
            os.path.join(slog_root, "generalization_sets", "gen_cogsLF.tsv"),
            os.path.join(slog_root, "gen.tsv"),
        ]
        gen_path = next((p for p in gen_candidates if os.path.exists(p)), gen_candidates[0])
        return {"train": os.path.join(slog_root, "train.tsv"), "gen": gen_path}
    raise ValueError(f"Unknown dataset: {dataset}")


# ══════════════════════════════════════════════════════════════
# Pool + valency analysis functions (dataset-agnostic)
# ══════════════════════════════════════════════════════════════
def build_pt_pp_maps(train):
    """Same logic as cogs_compositional.build_verb_form_maps: extract the verb
    lemma via `.agent` in LF and record the input token at that position."""
    pt_map, pp_map = {}, {}
    for inp, lf, _ in train:
        verb = extract_verb_from_lf(lf)
        if verb is None:
            continue
        toks = inp.split()
        lf_toks = lf.split()
        verb_pos = None
        for j in range(len(lf_toks) - 6):
            if lf_toks[j + 1] == "." and lf_toks[j + 2] == "agent" \
                    and lf_toks[j + 3] == "(" and lf_toks[j + 4] == "x" \
                    and lf_toks[j + 5] == "_" and lf_toks[j + 6].isdigit():
                verb_pos = int(lf_toks[j + 6])
                break
        if verb_pos is None or verb_pos >= len(toks):
            continue
        verb_word = toks[verb_pos]
        if "was" in toks and "by" in toks:
            pp_map.setdefault(verb, verb_word)
        else:
            pt_map.setdefault(verb, verb_word)
    return pt_map, pp_map


def build_pool_verbs(train):
    pt_map, pp_map = build_pt_pp_maps(train)
    pp_with_morph = dict(pp_map)
    for lemma, past in pt_map.items():
        if lemma not in pp_with_morph and past.endswith("ed"):
            pp_with_morph[lemma] = past
    return set(pt_map.keys()) & set(pp_with_morph.keys()), pt_map, pp_with_morph


def verbs_with_roles(lf):
    toks = lf.split()
    with_agent, with_theme = set(), set()
    for j in range(len(toks) - 2):
        if toks[j + 1] == "." and toks[j + 2] in ("agent", "theme"):
            v = toks[j]
            if toks[j + 2] == "agent":
                with_agent.add(v)
            elif toks[j + 2] == "theme":
                with_theme.add(v)
    return with_agent, with_theme


def classify_valency(train):
    agent_verbs, theme_verbs = set(), set()
    transitive_example_verbs = set()
    for _inp, lf, _cat in train:
        wa, wt = verbs_with_roles(lf)
        agent_verbs |= wa
        theme_verbs |= wt
        transitive_example_verbs |= (wa & wt)
    return {
        "agent": agent_verbs,
        "theme": theme_verbs,
        "transitive_example": transitive_example_verbs,
        "unacc_only": theme_verbs - agent_verbs,
    }


def classify_voice(train):
    """COGS voice heuristic: input has was+by ⇒ passive. (Good for COGS, noisy
    on SLOG but still informative.)"""
    passive, active = set(), set()
    for inp, lf, _ in train:
        v = extract_verb_from_lf(lf)
        if v is None:
            continue
        toks = inp.split()
        if "was" in toks and "by" in toks:
            passive.add(v)
        else:
            active.add(v)
    return passive, active


# ══════════════════════════════════════════════════════════════
# Per-category summary
# ══════════════════════════════════════════════════════════════
def summarise_pool_coverage(label, verbs_counter, pool_verbs):
    unique_verbs = set(verbs_counter.keys())
    in_pool = unique_verbs & pool_verbs
    out_pool = unique_verbs - pool_verbs
    examples_in = sum(verbs_counter[v] for v in in_pool)
    examples_out = sum(verbs_counter[v] for v in out_pool)
    total = sum(verbs_counter.values())
    print(f"  {label} ({total} examples, {len(unique_verbs)} unique verbs):")
    print(f"    verbs in permute pool: {len(in_pool)}/{len(unique_verbs)}  "
          f"→ {examples_in}/{total} examples ({100*examples_in/max(total,1):.1f}%)")
    if out_pool:
        top_out = sorted(out_pool, key=lambda v: -verbs_counter[v])
        print(f"    verbs OUT of pool: {top_out[:10]}"
              + (f" (+{len(top_out)-10} more)" if len(top_out) > 10 else ""))
        print(f"    (example-level loss: {examples_out}/{total} = "
              f"{100*examples_out/max(total,1):.1f}%)")


def detailed_per_verb(verbs_counter, pool_verbs, valency, voice):
    """Print per-verb detail for small categories (AGENT/THEME/TRANSITIVE/UNACC
    flags + voice hint)."""
    passive_verbs = voice[0]
    active_verbs = voice[1]
    for v, c in verbs_counter.most_common():
        flags = [
            ("pool",               v in pool_verbs),
            ("seen_transitive",    v in valency["transitive_example"]),
            ("unacc_only",         v in valency["unacc_only"]),
            ("train_passive_seen", v in passive_verbs),
            ("train_active_seen",  v in active_verbs),
        ]
        flags_str = "  ".join(f"{n}={'✓' if ok else '✗'}" for n, ok in flags)
        print(f"      {v:20s} {c:4d}  {flags_str}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="cogs",
                    choices=["cogs", "slog"])
    ap.add_argument("--all-structural", action="store_true",
                    help="Print pool coverage for every category whose name contains a structural keyword (not just the voice categories).")
    args = ap.parse_args()

    paths = dataset_paths(args.dataset)
    print(f"Dataset: {args.dataset}")
    print(f"  train: {paths['train']}")
    print(f"  gen:   {paths['gen']}")
    train = load_tsv(paths["train"])
    gen = load_tsv(paths["gen"])
    print(f"Loaded {len(train)} train, {len(gen)} gen.\n")

    # --- Voice heuristic (COGS-style) ---
    passive_verbs, active_verbs = classify_voice(train)
    print(f"Voice heuristic from train:")
    print(f"  verbs ever passive (was + by): {len(passive_verbs)}")
    print(f"  verbs ever active (no was+by): {len(active_verbs)}")
    print(f"  intersection: {len(passive_verbs & active_verbs)}\n")

    # --- Permute-verbs pool ---
    pool_verbs, pt_map, pp_map_morph = build_pool_verbs(train)
    print(f"Permute-verbs pool (pt_map ∩ pp_map with -ed morphology fallback): "
          f"{len(pool_verbs)} verbs\n")

    # --- Valency classification ---
    valency = classify_valency(train)
    print(f"Valency analysis from train:")
    print(f"  verbs with .agent:                       {len(valency['agent'])}")
    print(f"  verbs with .theme:                       {len(valency['theme'])}")
    print(f"  verbs seen transitive (.agent + .theme): {len(valency['transitive_example'])}")
    print(f"  verbs unaccusative only (.theme no .agent): {len(valency['unacc_only'])}")
    if valency["unacc_only"]:
        print(f"    list: {sorted(valency['unacc_only'])}")
    print()

    # --- Group gen examples by category ---
    gen_by_cat = {}
    for inp, lf, cat in gen:
        v = extract_verb_from_lf(lf)
        if v is None:
            continue
        gen_by_cat.setdefault(cat, Counter())[v] += 1

    # --- Per-voice-category detail ---
    print("=" * 60)
    print("Per-category verb coverage against permute-verbs pool:")
    print("=" * 60)
    categories_to_show = sorted(set(gen_by_cat) & VOICE_CATEGORIES)
    if args.all_structural:
        for cat in sorted(gen_by_cat):
            if cat in categories_to_show:
                continue
            low = cat.lower()
            if any(k in low for k in STRUCTURAL_KEYWORDS):
                categories_to_show.append(cat)

    for cat in categories_to_show:
        counter = gen_by_cat[cat]
        print()
        summarise_pool_coverage(cat, counter, pool_verbs)
        # Detailed per-verb table for the 3 voice/valency categories
        if cat in {"active_to_passive", "passive_to_active", "unacc_to_transitive"}:
            print(f"    per-verb flags:")
            detailed_per_verb(counter, pool_verbs, valency,
                              (passive_verbs, active_verbs))

    # --- Recap: verbs that would benefit from being force-added to the pool ---
    # A verb benefits if it is in gen (any structural category) AND is out of pool
    # but exists in train (as .theme-only or passive-only) and its regular -ed form
    # would resolve it.
    candidates = set()
    for cat, counter in gen_by_cat.items():
        if cat not in categories_to_show:
            continue
        candidates |= (set(counter.keys()) - pool_verbs)
    # Filter to those known to exist in train (via valency.theme)
    candidates_in_train = candidates & (valency["agent"] | valency["theme"])
    print("\n" + "=" * 60)
    print("Candidates for FORCED_REGULAR_VERBS_FOR_POOL:")
    print("=" * 60)
    if not candidates_in_train:
        print("  (none — current pool already covers all structural-test verbs)")
    else:
        for v in sorted(candidates_in_train):
            reasons = []
            if v in valency["unacc_only"]:
                reasons.append("unacc_only")
            if v in passive_verbs and v not in active_verbs:
                reasons.append("passive_only")
            if v in active_verbs and v not in passive_verbs:
                reasons.append("active_only")
            # Count impact across gen
            impact = sum(gen_by_cat[c].get(v, 0) for c in categories_to_show)
            print(f"  {v:20s}  impact={impact:4d} examples  reasons={reasons}")
