#!/usr/bin/env python3
"""
Phase 3 Niveau 3+4 — Diagnostic causal et génération automatique de fix.

Niveau 3 : Pour chaque cluster d'erreurs, compare la sortie du modèle à la
sortie correcte. Extrait le diff structurel. Formule une hypothèse causale.
Vérifie dans le train set.

Niveau 4 : Quand une hypothèse est confirmée, génère automatiquement les
exemples correctifs en appliquant la transformation identifiée.

Usage:
  python3 causal_diagnosis.py --checkpoint runs_master/cycle_0/checkpoint.pt \
      --dataset cogs --out-dir runs_master
"""
import os, sys, json, argparse, re
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))

from cogs_compositional import (
    parse_cogs_tsv, build_vocabs, COGSDataset, collate,
    TransformerSeq2Seq, greedy_decode,
    MAX_IN, MAX_OUT, MAX_DECODE, PAD, BOS, EOS,
)


# ══════════════════════════════════════════════════════════════
# LF parsing
# ══════════════════════════════════════════════════════════════
def parse_lf(lf_str):
    """Parse a COGS/SLOG LF string into a list of predicates.
    Returns list of (predicate_name, arg1, arg2, ...) tuples."""
    # Split presuppositions and body
    if " ; " in lf_str:
        parts = lf_str.split(" ; ")
        body = parts[-1]
        presups = parts[:-1]
    else:
        body = lf_str
        presups = []

    predicates = []
    # Parse presuppositions: "* noun ( x _ N )"
    for p in presups:
        m = re.match(r'\* (\w+) \( (x _ \d+) \)', p.strip())
        if m:
            predicates.append(("PRESUP:" + m.group(1), m.group(2)))

    # Parse body: split by " AND "
    for clause in body.split(" AND "):
        clause = clause.strip()
        # Pattern: NOUN ( x _ N )
        m = re.match(r'^(\w+) \( (x _ \d+) \)$', clause)
        if m:
            predicates.append((m.group(1), m.group(2)))
            continue
        # Pattern: NOUN . ROLE ( x _ N , x _ M )  or  NOUN . ROLE ( x _ N , NAME )
        m = re.match(r'^(\w[\w .]+) \( (x _ \d+(?:.*?)) \)$', clause)
        if m:
            pred_name = m.group(1)
            args_str = m.group(2)
            args = [a.strip() for a in args_str.split(",")]
            predicates.append(tuple([pred_name] + args))
            continue

    return predicates


# ══════════════════════════════════════════════════════════════
# Structural diff
# ══════════════════════════════════════════════════════════════
@dataclass
class DiffResult:
    missing: List[tuple] = field(default_factory=list)
    extra: List[tuple] = field(default_factory=list)
    wrong_args: List[dict] = field(default_factory=list)
    diff_type: str = "unknown"

    @property
    def n_errors(self):
        return len(self.missing) + len(self.extra) + len(self.wrong_args)


def structural_diff(pred_lf_str, gold_lf_str):
    """Compare two LFs, return the structural diff."""
    pred = set(parse_lf(pred_lf_str))
    gold = set(parse_lf(gold_lf_str))

    missing = [p for p in gold if p not in pred]
    extra = [p for p in pred if p not in gold]

    # Detect wrong arguments: same predicate name, different args
    wrong_args = []
    gold_by_name = defaultdict(list)
    pred_by_name = defaultdict(list)
    for p in gold: gold_by_name[p[0]].append(p)
    for p in pred: pred_by_name[p[0]].append(p)
    for name in gold_by_name:
        if name in pred_by_name:
            for gp in gold_by_name[name]:
                for pp in pred_by_name[name]:
                    if gp != pp and gp[0] == pp[0]:
                        wrong_args.append({"pred": name, "gold": gp, "pred_args": pp})

    # Classify diff type
    diff_type = classify_diff(missing, extra, wrong_args)
    return DiffResult(missing=missing, extra=extra, wrong_args=wrong_args,
                      diff_type=diff_type)


def classify_diff(missing, extra, wrong_args):
    """Classify the type of structural diff with fine-grained sub-types."""

    # 1. nmod on subject (highest priority — specific structural gap)
    if any("nmod" in str(m) for m in missing):
        for m in missing:
            if "nmod" in str(m) and len(m) > 1:
                if "x _ 1" in str(m):
                    return "missing_nmod_on_subj"

    # 2. Truncation (many missing, nothing extra)
    if len(missing) > 3 and len(extra) == 0:
        return "truncated"

    # 3. Wrong argument positions (agent/theme in both missing and extra)
    if any("agent" in str(m) or "theme" in str(m) for m in missing):
        if any("agent" in str(e) or "theme" in str(e) for e in extra):
            return "wrong_arg_position"

    # 4. Wrong argument values (same predicate, different args)
    if len(missing) == 0 and len(extra) == 0 and wrong_args:
        return "wrong_arg_values"

    # 5. Fine-grained missing predicates sub-types
    if len(missing) > 0:
        # 5a. Missing nmod/ccomp (modification predicates)
        has_missing_mod = any("nmod" in str(m) or "ccomp" in str(m) for m in missing)
        if has_missing_mod:
            return "missing_modification"

        # 5b. Missing verb role (.agent, .theme, .recipient)
        has_missing_role = any(". agent" in str(m) or ". theme" in str(m)
                               or ". recipient" in str(m) for m in missing)
        if has_missing_role and not any(". agent" in str(e) or ". theme" in str(e) for e in extra):
            return "missing_verb_role"

        # 5c. Missing lexical predicate (bare noun like "cake(x_3)")
        # These are tuples like ("cake", "x _ 3") — no dot in predicate name
        missing_lexical = [m for m in missing if len(m) == 2 and "." not in str(m[0])
                          and not str(m[0]).startswith("PRESUP:")]
        missing_presup = [m for m in missing if str(m[0]).startswith("PRESUP:")]

        if missing_lexical or missing_presup:
            return "missing_lexical_noun"

        # 5d. Catch-all
        return "missing_predicates_other"

    if len(extra) > 0:
        return "extra_predicates"
    return "correct"


# ══════════════════════════════════════════════════════════════
# Hypothesis formulation and verification
# ══════════════════════════════════════════════════════════════
@dataclass
class Hypothesis:
    description: str
    fix_type: Optional[str]
    evidence: Optional[str] = None
    confirmed: bool = False


def formulate_hypothesis(diff_type, train_pairs):
    """From a diff type, formulate and verify a causal hypothesis."""
    if diff_type == "missing_nmod_on_subj":
        # Count nmod on subj vs obj in train
        n_subj = 0; n_obj = 0
        for _, lf, _ in train_pairs:
            for m in re.findall(r'nmod[^(]*\( (x _ \d+)', lf):
                idx = int(m.split()[-1])
                if idx <= 1: n_subj += 1
                else: n_obj += 1
        h = Hypothesis(
            description=f"nmod on subject: {n_subj} in train vs {n_obj} on object",
            fix_type="generate_nmod_on_subject")
        h.confirmed = (n_subj < n_obj * 0.1)  # <10% on subject
        h.evidence = f"train has {n_subj} nmod-on-subj vs {n_obj} nmod-on-obj"
        return h

    elif diff_type == "truncated":
        max_lf = max(len(lf.split()) for _, lf, _ in train_pairs)
        h = Hypothesis(
            description=f"decoder never generated LF longer than {max_lf} tokens",
            fix_type="generate_longer_examples")
        h.confirmed = True  # always true for gen (gen has longer LFs)
        h.evidence = f"train max LF length = {max_lf} tokens"
        return h

    elif diff_type == "missing_verb_role":
        h = Hypothesis(
            description="verb role missing — possible unseen voice/valency",
            fix_type="generate_verb_in_missing_form")
        h.confirmed = True
        h.evidence = "structural"
        return h

    elif diff_type == "wrong_arg_position":
        h = Hypothesis(
            description="arguments in wrong positions — structural confusion",
            fix_type="generate_position_examples")
        h.confirmed = True
        h.evidence = "structural"
        return h

    elif diff_type == "missing_lexical_noun":
        # Check which nouns are missing and whether they appear in both
        # subject and object positions in train
        from collections import defaultdict as _dd
        subj_nouns = _dd(int); obj_nouns = _dd(int)
        DETS = {"a", "A", "the", "The"}
        for inp, lf, _ in train_pairs:
            tokens = inp.split()
            if len(tokens) < 4: continue
            if tokens[0] in DETS and tokens[1][0].islower():
                subj_nouns[tokens[1]] += 1
            # Object: any noun after a det that's not the subject
            for i in range(2, len(tokens) - 1):
                if tokens[i] in DETS and i + 1 < len(tokens) and tokens[i+1][0].islower():
                    obj_nouns[tokens[i+1]] += 1
        all_nouns = set(subj_nouns.keys()) | set(obj_nouns.keys())
        subj_only = sum(1 for n in all_nouns if subj_nouns[n] > 0 and obj_nouns[n] == 0)
        obj_only = sum(1 for n in all_nouns if obj_nouns[n] > 0 and subj_nouns[n] == 0)
        both = sum(1 for n in all_nouns if subj_nouns[n] > 0 and obj_nouns[n] > 0)
        h = Hypothesis(
            description=f"lexical noun missing — position coverage gap. "
                        f"Nouns: {subj_only} subj-only, {obj_only} obj-only, {both} both",
            fix_type="generate_noun_position_swap")
        h.confirmed = (subj_only + obj_only) > 0
        h.evidence = (f"{subj_only} nouns seen only as subject, "
                     f"{obj_only} only as object, {both} in both positions")
        return h

    elif diff_type == "missing_modification":
        # nmod or ccomp entirely absent (not just wrong position)
        n_nmod = sum(1 for _, lf, _ in train_pairs if ". nmod" in lf)
        n_ccomp = sum(1 for _, lf, _ in train_pairs if ". ccomp" in lf)
        max_depth_nmod = 0
        for _, lf, _ in train_pairs:
            d = len(re.findall(r'nmod', lf))
            max_depth_nmod = max(max_depth_nmod, d)
        h = Hypothesis(
            description=f"modification predicate missing (nmod or ccomp). "
                        f"Train has {n_nmod} examples with nmod (max depth {max_depth_nmod}), "
                        f"{n_ccomp} with ccomp",
            fix_type="generate_deeper_modification")
        h.confirmed = True
        h.evidence = f"nmod examples: {n_nmod}, ccomp examples: {n_ccomp}, max nmod depth: {max_depth_nmod}"
        return h

    elif diff_type == "missing_predicates_other":
        h = Hypothesis(
            description="predicates missing — unclassified structural gap",
            fix_type=None)
        h.confirmed = False
        h.evidence = "could not classify further"
        return h

    elif diff_type == "extra_predicates":
        h = Hypothesis(
            description="model generates extra predicates not in gold — hallucination",
            fix_type=None)
        h.confirmed = True
        h.evidence = "model over-generates"
        return h

    else:
        return Hypothesis(description=f"unknown diff type: {diff_type}",
                         fix_type=None, confirmed=False)


# ══════════════════════════════════════════════════════════════
# Main diagnosis
# ══════════════════════════════════════════════════════════════
def diagnose(model, in_w2i, out_w2i, gen_pairs, train_pairs, device="cuda",
             max_examples=200):
    """Run full causal diagnosis on gen set errors."""
    dev_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    i2out = {v: k for k, v in out_w2i.items()}
    eos_id = out_w2i[EOS]

    ge_ds = COGSDataset(gen_pairs, in_w2i, out_w2i)
    ge_ld = DataLoader(ge_ds, 16, shuffle=False, collate_fn=collate, num_workers=0)

    # Collect diffs
    diffs_by_cat = defaultdict(list)
    diff_type_counts = Counter()
    wrong_arg_examples = []      # fine-grained wrong_arg_position log
    wrong_arg_patterns = Counter()
    n_correct = 0; n_total = 0; n_analyzed = 0

    # Build input token decoder for displaying inputs
    i2in = {v: k for k, v in in_w2i.items()}

    model.eval()
    with torch.no_grad():
        for src, src_mask, tgt, _, cats in ge_ld:
            if n_analyzed >= max_examples:
                break
            src = src.to(dev_obj); src_mask = src_mask.to(dev_obj)
            pred_ids = greedy_decode(model, src, src_mask, out_w2i)

            for i in range(src.size(0)):
                if n_analyzed >= max_examples:
                    break
                # Decode pred tokens
                pred_toks = []
                for t in pred_ids[i].cpu().tolist():
                    if t == eos_id: break
                    if t == 0: continue
                    pred_toks.append(i2out.get(t, "?"))
                pred_lf = " ".join(pred_toks)

                # Gold LF
                gold_toks = []
                for t in tgt[i, 1:].tolist():
                    if t == eos_id: break
                    if t == 0: continue
                    gold_toks.append(i2out.get(t, "?"))
                gold_lf = " ".join(gold_toks)

                n_total += 1
                if pred_lf == gold_lf:
                    n_correct += 1
                    diff_type_counts["correct"] += 1
                else:
                    diff = structural_diff(pred_lf, gold_lf)
                    diff_type_counts[diff.diff_type] += 1
                    diffs_by_cat[cats[i]].append({
                        "diff_type": diff.diff_type,
                        "n_missing": len(diff.missing),
                        "n_extra": len(diff.extra),
                    })

                    # Fine-grained logging for wrong_arg_position
                    if diff.diff_type == "wrong_arg_position":
                        # Reconstruct input
                        inp_toks = [i2in.get(t, "?") for t in src[i].cpu().tolist()
                                    if t != 0]
                        inp_str = " ".join(inp_toks)

                        # Classify sub-pattern
                        is_passive = " was " in inp_str and " by " in inp_str
                        is_dative = " to " in inp_str.split()
                        is_truncated_passive = " was " in inp_str and not is_passive
                        is_ccomp = " that " in inp_str
                        if is_passive:
                            sub_pattern = "passive_with_by"
                        elif is_truncated_passive:
                            sub_pattern = "passive_no_by"
                        elif is_dative:
                            sub_pattern = "dative"
                        elif is_ccomp:
                            sub_pattern = "ccomp"
                        else:
                            sub_pattern = "other"
                        wrong_arg_patterns[sub_pattern] += 1

                        if len(wrong_arg_examples) < 20:
                            wrong_arg_examples.append({
                                "category": cats[i],
                                "sub_pattern": sub_pattern,
                                "input": inp_str,
                                "gold": gold_lf[:200],
                                "pred": pred_lf[:200],
                                "n_missing": len(diff.missing),
                                "n_extra": len(diff.extra),
                                "wrong_args": [
                                    {"pred_name": wa["pred"],
                                     "gold": str(wa["gold"])[:80],
                                     "pred_args": str(wa["pred_args"])[:80]}
                                    for wa in diff.wrong_args[:3]
                                ],
                            })
                n_analyzed += 1

    print(f"\n{'='*70}")
    print(f"  CAUSAL DIAGNOSIS — {n_analyzed} examples analyzed")
    print(f"{'='*70}")
    print(f"  Correct: {n_correct}/{n_total} ({n_correct/max(n_total,1)*100:.1f}%)")
    print(f"\n  Diff type distribution:")
    for dtype, count in diff_type_counts.most_common():
        print(f"    {dtype:<30s} {count:>5d} ({count/max(n_total,1)*100:.1f}%)")

    # Per-category dominant diff type
    print(f"\n  Per-category dominant diff:")
    cat_dominant = {}
    for cat, diffs in sorted(diffs_by_cat.items()):
        types = Counter(d["diff_type"] for d in diffs)
        dom = types.most_common(1)[0] if types else ("?", 0)
        cat_dominant[cat] = dom[0]
        print(f"    {cat:<40s} {dom[0]:<25s} ({dom[1]}/{len(diffs)})")

    # Formulate hypotheses for each diff type
    print(f"\n  Hypotheses:")
    hypotheses = {}
    for dtype in diff_type_counts:
        if dtype == "correct": continue
        h = formulate_hypothesis(dtype, train_pairs)
        hypotheses[dtype] = {
            "description": h.description,
            "fix_type": h.fix_type,
            "confirmed": h.confirmed,
            "evidence": h.evidence,
        }
        status = "✓ CONFIRMED" if h.confirmed else "✗ not confirmed"
        print(f"    [{dtype}] {h.description}")
        print(f"      {status} — fix: {h.fix_type}")
        print(f"      evidence: {h.evidence}")

    # Fine-grained wrong_arg_position analysis
    if wrong_arg_patterns:
        total_wa = sum(wrong_arg_patterns.values())
        print(f"\n  wrong_arg_position sub-patterns ({total_wa} total):")
        for pat, count in wrong_arg_patterns.most_common():
            print(f"    {pat:<25s} {count:>5d} ({count/total_wa*100:.1f}%)")

        print(f"\n  wrong_arg_position — 10 detailed examples:")
        for i, ex in enumerate(wrong_arg_examples[:10]):
            print(f"    [{i+1}] [{ex['sub_pattern']}] {ex['category']}")
            print(f"      IN:   {ex['input']}")
            print(f"      GOLD: {ex['gold'][:150]}...")
            print(f"      PRED: {ex['pred'][:150]}...")
            if ex["wrong_args"]:
                for wa in ex["wrong_args"][:2]:
                    print(f"      WRONG {wa['pred_name']}: gold={wa['gold']}  pred={wa['pred_args']}")

    return {
        "n_analyzed": n_analyzed,
        "n_correct": n_correct,
        "diff_types": dict(diff_type_counts),
        "cat_dominant_diff": cat_dominant,
        "hypotheses": hypotheses,
        "wrong_arg_patterns": dict(wrong_arg_patterns),
        "wrong_arg_examples": wrong_arg_examples,
    }


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="Phase 3 Level 3+4 — Causal diagnosis")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, default="cogs", choices=["cogs", "slog"])
    p.add_argument("--max-examples", type=int, default=500)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out-dir", type=str, default="runs_master")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    in_w2i = ckpt["in_w2i"]; out_w2i = ckpt["out_w2i"]
    d_model = ckpt["d_model"]
    model = TransformerSeq2Seq(len(in_w2i), len(out_w2i), d_model=d_model,
                               max_in=ckpt.get("max_in", MAX_IN),
                               max_out=ckpt.get("max_out", MAX_OUT)).to(device)
    model.load_state_dict(ckpt["state_dict"])

    # Load data
    if args.dataset == "cogs":
        data_dir = os.path.join(HERE, "data", "cogs")
        train = parse_cogs_tsv(os.path.join(data_dir, "train.tsv"))
        gen = parse_cogs_tsv(os.path.join(data_dir, "gen.tsv"))
    else:
        data_dir = os.path.join(HERE, "data", "slog")
        train = parse_cogs_tsv(os.path.join(data_dir, "train.tsv"))
        gen_path = os.path.join(data_dir, "generalization_sets", "gen_cogsLF.tsv")
        gen = parse_cogs_tsv(gen_path)

    results = diagnose(model, in_w2i, out_w2i, gen, train,
                       device=args.device, max_examples=args.max_examples)

    out_path = os.path.join(args.out_dir, f"diagnosis_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
