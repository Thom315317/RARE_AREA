#!/usr/bin/env python3
"""Étape B+ — Ajoute des features distributionnelles au meta-dataset.

Pour chaque exemple meta, calcule :
  - verb_construction_count_train : nombre d'occurrences (verbe, pattern) dans
    le train COGS
  - verb_construction_seen_binary  : 1 si count > 0, sinon 0

Patterns d'input (heuristiques inférence-time, sans LF) :
  passive_by, passive_simple, dative_to, intransitive, active, other

Usage :
  python3 add_distributional_features.py \\
      --train data/cogs/train.tsv \\
      --in-jsonl  meta_data/cogs_meta_dataset.jsonl \\
      --out-jsonl meta_data/cogs_meta_dataset_bplus.jsonl \\
      --stats-md  meta_data/bplus_distributional_stats.md
"""
import argparse
import json
import os
from collections import Counter, defaultdict


# ══════════════════════════════════════════════════════════════════════════
# Verb-word → lemma map depuis le train
# ══════════════════════════════════════════════════════════════════════════
def _parse_train_tsv(path):
    pairs = []
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            inp = parts[0]
            lf = parts[1]
            cat = parts[2] if len(parts) >= 3 else ""
            pairs.append((inp, lf, cat))
    return pairs


def _extract_lemma_at_pos(lf_toks, pos_target):
    """Cherche dans la LF un prédicat `<lemma> . <role> ( x _ N , …)` avec N == pos_target.
    Renvoie le lemma trouvé ou None."""
    for j in range(len(lf_toks) - 6):
        if (lf_toks[j + 1] == "." and lf_toks[j + 2] in
                ("agent", "theme", "recipient", "ccomp", "xcomp")
                and lf_toks[j + 3] == "(" and lf_toks[j + 4] == "x"
                and lf_toks[j + 5] == "_" and lf_toks[j + 6].isdigit()):
            pos = int(lf_toks[j + 6])
            if pos == pos_target:
                return lf_toks[j]
    return None


def build_word_to_lemma_map(train_pairs):
    """Pour chaque mot d'input, trouve le lemma le plus probable en
    cherchant les prédicats LF dont le x_N pointe sur ce mot."""
    counts = defaultdict(Counter)  # word → Counter(lemma)
    for inp, lf, _ in train_pairs:
        toks = inp.split()
        lf_toks = lf.split()
        # On scanne toutes les positions x_N rencontrées dans la LF
        positions_seen = set()
        for j in range(len(lf_toks) - 4):
            if (lf_toks[j] == "x" and lf_toks[j + 1] == "_"
                    and lf_toks[j + 2].isdigit()):
                positions_seen.add(int(lf_toks[j + 2]))
        for pos in positions_seen:
            if 0 <= pos < len(toks):
                lem = _extract_lemma_at_pos(lf_toks, pos)
                if lem is not None and lem != toks[pos]:
                    # On veut une mise en correspondance forme→lemma seulement
                    # quand le mot diffère du lemma (sinon trivial), mais en
                    # garder aussi pour matcher les bare forms
                    counts[toks[pos]][lem] += 1
                elif lem is not None:
                    counts[toks[pos]][lem] += 1
    word_to_lemma = {}
    for w, c in counts.items():
        word_to_lemma[w] = c.most_common(1)[0][0]
    return word_to_lemma


# ══════════════════════════════════════════════════════════════════════════
# Pattern extraction (input-only)
# ══════════════════════════════════════════════════════════════════════════
def extract_verb_and_pattern(input_tokens, word_to_lemma):
    """Renvoie (verb_lemma_or_None, pattern_str)."""
    toks = list(input_tokens)
    verb_lemma = None
    verb_pos = None
    for i, t in enumerate(toks):
        if t in word_to_lemma:
            verb_lemma = word_to_lemma[t]
            verb_pos = i
            break
    if verb_lemma is None:
        return None, "other"

    has_was = "was" in toks
    has_by = "by" in toks
    rest = toks[verb_pos + 1:]
    if rest and rest[-1] == ".":
        rest = rest[:-1]
    has_to_after_verb = "to" in rest

    if has_was and has_by:
        pattern = "passive_by"
    elif has_was:
        pattern = "passive_simple"
    elif has_to_after_verb:
        pattern = "dative_to"
    elif len(rest) == 0:
        pattern = "intransitive"
    else:
        pattern = "active"
    return verb_lemma, pattern


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True,
                    help="Chemin vers data/cogs/train.tsv")
    ap.add_argument("--in-jsonl", required=True,
                    help="Meta-dataset existant (sortie Phase B)")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--stats-md", required=True)
    args = ap.parse_args()

    print(f"Loading train from {args.train}…")
    train_pairs = _parse_train_tsv(args.train)
    print(f"  {len(train_pairs)} train pairs")

    print("Building word→lemma map…")
    word_to_lemma = build_word_to_lemma_map(train_pairs)
    print(f"  {len(word_to_lemma)} input words mapped to a lemma")

    print("Counting (verb_lemma, pattern) on train…")
    train_counts = Counter()
    skipped_no_verb = 0
    pattern_dist_train = Counter()
    for inp, lf, _ in train_pairs:
        verb, pat = extract_verb_and_pattern(inp.split(), word_to_lemma)
        if verb is None:
            skipped_no_verb += 1
            continue
        train_counts[(verb, pat)] += 1
        pattern_dist_train[pat] += 1
    print(f"  {len(train_counts)} unique (verb, pattern) combos")
    print(f"  {skipped_no_verb} train ex without identifiable verb")
    print(f"  pattern dist train: {dict(pattern_dist_train)}")

    print(f"\nProcessing meta-dataset {args.in_jsonl}…")
    n_total = 0
    n_with_count_zero = 0
    per_cat = defaultdict(lambda: {"n": 0, "n_zero": 0,
                                    "n_errors": 0, "n_zero_errors": 0,
                                    "patterns": Counter()})
    with open(args.in_jsonl, "r") as fin, open(args.out_jsonl, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            input_tokens = rec.get("input_tokens", [])
            verb, pattern = extract_verb_and_pattern(input_tokens, word_to_lemma)
            if verb is None:
                count = 0
            else:
                count = train_counts.get((verb, pattern), 0)
            seen = 1 if count > 0 else 0
            rec.setdefault("features_inference", {})
            rec["features_inference"]["verb_construction_count_train"] = int(count)
            rec["features_inference"]["verb_construction_seen_binary"] = int(seen)
            rec.setdefault("metadata", {})
            rec["metadata"]["bplus_verb"] = verb
            rec["metadata"]["bplus_pattern"] = pattern
            fout.write(json.dumps(rec) + "\n")

            n_total += 1
            cat = rec.get("category", "unknown")
            per_cat[cat]["n"] += 1
            per_cat[cat]["patterns"][pattern] += 1
            if count == 0:
                n_with_count_zero += 1
                per_cat[cat]["n_zero"] += 1
            if not rec.get("exact_match", True):
                per_cat[cat]["n_errors"] += 1
                if count == 0:
                    per_cat[cat]["n_zero_errors"] += 1

    print(f"  {n_total} records augmented; {n_with_count_zero} "
          f"({100*n_with_count_zero/max(n_total,1):.1f}%) have count == 0")

    # ── stats markdown
    PROBLEM_CATS = {"subj_to_obj_common", "passive_to_active",
                    "unacc_to_transitive", "obj_omitted_transitive_to_transitive"}
    lines = []
    lines.append("# B+ — sanity stats sur features distributionnelles")
    lines.append("")
    lines.append(f"- Train pairs : {len(train_pairs)}")
    lines.append(f"- Verbes mappés : {len(word_to_lemma)} formes → lemmas")
    lines.append(f"- Combos (verbe, pattern) uniques sur train : {len(train_counts)}")
    lines.append(f"- Meta-records totaux : {n_total}")
    lines.append(f"- Meta-records avec count==0 : {n_with_count_zero} "
                 f"({100*n_with_count_zero/max(n_total,1):.1f}%)")
    lines.append("")
    lines.append("## Distribution patterns (train)")
    lines.append("")
    for p, c in pattern_dist_train.most_common():
        lines.append(f"- `{p}` : {c} ({100*c/max(sum(pattern_dist_train.values()),1):.1f}%)")
    lines.append("")
    lines.append("## Sanity check — count==0 par catégorie meta-test")
    lines.append("")
    lines.append("L'hypothèse distributionnelle prédit que les 4 catégories problématiques "
                 "(en gras) ont plus d'exemples count==0 que les autres.")
    lines.append("")
    lines.append("| Catégorie | n | n_zero | %_zero | n_errors | %_zero_among_errors |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    rows = []
    for cat, s in per_cat.items():
        if s["n"] == 0:
            continue
        pct_zero = 100 * s["n_zero"] / s["n"]
        pct_zero_err = (100 * s["n_zero_errors"] / s["n_errors"]
                        if s["n_errors"] > 0 else float("nan"))
        rows.append((cat, s["n"], s["n_zero"], pct_zero,
                     s["n_errors"], pct_zero_err))
    rows.sort(key=lambda r: -r[3])  # by %_zero desc
    for cat, n, nz, pct, nerr, pct_err in rows:
        bold = "**" if cat in PROBLEM_CATS else ""
        pct_err_str = f"{pct_err:.1f}%" if pct_err == pct_err else "N/A"
        lines.append(f"| {bold}{cat}{bold} | {n} | {nz} | {pct:.1f}% | "
                     f"{nerr} | {pct_err_str} |")
    lines.append("")

    # Verdict sanity hypothèse
    problem_pct = []
    other_pct = []
    for cat, n, nz, pct, *_ in rows:
        if cat in PROBLEM_CATS:
            problem_pct.append(pct)
        else:
            other_pct.append(pct)
    if problem_pct and other_pct:
        avg_p = sum(problem_pct) / len(problem_pct)
        avg_o = sum(other_pct) / len(other_pct)
        lines.append(f"### Lecture")
        lines.append("")
        lines.append(f"- %_zero moyen sur les 4 catégories problématiques : **{avg_p:.1f}%**")
        lines.append(f"- %_zero moyen sur les autres catégories : {avg_o:.1f}%")
        if avg_p > avg_o + 5:
            lines.append("- ✓ Hypothèse distributionnelle plausible : les catégories "
                         "problématiques ont nettement plus d'exemples count==0.")
        elif avg_p > avg_o:
            lines.append("- ⚠️ Hypothèse marginalement supportée : écart faible (<5pts).")
        else:
            lines.append("- ❌ Hypothèse pré-réfutée : les catégories problématiques n'ont "
                         "PAS plus de count==0 que les autres. Le signal n'est pas "
                         "(seulement) distributionnel.")
        lines.append("")

    os.makedirs(os.path.dirname(args.stats_md) or ".", exist_ok=True)
    with open(args.stats_md, "w") as f:
        f.write("\n".join(lines))
    print(f"\nWrote augmented JSONL : {args.out_jsonl}")
    print(f"Wrote stats markdown  : {args.stats_md}")


if __name__ == "__main__":
    main()
