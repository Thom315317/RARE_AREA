#!/usr/bin/env python3
"""Phase A.2 — Étape 2/3 : à partir de flagged.json, génère des variantes
ciblées par cluster et écrit un fichier extra_train.tsv compatible avec le
flag --extra-train-file de cogs_compositional.py.

Stratégie pragmatique (v1) :
  - 4 clusters par features (KMeans k=4 sur surprise + entropy + length + nesting).
  - Pour chaque flagged example, applique 1-2 transformations simples et concrètes :
       * proper-name swap (uniforme aléatoire dans la liste des noms COGS)
       * common-noun swap (uniforme dans top-30 communs)
       * peel-cp prefix : "<NAME> said that <input>"
  - Ces transformations préservent la LF en l'adaptant (offset des x_N quand on
    ajoute un préfixe ; substitution des proper-names dans la LF).
  - Total : ~3-4 variantes par flagged example.

Usage :
  python3 meta_a2_generate_variants.py \\
      --flagged runs_meta/etape_a2/cycle_1/flagged.json \\
      --output  runs_meta/etape_a2/cycle_1/extra_train.tsv \\
      --max-per-example 3 \\
      --max-total 2000 \\
      --seed 42
"""
import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict


# ══════════════════════════════════════════════════════════════════════════
# Pools (alignés sur les permutations COGS existantes)
# ══════════════════════════════════════════════════════════════════════════
PROPER_POOL_DEFAULT = [
    "Emma", "Liam", "Olivia", "Noah", "Ava", "Charlotte", "Mia", "Ethan",
    "Sophia", "Lucas", "Isabella", "Mason", "Aiden", "Elijah", "Amelia",
    "Harper", "Hazel", "James", "Henry", "Owen", "Sebastian", "Ella",
    "Jacob", "William", "Daniel", "Jackson", "Levi", "Carter", "Wyatt",
    "Grayson", "Evelyn", "Abigail", "Emily", "Camila", "Aria",
]
COMMON_POOL_DEFAULT = [
    "boy", "girl", "cake", "cookie", "table", "rose", "boat", "house", "dog",
    "cat", "bird", "frog", "donkey", "monkey", "rabbit", "cup", "book",
    "chair", "ball", "hat", "shoe", "tree", "river", "mountain", "letter",
    "ring", "stone", "key", "door", "window",
]
DETERMINERS = {"A", "An", "The", "a", "an", "the"}
FUNCTION_WORDS = {"that", "was", "were", "by", "to", "in", "on", "."}


def _proper_used(tokens):
    """Identifie les proper nouns dans une liste de tokens (capitalisés
    non-déterminants, pas en début de phrase)."""
    out = set()
    for t in tokens:
        if (t and t[0].isupper() and len(t) > 1 and t.isalpha()
                and t not in DETERMINERS):
            out.add(t)
    return out


def _common_after_det(tokens):
    """Common nouns post-déterminant."""
    out = set()
    for i, t in enumerate(tokens):
        if (i > 0 and tokens[i - 1] in DETERMINERS and t.islower()
                and t.isalpha() and len(t) > 1
                and t not in FUNCTION_WORDS):
            out.add(t)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Transformations
# ══════════════════════════════════════════════════════════════════════════
def swap_proper_names(input_str, lf_str, rng, pool):
    used = sorted(_proper_used(input_str.split()))
    if not used:
        return None
    cands = [p for p in pool if p not in used]
    if not cands:
        return None
    mapping = {old: rng.choice(cands) for old in used}
    new_input = " ".join(mapping.get(t, t) for t in input_str.split())
    new_lf_tokens = []
    for t in lf_str.split():
        new_lf_tokens.append(mapping.get(t, t))
    return new_input, " ".join(new_lf_tokens), "swap_proper"


def swap_common_nouns(input_str, lf_str, rng, pool):
    """Substitue les common nouns post-déterminant. La LF utilise les mêmes
    formes lemma — on remplace partout dans la LF aussi."""
    toks = input_str.split()
    used = sorted(_common_after_det(toks))
    if not used:
        return None
    cands = [c for c in pool if c not in used]
    if not cands:
        return None
    mapping = {old: rng.choice(cands) for old in used}
    new_input = []
    for i, t in enumerate(toks):
        if (i > 0 and toks[i - 1] in DETERMINERS and t in mapping):
            new_input.append(mapping[t])
        else:
            new_input.append(t)
    new_lf = []
    for t in lf_str.split():
        new_lf.append(mapping.get(t, t))
    return " ".join(new_input), " ".join(new_lf), "swap_common"


_X_RE = re.compile(r"^x$")


def peel_cp_prefix(input_str, lf_str, rng, pool):
    """Préfixe avec 'NAME said that' et incrémente toutes les positions x_N
    de la LF de 3 (3 tokens insérés avant le sujet original)."""
    if input_str.endswith(" ."):
        body = input_str[:-2]
    else:
        body = input_str
    name = rng.choice(pool)
    new_input = f"{name} said that {body} ."

    # Increment x_N positions in LF by 3 (3 tokens : NAME, said, that)
    # Pattern: "x _ N" → "x _ (N+3)"
    new_lf = re.sub(
        r"x _ (\d+)",
        lambda m: f"x _ {int(m.group(1)) + 3}",
        lf_str,
    )
    # Wrap with "say . agent ( x _ 1 , NAME ) AND say . ccomp ( x _ 1 , x _ M )"
    # where M is the position of the verb in the inner clause. We approximate
    # by leaving it implicit and just prepending the matrix predicates.
    # Pour rester simple : on ajoute la forme matrix correcte si l'input body
    # commence par un déterminant ou un nom propre.
    body_tokens = body.split()
    if not body_tokens:
        return None
    # Find the verb position : first token that maps to .agent in lf_str
    # Use a heuristic : the verb is at position N where the smallest x_N appears
    # that is the head of a predicate (i.e. has ".something" right after).
    # Simpler : take the smallest verb_pos = position of first .agent / .theme head.
    m_first = re.search(r"(\w+) \. (?:agent|theme) \(", lf_str)
    if not m_first:
        return None
    inner_verb_lemma = m_first.group(1)
    # Heuristic verb_pos in body : where this verb form appears in body tokens.
    # We don't know the inflection, so just place ccomp at the new position of
    # the original first .agent/.theme x_N (already shifted).
    m_pos = re.search(r"x _ (\d+)", lf_str)
    if not m_pos:
        return None
    inner_pos = int(m_pos.group(1)) + 3
    matrix = (f"say . agent ( x _ 1 , {name} ) AND "
              f"say . ccomp ( x _ 1 , x _ {inner_pos} ) AND ")
    new_lf = matrix + new_lf
    return new_input, new_lf, "peel_cp_prefix"


# ══════════════════════════════════════════════════════════════════════════
# Cluster + dispatch
# ══════════════════════════════════════════════════════════════════════════
def cluster_features(flagged, n_clusters=4, seed=42):
    """KMeans sur surprise/entropy/length/nesting. Renvoie cluster_id par
    flagged example + nom dominant ('high_nesting', 'high_length', etc.)."""
    try:
        from sklearn.cluster import KMeans
        import numpy as np
    except ImportError:
        return [0] * len(flagged), {0: "all"}

    feats = []
    for f in flagged:
        ft = f["features"]
        feats.append([ft.get("surprise_mean", 0.0),
                      ft.get("entropy_mean_greedy", 0.0),
                      ft.get("input_length", 0),
                      ft.get("nesting_depth", 0)])
    feats = np.array(feats, dtype=float)
    if feats.std(axis=0).min() == 0 or len(feats) < n_clusters:
        return [0] * len(flagged), {0: "all"}
    # Standardise
    mu = feats.mean(axis=0); sd = feats.std(axis=0); sd[sd == 0] = 1.0
    feats_s = (feats - mu) / sd
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(feats_s)

    # Name each cluster by its dominant feature
    cluster_names = {}
    feat_names = ["surprise_mean", "entropy_mean_greedy",
                  "input_length", "nesting_depth"]
    for c in range(n_clusters):
        centroid = km.cluster_centers_[c]
        dom_idx = int(centroid.argmax())
        cluster_names[c] = f"high_{feat_names[dom_idx]}"
    return labels.tolist(), cluster_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flagged", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-per-example", type=int, default=3,
                    help="Nombre max de variantes par exemple flagged")
    ap.add_argument("--max-total", type=int, default=2000,
                    help="Cap global sur le nombre de variantes générées")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--proper-pool", default=None,
                    help="Path à un JSON list ; si absent, utilise le pool par défaut.")
    ap.add_argument("--common-pool", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    proper_pool = (json.load(open(args.proper_pool))
                   if args.proper_pool else list(PROPER_POOL_DEFAULT))
    common_pool = (json.load(open(args.common_pool))
                   if args.common_pool else list(COMMON_POOL_DEFAULT))

    with open(args.flagged, "r") as f:
        flagged_data = json.load(f)
    flagged = flagged_data["examples"]
    print(f"Loaded {len(flagged)} flagged examples")

    # Cluster
    cluster_labels, cluster_names = cluster_features(flagged,
                                                       n_clusters=4,
                                                       seed=args.seed)
    cluster_dist = Counter(cluster_labels)
    print(f"Cluster distribution: {dict(cluster_dist)} → {cluster_names}")

    # Sort flagged by score desc to bias generation toward most-risky
    flagged_sorted = sorted(zip(flagged, cluster_labels),
                            key=lambda x: -x[0]["score"])

    # Cluster → preferred transformations (if we want to be cluster-aware)
    # In v1 we apply all 3 to every flagged example, capped by max-per-example.
    transforms_all = [swap_proper_names, swap_common_nouns, peel_cp_prefix]

    out_lines = []
    n_per_cluster = Counter()
    n_per_transform = Counter()
    for ex, cluster_id in flagged_sorted:
        if len(out_lines) >= args.max_total:
            break
        # Apply up to max_per_example transformations
        rng.shuffle(transforms_all)
        applied = 0
        for tr in transforms_all:
            if applied >= args.max_per_example:
                break
            try:
                result = tr(ex["input"], ex["lf"], rng,
                             proper_pool if tr is not swap_common_nouns else common_pool)
            except Exception:
                continue
            if result is None:
                continue
            new_inp, new_lf, tag = result
            cat = f"a2_aug_{tag}"
            out_lines.append(f"{new_inp}\t{new_lf}\t{cat}")
            n_per_cluster[cluster_id] += 1
            n_per_transform[tag] += 1
            applied += 1

    print(f"\nGenerated {len(out_lines)} variants")
    print(f"  by cluster: {dict(n_per_cluster)}")
    print(f"  by transform: {dict(n_per_transform)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"Wrote {args.output}")

    # Also write clusters.json metadata
    clusters_path = os.path.join(os.path.dirname(args.output), "clusters.json")
    with open(clusters_path, "w") as f:
        json.dump({
            "cluster_names": {str(k): v for k, v in cluster_names.items()},
            "cluster_sizes": {str(k): v for k, v in cluster_dist.items()},
            "n_variants_per_cluster": {str(k): v for k, v in n_per_cluster.items()},
            "n_variants_per_transform": dict(n_per_transform),
            "total_variants": len(out_lines),
        }, f, indent=2)
    print(f"Wrote {clusters_path}")


if __name__ == "__main__":
    main()
