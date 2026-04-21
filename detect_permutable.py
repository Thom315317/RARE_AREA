#!/usr/bin/env python3
"""Skeleton: automatic detection of permutable token classes in a seq2seq dataset.

Algorithm
---------
For each input token v in vocab:
  1. Collect all contexts in which v occurs (window = ±W tokens).
  2. Build a distributional fingerprint: a sparse vector over
     (offset, neighbor_token) pairs counting co-occurrences.
  3. Compute pairwise cosine similarity between fingerprints.
  4. Union-find over pairs whose similarity > sim_thr → clusters.
Returned clusters with ≥2 members are permutable candidates.

Usage
-----
    from detect_permutable import detect_permutable
    clusters = detect_permutable(pairs, window=2, sim_thr=0.8, min_freq=5)
    # clusters: List[List[str]] — each inner list is a class of input tokens

Notes
-----
- Input-side only for now. Output-side mapping (needed for seq2seq augmentation)
  must be resolved separately — e.g. via output tokens that co-occur
  exclusively with a given input token across the training set, or via
  lowercasing heuristic (COGS proper names).
- min_freq filters rare tokens whose fingerprints are noisy.
"""
from collections import defaultdict
from typing import Callable, List, Tuple, Iterable, Optional
import math


# ══════════════════════════════════════════════════════════════
# Union-Find
# ══════════════════════════════════════════════════════════════
class UnionFind:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb

    def clusters(self):
        groups = defaultdict(list)
        for x in self.p:
            groups[self.find(x)].append(x)
        return list(groups.values())


# ══════════════════════════════════════════════════════════════
# Fingerprints
# ══════════════════════════════════════════════════════════════
def _tokenize_default(s: str) -> List[str]:
    return s.split()


def build_fingerprints(pairs: Iterable[Tuple[str, str]],
                       window: int = 2,
                       tokenize: Callable[[str], List[str]] = _tokenize_default):
    """Return (freq, fprint) where
      freq: dict token -> total count
      fprint: dict token -> dict[(offset, neighbor)] -> count
    Offsets range over [-window, -1] ∪ [1, window].
    """
    freq = defaultdict(int)
    fprint = defaultdict(lambda: defaultdict(int))
    for inp, _ in pairs:
        toks = tokenize(inp)
        L = len(toks)
        for i, v in enumerate(toks):
            freq[v] += 1
            for d in range(-window, window + 1):
                if d == 0: continue
                j = i + d
                if 0 <= j < L:
                    fprint[v][(d, toks[j])] += 1
    return freq, fprint


def _cosine(a: dict, b: dict) -> float:
    if not a or not b: return 0.0
    # Iterate smaller dict
    if len(a) > len(b): a, b = b, a
    dot = 0.0
    for k, va in a.items():
        vb = b.get(k)
        if vb is not None: dot += va * vb
    if dot == 0.0: return 0.0
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))
    if na == 0 or nb == 0: return 0.0
    return dot / (na * nb)


# ══════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════
def detect_permutable(pairs: Iterable[Tuple[str, str]],
                      window: int = 2,
                      sim_thr: float = 0.8,
                      min_freq: int = 5,
                      exclude: Optional[set] = None,
                      tokenize: Callable[[str], List[str]] = _tokenize_default
                     ) -> List[List[str]]:
    """Detect permutable token classes from input-side context fingerprints.

    Args:
        pairs: iterable of (input, output) strings (output is currently ignored)
        window: ± context window size
        sim_thr: minimum cosine similarity to merge two tokens into a class
        min_freq: discard tokens with total count < min_freq (noise filter)
        exclude: set of tokens to never consider (e.g. <pad>, <bos>, punctuation)
        tokenize: how to split strings into tokens

    Returns:
        List of clusters, each a list of input-vocab tokens (len ≥ 2).
    """
    pairs = list(pairs)
    freq, fprint = build_fingerprints(pairs, window=window, tokenize=tokenize)
    keep = [t for t, c in freq.items() if c >= min_freq]
    if exclude:
        keep = [t for t in keep if t not in exclude]

    uf = UnionFind(keep)
    # Naive O(V²) pairwise — fine for V ≲ a few thousand
    keep_sorted = sorted(keep)
    N = len(keep_sorted)
    for i in range(N):
        a = keep_sorted[i]
        fa = fprint[a]
        for j in range(i + 1, N):
            b = keep_sorted[j]
            if _cosine(fa, fprint[b]) >= sim_thr:
                uf.union(a, b)

    clusters = [sorted(c) for c in uf.clusters() if len(c) >= 2]
    clusters.sort(key=lambda c: (-len(c), c[0]))
    return clusters


# ══════════════════════════════════════════════════════════════
# CLI — try on a given seq2seq dataset file
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse, json, os
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True,
                   help="Path to TSV (input<TAB>output) or plain text")
    p.add_argument("--format", type=str, choices=["tsv", "scan"], default="tsv",
                   help="tsv: COGS-style. scan: 'IN: ... OUT: ...' lines")
    p.add_argument("--window", type=int, default=2)
    p.add_argument("--sim-thr", type=float, default=0.8)
    p.add_argument("--min-freq", type=int, default=5)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    pairs = []
    with open(args.dataset) as f:
        for line in f:
            line = line.rstrip("\n")
            if args.format == "tsv":
                parts = line.split("\t")
                if len(parts) >= 2: pairs.append((parts[0], parts[1]))
            else:  # scan
                import re
                m = re.match(r"IN:\s*(.*?)\s*OUT:\s*(.*)$", line.strip())
                if m: pairs.append((m.group(1), m.group(2)))

    clusters = detect_permutable(pairs, window=args.window,
                                 sim_thr=args.sim_thr, min_freq=args.min_freq)
    print(f"Found {len(clusters)} candidate classes "
          f"(sim_thr={args.sim_thr}, window=±{args.window}, min_freq={args.min_freq})")
    for k, c in enumerate(clusters):
        preview = c if len(c) <= 15 else c[:15] + [f"... (+{len(c)-15})"]
        print(f"  class {k+1} ({len(c)}): {preview}")
    if args.out:
        json.dump({"clusters": clusters, "window": args.window,
                   "sim_thr": args.sim_thr, "min_freq": args.min_freq},
                  open(args.out, "w"), indent=2)
        print(f"Saved: {args.out}")
