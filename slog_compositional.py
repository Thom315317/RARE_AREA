#!/usr/bin/env python3
"""SLOG structural generalization benchmark — wraps cogs_compositional.py.

SLOG (Li et al., EMNLP 2023) extends COGS with 17 harder structural
generalization categories (deep PP recursion, center embedding, relative
clauses, questions, etc.). Same format, same architecture, same augmentations.

This wrapper reuses the full CLI of cogs_compositional.py so every mechanism
added to COGS (peel-stack, peel-cp, cross-voice, permute-verbs, causal
curriculum, scaling flags, …) is automatically available for SLOG.

SLOG-specific additions implemented directly here (never in cogs_compositional):
  --unaccusative-aug   enables injection of Wh + declarative unaccusative
                       examples targeting Q_subj_active errors.
  --unaccusative-count N  number of examples to inject (default 300).

Data must be pre-downloaded to data/slog/ (see sweep_pub.sh or README).

Usage:
  python3 slog_compositional.py --variant B0 --seed 42 --epochs 60
  python3 slog_compositional.py --variant B4 --copy --peel-stack \\
      --permute-verbs --tfr-start=20 --seed 42 --epochs 60
"""
import os, sys, glob, json, random, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
SLOG_DATA = os.path.join(HERE, "data", "slog")
SLOG_RUNS = os.path.join(HERE, "runs", "slog")

# Reconfigure cogs_compositional paths before importing its pipeline
import cogs_compositional as cogs
cogs.DATA_DIR = SLOG_DATA
cogs.RUNS_DIR = SLOG_RUNS
os.makedirs(SLOG_DATA, exist_ok=True)
os.makedirs(SLOG_RUNS, exist_ok=True)

# SLOG data is local (already downloaded), override download to use local files
_SLOG_LOCAL = {
    "train": os.path.join(SLOG_DATA, "train.tsv"),
    "dev":   os.path.join(SLOG_DATA, "dev.tsv"),
    "gen":   os.path.join(SLOG_DATA, "generalization_sets", "gen_cogsLF.tsv"),
}


# ══════════════════════════════════════════════════════════════════════════
# Unaccusative augmentation patch (SLOG-only)
# ══════════════════════════════════════════════════════════════════════════
# 16 English unaccusative verbs. For each: (present, past-form-in-input, lemma-for-LF).
UNACCUSATIVE_VERBS = [
    ("shorten",     "shortened",     "shorten"),
    ("float",       "floated",       "float"),
    ("sink",        "sank",          "sink"),
    ("melt",        "melted",        "melt"),
    ("collapse",    "collapsed",     "collapse"),
    ("disintegrate","disintegrated", "disintegrate"),
    ("shatter",     "shattered",     "shatter"),
    ("freeze",      "froze",         "freeze"),
    ("burn",        "burned",        "burn"),
    ("grow",        "grew",          "grow"),
    ("shrink",      "shrank",        "shrink"),
    ("rise",        "rose",          "rise"),
    ("fall",        "fell",          "fall"),
    ("break",       "broke",         "break"),
    ("roll",        "rolled",        "roll"),
    ("open",        "opened",        "open"),
]

# Runtime flags, set from args in __main__ before train_one_run is called
_UNACC_INJECT = False        # inject the 300 unaccusative examples
_UNACC_POOL_EXCLUDE = False  # remove unaccusative verbs from permutation pool
_UNACC_COUNT = 300
_UNACC_SEED = 42
_UNACC_POOL_USED = []       # populated after generation (logged)
_UNACC_POOL_SKIPPED = []    # (lemma, reason)
_UNACC_REMOVED_FROM_PERM = []


def _train_vocab_from_pairs(pairs):
    in_vocab = set()
    out_vocab = set()
    for inp, lf, _ in pairs:
        in_vocab.update(inp.split())
        out_vocab.update(lf.split())
    return in_vocab, out_vocab


def _proper_names_from_pairs(pairs, in_vocab):
    det_like = {"The", "A", "Who", "What"}
    names = set()
    for inp, _lf, _cat in pairs:
        for tok in inp.split():
            if tok and tok[0].isupper() and tok not in det_like and tok.isalpha():
                if tok in in_vocab:
                    names.add(tok)
    return sorted(names)


def _generate_unaccusative_examples(train_pairs, count, seed):
    """Return (valid_examples, n_invalid_dropped, usable_pool, skipped_pool)
    where examples is a list of (inp, lf, tag)."""
    rng = random.Random(seed)
    in_vocab, out_vocab = _train_vocab_from_pairs(train_pairs)

    usable, skipped = [], []
    for pres, past, lemma in UNACCUSATIVE_VERBS:
        if past not in in_vocab:
            skipped.append((lemma, f"past '{past}' not in train input vocab"))
            continue
        if lemma not in out_vocab:
            skipped.append((lemma, f"lemma '{lemma}' not in train output vocab"))
            continue
        usable.append((pres, past, lemma))

    if len(usable) < 8:
        raise RuntimeError(
            f"Unaccusative pool too small: {len(usable)} verbs usable "
            f"(need ≥8). Skipped: {skipped}"
        )

    proper_names = _proper_names_from_pairs(train_pairs, in_vocab)
    if not proper_names:
        proper_names = ["Emma"]

    n_who  = int(count * 0.40)
    n_what = int(count * 0.40)
    n_decl = count - n_who - n_what

    per_template = {"who": 0, "what": 0, "decl": 0}
    examples = []
    for _ in range(n_who):
        _, past, lemma = rng.choice(usable)
        examples.append((f"Who {past} ?",
                         f"{lemma} . theme ( x _ 1 , ? )",
                         "aug_unaccusative_who"))
        per_template["who"] += 1
    for _ in range(n_what):
        _, past, lemma = rng.choice(usable)
        examples.append((f"What {past} ?",
                         f"{lemma} . theme ( x _ 1 , ? )",
                         "aug_unaccusative_what"))
        per_template["what"] += 1
    for _ in range(n_decl):
        _, past, lemma = rng.choice(usable)
        name = rng.choice(proper_names)
        examples.append((f"{name} {past} .",
                         f"{lemma} . theme ( x _ 1 , {name} )",
                         "aug_unaccusative_decl"))
        per_template["decl"] += 1

    # Validate every generated example against the train-derived vocab.
    # (Vocab is rebuilt from train+dev+gen later, but using only train-vocab
    # is the strictest safe filter: all our tokens come from train anyway.)
    valid, invalid = [], 0
    for inp, lf, tag in examples:
        if not all(t in in_vocab for t in inp.split()):
            invalid += 1
            continue
        if not all(t in out_vocab for t in lf.split()):
            invalid += 1
            continue
        valid.append((inp, lf, tag))

    return valid, invalid, usable, skipped, per_template


# ══════════════════════════════════════════════════════════════════════════
# Monkey-patches on cogs_compositional
# ══════════════════════════════════════════════════════════════════════════
_orig_parse_cogs_tsv = cogs.parse_cogs_tsv


def _patched_parse_cogs_tsv(path):
    global _UNACC_POOL_USED, _UNACC_POOL_SKIPPED
    pairs = _orig_parse_cogs_tsv(path)
    # Inject only on the train file, only when injection is enabled
    if _UNACC_INJECT and "train" in os.path.basename(path):
        valid, invalid, usable, skipped, per_tpl = _generate_unaccusative_examples(
            pairs, _UNACC_COUNT, _UNACC_SEED)
        _UNACC_POOL_USED = [u[2] for u in usable]
        _UNACC_POOL_SKIPPED = skipped
        print("UNACCUSATIVE augmentation ON:")
        print(f"  pool size: {len(usable)} verbs "
              f"({len(skipped)} skipped for vocab mismatch)")
        print(f"  pool used: {[u[2] for u in usable]}")
        for lemma, reason in skipped:
            print(f"    SKIP {lemma}: {reason}")
        print(f"  examples generated: {_UNACC_COUNT}")
        print(f"  examples after validation: {len(valid)} "
              f"({invalid} invalid dropped)")
        print(f"  breakdown: {per_tpl['who']} Who-questions, "
              f"{per_tpl['what']} What-questions, "
              f"{per_tpl['decl']} declaratives")
        print(f"  3 sample examples:")
        shown_tags = set()
        for inp, lf, tag in valid:
            short_tag = tag.split("_")[-1]
            if short_tag in shown_tags:
                continue
            shown_tags.add(short_tag)
            label = {"who": "A", "what": "B", "decl": "C"}.get(short_tag, "?")
            print(f"    [{label}] {inp}   →   {lf}")
            if len(shown_tags) >= 3:
                break
        if len(valid) < 100:
            raise RuntimeError(
                f"Only {len(valid)} unaccusative examples passed validation "
                f"(<100). Abort rather than train on a broken pool."
            )
        pairs = pairs + valid
    return pairs


cogs.parse_cogs_tsv = _patched_parse_cogs_tsv


_orig_build_verb_perm_pool = cogs.build_verb_perm_pool


def _patched_build_verb_perm_pool(train_pairs, in_w2i, out_w2i,
                                   morphology_fallback=True,
                                   filter_unaccusative=False):
    global _UNACC_REMOVED_FROM_PERM
    pool, stats = _orig_build_verb_perm_pool(train_pairs, in_w2i, out_w2i,
                                              morphology_fallback,
                                              filter_unaccusative=filter_unaccusative)
    if _UNACC_POOL_EXCLUDE:
        unacc_lemma_ids = {}
        for _pres, _past, lemma in UNACCUSATIVE_VERBS:
            lid = out_w2i.get(lemma)
            if lid is not None:
                unacc_lemma_ids[lid] = lemma
        new_pool = [(p, pp, l) for (p, pp, l) in pool
                    if l not in unacc_lemma_ids]
        removed_lemmas = sorted({
            lemma for (p, pp, l), lemma
            in [((x[0], x[1], x[2]), unacc_lemma_ids.get(x[2]))
                for x in pool]
            if lemma is not None
        })
        n_before = len(pool)
        n_after = len(new_pool)
        _UNACC_REMOVED_FROM_PERM = removed_lemmas
        if _UNACC_INJECT:
            print("VERB PERMUTATION pool adjusted (full unaccusative patch):")
        else:
            print("POOL FILTERING ON (no aug):")
        print(f"  removed from permutation pool: {removed_lemmas}")
        print(f"  new pool size: {n_after} (was {n_before})")
        stats["pool_size"] = n_after
        stats["unaccusative_removed"] = n_before - n_after
        return new_pool, stats
    return pool, stats


cogs.build_verb_perm_pool = _patched_build_verb_perm_pool


# Patch the data loading in train_one_run (legacy behaviour preserved)
_orig_train_one_run = cogs.train_one_run


def _patched_train_one_run(variant, seed, args):
    # Patch _download to be a no-op (data already local)
    orig_download = cogs._download
    cogs._download = lambda url, dest: None

    import shutil
    gen_link = os.path.join(SLOG_DATA, "gen.tsv")
    if not os.path.exists(gen_link):
        shutil.copy2(_SLOG_LOCAL["gen"], gen_link)

    cogs_aug = os.path.join(HERE, "data", "cogs", "aug_struct.tsv")
    slog_aug = os.path.join(SLOG_DATA, "aug_struct.tsv")
    if not os.path.exists(slog_aug) and os.path.exists(cogs_aug):
        shutil.copy2(cogs_aug, slog_aug)

    result = _orig_train_one_run(variant, seed, args)
    cogs._download = orig_download
    return result


cogs.train_one_run = _patched_train_one_run


# ══════════════════════════════════════════════════════════════════════════
# Post-training analysis (delta vs baseline, targeted check, pool_used)
# ══════════════════════════════════════════════════════════════════════════
BASELINE_PATTERN = os.path.join(
    HERE, "runs_innov_slog", "permute_verbs_peel10_whpassive", "B4_s42_*"
)
TARGETED_EXAMPLES = [
    # 10 Q_subj_active witnesses used by the previous diagnostic
    ("Who hated to dust ?",
     "hate . agent ( x _ 1 , ? ) AND hate . xcomp ( x _ 1 , x _ 3 ) AND dust . agent ( x _ 3 , ? )"),
    ("Who forwarded Ava a box ?",
     "forward . agent ( x _ 1 , ? ) AND forward . recipient ( x _ 1 , Ava ) AND forward . theme ( x _ 1 , x _ 4 ) AND box ( x _ 4 )"),
    ("Who fed the plate to the boy ?",
     "* plate ( x _ 3 ) ; * boy ( x _ 6 ) ; feed . agent ( x _ 1 , ? ) AND feed . theme ( x _ 1 , x _ 3 ) AND feed . recipient ( x _ 1 , x _ 6 )"),
    ("Who gave a cake to Charlotte ?",
     "give . agent ( x _ 1 , ? ) AND give . theme ( x _ 1 , x _ 3 ) AND give . recipient ( x _ 1 , Charlotte ) AND cake ( x _ 3 )"),
    ("Who preferred to run ?",
     "prefer . agent ( x _ 1 , ? ) AND prefer . xcomp ( x _ 1 , x _ 3 ) AND run . agent ( x _ 3 , ? )"),
    ("Who slept ?",      "sleep . agent ( x _ 1 , ? )"),
    ("Who shortened ?",  "shorten . theme ( x _ 1 , ? )"),
    ("Who hunted ?",     "hunt . agent ( x _ 1 , ? )"),
    ("Who sent a researcher a cake ?",
     "send . agent ( x _ 1 , ? ) AND send . recipient ( x _ 1 , x _ 3 ) AND send . theme ( x _ 1 , x _ 5 ) AND researcher ( x _ 3 ) AND cake ( x _ 5 )"),
    ("Who floated ?",    "float . theme ( x _ 1 , ? )"),
]


def _find_latest_run_dir(base_pattern):
    matches = sorted(glob.glob(base_pattern))
    return matches[-1] if matches else None


def _load_summary(run_dir):
    path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _write_delta_vs_baseline(patch_run_dir, baseline_run_dir):
    """Write delta_vs_baseline.md comparing categories between two runs."""
    patch_summary = _load_summary(patch_run_dir)
    baseline_summary = _load_summary(baseline_run_dir) if baseline_run_dir else None
    if patch_summary is None:
        return
    out_path = os.path.join(patch_run_dir, "delta_vs_baseline.md")
    lines = [
        "# Unaccusative patch — delta vs baseline",
        f"**Baseline :** `{baseline_run_dir}`",
        f"**Patch    :** `{patch_run_dir}`",
        "",
    ]
    pcat = patch_summary.get("greedy_gen_by_cat", {})
    bcat = (baseline_summary.get("greedy_gen_by_cat", {})
            if baseline_summary else {})
    keys = sorted(set(list(pcat.keys()) + list(bcat.keys())))
    lines.append("| Catégorie | Baseline | Patch | Delta |")
    lines.append("|---|---:|---:|---:|")
    for k in keys:
        b = bcat.get(k, None)
        p = pcat.get(k, None)
        d = (p - b) if (b is not None and p is not None) else None
        b_s = f"{b:.2f}%" if b is not None else "—"
        p_s = f"{p:.2f}%" if p is not None else "—"
        d_s = f"{d:+.2f}" if d is not None else "—"
        lines.append(f"| {k} | {b_s} | {p_s} | {d_s} |")
    lines.append("")
    b_g = (baseline_summary.get("final_gen_exact_greedy")
           if baseline_summary else None)
    p_g = patch_summary.get("final_gen_exact_greedy")
    b_d = (baseline_summary.get("final_dev_exact_greedy")
           if baseline_summary else None)
    p_d = patch_summary.get("final_dev_exact_greedy")
    lines.append("| **gen_greedy global** | "
                 f"{(f'{b_g:.2f}%' if b_g is not None else '—')} | "
                 f"{(f'{p_g:.2f}%' if p_g is not None else '—')} | "
                 f"{(f'{p_g - b_g:+.2f}' if (b_g is not None and p_g is not None) else '—')} |")
    lines.append("| **dev_greedy** | "
                 f"{(f'{b_d:.2f}%' if b_d is not None else '—')} | "
                 f"{(f'{p_d:.2f}%' if p_d is not None else '—')} | "
                 f"{(f'{p_d - b_d:+.2f}' if (b_d is not None and p_d is not None) else '—')} |")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {out_path}")


def _write_targeted_check(patch_run_dir):
    """Greedy-decode the 10 witness examples and write targeted_check.txt."""
    import torch
    ckpt_path = os.path.join(patch_run_dir, "checkpoint.pt")
    if not os.path.exists(ckpt_path):
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    in_w2i, out_w2i = ckpt["in_w2i"], ckpt["out_w2i"]
    out_i2w = {v: k for k, v in out_w2i.items()}

    n_layers = ckpt.get("n_layers", 2)
    n_heads = ckpt.get("n_heads", 4)
    ffn = ckpt.get("ffn", 256)
    use_jepa = bool(ckpt.get("use_jepa", False))
    use_jepa_ema = bool(ckpt.get("use_jepa_ema", False))
    jepa_ema_decay = float(ckpt.get("jepa_ema_decay", 0.99))

    if ckpt.get("use_tags"):
        in_to_out = cogs.build_in_to_out_map(in_w2i, out_w2i)
        token_cats = cogs.build_token_categories(in_w2i)
        model = cogs.TransformerSeq2SeqCopyTags(
            len(in_w2i), len(out_w2i), d_model=ckpt["d_model"],
            n_heads=n_heads, n_layers=n_layers, ffn=ffn,
            max_in=ckpt["max_in"], max_out=ckpt["max_out"],
            in_to_out_map=in_to_out, token_categories=token_cats,
            use_jepa=use_jepa, use_jepa_ema=use_jepa_ema,
            jepa_ema_decay=jepa_ema_decay).to(device)
    elif ckpt.get("use_copy"):
        in_to_out = cogs.build_in_to_out_map(in_w2i, out_w2i)
        model = cogs.TransformerSeq2SeqWithCopy(
            len(in_w2i), len(out_w2i), d_model=ckpt["d_model"],
            n_heads=n_heads, n_layers=n_layers, ffn=ffn,
            max_in=ckpt["max_in"], max_out=ckpt["max_out"],
            in_to_out_map=in_to_out, use_jepa=use_jepa,
            use_jepa_ema=use_jepa_ema,
            jepa_ema_decay=jepa_ema_decay).to(device)
    else:
        model = cogs.TransformerSeq2Seq(
            len(in_w2i), len(out_w2i), d_model=ckpt["d_model"],
            n_heads=n_heads, n_layers=n_layers, ffn=ffn,
            max_in=ckpt["max_in"], max_out=ckpt["max_out"],
            use_jepa=use_jepa, use_jepa_ema=use_jepa_ema,
            jepa_ema_decay=jepa_ema_decay).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    def _ids_to_tokens(ids):
        toks = []
        for i in ids:
            i = int(i)
            if i == 0:
                break
            w = out_i2w.get(i, f"<{i}>")
            if w == cogs.EOS:
                break
            if w == cogs.BOS:
                continue
            toks.append(w)
        return " ".join(toks)

    lines = ["# Targeted check — 10 Q_subj_active witnesses", ""]
    n_correct = 0
    with torch.no_grad():
        for inp, gold in TARGETED_EXAMPLES:
            src_ids = [in_w2i.get(t, 0) for t in inp.split()]
            src = torch.tensor([src_ids], dtype=torch.long, device=device)
            src_mask = (src != 0)
            pred_ids = cogs.greedy_decode(model, src, src_mask, out_w2i,
                                           max_len=ckpt["max_out"])
            pred = _ids_to_tokens(pred_ids[0].tolist()).strip()
            ok = pred == gold.strip()
            if ok:
                n_correct += 1
            lines.append(f"[{'✓' if ok else '✗'}] {inp}")
            lines.append(f"   GOLD: {gold}")
            lines.append(f"   PRED: {pred}")
            lines.append("")
    lines.insert(2, f"**Score : {n_correct}/{len(TARGETED_EXAMPLES)} corrects**")
    lines.insert(3, "")
    out_path = os.path.join(patch_run_dir, "targeted_check.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {out_path}  ({n_correct}/{len(TARGETED_EXAMPLES)} correct)")


def _write_pool_used(patch_run_dir):
    out_path = os.path.join(patch_run_dir, "pool_used.txt")
    lines = [
        "=== Unaccusative verb pool used ===",
        f"Usable ({len(_UNACC_POOL_USED)}): {_UNACC_POOL_USED}",
        f"Skipped ({len(_UNACC_POOL_SKIPPED)}):",
    ]
    for lemma, reason in _UNACC_POOL_SKIPPED:
        lines.append(f"  {lemma}: {reason}")
    lines.append("")
    lines.append("=== Removed from permutation pool ===")
    lines.append(f"{_UNACC_REMOVED_FROM_PERM}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {out_path}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Reuse COGS parser so SLOG auto-inherits every new mechanism.
    p = cogs.build_arg_parser(description="SLOG structural generalization")
    # SLOG-only additions:
    p.add_argument("--unaccusative-aug", action="store_true",
                   help="Full patch: inject 300 unaccusative examples AND remove unaccusative verbs from permutation pool.")
    p.add_argument("--unaccusative-pool-only", action="store_true",
                   help="Control: remove unaccusative verbs from permutation pool WITHOUT injecting the 300 examples. Isolates the pool-change effect.")
    # NOTE: --filter-unaccusative-from-pool is already defined in cogs.build_arg_parser()
    # (canonical name from the COGS sweep spec). We just consume args.filter_unaccusative_from_pool below.
    p.add_argument("--unaccusative-count", type=int, default=300,
                   help="Number of unaccusative examples to inject (default 300).")
    args = p.parse_args()
    args._bench_name = "SLOG"

    # Wire the runtime flags used by the monkey-patches.
    # --unaccusative-aug turns both knobs on; --unaccusative-pool-only (legacy name)
    # or --filter-unaccusative-from-pool (canonical, sweep spec) turn only the
    # pool exclusion on. All three names are compatible and additive.
    full_on = bool(getattr(args, "unaccusative_aug", False))
    pool_only = (bool(getattr(args, "unaccusative_pool_only", False))
                 or bool(getattr(args, "filter_unaccusative_from_pool", False)))
    _UNACC_INJECT = full_on
    _UNACC_POOL_EXCLUDE = full_on or pool_only
    _UNACC_COUNT = int(getattr(args, "unaccusative_count", 300))
    _UNACC_SEED = int(args.seed)

    if args.runs_dir:
        cogs.RUNS_DIR = args.runs_dir
        os.makedirs(args.runs_dir, exist_ok=True)

    summary = cogs.train_one_run(args.variant, args.seed, args)

    # Post-analysis — for the full patch AND the pool-only control, so we can
    # compare both runs against the same baseline with one script path.
    if (_UNACC_INJECT or _UNACC_POOL_EXCLUDE) and summary is not None:
        patch_run_dir = summary.get("run_dir")
        if patch_run_dir and os.path.isdir(patch_run_dir):
            print("\n=== Post-training analysis ===")
            _write_pool_used(patch_run_dir)
            baseline_run_dir = _find_latest_run_dir(BASELINE_PATTERN)
            if baseline_run_dir is None:
                print(f"  baseline not found under {BASELINE_PATTERN} — skipping delta")
            else:
                _write_delta_vs_baseline(patch_run_dir, baseline_run_dir)
            try:
                _write_targeted_check(patch_run_dir)
            except Exception as e:
                print(f"  targeted_check failed: {e}")
