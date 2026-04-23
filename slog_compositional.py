#!/usr/bin/env python3
"""SLOG structural generalization benchmark — wraps cogs_compositional.py.

SLOG (Li et al., EMNLP 2023) extends COGS with 17 harder structural
generalization categories (deep PP recursion, center embedding, relative
clauses, questions, etc.). Same format, same architecture, same augmentations.

This wrapper reuses the full CLI of cogs_compositional.py so every mechanism
added to COGS (peel-stack, peel-cp, cross-voice, permute-verbs, causal
curriculum, scaling flags, …) is automatically available for SLOG.

Data must be pre-downloaded to data/slog/ (see sweep_pub.sh or README).

Usage:
  python3 slog_compositional.py --variant B0 --seed 42 --epochs 60
  python3 slog_compositional.py --variant B4 --copy --peel-stack \\
      --permute-verbs --tfr-start=20 --seed 42 --epochs 60
"""
import os, sys, argparse

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

# Patch the data loading in train_one_run
_orig_train_one_run = cogs.train_one_run


def _patched_train_one_run(variant, seed, args):
    # Patch _download to be a no-op (data already local)
    orig_download = cogs._download
    cogs._download = lambda url, dest: None

    # Patch parse paths: train_one_run reads from DATA_DIR/train.tsv etc.
    # We need gen to point to the right file
    import shutil
    gen_link = os.path.join(SLOG_DATA, "gen.tsv")
    if not os.path.exists(gen_link):
        shutil.copy2(_SLOG_LOCAL["gen"], gen_link)

    # Also copy aug_struct.tsv from cogs data if B3 needs it
    cogs_aug = os.path.join(HERE, "data", "cogs", "aug_struct.tsv")
    slog_aug = os.path.join(SLOG_DATA, "aug_struct.tsv")
    if not os.path.exists(slog_aug) and os.path.exists(cogs_aug):
        shutil.copy2(cogs_aug, slog_aug)

    result = _orig_train_one_run(variant, seed, args)
    cogs._download = orig_download
    return result


cogs.train_one_run = _patched_train_one_run


if __name__ == "__main__":
    # Use the SAME CLI as cogs_compositional.py so SLOG auto-inherits every
    # new mechanism (peel-cp, permute-verbs, causal-curriculum, scaling, …).
    p = cogs.build_arg_parser(description="SLOG structural generalization")
    args = p.parse_args()
    args._bench_name = "SLOG"

    if args.runs_dir:
        cogs.RUNS_DIR = args.runs_dir
        os.makedirs(args.runs_dir, exist_ok=True)

    cogs.train_one_run(args.variant, args.seed, args)
