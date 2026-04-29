#!/bin/bash
# Diagnostic SLOG — 2 runs ciblés pour identifier la cause de la dégradation
# JEPA sur SLOG (P2 = 26.86% vs ref non-JEPA = 30.14%).
#
# Lancement : bash run_slog_diag.sh
# Compute : ~2-2h30 total (D1 ~50min, D2 ~80min)

set -e
export PYTHONUNBUFFERED=1

LOG_DIR="runs_meta/slog_diag_logs"
mkdir -p "$LOG_DIR"

echo "════════════════════════════════════════════════════════════"
echo "  SLOG diagnostic started at $(date)"
echo "════════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────────────────────────────
# D1 — H1 : augmentations réduites (peel 5/3, pas de wh-passive)
# ─────────────────────────────────────────────────────────────────────
echo
echo "─── D1 : augmentations réduites (peel-stack=5, peel-cp=3, sans wh-passive) ───"
python3 slog_compositional.py \
    --variant B4 --copy \
    --peel-stack --peel-stack-depth=5 \
    --peel-cp --peel-cp-depth=3 \
    --permute-verbs \
    --filter-unaccusative-from-pool \
    --jepa --jepa-lambda 0.1 \
    --tfr-start=20 \
    --seed 42 --epochs 90 \
    --runs-dir runs/B4_slog_jepa_diag_H1_smaller_aug 2>&1 | tee "$LOG_DIR/d1.log" || \
    echo "WARNING: D1 failed ; continuing"

python3 write_baseline_sanity.py --piste D1_smaller_aug \
    --run-dir 'runs/B4_slog_jepa_diag_H1_smaller_aug/B4_s42_*' || true

# ─────────────────────────────────────────────────────────────────────
# D2 — H2 : sans scheduled sampling (tfr-start=999)
# ─────────────────────────────────────────────────────────────────────
echo
echo "─── D2 : sans scheduled sampling (tfr-start=999) ───"
python3 slog_compositional.py \
    --variant B4 --copy \
    --peel-stack --peel-stack-depth=10 \
    --peel-cp --peel-cp-depth=10 \
    --permute-verbs \
    --wh-passive-aug \
    --filter-unaccusative-from-pool \
    --jepa --jepa-lambda 0.1 \
    --tfr-start=999 \
    --seed 42 --epochs 90 \
    --runs-dir runs/B4_slog_jepa_diag_H2_no_tfr 2>&1 | tee "$LOG_DIR/d2.log" || \
    echo "WARNING: D2 failed ; continuing"

python3 write_baseline_sanity.py --piste D2_no_tfr \
    --run-dir 'runs/B4_slog_jepa_diag_H2_no_tfr/B4_s42_*' || true

# ─────────────────────────────────────────────────────────────────────
# Verdict
# ─────────────────────────────────────────────────────────────────────
echo
echo "─── Verdict ───"
python3 diag_slog_summary.py \
    --p2-run 'runs/B4_slog_jepa_p2_epochs90/B4_s42_*' \
    --d1-run 'runs/B4_slog_jepa_diag_H1_smaller_aug/B4_s42_*' \
    --d2-run 'runs/B4_slog_jepa_diag_H2_no_tfr/B4_s42_*' \
    --output runs/diag_slog_summary.md | tee "$LOG_DIR/verdict.log"

echo
echo "════════════════════════════════════════════════════════════"
echo "  SLOG diagnostic finished at $(date)"
echo "  Verdict : runs/diag_slog_summary.md"
echo "════════════════════════════════════════════════════════════"
