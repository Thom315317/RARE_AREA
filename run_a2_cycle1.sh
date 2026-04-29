#!/bin/bash
# Phase A.2 — Cycle 1 : un cycle complet d'augmentation auto sur COGS.
# Compute total : ~50 min.
#
# Pré-requis :
#   - runs/B4_jepa_s42/B4_s42_*/checkpoint.pt (B4+JEPA cycle 0)
#   - meta_data/cogs_meta_dataset_bplus.jsonl + cogs_meta_splits.json
#   - meta_data/token_freq.json
#   - data/cogs/train.tsv

set -e
export PYTHONUNBUFFERED=1

CYCLE_DIR="runs_meta/etape_a2/cycle_1"
LOG_DIR="$CYCLE_DIR/logs"
mkdir -p "$LOG_DIR"

echo "════════════════════════════════════════════════════════════"
echo "  A.2 Cycle 1 started at $(date)"
echo "════════════════════════════════════════════════════════════"

# ── Étape 1 — identifier les zones de risque dans le train
echo
echo "─── Étape 1 : meta-encoder scoring sur train ───"
python3 meta_a2_identify_risk.py \
    --checkpoint 'runs/B4_jepa_s42/B4_s42_*/checkpoint.pt' \
    --meta-dataset meta_data/cogs_meta_dataset_bplus.jsonl \
    --meta-splits  meta_data/cogs_meta_splits.json \
    --train-file   data/cogs/train.tsv \
    --token-freq   meta_data/token_freq.json \
    --output-dir   "$CYCLE_DIR" \
    --threshold    0.5 \
    --seed 42 2>&1 | tee "$LOG_DIR/etape1_identify.log"

# ── Étape 2/3 — clustering + génération de variantes
echo
echo "─── Étape 2/3 : génération de variantes ───"
python3 meta_a2_generate_variants.py \
    --flagged "$CYCLE_DIR/flagged.json" \
    --output  "$CYCLE_DIR/extra_train.tsv" \
    --max-per-example 3 \
    --max-total 2000 \
    --seed 42 2>&1 | tee "$LOG_DIR/etape23_generate.log"

# ── Étape 4 — réentraînement avec extra-train
echo
echo "─── Étape 4 : réentraînement B4+JEPA + extra_train ───"
python3 cogs_compositional.py \
    --variant B4 --copy \
    --peel-stack --peel-stack-depth=5 \
    --peel-cp --peel-cp-depth=3 \
    --permute-verbs \
    --jepa --jepa-lambda 0.1 \
    --tfr-start=20 \
    --extra-train-file "$CYCLE_DIR/extra_train.tsv" \
    --seed 42 --epochs 60 \
    --runs-dir runs/B4_a2_cycle1 2>&1 | tee "$LOG_DIR/etape4_retrain.log"

# ── Étape 5 — évaluation et delta
echo
echo "─── Étape 5 : delta vs cycle 0 ───"
python3 meta_a2_evaluate.py \
    --current  'runs/B4_a2_cycle1/B4_s42_*/summary.json' \
    --previous 'runs/B4_jepa_s42/B4_s42_*/summary.json' \
    --output   "$CYCLE_DIR/delta_vs_cycle0.md" \
    --cycle-name 1 2>&1 | tee "$LOG_DIR/etape5_eval.log"

# ── Validation auto (module C) sur le nouveau modèle (sanity uniquement)
# meta_validation.py ne s'applique pas directement sur summary.json, on skip.

echo
echo "════════════════════════════════════════════════════════════"
echo "  A.2 Cycle 1 finished at $(date)"
echo "  Verdict : $CYCLE_DIR/delta_vs_cycle0.md"
echo "════════════════════════════════════════════════════════════"
