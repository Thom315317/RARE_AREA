#!/bin/bash
# TEST 1 — Reproduction A.2 cycle 1 sur 3 seeds
# Compute estimé : ~3-5h
#
# Pour chaque seed s ∈ {42, 123, 456} :
#   1. Train cycle 0 (B4+JEPA) si manquant
#   2. Run A.2 cycle 1 (identify_risk + generate + retrain)
# Puis aggregator des deltas.

set -e
export PYTHONUNBUFFERED=1

OUT_DIR="tests/test1_a2_repro"
mkdir -p "$OUT_DIR/logs"

SEEDS=(42 123 456)

echo "════════════════════════════════════════════════════════════"
echo "  TEST 1 — A.2 reproduction (3 seeds) started at $(date)"
echo "════════════════════════════════════════════════════════════"

for SEED in "${SEEDS[@]}"; do
    echo
    echo "═══ SEED $SEED ═══"

    # ── Cycle 0 : B4+JEPA si manquant
    CYCLE0_DIR="runs/B4_jepa_s${SEED}"
    if [ -n "$(ls "$CYCLE0_DIR"/B4_s${SEED}_*/checkpoint.pt 2>/dev/null)" ]; then
        echo "  Cycle 0 already present in $CYCLE0_DIR — skipping training"
    else
        echo "  Training cycle 0 (B4+JEPA seed $SEED)..."
        python3 cogs_compositional.py \
            --variant B4 --copy \
            --peel-stack --peel-stack-depth=5 \
            --peel-cp --peel-cp-depth=3 \
            --permute-verbs \
            --jepa --jepa-lambda 0.1 \
            --tfr-start=20 \
            --seed "$SEED" --epochs 60 \
            --runs-dir "$CYCLE0_DIR" 2>&1 | tee "$OUT_DIR/logs/cycle0_s${SEED}.log" || \
            { echo "WARNING: cycle 0 seed $SEED failed"; continue; }
    fi

    # ── Cycle 1 : A.2 identify_risk + generate + retrain
    CYCLE1_DIR="runs/B4_a2_cycle1_s${SEED}"
    META_OUT="tests/test1_a2_repro/cycle1_meta_s${SEED}"
    if [ -n "$(ls "$CYCLE1_DIR"/B4_s${SEED}_*/summary.json 2>/dev/null)" ]; then
        echo "  Cycle 1 seed $SEED already present — skipping"
    else
        mkdir -p "$META_OUT"
        echo "  Running A.2 identify_risk seed $SEED..."
        python3 meta_a2_identify_risk.py \
            --checkpoint "$CYCLE0_DIR/B4_s${SEED}_*/checkpoint.pt" \
            --meta-dataset meta_data/cogs_meta_dataset_bplus.jsonl \
            --meta-splits  meta_data/cogs_meta_splits.json \
            --train-file   data/cogs/train.tsv \
            --token-freq   meta_data/token_freq.json \
            --output-dir   "$META_OUT" \
            --threshold    0.5 \
            --seed "$SEED" 2>&1 | tee "$OUT_DIR/logs/identify_s${SEED}.log" || \
            { echo "WARNING: identify_risk seed $SEED failed"; continue; }

        echo "  Generating variants seed $SEED..."
        python3 meta_a2_generate_variants.py \
            --flagged "$META_OUT/flagged.json" \
            --output  "$META_OUT/extra_train.tsv" \
            --max-per-example 3 \
            --max-total 2000 \
            --seed "$SEED" 2>&1 | tee "$OUT_DIR/logs/generate_s${SEED}.log" || \
            { echo "WARNING: generate seed $SEED failed"; continue; }

        echo "  Training cycle 1 seed $SEED..."
        python3 cogs_compositional.py \
            --variant B4 --copy \
            --peel-stack --peel-stack-depth=5 \
            --peel-cp --peel-cp-depth=3 \
            --permute-verbs \
            --jepa --jepa-lambda 0.1 \
            --tfr-start=20 \
            --extra-train-file "$META_OUT/extra_train.tsv" \
            --seed "$SEED" --epochs 60 \
            --runs-dir "$CYCLE1_DIR" 2>&1 | tee "$OUT_DIR/logs/cycle1_s${SEED}.log" || \
            { echo "WARNING: cycle 1 retrain seed $SEED failed"; continue; }
    fi
done

echo
echo "─── Aggregation 3 seeds ───"
python3 test1_a2_aggregate.py \
    --cycle0-glob 'runs/B4_jepa_s{seed}/B4_s{seed}_*/summary.json' \
    --cycle1-glob 'runs/B4_a2_cycle1_s{seed}/B4_s{seed}_*/summary.json' \
    --seeds 42,123,456 \
    --output-dir "$OUT_DIR" 2>&1 | tee "$OUT_DIR/logs/aggregate.log"

echo
echo "════════════════════════════════════════════════════════════"
echo "  TEST 1 finished at $(date)"
echo "  Verdict : $OUT_DIR/verdict.txt"
echo "════════════════════════════════════════════════════════════"
