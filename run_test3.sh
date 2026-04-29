#!/bin/bash
# TEST 3 — Diagnostic conflit gradient task/JEPA sur COGS et SLOG
# Compute estimé : ~5h
# Pour chaque benchmark × seed : entraîner B4+JEPA avec --grad-snapshots-out
# Puis aggréger pour calculer P_neg et appliquer le verdict.

set -e
export PYTHONUNBUFFERED=1

OUT_DIR="tests/test3_grad_conflict"
mkdir -p "$OUT_DIR/logs"

SEEDS=(42 123 456)

echo "════════════════════════════════════════════════════════════"
echo "  TEST 3 — gradient conflict diag started at $(date)"
echo "════════════════════════════════════════════════════════════"

# ── COGS
for SEED in "${SEEDS[@]}"; do
    GRAD_OUT="$OUT_DIR/cogs_s${SEED}_grad.json"
    if [ -f "$GRAD_OUT" ]; then
        echo "  COGS seed $SEED : grad json already present"
        continue
    fi
    echo
    echo "─── COGS seed $SEED ───"
    python3 cogs_compositional.py \
        --variant B4 --copy \
        --peel-stack --peel-stack-depth=5 \
        --peel-cp --peel-cp-depth=3 \
        --permute-verbs \
        --jepa --jepa-lambda 0.1 \
        --tfr-start=20 \
        --grad-snapshots-out "$GRAD_OUT" \
        --seed "$SEED" --epochs 60 \
        --runs-dir "runs/test3_cogs_s${SEED}" 2>&1 | tee "$OUT_DIR/logs/cogs_s${SEED}.log" || \
        echo "WARNING: COGS seed $SEED failed"
done

# ── SLOG
for SEED in "${SEEDS[@]}"; do
    GRAD_OUT="$OUT_DIR/slog_s${SEED}_grad.json"
    if [ -f "$GRAD_OUT" ]; then
        echo "  SLOG seed $SEED : grad json already present"
        continue
    fi
    echo
    echo "─── SLOG seed $SEED ───"
    python3 slog_compositional.py \
        --variant B4 --copy \
        --peel-stack --peel-stack-depth=10 \
        --peel-cp --peel-cp-depth=10 \
        --permute-verbs --wh-passive-aug --filter-unaccusative-from-pool \
        --jepa --jepa-lambda 0.1 \
        --tfr-start=20 \
        --grad-snapshots-out "$GRAD_OUT" \
        --seed "$SEED" --epochs 60 \
        --runs-dir "runs/test3_slog_s${SEED}" 2>&1 | tee "$OUT_DIR/logs/slog_s${SEED}.log" || \
        echo "WARNING: SLOG seed $SEED failed"
done

# ── Aggregator
echo
echo "─── Aggregation ───"
python3 test3_grad_aggregate.py \
    --cogs-glob "$OUT_DIR/cogs_s{seed}_grad.json" \
    --slog-glob "$OUT_DIR/slog_s{seed}_grad.json" \
    --seeds 42,123,456 \
    --output-dir "$OUT_DIR" 2>&1 | tee "$OUT_DIR/logs/aggregate.log"

echo
echo "════════════════════════════════════════════════════════════"
echo "  TEST 3 finished at $(date)"
echo "  Verdict : $OUT_DIR/verdict.txt"
echo "════════════════════════════════════════════════════════════"
