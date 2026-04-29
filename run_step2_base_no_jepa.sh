#!/bin/bash
# Pipeline propre — ÉTAPE 2 : re-mesure modèle de base SANS JEPA
# Compute estimé : ~2.5h (3 seeds × 2 benchmarks × ~25 min)
#
# Hypothèse à vérifier :
#   COGS Δ ∈ [-1pp, +1pp]   (neutre)
#   SLOG Δ ∈ [+2pp, +5pp]   (gain attendu de retirer JEPA)

set +e
export PYTHONUNBUFFERED=1

OUT="pipeline_clean/step2_base_model"
mkdir -p "$OUT/logs" "$OUT/runs"
SEEDS=(42 123 456)

echo "════════════════════════════════════════════════════════════"
echo "  ÉTAPE 2 — base model NO JEPA started at $(date)"
echo "════════════════════════════════════════════════════════════"

for SEED in "${SEEDS[@]}"; do
    # ── COGS B4 sans --jepa
    DIR="$OUT/runs/cogs_s${SEED}"
    if [ -n "$(ls $DIR/B4_s${SEED}_*/summary.json 2>/dev/null)" ]; then
        echo "  COGS seed $SEED already present in $DIR — skipping"
    else
        echo
        echo "─── COGS seed $SEED (no JEPA) ───"
        python3 cogs_compositional.py \
            --variant B4 --copy \
            --peel-stack --peel-stack-depth=5 \
            --peel-cp --peel-cp-depth=3 \
            --permute-verbs \
            --tfr-start=20 \
            --seed "$SEED" --epochs 60 \
            --runs-dir "$DIR" 2>&1 | tee "$OUT/logs/cogs_s${SEED}.log" || \
            echo "WARNING: COGS seed $SEED failed"
    fi

    # ── SLOG B4 sans --jepa
    DIR="$OUT/runs/slog_s${SEED}"
    if [ -n "$(ls $DIR/B4_s${SEED}_*/summary.json 2>/dev/null)" ]; then
        echo "  SLOG seed $SEED already present in $DIR — skipping"
    else
        echo
        echo "─── SLOG seed $SEED (no JEPA) ───"
        python3 slog_compositional.py \
            --variant B4 --copy \
            --peel-stack --peel-stack-depth=10 \
            --peel-cp --peel-cp-depth=10 \
            --permute-verbs --wh-passive-aug --filter-unaccusative-from-pool \
            --tfr-start=20 \
            --seed "$SEED" --epochs 60 \
            --runs-dir "$DIR" 2>&1 | tee "$OUT/logs/slog_s${SEED}.log" || \
            echo "WARNING: SLOG seed $SEED failed"
    fi
done

echo
echo "─── Comparaison vs runs JEPA historiques ───"
python3 step2_compare.py \
    --new-cogs-glob "$OUT/runs/cogs_s{seed}/B4_s{seed}_*/summary.json" \
    --new-slog-glob "$OUT/runs/slog_s{seed}/B4_s{seed}_*/summary.json" \
    --hist-cogs-glob "runs/B4_jepa_s{seed}/B4_s{seed}_*/summary.json" \
    --hist-slog-glob "runs/B4_slog_jepa/B4_s{seed}_*/summary.json" \
    --seeds 42,123,456 \
    --output-dir "$OUT" 2>&1 | tee "$OUT/logs/compare.log"

echo
echo "════════════════════════════════════════════════════════════"
echo "  ÉTAPE 2 finished at $(date)"
echo "  Verdict : $OUT/verdict.txt"
echo "════════════════════════════════════════════════════════════"
