#!/bin/bash
# Pipeline SLOG complet : 3 baselines → sélection → meta-modèle → A.1 → comparaison COGS vs SLOG
# Lancement : bash run_slog_full_pipeline.sh
# Compute total estimé : ~7h30. À lancer en début de nuit.
#
# Hypothèses :
#   - .venv activé OU le shebang #!/usr/bin/env python3 fonctionne avec torch dispo
#   - Les phases COGS sont déjà finies (pour comparaison finale)
#   - Le dossier runs/ existe

set -e  # arrêt sur erreur. Pour les phases dont l'échec ne doit pas bloquer
        # le pipeline, encadrer avec || true.

# Forcer Python en mode unbuffered pour que les logs par epoch
# (cogs_compositional `E000 ★ loss=…`) apparaissent en temps réel à travers
# tee, pas par paquets.
export PYTHONUNBUFFERED=1

LOG_DIR="runs_meta/slog_pipeline_logs"
mkdir -p "$LOG_DIR"

echo "════════════════════════════════════════════════════════════════"
echo "  SLOG full pipeline started at $(date)"
echo "════════════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────────────────────────────
# Étape A — 3 baselines candidates (séries)
# ─────────────────────────────────────────────────────────────────────

echo
echo "─── Piste 1 : λ=0.05 ───"
python3 slog_compositional.py \
    --variant B4 --copy \
    --peel-stack --peel-stack-depth=10 --peel-cp --peel-cp-depth=10 \
    --permute-verbs --wh-passive-aug --filter-unaccusative-from-pool \
    --jepa --jepa-lambda 0.05 \
    --tfr-start=20 --seed 42 --epochs 60 \
    --runs-dir runs/B4_slog_jepa_p1_lambda005 2>&1 | tee "$LOG_DIR/piste1.log" || \
    echo "WARNING: piste 1 failed ; continuing"

python3 write_baseline_sanity.py --piste p1_lambda005 \
    --run-dir 'runs/B4_slog_jepa_p1_lambda005/B4_s42_*' || true

echo
echo "─── Piste 2 : λ=0.1, 90 epochs ───"
python3 slog_compositional.py \
    --variant B4 --copy \
    --peel-stack --peel-stack-depth=10 --peel-cp --peel-cp-depth=10 \
    --permute-verbs --wh-passive-aug --filter-unaccusative-from-pool \
    --jepa --jepa-lambda 0.1 \
    --tfr-start=20 --seed 42 --epochs 90 \
    --runs-dir runs/B4_slog_jepa_p2_epochs90 2>&1 | tee "$LOG_DIR/piste2.log" || \
    echo "WARNING: piste 2 failed ; continuing"

python3 write_baseline_sanity.py --piste p2_epochs90 \
    --run-dir 'runs/B4_slog_jepa_p2_epochs90/B4_s42_*' || true

echo
echo "─── Piste 3 : λ=0.1 + EMA target ───"
python3 slog_compositional.py \
    --variant B4 --copy \
    --peel-stack --peel-stack-depth=10 --peel-cp --peel-cp-depth=10 \
    --permute-verbs --wh-passive-aug --filter-unaccusative-from-pool \
    --jepa --jepa-lambda 0.1 --jepa-ema --jepa-ema-decay 0.99 \
    --tfr-start=20 --seed 42 --epochs 60 \
    --runs-dir runs/B4_slog_jepa_p3_ema 2>&1 | tee "$LOG_DIR/piste3.log" || \
    echo "WARNING: piste 3 failed ; continuing"

python3 write_baseline_sanity.py --piste p3_ema \
    --run-dir 'runs/B4_slog_jepa_p3_ema/B4_s42_*' || true

# ─────────────────────────────────────────────────────────────────────
# Sélection automatique du meilleur baseline
# ─────────────────────────────────────────────────────────────────────

echo
echo "─── Sélection du meilleur baseline ───"
python3 select_best_slog_baseline.py \
    --pistes p1_lambda005,p2_epochs90,p3_ema \
    --runs-prefix runs/B4_slog_jepa_ \
    --output runs/best_slog_baseline.json | tee "$LOG_DIR/select.log"

BEST_RUN_DIR=$(python3 -c "import json; print(json.load(open('runs/best_slog_baseline.json'))['metrics']['run_dir'])")
BEST_PISTE=$(python3 -c "import json; print(json.load(open('runs/best_slog_baseline.json'))['choice'])")
echo "Best : $BEST_PISTE → $BEST_RUN_DIR"

# ─────────────────────────────────────────────────────────────────────
# Phase 2 — Dataset meta SLOG
# ─────────────────────────────────────────────────────────────────────

echo
echo "─── Phase 2 : meta-dataset SLOG ───"
python3 meta_dataset_builder.py \
    --checkpoint "$BEST_RUN_DIR/checkpoint.pt" \
    --benchmark slog \
    --output meta_data/slog_meta_dataset.jsonl 2>&1 | tee "$LOG_DIR/phase2.log"

# ─────────────────────────────────────────────────────────────────────
# Phase 2.5 — identifier cats problématiques SLOG
# ─────────────────────────────────────────────────────────────────────

echo
echo "─── Phase 2.5 : identifier cats prob SLOG ───"
python3 slog_identify_problem_cats.py \
    --dataset meta_data/slog_meta_dataset.jsonl \
    --top 4 \
    --output meta_data/slog_problem_cats.txt | tee "$LOG_DIR/phase25.log"

PROB_CATS=$(cat meta_data/slog_problem_cats.txt)
echo "Problem cats SLOG : $PROB_CATS"

# ─────────────────────────────────────────────────────────────────────
# Phase 3 — MetaEncoder robust SLOG (3 seeds)
# ─────────────────────────────────────────────────────────────────────

echo
echo "─── Phase 3 : MetaEncoder robust SLOG ───"
python3 meta_train_etape3_robust.py \
    --dataset meta_data/slog_meta_dataset.jsonl \
    --splits  meta_data/slog_meta_splits.json \
    --output-dir runs_meta/etape_slog_robust \
    --seeds 42,123,456 \
    --p-perturb 0.5 \
    --problem-cats "$PROB_CATS" 2>&1 | tee "$LOG_DIR/phase3.log"

# ─────────────────────────────────────────────────────────────────────
# Phase 4 — Validation auto
# ─────────────────────────────────────────────────────────────────────

echo
echo "─── Phase 4 : validation auto ───"
python3 meta_validation.py \
    --metrics runs_meta/etape_slog_robust/seed_42/metrics.json \
    --output runs_meta/etape_slog_robust/auto_alerts.md 2>&1 | tee "$LOG_DIR/phase4.log" || true

# ─────────────────────────────────────────────────────────────────────
# Phase 5 — A.1 selective prediction
# ─────────────────────────────────────────────────────────────────────

echo
echo "─── Phase 5 : A.1 selective prediction ───"
python3 meta_etape_A1_selective.py \
    --runs runs_meta/etape_slog_robust/seed_42,runs_meta/etape_slog_robust/seed_123,runs_meta/etape_slog_robust/seed_456 \
    --output-dir runs_meta/etape_slog_A1 2>&1 | tee "$LOG_DIR/phase5.log"

# ─────────────────────────────────────────────────────────────────────
# Bilan : tableau COGS vs SLOG
# ─────────────────────────────────────────────────────────────────────

echo
echo "─── Bilan COGS vs SLOG ───"
python3 meta_cogs_vs_slog.py \
    --cogs-robust runs_meta/etapeB_robust \
    --cogs-a1     runs_meta/etapeA1_robust \
    --slog-robust runs_meta/etape_slog_robust \
    --slog-a1     runs_meta/etape_slog_A1 \
    --output      runs_meta/cogs_vs_slog_final.md 2>&1 | tee "$LOG_DIR/cogs_vs_slog.log" || true

echo
echo "════════════════════════════════════════════════════════════════"
echo "  SLOG full pipeline finished at $(date)"
echo "  Best baseline : $BEST_PISTE"
echo "  Comparison    : runs_meta/cogs_vs_slog_final.md"
echo "════════════════════════════════════════════════════════════════"
