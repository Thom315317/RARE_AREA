#!/bin/bash
# Pipeline propre post-Tests v2 — orchestration complète
# Étape 2 (~2.5h) → Étape 3 (~3h) → Étape 4 (synthèse, instantané)
# Total : ~5.5-6h compute. Peut tourner sur 1 nuit.
#
# Étape 1 (archivage) : déjà fait à la main (commit + branche, à pusher)

set +e
export PYTHONUNBUFFERED=1

mkdir -p pipeline_clean

echo "════════════════════════════════════════════════════════════"
echo "  Pipeline propre — start at $(date)"
echo "════════════════════════════════════════════════════════════"

# ── Étape 2
echo
echo "######## ÉTAPE 2 — modèle de base sans JEPA ########"
bash run_step2_base_no_jepa.sh 2>&1 | tee pipeline_clean/step2_full.log

# ── Étape 3
echo
echo "######## ÉTAPE 3 — MetaEncoder C5 (sans structure) ########"
python3 meta_step3_clean.py \
    --cogs-dataset meta_data/cogs_meta_dataset_bplus.jsonl \
    --cogs-splits  meta_data/cogs_meta_splits.json \
    --slog-dataset meta_data/slog_meta_dataset.jsonl \
    --slog-splits  meta_data/slog_meta_splits.json \
    --output-dir   pipeline_clean/step3_meta_encoder \
    --seeds 42,123,456 2>&1 | tee pipeline_clean/step3_full.log

# ── Étape 4 : synthèse
echo
echo "######## ÉTAPE 4 — synthèse PIPELINE_STATE.md ########"
python3 step4_pipeline_state.py \
    --step2-dir pipeline_clean/step2_base_model \
    --step3-dir pipeline_clean/step3_meta_encoder \
    --output    pipeline_clean/PIPELINE_STATE.md 2>&1 | tee pipeline_clean/step4_synthesis.log

echo
echo "════════════════════════════════════════════════════════════"
echo "  Pipeline propre finished at $(date)"
echo "  Verdicts :"
echo "    pipeline_clean/step2_base_model/verdict.txt"
echo "    pipeline_clean/step3_meta_encoder/verdict.txt"
echo "    pipeline_clean/step3_meta_encoder/slog_risk_diagnosis.md"
echo "    pipeline_clean/PIPELINE_STATE.md  (synthèse)"
echo "════════════════════════════════════════════════════════════"
