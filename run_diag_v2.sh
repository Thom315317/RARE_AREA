#!/bin/bash
# Orchestration des 3 tests diagnostiques pré-refonte v2.
# Lance les 3 en série. Chaque test continue même si les autres échouent.
#
# Compute total estimé : ~10-12h (à laisser tourner la nuit sur 1-2 nuits)
#
# Lancement :
#   chmod +x run_diag_v2.sh
#   bash run_diag_v2.sh 2>&1 | tee tests/diag_v2.log

set +e  # NE PAS bloquer si un test échoue
export PYTHONUNBUFFERED=1

mkdir -p tests

echo "════════════════════════════════════════════════════════════════"
echo "  3 tests diag pré-refonte v2 — start at $(date)"
echo "════════════════════════════════════════════════════════════════"

echo
echo "######## TEST 1 — A.2 reproduction (3 seeds) ########"
bash run_test1.sh 2>&1 | tee tests/test1_full.log

echo
echo "######## TEST 2 — Ablation factorielle MetaEncoder ########"
python3 meta_test2_ablation.py \
    --cogs-dataset meta_data/cogs_meta_dataset_bplus.jsonl \
    --cogs-splits  meta_data/cogs_meta_splits.json \
    --slog-dataset meta_data/slog_meta_dataset.jsonl \
    --slog-splits  meta_data/slog_meta_splits.json \
    --output-dir   tests/test2_ablation \
    --seeds 42,123,456 \
    --skip-loco 2>&1 | tee tests/test2_full.log

echo
echo "######## TEST 3 — Gradient conflict task/JEPA ########"
bash run_test3.sh 2>&1 | tee tests/test3_full.log

echo
echo "######## SYNTHESE finale ########"
python3 diag_synthesis.py \
    --test1-dir tests/test1_a2_repro \
    --test2-dir tests/test2_ablation \
    --test3-dir tests/test3_grad_conflict \
    --output    tests/SYNTHESIS.md 2>&1 | tee tests/synthesis.log

echo
echo "════════════════════════════════════════════════════════════════"
echo "  3 tests finished at $(date)"
echo "  Verdicts par test :"
echo "    tests/test1_a2_repro/verdict.txt"
echo "    tests/test2_ablation/verdict.txt"
echo "    tests/test3_grad_conflict/verdict.txt"
echo "  Synthèse + orientation refonte :"
echo "    tests/SYNTHESIS.md"
echo "════════════════════════════════════════════════════════════════"
