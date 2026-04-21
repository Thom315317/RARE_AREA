#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# SWEEP FOR PUBLICATION — all runs go to runs_pub/
#
# RTX 3070 8GB. Each SCAN run ~10-20 min. Each COGS run ~30-40 min.
# Total estimated: ~8-10 hours.
#
# Usage:
#   cd ~/rare_jepa && source .venv/bin/activate
#   bash sweep_pub.sh          # run everything sequentially
#   bash sweep_pub.sh scan     # SCAN only
#   bash sweep_pub.sh cogs     # COGS only
# ═══════════════════════════════════════════════════════════════════════
set -e

SCAN_DIR="runs_pub/scan"
COGS_DIR="runs_pub/cogs"
SLOG_DIR="runs_pub/slog"
SEEDS="42 123 456"

run_scan() {
    local variant=$1 epochs=$2
    for seed in $SEEDS; do
        echo ""
        echo "▶ SCAN $variant seed=$seed epochs=$epochs"
        python3 scan_compositional.py --variant "$variant" --seed "$seed" \
            --epochs "$epochs" --runs-dir "$SCAN_DIR"
    done
}

run_cogs() {
    local variant=$1 epochs=$2 patience=$3
    for seed in $SEEDS; do
        echo ""
        echo "▶ COGS $variant seed=$seed epochs=$epochs patience=$patience"
        python3 cogs_compositional.py --variant "$variant" --seed "$seed" \
            --epochs "$epochs" --patience "$patience" --runs-dir "$COGS_DIR"
    done
}

# ─────────────────────────────────────────────────────────
# SCAN addprim_jump (3 seeds × 2 variants)
# ─────────────────────────────────────────────────────────
run_scan_all() {
    echo "═══════════════════════════════════════════"
    echo " SCAN addprim_jump — publication sweep"
    echo "═══════════════════════════════════════════"

    # A0: baseline (~0% test, confirms the split)
    run_scan A0 50

    # A4_permute: the result (99.8%+)
    run_scan A4_permute 30
}

# ─────────────────────────────────────────────────────────
# COGS (3 seeds × 5 variants)
# ─────────────────────────────────────────────────────────
run_cogs_all() {
    echo "═══════════════════════════════════════════"
    echo " COGS — publication sweep"
    echo "═══════════════════════════════════════════"

    # B0: baseline (gen ~0%)
    run_cogs B0 60 20

    # B1: proper name permutation (gen ~14%)
    run_cogs B1 60 20

    # B3_auto: structural PP augmentation (gen ~2%, obj_pp_to_subj_pp ~40%)
    run_cogs B3_auto 60 20

    # B4: B1 + B3_auto combined (gen ~16%)
    run_cogs B4 60 20

    # B5: B4 + active↔passive + dative swap + common noun permutation
    run_cogs B5 60 20
}

# ─────────────────────────────────────────────────────────
# SLOG (3 seeds × 3 variants — same augmentations, harder splits)
# ─────────────────────────────────────────────────────────
run_slog() {
    local variant=$1 epochs=$2 patience=$3
    for seed in $SEEDS; do
        echo ""
        echo "▶ SLOG $variant seed=$seed epochs=$epochs patience=$patience"
        python3 slog_compositional.py --variant "$variant" --seed "$seed" \
            --epochs "$epochs" --patience "$patience" --runs-dir "$SLOG_DIR"
    done
}

run_slog_all() {
    echo "═══════════════════════════════════════════"
    echo " SLOG — publication sweep"
    echo "═══════════════════════════════════════════"

    # B0: baseline
    run_slog B0 60 20

    # B5: all augmentations (the full method)
    run_slog B5 60 20

    # B1: proper name permutation only (ablation)
    run_slog B1 60 20
}

# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
mkdir -p "$SCAN_DIR" "$COGS_DIR" "$SLOG_DIR"

case "${1:-all}" in
    scan) run_scan_all ;;
    cogs) run_cogs_all ;;
    slog) run_slog_all ;;
    all)  run_scan_all; run_cogs_all; run_slog_all ;;
    *)    echo "Usage: $0 [scan|cogs|slog|all]"; exit 1 ;;
esac

echo ""
echo "═══════════════════════════════════════════"
echo " SWEEP COMPLETE"
echo " Results in: runs_pub/"
echo "═══════════════════════════════════════════"

# Generate summary
echo ""
echo "Generating summaries..."
python3 results_summary.py --variant A0 --runs-dir "$SCAN_DIR" \
    --out runs_pub/scan_A0_summary.json 2>/dev/null || true
python3 results_summary.py --variant A4_permute --runs-dir "$SCAN_DIR" \
    --out runs_pub/scan_A4_summary.json 2>/dev/null || true
echo "Done. Check runs_pub/ for all results."
