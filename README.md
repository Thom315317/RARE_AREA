# RARE-AREA

**Autonomous Reasoning through Error Awareness**

A self-diagnosing neural system that detects, explains, and corrects its own compositional generalization failures — without external supervision.

---

## Results

| Model | Params | COGS gen (greedy) | Method |
|---|---|---|---|
| Transformer baseline (Kim & Linzen 2020) | 9.5M | 16-35% | Standard seq2seq |
| Lake & Baroni 2023 | ~3M | >99% lexical, 0% structural | Meta-learning |
| **RARE-AREA B0+copy** | **1.02M** | **52.48%** | Copy mechanism + scheduled sampling |
| **RARE-AREA B4+copy** | **1.02M** | **54.64%** | Copy + targeted augmentation |
| **RARE-AREA auto-loop** | **1.02M** | **16.67%** | Fully autonomous (no human intervention) |

SCAN addprim_jump: **99.84% ± 0.19%** (3 seeds) via action permutation.

### Key findings

1. **76% of "structural confusion" errors are lexical hallucinations.** The model says "Ella" instead of "Paula", not because it confuses agent/theme, but because it generates tokens from memory instead of copying from input. A copy mechanism (+128 params) resolves all lexical categories at once (0% → 97-99%).

2. **Causal diagnosis recovers human-designed fixes automatically.** The system identifies "0 nmod on subject in train vs 5495 on object" and "202 nouns seen only as object" — the same gaps a human researcher found manually — in 5 seconds of CPU time. 100% of hypotheses confirmed.

3. **Autonomous loop: 0.06% → 16.67% without human intervention.** The system measures its own uncertainty (JEPA-surprise), clusters errors without labels, identifies generators, and retrains. On SLOG: PP_modif_subj 1% → 31.7% autonomously.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  RARE-AREA                       │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Encoder  │→│  JEPA    │→│  Surprise map │  │
│  │ (2-layer)│  │ predictor│  │  (per token)  │  │
│  └────┬─────┘  └──────────┘  └───────┬───────┘  │
│       │                              │           │
│       ▼                              ▼           │
│  ┌──────────┐              ┌─────────────────┐   │
│  │ Decoder  │              │ Causal diagnosis│   │
│  │ + copy   │              │ (diff pred/gold)│   │
│  │ + sched. │              │ → hypotheses    │   │
│  │ sampling │              │ → verification  │   │
│  └────┬─────┘              │ → auto-fix      │   │
│       │                    └─────────────────┘   │
│       ▼                                          │
│  ┌──────────┐                                    │
│  │  Output  │                                    │
│  │  (LF)    │                                    │
│  └──────────┘                                    │
└─────────────────────────────────────────────────┘
```

---

## Project structure

Flat layout — all Python files at root for simple imports.

```
RARE_AREA/
├── README.md                      # This file
├── RESULTS.md                     # Chronological results log
├── CHANGELOG.md                   # Code evolution
├── MACHINE_B_SETUP.md             # Setup for secondary compute machine
├── requirements.txt
├── .gitignore
│
├── cogs_compositional.py          # COGS model + copy + tags + scheduled sampling
├── scan_compositional.py          # SCAN ablation (7 variants)
├── slog_compositional.py          # SLOG wrapper (reuses COGS model)
│
├── error_mirror.py                # JEPA-surprise + decoder entropy
├── cluster_gaps.py                # KMeans on structural features
├── causal_diagnosis.py            # Structural diff + hypotheses + verification
│
├── auto_augment.py                # Route clusters → generators (single pass)
├── auto_loop.py                   # Iterative loop v1 (cluster-based routing)
├── auto_loop_v2.py                # v2 (causal diagnosis-based routing)
├── auto_loop_v3.py                # v3 (lexical + structural fusion)
│
├── detect_permutable.py           # Distributional role-class detection
├── cogs_compare.py                # Per-category B0 vs Bx comparison
├── results_summary.py             # Multi-seed aggregation with IC95
│
├── sweep_final.py                 # 5-seed sweep orchestration
├── eval_copy.py                   # Re-run greedy eval on a copy checkpoint
│
└── data/                          # Datasets (gitignored, download separately)
    ├── cogs/
    ├── scan/
    └── slog/
```

---

## Quick start

### Requirements
```
Python 3.10+
PyTorch 2.0+ (CUDA)
GPU: 4+ GB VRAM (tested on RTX 3070 8GB)
```

### Install
```bash
git clone https://github.com/Thom315317/RARE_AREA.git
cd rare-area
pip install -r requirements.txt
```

### Download data
```bash
# COGS
git clone https://github.com/najoungkim/COGS.git data/cogs

# SCAN
git clone https://github.com/brendenlake/SCAN.git data/scan

# SLOG (password-protected zip; password: SLOG)
git clone https://github.com/bingzhilee/SLOG.git data/slog
```

### Run baseline + copy mechanism
```bash
# B0 with copy mechanism + scheduled sampling on COGS
python cogs_compositional.py --variant B0 --seed 42 --epochs 60 --copy

# Add structural role tagger (marginal gain)
python cogs_compositional.py --variant B0 --seed 42 --epochs 60 --tags

# B4 + copy (best single-seed result: 54.64%)
python cogs_compositional.py --variant B4 --seed 42 --epochs 60 --copy
```

### Run causal diagnosis
```bash
# Measure JEPA-surprise + decoder entropy
python error_mirror.py --checkpoint runs/cogs/B0_s42_.../checkpoint.pt --dataset cogs

# Structural diff + hypothesis verification
python causal_diagnosis.py --checkpoint runs/cogs/B0_s42_.../checkpoint.pt --dataset cogs
```

### Run autonomous loop
```bash
# v1 — cluster-based routing (0% → ~17% on COGS)
python auto_loop.py --dataset cogs --max-cycles 3

# v3 — lexical (v1) + causal diagnosis (v2) fusion
python auto_loop_v3.py --dataset cogs --max-cycles 3
```

### Run 5-seed sweep
```bash
# SCAN (3 splits × 4 variants × 5 seeds)
python sweep_final.py --benchmark scan

# COGS (variants × 5 seeds)
python sweep_final.py --benchmark cogs

# SLOG
python sweep_final.py --benchmark slog

# Recap (no training, just aggregate tables)
python sweep_final.py --recap
```

---

## Benchmarks

### SCAN (addprim_jump)

| Variant | Test exact match | Description |
|---|---|---|
| A0 Baseline | 0.0% | Standard seq2seq |
| A1 Role supervision | 0.1% | Auxiliary role head on encoder |
| A2 VQ post-encoder | 1.1% → 0.0% | Gumbel-Softmax prototypes |
| A4 Permute seen only | 0.4% → 0.1% | Permute {walk, run, look} |
| **A4 Permute all** | **99.84% ± 0.19%** | **Permute {walk, run, look, jump}** |

### COGS

| Variant | Gen greedy | Description |
|---|---|---|
| B0 Baseline | 0.02% | Standard seq2seq |
| B4 (augmentation) | 17.8% | Proper name perm + PP augmentation |
| **B0 + copy** | **52.48%** | **Copy mechanism + scheduled sampling** |
| **B4 + copy** | **54.64%** | **Copy + augmentation** |
| Auto-loop v1 | 16.67% | Fully autonomous |

### Causal diagnosis

| Diff type | % of errors | Hypothesis | Confirmed |
|---|---|---|---|
| wrong_arg_position | 42.6% | Lexical hallucination (76%) + structural swap (24%) | ✓ |
| missing_lexical_noun | 25.2% | 202 nouns seen only as object | ✓ |
| missing_nmod_on_subj | 11.8% | 0 nmod on subject in train vs 5495 on object | ✓ |
| missing_modification | 4.0% | Max nmod depth = 2 in train | ✓ |
| missing_verb_role | 0.8% | Verb unseen in this voice/valency | ✓ |

---

## Status

Work in progress. Results are from 1-3 seed runs; a 5-seed sweep is being prepared.
See `RESULTS.md` for the up-to-date chronological log.

---

## License

MIT
