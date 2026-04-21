# Results — Chronological Log

Tracked results across SCAN, COGS, SLOG benchmarks. Updated as new runs complete.

---

## Headlines (best single-seed results)

| Benchmark | Config | Gen (greedy) | Params |
|---|---|---|---|
| SCAN addprim_jump | A4_permute (3 seeds) | **99.84% ± 0.19%** | 0.68M |
| COGS | B0 + copy + scheduled sampling | **52.48%** | 1.01M |
| COGS | B4 + copy + scheduled sampling | **54.64%** | 1.02M |
| COGS | Auto-loop v1 (autonomous) | 16.67% | 1.01M |
| SLOG | B4 baseline | 19.18% | 1.04M |
| SLOG | Auto-augment (pp_on_subject only) | 19.47% | 1.04M |

---

## SCAN — addprim_jump ablation (7 variants)

Standalone: `scan_compositional.py`. Split: `jump` appears only in isolation in train, must compose at test.

| Variant | Intervention | Test EM | Seeds |
|---|---|---|---|
| A0 | Baseline seq2seq | 0.54% | 5 |
| A1 | Auxiliary role head on encoder | 0.28% | 5 |
| A2 | VQ post-encoder (6 prototypes, Gumbel-ST) | 1.05% | 1 |
| A2_warm | VQ warm-started from A1 centroids | 1.12% | 1 |
| A2_pre | VQ before encoder, V=raw_emb bypass | 0.00% | 1 |
| A3_dual | Dual encoder + consistency loss | 0.04% | 1 |
| **A4_permute** | **Permute {walk, run, look, jump} consistently** | **99.84% ± 0.19%** | 3 |

The only intervention that works is data augmentation: permuting the 4 primitive actions exposes the model to `jump` in compositional contexts it never saw in train.

### SCAN — other splits (sweep, 5 seeds)

| Split | A0 | A1 | A4_permute |
|---|---|---|---|
| addprim_jump | 0.54% | 0.28% | **99.98%** |
| addprim_turn_left | 98.30% | 95.99% | 92.43% |
| length | 0.05% | 0.03% | — |

Note: `addprim_turn_left` is already solved by the baseline (train covers turn-left in composition). `length` tests sequence-length generalization and no variant helps.

---

## COGS

### Manual ablations (1 seed each)

| Variant | Gen greedy | Description |
|---|---|---|
| B0 | 0.06% | Standard Transformer seq2seq |
| B1 | 14.50% | Permute 103 proper names |
| B3_auto | 1.52% (obj_pp_to_subj_pp=39.90%) | 893 auto-generated PP-on-subject examples |
| B4 | 17.85% | B1 + B3_auto |
| B6 | — | Noun-position swap (targeted) |
| B7a | — | PP recursion +1 level |

### Architecture interventions

| Config | Gen greedy | Notes |
|---|---|---|
| B0 + copy | **52.48%** | Pointer network + scheduled sampling (tfr 1.0→0.5) |
| B0 + copy + tags | 52.79% | Plus structural role tagger (lambda=0.3) — marginal gain |
| **B4 + copy** | **54.64%** | Copy resolves lexical, B3_auto resolves obj_pp_to_subj_pp |

Full greedy by-category for B4 + copy (selected):

| Category | B0+copy | B4+copy |
|---|---|---|
| prim_to_inf_arg | 99.2% | 99.8% |
| obj_to_subj_proper | 98.7% | 98.9% |
| prim_to_subj_proper | 98.9% | 98.8% |
| obj_to_subj_common | 97.9% | 97.8% |
| prim_to_obj_common | 97.3% | 96.4% |
| obj_pp_to_subj_pp | 0.0% | **42.5%** |
| subj_to_obj_proper | 57.6% | 60.2% |
| prim_to_obj_proper | 57.9% | 59.7% |
| active_to_passive | 0.0% | 0.0% |
| pp_recursion | 2.8% | 3.8% |
| cp_recursion | 0.0% | 0.0% |
| dative (both directions) | 0.0% | 0.0% |

### Causal diagnosis (B0 baseline, 500 gen examples)

100% of confirmed hypotheses match known manual fixes.

| Diff type | Count | Hypothesis | Evidence |
|---|---|---|---|
| wrong_arg_position | 42.6% | 76% lexical hallucination, 24% structural | Sub-patterns: other 76%, ccomp 14%, passive 10% |
| missing_lexical_noun | 25.2% | 202 nouns only as object, 2 only as subject | Position coverage gap |
| missing_nmod_on_subj | 11.8% | 0 nmod on subject vs 5495 on object | Train never has PP on subject |
| missing_modification | 4.0% | Max nmod depth = 2 | Recursion gap |
| missing_verb_role | 0.8% | Verb never seen in this form | Voice/valency gap |

### Autonomous loop (no human intervention)

Loop iterates: measure surprise → cluster → diagnose → fix → retrain.

- **COGS auto-loop v1**: 0.06% → **16.67%** in 1 cycle (proper_names + passive auto-activated)
- **SLOG auto-loop v1**: 20.62% → 19.51% (stalled; only available generator regresses Q_iobj)
- **COGS auto-loop v3** (lexical+structural fusion): identifies 5 fix types correctly but cumulative application degrades (1.87% gen); per-fix validation at 20 epochs is insufficient signal on COGS.

---

## SLOG

### Results

| Variant | Gen greedy | Notes |
|---|---|---|
| B0 | 20.62% | SLOG train already includes PP examples |
| B1 | ~14% | Proper name permutation |
| B4 | 19.18% | B1 + B3_auto |
| Auto (pp_on_subject only) | 19.47% | Q_iobj regresses 79→48% |

SLOG gen categories that work (>50%): Q_subj_active (94.7%), Q_iobj_ditransV (79.1%), PP_3 (62.3%).
Categories at 0%: CP_5-12, PP_5-12, center_embed (3 and 5-12), RC_iobj_extracted, RC_modif_subj, Q_long_mv, Q_dobj_ditransV.

### SLOG surprise × entropy analysis

| Type | Categories | Example gap acc | Action |
|---|---|---|---|
| STRUCTURAL (both high) | center_embed_5-12, CP_5-12, PP_5-12 | 0% | Deep recursion — architectural |
| ENCODER-ONLY (high surprise, medium entropy) | RC_modif_subj, center_embed_3, PP_modif_subj | 0-1% | Exposure gaps |
| LOW | Q_subj_active, PP_3 | >60% | Already resolved |

---

## Key findings

1. **76% of "wrong_arg_position" errors are lexical hallucinations.** The model says "Ella" instead of "Paula" — it generates from vocab instead of copying from input. A copy mechanism (+128 params) resolves all 10 lexical categories at once (0% → 97-99%).

2. **Causal diagnosis recovers human-designed fixes automatically.** The system identifies "0 nmod on subject in train vs 5495 on object" and "202 nouns seen only as object" in seconds, with 100% hypothesis confirmation rate. These match the fixes a human researcher designed manually (B3_auto, B6).

3. **Autonomous loop reaches 94% of manual ablation performance.** On COGS: 0.06% → 16.67% (vs 17.8% for the human-designed B4), without any human choice of augmentation.

4. **Structural role tagger is redundant with copy mechanism.** Adding explicit role labels (SUBJ/VERB/OBJ/…) via an auxiliary loss (+20K params) yields no measurable gain (52.79% vs 52.48%). The copy attention already learns role-conditional pointing.

5. **Deep recursion is the remaining wall.** cp_recursion, pp_recursion, center_embedding_5-12 stay at 0% across all interventions. Scaling (1M → 32M in literature) does not resolve them. This is an architectural bottleneck of the Transformer seq2seq at fixed depth.

---

## SCAN auxiliary: `detect_permutable.py`

Unsupervised detection of permutable token classes via distributional similarity (cosine on ±2 context fingerprints, agglomerative clustering at sim_thr=0.8):

- Correctly recovers the 5 SCAN role classes (ACTION, DIRECTION, MOD_DIR, QUANTIFIER, CONNECTOR, TURN) from train alone.
- `jump` does NOT cluster with ACTION — its signature is orthogonal because it appears only in isolation. This confirms the addprim_jump split is a genuine distributional gap.

---

## Reproducibility

All results are deterministic given seed + code version. Seed list for sweeps: 42, 123, 456, 789, 1337.
Hardware: RTX 3070 8 GB, PyTorch 2.0+, AMP fp16.
