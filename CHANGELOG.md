# RARE-JEPA Changelog

Chronological log of every modification, with rationale and result.

---

## v13 — Cleanup post-Tests v2 — 2026-04-29

### Composants archivés (branche `archive/jepa_a2_v1`)

État pré-nettoyage figé sur la branche `archive/jepa_a2_v1` (commit
`0dcd03e`). Sur main, les composants suivants ne sont plus utilisés par le
pipeline principal — leur code reste présent (non instancié) :

- **JEPA loss** (`--jepa`, `--jepa-lambda`, `--jepa-ema`) :
  - Verdict TEST 2 (ablation factorielle) : Δ macro intra-cat = +0.001 (COGS),
    -0.001 (SLOG) avec IC95 incluant 0 → **NO-GO causalement utile**.
  - Verdict TEST 3 (gradient conflict) : reporté (snapshot bug), réutiliser
    les checkpoints existants pour diagnostic post-hoc si besoin.
  - Code `JEPAPredictor`, `_jepa_loss`, `encode_ema`, `update_ema` reste dans
    `cogs_compositional.py` mais n'est plus activé dans les commandes du
    pipeline propre (`pipeline_clean/`).

- **Feature `nesting_depth`** dans le MetaEncoder :
  - Verdict TEST 2 : C2 (tokens+structure) ne bat pas C1 (tokens-only) sur
    macro intra-cat. Feature dégrade légèrement sur SLOG.
  - Le pipeline propre utilise désormais C5 (tokens + entropy + surprise).

- **A.2 auto-correction** (`meta_a2_*.py`, `run_a2_cycle1.sh`) :
  - Verdict TEST 1 : Δ_pria moyen = -19.85pp sur 3 seeds (catastrophic
    forgetting structurel confirmé). Le `peel_cp_prefix` casse
    `prim_to_inf_arg`.
  - Modules conservés mais non importés par le pipeline principal.

Ces composants peuvent être réactivés si le diagnostic de fuite (à venir)
invalide les conclusions actuelles.

### Bug fixé

- `_grad_snapshot` (TEST 3) : ordre du tuple `collate()` incorrect
  → `batch[3]` n'est PAS `src_mask` mais `in_lens` (1-D). Fixé : utiliser
  `batch[1]` pour src_mask et dériver `tgt_in/tgt_out` de `batch[2]`.

### Anomalie résolue

- **risk@cov=0.80 SLOG** : 0.103 (A.1) vs 0.4150 (TEST 2) — **pas un bug**
  d'implémentation. Sur SLOG (err_rate=0.532), la borne théorique à
  coverage=0.80 est `(0.532-0.20)/0.80 = 0.4150`. Le 0.103 d'A.1 vient d'un
  bug `np.interp` (extrapolation hors plage) car les scores SLOG sont
  bimodaux (coverage observée ∈ [0.43, 0.50]).
- À cov=0.50 sur SLOG : risk = 0.072 (utilisable). À cov=0.30 : 0.001.
  Le coverage 80% est inadapté à un benchmark à 53% d'erreurs.

---

## v12 — JEPA + meta-modèle (Phase A/B/C) — 2026-04-26

### Phase A : intégration JEPA dans `cogs_compositional.py`
- Nouvelle classe `JEPAPredictor` (MLP d→d→d avec stop-gradient sur la cible)
- Helper `_jepa_loss(pred, target, src_mask)` — MSE masqué sur encoder shifted-by-1
- Argument `use_jepa=False` propagé dans les 3 classes (`TransformerSeq2Seq`, `…WithCopy`, `…CopyTags`)
- CLI : `--jepa`, `--jepa-lambda` (défaut 0.1), `--filter-unaccusative-from-pool`
- Boucle d'entraînement : `total_loss = task + λ × jepa`, log per-epoch `jepa=X.XXXX`
- Trajectoire échantillonnée tous les 5 epochs sur 200 train + 200 gen stratifié → `jepa_trajectory.json`
- `jepa_curves.png` à la fin du run
- Checkpoint étendu : `use_jepa`, `jepa_state_dict`, `jepa_lambda` (backward-compat)

**Run B4+JEPA seed 42** : gen_greedy 81.17% (vs 81.56% sans JEPA), jepa_loss 2.03→0.07, sanity checks OK.

### Phase B : `meta_dataset_builder.py` (nouveau)
- Charge un checkpoint JEPA (`use_jepa=True` requis)
- Pour chaque exemple dev+gen : surprise (depuis JEPA), greedy decode + entropy (sans gold), structural features
- Schema strict inférence-time-valid (correctif review hostile #1) : surprise_mean/max, entropy_mean/max_greedy, input_length, rare_token_count, nesting_depth
- Sortie : JSONL + splits stratifiés 70/15/15 + stats.md (sanity checks) + feature_distributions.png + token_freq.json
- 24 000 records produits sur le run B4 ; ratio surprise(error)/surprise(exact) = 3.8×, entropy = 13.7×

### Phase C : `meta_train_etape1.py` (nouveau)
- MetaMLP 3-layer (hidden=128, dropout 0.1)
- Features par défaut : surprise_mean + input_length (≥2 imposé par correctif #4)
- Flag `--features` ajouté pour configurer la liste
- 3 baselines : raw_surprise (correctif #3), Logreg(1D), GBDT(1D, n_estimators=100, depth=3)
- Bootstrap CI (1000 resamples) sur deltas
- Test de décorrélation : AUC du MLP sur les exemples mal classés par GBDT_1D
- Verdict auto per spec C.8 : GO/MARGINAL/NO-GO/BUG

**Résultats étape 1 (3 seeds)** : AUC MLP = 0.940 (±0.001), Δ vs raw = +0.039 IC>0, **decorrelation = 0.255 < 0.55 → verdict MARGINAL**. Cohérent avec corr(surprise, input_length) = 0.92.

### Phase C étape 2 : `meta_train_etape2.py` (nouveau)
- Features hardcodées : surprise_mean + entropy_mean_greedy
- Ajoute 2 baselines 1D pour l'entropie (raw_entropy + GBDT_1D-entropy)
- Corrélation feature loggée (`feature_correlation.txt`) avec alerte si |corr| > 0.85
- Double test de décorrélation : MLP sur erreurs GBDT-surprise ET sur erreurs GBDT-entropy
- 6 modèles sur ROC superposée
- Verdict auto §5 : GO étape 3 / PARTIEL / NO-GO / BUG

### Phase C analyse par catégorie : `meta_analyse_par_categorie.py` (nouveau)
- Re-entraîne MetaMLP sur 3 seeds (déterministe) avec --features configurable
- Per catégorie : AUC (si ≥20 ex, 2 classes), TP/FN/TN/FP au seuil val-optimal
- Tagging spécifique : cp/pp_recursion (impossibles, doivent être flag>70%) ; a2p/p2a/unacc_to_trans (faciles, FP doivent rester ≤5%)
- Agrégation 3 seeds : mean ± std AUC, somme TP/FN/TN/FP
- Sortie : single MD avec synthèse 5-10 lignes

**Résultat 4 features** (surprise+entropy+rare+nesting) **3 seeds** :
- AUC global = 0.9714 ± 0.0004
- cp_recursion AUC 0.994, flag 93.3% (cible 92.7%) — détection parfaite
- pp_recursion AUC 0.993, flag 81.3% (cible 80%) — idem
- 0 FP sur 1350 catégories quasi-parfaites
- 4 catégories AUC < 0.5 : bruit statistique (1-3 vraies erreurs, FP=0)
- only_seen_as_unacc_subj_* : AUC 0.74-0.77, TP=0 (erreurs faites avec confiance) → cible étape 3

---

## v11 — COGS compositional benchmark — 2026-04-17

### New file `cogs_compositional.py` (standalone)
Nouveau benchmark de parsing sémantique : COGS (Kim & Linzen, 2020). Input = phrase anglaise, output = forme logique. Split gen = combinaisons inédites.

### Variantes
- **B0** : Transformer seq2seq baseline (même archi que A0 : 2 layers, d=128, 4 heads). Résultat : dev_ex ~95%, gen_ex **0%**. Le modèle apprend bien l'in-distribution mais ne généralise pas.
- **B1** : B0 + permutation aléatoire des noms propres (input + output) au training. 103 noms propres détectés automatiquement (heuristique : tokens capitalisés alphabétiques, len>1, pas des détermineurs, présents dans les deux vocabs). Résultat préliminaire : gen_ex **9.16%** à E005 (en cours).

### Détection automatique des noms propres
Heuristique : `t.isalpha() and t[0].isupper() and len(t) > 1 and t not in COGS_DETERMINERS and t in out_w2i`. Bug fix : la première version cherchait `t.lower()` dans out_w2i, mais COGS garde la casse (Emma→Emma, pas emma). Résultat : 2 faux positifs (A, TV) → 103 vrais noms propres après fix.

### Problème de performance résolu : eval greedy decode quadratique
Le greedy decode autorégressif (sans KV cache) est O(T²) par séquence. COGS a des outputs jusqu'à 484 tokens. Aux premières epochs le modèle ne produit pas EOS → chaque séquence décode les T tokens max. Avec 24000 exemples, chaque eval prenait >30 minutes.

**Itérations du fix** :
1. Réduction eval batch (64→16→8) : insuffisant, le quadratique domine
2. Cap max_decode (484→200→120) : amélioration ~3x mais toujours lent
3. Skip eval aux premières epochs (ep<3, puis ep<8) : accélère le départ mais ne résout pas le fond
4. **Solution finale : teacher-forced eval** (une seule passe forward parallèle par batch, ~5s pour tout le dataset). Les métriques teacher-forced sont un upper-bound sur greedy (chaque position voit les vrais tokens précédents, pas ses propres prédictions), mais fidèles en tendance. Le greedy decode est réservé au summary final.

### Fichier `detect_permutable.py` (squelette)
Détection automatique de classes de tokens permutables par similarité distributionnelle :
1. Pour chaque token, collecter les contextes (fenêtre ±2)
2. Cosine similarity entre fingerprints contextuels
3. Union-find sur les paires sim > seuil (0.8)
4. Retourner les clusters ≥2 membres

Testé sur SCAN train : retrouve exactement les 5 classes de rôle de la grammaire (ACTION/CONNECTOR/MOD_DIR/DIRECTION/QUANTIFIER) sans supervision. Note : `jump` ne tombe PAS dans la classe ACTION car il n'apparaît qu'en isolation dans le train addprim_jump → signature distributionnelle différente.

### Fichier `results_summary.py`
Helper pour agréger les runs d'une variante sur plusieurs seeds → `results_summary.json` avec mean/std.

---

## v10 — SCAN A4_permute (data augmentation breakthrough) — 2026-04-16

### New variant A4_permute in `scan_compositional.py`
Architecture **identique** à A0 (zero changement). Seule différence : data augmentation au training.

À chaque `__getitem__`, les 4 primitives d'action {jump, walk, run, look} sont permutées aléatoirement de façon cohérente input↔output (ex: `walk around right twice → I_TURN_RIGHT I_WALK ...` peut devenir `jump around right twice → I_TURN_RIGHT I_JUMP ...`). La permutation est tirée par exemple, pas par batch.

### Résultat (3 seeds)
| Seed | Final test | Best test |
|---|---|---|
| 42  | 99.57% | 99.57% |
| 123 | 99.96% | 100.0% |
| 456 | 100.0% | 100.0% |
| **Mean** | **99.84% ± 0.19%** | **99.84% ± 0.19%** |

Le modèle converge à ~E009 (9 epochs). Saturation rapide et stable.

### Interprétation
Le problème de SCAN addprim_jump n'était jamais architectural. Le Transformer A0 sait déjà composer — il n'avait juste jamais vu `jump` dans un contexte compositionnel pendant le training. En permutant les 4 actions, on montre au modèle que jump est interchangeable avec walk/run/look → généralisation immédiate.

Toutes les variantes A1-A3 (supervision structurelle, VQ, dual encoder, consistency loss) étaient de l'over-engineering pour un problème de **couverture distributionnelle**.

---

## v9 — SCAN compositional : A3_dual (dual encoder + consistency loss) — 2026-04-16

### New variant A3_dual in `scan_compositional.py`
Architecture à deux encodeurs, même Transformer de base que A0 :
- **Structure encoder** (self.encoder, 2 layers d=128, hérité de BaseSeq2Seq) : entraîné avec task loss + L_consist
- **Content encoder** (self.enc_content, 2 layers d=128) : entraîné avec task loss uniquement
- Poids **non partagés** entre les deux encodeurs

### L_consist
Pour chaque batch, on crée une version augmentée en remplaçant chaque primitive d'action (jump/walk/run/look) par une autre action tirée uniformément parmi les 4. Puis :
```
L_consist = MSE(struct_out_orig, struct_out_aug) sur positions (non-action ∧ non-padding)
```
Le gradient de L_consist atteint uniquement la Structure encoder (pas la Content encoder, pas les embeddings partagés via detach implicite). Hypothèse : si la Structure doit représenter le contexte indépendamment de l'action, elle apprend la factorisation rôle/identité.

### Decoder cross-attention
`Q` = decoder state, `K` = struct_out, `V` = content_out. Le décodeur regarde OÙ via la structure, lit QUOI via le contenu.

### Loss totale
```
L = L_action + λ_consist * L_consist     (λ default = 1.0)
```

### CLI
- `--variant A3_dual` (nouveau choix)
- `--lambda-consist FLOAT` (défaut 1.0)
- Log shell : `L_act=… L_con=…` affichés chaque epoch
- `L_consist` logé dans `metrics.json` sous `train_loss_parts`

### Run
```
python3 scan_compositional.py --variant A3_dual --seed 42 --epochs 100 --lambda-consist 1.0
```

### Premier lancement (partiel, 8 epochs)
val_best 95.6% / test_best 0.04% — en cours, trop tôt pour conclure.

---

## v8 — SCAN compositional benchmark (new axis) — 2026-04-16

### Standalone file `scan_compositional.py`
Indépendant de rare_jepa.py. Objectif : tester si un bottleneck de rôle structurel résout addprim_jump (compositional generalization).

### Variantes implémentées (avant A3_dual)
- **A0** : Transformer seq2seq baseline
- **A1** : A0 + tête auxiliaire de rôle sur encoder (CE contre labels dérivés de la grammaire : ACTION, DIRECTION, MOD_DIR, QUANTIFIER, CONNECTOR, TURN)
- **A2** : A0 + VQ bottleneck (6 prototypes, Gumbel-ST, τ 2.0→0.5) sur encoder_out. K=quantifié, V=original.
- **A2_nosup** : A2 sans L_commitment (ablation)
- **A2_warm** : A2 avec codebook warm-started depuis les centroids par rôle d'un checkpoint A1
- **A2_pre** : VQ AVANT l'encoder (sur embeddings d'entrée), V=raw_emb bypass

### Commun à toutes
- 3e-4 lr, Adam, warmup 500 steps, cosine decay
- AMP fp16, batch 64, seed 42
- Label smoothing 0.1, gradient clip 1.0
- Checkpoint sauvé à la fin (pour A2_warm init)
- `probe_a1_swap.py` : probe d'interprétabilité qui remplace enc(jump) par enc(walk) aux positions de `jump` puis décode

### Résultats (seed 42)
| Variante | Val (best) | Test (best) | Test (final) | Epochs |
|---|---|---|---|---|
| A0 | 97.5% | 0.83% | 0.83% | 11 |
| A1 | 100.0% | 0.23% | 0.09% | 50 |
| A2 | 93.4% | 1.05% | 0.08% | 15 |
| A2_warm | 99.8% | 1.12% | 0.51% | 41 |
| A2_pre | 96.9% | 0.00% | 0.00% | 34 |

Toutes les variantes saturent sur val (93-100%) mais échouent (<2%) sur test addprim_jump. La supervision de rôle, le VQ, et le warm-start ne suffisent pas. → motivation pour A3_dual.

---

## v7 — JEPA Co-Learning (optional mode) — 2026-04-15

### JEPA Co-Learning flag
- New model param `jepa_colearn: bool = False` (default off = backward compat)
- When True: target_all = live h_real (gradient flows into experts)
- When False: target_all = EMA h_ema.detach() (current behavior)
- `JEPA_COEFF_COLEARN = 0.05` config (reduced from 0.1 to avoid expert collapse)
- Propagated via `info["_jepa_coeff"]` to compute_loss
- CLI: `test_unified_simple.py --jepa-colearn`

Purpose: test if experts learning to be "predictable by JEPA" helps routing.
Saved in runs with `"jepa_colearn": true/false` for reproducibility.

---

## v6 — Unified routing family — 2026-04-14 evening

### RARE-UNIFIED (Run #5, 8 actions)
- Router produces 8 logits: advance, revise, jump×6
- JEPA + ValueHead + Bary all available as submodes via pilot
- Phase 1 forced sequential (shuffled), Phase 2 pilot, Phase 3 winner
- **Result**: Task 2 explodes (99.7%) via "advance" shortcut; Task 3 regresses (75%) from redundancy.

### RARE-UNIFIED dedup (Run #6, 6 actions)
- Router outputs 6 logits = one per expert
- advance/revise/jump classified a posteriori (chosen == current+1 / current / else)
- **Result**: seed-sensitive inversion — T1 97.8% but T2 drops to 70.7%.

### Bug fix: Phase 1 trains JEPA for unified/bary
- Added `"unified"` to Phase 1 JEPA observation branch (was missing)
- Previously: JEPA predictor init was random entering Phase 2 → routing based on noise

### Unified Simple (Run #7, THE winner)
- No pilot, no submode alternation. Just `unified_submode = "jepa"`, 50 epochs total.
- Task 1 → 100% @ E36
- Task 3 → 100% @ E40  
- Task 2 → 82.8% (still rising)

### Unified Simple 70ep (Run #8)
- Same as Run #7, but 70 epochs total for Task 2
- Task 2 → 100% @ E66

---

## v5 — Meta Pilot + Barycentric — 2026-04-14 afternoon

### Meta Pilot (Run #4)
- Single model with all routing components (jepa + gumbel_router + bary + value_head)
- Phase 1: forced sequential (trains JEPA)
- Phase 1.5 (ep 20-28): pilot each of gumbel/jepa/bary for 3 epochs
- Phase 2 (ep 29-34): winner continues with ε decay
- **Result**: T1 jepa 63.8%, T2 gumbel 93.7%, T3 bary 94.2%. Different winners per task.

### Barycentric Routing (Run #3)
- New `routing="bary"` mode: geometric routing via JEPA prediction barycenter
- CCV (Cluster Convergence Value) = mean pairwise cosine distance between predictions
- Stability mode (CCV < threshold): pick most central expert
- Exploration mode (CCV > threshold): pick most divergent
- Halt by classifier confidence on barycenter > 0.95
- `L_ccv = 0.01 × CCV` loss to encourage convergence
- Learnable threshold (sigmoid-wrapped scalar)
- **Solo result** (Run #3): T1 63.5%, T2 82.5%, T3 72.2% — mitigé.
- **Under meta pilot** (Run #4): T3 = 94.2% 🔥 — warm-up JEPA transfers.

### JEPA Hybrid (Run #2, FAILED)
- Gumbel-Softmax on JEPA value_scores with step-dependent tau (2.0 @t=0, 0.5 @t≥2)
- Rationale: H1 diagnostic showed JEPA error high at t=0
- **Result**: no improvement on Task 3 (73.3% vs jepa solo 73.8%).

### H1 Diagnostic (Run #1)
- Track per-step JEPA prediction error vs step t
- **Finding**: U-shape (not linear growth). High @t=0 (0.87), drops to 0.28 @t=2, rises to 0.47 @t=6.
- H1 (linear accumulation) REJECTED.

### Infrastructure
- `save_run()` with subfolder support + JSON serialization
- Never overwrites runs (timestamp prefix)
- Per-step JEPA error tracking in info dict
- All scripts support `--tag`, `--tasks` CLI flags

---

## v4 — Anti-Shortcut Data + Attention Pooling — earlier run

### Anti-Shortcut Data Generation
- **Problem**: with 4 people × 6 locations, target appears in last sentence ~25% of the time. Model learns shortcut "predict last sentence's location" → 37.5% accuracy (matches observed).
- **Fix**: all 3 task generators now ALWAYS add 1-3 distractor sentences about OTHER people AFTER the key fact. Last sentence is never about the target.

### Attention Pooling
- Mean pooling replaced by attention pooling (last token "?" as query)
- fp16 safety: `-6e4` instead of `-1e9` in masked_fill

### Result
| | T1 | T2 | T3 | Avg |
|---|---|---|---|---|
| baseline | 88.8% | 100% | 100% | 96.3% |
| gumbel | 66.8% | 71.5% | 89.3% | 75.9% |
| jepa | 70.8% | 99.8% | 69.0% | 79.9% |
| jepa_nogru | 73.0% | 100% | 72.8% | 81.9% |

---

## v3 — Barycentric v1 + JEPA_CCV — earlier (deprecated, merged into v5)

Initial bary implementation. Superseded by v5.

---

## v2 — No-Revise + Shuffled Phase 1 + Anti-Collapse — earlier

### JEPA No-Revise
- New `allow_revise=False` mode: each expert used AT MOST ONCE per sample
- Max steps = K=6 (same budget as baseline)

### Shuffled Phase 1
- Random permutation per batch in Phase 1 for non-baseline models
- Breaks co-adaptation: experts learn to be order-independent

### Gumbel hard routing
- Removed 6x-compute weighted sum
- `F.gumbel_softmax(hard=True)` with proper ST gradients

### Anti-collapse stronger
- COVERAGE_THR 5% → 10%
- COVERAGE_PCT 10% → 20%

### Result
| | T1 | T2 | T3 | Avg |
|---|---|---|---|---|
| baseline | 83.7% | 100% | 100% | 94.6% |
| gumbel | 62.2% | 72.8% | 88.2% | 74.4% |
| jepa | 67.3% | 77.0% | 72.0% | 72.1% |
| jepa_norevise | 64.2% | 84.7% | 73.2% | 74.0% |
| jepa_nogru | 64.5% | 73.5% | 74.8% | 70.9% |

---

## v1 — Initial attention pool + EMA + Gumbel coverage — earlier

### EMA target experts
- EMA shadow of experts (decay=0.99) used as stable JEPA target
- Prevents L_jepa stagnation from moving-target problem

### Attention pooling
- Last token attends to sequence for classification

### Gumbel coverage forcing
- Underused experts (<5% usage) forced on 10% of samples

### ENT_COEFF 0.01 → 0.05
- Stronger entropy pressure on expert distribution

### Batched JEPA simulation
- 6 MLP forwards batched into one `(B*6, dim)` forward
- 4x speedup on JEPA routing

---

## v0 — Baseline implementation

### Architecture
- 6 TransformerExpert blocks (Pre-LN, 256-dim, 4 heads, FFN 512, dropout 0.2)
- Embedding: token + position, 256-dim, max_len 512
- Mean pooling for classification
- GRU memory (256-dim)
- Max steps T=8, min steps=2
- Phase curriculum: Phase 1 forced, Phase 2 ε-greedy 0.3, Phase 3 ε decay

### Initial issues found
- Severe overfitting (train 98% / test 37% on T1)
- Gumbel halt collapse to step 0
- L_jepa stuck at ~0.65 (moving target)
- Mean pooling bottleneck at ~40% on T1
- All resolved by v4+.

---

## Reproducibility

- SEED=42 throughout
- Data: 3000 train / 600 test per task
- Dropout 0.2, weight_decay 1e-4
- Each run is deterministic given same code version
- Different versions diverge RNG state (different model/arch → different init sequences)
- All runs saved as `runs/[subfolder/]YYYYMMDD_HHMMSS_<tag>.json`, never overwritten

## Files index

- `rare_jepa.py` — full implementation (~1500 lines)
- `test_*.py` — each bAbI experiment runner
- `diagnostic_h1.py` — per-step JEPA diagnostic
- `scan_compositional.py` — SCAN addprim_jump ablation study (A0/A1/A2/A2_nosup/A2_warm/A2_pre/A3_dual)
- `probe_a1_swap.py` — swap-interpretation probe for A1
- `RESULTS.md` — chronological results
- `CHANGELOG.md` — this file
- (internal dev notes kept local)
- `plots/*.png` — visualizations
- `runs/**/*.json` — all saved runs
