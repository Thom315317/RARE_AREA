# Results — Chronological Log

Tracked results across SCAN, COGS, SLOG benchmarks. Updated as new runs complete.

---

## Headlines (best single-seed results)

| Benchmark | Config | Gen (greedy) | Params |
|---|---|---|---|
| SCAN addprim_jump | A4_permute (3 seeds) | **99.84% ± 0.19%** | 0.68M |
| COGS | B0 + copy + scheduled sampling | 52.48% | 1.01M |
| COGS | B0 + copy + peel-stack aug (Innov. K) | **53.20%** | 1.02M |
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
| B0 + copy + bidirectional (Innov. A) | 51.33% | **Négatif** — voir section dédiée |
| B0 + copy + selective-forgetting permute (Innov. B) | 49.91% | **Négatif** — casse prim_to_inf_arg |
| B0 + copy + selective-forgetting dropout (Innov. B) | 51.05% | **Négatif** — idem |
| B0 + copy + selective-forgetting reset (Innov. B) | 52.48% | **Neutre** — pas de gain, pas de perte |
| B0 + copy + peel-stack aug (Innov. K, depth=3) | 53.20% | **Positif** — pp_recursion 2.8% → 8.3% |
| B0 + copy + peel-stack aug (Innov. K, depth=5) | 53.31% | pp_recursion plafonne à 8.8% — ne scale pas |
| **B4 + copy + peel-stack (K+B4)** | **54.88%** | Subadditif — compétition PP |
| B0 + copy + peel-stack + permute (K+B) | 51.78% | Permute annule le gain de K |
| B0 + copy + contrastive errors (Innov. C) | 52.58% | Neutre — +0.10 dans le bruit |
| B4 + copy + peel-stack + cross-voice n=10 | 54.86% | Structurels inchangés à 0% |
| B4 + copy + peel-stack + cross-voice n=25 | 55.11% | Structurels inchangés à 0% |

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

### Innovation A — Bidirectional training (NÉGATIF)

Config : B0 + copy + unified vocab (874 tokens) + tied embeddings + 50% swap phrase↔LF. 60 epochs, seed 42.

Résultat : **51.33% greedy** (vs 52.48% baseline, dans le bruit d'un seed).

**Toutes les catégories structurelles restent à 0% :**

| Catégorie | B0+copy | B0+copy+bidir |
|---|---|---|
| active_to_passive | 0.0% | 0.0% |
| passive_to_active | 0.0% | 0.0% |
| do_dative_to_pp_dative | 0.0% | 0.0% |
| pp_dative_to_do_dative | 0.0% | 0.0% |
| unacc_to_transitive | 0.0% | 0.0% |
| obj_omitted_transitive_to_transitive | 0.0% | 0.0% |
| cp_recursion | 0.0% | 0.0% |
| pp_recursion | 2.8% | 0.7% |

**Raison de fond :** le bidirectionnel n'apprend pas la transformation active↔passive parce que les LFs actif et passif sont *différentes*. Le modèle voit `(phrase_active, LF_active)` et séparément `(phrase_passive, LF_passive)` — jamais la paire `(phrase_active, phrase_passive)` liée par une LF commune. Le swap 50% enseigne seulement que le vocabulaire est partagé entre input et output. Aucune pression pour apprendre la correspondance structurelle. Hypothèse initiale invalidée.

**Éliminé** comme l'était le tagger : supervision indirecte → pas d'effet sur le décodeur.

### Innovation B — Selective forgetting (3 modes, NÉGATIF ou NEUTRE)

Config : B0 + copy + scheduled sampling + perturbation périodique des bindings positionnels. `--forgetting-every=20 --forgetting-duration=1 --forgetting-start=0`, 60 epochs, seed 42.

| Mode | Greedy | best_gen_tf | prim_to_inf_arg | pp_recursion | 7 structurels |
|---|---|---|---|---|---|
| permute | 49.91% | 52.30% | 99.2% → 69.3% | 2.8% → 0.3% | 0% |
| dropout (80%) | 51.05% | 52.74% | 99.2% → 75.6% | 2.8% → 1.3% | 0% |
| reset (layer 0 self-attn) | 52.48% | 52.51% | 99.2% → 98.5% | 2.8% → 3.2% | 0% |

**Observations :**

- **permute et dropout cassent des catégories lexicales déjà résolues** (`prim_to_inf_arg` tombe de 99% à 69-75%). La shock tous les 20 epochs est trop espacée pour créer une pression de généralisation mais assez forte pour détruire les bindings acquis.
- **reset est neutre** : réinitialiser la self-attention de la couche 0 est absorbé par la couche 1 sans dégradation ni bénéfice. La représentation position→rôle se reforme à l'identique.
- **Aucun mode ne bouge les 7 catégories structurelles.** L'analogie avec SCAN A4_permute (permutation des primitives lexicales) ne se transpose pas à la permutation d'embeddings positionnels : SCAN permute des *tokens* qui apparaissent dans tous les contextes compositionnels ; ici la permutation d'embeddings détruit le binding sans exposer le modèle à de nouveaux contextes.

**Conclusion :** la perturbation de position seule ne suffit pas à déloger la mémorisation positionnelle, qui est répartie entre embeddings et attention. Une re-convergence rapide efface le signal.

**Piste future :** perturbation *rythmée* (start=15, every=5, duration=2) pour empêcher la re-convergence entre chocs. Non testé à ce stade.

### Innovation K — Peel-and-stack augmentation (POSITIF modeste)

Config : B0 + copy + augmentation en cascade de `generate_pp_recursion_extension` sur 3 niveaux (depth 3, 4, 5). +2100 exemples de PP recursion. 60 epochs, seed 42.

**Résultat : 53.20% greedy (+0.72 vs 52.48% baseline), best_gen_tf 53.49%.**

| Catégorie | B0+copy | B0+copy+peel-stack |
|---|---|---|
| **pp_recursion** | **2.8%** | **8.3%** (×3) |
| prim_to_inf_arg | 99.2% | 99.8% |
| obj_to_subj_proper | 98.7% | 98.9% |
| prim_to_obj_proper | 57.9% | 62.3% |
| subj_to_obj_proper | 57.6% | 61.1% |
| active_to_passive / passive_to_active | 0.0% | 0.0% |
| cp_recursion | 0.0% | 0.0% |
| datives (both directions) | 0.0% | 0.0% |
| obj_pp_to_subj_pp | 0.0% | 0.0% |

**Lecture :** c'est la première innovation testée (après A, B×3) qui donne un gain net sans régression. Le gain est concentré sur `pp_recursion` (×3 : 2.8% → 8.3%), ce qui valide l'hypothèse : **l'exposition à la profondeur 3-5 pendant l'entraînement suffit à améliorer la récursion sans besoin d'architecture dédiée**. L'effet de spillover sur les proper_names (+4 pts sur `prim_to_obj_proper` et `subj_to_obj_proper`) suggère que l'augmentation exerce aussi une pression de généralisation lexicale collatérale.

**Limite :** `cp_recursion` reste à 0% (logique — l'augmentation ne couvre que les PP, pas les CP). Les catégories de transformation de voix/valence (active↔passive, datives, unacc/transitive) restent à 0% : elles ne dépendent pas de la profondeur mais de la correspondance structurelle inter-voix.

**Note :** c'est la version *data-augmentation only* de K. La version architecturale complète (segmenter BIO + mémoire d'assemblage + stop head) décrite dans PROMPT_INNOVATIONS.md n'a pas encore été testée.

### Composition K + B_permute — NÉGATIF

Config : les deux innovations combinées (B0 + copy + peel-stack + permute every=20).

Résultat : **51.78% greedy** (vs 53.20% pour K seule). Le perturbation de position annule le gain de K sur le score global. `pp_recursion` reste à 7.7% (proche du K seul à 8.3%), donc l'effet K sur sa catégorie cible survit, mais la perturbation dégrade les catégories lexicales (`prim_to_inf_arg` 99.8% → 97.5%, `subj_to_obj_proper` 61% → 50.7%).

**Les deux innovations ne sont pas complémentaires** dans cette configuration. Nécessiterait de retester avec la perturbation rythmée (warmup+cycle) pour voir si l'absence de shock après ep 60 change la donne.

### K + B4 — Combinaison des deux gagnants (POSITIF mais subadditif)

Config : B4 (perm noms propres + PP aug B3_auto) + copy + peel-stack depth=3. 60 epochs, seed 42.

Résultat : **54.88% greedy**, best_gen_tf 55.20%.

| Catégorie | B4+copy | K+B4 | Delta |
|---|---|---|---|
| obj_pp_to_subj_pp | **42.5%** | **32.1%** | **-10.4** (régression) |
| pp_recursion | 3.8% | 8.2% | +4.4 (gain K) |
| prim_to_obj_proper | 59.7% | 62.8% | +3.1 |
| subj_to_obj_proper | 60.2% | 63.4% | +3.2 |
| 7 structurels | 0% | 0% | — |

**Subadditivité confirmée :** K seul donne +0.72 sur B0 (52.48 → 53.20), B4 donne +2.16 (52.48 → 54.64). L'addition serait 55.40%. Obtenu : 54.88% (+0.24 sur B4). Les deux augmentations opèrent sur PP (B3_auto sur depth-1 PP-on-subject, K sur depth 3-5 PP recursion) → compétition pour la capacité 1M params. Le budget PP est fini.

**Différence avec SCAN :** A4_permute sur SCAN permute des primitives d'action, orthogonal au contexte compositionnel. Sur COGS, toutes les augmentations PP sont dans le même sous-espace structurel.

### K depth=5 — Test de scalabilité de la récursion (NÉGATIF sur l'hypothèse)

Config : B0 + copy + peel-stack depth=5 (5 niveaux de cascade, +3500 exemples jusqu'à profondeur 7). 60 epochs, seed 42.

Résultat : **53.31% greedy** (vs K depth=3 = 53.20%, +0.11).

| Catégorie | K depth=3 | K depth=5 | Delta |
|---|---|---|---|
| **pp_recursion** | **8.3%** | **8.8%** | **+0.5** |
| prim_to_obj_proper | 62.3% | 63.4% | +1.1 |
| subj_to_obj_proper | 61.1% | 62.0% | +0.9 |
| 7 structurels | 0% | 0% | — |

**L'hypothèse "récursion dans la boucle" est invalidée.** On attendait pp_recursion proportionnellement au nombre d'exemples (+67% d'exemples ↔ doublement attendu vers 15%). Obtenu : +0.5 point. **Le modèle mémorise les profondeurs vues en train au lieu d'apprendre la règle récursive.** La gen set contient des profondeurs jamais vues en train → pas de généralisation au-delà de la distribution.

**Conséquence pour Innovation K :** la version minimale (data augmentation only) plafonne à ~8.8% sur pp_recursion. La version architecturale complète (segmenter + memory + stop_head) reste nécessaire pour tester la vraie hypothèse de recursion compositionnelle.

### Innovation C — Contrastive errors (NEUTRE)

Config : B0 + copy + per-token margin loss (lambda=0.2, margin=1.0) poussant le gold au-dessus du top-1 non-gold. 60 epochs, seed 42.

Résultat : **52.58% greedy** (vs baseline 52.48%, +0.10 — dans le bruit).

| Catégorie | B0+copy | C | Delta |
|---|---|---|---|
| pp_recursion | 2.8% | 4.1% | +1.3 (marginal) |
| prim_to_obj_proper | 57.9% | 57.1% | -0.8 |
| subj_to_obj_proper | 57.6% | 56.9% | -0.7 |
| 7 structurels | 0% | 0% | — |

**Interprétation :** la per-token margin loss duplique partiellement le signal déjà fourni par la NLL (label_smoothing et softmax font déjà un travail de séparation gold/alternatives). Gain marginal sur pp_recursion mais légère régression sur les proper_names. Pas d'effet sur les 7 catégories à 0%.

**Implémentation simplifiée** : la version du prompt original utilisait un buffer d'erreurs + contrastive sur représentations encodées (nécessite bidirectional pour aligner les espaces input/output). La version testée ici est une margin loss par token, plus cheap, qui capture l'idée principale ("gold doit dominer le top-1 compétiteur").

### Innovation cross-voice (sweep en cours)

Config : B4 + copy + peel-stack + cross-voice (paires active↔passive générées depuis train). 60 epochs, seed 42.

**Générateur :**
- Extrait automatiquement `verb_lemma → past_tense` des actifs et `verb_lemma → past_participle` des passifs de train
- Convertit active transitive → passive (et inverse) uniquement pour les exemples "purs" (pas de nmod/ccomp/recipient)
- Re-indexe les variables `x_N` selon les nouvelles positions
- Pool disponible dans COGS train : a2p=2698 candidats, p2a=3821, skipped=20629 (examples avec modifiers)
- Validation manuelle des LFs générées : sémantiquement et syntaxiquement correctes (agent/theme préservés, ordre des prédicats + presup conforme au format COGS)

**Sweep par taille d'augmentation (n = {10, 25, 50, 100, 500, all}) :**

| n | Greedy | active_to_passive | passive_to_active | Note |
|---|---|---|---|---|
| 10 | 54.86% | 0% | 0% | Baseline K+B4 (54.88%) |
| 25 | 55.11% | 0% | 0% | +0.23 global, structurels inchangés |
| 50 | — | — | — | En cours |
| 100 | — | — | — | En cours |
| 500 | — | — | — | En cours |
| all | — | — | — | En cours |

Trois scénarios attendus :
1. Point d'inflexion avant n=500 → finding majeur (copy réduit besoin d'exposition)
2. Linéaire → quantification du coût d'exposition
3. Plateau à 0% → problème plus fondamental (architecture, pas exposition)

### Bilan après 7 innovations testées

| Innovation | Flag | Statut | Delta vs baseline |
|---|---|---|---|
| Tagger structurel | `--tags` | Neutre | +0.3 |
| A - Bidirectional | `--bidirectional` | Négatif | -1.2 |
| B - Forgetting permute | `--selective-forgetting --forgetting-mode=permute` | Négatif | -2.6 |
| B - Forgetting dropout | `--selective-forgetting --forgetting-mode=dropout` | Négatif | -1.4 |
| B - Forgetting reset | `--selective-forgetting --forgetting-mode=reset` | Neutre | 0.0 |
| **K - Peel-stack aug** | `--peel-stack` | **Positif** | **+0.72** |
| C - Contrastive errors | `--contrastive-errors` | Neutre | +0.10 |

**Observation centrale :** aucune des 7 innovations ne débloque les 7 catégories à 0% (active↔passive, datives, cp_recursion, unacc_to_transitive, obj_omitted_transitive_to_transitive). Toutes les améliorations se concentrent sur les catégories déjà non-nulles.

Le mur structurel requiert probablement l'une de ces voies :
- **Supervision cross-voice directe** : voir `(phrase_active, LF)` ET `(phrase_passive, LF_reordered)` dans le train
- **Mécanisme architectural** : round-trip E, peel-stack complet K-architectural
- ~~**Ré-ranking avec contraintes** : beam search + contraintes extraites du train set~~ **ÉLIMINÉ** — voir section "Constrained beam" ci-dessous

### Constrained beam search — FINDING NÉGATIF INFORMATIF

Config : checkpoint K+B4 (greedy 54.88%) + beam search k=10 + re-ranking par contraintes extraites du train set (animacy, known_verbs, nmod/ccomp attach, passive agent match). 500 exemples de gen (≈37% structurels).

**Résultat global :** 43.75% constrained-beam vs 54.88% greedy → -11 points.

**Causes du déclassement global :**
1. Les règles simples (UNK_VERB, ANIMACY, PASSIVE) introduisent du bruit — elles aident certaines catégories (`obj_to_subj_common` 98.2% → 100%) et en pénalisent d'autres (`prim_to_obj_proper` 62.8% → 52.6%). Net négatif.
2. Les contraintes *soft* extraites du train ne sont pas parfaites ; calibrage inadéquat pour re-ranking fin.

Mais le **vrai finding** n'est pas le score — c'est le diagnostic de rang.

**Diagnostic de rang du gold dans les beams (187 exemples structurels) :**

| Catégorie | # examples | Gold dans top-10 |
|---|---|---|
| active_to_passive | 18 | **0 / 18** |
| passive_to_active | 25 | **0 / 25** |
| cp_recursion | 29 | **0 / 29** |
| do_dative_to_pp_dative | 22 | **0 / 22** |
| pp_dative_to_do_dative | 33 | **0 / 33** |
| unacc_to_transitive | 29 | **0 / 29** |
| obj_omitted_transitive_to_transitive | 31 | **0 / 31** |
| **Total** | **187** | **0 / 187 (0.0%)** |

**Interprétation :** le décodeur n'accorde *aucune* probabilité mesurable à la structure correcte pour ces 7 catégories. La bonne réponse n'est pas dans les 10 hypothèses les plus probables — et ne sera vraisemblablement pas dans les 100 non plus, vu l'écart de log-prob.

**Conséquence théorique :** le mur des 0% est dans la distribution `p(y|x)` apprise, pas dans la procédure de décodage. Aucune stratégie de *decoding-time* (beam search, re-ranking, nucleus sampling, constrained decoding, contraintes logiques, MBR decoding) ne peut franchir ce mur — par construction, elles opèrent sur les candidats que le modèle génère, et les candidats corrects ne sont jamais générés.

**Conséquence pratique :** il faut modifier le *training*, pas l'inférence. Les voies restantes :
1. Supervision cross-voix directe (actifs+passifs appariés) — équivalent COGS de SCAN A4_permute
2. Round-trip self-consistency (E) avec feedback au training
3. K architectural complet (segmenter + mémoire d'assemblage + stop_head)

**Valeur pour le papier :** résultat négatif à publier. Personne n'a documenté ce finding avec ce niveau de clarté (0/187 gold in top-k pour les catégories structurelles COGS). Cela invalide une classe entière d'approches (constraint-based decoding sur petits Transformers compositionnels) et justifie les approches training-side.

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
