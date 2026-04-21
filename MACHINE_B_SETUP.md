# RARE-AREA — Setup Machine B (Scaling + Sweep)

## Rôle de cette machine

Machine B fait tourner le compute lourd pendant que Machine A (RTX 3070) fait le dev/innovations.

**Machine A** = développement, innovations 1-9, itérations rapides
**Machine B** = sweep 5 seeds, tests de scalabilité, runs longs

---

## Setup

### 1. Cloner le repo

```bash
git clone https://github.com/[username]/rare-area.git
cd rare-area
```

### 2. Installer les dépendances

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy scikit-learn
```

### 3. Télécharger les données

```bash
# COGS
git clone https://github.com/najoungkim/COGS.git data/cogs

# SCAN
git clone https://github.com/brendenlake/SCAN.git data/scan

# SLOG — vérifier le repo exact avec Julien
```

### 4. Vérifier que ça tourne

```bash
# Test rapide : B0 COGS 5 epochs
python models/cogs_compositional.py \
    --dataset cogs \
    --epochs 5 \
    --seed 42

# Doit afficher : loss qui baisse, dev qui monte
```

---

## Tâche 1 — Sweep 5 seeds (priorité haute)

Le sweep produit les résultats publiables avec statistiques.

### SCAN (4 variantes × 3 splits × 5 seeds = 60 runs)

```bash
python experiments/sweep_final.py --benchmark scan
```

| Variante | Description |
|---|---|
| A0 | Baseline |
| A1 | Role supervision (lambda=0.3) |
| A4-seen | Permutation primitives vues seulement |
| A4-all | Permutation toutes primitives |

Splits : addprim_jump, addprim_turn_left, length
Seeds : 42, 123, 456, 789, 1337
Temps estimé : ~15-20h

### COGS (5 variantes × 5 seeds = 25 runs)

```bash
python experiments/sweep_final.py --benchmark cogs
```

| Variante | Description |
|---|---|
| B0 | Baseline |
| B1 | Permutation 103 noms propres |
| B3-auto | 893 exemples PP obj→subj |
| B4 | B1 + B3-auto |
| B0+copy | Copy mechanism + scheduled sampling |
| B4+copy | Copy + B4 augmentation |

Seeds : 42, 123, 456, 789, 1337
Temps estimé : ~20h (greedy eval lent mais nécessaire)

### SLOG (3 variantes × 5 seeds = 15 runs)

```bash
python experiments/sweep_final.py --benchmark slog
```

| Variante | Description |
|---|---|
| B0 | Baseline |
| B4 | Permutation + PP |
| Auto | Boucle autonome |

Seeds : 42, 123, 456, 789, 1337
Temps estimé : ~10h

### Total sweep : ~45-50h

Peut tourner en continu. Le script reprend automatiquement après un crash (skip les runs déjà terminés).

### Output

Le sweep produit :
```
sweep/
  sweep_results_scan.json    # Métriques brutes
  sweep_results_cogs.json
  sweep_results_slog.json
  sweep_tables.md            # Tableaux avec IC 95%
  cogs_category_sizes.json   # Tailles des catégories
  qualitative_analysis.json  # 5 exemples corrects/incorrects par catégorie
```

Pousser les résultats sur GitHub quand c'est fini :
```bash
git add sweep/
git commit -m "Sweep results: SCAN/COGS/SLOG, 5 seeds, IC 95%"
git push
```

---

## Tâche 2 — Test de scalabilité (priorité moyenne)

Après le sweep, ou en parallèle si la GPU a de la marge.

### Configs à tester

| Config | d_model | n_layers | n_heads | Params | VRAM | Temps |
|---|---|---|---|---|---|---|
| S (actuel) | 128 | 2+2 | 4 | 1M | ~1.5 Go | 40 min |
| M | 256 | 2+2 | 8 | 4M | ~3 Go | 80 min |
| L | 256 | 4+4 | 8 | 8M | ~4.5 Go | 2h |
| XL | 384 | 4+4 | 8 | 16M | ~6 Go | 3h |

### Pour chaque config, lancer :

```bash
# B0 + copy (sans augmentation)
python models/cogs_compositional.py \
    --dataset cogs \
    --copy \
    --scheduled-sampling \
    --d-model 256 \
    --n-layers 2 \
    --n-heads 8 \
    --seed 42 \
    --epochs 60

# Répéter avec --d-model 256 --n-layers 4
# Répéter avec --d-model 384 --n-layers 4
```

### Ce qu'on regarde

Le by_cat greedy. Deux questions :

1. **Est-ce que le copy scale ?** Les catégories lexicales (97-99% à 1M) restent-elles à 97%+ à 4M/8M/16M ? Si elles baissent, le modèle plus gros mémorise au lieu de copier.

2. **Est-ce que les 0% structurels bougent ?** active_to_passive, passive_to_active, do_dative_to_pp_dative... Si elles passent de 0% à >0% avec plus de couches, le scaling suffit et les innovations sont optionnelles. Si elles restent à 0%, les innovations sont nécessaires.

### Output

```
scaling/
  scaling_S_s42.json     # by_cat pour chaque config
  scaling_M_s42.json
  scaling_L_s42.json
  scaling_XL_s42.json
  scaling_summary.md     # Tableau comparatif
```

Pousser sur GitHub :
```bash
git add scaling/
git commit -m "Scaling test: 1M/4M/8M/16M with copy mechanism"
git push
```

---

## Tâche 3 — Sweep innovations (après Machine A)

Quand Machine A a trouvé les innovations qui marchent, Machine B fait le sweep 5 seeds dessus.

```bash
# Exemple : si innovation 1 (gate surprise) marche
python experiments/sweep_final.py \
    --benchmark cogs \
    --variant B0_copy_surprise_gate \
    --seeds 42,123,456,789,1337
```

Temps : dépend du nombre d'innovations gagnantes. ~4h par variante × 5 seeds.

---

## Communication entre machines

### GitHub = source de vérité

```
Machine A (dev) :
  1. Implémente une innovation
  2. Teste sur 1 seed
  3. Si ça marche : commit + push
  4. Tague : "ready-for-sweep"

Machine B (compute) :
  1. git pull
  2. Lance le sweep sur la nouvelle variante
  3. Commit les résultats
  4. Push
```

### Convention de branches

```
main              — code stable, résultats validés
dev/innovation-N  — innovation en cours sur Machine A
sweep/variant-X   — sweep en cours sur Machine B
```

### Convention de commits

```
[SWEEP] SCAN A4-all 5 seeds complete
[SCALE] COGS copy d=256 2+2 s42
[INNOV] Innovation 1: surprise gate implemented
[FIX] auto_loop proper_names disabled on SLOG
[DOC] Updated README with copy results
```

---

## Monitoring

### Vérifier qu'un run tourne

```bash
# Voir les runs en cours
ls -lt runs_master/ | head -5

# Voir les logs du dernier run
tail -f runs_master/$(ls -t runs_master | head -1)/train.log
```

### Vérifier l'utilisation GPU

```bash
nvidia-smi
# Doit montrer ~1-6 Go utilisés selon la config
```

### Si un run crashe

Le sweep script reprend automatiquement. Vérifier :
```bash
# Quels runs sont terminés ?
find sweep/runs/ -name "summary.json" | wc -l

# Quels runs manquent ?
python experiments/sweep_final.py --benchmark cogs --recap
```

---

## Sanity checks post-sweep

Avant de considérer les résultats comme finaux :

- [ ] A0 sur addprim_jump = ~0% (5 seeds)
- [ ] A4-all sur addprim_jump = ~99%+ (5 seeds)
- [ ] B0 sur COGS gen = ~0% (5 seeds)
- [ ] B0+copy sur COGS gen = ~52% (5 seeds)
- [ ] Dev accuracy > 95% pour toutes les variantes
- [ ] Aucune variante avec std > 10% sur 5 seeds
- [ ] Tailles des catégories COGS loggées
- [ ] Greedy eval fait pour toutes les variantes COGS

Si un check échoue, ne pas publier. Investiguer d'abord.

---

## Contact

Résultats et questions → pousser sur GitHub ou contacter Julien directement.
Ne pas modifier le code sur Machine B sauf pour des fixes de bugs.
Tout le développement se fait sur Machine A.
