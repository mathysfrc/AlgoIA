# Documentation Technique - Système NutriAI

## 📋 Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Flux de Traitement des Données](#flux-de-traitement-des-données)
3. [Préprocessing des Données](#préprocessing-des-données)
4. [Système de Classification](#système-de-classification)
5. [Modèles d'Apprentissage Machine](#modèles-dapprentissage-machine)
6. [Planificateur de Repas](#planificateur-de-repas)
7. [Évaluation des Modèles](#évaluation-des-modèles)
8. [Architecture Globale](#architecture-globale)

---

## Vue d'ensemble

**NutriAI** est un système de recommandation nutritionnelle personnalisé qui utilise plusieurs algorithmes d'apprentissage machine pour classer et recommander des aliments selon le profil unique de chaque utilisateur.

### Composants Principaux

- **Classification personnalisée** : Attribue un rôle à chaque aliment (Privilégier, Modération, Neutre, Éviter)
- **KNN (K-Nearest Neighbors)** : Recommandations basées sur la similarité
- **Arbre de Décision** : Classification explicable avec règles claires
- **Random Forest** : Ensemble d'arbres pour robustesse
- **Gradient Boosting** : Modèle avancé pour meilleure précision
- **Planificateur de Repas** : Algorithme glouton pour générer des plans quotidiens

---

## Flux de Traitement des Données

### 1. Sources de Données

Le système commence avec plusieurs datasets bruts dans le dossier `datasets/` :

```
datasets/
├── dataset1.csv
├── dataset2.csv
├── dataset3.csv
└── 5_to_1_dataset/
    ├── dataset1.csv
    ├── dataset2.csv
    ├── dataset3.csv
    ├── dataset4.csv
    └── dataset5.csv
```

### 2. Étape 1 : Combinaison des Datasets

**Fichier** : `preprocessing/create_dataset.py` et `preprocessing/5_to_1_dataset.py`

**Processus** :
- Les fichiers dans `datasets/` sont combinés en un seul dataset
- Normalisation des colonnes (noms différents selon les sources)
- Remplissage intelligent des valeurs manquantes :
  - Calories calculées : `4×Protein + 4×Carbs + 9×Fat`
  - Fat estimé à partir de Saturated Fat
  - Carbs = Sugar + Fiber + moyenne globale
  - Meal Type assigné aléatoirement si manquant

**Résultat** : `data/final_food.csv`

### 3. Étape 2 : Nettoyage et Feature Engineering

**Fichier** : `preprocessing/data_preprocessing.py`

**Traitements appliqués** :

#### a) Suppression des valeurs manquantes
- Suppression des lignes avec NaN
- Suppression des aliments avec Calories ≤ 0

#### b) Suppression des outliers (méthode IQR)
Pour chaque colonne nutritionnelle :
```
Q1 = 25ème percentile
Q3 = 75ème percentile
IQR = Q3 - Q1
Limite inférieure = Q1 - 1.5×IQR
Limite supérieure = Q3 + 1.5×IQR
```
Les valeurs en dehors de ces limites sont supprimées.

#### c) Feature Engineering

**Ratios nutritionnels** :
- `prot_ratio = (Protein × 4) / Calories`
- `carb_ratio = (Carbs × 4) / Calories`
- `fat_ratio = (Fat × 9) / Calories`

**Indices dérivés** :
- `fiber_per_100kcal = Fiber / (Calories / 100)`
- `sugar_per_100kcal = Sugar / (Calories / 100)`
- `density_kcal_100g = Calories / 100`

**Indice de satiété** :
```
satiety_index = (Protein × 1.0 + Fiber × 0.8 + Water × 0.05 - Sugar × 0.3) / (Calories/100 + 1)
```

**Score d'équilibre** :
```
balance_score = 0.35×prot_ratio + 0.25×(1-carb_ratio) + 0.20×(1-fat_ratio) + 0.15×(fiber_per_100kcal/10) + 0.05×(1-sugar_per_100kcal/50)
```
Normalisé entre 0 et 1.

#### d) Discrétisation
- `calorie_level` : Faible (0-100), Moyen (100-300), Élevé (300+)
- `protein_level` : Faible (0-5), Moyen (5-15), Élevé (15+)
- `fiber_level` : Faible (0-2), Moyen (2-5), Élevé (5+)

**Résultat** : `data/processed_nutrition.csv`

### 4. Étape 3 : Ajout de Bruit (Optionnel)

**Fichier** : `preprocessing/add_noise.py` et `add_noise_simple.py`

**Objectif** : Rendre les résultats plus réalistes (éviter des scores parfaits de 1.0000)

**Types de bruit** :

#### a) Bruit nutritionnel (5% par défaut)
- Bruit gaussien ajouté aux valeurs nutritionnelles
- `valeur_bruitée = valeur_originale × (1 + ε)` où ε ~ N(0, 0.05)
- Valeurs garanties positives

#### b) Bruit sur les labels (3% par défaut)
- Change aléatoirement quelques classifications
- Simule des cas ambigus ou erreurs

#### c) Valeurs manquantes (1% par défaut)
- Ajoute quelques NaN pour tester la robustesse

**Résultat** : `data/processed_nutrition_noisy.csv`

---

## Préprocessing des Données

### Profil Utilisateur

**Fichier** : `models.py` - méthode `set_user_profile()`

Le système calcule un profil complet pour chaque utilisateur :

#### 1. BMR (Basal Metabolic Rate) - Formule Mifflin-St Jeor

**Homme** :
```
BMR = 88.362 + (13.397 × poids) + (4.799 × taille) - (5.677 × âge)
```

**Femme** :
```
BMR = 447.593 + (9.247 × poids) + (3.098 × taille) - (4.330 × âge)
```

#### 2. TDEE (Total Daily Energy Expenditure)

```
TDEE = BMR × multiplicateur_activité
```

Multiplicateurs :
- Sédentaire : 1.2
- Léger : 1.375
- Modéré : 1.55
- Intense : 1.725
- Athlète : 1.9

#### 3. Objectifs Nutritionnels

**Perte de poids** :
- Calories = TDEE - 500
- Protéines = poids × 2.2 g/kg
- Lipides = poids × 0.8 g/kg

**Gain de masse** :
- Calories = TDEE + 500
- Protéines = poids × 1.8 g/kg
- Lipides = poids × 1.0 g/kg

**Maintien** :
- Calories = TDEE
- Protéines = poids × 2.0 g/kg
- Lipides = poids × 0.9 g/kg

**Glucides** (toujours) :
```
Glucides = (Calories - (Protéines × 4 + Lipides × 9)) / 4
```

#### 4. Encodages Numériques

Pour intégration dans les modèles ML :
- `gender_encoded` : 1 = Homme, 0 = Femme
- `activity_encoded` : 0-4 (Sédentaire à Athlète)
- `objective_encoded` : 0 = perte, 1 = maintien, 2 = gain
- `age_group` : 0 = jeune (<25), 1 = adulte (25-50), 2 = senior (50+)

### Ajout des Features Utilisateur au Dataset

**Méthode** : `_add_user_profile_features()`

Pour chaque aliment, le système ajoute les colonnes suivantes :
- `user_weight`, `user_height`, `user_age`, `user_bmi`
- `user_bmr`, `user_tdee`
- `user_gender`, `user_activity`, `user_objective`, `user_age_group`
- `user_target_calories`, `user_target_protein`, `user_target_carbs`, `user_target_fat`

### Calcul des Scores Personnalisés

**Méthode** : `_calculate_personalized_scores()`

#### a) Fit Scores (0-1)

Mesurent à quel point un aliment correspond aux besoins :

```
protein_fit = 1 - |Protein - target_protein/6| / (target_protein/6 + 1)
carbs_fit = 1 - |Carbs - target_carbs/6| / (target_carbs/6 + 1)
fat_fit = 1 - |Fat - target_fat/6| / (target_fat/6 + 1)
calorie_fit = 1 - |Calories - target_calories/4| / (target_calories/4 + 1)
```

#### b) User Score Global

**Perte de poids** :
```
user_score = 0.35×protein_fit + 0.25×calorie_fit + 0.20×(Fiber/max) + 0.15×(1-Sugar/max) + 0.05×(satiety_index/max)
```

**Gain de masse** :
```
user_score = 0.30×protein_fit + 0.30×carbs_fit + 0.25×(Calories/max) + 0.15×calorie_fit
```

**Maintien** :
```
user_score = 0.30×protein_fit + 0.25×carbs_fit + 0.25×fat_fit + 0.20×calorie_fit
```

---

## Système de Classification

**Fichier** : `models.py` - méthode `classify_food_role_personalized()`

### Rôles Attribués

1. **Privilégier** : Aliments très adaptés au profil
2. **Modération** : Aliments acceptables mais à consommer avec modération
3. **Neutre** : Aliments ni particulièrement bons ni mauvais
4. **Éviter** : Aliments déconseillés pour ce profil

### Règles de Classification (Dynamiques selon Objectif)

#### Objectif : Perte de Poids

**Privilégier** si :
- Protéines ≥ 80% du seuil par repas
- Calories ≤ 120% du seuil par repas
- Fibres ≥ 3g
- Sucre ≤ 8g

**Modération** si :
- Lipides > 150% du seuil OU
- Sucre entre 8-15g

**Éviter** si :
- Calories > 180% du seuil OU
- Sucre > 15g OU
- Lipides > 200% du seuil

#### Objectif : Gain de Masse

**Privilégier** si :
- Calories ≥ 120% du seuil
- Protéines ≥ 70% du seuil
- Glucides ≥ 80% du seuil

**Modération** si :
- Calories < 80% du seuil

**Éviter** si :
- Sucre > 20g ET Protéines < 50% du seuil

#### Objectif : Maintien

**Privilégier** si :
- Protéines entre 70-130% du seuil
- Calories entre 80-120% du seuil

**Modération** si :
- Lipides > 150% du seuil OU
- Sucre > 12g

**Éviter** si :
- Calories > 200% du seuil OU
- Sucre > 20g

### Filtrage des Aliments Malsains

**Blacklist** automatique :
- Fast food (McDonald's, Burger King, KFC, etc.)
- Sodas (Coca, Pepsi, Fanta, etc.)
- Snacks industriels (chips, bonbons, etc.)

---

## Modèles d'Apprentissage Machine

### 1. KNN (K-Nearest Neighbors)

**Fichier** : `models.py` - méthode `train_knn_personalized()`

#### Principe

Trouve les K aliments les plus similaires à un aliment donné en fonction de leurs caractéristiques nutritionnelles ET du profil utilisateur.

#### Features Utilisées

**Base** :
- Calories, Protein, Carbs, Fat, Fiber, Sugar, Water
- density_kcal_100g, satiety_index

**Personnalisées** :
- user_score, protein_fit, carbs_fit, fat_fit
- user_weight, user_age, user_bmi
- user_gender, user_activity, user_objective
- user_target_protein, user_target_carbs, user_target_fat

**Total** : ~20 features

#### Processus

1. Normalisation avec `StandardScaler`
2. Calcul des distances euclidiennes
3. Sélection des 6 plus proches voisins (K=6)
4. Recommandations basées sur ces voisins

#### Utilisation

```python
recommendations = ai.recommend_similar_personalized("poulet")
# Retourne les 5 aliments les plus similaires avec leur score
```

### 2. Arbre de Décision

**Fichier** : `models.py` - méthode `train_decision_tree_personalized()`

#### Principe

Arbre binaire qui classe les aliments selon une série de règles "si-alors" basées sur les features.

#### Features Utilisées

**Adaptatives** (basées sur le profil) :
- `meets_protein_need` : Protéines ≥ 80% du seuil
- `meets_carbs_need` : Glucides ≥ 80% du seuil
- `within_calorie_target` : Calories entre 70-130% du seuil
- `low_sugar` : Sucre < 10g
- `high_fiber` : Fibres > 4g
- `healthy_fat_ratio` : Lipides entre 50-150% du seuil

**Profil utilisateur** :
- user_age, user_bmi, user_gender
- user_activity, user_objective
- user_target_protein, user_target_calories

**Scores** :
- user_score, satiety_index

#### Hyperparamètres

- `max_depth=8` : Profondeur maximale
- `min_samples_leaf=15` : Minimum d'échantillons par feuille
- `class_weight='balanced'` : Équilibre les classes

#### Avantages

- **Explicable** : Règles claires et compréhensibles
- **Personnalisé** : S'adapte au profil utilisateur
- **Rapide** : Inférence très rapide

### 3. Random Forest

**Fichier** : `model_evaluation.py` - méthode `tune_random_forest()`

#### Principe

Ensemble de plusieurs arbres de décision (80-150 arbres) qui votent pour la classification finale.

#### Processus

1. **Bootstrap** : Chaque arbre est entraîné sur un échantillon aléatoire (avec remise)
2. **Feature Randomness** : À chaque split, seulement un sous-ensemble de features est considéré
3. **Vote** : La classe majoritaire parmi tous les arbres est choisie

#### Hyperparamètres Optimisés

- `n_estimators` : 80-150 arbres
- `max_depth` : 10-20 niveaux
- `min_samples_split` : 2-10
- `min_samples_leaf` : 1-2

#### Avantages

- **Robuste** : Moins sensible au surapprentissage
- **Précis** : Généralement meilleur qu'un seul arbre
- **Gère les features importantes** : Identifie automatiquement les features clés

### 4. Gradient Boosting

**Fichier** : `model_evaluation.py` - méthode `tune_gradient_boosting()`

#### Principe

Séquence d'arbres où chaque arbre corrige les erreurs de l'arbre précédent.

#### Processus

1. Premier arbre fait des prédictions
2. Calcul des erreurs (résidus)
3. Nouvel arbre entraîné pour prédire ces erreurs
4. Répété 80-120 fois
5. Prédiction finale = somme de toutes les prédictions

#### Hyperparamètres Optimisés

- `n_estimators` : 80-120 arbres
- `learning_rate` : 0.05-0.1 (vitesse d'apprentissage)
- `max_depth` : 3-5 niveaux

#### Avantages

- **Très précis** : Souvent le meilleur modèle
- **Apprentissage séquentiel** : Chaque arbre améliore le précédent
- **Gère les relations complexes** : Capture des patterns non-linéaires

### Comparaison des Modèles

| Modèle | Précision | Vitesse | Explicabilité | Robustesse |
|--------|-----------|---------|---------------|------------|
| KNN | Moyenne | Rapide | Faible | Moyenne |
| Decision Tree | Bonne | Très rapide | **Excellente** | Faible |
| Random Forest | **Très bonne** | Moyenne | Bonne | **Excellente** |
| Gradient Boosting | **Excellente** | Lente | Moyenne | Bonne |

---

## Planificateur de Repas

**Fichier** : `meal_planner.py`

### Principe

Algorithme **glouton** (greedy) qui construit un plan de repas quotidien en sélectionnant itérativement les meilleurs aliments pour chaque repas.

### Répartition Calorique

- **Petit-déjeuner** : 25% des calories
- **Déjeuner** : 35% des calories
- **Dîner** : 30% des calories
- **Collation** : 10% des calories

### Algorithme Glouton

Pour chaque repas :

1. **Filtrage initial** :
   - Par type de repas (ex: Petit-déjeuner → Grains, Fruits, Dairy, Eggs)
   - Exclusion des aliments déjà utilisés
   - Exclusion des aliments malsains

2. **Tri par priorité** :
   - D'abord par rôle : Privilégier > Modération > Neutre > Éviter
   - Puis par `user_score` décroissant

3. **Sélection itérative** (3-4 aliments par repas) :

   Pour chaque position :
   - Évaluer tous les candidats restants
   - Calculer l'erreur si on ajoute cet aliment :
     ```
     error = |calories_total - target_calories| × 1.0
            + |protein_total - target_protein| × 2.0  (priorité protéines)
            + |carbs_total - target_carbs| × 1.5
            + |fat_total - target_fat| × 1.2
            - user_score × 50  (bonus si score élevé)
            + 200 si rôle = "Éviter"  (pénalité)
     ```
   - Sélectionner l'aliment avec l'erreur minimale
   - Ajouter au repas et mettre à jour les totaux

4. **Contraintes** :
   - Maximum 1 aliment "Éviter" par repas
   - Pas de doublons dans un même repas
   - Maximum 4 aliments pour Déjeuner/Dîner, 3 pour les autres

### Résultat

Retourne :
- **Plan** : Dictionnaire `{repas: [liste_aliments]}`
- **Totaux** : Calories, Protéines, Glucides, Lipides totaux

---

## Évaluation des Modèles

**Fichier** : `model_evaluation.py`

### Processus d'Évaluation

#### 1. Préparation des Données

- Filtrage des classes trop rares (< 2 échantillons)
- Vérification d'au moins 2 classes restantes
- Features sélectionnées (même liste que KNN)
- Split train/test : 80/20 avec stratification

#### 2. Fine-tuning des Hyperparamètres

**Méthode** : GridSearchCV avec validation croisée

- **Decision Tree** : 3-fold CV
- **Random Forest** : 3-fold CV
- **KNN** : 5-fold CV
- **Gradient Boosting** : 3-fold CV

**Métrique d'optimisation** : F1-Score pondéré

#### 3. Métriques Calculées

Pour chaque modèle :

- **Accuracy** : Pourcentage de prédictions correctes
- **F1-Score** : Moyenne harmonique de précision et rappel (pondéré par classe)
- **Precision** : Pourcentage de vrais positifs parmi les prédictions positives
- **Recall** : Pourcentage de vrais positifs détectés
- **Temps d'entraînement** : Durée du fine-tuning
- **Temps d'inférence** : Temps pour prédire sur le test set

#### 4. Visualisations Générées

- **Matrices de confusion** : Pour chaque modèle (PNG)
- **Graphique F1-Score** : Comparaison visuelle (PNG)
- **Rapport texte** : Classification report détaillé (TXT)
- **Tableau comparatif** : CSV avec toutes les métriques

#### 5. Sauvegarde des Meilleurs Modèles

Les modèles optimisés sont sauvegardés dans `models/` :
- `decision_tree_best.pkl`
- `random_forest_best.pkl`
- `knn_best.pkl`
- `gradient_boosting_best.pkl`

### Interprétation des Résultats

**F1-Score > 0.85** : Excellent
**F1-Score 0.75-0.85** : Bon
**F1-Score 0.65-0.75** : Acceptable
**F1-Score < 0.65** : À améliorer

**Accuracy** : Doit être interprété avec précaution si classes déséquilibrées (d'où l'importance du F1-Score)

---

**Gestion des modèles**
1. Bagging
Random Forest (RandomForestClassifier) utilise le bagging
Plusieurs arbres entraînés sur des sous-ensembles bootstrap différents
```
model_evaluation.pyLine 14
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
```
2. Boosting
Gradient Boosting (GradientBoostingClassifier) utilise le boosting
Séquentiel, chaque arbre corrige les erreurs du précédent
```
model_evaluation.pyLine 14
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
```
3. Pré-pruning (limitation de croissance)
Pour l'arbre de décision dans models.py :
max_depth=8 : limite la profondeur
min_samples_leaf=15 : minimum d'échantillons par feuille
```
models.pyLines 481-489
        self.dt = DecisionTreeClassifier(            # max_depth=8 : Profondeur maximale            # min_samples_leaf=15 : Minimum d'échantillons par feuille            # class_weight='balanced' : Équilibre les classes            max_depth=8,            min_samples_leaf=15,            random_state=42,            class_weight='balanced'        )
```
4. Optimisation des hyperparamètres (GridSearchCV)
Utilisé dans model_evaluation.py pour tous les modèles
Teste différentes combinaisons d'hyperparamètres avec validation croisée
```
model_evaluation.pyLines 105-110
        gs = GridSearchCV(            model, param_grid,            cv=cv_folds,            scoring='f1_weighted', # Precision + Recall            n_jobs=-1        )
```
5. Validation croisée (Cross-Validation)
Utilisée via GridSearchCV avec un nombre de folds adaptatif (2-5 selon les classes)
```
model_evaluation.pyLines 33-36
    def _get_safe_cv(self, y, max_cv=5):        """Détermine un nombre de folds valide selon la classe la plus rare"""        min_class_size = min(Counter(y).values())        return max(2, min(max_cv, min_class_size))
```
6. Équilibrage des classes
class_weight='balanced' pour l'arbre de décision
Ajuste automatiquement les poids selon la fréquence des classes
```
models.pyLine 488
            class_weight='balanced'
```
7. Normalisation des features
StandardScaler pour normaliser les données avant l'entraînement
```
model_evaluation.pyLine 26
        self.scaler = StandardScaler()
```
8. Stratification
Utilisée dans train_test_split pour préserver la distribution des classes
```
model_evaluation.pyLine 77
            stratify=y_encoded if min_class_size >= 2 else
```

## Architecture Globale

### Flux Complet

```
┌─────────────────┐
│  datasets/      │  Datasets bruts
│  (sources)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ create_dataset  │  Combinaison et normalisation
│ 5_to_1_dataset  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ final_food.csv  │  Dataset combiné
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ data_preprocessing│  Nettoyage, feature engineering
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│processed_nutrition│  Dataset prêt
│     .csv        │
└────────┬────────┘
         │
         ├─────────────────┐
         │                 │
         ▼                 ▼
┌─────────────────┐  ┌─────────────────┐
│ add_noise       │  │  Sans bruit     │
│ (optionnel)     │  │                 │
└────────┬────────┘  └────────┬────────┘
         │                    │
         ▼                    ▼
┌─────────────────┐  ┌─────────────────┐
│processed_nutrition│ │processed_nutrition│
│  _noisy.csv     │  │     .csv        │
└────────┬────────┘  └────────┬────────┘
         │                    │
         └────────┬───────────┘
                  │
                  ▼
         ┌─────────────────┐
         │   NutriAI       │  Initialisation
         │   (models.py)   │
         └────────┬────────┘
                  │
         ┌────────┴────────┐
         │                 │
         ▼                 ▼
┌─────────────────┐  ┌─────────────────┐
│ set_user_profile│  │  Classification │
│  (BMR, TDEE,    │  │  (Rôles)        │
│   objectifs)    │  │                 │
└────────┬────────┘  └────────┬────────┘
         │                    │
         └────────┬───────────┘
                  │
         ┌────────┴────────┐
         │                 │
         ▼                 ▼
┌─────────────────┐  ┌─────────────────┐
│ train_knn       │  │ train_decision  │
│  (similarité)   │  │  _tree          │
└────────┬────────┘  └────────┬────────┘
         │                    │
         └────────┬───────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ model_evaluation│  Fine-tuning et comparaison
         │  (4 modèles)    │
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  Recommandations│  Utilisation
         │  Plan de repas  │
         └─────────────────┘
```

### Fichiers Clés

| Fichier | Rôle |
|---------|------|
| `main.py` | Point d'entrée, orchestre tout le processus |
| `models.py` | Classe NutriAI, KNN, Decision Tree personnalisés |
| `model_evaluation.py` | Évaluation comparative des 4 modèles |
| `meal_planner.py` | Algorithme glouton pour plans de repas |
| `preprocessing/data_preprocessing.py` | Nettoyage et feature engineering |
| `preprocessing/create_dataset.py` | Combinaison des datasets sources |
| `app.py` | Interface Streamlit pour utilisation interactive |

### Données Utilisées

**Tous les modèles utilisent les données du dossier `data/`** :

- Source initiale : `data/final_food.csv`
- Traité : `data/processed_nutrition.csv`
- Avec bruit (optionnel) : `data/processed_nutrition_noisy.csv`

**Aucun modèle n'utilise directement les fichiers dans `datasets/`** - ceux-ci servent uniquement à créer `final_food.csv`.

---

## Résumé

### Points Clés

1. **Traitement complet** : Les données passent par plusieurs étapes de nettoyage et d'enrichissement
2. **Personnalisation** : Chaque modèle intègre le profil utilisateur complet
3. **Multi-modèles** : 4 algorithmes différents pour robustesse et comparaison
4. **Explicabilité** : L'arbre de décision fournit des règles claires
5. **Optimisation** : Fine-tuning automatique des hyperparamètres
6. **Évaluation rigoureuse** : Validation croisée et métriques multiples















