# Fitness Nutrition Recommender

## Objectif du projet
Ce projet a pour but de développer une application web intelligente d’aide à la recommandation nutritionnelle personnalisée.  
L’utilisateur saisit ses informations (poids, taille, âge, objectif, activité physique…), et l’application :

- Calcule ses besoins caloriques journaliers (TDEE)
- Propose une répartition calorique par repas (petit-déjeuner, déjeuner, dîner, collation)
- Recommande des aliments similaires ou complémentaires selon l’objectif visé

L’application repose sur un modèle de Machine Learning (Decision Tree, KNN, Random Forest, Linear Regression) appliqué sur un dataset de nutrition fusionné à partir de sources Kaggle.

---

## Technologies et librairies utilisées

### Backend : Flask
Flask est utilisé pour créer le serveur web et gérer les routes :

| Route       | Fonctionnalité |
|------------|----------------|
| `/`        | Formulaire de profil utilisateur |
| `/overview`| Affichage du plan calorique et choix d’un aliment |
| `/meal`    | Composition de repas et affichage des graphiques |

Les sessions Flask (`session`) permettent de conserver les données entre les pages (profil, repas en cours, recommandations).

### Modélisation et Machine Learning : Scikit-Learn
La classe `NutritionRecommender` (dans `model.py`) intègre plusieurs techniques de Machine Learning complémentaires :

| Technique                | But / Utilisation                                           | Pourquoi                                                                 |
|---------------------------|------------------------------------------------------------|-------------------------------------------------------------------------|
| Decision Tree Classifier  | Prédire le type de repas selon la composition nutritionnelle | Interprétable, rapide à entraîner, fonctionne bien sur des données tabulaires |
| KNN (Nearest Neighbors)   | Trouver des aliments similaires sur la base de leur profil nutritionnel | Permet de recommander des aliments “proches” dans l’espace des nutriments |
| Linear Regression         | Estimer les calories d’un aliment à partir des autres nutriments | Fournit une estimation simple et continue en cas de données manquantes |
| Random Forest Regressor   | Estimation robuste des calories                                 | Réduit le surapprentissage et améliore la précision sur des données réelles |
| MinMaxScaler              | Mise à l’échelle des données nutritionnelles avant le KNN   | Évite qu’un nutriment domine les autres (ex : sodium ou sucre très élevés) |
| LabelEncoder              | Encodage des types de repas en valeurs numériques          | Nécessaire pour entraîner les modèles supervisés                        |

Chaque modèle est entraîné au chargement du dataset, avec un train/test split 70/30 pour vérifier les performances (accuracy et RMSE).

---

## Calculs personnalisés
La fonction `compute_calorie_plan()` calcule automatiquement :

- **BMR (Basal Metabolic Rate)** selon la formule Mifflin-St Jeor  
  → adaptée au sexe, à l’âge, au poids et à la taille
- **TDEE (Total Daily Energy Expenditure)**  
  → en fonction du niveau d’activité physique

### Plan calorique journalier personnalisé
Répartition des calories par repas :

- Petit-déjeuner → 25 %
- Déjeuner → 35 %
- Dîner → 30 %
- Snack → 10 %

Ajusté selon l’objectif (perte ou prise de poids sur X semaines).

---

## Systèmes de recommandation
Deux approches sont intégrées :

### 1. Pipeline classique (KNN)

```python
meal_type, recs, total_cal = recommender.pipeline(food_name)
```

## Systèmes de recommandation (suite)

### 1. Pipeline classique (KNN)
- Trouve les aliments les plus similaires en valeurs nutritionnelles
- Prédit le type de repas le plus probable (Decision Tree)
- Estime les calories totales du groupe d’aliments

**Exemple** : si l’utilisateur choisit “Chicken”, le système peut recommander “Turkey”, “Fish”, “Tofu”…

### 2. Recommandation intelligente (objectif calorique)
- Cherche des combinaisons optimales d’aliments pour atteindre un objectif calorique précis
- Génère et évalue toutes les combinaisons possibles dans un petit ensemble
- Trie les résultats selon la différence minimale avec la cible calorique

**Exemple** : pour un dîner de 600 kcal avec “Rice” comme base, le modèle peut proposer :  
- Meilleur menu : Rice + Chicken + Vegetables (≈ 590 kcal)  
- Alternatives : Rice + Beef, Rice + Egg + Spinach…

---

## Conception du dataset

Un dataset unique a été créé à partir de 3 datasets Kaggle différents. Chaque dataset contient au moins 11 colonnes similaires :  

**Colonnes :** Food Category, Meal Type, Calories, Protein, Carbs, Fat, Saturated Fat, Fiber, Sugar, Sodium, Water

### Étapes de construction
1. Sélection des datasets sources sur Kaggle (nutrition, aliments communs, calories)
2. Normalisation des colonnes (noms, unités, ordre)
3. Fusion automatique via un script Python :  
   - Regroupement des lignes similaires  
   - Suppression des doublons  
   - Remplacement des valeurs manquantes par la médiane
4. Ajout manuel de quelques entrées `Unknown` pour compléter les catégories manquantes

---

## Structure du projet

project/
│
├── app.py                  # Application Flask principale
├── model.py                # Classe NutritionRecommender (ML & data)
├── all_datas.csv           # Dataset fusionné et nettoyé
│
├── templates/              # Pages HTML
│   ├── base.html
│   ├── profile.html
│   ├── overview.html
│   ├── meal.html
│
├── static/
│   └── style.css           # Style global
│
└── README.md               # Documentation (ce fichier)

---

## Fonctionnement global

- L’utilisateur renseigne son profil
- L’application calcule son TDEE et son plan calorique cible
- L’utilisateur choisit un aliment de base → le modèle propose :
  - Des aliments similaires (KNN)
  - Un menu équilibré selon l’objectif (Smart recommender)
- Le tout est illustré par des graphiques nutritionnels (Matplotlib).

---

## Visualisation des nutriments

Grâce à Matplotlib, l’application génère dynamiquement des graphiques empilés montrant :  
- Les macronutriments (Calories, Protéines, Glucides, Lipides)
- Les fibres, sucres, sels et eau
- Les différences entre les aliments choisis

Les graphiques sont convertis en base64 pour affichage direct dans la page HTML.

---

## Outils de développement

- Python
- Flask
- Pandas / Numpy
- Scikit-learn
- Matplotlib
- HTML
