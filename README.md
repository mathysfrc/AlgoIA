# 🥗 NutriAI - Système de Recommandation Alimentaire Personnalisé

## 📋 Description du Projet

NutriAI est un système de recommandation alimentaire intelligent basé sur le **Machine Learning** qui s'adapte entièrement au profil utilisateur (âge, poids, taille, genre, niveau d'activité, objectif nutritionnel).

### 🎯 Objectifs

- Recommander des aliments personnalisés selon le profil nutritionnel
- Générer des plans de repas équilibrés automatiquement
- Classifier les aliments en 4 catégories (Privilégier/Modération/Neutre/Éviter)
- Comparer scientifiquement plusieurs algorithmes de ML

---

## 🤖 Modèles de Machine Learning Utilisés

### 1. **Decision Tree Classifier** (Arbre de Décision)
- **Usage** : Classification des aliments en 4 catégories
- **Avantages** : Interprétable, visualisable, explique les décisions
- **Optimisation** : GridSearchCV sur `max_depth`, `min_samples_leaf`, `criterion`

### 2. **Random Forest Classifier** (Forêt Aléatoire)
- **Usage** : Classification ensemble pour améliorer la précision
- **Avantages** : Robuste aux outliers, réduit l'overfitting
- **Optimisation** : GridSearchCV sur `n_estimators`, `max_depth`, `min_samples_split`

### 3. **K-Nearest Neighbors (KNN)**
- **Usage** : Recommandation d'aliments similaires
- **Avantages** : Simple, efficace pour la similarité
- **Optimisation** : GridSearchCV sur `n_neighbors`, `weights`, `metric`

### 4. **Gradient Boosting Classifier**
- **Usage** : Classification avancée avec boosting
- **Avantages** : Très performant, gère bien les données complexes
- **Optimisation** : GridSearchCV sur `n_estimators`, `learning_rate`, `max_depth`

### 5. **Algorithme de Classification par Seuils Dynamiques**
- **Type** : Système de règles basé sur le profil
- **Usage** : Attribution des rôles alimentaires selon les besoins nutritionnels
- **Avantages** : 100% personnalisé, s'adapte en temps réel

### 6. **Algorithme Glouton d'Optimisation**
- **Type** : Heuristique d'optimisation combinatoire
- **Usage** : Génération de plans de repas équilibrés
- **Avantages** : Rapide, atteint les objectifs nutritionnels

---

## 📊 Dataset

**Source** : `data/food_data.csv`

**Dimensions requises** :
- ✅ Minimum 1000 lignes (aliments)
- ✅ Minimum 10 colonnes (features nutritionnelles)

**Colonnes principales** :
- `Food Category` : Nom de l'aliment
- `Meal Type` : Type de repas
- `Calories`, `Protein`, `Carbs`, `Fat` : Macronutriments
- `Fiber`, `Sugar`, `Water` : Micronutriments

**Preprocessing** :
- Suppression des valeurs manquantes
- Retrait des outliers (IQR method)
- Feature engineering (ratios, indices)
- Discrétisation des variables continues

---

## 🛠️ Installation

### Prérequis

```bash
Python >= 3.8
```

### Installation des dépendances

```bash
pip install -r requirements.txt
```

**Contenu de `requirements.txt`** :
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
pillow>=10.0.0
```

---

## 🚀 Utilisation

### 1. Entraînement Initial

```bash
python main.py
```

**Ce script va** :
1. ✅ Nettoyer le dataset
2. ✅ Entraîner les modèles pour 4 profils types
3. ✅ Effectuer le fine-tuning des hyperparamètres
4. ✅ Comparer les 4 algorithmes de ML
5. ✅ Générer les visualisations et rapports

**Durée** : 5-10 minutes

**Fichiers générés** :
- `data/processed_nutrition.csv`
- `models/knn_personalized.pkl`
- `models/decision_tree_personalized.pkl`
- `models/best_*.pkl` (meilleurs modèles)
- `models/model_comparison.csv`
- `models/confusion_matrices.png`
- `models/performance_comparison.png`
- `models/evaluation_report.txt`

### 2. Lancement de l'Application

```bash
streamlit run app.py
```

**Interface Web** : http://localhost:8501

---

## 📱 Fonctionnalités de l'Application

### 🎯 Onglet 1 : Recommandations Personnalisées
- Filtrage par rôle alimentaire (Privilégier/Modération/Neutre/Éviter)
- Score de pertinence personnalisé
- Justification des recommandations

### 🍽️ Onglet 2 : Plan de Repas
- Génération automatique d'un plan quotidien
- Équilibrage des macronutriments
- Graphique comparatif cible vs réel

### 🔍 Onglet 3 : Analyse d'Aliment
- Recherche d'aliments
- Analyse nutritionnelle détaillée
- Recommandations similaires

### 🌳 Onglet 4 : Arbre Explicatif
- Visualisation de l'arbre de décision
- Règles de classification
- Importance des features

### 📊 Onglet 5 : Performances & Comparaison
- Tableau comparatif des 4 modèles
- Matrices de confusion
- Graphiques de performance
- Rapport détaillé téléchargeable

---

## 🔬 Métriques d'Évaluation

### Métriques Calculées

- **Accuracy** : Proportion de prédictions correctes
- **F1-Score** : Moyenne harmonique de précision et recall
- **Precision** : Taux de vrais positifs
- **Recall** : Taux de détection
- **Temps d'entraînement** : Performance computationnelle
- **Temps d'inférence** : Vitesse de prédiction

### Validation

- **Validation croisée 5-fold** : Pour tous les modèles
- **Split 80/20** : Train/Test
- **GridSearchCV** : Optimisation automatique des hyperparamètres

---

## 📂 Structure du Projet

```
nutriai/
│
├── data/
│   ├── food_data.csv              # Dataset brut
│   └── processed_nutrition.csv    # Dataset nettoyé
│
├── models/
│   ├── knn_personalized.pkl
│   ├── decision_tree_personalized.pkl
│   ├── best_*.pkl
│   ├── model_comparison.csv
│   ├── confusion_matrices.png
│   ├── performance_comparison.png
│   └── evaluation_report.txt
│
├── data_preprocessing.py          # Nettoyage et feature engineering
├── models.py                      # Modèles ML personnalisés
├── model_evaluation.py            # Évaluation et comparaison
├── meal_planner.py                # Générateur de plans de repas
├── user_profile.py                # Calculs nutritionnels
├── main.py                        # Script d'entraînement
├── app.py                         # Application Streamlit
├── requirements.txt               # Dépendances
└── README.md                      # Documentation
```

---

## 🧪 Exemple de Résultats Attendus

### Comparaison des Modèles

| Modèle              | Accuracy | F1-Score | Temps Train (s) |
|---------------------|----------|----------|-----------------|
| Decision Tree       | 0.8520   | 0.8485   | 12.34           |
| Random Forest       | 0.8735   | 0.8702   | 45.67           |
| KNN                 | 0.8312   | 0.8278   | 3.21            |
| Gradient Boosting   | 0.8898   | 0.8865   | 78.92           |

🏆 **Meilleur modèle** : Gradient Boosting (F1: 0.8865)

---

## 🎓 Conformité Académique

### ✅ Critères Respectés

| Critère | Requis | Implémenté |
|---------|--------|------------|
| Dataset | ≥1000 lignes, ≥10 colonnes | ✅ |
| Algorithmes ML | Plusieurs algorithmes | ✅ 6 modèles |
| Justification | Expliquer les choix | ✅ Documentation |
| Fine-tuning | Optimisation hyperparamètres | ✅ GridSearchCV |
| Métriques | Mesures de performance | ✅ 6 métriques |
| Comparaison | Comparer les modèles | ✅ 4 algorithmes |
| Validation | Cross-validation | ✅ 5-fold CV |

---

## 💡 Justification des Algorithmes

### Decision Tree
**Choix** : Interprétabilité maximale  
**Raison** : L'utilisateur peut voir POURQUOI un aliment est recommandé

### Random Forest
**Choix** : Amélioration de la robustesse  
**Raison** : Réduit l'overfitting, meilleure généralisation

### KNN
**Choix** : Recommandation par similarité  
**Raison** : Trouve des aliments similaires dans l'espace nutritionnel personnalisé

### Gradient Boosting
**Choix** : Performance maximale  
**Raison** : Meilleur F1-Score, idéal pour la production

### Seuils Dynamiques
**Choix** : Personnalisation totale  
**Raison** : S'adapte instantanément au profil utilisateur

### Algorithme Glouton
**Choix** : Optimisation combinatoire  
**Raison** : Génère des plans de repas équilibrés rapidement

---

## 🔮 Améliorations Futures

- [ ] Deep Learning (CNN pour images d'aliments)
- [ ] Système de recommandation collaboratif
- [ ] Gestion des allergies et préférences
- [ ] API REST pour intégration mobile
- [ ] Base de données PostgreSQL
- [ ] Authentification utilisateur
- [ ] Historique des plans de repas

---

## 👥 Auteurs

**Projet académique** - Master 2 Sciences de l'Ingénieur / Architecture des Systèmes Informatiques

**Cours** : Algorithmes avancés de Machine Learning (2025-2026)

---

## 📄 Licence

Ce projet est réalisé dans un cadre académique.

---

## 📞 Support

Pour toute question concernant le projet :
- Consulter la documentation dans `/docs`
- Voir les exemples dans `/examples`
- Lire le rapport d'évaluation : `models/evaluation_report.txt`

---

## 🎉 Démo

```bash
# Installation rapide
git clone <repository>
cd nutriai
pip install -r requirements.txt

# Entraînement (5-10 min)
python main.py

# Lancement de l'app
streamlit run app.py
```

**Enjoy! 🥗**