# 🚀 Quick Start Guide - NutriAI

## Installation Rapide (5 minutes)

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Vérifier le dataset

Assurez-vous d'avoir le fichier `data/food_data.csv` avec :
- ✅ Au moins 1000 lignes
- ✅ Au moins 10 colonnes
- ✅ Colonnes requises : `Food Category`, `Calories`, `Protein`, `Carbs`, `Fat`, `Fiber`, `Sugar`, `Water`, `Meal Type`

### 3. Entraîner les modèles

```bash
python main.py
```

**Durée : 5-10 minutes**

✅ Cela va :
- Nettoyer le dataset
- Entraîner 4 profils types
- Fine-tuner 4 algorithmes de ML
- Générer les rapports d'évaluation

### 4. Lancer l'application

```bash
streamlit run app.py
```

L'app s'ouvrira automatiquement sur http://localhost:8501

---

## 🎯 Utilisation de l'Application

### Étape 1 : Configurer votre profil

👈 Dans la **barre latérale** :
1. Entrez votre poids, taille, âge
2. Sélectionnez votre sexe
3. Choisissez votre niveau d'activité
4. Définissez votre objectif (perte/maintien/gain)
5. Cliquez sur **"Calculer mes besoins personnalisés"**

⏳ Patientez 10-15 secondes pendant le calcul

### Étape 2 : Explorer les onglets

#### 🎯 Onglet "Recommandations"
- Sélectionnez "Privilégier" pour voir les meilleurs aliments
- Cliquez sur un aliment pour voir ses détails

#### 🍽️ Onglet "Plan de Repas"
- Cliquez sur **"Générer mon plan complet"**
- Consultez vos 4 repas de la journée
- Comparez avec vos objectifs

#### 🔍 Onglet "Analyse Aliment"
- Recherchez un aliment (ex: "poulet")
- Analysez sa pertinence pour VOUS
- Découvrez des alternatives similaires

#### 🌳 Onglet "Arbre Explicatif"
- Visualisez l'arbre de décision
- Comprenez les règles de classification
- Identifiez les critères importants

#### 📊 Onglet "Performances & Comparaison"
- Consultez les performances des 4 modèles
- Analysez les matrices de confusion
- Téléchargez le rapport détaillé

---

## 📊 Pour la Présentation Académique

### Documents à préparer :

1. **Slides de présentation (10 min)** :
   - Contexte et problème
   - Dataset et preprocessing
   - 6 algorithmes utilisés
   - Fine-tuning et optimisation
   - Résultats et comparaisons
   - Démo live de l'app

2. **Fichiers à montrer** :
   - `models/model_comparison.csv` (tableau comparatif)
   - `models/confusion_matrices.png` (visualisations)
   - `models/evaluation_report.txt` (rapport complet)

3. **Points clés à mentionner** :
   - ✅ 6 modèles différents (Decision Tree, Random Forest, KNN, Gradient Boosting, Seuils Dynamiques, Algo Glouton)
   - ✅ GridSearchCV pour le fine-tuning
   - ✅ Validation croisée 5-fold
   - ✅ 6 métriques d'évaluation
   - ✅ 100% personnalisé au profil utilisateur
   - ✅ Dataset conforme (>1000 lignes, >10 colonnes)

---

## 🔧 Troubleshooting

### Erreur "Dataset non trouvé"

```bash
# Vérifier le chemin
ls data/food_data.csv

# Si absent, placer votre dataset dans data/
mkdir data
cp votre_dataset.csv data/food_data.csv
```

### Erreur "Modules non trouvés"

```bash
# Réinstaller les dépendances
pip install --upgrade -r requirements.txt
```

### L'évaluation est trop longue

C'est normal ! Le fine-tuning avec GridSearchCV peut prendre :
- Decision Tree : ~1-2 min
- Random Forest : ~3-5 min
- KNN : ~30 sec
- Gradient Boosting : ~2-4 min

**Total : 5-10 minutes**

### Dataset trop petit

Si votre dataset a < 1000 lignes :
1. Utilisez `data_preprocessing.enrich_with_profiles()` pour augmenter les données
2. Combinez plusieurs datasets
3. Générez des données synthétiques

---

## 💡 Conseils pour la Démo

### Scénario de démonstration (5 min) :

1. **Configuration profil** (1 min)
   - "Je configure un profil homme, 30 ans, 75kg, objectif perte de poids"
   
2. **Recommandations** (1 min)
   - "Voici les aliments à privilégier selon MON profil"
   - "Le poulet a un score élevé car riche en protéines"

3. **Plan de repas** (1 min)
   - "Le système génère un plan équilibré automatiquement"
   - "Regardez : 2100 kcal, proche de mon objectif 2000 kcal"

4. **Arbre explicatif** (1 min)
   - "L'arbre de décision explique pourquoi ces recommandations"
   - "Critères : protéines > 15g, calories < 300, sucres < 10g"

5. **Évaluation** (1 min)
   - "Nous avons testé 4 algorithmes"
   - "Gradient Boosting est le plus performant (F1: 0.89)"

---

## 📝 Checklist Avant Présentation

- [ ] Dataset conforme (≥1000 lignes, ≥10 colonnes)
- [ ] `python main.py` exécuté sans erreur
- [ ] Fichiers générés dans `models/`
- [ ] Application testée avec plusieurs profils
- [ ] Screenshots préparés
- [ ] Rapport `evaluation_report.txt` lu
- [ ] Présentation PowerPoint prête
- [ ] Questions potentielles anticipées
- [ ] Démo répétée au moins 2 fois

---

## 🎓 Questions Fréquentes (Q&A)

### Q1 : Pourquoi 6 modèles ?
**R** : Diversité des approches :
- Decision Tree : interprétabilité
- Random Forest : robustesse
- KNN : similarité
- Gradient Boosting : performance
- Seuils dynamiques : personnalisation
- Algo glouton : optimisation

### Q2 : Comment gérez-vous l'overfitting ?
**R** : 
- Validation croisée 5-fold
- Régularisation (max_depth, min_samples_leaf)
- Ensemble methods (Random Forest)

### Q3 : Pourquoi pas du Deep Learning ?
**R** : 
- Dataset trop petit pour DL
- ML classique suffit pour ce problème
- Interprétabilité prioritaire
- (Mais mentionner comme amélioration future)

### Q4 : Comment validez-vous la personnalisation ?
**R** : 
- Test avec 4 profils différents
- Features incluant TOUS les paramètres du profil
- Score de pertinence adaptatif

---

## 🏆 Checklist pour Note Maximale

- [ ] Dataset valide (>1000 lignes, >10 colonnes)
- [ ] Plusieurs algorithmes testés (6 modèles)
- [ ] Fine-tuning implémenté (GridSearchCV)
- [ ] Métriques complètes (Accuracy, F1, Precision, Recall)
- [ ] Comparaison objective (tableau + graphiques)
- [ ] Validation croisée (5-fold)
- [ ] Code propre et commenté
- [ ] Documentation complète (README)
- [ ] Justification des choix (algorithmiques)
- [ ] Résultats interprétés (analyse)
- [ ] Démo fonctionnelle (Streamlit)
- [ ] Présentation claire (10 min)

---

**Bon courage pour votre présentation ! 🎓**