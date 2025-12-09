# Quick Start Guide - NutriAI

## Installation Rapide (5 minutes)

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Vérifier le dataset

Assurez-vous d'avoir le fichier `data/final_food.csv` avec :
- +- 19.000 lignes
- 10 colonnes
- Colonnes requises : `Food Category`, `Calories`, `Protein`, `Carbs`, `Fat`, `Fiber`, `Sugar`, `Water`, `Meal Type`

### 3. Entraîner les modèles

```bash
python main.py
```

**Durée : 5-10 minutes**

Cela va :
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

## Utilisation de l'Application

### Étape 1 : Configurer votre profil

Dans la **barre latérale** :
1. Entrez votre poids, taille, âge
2. Sélectionnez votre sexe
3. Choisissez votre niveau d'activité
4. Définissez votre objectif (perte/maintien/gain)
5. Cliquez sur **"Calculer mes besoins personnalisés"**

Patientez 10-15 secondes pendant le calcul

### Étape 2 : Explorer les onglets

#### Onglet "Recommandations"
- Sélectionnez "Privilégier" pour voir les meilleurs aliments
- Cliquez sur un aliment pour voir ses détails

#### Onglet "Plan de Repas"
- Cliquez sur **"Générer mon plan complet"**
- Consultez vos 4 repas de la journée
- Comparez avec vos objectifs

#### Onglet "Analyse Aliment"
- Recherchez un aliment (ex: "poulet")
- Analysez sa pertinence pour VOUS
- Découvrez des alternatives similaires

#### Onglet "Arbre Explicatif"
- Visualisez l'arbre de décision
- Comprenez les règles de classification
- Identifiez les critères importants

#### Onglet "Performances & Comparaison"
- Consultez les performances des 4 modèles
- Analysez les matrices de confusion
- Téléchargez le rapport détaillé

---

## Troubleshooting

### Erreur "Dataset non trouvé"

```bash
# Vérifier le chemin
ls data/final_food.csv

# Si absent, placer votre dataset dans data/
mkdir data
cp votre_dataset.csv data/final_food.csv
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
1. Utilisez `data_augmentation` pour augmenter les données
2. Combinez plusieurs datasets
3. Générez des données synthétiques
