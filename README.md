# **NutriAI – Recommandations Nutritionnelles Personnalisées**

> Application web intelligente basée sur le ML pour un plan alimentaire adapté à vos objectifs : **perte**, **maintien** ou **gain de poids**.

---

## **Auteurs (Groupe de 3)**
- **Mathys FRANCO**  
- **Jobelin KOM**  
- **Cameron NOUPOUE**

---

## **Objectif du projet**
Développer une application capable de :

- Calculer les besoins énergétiques (**BMR**, **TDEE**) selon le profil utilisateur  
- Générer un **plan journalier équilibré** (petit-déjeuner, déjeuner, dîner, collation)  
- Classifier les aliments en 4 rôles :  
  **Privilégier**, **Modération**, **Neutre**, **Éviter**  
- Recommander des **aliments similaires** via **KNN**  
- Expliquer les décisions grâce à un **arbre de décision interprétable**

---

## **Dataset**
- **Fusion de 2 datasets Kaggle** : nutrition + aliments courants  
- **≈ 1 200 aliments**
- **Colonnes principales** :

```bash
Food Category | Meal Type | Calories | Protein | Carbs | Fat | Fiber | Sugar | Water
```

---

## **Prétraitement (`data_preprocessing.py`)**
- Suppression des **valeurs manquantes** et des **calories ≤ 0**  
- Détection des **outliers (IQR)**  
- **Discrétisation** :
- `calorie_level`
- `protein_level`
- `fiber_level`  
- **Feature engineering** :
- `prot_ratio`, `carb_ratio`, `fat_ratio`
- `density_kcal_100g`
- `satiety_index`
- `balance_score`

---

## **Modèles de Machine Learning (NutriAI)**

| Modèle | Rôle | Performance |
|--------|------|-------------|
| **Random Forest Regressor** | Prédiction du `balance_score` | MAE: **0.0108** |
| **Random Forest Classifier** | Classification du rôle de l’aliment | **97% accuracy** |
| **KNN (StandardScaler)** | Recommandation de 5 aliments similaires | Distance **euclidienne** |
| **Decision Tree (max_depth=6)** | Interprétation des règles de classification | **Arbre explicatif** |

>  **Hyperparamètres** optimisés via `GridSearchCV`  
> Modèles sauvegardés dans `/models/*.pkl`

---

## **Calculs personnalisés (`user_profile.py`)**

- **BMR** : Formule de **Harris-Benedict**  
- **TDEE** : Multiplicateur selon le **niveau d’activité**  
- **Macros cibles** :

| Objectif | Ajustement kcal | Protéines |
|-----------|----------------|------------|
| Perte | -500 kcal | 2.2 g/kg |
| Gain | +500 kcal | 1.8 g/kg |
| Maintien | ±0 kcal | 2.0 g/kg |

**Répartition des repas :**  
**Petit-déjeuner : 25 % | Déjeuner : 35 % | Dîner : 30 % | Collation : 10 %**

---

## **Planificateur de repas (`meal_planner.py`)**
- Sélectionne **3 aliments par repas**
- Respecte les **catégories** (Meat, Vegetables, etc.)
- Évite les **doublons**
- Minimise l’**écart avec les macros cibles**

---

## **Interface utilisateur (Streamlit – `app.py`)**

### 🧭 5 onglets interactifs :
1. **Recommandations**  
 → 5 aliments à *privilégier* ou *éviter* selon l’objectif  
 → Score d’équilibre + explication des choix  

2. **Repas**  
 → Plan complet avec macros réelles vs cibles  
 → Graphique comparatif (barres)  

3. **Analyse Aliment**  
 → Recherche par nom  
 → Rôle, ratios, densité, satiété  
 → 4 aliments similaires  

4. **Modèles & Performances**  
 → MAE, accuracy, précision par classe  
 → Hyperparamètres optimaux  

5. **Arbre Explicatif**  
 → Règles textuelles + visualisation graphique  
 → Légende claire des rôles

---

## **Structure du projet**

```bash
NutriAI/
├── app.py # Interface Streamlit
├── main.py # Entraînement des modèles
├── models/ # .pkl (RF, KNN, Tree, Scaler)
├── data/
│ ├── food_data.csv # Dataset brut
│ └── processed_nutrition.csv
├── user_profile.py # BMR, TDEE, macros
├── meal_planner.py # Générateur de repas
├── models.py # Classe NutriAI
├── data_preprocessing.py # Nettoyage + features
└── README.md
```
---

## **Installation & Lancement**

```bash
git clone https://github.com/votre-repo/NutriAI.git
cd NutriAI
python -m venv .venv
source .venv/bin/activate    # ou .venv\Scripts\activate sous Windows
pip install -r requirements.txt

# Entraînement des modèles
python main.py

# Lancement de l’application
streamlit run app.py
``` 

## **Défis rencontrés & solutions**

| Problème | Solution |
|------------|------------|
| `:success` non affiché | Utilisation de `st.success()` et `st.error()` |
| Arbre trop profond | Limitation avec `max_depth=6`, `min_samples_leaf=5` |
| Recommandations redondantes | Utilisation d’une liste `used_foods` pour assurer la diversité |
| Données bruitées | Nettoyage via **IQR** + **discrétisation** |

---

## **Fonctionnalités futures**
- **Export PDF** du plan  
- **Historique utilisateur**  
- **Recherche avancée** (filtre par rôle)  
- **Intégration API USDA** en temps réel  

---

## **Outils utilisés**
**Python** | **Pandas** | **Scikit-learn** | **Streamlit** | **Matplotlib** | **Joblib**

---

## **Liens GitHub**
- [Mathys FRANCO](https://github.com/mathysfranco)  
- [Jobelin KOM](https://github.com/jobelinkom)  
- [Cameron NOUPOUE](https://github.com/cameronnoupoue)

---

## **LinkedIn**
- [Mathys FRANCO](https://linkedin.com/in/mathysfranco)  
- [Jobelin KOM](https://linkedin.com/in/jobelinkom)  
- [Cameron NOUPOUE](https://linkedin.com/in/cameronnoupoue)
