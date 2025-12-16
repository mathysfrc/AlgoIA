import pandas as pd
import numpy as np

FEATURES = [
    'Calories', 'Protein', 'Carbs', 'Fat',
    'Fiber', 'Sugar', 'Water',
    'prot_ratio', 'carb_ratio', 'fat_ratio',
    'fiber_per_100kcal', 'sugar_per_100kcal',
    'density_kcal_100g', 'satiety_index',
    'balance_score'
]

# Remove des outliers avec la méthode IQR  => 3ème quantile (75%) - 1er quantile (25%)
def remove_outliers(df, columns, factor=1.5):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        # Défini le min et max
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        # Conservation uniquement des valeurs comprises entre les bornes définies
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

# Discretisation
def discretize_nutrients(df):
    # Entre 0 et 100 = Faible, 100 et 300 = Moyen, +300 = Elevé
    df['calorie_level'] = pd.cut(df['Calories'], bins=[0, 100, 300, 1000], labels=['Faible', 'Moyen', 'Élevé'])
    # Entre 0 et 5 = Faible, 5 et 15 = Moyen, +15 = Elevé
    df['protein_level'] = pd.cut(df['Protein'], bins=[0, 5, 15, 100], labels=['Faible', 'Moyen', 'Élevé'])
    # Entre 0 et 2 = Faible, 2 et 5 = Moyen, +5 = Elevé
    df['fiber_level'] = pd.cut(df['Fiber'], bins=[0, 2, 5, 20], labels=['Faible', 'Moyen', 'Élevé'])
    return df

def load_and_clean(filepath="data/final_food.csv"):
    df = pd.read_csv(filepath)
    # Supprime les lignes ayant un NaN
    df = df.dropna().copy()
    # Conserve uniquement les lignes ou Calories > 0
    df = df[df['Calories'] > 0]

    numeric_cols = ['Calories', 'Protein', 'Carbs', 'Fat', 'Fiber', 'Sugar', 'Water']
    # Converti en float, mieux pour les calculs
    df[numeric_cols] = df[numeric_cols].astype(float)

    # Pré-processing
    # On applique les deux méthodes précédentes
    df = remove_outliers(df, numeric_cols)
    df = discretize_nutrients(df)

    # Feature Engineering
    # On estime des données sur base de calculs prouvés
    df['prot_ratio'] = df['Protein'] * 4 / df['Calories']
    df['carb_ratio'] = df['Carbs'] * 4 / df['Calories']
    df['fat_ratio'] = df['Fat'] * 9 / df['Calories']
    df['fiber_per_100kcal'] = df['Fiber'] / (df['Calories'] / 100)
    df['sugar_per_100kcal'] = df['Sugar'] / (df['Calories'] / 100)
    df['density_kcal_100g'] = df['Calories'] / 100

    # Indice de satiété (capacité d'un aliment à rassasier)
    df['satiety_index'] = (
        df['Protein'] * 1.0 +
        df['Fiber'] * 0.8 +
        df['Water'] * 0.05 -
        df['Sugar'] * 0.3
    ) / (df['Calories'] / 100 + 1)

    # Score d'équilibre
    df['balance_score'] = (
        0.35 * df['prot_ratio'] +
        0.25 * (1 - df['carb_ratio']) +
        0.20 * (1 - df['fat_ratio']) +
        0.15 * df['fiber_per_100kcal'] / 10 +
        0.05 * (1 - df['sugar_per_100kcal'] / 50)
    )
    # Normalisation min-max => 0 = moins équilibré, 1 = équilibré
    df['balance_score'] = (df['balance_score'] - df['balance_score'].min()) / (
        df['balance_score'].max() - df['balance_score'].min())

    return df

def enrich_with_profiles(df):
    # Création de profils génériques
    profiles = [
        {"profil": "sedentaire", "activity_factor": 1.2, "objective": "perte"},
        {"profil": "modere", "activity_factor": 1.55, "objective": "maintien"},
        {"profil": "intense", "activity_factor": 1.725, "objective": "gain"},
        {"profil": "athlete", "activity_factor": 1.9, "objective": "gain"},
    ]

    # On créé 4 versions de chaque élément basé sur un profil différent
    # Par exemple, avocat : bon pour prise de masse mais modéré pour perte
    augmented_rows = []
    for _, row in df.iterrows():
        # 4 fois car 4 profiles
        for p in profiles:
            r = row.copy()
            r["profil"] = p["profil"]
            r["activity_factor"] = p["activity_factor"]
            r["objective"] = p["objective"]
            augmented_rows.append(r)

    df_aug = pd.DataFrame(augmented_rows)
    return df_aug


def add_noise_to_nutritional_data(df, noise_level=0.05, random_seed=42):
    """
    Ajoute du bruit gaussien aux valeurs nutritionnelles pour rendre les résultats plus réalistes
    """
    np.random.seed(random_seed)
    df_noisy = df.copy()
    
    # Colonnes nutritionnelles à perturber
    nutritional_cols = ['Calories', 'Protein', 'Carbs', 'Fat', 'Fiber', 'Sugar', 'Water']
    
    for col in nutritional_cols:
        if col in df_noisy.columns:
            # Bruit gaussien avec écart-type = noise_level * valeur
            noise = np.random.normal(0, noise_level, size=len(df_noisy))
            df_noisy[col] = df_noisy[col] * (1 + noise)
            
            # S'assurer que les valeurs restent positives
            df_noisy[col] = np.maximum(df_noisy[col], 0)
    
    # Recalculer les features dérivées
    if 'Calories' in df_noisy.columns and 'Protein' in df_noisy.columns:
        df_noisy['prot_ratio'] = df_noisy['Protein'] * 4 / (df_noisy['Calories'] + 1e-6)
        df_noisy['carb_ratio'] = df_noisy['Carbs'] * 4 / (df_noisy['Calories'] + 1e-6)
        df_noisy['fat_ratio'] = df_noisy['Fat'] * 9 / (df_noisy['Calories'] + 1e-6)
        
        if 'Fiber' in df_noisy.columns:
            df_noisy['fiber_per_100kcal'] = df_noisy['Fiber'] / ((df_noisy['Calories'] / 100) + 1e-6)
        if 'Sugar' in df_noisy.columns:
            df_noisy['sugar_per_100kcal'] = df_noisy['Sugar'] / ((df_noisy['Calories'] / 100) + 1e-6)
        
        df_noisy['density_kcal_100g'] = df_noisy['Calories'] / 100
        
        if all(col in df_noisy.columns for col in ['Protein', 'Fiber', 'Water', 'Sugar', 'Calories']):
            df_noisy['satiety_index'] = (
                df_noisy['Protein'] * 1.0 +
                df_noisy['Fiber'] * 0.8 +
                df_noisy['Water'] * 0.05 -
                df_noisy['Sugar'] * 0.3
            ) / ((df_noisy['Calories'] / 100) + 1)
    
    return df_noisy