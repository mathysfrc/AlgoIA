import pandas as pd
import numpy as np

def remove_outliers(df, columns, factor=1.5):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

def discretize_nutrients(df):
    df['calorie_level'] = pd.cut(df['Calories'], bins=[0, 100, 300, 1000], labels=['Faible', 'Moyen', 'Élevé'])
    df['protein_level'] = pd.cut(df['Protein'], bins=[0, 5, 15, 100], labels=['Faible', 'Moyen', 'Élevé'])
    df['fiber_level'] = pd.cut(df['Fiber'], bins=[0, 2, 5, 20], labels=['Faible', 'Moyen', 'Élevé'])
    return df

def load_and_clean(filepath="data/food_data.csv"):
    df = pd.read_csv(filepath)
    df = df.dropna().copy()
    df = df[df['Calories'] > 0]

    numeric_cols = ['Calories', 'Protein', 'Carbs', 'Fat', 'Fiber', 'Sugar', 'Water']
    df[numeric_cols] = df[numeric_cols].astype(float)

    # PRÉ-PROCESSING
    df = remove_outliers(df, numeric_cols)
    df = discretize_nutrients(df)

    # Feature Engineering
    df['prot_ratio'] = df['Protein'] * 4 / df['Calories']
    df['carb_ratio'] = df['Carbs'] * 4 / df['Calories']
    df['fat_ratio'] = df['Fat'] * 9 / df['Calories']
    df['fiber_per_100kcal'] = df['Fiber'] / (df['Calories'] / 100)
    df['sugar_per_100kcal'] = df['Sugar'] / (df['Calories'] / 100)
    df['density_kcal_100g'] = df['Calories'] / 100

    # Indice de satiété
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
    df['balance_score'] = (df['balance_score'] - df['balance_score'].min()) / (
        df['balance_score'].max() - df['balance_score'].min())

    return df