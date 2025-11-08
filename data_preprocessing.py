# data_preprocessing.py
import pandas as pd
import numpy as np


def load_and_clean(filepath="data/food_data.csv"):
    df = pd.read_csv(filepath)

    # Nettoyage
    df = df.dropna().copy()
    df = df[df['Calories'] > 0]

    # Conversion
    numeric_cols = ['Calories', 'Protein', 'Carbs', 'Fat', 'Saturated Fat', 'Fiber', 'Sugar', 'Sodium', 'Water']
    df[numeric_cols] = df[numeric_cols].astype(float)

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


if __name__ == "__main__":
    df = load_and_clean()
    df.to_csv("data/processed_nutrition.csv", index=False)
    print(f"Dataset traité : {len(df)} lignes, {len(df.columns)} colonnes")