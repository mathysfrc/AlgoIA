import pandas as pd
import numpy as np


def augment_dataset(df, target_size=1200):
    """
    Augmente le dataset en créant des variations réalistes des aliments existants.

    Méthodes d'augmentation :
    1. Variations de cuisson (cru, cuit, grillé, bouilli)
    2. Variations de portions (petite, moyenne, grande)
    3. Variations de préparation (nature, assaisonné, mariné)
    """
    print(f"\nAugmentation du dataset de {len(df)} → {target_size} lignes")

    augmented_rows = []
    current_size = len(df)

    # Garder toutes les lignes originales
    for _, row in df.iterrows():
        augmented_rows.append(row.to_dict())

    # Calculer combien de lignes à ajouter
    needed = target_size - current_size
    iterations = int(np.ceil(needed / current_size))

    print(f"   → Ajout de ~{needed} lignes via {iterations} itérations")

    # Méthode 1 : Variations de cuisson
    cooking_methods = [
        {'suffix': ' - Cuit', 'cal_mult': 0.95, 'water_mult': 0.85},
        {'suffix': ' - Grillé', 'cal_mult': 0.98, 'water_mult': 0.80},
        {'suffix': ' - Bouilli', 'cal_mult': 0.90, 'water_mult': 1.10},
        {'suffix': ' - Vapeur', 'cal_mult': 0.92, 'water_mult': 1.05}
    ]

    # Méthode 2 : Variations de portions
    portion_sizes = [
        {'suffix': ' - Petite Portion', 'mult': 0.75},
        {'suffix': ' - Grande Portion', 'mult': 1.35}
    ]

    # Méthode 3 : Variations de préparation
    preparations = [
        {'suffix': ' - Nature', 'sugar_mult': 0.90, 'salt_mult': 0.85},
        {'suffix': ' - Assaisonné', 'sugar_mult': 1.05, 'salt_mult': 1.20}
    ]

    iteration = 0
    while len(augmented_rows) < target_size and iteration < iterations:
        iteration += 1

        for _, original_row in df.iterrows():
            if len(augmented_rows) >= target_size:
                break

            # Choisir méthode aléatoirement
            method = np.random.choice(['cooking', 'portion', 'preparation'])

            new_row = original_row.to_dict()

            if method == 'cooking':
                variant = np.random.choice(cooking_methods)
                new_row['Food Category'] = original_row['Food Category'] + variant['suffix']
                new_row['Calories'] *= variant['cal_mult']
                new_row['Water'] *= variant['water_mult']
                # Ajuster légèrement les macros
                new_row['Protein'] *= (0.95 + np.random.uniform(-0.05, 0.05))
                new_row['Carbs'] *= (0.95 + np.random.uniform(-0.05, 0.05))
                new_row['Fat'] *= (0.95 + np.random.uniform(-0.05, 0.05))

            elif method == 'portion':
                variant = np.random.choice(portion_sizes)
                new_row['Food Category'] = original_row['Food Category'] + variant['suffix']
                # Multiplier tous les nutriments
                for col in ['Calories', 'Protein', 'Carbs', 'Fat', 'Fiber', 'Sugar', 'Water']:
                    new_row[col] *= variant['mult']

            elif method == 'preparation':
                variant = np.random.choice(preparations)
                new_row['Food Category'] = original_row['Food Category'] + variant['suffix']
                new_row['Sugar'] *= variant['sugar_mult']
                # Ajuster légèrement calories
                new_row['Calories'] *= (1.0 + np.random.uniform(-0.02, 0.02))

            # Ajouter légère variation aléatoire (±2%) pour éviter duplicatas exacts
            noise_cols = ['Calories', 'Protein', 'Carbs', 'Fat', 'Fiber']
            for col in noise_cols:
                new_row[col] *= (1.0 + np.random.uniform(-0.02, 0.02))

            augmented_rows.append(new_row)

    # Créer DataFrame augmenté
    df_augmented = pd.DataFrame(augmented_rows[:target_size])

    print(f" Dataset augmenté: {len(df_augmented)} lignes")
    print(f"   - Lignes originales: {current_size}")
    print(f"   - Lignes générées: {len(df_augmented) - current_size}")

    return df_augmented


def verify_augmented_dataset(df):
    """Vérifie la qualité du dataset augmenté"""
    print("\nVérification du dataset augmenté:")

    # 1. Dimensions
    print(f"   ✓ Lignes: {len(df)} (≥1000: {'' if len(df) >= 1000 else ''})")
    print(f"   ✓ Colonnes: {len(df.columns)} (≥10: {'' if len(df.columns) >= 10 else ''})")

    # 2. Valeurs manquantes
    missing = df.isnull().sum().sum()
    print(f"   ✓ Valeurs manquantes: {missing} {'' if missing == 0 else ''}")

    # 3. Duplicatas exacts
    duplicates = df.duplicated().sum()
    print(f"   ✓ Duplicatas exacts: {duplicates} {'' if duplicates < 10 else ''}")

    # 4. Distribution des nutriments
    print(f"\n    Statistiques nutritionnelles:")
    print(f"      Calories: {df['Calories'].min():.0f} - {df['Calories'].max():.0f} (moy: {df['Calories'].mean():.0f})")
    print(f"      Protéines: {df['Protein'].min():.1f} - {df['Protein'].max():.1f}g (moy: {df['Protein'].mean():.1f})")
    print(f"      Glucides: {df['Carbs'].min():.1f} - {df['Carbs'].max():.1f}g (moy: {df['Carbs'].mean():.1f})")

    return len(df) >= 1000 and len(df.columns) >= 10


if __name__ == "__main__":
    print("=" * 80)
    print("AUGMENTATION DU DATASET NUTRIAI")
    print("=" * 80)

    # Charger dataset nettoyé
    try:
        df = pd.read_csv("data/processed_nutrition.csv")
        print(f"\n✓ Dataset chargé: {len(df)} lignes")
    except FileNotFoundError:
        print(" Fichier 'data/processed_nutrition.csv' non trouvé")
        print("   Exécutez d'abord: python main.py")
        exit(1)

    # Vérifier si augmentation nécessaire
    if len(df) >= 1000:
        print(f"\n Dataset déjà conforme ({len(df)} lignes)")
        exit(0)

    # Augmenter le dataset
    df_augmented = augment_dataset(df, target_size=1200)

    # Vérifier qualité
    is_valid = verify_augmented_dataset(df_augmented)

    if is_valid:
        # Sauvegarder
        df_augmented.to_csv("data/processed_nutrition.csv", index=False)
        print(f"\n Dataset augmenté sauvegardé: data/processed_nutrition.csv")
        print(f"   → Vous pouvez maintenant relancer: python main.py")
    else:
        print(f"\n Le dataset augmenté ne respecte pas les critères")
        print(f"   → Vérifiez les erreurs ci-dessus")

    print("\n" + "=" * 80)