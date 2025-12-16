import pandas as pd
import numpy as np
import os


def add_nutritional_noise(df, noise_level=0.05, random_seed=42):
    """
    Ajoute du bruit gaussien aux valeurs nutritionnelles
    """
    np.random.seed(random_seed)
    
    df_noisy = df.copy()
    
    # Colonnes nutritionnelles à perturber
    nutritional_cols = ['Calories', 'Protein', 'Carbs', 'Fat', 'Fiber', 'Sugar', 'Water']
    
    # Colonnes qui doivent rester positives
    positive_cols = ['Calories', 'Protein', 'Carbs', 'Fat', 'Fiber', 'Sugar', 'Water']


    # Statistiques avant
    stats_before = {}
    for col in nutritional_cols:
        if col in df_noisy.columns:
            stats_before[col] = {
                'mean': df_noisy[col].mean(),
                'std': df_noisy[col].std(),
                'min': df_noisy[col].min(),
                'max': df_noisy[col].max()
            }
    
    # Ajouter bruit gaussien
    for col in nutritional_cols:
        if col in df_noisy.columns:
            # Bruit gaussien avec écart-type = noise_level * valeur
            noise = np.random.normal(0, noise_level, size=len(df_noisy))
            df_noisy[col] = df_noisy[col] * (1 + noise)
            
            # S'assurer que les valeurs restent positives
            if col in positive_cols:
                df_noisy[col] = np.maximum(df_noisy[col], 0)

    for col in nutritional_cols:
        if col in df_noisy.columns:
            mean_before = stats_before[col]['mean']
            mean_after = df_noisy[col].mean()
            diff = abs(mean_after - mean_before) / mean_before * 100
            print(f"{col:<15} {mean_before:>12.2f} {mean_after:>15.2f} {diff:>8.2f}%")
    
    return df_noisy


def add_label_noise(df, noise_probability=0.05, random_seed=42):
    """
    Ajoute du bruit aux labels (change aléatoirement quelques labels)
    Simule des erreurs de classification ou des cas ambigus
    """
    if 'role' not in df.columns:
        return df
    
    np.random.seed(random_seed)
    df_noisy = df.copy()
    
    # Classes possibles
    classes = df['role'].unique()
    
    # Nombre de labels à changer
    n_to_change = int(len(df) * noise_probability)
    
    # Indices aléatoires à changer
    indices_to_change = np.random.choice(df.index, size=n_to_change, replace=False)

    # Distribution avant
    print(f"\nDistribution AVANT:")
    for cls in classes:
        count = (df['role'] == cls).sum()
        pct = count / len(df) * 100
        print(f"  {cls}: {count} ({pct:.1f}%)")
    
    # Changer les labels
    for idx in indices_to_change:
        original_label = df.loc[idx, 'role']
        # Choisir une classe différente aléatoirement
        other_classes = [c for c in classes if c != original_label]
        if other_classes:
            new_label = np.random.choice(other_classes)
            df_noisy.loc[idx, 'role'] = new_label
    
    # Distribution après
    print(f"\nDistribution APRÈS:")
    for cls in classes:
        count = (df_noisy['role'] == cls).sum()
        pct = count / len(df_noisy) * 100
        print(f"  {cls}: {count} ({pct:.1f}%)")
    
    return df_noisy


def add_missing_values_noise(df, missing_probability=0.02, random_seed=42):
    """
    Ajoute des valeurs manquantes aléatoirement (simule des données incomplètes)
    """
    np.random.seed(random_seed)
    df_noisy = df.copy()
    
    # Colonnes numériques à perturber
    numeric_cols = ['Calories', 'Protein', 'Carbs', 'Fat', 'Fiber', 'Sugar', 'Water']
    
    n_missing = 0
    for col in numeric_cols:
        if col in df_noisy.columns:
            # Indices à mettre à NaN
            n_col_missing = int(len(df_noisy) * missing_probability)
            indices = np.random.choice(df_noisy.index, size=n_col_missing, replace=False)
            df_noisy.loc[indices, col] = np.nan
            n_missing += n_col_missing
    
    return df_noisy


def add_noise_to_dataset(input_path="data/processed_nutrition.csv",
                         output_path="data/processed_nutrition_noisy.csv",
                         noise_level=0.05,
                         label_noise_prob=0.03,
                         missing_prob=0.01,
                         random_seed=42):
    """
    Fonction principale pour ajouter du bruit au dataset complet
    """

    # Charger le dataset
    if not os.path.exists(input_path):
        print(f"❌ Fichier non trouvé: {input_path}")
        return None
    
    df = pd.read_csv(input_path)
    print(f"\n✓ Dataset chargé: {len(df)} lignes, {len(df.columns)} colonnes")
    
    # 1. Ajouter bruit aux valeurs nutritionnelles
    df = add_nutritional_noise(df, noise_level=noise_level, random_seed=random_seed)
    
    # 2. Ajouter bruit aux labels (si colonne 'role' existe)
    if 'role' in df.columns:
        df = add_label_noise(df, noise_probability=label_noise_prob, random_seed=random_seed+1)
    
    # 3. Ajouter valeurs manquantes
    df = add_missing_values_noise(df, missing_probability=missing_prob, random_seed=random_seed+2)
    
    # 4. Recalculer les features dérivées (car les valeurs de base ont changé)
    if 'Calories' in df.columns and 'Protein' in df.columns:
        # Recalculer les ratios
        df['prot_ratio'] = df['Protein'] * 4 / (df['Calories'] + 1e-6)  # +1e-6 pour éviter division par 0
        df['carb_ratio'] = df['Carbs'] * 4 / (df['Calories'] + 1e-6)
        df['fat_ratio'] = df['Fat'] * 9 / (df['Calories'] + 1e-6)
        
        if 'Fiber' in df.columns:
            df['fiber_per_100kcal'] = df['Fiber'] / ((df['Calories'] / 100) + 1e-6)
        if 'Sugar' in df.columns:
            df['sugar_per_100kcal'] = df['Sugar'] / ((df['Calories'] / 100) + 1e-6)
        
        df['density_kcal_100g'] = df['Calories'] / 100
        
        # Recalculer satiety_index
        if all(col in df.columns for col in ['Protein', 'Fiber', 'Water', 'Sugar', 'Calories']):
            df['satiety_index'] = (
                df['Protein'] * 1.0 +
                df['Fiber'] * 0.8 +
                df['Water'] * 0.05 -
                df['Sugar'] * 0.3
            ) / ((df['Calories'] / 100) + 1)
    
    # 5. Nettoyer les valeurs infinies ou invalides
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # 6. Sauvegarder
    df.to_csv(output_path, index=False)
    
    print(f"  ai = NutriAI(data_path='{output_path}')")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ajouter du bruit au dataset nutritionnel")
    parser.add_argument("--input", default="data/processed_nutrition.csv",
                       help="Chemin vers le dataset d'entrée")
    parser.add_argument("--output", default="data/processed_nutrition_noisy.csv",
                       help="Chemin vers le dataset de sortie")
    parser.add_argument("--noise-level", type=float, default=0.05,
                       help="Niveau de bruit sur valeurs nutritionnelles (0.05 = 5%%)")
    parser.add_argument("--label-noise", type=float, default=0.03,
                       help="Probabilité de changer un label (0.03 = 3%%)")
    parser.add_argument("--missing", type=float, default=0.01,
                       help="Probabilité de valeurs manquantes (0.01 = 1%%)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Seed pour reproductibilité")
    
    args = parser.parse_args()
    
    add_noise_to_dataset(
        input_path=args.input,
        output_path=args.output,
        noise_level=args.noise_level,
        label_noise_prob=args.label_noise,
        missing_prob=args.missing,
        random_seed=args.seed
    )










