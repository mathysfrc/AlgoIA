import os
import pandas as pd
from preprocessing.add_noise import add_noise_to_dataset

if __name__ == "__main__":

    # Vérifier que le dataset existe
    input_file = "data/processed_nutrition.csv"
    if not os.path.exists(input_file):
        print(f"\n Fichier non trouvé: {input_file}")
        exit(1)
    
    # Paramètres par défaut
    noise_level = 0.05      # 5% de bruit sur les valeurs nutritionnelles
    label_noise = 0.03      # 3% de labels changés
    missing_prob = 0.01     # 1% de valeurs manquantes

    # Ajouter le bruit
    df_noisy = add_noise_to_dataset(
        input_path=input_file,
        output_path="data/processed_nutrition_noisy.csv",
        noise_level=noise_level,
        label_noise_prob=label_noise,
        missing_prob=missing_prob,
        random_seed=42
    )
    
    if df_noisy is not None:
        print(f"{'='*60}")
        print(f"\nPour utiliser le dataset bruité:")
        print(f"  1. Modifiez main.py ligne 79:")
        print(f"     ai = NutriAI(data_path='data/processed_nutrition_noisy.csv')")
        print(f"\n  2. Ou modifiez main.py ligne 64:")
        print(f"     df = pd.read_csv('data/processed_nutrition_noisy.csv')")
        print(f"\n  3. Relancez: python main.py")















