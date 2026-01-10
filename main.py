import os

import pandas as pd

from models import NutriAI
from model_evaluation import run_full_evaluation
from preprocessing.data_preprocessing import load_and_clean


def train_for_profile(ai, weight, height, age, gender, activity, objective, profile_name):
    """Entraîne les modèles pour un profil spécifique"""
    print(f"\n{'=' * 60}")
    print(f"ENTRAÎNEMENT POUR PROFIL: {profile_name}")
    print(f"{'=' * 60}")

    # Configuration profil => models.py, on lui passe les datas
    profile = ai.set_user_profile(
        weight=weight,
        height=height,
        age=age,
        gender=gender,
        activity=activity,
        objective=objective
    )

    # Classification => models.py (doit être fait en premier pour créer la colonne 'role')
    print("\n1. Classification des rôles alimentaires (règles conditionnelles)...")
    ai.classify_food_role_personalized(use_tree=False)

    # Arbre de décision => models.py (s'entraîne sur les rôles créés par la classification)
    print("\n2. Entraînement de l'arbre de décision...")
    rules = ai.train_decision_tree_personalized()

    # KNN => models.py
    print("\n3. Entraînement KNN personnalisé...")
    ai.train_knn_personalized()

    print("\nEntraînement terminé pour ce profil!")
    print(f"   TDEE: {int(profile['tdee'])} kcal")
    print(f"   Objectif: {int(profile['target_calories'])} kcal")
    print(f"   Macros: P:{int(profile['target_protein'])}g "
          f"G:{int(profile['target_carbs'])}g "
          f"L:{int(profile['target_fat'])}g")

    # Sauvegarder
    ai.save_all_models()

    return profile


if __name__ == "__main__":
    print("=" * 60)
    print("NUTRIAI - SYSTÈME DE RECOMMANDATION PERSONNALISÉ")
    print("=" * 60)

    # Créer dossiers
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # 1. Nettoyage du dataset => models.py
    print("\nChargement et nettoyage du dataset...")
    if os.path.exists("data/processed_nutrition_noisy.csv"):
        print("Dataset déjà traité trouvé, utilisation sans écraser")
        df = pd.read_csv('data/processed_nutrition_noisy.csv')
    else:
        print("Aucun dataset traité, création…")
        # On appel final food et on traite avec load and clean => résultat : processed_nutrition.csv
        df = load_and_clean("data/final_food.csv")
        df.to_csv("data/processed_nutrition.csv", index=False)
    print(f"✓ Dataset nettoyé: {len(df)} lignes, {len(df.columns)} colonnes")

    # Vérification conformité
    if len(df) < 1000:
        print(f"ATTENTION: Dataset < 1000 lignes ({len(df)} lignes)")
    if len(df.columns) < 10:
        print(f"ATTENTION: Dataset < 10 colonnes ({len(df.columns)} colonnes)")

    # 2. Initialiser le système
    print("\nInitialisation du système NutriAI...")
    ai = NutriAI(data_path='data/processed_nutrition_noisy.csv')

    print("\nEntraînement pour TOUS les profils Homme / Femme...")

    # Profils de référence générique
    genders = {
        "Homme": {
            "weight": 80,
            "height": 178,
            "age": 30
        },
        "Femme": {
            "weight": 65,
            "height": 165,
            "age": 30
        }
    }

    activities = ["Sédentaire", "Léger", "Modéré", "Intense", "Athlète"]
    objectives = {
        "perte": "Perte",
        "maintien": "Maintien",
        "gain": "Gain"
    }

    results = []

    for gender, base in genders.items():
        for activity in activities:
            for objective_key, objective_label in objectives.items():
                profile_name = f"{gender} {activity} {objective_label}"

                print(f"\n🚀 Lancement profil: {profile_name}")

                # Sur base des profils fournis
                result = train_for_profile(
                    ai=ai,
                    weight=base["weight"],
                    height=base["height"],
                    age=base["age"],
                    gender=gender,
                    activity=activity,
                    objective=objective_key,
                    profile_name=profile_name
                )

                results.append({
                    "name": profile_name,
                    "gender": gender,
                    "activity": activity,
                    "objective": objective_key,
                    "result": result
                })

    # 4. Résumé
    print("\n" + "=" * 60)
    print("RÉSUMÉ DE L'ENTRAÎNEMENT")
    print("=" * 60)

    for r in results:
        print(f"\n{r['name']}:")
        print(f"  TDEE: {int(r['result']['tdee'])} kcal")
        print(f"  Objectif: {int(r['result']['target_calories'])} kcal")
        print(f"  Protéines: {int(r['result']['target_protein'])}g")

    # Utiliser le dernier profil pour l'évaluation
    last_profile = results[-1]['result']

    print("\nLancement de l'évaluation comparative...")
    print("   - Fine-tuning des hyperparamètres (GridSearchCV)")
    print("   - Validation croisée 5-fold")
    print("   - Test d'algorithmes différents")
    print("   - Génération des visualisations")

    try:
        evaluator, comparison_df = run_full_evaluation(ai.df, last_profile)

        print("\n" + "=" * 60)
        print(" Évaluation terminée")
        print("=" * 60)
        print("\n Résultats de comparaison:")
        print(comparison_df.to_string(index=False))

    except Exception as e:
        print(f"\n Erreur lors de l'évaluation: {e}")
        print("   L'entraînement de base a réussi, mais l'évaluation comparative a échoué.")


    # Tester pour le dernier profil
    last_profile_name = results[-1]['name']
    print(f"\nProfil de test: {last_profile_name}")

    # Aliments privilégiés
    privileged = ai.df[ai.df['role'] == 'Privilégier'].head(5)
    print("\n Top 5 aliments à privilégier:")
    for idx, (_, food) in enumerate(privileged.iterrows(), 1):
        print(f"  {idx}. {food['Food Category']} - Score: {food['user_score']:.2f}")

    # Aliments à éviter
    avoid = ai.df[ai.df['role'] == 'Éviter'].head(5)
    print("\n Top 5 aliments à éviter:")
    for idx, (_, food) in enumerate(avoid.iterrows(), 1):
        print(f"  {idx}. {food['Food Category']} - Score: {food['user_score']:.2f}")

    print("\n" + "=" * 60)
    print("=" * 60)
    print("   streamlit run app.py")
    print("=" * 60)