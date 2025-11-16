# main.py - ENTRAÎNEMENT DU SYSTÈME PERSONNALISÉ AVEC ÉVALUATION
import os

import pandas as pd

from data_preprocessing import load_and_clean
from models import NutriAI
from model_evaluation import run_full_evaluation


def train_for_profile(ai, weight, height, age, gender, activity, objective, profile_name):
    """Entraîne les modèles pour un profil spécifique"""
    print(f"\n{'=' * 60}")
    print(f"ENTRAÎNEMENT POUR PROFIL: {profile_name}")
    print(f"{'=' * 60}")

    # Configuration profil
    profile = ai.set_user_profile(
        weight=weight,
        height=height,
        age=age,
        gender=gender,
        activity=activity,
        objective=objective
    )

    # Classification personnalisée
    print("\n1. Classification des rôles alimentaires...")
    ai.classify_food_role_personalized()

    # KNN personnalisé
    print("\n2. Entraînement KNN personnalisé...")
    ai.train_knn_personalized()

    # Arbre de décision personnalisé
    print("\n3. Entraînement de l'arbre de décision...")
    rules = ai.train_decision_tree_personalized()

    print("\n✅ Entraînement terminé pour ce profil!")
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

    # 1. Nettoyage du dataset
    print("\n📊 Chargement et nettoyage du dataset...")
    if os.path.exists("data/processed_nutrition.csv"):
        print("📂 Dataset déjà traité trouvé, utilisation sans écraser")
        df = pd.read_csv("data/processed_nutrition.csv")
    else:
        print("🧹 Aucun dataset traité, création…")
        df = load_and_clean("data/food_data.csv")
        df.to_csv("data/processed_nutrition.csv", index=False)
    print(f"✓ Dataset nettoyé: {len(df)} lignes, {len(df.columns)} colonnes")

    # Vérification conformité
    if len(df) < 1000:
        print(f"⚠️  ATTENTION: Dataset < 1000 lignes ({len(df)} lignes)")
    if len(df.columns) < 10:
        print(f"⚠️  ATTENTION: Dataset < 10 colonnes ({len(df.columns)} colonnes)")

    # 2. Initialiser le système
    print("\n🤖 Initialisation du système NutriAI...")
    ai = NutriAI(data_path="data/processed_nutrition.csv")

    # 3. Entraîner pour différents profils types
    print("\n🎯 Entraînement pour profils types...")

    profiles_to_train = [
        # Profil 1: Homme sédentaire en perte de poids
        {
            'name': 'Homme Sédentaire Perte',
            'weight': 85,
            'height': 175,
            'age': 35,
            'gender': 'Homme',
            'activity': 'Sédentaire',
            'objective': 'perte'
        },
        # Profil 2: Femme modérée maintien
        {
            'name': 'Femme Modérée Maintien',
            'weight': 65,
            'height': 165,
            'age': 28,
            'gender': 'Femme',
            'activity': 'Modéré',
            'objective': 'maintien'
        },
        # Profil 3: Homme athlète gain
        {
            'name': 'Homme Athlète Gain',
            'weight': 75,
            'height': 180,
            'age': 25,
            'gender': 'Homme',
            'activity': 'Athlète',
            'objective': 'gain'
        },
        # Profil 4: Femme intense perte
        {
            'name': 'Femme Intense Perte',
            'weight': 70,
            'height': 170,
            'age': 30,
            'gender': 'Femme',
            'activity': 'Intense',
            'objective': 'perte'
        }
    ]

    results = []

    for profile_config in profiles_to_train:
        profile_config_copy = profile_config.copy()
        name = profile_config_copy.pop('name')
        result = train_for_profile(ai, **profile_config_copy, profile_name=name)
        results.append({
            'name': name,
            'config': profile_config_copy,
            'result': result
        })

    # 4. Résumé
    print("\n" + "=" * 60)
    print("📋 RÉSUMÉ DE L'ENTRAÎNEMENT")
    print("=" * 60)

    for r in results:
        print(f"\n{r['name']}:")
        print(f"  TDEE: {int(r['result']['tdee'])} kcal")
        print(f"  Objectif: {int(r['result']['target_calories'])} kcal")
        print(f"  Protéines: {int(r['result']['target_protein'])}g")

    # 5. ÉVALUATION COMPLÈTE DES MODÈLES
    print("\n" + "=" * 60)
    print("🧪 ÉVALUATION COMPLÈTE DES MODÈLES")
    print("=" * 60)

    # Utiliser le dernier profil pour l'évaluation
    last_profile = results[-1]['result']

    print("\n📊 Lancement de l'évaluation comparative...")
    print("   - Fine-tuning des hyperparamètres (GridSearchCV)")
    print("   - Validation croisée 5-fold")
    print("   - Test de 4 algorithmes différents")
    print("   - Génération des visualisations")
    print("\n⏳ Cela peut prendre 5-10 minutes...")

    try:
        evaluator, comparison_df = run_full_evaluation(ai.df, last_profile)

        print("\n" + "=" * 60)
        print("✅ ÉVALUATION TERMINÉE AVEC SUCCÈS")
        print("=" * 60)
        print("\n📊 Résultats de comparaison:")
        print(comparison_df.to_string(index=False))

    except Exception as e:
        print(f"\n❌ Erreur lors de l'évaluation: {e}")
        print("   L'entraînement de base a réussi, mais l'évaluation comparative a échoué.")

    # 6. Test de recommandations
    print("\n" + "=" * 60)
    print("🧪 TEST DES RECOMMANDATIONS")
    print("=" * 60)

    # Tester pour le dernier profil
    last_profile_name = results[-1]['name']
    print(f"\nProfil de test: {last_profile_name}")

    # Aliments privilégiés
    privileged = ai.df[ai.df['role'] == 'Privilégier'].head(5)
    print("\n✅ Top 5 aliments à PRIVILÉGIER:")
    for idx, (_, food) in enumerate(privileged.iterrows(), 1):
        print(f"  {idx}. {food['Food Category']} - Score: {food['user_score']:.2f}")

    # Aliments à éviter
    avoid = ai.df[ai.df['role'] == 'Éviter'].head(5)
    print("\n❌ Top 5 aliments à ÉVITER:")
    for idx, (_, food) in enumerate(avoid.iterrows(), 1):
        print(f"  {idx}. {food['Food Category']} - Score: {food['user_score']:.2f}")

    print("\n" + "=" * 60)
    print("✅ TOUT EST PRÊT!")
    print("=" * 60)
    print("\n🎉 Le système est 100% personnalisé et évalué scientifiquement.")
    print("\n📁 Fichiers générés:")
    print("   - data/processed_nutrition.csv (dataset nettoyé)")
    print("   - models/knn_personalized.pkl (modèle KNN)")
    print("   - models/decision_tree_personalized.pkl (arbre de décision)")
    print("   - models/model_comparison.csv (comparaison des modèles)")
    print("   - models/confusion_matrices.png (visualisations)")
    print("   - models/evaluation_report.txt (rapport détaillé)")
    print("\n🚀 Lancez l'application avec:")
    print("   streamlit run app.py")
    print("\n💡 Consultez l'onglet 'Performances & Comparaison' dans l'app!")
    print("=" * 60)