# diagnostic.py - Diagnostique complet du projet NutriAI
import os
import pandas as pd
import sys


def check_files():
    """Vérifie la présence de tous les fichiers nécessaires"""
    print("\n" + "=" * 80)
    print("📁 VÉRIFICATION DES FICHIERS")
    print("=" * 80)

    required_files = {
        'Scripts Python': [
            'data_preprocessing.py',
            'user_profile.py',
            'models.py',
            'model_evaluation.py',
            'meal_planner.py',
            'main.py',
            'app.py'
        ],
        'Configuration': [
            'requirements.txt'
        ],
        'Data': [
            'data/food_data.csv'
        ]
    }

    all_present = True

    for category, files in required_files.items():
        print(f"\n{category}:")
        for file in files:
            exists = os.path.exists(file)
            status = "✅" if exists else "❌"
            print(f"  {status} {file}")
            if not exists:
                all_present = False

    return all_present


def check_dataset():
    """Vérifie le dataset"""
    print("\n" + "=" * 80)
    print("📊 VÉRIFICATION DU DATASET")
    print("=" * 80)

    # Dataset brut
    if os.path.exists("data/food_data.csv"):
        df_raw = pd.read_csv("data/food_data.csv")
        print(f"\n✓ Dataset brut (food_data.csv):")
        print(f"  - Lignes: {len(df_raw)} {'✅' if len(df_raw) >= 1000 else '⚠️ < 1000'}")
        print(f"  - Colonnes: {len(df_raw.columns)} {'✅' if len(df_raw.columns) >= 10 else '❌ < 10'}")
        print(f"  - Colonnes: {list(df_raw.columns)[:5]}...")

        # Vérifier colonnes requises
        required_cols = ['Food Category', 'Calories', 'Protein', 'Carbs', 'Fat', 'Fiber', 'Sugar', 'Water']
        missing_cols = [col for col in required_cols if col not in df_raw.columns]

        if missing_cols:
            print(f"\n  ❌ Colonnes manquantes: {missing_cols}")
            return False
        else:
            print(f"  ✅ Toutes les colonnes requises sont présentes")

        if len(df_raw) < 1000:
            print(f"\n  ⚠️  SOLUTION: Exécutez 'python data_augmentation.py' pour augmenter le dataset")
            return False
    else:
        print("\n❌ Fichier data/food_data.csv non trouvé")
        return False

    # Dataset nettoyé
    if os.path.exists("data/processed_nutrition.csv"):
        df_processed = pd.read_csv("data/processed_nutrition.csv")
        print(f"\n✓ Dataset nettoyé (processed_nutrition.csv):")
        print(f"  - Lignes: {len(df_processed)} {'✅' if len(df_processed) >= 1000 else '⚠️'}")
        print(f"  - Colonnes: {len(df_processed.columns)}")
    else:
        print(f"\n⚠️  Dataset nettoyé non trouvé (sera créé par main.py)")

    return len(df_raw) >= 1000


def check_dependencies():
    """Vérifie les dépendances Python"""
    print("\n" + "=" * 80)
    print("📦 VÉRIFICATION DES DÉPENDANCES")
    print("=" * 80)

    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'sklearn',
        'matplotlib',
        'seaborn',
        'joblib',
        'PIL'
    ]

    all_installed = True

    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (non installé)")
            all_installed = False

    if not all_installed:
        print(f"\n⚠️  SOLUTION: pip install -r requirements.txt")

    return all_installed


def check_models():
    """Vérifie les modèles entraînés"""
    print("\n" + "=" * 80)
    print("🤖 VÉRIFICATION DES MODÈLES")
    print("=" * 80)

    if not os.path.exists("models"):
        print("\n⚠️  Dossier 'models/' non trouvé")
        print("   SOLUTION: Exécutez 'python main.py' pour entraîner les modèles")
        return False

    model_files = [
        'knn_personalized.pkl',
        'decision_tree_personalized.pkl',
        'scaler_personalized.pkl',
        'label_encoder.pkl'
    ]

    all_present = True

    print("\nModèles de base:")
    for file in model_files:
        path = f"models/{file}"
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"  {status} {file}")
        if not exists:
            all_present = False

    # Modèles d'évaluation
    evaluation_files = [
        'model_comparison.csv',
        'confusion_matrices.png',
        'performance_comparison.png',
        'evaluation_report.txt'
    ]

    print("\nFichiers d'évaluation:")
    eval_present = True
    for file in evaluation_files:
        path = f"models/{file}"
        exists = os.path.exists(path)
        status = "✅" if exists else "⚠️"
        print(f"  {status} {file}")
        if not exists:
            eval_present = False

    if not all_present:
        print(f"\n⚠️  SOLUTION: Exécutez 'python main.py' pour entraîner les modèles")

    if not eval_present:
        print(f"\n⚠️  Évaluation non effectuée (sera faite par main.py)")

    return all_present


def check_streamlit():
    """Vérifie si Streamlit peut démarrer"""
    print("\n" + "=" * 80)
    print("🌐 VÉRIFICATION STREAMLIT")
    print("=" * 80)

    if not os.path.exists("app.py"):
        print("\n❌ Fichier app.py non trouvé")
        return False

    try:
        import streamlit as st
        print("\n✅ Streamlit installé")
        print(f"   Version: {st.__version__}")
        print(f"\n💡 Pour lancer l'app: streamlit run app.py")
        return True
    except ImportError:
        print("\n❌ Streamlit non installé")
        print("   SOLUTION: pip install streamlit")
        return False


def generate_report():
    """Génère un rapport de diagnostic complet"""
    print("\n" + "=" * 80)
    print("📋 RAPPORT DE DIAGNOSTIC")
    print("=" * 80)

    checks = {
        'Fichiers': check_files(),
        'Dataset': check_dataset(),
        'Dépendances': check_dependencies(),
        'Modèles': check_models(),
        'Streamlit': check_streamlit()
    }

    print("\n" + "=" * 80)
    print("RÉSUMÉ")
    print("=" * 80)

    for check_name, status in checks.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {check_name}")

    all_ok = all(checks.values())

    print("\n" + "=" * 80)

    if all_ok:
        print("✅ TOUT EST PRÊT !")
        print("\nÉtapes suivantes:")
        print("  1. python main.py           (Entraîner les modèles)")
        print("  2. streamlit run app.py     (Lancer l'application)")
    else:
        print("⚠️  CORRECTIONS NÉCESSAIRES")
        print("\nPlan d'action:")

        if not checks['Fichiers']:
            print("\n1. ❌ Fichiers manquants")
            print("   → Copiez tous les fichiers .py fournis")

        if not checks['Dataset']:
            print("\n2. ❌ Dataset non conforme")
            print("   → Si < 1000 lignes: python data_augmentation.py")
            print("   → Vérifiez data/food_data.csv")

        if not checks['Dépendances']:
            print("\n3. ❌ Dépendances manquantes")
            print("   → pip install -r requirements.txt")

        if not checks['Modèles']:
            print("\n4. ⚠️  Modèles non entraînés")
            print("   → python main.py")

        if not checks['Streamlit']:
            print("\n5. ❌ Streamlit non installé")
            print("   → pip install streamlit")

    print("=" * 80)

    return all_ok


def quick_fix():
    """Propose des corrections rapides"""
    print("\n" + "=" * 80)
    print("🔧 CORRECTIONS AUTOMATIQUES")
    print("=" * 80)

    # Créer dossiers manquants
    for folder in ['data', 'models']:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"✓ Dossier '{folder}/' créé")

    # Vérifier dataset
    if os.path.exists("data/food_data.csv"):
        df = pd.read_csv("data/food_data.csv")
        if len(df) < 1000:
            print(f"\n⚠️  Dataset trop petit ({len(df)} lignes)")
            response = input("\n   Voulez-vous augmenter le dataset automatiquement ? (o/n): ")
            if response.lower() == 'o':
                print("\n   Exécution de data_augmentation.py...")
                os.system("python data_augmentation.py")

    print("\n✓ Corrections automatiques terminées")


if __name__ == "__main__":
    print("=" * 80)
    print("🔍 DIAGNOSTIC NUTRIAI")
    print("=" * 80)
    print("\nCe script vérifie que tout est prêt pour lancer le projet.\n")

    # Option: corrections automatiques
    if len(sys.argv) > 1 and sys.argv[1] == "--fix":
        quick_fix()

    # Diagnostic complet
    all_ok = generate_report()

    # Code de sortie
    sys.exit(0 if all_ok else 1)