# main.py
import os
from data_preprocessing import load_and_clean
from models import NutriAI

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    print("Nettoyage du dataset...")
    df = load_and_clean("data/food_data.csv")
    df.to_csv("data/processed_nutrition.csv", index=False)

    print("Entraînement des modèles...")
    ai = NutriAI()
    ai.train_balance_predictor()
    ai.classify_food_role(objective="perte")
    ai.train_knn_recommender()

    print("Tout est prêt ! Lancez : streamlit run app.py")