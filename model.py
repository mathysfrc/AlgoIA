import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

class NutritionRecommender:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        # Colonnes nutritionnelles
        self.nutrition_cols = ["Calories", "Protein", "Carbs", "Fat",
                               "Saturated Fat", "Fiber", "Sugar", "Sodium", "Water"]
        # Classes pour graphiques
        self.classes = {
            "Macronutrients": ["Calories", "Protein", "Carbs"],
            "Lipids": ["Fat", "Saturated Fat"],
            "Fibres & Sugar": ["Fiber", "Sugar"],
            "Minerals & Water": ["Sodium", "Water"]
        }

        # Profil nutritionnel cible pour fitness homme moyen
        self.target_profile = {
            "Breakfast": np.array([500, 30, 60, 10, 2, 5, 20, 150, 100]),
            "Lunch":     np.array([700, 40, 80, 15, 3, 7, 25, 200, 200]),
            "Dinner":    np.array([700, 40, 80, 15, 3, 7, 25, 200, 200]),
            "Snack":     np.array([250, 15, 30, 5, 1, 3, 15, 50, 50])
        }

    def recommend(self, food_name, meal_type=None, n_neighbors=3):
        df = self.data.copy()

        if meal_type not in self.target_profile:
            meal_type = "Dinner"

        # Vérifier que l'aliment existe
        if food_name not in df['Food Category'].values:
            return []

        # Besoin restant pour le repas
        target = self.target_profile[meal_type].copy()
        row = df[df['Food Category'] == food_name][self.nutrition_cols].values[0]
        remaining = target - row
        remaining[remaining < 0] = 0  # éviter valeurs négatives

        # Préparer features
        X = df[self.nutrition_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        remaining_scaled = scaler.transform(remaining.reshape(1, -1))

        # Pondération pour fitness : protéines et glucides plus importantes, graisses limitées
        weights = np.ones(len(self.nutrition_cols))
        weights[1] = 2.0  # Protein
        weights[2] = 1.5  # Carbs
        weights[3] = 0.5  # Fat
        weights[4] = 0.5  # Saturated Fat
        X_scaled_weighted = X_scaled * weights
        remaining_scaled_weighted = remaining_scaled * weights

        # K-NN
        knn = NearestNeighbors(n_neighbors=n_neighbors*2, metric='euclidean')  # *2 pour plus de diversité
        knn.fit(X_scaled_weighted)
        distances, indices = knn.kneighbors(remaining_scaled_weighted)

        recommended = []
        for i in indices[0]:
            candidate = df.iloc[i]['Food Category']
            if candidate != food_name and candidate not in recommended:
                recommended.append(candidate)
            if len(recommended) >= n_neighbors:
                break

        return recommended

    def get_nutrition_data(self, food_list, class_name):
        df = self.data.copy()
        cols = self.classes[class_name]
        values = []
        for f in food_list:
            row = df[df['Food Category'] == f][cols].values
            if len(row) == 0:
                values.append([0]*len(cols))
            else:
                values.append(row[0])
        return cols, values
