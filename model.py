# model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

class NutritionRecommender:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)

        # colonnes de données
        # On garde Calories aussi pour certaines étapes
        self.nutrition_cols = ["Calories", "Protein", "Carbs", "Fat",
                               "Saturated Fat", "Fiber", "Sugar", "Sodium", "Water"]

        # catégories pour graphiques
        self.classes = {
            "Macronutrients": ["Calories", "Protein", "Carbs"],
            "Lipids": ["Fat", "Saturated Fat"],
            "Fibres & Sugar": ["Fiber", "Sugar"],
            "Minerals & Water": ["Sodium", "Water"]
        }

        # entraîner les modèles
        self.train_models()

    def train_models(self):
        # Decision Tree -> prédire Meal Type à partir des nutriments
        X = self.data[self.nutrition_cols].values
        y = self.data["Meal Type"].values
        self.le_meal = LabelEncoder()
        y_enc = self.le_meal.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
        self.decision_tree = DecisionTreeClassifier(max_depth=6, random_state=42)
        self.decision_tree.fit(X_train, y_train)
        y_pred = self.decision_tree.predict(X_test)
        print("Decision Tree Accuracy (Meal Type):", round(accuracy_score(y_test, y_pred), 3))

        # Linear Regression -> prédire Calories à partir des autres nutriments (sans Calories input)
        # ici on utilise toutes les colonnes *sauf* Calories comme features
        features = ["Protein", "Carbs", "Fat", "Saturated Fat", "Fiber", "Sugar", "Sodium", "Water"]
        X_cal = self.data[features].values
        y_cal = self.data["Calories"].values
        X_train, X_test, y_train, y_test = train_test_split(X_cal, y_cal, test_size=0.2, random_state=42)
        self.linear_reg = LinearRegression()
        self.linear_reg.fit(X_train, y_train)
        y_pred_cal = self.linear_reg.predict(X_test)
        print("Linear Regression RMSE (Calories):", round(np.sqrt(mean_squared_error(y_test, y_pred_cal)), 3))

        # KNN preparation (will use same features as Decision Tree for similarity)
        self.scaler = StandardScaler()
        self.scaler.fit(X)

    # renvoie la liste complète des aliments (utile pour select)
    def all_foods(self):
        return list(self.data["Food Category"].unique())

    # pipeline complet pour un aliment de base -> prédiction meal_type, recommandations (knn), et estimation calories du combo
    def pipeline(self, food_name, n_neighbors=3):
        if food_name not in self.data["Food Category"].values:
            return None, [], 0.0

        # récupérer vecteur nutrition pour l'aliment
        row = self.data[self.data["Food Category"] == food_name][self.nutrition_cols].values[0]

        # Decision Tree : prédire Meal Type
        meal_pred = self.decision_tree.predict([row])[0]
        meal_type = self.le_meal.inverse_transform([meal_pred])[0]

        # KNN : recommander aliments proches
        X = self.data[self.nutrition_cols].values
        X_scaled = self.scaler.transform(X)
        target_scaled = self.scaler.transform([row])
        knn = NearestNeighbors(n_neighbors=n_neighbors+1, metric='euclidean')
        knn.fit(X_scaled)
        distances, indices = knn.kneighbors(target_scaled)
        recommendations = []
        for idx in indices[0]:
            candidate = self.data.iloc[idx]["Food Category"]
            if candidate != food_name and candidate not in recommendations:
                recommendations.append(candidate)
            if len(recommendations) >= n_neighbors:
                break

        # Estimer calories pour chaque aliment listé (on utilise linear_reg.predict sur features sans 'Calories')
        features = ["Protein", "Carbs", "Fat", "Saturated Fat", "Fiber", "Sugar", "Sodium", "Water"]
        total_cal = 0.0
        for f in [food_name] + recommendations:
            r = self.data[self.data["Food Category"] == f]
            if not r.empty:
                vals = r[features].values[0]
                pred_cal = float(self.linear_reg.predict([vals])[0])
                total_cal += pred_cal

        return meal_type, recommendations, float(total_cal)

    # calcule la somme des calories (estimation LR) pour une liste d'aliments
    def calories_for_list(self, food_list):
        features = ["Protein", "Carbs", "Fat", "Saturated Fat", "Fiber", "Sugar", "Sodium", "Water"]
        total = 0.0
        for f in food_list:
            r = self.data[self.data["Food Category"] == f]
            if r.empty:
                continue
            vals = r[features].values[0]
            total += float(self.linear_reg.predict([vals])[0])
        return float(total)

    # utile pour graphiques : renvoie cols et valeurs pour chaque aliment donné
    def get_nutrition_data(self, food_list, class_name):
        df = self.data.copy()
        cols = self.classes[class_name]
        values = []
        for f in food_list:
            row = df[df['Food Category'] == f][cols].values
            if len(row) == 0:
                values.append([0]*len(cols))
            else:
                values.append(row[0].tolist())
        return cols, values
