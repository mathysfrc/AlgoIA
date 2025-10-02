import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import random
import warnings

warnings.filterwarnings("ignore")


class NutritionRecommender:
    def __init__(self, csv_path):
        # charge et nettoie le dataset
        self.data = pd.read_csv(csv_path)

        # colonnes nutritionnelles attendues (ordre important)
        self.nutrition_cols = [
            "Calories", "Protein", "Carbs", "Fat",
            "Saturated Fat", "Fiber", "Sugar", "Sodium", "Water"
        ]

        # classes/ groupes pour graphiques
        self.classes = {
            "Macronutrients": ["Calories", "Protein", "Carbs"],
            "Lipids": ["Fat", "Saturated Fat"],
            "Fibres & Sugar": ["Fiber", "Sugar"],
            "Minerals & Water": ["Sodium", "Water"]
        }

        # nettoyage : s'assurer que toutes les colonnes nutritionnelles existent
        for col in self.nutrition_cols:
            if col not in self.data.columns:
                # si une colonne manque, la créer à 0 (prévenir crash)
                self.data[col] = 0.0

        # supprimer lignes sans 'Food Category' valides
        self.data['Food Category'] = self.data['Food Category'].astype(str)
        self.data = self.data[self.data['Food Category'].str.strip() != ""]

        # remplir NaNs numériques par la médiane de la colonne
        for col in self.nutrition_cols:
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                # tenter de convertir en numérique
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            median = float(self.data[col].median(skipna=True))
            self.data[col].fillna(median, inplace=True)

        # préparer colonnes de présence pour matching insensitive
        self._normalized_foods = [str(f).strip().lower() for f in self.data['Food Category'].values]

        # entraîner modèles
        self.train_models()

    def train_models(self):
        # X pour Decision Tree et KNN : vecteur nutrition complet
        X = self.data[self.nutrition_cols].values
        y = self.data["Meal Type"].astype(str).values if "Meal Type" in self.data.columns else np.array(["Unknown"] * len(self.data))

        # label encoder pour meal type
        self.le_meal = LabelEncoder()
        try:
            y_enc = self.le_meal.fit_transform(y)
        except Exception:
            # fallback si erreur
            y_enc = np.zeros(len(y), dtype=int)
            self.le_meal.fit(["Unknown"])

        # split et training decision tree (prédiction meal type)
        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
        self.decision_tree = DecisionTreeClassifier(max_depth=6, random_state=42)
        try:
            self.decision_tree.fit(X_train, y_train)
            y_pred = self.decision_tree.predict(X_test)
            print("Decision Tree Accuracy (Meal Type):", round(accuracy_score(y_test, y_pred), 3))
        except Exception as e:
            print("Warning - decision tree training failed:", e)
            # fallback model minimal
            self.decision_tree = DecisionTreeClassifier(max_depth=1, random_state=42)
            self.decision_tree.fit(X_train, y_train)

        # Linear regression pour estimer calories à partir des autres features
        features_no_cal = ["Protein", "Carbs", "Fat", "Saturated Fat", "Fiber", "Sugar", "Sodium", "Water"]
        X_cal = self.data[features_no_cal].values
        y_cal = self.data["Calories"].values
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cal, y_cal, test_size=0.2, random_state=42)
        self.linear_reg = LinearRegression()
        try:
            self.linear_reg.fit(X_train_c, y_train_c)
            y_pred_cal = self.linear_reg.predict(X_test_c)
            print("Linear Regression RMSE (Calories):", round(np.sqrt(mean_squared_error(y_test_c, y_pred_cal)), 3))
        except Exception as e:
            print("Warning - linear regression training failed:", e)
            # fallback: prédicteur constant (moyenne)
            mean_cal = float(np.mean(y_cal)) if len(y_cal) > 0 else 0.0
            class Dummy:
                def predict(self, X):
                    return np.array([mean_cal] * len(X))
            self.linear_reg = Dummy()

        # scaler pour KNN
        self.scaler = StandardScaler()
        try:
            self.scaler.fit(X)
        except Exception as e:
            print("Warning - scaler fit failed:", e)
            # en cas d'erreur, utiliser scaler qui renvoie l'entrée
            class IdentityScaler:
                def fit(self, X): return self
                def transform(self, X): return X
            self.scaler = IdentityScaler()

    def all_foods(self):
        # renvoie la liste "affichable" (originale)
        return list(self.data["Food Category"].values)

    def pipeline(self, food_name, n_neighbors=10):
        """
        Retourne (meal_type, recommendations_list, total_calories_estimated)
        - recommendations_list contient <= n_neighbors éléments (10 par défaut)
        - recommandations sont proposées, mais ne sont pas ajoutées automatiquement au repas
        """
        if food_name is None:
            return None, [], 0.0

        # normalisation du nom fourni
        food_name_norm = str(food_name).strip().lower()

        # debug
        print(f">>> pipeline called for '{food_name}' -> norm '{food_name_norm}'")

        # vérifier présence (tolérance espaces/majuscules)
        if food_name_norm not in self._normalized_foods:
            print(">>> pipeline: aliment non trouvé dans dataset (après normalisation).")
            return None, [], 0.0

        # récupérer index de la première occurrence
        idx_base = self._normalized_foods.index(food_name_norm)

        # récupérer vecteur nutritionnel (en float)
        try:
            row = self.data.iloc[idx_base][self.nutrition_cols].values.astype(float)
        except Exception as e:
            print(">>> pipeline: impossible de lire les valeurs nutritionnelles:", e)
            return None, [], 0.0

        # prédiction meal type
        try:
            meal_pred_enc = self.decision_tree.predict([row])[0]
            meal_type = self.le_meal.inverse_transform([meal_pred_enc])[0]
        except Exception as e:
            print(">>> pipeline: erreur decision tree predict:", e)
            meal_type = None

        # Préparer X et scaled
        X = self.data[self.nutrition_cols].values.astype(float)
        try:
            X_scaled = self.scaler.transform(X)
            target_scaled = self.scaler.transform([row])
        except Exception as e:
            print(">>> pipeline: scaler.transform failed:", e)
            X_scaled = X
            target_scaled = np.array([row])

        # KNN: essayer d'obtenir voisins
        recommendations = []
        try:
            nn = min(n_neighbors + 1, len(self.data))  # +1 car inclut lui-même
            knn = NearestNeighbors(n_neighbors=nn, metric="euclidean")
            knn.fit(X_scaled)
            distances, indices = knn.kneighbors(target_scaled)
            print(">>> pipeline: KNN distances sample:", distances.flatten()[:5])
            # parcourir indices et construire recommendations (exclure lui-même)
            for cand_idx in indices[0]:
                cand_name = str(self.data.iloc[cand_idx]["Food Category"]).strip()
                if cand_name.lower() != food_name_norm and cand_name not in recommendations:
                    recommendations.append(cand_name)
                if len(recommendations) >= n_neighbors:
                    break
        except Exception as e:
            print(">>> pipeline: KNN failed with error:", e)
            recommendations = []

        # Si KNN n'a rien retourné, fallback deterministe: prendre les plus proches selon distance manuelle
        if not recommendations:
            try:
                # calcul des distances euclidiennes manuellement (non-scaled) si needed
                diffs = X - row
                dists = np.linalg.norm(diffs, axis=1)
                # trier par distance et prendre les plus proches (excluant index base)
                order = np.argsort(dists)
                for cand_idx in order:
                    if int(cand_idx) == int(idx_base):
                        continue
                    cand_name = str(self.data.iloc[cand_idx]["Food Category"]).strip()
                    if cand_name not in recommendations:
                        recommendations.append(cand_name)
                    if len(recommendations) >= n_neighbors:
                        break
                print(">>> pipeline: fallback deterministic used, got", len(recommendations), "recommendations.")
            except Exception as e:
                print(">>> pipeline: deterministic fallback failed:", e)
                recommendations = []

        # Si toujours vide (très improbable ici), fallback aléatoire
        if not recommendations:
            try:
                all_names = [str(f).strip() for f in self.data["Food Category"].values if str(f).strip().lower() != food_name_norm]
                random.shuffle(all_names)
                recommendations = all_names[:min(n_neighbors, len(all_names))]
                print(">>> pipeline: random fallback used, got", len(recommendations), "recommendations.")
            except Exception as e:
                print(">>> pipeline: random fallback failed:", e)
                recommendations = []

        # Estimation calories pour base + recommandations (utilise linear_reg)
        total_cal = 0.0
        features_no_cal = ["Protein", "Carbs", "Fat", "Saturated Fat", "Fiber", "Sugar", "Sodium", "Water"]
        # calculer pour base + chaque recommendation si présent dans dataset
        to_calc = [str(self.data.iloc[idx_base]["Food Category"]).strip()] + recommendations
        added = 0
        for name in to_calc:
            name_norm = str(name).strip().lower()
            if name_norm in self._normalized_foods:
                idx = self._normalized_foods.index(name_norm)
                vals = self.data.iloc[idx][features_no_cal].values.astype(float)
                try:
                    pred_cal = float(self.linear_reg.predict([vals])[0])
                except Exception:
                    # si linear_reg a planté, utiliser la valeur Calories réelle comme fallback
                    pred_cal = float(self.data.iloc[idx]["Calories"])
                total_cal += pred_cal
                added += 1
        if added == 0:
            total_cal = 0.0

        # limiter la taille de recommendations à n_neighbors (déjà géré mais on s'assure)
        recommendations = recommendations[:n_neighbors]

        print(f">>> pipeline result: meal_type={meal_type}, recommendations_count={len(recommendations)}, total_cal={total_cal:.2f}")
        return meal_type, recommendations, float(total_cal)

    def calories_for_list(self, food_list):
        # somme des calories estimées via linear_reg pour une liste d'aliments
        features_no_cal = ["Protein", "Carbs", "Fat", "Saturated Fat", "Fiber", "Sugar", "Sodium", "Water"]
        total = 0.0
        for f in food_list:
            if f is None:
                continue
            f_norm = str(f).strip().lower()
            if f_norm not in self._normalized_foods:
                continue
            idx = self._normalized_foods.index(f_norm)
            vals = self.data.iloc[idx][features_no_cal].values.astype(float)
            try:
                pred = float(self.linear_reg.predict([vals])[0])
            except Exception:
                pred = float(self.data.iloc[idx]["Calories"])
            total += pred
        return float(total)

    def get_nutrition_data(self, food_list, class_name):
        # renvoie (cols, values) où values est une liste de listes (une par aliment)
        cols = self.classes.get(class_name, [])
        values = []
        for f in food_list:
            f_norm = str(f).strip().lower()
            row = self.data[self.data['Food Category'].str.strip().str.lower() == f_norm]
            if row.empty:
                values.append([0] * len(cols))
            else:
                values.append([float(row.iloc[0][c]) for c in cols])
        return cols, values
