import random

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
import itertools
import warnings
warnings.filterwarnings("ignore")


class NutritionRecommender:
    def __init__(self, csv_path):
        # Chargement et nettoyage du dataset
        self.data = pd.read_csv(csv_path)

        self.nutrition_cols = [
            "Calories", "Protein", "Carbs", "Fat", "Saturated Fat",
            "Fiber", "Sugar", "Sodium", "Water"
        ]

        self.classes = {
            "Macronutrients": ["Calories", "Protein", "Carbs"],
            "Lipids": ["Fat", "Saturated Fat"],
            "Fibres & Sugar": ["Fiber", "Sugar"],
            "Minerals & Water": ["Sodium", "Water"]
        }

        # Garantir présence des colonnes nutritionnelles
        for col in self.nutrition_cols:
            if col not in self.data.columns:
                self.data[col] = 0.0
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce').fillna(0)

        if 'Food Category' not in self.data.columns:
            raise ValueError("Colonne 'Food Category' manquante dans le CSV.")
        self.data['Food Category'] = self.data['Food Category'].astype(str).str.strip()
        self.data = self.data[self.data['Food Category'] != ""]

        # Normalisation des noms d'aliments
        self._normalized_foods = [f.lower() for f in self.data['Food Category'].values]

        # Placeholders pour modèles et données
        self.scaler = None
        self.le_meal = None
        self.decision_tree = None
        self.linear_reg = None
        self.ridge_reg = None
        self.cal_reg = None

        # Stockage des splits de test pour évaluation cohérente
        self.X_test_cls = self.y_test_cls = None
        self.X_test_reg = self.y_test_reg = None

        self.train_models()

    def train_models(self):
        # --- Données complètes ---
        X = self.data[self.nutrition_cols].values.astype(float)
        y = (self.data["Meal Type"].astype(str).values
             if "Meal Type" in self.data.columns
             else np.array(["Unknown"] * len(self.data)))

        # --- Encodage des labels (classification) ---
        self.le_meal = LabelEncoder()
        y_enc = self.le_meal.fit_transform(y)

        # --- Split stratifié pour classification ---
        X_train_cls, self.X_test_cls, y_train_cls, self.y_test_cls = train_test_split(
            X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
        )

        # --- RandomForestClassifier (équilibré) ---
        rf_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf_clf.fit(X_train_cls, y_train_cls)
        self.decision_tree = rf_clf

        # --- Régression : prédiction des calories ---
        features_no_cal = [
            "Protein", "Carbs", "Fat", "Saturated Fat",
            "Fiber", "Sugar", "Sodium", "Water"
        ]
        X_cal = self.data[features_no_cal].values.astype(float)
        y_cal = self.data["Calories"].values.astype(float)

        X_train_reg, self.X_test_reg, y_train_reg, self.y_test_reg = train_test_split(
            X_cal, y_cal, test_size=0.3, random_state=42
        )

        # Modèles de base
        self.linear_reg = LinearRegression().fit(X_train_reg, y_train_reg)
        self.ridge_reg = Ridge(alpha=1.0).fit(X_train_reg, y_train_reg)

        # RandomForestRegressor optimisé via GridSearchCV
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 15],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        gs = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        gs.fit(X_train_reg, y_train_reg)
        self.cal_reg = gs.best_estimator_

        # Scaler pour KNN (recommandations similaires)
        self.scaler = MinMaxScaler().fit(X)

    def all_foods(self):
        return list(self.data["Food Category"].values)

    def pipeline(self, food_name, n_neighbors=10):
        if not food_name:
            return None, [], 0.0

        food_name_norm = str(food_name).strip().lower()
        if food_name_norm not in self._normalized_foods:
            return None, [], 0.0

        idx = self._normalized_foods.index(food_name_norm)
        row = self.data.iloc[idx][self.nutrition_cols].values.astype(float)

        # Prédiction du type de repas
        try:
            meal_enc = self.decision_tree.predict([row])[0]
            meal_type = self.le_meal.inverse_transform([meal_enc])[0]
        except:
            meal_type = None

        # Recommandations via KNN
        X_scaled = self.scaler.transform(self.data[self.nutrition_cols].values.astype(float))
        target_scaled = self.scaler.transform([row])

        knn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(self.data)), metric="euclidean")
        knn.fit(X_scaled)
        _, indices = knn.kneighbors(target_scaled)

        recommendations = []
        for i in indices.flatten():
            if int(i) == idx:
                continue
            name = str(self.data.iloc[i]["Food Category"]).strip()
            if name not in recommendations:
                recommendations.append(name)
            if len(recommendations) >= n_neighbors:
                break

        # Estimation calorique totale
        total_cal = self.calories_for_list([food_name] + recommendations)
        return meal_type, recommendations, float(total_cal)

    def calories_for_list(self, food_list):
        total = 0.0
        features = ["Protein", "Carbs", "Fat", "Saturated Fat", "Fiber", "Sugar", "Sodium", "Water"]
        for f in food_list:
            if not f:
                continue
            f_norm = str(f).strip().lower()
            if f_norm not in self._normalized_foods:
                continue
            idx = self._normalized_foods.index(f_norm)
            vals = self.data.iloc[idx][features].values.astype(float)
            try:
                total += float(self.cal_reg.predict([vals])[0])
            except:
                total += float(self.data.iloc[idx]["Calories"])
        return float(total)

    def get_nutrition_data(self, food_list, class_name):
        cols = self.classes.get(class_name, [])
        values = []
        for f in food_list:
            f_norm = str(f).strip().lower()
            row = self.data[self.data['Food Category'].str.strip().str.lower() == f_norm]
            if row.empty:
                values.append([0.0] * len(cols))
            else:
                values.append([float(row.iloc[0][c]) for c in cols])
        return cols, values

    def recommend_menu_for_target(self, base_food, target_calories, max_items=4, candidate_pool=20, top_k=5):
        if not base_food:
            return {"best": None, "alternatives": []}

        base_norm = str(base_food).strip().lower()
        if base_norm not in self._normalized_foods:
            return {"best": None, "alternatives": []}

        idx_base = self._normalized_foods.index(base_norm)
        row_base = self.data.iloc[idx_base][self.nutrition_cols].values.astype(float)

        # Pool de candidats via KNN
        X = self.data[self.nutrition_cols].values.astype(float)
        X_scaled = self.scaler.transform(X)
        base_scaled = self.scaler.transform([row_base])

        knn = NearestNeighbors(n_neighbors=min(candidate_pool + 5, len(self.data)), metric='euclidean')
        knn.fit(X_scaled)
        _, inds = knn.kneighbors(base_scaled)
        cand_indices = [int(i) for i in inds.flatten() if int(i) != idx_base][:candidate_pool]

        if not cand_indices:
            dists = np.linalg.norm(X - row_base, axis=1)
            order = np.argsort(dists)
            cand_indices = [int(i) for i in order if i != idx_base][:candidate_pool]

        candidates = []
        for i in cand_indices:
            cand = str(self.data.iloc[i]['Food Category']).strip()
            if cand.lower() == base_norm:
                continue
            if cand not in candidates:
                candidates.append(cand)
            if len(candidates) >= candidate_pool:
                break

        if not candidates:
            all_foods = [f for f in self.data['Food Category'].values if str(f).strip().lower() != base_norm]
            random.shuffle(all_foods)
            candidates = all_foods[:candidate_pool]

        # Prédiction calorique
        features_no_cal = ["Protein", "Carbs", "Fat", "Saturated Fat", "Fiber", "Sugar", "Sodium", "Water"]

        def pred_cal(name):
            nrm = str(name).strip().lower()
            if nrm in self._normalized_foods:
                idx = self._normalized_foods.index(nrm)
            else:
                matches = [i for i, f in enumerate(self._normalized_foods) if nrm in f or f in nrm]
                if not matches:
                    return None
                idx = matches[0]
            vals = self.data.iloc[idx][features_no_cal].values.astype(float)
            try:
                return float(self.cal_reg.predict([vals])[0])
            except:
                return float(self.data.iloc[idx]['Calories'])

        base_pred = pred_cal(base_food) or 0.0
        cand_preds = {c: pred_cal(c) for c in candidates if pred_cal(c) is not None}

        # Générer toutes les combinaisons
        best_list = []
        for k in range(1, max_items + 1):
            for combo in itertools.combinations(cand_preds.keys(), k):
                total = base_pred + sum(cand_preds[c] for c in combo)
                diff = abs(total - target_calories)
                best_list.append((diff, combo, total))

        if not best_list:
            if candidates:
                c = candidates[0]
                total = base_pred + (cand_preds.get(c, 0))
                return {
                    "best": {"foods": [base_food, c], "pred_cal": total, "diff": abs(total - target_calories)},
                    "alternatives": []
                }
            return {
                "best": {"foods": [base_food], "pred_cal": base_pred, "diff": abs(base_pred - target_calories)},
                "alternatives": []
            }

        best_list.sort(key=lambda x: x[0])
        alternatives = []
        seen = set()
        for diff, combo, total in best_list:
            foods = [base_food] + list(combo)
            key = tuple(sorted(foods))
            if key in seen:
                continue
            seen.add(key)
            alternatives.append({"foods": foods, "pred_cal": total, "diff": diff})
            if len(alternatives) >= top_k:
                break

        return {"best": alternatives[0], "alternatives": alternatives[1:top_k]}

    def smart_recommendations(self, base_food, target_calories=None, n_neighbors=10, max_items=4, candidate_pool=20):
        if target_calories is None:
            meal_pred, recs, total_cal = self.pipeline(base_food, n_neighbors=n_neighbors)
            return {"meal_type": meal_pred, "recommendations": recs, "pred_cal": total_cal}
        else:
            res = self.recommend_menu_for_target(
                base_food, target_calories,
                max_items=max_items, candidate_pool=candidate_pool
            )
            return res