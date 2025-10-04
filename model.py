import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import itertools
import random
import warnings

warnings.filterwarnings("ignore")

class NutritionRecommender:
    def __init__(self, csv_path):
        # charge et nettoie le dataset
        self.data = pd.read_csv(csv_path)

        # colonnes nutritionnelles attendues (ordre important)
        self.nutrition_cols = [
            "Calories", "Protein", "Carbs", "Fat", "Saturated Fat", "Fiber", "Sugar", "Sodium", "Water"
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
        # 70% learning set and 30% validation set
        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.3, random_state=42)
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
        # 70% learning set and 30% validation set
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cal, y_cal, test_size=0.3, random_state=42)

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

        # --- NEW: Random Forest (ou DecisionTree) regressor pour estimer les calories de façon plus robuste
        try:
            self.cal_reg = RandomForestRegressor(n_estimators=100, random_state=42)
            self.cal_reg.fit(X_train_c, y_train_c)
            y_pred_rf = self.cal_reg.predict(X_test_c)
            print("RandomForestRegressor RMSE (Calories):", round(np.sqrt(mean_squared_error(y_test_c, y_pred_rf)), 3))
        except Exception as e:
            print("Warning - RandomForest training failed, falling back to linear reg:", e)
            self.cal_reg = self.linear_reg

        # scaler pour KNN
        self.scaler = MinMaxScaler()
        try:
            self.scaler.fit(X)
        except Exception as e:
            print("Warning - scaler fit failed:", e)
            # en cas d'erreur, utiliser scaler qui renvoie l'entrée
            class IdentityScaler:
                def fit(self, X):
                    return self
                def transform(self, X):
                    return X
            self.scaler = IdentityScaler()

    def all_foods(self):
        # renvoie la liste "affichable" (originale)
        return list(self.data["Food Category"].values)

    def pipeline(self, food_name, n_neighbors=10):
        """ Retourne (meal_type, recommendations_list, total_calories_estimated)
        - recommendations_list contient <= n_neighbors éléments (10 par défaut)
        - recommandations sont proposées, mais ne sont pas ajoutées automatiquement au repas
        """
        if food_name is None:
            return None, [], 0.0

        # normalisation du nom fourni
        food_name_norm = str(food_name).strip().lower()

        # debug
        # print(f">>> pipeline called for '{food_name}' -> norm '{food_name_norm}'")

        # vérifier présence (tolérance espaces/majuscules)
        if food_name_norm not in self._normalized_foods:
            # print(">>> pipeline: aliment non trouvé dans dataset (après normalisation).")
            return None, [], 0.0

        # récupérer index de la première occurrence
        idx_base = self._normalized_foods.index(food_name_norm)

        # récupérer vecteur nutritionnel (en float)
        try:
            row = self.data.iloc[idx_base][self.nutrition_cols].values.astype(float)
        except Exception as e:
            # print(">>> pipeline: impossible de lire les valeurs nutritionnelles:", e)
            return None, [], 0.0

        # prédiction meal type
        try:
            meal_pred_enc = self.decision_tree.predict([row])[0]
            meal_type = self.le_meal.inverse_transform([meal_pred_enc])[0]
        except Exception as e:
            # print(">>> pipeline: erreur decision tree predict:", e)
            meal_type = None

        # Préparer X et scaled
        X = self.data[self.nutrition_cols].values.astype(float)
        try:
            X_scaled = self.scaler.transform(X)
            target_scaled = self.scaler.transform([row])
        except Exception as e:
            # print(">>> pipeline: scaler.transform failed:", e)
            X_scaled = X
            target_scaled = np.array([row])

        # KNN: essayer d'obtenir voisins
        recommendations = []
        try:
            nn = min(n_neighbors + 1, len(self.data))  # +1 car inclut lui-même
            knn = NearestNeighbors(n_neighbors=nn, metric="euclidean")
            knn.fit(X_scaled)
            distances, indices = knn.kneighbors(target_scaled)
            # print(">>> pipeline: KNN distances sample:", distances.flatten()[:5])
            # parcourir indices et construire recommendations (exclure lui-même)
            for cand_idx in indices[0]:
                cand_name = str(self.data.iloc[cand_idx]["Food Category"]).strip()
                if cand_name.lower() != food_name_norm and cand_name not in recommendations:
                    recommendations.append(cand_name)
                if len(recommendations) >= n_neighbors:
                    break
        except Exception as e:
            # print(">>> pipeline: KNN failed with error:", e)
            recommendations = []

        # If KNN empty, fallback deterministic
        if not recommendations:
            try:
                diffs = X - row
                dists = np.linalg.norm(diffs, axis=1)
                order = np.argsort(dists)
                for cand_idx in order:
                    if int(cand_idx) == int(idx_base):
                        continue
                    cand_name = str(self.data.iloc[cand_idx]["Food Category"]).strip()
                    if cand_name not in recommendations:
                        recommendations.append(cand_name)
                    if len(recommendations) >= n_neighbors:
                        break
                # print(">>> pipeline: fallback deterministic used, got", len(recommendations), "recommendations.")
            except Exception as e:
                # print(">>> pipeline: deterministic fallback failed:", e)
                recommendations = []

        if not recommendations:
            try:
                all_names = [str(f).strip() for f in self.data["Food Category"].values if str(f).strip().lower() != food_name_norm]
                random.shuffle(all_names)
                recommendations = all_names[:min(n_neighbors, len(all_names))]
                # print(">>> pipeline: random fallback used, got", len(recommendations), "recommendations.")
            except Exception as e:
                # print(">>> pipeline: random fallback failed:", e)
                recommendations = []

        # Estimation calories pour base + recommandations (utilise cal_reg)
        total_cal = 0.0
        features_no_cal = ["Protein", "Carbs", "Fat", "Saturated Fat", "Fiber", "Sugar", "Sodium", "Water"]
        to_calc = [str(self.data.iloc[idx_base]["Food Category"]).strip()] + recommendations
        added = 0
        for name in to_calc:
            name_norm = str(name).strip().lower()
            if name_norm in self._normalized_foods:
                idx = self._normalized_foods.index(name_norm)
                vals = self.data.iloc[idx][features_no_cal].values.astype(float)
                try:
                    pred_cal = float(self.cal_reg.predict([vals])[0])
                except Exception:
                    pred_cal = float(self.data.iloc[idx]["Calories"])
                total_cal += pred_cal
                added += 1
        if added == 0:
            total_cal = 0.0

        # limiter la taille de recommendations à n_neighbors
        recommendations = recommendations[:n_neighbors]

        # print(f">>> pipeline result: meal_type={meal_type}, recommendations_count={len(recommendations)}, total_cal={total_cal:.2f}")
        return meal_type, recommendations, float(total_cal)

    def calories_for_list(self, food_list):
        # somme des calories estimées via cal_reg pour une liste d'aliments
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
                pred = float(self.cal_reg.predict([vals])[0])
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

    # ---- NEW: recommandations "intelligentes" pour atteindre un objectif calorique
    def recommend_menu_for_target(self, base_food, target_calories, max_items=4, candidate_pool=20, top_k=5):
        if base_food is None:
            return {"best": None, "alternatives": []}

        base_norm = str(base_food).strip().lower()
        if base_norm not in self._normalized_foods:
            return {"best": None, "alternatives": []}

        idx_base = self._normalized_foods.index(base_norm)
        row_base = self.data.iloc[idx_base][self.nutrition_cols].values.astype(float)

        # --- pool de candidats (KNN ou fallback) ---
        X = self.data[self.nutrition_cols].values.astype(float)
        try:
            X_scaled = self.scaler.transform(X)
            base_scaled = self.scaler.transform([row_base])
            knn = NearestNeighbors(
                n_neighbors=min(candidate_pool + 5, len(self.data)), metric='euclidean'
            )
            knn.fit(X_scaled)
            _, inds = knn.kneighbors(base_scaled)
            print("DEBUG >>> raw inds:", inds)

            cand_indices = [int(i) for i in inds[0] if int(i) != idx_base]

            # fallback si vide
            if not cand_indices:
                dists = np.linalg.norm(X - row_base, axis=1)
                order = np.argsort(dists)
                cand_indices = [int(i) for i in order if i != idx_base][:candidate_pool]

        except Exception as e:
            print("DEBUG >>> KNN failed:", e)
            dists = np.linalg.norm(X - row_base, axis=1)
            order = np.argsort(dists)
            cand_indices = [int(i) for i in order if i != idx_base][:candidate_pool]

            print("DEBUG >>> cand_indices:", cand_indices[:10])
            print("DEBUG >>> base_food:", base_food, "| idx_base:", idx_base)

        candidates = []
        for i in cand_indices:
            cand = str(self.data.iloc[i]['Food Category']).strip()
            if cand.lower() == base_norm:
                continue  # on évite uniquement le même exact aliment
            if cand not in candidates:  # éviter 100x "Fruits"
                candidates.append(cand)

        if not candidates:
            print("DEBUG >>> fallback: random candidates")
            all_foods = list(set(self.data['Food Category'].values))
            random.shuffle(all_foods)
            candidates = [f for f in all_foods if f.lower() != base_norm][:candidate_pool]
        print("DEBUG >>> Candidates:", candidates[:10])

        # --- calories prédictives ---
        features_no_cal = ["Protein","Carbs","Fat","Saturated Fat","Fiber","Sugar","Sodium","Water"]

        def pred_cal(name):
            nrm = str(name).strip().lower()
            # essayer correspondance exacte
            if nrm in self._normalized_foods:
                idx = self._normalized_foods.index(nrm)
            else:
                # fallback: correspondance approximative (contient le mot)
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
        print("DEBUG >>> Candidates:", candidates[:10])
        print("DEBUG >>> _normalized_foods (sample):", self._normalized_foods[:20])
        cand_preds = {c: pred_cal(c) for c in candidates if pred_cal(c) is not None}

        # --- générer combinaisons ---
        best_list = []
        for k in range(1, max_items+1):
            for combo in itertools.combinations(cand_preds.keys(), k):
                total = base_pred + sum(cand_preds[c] for c in combo)
                diff = abs(total - target_calories)
                best_list.append((diff, combo, total))

        print("Nb combos générées:", len(best_list))

        if not best_list:
            # fallback : au moins base_food + un candidat brut
            if candidates:
                c = candidates[0]
                total = base_pred + cand_preds.get(c,0)
                return {
                    "best": {"foods":[base_food,c],"pred_cal":total,"diff":abs(total-target_calories)},
                    "alternatives":[]
                }
            else:
                return {"best":{"foods":[base_food],"pred_cal":base_pred,"diff":abs(base_pred-target_calories)},"alternatives":[]}

        print("=== DEBUG SMART MENU ===")
        print("Base food:", base_food, "pred:", base_pred)
        print("Candidates:", candidates[:10])
        print("Cand preds:", list(cand_preds.items())[:10])
        # --- trier et garder top_k distincts ---
        best_list.sort(key=lambda x: x[0])
        alternatives, seen = [], set()
        for diff, combo, total in best_list:
            foods = [base_food] + list(combo)
            key = tuple(sorted(foods))
            if key in seen:
                continue
            seen.add(key)
            alternatives.append({"foods":foods,"pred_cal":total,"diff":diff})
            if len(alternatives) >= top_k:
                break

        return {"best":alternatives[0], "alternatives":alternatives}

    # small helper to expose smart recommendations in a single call
    def smart_recommendations(self, base_food, target_calories=None, n_neighbors=10, max_items=4, candidate_pool=20):
        """
        Si target_calories fourni -> utilise recommend_menu_for_target pour proposer des combinaisons.
        Sinon -> comportement classique pipeline (KNN similaires).
        """
        if target_calories is None:
            meal_pred, recs, total_cal = self.pipeline(base_food, n_neighbors=n_neighbors)
            return {"meal_type": meal_pred, "recommendations": recs, "pred_cal": total_cal}
        else:
            res = self.recommend_menu_for_target(base_food, target_calories, max_items=max_items, candidate_pool=candidate_pool)
            return res
