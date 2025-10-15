import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, mean_squared_error
import itertools
import random
import warnings

warnings.filterwarnings("ignore")


class NutritionRecommender:
    """
    Classe principal: charge le CSV, nettoie, entraine plusieurs modèles (classification & régression)
    et expose les fonctions utilisées par l'app Flask (pipeline, smart_recommendations, etc.).
    """

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

        # garantir présence des colonnes nutritionnelles
        for col in self.nutrition_cols:
            if col not in self.data.columns:
                self.data[col] = 0.0

        # forcer 'Food Category' en str et retirer lignes vides
        if 'Food Category' not in self.data.columns:
            raise ValueError("CSV must contain 'Food Category' column.")
        self.data['Food Category'] = self.data['Food Category'].astype(str)
        self.data = self.data[self.data['Food Category'].str.strip() != ""]

        # convertir numériques et remplir NaN par médiane
        for col in self.nutrition_cols:
            if col not in self.data.columns:
                self.data[col] = 0.0
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            median = float(self.data[col].median(skipna=True)) if not self.data[col].isna().all() else 0.0
            self.data[col].fillna(median, inplace=True)

        # normalisation des noms pour matching insensitive
        self._normalized_foods = [str(f).strip().lower() for f in self.data['Food Category'].values]

        # prepare scalers / placeholders qui seront créés dans train_models()
        self.scaler = None
        self.le_meal = None
        self.decision_tree = None
        self.linear_reg = None
        self.cal_reg = None  # regressor final pour calories (RandomForest ou fallback)
        self.poly = None
        self.poly_reg = None

        # entrainer les modèles dès l'init
        self.train_models()

    def train_models(self):
        """
        Entraine/optimise:
         - DecisionTreeClassifier (pruning via ccp_alpha selection)
         - LinearRegression (baseline) + Ridge
         - RandomForestRegressor (tuning via GridSearchCV) -> cal_reg
         - PolynomialFeatures + LinearRegression (fallback / capture non-linéarités)
         - MinMaxScaler pour KNN
         - Optional: suppression d'outliers (LOF) pour stabiliser les modèles
        """

        # X complet (vecteur nutritionnel)
        X = self.data[self.nutrition_cols].values.astype(float)

        # target meal type si existant, sinon 'Unknown'
        y = self.data["Meal Type"].astype(str).values if "Meal Type" in self.data.columns else np.array(["Unknown"] * len(self.data))

        # label encoder pour meal type (classification)
        self.le_meal = LabelEncoder()
        try:
            y_enc = self.le_meal.fit_transform(y)
        except Exception:
            y_enc = np.zeros(len(y), dtype=int)
            self.le_meal.fit(["Unknown"])

        # --- optional outlier removal (LOF) pour stabiliser KNN/trees/regressions ---
        try:
            # LOF appliqué uniquement sur les colonnes nutritives (pré-scaling)
            lof = LocalOutlierFactor(n_neighbors=20, contamination="auto")
            lof_mask = lof.fit_predict(X) > 0
            # si LOF retourne peu d'éléments (ex: petits dataset), on skip
            if np.sum(lof_mask) > max(10, 0.8 * len(lof_mask)):
                X = X[lof_mask]
                self.data = self.data.iloc[np.where(lof_mask)[0]]
                y_enc = y_enc[np.where(lof_mask)[0]]
                self._normalized_foods = [str(f).strip().lower() for f in self.data['Food Category'].values]
        except Exception:
            # si LOF échoue, on continue sans suppression
            pass

        # split pour classification (70/30)
        try:
            X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_enc, test_size=0.3, random_state=42)
        except Exception:
            # fallback simple split if something odd
            n = len(X)
            cut = int(n * 0.7)
            X_train_cls, X_test_cls = X[:cut], X[cut:]
            y_train_cls, y_test_cls = y_enc[:cut], y_enc[cut:]

        # --- Decision Tree with cost-complexity pruning selection via cross-valid.
        try:
            # compute path
            base_tree = DecisionTreeClassifier(random_state=42)
            path = base_tree.cost_complexity_pruning_path(X_train_cls, y_train_cls)
            ccp_alphas = path.ccp_alphas
            # evaluate several alphas with cross_val and pick best
            best_alpha = 0.0
            best_score = -np.inf
            for a in np.unique(np.clip(ccp_alphas, 0.0, None))[:10]:
                dt = DecisionTreeClassifier(random_state=42, ccp_alpha=a)
                scores = cross_val_score(dt, X_train_cls, y_train_cls, cv=3)
                mean_s = scores.mean()
                if mean_s > best_score:
                    best_score = mean_s
                    best_alpha = a
            # train final tree (if best_alpha==0 -> no pruning beyond defaults)
            self.decision_tree = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha, max_depth=12)
            self.decision_tree.fit(X_train_cls, y_train_cls)
            # diagnostic
            try:
                y_pred = self.decision_tree.predict(X_test_cls)
                print("Decision Tree Accuracy (Meal Type):", round(accuracy_score(y_test_cls, y_pred), 3))
            except Exception:
                pass
        except Exception as e:
            # fallback simple shallow tree
            print("Warning - decision tree training failed:", e)
            self.decision_tree = DecisionTreeClassifier(max_depth=6, random_state=42)
            try:
                self.decision_tree.fit(X_train_cls, y_train_cls)
            except Exception:
                pass

        # --- Régressions pour Calories ---
        features_no_cal = ["Protein", "Carbs", "Fat", "Saturated Fat", "Fiber", "Sugar", "Sodium", "Water"]
        X_cal = self.data[features_no_cal].values.astype(float)
        y_cal = self.data["Calories"].values.astype(float)

        # split pour régression (70/30)
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cal, y_cal, test_size=0.3, random_state=42)

        # baseline linear regression
        try:
            self.linear_reg = LinearRegression()
            self.linear_reg.fit(X_train_c, y_train_c)
            y_pred_cal = self.linear_reg.predict(X_test_c)
            print("Linear Regression RMSE (Calories):", round(np.sqrt(mean_squared_error(y_test_c, y_pred_cal)), 3))
        except Exception as e:
            print("Warning - linear regression training failed:", e)
            # fallback constant predictor
            mean_cal = float(np.mean(y_cal)) if len(y_cal) > 0 else 0.0
            class Dummy:
                def predict(self, X):
                    return np.array([mean_cal] * len(X))
            self.linear_reg = Dummy()

        # Ridge (regularized linear)
        try:
            self.ridge_reg = Ridge(alpha=1.0)
            self.ridge_reg.fit(X_train_c, y_train_c)
        except Exception:
            self.ridge_reg = self.linear_reg  # fallback

        # polynomial features + simple linear (captures non-lin)
        try:
            self.poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly_train = self.poly.fit_transform(X_train_c)
            self.poly_reg = LinearRegression().fit(X_poly_train, y_train_c)
        except Exception:
            self.poly = None
            self.poly_reg = None

        # RandomForestRegressor (ensemble): tuning léger via GridSearchCV
        try:
            rf = RandomForestRegressor(random_state=42)
            param_grid = {"n_estimators": [50, 100], "max_depth": [None, 8, 12]}
            gs = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1)
            gs.fit(X_train_c, y_train_c)
            self.cal_reg = gs.best_estimator_
            # diagnostic
            y_pred_rf = self.cal_reg.predict(X_test_c)
            print("RandomForestRegressor RMSE (Calories):", round(np.sqrt(mean_squared_error(y_test_c, y_pred_rf)), 3))
        except Exception as e:
            print("Warning - RandomForest training failed, falling back to linear reg:", e)
            self.cal_reg = self.ridge_reg if hasattr(self, 'ridge_reg') else self.linear_reg

        # scaler pour KNN (utilisé dans pipeline et recommend_menu_for_target)
        try:
            self.scaler = MinMaxScaler()
            self.scaler.fit(self.data[self.nutrition_cols].values.astype(float))
        except Exception as e:
            print("Warning - scaler fit failed:", e)
            # Identity scaler fallback
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
        - recommendations_list contient <= n_neighbors éléments
        - utilise decision tree (meal type), KNN pondéré pour recommandations, cal_reg pour calories
        """

        if food_name is None:
            return None, [], 0.0

        food_name_norm = str(food_name).strip().lower()

        # présence dans dataset ?
        if food_name_norm not in self._normalized_foods:
            return None, [], 0.0

        idx_base = self._normalized_foods.index(food_name_norm)

        # vecteur nutritionnel complet
        try:
            row = self.data.iloc[idx_base][self.nutrition_cols].values.astype(float)
        except Exception:
            return None, [], 0.0

        # prédiction meal type
        try:
            meal_pred_enc = self.decision_tree.predict([row])[0]
            meal_type = self.le_meal.inverse_transform([meal_pred_enc])[0]
        except Exception:
            meal_type = None

        # préparer données pour KNN
        X = self.data[self.nutrition_cols].values.astype(float)
        try:
            X_scaled = self.scaler.transform(X)
            target_scaled = self.scaler.transform([row])
        except Exception:
            X_scaled = X
            target_scaled = np.array([row])

        # KNN pondéré par distance (retourne noms)
        recommendations = []
        try:
            nn = min(n_neighbors + 1, len(self.data))
            knn = NearestNeighbors(n_neighbors=nn, metric="euclidean")
            knn.fit(X_scaled)
            distances, indices = knn.kneighbors(target_scaled)

            # build recommendations excluding the item itself, and prefer nearer items
            cand_pairs = list(zip(distances.flatten(), indices.flatten()))
            cand_pairs = [p for p in cand_pairs if int(p[1]) != int(idx_base)]
            # sort by distance
            cand_pairs.sort(key=lambda x: x[0])
            for dist, cand_idx in cand_pairs:
                cand_name = str(self.data.iloc[int(cand_idx)]["Food Category"]).strip()
                if cand_name.lower() != food_name_norm and cand_name not in recommendations:
                    recommendations.append(cand_name)
                if len(recommendations) >= n_neighbors:
                    break
        except Exception:
            recommendations = []

        # fallback deterministic (distance euclidienne brute)
        if not recommendations:
            try:
                diffs = X - row
                dists = np.linalg.norm(diffs, axis=1)
                order = np.argsort(dists)
                for cand_idx in order:
                    if int(cand_idx) == int(idx_base):
                        continue
                    cand_name = str(self.data.iloc[int(cand_idx)]["Food Category"]).strip()
                    if cand_name not in recommendations:
                        recommendations.append(cand_name)
                    if len(recommendations) >= n_neighbors:
                        break
            except Exception:
                recommendations = []

        # fallback random
        if not recommendations:
            try:
                all_names = [str(f).strip() for f in self.data["Food Category"].values if str(f).strip().lower() != food_name_norm]
                random.shuffle(all_names)
                recommendations = all_names[:min(n_neighbors, len(all_names))]
            except Exception:
                recommendations = []

        # Estimation calories pour base + recommandations (utilise cal_reg si dispo)
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

        recommendations = recommendations[:n_neighbors]
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

    # ---- recommandations "intelligentes" pour atteindre un objectif calorique
    def recommend_menu_for_target(self, base_food, target_calories, max_items=4, candidate_pool=20, top_k=5):
        if base_food is None:
            return {"best": None, "alternatives": []}

        base_norm = str(base_food).strip().lower()
        if base_norm not in self._normalized_foods:
            return {"best": None, "alternatives": []}

        idx_base = self._normalized_foods.index(base_norm)
        row_base = self.data.iloc[idx_base][self.nutrition_cols].values.astype(float)

        # pool de candidats via KNN (sur nutrition_cols)
        X = self.data[self.nutrition_cols].values.astype(float)
        try:
            X_scaled = self.scaler.transform(X)
            base_scaled = self.scaler.transform([row_base])
            knn = NearestNeighbors(n_neighbors=min(candidate_pool + 5, len(self.data)), metric='euclidean')
            knn.fit(X_scaled)
            _, inds = knn.kneighbors(base_scaled)
            cand_indices = [int(i) for i in inds[0] if int(i) != idx_base]
            if not cand_indices:
                dists = np.linalg.norm(X - row_base, axis=1)
                order = np.argsort(dists)
                cand_indices = [int(i) for i in order if i != idx_base][:candidate_pool]
        except Exception:
            dists = np.linalg.norm(X - row_base, axis=1)
            order = np.argsort(dists)
            cand_indices = [int(i) for i in order if i != idx_base][:candidate_pool]

        # liste de candidats (noms uniques)
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
            # fallback random
            all_foods = list(set(self.data['Food Category'].values))
            random.shuffle(all_foods)
            candidates = [f for f in all_foods if f.lower() != base_norm][:candidate_pool]

        # prédictions caloriques pour candidats
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
            except Exception:
                return float(self.data.iloc[idx]['Calories'])

        base_pred = pred_cal(base_food) or 0.0
        cand_preds = {c: pred_cal(c) for c in candidates if pred_cal(c) is not None}

        # générer combinaisons (brute-force, limité par max_items & candidate_pool)
        best_list = []
        for k in range(1, max_items + 1):
            # limiter nombre de candidats combinés pour réduire explosion combinatoire
            for combo in itertools.combinations(cand_preds.keys(), k):
                total = base_pred + sum(cand_preds[c] for c in combo)
                diff = abs(total - target_calories)
                best_list.append((diff, combo, total))

        if not best_list:
            # fallback minimal
            if candidates:
                c = candidates[0]
                total = base_pred + cand_preds.get(c, 0)
                return {
                    "best": {"foods": [base_food, c], "pred_cal": total, "diff": abs(total - target_calories)},
                    "alternatives": []
                }
            else:
                return {"best": {"foods": [base_food], "pred_cal": base_pred, "diff": abs(base_pred - target_calories)}, "alternatives": []}

        # trier et garder top_k distincts
        best_list.sort(key=lambda x: x[0])
        alternatives, seen = [], set()
        for diff, combo, total in best_list:
            foods = [base_food] + list(combo)
            key = tuple(sorted(foods))
            if key in seen:
                continue
            seen.add(key)
            alternatives.append({"foods": foods, "pred_cal": total, "diff": diff})
            if len(alternatives) >= top_k:
                break

        return {"best": alternatives[0], "alternatives": alternatives}

    def smart_recommendations(self, base_food, target_calories=None, n_neighbors=10, max_items=4, candidate_pool=20):
        """
        Si target_calories fourni -> recommend_menu_for_target (combinaisons)
        Sinon -> pipeline classique (KNN similaires)
        """
        if target_calories is None:
            meal_pred, recs, total_cal = self.pipeline(base_food, n_neighbors=n_neighbors)
            return {"meal_type": meal_pred, "recommendations": recs, "pred_cal": total_cal}
        else:
            res = self.recommend_menu_for_target(base_food, target_calories, max_items=max_items, candidate_pool=candidate_pool)
            return res
