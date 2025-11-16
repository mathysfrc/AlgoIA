# models.py
"""NutriAI

Version corrigée du module models.py — spécialement :
- entraînement du KNN avec sauvegarde DES FEATURES utilisées
- recommend_similar robuste (compatible sans casser scaler/knn)
- fallback propres si fichiers manquants

Design goals:
- Ne jamais tenter de transformer des colonnes non-connues par le scaler.
- Ne pas changer la dimension d'entrée du KNN après entraînement.
- Si un user_profile est fourni, on réordonne / re-score les voisins retournés
  plutôt que d'ajouter des colonnes non prévues au scaler/KNN.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from user_profile import calculate_bmr, calculate_tdee, get_macro_targets


class NutriAI:
    """
    Classe centrale : gère entraînement & usage de tous les modèles.
    Tout est modulable par profil / objectif via set_user_profile().

    Notes importantes pour KNN / recommend_similar :
    - train_knn_recommender() sauve la liste exacte de features dans models/knn_features.json
    - recommend_similar() utilise strictement ces features pour scaler et knn
    - Si user_profile est fourni, on **ne modifie pas** la dimension envoyée au scaler/knn :
      on récupère d'abord les voisins selon le KNN, puis on re-score ces candidats en
      tenant compte des macros/utilisateur pour produire un classement personnalisé.
    """

    def __init__(self, data_path="data/processed_nutrition.csv"):
        # Chargement données
        self.df = pd.read_csv(data_path)

        # Features de base utilisées partout
        self.base_features = [
            'Calories', 'Protein', 'Carbs', 'Fat', 'Fiber',
            'Sugar', 'Water', 'density_kcal_100g', 'satiety_index'
        ]

        # Assure colonnes profil/objective/activity_factor
        if 'profil' not in self.df.columns:
            self.df['profil'] = 'modere'
        if 'objective' not in self.df.columns:
            self.df['objective'] = 'maintien'
        if 'activity_factor' not in self.df.columns:
            self.df['activity_factor'] = self.df['profil'].map({
                'sedentaire': 1.2,
                'leger': 1.4,
                'modere': 1.55,
                'intense': 1.8,
                'athlete': 2.0
            }).fillna(1.5)

        # Calculs de base (au cas où)
        if 'prot_ratio' not in self.df.columns:
            # defensive: avoid division by zero
            self.df['prot_ratio'] = self.df['Protein'] * 4 / self.df['Calories'].replace(0, np.nan)
            self.df['prot_ratio'] = self.df['prot_ratio'].fillna(0.0)
        if 'carb_ratio' not in self.df.columns:
            self.df['carb_ratio'] = self.df['Carbs'] * 4 / self.df['Calories'].replace(0, np.nan)
            self.df['carb_ratio'] = self.df['carb_ratio'].fillna(0.0)
        if 'fat_ratio' not in self.df.columns:
            self.df['fat_ratio'] = self.df['Fat'] * 9 / self.df['Calories'].replace(0, np.nan)
            self.df['fat_ratio'] = self.df['fat_ratio'].fillna(0.0)

        # Initialisations
        self.scaler = StandardScaler()
        self.rf_reg = None
        self.rf_clf = None
        self.knn = None
        self.dt = None

        # profil courant (utilisé pour recommandations runtime)
        self.current_profile = 'modere'
        self.current_objective = 'maintien'

        # dossiers modèles
        os.makedirs("models", exist_ok=True)

    # ---------------------------
    # Helpers
    # ---------------------------
    def _get_encoded_X_and_save_features(self, df_local):
        """
        Encode profil+objective et retourne X complet. Ne réordonne pas.
        """
        df_encoded = pd.get_dummies(df_local, columns=['profil', 'objective'], drop_first=True)
        extra = [c for c in df_encoded.columns if c.startswith('profil_') or c.startswith('objective_')]
        # ensure base_features present in df_encoded; if not, add zeros
        for f in self.base_features:
            if f not in df_encoded.columns:
                df_encoded[f] = 0.0
        X = df_encoded[self.base_features + extra]
        return X

    # ---------------------------
    # Train regression (balance_score)
    # ---------------------------
    def train_balance_predictor(self):
        if 'balance_score' not in self.df.columns:
            raise ValueError("La colonne 'balance_score' est requise dans processed_nutrition.csv")

        X = self._get_encoded_X_and_save_features(self.df)
        y = self.df['balance_score']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.rf_reg = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
        self.rf_reg.fit(X_train, y_train)

        y_pred = self.rf_reg.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"RF Regressor entraîné — MAE: {mae:.4f}")

        joblib.dump(self.rf_reg, "models/rf_balance.pkl")

    # ---------------------------
    # Classification rôle foods
    # ---------------------------
    def classify_food_role(self, objective=None):
        if objective:
            self.current_objective = objective

        # Recalculate ratios (defensive)
        self.df['prot_ratio'] = (self.df['Protein'] * 4 / self.df['Calories'].replace(0, np.nan)).fillna(0.0)
        self.df['carb_ratio'] = (self.df['Carbs'] * 4 / self.df['Calories'].replace(0, np.nan)).fillna(0.0)
        self.df['fat_ratio'] = (self.df['Fat'] * 9 / self.df['Calories'].replace(0, np.nan)).fillna(0.0)

        # règles dynamiques selon objectif
        if self.current_objective == 'perte':
            conditions = [
                (self.df['prot_ratio'] > 0.25) & (self.df['Fiber'] > 3) & (self.df['Sugar'] < 5),
                (self.df['Fat'] > 22) | (self.df['Calories'] > 420),
                (self.df['Sugar'] > 15)
            ]
            choices = ['Privilégier', 'Modération', 'Éviter']

        elif self.current_objective == 'gain':
            conditions = [
                (self.df['Calories'] >= 300) & (self.df['Protein'] >= 12),
                (self.df['Fat'] < 12),
                (self.df['Sugar'] > 18)
            ]
            choices = ['Privilégier', 'Modération', 'Éviter']

        else:  # maintien
            conditions = [
                (self.df['prot_ratio'] > 0.22),
                (self.df['Fat'] > 25),
                (self.df['Sugar'] > 12)
            ]
            choices = ['Privilégier', 'Modération', 'Éviter']

        self.df['role'] = np.select(conditions, choices, default='Neutre')

        # Encodage & entraînement classifier
        df_encoded = pd.get_dummies(self.df, columns=['profil', 'objective'], drop_first=True)
        extra_cols = [c for c in df_encoded.columns if c.startswith('profil_') or c.startswith('objective_')]

        # make sure base_features exist
        for f in self.base_features:
            if f not in df_encoded.columns:
                df_encoded[f] = 0.0

        X = df_encoded[self.base_features + extra_cols]
        y = df_encoded['role']

        self.rf_clf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, class_weight='balanced')
        self.rf_clf.fit(X, y)

        # Sauvegarde modèle et features exactes
        joblib.dump(self.rf_clf, "models/rf_classifier.pkl")
        joblib.dump(list(X.columns), "models/rf_classifier_features.pkl")
        print(f"RF Classifier entraîné et sauvegardé (objectif={self.current_objective}).")

    # ---------------------------
    # KNN recommender
    # ---------------------------
    def train_knn_recommender(self):
        """
        Entraîne le KNN et sauvegarde :
          - le scaler (models/scaler.pkl)
          - le knn (models/knn_recommender.pkl)
          - la liste exacte des features utilisés par le KNN (models/knn_features.json)

        IMPORTANT : le KNN est entraîné *sur les seules features tabulaires* (base_features + one-hot profil/objective)
        et **n'inclut pas** de colonnes utilisateur dynamiques (u_calories, ...).
        """
        df_encoded = pd.get_dummies(self.df, columns=['profil', 'objective'], drop_first=True)
        extra_cols = [c for c in df_encoded.columns if c.startswith('profil_') or c.startswith('objective_')]

        # ensure base features present
        for f in self.base_features:
            if f not in df_encoded.columns:
                df_encoded[f] = 0.0

        official_features = self.base_features + extra_cols
        X = df_encoded[official_features].fillna(0.0)

        # fit scaler and knn
        X_scaled = self.scaler.fit_transform(X)
        self.knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
        self.knn.fit(X_scaled)

        # persist
        joblib.dump(self.knn, "models/knn_recommender.pkl")
        joblib.dump(self.scaler, "models/scaler.pkl")
        with open("models/knn_features.json", "w") as f:
            json.dump(official_features, f)

        print("KNN entraîné avec succès + features sauvegardées.")

    def recommend_similar(self, food_name, profil=None, objective=None, user_profile=None, top_k=6):
        """
        Recommande aliments similaires.

        - Utilise strictement les features enregistrées par train_knn_recommender().
        - Si user_profile est fourni, le classement est post-traité pour favoriser
          des aliments plus proches des macros cibles (sans modifier la taille d'entrée du KNN).
        """
        # find the food row
        row = self.df[self.df['Food Category'].str.contains(food_name, case=False, na=False)]
        if row.empty:
            return []

        profil_sel = profil or self.current_profile
        objective_sel = objective or self.current_objective

        subset = self.df[(self.df['profil'] == profil_sel) & (self.df['objective'] == objective_sel)]
        if subset.empty:
            subset = self.df.copy()

        # load official features used by KNN
        features_path = "models/knn_features.json"
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                official_features = json.load(f)
        else:
            # fallback: try to reconstruct sensibly from df
            # choose numeric columns present in df that are in base_features or are one-hot profil/objective
            candidate = [c for c in self.df.columns if self.df[c].dtype != 'object']
            official_features = [c for c in candidate if (c in self.base_features) or c.startswith('profil_') or c.startswith('objective_')]
            if not official_features:
                # absolute fallback
                official_features = [c for c in self.base_features if c in self.df.columns]

        # ensure present in subset; add missing with zeros
        for col in official_features:
            if col not in subset.columns:
                subset[col] = 0.0

        # build X (respecting the exact column order)
        X = subset[official_features].copy().fillna(0.0)

        # load scaler
        scaler_path = "models/scaler.pkl"
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            try:
                X_scaled = scaler.transform(X)
            except Exception as e:
                # If feature names mismatch (for example scaler was trained on different columns),
                # attempt to align columns: pad missing, drop extras, reorder
                trained_features = None
                try:
                    # if scaler was saved from sklearn's StandardScaler after fit, it may have feature_names_in_
                    trained_features = getattr(scaler, "feature_names_in_", None)
                except Exception:
                    trained_features = None

                if trained_features is not None:
                    trained_features = list(trained_features)
                    # add missing cols
                    for col in trained_features:
                        if col not in X.columns:
                            X[col] = 0.0
                    # drop extras
                    for col in list(X.columns):
                        if col not in trained_features:
                            X.drop(col, axis=1, inplace=True)
                    # reorder
                    X = X[trained_features]
                    X_scaled = scaler.transform(X)
                else:
                    # last resort: fit_transform a local scaler (less ideal)
                    from sklearn.preprocessing import StandardScaler
                    local_scaler = StandardScaler()
                    X_scaled = local_scaler.fit_transform(X)
        else:
            # no scaler saved -> fit a local one (best-effort)
            from sklearn.preprocessing import StandardScaler
            local_scaler = StandardScaler()
            X_scaled = local_scaler.fit_transform(X)

        # load knn
        knn_path = "models/knn_recommender.pkl"
        if os.path.exists(knn_path):
            knn = joblib.load(knn_path)
        else:
            # if no knn saved, train new one on-the-fly using X_scaled
            knn = NearestNeighbors(n_neighbors=min(top_k + 1, len(X_scaled)), metric='euclidean')
            knn.fit(X_scaled)

        # find index of the requested food within subset
        try:
            idx = subset.index.get_loc(row.index[0])
        except Exception:
            # fallback: use the first row of X_scaled as query
            idx = 0

        # ask knn for k+1 neighbors (first is the item itself)
        k_query = min(top_k + 1, len(X_scaled))
        distances, indices = knn.kneighbors(X_scaled[idx].reshape(1, -1), n_neighbors=k_query)
        cand_idx = indices[0]

        # remove the item itself if present
        cand_idx = [i for i in cand_idx if i != idx]
        candidates = subset.iloc[cand_idx].copy().reset_index(drop=True)

        # If no user_profile, return top_k directly
        if (user_profile is None) and (not hasattr(self, '_last_user_profile')):
            return candidates['Food Category'].tolist()[:top_k]

        # if user_profile not given explicitly but we have a last one, use it
        if user_profile is None and hasattr(self, '_last_user_profile'):
            user_profile = self._last_user_profile

        # Re-score candidates by proximity to user's macro targets
        try:
            user_vec = np.array([
                user_profile.get('calories', 0),
                user_profile.get('protein', 0),
                user_profile.get('carbs', 0),
                user_profile.get('fat', 0),
            ], dtype=float)

            # Build candidate macro vectors
            cand_macros = candidates[['Calories', 'Protein', 'Carbs', 'Fat']].fillna(0.0).values.astype(float)

            # compute euclidean distance in macro-space (smaller = better)
            macro_dist = np.linalg.norm((cand_macros - user_vec.reshape(1, -1)), axis=1)

            # we'll combine original KNN distance (if available) and macro distance
            # normalize both to [0,1]
            knn_dist = distances[0][1:1 + len(candidates)] if distances.shape[1] > 1 else np.zeros(len(candidates))
            if len(knn_dist) != len(macro_dist):
                knn_norm = np.interp(range(len(macro_dist)), [0, len(macro_dist) - 1], [0, 1])
            else:
                knn_norm = (knn_dist - np.min(knn_dist)) / (np.ptp(knn_dist) + 1e-9)

            macro_norm = (macro_dist - np.min(macro_dist)) / (np.ptp(macro_dist) + 1e-9)

            # weighting: give primary importance to food similarity (knn) but favor macros moderately
            alpha = 0.7  # weight for knn distance
            beta = 0.3   # weight for macro distance
            combined_score = alpha * knn_norm + beta * macro_norm

            # sort ascending (lower combined_score = better)
            order = np.argsort(combined_score)
            sorted_names = candidates['Food Category'].values[order].tolist()
            return sorted_names[:top_k]
        except Exception:
            # if anything fails, fallback to the raw knn candidates
            return candidates['Food Category'].tolist()[:top_k]

    # ---------------------------
    # Decision tree explanation
    # ---------------------------
    def train_decision_tree(self):
        df = self.df.copy()
        df['high_protein'] = (df['Protein'] > 15).astype(int)
        df['low_sugar'] = (df['Sugar'] < 8).astype(int)
        df['high_fiber'] = (df['Fiber'] > 4).astype(int)
        df['high_fat'] = (df['Fat'] > 20).astype(int)

        features_tree = ['Calories', 'high_protein', 'low_sugar', 'high_fiber', 'high_fat', 'satiety_index', 'activity_factor']
        df_encoded = pd.get_dummies(df[features_tree + ['profil', 'objective']], drop_first=True)

        y = df['role'] if 'role' in df.columns else np.zeros(len(df_encoded))
        self.dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=42, class_weight='balanced')
        self.dt.fit(df_encoded, y)

        joblib.dump(self.dt, "models/decision_tree.pkl")
        joblib.dump(list(df_encoded.columns), "models/tree_features.pkl")
        print("Decision tree entraîné et sauvegardé.")

    def get_tree_rules(self):
        if not os.path.exists("models/decision_tree.pkl"):
            return "Arbre non entraîné."
        dt = joblib.load("models/decision_tree.pkl")
        features = joblib.load("models/tree_features.pkl")
        return export_text(dt, feature_names=features, max_depth=5)

    # ---------------------------
    # Profil utilisateur runtime
    # ---------------------------
    def build_personal_tree(self, user_profile):
        required = ["weight", "height", "age", "gender", "activity", "objective"]
        for k in required:
            if k not in user_profile:
                raise ValueError(f"user_profile manque la clé '{k}'")

        if not all(k in user_profile for k in ["calories", "protein", "carbs", "fat"]):
            bmr = calculate_bmr(user_profile["weight"], user_profile["height"], user_profile["age"], user_profile["gender"])
            activity_map_display = {'sedentaire': "Sédentaire", 'leger': "Léger", 'modere': "Modéré", 'intense': "Intense", 'athlete': "Athlète"}
            activity_display = activity_map_display.get(user_profile["activity"], "Modéré")
            tdee = calculate_tdee(bmr, activity_display)
            macros = get_macro_targets(tdee, user_profile["objective"], user_profile["weight"])
            user_profile["calories"] = macros["calories"]
            user_profile["protein"] = macros["protein"]
            user_profile["carbs"] = macros["carbs"]
            user_profile["fat"] = macros["fat"]

        df = self.df.copy()
        df["user_weight"] = float(user_profile["weight"])
        df["user_height"] = float(user_profile["height"])
        df["user_age"] = float(user_profile["age"])
        df["user_is_male"] = 1 if str(user_profile["gender"]).lower().startswith("h") or str(user_profile["gender"]).lower().startswith("m") else 0
        mapping = {'sedentaire': 1.2, 'leger': 1.4, 'modere': 1.55, 'intense': 1.8, 'athlete': 2.0}
        df["user_activity_factor"] = mapping.get(user_profile["activity"], 1.55)
        obj_map = {'perte': -1, 'maintien': 0, 'gain': 1}
        df["user_objective_val"] = obj_map.get(user_profile["objective"], 0)
        df["user_calories"] = float(user_profile["calories"])
        df["user_protein"] = float(user_profile["protein"])
        df["user_carbs"] = float(user_profile["carbs"])
        df["user_fat"] = float(user_profile["fat"])

        numeric_food_cols = [
            "Calories", "Protein", "Carbs", "Fat", "Fiber", "Sugar",
            "density_kcal_100g", "satiety_index", "balance_score",
            # nouvelles colonnes dynamiques :
            "rel_protein", "rel_fat", "rel_carbs", "rel_cal",
            "sugar_penalty", "fiber_score", "dyn_density"
        ]
        for c in numeric_food_cols:
            if c not in df.columns:
                df[c] = 0.0

        # 1) Ratios dynamiques en fonction des cibles du user
        df["rel_protein"] = df["Protein"] / user_profile["protein"]  # % de la cible protéine
        df["rel_fat"] = df["Fat"] / user_profile["fat"]
        df["rel_carbs"] = df["Carbs"] / user_profile["carbs"]
        df["rel_cal"] = df["Calories"] / user_profile["calories"]

        # 2) Importance dynamique du sucre selon objectif
        if user_profile["objective"] == "maintien":
            df["sugar_penalty"] = df["Sugar"] * 0.3
        elif user_profile["objective"] == "gain":
            df["sugar_penalty"] = df["Sugar"] * 0.1
        else:  # perte
            df["sugar_penalty"] = df["Sugar"] * 1.2

        # 3) Fibre pondérée selon objectif
        if user_profile["objective"] == "perte":
            df["fiber_score"] = df["Fiber"] * 1.4
        elif user_profile["objective"] == "maintien":
            df["fiber_score"] = df["Fiber"] * 1.0
        else:  # gain
            df["fiber_score"] = df["Fiber"] * 0.7

        # 4) Densité kcal selon activité
        activity_boost = {
            "sedentaire": 0.9,
            "leger": 1.0,
            "modere": 1.1,
            "intense": 1.2,
            "athlete": 1.4
        }
        df["dyn_density"] = df["density_kcal_100g"] * activity_boost[user_profile["activity"]]

        X = df.select_dtypes(include=["number"]).copy()

        if "role" not in df.columns:
            temp = df.copy()
            temp["prot_ratio"] = temp["Protein"] * 4 / temp["Calories"].replace(0, 1)
            temp["role"] = "Neutre"
            temp.loc[(temp['prot_ratio'] > 0.22), "role"] = "Privilégier"
            temp.loc[(temp['Fat'] > 25), "role"] = "Modération"
            temp.loc[(temp['Sugar'] > 12), "role"] = "Éviter"
            y = temp["role"]
        else:
            y = df["role"]

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=8, class_weight="balanced", random_state=42)
        tree.fit(X, y)

        self.personal_tree = tree
        self.personal_tree_features = X.columns.tolist()
        os.makedirs("models", exist_ok=True)
        joblib.dump(tree, os.path.join("models", f"personal_tree_{user_profile['activity']}_{user_profile['objective']}.pkl"))
        joblib.dump(self.personal_tree_features, os.path.join("models", f"tree_features_{user_profile['activity']}_{user_profile['objective']}.pkl"))

        self._last_user_profile = user_profile.copy()
        return tree

    def get_personal_tree_rules(self, profil=None, objective=None):
        from sklearn.tree import export_text
        if not hasattr(self, "personal_tree") or self.personal_tree is None:
            return "❌ Aucun arbre personnalisé n'est construit.\nClique sur 'Appliquer le profil' → puis 'Construire l’arbre'."
        if not hasattr(self, "personal_tree_features") or self.personal_tree_features is None:
            return "❌ Aucune liste de features enregistrée pour l’arbre personnalisé."
        try:
            rules = export_text(self.personal_tree, feature_names=self.personal_tree_features)
            return rules
        except Exception as e:
            return f"❌ Erreur lors de l’export des règles : {e}"

    def get_role_subset(self, role_target, profil=None, objective=None):
        profil = profil or self.current_profile
        objective = objective or self.current_objective
        df = self.df.copy()
        df["prot_ratio"] = (df["Protein"] * 4 / df["Calories"].replace(0, 1)).fillna(0)
        df["carb_ratio"] = (df["Carbs"] * 4 / df["Calories"].replace(0, 1)).fillna(0)
        df["fat_ratio"] = (df["Fat"] * 9 / df["Calories"].replace(0, 1)).fillna(0)

        if objective == "perte":
            conditions = [
                (df['prot_ratio'] > 0.25) & (df['Fiber'] > 3) & (df['Sugar'] < 5),
                (df['Fat'] > 22) | (df['Calories'] > 420),
                (df['Sugar'] > 15)
            ]
            choices = ["Privilégier", "Modération", "Éviter"]
        elif objective == "gain":
            conditions = [
                (df['Calories'] >= 300) & (df['Protein'] >= 12),
                (df['Fat'] < 12),
                (df['Sugar'] > 18)
            ]
            choices = ["Privilégier", "Modération", "Éviter"]
        else:
            conditions = [
                (df['prot_ratio'] > 0.22),
                (df['Fat'] > 25),
                (df['Sugar'] > 12)
            ]
            choices = ["Privilégier", "Modération", "Éviter"]

        df["role"] = np.select(conditions, choices, default="Neutre")
        subset = df[(df["profil"] == profil) & (df["objective"] == objective) & (df["role"] == role_target)]
        if subset.empty:
            subset = df[df["role"] == role_target]
        return subset

    def set_user_profile(self, weight, height, age, gender, activity, objective):
        self.user_weight = weight
        self.user_height = height
        self.user_age = age
        self.user_gender = gender
        self.user_activity = activity
        self.user_objective = objective

        bmr = calculate_bmr(weight, height, age, gender)
        tdee = calculate_tdee(bmr, activity)
        targets = get_macro_targets(tdee, objective, weight)

        self.df['user_bmr'] = bmr
        self.df['user_tdee'] = tdee
        self.df['user_calories'] = targets["calories"]
        self.df['user_protein'] = targets["protein"]
        self.df['user_fat'] = targets["fat"]
        self.df['user_carbs'] = targets["carbs"]

        self.df['user_weight'] = weight
        self.df['user_height'] = height
        self.df['user_age'] = age
        self.df['user_gender'] = 1 if gender.lower().startswith("h") else 0

        mapping = {"Sédentaire": 1.2, "Léger": 1.375, "Modéré": 1.55, "Intense": 1.725, "Athlète": 1.9}
        self.df['user_activity_factor'] = mapping.get(activity, 1.55)

        obj_enc = {"perte": 0, "maintien": 1, "gain": 2}
        self.df['user_objective_enc'] = obj_enc.get(objective, 1)

        print("=== Profil COMPLET appliqué ===")
        print(f"- Poids : {weight} kg")
        print(f"- Taille : {height} cm")
        print(f"- Âge : {age}")
        print(f"- Genre : {gender}")
        print(f"- Activité : {activity}")
        print(f"- Objectif : {objective}")
        print(f"> Calories quotidiennes recommandées : {targets['calories']}")

    def save_all_models(self):
        if self.rf_reg is not None:
            joblib.dump(self.rf_reg, "models/rf_balance.pkl")
        if self.rf_clf is not None:
            joblib.dump(self.rf_clf, "models/rf_classifier.pkl")
        if self.knn is not None:
            joblib.dump(self.knn, "models/knn_recommender.pkl")
        if self.dt is not None:
            joblib.dump(self.dt, "models/decision_tree.pkl")
        print("Tous les modèles (disponibles) ont été sauvegardés.")
