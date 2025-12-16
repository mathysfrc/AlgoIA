import os
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text


class NutriAI:
    """
    Système 100% personnalisé basé sur le profil utilisateur complet.
    Chaque prédiction intègre : âge, poids, taille, genre, activité, objectif.
    """

    UNHEALTHY_BLACKLIST = [
        'mcdo', 'mcdonald', 'burger king', 'kfc', 'pizza hut', 'domino',
        'fast food', 'soda', 'coca', 'pepsi', 'fanta', 'sprite',
        'chips', 'doritos', 'cheetos', 'candy', 'bonbon', 'chocolat industriel',
        'nuggets', 'fries', 'frites', 'donut', 'croissant industriel'
    ]

    def __init__(self, data_path="data/processed_nutrition.csv"):
        self.df = pd.read_csv(data_path)

        self.base_features = [
            'Calories', 'Protein', 'Carbs', 'Fat', 'Fiber',
            'Sugar', 'Water', 'density_kcal_100g', 'satiety_index'
        ]

        # PRofil utilisateur complet générique => défini de manière générale
        self.user_profile = {
            'weight': 70,
            'height': 175,
            'age': 30,
            'gender': 'Homme',
            'activity': 'Modéré',
            'objective': 'maintien',
            'bmr': 0,
            'tdee': 0,
            'bmi': 0,
            'target_protein': 0,
            'target_carbs': 0,
            'target_fat': 0,
            'target_calories': 0,
            # Encodages pour ML
            'gender_encoded': 1,  # 1=Homme, 0=Femme
            'activity_encoded': 2,  # 0-4
            'objective_encoded': 1,  # 0=perte, 1=maintien, 2=gain
            'age_group': 1  # 0=jeune, 1=adulte, 2=senior
        }

        self._initialize_columns()
        # Pré-processing
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        self.class_order = ['Privilégier', 'Modération', 'Neutre', 'Éviter']

        self.rf_reg = None
        self.rf_clf = None
        self.knn = None
        self.dt = None

        os.makedirs("models", exist_ok=True)

    def _initialize_columns(self):
        # Si les colonnes existent pas, on donne une valeur
        if 'profil' not in self.df.columns:
            self.df['profil'] = 'modere'
        if 'objective' not in self.df.columns:
            self.df['objective'] = 'maintien'

        # Défini des valeurs selon calculs
        self.df['prot_ratio'] = self.df['Protein'] * 4 / self.df['Calories']
        self.df['carb_ratio'] = self.df['Carbs'] * 4 / self.df['Calories']
        self.df['fat_ratio'] = self.df['Fat'] * 9 / self.df['Calories']

    # Calcul mathématique fitness du profil complet pour savoir les besoins
    def set_user_profile(self, weight, height, age, gender, activity, objective):
        """
        Calcule tous les paramètres du profil utilisateur.
        Crée des encodages numériques pour intégration dans les modèles ML.
        """
        # 1. Stocker paramètres de base
        self.user_profile.update({
            'weight': weight,
            'height': height,
            'age': age,
            'gender': gender,
            'activity': activity,
            'objective': objective
        })

        # 2. Calculer BMR
        if gender.lower() == "homme":
            bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
        else:
            bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)

        # 3. Calculer TDEE
        activity_multipliers = {
            "Sédentaire": 1.2,
            "Léger": 1.375,
            "Modéré": 1.55,
            "Intense": 1.725,
            "Athlète": 1.9
        }
        tdee = bmr * activity_multipliers.get(activity, 1.55)

        # 4. Calculer BMI
        bmi = weight / ((height / 100) ** 2)

        # 5. Ajuster selon objectif => mathématiques
        if objective == "perte":
            target_cal = tdee - 500
            protein_ratio = 2.2
            fat_ratio = 0.8
        elif objective == "gain":
            target_cal = tdee + 500
            protein_ratio = 1.8
            fat_ratio = 1.0
        else:
            target_cal = tdee
            protein_ratio = 2.0
            fat_ratio = 0.9

        target_protein = weight * protein_ratio
        target_fat = weight * fat_ratio
        target_carbs = (target_cal - (target_protein * 4 + target_fat * 9)) / 4

        # 6. Encodages numérique
        gender_encoded = 1 if gender.lower() == "homme" else 0

        activity_map = {
            "Sédentaire": 0,
            "Léger": 1,
            "Modéré": 2,
            "Intense": 3,
            "Athlète": 4
        }
        activity_encoded = activity_map.get(activity, 2)

        # Encodage des objectifs
        objective_map = {"perte": 0, "maintien": 1, "gain": 2}
        objective_encoded = objective_map.get(objective, 1)

        # Encodage des groupes d'ages
        if age < 25:
            age_group = 0  # Jeune
        elif age < 50:
            age_group = 1  # Adulte
        else:
            age_group = 2  # Senior

        # 7. Mettre à jour profil complet avec les datas réelles du profil
        self.user_profile.update({
            'bmr': bmr,
            'tdee': tdee,
            'bmi': bmi,
            'target_protein': target_protein,
            'target_carbs': target_carbs,
            'target_fat': target_fat,
            'target_calories': target_cal,
            'gender_encoded': gender_encoded,
            'activity_encoded': activity_encoded,
            'objective_encoded': objective_encoded,
            'age_group': age_group
        })

        print(f"✓ Profil utilisateur configuré:")
        print(f"  Genre: {gender} (encoded: {gender_encoded})")
        print(f"  Âge: {age} ans (groupe: {age_group})")
        print(f"  Poids: {weight}kg | Taille: {height}cm | IMC: {bmi:.1f}")
        print(f"  Activité: {activity} (encoded: {activity_encoded})")
        print(f"  Objectif: {objective} (encoded: {objective_encoded})")
        print(f"  BMR: {int(bmr)} kcal | TDEE: {int(tdee)} kcal")
        print(f"  Cible: {int(target_cal)} kcal | P:{int(target_protein)}g G:{int(target_carbs)}g L:{int(target_fat)}g")

        # On retourne user_profile qu'on utilise partout par la suite
        return self.user_profile

    # filtrage + scoring
    def _filter_unhealthy_foods(self, df):
        # Exclut aliments malsains
        mask = df['Food Category'].str.lower().apply(
            lambda x: not any(bad in x for bad in self.UNHEALTHY_BLACKLIST)
        )
        filtered = df[mask].copy()
        removed = len(df) - len(filtered)
        if removed > 0:
            print(f"{removed} aliments malsains exclus")
        return filtered

    def _add_user_profile_features(self, df):
        """
        Ajoute les paramètres du profil utilisateur comme colonnes.
        C'est la clé pour que les modèles utilisent le profil complet !
        """
        profile = self.user_profile

        # Ajouter toutes les infos du profil
        df['user_weight'] = profile['weight']
        df['user_height'] = profile['height']
        df['user_age'] = profile['age']
        df['user_bmi'] = profile['bmi']
        df['user_bmr'] = profile['bmr']
        df['user_tdee'] = profile['tdee']
        df['user_gender'] = profile['gender_encoded']
        df['user_activity'] = profile['activity_encoded']
        df['user_objective'] = profile['objective_encoded']
        df['user_age_group'] = profile['age_group']

        # Ajouter les cibles nutritionnelles
        df['user_target_calories'] = profile['target_calories']
        df['user_target_protein'] = profile['target_protein']
        df['user_target_carbs'] = profile['target_carbs']
        df['user_target_fat'] = profile['target_fat']

        return df

    def _calculate_personalized_scores(self, df):
        """Calcule scores basés sur le profil => mathématiques"""
        target = self.user_profile

        # Distances normalisées
        # calculent des scores d’adéquation personnalisés
        # comparent chaque aliment aux besoins de l’utilisateur
        # sont normalisés entre 0 et 1
        # servent à :
        # - scorer
        # - classer
        # - expliquer
        #-  recommander
        df['protein_fit'] = 1 - abs(df['Protein'] - target['target_protein'] / 6 ) / (target['target_protein'] / 6 + 1)
        df['carbs_fit'] = 1 - abs(df['Carbs'] - target['target_carbs'] / 6) / (target['target_carbs'] / 6 + 1)
        df['fat_fit'] = 1 - abs(df['Fat'] - target['target_fat'] / 6) / (target['target_fat'] / 6 + 1)
        df['calorie_fit'] = 1 - abs(df['Calories'] - target['target_calories'] / 4) / (
                    target['target_calories'] / 4 + 1)

        for col in ['protein_fit', 'carbs_fit', 'fat_fit', 'calorie_fit']:
            df[col] = df[col].clip(0, 1)

        # Score personnalisé selon objectif
        if target['objective'] == 'perte':
            df['user_score'] = (
                    0.35 * df['protein_fit'] +
                    0.25 * df['calorie_fit'] +
                    0.20 * (df['Fiber'] / (df['Fiber'].max() + 0.1)) +
                    0.15 * (1 - df['Sugar'] / (df['Sugar'].max() + 0.1)) +
                    0.05 * (df['satiety_index'] / (df['satiety_index'].max() + 0.1))
            )
        elif target['objective'] == 'gain':
            df['user_score'] = (
                    0.30 * df['protein_fit'] +
                    0.30 * df['carbs_fit'] +
                    0.25 * (df['Calories'] / (df['Calories'].max() + 0.1)) +
                    0.15 * df['calorie_fit']
            )
        else:
            df['user_score'] = (
                    0.30 * df['protein_fit'] +
                    0.25 * df['carbs_fit'] +
                    0.25 * df['fat_fit'] +
                    0.20 * df['calorie_fit']
            )

        # User_score adapté au profil retourné dans le dataset
        return df

   # Classification
    def classify_food_role_personalized(self):
        # Classification avec seuils dynamiques
        # On récupère le profil et les scores et on exclu la malbouffe
        df = self._filter_unhealthy_foods(self.df.copy())
        df = self._add_user_profile_features(df)
        df = self._calculate_personalized_scores(df)

        target = self.user_profile

        # Seuils dynamiques
        protein_threshold = target['target_protein'] / 6
        carbs_threshold = target['target_carbs'] / 6
        fat_threshold = target['target_fat'] / 6
        cal_threshold = target['target_calories'] / 4

        # Défini les règles pour chaque objectif
        if target['objective'] == 'perte':
            conditions = [
                # Privilégier
                (df['Protein'] >= protein_threshold * 0.8) &
                (df['Calories'] <= cal_threshold * 1.2) &
                (df['Fiber'] >= 3) &
                (df['Sugar'] <= 8),

                # Modération
                (df['Fat'] > fat_threshold * 1.5) |
                ((df['Sugar'] > 8) & (df['Sugar'] <= 15)),

                # éviter
                (df['Calories'] > cal_threshold * 1.8) |
                (df['Sugar'] > 15) |
                (df['Fat'] > fat_threshold * 2)
            ]

        elif target['objective'] == 'gain':
            conditions = [
                # Privilégier
                (df['Calories'] >= cal_threshold * 1.2) &
                (df['Protein'] >= protein_threshold * 0.7) &
                (df['Carbs'] >= carbs_threshold * 0.8),

                # Modération
                (df['Calories'] < cal_threshold * 0.8),

                # éviter
                (df['Sugar'] > 20) & (df['Protein'] < protein_threshold * 0.5)
            ]

        else:  # maintien
            conditions = [
                # Privilégier
                (df['Protein'] >= protein_threshold * 0.7) &
                (df['Protein'] <= protein_threshold * 1.3) &
                (df['Calories'] >= cal_threshold * 0.8) &
                (df['Calories'] <= cal_threshold * 1.2),

                # Modération
                (df['Fat'] > fat_threshold * 1.5) |
                (df['Sugar'] > 12),

                # éviter
                (df['Calories'] > cal_threshold * 2) |
                (df['Sugar'] > 20)
            ]

        # Classification via choix de seuils
        choices = ['Privilégier', 'Modération', 'Éviter']
        df['role'] = np.select(conditions, choices, default='Neutre')

        self.df = df

        print(f"✓ Classification personnalisée :")
        # Calcul le %  d'aliment ayant un des rôles
        for role in self.class_order:
            count = len(df[df['role'] == role])
            pct = count / len(df) * 100
            print(f"  {role}: {count} ({pct:.1f}%)")

        return df

   # KN
    def train_knn_personalized(self):
        """
        KNN qui utilise directement les paramètres du profil utilisateur.
        Les aliments similaires sont ceux adaptés au même profil
        """
        # On récupère le profil et les scores et on exclu la malbouffe
        df = self._filter_unhealthy_foods(self.df.copy())
        df = self._add_user_profile_features(df)
        df = self._calculate_personalized_scores(df)

        # Features complètes incluant tout le profil, on se base sur toutes les features du profil
        features = self.base_features + [
            'user_score', 'protein_fit', 'carbs_fit', 'fat_fit',
            'user_weight', 'user_age', 'user_bmi',
            'user_gender', 'user_activity', 'user_objective',
            'user_target_protein', 'user_target_carbs', 'user_target_fat'
        ]

        # Normalisation
        # construis la matrice d’entrée du modèle on enleves les NaN
        X = df[features].fillna(0)
        # Normalise les données pour résultat + précis
        X_scaled = self.scaler.fit_transform(X)

        # Calcul des distances euclidiennes
        # Sélection des 6 plus proches voisins (K=6)
        self.knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
        # Recommandations basées sur ces voisins
        self.knn.fit(X_scaled)

        # Utilise le meilleur modèle par rapport au profil
        joblib.dump(self.knn, "models/knn_personalized.pkl")
        joblib.dump(self.scaler, "models/scaler_personalized.pkl")
        joblib.dump(features, "models/knn_features.pkl")

        print(f"✓ KNN entraîné avec {len(features)} features incluant profil complet")

    def recommend_similar_personalized(self, food_name):
        """Recommandations basées UNIQUEMENT sur similarité nutritionnelle réelle (calories, protéines, glucides, etc.)"""
        # On récupère le profil et les scores et on exclu la malbouffe
        df = self._filter_unhealthy_foods(self.df.copy())
        df = self._add_user_profile_features(df)
        df = self._calculate_personalized_scores(df)

        # Trouver l'aliment de référence par rapport a food category dans le dataset
        row = df[df['Food Category'].str.contains(food_name, case=False, na=False)]
        if row.empty:
            return []

        # Features pour similarité : Uniquement macronutriments principaux (calories, protéines, glucides, lipides)
        # On exclut Water, density, satiety_index qui peuvent brouiller la similarité
        similarity_features = ['Calories', 'Protein', 'Carbs', 'Fat']

        # Créer un KNN local avec uniquement les valeurs nutritionnelles principales pour trouver des aliments similaires
        X_similarity = df[similarity_features].fillna(0)
        similarity_scaler = StandardScaler()
        X_similarity_scaled = similarity_scaler.fit_transform(X_similarity)
        
        similarity_knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
        similarity_knn.fit(X_similarity_scaled)

        # Trouve les aliments les plus proches basés sur similarité nutritionnelle réelle
        _, indices = similarity_knn.kneighbors([X_similarity_scaled[row.index[0]]])

        recommendations = []
        reference_food = df.iloc[row.index[0]]
        
        for idx in indices[0][1:]:
            food = df.iloc[idx]
            
            # Filtrer les aliments trop différents (écart de plus de 50% sur les calories ou protéines principales)
            # pour éviter les recommandations non pertinentes
            cal_diff_ratio = abs(food['Calories'] - reference_food['Calories']) / max(reference_food['Calories'], 1)
            protein_diff_ratio = abs(food['Protein'] - reference_food['Protein']) / max(reference_food['Protein'], 1)
            
            # Ne garder que les aliments avec une différence raisonnable (max 80% d'écart)
            if cal_diff_ratio > 0.8 or protein_diff_ratio > 0.8:
                continue
            
            # Renvoie ces infos pour chaque voisin
            recommendations.append({
                'name': food['Food Category'],
                'role': food['role'],
                'user_score': food['user_score'],
                'calories': food['Calories'],
                'protein': food['Protein']
            })

        return recommendations

    # Arbre de décision
    def train_decision_tree_personalized(self):
        """
        Arbre qui utilise les paramètres du profil et des features adaptatives.
        """
        # On récupère le profil et les scores et on exclu la malbouffe
        df = self._filter_unhealthy_foods(self.df.copy())
        df = self._add_user_profile_features(df)
        df = self._calculate_personalized_scores(df)

        target = self.user_profile

        # Features adaptative basées sur le profil selon le nombres de repas (6,6,6,4)
        protein_threshold = target['target_protein'] / 6
        carbs_threshold = target['target_carbs'] / 6
        fat_threshold = target['target_fat'] / 6
        cal_threshold = target['target_calories'] / 4

        # Transforme les features en binaire s'ils correspondent au seuil défini 1 au sinon 0
        df['meets_protein_need'] = (df['Protein'] >= protein_threshold * 0.8).astype(int)
        df['meets_carbs_need'] = (df['Carbs'] >= carbs_threshold * 0.8).astype(int)
        df['within_calorie_target'] = (
                (df['Calories'] >= cal_threshold * 0.7) &
                (df['Calories'] <= cal_threshold * 1.3)
        ).astype(int)
        df['low_sugar'] = (df['Sugar'] < 10).astype(int)
        df['high_fiber'] = (df['Fiber'] > 4).astype(int)
        df['healthy_fat_ratio'] = ((df['Fat'] >= fat_threshold * 0.5) &
                                   (df['Fat'] <= fat_threshold * 1.5)).astype(int)

        # Features incluant directement le profil utilisateur
        features_tree = [
            # Features adaptatives
            'meets_protein_need', 'meets_carbs_need', 'within_calorie_target',
            'low_sugar', 'high_fiber', 'healthy_fat_ratio',
            'user_score', 'satiety_index',
            # Paramètres du profil utilisateur
            'user_age', 'user_bmi', 'user_gender',
            'user_activity', 'user_objective',
            'user_target_protein', 'user_target_calories'
        ]

        # Remplace les NaN par 0
        X = df[features_tree].fillna(0)
        y = df['role'] # Privilégier, Modération, Neutre, Eviter

        # Identifier les classes réellement présentes dans les données (Privilégié, Modération...)
        present_classes = sorted(y.unique())

        #  Au moins 2 classes nécessaires pour un arbre
        if len(present_classes) < 2:
            print(f"ATTENTION: Seulement {len(present_classes)} classe(s) présente(s): {present_classes}")
            print(f"   Les règles de classification sont trop strictes pour ce profil.")
            print(f"   L'arbre ne sera pas entraîné (nécessite au moins 2 classes).")

            # Créer un arbre minimal pour compatibilité, garantit que un objet modèle existe toujours même quand l’arbre “normal” est impossible
            from sklearn.dummy import DummyClassifier
            self.dt = DummyClassifier(strategy='most_frequent')
            self.dt.fit(X, y)

            # Chargement de l'abre entrainé
            joblib.dump(self.dt, "models/decision_tree_personalized.pkl")
            joblib.dump(features_tree, "models/tree_features_personalized.pkl")
            joblib.dump(present_classes, "models/tree_classes.pkl")

            return f"Arbre non entraîné: une seule classe présente ({present_classes[0]})\nLes règles de classification nécessitent un ajustement."

        # Encoder uniquement les classes présentes
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(present_classes)
        y_encoded = self.label_encoder.transform(y)

        self.dt = DecisionTreeClassifier(
            # max_depth=8 : Profondeur maximale
            # min_samples_leaf=15 : Minimum d'échantillons par feuille
            # class_weight='balanced' : Équilibre les classes
            max_depth=8,
            min_samples_leaf=15,
            random_state=42,
            class_weight='balanced'
        )
        self.dt.fit(X, y_encoded)

        joblib.dump(self.dt, "models/decision_tree_personalized.pkl")
        joblib.dump(features_tree, "models/tree_features_personalized.pkl")
        joblib.dump(self.label_encoder, "models/label_encoder.pkl")
        joblib.dump(present_classes, "models/tree_classes.pkl")  # Sauvegarder les classes présentes

        print(f"✓ Arbre entraîné avec {len(features_tree)} features")
        print(f"  Features profil utilisateur: age, bmi, genre, activité, objectif, cibles nutritionnelles")
        print(f"  Classes présentes dans ce profil: {', '.join(present_classes)}")

        # Retourner règles avec classes RÉELLEMENT présentes
        rules_text = export_text(
            self.dt,
            feature_names=features_tree,
            class_names=present_classes,  # Utiliser classes présentes
            max_depth=6 # Max 6 niveaux
        )

        return rules_text

    def get_tree_rules(self):
        """Récupère les règles avec les bons noms de classes"""
        if not os.path.exists("models/decision_tree_personalized.pkl"):
            return "Arbre non entraîné."

        dt = joblib.load("models/decision_tree_personalized.pkl")
        features = joblib.load("models/tree_features_personalized.pkl")

        # Charger les classes qui étaient présentes lors de l'entraînement
        if os.path.exists("models/tree_classes.pkl"):
            class_names = joblib.load("models/tree_classes.pkl")
        else:
            # Fallback sur l'ordre par défaut si fichier absent
            class_names = self.class_order

        return export_text(
            dt,
            feature_names=features,
            class_names=class_names,
            max_depth=6
        )

    # Sauvegarde des modèles

    def save_all_models(self):
        """Sauvegarde tous les modèles"""
        if self.knn:
            joblib.dump(self.knn, "models/knn_personalized.pkl")
        if self.dt:
            joblib.dump(self.dt, "models/decision_tree_personalized.pkl")
        if self.scaler:
            joblib.dump(self.scaler, "models/scaler_personalized.pkl")
        if self.label_encoder:
            joblib.dump(self.label_encoder, "models/label_encoder.pkl")

        joblib.dump(self.user_profile, "models/user_profile.pkl")
        print("✓ Tous les modèles sauvegardés")