import os
import time
import pandas as pd
from collections import Counter
import joblib
from matplotlib import pyplot as plt

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns


class ModelEvaluator:
    """Évaluation rapide, robuste et scientifiquement défendable des modèles ML"""

    def __init__(self, df, user_profile):
        self.df = df
        self.user_profile = user_profile
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.results = {}
        self.best_models = {}

  # Outils interne

    def _get_safe_cv(self, y, max_cv=5):
        """Détermine un nombre de folds valide selon la classe la plus rare"""
        min_class_size = min(Counter(y).values())
        return max(2, min(max_cv, min_class_size))

   # Préparation des données
    def prepare_data(self):
        """Nettoyage, filtrage et normalisation"""

        # Suppression des classes trop rares
        class_counts = Counter(self.df['role'])
        valid_classes = [cls for cls, cnt in class_counts.items() if cnt >= 2]

        if len(valid_classes) < len(class_counts):
            print("Classes supprimées (trop rares):",
                  set(class_counts) - set(valid_classes))

        df_filtered = self.df[self.df['role'].isin(valid_classes)].copy()

        if df_filtered['role'].nunique() < 2:
            raise ValueError("Évaluation impossible : une seule classe restante")

        features = [
            'Calories', 'Protein', 'Carbs', 'Fat', 'Fiber', 'Sugar',
            'density_kcal_100g', 'satiety_index', 'user_score',
            'protein_fit', 'carbs_fit', 'fat_fit', 'calorie_fit',
            'user_weight', 'user_age', 'user_bmi', 'user_gender',
            'user_activity', 'user_objective', 'user_target_protein',
            'user_target_carbs', 'user_target_calories'
        ]

        # Séparation X, Y
        X = df_filtered[features].fillna(0)
        y = df_filtered['role']

        # Encode les rôles (Privilégier...)
        y_encoded = self.label_encoder.fit_transform(y)

        # Sécurisation du split, taille de la classe la plus rare
        min_class_size = min(Counter(y_encoded).values())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, # X ce que le modèle voit, y ce qu'il doit prédire
            # 80/20 learning/test
            test_size=0.2,
            random_state=42,
            # force le split à garder les mêmes proportions de classes dans train et test uniquement si +2 classes
            stratify=y_encoded if min_class_size >= 2 else None
        )

        # Scaling : Met toutes les features sur une échelle comparable
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    # fine tunning des modèles
    # cherche la meilleure configuration possible de l’arbre afin d’avoir de meilleures prédictions et moins d’overfitting
    def tune_decision_tree(self, X, y):
        print("\nTuning Decision Tree...")

        param_grid = {
            # Plusieurs choix
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 10],
            'min_samples_leaf': [5, 15],
            'criterion': ['gini', 'entropy']
        }

        model = DecisionTreeClassifier(random_state=42)
        cv_folds = self._get_safe_cv(y, max_cv=3)

        # Pour chaque combinaison :
        # entraînement sur cv-1 folds
        # test sur le fold restant
        # score F1 pondéré (équilibre entre classes)
        gs = GridSearchCV(
            model, param_grid,
            cv=cv_folds,
            scoring='f1_weighted', # Precision + Recall
            n_jobs=-1
        )

        start = time.time()
        gs.fit(X, y)
        elapsed = time.time() - start

        return gs.best_estimator_, gs.best_params_, elapsed

    def tune_random_forest(self, X, y):
        print("\nTuning Random Forest...")

        # n_estimators : 80-150 arbres
        # max_depth : 10-20 niveaux
        # min_samples_split : 2-10
        # min_samples_leaf : 1-2
        param_grid = {
            'n_estimators': [80, 150],
            'max_depth': [10, 20],
            'min_samples_split': [2, 10],
            'min_samples_leaf': [1, 2]
        }

        model = RandomForestClassifier(random_state=42)
        cv_folds = self._get_safe_cv(y, max_cv=3)

        gs = GridSearchCV(
            model, param_grid,
            cv=cv_folds,
            # Recall + Precision
            scoring='f1_weighted',
            n_jobs=-1
        )

        start = time.time()
        gs.fit(X, y)
        elapsed = time.time() - start

        return gs.best_estimator_, gs.best_params_, elapsed

    def tune_knn(self, X, y):
        print("\nTuning KNN...")

        param_grid = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }

        model = KNeighborsClassifier()
        cv_folds = self._get_safe_cv(y, max_cv=5)

        gs = GridSearchCV(
            model, param_grid,
            cv=cv_folds,
            scoring='f1_weighted',
            n_jobs=-1
        )

        start = time.time()
        gs.fit(X, y)
        elapsed = time.time() - start

        return gs.best_estimator_, gs.best_params_, elapsed

    def tune_gradient_boosting(self, X, y):
        print("\nTuning Gradient Boosting...")

        # Premier arbre fait des prédictions
        # Calcul des erreurs (résidus)
        # Nouvel arbre entraîné pour prédire ces erreurs
        # Répété 80-120 fois
        # Prédiction finale = somme de toutes les prédictions
        param_grid = {
            'n_estimators': [80, 120],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }

        model = GradientBoostingClassifier(random_state=42)
        # Max 3 fold pour la validation croisée
        cv_folds = self._get_safe_cv(y, max_cv=3)

        gs = GridSearchCV(
            model, param_grid,
            cv=cv_folds,
            scoring='f1_weighted',
            n_jobs=-1
        )

        start = time.time()
        gs.fit(X, y)
        elapsed = time.time() - start

        return gs.best_estimator_, gs.best_params_, elapsed

    # évaluation
    def evaluate_model(self, model, X_test, y_test, name):
        # Après le résultats de l'entrainement on évalue les modèles sur les mêmes datas
        start = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start

        return {
            'model_name': name,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'inference_time': inference_time,
            'classification_report': classification_report(
                y_test, y_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True
            ),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

    # Comparaison
    def compare_models(self):
        print("\n=== Comparaison des modèles ===")

        # Sécurité : dossier de sortie
        os.makedirs("models", exist_ok=True)

        # Préparation des données
        X_train, X_test, y_train, y_test = self.prepare_data()

        models = [
            ("Decision Tree", self.tune_decision_tree),
            ("Random Forest", self.tune_random_forest),
            ("KNN", self.tune_knn),
            ("Gradient Boosting", self.tune_gradient_boosting)
        ]

        # Entrainement + évaluation
        for model_name, tuner in models:
            print(f"\n Évaluation : {model_name}")

            model, best_params, train_time = tuner(X_train, y_train)
            res = self.evaluate_model(model, X_test, y_test, model_name)

            res["training_time"] = train_time
            res["best_params"] = best_params

            self.results[model_name] = res
            self.best_models[model_name] = model

            # Matrice de confusion
            cm = res["confusion_matrix"]
            cm_path = f"models/confusion_{model_name.lower().replace(' ', '_')}.png"

            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Matrice de confusion – {model_name}")
            plt.tight_layout()
            plt.savefig(cm_path)
            plt.close()

            res["confusion_path"] = cm_path

        # Rapport texte global
        report_path = "models/evaluation_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            for model_name, res in self.results.items():
                f.write(f"\n=== {model_name} ===\n")
                f.write(pd.DataFrame(res["classification_report"]).to_string())
                f.write("\n\n")

        # Tableau comparatif
        df_results = pd.DataFrame([
            {
                "Modèle": res["model_name"],
                "Accuracy": res["accuracy"],
                "F1-Score": res["f1_score"],
                "Precision": res["precision"],
                "Recall": res["recall"],
                "Temps Train (s)": res["training_time"],
                "Temps Inférence (s)": res["inference_time"]
            }
            for res in self.results.values()
        ])

        print("\n=== Résultats finaux ===")
        print(df_results.to_string(index=False))

        df_results.to_csv("models/model_comparison.csv", index=False)

        # Graphique comparatif F1-score
        plt.figure(figsize=(8, 4))
        sns.barplot(x="Modèle", y="F1-Score", data=df_results)
        plt.title("Comparaison des F1-Scores")
        plt.tight_layout()
        plt.savefig("models/f1_comparison.png")
        plt.close()

        return df_results

    # Sauvegarde des modèles
    def save_best_models(self):
        os.makedirs("models", exist_ok=True)

        for name, model in self.best_models.items():
            filename = f"models/{name.lower().replace(' ', '_')}_best.pkl"
            joblib.dump(model, filename)

        print("Meilleurs modèles sauvegardés")


# Point d'entrée => appelé lors d'une demande de re-évaluation dans app.py
def run_full_evaluation(df, user_profile):
    print("\n=== Démarrage Évaluation Accélérée ===")

    evaluator = ModelEvaluator(df, user_profile)
    comparison = evaluator.compare_models()
    evaluator.save_best_models()

    print("\n=== Fin de l'évaluation ===")
    return evaluator, comparison
