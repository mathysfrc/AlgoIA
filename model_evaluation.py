# model_evaluation.py - Évaluation complète et comparaison des modèles
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """
    Classe pour l'évaluation complète des modèles :
    - Fine-tuning des hyperparamètres
    - Métriques de performance
    - Comparaison entre modèles
    """

    def __init__(self, df, user_profile):
        self.df = df
        self.user_profile = user_profile
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.results = {}
        self.best_models = {}

    def prepare_data(self):
        """Prépare les données pour l'entraînement"""
        # Features complètes avec profil utilisateur
        features = [
            'Calories', 'Protein', 'Carbs', 'Fat', 'Fiber', 'Sugar',
            'density_kcal_100g', 'satiety_index', 'user_score',
            'protein_fit', 'carbs_fit', 'fat_fit', 'calorie_fit',
            'user_weight', 'user_age', 'user_bmi', 'user_gender',
            'user_activity', 'user_objective', 'user_target_protein',
            'user_target_carbs', 'user_target_calories'
        ]

        X = self.df[features].fillna(0)
        y = self.df['role']

        # Encoder les labels
        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)

        # Split données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Normaliser
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, features

    def tune_decision_tree(self, X_train, y_train):
        """Fine-tuning de l'arbre de décision avec GridSearchCV"""
        print("\n" + "=" * 60)
        print("🔧 FINE-TUNING: Decision Tree")
        print("=" * 60)

        param_grid = {
            'max_depth': [5, 8, 10, 12, 15],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [5, 10, 15, 20, 25],
            'criterion': ['gini', 'entropy'],
            'class_weight': ['balanced', None]
        }

        dt = DecisionTreeClassifier(random_state=42)

        grid_search = GridSearchCV(
            estimator=dt,
            param_grid=param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )

        start = time.time()
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start

        print(f"\n✓ Meilleurs paramètres: {grid_search.best_params_}")
        print(f"✓ Meilleur score F1 (CV): {grid_search.best_score_:.4f}")
        print(f"✓ Temps d'entraînement: {training_time:.2f}s")

        return grid_search.best_estimator_, grid_search.best_params_, training_time

    def tune_random_forest(self, X_train, y_train):
        """Fine-tuning de Random Forest"""
        print("\n" + "=" * 60)
        print("🔧 FINE-TUNING: Random Forest")
        print("=" * 60)

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', None]
        }

        rf = RandomForestClassifier(random_state=42)

        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=3,  # Moins de CV pour RF (plus lent)
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )

        start = time.time()
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start

        print(f"\n✓ Meilleurs paramètres: {grid_search.best_params_}")
        print(f"✓ Meilleur score F1 (CV): {grid_search.best_score_:.4f}")
        print(f"✓ Temps d'entraînement: {training_time:.2f}s")

        return grid_search.best_estimator_, grid_search.best_params_, training_time

    def tune_knn(self, X_train, y_train):
        """Fine-tuning de K-Nearest Neighbors"""
        print("\n" + "=" * 60)
        print("🔧 FINE-TUNING: K-Nearest Neighbors")
        print("=" * 60)

        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': [1, 2]
        }

        knn = KNeighborsClassifier()

        grid_search = GridSearchCV(
            estimator=knn,
            param_grid=param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )

        start = time.time()
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start

        print(f"\n✓ Meilleurs paramètres: {grid_search.best_params_}")
        print(f"✓ Meilleur score F1 (CV): {grid_search.best_score_:.4f}")
        print(f"✓ Temps d'entraînement: {training_time:.2f}s")

        return grid_search.best_estimator_, grid_search.best_params_, training_time

    def tune_gradient_boosting(self, X_train, y_train):
        """Fine-tuning de Gradient Boosting"""
        print("\n" + "=" * 60)
        print("🔧 FINE-TUNING: Gradient Boosting")
        print("=" * 60)

        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'subsample': [0.8, 0.9, 1.0]
        }

        gb = GradientBoostingClassifier(random_state=42)

        grid_search = GridSearchCV(
            estimator=gb,
            param_grid=param_grid,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )

        start = time.time()
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start

        print(f"\n✓ Meilleurs paramètres: {grid_search.best_params_}")
        print(f"✓ Meilleur score F1 (CV): {grid_search.best_score_:.4f}")
        print(f"✓ Temps d'entraînement: {training_time:.2f}s")

        return grid_search.best_estimator_, grid_search.best_params_, training_time

    def evaluate_model(self, model, X_test, y_test, model_name):
        """Évalue un modèle sur le test set"""
        start = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start

        # Métriques
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted')

        # Rapport détaillé
        class_names = self.label_encoder.classes_
        report = classification_report(
            y_test, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )

        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)

        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'inference_time': inference_time,
            'classification_report': report,
            'confusion_matrix': cm
        }

    def compare_models(self):
        """Compare tous les modèles"""
        print("\n" + "=" * 60)
        print("📊 COMPARAISON DES MODÈLES")
        print("=" * 60)

        # Préparer données
        X_train, X_test, y_train, y_test, features = self.prepare_data()

        # 1. Decision Tree
        dt_model, dt_params, dt_time = self.tune_decision_tree(X_train, y_train)
        dt_results = self.evaluate_model(dt_model, X_test, y_test, "Decision Tree")
        dt_results['training_time'] = dt_time
        dt_results['best_params'] = dt_params
        self.results['Decision Tree'] = dt_results
        self.best_models['Decision Tree'] = dt_model

        # 2. Random Forest
        rf_model, rf_params, rf_time = self.tune_random_forest(X_train, y_train)
        rf_results = self.evaluate_model(rf_model, X_test, y_test, "Random Forest")
        rf_results['training_time'] = rf_time
        rf_results['best_params'] = rf_params
        self.results['Random Forest'] = rf_results
        self.best_models['Random Forest'] = rf_model

        # 3. KNN
        knn_model, knn_params, knn_time = self.tune_knn(X_train, y_train)
        knn_results = self.evaluate_model(knn_model, X_test, y_test, "KNN")
        knn_results['training_time'] = knn_time
        knn_results['best_params'] = knn_params
        self.results['KNN'] = knn_results
        self.best_models['KNN'] = knn_model

        # 4. Gradient Boosting
        gb_model, gb_params, gb_time = self.tune_gradient_boosting(X_train, y_train)
        gb_results = self.evaluate_model(gb_model, X_test, y_test, "Gradient Boosting")
        gb_results['training_time'] = gb_time
        gb_results['best_params'] = gb_params
        self.results['Gradient Boosting'] = gb_results
        self.best_models['Gradient Boosting'] = gb_model

        # Tableau comparatif
        comparison_df = pd.DataFrame({
            'Modèle': [r['model_name'] for r in self.results.values()],
            'Accuracy': [r['accuracy'] for r in self.results.values()],
            'F1-Score': [r['f1_score'] for r in self.results.values()],
            'Precision': [r['precision'] for r in self.results.values()],
            'Recall': [r['recall'] for r in self.results.values()],
            'Temps Train (s)': [r['training_time'] for r in self.results.values()],
            'Temps Inférence (s)': [r['inference_time'] for r in self.results.values()]
        })

        print("\n" + "=" * 60)
        print("RÉSULTATS COMPARATIFS")
        print("=" * 60)
        print(comparison_df.to_string(index=False))

        # Sauvegarder
        comparison_df.to_csv("models/model_comparison.csv", index=False)
        print("\n✓ Résultats sauvegardés dans 'models/model_comparison.csv'")

        # Identifier le meilleur modèle
        best_model_name = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Modèle']
        print(f"\n🏆 MEILLEUR MODÈLE: {best_model_name}")
        print(f"   F1-Score: {comparison_df['F1-Score'].max():.4f}")

        return comparison_df

    def plot_confusion_matrices(self):
        """Visualise les matrices de confusion"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()

        class_names = self.label_encoder.classes_

        for idx, (model_name, results) in enumerate(self.results.items()):
            cm = results['confusion_matrix']

            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=axes[idx]
            )
            axes[idx].set_title(f'{model_name}\nAccuracy: {results["accuracy"]:.3f}')
            axes[idx].set_xlabel('Prédictions')
            axes[idx].set_ylabel('Réalité')

        plt.tight_layout()
        plt.savefig('models/confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("✓ Matrices de confusion sauvegardées: 'models/confusion_matrices.png'")
        return fig

    def plot_performance_comparison(self):
        """Visualise la comparaison des performances"""
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        model_names = list(self.results.keys())

        data = {
            metric: [self.results[model][metric] for model in model_names]
            for metric in metrics
        }

        df = pd.DataFrame(data, index=model_names)

        fig, ax = plt.subplots(figsize=(12, 6))
        df.plot(kind='bar', ax=ax, width=0.8)

        ax.set_title('Comparaison des Performances des Modèles', fontsize=16, fontweight='bold')
        ax.set_xlabel('Modèle', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(0, 1)
        ax.legend(['Accuracy', 'F1-Score', 'Precision', 'Recall'], loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig('models/performance_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Graphique de comparaison sauvegardé: 'models/performance_comparison.png'")
        return fig

    def plot_training_time_comparison(self):
        """Compare les temps d'entraînement"""
        model_names = list(self.results.keys())
        train_times = [self.results[model]['training_time'] for model in model_names]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(model_names, train_times, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])

        ax.set_title('Temps d\'Entraînement des Modèles', fontsize=16, fontweight='bold')
        ax.set_xlabel('Modèle', fontsize=12)
        ax.set_ylabel('Temps (secondes)', fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        # Ajouter valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=10
            )

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('models/training_time_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Graphique des temps sauvegardé: 'models/training_time_comparison.png'")
        return fig

    def generate_detailed_report(self):
        """Génère un rapport détaillé en texte"""
        report_path = "models/evaluation_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RAPPORT D'ÉVALUATION DES MODÈLES - NutriAI\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Profil utilisateur:\n")
            f.write(f"  Poids: {self.user_profile['weight']}kg\n")
            f.write(f"  Âge: {self.user_profile['age']} ans\n")
            f.write(f"  Objectif: {self.user_profile['objective']}\n")
            f.write(f"  TDEE: {int(self.user_profile['tdee'])} kcal\n\n")

            for model_name, results in self.results.items():
                f.write("-" * 80 + "\n")
                f.write(f"MODÈLE: {model_name}\n")
                f.write("-" * 80 + "\n\n")

                f.write(f"Meilleurs hyperparamètres:\n")
                for param, value in results['best_params'].items():
                    f.write(f"  {param}: {value}\n")
                f.write("\n")

                f.write(f"Métriques globales:\n")
                f.write(f"  Accuracy: {results['accuracy']:.4f}\n")
                f.write(f"  F1-Score: {results['f1_score']:.4f}\n")
                f.write(f"  Precision: {results['precision']:.4f}\n")
                f.write(f"  Recall: {results['recall']:.4f}\n")
                f.write(f"  Temps d'entraînement: {results['training_time']:.2f}s\n")
                f.write(f"  Temps d'inférence: {results['inference_time']:.4f}s\n\n")

                f.write("Rapport par classe:\n")
                report = results['classification_report']
                for class_name in self.label_encoder.classes_:
                    if class_name in report:
                        f.write(f"  {class_name}:\n")
                        f.write(f"    Precision: {report[class_name]['precision']:.4f}\n")
                        f.write(f"    Recall: {report[class_name]['recall']:.4f}\n")
                        f.write(f"    F1-Score: {report[class_name]['f1-score']:.4f}\n")
                        f.write(f"    Support: {report[class_name]['support']}\n\n")

        print(f"✓ Rapport détaillé généré: '{report_path}'")

    def save_best_models(self):
        """Sauvegarde les meilleurs modèles"""
        for model_name, model in self.best_models.items():
            filename = f"models/best_{model_name.lower().replace(' ', '_')}.pkl"
            joblib.dump(model, filename)
            print(f"✓ Modèle sauvegardé: {filename}")

        # Sauvegarder aussi le scaler et label encoder
        joblib.dump(self.scaler, "models/scaler_tuned.pkl")
        joblib.dump(self.label_encoder, "models/label_encoder_tuned.pkl")


def run_full_evaluation(df, user_profile):
    """Lance l'évaluation complète"""
    print("\n" + "=" * 80)
    print("DÉMARRAGE DE L'ÉVALUATION COMPLÈTE")
    print("=" * 80)

    evaluator = ModelEvaluator(df, user_profile)

    # 1. Comparaison des modèles
    comparison_df = evaluator.compare_models()

    # 2. Visualisations
    print("\n📊 Génération des visualisations...")
    evaluator.plot_confusion_matrices()
    evaluator.plot_performance_comparison()
    evaluator.plot_training_time_comparison()

    # 3. Rapport détaillé
    print("\n📝 Génération du rapport...")
    evaluator.generate_detailed_report()

    # 4. Sauvegarder meilleurs modèles
    print("\n💾 Sauvegarde des modèles...")
    evaluator.save_best_models()

    print("\n" + "=" * 80)
    print("✅ ÉVALUATION TERMINÉE")
    print("=" * 80)
    print("\nFichiers générés:")
    print("  - models/model_comparison.csv")
    print("  - models/confusion_matrices.png")
    print("  - models/performance_comparison.png")
    print("  - models/training_time_comparison.png")
    print("  - models/evaluation_report.txt")
    print("  - models/best_*.pkl (meilleurs modèles)")

    return evaluator, comparison_df