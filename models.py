# models.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib


class NutriAI:
    def __init__(self):
        self.df = pd.read_csv("data/processed_nutrition.csv")
        self.scaler = StandardScaler()
        self.features = ['Calories', 'Protein', 'Carbs', 'Fat', 'Fiber', 'Sugar', 'Water', 'density_kcal_100g',
                         'satiety_index']
        self.rf_reg = None
        self.rf_clf = None
        self.knn = None

    def train_balance_predictor(self):
        X = self.df[self.features]
        y = self.df['balance_score']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        param_grid = {
            'n_estimators': [100, 300],
            'max_depth': [None, 15, 25],
            'min_samples_split': [2, 5]
        }
        self.rf_reg = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5,
                                   scoring='neg_mean_absolute_error')
        self.rf_reg.fit(X_train, y_train)

        y_pred = self.rf_reg.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"RF Regressor - MAE: {mae:.4f} | Best params: {self.rf_reg.best_params_}")
        joblib.dump(self.rf_reg, "models/rf_balance.pkl")

    def classify_food_role(self, objective="perte"):
        conditions = [
            (self.df['prot_ratio'] > 0.25) & (self.df['Fiber'] > 3) & (self.df['Sugar'] < 5),
            (self.df['Fat'] > 20) | (self.df['Calories'] > 400) | (self.df['Saturated Fat'] > 5),
            (self.df['Carbs'] > 50) & (self.df['Fiber'] < 2) & (self.df['Sugar'] > 15)
        ]
        choices = ["Privilégier", "Modération", "Éviter"]
        if objective == "gain":
            choices = choices[::-1]
        self.df['role'] = np.select(conditions, choices, default="Neutre")

        X = self.df[self.features]
        y = self.df['role']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.rf_clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
        self.rf_clf.fit(X_train, y_train)
        y_pred = self.rf_clf.predict(X_test)
        print("Classification Report:\n", classification_report(y_test, y_pred))
        joblib.dump(self.rf_clf, "models/rf_classifier.pkl")

    def train_knn_recommender(self):
        X = self.scaler.fit_transform(self.df[self.features])
        self.knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
        self.knn.fit(X)
        joblib.dump(self.knn, "models/knn_recommender.pkl")
        joblib.dump(self.scaler, "models/scaler.pkl")

    def recommend_similar(self, food_name):
        row = self.df[self.df['Food Category'].str.contains(food_name, case=False, na=False)]
        if row.empty:
            return ["Aucun aliment similaire"]
        idx = row.index[0]
        X = self.scaler.transform(self.df.loc[idx:idx, self.features])
        _, indices = self.knn.kneighbors(X)
        return self.df.iloc[indices[0][1:]]['Food Category'].tolist()