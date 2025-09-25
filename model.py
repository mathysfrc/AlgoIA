import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

class NutritionRecommender:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.nutrition_cols = ["Calories", "Protein", "Carbs", "Fat",
                               "Saturated Fat", "Fiber", "Sugar", "Sodium", "Water"]
        self.classes = {
            "Macronutrients": ["Calories", "Protein", "Carbs"],
            "Lipids": ["Fat", "Saturated Fat"],
            "Fibres & Sugar": ["Fiber", "Sugar"],
            "Minerals & Water": ["Sodium", "Water"]
        }

    def recommend(self, food_name, meal_type=None, n_neighbors=5):
        df = self.data
        if meal_type:
            df = df[df["Meal Type"] == meal_type]

        if food_name not in df['Food Category'].values:
            return []

        X = df[self.nutrition_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        knn = NearestNeighbors(n_neighbors=n_neighbors+1, metric='euclidean')
        knn.fit(X_scaled)

        idx = df.index[df['Food Category'] == food_name][0]
        # Trouver la position dans le subset filtré
        idx_in_subset = df.index.get_loc(idx)
        food_vector = X_scaled[idx_in_subset].reshape(1, -1)
        distances, indices = knn.kneighbors(food_vector)

        recommended = []
        for i in indices[0]:
            if i != idx_in_subset:
                recommended.append(df.iloc[i]['Food Category'])
        return recommended

    def get_nutrition_data(self, food_list, class_name, meal_type=None):
        df = self.data
        if meal_type:
            df = df[df["Meal Type"] == meal_type]
        cols = self.classes[class_name]
        values = []
        for f in food_list:
            row = df[df['Food Category']==f][cols].values
            if len(row)==0:
                values.append([0]*len(cols))
            else:
                values.append(row[0])
        return cols, values
