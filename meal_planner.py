# meal_planner.py
import pandas as pd
import joblib


class MealPlanner:
    def __init__(self):
        self.df = pd.read_csv("data/processed_nutrition.csv")
        self.df['role'] = joblib.load("models/rf_classifier.pkl").predict(
            self.df[['Calories', 'Protein', 'Carbs', 'Fat', 'Fiber', 'Sugar', 'Water', 'density_kcal_100g',
                     'satiety_index']]
        )
        self.categories = {
            "Petit-déjeuner": ["Grains", "Fruits", "Dairy"],
            "Déjeuner": ["Meat", "Fish", "Grains", "Vegetables", "Legumes"],
            "Dîner": ["Meat", "Fish", "Vegetables", "Grains"],
            "Collation": ["Fruits", "Nuts", "Dairy", "Grains"]
        }

    def generate_daily_plan(self, target_macros):
        plan = {}
        daily_totals = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
        used_foods = set()

        meal_targets = {
            "Petit-déjeuner": 0.25,
            "Déjeuner": 0.35,
            "Dîner": 0.30,
            "Collation": 0.10
        }

        for meal, ratio in meal_targets.items():
            target = {
                'calories': target_macros['calories'] * ratio,
                'protein': target_macros['protein'] * ratio,
                'carbs': target_macros['carbs'] * ratio,
                'fat': target_macros['fat'] * ratio
            }
            foods, totals = self._generate_diverse_meal(target, meal, used_foods)
            plan[meal] = foods
            for k in daily_totals:
                daily_totals[k] += totals[k]
            used_foods.update([f['Food Category'] for f in foods])

        return plan, daily_totals

    def _generate_diverse_meal(self, target, meal_type, used_foods):
        candidates = self.df.copy()
        if meal_type in self.categories:
            candidates = candidates[
                candidates['Food Category'].str.contains('|'.join(self.categories[meal_type]), case=False, na=False)]
        candidates = candidates[~candidates['Food Category'].isin(used_foods)]

        selected = []
        total = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
        roles_used = set()

        for _ in range(3):
            best = None
            best_score = float('inf')
            for _, food in candidates.iterrows():
                if food['Food Category'] in used_foods or (food['role'] in roles_used and len(roles_used) > 1):
                    continue
                temp = total.copy()
                temp['calories'] += food['Calories']
                temp['protein'] += food['Protein']
                temp['carbs'] += food['Carbs']
                temp['fat'] += food['Fat']

                error = abs(temp['calories'] - target['calories']) + \
                        abs(temp['protein'] - target['protein']) + \
                        abs(temp['carbs'] - target['carbs']) + \
                        abs(temp['fat'] - target['fat'])
                if error < best_score:
                    best_score = error
                    best = food
            if best is not None:
                selected.append(best)
                total['calories'] += best['Calories']
                total['protein'] += best['Protein']
                total['carbs'] += best['Carbs']
                total['fat'] += best['Fat']
                roles_used.add(best['role'])

        return selected, total