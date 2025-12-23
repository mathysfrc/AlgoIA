import numpy as np


class MealPlanner:
    """
    Générateur de plans de repas personnalisés.
    S'adapte aux besoins nutritionnels individuels.
    """

    # Blacklist d'aliments malsains
    UNHEALTHY_KEYWORDS = [
        'mcdonalds', 'burger king', 'kfc', 'pizza hut', 'dominos',
        'fast food', 'soda', 'coke', 'pepsi', 'fanta', 'sprite',
        'chips', 'doritos', 'cheetos', 'candy', 'sweets', 'industrial chocolate',
        'nuggets', 'fries', 'donut', 'industrial croissant'
    ]

    def __init__(self, df):
        """
        Initialise avec un DataFrame déjà filtré et scoré par NutriAI.
        """
        # Le dataset contient déja les résultats des modèles
        self.df = df.copy()

        # Filtrer aliments malsains
        self.df = self._filter_unhealthy()

        # Catégories d'aliments par type de repas (pas utilisé, juste en dure au cas ou pour faciliter gestion)
        self.meal_categories = {
            "Petit-déjeuner": ["Grains", "Fruits", "Dairy", "Eggs"],
            "Déjeuner": ["Meat", "Fish", "Grains", "Vegetables", "Legumes", "Dairy"],
            "Dîner": ["Meat", "Fish", "Vegetables", "Grains", "Legumes"],
            "Collation": ["Fruits", "Nuts", "Dairy", "Grains", "Vegetables"]
        }

    def _filter_unhealthy(self):
        """Exclut les aliments de la blacklist"""
        mask = self.df['Food Category'].str.lower().apply(
            lambda x: not any(bad in x for bad in self.UNHEALTHY_KEYWORDS)
        )
        return self.df[mask]

    def generate_daily_plan_personalized(self, target_macros):
        """
        Génère un plan quotidien adapté aux objectifs nutritionnels.
        """
        plan = {}
        daily_totals = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
        used_foods = set()

        # Répartition calorique par repas (personnalisable)
        meal_distribution = {
            "Petit-déjeuner": 0.25,  # 25% des calories
            "Déjeuner": 0.35,  # 35% des calories
            "Dîner": 0.30,  # 30% des calories
            "Collation": 0.10  # 10% des calories
        }

        for meal_name, ratio in meal_distribution.items():
            # Calculer cibles pour ce repas
            meal_target = {
                'calories': target_macros['calories'] * ratio,
                'protein': target_macros['protein'] * ratio,
                'carbs': target_macros['carbs'] * ratio,
                'fat': target_macros['fat'] * ratio
            }

            # Générer le repas
            foods, meal_totals = self._generate_optimized_meal(
                meal_target,
                meal_name,
                used_foods
            )

            plan[meal_name] = foods

            # Mettre à jour totaux
            for nutrient in daily_totals:
                daily_totals[nutrient] += meal_totals[nutrient]

            # Marquer aliments utilisés
            used_foods.update([f['Food Category'] for f in foods])

        return plan, daily_totals

    def _generate_optimized_meal(self, target, meal_type, used_foods):
        """
        Génère un repas optimisé pour atteindre les cibles nutritionnelles.
        Utilise un algorithme glouton avec score de pertinence.
        """
        # Filtrer candidats par type de repas
        candidates = self.df.copy()

        if meal_type in self.meal_categories:
            pattern = '|'.join(self.meal_categories[meal_type])
            candidates = candidates[
                candidates['Food Category'].str.contains(pattern, case=False, na=False)
            ]

        # Exclure aliments déjà utilisés
        candidates = candidates[~candidates['Food Category'].isin(used_foods)]

        # Priorité aux aliments "Privilégier"
        candidates = candidates.sort_values(
            ['role', 'user_score'],
            ascending=[True, False]  # Privilégier d'abord, puis par score
        )

        selected_foods = []
        current_totals = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
        roles_used = set()

        # Sélection itérative (3-4 aliments par repas)
        max_items = 4 if meal_type in ["Déjeuner", "Dîner"] else 3

        for _ in range(max_items):
            best_food = None
            best_score = float('inf')

            for _, food in candidates.iterrows():
                # Éviter duplicata et trop d'aliments "Éviter"
                if food['Food Category'] in [f['Food Category'] for f in selected_foods]:
                    continue

                if food['role'] == 'Éviter' and 'Éviter' in roles_used:
                    continue

                # Simuler ajout
                temp_totals = current_totals.copy()
                temp_totals['calories'] += food['Calories']
                temp_totals['protein'] += food['Protein']
                temp_totals['carbs'] += food['Carbs']
                temp_totals['fat'] += food['Fat']

                # Calculer erreur pondérée
                error = (
                        abs(temp_totals['calories'] - target['calories']) * 1.0 +
                        abs(temp_totals['protein'] - target['protein']) * 2.0 +  # Protéines prioritaires
                        abs(temp_totals['carbs'] - target['carbs']) * 1.5 +
                        abs(temp_totals['fat'] - target['fat']) * 1.2
                )

                # Bonus pour score utilisateur élevé
                error -= food.get('user_score', 0) * 50

                # Pénalité pour aliments "Éviter"
                if food['role'] == 'Éviter':
                    error += 200

                if error < best_score:
                    best_score = error
                    best_food = food

            # Ajouter meilleur aliment trouvé
            if best_food is not None:
                selected_foods.append(best_food.to_dict())
                current_totals['calories'] += best_food['Calories']
                current_totals['protein'] += best_food['Protein']
                current_totals['carbs'] += best_food['Carbs']
                current_totals['fat'] += best_food['Fat']
                roles_used.add(best_food['role'])

            else:
                break  # Plus de candidats valides

        return selected_foods, current_totals

    def adjust_portions(self, foods, target_calories):
        """
        Ajuste les portions pour mieux correspondre aux calories cibles.
        (Optionnel - pour amélioration future)
        """
        total_cal = sum(f['Calories'] for f in foods)

        if total_cal == 0:
            return foods

        ratio = target_calories / total_cal

        # Ajuster proportionnellement (entre 0.5x et 2x)
        ratio = max(0.5, min(2.0, ratio))

        adjusted = []
        for food in foods:
            adj_food = food.copy()
            adj_food['Calories'] *= ratio
            adj_food['Protein'] *= ratio
            adj_food['Carbs'] *= ratio
            adj_food['Fat'] *= ratio
            adjusted.append(adj_food)

        return adjusted

    def get_meal_summary(self, foods):
        """Résumé nutritionnel d'un repas"""
        totals = {
            'calories': sum(f['Calories'] for f in foods),
            'protein': sum(f['Protein'] for f in foods),
            'carbs': sum(f['Carbs'] for f in foods),
            'fat': sum(f['Fat'] for f in foods),
            'fiber': sum(f.get('Fiber', 0) for f in foods),
            'sugar': sum(f.get('Sugar', 0) for f in foods)
        }

        # Statistiques supplémentaires
        avg_score = np.mean([f.get('user_score', 0) for f in foods])
        roles = [f['role'] for f in foods]

        return {
            **totals,
            'avg_user_score': avg_score,
            'food_count': len(foods),
            'roles': roles
        }