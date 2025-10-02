from flask import Flask, render_template, request, redirect, url_for, session
from model import NutritionRecommender
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)
recommender = NutritionRecommender("all_datas.csv")

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def compute_calorie_plan(weight, target_weight, height, age, sex, workouts_per_week, duration_weeks):
    """
    Calcule TDEE, besoin calorique cible, et répartition par repas
    """

    # BMR (Mifflin-St Jeor)
    if sex == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    # facteur activité
    if workouts_per_week <= 1:
        factor = 1.2
    elif workouts_per_week <= 3:
        factor = 1.375
    elif workouts_per_week <= 5:
        factor = 1.55
    else:
        factor = 1.725

    tdee = bmr * factor

    # déficit ou surplus calorique en fonction de l’objectif
    diff_poids = target_weight - weight  # positif = prise, négatif = perte
    calories_diff_totales = diff_poids * 7700  # 7700 kcal par kg
    jours = max(1, duration_weeks * 7)
    delta_journalier = calories_diff_totales / jours

    calories_cibles = tdee + delta_journalier

    # répartition par repas
    repartition = {
        "Breakfast": 0.25,
        "Lunch": 0.35,
        "Dinner": 0.30,
        "Snack": 0.10
    }
    calories_par_repas = {rep: round(calories_cibles * pct) for rep, pct in repartition.items()}

    return round(calories_cibles), calories_par_repas

@app.route("/", methods=["GET", "POST"])
def profile():
    if request.method == "POST":
        sex = request.form.get("sex")
        weight = safe_float(request.form.get("weight"))
        target_weight = safe_float(request.form.get("target_weight"))
        height = safe_float(request.form.get("height"))
        age = safe_int(request.form.get("age"))
        workouts = safe_int(request.form.get("workouts"))
        duration = safe_int(request.form.get("duration"))

        session['user'] = {
            "sex": sex,
            "weight": weight,
            "target_weight": target_weight,
            "height": height,
            "age": age,
            "workouts": workouts,
            "duration": duration
        }

        if weight > 0 and target_weight > 0 and height > 0 and age > 0 and duration > 0:
            tdee, targets = compute_calorie_plan(weight, target_weight, height, age, sex, workouts, duration)
            session['tdee'] = tdee
            session['targets'] = targets
        else:
            session.pop('tdee', None)
            session.pop('targets', None)

        session['current_meal'] = {"meal_type": None, "base_food": None, "selected_foods": []}
        return redirect(url_for('overview'))

    return render_template("profile.html")

@app.route("/overview", methods=["GET", "POST"])
def overview():
    user = session.get('user')
    tdee = session.get('tdee')
    targets = session.get('targets', {})
    foods = recommender.all_foods()

    if request.method == "POST":
        base_food = request.form.get("food_name")
        meal_pred, recs, total_cal = recommender.pipeline(base_food)
        cm = session.get('current_meal', {})
        cm['base_food'] = base_food
        cm['meal_type'] = meal_pred
        cm['selected_foods'] = [base_food]
        session['current_meal'] = cm
        session['last_total_cal'] = total_cal
        session['last_recs'] = recs
        return redirect(url_for('meal'))

    return render_template("overview.html", user=user, tdee=tdee, targets=targets, foods=foods)

@app.route("/meal", methods=["GET", "POST"])
def meal():
    foods_all = recommender.all_foods()
    cm = session.get('current_meal', {"meal_type": None, "base_food": None, "selected_foods": []})
    targets = session.get('targets', {})
    tdee = session.get('tdee')
    suggestions = session.get('last_recs', [])

    if request.method == "POST":
        action = request.form.get("action")
        if action == "set_meal_type":
            cm['meal_type'] = request.form.get("meal_type_select")
        elif action == "add_food":
            add_food = request.form.get("add_food_select")
            if add_food and add_food not in cm['selected_foods']:
                cm['selected_foods'].append(add_food)
        elif action == "remove_food":
            rem = request.form.get("remove_food_name")
            if rem and rem in cm['selected_foods']:
                cm['selected_foods'].remove(rem)
        elif action == "refresh_recs":
            if cm.get('base_food'):
                meal_pred, recs, total_cal = recommender.pipeline(cm['base_food'])
                cm['meal_type'] = meal_pred
                session['last_recs'] = recs
                session['last_total_cal'] = total_cal
        session['current_meal'] = cm
        return redirect(url_for('meal'))

    selected = cm.get('selected_foods', [])
    total_calories = recommender.calories_for_list(selected) if selected else 0.0

    target_for_meal = None
    if cm.get('meal_type') and cm['meal_type'] in targets:
        target_for_meal = targets[cm['meal_type']]

    diff_message = ""
    if target_for_meal is not None:
        diff = total_calories - target_for_meal
        if diff > 50:
            diff_message = f"⚠️ Ce repas dépasse l’objectif de {diff:.0f} kcal."
        elif diff < -50:
            diff_message = f"ℹ️ Ce repas est en dessous de l’objectif de {-diff:.0f} kcal."
        else:
            diff_message = "✅ Ce repas correspond parfaitement à votre objectif calorique."

    graphs = []
    if selected:
        for class_name in recommender.classes.keys():
            cols, values = recommender.get_nutrition_data(selected, class_name)
            fig, ax = plt.subplots(figsize=(7, 4))
            bottom = [0] * len(selected)
            for i, nutrient_values in enumerate(zip(*values)):
                ax.bar(selected, nutrient_values, bottom=bottom, label=cols[i])
                bottom = [sum(x) for x in zip(bottom, nutrient_values)]
            ax.set_ylabel("Nutrition value")
            ax.set_title(class_name)
            ax.legend()
            buf = BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format="png")
            buf.seek(0)
            graphs.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
            plt.close(fig)

    return render_template("meal.html",
                           cm=cm,
                           foods_all=foods_all,
                           selected=selected,
                           total_calories=total_calories,
                           target_for_meal=target_for_meal,
                           diff_message=diff_message,
                           graphs=graphs,
                           tdee=tdee,
                           suggestions=suggestions)

if __name__ == "__main__":
    app.run(debug=True)
