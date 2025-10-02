# app.py
from flask import Flask, render_template, request, redirect, url_for, session
from model import NutritionRecommender
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # pour session, en prod choisis une clé stable
recommender = NutritionRecommender("all_datas.csv")

# utilitaires
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

def calculate_calorie_needs(weight, height, age, sex, goal):
    if sex == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    if goal == "lose":
        tdee = bmr * 0.85
    elif goal == "gain":
        tdee = bmr * 1.15
    else:
        tdee = bmr

    targets = {
        "Breakfast": tdee * 0.25,
        "Lunch": tdee * 0.35,
        "Dinner": tdee * 0.35,
        "Snack": tdee * 0.05,
    }
    return float(tdee), {k: float(v) for k, v in targets.items()}

# -----------------
# ROUTES
# -----------------

# Page profil ("/") -> sauvegarde profil en session
@app.route("/", methods=["GET", "POST"])
def profile():
    if request.method == "POST":
        goal = request.form.get("goal")
        sex = request.form.get("sex")
        weight = safe_float(request.form.get("weight"))
        height = safe_float(request.form.get("height"))
        age = safe_int(request.form.get("age"))
        # stocker en session
        session['user'] = {
            "goal": goal,
            "sex": sex,
            "weight": weight,
            "height": height,
            "age": age
        }
        # calculer tdee et targets et sauvegarder
        if weight > 0 and height > 0 and age > 0:
            tdee, targets = calculate_calorie_needs(weight, height, age, sex, goal)
            session['tdee'] = tdee
            session['targets'] = targets
        else:
            session.pop('tdee', None)
            session.pop('targets', None)

        # initialiser espace repas (vide)
        session['current_meal'] = {
            "meal_type": None,         # le type choisi pour la page meal
            "base_food": None,         # aliment de base initial choisi
            "selected_foods": []       # liste d'aliments ajoutés par l'utilisateur (inclut base s'il validé)
        }
        return redirect(url_for('overview'))

    # GET -> afficher formulaire profil
    return render_template("profile.html")

# Page overview -> montre tdee & targets, permet démarrer analyse d'un aliment
@app.route("/overview", methods=["GET", "POST"])
def overview():
    user = session.get('user')
    tdee = session.get('tdee')
    targets = session.get('targets', {})
    foods = recommender.all_foods()

    if request.method == "POST":
        base_food = request.form.get("food_name")
        # appeler pipeline pour avoir prédiction et recommandations initiales
        meal_pred, recs, total_cal = recommender.pipeline(base_food)
        # sauvegarder en session
        cm = session.get('current_meal', {})
        cm['base_food'] = base_food
        cm['meal_type'] = meal_pred  # par défaut on met la prédiction, user pourra changer
        # initial list = base_food + recs
        cm['selected_foods'] = [base_food] + recs
        session['current_meal'] = cm
        # on peut stocker aussi la dernière estimation
        session['last_total_cal'] = total_cal
        session['last_recs'] = recs
        return redirect(url_for('meal'))

    return render_template("overview.html",
                           user=user, tdee=tdee, targets=targets, foods=foods)

# Page meal -> choisir type, ajouter/enlever aliments, voir totals et graphiques
@app.route("/meal", methods=["GET", "POST"])
def meal():
    foods_all = recommender.all_foods()
    cm = session.get('current_meal', {"meal_type": None, "base_food": None, "selected_foods": []})
    targets = session.get('targets', {})
    tdee = session.get('tdee')
    message = ""

    # POST actions: change meal type, add food, remove food, accept/reset recommendations
    if request.method == "POST":
        action = request.form.get("action")
        if action == "set_meal_type":
            new_type = request.form.get("meal_type_select")
            cm['meal_type'] = new_type
        elif action == "add_food":
            add_food = request.form.get("add_food_select")
            if add_food and add_food not in cm['selected_foods']:
                cm['selected_foods'].append(add_food)
        elif action == "remove_food":
            rem = request.form.get("remove_food_name")
            if rem and rem in cm['selected_foods']:
                cm['selected_foods'].remove(rem)
        elif action == "refresh_recs":
            # rerun pipeline on base_food to get fresh recs (could be useful)
            if cm.get('base_food'):
                meal_pred, recs, total_cal = recommender.pipeline(cm['base_food'])
                cm['meal_type'] = meal_pred
                # replace recommendations (keep any user-added unique)
                cm['selected_foods'] = [cm['base_food']] + recs
                session['last_recs'] = recs
                session['last_total_cal'] = total_cal
        session['current_meal'] = cm
        return redirect(url_for('meal'))

    # GET -> calculs affichage
    selected = cm.get('selected_foods', [])
    total_calories = recommender.calories_for_list(selected) if selected else 0.0
    # target for chosen meal type (if known)
    target_for_meal = None
    if cm.get('meal_type') and cm['meal_type'] in targets:
        target_for_meal = targets[cm['meal_type']]

    # message diff
    diff_message = ""
    if target_for_meal is not None:
        diff = total_calories - target_for_meal
        if diff > 50:
            diff_message = f"⚠️ Ce repas dépasse l’objectif de {diff:.0f} kcal."
        elif diff < -50:
            diff_message = f"ℹ️ Ce repas est en dessous de l’objectif de {-diff:.0f} kcal."
        else:
            diff_message = "✅ Ce repas correspond parfaitement à votre objectif calorique."

    # graphiques (création de 1 ou plusieurs images base64)
    graphs = []
    if selected:
        for class_name in recommender.classes.keys():
            cols, values = recommender.get_nutrition_data(selected, class_name)
            fig, ax = plt.subplots(figsize=(7,4))
            bottom = [0]*len(selected)
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
                           tdee=tdee)

if __name__ == "__main__":
    app.run(debug=True)