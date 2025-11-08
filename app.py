import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from model import NutritionRecommender
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialisation du modèle (entraîne les modèles une seule fois)
recommender = NutritionRecommender("all_datas.csv")

def compute_model_results():
    """
    Évalue les modèles déjà entraînés dans `recommender`.
    Utilise les splits sauvegardés dans l'objet pour éviter les fuites de données.
    """
    results = []

    # --- Utilisation des splits pré-calculés dans train_models() ---
    Xc_test = recommender.X_test_cls
    yc_test = recommender.y_test_cls  # labels encodés
    Xr_test = recommender.X_test_reg
    yr_test = recommender.y_test_reg

    # --- Classification (RandomForestClassifier) ---
    try:
        clf = recommender.decision_tree
        if clf is not None:
            # Prédiction
            y_pred_enc = clf.predict(Xc_test)
            # Décodage
            try:
                y_true_labels = recommender.le_meal.inverse_transform(yc_test)
                y_pred_labels = recommender.le_meal.inverse_transform(y_pred_enc)
            except Exception as e:
                y_true_labels = yc_test
                y_pred_labels = y_pred_enc

            acc = accuracy_score(y_true_labels, y_pred_labels)
            cm = confusion_matrix(y_true_labels, y_pred_labels, labels=np.unique(y_true_labels))
            report = classification_report(y_true_labels, y_pred_labels, zero_division=0)

            # Validation croisée
            cv_scores = cross_val_score(clf, recommender.data[recommender.nutrition_cols].values,
                                      recommender.le_meal.transform(recommender.data["Meal Type"].astype(str)),
                                      cv=5, scoring='accuracy')

            results.append({
                "name": "RandomForestClassifier",
                "type": "classification",
                "metrics": {
                    "Accuracy": f"{acc:.3f}",
                    "CV Accuracy (mean ± std)": f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}",
                    "Confusion matrix shape": f"{cm.shape}",
                    "Classification report": report.replace("\n", " | ")
                },
                "interpretation": "Forêt aléatoire pour prédire le type de repas. Stratification + équilibrage des classes."
            })
        else:
            results.append({"name": "Classifier", "type": "classification", "metrics": {"Status": "Non disponible"}})
    except Exception as e:
        results.append({"name": "Classifier", "type": "classification", "metrics": {"Error": str(e)}})

    # --- Régressions (calories) ---
    regressors = [
        ("LinearRegression", getattr(recommender, "linear_reg", None)),
        ("Ridge", getattr(recommender, "ridge_reg", None)),
        ("CalReg (RandomForest)", getattr(recommender, "cal_reg", None))
    ]

    for name, model in regressors:
        if model is None:
            results.append({"name": name, "type": "regression", "metrics": {"Status": "Non disponible"}})
            continue
        try:
            y_pred = model.predict(Xr_test)
            rmse = np.sqrt(mean_squared_error(yr_test, y_pred))
            r2 = r2_score(yr_test, y_pred)
            mean_err = float(np.mean(y_pred - yr_test))

            # Validation croisée
            X_full = recommender.data[["Protein", "Carbs", "Fat", "Saturated Fat", "Fiber", "Sugar", "Sodium", "Water"]].values
            y_full = recommender.data["Calories"].values
            cv_rmse = -cross_val_score(model, X_full, y_full, cv=5, scoring='neg_root_mean_squared_error')

            results.append({
                "name": name,
                "type": "regression",
                "metrics": {
                    "RMSE": f"{rmse:.3f}",
                    "CV RMSE (mean ± std)": f"{cv_rmse.mean():.3f} ± {cv_rmse.std():.3f}",
                    "R2": f"{r2:.3f}",
                    "Mean error": f"{mean_err:.3f}"
                },
                "interpretation": "RMSE bas + R2 proche de 1 = bon modèle. CV pour robustesse."
            })
        except Exception as e:
            results.append({"name": name, "type": "regression", "metrics": {"Error": str(e)}})

    # --- Génération du plot récapitulatif ---
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        # 1) Matrice de confusion
        try:
            if recommender.decision_tree is not None:
                y_pred_enc = recommender.decision_tree.predict(Xc_test)
                y_true_labels = recommender.le_meal.inverse_transform(yc_test)
                y_pred_labels = recommender.le_meal.inverse_transform(y_pred_enc)
                cm = confusion_matrix(y_true_labels, y_pred_labels, labels=np.unique(y_true_labels))
                im = axes[0].imshow(cm, cmap='Blues')
                axes[0].set_title("Confusion Matrix (Meal Type)")
                axes[0].set_xticks(np.arange(len(np.unique(y_true_labels))))
                axes[0].set_yticks(np.arange(len(np.unique(y_true_labels))))
                axes[0].set_xticklabels(np.unique(y_true_labels), rotation=45, ha='right')
                axes[0].set_yticklabels(np.unique(y_true_labels))
                plt.colorbar(im, ax=axes[0])
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        axes[0].text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
            else:
                axes[0].text(0.5, 0.5, "Classifier absent", ha='center', va='center')
        except Exception as e:
            axes[0].text(0.5, 0.5, f"Erreur: {e}", ha='center', va='center')

        # 2) Résidus (Linear)
        try:
            if recommender.linear_reg:
                y_pred = recommender.linear_reg.predict(Xr_test)
                residuals = yr_test - y_pred
                axes[1].scatter(y_pred, residuals, s=8, alpha=0.6)
                axes[1].axhline(0, color='red', linestyle='--')
                axes[1].set_xlabel("Prédit")
                axes[1].set_ylabel("Résidu")
                axes[1].set_title("Résidus (LinearRegression)")
        except:
            axes[1].text(0.5, 0.5, "Erreur", ha='center')

        # 3) Prédit vs Réel (RandomForest)
        try:
            if recommender.cal_reg:
                y_pred = recommender.cal_reg.predict(Xr_test)
                axes[2].scatter(yr_test, y_pred, s=8, alpha=0.6)
                axes[2].plot([yr_test.min(), yr_test.max()], [yr_test.min(), yr_test.max()], 'r--')
                axes[2].set_xlabel("Réel")
                axes[2].set_ylabel("Prédit")
                axes[2].set_title("Prédit vs Réel (cal_reg)")
        except:
            axes[2].text(0.5, 0.5, "Erreur", ha='center')

        # 4) Feature Importance (Classifier)
        try:
            if hasattr(recommender.decision_tree, 'feature_importances_'):
                importances = recommender.decision_tree.feature_importances_
                axes[3].barh(recommender.nutrition_cols, importances)
                axes[3].set_title("Importance des Features (Meal Type)")
            else:
                axes[3].text(0.5, 0.5, "Pas d'importance", ha='center')
        except:
            axes[3].text(0.5, 0.5, "Erreur", ha='center')

        fig.tight_layout()
        static_dir = os.path.join(os.getcwd(), "static")
        os.makedirs(static_dir, exist_ok=True)
        plot_path = os.path.join(static_dir, "ml_models_summary.png")
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        plot_path = plot_path.replace("\\", "/")
    except Exception as e:
        print("Erreur plot:", e)
        plot_path = None

    return results, plot_path

def safe_float(value, default=0.0):
    try: return float(value)
    except: return default

def safe_int(value, default=0):
    try: return int(value)
    except: return default

def compute_calorie_plan(weight, target_weight, height, age, sex, workouts_per_week, duration_weeks):
    if sex == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    factor = {0:1.2, 1:1.2, 2:1.375, 3:1.375, 4:1.55, 5:1.55, 6:1.725, 7:1.725}.get(workouts_per_week, 1.55)
    tdee = bmr * factor
    diff_poids = target_weight - weight
    calories_diff_totales = diff_poids * 7700
    jours = max(1, duration_weeks * 7)
    delta_journalier = calories_diff_totales / jours
    calories_cibles = tdee + delta_journalier
    repartition = {"Breakfast": 0.25, "Lunch": 0.35, "Dinner": 0.30, "Snack": 0.10}
    calories_par_repas = {rep: round(calories_cibles * pct) for rep, pct in repartition.items()}
    return round(calories_cibles), calories_par_repas

# === Routes Flask ===
@app.route("/", methods=["GET", "POST"])
def profile():
    if request.method == "POST":
        session['user'] = {
            "sex": request.form.get("sex"),
            "weight": safe_float(request.form.get("weight")),
            "target_weight": safe_float(request.form.get("target_weight")),
            "height": safe_float(request.form.get("height")),
            "age": safe_int(request.form.get("age")),
            "workouts": safe_int(request.form.get("workouts")),  # nombre d'entraînements
            "duration": safe_int(request.form.get("duration"))
        }
        user = session['user']

        if all(user[k] > 0 for k in ["weight", "target_weight", "height", "age", "duration"]):
            tdee, targets = compute_calorie_plan(
                weight=user["weight"],
                target_weight=user["target_weight"],
                height=user["height"],
                age=user["age"],
                sex=user["sex"],
                workouts_per_week=user["workouts"],  # <--- CORRIGÉ
                duration_weeks=user["duration"]
            )
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
    if request.method == "POST":
        base_food = request.form.get("food_name")
        meal_pred, recs, total_cal = recommender.pipeline(base_food)
        session['current_meal'] = {'base_food': base_food, 'meal_type': meal_pred, 'selected_foods': [base_food]}
        session['last_recs'] = recs
        session['last_total_cal'] = total_cal
        return redirect(url_for('meal'))
    return render_template("overview.html", user=session.get('user'), tdee=session.get('tdee'), targets=session.get('targets', {}), foods=recommender.all_foods())

@app.route("/analysis", methods=["GET", "POST"])
def analysis():
    foods = recommender.all_foods()
    if request.method == "POST":
        base_food = request.form.get("food_name")
        meal_pred, recs, _ = recommender.pipeline(base_food)
        session["current_meal"] = {"base_food": base_food, "meal_type": meal_pred, "selected_foods": [base_food]}
        session["last_recs"] = recs
        return render_template("analysis.html", base_food=base_food, meal_type=meal_pred, suggestions=recs, foods=foods)
    return render_template("analysis.html", foods=foods)

@app.route("/meal", methods=["GET", "POST"])
def meal():
    cm = session.get('current_meal', {"meal_type": None, "base_food": None, "selected_foods": []})
    targets = session.get('targets', {})
    if request.method == "POST":
        action = request.form.get("action")
        if action == "set_meal_type":
            cm['meal_type'] = request.form.get("meal_type_select")
        elif action == "add_food":
            f = request.form.get("add_food_select")
            if f and f not in cm['selected_foods']: cm['selected_foods'].append(f)
        elif action == "remove_food":
            f = request.form.get("remove_food_name")
            if f in cm['selected_foods']: cm['selected_foods'].remove(f)
        elif action == "refresh_recs" and cm.get('base_food'):
            meal_pred, recs, total_cal = recommender.pipeline(cm['base_food'])
            cm['meal_type'] = meal_pred
            session['last_recs'] = recs
            session['last_total_cal'] = total_cal
        session['current_meal'] = cm
        return redirect(url_for('meal'))

    selected = cm.get('selected_foods', [])
    total_calories = recommender.calories_for_list(selected)
    target_for_meal = targets.get(cm.get('meal_type')) if cm.get('meal_type') in targets else None
    smart_res = recommender.smart_recommendations(cm['base_food'], target_for_meal) if target_for_meal and cm.get('base_food') else None

    diff_message = ""
    if target_for_meal:
        diff = total_calories - target_for_meal
        if abs(diff) <= 50:
            diff_message = "Objectif atteint !"
        elif diff > 0:
            diff_message = f"Dépassement de {diff:.0f} kcal."
        else:
            diff_message = f"Manque de {-diff:.0f} kcal."

    graphs = []
    if selected:
        for class_name in recommender.classes:
            cols, values = recommender.get_nutrition_data(selected, class_name)
            fig, ax = plt.subplots(figsize=(7, 4))
            bottom = [0] * len(selected)
            for i, nutrient_values in enumerate(zip(*values)):
                ax.bar(selected, nutrient_values, bottom=bottom, label=cols[i])
                bottom = [sum(x) for x in zip(bottom, nutrient_values)]
            ax.set_ylabel("Valeur")
            ax.set_title(class_name)
            ax.legend()
            buf = BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            graphs.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
            plt.close(fig)

    return render_template("meal.html", cm=cm, foods_all=recommender.all_foods(), selected=selected,
                           total_calories=total_calories, target_for_meal=target_for_meal, diff_message=diff_message,
                           graphs=graphs, tdee=session.get('tdee'), suggestions=session.get('last_recs', []),
                           smart_res=smart_res)

@app.route("/ml_analysis")
def ml_analysis():
    results, plot_path = compute_model_results()
    return render_template("ml_analysis.html", results=results, plot_path=plot_path)

if __name__ == "__main__":
    app.run(debug=True)