import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score, classification_report, accuracy_score
from model import NutritionRecommender
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)
recommender = NutritionRecommender("all_datas.csv")

def compute_model_results():
    """
    Évalue les modèles déjà entraînés dans `recommender`.
    Retourne: (results_list, plot_path)
    - results_list: liste de dict {name, type, metrics: {...}, interpretation}
    - plot_path: chemin relatif vers un PNG sauvegardé dans static/
    """
    results = []

    # --- Préparer jeux de test similaires à ceux utilisés en training ---
    df = recommender.data.copy()
    # Classification (Meal Type) : X = nutrition_cols, y = Meal Type (encoded)
    X_cls = df[recommender.nutrition_cols].values.astype(float)
    if "Meal Type" in df.columns:
        y_cls = df["Meal Type"].astype(str).values
    else:
        y_cls = np.array(["Unknown"] * len(df))

    # Regression (Calories) : features sans 'Calories'
    features_no_cal = ["Protein", "Carbs", "Fat", "Saturated Fat", "Fiber", "Sugar", "Sodium", "Water"]
    X_reg = df[features_no_cal].values.astype(float)
    y_reg = df["Calories"].values.astype(float)

    # split simple (deterministe) pour évaluation
    from sklearn.model_selection import train_test_split
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_cls, y_cls, test_size=0.3, random_state=42)
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

    # ----- ÉVALUATION classification: decision_tree (si présent) -----
    try:
        dt = recommender.decision_tree
        if dt is not None:
            # Prédire sur Xc_test
            y_pred_enc = dt.predict(Xc_test)
            # Vérifier les étiquettes et utiliser le LabelEncoder si nécessaire
            try:
                if hasattr(recommender, 'le_meal') and recommender.le_meal is not None:
                    # Vérifier la compatibilité des étiquettes
                    unique_labels = np.unique(yc_test)
                    known_labels = recommender.le_meal.classes_
                    print("Étiquettes yc_test:", unique_labels)
                    print("Classes du LabelEncoder:", known_labels)
                    # Filtrer les étiquettes non reconnues
                    valid_mask = np.isin(yc_test, known_labels)
                    if not valid_mask.any():
                        raise ValueError("Aucune étiquette valide pour la matrice de confusion")
                    yc_test_valid = yc_test[valid_mask]
                    y_pred_enc_valid = y_pred_enc[valid_mask]
                    Xc_test_valid = Xc_test[valid_mask]
                    # Encoder les étiquettes valides
                    y_true_enc = recommender.le_meal.transform(yc_test_valid)
                    y_pred_labels = recommender.le_meal.inverse_transform(y_pred_enc_valid)
                else:
                    raise ValueError("LabelEncoder non disponible")
                # Calculer les métriques
                acc = accuracy_score(yc_test_valid, y_pred_labels)
                cm = confusion_matrix(yc_test_valid, y_pred_labels, labels=np.unique(yc_test_valid))
                report = classification_report(yc_test_valid, y_pred_labels, zero_division=0)
                print("Matrice de confusion:", cm)
                results.append({
                    "name": "DecisionTreeClassifier",
                    "type": "classification",
                    "metrics": {
                        "Accuracy": f"{acc:.3f}",
                        "Confusion matrix shape": f"{cm.shape}",
                        "Classification report (brief)": report.replace("\n", " | ")
                    },
                    "interpretation": "Arbre entraîné pour prédire le type de repas. Vérifier classes peu représentées dans le rapport."
                })
            except Exception as e:
                print("Erreur dans le traitement des étiquettes:", str(e))
                results.append({
                    "name": "DecisionTreeClassifier",
                    "type": "classification",
                    "metrics": {"Error": f"Échec traitement étiquettes: {str(e)}"},
                    "interpretation": "Vérifiez les données de 'Meal Type' ou le LabelEncoder."
                })
        else:
            print("DecisionTree absent")
            results.append({
                "name": "DecisionTreeClassifier",
                "type": "classification",
                "metrics": {"Status": "Non disponible"},
                "interpretation": "Modèle absent."
            })
    except Exception as e:
        print("Erreur dans l'évaluation du DecisionTree:", str(e))
        results.append({
            "name": "DecisionTreeClassifier",
            "type": "classification",
            "metrics": {"Error": str(e)},
            "interpretation": "Échec évaluation."
        })

    # ----- ÉVALUATION régressions calories -----
    regressors = [
        ("LinearRegression", getattr(recommender, "linear_reg", None)),
        ("Ridge", getattr(recommender, "ridge_reg", None)),
        ("Polynomial(2)+Linear", getattr(recommender, "poly_reg", None)),
        ("CalReg(RandomForest or fallback)", getattr(recommender, "cal_reg", None))
    ]

    for name, model in regressors:
        if model is None:
            results.append({
                "name": name,
                "type": "regression",
                "metrics": {"Status": "Non disponible"},
                "interpretation": "Aucun modèle pour cette entrée."
            })
            continue
        try:
            if name.startswith("Polynomial") and getattr(recommender, "poly", None):
                X_test_trans = recommender.poly.transform(Xr_test)
            else:
                X_test_trans = Xr_test
            y_pred = model.predict(X_test_trans)
            rmse = np.sqrt(mean_squared_error(yr_test, y_pred))
            r2 = r2_score(yr_test, y_pred)
            mean_err = float(np.mean(y_pred - yr_test))
            results.append({
                "name": name,
                "type": "regression",
                "metrics": {
                    "RMSE": f"{rmse:.3f}",
                    "R2": f"{r2:.3f}",
                    "Mean error (pred - réel)": f"{mean_err:.3f}"
                },
                "interpretation": "Comparer RMSE/R2 pour juger la qualité. R2 proche de 1 = bon, RMSE bas = erreurs faibles."
            })
        except Exception as e:
            results.append({
                "name": name,
                "type": "regression",
                "metrics": {"Error": str(e)},
                "interpretation": "Échec évaluation."
            })

    # ----- Génération d'un plot récapitulatif (matrice confusion + résidus) -----
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        # 1) Matrice de confusion pour decision tree (si possible)
        try:
            if recommender.decision_tree is not None:
                y_pred_enc = recommender.decision_tree.predict(Xc_test)
                if hasattr(recommender, 'le_meal') and recommender.le_meal is not None:
                    unique_labels = np.unique(yc_test)
                    valid_mask = np.isin(yc_test, recommender.le_meal.classes_)
                    if valid_mask.any():
                        yc_test_valid = yc_test[valid_mask]
                        y_pred_enc_valid = y_pred_enc[valid_mask]
                        y_true_enc = recommender.le_meal.transform(yc_test_valid)
                        y_pred_labels = recommender.le_meal.inverse_transform(y_pred_enc_valid)
                        cm = confusion_matrix(yc_test_valid, y_pred_labels, labels=np.unique(yc_test_valid))
                        print("Matrice de confusion (plot):", cm)
                        ax = axes[0]
                        im = ax.imshow(cm, cmap='Blues', aspect='auto')
                        ax.set_title("Confusion matrix (DecisionTree)")
                        ax.set_xticks(np.arange(len(unique_labels)))
                        ax.set_yticks(np.arange(len(unique_labels)))
                        ax.set_xticklabels(unique_labels, rotation=45, ha='right')
                        ax.set_yticklabels(unique_labels)
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("True")
                        plt.colorbar(im, ax=ax)
                        for i in range(len(unique_labels)):
                            for j in range(len(unique_labels)):
                                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8, color='black')
                    else:
                        axes[0].text(0.5, 0.5, "Aucune étiquette valide", ha='center', va='center')
                        axes[0].set_title("Confusion matrix (DecisionTree)")
                else:
                    axes[0].text(0.5, 0.5, "LabelEncoder absent", ha='center', va='center')
                    axes[0].set_title("Confusion matrix (DecisionTree)")
            else:
                print("DecisionTree absent (plot)")
                axes[0].text(0.5, 0.5, "DecisionTree absent", ha='center', va='center')
                axes[0].set_title("Confusion matrix (DecisionTree)")
        except Exception as e:
            print("Erreur dans le plot de la matrice de confusion:", str(e))
            axes[0].text(0.5, 0.5, f"Erreur: {str(e)}", ha='center', va='center')
            axes[0].set_title("Confusion matrix (DecisionTree)")

        # 2) Résidus - LinearRegression
        try:
            lr = recommender.linear_reg
            if lr is not None:
                y_pred_lr = lr.predict(Xr_test)
                residuals = yr_test - y_pred_lr
                ax = axes[1]
                ax.scatter(y_pred_lr, residuals, s=8)
                ax.axhline(0, linestyle='--')
                ax.set_xlabel("Predicted Calories")
                ax.set_ylabel("Residuals (real - pred)")
                ax.set_title("Residuals (LinearRegression)")
            else:
                axes[1].text(0.5, 0.5, "LinearRegression absent", ha='center', va='center')
                axes[1].set_title("Residuals (LinearRegression)")
        except Exception:
            axes[1].text(0.5, 0.5, "Erreur residuals", ha='center', va='center')
            axes[1].set_title("Residuals (LinearRegression)")

        # 3) Pred vs Réel - RandomForest (cal_reg)
        try:
            rf = recommender.cal_reg
            if rf is not None:
                y_pred_rf = rf.predict(Xr_test)
                ax = axes[2]
                ax.scatter(yr_test, y_pred_rf, s=8)
                ax.plot([yr_test.min(), yr_test.max()], [yr_test.min(), yr_test.max()], linestyle='--')
                ax.set_xlabel("Real Calories")
                ax.set_ylabel("Predicted Calories")
                ax.set_title("Predicted vs Real (cal_reg)")
            else:
                axes[2].text(0.5, 0.5, "cal_reg absent", ha='center', va='center')
                axes[2].set_title("Pred vs Real (cal_reg)")
        except Exception:
            axes[2].text(0.5, 0.5, "Erreur pred vs real", ha='center', va='center')
            axes[2].set_title("Pred vs Real (cal_reg)")

        # 4) Histogramme des erreurs (régression la plus performante trouvée)
        try:
            best_name, best_err = None, float('inf')
            for name, model in regressors:
                if model is None:
                    continue
                try:
                    if name.startswith("Polynomial") and getattr(recommender, "poly", None):
                        ypred = model.predict(recommender.poly.transform(Xr_test))
                    else:
                        ypred = model.predict(Xr_test)
                    rmse = np.sqrt(mean_squared_error(yr_test, ypred))
                    if rmse < best_err:
                        best_err = rmse
                        best_name = name
                        best_pred = ypred
                except Exception:
                    continue
            ax = axes[3]
            if best_name is not None:
                errors = yr_test - best_pred
                ax.hist(errors, bins=30)
                ax.set_title(f"Histogramme des erreurs (meilleur: {best_name})")
                ax.set_xlabel("Erreur (real - pred)")
            else:
                ax.text(0.5, 0.5, "Aucun modèle régression disponible", ha='center', va='center')
                ax.set_title("Histogramme des erreurs")
        except Exception:
            axes[3].text(0.5, 0.5, "Erreur histogramme", ha='center', va='center')
            axes[3].set_title("Histogramme des erreurs")

        fig.tight_layout()

        # Sauvegarder dans static/
        static_dir = os.path.join(os.getcwd(), "static")
        if not os.path.isdir(static_dir):
            os.makedirs(static_dir, exist_ok=True)
        out_path = os.path.join(static_dir, "ml_models_summary.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        # Path relatif pour utilisation dans template
        plot_path = out_path.replace("\\", "/")
    except Exception as e:
        print("Erreur génération plot récapitulatif:", e)
        plot_path = None

    return results, plot_path

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
    if sex == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    if workouts_per_week <= 1:
        factor = 1.2
    elif workouts_per_week <= 3:
        factor = 1.375
    elif workouts_per_week <= 5:
        factor = 1.55
    else:
        factor = 1.725
    tdee = bmr * factor
    diff_poids = target_weight - weight
    calories_diff_totales = diff_poids * 7700
    jours = max(1, duration_weeks * 7)
    delta_journalier = calories_diff_totales / jours
    calories_cibles = tdee + delta_journalier
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

@app.route("/analysis", methods=["GET", "POST"])
def analysis():
    foods = recommender.all_foods()
    if request.method == "POST":
        base_food = request.form.get("food_name")
        meal_pred, recs, _ = recommender.pipeline(base_food)
        cm = {"base_food": base_food, "meal_type": meal_pred, "selected_foods": [base_food]}
        session["current_meal"] = cm
        session["last_recs"] = recs
        return render_template("analysis.html", base_food=base_food, meal_type=meal_pred, suggestions=recs, foods=foods)
    return render_template("analysis.html", foods=foods)

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
    smart_res = None
    if target_for_meal and cm.get('base_food'):
        smart_res = recommender.smart_recommendations(
            base_food=cm['base_food'],
            target_calories=target_for_meal
        )
    diff_message = ""
    if target_for_meal is not None:
        diff = total_calories - target_for_meal
        if diff > 50:
            diff_message = f"Ce repas dépasse l’objectif de {diff:.0f} kcal."
        elif diff < -50:
            diff_message = f"Ce repas est en dessous de l’objectif de {-diff:.0f} kcal."
        else:
            diff_message = "Ce repas correspond parfaitement à votre objectif calorique."
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
                           suggestions=suggestions,
                           smart_res=smart_res)

@app.route("/ml_analysis")
def ml_analysis():
    results, plot_path = compute_model_results()
    plot_path = plot_path.replace("\\", "/")
    return render_template("ml_analysis.html", results=results, plot_path=plot_path)


if __name__ == "__main__":
    app.run(debug=True)