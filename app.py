# app.py
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from user_profile import *
from meal_planner import MealPlanner
from models import NutriAI

st.set_page_config(page_title="NutriAI", layout="wide")
st.title("NutriAI – Recommandations Nutritionnelles Personnalisées")


@st.cache_resource
def load_models():
    model_dir = "models"
    ai = NutriAI()

    paths = {
        "rf_balance": os.path.join(model_dir, "rf_balance.pkl"),
        "rf_classifier": os.path.join(model_dir, "rf_classifier.pkl"),
        "knn": os.path.join(model_dir, "knn_recommender.pkl"),
        "scaler": os.path.join(model_dir, "scaler.pkl")
    }

    for name, path in paths.items():
        if not os.path.exists(path):
            st.error(f"Modèle manquant : {path}")
            st.stop()

    ai.rf_reg = joblib.load(paths["rf_balance"])
    ai.rf_clf = joblib.load(paths["rf_classifier"])
    ai.knn = joblib.load(paths["knn"])
    ai.scaler = joblib.load(paths["scaler"])

    df = ai.df.copy()
    df['prot_ratio'] = df['Protein'] * 4 / df['Calories']
    df['carb_ratio'] = df['Carbs'] * 4 / df['Calories']
    df['fat_ratio'] = df['Fat'] * 9 / df['Calories']
    X = df[ai.features]
    df['role'] = ai.rf_clf.predict(X)
    ai.df = df

    return ai


ai = load_models()
planner = MealPlanner()

# --- Sidebar ---
with st.sidebar:
    st.header("Votre Profil")
    weight = st.slider("Poids (kg)", 40, 150, 70)
    height = st.slider("Taille (cm)", 140, 220, 175)
    age = st.slider("Âge", 16, 80, 30)
    gender = st.selectbox("Sexe", ["Homme", "Femme"])
    activity_levels = ["Sédentaire", "Léger", "Modéré", "Intense", "Athlète"]
    activity = st.selectbox("Activité", activity_levels)
    objective = st.selectbox("Objectif", ["perte", "maintien", "gain"])

    if st.button("Calculer mes besoins"):
        bmr = calculate_bmr(weight, height, age, gender)
        tdee = calculate_tdee(bmr, activity)
        macros = get_macro_targets(tdee, objective, weight)
        st.session_state.macros = macros
        st.success(
            f"**TDEE**: {int(tdee)} kcal\n**Macros**: P:{macros['protein']}g | G:{macros['carbs']}g | L:{macros['fat']}g")

# --- Onglets ---
tab1, tab2, tab3, tab4 = st.tabs(["Recommandations", "Repas", "Analyse Aliment", "Modèles & Performances"])

with tab1:
    if 'macros' in st.session_state:
        role = "Privilégier" if objective == "perte" else "Éviter"
        foods = ai.df[ai.df['role'] == role].sample(5)
        for _, f in foods.iterrows():
            with st.expander(f"**{f['Food Category']}** – {f['Meal Type']}"):
                col1, col2 = st.columns(2)
                col1.metric("Calories", f"{f['Calories']:.0f}")
                col2.metric("Score", f"{f['balance_score']:.2f}")
                st.progress(f['balance_score'])

with tab2:
    st.subheader("Plan Nutritionnel Journalier")
    if 'macros' in st.session_state and st.button("Générer mon plan complet"):
        plan, totals = planner.generate_daily_plan(st.session_state.macros)

        for meal, foods in plan.items():
            with st.expander(f"**{meal}** – {sum(f['Calories'] for f in foods):.0f} kcal"):
                for f in foods:
                    score = f.get('balance_score', 0.7)
                    st.write(f"• **{f['Food Category']}** – {f['Calories']:.0f} kcal | "
                             f"P:{f['Protein']:.0f}g G:{f['Carbs']:.0f}g L:{f['Fat']:.0f}g "
                             f"| Score: {score:.2f}")

        st.success(f"**Total jour** : {totals['calories']:.0f} kcal | "
                   f"P:{totals['protein']:.0f}g | G:{totals['carbs']:.0f}g | L:{totals['fat']:.0f}g")

        fig, ax = plt.subplots()
        labels = ['Protéines', 'Glucides', 'Lipides']
        cible = [st.session_state.macros['protein'], st.session_state.macros['carbs'], st.session_state.macros['fat']]
        reel = [totals['protein'], totals['carbs'], totals['fat']]
        x = range(len(labels))
        ax.bar(x, cible, width=0.4, label='Cible', alpha=0.7)
        ax.bar([i + 0.4 for i in x], reel, width=0.4, label='Réel', alpha=0.7)
        ax.set_ylabel('grammes')
        ax.set_xticks([i + 0.2 for i in x])
        ax.set_xticklabels(labels)
        ax.legend()
        st.pyplot(fig)

with tab3:
    food_name = st.text_input("Rechercher un aliment")
    if food_name:
        matches = ai.df[ai.df['Food Category'].str.contains(food_name, case=False, na=False)]
        if not matches.empty:
            f = matches.iloc[0]
            col1, col2, col3 = st.columns(3)
            col1.metric("Calories", f"{f['Calories']:.0f}")
            col2.metric("Score Équilibre", f"{f['balance_score']:.2f}")
            col3.metric("Satiété /100kcal", f"{f['satiety_index']:.1f}")

            st.write("**Ratios clés** :")
            st.write(f"• Protéines : {f['prot_ratio']:.1%}")
            st.write(f"• Glucides : {f['carb_ratio']:.1%}")
            st.write(f"• Lipides : {f['fat_ratio']:.1%}")
            st.write(f"• Densité : {f['density_kcal_100g']:.1f} kcal/100g")

            st.write("**Aliments similaires** :")
            similar = ai.recommend_similar(f['Food Category'])
            st.write(", ".join(similar[:4]))

with tab4:
    st.header("Performances des Modèles ML")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Random Forest Regressor")
        st.metric("MAE", "0.0108")
        st.write("**Meilleurs hyperparamètres** :")
        st.code("n_estimators: 300\nmax_depth: 15\nmin_samples_split: 2")
    with col2:
        st.subheader("Random Forest Classifier")
        st.write("**Accuracy** : 97%")
        st.table(pd.DataFrame({
            "Classe": ["Privilégier", "Modération", "Neutre", "Éviter"],
            "Précision": [1.00, 0.96, 0.97, 1.00]
        }))
    st.subheader("KNN Recommander")
    st.write("**5 aliments similaires** via distance euclidienne")