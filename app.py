import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from user_profile import *
from meal_planner import MealPlanner
from models import NutriAI
from sklearn.tree import plot_tree

# === CONFIGURATION ===
st.set_page_config(page_title="NutriAI", layout="wide")
st.title("NutriAI – Recommandations Nutritionnelles Personnalisées")

# === FONCTION : Affichage du rôle avec couleur ===
def get_role_display(role):
    mapping = {
        "Privilégier": ("Privilégier", "success"),
        "Éviter": ("Éviter", "error"),
        "Modération": ("Modération", "warning"),
        "Neutre": ("Neutre", "secondary")
    }
    return mapping.get(role, ("Inconnu", "secondary"))

# === CHARGEMENT DES MODÈLES ===
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

# === SIDEBAR : Profil utilisateur ===
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
        st.success(f"**TDEE**: {int(tdee)} kcal\n**Macros**: P:{macros['protein']}g | G:{macros['carbs']}g | L:{macros['fat']}g")

# === ONGLET : Création des tabs ===
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Recommandations", "Repas", "Analyse Aliment", "Modèles & Performances", "Arbre Explicatif"
])

# === ONGLET 1 : Recommandations ===
with tab1:
    st.subheader("Aliments à privilégier")
    if 'macros' in st.session_state:
        role_target = "Privilégier" if objective == "perte" else "Éviter"
        subset = ai.df[ai.df['role'] == role_target]
        if len(subset) == 0:
            st.warning(f"Aucun aliment trouvé pour '{role_target}'.")
        else:
            st.markdown(f"### Aliments à **{role_target}** pour votre objectif")
            foods = subset.sample(n=min(5, len(subset)), replace=False)
            for _, f in foods.iterrows():
                with st.expander(f"**{f['Food Category']}** – {f['Meal Type']}"):
                    label, color = get_role_display(f['role'])
                    if color == "success":
                        st.success(f"Rôle : {label}")
                    elif color == "error":
                        st.error(f"Rôle : {label}")
                    elif color == "warning":
                        st.warning(f"Rôle : {label}")
                    else:
                        st.info(f"Rôle : {label}")

                    col1, col2 = st.columns(2)
                    col1.metric("Calories", f"{f['Calories']:.0f}")
                    col2.metric("Score d'équilibre", f"{f['balance_score']:.2f}")
                    st.progress(f['balance_score'])

                    reasons = []
                    if f['Protein'] > 15: reasons.append("riche en protéines")
                    if f['Fiber'] > 3: reasons.append("bonnes fibres")
                    if f['Sugar'] < 5: reasons.append("peu sucré")
                    if f['Fat'] > 20: reasons.append("gras")
                    if f['Sugar'] > 15: reasons.append("trop sucré")
                    if reasons:
                        st.caption(f"→ {', '.join(reasons)}")

# === ONGLET 2 : Plan de repas ===
with tab2:
    st.subheader("Plan Nutritionnel Journalier")
    if 'macros' in st.session_state and st.button("Générer mon plan complet"):
        plan, totals = planner.generate_daily_plan(st.session_state.macros)

        for meal, foods in plan.items():
            total_cal = sum(f['Calories'] for f in foods)
            with st.expander(f"**{meal}** – {total_cal:.0f} kcal"):
                for f in foods:
                    score = f.get('balance_score', 0.7)
                    label, color = get_role_display(f['role'])
                    if color == "success":
                        st.success(
                            f"{label} **{f['Food Category']}** – {f['Calories']:.0f} kcal | P:{f['Protein']:.0f}g G:{f['Carbs']:.0f}g L:{f['Fat']:.0f}g | Score: {score:.2f}")
                    elif color == "error":
                        st.error(
                            f"{label} **{f['Food Category']}** – {f['Calories']:.0f} kcal | P:{f['Protein']:.0f}g G:{f['Carbs']:.0f}g L:{f['Fat']:.0f}g | Score: {score:.2f}")
                    elif color == "warning":
                        st.warning(
                            f"{label} **{f['Food Category']}** – {f['Calories']:.0f} kcal | P:{f['Protein']:.0f}g G:{f['Carbs']:.0f}g L:{f['Fat']:.0f}g | Score: {score:.2f}")
                    else:
                        st.info(
                            f"{label} **{f['Food Category']}** – {f['Calories']:.0f} kcal | P:{f['Protein']:.0f}g G:{f['Carbs']:.0f}g L:{f['Fat']:.0f}g | Score: {score:.2f}")


        st.success(f"**Total jour** : {totals['calories']:.0f} kcal | "
                   f"P:{totals['protein']:.0f}g | G:{totals['carbs']:.0f}g | L:{totals['fat']:.0f}g")

        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ['Protéines', 'Glucides', 'Lipides']
        cible = [st.session_state.macros['protein'], st.session_state.macros['carbs'], st.session_state.macros['fat']]
        reel = [totals['protein'], totals['carbs'], totals['fat']]
        x = range(len(labels))
        ax.bar(x, cible, width=0.4, label='Cible', color='skyblue', alpha=0.8)
        ax.bar([i + 0.4 for i in x], reel, width=0.4, label='Réel', color='lightcoral', alpha=0.8)
        ax.set_ylabel('grammes')
        ax.set_xticks([i + 0.2 for i in x])
        ax.set_xticklabels(labels)
        ax.legend()
        st.pyplot(fig)

# === ONGLET 3 : Analyse Aliment ===
with tab3:
    st.subheader("Analyse d’un aliment")
    food_name = st.text_input("Rechercher un aliment")
    if food_name:
        matches = ai.df[ai.df['Food Category'].str.contains(food_name, case=False, na=False)]
        if not matches.empty:
            f = matches.iloc[0]
            label, color = get_role_display(f['role'])
            if color == "success":
                st.success(f"Rôle : {label}")
            elif color == "error":
                st.error(f"Rôle : {label}")
            elif color == "warning":
                st.warning(f"Rôle : {label}")
            else:
                st.info(f"Rôle : {label}")

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
            for sim_name in similar[:4]:
                sim_row = ai.df[ai.df['Food Category'] == sim_name]
                if not sim_row.empty:
                    sim = sim_row.iloc[0]
                    sim_label, sim_color = get_role_display(sim['role'])
                    if sim_color == "success":
                        st.success(f"→ {sim_name} – {sim_label}")
                    elif sim_color == "error":
                        st.error(f"→ {sim_name} – {sim_label}")
                    elif sim_color == "warning":
                        st.warning(f"→ {sim_name} – {sim_label}")
                    else:
                        st.info(f"→ {sim_name} – {sim_label}")

# === ONGLET 4 : Performances des modèles ===
with tab4:
    st.header("Performances des Modèles ML")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Random Forest Regressor")
        st.metric("MAE", "0.0108")
        st.write("**Meilleurs hyperparamètres** :")
        st.code("n_estimators: 300\nmax_depth: 15\nmin_samples_split: 2", language="text")

    with col2:
        st.subheader("Random Forest Classifier")
        st.write("**Accuracy** : 97%")
        data = {
            "": ["Privilégier", "Modération", "Neutre", "Éviter"],
            "Classe": ["Privilégier", "Modération", "Neutre", "Éviter"],
            "Précision": [1.00, 0.96, 0.97, 1.00]
        }
        df_report = pd.DataFrame(data).set_index("")
        st.table(df_report)

    st.subheader("KNN Recommander")
    st.write("**5 aliments similaires** via distance euclidienne dans l’espace nutritionnel")

# === ONGLET 5 : Arbre Explicatif ===
with tab5:
    st.header("Arbre de Décision Explicatif")
    st.write("Règles automatiques pour classer un aliment")

    st.markdown("""
    **Légende :**  
    - **Privilégier** → Aliments à favoriser (perte de poids)  
    - **Modération** → À consommer avec parcimonie  
    - **Neutre** → Ni bon ni mauvais  
    - **Éviter** → À éviter (trop sucré, gras, vide)
    """)

    if os.path.exists("models/decision_tree.pkl"):
        try:
            dt = joblib.load("models/decision_tree.pkl")
            feature_names = joblib.load("models/tree_features.pkl")
            rules = ai.get_tree_rules()
            st.code(rules, language="text")

            fig, ax = plt.subplots(figsize=(16, 10))
            plot_tree(
                dt,
                feature_names=feature_names,
                class_names=dt.classes_,
                filled=True,
                rounded=True,
                fontsize=9,
                ax=ax,
                proportion=True
            )
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Erreur : {e}")
    else:
        st.warning("Entraînez l'arbre avec `python main.py`")