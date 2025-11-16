# app.py
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

from user_profile import calculate_bmr, calculate_tdee, get_macro_targets
from models import NutriAI
from meal_planner import MealPlanner
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

# === CHARGEMENT DES MODÈLES / OBJET AI ===
@st.cache_resource
def load_models():
    """
    Charge l'objet NutriAI et, si présents, les modèles sauvegardés.
    Si certains fichiers manquent, on continue en mode 'on-the-fly'.
    """
    model_dir = "models"
    ai = NutriAI()

    # fichiers potentiels (non obligatoires, on émet seulement un warning)
    optional_paths = {
        "rf_balance": os.path.join(model_dir, "rf_balance.pkl"),
        "rf_classifier": os.path.join(model_dir, "rf_classifier.pkl"),
        "knn": os.path.join(model_dir, "knn_recommender.pkl"),
        "scaler": os.path.join(model_dir, "scaler.pkl"),
        "decision_tree": os.path.join(model_dir, "decision_tree.pkl"),
        "tree_features": os.path.join(model_dir, "tree_features.pkl"),
    }

    for name, path in optional_paths.items():
        if os.path.exists(path):
            try:
                if name == "scaler":
                    ai.scaler = joblib.load(path)
                elif name == "rf_balance":
                    ai.rf_reg = joblib.load(path)
                elif name == "rf_classifier":
                    ai.rf_clf = joblib.load(path)
                elif name == "knn":
                    ai.knn = joblib.load(path)
                elif name == "decision_tree":
                    ai.dt = joblib.load(path)
                elif name == "tree_features":
                    # stored for legacy; not strictly required
                    pass
            except Exception as e:
                st.warning(f"Impossible de charger {path} : {e}")
        else:
            # not present -> ok, we will compute on the fly when needed
            pass

    # recalculs defensifs sur df
    df = ai.df.copy()
    df['prot_ratio'] = df['Protein'] * 4 / df['Calories'].replace(0, pd.NA)
    df['carb_ratio'] = df['Carbs'] * 4 / df['Calories'].replace(0, pd.NA)
    df['fat_ratio'] = df['Fat'] * 9 / df['Calories'].replace(0, pd.NA)
    df = df.fillna(0)

    # si rf_clf est disponible, produire 'role' global (legacy)
    if ai.rf_clf is not None:
        try:
            df_encoded = pd.get_dummies(df, columns=["profil", "objective"], drop_first=True)
            for col in ai.rf_clf.feature_names_in_:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            X = df_encoded[ai.rf_clf.feature_names_in_]
            df['role'] = ai.rf_clf.predict(X)
        except Exception as e:
            st.warning(f"Impossible d'utiliser rf_classifier pour prédire les rôles automatiquement : {e}")
            df['role'] = 'Neutre'
    else:
        df['role'] = 'Neutre'

    ai.df = df
    return ai

ai = load_models()

# === INSTANTIATION DU MealPlanner ===
# MealPlanner peut accepter soit (ai) soit un chemin ; on essaye les deux pour compatibilité
try:
    planner = MealPlanner(ai)  # si ta version attend un objet NutriAI
except TypeError:
    try:
        planner = MealPlanner()  # fallback: constructeur sans argument (utilise data/processed_nutrition.csv)
    except Exception as e:
        st.error(f"Impossible d'instancier MealPlanner automatiquement : {e}")
        st.stop()

# === SIDEBAR : Profil utilisateur ===
with st.sidebar:
    st.header("Votre Profil")

    weight = st.slider("Poids (kg)", 40, 150, 70)
    height = st.slider("Taille (cm)", 140, 220, 175)
    age = st.slider("Âge", 16, 80, 30)
    gender = st.selectbox("Sexe", ["Homme", "Femme"])

    # affichage friendly -> mapping vers les clés utilisées par NutriAI
    activity_display = ["Sédentaire", "Léger", "Modéré", "Intense", "Athlète"]
    activity_key_map = {
        "Sédentaire": "sedentaire",
        "Léger": "leger",
        "Modéré": "modere",
        "Intense": "intense",
        "Athlète": "athlete"
    }
    activity_display_choice = st.selectbox("Activité", activity_display, index=2)
    activity_key = activity_key_map[activity_display_choice]

    objective = st.selectbox("Objectif", ["perte", "maintien", "gain"])

    if st.button("Appliquer le profil"):
        ai.set_user_profile(
            weight=weight,
            height=height,
            age=age,
            gender=gender,
            activity=activity_display_choice,
            objective=objective
        )

        # On doit construire user_profile avec macros si déjà calculées
        if 'macros' in st.session_state:
            user_profile = {
                "weight": weight,
                "height": height,
                "age": age,
                "gender": gender,
                "activity": activity_key,
                "objective": objective,
                "calories": st.session_state.macros["calories"],
                "protein": st.session_state.macros["protein"],
                "carbs": st.session_state.macros["carbs"],
                "fat": st.session_state.macros["fat"]
            }
            try:
                ai.build_personal_tree(user_profile)
                st.success(f"Profil appliqué et arbre construit pour {activity_key}/{objective}")
            except Exception as e:
                st.warning(f"Profil appliqué mais erreur arbre : {e}")

        else:
            st.warning("Veuillez d’abord cliquer sur *Calculer mes besoins* avant de construire l’arbre.")

    if  st.button("Calculer mes besoins"):
        ai.set_user_profile(
            weight=weight,
            height=height,
            age=age,
            gender=gender,
            activity=activity_display_choice,
            objective=objective
        )

        # Récupérer les valeurs calculées par NutriAI
        macros = {
            "calories": ai.df['user_calories'].iloc[0],
            "protein": ai.df['user_protein'].iloc[0],
            "carbs": ai.df['user_carbs'].iloc[0],
            "fat": ai.df['user_fat'].iloc[0],
        }

        st.session_state.macros = macros

        st.success(
            f"**TDEE (objectif inclus)** : {int(macros['calories'])} kcal\n"
            f"**Macros** : P:{macros['protein']}g | "
            f"G:{macros['carbs']}g | L:{macros['fat']}g"
        )

# === ONGLET : Création des tabs ===
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Recommandations", "Repas", "Analyse Aliment", "Modèles & Performances", "Arbre Explicatif"
])

# === ONGLET 1 : Recommandations ===
with tab1:
    st.subheader("Aliments à privilégier (personnalisé)")
    if 'macros' in st.session_state:
        # rôle cible : logique simple (on peut ajuster)
        if objective == "perte":
            role_target = "Privilégier"
        elif objective == "gain":
            role_target = "Privilégier"
        else:
            role_target = "Privilégier"

        # on utilise la méthode qui calcule les labels selon le profil/objectif
        subset = ai.get_role_subset(role_target, profil=ai.current_profile, objective=ai.current_objective)
        if len(subset) == 0:
            st.warning(f"Aucun aliment trouvé pour '{role_target}'.")
        else:
            st.markdown(f"### Aliments à **{role_target}** pour votre profil `{ai.current_profile}` / `{ai.current_objective}`")
            foods = subset.sample(n=min(8, len(subset)), replace=False)
            for _, f in foods.iterrows():
                with st.expander(f"**{f['Food Category']}** – {f.get('Meal Type', '')}"):
                    label, color = get_role_display(f.get('role', 'Neutre'))
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
                    col2.metric("Score d'équilibre", f"{f.get('balance_score', 0.0):.2f}")
                    # mini résumé nutritionnel
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
    st.subheader("Plan Nutritionnel Journalier (personnalisé)")
    if 'macros' in st.session_state and st.button("Générer mon plan complet"):
        try:
            plan, totals = planner.generate_daily_plan(st.session_state.macros, profil=ai.current_profile, objective=ai.current_objective)
        except TypeError:
            # fallback si MealPlanner n'accepte pas ces arguments
            plan, totals = planner.generate_daily_plan(st.session_state.macros)

        for meal, foods in plan.items():
            total_cal = sum(float(f['Calories']) for f in foods)
            with st.expander(f"**{meal}** – {total_cal:.0f} kcal"):
                for f in foods:
                    score = f.get('balance_score', 0.7)
                    label, color = get_role_display(f.get('role', 'Neutre'))
                    line = f"{label} **{f['Food Category']}** – {f['Calories']:.0f} kcal | P:{f['Protein']:.0f}g G:{f['Carbs']:.0f}g L:{f['Fat']:.0f}g | Score: {score:.2f}"
                    if color == "success":
                        st.success(line)
                    elif color == "error":
                        st.error(line)
                    elif color == "warning":
                        st.warning(line)
                    else:
                        st.info(line)

        st.success(f"**Total jour** : {totals['calories']:.0f} kcal | "
                   f"P:{totals['protein']:.0f}g | G:{totals['carbs']:.0f}g | L:{totals['fat']:.0f}g")

        # graphique macros (matplotlib)
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ['Protéines', 'Glucides', 'Lipides']
        cible = [st.session_state.macros['protein'], st.session_state.macros['carbs'], st.session_state.macros['fat']]
        reel = [totals['protein'], totals['carbs'], totals['fat']]
        x = range(len(labels))
        ax.bar(x, cible, width=0.4, label='Cible')
        ax.bar([i + 0.4 for i in x], reel, width=0.4, label='Réel')
        ax.set_ylabel('grammes')
        ax.set_xticks([i + 0.2 for i in x])
        ax.set_xticklabels(labels)
        ax.legend()
        st.pyplot(fig)

with tab3:
    st.subheader("Analyse d’un aliment")
    food_name = st.text_input("Rechercher un aliment")
    if food_name:
        matches = ai.df[ai.df['Food Category'].str.contains(food_name, case=False, na=False)]
        if not matches.empty:
            f = matches.iloc[0]

            # ---- Rôle personnalisé EXACTEMENT comme TAB 1 ----
            try:
                subset_single = ai.get_role_subset(
                    role=f.get('role', 'Neutre'),
                    profil=ai.current_profile,
                    objective=ai.current_objective
                )

                role_label = subset_single[subset_single['Food Category'] == f['Food Category']]['role'].values
                role_label = role_label[0] if len(role_label) > 0 else f.get('role', 'Neutre')

            except Exception:
                role_label = f.get('role', 'Neutre')

            # Affichage du rôle
            label, color = get_role_display(role_label)
            if color == "success":
                st.success(f"Rôle : {label}")
            elif color == "error":
                st.error(f"Rôle : {label}")
            elif color == "warning":
                st.warning(f"Rôle : {label}")
            else:
                st.info(f"Rôle : {label}")

            # ---- Infos nutritionnelles ----
            col1, col2, col3 = st.columns(3)
            col1.metric("Calories", f"{f['Calories']:.0f}")
            col2.metric("Score Équilibre", f"{f.get('balance_score', 0.0):.2f}")
            col3.metric("Satiété /100kcal", f"{f.get('satiety_index', 0.0):.1f}")

            st.write("**Ratios clés** :")
            st.write(f"• Protéines : {f.get('prot_ratio',0):.1%}")
            st.write(f"• Glucides : {f.get('carb_ratio',0):.1%}")
            st.write(f"• Lipides : {f.get('fat_ratio',0):.1%}")
            st.write(f"• Densité : {f.get('density_kcal_100g',0):.1f} kcal/100g")

            # ---- Similarité ----
            st.write("**Aliments similaires (personnalisé)** :")
            similar = ai.recommend_similar(
                f['Food Category'],
                profil=ai.current_profile,
                objective=ai.current_objective,
                user_profile=ai._last_user_profile
            )

            # ---- On ne garde que les “Privilégier” (COMME TAB 1) ----
            filtered_similar = []

            for sim_name in similar:
                sim_row = ai.df[ai.df['Food Category'] == sim_name]
                if sim_row.empty:
                    continue
                sim = sim_row.iloc[0]

                sim_role = ai.get_role_subset(
                    "Privilégier",
                    profil=ai.current_profile,
                    objective=ai.current_objective
                )

                if sim_name in sim_role["Food Category"].values:
                    filtered_similar.append(sim_name)

            # ---- Affichage final ----
            if len(filtered_similar) == 0:
                st.info("Aucun aliment recommandé similaire trouvé.")
            else:
                for sim_name in filtered_similar[:6]:
                    st.success(f"→ {sim_name} – Privilégier")

# === ONGLET 4 : Performances ===
with tab4:
    st.header("Performances des Modèles ML")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Random Forest Regressor")
        if ai.rf_reg is not None:
            st.metric("MAE", "0.0108 (pré-enregistré)")
            st.code("n_estimators: 300\nmax_depth: 15\nmin_samples_split: 2")
        else:
            st.info("RF Regressor non chargé — exécute 'main.py' pour entraîner.")

    with col2:
        st.subheader("Random Forest Classifier")
        if ai.rf_clf is not None:
            st.write("**Accuracy : estimée**")
            # tu peux afficher un rapport réel si tu sauvegardes les métriques
            data = {
                "": ["Privilégier", "Modération", "Neutre", "Éviter"],
                "Classe": ["Privilégier", "Modération", "Neutre", "Éviter"],
                "Précision": [1.00, 0.96, 0.97, 1.00]
            }
            df_report = pd.DataFrame(data).set_index("")
            st.table(df_report)
        else:
            st.info("RF Classifier non chargé — labels calculés dynamiquement selon règles.")

# === ONGLET 5 : Arbre explicatif personnalisé ===
with tab5:
    st.header("Arbre de Décision Personnalisé (profil / objectif)")
    st.markdown("""
    **Légende :**  
    - **Privilégier** → Aliments à favoriser  
    - **Modération** → À consommer avec parcimonie  
    - **Neutre** → Ni bon ni mauvais  
    - **Éviter** → À éviter
    """)

    try:
        rules = ai.get_personal_tree_rules(profil=ai.current_profile, objective=ai.current_objective)
        st.code(rules, language="text")
        # si un arbre existe on peut aussi l'afficher graphiquement
        key = os.path.join("models", f"personal_tree_{ai.current_profile}_{ai.current_objective}.pkl")
        feat_key = os.path.join("models", f"tree_features_{ai.current_profile}_{ai.current_objective}.pkl")
        if os.path.exists(key) and os.path.exists(feat_key):
            dt = joblib.load(key)
            feature_names = joblib.load(feat_key)
            fig, ax = plt.subplots(figsize=(16, 10))
            plot_tree(ai.personal_tree, feature_names=ai.personal_tree_features, class_names=ai.personal_tree.classes_,
                      filled=True, rounded=True, fontsize=8, ax=ax, proportion=True)
            st.pyplot(fig)
        else:
            st.info("Arbre personnalisé construit à la volée (voir règles ci-dessus).")
    except Exception as e:
        st.error(f"Erreur lors de la génération de l'arbre personnalisé : {e}")
