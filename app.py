import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from meal_planner import MealPlanner
from models import NutriAI

st.set_page_config(page_title="NutriAI Personnalisé", layout="wide")
st.title("NutriAI – Recommandations 100% Personnalisées")


# Chargement du système
@st.cache_resource
def load_ai_system():
    ai = NutriAI()
    return ai


ai = load_ai_system()

# Profil utilisateur
with st.sidebar:
    st.header("Votre Profil")

    col1, col2 = st.columns(2)
    with col1:
        weight = st.number_input("Poids (kg)", 40, 150, 70, 1)
        age = st.number_input("Âge", 16, 80, 30, 1)
        gender = st.selectbox("Sexe", ["Homme", "Femme"])

    with col2:
        height = st.number_input("Taille (cm)", 140, 220, 175, 1)
        activity = st.selectbox("Activité", [
            "Sédentaire", "Léger", "Modéré", "Intense", "Athlète"
        ])
        objective = st.selectbox("Objectif", [
            ("perte", "Perte de poids"),
            ("maintien", "Maintien"),
            ("gain", "Prise de masse")
        ], format_func=lambda x: x[1])

    st.divider()

    if st.button("Calculer mes besoins personnalisés", type="primary"):
        with st.spinner("Calcul en cours..."):
            # Configuration du profil utilisateur
            profile = ai.set_user_profile(
                weight=weight,
                height=height,
                age=age,
                gender=gender,
                activity=activity,
                objective=objective[0]
            )

            # Classification
            ai.classify_food_role_personalized()

            # Entraînement KNN
            ai.train_knn_personalized()

            # Entraînement arbre
            ai.train_decision_tree_personalized()

            st.session_state.profile = profile
            st.session_state.ai_ready = True

        st.success("Profil configuré !")

        # Affichage des besoins
        st.metric("TDEE", f"{int(profile['tdee'])} kcal")
        st.metric("Objectif", f"{int(profile['target_calories'])} kcal")

        col1, col2, col3 = st.columns(3)
        col1.metric("Protéines", f"{int(profile['target_protein'])}g")
        col2.metric("Glucides", f"{int(profile['target_carbs'])}g")
        col3.metric("Lipides", f"{int(profile['target_fat'])}g")

# Vérification profil configuré
if 'ai_ready' not in st.session_state:
    st.info("Configurez votre profil dans la barre latérale pour commencer")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Recommandations",
    "Plan de Repas",
    "Analyse Aliment",
    "Arbre Explicatif",
    "Performances & Comparaison"
])

# Recommandations
with tab1:
    st.header("Aliments Recommandés pour VOUS")

    profile = st.session_state.profile

    col1, col2, col3 = st.columns(3)
    col1.metric("Votre Objectif", f"{int(profile['target_calories'])} kcal/jour")
    col2.metric("Activité", profile['activity'])
    col3.metric("IMC", f"{weight / (height / 100) ** 2:.1f}")

    st.divider()

    # Filtrer aliments selon rôle
    role_filter = st.radio(
        "Afficher les aliments à :",
        ["Privilégier", "Modération", "Neutre", "Éviter"],
        horizontal=True
    )

    foods = ai.df[ai.df['role'] == role_filter].sort_values('user_score', ascending=False)

    if len(foods) == 0:
        st.warning(f"Aucun aliment '{role_filter}' trouvé.")
    else:
        st.subheader(f"{len(foods)} aliments à {role_filter}")

        # Top 10
        for idx, (_, food) in enumerate(foods.head(10).iterrows(), 1):
            with st.expander(
                    f"#{idx} **{food['Food Category']}** "
                    f"({food['Meal Type']}) - Score: {food['user_score']:.2f}"
            ):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Calories", f"{food['Calories']:.0f}")
                col2.metric("Protéines", f"{food['Protein']:.1f}g")
                col3.metric("Glucides", f"{food['Carbs']:.1f}g")
                col4.metric("Lipides", f"{food['Fat']:.1f}g")

                st.progress(food['user_score'], "Score de pertinence pour VOUS")

                # Pourquoi cet aliment ?
                reasons = []
                if food['protein_fit'] > 0.7:
                    reasons.append("✓ Protéines adaptées")
                if food['calorie_fit'] > 0.7:
                    reasons.append("✓ Calories appropriées")
                if food['Fiber'] > 4:
                    reasons.append("✓ Riche en fibres")
                if food['Sugar'] < 8:
                    reasons.append("✓ Faible en sucre")
                if food['satiety_index'] > 2:
                    reasons.append("✓ Effet satiété élevé")

                if reasons:
                    st.info("Pourquoi cet aliment ? " + " | ".join(reasons))

# Plan de repas
with tab2:
    st.header("Plan Nutritionnel Journalier Personnalisé")

    profile = st.session_state.profile

    if st.button("Générer mon plan complet", type="primary"):
        planner = MealPlanner(ai.df)

        target_macros = {
            'calories': profile['target_calories'],
            'protein': profile['target_protein'],
            'carbs': profile['target_carbs'],
            'fat': profile['target_fat']
        }

        with st.spinner("Génération du plan optimal..."):
            plan, totals = planner.generate_daily_plan_personalized(target_macros)

        st.success("Plan généré avec succès !")

        # Affichage par repas
        for meal, foods in plan.items():
            total_cal = sum(f['Calories'] for f in foods)

            with st.expander(f"**{meal}** – {total_cal:.0f} kcal", expanded=True):
                for food in foods:
                    role_emoji = {
                        'Privilégier': '✅',
                        'Modération': '⚠️',
                        'Neutre': 'ℹ️',
                        'Éviter': '❌'
                    }.get(food['role'], '•')

                    st.markdown(
                        f"{role_emoji} **{food['Food Category']}** "
                        f"({food['role']}) - "
                        f"{food['Calories']:.0f} kcal | "
                        f"P:{food['Protein']:.0f}g "
                        f"G:{food['Carbs']:.0f}g "
                        f"L:{food['Fat']:.0f}g | "
                        f"Score: {food.get('user_score', 0):.2f}"
                    )

        # Comparaison cible vs réel
        st.divider()
        st.subheader("Bilan Nutritionnel")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Calories totales",
                f"{totals['calories']:.0f} kcal",
                f"{totals['calories'] - target_macros['calories']:.0f} kcal"
            )
            st.metric(
                "Protéines",
                f"{totals['protein']:.0f}g",
                f"{totals['protein'] - target_macros['protein']:.0f}g"
            )

        with col2:
            st.metric(
                "Glucides",
                f"{totals['carbs']:.0f}g",
                f"{totals['carbs'] - target_macros['carbs']:.0f}g"
            )
            st.metric(
                "Lipides",
                f"{totals['fat']:.0f}g",
                f"{totals['fat'] - target_macros['fat']:.0f}g"
            )

        # Graphique comparatif
        fig, ax = plt.subplots(figsize=(10, 5))

        labels = ['Protéines', 'Glucides', 'Lipides']
        cible = [target_macros['protein'], target_macros['carbs'], target_macros['fat']]
        reel = [totals['protein'], totals['carbs'], totals['fat']]

        x = range(len(labels))
        width = 0.35

        ax.bar([i - width / 2 for i in x], cible, width, label='Objectif', color='#4CAF50', alpha=0.8)
        ax.bar([i + width / 2 for i in x], reel, width, label='Réel', color='#2196F3', alpha=0.8)

        ax.set_ylabel('Grammes', fontsize=12)
        ax.set_xlabel('Macronutriments', fontsize=12)
        ax.set_title('Comparaison Objectif vs Réel', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        st.pyplot(fig)

# Analyse aliment
with tab3:
    st.header("Analyse Détaillée d'un Aliment")

    food_name = st.text_input("Rechercher un aliment", placeholder="Ex: poulet, riz, banane...")

    if food_name:
        matches = ai.df[ai.df['Food Category'].str.contains(food_name, case=False, na=False)]

        if matches.empty:
            st.warning("Aucun aliment trouvé. Essayez un autre terme.")
        else:
            # Prendre le meilleur match
            food = matches.sort_values('user_score', ascending=False).iloc[0]

            # Header
            role_color = {
                'Privilégier': 'green',
                'Modération': 'orange',
                'Neutre': 'blue',
                'Éviter': 'red'
            }

            st.markdown(f"### {food['Food Category']}")
            st.markdown(f"**Rôle:** :{role_color[food['role']]}[{food['role']}]")
            st.markdown(f"**Type de repas:** {food['Meal Type']}")

            st.divider()

            # Métriques
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Calories", f"{food['Calories']:.0f}")
            col2.metric("Protéines", f"{food['Protein']:.1f}g")
            col3.metric("Glucides", f"{food['Carbs']:.1f}g")
            col4.metric("Lipides", f"{food['Fat']:.1f}g")

            col1, col2, col3 = st.columns(3)
            col1.metric("Fibres", f"{food['Fiber']:.1f}g")
            col2.metric("Sucres", f"{food['Sugar']:.1f}g")
            col3.metric("Score perso", f"{food['user_score']:.2f}")

            # Analyse de pertinence pour VOUS
            st.subheader("Pertinence pour votre profil")

            profile = st.session_state.profile

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Fit Protéines",
                    f"{food['protein_fit']:.0%}",
                    f"Cible: {profile['target_protein'] / 6:.0f}g/repas"
                )
                st.metric(
                    "Fit Glucides",
                    f"{food['carbs_fit']:.0%}",
                    f"Cible: {profile['target_carbs'] / 6:.0f}g/repas"
                )

            with col2:
                st.metric(
                    "Fit Lipides",
                    f"{food['fat_fit']:.0%}",
                    f"Cible: {profile['target_fat'] / 6:.0f}g/repas"
                )
                st.metric(
                    "Fit Calories",
                    f"{food['calorie_fit']:.0%}",
                    f"Cible: {profile['target_calories'] / 4:.0f} kcal/repas"
                )

            # Recommandations similaires
            st.subheader("Aliments Similaires Recommandés")

            similar = ai.recommend_similar_personalized(food['Food Category'])

            if similar:
                for sim in similar[:5]:
                    role_emoji = {
                        'Privilégier': '✅',
                        'Modération': '⚠️',
                        'Neutre': 'ℹ️',
                        'Éviter': '❌'
                    }.get(sim['role'], '•')

                    st.markdown(
                        f"{role_emoji} **{sim['name']}** - "
                        f"{sim['role']} | "
                        f"Score: {sim['user_score']:.2f} | "
                        f"{sim['calories']:.0f} kcal, {sim['protein']:.0f}g protéines"
                    )

# Arbre explicatif
with tab4:
    st.header("Arbre de Décision Personnalisé")

    st.info(
        "Cet arbre est **unique à votre profil**. "
    )

    profile = st.session_state.profile

    # Afficher tous les paramètres du profil utilisés
    st.subheader("Paramètres de Votre Profil Utilisés dans l'Arbre")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Âge", f"{profile['age']} ans")
    col1.metric("Genre", profile['gender'])
    col2.metric("Poids", f"{profile['weight']} kg")
    col2.metric("IMC", f"{profile['bmi']:.1f}")
    col3.metric("Activité", profile['activity'])
    col3.metric("TDEE", f"{int(profile['tdee'])} kcal")
    col4.metric("Objectif", profile['objective'])
    col4.metric("Cible", f"{int(profile['target_calories'])} kcal")

    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric("Protéines/repas", f"{profile['target_protein'] / 6:.0f}g")
    col2.metric("Glucides/repas", f"{profile['target_carbs'] / 6:.0f}g")
    col3.metric("Calories/repas", f"{profile['target_calories'] / 4:.0f} kcal")

    st.divider()

    if os.path.exists("models/decision_tree_personalized.pkl"):
        try:
            dt = joblib.load("models/decision_tree_personalized.pkl")
            features = joblib.load("models/tree_features_personalized.pkl")

            # Charger les classes présentes lors de l'entraînement
            if os.path.exists("models/tree_classes.pkl"):
                tree_classes = joblib.load("models/tree_classes.pkl")
            else:
                tree_classes = ['Privilégier', 'Modération', 'Neutre', 'Éviter']

            # Règles textuelles
            st.subheader("Règles de Décision")

            from sklearn.tree import export_text

            rules = export_text(dt, feature_names=features, class_names=tree_classes, max_depth=5)
            st.code(rules, language="text")

            # Note sur les classes
            st.info(f"Classes présentes dans ce modèle : {', '.join(tree_classes)}")

            # Visualisation graphique
            st.subheader("Visualisation de l'Arbre")

            from sklearn.tree import plot_tree

            fig, ax = plt.subplots(figsize=(20, 12))
            plot_tree(
                dt,
                feature_names=features,
                class_names=tree_classes,
                filled=True,
                rounded=True,
                fontsize=10,
                ax=ax,
                proportion=True
            )
            plt.tight_layout()
            st.pyplot(fig)

            # Importance des features
            st.subheader("Importance des Critères")

            importance_df = pd.DataFrame({
                'Critère': features,
                'Importance': dt.feature_importances_
            }).sort_values('Importance', ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(importance_df['Critère'], importance_df['Importance'], color='#4CAF50')
            ax.set_xlabel('Importance')
            ax.set_title('Importance des Critères dans la Classification')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Erreur lors du chargement de l'arbre : {e}")
    else:
        st.warning("L'arbre n'a pas encore été entraîné. Configurez votre profil d'abord.")

# Évaluation et Comparaison
with tab5:
    st.header("Performances et Comparaison des Modèles")

    # Vérifier si l'évaluation a déjà été effectuée
    if not os.path.exists("models/model_comparison.csv"):
        st.warning("Les modèles n'ont pas encore été évalués.")
        st.info(
            "L'évaluation prendra 5-10 minutes et testera 4 algorithmes différents avec optimisation des hyperparamètres.")

        if st.button("Lancer l'Évaluation Complète", type="primary"):
            with st.spinner("Évaluation en cours... Cela peut prendre plusieurs minutes."):
                from model_evaluation import run_full_evaluation

                # Lancer l'évaluation
                evaluator, comparison_df = run_full_evaluation(
                    ai.df,
                    st.session_state.profile
                )

                st.session_state.evaluation_done = True
                st.rerun()
    else:
        st.success("Évaluation disponible")

        #TABLEAU COMPARATIF
        st.subheader("Tableau Comparatif des Modèles")

        comparison_df = pd.read_csv("models/model_comparison.csv")

        # Formater le tableau
        styled_df = comparison_df.style.highlight_max(
            subset=['Accuracy', 'F1-Score', 'Precision', 'Recall'],
            color='lightgreen'
        ).format({
            'Accuracy': '{:.4f}',
            'F1-Score': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'Temps Train (s)': '{:.2f}',
            'Temps Inférence (s)': '{:.4f}'
        })

        st.dataframe(styled_df, use_container_width=True)

        # Identifier le meilleur
        best_idx = comparison_df['F1-Score'].idxmax()
        best_model = comparison_df.loc[best_idx, 'Modèle']
        best_f1 = comparison_df.loc[best_idx, 'F1-Score']

        st.success(f"**Meilleur modèle :** {best_model} (F1-Score: {best_f1:.4f})")

        st.divider()

        # GRAPHIQUES DE PERFORMANCE
        st.subheader("Visualisations des Performances")

        col1, col2 = st.columns(2)

        with col1:
            if os.path.exists("models/f1_comparison.png"):
                st.image("models/f1_comparison.png", caption="Comparaison des F1-Scores")
            else:
                st.warning("Graphique non disponible")

        st.divider()

        # MATRICES DE CONFUSION
        st.subheader("Matrices de Confusion")

        confusion_files = [
            "models/confusion_decision_tree.png",
            "models/confusion_random_forest.png",
            "models/confusion_knn.png",
            "models/confusion_gradient_boosting.png"
        ]

        available = [f for f in confusion_files if os.path.exists(f)]

        if available:
            for f in available:
                st.image(f, caption=os.path.basename(f))
        else:
            st.warning("Matrices de confusion non disponibles")

        st.divider()

        # RAPPORT DÉTAILLÉ
        st.subheader("Rapport Détaillé")

        if os.path.exists("models/evaluation_report.txt"):
            with open("models/evaluation_report.txt", 'r', encoding='utf-8') as f:
                report_content = f.read()

            with st.expander("Voir le rapport complet", expanded=False):
                st.text(report_content)

            # Bouton téléchargement
            st.download_button(
                label="Télécharger le rapport",
                data=report_content,
                file_name="nutriai_evaluation_report.txt",
                mime="text/plain"
            )
        else:
            st.warning("Rapport détaillé non disponible")

        st.divider()

        # ANALYSE DES RÉSULTATS
        st.subheader("Analyse des Résultats")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Points Forts")
            st.markdown(f"""
            - **{len(comparison_df)} modèles** testés et comparés
            - Optimisation automatique des hyperparamètres
            - Validation croisée 5-fold
            - Métriques multiples (Accuracy, F1, Precision, Recall)
            - Modèles adaptés au profil utilisateur
            """)

        with col2:
            st.markdown("### Insights")

            # Calculer quelques stats
            avg_accuracy = comparison_df['Accuracy'].mean()
            avg_f1 = comparison_df['F1-Score'].mean()
            fastest_model = comparison_df.loc[comparison_df['Temps Train (s)'].idxmin(), 'Modèle']
            slowest_model = comparison_df.loc[comparison_df['Temps Train (s)'].idxmax(), 'Modèle']

            st.markdown(f"""
            - Accuracy moyenne: **{avg_accuracy:.3f}**
            - F1-Score moyen: **{avg_f1:.3f}**
            - Plus rapide: **{fastest_model}**
            - Plus lent: **{slowest_model}**
            - Meilleur: **{best_model}**
            """)

        st.divider()

        # RE-ÉVALUATION
        st.subheader("Re-évaluation")

        col1, col2 = st.columns([3, 1])

        with col1:
            st.info("Vous pouvez relancer l'évaluation si vous avez modifié votre profil ou les données.")

        with col2:
            if st.button("Re-évaluer", type="secondary"):
                files_to_remove = [
                    "models/model_comparison.csv",
                    "models/f1_comparison.png",
                    "models/evaluation_report.txt",
                    "models/confusion_decision_tree.png",
                    "models/confusion_random_forest.png",
                    "models/confusion_knn.png",
                    "models/confusion_gradient_boosting.png"
                ]

                for f in files_to_remove:
                    if os.path.exists(f):
                        os.remove(f)

                st.rerun()