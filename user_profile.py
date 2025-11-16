# user_profile.py

def calculate_bmr(weight, height, age, gender):
    """
    BMR Mifflin-St Jeor — plus précis que Harris-Benedict
    """
    gender = gender.lower()
    if gender.startswith("h") or gender.startswith("m"):
        # homme
        return (10 * weight) + (6.25 * height) - (5 * age) + 5
    else:
        # femme
        return (10 * weight) + (6.25 * height) - (5 * age) - 161


def calculate_tdee(bmr, activity):
    """
    activity doit être la version affichée dans Streamlit : 'Sédentaire', 'Modéré'...
    """
    multipliers = {
        "Sédentaire": 1.2,
        "Léger": 1.375,
        "Modéré": 1.55,
        "Intense": 1.725,
        "Athlète": 1.9
    }
    return bmr * multipliers.get(activity, 1.55)


def get_macro_targets(tdee, objective, weight):
    """
    Répartition intelligente selon objectif + poids
    """
    # Ajustement calories
    if objective == "perte":
        cal = tdee - 400        # déficit modéré
        prot = weight * 2.3     # haute protéine en sèche
    elif objective == "gain":
        cal = tdee + 350        # Lean bulk
        prot = weight * 1.8
    else:
        cal = tdee              # maintien
        prot = weight * 2.0

    # Lipides ≈ 0.9g/kg
    fat = weight * 0.9

    # Glucides = calories restantes
    carb = (cal - (prot * 4 + fat * 9)) / 4

    return {
        "calories": int(cal),
        "protein": int(prot),
        "fat": int(fat),
        "carbs": int(carb)
    }
