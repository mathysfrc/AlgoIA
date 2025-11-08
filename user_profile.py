# user_profile.py
def calculate_bmr(weight, height, age, gender):
    if gender.lower() == "homme":
        return 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        return 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)


def calculate_tdee(bmr, activity):
    multipliers = {
        "Sédentaire": 1.2,
        "Léger": 1.375,
        "Modéré": 1.55,
        "Intense": 1.725,
        "Athlète": 1.9
    }
    return bmr * multipliers.get(activity, 1.55)


def get_macro_targets(tdee, objective, weight):
    if objective == "perte":
        cal = tdee - 500
        prot = weight * 2.2
    elif objective == "gain":
        cal = tdee + 500
        prot = weight * 1.8
    else:
        cal = tdee
        prot = weight * 2.0

    fat = weight * 0.9
    carb = (cal - (prot * 4 + fat * 9)) / 4
    return {
        "calories": int(cal),
        "protein": int(prot),
        "fat": int(fat),
        "carbs": int(carb)
    }