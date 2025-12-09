import pandas as pd
import numpy as np
import random

FINAL_COLUMNS = [
    "Food Category","Meal Type","Calories","Protein","Carbs","Fat",
    "Saturated Fat","Fiber","Sugar","Sodium","Water"
]

MEAL_TYPES = ["Lunch", "Snack", "Breakfast", "Meal"]

def smart_fill(df):
    """
    Remplit les valeurs manquantes avec des calculs cohérents.
    (moyenne, ratios, approximation nutritionnelle)
    """
    # Si Calories manquant → approx via macro nutriments
    # Calories = 4*Protein + 4*Carbs + 9*Fat
    missing_cal = df["Calories"].isna()
    df.loc[missing_cal, "Calories"] = (
        4 * df.loc[missing_cal, "Protein"].fillna(0) +
        4 * df.loc[missing_cal, "Carbs"].fillna(0) +
        9 * df.loc[missing_cal, "Fat"].fillna(0)
    )

    # Si Fat manque → approx via Lipid saturated + 2×non-saturated
    if "Saturated Fat" in df.columns:
        missing_fat = df["Fat"].isna()
        df.loc[missing_fat, "Fat"] = (
            df.loc[missing_fat, "Saturated Fat"].fillna(0) * 1.3
        )

    # Si Carbs manque → approx via Sugar + Fiber + moyenne globale
    if "Carbs" in df.columns and "Fiber" in df.columns and "Sugar" in df.columns:
        missing_carbs = df["Carbs"].isna()
        df.loc[missing_carbs, "Carbs"] = (
            df["Sugar"].fillna(df["Sugar"].mean()) +
            df["Fiber"].fillna(df["Fiber"].mean())
        )

    # Meal Type = random si manque
    if "Meal Type" in df.columns:
        df["Meal Type"] = df["Meal Type"].apply(
            lambda x: x if pd.notna(x) else random.choice(MEAL_TYPES)
        )


    df = df.fillna(df.mean(numeric_only=True))

    return df


df1 = pd.read_csv("../datasets/dataset1.csv")
df2 = pd.read_csv("../datasets/dataset2.csv")
df3 = pd.read_csv("../datasets/dataset3.csv")

df1_norm = pd.DataFrame()
df1_norm["Food Category"] = df1.get("food", None)
df1_norm["Meal Type"] = None
df1_norm["Calories"] = df1["Caloric Value"]
df1_norm["Protein"] = df1["Protein"]
df1_norm["Carbs"] = df1["Carbohydrates"]
df1_norm["Fat"] = df1["Fat"]
df1_norm["Saturated Fat"] = df1["Saturated Fats"]
df1_norm["Fiber"] = df1["Dietary Fiber"]
df1_norm["Sugar"] = df1["Sugars"]
df1_norm["Sodium"] = df1["Sodium"]
df1_norm["Water"] = df1["Water"]

df1_norm = smart_fill(df1_norm)

df2_norm = pd.DataFrame()
df2_norm["Food Category"] = df2["Food Category"]
df2_norm["Meal Type"] = df2["Meal Type"]
df2_norm["Calories"] = df2["Calories"]
df2_norm["Protein"] = df2["Protein"]
df2_norm["Carbs"] = df2["Carbs"]
df2_norm["Fat"] = df2["Fat"]
df2_norm["Saturated Fat"] = df2["Saturated Fat"]
df2_norm["Fiber"] = df2["Fiber"]
df2_norm["Sugar"] = df2["Sugar"]
df2_norm["Sodium"] = df2["Sodium"]
df2_norm["Water"] = df2["Water"]

df2_norm = smart_fill(df2_norm)

df3_norm = pd.DataFrame()
df3_norm["Food Category"] = df3["Category"]
df3_norm["Meal Type"] = None
df3_norm["Calories"] = None
df3_norm["Protein"] = df3["Data.Protein"]
df3_norm["Carbs"] = df3["Data.Carbohydrate"]
df3_norm["Fat"] = df3["Data.Fat.Total Lipid"]
df3_norm["Saturated Fat"] = df3["Data.Fat.Saturated Fat"]
df3_norm["Fiber"] = df3["Data.Fiber"]
df3_norm["Sugar"] = df3["Data.Sugar Total"]
df3_norm["Sodium"] = df3["Data.Major Minerals.Sodium"]
df3_norm["Water"] = df3["Data.Water"]

df3_norm = smart_fill(df3_norm)

combined = pd.concat([df1_norm, df2_norm, df3_norm], ignore_index=True)
combined = combined[FINAL_COLUMNS]


combined.to_csv("../data/final_food.csv", index=False)

print("Fichier généré : final_food.csv")
