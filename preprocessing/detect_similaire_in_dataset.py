import pandas as pd
from fuzzywuzzy import fuzz

df = pd.read_csv("../data/final_food.csv")

categories = df["Food Category"].unique()

groups = {}
visited = set()

SIMILARITY_THRESHOLD = 90

for cat in categories:
    if cat in visited:
        continue

    groups[cat] = [cat]
    visited.add(cat)

    for other in categories:
        if other not in visited:
            similarity = fuzz.ratio(cat.lower(), other.lower())
            if similarity >= SIMILARITY_THRESHOLD:
                groups[cat].append(other)
                visited.add(other)

print("Groupes de catégories similaires trouvés :")
for main_cat, similars in groups.items():
    if len(similars) > 1:
        print(f"- {main_cat} : {similars}")


replacement_map = {}
for main_cat, similars in groups.items():
    for s in similars:
        replacement_map[s] = main_cat

df["Food Category"] = df["Food Category"].map(replacement_map)

df = df.drop_duplicates()

df.to_csv("../data/final_food.csv", index=False)

print("Dataset nettoyé enregistré sous final_food.csv")
