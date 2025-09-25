from flask import Flask, render_template, request
from model import NutritionRecommender
import base64
from io import BytesIO
import matplotlib.pyplot as plt

app = Flask(__name__)
recommender = NutritionRecommender("dataset_clean.csv")

# -----------------------------
# Analyse calories par catégorie
# -----------------------------
calories_avg = recommender.data.groupby("Food Category")["Calories"].mean().sort_values(ascending=False)

print("Calories moyennes par catégorie :")
print(calories_avg)

print("\nDétails Fruits :")
print(recommender.data[recommender.data["Food Category"]=="Fruits"][["Food Category","Meal Type","Calories"]])

print("\nDétails Meat :")
print(recommender.data[recommender.data["Food Category"]=="Meat"][["Food Category","Meal Type","Calories"]])

# -----------------------------
# Routes Flask
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    graphs = []
    food_name = ""
    meal_type = ""
    if request.method == "POST":
        food_name = request.form.get("food_name")
        meal_type = request.form.get("meal_type")
        recommendations = recommender.recommend(food_name, meal_type=meal_type)
        if recommendations:
            food_list = [food_name] + recommendations
            for class_name in recommender.classes.keys():
                cols, values = recommender.get_nutrition_data(food_list, class_name, meal_type=meal_type)
                fig, ax = plt.subplots(figsize=(7,4))
                bottom = [0]*len(food_list)
                for i, nutrient_values in enumerate(zip(*values)):
                    ax.bar(food_list, nutrient_values, bottom=bottom, label=cols[i])
                    bottom = [sum(x) for x in zip(bottom, nutrient_values)]
                ax.set_ylabel("Nutrition value")
                ax.set_title(class_name)
                ax.legend()
                buf = BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                graphs.append(image_base64)
                plt.close(fig)

    return render_template("index.html", recommendations=recommendations, graphs=graphs, food_name=food_name)

if __name__ == "__main__":
    app.run(debug=True)
