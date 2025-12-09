import pandas as pd

files = [
    "../datasets/5_to_1_dataset/dataset1.csv",
    "../datasets/5_to_1_dataset/dataset2.csv",
    "../datasets/5_to_1_dataset/dataset3.csv",
    "../datasets/5_to_1_dataset/dataset4.csv",
    "../datasets/5_to_1_dataset/dataset5.csv"
]

dfs = [pd.read_csv(f) for f in files]
combined_df = pd.concat(dfs, ignore_index=True)

if "Unnamed: 0" in combined_df.columns:
    combined_df = combined_df.drop(columns=["Unnamed: 0"])

combined_df.to_csv("../datasets/dataset1.csv", index=False)

print("Dataset combiné créé : datasets/dataset1.csv")
