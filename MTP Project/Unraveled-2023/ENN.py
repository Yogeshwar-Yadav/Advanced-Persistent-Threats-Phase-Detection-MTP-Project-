import pandas as pd
from imblearn.under_sampling import EditedNearestNeighbours
from collections import Counter

# === Load SMOTENC balanced dataset ===
balanced_df = pd.read_csv("/home/yogeshwar/Yogesh-MTP/csv/Combined/combined_cleaned_balanced_numeric.csv")

# Split features and target
X = balanced_df.drop("Stage", axis=1)
y = balanced_df["Stage"]

print("Before ENN:", Counter(y))

# === Apply Edited Nearest Neighbours (ENN) ===
enn = EditedNearestNeighbours(n_neighbors=3, n_jobs=-1)  # n_jobs=-1 uses all cores
X_res, y_res = enn.fit_resample(X, y)

print("After ENN:", Counter(y_res))

# === Save ENN cleaned dataset ===
enn_df = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.DataFrame(y_res, columns=["Stage"])], axis=1)
enn_df.to_csv("/home/yogeshwar/Yogesh-MTP/csv/Combined/combined_cleaned_balanced_ENN.csv", index=False)

print("✅ ENN-cleaned dataset saved as combined_cleaned_balanced_ENN.csv")
