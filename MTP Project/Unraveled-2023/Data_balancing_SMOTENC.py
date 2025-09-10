# # It is Normal SMOTENC code , It is for DAPT 2020 dataset

# import pandas as pd
# from imblearn.over_sampling import SMOTENC
# from collections import Counter
# import matplotlib.pyplot as plt

# # Load your dataset
# df = pd.read_csv("/home/yogeshwar/Yogesh-MTP/csv/Combined/combined_cleaned.csv")

# # Separate features and target
# X = df.drop("Stage", axis=1)   # features
# y = df["Stage"]                # target

# # Identify categorical feature indices
# categorical_cols = [
#     'Activity'
# ]
# categorical_indices = [X.columns.get_loc(col) for col in categorical_cols]

# print("Categorical feature indices:", categorical_indices)

# # Count before balancing
# before_counts = Counter(y)

# # Apply SMOTENC
# smote_nc = SMOTENC(categorical_features=categorical_indices, random_state=42)
# X_res, y_res = smote_nc.fit_resample(X, y)

# # Count after balancing
# after_counts = Counter(y_res)

# print("Before balancing:", before_counts)
# print("After balancing:", after_counts)

# # Save balanced dataset
# balanced_df = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.DataFrame(y_res, columns=["Stage"])], axis=1)
# balanced_df.to_csv("/home/yogeshwar/Yogesh-MTP/csv/Combined/combined_cleaned_balanced.csv", index=False)

# print("✅ Balanced dataset saved successfully!")

# # ------------------ Visualization ------------------
# fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# # Before balancing
# axes[0].bar(before_counts.keys(), before_counts.values(), color="skyblue")
# axes[0].set_title("Before Balancing")
# axes[0].set_xlabel("Stage")
# axes[0].set_ylabel("Count")
# axes[0].tick_params(axis='x', rotation=45)

# # After balancing
# axes[1].bar(after_counts.keys(), after_counts.values(), color="lightgreen")
# axes[1].set_title("After Balancing (SMOTENC)")
# axes[1].set_xlabel("Stage")
# axes[1].set_ylabel("Count")
# axes[1].tick_params(axis='x', rotation=45)

# plt.tight_layout()
# # Save the figure as an image file
# output_plot = "/home/yogeshwar/Yogesh-MTP/csv/Combined/stage_balancing_plot.png"
# plt.savefig(output_plot)

# print(f"📊 Bar chart saved as: {output_plot}")







# It is SMOTENN code without any cap

# import pandas as pd
# from imblearn.combine import SMOTEENN
# from collections import Counter
# import matplotlib.pyplot as plt

# # Load your dataset
# df = pd.read_csv("/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/combined_cleaned_numeric.csv")

# # Separate features and target
# X = df.drop("Stage", axis=1)   # features
# y = df["Stage"]                # target

# print("✅ All features (except Stage) are numeric, ready for SMOTEENN.")

# # Count before balancing
# before_counts = Counter(y)

# # Apply Hybrid SMOTE (SMOTEENN)
# smote_enn = SMOTEENN(random_state=42, sampling_strategy="auto")
# X_res, y_res = smote_enn.fit_resample(X, y)

# # Count after balancing
# after_counts = Counter(y_res)

# print("Before balancing:", before_counts)
# print("After balancing:", after_counts)

# # Save balanced dataset
# balanced_df = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.DataFrame(y_res, columns=["Stage"])], axis=1)
# balanced_df.to_csv("/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/combined_cleaned_balanced_hybrid.csv", index=False)

# print("✅ Hybrid SMOTE-balanced dataset saved successfully!")

# # ------------------ Visualization ------------------
# fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# # Before balancing
# axes[0].bar(before_counts.keys(), before_counts.values(), color="skyblue")
# axes[0].set_title("Before Balancing")
# axes[0].set_xlabel("Stage")
# axes[0].set_ylabel("Count")
# axes[0].tick_params(axis='x', rotation=45)

# # After balancing
# axes[1].bar(after_counts.keys(), after_counts.values(), color="lightgreen")
# axes[1].set_title("After Balancing (Hybrid SMOTEENN)")
# axes[1].set_xlabel("Stage")
# axes[1].set_ylabel("Count")
# axes[1].tick_params(axis='x', rotation=45)

# plt.tight_layout()
# # Save the figure as an image file
# output_plot = "/home/yogeshwar/Yogesh-MTP/Unraveled_Dataset/unraveled/data/Combined_New/stage_balancing_plot_hybrid.png"
# plt.savefig(output_plot)

# print(f"📊 Hybrid SMOTE bar chart saved as: {output_plot}")






#Makes samples of each class as 100k => SMOTEEN code with cap as 100k

# import pandas as pd
# from imblearn.combine import SMOTEENN
# from imblearn.under_sampling import RandomUnderSampler
# from collections import Counter
# import matplotlib.pyplot as plt

# # Load dataset
# df = pd.read_csv("/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/New/Unraveled_train_data.csv")

# # Separate features and target
# X = df.drop("Stage", axis=1)
# y = df["Stage"]

# print("✅ All features (except Stage) are numeric, ready for SMOTEENN.")

# # ---------------- Step 1: Undersample Benign ----------------
# rus = RandomUnderSampler(sampling_strategy={'Benign': 100000}, random_state=42)
# X_resampled, y_resampled = rus.fit_resample(X, y)
# print("After undersampling:", Counter(y_resampled))

# # ---------------- Step 2: Apply SMOTEENN ----------------
# # Target: max 70k for each class
# target_counts = {cls: 70000 for cls in y_resampled.unique()}

# smote_enn = SMOTEENN(sampling_strategy=target_counts, random_state=42)
# X_final, y_final = smote_enn.fit_resample(X_resampled, y_resampled)

# print("Final balanced counts:", Counter(y_final))

# # Save balanced dataset
# balanced_df = pd.concat([pd.DataFrame(X_final, columns=X.columns), pd.DataFrame(y_final, columns=["Stage"])], axis=1)
# balanced_df.to_csv("/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/New/Unraveled_train_data_balanced.csv", index=False)

# print("✅ Hybrid balanced dataset saved successfully!")

# # ---------------- Visualization ----------------
# fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# # Before
# before_counts = Counter(y)
# axes[0].bar(before_counts.keys(), before_counts.values(), color="skyblue")
# axes[0].set_title("Before Balancing")
# axes[0].tick_params(axis='x', rotation=45)

# # After
# after_counts = Counter(y_final)
# axes[1].bar(after_counts.keys(), after_counts.values(), color="lightgreen")
# axes[1].set_title("After Balancing (Hybrid SMOTEENN + Undersampling)")
# axes[1].tick_params(axis='x', rotation=45)

# plt.tight_layout()
# plt.savefig("/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/New/stage_balancing_plot_fixed.png")
# print("📊 Bar chart saved.")



#Makes samples of each class as 70k => SMOTEEN code with cap as 70k

import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/New/Unraveled_train_data.csv")

# Separate features and target
X = df.drop("Stage", axis=1)
y = df["Stage"]

print("✅ All features (except Stage) are numeric, ready for SMOTEENN.")

# ---------------- Step 1: Undersample Benign ----------------
rus = RandomUnderSampler(sampling_strategy={'Benign': 70000}, random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)
print("After undersampling:", Counter(y_resampled))

# ---------------- Step 2: Apply SMOTEENN ----------------
smote_enn = SMOTEENN(sampling_strategy="not majority", random_state=42)
X_final, y_final = smote_enn.fit_resample(X_resampled, y_resampled)

print("Final balanced counts:", Counter(y_final))

# Save balanced dataset
balanced_df = pd.concat([pd.DataFrame(X_final, columns=X.columns), pd.DataFrame(y_final, columns=["Stage"])], axis=1)
balanced_df.to_csv("/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/New/Unraveled_train_data_balanced.csv", index=False)

print("✅ Hybrid balanced dataset saved successfully!")

# ---------------- Visualization ----------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Before
before_counts = Counter(y)
axes[0].bar(before_counts.keys(), before_counts.values(), color="skyblue")
axes[0].set_title("Before Balancing")
axes[0].tick_params(axis='x', rotation=45)

# After
after_counts = Counter(y_final)
axes[1].bar(after_counts.keys(), after_counts.values(), color="lightgreen")
axes[1].set_title("After Balancing (SMOTEENN after Benign undersampling)")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/New/stage_balancing_plot_fixed.png")
print("📊 Bar chart saved.")
