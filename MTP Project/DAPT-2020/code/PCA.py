import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the scaled data
df = pd.read_csv("/home/yogeshwar/Yogesh-MTP/Models/Transformer/transformer_test_data.csv")

# Separate features and label
X = df.drop(columns=["Stage"])
y = df["Stage"]

# ----- Change this value to experiment -----
N_COMPONENTS = 60# Try values like 20, 40, 60, etc.

# Perform PCA
pca = PCA(n_components=N_COMPONENTS)
X_pca = pca.fit_transform(X)

# Save PCA-reduced dataset
pca_df = pd.DataFrame(X_pca)
pca_df["Stage"] = y
pca_df.to_csv(f"/home/yogeshwar/Yogesh-MTP/code/trial/pca_{N_COMPONENTS}_components_test_data.csv", index=False)

# Plot explained variance
plt.figure(figsize=(8, 4))
plt.plot(range(1, N_COMPONENTS + 1), pca.explained_variance_ratio_, marker='o')
plt.title(f"PCA Explained Variance (n={N_COMPONENTS})")
plt.xlabel("Principal Component")
plt.ylabel("Variance Ratio")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"pca_{N_COMPONENTS}_variance_plot.png")
plt.close()

print(f"Saved PCA-reduced file: pca_{N_COMPONENTS}_components.csv")
