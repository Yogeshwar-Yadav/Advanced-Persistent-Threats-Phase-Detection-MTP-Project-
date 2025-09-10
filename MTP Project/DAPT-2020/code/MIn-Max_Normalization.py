import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv("/home/yogeshwar/Yogesh-MTP/csv/new_csv/custom_combined_numeric.csv")

# Separate features and label
features = df.iloc[:, :-1]  # All columns except the last
label = df.iloc[:, -1]      # Last column (Stage)

# Apply Min-Max Scaling to features only
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Combine scaled features and original label
scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
scaled_df['Stage'] = label  # Append the label column as-is

# Save the final scaled dataset
scaled_df.to_csv("/home/yogeshwar/Yogesh-MTP/csv/new_csv/custom_combined_scaled.csv", index=False)

print("Scaling complete (excluding label column). Saved as 'custom_combined_scaled.csv'")
