import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/New/Unraveled_test_data.csv")

# Separate features and label
features = df.iloc[:, :-1]  # All columns except the last
label = df.iloc[:, -1]      # Last column (Stage)

# Apply Standard Scaling (Z-score normalization) to features only
scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)

# Combine standardized features and original label
standardized_df = pd.DataFrame(standardized_features, columns=features.columns)
standardized_df['Stage'] = label  # Append the label column as-is

# Save the standardized dataset
standardized_df.to_csv("/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/New/Unraveled_test_data_standard_scaled.csv", index=False)

print("✅ Standard normalization complete (excluding label column). Saved as 'custom_combined_standard_scaled.csv'")
