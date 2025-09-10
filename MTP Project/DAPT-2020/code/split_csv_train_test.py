import pandas as pd
from sklearn.model_selection import train_test_split

# Load the normalized, numeric CSV
df = pd.read_csv("/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/combined_cleaned_numeric.csv")

# Check if 'Stage' column exists
if 'Stage' not in df.columns:
    raise ValueError("The CSV must contain a column named 'Stage' as the label.")

# Split features and label
X = df.drop(columns=["Stage"])
y = df["Stage"]

# Perform stratified split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

# Combine back features and label for saving
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Save to CSV files
train_df.to_csv("/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/New/Unraveled_train_data.csv", index=False)
test_df.to_csv("/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/New/Unraveled_test_data.csv", index=False)

print("Splitting done.")
print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")
