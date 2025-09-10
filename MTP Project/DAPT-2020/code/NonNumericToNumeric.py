# import pandas as pd
# from sklearn.preprocessing import LabelEncoder

# # Load the CSV file
# df = pd.read_csv("/home/yogeshwar/Yogesh-MTP/csv/custom_dataset_trans/trans_combined.csv")

# # Initialize encoders dictionary to store encoders (optional, for reverse mapping)
# encoders = {}

# # 1. Label encode Flow ID, Src IP, Dst IP, Stage
# for col in ['Flow ID', 'Src IP', 'Dst IP', 'Stage']:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col].astype(str))
#     encoders[col] = le  # Save encoder if you want to decode later

# # 2. Convert Timestamp to numeric (UNIX timestamp)
# df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%d/%m/%Y %I:%M:%S %p", errors='coerce')
# df['Timestamp'] = df['Timestamp'].astype('int64') // 10**9  # Convert to seconds

# # Final check
# non_numeric_cols = df.select_dtypes(exclude=['number']).columns
# print("Remaining non-numeric columns (should be 0):", list(non_numeric_cols))

# # Optional: Save the numeric DataFrame
# df.to_csv("/home/yogeshwar/Yogesh-MTP/csv/custom_dataset_trans/trans_combined_numeric.csv", index=False)


import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the CSV file
df = pd.read_csv("/home/yogeshwar/Yogesh-MTP/csv/Combined/combined_cleaned_balanced.csv")

# Initialize dictionary to store encoders (useful if you want to decode later)
encoders = {}

# List of non-numeric columns (excluding Stage)
cols_to_encode = [
    'Activity'   
]

# Apply LabelEncoder to each column except Stage
for col in cols_to_encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Final check
non_numeric_cols = df.select_dtypes(exclude=['number']).columns
print("Remaining non-numeric columns (should include only Stage):", list(non_numeric_cols))

# Save the numeric DataFrame
df.to_csv("/home/yogeshwar/Yogesh-MTP/csv/Combined/combined_cleaned_balanced_numeric.csv", index=False)

print(f"✅ Saved transformed CSV with {df.shape[1]} columns at /home/yogeshwar/Yogesh-MTP/csv/Combined/combined_cleaned_balanced_numeric.csv")
