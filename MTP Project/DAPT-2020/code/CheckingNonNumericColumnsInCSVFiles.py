import pandas as pd
import numpy as np

# Load the CSV file (replace with your actual file path)
df = pd.read_csv("/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/combined_cleaned.csv")

# Find non-numeric columns and their data types
non_numeric_columns = {col: df[col].dtype for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])}

# Print the result
print("Non-numeric columns and their data types:")
for col, dtype in non_numeric_columns.items():
    print(f"{col}: {dtype}")

print(f"\nTotal number of non-numeric columns: {len(non_numeric_columns)}")
