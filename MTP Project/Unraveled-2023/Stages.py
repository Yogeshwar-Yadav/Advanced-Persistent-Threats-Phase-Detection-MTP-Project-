import os
import pandas as pd

# Path to the parent directory
base_dir = "/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/network-flows"

# List to collect final results
final_rows = []

for root, dirs, files in os.walk(base_dir):
    csv_files = sorted([f for f in files if f.endswith(".csv")])
    if not csv_files:
        continue

    folder_name = os.path.basename(root)
    folder_written = False  # To ensure folder name appears only once

    for csv_file in csv_files:
        file_path = os.path.join(root, csv_file)
        try:
            # Read only header
            header = pd.read_csv(file_path, nrows=0).columns
            if len(header) < 3:
                continue  # Skip files with insufficient columns

            stage_col = header[-3]  # Third-last column

            # Read only the stage column
            df = pd.read_csv(file_path, usecols=[stage_col])
            stage_counts = df[stage_col].value_counts()

            csv_written = False  # To ensure CSV file name appears only once

            for stage, count in stage_counts.items():
                final_rows.append({
                    "Folder": folder_name if not folder_written else "",
                    "CSV_File": csv_file if not csv_written else "",
                    "Stage_Name": stage,
                    "Stage_Count": count
                })
                folder_written = True
                csv_written = True

        except Exception as e:
            print(f"⚠️ Error in {csv_file} from {folder_name}: {e}")

# Save final result
output_df = pd.DataFrame(final_rows)
output_df.to_csv("stage_summary_formatted.csv", index=False)
print("✅ Output saved to stage_summary_formatted.csv")
