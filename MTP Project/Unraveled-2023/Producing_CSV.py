# import pandas as pd
# import os

# # ----------- Configurable Inputs ------------
# base_folder = "/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/network-flows"
# stage_to_collect = "Data Exfiltration"   # change this for other stages
# output_csv = "/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Data_Exfiltration/final.csv"
# # --------------------------------------------

# final_entries = []

# # Walk through all subdirectories and CSV files
# for root, dirs, files in os.walk(base_folder):
#     for file in files:
#         if file.endswith(".csv"):
#             path = os.path.join(root, file)
#             try:
#                 # Get the stage column name (3rd last column)
#                 header = pd.read_csv(path, nrows=0).columns
#                 if len(header) < 3:
#                     print(f"⚠️ Skipping {path}, not enough columns.")
#                     continue

#                 stage_col = header[-3]

#                 try:
#                     df = pd.read_csv(path, on_bad_lines='skip', engine='python')
#                 except Exception as e:
#                     print(f"❌ Error reading {path}: {e}")
#                     continue

#                 filtered = df[df[stage_col] == stage_to_collect]

#                 if len(filtered) == 0:
#                     print(f"⚠️ No entries for stage '{stage_to_collect}' in {path}")
#                     continue

#                 final_entries.append(filtered)
#                 print(f"✅ Collected {len(filtered)} entries of stage '{stage_to_collect}' from {path}")

#             except Exception as e:
#                 print(f"❌ Error processing {path}: {e}")

# # Save the final CSV (with single header)
# if final_entries:
#     final_df = pd.concat(final_entries, ignore_index=True)
#     final_df.to_csv(output_csv, index=False)
#     print(f"\n✅ Final combined CSV saved as: {output_csv} ({len(final_df)} total rows)")
# else:
#     print("⚠️ No data collected.")









#for DAPT 2020
import pandas as pd
import os

# ----------- Configurable Inputs ------------
base_folder = "/home/yogeshwar/Yogesh-MTP/DAPT_CSV"   # folder containing 10 CSVs
stage_to_collect = "Data Exfiltration"                # change this for other stages
output_csv = "/home/yogeshwar/Yogesh-MTP/csv/Data_Exfiltration/final_stage_data.csv"
# --------------------------------------------

final_entries = []

# Iterate through all CSV files in the folder
for file in os.listdir(base_folder):
    if file.endswith(".csv"):
        path = os.path.join(base_folder, file)
        try:
            df = pd.read_csv(path, on_bad_lines='skip', engine='python')

            # last column is always Stage
            stage_col = df.columns[-1]

            # filter rows by stage
            filtered = df[df[stage_col] == stage_to_collect]

            if not filtered.empty:
                final_entries.append(filtered)
                print(f"✅ Collected {len(filtered)} entries of stage '{stage_to_collect}' from {file}")
            else:
                print(f"⚠️ No entries for stage '{stage_to_collect}' in {file}")

        except Exception as e:
            print(f"❌ Error processing {path}: {e}")

# Save final CSV with a single header
if final_entries:
    final_df = pd.concat(final_entries, ignore_index=True)
    final_df.to_csv(output_csv, index=False)
    print(f"\n✅ Final combined CSV saved as: {output_csv} ({len(final_df)} total rows)")
else:
    print("⚠️ No data collected.")
