import pandas as pd

# load your final CSV used for training/testing (the one you split)
# if you have separate train/test files, load both. Here I'll assume train.csv & test.csv paths:
train_path = "/home/yogeshwar/Yogesh-MTP/csv/Combined/DAPT_train_data.csv"
test_path  = "/home/yogeshwar/Yogesh-MTP/csv/Combined/DAPT_test_data.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Make sure columns have same order
test = test[train.columns]

# How many exact duplicates?
merged = pd.merge(test.assign(_in_test=1), train.assign(_in_train=1), how='inner', on=list(train.columns))
print("Exact duplicate rows found between test and train:", len(merged))

# If >0, show example rows
if len(merged) > 0:
    print(merged.head())
