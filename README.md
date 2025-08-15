# Rna3d-Minimal-Train
Overview
If you sat down to complete a puzzle without knowing what it should look like, you’d have to rely on patterns and logic to piece it together. In the same way, predicting Ribonucleic acid (RNA)’s 3D structure involves using only its sequence to figure out how it folds into the structures that define its function.

In this competition, you’ll develop machine learning models to predict an RNA molecule’s 3D structure from its sequence. The goal is to improve our understanding of biological processes and drive new advancements in medicine and biotechnology.
# first the dependemcies
import os
import pandas as pd

# === SETTINGS ===
excel_file = "data_overview.xlsx"  # Excel with any metadata
train_folder = "train_sequence"
test_folder = "test_sequence"
output_folder = "processed_data"

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# === 1. LOAD EXCEL FILE ===
if os.path.exists(excel_file):
    excel_df = pd.read_excel(excel_file)
    print(f"[INFO] Loaded Excel file with shape: {excel_df.shape}")
else:
    excel_df = pd.DataFrame()
    print("[WARNING] Excel file not found, skipping metadata load.")

# === 2. LOAD TRAINING DATA FROM FOLDER ===
train_files = [f for f in os.listdir(train_folder) if f.endswith(".csv")]
train_dfs = []

for file in train_files:
    path = os.path.join(train_folder, file)
    df = pd.read_csv(path)
    train_dfs.append(df)
    print(f"[INFO] Loaded train file: {file} shape: {df.shape}")

if train_dfs:
    train_df = pd.concat(train_dfs, ignore_index=True)
    print(f"[INFO] Combined train data shape: {train_df.shape}")
else:
    train_df = pd.DataFrame()
    print("[WARNING] No train files found.")

# === 3. LOAD TEST DATA FROM FOLDER ===
test_files = [f for f in os.listdir(test_folder) if f.endswith(".csv")]
test_dfs = []

for file in test_files:
    path = os.path.join(test_folder, file)
    df = pd.read_csv(path)
    test_dfs.append(df)
    print(f"[INFO] Loaded test file: {file} shape: {df.shape}")

if test_dfs:
    test_df = pd.concat(test_dfs, ignore_index=True)
    print(f"[INFO] Combined test data shape: {test_df.shape}")
else:
    test_df = pd.DataFrame()
    print("[WARNING] No test files found.")

# === 4. BASIC CLEANING ===
def clean_dataframe(df):
    if df.empty:
        return df
    df.columns = [c.strip().lower() for c in df.columns]  # lowercase column names
    df = df.drop_duplicates()
    df = df.fillna("")  # fill missing values
    return df

train_df = clean_dataframe(train_df)
test_df = clean_dataframe(test_df)

# === 5. SAVE PROCESSED FILES ===
train_df.to_csv(os.path.join(output_folder, "train.csv"), index=False)
test_df.to_csv(os.path.join(output_folder, "test.csv"), index=False)

print("[INFO] Saved processed train and test data.")

# Optional: Save metadata if available
if not excel_df.empty:
    excel_df.to_csv(os.path.join(output_folder, "metadata.csv"), index=False)
    print("[INFO] Saved metadata file.")
