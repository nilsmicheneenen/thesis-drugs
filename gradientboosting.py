import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import classification_report

# Config and importation of the training dataset

FILE_PATH = "CDER BLA.xlsx"
SHEET_NAME = "Complete Life Cycle"
RANDOM_STATE = 42
TEST_SIZE = 0.20  # as per your request


# Formula to cleam
def find_year_cols(columns):
    """Finds and sorts columns that are valid years."""
    years = [str(c) for c in columns if re.fullmatch(r"\d{4}", str(c))]
    return sorted(years, key=lambda x: int(x))

def coerce_years_to_numeric(df, year_cols):
    """Cleans and converts year columns to numeric types."""
    for c in year_cols:
        s = df[c].astype(str)
        s = s.str.replace(r"[\s,]", "", regex=True)
        s = s.str.replace(r"[^0-9\.\-]", "", regex=True)
        df[c] = pd.to_numeric(s, errors="coerce")
    return df

def sales_to_bucket(s):
    """Converts a continuous sales value into a discrete bucket."""
    if s >= 1000:
        return "blockbuster"
    elif s >= 500:
        return "high"
    elif s >= 100:
        return "medium"
    else:
        return "below_100"


# Data Loading

print(f"Loading data from {FILE_PATH}...")
try:
    df_full = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
except FileNotFoundError:
    print(f"Error: The file '{FILE_PATH}' was not found.")
    exit()

if "Modaality" in df_full.columns:
    df_full = df_full.rename(columns={"Modaality": "Modality"})

year_cols = find_year_cols(df_full.columns)
df_full = coerce_years_to_numeric(df_full, year_cols)


# Target: Simple mean of the first 13 years after approval
# Create ApprovalYear first (used both for target and features)
df_full["ApprovalYear"] = pd.to_datetime(df_full["Approval Date"], errors="coerce").dt.year

def mean_first_13_years(row):
    ay = row["ApprovalYear"]
    if pd.isna(ay):
        return np.nan
    # Build the list of calendar-year columns for years [ay, ay+12] that exist in the data
    cols = [str(int(ay + i)) for i in range(13) if str(int(ay + i)) in df_full.columns]
    if not cols:
        return np.nan
    # Simple arithmetic mean across available years (skips NaN)
    return row[cols].mean(skipna=True)

print("Calculating simple 13-year mean sales from approval year...")
df_full["Avg13yrSales"] = df_full.apply(mean_first_13_years, axis=1)
df_full = df_full.dropna(subset=["Avg13yrSales"])
df_full = df_full[df_full["Avg13yrSales"] > 0].copy()


# Main Evaluation Loop

results = []
years_to_review_list = range(2, 11)

for years in years_to_review_list:
    print(f"\n--- Training model with years_of_review = {years} ---")
    df = df_full.copy()

    # Feature Engineering based on the current 'years_of_review'
    # (ApprovalYear already computed above)
    for i in range(1, years + 1):
        df[f"Sales_Year_{i}"] = df.apply(
            lambda row: row.get(str(int(row["ApprovalYear"] + i - 1)) if pd.notna(row["ApprovalYear"]) else np.nan, np.nan),
            axis=1
        )

    cat_features = ["Therapeutic Area", "Modality"]
    num_features = [f"Sales_Year_{i}" for i in range(1, years + 1)]
    X = df[cat_features + num_features]
    y = df["Avg13yrSales"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ],
        remainder='passthrough'
    )

    # Fixed hyperparameters (fast loop)
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(
            random_state=RANDOM_STATE,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=10,
            subsample=1,
            max_features='sqrt'
        )),
    ])

    # Train the gradient boosting regressor
    model.fit(X_train, y_train)

    # --- Performance Calculation ---
    y_pred_train_cont = model.predict(X_train)
    y_pred_test_cont = model.predict(X_test)

    y_true_train_bucket = y_train.apply(sales_to_bucket)
    y_pred_train_bucket = pd.Series(y_pred_train_cont, index=y_train.index).apply(sales_to_bucket)
    y_true_test_bucket = y_test.apply(sales_to_bucket)
    y_pred_test_bucket = pd.Series(y_pred_test_cont, index=y_test.index).apply(sales_to_bucket)

    report_train = classification_report(y_true_train_bucket, y_pred_train_bucket, output_dict=True, zero_division=0)
    report_test = classification_report(y_true_test_bucket, y_pred_test_bucket, output_dict=True, zero_division=0)

    results.append({
        "years": years,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_accuracy": report_train['accuracy'],
        "test_accuracy": report_test['accuracy'],
        "train_recall_macro": report_train['macro avg']['recall'],
        "test_recall_macro": report_test['macro avg']['recall'],
        "train_f1_macro": report_train['macro avg']['f1-score'],
        "test_f1_macro": report_test['macro avg']['f1-score'],
        "train_f1_blockbuster": report_train.get('blockbuster', {}).get('f1-score', 0),
        "test_f1_blockbuster": report_test.get('blockbuster', {}).get('f1-score', 0),
        "train_f1_high": report_train.get('high', {}).get('f1-score', 0),
        "test_f1_high": report_test.get('high', {}).get('f1-score', 0),
        "train_f1_medium": report_train.get('medium', {}).get('f1-score', 0),
        "test_f1_medium": report_test.get('medium', {}).get('f1-score', 0),
    })


# export of the table for the thesis
print("\n\n" + "="*130)
print("                                   Model Performance vs. Years of Review (Overall Metrics)")
print("="*130)
print("| Years | Train Size | Test Size | Train Acc | Test Acc | Train Recall (Macro) | Test Recall (Macro) | Train F1 (Macro) | Test F1 (Macro) |")
print("| :---: | :--------: | :-------: | :-------: | :------: | :------------------: | :-----------------: | :--------------: | :-------------: |")
for res in results:
    print(
        f"| {res['years']:<5} | {res['train_size']:<10} | {res['test_size']:<9} | "
        f"{res['train_accuracy']:.2f}      | {res['test_accuracy']:.2f}     | "
        f"{res['train_recall_macro']:.2f}                  | {res['test_recall_macro']:.2f}                | "
        f"{res['train_f1_macro']:.2f}             | {res['test_f1_macro']:.2f}            |"
    )
print("="*130)

print("\n\n" + "="*120)
print("                                  Model Performance vs. Years of Review (Per Category F1-Score)")
print("="*120)
print("| Years | Train Size | Test Size | F1 (Blockbuster) |   F1 (High)    |   F1 (Medium)  |")
print("|-------|------------|-----------|------------------|----------------|----------------|")
print("|       |            |           |  Train |  Test   |  Train |  Test  |  Train |  Test  |")
print("| :---: | :--------: | :-------: | :----: | :-----: | :----: | :----: | :----: | :----: |")
for res in results:
    print(
        f"| {res['years']:<5} | {res['train_size']:<10} | {res['test_size']:<9} | "
        f"{res['train_f1_blockbuster']:.2f}   |  {res['test_f1_blockbuster']:.2f}   | "
        f"{res['train_f1_high']:.2f}   | {res['test_f1_high']:.2f}  | "
        f"{res['train_f1_medium']:.2f}   | {res['test_f1_medium']:.2f}  |"
    )
print("="*120)
