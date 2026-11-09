"""
Stage 1: Data Loading + EDA
Project: Credit Card Share of Wallet (SoW) Analytics
Dataset: BankChurners (Credit Card Customers) - Kaggle

What this script does:
1. Loads the raw CSV
2. Drops junk columns (CLIENTNUM, the 2 Naive Bayes classifier columns)
3. Checks shape, nulls, dtypes
4. Compares key metrics between Attrited vs Existing customers
5. Saves a cleaned CSV for use in later stages
"""

import pandas as pd

# ---------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------
RAW_PATH = "BankChurners.csv"   # change path if needed
df = pd.read_csv(RAW_PATH)

print("Raw shape:", df.shape)
print("\nNulls per column:\n", df.isnull().sum())
print("\nDtypes:\n", df.dtypes)

# ---------------------------------------------------------
# 2. Drop junk columns
#    - CLIENTNUM: just an ID, no predictive value
#    - Naive_Bayes_Classifier_*: pre-computed junk columns bundled
#      with this dataset, not meant to be used as features
# ---------------------------------------------------------
cols_to_drop = ["CLIENTNUM"] + [c for c in df.columns if c.startswith("Naive_Bayes")]
df = df.drop(columns=cols_to_drop)

print("\nDropped columns:", cols_to_drop)
print("Cleaned shape:", df.shape)

# ---------------------------------------------------------
# 3. Target variable check
# ---------------------------------------------------------
print("\nAttrition_Flag value counts:")
print(df["Attrition_Flag"].value_counts())

attrition_rate = (df["Attrition_Flag"] == "Attrited Customer").mean() * 100
print(f"\nAttrition rate: {attrition_rate:.2f}%")
# Note: ~16% attrition -> real class imbalance (~1:5).
# This is why later stages use cost-sensitive thresholds instead
# of a default 0.5 cutoff.

# ---------------------------------------------------------
# 4. EDA - compare key metrics: Attrited vs Existing customers
#    This is the core validation for the whole project:
#    do attrited customers show a "quiet decline" before leaving?
# ---------------------------------------------------------
key_cols = [
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Total_Amt_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
    "Credit_Limit",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
]

comparison = df.groupby("Attrition_Flag")[key_cols].mean().round(2).T
print("\n=== Numeric summary by Attrition_Flag ===")
print(comparison)

# ---------------------------------------------------------
# 5. Correlation check (numeric features only)
# ---------------------------------------------------------
numeric_df = df.select_dtypes(include="number")
corr = numeric_df.corr(numeric_only=True)
print("\nTop correlations with Avg_Utilization_Ratio:")
print(corr["Avg_Utilization_Ratio"].sort_values(ascending=False).head(10))

# ---------------------------------------------------------
# 6. Save cleaned dataset for Stage 2 onward
# ---------------------------------------------------------
OUT_PATH = "bankchurners_clean.csv"
df.to_csv(OUT_PATH, index=False)
print(f"\nSaved cleaned dataset to: {OUT_PATH}")
