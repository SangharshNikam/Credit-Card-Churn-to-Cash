"""
Stage 4: Chi-square Test of Independence
Project: Credit Card Share of Wallet (SoW) Analytics

Question: Is attrition related to a categorical feature
(Gender, Card_Category, Income_Category, ...) or is it independent of it?

H0: the feature and attrition are independent (no relationship)
H1: the feature and attrition are related

Tests used:
1. Chi-square test of independence -> is there a relationship?
2. Cramer's V                      -> how STRONG is it (0 = none, 1 = perfect)
3. Expected-count check            -> chi-square needs expected counts >= 5 per cell
"""

import numpy as np
import pandas as pd
from scipy import stats

# Change this path to your Stage 2 output file
PATH = "/kaggle/working/bankchurners_with_decline_score.csv"
df = pd.read_csv(PATH)

df["Attrited"] = (df["Attrition_Flag"] == "Attrited Customer").astype(int)

# Card_Category has very few Platinum (20) and Gold (116) customers, which breaks
# the expected-count rule. Merge them into one group for testing.
df["Card_Category_Merged"] = df["Card_Category"].replace({"Gold": "Gold/Platinum", "Platinum": "Gold/Platinum"})

cat_cols = ["Gender", "Education_Level", "Marital_Status",
            "Income_Category", "Card_Category_Merged", "Dependent_count"]


def cramers_v(table: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    r, k = table.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


rows = []
for col in cat_cols:
    table = pd.crosstab(df[col], df["Attrited"])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    rows.append({
        "feature": col,
        "chi2": chi2,
        "dof": dof,
        "p_value": p,
        "cramers_v": cramers_v(table),
        "min_expected_count": expected.min(),
    })

result = pd.DataFrame(rows).sort_values("cramers_v", ascending=False)
print("Chi-square results (sorted by strength):")
print(result.round(4).to_string(index=False))

# Attrition rate per category, so you can see WHAT drives the relationship
for col in cat_cols:
    print(f"\nAttrition rate by {col}:")
    print((df.groupby(col)["Attrited"].mean() * 100).round(1).sort_values(ascending=False).to_string())

# Multiple tests -> Bonferroni-adjusted alpha
alpha = 0.05 / len(cat_cols)
print(f"\nBonferroni-adjusted alpha for {len(cat_cols)} tests: {alpha:.4f}")
print("Significant features:", result.loc[result["p_value"] < alpha, "feature"].tolist())
