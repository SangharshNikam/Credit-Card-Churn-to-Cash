"""
Stage 7: Model Evaluation - Gini, Gain/Lift charts, Calibration
Project: Credit Card Share of Wallet (SoW) Analytics

Input: stage6_test_predictions.csv (test-set probabilities saved by Stage 6)

What this script does:
1. Gini for each model (Gini = 2 x ROC-AUC - 1)
2. Gain & Lift table + charts (decile-wise) for the final model, LightGBM
   - Gain: "if we contact the top X% riskiest customers, what % of all
     attrition do we catch?"
   - Lift: "how many times better than picking customers at random?"
3. Calibration check: when the model says 70%, do ~70% really attrite?
   (This matters because Stage 8 multiplies revenue by the probability.)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss

# Copy the real path from the Kaggle Input panel (click the file -> copy path)
PATH = "/kaggle/input/YOUR-STAGE6-NOTEBOOK-NAME/stage6_test_predictions.csv"
FINAL_MODEL = "proba_LightGBM"

df = pd.read_csv(PATH, index_col=0)
print("Loaded:", df.shape)
print("Columns:", df.columns.tolist())
print(f"Attrition rate in test set: {df['y_true'].mean():.4f}")

# ---------------------------------------------------------
# 1. Gini for every model
# ---------------------------------------------------------
print("\n=== Gini per model ===")
for col in [c for c in df.columns if c.startswith("proba_")]:
    auc = roc_auc_score(df["y_true"], df[col])
    print(f"{col:28s} ROC-AUC = {auc:.4f}   Gini = {2 * auc - 1:.4f}")

# ---------------------------------------------------------
# 2. Gain / Lift table (deciles) for the final model
# ---------------------------------------------------------
d = df[["y_true", FINAL_MODEL]].copy()
d["rank"] = d[FINAL_MODEL].rank(method="first", ascending=False)
d["decile"] = pd.qcut(d["rank"], 10, labels=range(1, 11)).astype(int)   # 1 = riskiest 10%

base_rate = d["y_true"].mean()
table = d.groupby("decile").agg(customers=("y_true", "size"), attrited=("y_true", "sum"))
table["attrition_rate"] = table["attrited"] / table["customers"]
table["cum_attrited"] = table["attrited"].cumsum()
table["cum_gain_pct"] = table["cum_attrited"] / table["attrited"].sum() * 100
table["lift"] = table["attrition_rate"] / base_rate
table["cum_lift"] = (table["cum_attrited"] / table["customers"].cumsum()) / base_rate

print(f"\n=== Gain / Lift table ({FINAL_MODEL}) ===")
print(table.round(3).to_string())
print(f"\nTop 10% riskiest customers capture {table.loc[1, 'cum_gain_pct']:.1f}% of all attrition")
print(f"Top 20% riskiest customers capture {table.loc[2, 'cum_gain_pct']:.1f}% of all attrition")
print(f"Top-decile lift: {table.loc[1, 'lift']:.2f}x better than random")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
x = np.arange(0, 101, 10)
ax[0].plot(x, [0] + table["cum_gain_pct"].tolist(), marker="o", label="Model")
ax[0].plot([0, 100], [0, 100], "--", color="gray", label="Random")
ax[0].set_title("Cumulative Gain")
ax[0].set_xlabel("% of customers contacted (riskiest first)")
ax[0].set_ylabel("% of attrition captured")
ax[0].legend()
ax[1].bar(table.index, table["lift"])
ax[1].axhline(1, linestyle="--", color="gray")
ax[1].set_title("Lift by Decile")
ax[1].set_xlabel("Decile (1 = riskiest 10%)")
ax[1].set_ylabel("Lift vs random")
plt.tight_layout()
plt.savefig("/kaggle/working/stage7_gain_lift.png", dpi=150)
plt.show()

# ---------------------------------------------------------
# 3. Calibration
# ---------------------------------------------------------
y, p = df["y_true"].to_numpy(), df[FINAL_MODEL].to_numpy()
frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")

bins = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
ece = sum(abs(y[bins == b].mean() - p[bins == b].mean()) * (bins == b).mean() for b in range(10))

print(f"\n=== Calibration ({FINAL_MODEL}) ===")
print(f"Brier score: {brier_score_loss(y, p):.4f}")
print(f"Expected Calibration Error (ECE): {ece:.4f}")
print(pd.DataFrame({"mean_predicted": mean_pred, "actual_rate": frac_pos}).round(3).to_string())

plt.figure(figsize=(5.5, 5))
plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
plt.plot(mean_pred, frac_pos, marker="o", label=FINAL_MODEL)
plt.xlabel("Mean predicted probability")
plt.ylabel("Actual attrition rate")
plt.title("Calibration curve")
plt.legend()
plt.tight_layout()
plt.savefig("/kaggle/working/stage7_calibration.png", dpi=150)
plt.show()
