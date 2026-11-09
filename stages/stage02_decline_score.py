"""
Stage 2 (v2): Build Custom Decline Score
Project: Credit Card Share of Wallet (SoW) Analytics

What changed from v1:
- v1 used min-max normalization. Total_Ct_Chng_Q4_Q1 and Total_Amt_Chng_Q4_Q1
  have outliers (max ~3.7), so min-max squashed them into a tiny range and
  Avg_Utilization_Ratio dominated the score (std 0.28 vs 0.06).
- v2 uses PERCENTILE RANK instead, so every signal has the same spread
  and equal weights are really equal.
- v2 adds Total_Trans_Ct (low overall activity = disengagement).

Logic (decided by business meaning, NOT by looking at the label):
  lower transaction-count change, lower amount change, lower utilization,
  lower transaction count  ->  more disengagement  ->  higher decline score.
"""

import pandas as pd
from sklearn.metrics import roc_auc_score

IN_PATH = "bankchurners_clean.csv"
OUT_PATH = "bankchurners_with_decline_score.csv"

df = pd.read_csv(IN_PATH)


def decline_signal(series: pd.Series) -> pd.Series:
    """Percentile rank (0-1), inverted so LOWER raw value = HIGHER decline signal."""
    return 1 - series.rank(pct=True)


signals = pd.DataFrame({
    "ct_chng_signal": decline_signal(df["Total_Ct_Chng_Q4_Q1"]),
    "amt_chng_signal": decline_signal(df["Total_Amt_Chng_Q4_Q1"]),
    "util_signal": decline_signal(df["Avg_Utilization_Ratio"]),
    "trans_ct_signal": decline_signal(df["Total_Trans_Ct"]),
})

# Equal weights across the 4 signals
df["Decline_Score"] = signals.mean(axis=1)

print("Decline_Score summary:")
print(df["Decline_Score"].describe())

# Check that no single signal dominates (spread should be similar)
print("\nStd of each signal (should be similar):")
print(signals.std().round(3))
print("\nCorrelation of each signal with Decline_Score:")
print(signals.corrwith(df["Decline_Score"]).round(2))

# ---------------------------------------------------------
# Sanity check ONLY (not used to pick signals):
# does the score line up with actual attrition?
# ---------------------------------------------------------
y = (df["Attrition_Flag"] == "Attrited Customer").astype(int)
print("\nMean Decline_Score by Attrition_Flag:")
print(df.groupby("Attrition_Flag")["Decline_Score"].mean().round(3))
print(f"\nAUC of Decline_Score vs actual attrition: {roc_auc_score(y, df['Decline_Score']):.3f}")

df.to_csv(OUT_PATH, index=False)
print(f"\nSaved dataset with Decline_Score to: {OUT_PATH}")
