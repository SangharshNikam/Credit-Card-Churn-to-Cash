"""
Stage 3: Hypothesis Testing (t-test)
Project: Credit Card Share of Wallet (SoW) Analytics

Question: Do Attrited customers really have a HIGHER Decline_Score
than Existing customers, or could the gap just be random chance?

H0 (null):        mean Decline_Score of Attrited == mean of Existing
H1 (alternative): mean Decline_Score of Attrited  > mean of Existing

Tests used:
1. Welch's t-test        -> compares the two group means (does not assume equal variance)
2. Mann-Whitney U test   -> backup, makes no normal-distribution assumption
3. Cohen's d             -> effect size (how BIG the gap is, not just if it exists)
"""

import numpy as np
import pandas as pd
from scipy import stats

# Change this path to your Stage 2 output file
PATH = "/kaggle/working/bankchurners_with_decline_score.csv"
df = pd.read_csv(PATH)

attrited = df.loc[df["Attrition_Flag"] == "Attrited Customer", "Decline_Score"]
existing = df.loc[df["Attrition_Flag"] == "Existing Customer", "Decline_Score"]

print(f"Attrited n={len(attrited)}, mean={attrited.mean():.3f}")
print(f"Existing n={len(existing)}, mean={existing.mean():.3f}")


def cohens_d(a, b):
    """Effect size: gap between means in units of pooled standard deviation."""
    pooled_sd = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled_sd


# ---------------------------------------------------------
# 1. Welch's t-test (one-sided: Attrited > Existing)
# ---------------------------------------------------------
t_stat, p_t = stats.ttest_ind(attrited, existing, equal_var=False, alternative="greater")
print(f"\nWelch t-test:   t = {t_stat:.2f},  p-value = {p_t:.2e}")

# ---------------------------------------------------------
# 2. Mann-Whitney U (non-parametric backup)
# ---------------------------------------------------------
u_stat, p_u = stats.mannwhitneyu(attrited, existing, alternative="greater")
print(f"Mann-Whitney U: U = {u_stat:.0f},  p-value = {p_u:.2e}")

# ---------------------------------------------------------
# 3. Effect size
# ---------------------------------------------------------
d = cohens_d(attrited, existing)
print(f"Cohen's d:      {d:.2f}   (0.2 small, 0.5 medium, 0.8 large)")

# ---------------------------------------------------------
# 4. Same test on each individual signal (which signals separate the groups?)
# ---------------------------------------------------------
def decline_signal(s):
    return 1 - s.rank(pct=True)

signals = pd.DataFrame({
    "ct_chng_signal": decline_signal(df["Total_Ct_Chng_Q4_Q1"]),
    "amt_chng_signal": decline_signal(df["Total_Amt_Chng_Q4_Q1"]),
    "util_signal": decline_signal(df["Avg_Utilization_Ratio"]),
    "trans_ct_signal": decline_signal(df["Total_Trans_Ct"]),
})

rows = []
mask = df["Attrition_Flag"] == "Attrited Customer"
for col in signals.columns:
    a, b = signals.loc[mask, col], signals.loc[~mask, col]
    t, p = stats.ttest_ind(a, b, equal_var=False, alternative="greater")
    rows.append({"signal": col, "mean_attrited": a.mean(), "mean_existing": b.mean(),
                 "t": t, "p_value": p, "cohens_d": cohens_d(a, b)})

print("\nPer-signal results:")
print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---------------------------------------------------------
# Decision rule
# ---------------------------------------------------------
alpha = 0.05
print(f"\nDecision at alpha = {alpha}:",
      "REJECT H0 -> the gap is statistically significant" if p_t < alpha
      else "FAIL TO REJECT H0 -> gap could be chance")
