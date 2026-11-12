"""
Stage 5: ANOVA across Card Categories
Project: Credit Card Share of Wallet (SoW) Analytics

Question: Do the 4 card categories (Blue, Silver, Gold, Platinum) differ in
(a) how much customers spend and (b) their Decline_Score?

H0: all card categories have the same mean
H1: at least one card category has a different mean

Tests used:
1. Levene test          -> are the group variances similar? (ANOVA assumption)
2. One-way ANOVA (F-test) -> do the group means differ?
3. Kruskal-Wallis       -> backup, no normal-distribution assumption
4. Eta-squared          -> effect size (how much variance the groups explain)
5. Tukey HSD            -> WHICH pairs of categories differ (post-hoc)
"""

import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Change this path to your Stage 2 output file
PATH = "/kaggle/working/bankchurners_with_decline_score.csv"
df = pd.read_csv(PATH)

order = ["Blue", "Silver", "Gold", "Platinum"]
print("Customers per card category:")
print(df["Card_Category"].value_counts().reindex(order))


def eta_squared(groups):
    """Share of total variance explained by the group (0 to 1)."""
    all_vals = pd.concat(groups)
    grand_mean = all_vals.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((all_vals - grand_mean) ** 2).sum()
    return ss_between / ss_total


for metric in ["Total_Trans_Amt", "Decline_Score"]:
    print(f"\n{'=' * 60}\nMetric: {metric}\n{'=' * 60}")
    groups = [df.loc[df["Card_Category"] == c, metric] for c in order]

    print("\nMean by card category:")
    print(df.groupby("Card_Category")[metric].mean().reindex(order).round(3).to_string())

    lev_stat, lev_p = stats.levene(*groups)
    f_stat, f_p = stats.f_oneway(*groups)
    h_stat, h_p = stats.kruskal(*groups)
    print(f"\nLevene (equal variance):  p = {lev_p:.4g}")
    print(f"One-way ANOVA:            F = {f_stat:.2f}, p = {f_p:.4g}")
    print(f"Kruskal-Wallis (backup):  H = {h_stat:.2f}, p = {h_p:.4g}")
    print(f"Eta-squared (effect size): {eta_squared(groups):.4f}   (0.01 small, 0.06 medium, 0.14 large)")

    print("\nTukey HSD (which pairs differ):")
    tukey = pairwise_tukeyhsd(df[metric], df["Card_Category"], alpha=0.05)
    print(tukey.summary())
