"""
Stage 8: Revenue-at-Risk
Project: Credit Card Share of Wallet (SoW) Analytics

What this script does:
1. Out-of-fold (5-fold) LightGBM predictions for ALL 10,127 customers.
   Every customer's probability comes from a model that never saw that customer.
   Decline_Score is fitted on each fold's TRAIN part only (no leakage).
2. Platt-scaling calibration (cross-fitted) to fix the ~5% underestimate of total attrition
3. Annual revenue per customer (ASSUMPTIONS below) and
   revenue_at_risk = annual_revenue x calibrated probability
4. Back-test: expected revenue loss vs revenue of customers who really attrited
5. Breakdown by card category / income, top customers, and a CSV for Stages 9-13

BUSINESS ASSUMPTIONS (the dataset has no APR / interchange columns - change freely;
Stage 9 Monte Carlo will vary them as ranges):
  APR          : interest earned on the revolving balance
  INTERCHANGE  : fee earned on card spend (Total_Trans_Amt = last 12 months)
Amounts are presumably USD (the income bands are in $), but the dataset does not state it.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from lightgbm import LGBMClassifier

PATH = "/kaggle/input/datasets/sakshigoyal7/credit-card-customers/BankChurners.csv"
OUT_PATH = "/kaggle/working/stage8_customer_risk.csv"
SEED = 42

APR = 0.20            # assumption: 20% annual interest on revolving balance
INTERCHANGE = 0.015   # assumption: 1.5% of annual card spend

# ---------------------------------------------------------
# 1. Load + clean (drop leaky Naive Bayes columns; keep CLIENTNUM only as an ID)
# ---------------------------------------------------------
df = pd.read_csv(PATH)
df = df.drop(columns=[c for c in df.columns if c.startswith("Naive_Bayes")])
y = (df["Attrition_Flag"] == "Attrited Customer").astype(int)
ids = df["CLIENTNUM"]
X = df.drop(columns=["CLIENTNUM", "Attrition_Flag"])

cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
Xenc = pd.get_dummies(X, columns=cat_cols).astype(float)
print("Customers:", len(df), "| attrited:", int(y.sum()), f"({y.mean():.2%})")

# ---------------------------------------------------------
# 2. Out-of-fold predictions for every customer
# ---------------------------------------------------------
SIGNAL_COLS = ["Total_Ct_Chng_Q4_Q1", "Total_Amt_Chng_Q4_Q1", "Avg_Utilization_Ratio", "Total_Trans_Ct"]


def make_ecdf(values: pd.Series):
    s = np.sort(values.to_numpy())
    n = len(s)
    return lambda v: (np.searchsorted(s, v, side="left") + np.searchsorted(s, v, side="right")) / 2 / n


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
p_raw = np.zeros(len(df))
decline_score = np.zeros(len(df))

for fold, (tr, va) in enumerate(skf.split(Xenc, y), start=1):
    ecdfs = {c: make_ecdf(Xenc.iloc[tr][c]) for c in SIGNAL_COLS}

    def add_score(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        d["Decline_Score"] = np.mean([1 - ecdfs[c](d[c].to_numpy()) for c in SIGNAL_COLS], axis=0)
        return d

    A, B = add_score(Xenc.iloc[tr]), add_score(Xenc.iloc[va])
    model = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                           random_state=SEED, verbose=-1).fit(A, y.iloc[tr])
    p_raw[va] = model.predict_proba(B)[:, 1]
    decline_score[va] = B["Decline_Score"].to_numpy()
    print(f"fold {fold} done")

# ---------------------------------------------------------
# 3. Platt-scaling calibration (cross-fitted, so no in-sample optimism)
# ---------------------------------------------------------
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


p_cal = cross_val_predict(
    LogisticRegression(), logit(p_raw), y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1),
    method="predict_proba",
)[:, 1]


def ece(y_true, p, bins=10):
    b = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    y_true = np.asarray(y_true)
    return sum(abs(y_true[b == i].mean() - p[b == i].mean()) * (b == i).mean() for i in range(bins))


print("\n=== Calibration check (all customers, out-of-fold) ===")
print(f"{'':18s} {'sum of probs':>13s} {'actual':>8s} {'Brier':>8s} {'ECE':>8s} {'AUC':>8s}")
for name, p in [("Raw LightGBM", p_raw), ("Platt calibrated", p_cal)]:
    print(f"{name:18s} {p.sum():13.1f} {int(y.sum()):8d} {brier_score_loss(y, p):8.4f} "
          f"{ece(y, p):8.4f} {roc_auc_score(y, p):8.4f}")

# ---------------------------------------------------------
# 4. Revenue and revenue-at-risk
# ---------------------------------------------------------
out = pd.DataFrame({
    "CLIENTNUM": ids,
    "Card_Category": df["Card_Category"],
    "Income_Category": df["Income_Category"],
    "Credit_Limit": df["Credit_Limit"],
    "Total_Revolving_Bal": df["Total_Revolving_Bal"],
    "Total_Trans_Amt": df["Total_Trans_Amt"],
    "Decline_Score": decline_score,
    "y_true": y,
    "p_raw": p_raw,
    "p_cal": p_cal,
})
out["annual_revenue"] = out["Total_Revolving_Bal"] * APR + out["Total_Trans_Amt"] * INTERCHANGE
out["revenue_at_risk"] = out["annual_revenue"] * out["p_cal"]

total_rev = out["annual_revenue"].sum()
exp_loss = out["revenue_at_risk"].sum()
real_loss = (out["annual_revenue"] * out["y_true"]).sum()

print(f"\n=== Revenue-at-risk (APR {APR:.0%}, interchange {INTERCHANGE:.1%}) ===")
print(f"Total annual revenue (all customers):      {total_rev:>14,.0f}")
print(f"Expected revenue at risk (model):          {exp_loss:>14,.0f}  ({exp_loss / total_rev:.1%} of total)")
print(f"Back-test: revenue of customers who left:  {real_loss:>14,.0f}  ({real_loss / total_rev:.1%} of total)")
print(f"Gap model vs back-test:                    {exp_loss / real_loss - 1:>+14.1%}")


def breakdown(col):
    g = out.groupby(col).agg(customers=("y_true", "size"),
                             annual_revenue=("annual_revenue", "sum"),
                             revenue_at_risk=("revenue_at_risk", "sum"))
    g["risk_pct_of_revenue"] = g["revenue_at_risk"] / g["annual_revenue"] * 100
    return g.sort_values("revenue_at_risk", ascending=False).round(1)


print("\n=== By Card_Category ===")
print(breakdown("Card_Category").to_string())
print("\n=== By Income_Category ===")
print(breakdown("Income_Category").to_string())

print("\n=== Top 10 customers by revenue at risk ===")
cols = ["CLIENTNUM", "Card_Category", "annual_revenue", "p_cal", "revenue_at_risk", "y_true"]
print(out.sort_values("revenue_at_risk", ascending=False).head(10)[cols].round(3).to_string(index=False))

# ---------------------------------------------------------
# 5. Who can we still save? Customers who have NOT left yet (y_true = 0)
#    Most of the biggest revenue-at-risk above are customers who already left, so
#    the retention target list must come from the customers still on the book.
# ---------------------------------------------------------
existing = out[out["y_true"] == 0]
print("\n=== Existing (not yet attrited) customers ===")
print(f"Customers: {len(existing)}")
print(f"Expected revenue at risk: {existing['revenue_at_risk'].sum():,.0f}  "
      f"({existing['revenue_at_risk'].sum() / existing['annual_revenue'].sum():.1%} of their revenue)")
print(f"Customers with calibrated risk > 50%: {(existing['p_cal'] > 0.5).sum()}  "
      f"(revenue at risk {existing.loc[existing['p_cal'] > 0.5, 'revenue_at_risk'].sum():,.0f})")
print("\nTop 10 existing customers to save (by revenue at risk):")
print(existing.sort_values("revenue_at_risk", ascending=False).head(10)[cols].round(3).to_string(index=False))

out.to_csv(OUT_PATH, index=False)
print(f"\nSaved customer-level file to: {OUT_PATH}  ({out.shape[0]} rows, {out.shape[1]} columns)")
