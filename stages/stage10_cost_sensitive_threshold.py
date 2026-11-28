"""
Stage 10: Cost-Sensitive Threshold
Project: Credit Card Share of Wallet (SoW) Analytics

Question: which customers should get a retention offer?
A default 0.5 cutoff ignores money. A retention offer COSTS something, and it only
SAVES revenue if the customer would really leave AND the offer works. So we pick the
cutoff that maximises net benefit, not accuracy.

Only EXISTING customers (y_true = 0) are considered: people who already left cannot
be contacted any more.

For one customer i:
    expected gain if contacted = SUCCESS_RATE x p_i x annual_revenue_i  -  OFFER_COST
Net benefit of a campaign = sum of that over everyone contacted.

BUSINESS ASSUMPTIONS (not in the dataset - change them at the top; the sensitivity
table below shows how the answer moves when they change):
    OFFER_COST   : cost of one retention offer (cashback / fee waiver / call)
    SUCCESS_RATE : chance that the offer keeps a customer who would otherwise leave
    A saved customer is assumed to keep paying for one more year of revenue.

Input : stage8_customer_risk.csv (Stage 8 output added as a Kaggle input)
Output: stage10_retention_list.csv + stage10_threshold.png
"""

import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OFFER_COST = 25.0      # assumption: $25 per customer contacted
SUCCESS_RATE = 0.30    # assumption: 30% of would-be leavers are kept by the offer

found = glob.glob("/kaggle/input/**/stage8_customer_risk.csv", recursive=True)
assert found, "stage8_customer_risk.csv not found - add the Stage 8 notebook output as an input"
df = pd.read_csv(found[0])
ex = df[df["y_true"] == 0].reset_index(drop=True)
p = ex["p_cal"].to_numpy()
rev = ex["annual_revenue"].to_numpy()
print(f"Existing customers: {len(ex)} | assumptions: offer cost {OFFER_COST:.0f}, success rate {SUCCESS_RATE:.0%}")


def campaign(mask, cost=OFFER_COST, s=SUCCESS_RATE):
    n = int(mask.sum())
    gross = (s * p[mask] * rev[mask]).sum()
    return n, gross, cost * n, gross - cost * n


# ---------------------------------------------------------
# 1. Net benefit vs flat probability threshold
# ---------------------------------------------------------
grid = np.round(np.arange(0.0, 1.0001, 0.005), 3)


def best_threshold(cost, s):
    nets = np.array([campaign(p >= t, cost, s)[3] for t in grid])
    return grid[nets.argmax()], nets.max(), nets


t_best, net_best, nets = best_threshold(OFFER_COST, SUCCESS_RATE)

strategies = {
    "Contact nobody": np.zeros(len(p), dtype=bool),
    "Contact everyone": np.ones(len(p), dtype=bool),
    "Flat threshold 0.50 (default)": p >= 0.50,
    f"Best flat threshold ({t_best:.3f})": p >= t_best,
    "Value-aware (gain > 0 per customer)": SUCCESS_RATE * p * rev > OFFER_COST,
}
rows = []
for name, m in strategies.items():
    n, gross, cost, net = campaign(m)
    rows.append({"strategy": name, "contacted": n, "expected_saved_revenue": gross,
                 "offer_cost": cost, "net_benefit": net})
table = pd.DataFrame(rows)
print("\n=== Strategy comparison (existing customers) ===")
print(table.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))

# ---------------------------------------------------------
# 2. Sensitivity: how does the best threshold move with the assumptions?
# ---------------------------------------------------------
costs, rates = [10, 25, 50, 100], [0.10, 0.20, 0.30, 0.50]
sens = []
for c in costs:
    for s in rates:
        t, net, _ = best_threshold(c, s)
        n = campaign(p >= t, c, s)[0]
        sens.append({"offer_cost": c, "success_rate": s, "best_threshold": t,
                     "contacted": n, "net_benefit": net})
sens = pd.DataFrame(sens)
print("\n=== Sensitivity: best flat threshold for each assumption pair ===")
print(sens.pivot(index="offer_cost", columns="success_rate", values="best_threshold").round(3).to_string())
print("\nNet benefit at that threshold:")
print(sens.pivot(index="offer_cost", columns="success_rate", values="net_benefit").round(0).to_string())

# ---------------------------------------------------------
# 3. Chart + retention list
# ---------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(grid, nets)
ax[0].axvline(0.5, color="gray", linestyle="--", label="Default 0.50")
ax[0].axvline(t_best, color="red", linestyle="--", label=f"Best {t_best:.3f}")
ax[0].set_title("Net benefit vs probability threshold")
ax[0].set_xlabel("Contact customers with p_cal >= threshold")
ax[0].set_ylabel("Net benefit")
ax[0].legend()
ax[1].plot(grid, [int((p >= t).sum()) for t in grid])
ax[1].set_title("Customers contacted vs threshold")
ax[1].set_xlabel("Threshold")
ax[1].set_ylabel("Customers contacted")
plt.tight_layout()
plt.savefig("/kaggle/working/stage10_threshold.png", dpi=150)
plt.show()

ex["expected_gain"] = SUCCESS_RATE * ex["p_cal"] * ex["annual_revenue"] - OFFER_COST
lst = ex[ex["expected_gain"] > 0].sort_values("expected_gain", ascending=False)
lst[["CLIENTNUM", "Card_Category", "Income_Category", "p_cal", "annual_revenue", "expected_gain"]].to_csv(
    "/kaggle/working/stage10_retention_list.csv", index=False)
print(f"\nRetention list (value-aware rule): {len(lst)} customers, "
      f"total expected net benefit {lst['expected_gain'].sum():,.0f}")
print("Top 5:")
print(lst[["CLIENTNUM", "Card_Category", "p_cal", "annual_revenue", "expected_gain"]].head(5).round(3).to_string(index=False))
print("\nSaved: stage10_retention_list.csv and stage10_threshold.png")
