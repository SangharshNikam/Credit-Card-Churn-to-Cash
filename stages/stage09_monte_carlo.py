"""
Stage 9: Monte Carlo Simulation of Revenue-at-Risk
Project: Credit Card Share of Wallet (SoW) Analytics

Problem: Stage 8 gave ONE number, but it rests on assumptions (APR, interchange)
and on probabilities that are slightly uncertain. Instead of one number we run
10,000 "what-if" worlds. In each world we randomly pick:
  - APR            ~ Triangular(18%, 21%, 24%)   (Fed G.19 card APRs are ~21-22%)
  - Interchange    ~ Triangular(1.0%, 1.5%, 2.5%) (published rates ~1.5-2.4%)
  - Probability scale ~ Uniform(0.95, 1.05)       (residual calibration error, +/-5%)
and recompute revenue-at-risk. The spread of the 10,000 results is the answer.

Two scopes:
  - All customers (includes people who already left)
  - Existing customers only (the ones we can still save)  <- the actionable number
For existing customers we also simulate WHO actually leaves (coin flip with
probability p_cal), because even a perfect model cannot know individual outcomes.

Input : stage8_customer_risk.csv (from Stage 8, added as a Kaggle input)
Output: stage9_simulation_results.csv + stage9_monte_carlo.png
"""

import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 42
N_SIM = 10_000
rng = np.random.default_rng(SEED)

# Finds the Stage 8 file automatically, wherever Kaggle mounted it
found = glob.glob("/kaggle/input/**/stage8_customer_risk.csv", recursive=True)
assert found, "stage8_customer_risk.csv not found - add the Stage 8 notebook output as an input"
PATH = found[0]
print("Using:", PATH)

df = pd.read_csv(PATH)
bal = df["Total_Revolving_Bal"].to_numpy()
trans = df["Total_Trans_Amt"].to_numpy()
p = df["p_cal"].to_numpy()
is_existing = (df["y_true"] == 0).to_numpy()
print(f"Customers: {len(df)} | existing (not yet attrited): {is_existing.sum()}")


def exposure(apr, ic, scale=1.0, mask=None):
    """Expected revenue at risk = sum(annual revenue x attrition probability)."""
    rev = bal * apr + trans * ic
    pp = np.minimum(p * scale, 1.0)
    if mask is not None:
        return (rev[mask] * pp[mask]).sum()
    return (rev * pp).sum()


# Base case must reproduce Stage 8
base_all = exposure(0.20, 0.015)
base_ex = exposure(0.20, 0.015, mask=is_existing)
print(f"\nBase case (Stage 8: APR 20%, interchange 1.5%): all = {base_all:,.0f} "
      f"(Stage 8 file says {df['revenue_at_risk'].sum():,.0f}) | existing = {base_ex:,.0f}")

# ---------------------------------------------------------
# Simulation
# ---------------------------------------------------------
apr_s = rng.triangular(0.18, 0.21, 0.24, N_SIM)
ic_s = rng.triangular(0.010, 0.015, 0.025, N_SIM)
scale_s = rng.uniform(0.95, 1.05, N_SIM)

bal_e, trans_e, p_e = bal[is_existing], trans[is_existing], p[is_existing]
all_loss = np.empty(N_SIM)
ex_loss = np.empty(N_SIM)
ex_real = np.empty(N_SIM)

for i in range(N_SIM):
    all_loss[i] = exposure(apr_s[i], ic_s[i], scale_s[i])
    rev_e = bal_e * apr_s[i] + trans_e * ic_s[i]
    pp_e = np.minimum(p_e * scale_s[i], 1.0)
    ex_loss[i] = (rev_e * pp_e).sum()
    leaves = rng.random(len(p_e)) < pp_e            # who actually leaves in this world
    ex_real[i] = (rev_e * leaves).sum()


def summary(name, x):
    return {"scope": name, "mean": x.mean(), "median": np.median(x),
            "p2.5": np.percentile(x, 2.5), "p97.5": np.percentile(x, 97.5)}


res = pd.DataFrame([
    summary("All customers (expected exposure)", all_loss),
    summary("Existing customers (expected)", ex_loss),
    summary("Existing customers (incl. who really leaves)", ex_real),
])
print(f"\n=== Monte Carlo results ({N_SIM:,} simulations, 95% interval) ===")
print(res.round(0).to_string(index=False, float_format=lambda v: f"{v:,.0f}"))

# ---------------------------------------------------------
# Which assumption matters most? (one at a time, others at their middle value)
# ---------------------------------------------------------
mid = dict(apr=0.21, ic=0.015, scale=1.0)
ranges = {"APR (18% - 24%)": ("apr", 0.18, 0.24),
          "Interchange (1.0% - 2.5%)": ("ic", 0.010, 0.025),
          "Probability scale (0.95 - 1.05)": ("scale", 0.95, 1.05)}
mid_val = exposure(mid["apr"], mid["ic"], mid["scale"], mask=is_existing)
rows = []
for label, (key, lo, hi) in ranges.items():
    vals = []
    for v in (lo, hi):
        args = dict(mid)
        args[key] = v
        vals.append(exposure(args["apr"], args["ic"], args["scale"], mask=is_existing))
    rows.append({"assumption": label, "low_end": vals[0], "high_end": vals[1], "swing": abs(vals[1] - vals[0])})
tornado = pd.DataFrame(rows).sort_values("swing", ascending=False)
print(f"\n=== Sensitivity: existing customers, middle case = {mid_val:,.0f} ===")
print(tornado.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))

# ---------------------------------------------------------
# Charts + save the simulated draws
# ---------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].hist(all_loss, bins=50)
ax[0].axvline(base_all, color="red", linestyle="--", label=f"Stage 8 base {base_all:,.0f}")
ax[0].set_title("All customers: revenue at risk")
ax[0].legend(fontsize=8)
ax[1].hist(ex_loss, bins=50, alpha=0.7, label="Expected")
ax[1].hist(ex_real, bins=50, alpha=0.5, label="Incl. who really leaves")
ax[1].axvline(base_ex, color="red", linestyle="--", label=f"Stage 8 base {base_ex:,.0f}")
ax[1].set_title("Existing customers (savable)")
ax[1].legend(fontsize=8)
t = tornado.iloc[::-1]
ax[2].barh(t["assumption"], t["swing"])
ax[2].set_title("What moves the answer most (existing)")
ax[2].set_xlabel("Swing in revenue at risk")
plt.tight_layout()
plt.savefig("/kaggle/working/stage9_monte_carlo.png", dpi=150)
plt.show()

pd.DataFrame({"apr": apr_s, "interchange": ic_s, "prob_scale": scale_s,
              "all_customers_loss": all_loss, "existing_expected_loss": ex_loss,
              "existing_realized_loss": ex_real}).to_csv("/kaggle/working/stage9_simulation_results.csv", index=False)
print("\nSaved: stage9_simulation_results.csv and stage9_monte_carlo.png")
