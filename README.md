# Churn-to-Cash: Credit Card Revenue-at-Risk & Retention Targeting

Most churn projects stop at "who will leave". This one goes further: **how much revenue is at stake, which customers are worth an offer, and what to offer them.**

Built Sept-Oct 2026 on the public Kaggle [Credit Card customers (BankChurners)](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) dataset (10,127 customers, 16.07% attrited). Python, scikit-learn, LightGBM, SHAP.

## The question

A card issuer loses money long before a customer closes the card: spend and activity fade first. The project answers five connected questions:

1. Can we spot disengagement **without using the churn label**? (decline score)
2. How well can a model rank who leaves, and can its probabilities be trusted for money? (models, calibration)
3. How much revenue is at risk, and how much of it can still be saved? (revenue-at-risk, Monte Carlo)
4. Which customers are worth a retention offer once the offer's cost is counted? (cost-sensitive targeting)
5. Why is a customer risky, and what should the bank try? (segmentation, SHAP, action playbook)

## Headline results

| Question | Result |
|---|---|
| Label-free decline score | AUC 0.79 on a held-out test set with no training; Welch's t-test: 0.670 (attrited) vs 0.467 (existing), Cohen's d = 1.32 |
| Best model | LightGBM: ROC-AUC 0.993, Gini 0.986, KS 0.916 (Logistic Regression 0.916, Random Forest 0.981) |
| Ranking power | The riskiest 20% of customers contain 96.3% of attrition (top-decile lift 6.2x) |
| Calibration | Raw model under-counted attrition by 3.3% (1,572.7 vs 1,627). After Platt scaling: 1,626.8. ECE 0.0069 to 0.0019, AUC unchanged |
| Revenue at risk | 289,403 expected (9.6% of 3.02M annual revenue); back-test against customers who really left: 294,471 (gap -1.7%) |
| How much is still savable | Only **46,692** sits with customers who have not left yet; **84% of the exposure is already lost** |
| Uncertainty (Monte Carlo, 10,000 runs) | Savable revenue at risk: 95% range 43,439 to 55,918. APR is the biggest source of uncertainty, not the model |
| Targeting | Contacting everyone loses 198,492. A default 0.5 cutoff nets +2,284. **Value-aware targeting of 125 customers nets +4,381 (about 2.4x return on offer cost)** |
| Segments | "Grow" (4,169 customers) holds 62% of revenue at 1% average risk; "Protect" is 125 customers; 370 high-value customers have already left (win-back pool, 182K past annual revenue) |
| Drivers | For 81% of the 125 Protect customers the top SHAP driver is transaction count; demographics almost never appear |
| Actions | 98% of Protect customers fall into two themes: spend activation (103) and high-spender retention (19) |

All money figures are **under stated assumptions** (see below).

## Pipeline (13 stages, one script each in `stages/`)

| # | Script | What it does |
|---|---|---|
| 1 | `stage01_data_loading_eda.py` | Load, drop leaky columns, EDA |
| 2 | `stage02_decline_score.py` | Label-free decline score from 4 behaviour signals (percentile ranks) |
| 3 | `stage03_hypothesis_testing.py` | Welch's t-test, Mann-Whitney, Cohen's d |
| 4 | `stage04_chi_square.py` | Chi-square + Cramer's V + Bonferroni on categorical features |
| 5 | `stage05_anova.py` | ANOVA, Kruskal-Wallis, Tukey across card categories |
| 6 | `stage06_model_training.py` | Stratified split; Logistic Regression vs Random Forest vs LightGBM |
| 7 | `stage07_gini_lift_calibration.py` | Gini, gain/lift, calibration check |
| 8 | `stage08_revenue_at_risk.py` | Out-of-fold predictions, Platt calibration, revenue-at-risk, back-test |
| 9 | `stage09_monte_carlo.py` | 10,000 simulations over APR, interchange and probability error |
| 10 | `stage10_cost_sensitive_threshold.py` | Net-benefit-based targeting, sensitivity to offer cost and success rate |
| 11 | `stage11_segmentation.py` | Value x risk segments (Protect, Grow, Monitor, Maintain, Win-back, Closed) |
| 12 | `stage12_shap.py` | SHAP drivers for the Protect customers |
| 13 | `stage13_action_mapping.py` | Driver to action playbook with hypotheses and KPIs |

## Key design choices

- **No leakage:** the two `Naive_Bayes_*` columns that ship with the dataset are dropped; the decline score is fitted on training data only; revenue uses out-of-fold predictions so no customer is scored by a model that saw them.
- **No class weights:** revenue is probability x money, so probabilities are kept honest and imbalance is handled with a cost-sensitive threshold.
- **Effect sizes next to p-values** (Cohen's d, Cramer's V, eta-squared), because with 10,000+ rows p-values are always tiny.
- **Already-lost vs savable are separated.** Customers who left are reported as lost revenue, never as "at risk".

## Assumptions (not in the dataset)

| Assumption | Value | Basis |
|---|---|---|
| APR on revolving balance | 20% (simulated 18-24%) | US Federal Reserve G.19 card rates, about 21-22% |
| Interchange on spend | 1.5% (simulated 1.0-2.5%) | Published US credit interchange, about 1.5-2.4% |
| Offer cost per customer | 25 | Assumption |
| Offer success rate | 30% | Assumption; must be measured with an A/B test before any rollout |

Revenue = revolving balance x APR + 12-month spend x interchange. It is **gross revenue, not profit** (no cost of funds, rewards or credit losses). Currency is presumably USD (income bands are in $) but the dataset does not say.

## Limitations

- **Static snapshot with no dates.** Time-based validation and any "early warning" claim are not possible. The very high AUC (0.99) is typical for this dataset: many attrited customers have already gone quiet, so the model is better described as identifying disengaged customers than predicting future churn.
- **Association, not causation.** SHAP drivers and the action playbook are hypotheses to test with a control group, not proven levers.
- **Money figures depend on assumptions** above. Only the probabilities are validated against data.
- **Limited model comparison:** three model families were compared (linear, bagging, boosting). XGBoost, CatBoost and hyperparameter tuning were not run.
- **Not deployed.** This is an analysis, not a scoring service.

## How to run

The stages were run in order as Kaggle notebooks. Paths such as `/kaggle/input/...` and `/kaggle/working/...` are Kaggle-specific; edit them for a local run. Each script says which earlier output it needs. See `data/README.md` for the dataset.

```
pip install -r requirements.txt
```
