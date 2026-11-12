"""
Stage 6: Model Training + Comparison
Project: Credit Card Share of Wallet (SoW) Analytics

What this script does:
1. Stratified 80/20 train/test split (keeps ~16% attrition in both parts)
2. Builds Decline_Score from TRAIN data only (percentile ranks are fitted on
   train, then applied to test -> no information leaks from test into train)
3. Compares the heuristic score (baseline) with 3 models:
   Logistic Regression, Random Forest, LightGBM
4. Runs every model with AND without Decline_Score, to check if our
   custom score adds value on top of the raw features
5. Reports ROC-AUC, PR-AUC, KS, Gini, Brier on the held-out TEST set
   + 5-fold CV ROC-AUC on train for stability
6. Saves test-set probabilities for Stages 7-10
7. Plots ROC + Precision-Recall curves to compare all models visually

Note: no class weights are used on purpose. Revenue-at-risk (Stage 8) multiplies
by the predicted probability, so we keep probabilities honest and handle the
class imbalance later with a cost-sensitive threshold (Stage 10).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

# Change this path to your Stage 2 output file (any file with the cleaned columns works)
PATH = "/kaggle/working/bankchurners_with_decline_score.csv"
OUT_PATH = "/kaggle/working/stage6_test_predictions.csv"
SEED = 42

df = pd.read_csv(PATH)
# Works with the raw Kaggle file too: drop the ID + the 2 Naive Bayes columns
# (they are leaky, pre-computed attrition predictions)
df = df.drop(columns=[c for c in df.columns if c == "CLIENTNUM" or c.startswith("Naive_Bayes")])
y = (df["Attrition_Flag"] == "Attrited Customer").astype(int)
X = df.drop(columns=["Attrition_Flag", "Decline_Score"], errors="ignore")  # recompute score below

# ---------------------------------------------------------
# 1. Stratified split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
print(f"Train: {X_train.shape}, attrition rate {y_train.mean():.4f}")
print(f"Test:  {X_test.shape}, attrition rate {y_test.mean():.4f}")

# ---------------------------------------------------------
# 2. Decline_Score fitted on TRAIN only
# ---------------------------------------------------------
SIGNAL_COLS = ["Total_Ct_Chng_Q4_Q1", "Total_Amt_Chng_Q4_Q1", "Avg_Utilization_Ratio", "Total_Trans_Ct"]


def make_ecdf(train_values: pd.Series):
    """Percentile rank of any value relative to the TRAIN distribution."""
    s = np.sort(train_values.to_numpy())
    n = len(s)
    return lambda v: (np.searchsorted(s, v, side="left") + np.searchsorted(s, v, side="right")) / 2 / n


ecdfs = {c: make_ecdf(X_train[c]) for c in SIGNAL_COLS}


def add_decline_scores(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    sig = {c: 1 - ecdfs[c](data[c].to_numpy()) for c in SIGNAL_COLS}   # lower raw value = more decline
    data["Decline_Score"] = np.mean([sig[c] for c in SIGNAL_COLS], axis=0)
    data["Decline_Score_3sig"] = np.mean([sig[c] for c in SIGNAL_COLS if c != "Total_Trans_Ct"], axis=0)
    return data


X_train, X_test = add_decline_scores(X_train), add_decline_scores(X_test)

# ---------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------
def ks_stat(y_true, proba):
    return stats.ks_2samp(proba[y_true == 1], proba[y_true == 0]).statistic


def evaluate(name, y_true, proba):
    auc = roc_auc_score(y_true, proba)
    return {
        "model": name,
        "roc_auc": auc,
        "gini": 2 * auc - 1,
        "pr_auc": average_precision_score(y_true, proba),
        "ks": ks_stat(y_true.to_numpy(), np.asarray(proba)),
    }


results = []

# Baselines: the heuristic score alone (no model, no training)
results.append(evaluate("Heuristic: Decline_Score (4 signals)", y_test, X_test["Decline_Score"]))
results.append(evaluate("Heuristic: Decline_Score (3 signals, no Trans_Ct)", y_test, X_test["Decline_Score_3sig"]))

# ---------------------------------------------------------
# 4. Models, with and without Decline_Score
# ---------------------------------------------------------
# text columns (works on both older and newer pandas string types)
cat_cols = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]
score_cols = ["Decline_Score", "Decline_Score_3sig"]


def prepare(data: pd.DataFrame, with_score: bool) -> pd.DataFrame:
    d = data.drop(columns=score_cols if not with_score else ["Decline_Score_3sig"])
    return pd.get_dummies(d, columns=cat_cols, drop_first=False).astype(float)


models = {
    "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)),
    "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
    "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                               random_state=SEED, verbose=-1),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
preds = {}
cv_auc = {}
brier = {}

for with_score in [False, True]:
    tag = "with Decline_Score" if with_score else "without Decline_Score"
    Xtr, Xte = prepare(X_train, with_score), prepare(X_test, with_score)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)
    for name, model in models.items():
        label = f"{name} ({tag})"
        cv = cross_val_score(model, Xtr, y_train, cv=skf, scoring="roc_auc")
        model.fit(Xtr, y_train)
        proba = model.predict_proba(Xte)[:, 1]
        preds[label] = proba
        cv_auc[label] = (cv.mean(), cv.std())
        brier[label] = brier_score_loss(y_test, proba)
        results.append(evaluate(label, y_test, pd.Series(proba, index=y_test.index)))

res = pd.DataFrame(results)
res["brier"] = res["model"].map(brier)
res["cv_auc_mean"] = res["model"].map(lambda m: cv_auc.get(m, (np.nan, np.nan))[0])
res["cv_auc_std"] = res["model"].map(lambda m: cv_auc.get(m, (np.nan, np.nan))[1])
res = res.sort_values("roc_auc", ascending=False).reset_index(drop=True)

pd.set_option("display.width", 200)
print("\n=== TEST-SET RESULTS (sorted by ROC-AUC) ===")
print(res.round(4).to_string(index=False))

# ---------------------------------------------------------
# 5. Save test predictions for later stages
# ---------------------------------------------------------
out = X_test[["Credit_Limit", "Avg_Utilization_Ratio", "Total_Trans_Amt", "Card_Category", "Decline_Score"]].copy()
out["y_true"] = y_test
for label, p in preds.items():
    if "with Decline_Score" in label:
        out["proba_" + label.split(" ")[0]] = p
out.to_csv(OUT_PATH, index=True)
print(f"\nSaved test predictions to: {OUT_PATH}")

# ---------------------------------------------------------
# 6. Visual comparison: ROC + Precision-Recall curves (test set)
# ---------------------------------------------------------
curves = {
    "Logistic Regression (linear baseline)": preds["LogisticRegression (with Decline_Score)"],
    "Random Forest": preds["RandomForest (with Decline_Score)"],
    "LightGBM": preds["LightGBM (with Decline_Score)"],
    "Heuristic Decline_Score (no training)": X_test["Decline_Score"].to_numpy(),
}

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for name, s in curves.items():
    fpr, tpr, _ = roc_curve(y_test, s)
    prec, rec, _ = precision_recall_curve(y_test, s)
    ax[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, s):.3f})")
    ax[1].plot(rec, prec, label=f"{name} (AP {average_precision_score(y_test, s):.3f})")

ax[0].plot([0, 1], [0, 1], "--", color="gray", label="Random")
ax[0].set_title("ROC curves (test set)")
ax[0].set_xlabel("False positive rate")
ax[0].set_ylabel("True positive rate")
ax[0].legend(loc="lower right", fontsize=8)

ax[1].axhline(y_test.mean(), linestyle="--", color="gray", label=f"Random ({y_test.mean():.2f})")
ax[1].set_title("Precision-Recall curves (test set)")
ax[1].set_xlabel("Recall")
ax[1].set_ylabel("Precision")
ax[1].legend(loc="lower left", fontsize=8)

plt.tight_layout()
plt.savefig("/kaggle/working/stage6_model_comparison.png", dpi=150)
plt.show()
