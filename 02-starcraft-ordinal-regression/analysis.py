"""Ordinal logistic regression of StarCraft II league placement (Python re-implementation of an SPSS PLUM analysis)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.miscmodels.ordinal_model import OrderedModel

df = pd.read_csv("data/SkillCraft.csv", na_values="?").dropna()        # league 8 (professionals) has no age or hours, so it drops out
y = df["LeagueIndex"].astype(int)
predictors = ["Age", "HoursPerWeek", "SelectByHotkeys", "AssignToHotkeys", "UniqueHotkeys", "MinimapAttacks",
              "NumberOfPACs", "GapBetweenPACs", "ActionLatency", "WorkersMade", "UniqueUnitsMade"]
model = OrderedModel(y, df[predictors], distr="logit").fit(method="bfgs", maxiter=5000, disp=False)
print(model.summary())

n = len(y); ll_full = model.llf
p0 = y.value_counts(normalize=True); ll_null = float((y.value_counts() * np.log(p0)).sum())   # thresholds only
cox_snell = 1 - np.exp(2 * (ll_null - ll_full) / n)
nagelkerke = cox_snell / (1 - np.exp(2 * ll_null / n))
mcfadden = 1 - ll_full / ll_null
pred = np.asarray(model.predict()).argmax(axis=1) + y.min()
print(f"\nPseudo R-squared: Cox and Snell {cox_snell:.3f}, Nagelkerke {nagelkerke:.3f}, McFadden {mcfadden:.3f}")
print(f"Classification accuracy: {np.mean(pred == y.values)*100:.2f}%")

tab = pd.crosstab(y, pred, rownames=["Actual league"], colnames=["Predicted league"]); print("\n", tab)
plt.figure(figsize=(6, 5)); plt.imshow(tab.values, cmap="Blues"); plt.colorbar(label="Players")
plt.xticks(range(tab.shape[1]), tab.columns); plt.yticks(range(tab.shape[0]), tab.index)
plt.xlabel("Predicted league"); plt.ylabel("Actual league"); plt.title("Confusion matrix"); plt.tight_layout(); plt.savefig("figures/confusion_matrix.png", dpi=130)
