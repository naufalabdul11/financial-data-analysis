"""Diamond price regression on the 4C characteristics (Python re-implementation of a Minitab analysis).

Steps follow the report: recode the categorical 4Cs as numbers, check multicollinearity (VIF),
fit a multiple linear regression, and test the residual assumptions.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy import stats

df = pd.read_csv("data/diamonds.csv", index_col=0)
df["cut_code"] = df["cut"].map({"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4})
df["color_code"] = df["color"].map({c: i for i, c in enumerate("JIHGFED")})                # J worst ... D best
df["clarity_code"] = df["clarity"].map({c: i for i, c in enumerate(["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])})

def vif_table(cols):
    X = sm.add_constant(df[cols])
    return pd.Series([variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])], index=cols).round(2)

all_x = ["carat", "depth", "table", "x", "y", "z", "cut_code", "color_code", "clarity_code"]
print("VIF with every predictor:\n", vif_table(all_x), "\n")
# x and y are almost copies of carat and z (VIF far above 10), so they are dropped, as in the report
final_x = ["carat", "depth", "table", "z", "cut_code", "color_code", "clarity_code"]
print("VIF after dropping x and y:\n", vif_table(final_x), "\n")

model = sm.OLS(df["price"], sm.add_constant(df[final_x])).fit()
print(model.summary())
print(f"\nS = {np.sqrt(model.scale):.3f}, R-sq = {model.rsquared*100:.2f}%, R-sq(adj) = {model.rsquared_adj*100:.2f}%")

resid = model.resid
print(f"Kolmogorov-Smirnov on standardised residuals: p = {stats.kstest((resid - resid.mean()) / resid.std(), 'norm').pvalue:.4g}")
glejser = sm.OLS(np.abs(resid), sm.add_constant(df[final_x])).fit()
print(f"Glejser test (heteroskedasticity): F = {glejser.fvalue:.1f}, p = {glejser.f_pvalue:.4g}")
print(f"Durbin-Watson: {durbin_watson(resid):.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(model.fittedvalues, resid, s=2, alpha=.3); ax[0].axhline(0, color="k", lw=.8); ax[0].set(title="Residuals vs fitted", xlabel="Fitted price", ylabel="Residual")
sm.qqplot(resid, line="s", ax=ax[1], markersize=2); ax[1].set_title("Normal Q-Q plot of residuals")
plt.tight_layout(); plt.savefig("figures/residuals.png", dpi=130)
