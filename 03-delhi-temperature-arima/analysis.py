"""ARIMA forecasting of Delhi's daily mean temperature (Python re-implementation of a Minitab analysis)."""
import numpy as np, pandas as pd, matplotlib, warnings
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
warnings.filterwarnings("ignore")

train = pd.read_csv("data/DailyDelhiClimateTrain.csv", parse_dates=["date"])
test = pd.read_csv("data/DailyDelhiClimateTest.csv", parse_dates=["date"])
data = pd.concat([train, test.iloc[1:]]).set_index("date")["meantemp"]          # 1 Jan 2017 appears in both files
cut = int(round(len(data) * 0.9)); fit_part, hold_out = data.iloc[:cut], data.iloc[cut:]   # 90% training, 10% testing
print(f"Training {fit_part.index[0].date()} to {fit_part.index[-1].date()} ({len(fit_part)} days), testing {len(hold_out)} days")
print(f"ADF test p-value: level {adfuller(fit_part)[1]:.3f}, first difference {adfuller(fit_part.diff().dropna())[1]:.3g}")

rows = []
for order in [(0, 1, 1), (0, 1, 2), (0, 1, 3), (1, 1, 1)]:
    res = ARIMA(fit_part.values, order=order).fit(); fc = res.forecast(len(hold_out))
    mape = np.mean(np.abs((hold_out.values - fc) / hold_out.values)) * 100
    lb = acorr_ljungbox(res.resid[1:], lags=[12], return_df=True)["lb_pvalue"].iloc[0]
    ks = stats.kstest((res.resid[1:] - res.resid[1:].mean()) / res.resid[1:].std(), "norm").pvalue
    rows.append({"model": f"ARIMA{order}", "AIC": round(res.aic, 2), "MAPE %": round(mape, 2), "Ljung-Box p": round(lb, 3), "KS p": round(ks, 3)})
table = pd.DataFrame(rows); print(table.to_string(index=False))

best = ARIMA(fit_part.values, order=(0, 1, 3)).fit(); fc = best.get_forecast(len(hold_out)); ci = fc.conf_int(alpha=.05)
plt.figure(figsize=(11, 4)); plt.plot(data.index[-400:], data.values[-400:], label="Observed")
plt.plot(hold_out.index, fc.predicted_mean, label="ARIMA(0,1,3) forecast"); plt.fill_between(hold_out.index, ci[:, 0], ci[:, 1], alpha=.2)
plt.ylabel("Mean temperature (°C)"); plt.legend(); plt.title("Delhi daily mean temperature"); plt.tight_layout(); plt.savefig("figures/forecast.png", dpi=130)
