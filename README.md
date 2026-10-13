# Financial Data Analysis: four statistical modelling projects

Four team projects from the Financial Data Analysis course, Actuarial Science ITS (2024). The original analyses were done in Minitab and SPSS; this repository re-implements each one in Python so that it can be read and rerun. Every folder has `analysis.py`, its data, the printed output in `results.txt` and the figures.

| Folder | Question | Method | Report (Minitab or SPSS) | This Python version |
|---|---|---|---|---|
| `01-diamond-price-regression` | What drives a diamond's price? | Multiple linear regression on the 4Cs with VIF screening, normality, Glejser and Durbin–Watson tests | R² 92.72% | R² 90.57% |
| `02-starcraft-ordinal-regression` | What separates players across StarCraft II leagues? | Ordinal logistic regression (logit link) | Nagelkerke 0.560, Cox–Snell 0.542, McFadden 0.226, accuracy 40.41% | Nagelkerke 0.559, Cox–Snell 0.542, McFadden 0.226, accuracy 40.26% |
| `03-delhi-temperature-arima` | Can ARIMA forecast Delhi's daily temperature? | ARIMA(p, 1, q) on a 90/10 training and testing split, Ljung–Box and Kolmogorov–Smirnov checks | ARIMA(0,1,3), MAPE 28.4% | ARIMA(0,1,3), MAPE 26.5%; the only candidate with white-noise residuals |
| `04-electricity-clustering` | How do provinces group by electricity use, 2013–2022? | Hierarchical clustering and K-Means, silhouette and pseudo-F | K-Means, 3 clusters, pseudo-F 202.77 | K-Means, 3 clusters, pseudo-F 202.77 |

## Why some numbers differ
- **Diamonds:** Minitab's recoding grouped some colour and clarity grades into separate indicator variables, and those groupings cannot be fully recovered from the report, so this version uses one ordinal code per characteristic.
- **ARIMA:** Minitab and statsmodels estimate ARIMA parameters and likelihoods differently, so forecasts and AIC values differ even for the same model.
- **Clustering:** the pseudo-F values match exactly; the silhouette scores in this version differ from the ones in the report.

## Run it
```bash
pip install -r requirements.txt
cd 02-starcraft-ordinal-regression && python analysis.py
```

## Data
Diamonds, SkillCraft (UCI) and Daily Delhi Climate are public datasets from Kaggle; electricity distributed by province (GWh) is from Statistics Indonesia (BPS).

Credit: analyses originally done with my Financial Data Analysis project team at ITS. Portfolio: https://naufalabdul11.github.io
