# -*- coding: utf-8 -*-
"""
Check rapidi: overdispersion Poisson, alternativa NegBin, CV temporale out-of-sample.
"""
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
from patsy import dmatrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_poisson_deviance

DATA = Path("data/processed")
XH = pd.read_parquet(DATA/"model_matrix_hour.parquet")

# Poisson vs NegBin
X = dmatrix("bs(hour, df=8, include_intercept=False) + bs(tmax, df=6, include_intercept=False) + \
             precip + C(dow) + C(zone)", XH, return_type='dataframe')
y = XH["trips"].values
po = sm.GLM(y, X, family=sm.families.Poisson()).fit()
mu = po.mu
phi = ((y - mu)**2 - y)/mu   # test grezzo di overdispersione
print("Overdispersion (mean residual-based) ~", float(np.mean(phi)))

nb = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()
print("AIC Poisson:", po.aic, "| AIC NegBin:", nb.aic)

# CV temporale
tscv = TimeSeriesSplit(n_splits=5)
scores = []
for tr, te in tscv.split(X):
    m = sm.GLM(y[tr], X.iloc[tr], family=sm.families.Poisson()).fit()
    yhat = m.predict(X.iloc[te])
    # devianza di Poisson come metrica
    scores.append(mean_poisson_deviance(y[te], yhat))
print("TimeSeries CV mean deviance:", float(np.mean(scores)))
