# -*- coding: utf-8 -*-
"""
Stima GAM-GLM (tramite spline in statsmodels) per outcome di conteggio.
- Modello orario (Poisson/NegBin) con splines su ora e temperatura.
- Modello giornaliero con splines su tmax e precipitazione.
Salva coefficienti e partial effects in data/processed/ e figure in figures/.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrix
import matplotlib.pyplot as plt

DATA = Path("data/processed")
FIGS = Path("figures"); FIGS.mkdir(parents=True, exist_ok=True)

XH = pd.read_parquet(DATA/"model_matrix_hour.parquet")
XD = pd.read_parquet(DATA/"model_matrix_day.parquet")

def fit_glm_spline_hour(df):
    df = df.dropna(subset=["trips"]).copy()
    # spline basi
    df["s_hour"] = dmatrix("bs(hour, df=8, include_intercept=False)", df, return_type='dataframe')
    df["s_tmax"] = dmatrix("bs(tmax, df=6, include_intercept=False)", df, return_type='dataframe')
    # costruisci formula con colonne espanse
    X_hour = dmatrix("bs(hour, df=8, include_intercept=False) + bs(tmax, df=6, include_intercept=False) + \
                      precip + C(dow) + C(zone)", df, return_type='dataframe')
    y = df["trips"].values
    model = sm.GLM(y, X_hour, family=sm.families.Poisson())
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df["station_id"]})
    return res, X_hour.design_info.column_names

def fit_glm_spline_day(df):
    df = df.dropna(subset=["trips"]).copy()
    X_day = dmatrix("bs(tmax, df=6, include_intercept=False) + \
                     precip + gmr_work + gmr_transit + C(dow) + C(zone)",
                    df, return_type='dataframe')
    y = df["trips"].values
    model = sm.GLM(y, X_day, family=sm.families.Poisson())
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df["station_id"]})
    return res, X_day.design_info.column_names

def partial_plot_hour(df, res):
    grid = pd.DataFrame({"hour": np.arange(5, 24),
                         "tmax": df["tmax"].median(),
                         "precip": 0,
                         "dow": 2,
                         "zone": df["zone"].mode().iat[0]})
    Xg = dmatrix("bs(hour, df=8, include_intercept=False) + bs(tmax, df=6, include_intercept=False) + \
                  precip + C(dow) + C(zone)", grid, return_type='dataframe')
    pred = res.get_prediction(Xg).summary_frame()
    plt.figure()
    plt.plot(grid["hour"], np.exp(pred["mean"]), label="f(hour)")
    plt.fill_between(grid["hour"], np.exp(pred["mean_ci_lower"]), np.exp(pred["mean_ci_upper"]), alpha=0.2)
    plt.xlabel("Hour"); plt.ylabel("E[trips]"); plt.title("Partial effect of hour")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIGS/"gam_hour_partial.png", dpi=120)

def partial_plot_tmax_day(df, res):
    xs = np.linspace(df["tmax"].quantile(0.02), df["tmax"].quantile(0.98), 50)
    grid = pd.DataFrame({"tmax": xs, "precip": 0, "dow": 2,
                         "zone": df["zone"].mode().iat[0],
                         "gmr_work": 0, "gmr_transit": 0})
    Xg = dmatrix("bs(tmax, df=6, include_intercept=False) + \
                  precip + gmr_work + gmr_transit + C(dow) + C(zone)", grid, return_type='dataframe')
    pred = res.get_prediction(Xg).summary_frame()
    plt.figure()
    plt.plot(xs, np.exp(pred["mean"]), label="f(tmax)")
    plt.fill_between(xs, np.exp(pred["mean_ci_lower"]), np.exp(pred["mean_ci_upper"]), alpha=0.2)
    plt.xlabel("Tmax (°C)"); plt.ylabel("E[trips]"); plt.title("Partial effect of temperature (daily)")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIGS/"gam_day_tmax_partial.png", dpi=120)

if __name__ == "__main__":
    res_h, cols_h = fit_glm_spline_hour(XH)
    res_d, cols_d = fit_glm_spline_day(XD)
    print(res_h.summary())
    print(res_d.summary())
    # salva riassunti
    (DATA/"gam_hour_summary.txt").write_text(res_h.summary().as_text())
    (DATA/"gam_day_summary.txt").write_text(res_d.summary().as_text())
    partial_plot_hour(XH, res_h)
    partial_plot_tmax_day(XD, res_d)
    print("Figure salvate in:", FIGS)
