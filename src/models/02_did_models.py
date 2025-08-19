# -*- coding: utf-8 -*-
"""
DID generico su base giornaliera.
Definisci:
  - event_date: data di "trattamento" (es. inizio anno accademico, apertura ZTL estiva, ecc.)
  - treat_rule: quali stazioni sono "trattate" (es. zone=='center' oppure vicinanza ad atenei)
Modello:
  trips ~ treat*post + C(station_id) + C(dow) + week FE + controlli (meteo, GMR)
Salva tabella coefficienti e un plot con media per gruppo pre/post.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

DATA = Path("data/processed")
FIGS = Path("figures"); FIGS.mkdir(exist_ok=True, parents=True)

X = pd.read_parquet(DATA/"model_matrix_day.parquet")

# ---- parametrizza qui
EVENT_DATE = pd.Timestamp("2022-06-15")   # <- MODIFICA a tuo caso
def treat_rule(df):
    return (df["zone"].str.contains("cent", case=False, na=False)).astype(int)

def prep(df):
    df = df.copy()
    df["post"]   = (df["date"] >= EVENT_DATE).astype(int)
    df["treat"]  = treat_rule(df)
    df["did"]    = df["treat"]*df["post"]
    return df

def fit_did(df):
    df = prep(df)
    # finestra simmetrica (opzionale)
    pre  = df.loc[df["date"].between(EVENT_DATE - pd.Timedelta(days=60), EVENT_DATE - pd.Timedelta(days=1))]
    post = df.loc[df["date"].between(EVENT_DATE, EVENT_DATE + pd.Timedelta(days=60))]
    dat  = pd.concat([pre, post], ignore_index=True)
    # fattori fissi stazione e giorno-settimana, controlli meteo e GMR
    formula = "trips ~ did + treat + post + tmax + precip + gmr_work + gmr_transit + C(dow) + C(station_id)"
    mod = smf.ols(formula, data=dat).fit(cov_type="cluster", cov_kwds={"groups": dat["station_id"]})
    return mod, dat

def plot_group_means(dat):
    g = (dat.groupby(["treat","post"], as_index=False)
            ["trips"].mean().rename(columns={"trips":"mean_trips"}))
    labels = {0:"Controllo",1:"Trattamento"}
    fig, ax = plt.subplots()
    for t in [0,1]:
        sub = g[g["treat"]==t].sort_values("post")
        ax.plot(["Pre","Post"], sub["mean_trips"].values, marker="o", label=labels[t])
    ax.set_title("DID – medie di gruppo pre/post")
    ax.set_ylabel("Viaggi medi/giorno"); ax.legend(); fig.tight_layout()
    fig.savefig(FIGS/"did_group_means.png", dpi=120)

if __name__ == "__main__":
    mod, dat = fit_did(X)
    print(mod.summary().as_text())
    (DATA/"did_summary.txt").write_text(mod.summary().as_text())
    plot_group_means(dat)
    print("Figura DID salvata.")
