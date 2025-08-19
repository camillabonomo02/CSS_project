# -*- coding: utf-8 -*-
"""
Costruisce le matrici di modello a livello ORARIO e GIORNALIERO
mergiando outcome (trips) con covariate: GMR, meteo, GTFS, attributi stazione.
Salva:
  data/processed/model_matrix_hour.parquet
  data/processed/model_matrix_day.parquet
"""
from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path("data/processed")
OUT_H = DATA/"model_matrix_hour.parquet"
OUT_D = DATA/"model_matrix_day.parquet"

# ---- File attesi (rinomina qui se i nomi sono diversi)
OUTCOME_H = DATA/"station_hour_outcomes.parquet"   # colonne minime: station_id,date,hour,trips
OUTCOME_D = DATA/"station_day_outcomes.parquet"    # colonne minime: station_id,date,trips
STATION   = DATA/"station_index.csv"               # colonne consigliate: station_id,name,lat,lon,zone (opz.)
GMR       = DATA/"gmr_day.parquet"                 # date,gmr_transit,gmr_work,gmr_retail,gmr_parks
WEATHER   = DATA/"weather_2022.parquet"            # date,tmax,precip (o nomi simili)
GTFS_H    = DATA/"gtfs_station_hour.parquet"       # station_id,date,hour,gtfs_dep_300m (o simile)

def _safe_read(fn):
    if Path(fn).exists():
        return pd.read_parquet(fn) if fn.endswith(".parquet") else pd.read_csv(fn)
    raise FileNotFoundError(f"Manca {fn}")

def _prep_dates(df, col="date"):
    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    return df

def _infer_cols(df, wanted, rename=None):
    """mappa colonne se i nomi differiscono leggermente"""
    df = df.copy()
    if rename:
        df = df.rename(columns=rename)
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        raise ValueError(f"Mancano colonne {missing} in dataframe con colonne {list(df.columns)[:10]}...")
    return df[wanted]

def main():
    # ---- load
    y_h = _safe_read(str(OUTCOME_H))
    y_d = _safe_read(str(OUTCOME_D))
    st  = _safe_read(str(STATION))
    gmr = _safe_read(str(GMR))
    met = _safe_read(str(WEATHER))
    try:
        gtfs_h = _safe_read(str(GTFS_H))
    except FileNotFoundError:
        gtfs_h = pd.DataFrame(columns=["station_id","date","hour","gtfs_dep_300m"])

    # ---- standardizza chiavi temporali
    y_h = _prep_dates(y_h, "date")
    y_d = _prep_dates(y_d, "date")
    gmr  = _prep_dates(gmr, "date")
    met  = _prep_dates(met, "date")
    if not gtfs_h.empty:
        gtfs_h = _prep_dates(gtfs_h, "date")

    # ---- colonne minime & rinomina flessibile
    # Google Mobility
    gmr = gmr.rename(columns={
        "workplaces_percent_change_from_baseline":"gmr_work",
        "retail_and_recreation_percent_change_from_baseline":"gmr_retail",
        "transit_stations_percent_change_from_baseline":"gmr_transit",
        "parks_percent_change_from_baseline":"gmr_parks"
    })
    gmr = gmr[["date","gmr_transit","gmr_work","gmr_retail","gmr_parks"]]

    # Meteo
    met = met.rename(columns={
        "tmax":"tmax", "temp_max":"tmax", "temperature_2m_max":"tmax",
        "precip":"precip", "precipitation_sum":"precip"
    })
    met = met[["date","tmax","precip"]]

    # Statiche stazioni
    if "zone" not in st.columns:
        st["zone"] = "unknown"
    st = st[["station_id","zone"]].copy()

    # GTFS hourly (opzionale)
    if not gtfs_h.empty:
        gtfs_h = gtfs_h.rename(columns={
            "gtfs_dep_300m":"gtfs_dep_300m",
            "departures_300m":"gtfs_dep_300m"
        })
        gtfs_h = gtfs_h[["station_id","date","hour","gtfs_dep_300m"]]

    # ------------ ORARIO
    Xh = y_h.merge(st, on="station_id", how="left")\
            .merge(gmr, on="date", how="left")\
            .merge(met, on="date", how="left")
    if not gtfs_h.empty:
        Xh = Xh.merge(gtfs_h, on=["station_id","date","hour"], how="left")
    # features temporali
    Xh["dow"]   = Xh["date"].dt.dayofweek        # 0 lun - 6 dom
    Xh["is_we"] = (Xh["dow"]>=5).astype(int)
    Xh["month"] = Xh["date"].dt.month
    # sostituisci eventuali NaN in GTFS con 0
    if "gtfs_dep_300m" in Xh.columns:
        Xh["gtfs_dep_300m"] = Xh["gtfs_dep_300m"].fillna(0)

    # ------------ GIORNALIERO
    if not gtfs_h.empty:
        gtfs_d = (gtfs_h
                  .groupby(["station_id","date"], as_index=False)["gtfs_dep_300m"]
                  .sum().rename(columns={"gtfs_dep_300m":"gtfs_dep_300m_day"}))
    else:
        gtfs_d = pd.DataFrame(columns=["station_id","date","gtfs_dep_300m_day"])
    Xd = y_d.merge(st, on="station_id", how="left")\
            .merge(gmr, on="date", how="left")\
            .merge(met, on="date", how="left")\
            .merge(gtfs_d, on=["station_id","date"], how="left")
    Xd["dow"]   = Xd["date"].dt.dayofweek
    Xd["is_we"] = (Xd["dow"]>=5).astype(int)
    Xd["week"]  = Xd["date"].dt.isocalendar().week.astype(int)
    Xd["month"] = Xd["date"].dt.month
    if "gtfs_dep_300m_day" in Xd.columns:
        Xd["gtfs_dep_300m_day"] = Xd["gtfs_dep_300m_day"].fillna(0)

    # ---- sanity check minimi
    for df, name, cols in [
        (Xh, "hour", ["station_id","date","hour","trips"]),
        (Xd, "day",  ["station_id","date","trips"])
    ]:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"{name}: mancano colonne {missing}")

    # ---- save
    Xh.to_parquet(OUT_H, index=False)
    Xd.to_parquet(OUT_D, index=False)
    print("Saved:", OUT_H, Xh.shape, "|", OUT_D, Xd.shape)

if __name__ == "__main__":
    main()
