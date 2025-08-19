import pandas as pd
from pathlib import Path

OUT = Path("data/processed")
gmr = pd.read_parquet(OUT/"gmr_day.parquet")
gtfsH = pd.read_parquet(OUT/"gtfs_station_hour.parquet")

# aggrega GTFS a giorno: somma partenze sulle 24/48 ore
gtfsD = (gtfsH.groupby(["station_id","date"], as_index=False)
         .agg(dep_urb_day=("dep_hour_urb_300m","sum"),
              dep_ext_day=("dep_hour_ext_300m","sum"),
              dep_tot_day=("dep_hour_total_300m","sum")))

# prodotto cartesiano stazioni × date coperte da GTFS
stations = pd.read_parquet(OUT/"station_index.geo.parquet")[["station_id"]]
dates = gtfsD["date"].drop_duplicates().to_frame()
grid = dates.assign(key=1).merge(stations.assign(key=1), on="key").drop(columns="key")
cov = grid.merge(gtfsD, on=["station_id","date"], how="left").merge(gmr, on="date", how="left")
cov.to_parquet(OUT/"station_day_covariates.parquet", index=False)
print("Salvato:", OUT/"station_day_covariates.parquet", "→", cov.shape)
