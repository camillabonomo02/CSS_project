import pandas as pd
from pathlib import Path

P = Path("data/processed"); P.mkdir(parents=True, exist_ok=True)

gmr  = pd.read_parquet(P/"gmr_day.parquet")                                # 2022-01-01 .. 2022-10-15
gtfsH = pd.read_parquet(P/"gtfs_station_hour.parquet")                      # station_id, date, hour, dep_hour_*_300m
stidx = pd.read_parquet(P/"station_index.geo.parquet")[["station_id","capacity","zone","lon","lat"]]

# --- limita GTFS alla finestra GMR
d0, d1 = gmr["date"].min(), gmr["date"].max()
gtfsH = gtfsH[(gtfsH["date"]>=d0) & (gtfsH["date"]<=d1)].copy()

# --- aggrega GTFS a giorno per stazione
agg = (gtfsH.groupby(["station_id","date"], as_index=False)
       .agg(dep_urb_day=("dep_hour_urb_300m","sum"),
            dep_ext_day=("dep_hour_ext_300m","sum"),
            dep_tot_day=("dep_hour_total_300m","sum")))

# --- costruisci griglia stazione×giorno nella finestra e unisci
stations = stidx[["station_id"]].drop_duplicates()
dates = pd.DataFrame({"date": pd.date_range(d0, d1, freq="D")})
grid = stations.assign(key=1).merge(dates.assign(key=1), on="key").drop(columns="key")

df = (grid.merge(agg, on=["station_id","date"], how="left")
           .merge(gmr, on="date", how="inner")         # tiene solo giorni con GMR
           .merge(stidx, on="station_id", how="left"))

# --- variabili temporali
df["dow"] = df["date"].dt.dayofweek
df["offset"] = (df["capacity"].fillna(1)).clip(lower=1).pipe(lambda s: s.apply(lambda x: __import__("math").log(x)))

out = P/"station_day_features_2022.parquet"
df.to_parquet(out, index=False)
print("Salvato:", out, "→", df.shape, "| range:", df["date"].min(), "→", df["date"].max())
