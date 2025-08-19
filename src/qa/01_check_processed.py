from pathlib import Path
import pandas as pd

P = Path("data/processed")

gmr   = pd.read_parquet(P/"gmr_day.parquet")                         # date, gmr_*
gtfsH = pd.read_parquet(P/"gtfs_station_hour.parquet")               # station_id, date, hour, dep_hour_urb_300m, dep_hour_ext_300m, dep_hour_total_300m
covD  = pd.read_parquet(P/"station_day_covariates.parquet")          # station_id, date, dep_*_day, gmr_*
stidx = pd.read_csv(P/"station_index.csv")                           # station_id, name, lon, lat, ...

print("== Shapes ==")
for n,df in [("gmr_day", gmr), ("gtfs_station_hour", gtfsH), ("station_day_covariates", covD), ("station_index", stidx)]:
    print(n, df.shape)

# Unicità chiavi
assert gmr["date"].is_unique, "gmr_day: date non univoche"
assert gtfsH.duplicated(["station_id","date","hour"]).sum()==0, "gtfs_hour: chiave non univoca"
assert covD.duplicated(["station_id","date"]).sum()==0, "covariates_day: chiave non univoca"
assert stidx["station_id"].is_unique, "station_index: station_id non univoci"

# Intersezioni/coverage
dates_gtfs = (gtfsH["date"].min(), gtfsH["date"].max())
dates_gmr  = (gmr["date"].min(), gmr["date"].max())
print("GTFS date range:", dates_gtfs)
print("GMR  date range:", dates_gmr)

missing_stats = gtfsH.isna().mean().sort_values(ascending=False).head(10)
print("Missingness top10 (gtfs_hour):\n", missing_stats)

# Stazioni coperte da GTFS vs indice
s_gtfs = set(gtfsH["station_id"].unique())
s_idx  = set(stidx["station_id"].unique())
print("Stations in GTFS hour only:", len(s_gtfs - s_idx))
print("Stations in index only:", len(s_idx - s_gtfs))
