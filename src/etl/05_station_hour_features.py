import pandas as pd
from pathlib import Path

P = Path("data/processed"); P.mkdir(parents=True, exist_ok=True)

gtfsH = pd.read_parquet(P/"gtfs_station_hour.parquet")
stidx = pd.read_parquet(P/"station_index.geo.parquet")[["station_id","lon","lat","capacity","zone"]]

# Aggiungi metadati e variabili temporali
df = gtfsH.merge(stidx, on="station_id", how="left")
df["dow"]  = pd.to_datetime(df["date"]).dt.dayofweek
df["hour"] = df["hour"].astype(int)
# sicurezza: rimpiazza NA dep con zero
for c in ["dep_hour_urb_300m","dep_hour_ext_300m","dep_hour_total_300m"]:
    if c in df.columns: df[c] = df[c].fillna(0)

out = P/"station_hour_features.parquet"
df.to_parquet(out, index=False)
print("Salvato:", out, "→", df.shape)
