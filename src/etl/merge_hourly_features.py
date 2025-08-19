import pandas as pd
from pathlib import Path
P=Path("data/processed")

H = pd.read_parquet(P/"station_hour.parquet")                      # outcome
Xg = pd.read_parquet(P/"gtfs_station_hour.parquet")                # offerta TPL vicina
S = pd.read_csv(P/"station_index.csv")                             # anagrafica stazioni (lat/lon, zona)
out = (H.merge(Xg, on=["station_id","hour"], how="left")
         .merge(S, on="station_id", how="left"))
out.to_parquet(P/"model_hour_dataset.parquet", index=False)
print(out.shape, out.columns.tolist()[:12], "…")