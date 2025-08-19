import pandas as pd
from pathlib import Path

P = Path("data/processed"); P.mkdir(parents=True, exist_ok=True)
cov = pd.read_parquet(P/"station_day_covariates.parquet")
stidx = pd.read_parquet(P/"station_index.geo.parquet")[["station_id","capacity","zone","lon","lat"]]

# Variabili temporali
cov["dow"] = pd.to_datetime(cov["date"]).dt.dayofweek

# Coalesce dep_* se mancano alcune colonne
for c in ["dep_urb_day","dep_ext_day","dep_tot_day"]:
    if c not in cov.columns:
        cov[c] = 0

cov = cov.merge(stidx, on="station_id", how="left")
out = P/"station_day_features.parquet"
cov.to_parquet(out, index=False)
print("Salvato:", out, "→", cov.shape)
