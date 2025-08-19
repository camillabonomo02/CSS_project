import pandas as pd
from pathlib import Path
P=Path("data/processed")

D   = pd.read_parquet(P/"station_day.parquet")                     # outcome
GMR = pd.read_parquet(P/"gmr_day.parquet")                         # google mobility (PAT, daily)
W   = pd.read_parquet(P/"weather_2022.parquet")                    # meteo giornaliero (ERA5)
C   = pd.read_parquet(P/"station_day_covariates.parquet")          # dummies: weekday/holidays ecc.
S   = pd.read_csv(P/"station_index.csv")

# Make both date columns timezone-naive
D["date"] = D["date"].dt.tz_localize(None)
GMR["date"] = GMR["date"].dt.tz_localize(None)


df = (D.merge(GMR, on="date", how="left")
        .merge(W, on="date", how="left")
        .merge(C, on=["station_id", "date"], how="left")
        .merge(S, on="station_id", how="left"))

df.to_parquet(P/"model_day_dataset.parquet", index=False)
print(df.shape, df.date.min(), "→", df.date.max())