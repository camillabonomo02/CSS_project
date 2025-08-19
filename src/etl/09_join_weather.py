import pandas as pd
from pathlib import Path
P = Path("data/processed")
df = pd.read_parquet(P/"station_day_features_2022.parquet")
wx = pd.read_parquet(P/"weather_2022.parquet")
df = df.merge(wx, on="date", how="left")
df.to_parquet(P/"station_day_features_2022.parquet", index=False)
print("Aggiornato con meteo:", P/"station_day_features_2022.parquet", df.shape)
