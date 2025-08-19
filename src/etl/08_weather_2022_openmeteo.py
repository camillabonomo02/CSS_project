import pandas as pd, json
from urllib.request import urlopen
from pathlib import Path

OUT = Path("data/processed"); OUT.mkdir(parents=True, exist_ok=True)
url = ("https://archive-api.open-meteo.com/v1/era5?"
       "latitude=46.07&longitude=11.12&start_date=2022-01-01&end_date=2022-10-15&"
       "daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Europe%2FRome")
data = json.load(urlopen(url))
d = pd.DataFrame({
    "date": pd.to_datetime(data["daily"]["time"]),
    "tmax": data["daily"]["temperature_2m_max"],
    "tmin": data["daily"]["temperature_2m_min"],
    "prcp": data["daily"]["precipitation_sum"]
})
d.to_parquet(OUT/"weather_2022.parquet", index=False)
print("Salvato weather:", OUT/"weather_2022.parquet", d.shape)
