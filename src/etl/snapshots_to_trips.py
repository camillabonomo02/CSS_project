from pathlib import Path
import pandas as pd

IN = Path("data/raw/bikesharing_trento/status")
OUT = Path("data/processed"); OUT.mkdir(parents=True, exist_ok=True)
TZ = "Europe/Rome"

files = sorted(IN.glob("status_*.ndjson"))
if not files:
    raise SystemExit("Nessuna snapshot NDJSON trovata.")

df = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%dT%H%M%S").dt.tz_localize(
    TZ, nonexistent="shift_forward", ambiguous="NaT"
)
df = df.sort_values(["station_id","timestamp"])
for c in ["bikes","docks","lat","lon"]:
    if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

df["bikes_lag"] = df.groupby("station_id")["bikes"].shift()
df["delta"] = df["bikes"] - df["bikes_lag"]

# Filtra rebalancing (soglia euristica; regola dopo ispezione)
mask_rebal = df["delta"].abs() > 6
df.loc[mask_rebal, "delta"] = 0

df["pickup"]  = (-df["delta"]).clip(lower=0)
df["return"]  = ( df["delta"]).clip(lower=0)

df["date"] = df["timestamp"].dt.normalize()
df["hour"] = df["timestamp"].dt.hour

H = df.groupby(["station_id","date","hour"], as_index=False).agg(trips_hour=("pickup","sum"))
D = H.groupby(["station_id","date"], as_index=False).agg(trips_day=("trips_hour","sum"))

H.to_parquet(OUT/"station_hour.parquet", index=False)
D.to_parquet(OUT/"station_day.parquet", index=False)
print("Outcome creato:", H.shape, D.shape)
