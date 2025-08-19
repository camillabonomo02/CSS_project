from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

P = Path("data/processed"); P.mkdir(parents=True, exist_ok=True)

# A) Profili orari GTFS per quartili di offerta
H = pd.read_parquet(P/"station_hour_features.parquet")
H["date"] = pd.to_datetime(H["date"])
# somma per stazione-giorno e calcola quartile
S = (H.groupby(["station_id","date"], as_index=False)["dep_hour_total_300m"].sum()
       .rename(columns={"dep_hour_total_300m":"dep_day"}))
q = S["dep_day"].quantile([0.25,0.5,0.75]).to_list()
def qbin(x):
    if x<=q[0]: return "Q1"
    if x<=q[1]: return "Q2"
    if x<=q[2]: return "Q3"
    return "Q4"
S["quartile"] = S["dep_day"].map(qbin)
H = H.merge(S[["station_id","date","quartile"]], on=["station_id","date"], how="left")
prof = H.groupby(["hour","quartile"])["dep_hour_total_300m"].mean().reset_index()

plt.figure()
for qb in ["Q1","Q2","Q3","Q4"]:
    sub = prof[prof["quartile"]==qb]
    plt.plot(sub["hour"], sub["dep_hour_total_300m"], label=qb)
plt.xlabel("Hour")
plt.ylabel("Mean GTFS departures within 300m")
plt.title("Hourly GTFS supply by station-day quartile")
plt.legend()
plt.tight_layout()
plt.savefig("figures/gtfs_hour_profiles.png", dpi=150)

# B) Serie temporale GMR (transit vs workplaces)
D = pd.read_parquet(P/"gmr_day.parquet")
D = D.sort_values("date")
plt.figure()
plt.plot(D["date"], D["gmr_transit"], label="Transit")
plt.plot(D["date"], D["gmr_work"], label="Workplaces")
plt.axhline(0, linestyle="--")
plt.xlabel("Date")
plt.ylabel("% change from baseline")
plt.title("Google Mobility (PAT)")
plt.legend()
plt.tight_layout()
plt.savefig("figures/gmr_timeseries.png", dpi=150)

# C) Mappa stazioni dimensionate per offerta media GTFS
G = gpd.read_parquet(P/"station_index.geo.parquet")
avg = H.groupby("station_id")["dep_hour_total_300m"].mean().reset_index(name="dep_hour_avg")
G = G.merge(avg, on="station_id", how="left")
base = G.to_crs(3857)  # per plotting veloce
ax = base.plot(markersize=5 + (base["dep_hour_avg"].fillna(0) ** 0.5) * 2, alpha=0.8)
ax.set_axis_off()
plt.title("Stations sized by mean hourly GTFS departures within 300m")
plt.tight_layout()
plt.savefig("figures/stations_gtfs_map.png", dpi=150)
print("Saved figures to ./figures/")

