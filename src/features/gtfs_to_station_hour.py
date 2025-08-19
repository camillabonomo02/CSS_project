import pandas as pd, geopandas as gpd
from pathlib import Path
from shapely.geometry import Point

OUT = Path("data/processed"); OUT.mkdir(parents=True, exist_ok=True)
STATIONS = gpd.read_parquet(OUT/"station_index.geo.parquet").to_crs(32632)

GTFS_DIRS = [
    ("urb",  Path("data/external/gtfs/google_transit_urbano_tte")),
    ("ext",  Path("data/external/gtfs/google_transit_extraurbano_tte")),  
]

BUFFER_M = 300

def load_gtfs_folder(tag, d):
    """Carica tabelle base GTFS da una cartella."""
    req = ["stops.txt","routes.txt","trips.txt","stop_times.txt","calendar.txt","calendar_dates.txt"]
    for f in req:
        if not (d/f).exists():
            raise FileNotFoundError(f"Manca {f} in {d}")
    stops = pd.read_csv(d/"stops.txt")
    routes= pd.read_csv(d/"routes.txt")
    trips = pd.read_csv(d/"trips.txt")
    stt   = pd.read_csv(d/"stop_times.txt")
    cal   = pd.read_csv(d/"calendar.txt")
    cald  = pd.read_csv(d/"calendar_dates.txt")
    # keep minimal cols
    stops = stops[["stop_id","stop_lat","stop_lon"]]
    routes= routes[["route_id","route_type","route_short_name","agency_id"]] if "route_type" in routes.columns else routes
    trips = trips[["trip_id","route_id","service_id"]]
    stt   = stt[["trip_id","stop_id","departure_time"]]
    return {"tag":tag,"stops":stops,"routes":routes,"trips":trips,"stt":stt,"cal":cal,"cald":cald}

gtfs = [load_gtfs_folder(tag, d) for tag,d in GTFS_DIRS]

# mappa stazioni ↔ fermate entro 300m (una volta per tutte, per ciascun dataset)
def stations_near_stops(stations_gdf, stops_df):
    gs_stops = gpd.GeoDataFrame(
        stops_df, geometry=gpd.points_from_xy(stops_df.stop_lon, stops_df.stop_lat), crs="EPSG:4326"
    ).to_crs(32632)
    buf = stations_gdf[["station_id","geometry"]].copy()
    buf["geometry"] = buf.buffer(BUFFER_M)
    near = gpd.sjoin(buf, gs_stops, predicate="contains")[["station_id","stop_id"]]
    return near

near_maps = {g["tag"]: stations_near_stops(STATIONS, g["stops"]) for g in gtfs}

# utilità calendario GTFS
def active_services_on(cal, cald, day: pd.Timestamp) -> set:
    weekday = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"][day.weekday()]
    base = cal[
        (cal[weekday]==1) &
        (pd.to_datetime(cal["start_date"], format="%Y%m%d")<=day) &
        (pd.to_datetime(cal["end_date"],   format="%Y%m%d")>=day)
    ]["service_id"]
    add = cald[(cald["date"]==int(day.strftime("%Y%m%d"))) & (cald["exception_type"]==1)]["service_id"]
    rem = cald[(cald["date"]==int(day.strftime("%Y%m%d"))) & (cald["exception_type"]==2)]["service_id"]
    s = set(base).union(set(add))
    return s.difference(set(rem))

# intervallo di date: usa l'intersezione dei calendari (min start, max end)
def gtfs_date_range(gtfs_parts):
    starts, ends = [], []
    for g in gtfs_parts:
        starts.append(pd.to_datetime(g["cal"]["start_date"], format="%Y%m%d").min())
        ends.append(pd.to_datetime(g["cal"]["end_date"],   format="%Y%m%d").max())
    return max(starts), min(ends)

d0, d1 = gtfs_date_range(gtfs)
dates = pd.date_range(d0, d1, freq="D")

frames = []
for day in dates:
    for g in gtfs:
        tag = g["tag"]
        active = active_services_on(g["cal"], g["cald"], day)
        trips_today = g["trips"][g["trips"]["service_id"].isin(active)][["trip_id","route_id"]]
        stt_today = g["stt"].merge(trips_today, on="trip_id", how="inner")[["stop_id","route_id","departure_time"]]
        if stt_today.empty:
            continue
        stt_today = stt_today[stt_today["departure_time"].notna()]
        stt_today = stt_today[stt_today["departure_time"].str.match(r"^\d{1,2}:\d{2}(:\d{2})?$")]
        h = stt_today["departure_time"].str.split(":", expand=True).astype(int)
        stt_today["hour"] = (h[0] % 48)  # supporta orari 24–47
        dep = (stt_today.groupby(["stop_id","hour"]).size()
               .rename("dep_hour").reset_index())
        dep = dep.merge(near_maps[tag], on="stop_id", how="inner")
        agg = (dep.groupby(["station_id","hour"], as_index=False)["dep_hour"].sum()
                 .rename(columns={"dep_hour":f"dep_hour_{tag}_300m"}))
        agg["date"] = day.normalize()
        frames.append(agg)

feat = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["station_id","hour","date","dep_hour_urb_300m","dep_hour_ext_300m"])

feat = (
    feat.groupby(["station_id", "date", "hour"], as_index=False)
        .agg({
            "dep_hour_urb_300m": "sum",
            "dep_hour_ext_300m": "sum",
        })
)

feat["dep_hour_total_300m"] = feat[["dep_hour_urb_300m", "dep_hour_ext_300m"]].sum(axis=1)

feat.to_parquet(OUT/"gtfs_station_hour.parquet", index=False)
print("Salvato:", OUT/"gtfs_station_hour.parquet", "→", feat.shape)
