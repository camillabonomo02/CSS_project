#!/usr/bin/env python3
"""
Data cleaning pipeline for the Trento bike sharing project
- Weather (ERA5 daily JSON 2020-2022)
- Google Mobility (2022, Autonomous Province of Trento)
- Bike sharing stations (WKT in CRS UTM32N)
- GTFS (stops, routes, trips, stop_times, calendar, shapes, ... - 2025)

Usage:
    python scripts/clean_all.py --raw data/raw --interim data/interim --processed data/processed

Requirements:
    pandas, pyarrow, geopandas, shapely, pyproj

Main outputs:
    data/interim/meteo_daily.parquet
    data/interim/mobility_trento_2022.parquet
    data/interim/stations_clean.geojson
    data/interim/gtfs_stops_2025.geojson
    data/interim/gtfs_service_calendar_2025.parquet
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import re
import pandas as pd

try:
    import geopandas as gpd
    from shapely import wkt
except Exception as e:
    gpd = None
    wkt = None


# Utils

def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def info(msg: str) -> None:
    print(f"[clean] {msg}")


# Weather

def clean_meteo(raw_dir: Path, interim_dir: Path) -> Path:
    src = raw_dir / "trento_era5_daily_2020_2022.json"
    info(f"Weather: reading {src}")  # Weather: reading {src}
    with open(src, "r") as f:
        meteo_json = json.load(f)
    # The valid structure is inside 'daily'
    daily = pd.DataFrame(meteo_json["daily"]).rename(columns={
        "time": "date",
        "temperature_2m_max": "temp_max",
        "precipitation_sum": "precip_mm",
    })
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    # Basic checks
    assert daily["date"].notna().all(), "Weather dates could not be parsed"
    assert daily["date"].min().year == 2020 and daily["date"].max().year == 2022, \
        "The weather file should cover 2020-2022"
    out = interim_dir / "meteo_daily.parquet"
    daily.sort_values("date").to_parquet(out, index=False)
    info(f"Weather: saved {out} with {len(daily)} rows and {daily.shape[1]} columns") 


# Google Mobility

def clean_mobility(raw_dir: Path, interim_dir: Path) -> Path:
    src = raw_dir / "2022_IT_Region_Mobility_Report.csv"
    info(f"Mobility: reading {src}")  # Mobility: reading {src}
    mob = pd.read_csv(src)
    # Filter Autonomous Province of Trento
    mask = (
        (mob["sub_region_1"] == "Trentino-South Tyrol") &
        (mob["sub_region_2"] == "Autonomous Province of Trento")
    )
    trento = mob[mask].copy()
    if trento.empty:
        raise ValueError("Filtering for 'Autonomous Province of Trento' found no rows. Check the source file.")
    trento["date"] = pd.to_datetime(trento["date"], errors="coerce")
    # Rename main columns
    ren = {
        "retail_and_recreation_percent_change_from_baseline": "mob_retail",
        "grocery_and_pharmacy_percent_change_from_baseline": "mob_grocery",
        "parks_percent_change_from_baseline": "mob_parks",
        "transit_stations_percent_change_from_baseline": "mob_transit",
        "workplaces_percent_change_from_baseline": "mob_work",
        "residential_percent_change_from_baseline": "mob_residential",
    }
    trento = trento.rename(columns=ren)
    # Keep minimal set
    keep = ["date", *ren.values(), "place_id", "iso_3166_2_code"]
    trento = trento[keep].sort_values("date")
    # Sanity check: only 2022
    years = trento["date"].dt.year.unique().tolist()
    info(f"Mobility: years present {years}")  
    out = interim_dir / "mobility_trento_2022.parquet"
    trento.to_parquet(out, index=False)
    info(f"Mobility: saved {out} with {len(trento)} rows")  
    return out


# bike sharing stations (WKT UTM32N -> WGS84)

def clean_stations(raw_dir: Path, interim_dir: Path) -> Path:
    src = raw_dir / "stazioni_trento.csv"
    info(f"Stations: reading {src}")
    # The file uses ';' as separator and WKT column like 'POINT (663132.53 5104569.75)'
    df = pd.read_csv(src, sep=';')
    expected = {"WKT", "id", "fumetto", "desc", "cicloposteggi", "tipologia"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"Unexpected columns in {src}. Found: {df.columns.tolist()}")

    if gpd is None:
        raise RuntimeError("geopandas not available: install geopandas to handle geometries in the WKT column")

    gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.GeoSeries.from_wkt(df["WKT"]))
    gdf = gdf.set_crs(32632, allow_override=True).to_crs(4326)
    gdf["lat"] = gdf.geometry.y
    gdf["lon"] = gdf.geometry.x
    # Standardize names
    gdf = gdf.rename(columns={
        "id": "station_id",
        "fumetto": "name",
        "cicloposteggi": "capacity",
        "tipologia": "type"
    })
    # Order columns
    cols = ["station_id", "name", "desc", "capacity", "type", "lat", "lon", "geometry"]
    gdf = gdf[cols]
    # Duplicates (by coordinates)
    gdf = gdf.drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)
    out = interim_dir / "stations_clean.geojson"
    gdf.to_file(out, driver="GeoJSON")
    info(f"Stations: saved {out} with {len(gdf)} points")  
    return out


# GTFS (stops + calendar info)

def clean_gtfs(raw_dir: Path, interim_dir: Path) -> tuple[Path, Path]:
    stops_p = raw_dir / "stops.txt"
    info(f"GTFS: reading stops {stops_p}")
    stops = pd.read_csv(stops_p)
    # Standard GTFS columns
    stops = stops.rename(columns={
        "stop_id": "stop_id",
        "stop_name": "stop_name",
        "stop_lat": "lat",
        "stop_lon": "lon",
        "zone_id": "zone_id",
    })
    # Geometries
    if gpd is None:
        raise RuntimeError("geopandas not available: install geopandas to export stops as GeoJSON")
    gstops = gpd.GeoDataFrame(stops, geometry=gpd.points_from_xy(stops["lon"], stops["lat"], crs=4326))
    out_stops = interim_dir / "gtfs_stops_2025.geojson"
    gstops.to_file(out_stops, driver="GeoJSON")
    info(f"GTFS: saved {out_stops} with {len(gstops)} stops")  

    # Service calendar (useful for hourly/daily frequencies later)
    cal_p = raw_dir / "calendar.txt"
    if cal_p.exists():
        cal = pd.read_csv(cal_p)
        # normalize dates
        for c in ("start_date", "end_date"):
            cal[c] = pd.to_datetime(cal[c].astype(str), format="%Y%m%d", errors="coerce")
        out_cal = interim_dir / "gtfs_service_calendar_2025.parquet"
        cal.to_parquet(out_cal, index=False)
        info(f"GTFS: saved {out_cal} with {len(cal)} rows")  
    else:
        out_cal = None
        info("GTFS: calendar.txt not found — skipping") 

    return out_stops, out_cal

# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--interim", type=Path, default=Path("data/interim"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = ap.parse_args()

    ensure_dirs(args.raw, args.interim, args.processed)

    # Step execution
    meteo_p = clean_meteo(args.raw, args.interim)
    mob_p = clean_mobility(args.raw, args.interim)
    stations_p = clean_stations(args.raw, args.interim)
    gtfs_stops_p, gtfs_cal_p = clean_gtfs(args.raw, args.interim)

    info("Cleaning completed ✔")
    info(f"Interim: {args.interim}")
    for p in [meteo_p, mob_p, stations_p, gtfs_stops_p, gtfs_cal_p]:
        if p is not None:
            info(f" - {p}")


if __name__ == "__main__":
    main()
