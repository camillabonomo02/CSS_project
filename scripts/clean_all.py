"""
Pipeline di pulizia per il progetto bike sharing Trento
- Meteo (ERA5 daily JSON 2020-2022)
- Google Mobility (2022, Provincia Autonoma di Trento)
- Stazioni bike sharing (WKT in CRS UTM32N)
- GTFS (stops, routes, trips, stop_times, calendar, shapes, ... - 2025)
- Confini amministrativi (onData, CSV)

Uso:
    python scripts/clean_all.py --raw data/raw --interim data/interim --processed data/processed

Requisiti:
    pandas, pyarrow, geopandas, shapely, pyproj

Output (principali):
    data/interim/meteo_daily.parquet
    data/interim/mobility_trento_2022.parquet
    data/interim/stations_clean.geojson
    data/interim/gtfs_stops_2025.geojson
    data/interim/gtfs_service_calendar_2025.parquet
    data/interim/confini_trento.parquet
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import re
import pandas as pd

# --- Optional: geopandas only when needed ---
try:
    import geopandas as gpd
    from shapely import wkt
except Exception as e:  # pragma: no cover
    gpd = None
    wkt = None


# --------------------
# Utils
# --------------------

def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def info(msg: str) -> None:
    print(f"[clean] {msg}")


# --------------------
# Meteo
# --------------------

def clean_meteo(raw_dir: Path, interim_dir: Path) -> Path:
    src = raw_dir / "trento_era5_daily_2020_2022.json"
    info(f"Meteo: leggo {src}")
    with open(src, "r") as f:
        meteo_json = json.load(f)
    # La struttura valida è dentro 'daily'
    daily = pd.DataFrame(meteo_json["daily"]).rename(columns={
        "time": "date",
        "temperature_2m_max": "temp_max",
        # se presenti altre variabili, verranno mantenute automaticamente
        "precipitation_sum": "precip_mm",
    })
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    # Controlli basilari
    assert daily["date"].notna().all(), "Date meteo non parseabili"
    assert daily["date"].min().year == 2020 and daily["date"].max().year == 2022, \
        "Il file meteo dovrebbe coprire 2020–2022"
    out = interim_dir / "meteo_daily.parquet"
    daily.sort_values("date").to_parquet(out, index=False)
    info(f"Meteo: salvato {out} con {len(daily)} righe e {daily.shape[1]} colonne")
    return out


# --------------------
# Google Mobility
# --------------------

def clean_mobility(raw_dir: Path, interim_dir: Path) -> Path:
    src = raw_dir / "2022_IT_Region_Mobility_Report.csv"
    info(f"Mobility: leggo {src}")
    mob = pd.read_csv(src)
    # Filtro Provincia autonoma di Trento
    mask = (
        (mob["sub_region_1"] == "Trentino-South Tyrol") &
        (mob["sub_region_2"] == "Autonomous Province of Trento")
    )
    trento = mob[mask].copy()
    if trento.empty:
        raise ValueError("Filtrando per 'Autonomous Province of Trento' non trovo righe. Controllare il file sorgente.")
    trento["date"] = pd.to_datetime(trento["date"], errors="coerce")
    # Rinomina colonne principali
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
    # Sanity check: solo 2022
    years = trento["date"].dt.year.unique().tolist()
    info(f"Mobility: anni presenti {years}")
    out = interim_dir / "mobility_trento_2022.parquet"
    trento.to_parquet(out, index=False)
    info(f"Mobility: salvato {out} con {len(trento)} righe")
    return out


# --------------------
# Stazioni bike sharing (WKT UTM32N -> WGS84)
# --------------------

def clean_stations(raw_dir: Path, interim_dir: Path) -> Path:
    src = raw_dir / "stazioni_trento.csv"
    info(f"Stazioni: leggo {src}")
    # Il file ha separatore ';' e colonna WKT tipo 'POINT (663132.53 5104569.75)'
    df = pd.read_csv(src, sep=';')
    expected = {"WKT", "id", "fumetto", "desc", "cicloposteggi", "tipologia"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"Colonne inattese in {src}. Trovate: {df.columns.tolist()}")

    if gpd is None:
        raise RuntimeError("geopandas non disponibile: installa geopandas per gestire geometrie della colonna WKT")

    gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.GeoSeries.from_wkt(df["WKT"]))
    # I valori sembrano UTM 32N per Trento: EPSG:32632
    gdf = gdf.set_crs(32632, allow_override=True).to_crs(4326)
    gdf["lat"] = gdf.geometry.y
    gdf["lon"] = gdf.geometry.x
    # Standardizza nomi
    gdf = gdf.rename(columns={
        "id": "station_id",
        "fumetto": "name",
        "cicloposteggi": "capacity",
        "tipologia": "type"
    })
    # Ordina colonne
    cols = ["station_id", "name", "desc", "capacity", "type", "lat", "lon", "geometry"]
    gdf = gdf[cols]
    # Duplicati (per coordinate)
    gdf = gdf.drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)
    out = interim_dir / "stations_clean.geojson"
    gdf.to_file(out, driver="GeoJSON")
    info(f"Stazioni: salvato {out} con {len(gdf)} punti")
    return out


# --------------------
# GTFS (stops + info calendario servizio)
# --------------------

def clean_gtfs(raw_dir: Path, interim_dir: Path) -> tuple[Path, Path]:
    stops_p = raw_dir / "stops.txt"
    info(f"GTFS: leggo fermate {stops_p}")
    stops = pd.read_csv(stops_p)
    # Colonne standard GTFS
    stops = stops.rename(columns={
        "stop_id": "stop_id",
        "stop_name": "stop_name",
        "stop_lat": "lat",
        "stop_lon": "lon",
        "zone_id": "zone_id",
    })
    # Geometrie
    if gpd is None:
        raise RuntimeError("geopandas non disponibile: installa geopandas per esportare le fermate come GeoJSON")
    gstops = gpd.GeoDataFrame(stops, geometry=gpd.points_from_xy(stops["lon"], stops["lat"], crs=4326))
    out_stops = interim_dir / "gtfs_stops_2025.geojson"
    gstops.to_file(out_stops, driver="GeoJSON")
    info(f"GTFS: salvato {out_stops} con {len(gstops)} fermate")

    # Calendario servizio (utile per frequenze orarie/giornaliere più avanti)
    cal_p = raw_dir / "calendar.txt"
    if cal_p.exists():
        cal = pd.read_csv(cal_p)
        # normalizza date
        for c in ("start_date", "end_date"):
            cal[c] = pd.to_datetime(cal[c].astype(str), format="%Y%m%d", errors="coerce")
        out_cal = interim_dir / "gtfs_service_calendar_2025.parquet"
        cal.to_parquet(out_cal, index=False)
        info(f"GTFS: salvato {out_cal} con {len(cal)} righe")
    else:
        out_cal = None
        info("GTFS: calendar.txt non trovato — salto")

    return out_stops, out_cal


# --------------------
# Confini amministrativi (onData CSV)
# --------------------

def clean_confini(raw_dir: Path, interim_dir: Path) -> Path:
    src = raw_dir / "ondata_confini_amministrativi_api_v2_it_20250101_unita-territoriali-sovracomunali_22_comuni.csv"
    if not src.exists():
        info("Confini: file non trovato, salto")
        return None
    info(f"Confini: leggo {src}")
    df = pd.read_csv(src)
    keep = [
        "pro_com", "comune", "den_reg", "den_prov", "den_uts", "sigla",
        "shape_leng", "shape_area"
    ]
    keep = [c for c in keep if c in df.columns]
    out = interim_dir / "confini_trento.parquet"
    df[keep].to_parquet(out, index=False)
    info(f"Confini: salvato {out}")
    return out


# --------------------
# Main
# --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--interim", type=Path, default=Path("data/interim"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = ap.parse_args()

    ensure_dirs(args.raw, args.interim, args.processed)

    # Esecuzione step
    meteo_p = clean_meteo(args.raw, args.interim)
    mob_p = clean_mobility(args.raw, args.interim)
    stations_p = clean_stations(args.raw, args.interim)
    gtfs_stops_p, gtfs_cal_p = clean_gtfs(args.raw, args.interim)
    confini_p = clean_confini(args.raw, args.interim)

    info("Pulizia completata ✔")
    info(f"Interim: {args.interim}")
    for p in [meteo_p, mob_p, stations_p, gtfs_stops_p, gtfs_cal_p, confini_p]:
        if p is not None:
            info(f" - {p}")


if __name__ == "__main__":
    main()
