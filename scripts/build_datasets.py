#!/usr/bin/env python3
"""
Costruisce i dataset analitici da usare nei modelli/figure.

Output principali:
- data/processed/temporal_2022.parquet  # Mobility + Meteo + Calendario
- data/processed/station_accessibility_2025.parquet  # indicatori di intermodalità per stazione
- data/processed/station_accessibility_2025.geojson  # versione geospaziale per mappe

Dipendenze: pandas, pyarrow, geopandas, shapely, numpy, (opz.) holidays
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# opzionali
try:
    import geopandas as gpd
    from shapely.geometry import Point
except Exception:
    gpd = None

try:
    import holidays
    IT_HOL = holidays.country_holidays('IT')
except Exception:
    IT_HOL = None


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------

def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def info(msg: str) -> None:
    print(f"[build] {msg}")


# ------------------------------------------------------------
# 1) Dataset temporale 2022 (Mobility + Meteo + Calendario)
# ------------------------------------------------------------

def build_temporal(interim_dir: Path, processed_dir: Path) -> Path:
    meteo_p = interim_dir / "meteo_daily.parquet"
    mob_p = interim_dir / "mobility_trento_2022.parquet"
    if not meteo_p.exists() or not mob_p.exists():
        raise FileNotFoundError("Mancano meteo_daily.parquet o mobility_trento_2022.parquet: esegui prima clean_all.py")

    meteo = pd.read_parquet(meteo_p)
    mob = pd.read_parquet(mob_p)

    # Seleziona 2022 dal meteo e fai il merge per data
    meteo_2022 = meteo[meteo['date'].dt.year == 2022].copy()
    df = mob.merge(meteo_2022, on='date', how='left')

    # Calendario di utilità
    df['dow'] = df['date'].dt.dayofweek  # 0=Lunedì
    df['is_weekend'] = df['dow'] >= 5
    if IT_HOL is not None:
        df['is_holiday'] = df['date'].dt.date.apply(lambda d: d in IT_HOL)
    else:
        info("Pacchetto 'holidays' non disponibile: 'is_holiday' impostato a False")
        df['is_holiday'] = False

    # Feature meteo trasformate semplici
    if 'temp_max' in df:
        df['temp_max_sq'] = df['temp_max'] ** 2
    if 'precip_mm' in df:
        df['rain_binary'] = (df['precip_mm'] > 0).astype(int)
        df['rain_heavy'] = (df['precip_mm'] >= 10).astype(int)

    out = processed_dir / "temporal_2022.parquet"
    df.sort_values('date').to_parquet(out, index=False)
    info(f"Temporale: salvato {out} con shape {df.shape}")
    return out


# ------------------------------------------------------------
# 2) Dataset spaziale stazioni + GTFS (2025)
# ------------------------------------------------------------

def _routes_per_stop(raw_dir: Path) -> pd.DataFrame:
    """Ritorna per ogni stop_id il numero di route_id unici che vi servono.
    Usa stop_times -> trips -> routes. Riduce drast. la dimensione con aggregazioni.
    """
    stop_times = pd.read_csv(raw_dir / 'stop_times.txt', usecols=['stop_id', 'trip_id'])
    trips = pd.read_csv(raw_dir / 'trips.txt', usecols=['trip_id', 'route_id'])
    # merge e aggrega
    st_tr = stop_times.merge(trips, on='trip_id', how='left')
    agg = (st_tr.groupby('stop_id')['route_id']
                  .nunique(dropna=True)
                  .rename('n_routes_at_stop')
                  .reset_index())
    return agg


def build_spatial(raw_dir: Path, interim_dir: Path, processed_dir: Path,
                  buffer_m: list[int] = [300, 500]) -> tuple[Path, Path]:
    if gpd is None:
        raise RuntimeError("geopandas richiesto per la parte spaziale")

    stations_p = interim_dir / 'stations_clean.geojson'
    stops_p = interim_dir / 'gtfs_stops_2025.geojson'
    if not stations_p.exists() or not stops_p.exists():
        raise FileNotFoundError("Mancano stations_clean.geojson o gtfs_stops_2025.geojson: esegui clean_all.py")

    gstations = gpd.read_file(stations_p).to_crs(3857)  # metri
    gstops = gpd.read_file(stops_p).to_crs(3857)

    # Nearest stop (distanza in metri)
    nearest = gpd.sjoin_nearest(gstations, gstops[['stop_id', 'stop_name', 'geometry']], how='left', distance_col='dist_to_stop_m')

    # Precalcola quanti routes servono ciascuna fermata
    routes_at_stop = _routes_per_stop(raw_dir)
    gstops_routes = gstops.merge(routes_at_stop, on='stop_id', how='left').fillna({'n_routes_at_stop': 0})

    # Buffer analysis per ogni soglia
    results = []
    for buf in buffer_m:
        ring = gstops_routes.copy()
        ring['geometry'] = ring.buffer(buf)
        # spatial join: per ogni stazione, conta fermate e somma routes unici nella zona
        # per contare routes unici nell'area attorno a ciascuna stazione: prima associare stop->routes count,
        # poi aggregare numero di stop e somma delle route distinct proxy (approssimazione strutturale)
        joined = gpd.sjoin(gstations[['station_id', 'geometry']], ring[['n_routes_at_stop', 'geometry']], how='left', predicate='intersects')
        agg = (joined.groupby('station_id')
                      .agg(stops_in_buf=('n_routes_at_stop', 'size'),
                           routes_in_buf=('n_routes_at_stop', 'sum'))
                      .reset_index())
        agg = agg.rename(columns={'stops_in_buf': f'stops_{buf}m', 'routes_in_buf': f'routes_{buf}m'})
        results.append(agg)

    # Unisci tutti i risultati su gstations
    acc = nearest[['station_id', 'stop_id', 'stop_name', 'dist_to_stop_m']].copy()
    for r in results:
        acc = acc.merge(r, on='station_id', how='left')

    # riporta in EPSG:4326 per export geo
    gstations4326 = gstations.to_crs(4326)
    gacc = gstations4326.merge(acc, on='station_id', how='left')

    # Valori NaN -> 0 per conteggi
    for c in [f'stops_{b}m' for b in buffer_m] + [f'routes_{b}m' for b in buffer_m]:
        if c in gacc.columns:
            gacc[c] = gacc[c].fillna(0).astype(int)

    # Ordina colonne
    ordered = ['station_id', 'name', 'capacity', 'type', 'lat', 'lon', 'stop_id', 'stop_name', 'dist_to_stop_m'] \
              + [f'stops_{b}m' for b in buffer_m] + [f'routes_{b}m' for b in buffer_m]
    ordered = [c for c in ordered if c in gacc.columns] + [c for c in gacc.columns if c not in ordered and c != 'geometry']
    gacc = gacc[ordered + ['geometry']]

    out_parquet = processed_dir / 'station_accessibility_2025.parquet'
    out_geojson = processed_dir / 'station_accessibility_2025.geojson'
    ensure_dirs(processed_dir)
    pd.DataFrame(gacc.drop(columns='geometry')).to_parquet(out_parquet, index=False)
    gacc.to_file(out_geojson, driver='GeoJSON')
    info(f"Spaziale: salvati {out_parquet} e {out_geojson} (n={len(gacc)})")
    return out_parquet, out_geojson


# ------------------------------------------------------------
# Main CLI
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', type=Path, default=Path('data/raw'))
    ap.add_argument('--interim', type=Path, default=Path('data/interim'))
    ap.add_argument('--processed', type=Path, default=Path('data/processed'))
    args = ap.parse_args()

    ensure_dirs(args.processed)

    # 1) temporale
    temporal_p = build_temporal(args.interim, args.processed)

    # 2) spaziale
    spatial_pq, spatial_gj = build_spatial(args.raw, args.interim, args.processed)

    info('Build completato ✔')
    info(f" - {temporal_p}")
    info(f" - {spatial_pq}")
    info(f" - {spatial_gj}")


if __name__ == '__main__':
    main()
