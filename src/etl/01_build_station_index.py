"""
Costruisce l'indice delle stazioni a partire da:
  data/raw/bikesharing_trento/stazioni_trento.csv

Il file ha separatore ';' e colonne:
- WKT: geometria in UTM32N (es. 'POINT (663132.53 5104569.75)')
- id: identificativo numerico della stazione
- fumetto: etichetta sintetica (usata come 'name')
- desc: descrizione testuale
- cicloposteggi: capacità (numero stalli)
- tipologia: stringa informativa (non usata qui)

Output:
- data/processed/station_index.geo.parquet  (GeoDataFrame in EPSG:4326)
- (opz.) data/processed/station_index.csv   (tabellare con lon/lat)
"""

from pathlib import Path
import re
import pandas as pd
import geopandas as gpd

# --- Percorsi
IN_PATH  = Path("data/raw/bikesharing_trento/stazioni_trento.csv")
OUT_DIR  = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _clean_name(s: str) -> str:
    """Pulizia minima del nome: spazi doppi, maiuscole coerenti su parole comuni."""
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\s+", " ", s.strip())
    # uniforma apostrofi/accents basilari
    s = s.replace("UNIVERSITA'", "Università").replace("FF.SS.", "FS")
    # capitalizzazione soft mantenendo sigle
    return " ".join(w.capitalize() if not w.isupper() else w for w in s.split(" "))

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"CSV stazioni non trovato: {IN_PATH}")

    # 1) Leggi CSV (separatore ';')
    df = pd.read_csv(IN_PATH, sep=";")

    # Controllo colonne attese
    expected = {"WKT","id","fumetto","desc","cicloposteggi","tipologia"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel CSV: {missing}")

    # 2) Geometria da WKT -> GeoDataFrame in UTM32N
    g = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.GeoSeries.from_wkt(df["WKT"]),
        crs="EPSG:32632"  # UTM 32N coerente con valori (E≈663k, N≈5.1M)
    )

    # 3) Riproj in WGS84 e crea lon/lat
    g = g.to_crs(epsg=4326)
    g["lon"] = g.geometry.x
    g["lat"] = g.geometry.y

    # 4) Rinomina/normalizza campi chiave
    g = g.rename(columns={
        "id": "station_id",
        "fumetto": "name",
        "cicloposteggi": "capacity",
    })
    g["name"] = g["name"].map(_clean_name)
    # garantisci tipi
    g["station_id"] = g["station_id"].astype(int)
    g["capacity"]   = pd.to_numeric(g["capacity"], errors="coerce").astype("Int64")

    # 5) (Facoltativo) zona placeholder: da valorizzare più avanti con poligoni/heuristics
    g["zone"] = pd.Series(["unknown"]*len(g), dtype="string")

    # 6) Selezione colonne finali (mantieni desc per utilità)
    cols = ["station_id","name","desc","capacity","lon","lat","zone","geometry"]
    g_out = g[cols].copy()

    # 7) Salva
    out_parquet = OUT_DIR / "station_index.geo.parquet"
    g_out.to_parquet(out_parquet, index=False)

    # CSV “piatto” (senza geometria) utile per rapida ispezione
    out_csv = OUT_DIR / "station_index.csv"
    g_out.drop(columns=["geometry"]).to_csv(out_csv, index=False)

    print(f"Salvati:\n - {out_parquet}\n - {out_csv}")
    print(f"Righe: {len(g_out)} | Colonne: {list(g_out.columns)}")

if __name__ == "__main__":
    main()
