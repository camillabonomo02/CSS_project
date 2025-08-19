import pandas as pd
from pathlib import Path
import re

# --- INPUT: metti qui il file che hai effettivamente
SRC = Path("data/external/google_mobility/2022_IT_Region_Mobility_Report.csv")
# SRC = Path("data/external/google_mobility/Global_Mobility_Report.csv")

OUT = Path("data/processed"); OUT.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT/"gmr_day.parquet"

# --- Parametri di matching
REGION_REGEXES = [
    r"Trentino[-\s]Alto\sAdige(?:\s*[-/]?\s*S[uü]dtirol)?",  # copre molte varianti
]
PROVINCE_REGEXES = [
    r"^Trento$", r"Provincia.*Trento", r"Autonomous\s*Province\s*of\s*Trento"
]

# --- Carica
g = pd.read_csv(SRC, parse_dates=["date"])
for c in ["sub_region_1","sub_region_2","country_region"]:
    if c in g.columns:
        g[c] = g[c].fillna("")

# --- Filtra Italia
if "country_region" in g.columns:
    g = g[g["country_region"] == "Italy"]

# --- Tenta prima PROVINCIA (sub_region_2) se il file lo contiene
has_prov = (g["sub_region_2"] != "").any() if "sub_region_2" in g.columns else False
sel = pd.Series(False, index=g.index)

if has_prov:
    pat = re.compile("|".join(PROVINCE_REGEXES), flags=re.IGNORECASE)
    sel = g["sub_region_2"].str.match(pat)
    level = "province"
# Se non c'è livello provincia nel file, usa il livello REGIONE (sub_region_1)
if not sel.any():
    patR = re.compile("|".join(REGION_REGEXES), flags=re.IGNORECASE)
    sel = g["sub_region_1"].str.match(patR)
    level = "region"

g = g.loc[sel].copy()

if g.empty:
    raise ValueError(
        "Nessuna riga GMR trovata per Trentino/Trento nel file sorgente.\n"
        "Apri il CSV e verifica i valori esatti di sub_region_1/2; in caso, aggiorna le regex sopra."
    )

# --- Seleziona & rinomina colonne
rename_map = {
    "transit_stations_percent_change_from_baseline": "gmr_transit",
    "workplaces_percent_change_from_baseline":       "gmr_work",
    "retail_and_recreation_percent_change_from_baseline": "gmr_retail",
    "parks_percent_change_from_baseline":            "gmr_parks",
}
cols = ["date"] + [c for c in rename_map if c in g.columns]
g = g[cols].rename(columns=rename_map).sort_values("date").reset_index(drop=True)

# --- Tipi numerici coerenti
for c in ["gmr_transit","gmr_work","gmr_retail","gmr_parks"]:
    if c in g.columns:
        g[c] = pd.to_numeric(g[c], errors="coerce").astype("float64")

g["geo_level"] = level
g["region_name"] = "PAT/Trentino" if level=="province" else "Trentino-Alto Adige/Südtirol"

# --- Salva
g.to_parquet(OUT_PATH, index=False)
print("Salvato:", OUT_PATH, "→", g["date"].min(), "→", g["date"].max(), "| n=", len(g))
