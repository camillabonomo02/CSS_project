import pandas as pd
from pathlib import Path

# === INPUT/OUTPUT ===
IN_CSV = Path("data/external/google_mobility/2022_IT_Region_Mobility_Report.csv")  # <-- aggiorna se serve
OUT = Path("data/processed"); OUT.mkdir(parents=True, exist_ok=True)

# Quale regione vuoi estrarre (regex robusta ai diversi formati del nome)
REGION_REGEX = r"Trentino[-\s]Alto\sAdige(?:.?S[uü]dtirol)?"

# === LOAD ===
g = pd.read_csv(IN_CSV, parse_dates=["date"])

# Colonne utili possono variare poco tra release; normalizziamo i test
g["sub_region_1"] = g["sub_region_1"].fillna("")
g["sub_region_2"] = g["sub_region_2"].fillna("")

# === FILTER: Italia + livello REGIONE (sub_region_2 vuoto) + regione desiderata ===
flt = (
    (g["country_region"] == "Italy")
    & (g["sub_region_2"] == "")                      # livello REGIONALE (non provinciale)
    & (g["sub_region_1"].str.contains(REGION_REGEX, case=False, regex=True))
)

# === KEEP & RENAME (stesso schema che usavi prima) ===
rename_map = {
    "transit_stations_percent_change_from_baseline": "gmr_transit",
    "workplaces_percent_change_from_baseline":       "gmr_work",
    "retail_and_recreation_percent_change_from_baseline": "gmr_retail",
    "parks_percent_change_from_baseline":            "gmr_parks",
    # Se ti servono: "residential_percent_change_from_baseline": "gmr_residential",
    #                "grocery_and_pharmacy_percent_change_from_baseline": "gmr_grocery",
}

cols_presenti = ["date"] + [c for c in rename_map.keys() if c in g.columns]
if len(cols_presenti) == 1:
    raise ValueError("Le colonne GMR attese non sono presenti nel CSV delle regioni.")

g = (g.loc[flt, cols_presenti]
       .rename(columns=rename_map)
       .sort_values("date")
       .reset_index(drop=True))

# (Opzionale) annota che è livello regionale
g["geo_level"] = "region"
g["region_name"] = "Trentino-Alto Adige/Südtirol"

# === SAVE ===
out_path = OUT / "gmr_day.parquet"   # mantengo lo stesso nome per non rompere la pipeline a valle
g.to_parquet(out_path, index=False)
print("Salvato:", out_path, "→", g.shape)