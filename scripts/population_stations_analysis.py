#!/usr/bin/env python3
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import matplotlib.patheffects as pe

# -------------------- Settings --------------------
CRS = "EPSG:32632"
BUFFER = 300
OUT_FIG = Path("reports/figures")
OUT_TAB = Path("reports/tables")
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

# -------------------- Load Data --------------------
circ = gpd.read_file("data/interim/circoscrizioni.geojson", engine="pyogrio").to_crs(CRS)
pop = pd.read_csv("data/interim/famiglie_circoscrizioni_2024.csv")
stations = gpd.read_file("data/processed/station_accessibility_2025.geojson").to_crs(CRS)
stops = pd.read_csv("data/raw/stops.txt")
stops = gpd.GeoDataFrame(stops, geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat), crs="EPSG:4326").to_crs(CRS)

# -------------------- Merge Population --------------------
def norm(s): return s.strip().lower().replace("–", "-").replace("—", "-")
circ["key"] = circ["nome"].map(norm)
pop["key"] = pop["Circumscription"].map(norm)
circ = circ.merge(pop[["key", "Families_2024"]], on="key", how="left")

# -------------------- Identify Intermodal Stations --------------------
stops_buffer = gpd.GeoDataFrame(geometry=stops.buffer(BUFFER), crs=CRS)
joined = gpd.sjoin(stations, stops_buffer, how="inner", predicate="intersects")
stations["intermodal"] = stations.index.isin(joined.index)

# -------------------- Compute Coverage and Stats --------------------
cov = stations[stations["intermodal"]].buffer(BUFFER)
srv_poly = gpd.GeoSeries([cov.unary_union], crs=CRS)

circ["area_total"] = circ.geometry.area
circ["area_served"] = circ.geometry.intersection(srv_poly.iloc[0]).area
circ["share_served"] = circ["area_served"] / circ["area_total"]
circ["pop_served"] = circ["Families_2024"] * circ["share_served"]
circ["perc_served"] = 100 * (circ["pop_served"] / circ["Families_2024"])

# -------------------- Classification --------------------
bins = [0, 10, 20, 30, 40, 50, 100]
labels = ["0–10%", "10–20%", "20–30%", "30–40%", "40–50%", ">50%"]
circ["served_class"] = pd.cut(circ["perc_served"], bins=bins, labels=labels, include_lowest=True)

# -------------------- Clip Stops to Trento Area --------------------
stops = stops[stops.geometry.within(circ.unary_union)]

# -------------------- Plot: Classed Map --------------------
fig, ax = plt.subplots(figsize=(9, 10))
circ.plot(ax=ax, column="served_class", cmap="YlGnBu", legend=False, edgecolor="white", linewidth=0.5)

# --- Legend patches
served_handles = [mpatches.Patch(color=plt.cm.YlGnBu(i / (len(labels)-1)), label=label) for i, label in enumerate(labels)]
infra_handles = [
    mpatches.Patch(color="black", label="Bike stations"),
    mpatches.Patch(color="red", label="Transit stops")
]
legend_all = served_handles + infra_handles

ax.legend(handles=legend_all, title="% Families Served & Infrastructure", loc="lower left", frameon=True)


stations.plot(ax=ax, color="black", markersize=15, alpha=0.8, label="Bike stations")
stops.plot(ax=ax, color="red", markersize=4, alpha=0.4, label="Transit stops")

for idx, row in circ.iterrows():
    ax.annotate(row["nome"], xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                fontsize=8, ha="center", va="center", color="black",
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])

ax.set_title("Families served by intermodal hubs (classified)", fontsize=14)
ax.axis("off")
plt.tight_layout()
plt.savefig(OUT_FIG / "intermodal_classed_map.png", dpi=300)

# -------------------- Plot: Sorted Bar Chart --------------------
sorted_data = circ.sort_values("perc_served", ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(sorted_data["nome"], sorted_data["perc_served"], color="mediumseagreen")
ax.set_xlabel("% families served")
ax.set_title("Share of families served by intermodal hubs")
ax.invert_yaxis()
for bar in bars:
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", va='center', fontsize=8)

plt.tight_layout()
plt.savefig(OUT_FIG / "intermodal_served_barplot.png", dpi=300)

# -------------------- Save Summary Table --------------------
circ[["nome", "Families_2024", "perc_served"]].to_csv(OUT_TAB / "intermodal_population_coverage.csv", index=False)
