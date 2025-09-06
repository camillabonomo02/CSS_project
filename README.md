# CSS Project – Intermodal Urban Mobility in Trento

## Overview  
This repository contains the code and datasets for the final project of the *Computational Social Science* course (a.y. 2024/2025).  
The project investigates **intermodal urban mobility in Trento**, focusing on the integration between **bike-sharing stations**, the **public transport network**, and the role of **weather conditions** in shaping accessibility and usage patterns.  

**Research Questions**  
1. How are bike-sharing stations distributed relative to the local public transport network?  
2. Which urban areas appear underserved from an intermodal perspective?  
3. How do urban mobility patterns vary with weather?  

---

## Data  

### Bike-sharing and boundaries  
- `data/interim/circoscrizioni.geojson` – Administrative boundaries of Trento’s circoscrizioni.  
- `data/raw/stazioni_trento.csv` – Locations and attributes of bike-sharing stations.  

### Weather  
- `data/raw/trento_era5_daily_2020_2022.json` – Daily weather data (temperature and precipitation) for Trento, 2020–2022.  

### Public transport (GTFS)  
The project also includes a full GTFS feed from **Trentino Trasporti S.p.A.**. Not all files were directly used in the scripts, but they are preserved here for completeness:  

- `calendar.txt` – Weekly operating days and service periods:contentReference[oaicite:0]{index=0}.  
- `calendar_dates.txt` – Exceptions to the regular service calendar:contentReference[oaicite:1]{index=1}.  
- `feed_info.txt` – Publisher and feed metadata:contentReference[oaicite:2]{index=2}.  
- `routes.txt` – Route identifiers, names, and transport types:contentReference[oaicite:3]{index=3}.  
- `shapes.txt` – Geographic shapes of routes (polylines):contentReference[oaicite:4]{index=4}.  
- `stops.txt` – Locations and attributes of PT stops:contentReference[oaicite:5]{index=5}.  
- `stopslevel.txt` – Stop hierarchy information:contentReference[oaicite:6]{index=6}.  
- `stop_times.txt` – Scheduled arrival and departure times for each stop on a trip:contentReference[oaicite:7]{index=7}.  
- `transfers.txt` – Allowed transfers between stops:contentReference[oaicite:8]{index=8}.  
- `trips.txt` – Specific service trips linked to routes and shapes:contentReference[oaicite:9]{index=9}.  

---

## Pipeline  

Run the scripts in the following order to reproduce the project:  

### 1. `clean_all.py`  
- Cleans and prepares all raw datasets (bike-sharing, circoscrizioni, weather, GTFS).  
- Standardizes formats (CRS, dates, column names).  
- Outputs: processed versions under `data/processed/`.  

### 2. `build_datasets.py`  
- Integrates cleaned datasets into analysis-ready tables.  
- Constructs spatial features (e.g., station-to-circoscrizione assignments, PT proximity).  
- Outputs: consolidated datasets for later analysis.  

### 3. `analysis_suite.py`  
- Exploratory and descriptive analysis.  
- Produces summary statistics and plots (maps, distributions, temporal graphs).  
- Outputs: `results/figures/` and `results/tables/`.  

### 4. `population_stations_analysis.py`  
- Examines the relationship between **population distribution** and **bike-sharing accessibility**.  
- Computes accessibility per resident by circoscrizione.  
- Outputs: CSV tables and choropleth maps.  

### 5. `rq2_analysis.py`  
- Focuses on underserved areas from an intermodal perspective.  
- Computes **intermodality indicators** combining bike-sharing and PT accessibility.  
- Flags underserved circoscrizioni.  
- Outputs: maps and tables highlighting accessibility gaps.  

---

## Installation & Requirements  
To set up the environment:  

```bash
git clone https://github.com/camillabonomo02/CSS_project.git
cd CSS_project
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
---
## How to reproduce the project
Run the scripts in sequence:

```bash
python scripts/clean_all.py
python scripts/build_datasets.py
python scripts/analysis_suite.py
python scripts/population_stations_analysis.py
python scripts/rq2_analysis.py
```
Outputs (tables, figures, maps) will be saved in the results/ folder.

---

## Authors
Camilla Bonomo, Sara Lamouchi, Silvia Bortoluzzi, Diego Conti, Paolo Fabbri


