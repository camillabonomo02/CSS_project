from pathlib import Path

BASE = Path("data")
paths = {
    "GMR": BASE/"external/google_mobility/2022_IT_Region_Mobility_Report.csv",
    "GTFS_urb": BASE/"external/gtfs/google_transit_urbano_tte",
    "GTFS_extra": BASE/"external/gtfs/google_transit_extraurbano_tte",
    "BSS_dir": BASE/"raw/bikesharing_trento",
}

for k,p in paths.items():
    if p.suffix:  # file
        print(k, "OK" if p.exists() else "MISSING", "→", p)
    else:         # dir
        print(k, "OK" if p.exists() and p.is_dir() else "MISSING", "→", p)

# elenca file GTFS essenziali
for gt in ["GTFS_urb","GTFS_extra"]:
    d = paths[gt]
    if d.exists():
        need = ["stops.txt","routes.txt","trips.txt","stop_times.txt","calendar.txt","calendar_dates.txt"]
        missing = [f for f in need if not (d/f).exists()]
        print(f"{gt}: missing={missing}" if missing else f"{gt}: all good")