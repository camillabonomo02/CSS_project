import os, time, json, requests
from pathlib import Path
from datetime import datetime, timezone

CITY_ID   = os.environ.get("BICINCITTA_CITY_ID", "187")  # Trento
BASE      = "https://www.bicincitta.com/frmLeStazioniComune.aspx"
REFERER   = f"{BASE}?ID={CITY_ID}"
ENDPOINT  = f"{BASE}/RefreshStations"
FREQ_SEC  = int(os.environ.get("POLL_EVERY_SEC", "300"))  # ogni 5 min
OUTDIR    = Path("data/raw/bikesharing_trento/status"); OUTDIR.mkdir(parents=True, exist_ok=True)

UA = "css_project_gpt-research/1.0"
HEADERS = {
    "User-Agent": UA,
    "Accept": "application/json, text/javascript, */*;q=0.1",
    "Content-Type": "application/json; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
    "Origin": "https://www.bicincitta.com",
    "Referer": REFERER,
}

def parse_station_record(rec_str: str):
    """
    Rec string format (fields separated by '§'):
    0: station_id (es. 1121)
    1: lat
    2: lon
    3: name (es. '10.01 Bren Center')
    4: codice interno (non indispensabile)
    5: bikes (disponibili al momento)
    6: docks (posti liberi al momento)
    """
    parts = rec_str.split("§")
    if len(parts) < 7:
        return None
    try:
        station_id = parts[0].strip()
        lat  = float(parts[1].replace(",", "."))
        lon  = float(parts[2].replace(",", "."))
        name = parts[3].strip()
        bikes = int(parts[5])
        docks = int(parts[6])
        return {
            "station_id": station_id,
            "name": name,
            "lat": lat,
            "lon": lon,
            "bikes": bikes,
            "docks": docks
        }
    except Exception:
        return None

def fetch_once(sess: requests.Session):
    # 1) cookie di sessione
    sess.get(REFERER, headers={"User-Agent": UA}, timeout=20)
    # 2) POST -> RefreshStations (ASP.NET WebMethod)
    r = sess.post(ENDPOINT, headers=HEADERS,
                  data=json.dumps({"IDComune": CITY_ID}), timeout=20)
    r.raise_for_status()
    js = r.json()
    payload = js.get("d", [])
    if not isinstance(payload, list) or len(payload) == 0:
        return []

    # elemento 0 = url immagini (ignoralo); da 1 in poi: record '§'
    rows = []
    for s in payload[1:]:
        if not isinstance(s, str):
            continue
        row = parse_station_record(s)
        if row:
            rows.append(row)
    return rows

def save_snapshot(rows):
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y%m%dT%H%M%S")
    out = OUTDIR / f"status_{ts}.ndjson"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            r["timestamp"] = ts
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {out} | rows: {len(rows)} | sample: {rows[0] if rows else 'n/a'}")

if __name__ == "__main__":
    sess = requests.Session()
    try:
        once = os.environ.get("FETCH_ONCE", "1") == "1"
        rows = fetch_once(sess)
        if not rows:
            print("⚠️ Nessuna stazione parsata: controlla che la pagina mostri dati e riprova.")
        else:
            save_snapshot(rows)
        if not once:
            while True:
                time.sleep(FREQ_SEC)
                rows = fetch_once(sess)
                if rows:
                    save_snapshot(rows)
    finally:
        sess.close()
