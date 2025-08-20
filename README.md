# Obiettivo del progetto (aggiornato con GTFS)

Data la disponibilità di **dati sulle stazioni di bike sharing (localizzazione, caratteristiche)**, dati meteo (ERA5 daily fino al 2022), **Google Mobility Reports (fino al 2022)**, confini amministrativi e **GTFS di Trentino Trasporti (2025)**, l’obiettivo è costruire un’analisi **multilivello** che unisce dinamiche temporali (2020–2022) e struttura spaziale (2025).

L’idea centrale è studiare la relazione tra **meteo, mobilità urbana e accessibilità multimodale** (bike sharing + TPL), producendo insight utili per la pianificazione della mobilità sostenibile.

---

# Schema generale (overview)

1. **Domande di ricerca (RQ)**
2. **Raccolta dati e integrazione**
3. **Analisi temporale (EDA + GAM)**
4. **Analisi spaziale (bike sharing + GTFS)**
5. **Cluster analysis e mappe di copertura**
6. **Risultati e visualizzazioni**
7. **Discussione e limiti**

---

# 1) Domande di ricerca (adattate)

* **RQ1 (mobilità e meteo)**: Come variano i pattern di mobilità urbana (Google Mobility Reports 2020–2022) in relazione alle condizioni meteo (ERA5)?
* **RQ2 (potenziale bike sharing e intermodalità)**: Come sono distribuite le stazioni di bike sharing rispetto alla rete di trasporto pubblico locale (GTFS)?
* **RQ3 (integrazione e pianificazione)**: Quali aree urbane risultano sottoservite dal punto di vista intermodale (assenza di stazioni bici in prossimità di nodi TPL ad alta frequenza)?

---

# 2) Raccolta e integrazione dati

* **Stazioni** (`stazioni_trento.csv`): coordinate, nome stazione, eventuale capacità.
* **Meteo** (`trento_era5_daily_2020_2022.json`): temperatura media, massima, minima, precipitazioni, vento.
* **Google Mobility Reports** (`2022_IT_Region_Mobility_Report.csv`): variazioni percentuali rispetto a baseline (retail, workplaces, parks, transit stations, residential).
* **GTFS TPL (2025)**: fermate (`stops.txt`), orari (`stop_times.txt`), viaggi (`trips.txt`), percorsi (`routes.txt`), geometrie (`shapes.txt`), calendario (`calendar.txt`).
* **Confini amministrativi** (`ondata_confini...csv`): unità territoriali locali.

Dataset target:

* **Temporale (2020–2022)**: `date | mobility_metric | temp | rain | dow | holiday | ...`
* **Spaziale (2025)**: `station_id | lon | lat | nearest_GTFS_stop | num_routes | dist_center | ...`

---

# 3) Analisi temporale

* Serie storiche di mobilità (2020–2022), evidenziando fasi pandemiche e stagionalità.
* GAM: mobilità (*Transit stations*, *Workplaces*) \~ meteo + calendario.
* Output: curve non lineari (temperature, pioggia) con CI.

---

# 4) Analisi spaziale

* Join stazioni bici ↔ fermate GTFS (buffer 300/500m).
* Calcolo **indice di intermodalità**: numero di fermate e linee accessibili entro soglia.
* Analisi di copertura: aree urbane senza stazioni bici ma con TPL ad alta frequenza.

---

# 5) Cluster analysis (tipologie di stazioni)

* Variabili: distanza da centro, vicinanza a fermate TPL, numero di linee accessibili, densità servizi.
* Metodo: k‑means o clustering gerarchico.
* Output: tipologie di stazioni (centrali multimodali, periferiche ricreative, intermedie).

---

# 6) Validazione & robustness

* GAM: modelli alternativi (OLS, NegBin) e confronto categorie di mobilità (workplaces vs parks).
* Analisi spaziale: test con buffer diversi (200/500/800m).
* Cluster: silhouette score per scelta numero cluster.

---

# 7) Risultati attesi

* Effetti meteo chiari sulla mobilità urbana (es. riduzione spostamenti in giornate piovose >10mm).
* Identificazione di stazioni poco intermodali (lontane da fermate TPL).
* Cluster tipologici che distinguono stazioni centrali/pendolari da stazioni periferiche/ricreative.
* Mappe di opportunità per espansione bike sharing.

---

# 8) Discussione e limiti

* **Limite temporale**: mismatch dati 2022 (mobilità+meteo) vs 2025 (GTFS). → I GTFS sono usati come *proxy strutturale* e non per analisi temporale.
* **Valore aggiunto**: analisi integrata di **mobilità passata e infrastruttura attuale**, utile per pianificazione.
* Possibile estensione: ottenere GTFS storici o log bike sharing per validare meglio.

---

# 9) Prossimi passi

1. Costruire dataset integrato **mobility + meteo + calendario (2020–2022)**.
2. Stimare GAM su mobilità urbana.
3. Integrare stazioni con GTFS (prossimità, linee, intermodalità).
4. Cluster analysis delle stazioni.
5. Preparare 2 figure temporali + 2 mappe spaziali + 1 tabella regressione GAM.

---

Questa nuova impostazione chiarisce come usare i GTFS 2025 per l’analisi spaziale e la pianificazione, mantenendo coerenza con i dati 2020–2022 per la parte temporale.
