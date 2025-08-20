#!/usr/bin/env python3
"""
Tool unico per:
- EDA temporale su Mobility+Meteo (2022)
- Stima GAM su indicatori di mobilità (Gaussian) con smoothers su meteo
- Mappe dell'indice di intermodalità delle stazioni (GTFS 2025)

Uso tipico:
  python scripts/analysis_suite.py eda --processed data/processed --out reports
  python scripts/analysis_suite.py gam --processed data/processed --out reports
  python scripts/analysis_suite.py maps --processed data/processed --out reports

Dipendenze: pandas, numpy, matplotlib, (pygam), geopandas, contextily (opz.)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# opzionali
try:
    from pygam import LinearGAM, s, f as pf
except Exception:
    LinearGAM = None
    s = None
    f = None

try:
    import geopandas as gpd
    import contextily as ctx
except Exception:
    gpd = None
    ctx = None


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------

def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def info(msg: str) -> None:
    print(f"[analysis] {msg}")


# ------------------------------------------------------------
# 1) EDA temporale
# ------------------------------------------------------------

def run_eda(processed_dir: Path, out_dir: Path) -> None:
    temporal_p = processed_dir / 'temporal_2022.parquet'
    if not temporal_p.exists():
        raise FileNotFoundError("temporal_2022.parquet non trovato. Esegui build_datasets.py")

    df = pd.read_parquet(temporal_p)
    ensure_dirs(out_dir / 'figures')

    # 1) Serie temporali (media mobile 7 giorni)
    fig, ax = plt.subplots(figsize=(10, 4))
    for col, lab in [('mob_transit', 'Transit'), ('mob_work', 'Workplaces')]:
        if col in df:
            df.set_index('date')[col].rolling(7, min_periods=1).mean().plot(ax=ax, label=lab)
    ax.set_title('Mobilità (media mobile 7g) — Provincia Autonoma di Trento, 2022')
    ax.set_xlabel('Data'); ax.set_ylabel('% vs baseline'); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / 'figures/ts_mobility_rolling7.png', dpi=300)
    plt.close(fig)

    # 2) Scatter temp vs mobilità (con binned means)
    if 'temp_max' in df:
        for col in ['mob_transit', 'mob_work']:
            if col in df:
                bins = pd.cut(df['temp_max'], bins=10)
                means = df.groupby(bins)[col].mean()
                centers = [i.mid for i in means.index]
                fig, ax = plt.subplots(figsize=(5,4))
                ax.scatter(df['temp_max'], df[col], alpha=0.2, s=10)
                ax.plot(centers, means.values, linewidth=2)
                ax.set_title(f'{col} vs Temp max (binning)')
                ax.set_xlabel('Temp max (°C)'); ax.set_ylabel('% vs baseline')
                fig.tight_layout(); fig.savefig(out_dir / f'figures/scatter_{col}_temp.png', dpi=300)
                plt.close(fig)

    # 3) Boxplot precipitazioni (binario/heavy)
    if {'rain_binary','rain_heavy'}.issubset(df.columns):
        for flag, lab in [('rain_binary','Pioggia>0mm'), ('rain_heavy','Pioggia≥10mm')]:
            for col in ['mob_transit','mob_work']:
                if col in df:
                    fig, ax = plt.subplots(figsize=(4,4))
                    df.boxplot(column=col, by=flag, ax=ax)
                    ax.set_title(f'{col} per {lab}'); ax.set_xlabel(lab); ax.set_ylabel('% vs baseline')
                    fig.suptitle('')
                    fig.tight_layout(); fig.savefig(out_dir / f'figures/box_{col}_{flag}.png', dpi=300)
                    plt.close(fig)

    info("EDA completata — figure salvate in reports/figures/")


# ------------------------------------------------------------
# 2) GAM su mobilità
# ------------------------------------------------------------

def run_gam(processed_dir: Path, out_dir: Path) -> None:
    if LinearGAM is None:
        raise RuntimeError("pygam non disponibile o import incompleto. Installa 'pygam'.")

    temporal_p = processed_dir / 'temporal_2022.parquet'
    df = pd.read_parquet(temporal_p)

    # Feature set
    covs = []
    if 'temp_max' in df: covs.append('temp_max')
    if 'precip_mm' in df: covs.append('precip_mm')
    for cat in ['dow','is_weekend','is_holiday']:
        if cat in df: covs.append(cat)

    # Funzione ausiliaria per un target
    def fit_one(target: str):
        d = df.dropna(subset=[target] + covs).copy()
        # Costruiamo X con ordine controllato: numeriche prima, poi dummies
        num_cols = [c for c in ['temp_max','precip_mm'] if c in d.columns]
        cat_cols = [c for c in ['dow','is_weekend','is_holiday'] if c in d.columns]
        X_num = d[num_cols].copy() if num_cols else pd.DataFrame(index=d.index)
        X_cat = pd.get_dummies(d[cat_cols], columns=cat_cols, drop_first=True) if cat_cols else pd.DataFrame(index=d.index)
        X = pd.concat([X_num, X_cat], axis=1)
        y = d[target].astype(float).values
        # Termini GAM: smoothers sugli indici effettivi delle colonne numeriche
        terms = []
        term_idx_map = {}
        col_idx = {c:i for i,c in enumerate(X.columns)}
        if 'temp_max' in col_idx:
            terms.append(s(col_idx['temp_max']))
            term_idx_map['temp_max'] = len(terms)-1
        if 'precip_mm' in col_idx:
            terms.append(s(col_idx['precip_mm']))
            term_idx_map['precip_mm'] = len(terms)-1
        # Termini lineari (fattoriali) per il resto delle colonne
        used = set(num_cols)
        for i, c in enumerate(X.columns):
            if c in used:
                continue
            terms.append(pf(i))
        # Somma dei termini evitando sum()
        if not terms:
            # fallback: tutti lineari
            terms = [pf(i) for i in range(X.shape[1])]
        term_expr = terms[0]
        for t in terms[1:]:
            term_expr = term_expr + t
        gam = LinearGAM(term_expr).gridsearch(X.values, y)
        return gam, X.columns, term_idx_map

    ensure_dirs(out_dir / 'figures', out_dir / 'tables')

    for target in ['mob_transit', 'mob_work']:
        if target in df:
            gam, cols, term_idx = fit_one(target)
            # Salva riepilogo
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                gam.summary()
            with open(out_dir / f"tables/gam_{target}_summary.txt", 'w') as fh:
                fh.write(buf.getvalue())
            # Effetti parziali dei smoothers
            for label in [c for c in ['temp_max','precip_mm'] if c in cols and c in term_idx]:
                # Costruisci una griglia valida per il termine, con la stessa dimensionalità di X
                grid = gam.generate_X_grid(term=term_idx[label], n=100)
                feat_idx = list(cols).index(label)
                x_vals = grid[:, feat_idx]
                pdep = gam.partial_dependence(term=term_idx[label], X=grid)
                confi = gam.partial_dependence(term=term_idx[label], X=grid, width=0.95)
                fig, ax = plt.subplots(figsize=(5,4))
                ax.plot(x_vals, pdep)
                ax.plot(x_vals, confi[0], linestyle='--')
                ax.plot(x_vals, confi[1], linestyle='--')
                ax.set_title(f'GAM effetto parziale — {target} ~ s({label})')
                ax.set_xlabel(label); ax.set_ylabel(target)
                fig.tight_layout(); fig.savefig(out_dir / f'figures/gam_{target}_{label}.png', dpi=300)
                plt.close(fig)

    info("GAM completata — summary in reports/tables/, figure in reports/figures/")


# ------------------------------------------------------------
# 3) Mappe intermodalità
# ------------------------------------------------------------

def run_maps(processed_dir: Path, out_dir: Path) -> None:
    if gpd is None:
        raise RuntimeError("geopandas non disponibile per generare mappe")

    gpath = processed_dir / 'station_accessibility_2025.geojson'
    if not gpath.exists():
        raise FileNotFoundError("station_accessibility_2025.geojson non trovato. Esegui build_datasets.py")

    g = gpd.read_file(gpath)
    ensure_dirs(out_dir / 'figures', out_dir / 'tables')

    # Costruisci un indice semplice di intermodalità
    for b in (300, 500):
        s_col = f'stops_{b}m'; r_col = f'routes_{b}m'
        if s_col in g and r_col in g:
            g[f'idx_intermodal_{b}m'] = g[s_col].fillna(0) + 0.5 * g[r_col].fillna(0)

    # Tabella ranking top-10 per 300m
    if 'idx_intermodal_300m' in g:
        top = g[['station_id','name','dist_to_stop_m','stops_300m','routes_300m','idx_intermodal_300m']]
        top = top.sort_values('idx_intermodal_300m', ascending=False).head(10)
        top.to_csv(out_dir / 'tables/top10_intermodal_300m.csv', index=False)

    # Mappa semplice (EPSG:3857 per basemap se disponibile)
    try:
        g3857 = g.to_crs(3857)
        fig, ax = plt.subplots(figsize=(8,8))
        g3857.plot(ax=ax, markersize=30, alpha=0.8)
        if ctx is not None:
            try:
                ctx.add_basemap(ax, crs=g3857.crs)
            except Exception:
                pass
        ax.set_title('Stazioni bike sharing — indice di intermodalità (dimensione ~ 300m)')
        fig.tight_layout(); fig.savefig(out_dir / 'figures/map_stations_intermodal.png', dpi=300)
        plt.close(fig)
    except Exception as e:
        info(f"Impossibile generare mappa con basemap: {e}")

    info("Mappe completate — tabelle/figure salvate in reports/")


# ------------------------------------------------------------
# Main CLI
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task', choices=['eda','gam','maps'])
    ap.add_argument('--processed', type=Path, default=Path('data/processed'))
    ap.add_argument('--out', type=Path, default=Path('reports'))
    args = ap.parse_args()

    ensure_dirs(args.out)

    if args.task == 'eda':
        run_eda(args.processed, args.out)
    elif args.task == 'gam':
        run_gam(args.processed, args.out)
    elif args.task == 'maps':
        run_maps(args.processed, args.out)

    info('Done ✔')


if __name__ == '__main__':
    main()
