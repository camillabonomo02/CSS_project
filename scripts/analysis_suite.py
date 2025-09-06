#!/usr/bin/env python3
"""
Analysis suite:
- Temporal EDA on Mobility + Weather (2022)
- GAM estimation (Gaussian) with smoothers for weather
- Intermodality maps & rankings (GTFS 2025)

Default:
  python3 scripts/analysis_suite.py        # runs eda + gam + maps

Optional:
  python3 scripts/analysis_suite.py eda  --processed data/processed --out reports
  python3 scripts/analysis_suite.py gam  --processed data/processed --out reports
  python3 scripts/analysis_suite.py maps --processed data/processed --out reports

Requirements: pandas, numpy, matplotlib, (pygam), geopandas, contextily (optional)
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from pygam import LinearGAM, s, f as pf
except Exception:
    LinearGAM = None
    s = None
    pf = None

try:
    import geopandas as gpd
    import contextily as ctx
except Exception:
    gpd = None
    ctx = None


# ------------------------- Plot helpers -------------------------
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

def style_axes(ax, title: str, xlabel: str, ylabel: str, legend: bool = True):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", alpha=0.25)
    if legend:
        ax.legend(frameon=False)


# ------------------------- Utils -------------------------
def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def info(msg: str) -> None:
    print(f"[analysis] {msg}")


# ------------------------- 1) Temporal EDA -------------------------
def run_eda(processed_dir: Path, out_dir: Path) -> None:
    temporal_p = processed_dir / 'temporal_2022.parquet'
    if not temporal_p.exists():
        raise FileNotFoundError("temporal_2022.parquet not found. Run build_datasets.py first.")

    df = pd.read_parquet(temporal_p)
    ensure_dirs(out_dir / 'figures')

    # Time series (7-day rolling means)
    fig, ax = plt.subplots(figsize=(10, 4))
    for col, lab in [('mob_transit', 'Transit stations'), ('mob_work', 'Workplaces')]:
        if col in df:
            df.set_index('date')[col].rolling(7, min_periods=1).mean().plot(ax=ax, label=lab, linewidth=2)
    style_axes(ax,
               title='Mobility (7-day rolling average) — Autonomous Province of Trento, 2022',
               xlabel='Date', ylabel='% vs baseline', legend=True)
    fig.tight_layout()
    fig.savefig(out_dir / 'figures/ts_mobility_rolling7_en.png')
    plt.close(fig)

    # Scatter: temperature vs mobility (with binned means)
    if 'temp_max' in df:
        for col, lab in [('mob_transit','Transit stations'), ('mob_work','Workplaces')]:
            if col in df:
                bins = pd.cut(df['temp_max'], bins=10)
                means = df.groupby(bins)[col].mean()
                centers = np.array([b.mid for b in means.index])
                fig, ax = plt.subplots(figsize=(5.5,4))
                ax.scatter(df['temp_max'], df[col], alpha=0.25, s=10, label='Daily observations')
                ax.plot(centers, means.values, linewidth=2, label='Binned mean (10 bins)')
                style_axes(ax,
                           title=f'{lab} vs Max temperature (°C)',
                           xlabel='Max temperature (°C)', ylabel='% vs baseline', legend=True)
                fig.tight_layout()
                fig.savefig(out_dir / f'figures/scatter_{col}_temp_en.png')
                plt.close(fig)

    # Boxplots: precipitation (binary / heavy)
    if {'rain_binary','rain_heavy'}.issubset(df.columns):
        for flag, flag_lab in [('rain_binary','Rain > 0mm'), ('rain_heavy','Rain ≥ 10mm')]:
            for col, lab in [('mob_transit','Transit stations'), ('mob_work','Workplaces')]:
                if col in df:
                    fig, ax = plt.subplots(figsize=(5.5,3))
                    data0 = df.loc[df[flag] == 0, col].dropna().values
                    data1 = df.loc[df[flag] == 1, col].dropna().values
                    ax.boxplot([data0, data1], labels=['No', 'Yes'])
                    style_axes(ax,
                               title=f'{lab} by {flag_lab}',
                               xlabel=flag_lab, ylabel='% vs baseline', legend=False)
                    fig.tight_layout()
                    fig.savefig(out_dir / f'figures/box_{col}_{flag}_en.png')
                    plt.close(fig)

    info("EDA complete — figures saved to reports/figures/")


# ------------------------- 2) GAM on mobility -------------------------
def run_gam(processed_dir: Path, out_dir: Path) -> None:
    if LinearGAM is None:
        raise RuntimeError("pygam not available. Please `pip install pygam`.")

    temporal_p = processed_dir / 'temporal_2022.parquet'
    df = pd.read_parquet(temporal_p)

    # Covariates
    covs = []
    if 'temp_max' in df: covs.append('temp_max')
    if 'precip_mm' in df: covs.append('precip_mm')
    for cat in ['dow','is_weekend','is_holiday']:
        if cat in df: covs.append(cat)

    def fit_one(target: str):
        d = df.dropna(subset=[target] + covs).copy()
        # X: numeric first, then dummies (stable order)
        num_cols = [c for c in ['temp_max','precip_mm'] if c in d.columns]
        cat_cols = [c for c in ['dow','is_weekend','is_holiday'] if c in d.columns]
        X_num = d[num_cols].copy() if num_cols else pd.DataFrame(index=d.index)
        X_cat = pd.get_dummies(d[cat_cols], columns=cat_cols, drop_first=True) if cat_cols else pd.DataFrame(index=d.index)
        X = pd.concat([X_num, X_cat], axis=1)
        y = d[target].astype(float).values

        # Terms
        terms = []
        term_idx_map = {}
        col_idx = {c: i for i, c in enumerate(X.columns)}
        if 'temp_max' in col_idx:
            terms.append(s(col_idx['temp_max']))
            term_idx_map['temp_max'] = len(terms) - 1
        if 'precip_mm' in col_idx:
            terms.append(s(col_idx['precip_mm']))
            term_idx_map['precip_mm'] = len(terms) - 1

        used = set(num_cols)
        for i, c in enumerate(X.columns):
            if c in used:  # already smoothed
                continue
            terms.append(pf(i))

        if not terms:
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

            # Save textual summary by capturing stdout
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                gam.summary()
            with open(out_dir / f"tables/gam_{target}_summary.txt", 'w') as fh:
                fh.write(buf.getvalue())

            # Partial effects
            for label in [c for c in ['temp_max','precip_mm'] if c in cols and c in term_idx]:
                grid = gam.generate_X_grid(term=term_idx[label], n=200)
                feat_idx = list(cols).index(label)

                # x e partial effect → 1D
                x_vals = np.asarray(grid[:, feat_idx]).reshape(-1)
                pdep   = np.asarray(gam.partial_dependence(term=term_idx[label], X=grid)).reshape(-1)

                # CI: gestisci tuple/lista vs array (2,n) / (n,2) / colonne
                confi = gam.partial_dependence(term=term_idx[label], X=grid, width=0.95)
                if isinstance(confi, (list, tuple)) and len(confi) == 2:
                    ci_lo = np.asarray(confi[0]).reshape(-1)
                    ci_hi = np.asarray(confi[1]).reshape(-1)
                else:
                    ci_arr = np.asarray(confi)
                    # ora è un array; normalizza alle forme comuni
                    if ci_arr.ndim == 1:
                        ci_lo = ci_hi = ci_arr.reshape(-1)
                    elif ci_arr.shape[0] == 2:
                        ci_lo, ci_hi = ci_arr[0].reshape(-1), ci_arr[1].reshape(-1)
                    elif ci_arr.shape[-1] == 2:
                        ci_lo, ci_hi = ci_arr[..., 0].reshape(-1), ci_arr[..., 1].reshape(-1)
                    else:
                        # fallback prudente
                        ci_lo, ci_hi = ci_arr[0].reshape(-1), ci_arr[1].reshape(-1)

                # allinea lunghezze e ordina per x
                m = min(len(x_vals), len(pdep), len(ci_lo), len(ci_hi))
                x_vals, pdep, ci_lo, ci_hi = x_vals[:m], pdep[:m], ci_lo[:m], ci_hi[:m]
                order = np.argsort(x_vals)
                xv, mu, lo, hi = x_vals[order], pdep[order], ci_lo[order], ci_hi[order]

                fig, ax = plt.subplots(figsize=(6,4))
                ax.fill_between(xv, lo, hi, color="0.7", alpha=0.3, label="95% CI")  # banda grigia
                ax.plot(xv, mu, color="C0", linewidth=2, label="Partial effect")     # linea blu

                # Etichette asse x (come concordato)
                if label == "precip_mm":
                    ax.set_xlabel("Daily precipitation (mm)")
                    ax.set_xticks([0, 5, 10, 20])
                    ax.set_xticklabels(["0 (no rain)", "5", "10", "20 (heavy)"])
                else:
                    ax.set_xlabel("Daily max temperature (°C)")
                    ax.set_xticks([0, 5, 10, 15, 20, 30, 35])
                    ax.set_xticklabels(["0", "5", "10", "15", "20", "30", "35+"])

                ax.set_ylabel(f"{target} mobility (pp vs baseline)")
                ax.set_ylim(-15, 15)

                def _gam_title(target: str, label: str) -> str:
                    tgt = "Transit mobility" if target == "mob_transit" else "Workplace mobility"
                    var = "precipitation" if label == "precip_mm" else "temperature"
                    return f"{tgt} — GAM effect of {var}"

                style_axes(ax,
                        title=_gam_title(target, label),
                        xlabel=ax.get_xlabel(),
                        ylabel=ax.get_ylabel(),
                        legend=True)


                fig.tight_layout()
                fig.savefig(out_dir / f'figures/gam_{target}_{label}_en.png')
                plt.close(fig)


    info("GAM complete — summaries in reports/tables/, figures in reports/figures/")


# ------------------------- 3) Intermodality maps -------------------------
def run_maps(processed_dir: Path, out_dir: Path) -> None:
    # 1) requisiti
    if gpd is None:
        raise RuntimeError("geopandas not available. `pip install geopandas`")
    try:
        import contextily as ctx  # assicura import
    except Exception as e:
        raise RuntimeError("contextily not available. Install with `pip install contextily`") from e

    # 2) dati
    gpath = processed_dir / 'station_accessibility_2025.geojson'
    if not gpath.exists():
        raise FileNotFoundError("station_accessibility_2025.geojson not found. Run build_datasets.py")
    g = gpd.read_file(gpath)

    ensure_dirs(out_dir / 'figures', out_dir / 'tables')

    # 3) indice intermodalità (se mancante)
    for b in (300, 500):
        s_col = f'stops_{b}m'; r_col = f'routes_{b}m'
        if s_col in g.columns and r_col in g.columns and f'idx_intermodal_{b}m' not in g.columns:
            g[f'idx_intermodal_{b}m'] = g[s_col].fillna(0) + 0.5 * g[r_col].fillna(0)

    # 4) classifica top-10
    if 'idx_intermodal_300m' in g.columns:
        top = g[['station_id','name','dist_to_stop_m','stops_300m','routes_300m','idx_intermodal_300m']].copy()
        top = top.sort_values('idx_intermodal_300m', ascending=False).head(10)
        top.to_csv(out_dir / 'tables/top10_intermodal_300m.csv', index=False)

    # 5) mappa con basemap
    g3857 = g.to_crs(3857)
    minx, miny, maxx, maxy = g3857.total_bounds
    dx, dy = maxx - minx, maxy - miny

    # figura verticale, proporzioni reali
    base_w = 9.0
    aspect = dy / dx if dx > 0 else 1.3
    base_h = max(base_w * aspect * 1.05, 10.0)

    fig, ax = plt.subplots(figsize=(base_w, base_h))

    # scegli la colonna da usare per colore
    color_col = 'idx_intermodal_300m' if 'idx_intermodal_300m' in g3857.columns else (
                'stops_300m' if 'stops_300m' in g3857.columns else None)
    if color_col is None:
        raise ValueError("No intermodality columns found (idx_intermodal_300m / stops_300m).")

    # disegna i punti (prima dei tiles o dopo è uguale; qui prima)
    g3857.plot(
        ax=ax,
        column=color_col,
        cmap='viridis',
        markersize=70,
        alpha=0.95,
        legend=True,
        legend_kwds={'label': "Intermodality index (≈300m)", "shrink": 0.7}
    )

    # extent e aspetto
    pad = 0.05
    ax.set_xlim(minx - dx*pad, maxx + dx*pad)
    ax.set_ylim(miny - dy*pad, maxy + dy*pad)
    ax.set_aspect('equal')

    # aggiungi basemap (se fallisce, fai vedere l'errore chiaro)
    # provider alternativo molto stabile
    provider = ctx.providers.CartoDB.Positron
    try:
        ctx.add_basemap(ax, crs=g3857.crs, source=provider, zoom=13, attribution_size=6)
    except Exception as e:
        raise RuntimeError(
            f"Basemap download failed (provider={provider}). "
            "Check internet connection / firewall / provider availability."
        ) from e

    ax.set_axis_off()
    ax.set_title('Bike-sharing stations — Intermodality index (≈300m)', pad=10)

    fig.tight_layout()
    fig.savefig(out_dir / 'figures/map_stations_intermodal_en.png')
    plt.close(fig)

    info("Maps complete — tables/figures saved in reports/")


# ------------------------- Main CLI -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task', nargs='?', choices=['eda','gam','maps','all'], default='all',
                    help="Task to run (eda, gam, maps, all). Default = all")
    ap.add_argument('--processed', type=Path, default=Path('data/processed'))
    ap.add_argument('--out', type=Path, default=Path('reports'))
    args = ap.parse_args()

    ensure_dirs(args.out)

    if args.task == 'all':
        run_eda(args.processed, args.out)
        run_gam(args.processed, args.out)
        run_maps(args.processed, args.out)
    elif args.task == 'eda':
        run_eda(args.processed, args.out)
    elif args.task == 'gam':
        run_gam(args.processed, args.out)
    elif args.task == 'maps':
        run_maps(args.processed, args.out)

    info('Done ✔')


if __name__ == '__main__':
    main()
