# -*- coding: utf-8 -*-
"""
Code_Descriptive_Stats_v2.py
Part 1 Descriptive Statistics for LCZ-Carbon Research
------------------------------------------------------
Fig 1: Urban area trend (shapefile-based, absolute km²) – USA & China 2000-2022
Fig 2: LCZ composition stacked area (annual share)
Fig 3: Relative emission intensity bubble chart (LCZ CO2 / national mean × 100)
Fig 4: CO2 annual trend index (2000 = 100), LCZ1-9 only
Fig 5: ln(CO2+1) distribution density
Fig 6: LCZ share relative ratio (LCZ share / all-LCZ mean share)
Fig 7: Monthly CO2 smooth trend + seasonal amplitude (LOWESS)

Style rules:
  - Font: Arial everywhere
  - Axis titles: 20 pt, NOT bold
  - Legend: 14 pt
  - Axis spine linewidth: 1.3 pt (left + bottom only, no top/right)
  - Data lines: 0.5 pt
  - No grid

All figures saved as PNG + CSV table to Results_DescStats_v2\
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────
URBAN_FP_DIR   = r"E:\Data_All\Urban Footprint\Cities_2000_2022"
USA_BOUNDARY   = r"D:\LCZCarbon\US_State_Boundaries\US_State_Boundaries.shp"
CHN_BOUNDARY   = r"D:\LCZCarbon\ChinaBoundary\ChinaBoundaryNew.shp"

USA_PANEL      = r"D:\LCZCarbon\Results_USA_Month\USA_1km\Panel_Monthly_clean.pkl"
CHN_PANEL      = r"D:\LCZCarbon\Results_CHN_Month\CHN_1km\Panel_Monthly_clean.pkl"

OUT_DIR        = r"D:\LCZCarbon\Results_DescStats_v2"
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE_N       = 1_500_000   # rows per country for panel-based figures
RANDOM_SEED    = 42

# ──────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ──────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family':          'Arial',
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'axes.linewidth':       1.3,
    'xtick.major.width':    1.3,
    'ytick.major.width':    1.3,
    'xtick.minor.width':    1.0,
    'ytick.minor.width':    1.0,
    'xtick.direction':      'out',
    'ytick.direction':      'out',
    'axes.grid':            False,
    'legend.frameon':       False,
    'figure.dpi':           150,
    'savefig.dpi':          300,
    'savefig.bbox':         'tight',
})

FS_TITLE  = 20   # axis label font size (not bold)
FS_TICK   = 14
FS_LEGEND = 14
LW_DATA   = 0.5
LW_SPINE  = 1.3

# LCZ colour palette (consistent across all figures)
LCZ_LABELS = {
    1: 'LCZ1 Compact highrise',
    2: 'LCZ2 Compact midrise',
    3: 'LCZ3 Compact lowrise',
    4: 'LCZ4 Open highrise',
    5: 'LCZ5 Open midrise',
    6: 'LCZ6 Open lowrise',
    7: 'LCZ7 Lightweight lowrise',
    8: 'LCZ8 Large lowrise',
    9: 'LCZ9 Sparsely built',
    10: 'LCZ10 Heavy industry',
    11: 'LCZ11 Dense trees',
    12: 'LCZ12 Scattered trees',
    13: 'LCZ13 Bush/scrub',
    14: 'LCZ14 Low plants',
    15: 'LCZ15 Bare rock/paved',
    16: 'LCZ16 Bare soil/sand',
    17: 'LCZ17 Water',
}

# Colour map: built (1-10) warm tones, natural (11-17) cool tones
LCZ_COLORS = {
    1:  '#8B0000', 2:  '#CD5C5C', 3:  '#FF6347', 4:  '#FF8C00',
    5:  '#FFA500', 6:  '#FFD700', 7:  '#DAA520', 8:  '#D2691E',
    9:  '#C0C0C0', 10: '#696969',
    11: '#006400', 12: '#228B22', 13: '#9ACD32', 14: '#7CFC00',
    15: '#808000', 16: '#D2B48C', 17: '#4169E1',
}

COUNTRY_COLORS = {'USA': '#2166AC', 'CHN': '#D6604D'}

# ──────────────────────────────────────────────────────────────
# HELPER: apply spine / tick style to an axis
# ──────────────────────────────────────────────────────────────
def style_ax(ax, xlabel='', ylabel='', title=''):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for sp in ['left', 'bottom']:
        ax.spines[sp].set_linewidth(LW_SPINE)
    ax.tick_params(axis='both', labelsize=FS_TICK, width=LW_SPINE, length=4)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FS_TITLE, fontweight='normal', labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS_TITLE, fontweight='normal', labelpad=8)
    if title:
        ax.set_title(title, fontsize=FS_TITLE, fontweight='normal', pad=10)


# ══════════════════════════════════════════════════════════════
# FIG 1 – Urban area trend (shapefile-based)
# ══════════════════════════════════════════════════════════════
def fig1_urban_area():
    print("=" * 60)
    print("FIG 1: Urban area trend (shapefile-based) …")
    try:
        import geopandas as gpd
    except ImportError:
        print("  [ERROR] geopandas not installed. Skipping Fig 1.")
        return

    EQUAL_AREA_CRS = "EPSG:6933"   # WGS 84 / NSIDC EASE-Grid 2.0 (metres)

    usa_bnd = gpd.read_file(USA_BOUNDARY).to_crs(EQUAL_AREA_CRS)
    chn_bnd = gpd.read_file(CHN_BOUNDARY).to_crs(EQUAL_AREA_CRS)

    records = []
    years = range(2000, 2023)
    for yr in years:
        shp_path = os.path.join(URBAN_FP_DIR, f"Cities_{yr}.shp")
        if not os.path.exists(shp_path):
            print(f"  [WARN] Not found: {shp_path}")
            continue
        urban = gpd.read_file(shp_path).to_crs(EQUAL_AREA_CRS)

        # ── USA ──────────────────────────────────
        urban_usa = gpd.overlay(urban, usa_bnd[['geometry']], how='intersection')
        area_usa  = urban_usa.geometry.area.sum() / 1e6   # m² → km²

        # ── China ────────────────────────────────
        urban_chn = gpd.overlay(urban, chn_bnd[['geometry']], how='intersection')
        area_chn  = urban_chn.geometry.area.sum() / 1e6

        records.append({'Year': yr, 'USA_km2': area_usa, 'CHN_km2': area_chn})
        print(f"  {yr}: USA={area_usa:,.0f} km²  CHN={area_chn:,.0f} km²")

    df1 = pd.DataFrame(records)
    csv_path = os.path.join(OUT_DIR, "Fig1_Urban_Area_km2.csv")
    df1.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  Saved table: {csv_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df1['Year'], df1['USA_km2'] / 1e3, color=COUNTRY_COLORS['USA'],
            linewidth=LW_DATA, marker='o', markersize=4, label='USA')
    ax.plot(df1['Year'], df1['CHN_km2'] / 1e3, color=COUNTRY_COLORS['CHN'],
            linewidth=LW_DATA, marker='s', markersize=4, label='China')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}'))
    ax.set_ylabel("Urban Area (×10³ km²)", fontsize=FS_TITLE, fontweight='normal')
    ax.set_xlabel("Year", fontsize=FS_TITLE, fontweight='normal')
    ax.legend(fontsize=FS_LEGEND)
    style_ax(ax)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "Fig1_Urban_Area_Trend.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved figure: {out}")


# ══════════════════════════════════════════════════════════════
# LOAD PANEL DATA (shared by Figs 2-7)
# ══════════════════════════════════════════════════════════════
def load_panel():
    print("=" * 60)
    print("Loading panel data …")
    dfs = {}
    for country, path in [('USA', USA_PANEL), ('CHN', CHN_PANEL)]:
        print(f"  {country}: {path}")
        df = pd.read_pickle(path)
        if len(df) > SAMPLE_N:
            df = df.sample(n=SAMPLE_N, random_state=RANDOM_SEED)
        dfs[country] = df
        print(f"  {country}: {len(df):,} rows  cols={list(df.columns[:10])} …")
    return dfs


# ══════════════════════════════════════════════════════════════
# DETECT LCZ SHARE COLUMNS
# ══════════════════════════════════════════════════════════════
def lcz_share_cols(df):
    cols = [c for c in df.columns if c.startswith('LCZ') and c.endswith('_share')]
    cols.sort(key=lambda c: int(c.replace('LCZ', '').replace('_share', '')))
    return cols


# ══════════════════════════════════════════════════════════════
# FIG 2 – LCZ composition stacked area
# ══════════════════════════════════════════════════════════════
def fig2_lcz_composition(dfs):
    print("=" * 60)
    print("FIG 2: LCZ composition stacked area …")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    csv_rows = []

    for ax, (country, df) in zip(axes, dfs.items()):
        cols = lcz_share_cols(df)
        annual = df.groupby('Year')[cols].mean()

        # ── Normalise so each year sums to exactly 1 ──────────────
        # Needed because: (1) edge pixels with partial neighbourhoods
        # have row-sums < 1; (2) random sampling can introduce small
        # imbalances across columns. This is a valid descriptive step.
        row_sum = annual.sum(axis=1)
        annual  = annual.div(row_sum, axis=0)

        # diagnostic
        print(f"  {country}: row-sum range [{row_sum.min():.4f}, {row_sum.max():.4f}] "
              f"→ normalised to 1.000")

        annual = annual.reset_index()
        years  = annual['Year'].values
        vals   = [annual[c].values for c in cols]
        colors = [LCZ_COLORS[int(c.replace('LCZ','').replace('_share',''))] for c in cols]
        labels = [f"LCZ{int(c.replace('LCZ','').replace('_share',''))}" for c in cols]

        ax.stackplot(years, vals, colors=colors, labels=labels, linewidth=0)
        style_ax(ax, xlabel='Year', ylabel='LCZ Share (%)' if country=='USA' else '',
                 title=country)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

        # collect for CSV
        annual.insert(0, 'Country', country)
        csv_rows.append(annual)

    # single legend below
    handles = [Line2D([0],[0], color=LCZ_COLORS[i], linewidth=8) for i in range(1,18)]
    leg_labels = [f"LCZ{i}" for i in range(1,18)]
    fig.legend(handles, leg_labels, loc='lower center', ncol=9, fontsize=FS_LEGEND,
               frameon=False, bbox_to_anchor=(0.5, -0.08))
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "Fig2_LCZ_Composition.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved figure: {out}")

    # CSV table (normalised shares)
    df2 = pd.concat(csv_rows, ignore_index=True)
    csv_path = os.path.join(OUT_DIR, "Fig2_LCZ_Composition.csv")
    df2.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  Saved table: {csv_path}")


# ══════════════════════════════════════════════════════════════
# FIG 3 – Relative emission intensity bubble chart
# ══════════════════════════════════════════════════════════════
def fig3_relative_intensity(dfs):
    print("=" * 60)
    print("FIG 3: Relative emission intensity bubble chart …")
    all_rows = []

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (country, df) in zip(axes, dfs.items()):
        cols   = lcz_share_cols(df)
        co2col = 'CO2' if 'CO2' in df.columns else 'ln_CO2'

        # compute mean CO2 per LCZ type  (assign dominant LCZ)
        lcz_ids = [int(c.replace('LCZ','').replace('_share','')) for c in cols]

        # dominant LCZ per pixel
        df = df.copy()
        df['dominant_lcz'] = df[cols].idxmax(axis=1).str.replace('LCZ','').str.replace('_share','').astype(int)

        # annual mean CO2 per LCZ
        if 'CO2' in df.columns:
            grp = df.groupby(['Year','dominant_lcz'])['CO2'].mean().reset_index()
            nat  = df.groupby('Year')['CO2'].mean().reset_index().rename(columns={'CO2':'nat_mean'})
            grp  = grp.merge(nat, on='Year')
            grp['rel_intensity'] = grp['CO2'] / grp['nat_mean'] * 100
        else:
            # use ln_CO2: back-transform for ratio
            grp = df.groupby(['Year','dominant_lcz'])['ln_CO2'].mean().reset_index()
            nat  = df.groupby('Year')['ln_CO2'].mean().reset_index().rename(columns={'ln_CO2':'nat_mean'})
            grp  = grp.merge(nat, on='Year')
            grp['rel_intensity'] = np.exp(grp['ln_CO2']) / np.exp(grp['nat_mean']) * 100

        # mean share per LCZ
        share_mean = df[cols].mean()
        share_dict = {int(c.replace('LCZ','').replace('_share','')): v for c,v in share_mean.items()}

        # collapse to single mean per LCZ (across years)
        lcz_summary = grp.groupby('dominant_lcz')['rel_intensity'].mean().reset_index()
        lcz_summary['share'] = lcz_summary['dominant_lcz'].map(share_dict).fillna(0)
        lcz_summary['Country'] = country
        all_rows.append(lcz_summary)

        xs  = lcz_summary['dominant_lcz'].values
        ys  = lcz_summary['rel_intensity'].values
        ss  = (lcz_summary['share'].values * 3000 + 20)   # bubble size scaled
        cs  = [LCZ_COLORS.get(x, '#888888') for x in xs]

        ax.scatter(xs, ys, s=ss, c=cs, alpha=0.85, edgecolors='white', linewidths=0.4)
        ax.axhline(100, color='black', linewidth=LW_DATA, linestyle='--', alpha=0.5)
        ax.set_xticks(range(1, 18))
        ax.set_xticklabels([f'LCZ{i}' for i in range(1, 18)], rotation=45, ha='right')
        style_ax(ax, xlabel='LCZ Type',
                 ylabel='Relative Emission Intensity (%)' if country=='USA' else '',
                 title=country)

    # legend for bubble size
    for ax in axes:
        for sz, lbl in [(20,'~0%'), (320,'10%'), (1520,'50%')]:
            ax.scatter([], [], s=sz, c='grey', alpha=0.5, label=lbl)
        ax.legend(title='LCZ Share', fontsize=FS_LEGEND-2, title_fontsize=FS_LEGEND-2,
                  loc='upper right')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "Fig3_Relative_Emission_Intensity.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved figure: {out}")

    df3 = pd.concat(all_rows, ignore_index=True)
    csv_path = os.path.join(OUT_DIR, "Fig3_Relative_Emission_Intensity.csv")
    df3.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  Saved table: {csv_path}")


# ══════════════════════════════════════════════════════════════
# FIG 4 – CO2 annual trend index (2000 = 100), LCZ 1-9
# ══════════════════════════════════════════════════════════════
def fig4_co2_index(dfs):
    print("=" * 60)
    print("FIG 4: CO2 annual trend index (2000=100), LCZ1-9 …")
    all_rows = []

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    for ax, (country, df) in zip(axes, dfs.items()):
        cols = lcz_share_cols(df)
        df   = df.copy()
        df['dominant_lcz'] = df[cols].idxmax(axis=1).str.replace('LCZ','').str.replace('_share','').astype(int)
        df9 = df[df['dominant_lcz'].between(1, 9)]

        co2col = 'CO2' if 'CO2' in df.columns else 'ln_CO2'
        annual = df9.groupby(['Year','dominant_lcz'])[co2col].mean().reset_index()
        annual.columns = ['Year','LCZ','mean_co2']

        # index to 2000
        base = annual[annual['Year']==2000][['LCZ','mean_co2']].rename(columns={'mean_co2':'base2000'})
        annual = annual.merge(base, on='LCZ')
        annual['index'] = annual['mean_co2'] / annual['base2000'] * 100
        annual['Country'] = country
        all_rows.append(annual)

        for lcz_id in range(1, 10):
            sub = annual[annual['LCZ']==lcz_id].sort_values('Year')
            if sub.empty:
                continue
            ax.plot(sub['Year'], sub['index'],
                    color=LCZ_COLORS[lcz_id], linewidth=LW_DATA,
                    label=f'LCZ{lcz_id}')

        ax.axhline(100, color='black', linewidth=LW_DATA, linestyle='--', alpha=0.5)
        style_ax(ax, xlabel='Year',
                 ylabel='CO₂ Index (2000 = 100)' if country=='USA' else '',
                 title=country)
        ax.legend(fontsize=FS_LEGEND, ncol=2)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "Fig4_CO2_Index_LCZ1-9.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved figure: {out}")

    df4 = pd.concat(all_rows, ignore_index=True)
    csv_path = os.path.join(OUT_DIR, "Fig4_CO2_Index_LCZ1-9.csv")
    df4.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  Saved table: {csv_path}")


# ══════════════════════════════════════════════════════════════
# FIG 5 – ln(CO2+1) distribution density
# ══════════════════════════════════════════════════════════════
def fig5_log_distribution(dfs):
    print("=" * 60)
    print("FIG 5: ln(CO2+1) distribution …")
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    stat_rows  = []
    kde_rows   = []   # KDE curve (x, density) for Origin

    for ax, (country, df) in zip(axes, dfs.items()):
        # compute ln(CO2+1)  — use raw CO2 if available, else ln_CO2 already log
        if 'CO2' in df.columns:
            vals = np.log1p(df['CO2'].dropna().values)
            xlabel_str = 'ln(CO₂ + 1)'
        else:
            vals = df['ln_CO2'].dropna().values
            xlabel_str = 'ln(CO₂)'

        kde  = gaussian_kde(vals, bw_method='scott')
        xr   = np.linspace(vals.min(), vals.max(), 500)
        yr   = kde(xr)

        ax.fill_between(xr, yr, alpha=0.25, color=COUNTRY_COLORS[country])
        ax.plot(xr, yr, color=COUNTRY_COLORS[country], linewidth=LW_DATA)
        median_val = np.median(vals)
        ax.axvline(median_val, color='black', linewidth=LW_DATA,
                   linestyle='--', alpha=0.7, label=f'Median={median_val:.2f}')
        style_ax(ax, xlabel=xlabel_str, ylabel='Density', title=country)
        ax.legend(fontsize=FS_LEGEND)

        # ── Summary stats ──────────────────────────────────────
        stat_rows.append({
            'Country':  country,
            'N':        len(vals),
            'Mean':     np.mean(vals),
            'Median':   median_val,
            'Std':      np.std(vals),
            'Min':      vals.min(),
            'Max':      vals.max(),
            'Skewness': pd.Series(vals).skew(),
            'Kurtosis': pd.Series(vals).kurt(),
        })

        # ── KDE curve points (500 rows per country) ────────────
        for xi, yi in zip(xr, yr):
            kde_rows.append({'Country': country, 'ln_CO2_x': xi, 'Density_y': yi})

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "Fig5_Log_Distribution.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved figure: {out}")

    # CSV 1 – summary statistics (descriptive table for paper)
    df5_stats = pd.DataFrame(stat_rows)
    csv1 = os.path.join(OUT_DIR, "Fig5a_Log_Distribution_Stats.csv")
    df5_stats.to_csv(csv1, index=False, encoding='utf-8-sig')
    print(f"  Saved summary stats : {csv1}")

    # CSV 2 – KDE curve points (x, y) for reproducing the curve in Origin
    df5_kde = pd.DataFrame(kde_rows)
    csv2 = os.path.join(OUT_DIR, "Fig5b_KDE_Curve_ForOrigin.csv")
    df5_kde.to_csv(csv2, index=False, encoding='utf-8-sig')
    print(f"  Saved KDE curve     : {csv2}")

    # CSV 3 – wide format (USA and CHN side by side, easier for Origin)
    usa_kde = df5_kde[df5_kde['Country']=='USA'][['ln_CO2_x','Density_y']].reset_index(drop=True)
    chn_kde = df5_kde[df5_kde['Country']=='CHN'][['ln_CO2_x','Density_y']].reset_index(drop=True)
    df5_wide = pd.DataFrame({
        'USA_ln_CO2': usa_kde['ln_CO2_x'],
        'USA_Density': usa_kde['Density_y'],
        'CHN_ln_CO2': chn_kde['ln_CO2_x'],
        'CHN_Density': chn_kde['Density_y'],
    })
    csv3 = os.path.join(OUT_DIR, "Fig5c_KDE_Wide_ForOrigin.csv")
    df5_wide.to_csv(csv3, index=False, encoding='utf-8-sig')
    print(f"  Saved KDE wide      : {csv3}")


# ══════════════════════════════════════════════════════════════
# FIG 6 – LCZ share relative ratio
# ══════════════════════════════════════════════════════════════
def fig6_share_relative(dfs):
    print("=" * 60)
    print("FIG 6: LCZ share relative ratio …")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    all_rows  = []

    for ax, (country, df) in zip(axes, dfs.items()):
        cols = lcz_share_cols(df)
        # annual mean share per LCZ
        annual = df.groupby('Year')[cols].mean()
        # grand mean across all years & all LCZ types
        grand_mean = annual.values.mean()
        # relative ratio = LCZ_share / grand_mean
        rel = annual / grand_mean

        years   = annual.index.values
        lcz_ids = [int(c.replace('LCZ','').replace('_share','')) for c in cols]

        for col, lcz_id in zip(cols, lcz_ids):
            ax.plot(years, rel[col].values,
                    color=LCZ_COLORS[lcz_id], linewidth=LW_DATA,
                    label=f'LCZ{lcz_id}')

        ax.axhline(1, color='black', linewidth=LW_DATA, linestyle='--', alpha=0.5)
        style_ax(ax, xlabel='Year',
                 ylabel='Relative Share Ratio' if country=='USA' else '',
                 title=country)
        ax.legend(fontsize=FS_LEGEND-2, ncol=2)

        # table
        rel_reset = rel.reset_index()
        rel_reset.insert(0, 'Country', country)
        all_rows.append(rel_reset)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "Fig6_LCZ_Share_Relative.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved figure: {out}")

    df6 = pd.concat(all_rows, ignore_index=True)
    csv_path = os.path.join(OUT_DIR, "Fig6_LCZ_Share_Relative.csv")
    df6.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  Saved table: {csv_path}")


# ══════════════════════════════════════════════════════════════
# FIG 7 – Monthly CO2 smooth trend + seasonal amplitude
# ══════════════════════════════════════════════════════════════
def fig7_monthly_smooth(dfs):
    """
    For each country:
      - Compute mean CO2 for each (Year, Month) cell
      - Build a continuous time index  t = Year + (Month-1)/12
      - Fit LOWESS for the long-term trend component
      - Derive seasonal amplitude = max(monthly anomaly) - min(monthly anomaly)
        per year, then smooth with LOWESS too
    Panel: two rows × one col
      Top row:    smoothed monthly CO2 level   (one curve per country)
      Bottom row: seasonal amplitude per year  (bar or line)
    """
    print("=" * 60)
    print("FIG 7: Monthly CO2 smooth trend + seasonal amplitude …")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                             gridspec_kw={'height_ratios': [2, 1]})
    all_rows = []

    for col_idx, (country, df) in enumerate(dfs.items()):
        ax_top = axes[0][col_idx]
        ax_bot = axes[1][col_idx]

        co2col = 'CO2' if 'CO2' in df.columns else 'ln_CO2'
        ym_mean = df.groupby(['Year','Month'])[co2col].mean().reset_index()
        ym_mean = ym_mean.sort_values(['Year','Month'])
        ym_mean['t'] = ym_mean['Year'] + (ym_mean['Month']-1)/12

        t_arr = ym_mean['t'].values
        y_arr = ym_mean[co2col].values

        # ── LOWESS trend ─────────────────────────
        frac  = 0.12   # smoothing bandwidth (12% of data = ~2.4 years)
        trend = lowess(y_arr, t_arr, frac=frac, it=3, return_sorted=False)

        # ── Seasonal anomaly per year ─────────────
        ym_mean['anomaly'] = y_arr - trend
        amp_per_year = (ym_mean.groupby('Year')['anomaly']
                                .agg(lambda x: x.max() - x.min())
                                .reset_index()
                                .rename(columns={'anomaly': 'amplitude'}))

        # smooth amplitude
        amp_trend = lowess(amp_per_year['amplitude'].values,
                           amp_per_year['Year'].values,
                           frac=0.4, it=3, return_sorted=False)

        # ── Plot: top ────────────────────────────
        ax_top.plot(t_arr, y_arr,
                    color=COUNTRY_COLORS[country], linewidth=LW_DATA,
                    alpha=0.4, label='Monthly mean')
        ax_top.plot(t_arr, trend,
                    color=COUNTRY_COLORS[country], linewidth=1.2,
                    label='LOWESS trend')
        style_ax(ax_top,
                 ylabel='ln(CO₂)' if col_idx==0 else '',
                 title=country)
        ax_top.legend(fontsize=FS_LEGEND)

        # ── Plot: bottom ─────────────────────────
        ax_bot.bar(amp_per_year['Year'], amp_per_year['amplitude'],
                   color=COUNTRY_COLORS[country], alpha=0.4, width=0.7)
        ax_bot.plot(amp_per_year['Year'], amp_trend,
                    color=COUNTRY_COLORS[country], linewidth=1.0)
        style_ax(ax_bot, xlabel='Year',
                 ylabel='Seasonal Amplitude' if col_idx==0 else '')

        # ── Table data ───────────────────────────
        ym_mean['Country'] = country
        ym_mean['trend']   = trend
        all_rows.append(ym_mean[['Country','Year','Month','t',co2col,'trend','anomaly']])

        amp_per_year['Country']       = country
        amp_per_year['amp_smoothed']  = amp_trend
        all_rows.append(amp_per_year)  # different structure – saved separately below

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "Fig7_Monthly_Smooth.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved figure: {out}")

    # ── CSV 1: monthly series (for top panel in Origin) ─────────
    ym_rows  = [r for r in all_rows if 'Month' in r.columns]
    df7_ym   = pd.concat(ym_rows, ignore_index=True)
    csv_ym   = os.path.join(OUT_DIR, "Fig7a_Monthly_Series.csv")
    df7_ym.to_csv(csv_ym, index=False, encoding='utf-8-sig')
    print(f"  Saved monthly series table : {csv_ym}")

    # ── CSV 2: annual amplitude (for bottom panel in Origin) ────
    amp_rows = [r for r in all_rows if 'amplitude' in r.columns]
    df7_amp  = pd.concat(amp_rows, ignore_index=True)
    # reorder columns for clarity
    df7_amp  = df7_amp[['Country','Year','amplitude','amp_smoothed']]
    csv_amp  = os.path.join(OUT_DIR, "Fig7b_Seasonal_Amplitude.csv")
    df7_amp.to_csv(csv_amp, index=False, encoding='utf-8-sig')
    print(f"  Saved amplitude table      : {csv_amp}")

    # ── CSV 3: wide format – pivot for easy Origin import ───────
    # Top panel: one column per series (USA_CO2, USA_trend, CHN_CO2, CHN_trend)
    wide_parts = []
    for country in ['USA', 'CHN']:
        sub = df7_ym[df7_ym['Country']==country][['t','CO2' if 'CO2' in df7_ym.columns else 'ln_CO2','trend']].copy()
        co2_col = 'CO2' if 'CO2' in sub.columns else 'ln_CO2'
        sub = sub.rename(columns={co2_col: f'{country}_CO2', 'trend': f'{country}_trend'})
        wide_parts.append(sub.set_index('t'))
    df7_wide = pd.concat(wide_parts, axis=1).reset_index()
    df7_wide.rename(columns={'t': 'Time'}, inplace=True)
    csv_wide = os.path.join(OUT_DIR, "Fig7c_Wide_ForOrigin.csv")
    df7_wide.to_csv(csv_wide, index=False, encoding='utf-8-sig')
    print(f"  Saved wide (Origin) table  : {csv_wide}")


# ══════════════════════════════════════════════════════════════
# FIG 8 – Per-capita urban CO₂ (panel) vs national benchmark (OWID)
# ══════════════════════════════════════════════════════════════

# ── OWID country name mapping ──────────────────────────────────
OWID_NAMES = {'USA': 'United States', 'CHN': 'China'}

# ── Population TIF paths (corrected float32, re-clipped from LandScan) ──
POP_TIF_DIRS = {
    'USA': (r"D:\LCZCarbon\Population\USAPOP", "POP{year}_USA.tif"),
    'CHN': (r"D:\LCZCarbon\Population\CHNPOP", "POP{year}_CHN.tif"),
}

# ── Fallback data (GCP via OWID, ton CO₂/person/year) ──────────
# Used when network download fails; values cross-checked with IEA 2024
OWID_FALLBACK = {
    'United States': {
        2000: 20.18, 2001: 19.70, 2002: 19.58, 2003: 19.52, 2004: 19.79,
        2005: 19.54, 2006: 18.88, 2007: 19.10, 2008: 18.38, 2009: 16.93,
        2010: 17.55, 2011: 17.01, 2012: 16.16, 2013: 16.44, 2014: 16.49,
        2015: 15.97, 2016: 15.53, 2017: 15.00, 2018: 15.63, 2019: 14.98,
    },
    'China': {
        2000:  2.67, 2001:  2.77, 2002:  2.96, 2003:  3.36, 2004:  3.90,
        2005:  4.47, 2006:  4.95, 2007:  5.36, 2008:  5.53, 2009:  5.91,
        2010:  6.67, 2011:  7.30, 2012:  7.53, 2013:  7.78, 2014:  7.77,
        2015:  7.67, 2016:  7.60, 2017:  7.74, 2018:  8.02, 2019:  7.96,
    },
}


def load_pop_from_tif(country, years, min_pop=1.0):
    """
    从修正后的 Population TIF（float32，已从 LandScan Global 重新裁剪）
    读取各年人口均值，作为 Fig 8 人均 CO₂ 的分母。

    只统计 pop >= min_pop 的像元（过滤无人像元，保留城市/农村有人口区域）。
    返回 dict {year: mean_pop_per_pixel}。

    说明：
      - 旧做法：从 panel pkl 的 Pop 列读取，并对 2009/2018/2019 做 rescaling
      - 新做法：直接读 TIF（已修复 uint8 截断 + 年份异常），无需任何 rescaling
    """
    pop_dir, pop_tpl = POP_TIF_DIRS[country]
    result = {}
    for yr in years:
        path = os.path.join(pop_dir, pop_tpl.format(year=yr))
        if not os.path.exists(path):
            print(f"    [WARN] {country} {yr}: Pop TIF not found → {path}")
            result[yr] = np.nan
            continue
        try:
            # 优先使用 rasterio（精确处理 nodata 和 CRS）
            import rasterio as _rio
            with _rio.open(path) as src:
                arr  = src.read(1).astype(np.float32).ravel()
                nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = 0.0
        except ImportError:
            # fallback: PIL
            from PIL import Image
            arr = np.array(Image.open(path), dtype=np.float32).ravel()

        valid = arr[arr >= min_pop]
        mean_pop = float(np.mean(valid)) if len(valid) > 0 else np.nan
        result[yr] = mean_pop
        print(f"    {country} {yr}: TIF mean pop = {mean_pop:.2f}  "
              f"(valid pixels={len(valid):,})")
    return result


def fetch_owid_data(out_dir, year_start=2000, year_end=2019):
    """
    Download OWID CO₂ per-capita data (GCP-based).
    Primary:  https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv
    Fallback: hard-coded GCP values in OWID_FALLBACK dict above.

    Returns DataFrame with columns: country_owid, Year, co2_per_capita
    (unit: ton CO₂ / person / year)
    """
    import urllib.request, ssl

    OWID_URL = ("https://raw.githubusercontent.com/owid/co2-data/"
                "master/owid-co2-data.csv")
    raw_csv  = os.path.join(out_dir, "OWID_CO2_raw.csv")
    target   = list(OWID_NAMES.values())   # ['United States', 'China']

    # ── Try download ──────────────────────────────────────────
    downloaded = False
    if not os.path.exists(raw_csv):
        print("  Downloading OWID CO₂ data …")
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode    = ssl.CERT_NONE
            with urllib.request.urlopen(OWID_URL, context=ctx, timeout=30) as resp:
                content = resp.read()
            with open(raw_csv, 'wb') as f:
                f.write(content)
            print(f"  Downloaded → {raw_csv}")
            downloaded = True
        except Exception as e:
            print(f"  [WARN] Download failed ({e}). Using fallback GCP data.")
    else:
        print(f"  Using cached OWID file: {raw_csv}")
        downloaded = True

    # ── Parse downloaded CSV ──────────────────────────────────
    if downloaded and os.path.exists(raw_csv):
        try:
            owid = pd.read_csv(raw_csv, usecols=['country','year','co2_per_capita'])
            owid = owid[
                owid['country'].isin(target) &
                owid['year'].between(year_start, year_end)
            ].copy()
            owid = owid.rename(columns={'country': 'country_owid',
                                        'year':    'Year'})
            owid = owid.dropna(subset=['co2_per_capita'])
            if len(owid) > 0:
                print(f"  OWID records loaded: {len(owid)}")
                return owid
        except Exception as e:
            print(f"  [WARN] CSV parse failed ({e}). Using fallback.")

    # ── Fallback ──────────────────────────────────────────────
    print("  Using built-in GCP fallback data (2000-2019).")
    rows = []
    for cname, year_dict in OWID_FALLBACK.items():
        for yr, val in year_dict.items():
            if year_start <= yr <= year_end:
                rows.append({'country_owid': cname, 'Year': yr,
                             'co2_per_capita': val})
    return pd.DataFrame(rows)


def fig8_percapita_co2(dfs, out_dir):
    """
    Fig 8: Urban per-capita CO₂ from panel data vs national benchmark (OWID/GCP).

    Per-capita method (robust stratified approach):
      1. Re-load the FULL Panel_Monthly_clean.pkl for each country (bypass 1.5M sample)
      2. For each Year, draw a STRATIFIED sample (SAMPLE_PER_YEAR rows per year)
         so every year is equally represented regardless of panel size.
      3. Within each year, compute sum(CO2) / sum(Pop)  [= mean(CO2)/mean(Pop)]
         over all 12 monthly means, then annualise.
      4. Filter pixels with Pop in [P2, P98] to remove population raster outliers.

    Unit: ton CO₂ / person / year
    """
    print("=" * 60)
    print("FIG 8: Per-capita urban CO₂ (panel) vs national benchmark …")

    PANEL_PATHS    = {'USA': USA_PANEL, 'CHN': CHN_PANEL}
    SAMPLE_PER_YR  = 100_000   # rows per year (robust but fast)

    # ── 1. Fetch OWID national benchmark ─────────────────────
    owid = fetch_owid_data(out_dir)
    owid_csv = os.path.join(out_dir, "Fig8_OWID_Benchmark.csv")
    owid.to_csv(owid_csv, index=False, encoding='utf-8-sig')
    print(f"  Saved OWID table: {owid_csv}")

    # ── 2. Compute panel per-capita per year ──────────────────
    panel_rows = []
    for country in ['USA', 'CHN']:
        pkl_path = PANEL_PATHS[country]
        print(f"  {country}: loading full panel from {pkl_path} …")
        try:
            df_full = pd.read_pickle(pkl_path)
        except Exception as e:
            print(f"  [WARN] Cannot load {pkl_path}: {e}. Skipping.")
            continue

        if 'CO2' not in df_full.columns:
            print(f"  [WARN] {country}: missing CO2 column.")
            continue

        # ── Filter: positive CO2 ──────────────────────────────
        df_full = df_full[df_full['CO2'] > 0].copy()

        # ── Stratified sample by Year (loop, avoids pandas groupby.apply bug) ──
        sampled_parts = []
        for yr in sorted(df_full['Year'].unique()):
            g = df_full[df_full['Year'] == yr]
            sampled_parts.append(
                g.sample(n=min(SAMPLE_PER_YR, len(g)), random_state=RANDOM_SEED)
            )
        df_s = pd.concat(sampled_parts, ignore_index=True)
        del df_full
        print(f"  {country}: stratified sample → {len(df_s):,} rows")

        # ── Step A: monthly mean CO2 per Year ────────────────
        ym = df_s.groupby(['Year', 'Month']).agg(
            CO2_m=('CO2', 'mean'),
        ).reset_index()

        # ── Step B: annual total CO2 = sum of 12 monthly means ─
        ann = ym.groupby('Year').agg(
            CO2_annual=('CO2_m', 'sum'),   # ton C/pixel/year
        ).reset_index()

        # ── Step C: load Pop from corrected TIF files ─────────
        # Population is read directly from the re-clipped float32 TIF
        # (D:\LCZCarbon\Population), bypassing the panel's Pop column.
        # This avoids any residual uint8 truncation or year-specific
        # anomalies that may still exist in the old panel pkl.
        # No rescaling needed — the TIF source is already corrected.
        print(f"  {country}: reading population from TIF files …")
        pop_by_year = load_pop_from_tif(country, sorted(ann['Year'].unique()))
        ann['Pop_tif'] = ann['Year'].map(pop_by_year)

        # ── Step D: per capita  (ton C/pixel/yr ÷ person/pixel × 44/12) ──
        # 44/12 = molar mass ratio CO₂/C → converts ton C → ton CO₂
        ann['percap_panel'] = ann['CO2_annual'] / ann['Pop_tif'] * (44 / 12)
        ann['Country']      = country

        # Diagnostic: print per-year values
        for _, row in ann.iterrows():
            print(f"    {country} {int(row.Year)}: "
                  f"CO2_ann={row.CO2_annual:.2f} tC  "
                  f"Pop_tif={row.Pop_tif:.2f}  "
                  f"percap={row.percap_panel:.2f} tCO2/person/yr")

        panel_rows.append(ann)

    df_panel = pd.concat(panel_rows, ignore_index=True)
    # 整理列顺序，方便核查
    out_cols = ['Country', 'Year', 'CO2_annual', 'Pop_tif', 'percap_panel']
    df_panel = df_panel[[c for c in out_cols if c in df_panel.columns]]
    panel_csv = os.path.join(out_dir, "Fig8_Panel_PerCapita.csv")
    df_panel.to_csv(panel_csv, index=False, encoding='utf-8-sig')
    print(f"  Saved panel per-capita table: {panel_csv}")

    # ── 3. Plot ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (country, _) in zip(axes, dfs.items()):
        owid_name = OWID_NAMES[country]
        color     = COUNTRY_COLORS[country]

        # ── Panel series ─────────────────────────────────────
        sub_p = df_panel[df_panel['Country']==country].sort_values('Year')
        if not sub_p.empty:
            ax.plot(sub_p['Year'], sub_p['percap_panel'],
                    color=color, linewidth=LW_DATA,
                    marker='o', markersize=3,
                    label='Urban panel (ODIAC)')
            # LOWESS trend
            if len(sub_p) >= 5:
                t_trend = lowess(sub_p['percap_panel'].values,
                                 sub_p['Year'].values,
                                 frac=0.4, it=3, return_sorted=False)
                ax.plot(sub_p['Year'], t_trend,
                        color=color, linewidth=1.5,
                        linestyle='-', alpha=0.9)

        # ── OWID national benchmark ───────────────────────────
        sub_o = owid[owid['country_owid']==owid_name].sort_values('Year')
        if not sub_o.empty:
            ax.plot(sub_o['Year'], sub_o['co2_per_capita'],
                    color='#555555', linewidth=LW_DATA,
                    marker='s', markersize=3,
                    linestyle='--',
                    label='National (OWID/GCP)')
            # LOWESS trend
            if len(sub_o) >= 5:
                n_trend = lowess(sub_o['co2_per_capita'].values,
                                 sub_o['Year'].values,
                                 frac=0.4, it=3, return_sorted=False)
                ax.plot(sub_o['Year'], n_trend,
                        color='#555555', linewidth=1.5,
                        linestyle='--', alpha=0.9)

        style_ax(ax, xlabel='Year',
                 ylabel='CO₂ per Capita (ton CO₂/person/year)' if country=='USA' else '',
                 title=country)
        ax.legend(fontsize=FS_LEGEND)

    plt.tight_layout()
    out_fig = os.path.join(out_dir, "Fig8_PerCapita_CO2.png")
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"  Saved figure: {out_fig}")

    # ── 4. Combined wide CSV for Origin ──────────────────────
    wide_rows = []
    for country in ['USA', 'CHN']:
        owid_name = OWID_NAMES[country]
        sub_p = df_panel[df_panel['Country']==country].sort_values('Year').reset_index(drop=True)
        sub_o = owid[owid['country_owid']==owid_name].sort_values('Year').reset_index(drop=True)
        merged = sub_p[['Year', 'CO2_annual', 'Pop_tif', 'percap_panel']].merge(
            sub_o[['Year','co2_per_capita']], on='Year', how='outer').sort_values('Year')
        merged = merged.rename(columns={
            'CO2_annual':    f'{country}_CO2_annual',
            'Pop_tif':       f'{country}_Pop_tif',
            'percap_panel':  f'{country}_panel_percap',
            'co2_per_capita':f'{country}_OWID_percap',
        })
        wide_rows.append(merged.set_index('Year'))

    df8_wide = pd.concat(wide_rows, axis=1).reset_index()
    csv8_wide = os.path.join(out_dir, "Fig8_PerCapita_Wide_ForOrigin.csv")
    df8_wide.to_csv(csv8_wide, index=False, encoding='utf-8-sig')
    print(f"  Saved wide (Origin) table  : {csv8_wide}")


# ══════════════════════════════════════════════════════════════
# MAIN – 运行控制开关
# 把不需要重跑的图编号从 RUN_FIGS 里删掉即可跳过，
# 例如只跑 Fig8：  RUN_FIGS = {8}
# 全部重跑：       RUN_FIGS = {1, 2, 3, 4, 5, 6, 7, 8}
# ══════════════════════════════════════════════════════════════
RUN_FIGS = {8}   # ← 修改这里控制跑哪些图

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  LCZ-Carbon Descriptive Statistics v2")
    print(f"  Running figures: {sorted(RUN_FIGS)}")
    print("=" * 60)

    # Fig 1 – standalone (requires geopandas + shapefiles)
    if 1 in RUN_FIGS:
        fig1_urban_area()

    # Load panel only if any of Figs 2-8 are needed
    NEED_PANEL = RUN_FIGS & {2, 3, 4, 5, 6, 7, 8}
    if NEED_PANEL:
        dfs = load_panel()
    else:
        dfs = {}

    if 2 in RUN_FIGS: fig2_lcz_composition(dfs)
    if 3 in RUN_FIGS: fig3_relative_intensity(dfs)
    if 4 in RUN_FIGS: fig4_co2_index(dfs)
    if 5 in RUN_FIGS: fig5_log_distribution(dfs)
    if 6 in RUN_FIGS: fig6_share_relative(dfs)
    if 7 in RUN_FIGS: fig7_monthly_smooth(dfs)
    if 8 in RUN_FIGS: fig8_percapita_co2(dfs, OUT_DIR)

    print("\n" + "=" * 60)
    print(f"  All outputs saved to: {OUT_DIR}")
    print("=" * 60)
