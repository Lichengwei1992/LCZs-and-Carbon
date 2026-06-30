# -*- coding: utf-8 -*-
# Code_LCZ_CO2_Dynamics.py
# =============================================================
# Three analytical frameworks for LCZ-CO2 temporal dynamics
# (USA & China, 2000-2019)
#
# Method 1 - LOWESS Trend + Slope Heatmap
# Method 2 - Convergence Analysis (sigma + beta)
# Method 3 - Structural Break Detection (Chow optimal single-break)
#
# Style: Arial, axis-title=18, tick=16, legend=14, linewidth=1pt,
#        USA sky-blue/dark-blue, CHN light-pink/dark-red,
#        x-ticks 2000/2005/2010/2015/2020, left+bottom spines only
# =============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

warnings.filterwarnings("ignore")

# ─────────────────────────── Paths ───────────────────────────
USA_PANEL = r"D:\LCZCarbon\Results_USA_Month\USA_1km\Panel_Monthly_clean.pkl"
CHN_PANEL = r"D:\LCZCarbon\Results_CHN_Month\CHN_1km\Panel_Monthly_clean.pkl"
OUT_DIR   = r"D:\LCZCarbon\Results_DescStats_v2"
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE_PER_YR = 50_000
RANDOM_SEED   = 42
YEARS         = list(range(2000, 2020))
MIN_OBS_LCZ   = 200

# ─────────────────── Style (Origin-matching) ─────────────────
matplotlib.rcParams.update({
    'font.family':         'Arial',
    'axes.spines.top':     False,
    'axes.spines.right':   False,
    'axes.linewidth':      1.0,
    'xtick.major.width':   1.0,
    'ytick.major.width':   1.0,
    'xtick.minor.width':   0.7,
    'ytick.minor.width':   0.7,
    'xtick.major.size':    5,
    'ytick.major.size':    5,
    'xtick.direction':     'out',
    'ytick.direction':     'out',
    'axes.grid':           False,
    'legend.frameon':      False,
    'legend.fontsize':     14,
    'figure.dpi':          150,
    'savefig.dpi':         300,
    'savefig.bbox':        'tight',
    'figure.facecolor':    'white',
    'axes.facecolor':      'white',
})

FS_TITLE  = 18   # axis label
FS_TICK   = 16   # tick labels
FS_LEGEND = 14   # legend
LW        = 1.0  # universal line width

# Country colours  (light = raw data,  dark = trend / fit line)
USA_LIGHT  = '#87CEEB'   # sky blue
USA_DARK   = '#1B6BAF'   # dark blue
CHN_LIGHT  = '#F4A6A6'   # light pink
CHN_DARK   = '#C0392B'   # dark red

COUNTRY_STYLE = {
    'USA': dict(light=USA_LIGHT, dark=USA_DARK,  marker='s'),
    'CHN': dict(light=CHN_LIGHT, dark=CHN_DARK,  marker='o'),
}

# LCZ colour palette (warm=built, cool=natural)
LCZ_COLORS = {
    1:'#8B0000', 2:'#CD5C5C', 3:'#FF6347', 4:'#FF8C00',
    5:'#FFA500', 6:'#FFD700', 7:'#DAA520', 8:'#D2691E',
    9:'#C0C0C0', 10:'#696969',
    11:'#006400', 12:'#228B22', 13:'#9ACD32', 14:'#7CFC00',
    15:'#808000', 16:'#D2B48C', 17:'#4169E1',
}

# Known policy events (reference lines)
EVENTS = {2008: '2008\nCrisis', 2015: 'Paris\nAgreement'}

# X-axis ticks for all time-series plots
YEAR_TICKS = [2000, 2005, 2010, 2015, 2020]


# ─────────────────── Axis styling helper ─────────────────────
def style_ax(ax, xlabel='', ylabel='', title=''):
    """Apply Origin-matching style to an axis."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(LW)
    ax.spines['bottom'].set_linewidth(LW)
    ax.tick_params(axis='both', labelsize=FS_TICK, width=LW, length=5)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FS_TITLE, fontweight='normal', labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS_TITLE, fontweight='normal', labelpad=8)
    if title:
        ax.set_title(title, fontsize=FS_TITLE, fontweight='normal', pad=10)


def set_year_xticks(ax, xlim=(1999, 2020)):
    ax.set_xticks(YEAR_TICKS)
    ax.set_xticklabels([str(y) for y in YEAR_TICKS], fontsize=FS_TICK)
    ax.set_xlim(*xlim)


def sig_label(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.1:   return '†'
    return 'ns'


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════

def load_lcz_annual(panel_paths):
    """
    Stratified-sample panel by year, assign dominant LCZ,
    return annual mean CO2 matrix per (Country, Year, LCZ).

    Returns
    -------
    co2_matrix : {country: DataFrame(index=Year, columns=LCZ_ids)}
    share_dict : {country: {lcz_id: mean_share}}
    """
    co2_matrix = {}
    share_dict = {}

    for country, pkl_path in panel_paths.items():
        print(f"  {country}: {pkl_path}")
        df = pd.read_pickle(pkl_path)

        co2col = 'CO2' if 'CO2' in df.columns else 'ln_CO2'
        share_cols = sorted(
            [c for c in df.columns if c.startswith('LCZ') and c.endswith('_share')],
            key=lambda c: int(c.replace('LCZ', '').replace('_share', ''))
        )

        df = df[df[co2col] > 0].copy()
        df['dominant_lcz'] = (
            df[share_cols].idxmax(axis=1)
            .str.replace('LCZ', '').str.replace('_share', '').astype(int)
        )
        share_dict[country] = {
            int(c.replace('LCZ', '').replace('_share', '')): df[c].mean()
            for c in share_cols
        }

        # Stratified sample
        parts = []
        for yr in sorted(df['Year'].unique()):
            g = df[df['Year'] == yr]
            parts.append(g.sample(n=min(SAMPLE_PER_YR, len(g)), random_state=RANDOM_SEED))
        df_s = pd.concat(parts, ignore_index=True)
        del df

        ann = (df_s.groupby(['Year', 'dominant_lcz'])[co2col]
                   .agg(['mean', 'count']).reset_index())
        ann.columns = ['Year', 'LCZ', 'CO2_mean', 'N']
        ann = ann[ann['N'] >= MIN_OBS_LCZ]

        pivot = ann.pivot(index='Year', columns='LCZ', values='CO2_mean').reindex(YEARS)
        co2_matrix[country] = pivot
        print(f"    {country}: {len(pivot.columns)} LCZ types, {len(df_s):,} sample rows")

    return co2_matrix, share_dict


# ══════════════════════════════════════════════════════════════
# METHOD 1 – LOWESS TREND + SLOPE HEATMAP
# ══════════════════════════════════════════════════════════════

def method1_trend_slopes(co2_matrix):
    print("=" * 60)
    print("METHOD 1: LOWESS Trend + Slope Heatmap")

    slope_rows = []
    smoothed   = {}   # {country: {lcz_id: (yr_arr, sm_arr)}}

    for country, pivot in co2_matrix.items():
        smoothed[country] = {}
        for lcz_id in pivot.columns:
            series = pivot[lcz_id].dropna()
            if len(series) < 8:
                continue
            yr_sub  = series.index.values.astype(float)
            co2_sub = series.values
            sm = lowess(co2_sub, yr_sub, frac=0.4, it=3, return_sorted=False)
            smoothed[country][lcz_id] = (yr_sub, sm)
            log_sm = np.log(np.maximum(sm, 1e-9))
            slope, intercept, r, p, se = stats.linregress(yr_sub, log_sm)
            slope_rows.append({
                'Country':      country,
                'LCZ':          lcz_id,
                'Slope_pct_yr': slope * 100,
                'SE_pct_yr':    se * 100,
                'T_stat':       slope / se if se > 0 else np.nan,
                'P_value':      p,
                'R2':           r ** 2,
                'Sig':          sig_label(p),
            })

    df_slopes = pd.DataFrame(slope_rows)

    # ── Fig 1a: LOWESS Trend Lines (LCZ 1–9) ──────────────
    # Two separate panels: USA (left) and CHN (right)
    # Raw: thin + transparent;  LOWESS: thick + opaque
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (country, pivot) in zip(axes, co2_matrix.items()):
        cs = COUNTRY_STYLE[country]
        for lcz_id in range(1, 10):
            if lcz_id not in smoothed[country]:
                continue
            yr_sub, sm = smoothed[country][lcz_id]
            raw_s = pivot[lcz_id].dropna()
            color = LCZ_COLORS[lcz_id]

            # Raw: thin, low alpha
            ax.plot(raw_s.index, raw_s.values,
                    color=color, linewidth=0.5, alpha=0.25)
            # LOWESS: full linewidth=1
            ax.plot(yr_sub, sm,
                    color=color, linewidth=LW,
                    marker='o', markersize=3, markevery=2,
                    label=f'LCZ{lcz_id}')

        set_year_xticks(ax)
        style_ax(ax,
                 xlabel='Year',
                 ylabel='Mean CO\u2082 (ton C/cell/month)' if country == 'USA' else '',
                 title=country)
        ax.legend(fontsize=FS_LEGEND, ncol=2, loc='upper right',
                  handlelength=1.5, labelspacing=0.4)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "Fig_Dyn1a_LOWESS_Trends.png")
    fig.savefig(out); plt.close(fig)
    print(f"  Saved: {out}")

    # ── Fig 1b: Slope Heatmap (all LCZ, both countries) ───
    pivot_hm = df_slopes.pivot(index='LCZ', columns='Country',
                               values='Slope_pct_yr').sort_index()
    sig_hm   = df_slopes.pivot(index='LCZ', columns='Country',
                               values='Sig').reindex(pivot_hm.index)

    vals_flat = pivot_hm.values[~np.isnan(pivot_hm.values)]
    vmax = np.ceil(np.nanmax(np.abs(vals_flat)) * 10) / 10 if len(vals_flat) else 5.0

    fig, ax = plt.subplots(figsize=(4.5, 9))

    im = ax.imshow(pivot_hm.values, aspect='auto',
                   cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   interpolation='nearest')

    ax.set_xticks(range(len(pivot_hm.columns)))
    ax.set_xticklabels(pivot_hm.columns, fontsize=FS_TICK)
    ax.set_yticks(range(len(pivot_hm.index)))
    ax.set_yticklabels([f'LCZ{i}' for i in pivot_hm.index], fontsize=FS_TICK)

    # Divider: built (1-10) vs natural (11-17)
    if 11 in list(pivot_hm.index):
        div = list(pivot_hm.index).index(11) - 0.5
        ax.axhline(div, color='white', linewidth=2.0)

    for i, lcz in enumerate(pivot_hm.index):
        for j, ctry in enumerate(pivot_hm.columns):
            val = pivot_hm.loc[lcz, ctry]
            sg  = sig_hm.loc[lcz, ctry] if pd.notna(sig_hm.loc[lcz, ctry]) else ''
            if pd.notna(val):
                txt_color = 'white' if abs(val) > vmax * 0.65 else 'black'
                ax.text(j, i, f'{val:.2f}{sg}',
                        ha='center', va='center',
                        fontsize=9, color=txt_color)

    cb = plt.colorbar(im, ax=ax, shrink=0.55, pad=0.03)
    cb.set_label('Annual Slope (%/yr)', fontsize=FS_LEGEND)
    cb.ax.tick_params(labelsize=FS_TICK - 2)
    ax.set_title('CO\u2082 Trend Slope by LCZ\n\u2020p<0.1  *p<0.05  **p<0.01  ***p<0.001',
                 fontsize=FS_LEGEND + 1, pad=10)
    for sp in ax.spines.values():
        sp.set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "Fig_Dyn1b_Slope_Heatmap.png")
    fig.savefig(out); plt.close(fig)
    print(f"  Saved: {out}")

    df_slopes.sort_values(['Country', 'LCZ']).to_csv(
        os.path.join(OUT_DIR, "Table_Dyn1_Slopes.csv"),
        index=False, encoding='utf-8-sig')
    print(f"  Saved: Table_Dyn1_Slopes.csv")

    return df_slopes, smoothed


# ══════════════════════════════════════════════════════════════
# METHOD 2 – CONVERGENCE ANALYSIS
# ══════════════════════════════════════════════════════════════

def method2_convergence(co2_matrix, share_dict):
    print("=" * 60)
    print("METHOD 2: Convergence Analysis")

    sigma_data = {}
    beta_data  = {}
    conv_rows  = []

    for country, pivot in co2_matrix.items():
        # sigma-convergence: CV = std/mean × 100 across LCZ types per year
        cv_series = pivot.std(axis=1) / pivot.mean(axis=1) * 100
        sigma_data[country] = cv_series

        cv_clean = cv_series.dropna()
        yrs_f    = cv_clean.index.values.astype(float)
        sl_s, it_s, r_s, p_s, _ = stats.linregress(yrs_f, cv_clean.values)
        print(f"    sigma {country}: slope={sl_s:.3f}%/yr  p={p_s:.3f} {sig_label(p_s)}")

        # beta-convergence
        yr_min, yr_max = pivot.index.min(), pivot.index.max()
        n_yrs = yr_max - yr_min
        beta_rows = []
        for lcz_id in pivot.columns:
            c0 = pivot.loc[yr_min, lcz_id]
            cT = pivot.loc[yr_max, lcz_id]
            if pd.isna(c0) or pd.isna(cT) or c0 <= 0:
                continue
            growth = (np.log(cT) - np.log(c0)) / n_yrs * 100
            beta_rows.append({
                'Country':       country,
                'LCZ':           lcz_id,
                'ln_CO2_2000':   np.log(c0),
                'CO2_2000':      c0,
                'CO2_end':       cT,
                'Growth_pct_yr': growth,
                'Share':         share_dict[country].get(lcz_id, 0),
            })

        df_beta = pd.DataFrame(beta_rows)
        beta_data[country] = df_beta

        if len(df_beta) >= 5:
            b_sl, b_it, b_r, b_p, _ = stats.linregress(
                df_beta['ln_CO2_2000'], df_beta['Growth_pct_yr'])
        else:
            b_sl = b_it = b_r = b_p = np.nan

        print(f"    beta  {country}: beta={b_sl:.4f}  p={b_p:.3f} {sig_label(b_p)}"
              f"  R2={b_r**2:.3f}" if not np.isnan(b_r) else f"    beta  {country}: insufficient data")

        conv_rows.append({
            'Country':           country,
            'CV_2000':           cv_series.get(2000, np.nan),
            'CV_2019':           cv_series.get(2019, np.nan),
            'Sigma_slope_%/yr':  sl_s,
            'Sigma_P':           p_s,
            'Sigma_sig':         sig_label(p_s),
            'Sigma_convergence': 'YES' if sl_s < 0 and p_s < 0.1 else 'NO',
            'Beta_coef':         b_sl,
            'Beta_P':            b_p,
            'Beta_R2':           b_r**2 if not np.isnan(b_r) else np.nan,
            'Beta_sig':          sig_label(b_p),
            'Beta_convergence':  'YES' if (not np.isnan(b_sl)) and b_sl < 0 and b_p < 0.1 else 'NO',
        })

    df_conv = pd.DataFrame(conv_rows)

    # ── Fig 2a: sigma-Convergence ──────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))

    for country, cv_series in sigma_data.items():
        cs       = COUNTRY_STYLE[country]
        cv_clean = cv_series.dropna()
        yrs      = cv_clean.index.values.astype(float)
        vals     = cv_clean.values
        sl, it, r, p, _ = stats.linregress(yrs, vals)

        # Raw data: light color + marker
        ax.plot(yrs, vals,
                color=cs['light'], linewidth=LW,
                marker=cs['marker'], markersize=6,
                markerfacecolor=cs['light'], markeredgecolor=cs['dark'],
                markeredgewidth=0.6,
                label=country, zorder=3)

        # Trend line: dark color, dashed if not significant
        ls = '-' if p < 0.1 else '--'
        ax.plot(yrs, sl * yrs + it,
                color=cs['dark'], linewidth=LW, linestyle=ls,
                label=f'{country} trend ({sig_label(p)})', zorder=2)

    # Event reference lines
    y_top = ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 50
    for ev_yr, ev_lbl in EVENTS.items():
        ax.axvline(ev_yr, color='#AAAAAA', linewidth=0.8,
                   linestyle=':', zorder=0)

    set_year_xticks(ax)
    style_ax(ax,
             xlabel='Year',
             ylabel='Coefficient of Variation (%) across LCZ types',
             title='\u03c3-Convergence: Cross-LCZ CO\u2082 Dispersion')
    ax.legend(fontsize=FS_LEGEND, loc='best')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "Fig_Dyn2a_Sigma_Convergence.png")
    fig.savefig(out); plt.close(fig)
    print(f"  Saved: {out}")

    # ── Fig 2b: beta-Convergence Scatter ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (country, df_beta) in zip(axes, beta_data.items()):
        if df_beta.empty:
            continue
        cs = COUNTRY_STYLE[country]
        xs = df_beta['ln_CO2_2000'].values
        ys = df_beta['Growth_pct_yr'].values
        ss = np.clip(df_beta['Share'].values * 4000 + 60, 60, 2500)

        # Scatter: light fill, dark edge
        ax.scatter(xs, ys, s=ss,
                   facecolors=cs['light'], edgecolors=cs['dark'],
                   linewidths=0.8, alpha=0.9, zorder=3)

        # LCZ labels
        for _, row in df_beta.iterrows():
            ax.annotate(f'LCZ{int(row.LCZ)}',
                        (row.ln_CO2_2000, row.Growth_pct_yr),
                        fontsize=10, ha='center', va='bottom',
                        xytext=(0, 6), textcoords='offset points', zorder=4)

        # OLS line
        if len(df_beta) >= 5:
            b_sl, b_it, b_r, b_p, _ = stats.linregress(xs, ys)
            x_fit = np.linspace(xs.min() - 0.1, xs.max() + 0.1, 200)
            ax.plot(x_fit, b_sl * x_fit + b_it,
                    color=cs['dark'], linewidth=LW, linestyle='--', zorder=2)
            converging = b_sl < 0 and b_p < 0.1
            ax.text(0.05, 0.05,
                    f'\u03b2 = {b_sl:.3f} ({sig_label(b_p)})\n'
                    f'R\u00b2 = {b_r**2:.2f}\n'
                    f'{"Converging \u2713" if converging else "Diverging \u2717"}',
                    transform=ax.transAxes, fontsize=FS_LEGEND,
                    va='bottom', color=cs['dark'],
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.75))

        ax.axhline(0, color='#AAAAAA', linewidth=0.8, linestyle=':', zorder=0)
        style_ax(ax,
                 xlabel='ln(CO\u2082) in 2000  (initial emission level)',
                 ylabel='Avg Annual Growth Rate (%/yr)' if country == 'USA' else '',
                 title=f'{country}  \u2013  \u03b2-Convergence')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "Fig_Dyn2b_Beta_Convergence.png")
    fig.savefig(out); plt.close(fig)
    print(f"  Saved: {out}")

    # Tables
    df_conv.to_csv(os.path.join(OUT_DIR, "Table_Dyn2_Convergence.csv"),
                   index=False, encoding='utf-8-sig')
    pd.concat(beta_data.values(), ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "Table_Dyn2b_Beta_Detail.csv"),
        index=False, encoding='utf-8-sig')
    print(f"  Saved: Table_Dyn2_Convergence.csv  /  Table_Dyn2b_Beta_Detail.csv")

    return df_conv, beta_data


# ══════════════════════════════════════════════════════════════
# METHOD 3 – STRUCTURAL BREAK DETECTION
# ══════════════════════════════════════════════════════════════

def chow_optimal_break(years, values, min_seg=4):
    """
    Find optimal single structural break by maximising Chow F-statistic.
    Returns (break_year, f_stat, p_value, pre_slope, post_slope, delta_level).
    """
    n    = len(years)
    best = dict(yr=np.nan, f=0.0, p=1.0, pre=np.nan, post=np.nan, delta=np.nan)

    sl_r, it_r, _, _, _ = stats.linregress(years, values)
    rss_r = np.sum((values - (it_r + sl_r * years)) ** 2)

    for i in range(min_seg, n - min_seg + 1):
        y1, y2 = values[:i], values[i:]
        x1, x2 = years[:i],  years[i:]
        if len(y1) < 3 or len(y2) < 3:
            continue
        sl1, it1, _, _, _ = stats.linregress(x1, y1)
        sl2, it2, _, _, _ = stats.linregress(x2, y2)
        rss_u = (np.sum((y1 - (it1 + sl1 * x1))**2) +
                 np.sum((y2 - (it2 + sl2 * x2))**2))
        if rss_u < 1e-12:
            continue
        k = 2
        f = ((rss_r - rss_u) / k) / (rss_u / max(n - 2 * k, 1))
        p = 1 - stats.f.cdf(f, k, n - 2 * k)
        if f > best['f']:
            delta = (it2 + sl2 * years[i]) - (it1 + sl1 * years[i])
            best.update(yr=years[i], f=f, p=p, pre=sl1, post=sl2, delta=delta)

    return best['yr'], best['f'], best['p'], best['pre'], best['post'], best['delta']


def method3_structural_breaks(co2_matrix):
    print("=" * 60)
    print("METHOD 3: Structural Break Detection (Chow)")

    break_rows = []

    for country, pivot in co2_matrix.items():
        for lcz_id in sorted(pivot.columns):
            series = pivot[lcz_id].dropna()
            if len(series) < 10:
                continue
            yr_sub  = series.index.values.astype(float)
            co2_sub = series.values
            bk_yr, f, p, pre_sl, post_sl, delta = chow_optimal_break(yr_sub, co2_sub)
            direction = 'Up' if (not np.isnan(delta) and delta > 0) else 'Down'
            sg = sig_label(p)
            print(f"    {country} LCZ{lcz_id:2d}: "
                  f"break={int(bk_yr) if not np.isnan(bk_yr) else '?':4}  "
                  f"F={f:6.2f}  p={p:.3f}{sg:>4}  "
                  f"pre={pre_sl:.4f}  post={post_sl:.4f}")
            break_rows.append({
                'Country':     country,
                'LCZ':         lcz_id,
                'Break_Year':  int(bk_yr) if not np.isnan(bk_yr) else np.nan,
                'F_stat':      f,
                'P_value':     p,
                'Sig':         sg,
                'Pre_slope':   pre_sl,
                'Post_slope':  post_sl,
                'Delta_level': delta,
                'Direction':   direction,
            })

    df_breaks = pd.DataFrame(break_rows)

    # ── Fig 3a: 3×3 grid LCZ 1-9 (index 2000=100) ────────
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
    axes_flat  = axes.ravel()

    for idx, lcz_id in enumerate(range(1, 10)):
        ax = axes_flat[idx]

        for country in ['USA', 'CHN']:
            if country not in co2_matrix:
                continue
            pivot = co2_matrix[country]
            if lcz_id not in pivot.columns:
                continue

            cs     = COUNTRY_STYLE[country]
            series = pivot[lcz_id].dropna()
            yr_sub = series.index.values.astype(float)
            base   = series.values[0] if series.values[0] > 0 else 1.0
            idx_v  = series.values / base * 100

            # Raw: light color + marker (every 2 years for clarity)
            ax.plot(yr_sub, idx_v,
                    color=cs['light'], linewidth=LW,
                    marker=cs['marker'], markersize=4,
                    markerfacecolor=cs['light'], markeredgecolor=cs['dark'],
                    markeredgewidth=0.5, markevery=2,
                    label=country, zorder=3)

            # Break year marker (significant only)
            row_m = df_breaks[
                (df_breaks['Country'] == country) &
                (df_breaks['LCZ'] == lcz_id) &
                (df_breaks['P_value'] < 0.1)
            ]
            if not row_m.empty:
                bk = row_m.iloc[0]['Break_Year']
                if not np.isnan(bk):
                    ax.axvline(bk, color=cs['dark'], linewidth=LW,
                               linestyle='--', alpha=0.85, zorder=2)

        # Horizontal reference at 100
        ax.axhline(100, color='#AAAAAA', linewidth=0.7, linestyle=':', zorder=0)

        # Event reference lines
        for ev_yr in EVENTS:
            ax.axvline(ev_yr, color='#DDDDDD', linewidth=0.7,
                       linestyle=':', zorder=0)

        ax.set_title(f'LCZ{lcz_id}', fontsize=FS_TITLE - 2, fontweight='normal')
        ax.tick_params(labelsize=FS_TICK - 2, width=LW, length=4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(LW)
        ax.spines['bottom'].set_linewidth(LW)
        ax.set_xticks(YEAR_TICKS)
        ax.set_xticklabels([str(y) for y in YEAR_TICKS], fontsize=FS_TICK - 2, rotation=30)
        ax.set_xlim(1999, 2020)

        if idx % 3 == 0:
            ax.set_ylabel('Index (2000=100)', fontsize=FS_TITLE - 3)
        if idx >= 6:
            ax.set_xlabel('Year', fontsize=FS_TITLE - 3)

    # Global legend
    leg_handles = []
    for country in ['USA', 'CHN']:
        cs = COUNTRY_STYLE[country]
        leg_handles.append(
            Line2D([0],[0], color=cs['light'], linewidth=LW,
                   marker=cs['marker'], markersize=6,
                   markerfacecolor=cs['light'], markeredgecolor=cs['dark'],
                   markeredgewidth=0.5, label=country))
    leg_handles.append(
        Line2D([0],[0], color='#888888', linewidth=LW, linestyle='--',
               label='Structural break (p<0.1)'))

    fig.legend(handles=leg_handles, loc='lower center', ncol=3,
               fontsize=FS_LEGEND, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('CO\u2082 Trend Index (2000=100) with Structural Breaks  \u2013  LCZ 1\u20139',
                 fontsize=FS_TITLE, fontweight='normal', y=1.02)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "Fig_Dyn3a_Break_TimeSeries.png")
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved: {out}")

    # ── Fig 3b: Break Year Summary Chart ──────────────────
    fig, ax = plt.subplots(figsize=(11, 7))

    df_sig = df_breaks[df_breaks['P_value'] < 0.1].copy()
    offsets = {'USA': 0.18, 'CHN': -0.18}

    if not df_sig.empty:
        for _, row in df_sig.iterrows():
            if np.isnan(row['Break_Year']):
                continue
            cs     = COUNTRY_STYLE[row['Country']]
            marker = '^' if row['Direction'] == 'Up' else 'v'
            size   = np.clip(row['F_stat'] * 14, 50, 500)
            y_pos  = row['LCZ'] + offsets[row['Country']]
            ax.scatter(row['Break_Year'], y_pos,
                       c=cs['dark'], s=size, marker=marker,
                       alpha=0.9, edgecolors='white', linewidths=0.5, zorder=3)

    # Event lines
    for yr, lbl in EVENTS.items():
        ax.axvline(yr, color='#AAAAAA', linewidth=0.8, linestyle=':', zorder=0)
        ax.text(yr, 17.8, lbl, ha='center', va='top',
                fontsize=FS_LEGEND - 2, color='#888888')

    # Built / natural divider
    ax.axhline(10.5, color='#CCCCCC', linewidth=0.8, linestyle='-')
    ax.text(1998.6, 10.5, 'Built\n\u2191\n\u2193\nNatural',
            fontsize=FS_LEGEND - 3, ha='right', va='center', color='#888888')

    ax.set_yticks(range(1, 18))
    ax.set_yticklabels([f'LCZ{i}' for i in range(1, 18)], fontsize=FS_TICK)
    ax.set_xticks(YEAR_TICKS)
    ax.set_xticklabels([str(y) for y in YEAR_TICKS], fontsize=FS_TICK)
    ax.set_xlim(1998.5, 2020.5)
    ax.set_ylim(0.2, 18.5)

    leg_h = [
        Line2D([0],[0], color=COUNTRY_STYLE['USA']['dark'],
               marker='o', linestyle='', markersize=9, label='USA'),
        Line2D([0],[0], color=COUNTRY_STYLE['CHN']['dark'],
               marker='o', linestyle='', markersize=9, label='CHN'),
        Line2D([0],[0], color='grey', marker='^', linestyle='',
               markersize=9, label='Level increase (\u25b2)'),
        Line2D([0],[0], color='grey', marker='v', linestyle='',
               markersize=9, label='Level decrease (\u25bc)'),
        Line2D([0],[0], color='grey', marker='o', linestyle='',
               markersize=5, label='Size \u221d F-stat'),
    ]
    ax.legend(handles=leg_h, fontsize=FS_LEGEND,
              loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

    style_ax(ax,
             xlabel='Year of Structural Break',
             ylabel='LCZ Type',
             title='Structural Break Year by LCZ  (p\u00a0<\u00a00.1,  \u25b2\u202f=\u202flevel up,  \u25bc\u202f=\u202flevel down)')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "Fig_Dyn3b_Break_Summary.png")
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved: {out}")

    df_breaks.sort_values(['Country', 'LCZ']).to_csv(
        os.path.join(OUT_DIR, "Table_Dyn3_Breaks.csv"),
        index=False, encoding='utf-8-sig')
    print(f"  Saved: Table_Dyn3_Breaks.csv")

    return df_breaks


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  LCZ-CO2 Temporal Dynamics  |  Methods 1-2-3")
    print("=" * 60)

    PANEL_PATHS = {'USA': USA_PANEL, 'CHN': CHN_PANEL}

    print("\n[1/4] Loading panel data ...")
    co2_matrix, share_dict = load_lcz_annual(PANEL_PATHS)

    print("\n[2/4] Method 1: LOWESS Trend + Slope Heatmap ...")
    df_slopes, smoothed = method1_trend_slopes(co2_matrix)

    print("\n[3/4] Method 2: Convergence Analysis ...")
    df_conv, beta_data = method2_convergence(co2_matrix, share_dict)

    print("\n[4/4] Method 3: Structural Break Detection ...")
    df_breaks = method3_structural_breaks(co2_matrix)

    print("\n" + "=" * 60)
    print(f"  All outputs saved to: {OUT_DIR}")
    print("  Figures : Fig_Dyn1a/1b  2a/2b  3a/3b  (.png)")
    print("  Tables  : Table_Dyn1  Dyn2  Dyn2b  Dyn3  (.csv)")
    print("=" * 60)
