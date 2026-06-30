# -*- coding: utf-8 -*-
"""
Code_Robustness_IndustrialStructure.py
=======================================
Reviewer robustness check: add an INDUSTRIAL-STRUCTURE control to the
monthly fixed-effects morphology regression, using electricity intensity
(EC / GDP) as a proxy for the secondary-industry / manufacturing share.

Why EC/GDP:
    Electricity intensity (electricity consumed per unit of GDP) is a
    standard structural indicator. Industry and manufacturing are far more
    electricity-intensive per unit of output than services, so EC/GDP tracks
    the secondary-industry share. In this dataset it declines steadily over
    2000-2019 (CHN -21%, USA -10%), the structural fingerprint of the
    deindustrialisation the reviewer asked us to control for.

Design (admin-level merge — no panel rebuild required):
    The cleaned monthly panel stores only the administrative unit id
    (City_ID for CHN, State_ID for USA), NOT pixel coordinates. Industrial
    structure is a regional economic concept, so we aggregate EC/GDP to the
    admin x year level (mean electricity intensity per city/state per year),
    then merge it onto the panel by (fe_col, Year). Because it varies by
    admin unit AND year, ln(EC/GDP) is fully identified under the existing
    City/State FE + Year FE.

Two models compared, both with Month + Year + admin FE, LCZ9 reference,
cluster-robust SE at the admin level, stratified-sampled (8000 / year-month):
    Model 2  : ln_CO2 ~ controls + LCZ            (current published spec)
    Model 5  : ln_CO2 ~ controls + LCZ + ln_ECgdp (+ industrial structure)

Outputs (to each country's *_1km result dir):
    Model5_IndStruct_Coefs.csv
    Model5_IndStruct_Summary.txt
    Robustness_IndStruct_Comparison.csv   <- side-by-side LCZ & Pop/GDP coefs

Requirements (all present on the local D: workstation):
    rasterio, geopandas, statsmodels, pandas, numpy
Run time: ~10-20 min per country (raster aggregation + two regressions).
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.warp import reproject, Resampling
from rasterio.transform import array_bounds
import geopandas as gpd
from shapely.geometry import box
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

# ============================================================
# 1. CONFIG  (paths mirror Code_Monthly_Regression.py)
# ============================================================

YEARS = range(2000, 2020)

SAMPLE_PER_YM     = 8000
RANDOM_SEED       = 42
CO2_PERCENTILE    = 99
LCZ_MAX_ZERO_FRAC = 0.99
SEASON_BASE       = 'Spring'

SEASON_MAP = {12: 'Winter', 1: 'Winter', 2: 'Winter',
              3: 'Spring', 4: 'Spring', 5: 'Spring',
              6: 'Summer', 7: 'Summer', 8: 'Summer',
              9: 'Fall', 10: 'Fall', 11: 'Fall'}

COUNTRY_CFG = {
    'USA': {
        'panel':     r"D:\LCZCarbon\Results_USA_Month\USA_1km\Panel_Monthly_clean.pkl",
        'outdir':    r"D:\LCZCarbon\Results_USA_Month\USA_1km",
        'fe_col':    'State_ID',
        # EC/GDP electricity-intensity rasters
        'ecgdp_dir': r"D:\LCZCarbon\电力强度\USA",
        'ecgdp_tpl': "EC_GDP_{year}_USA.tif",
        # reference CO2 raster (defines the analysis grid) + admin boundary
        'ref_co2':   r"D:\LCZCarbon\CarbonSum\US\CarbonSum_2000_US.tif",
        'admin_shp': r"D:\LCZCarbon\US_State_Boundaries\US_State_Boundaries.shp",
        'admin_kw':  ['NAME', 'STATE', 'STNAME', 'NAME_1', 'STATENAME'],
    },
    'CHN': {
        'panel':     r"D:\LCZCarbon\Results_CHN_Month\CHN_1km\Panel_Monthly_clean.pkl",
        'outdir':    r"D:\LCZCarbon\Results_CHN_Month\CHN_1km",
        'fe_col':    'City_ID',
        'ecgdp_dir': r"D:\LCZCarbon\电力强度\CHN",
        'ecgdp_tpl': "EC_GDP_{year}_CHN.tif",
        'ref_co2':   r"D:\LCZCarbon\CarbonSum\CHN\CarbonSum_2000_CHN.tif",
        'admin_shp': r"D:\LCZCarbon\ChinaBoundary\ChinaBoundaryNew.shp",
        'admin_kw':  ['NAME', 'CITY', '市', '名称', 'DIJI', 'PREFEC', 'DNAME', 'CNAME'],
    },
}


# ============================================================
# 2. BUILD ADMIN x YEAR ELECTRICITY-INTENSITY TABLE
#    (rasterize admin boundary -> for each year, mean EC/GDP per admin unit)
#    NB: admin IDs are assigned EXACTLY as in the panel builder:
#        sort by name -> ID_Num = 1..N. This guarantees the merge key
#        matches the panel's City_ID / State_ID.
# ============================================================

def get_admin_raster(shp_path, name_kw, ref_meta):
    gdf = gpd.read_file(shp_path)
    if gdf.crs != ref_meta['crs']:
        gdf = gdf.to_crs(ref_meta['crs'])
    name_col = next((c for c in gdf.columns
                     if any(k in c.upper() for k in name_kw) and c != 'geometry'), None)
    if name_col is None:
        name_col = next((c for c in gdf.columns
                         if c != 'geometry' and gdf[c].dtype == object), None)
    gdf = gdf.sort_values(name_col).reset_index(drop=True)
    gdf['ID_Num'] = range(1, len(gdf) + 1)
    burned = features.rasterize(
        ((geom, val) for geom, val in zip(gdf.geometry, gdf['ID_Num'])),
        out_shape=(ref_meta['height'], ref_meta['width']),
        transform=ref_meta['transform'], fill=0, dtype='int16')
    return burned


def build_ecgdp_by_admin(cfg):
    """Return DataFrame: [fe_col, Year, EC_GDP] = mean electricity intensity."""
    with rasterio.open(cfg['ref_co2']) as src:
        ref_meta = src.meta.copy()
    print(f"  ref grid: {ref_meta['width']}x{ref_meta['height']} @ {ref_meta['crs']}")

    admin = get_admin_raster(cfg['admin_shp'], cfg['admin_kw'], ref_meta)
    print(f"  admin units rasterized: {len(np.unique(admin[admin > 0]))}")

    rows = []
    for year in YEARS:
        fp = os.path.join(cfg['ecgdp_dir'], cfg['ecgdp_tpl'].format(year=year))
        if not os.path.exists(fp):
            print(f"    [WARN] missing {fp}"); continue
        # resample EC/GDP to the reference grid
        with rasterio.open(fp) as src:
            dst = np.zeros((ref_meta['height'], ref_meta['width']), dtype='float32')
            reproject(source=rasterio.band(src, 1), destination=dst,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=ref_meta['transform'], dst_crs=ref_meta['crs'],
                      resampling=Resampling.average)
        valid = (admin > 0) & (dst > 0) & np.isfinite(dst)
        ids = admin[valid]; vals = dst[valid]
        # mean EC/GDP per admin unit this year
        df_y = pd.DataFrame({'fe': ids, 'v': vals}).groupby('fe')['v'].mean()
        for fe_id, v in df_y.items():
            rows.append({cfg['fe_col']: int(fe_id), 'Year': int(year), 'EC_GDP': float(v)})
        print(f"    {year}: {len(df_y)} admin units, median EC/GDP={np.median(vals):,.0f}")
        del dst; gc.collect()

    return pd.DataFrame(rows)


# ============================================================
# 3. REGRESSION:  Model 2  vs  Model 5 (+ ln_ECgdp)
# ============================================================

def run_models(df, fe_col, out_dir, label, ec_table):
    print(f"\n  [{label}] panel rows: {len(df):,}")

    # merge admin x year electricity intensity
    df = df.merge(ec_table, on=[fe_col, 'Year'], how='left')
    miss = df['EC_GDP'].isna().mean()
    print(f"    EC/GDP merged; missing fraction = {miss:.3%}")

    # log transforms (mirror existing pipeline)
    for col in ['CO2', 'Pop', 'GDP', 'HDD', 'CDD', 'GHI', 'EC_GDP']:
        df[f'ln_{col}'] = np.log(df[col].clip(lower=0) + 1).astype('float32')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df['Season'] = pd.Categorical(
        df['Month'].map(SEASON_MAP),
        categories=[SEASON_BASE] + [s for s in ['Summer', 'Fall', 'Winter']],
        ordered=False)

    # LCZ near-zero filter (identical rule to the main pipeline)
    all_lcz = [c for c in df.columns if c.endswith('_share')]
    zero_frac = (df[all_lcz] < 1e-6).mean()
    valid_lcz = zero_frac[zero_frac < LCZ_MAX_ZERO_FRAC].index.tolist()

    keep = ([f'ln_{c}' for c in ['CO2', 'Pop', 'GDP', 'HDD', 'CDD', 'GHI', 'EC_GDP']]
            + [fe_col, 'Year', 'Month', 'Season'] + valid_lcz)
    df.dropna(subset=keep, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"    after dropna: {len(df):,} rows")

    ref_lcz  = 'LCZ9_share' if 'LCZ9_share' in valid_lcz else valid_lcz[0]
    lcz_vars = [c for c in valid_lcz if c != ref_lcz]
    lcz_f    = ' + '.join(lcz_vars)
    controls = ("ln_Pop + ln_GDP + ln_HDD + ln_CDD + ln_GHI"
                f" + C(Month) + C(Year) + C({fe_col})")

    models = {
        'Model2_Morphology':  f"ln_CO2 ~ {controls} + {lcz_f}",
        'Model5_IndStruct':   f"ln_CO2 ~ {controls} + {lcz_f} + ln_EC_GDP",
    }

    fitted = {}
    for name, formula in models.items():
        print(f"\n    Running {name} ...")
        m = smf.ols(formula, data=df).fit(
            cov_type='cluster', cov_kwds={'groups': df[fe_col]})
        fitted[name] = m
        print(f"      R2={m.rsquared:.4f}  N={int(m.nobs):,}")
        if name == 'Model5_IndStruct':
            ci = m.conf_int()
            pd.DataFrame({'coef': m.params, 'std_err': m.bse, 'pvalue': m.pvalues,
                          'conf_low': ci[0], 'conf_high': ci[1]}).to_csv(
                os.path.join(out_dir, 'Model5_IndStruct_Coefs.csv'),
                encoding='utf-8-sig')
            with open(os.path.join(out_dir, 'Model5_IndStruct_Summary.txt'),
                      'w', encoding='utf-8') as f:
                f.write(f"Robustness: + ln(EC/GDP) industrial-structure control\n")
                f.write(f"LCZ ref={ref_lcz} | cluster SE on {fe_col}\n\n")
                f.write(m.summary().as_text())

    # ---- side-by-side comparison of the variables that matter ----
    focus = ['ln_Pop', 'ln_GDP', 'ln_EC_GDP'] + lcz_vars
    m2, m5 = fitted['Model2_Morphology'], fitted['Model5_IndStruct']
    comp = []
    for term in focus:
        comp.append({
            'term':        term,
            'Model2_coef': m2.params.get(term, np.nan),
            'Model2_p':    m2.pvalues.get(term, np.nan),
            'Model5_coef': m5.params.get(term, np.nan),
            'Model5_p':    m5.pvalues.get(term, np.nan),
            'abs_change':  abs(m5.params.get(term, np.nan) - m2.params.get(term, np.nan)),
            'pct_change':  (100 * (m5.params.get(term, np.nan) - m2.params.get(term, np.nan))
                            / abs(m2.params.get(term, np.nan))
                            if m2.params.get(term, 0) != 0 else np.nan),
        })
    comp_df = pd.DataFrame(comp)
    comp_df.to_csv(os.path.join(out_dir, 'Robustness_IndStruct_Comparison.csv'),
                   index=False, encoding='utf-8-sig')
    print(f"\n  [{label}] comparison saved. LCZ coef mean |% change| = "
          f"{comp_df[comp_df.term.str.startswith('LCZ')]['pct_change'].abs().mean():.2f}%")
    print(comp_df.round(4).to_string(index=False))
    return comp_df


# ============================================================
# 4. MAIN
# ============================================================

def run_country(country):
    cfg = COUNTRY_CFG[country]
    print(f"\n{'='*64}\n  {country} — Industrial-structure robustness\n{'='*64}")

    # 4a. admin x year EC/GDP table
    print("  [1/3] Building EC/GDP admin x year table from rasters ...")
    ec_table = build_ecgdp_by_admin(cfg)
    ec_table.to_csv(os.path.join(cfg['outdir'], 'EC_GDP_by_admin_year.csv'),
                    index=False, encoding='utf-8-sig')

    # 4b. load + stratified-sample panel
    print("  [2/3] Loading & stratified-sampling panel ...")
    df = pd.read_pickle(cfg['panel'])
    df = df[df['CO2'] <= df['CO2'].quantile(CO2_PERCENTILE / 100)].copy()
    ym = df['Year'].astype(str) + '_' + df['Month'].astype(str).str.zfill(2)
    df = (df.groupby(ym, group_keys=False)
            .apply(lambda g: g.sample(n=min(SAMPLE_PER_YM, len(g)),
                                      random_state=RANDOM_SEED))
            .reset_index(drop=True))
    print(f"    sampled rows: {len(df):,}")

    # 4c. regressions
    print("  [3/3] Estimating Model 2 vs Model 5 ...")
    run_models(df, cfg['fe_col'], cfg['outdir'], country, ec_table)
    del df; gc.collect()


if __name__ == '__main__':
    print("="*64)
    print("  Industrial-structure robustness (EC/GDP) — USA + CHN")
    print("="*64)
    for c in ['USA', 'CHN']:
        run_country(c)
    print("\nDone. Inspect Robustness_IndStruct_Comparison.csv in each result dir:")
    print("  - LCZ coefficients should stay similar in sign & magnitude")
    print("  - ln_EC_GDP coefficient = the industrial-structure effect")
    print("  - ln_GDP may shift (income & industrial structure are entangled)")
