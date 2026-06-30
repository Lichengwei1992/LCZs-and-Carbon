# -*- coding: utf-8 -*-
"""
Code_Robustness_IndStruct_v2.py
===============================================================
Hardened version that GUARANTEES CSV output for the industrial-structure
robustness check (EC/GDP electricity-intensity control).

What changed vs v1:
  * every CSV is written the moment it exists (no all-or-nothing at the end)
  * full try/except around each model fit -> one failing model never blocks
    the other model's output
  * verbose diagnostics: prints absolute paths, row counts, merge coverage
  * a final "FILES WRITTEN" report so you can see exactly what was produced
  * safe coefficient extraction (.get) so the comparison table never crashes
  * writes a plain CSV of EVERY coefficient for both models, plus the focused
    comparison table.

Run locally (D: workstation has rasterio/geopandas/statsmodels):
    python Code_Robustness_IndStruct_v2.py
Outputs land in each country's *_1km result dir.
"""

import os, gc, warnings, traceback
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.warp import reproject, Resampling
import geopandas as gpd
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

YEARS             = range(2000, 2020)
SAMPLE_PER_YM     = 8000
RANDOM_SEED       = 42
CO2_PERCENTILE    = 99
LCZ_MAX_ZERO_FRAC = 0.99
SEASON_BASE       = 'Spring'
SEASON_MAP = {12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',
              6:'Summer',7:'Summer',8:'Summer',9:'Fall',10:'Fall',11:'Fall'}

COUNTRY_CFG = {
    'USA': {
        'panel':     r"D:\LCZCarbon\Results_USA_Month\USA_1km\Panel_Monthly_clean.pkl",
        'outdir':    r"D:\LCZCarbon\Results_USA_Month\USA_1km",
        'fe_col':    'State_ID',
        'ecgdp_dir': r"D:\LCZCarbon\电力强度\USA",
        'ecgdp_tpl': "EC_GDP_{year}_USA.tif",
        'ref_co2':   r"D:\LCZCarbon\CarbonSum\US\CarbonSum_2000_US.tif",
        'admin_shp': r"D:\LCZCarbon\US_State_Boundaries\US_State_Boundaries.shp",
        'admin_kw':  ['NAME','STATE','STNAME','NAME_1','STATENAME'],
    },
    'CHN': {
        'panel':     r"D:\LCZCarbon\Results_CHN_Month\CHN_1km\Panel_Monthly_clean.pkl",
        'outdir':    r"D:\LCZCarbon\Results_CHN_Month\CHN_1km",
        'fe_col':    'City_ID',
        'ecgdp_dir': r"D:\LCZCarbon\电力强度\CHN",
        'ecgdp_tpl': "EC_GDP_{year}_CHN.tif",
        'ref_co2':   r"D:\LCZCarbon\CarbonSum\CHN\CarbonSum_2000_CHN.tif",
        'admin_shp': r"D:\LCZCarbon\ChinaBoundary\ChinaBoundaryNew.shp",
        'admin_kw':  ['NAME','CITY','市','名称','DIJI','PREFEC','DNAME','CNAME'],
    },
}

WRITTEN = []   # track every file we produce

def _save_csv(df, path, note=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=True if df.index.name else False, encoding='utf-8-sig')
    WRITTEN.append(path)
    print(f"      [WROTE] {path}  ({len(df)} rows) {note}")


# ---------- EC/GDP admin x year table (the only raster step) ----------

def build_ecgdp_table(cfg):
    with rasterio.open(cfg['ref_co2']) as src:
        ref = src.meta.copy()
    gdf = gpd.read_file(cfg['admin_shp'])
    if gdf.crs != ref['crs']:
        gdf = gdf.to_crs(ref['crs'])
    name_col = next((c for c in gdf.columns
                     if any(k in c.upper() for k in cfg['admin_kw']) and c != 'geometry'), None)
    if name_col is None:
        name_col = next((c for c in gdf.columns
                         if c != 'geometry' and gdf[c].dtype == object), None)
    gdf = gdf.sort_values(name_col).reset_index(drop=True)
    gdf['ID_Num'] = range(1, len(gdf) + 1)
    admin = features.rasterize(
        ((g, v) for g, v in zip(gdf.geometry, gdf['ID_Num'])),
        out_shape=(ref['height'], ref['width']),
        transform=ref['transform'], fill=0, dtype='int16')

    rows = []
    for yr in YEARS:
        fp = os.path.join(cfg['ecgdp_dir'], cfg['ecgdp_tpl'].format(year=yr))
        if not os.path.exists(fp):
            print(f"      [WARN] missing {fp}"); continue
        with rasterio.open(fp) as src:
            dst = np.zeros((ref['height'], ref['width']), dtype='float32')
            reproject(rasterio.band(src, 1), dst,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=ref['transform'], dst_crs=ref['crs'],
                      resampling=Resampling.average)
        m = (admin > 0) & (dst > 0) & np.isfinite(dst)
        g = pd.DataFrame({'fe': admin[m], 'v': dst[m]}).groupby('fe')['v'].mean()
        for fe_id, v in g.items():
            rows.append({cfg['fe_col']: int(fe_id), 'Year': int(yr), 'EC_GDP': float(v)})
        del dst; gc.collect()
    return pd.DataFrame(rows)


# ---------- regression with guaranteed output ----------

def run_models(df, cfg, ec_table):
    fe_col, out_dir = cfg['fe_col'], cfg['outdir']

    df = df.merge(ec_table, on=[fe_col, 'Year'], how='left')
    cov = 1 - df['EC_GDP'].isna().mean()
    print(f"      EC/GDP merge coverage: {cov:.1%}")
    if cov < 0.5:
        print("      [WARN] low coverage — check admin-id alignment, continuing on covered rows")

    for c in ['CO2','Pop','GDP','HDD','CDD','GHI','EC_GDP']:
        df[f'ln_{c}'] = np.log(df[c].clip(lower=0) + 1).astype('float32')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['Season'] = pd.Categorical(df['Month'].map(SEASON_MAP),
                                  categories=[SEASON_BASE,'Summer','Fall','Winter'])

    all_lcz   = [c for c in df.columns if c.endswith('_share')]
    valid_lcz = (df[all_lcz] < 1e-6).mean()
    valid_lcz = valid_lcz[valid_lcz < LCZ_MAX_ZERO_FRAC].index.tolist()

    keep = ([f'ln_{c}' for c in ['CO2','Pop','GDP','HDD','CDD','GHI','EC_GDP']]
            + [fe_col,'Year','Month'] + valid_lcz)
    df.dropna(subset=keep, inplace=True)
    print(f"      rows after dropna: {len(df):,}")
    if len(df) < 1000:
        print("      [ERROR] too few rows to fit — aborting this country"); return

    ref_lcz  = 'LCZ9_share' if 'LCZ9_share' in valid_lcz else valid_lcz[0]
    lcz_vars = [c for c in valid_lcz if c != ref_lcz]
    controls = (f"ln_Pop + ln_GDP + ln_HDD + ln_CDD + ln_GHI"
                f" + C(Month) + C(Year) + C({fe_col})")
    formulas = {
        'Model2_Morphology': f"ln_CO2 ~ {controls} + {' + '.join(lcz_vars)}",
        'Model5_IndStruct':  f"ln_CO2 ~ {controls} + {' + '.join(lcz_vars)} + ln_EC_GDP",
    }

    fitted = {}
    for name, formula in formulas.items():
        try:
            print(f"      fitting {name} ...")
            m = smf.ols(formula, data=df).fit(cov_type='cluster',
                                              cov_kwds={'groups': df[fe_col]})
            fitted[name] = m
            ci = m.conf_int()
            coefs = pd.DataFrame({'coef': m.params, 'std_err': m.bse,
                                  'pvalue': m.pvalues, 'conf_low': ci[0], 'conf_high': ci[1]})
            coefs.index.name = 'term'
            _save_csv(coefs, os.path.join(out_dir, f'{name}_Coefs.csv'),
                      note=f"R2={m.rsquared:.4f} N={int(m.nobs):,}")
        except Exception as e:
            print(f"      [ERROR] {name} failed: {e}")
            traceback.print_exc()

    # comparison table — written even if only one model succeeded
    focus = ['ln_Pop','ln_GDP','ln_EC_GDP'] + lcz_vars
    m2 = fitted.get('Model2_Morphology')
    m5 = fitted.get('Model5_IndStruct')
    rows = []
    for t in focus:
        c2 = m2.params.get(t, np.nan) if m2 is not None else np.nan
        p2 = m2.pvalues.get(t, np.nan) if m2 is not None else np.nan
        c5 = m5.params.get(t, np.nan) if m5 is not None else np.nan
        p5 = m5.pvalues.get(t, np.nan) if m5 is not None else np.nan
        pct = (100*(c5-c2)/abs(c2)) if (c2 not in (0, np.nan) and not np.isnan(c2) and not np.isnan(c5)) else np.nan
        rows.append({'term':t,'Model2_coef':c2,'Model2_p':p2,
                     'Model5_coef':c5,'Model5_p':p5,'pct_change':pct})
    comp = pd.DataFrame(rows)
    _save_csv(comp, os.path.join(out_dir, 'Robustness_IndStruct_Comparison.csv'))
    lcz_chg = comp[comp.term.str.startswith('LCZ')]['pct_change'].abs().mean()
    print(f"      >>> mean |%change| of LCZ coefficients = {lcz_chg:.2f}%")
    if m5 is not None:
        print(f"      >>> ln_EC_GDP coef = {m5.params.get('ln_EC_GDP', np.nan):.4f} "
              f"(p={m5.pvalues.get('ln_EC_GDP', np.nan):.3g})")


def run_country(country):
    cfg = COUNTRY_CFG[country]
    print(f"\n{'='*60}\n  {country}\n{'='*60}")
    os.makedirs(cfg['outdir'], exist_ok=True)
    try:
        print("  [1/3] EC/GDP admin x year table ...")
        ec = build_ecgdp_table(cfg)
        _save_csv(ec, os.path.join(cfg['outdir'], 'EC_GDP_by_admin_year.csv'))

        print("  [2/3] load + stratified sample panel ...")
        df = pd.read_pickle(cfg['panel'])
        df = df[df['CO2'] <= df['CO2'].quantile(CO2_PERCENTILE/100)].copy()
        ym = df['Year'].astype(str) + '_' + df['Month'].astype(str).str.zfill(2)
        df = (df.groupby(ym, group_keys=False)
                .apply(lambda g: g.sample(n=min(SAMPLE_PER_YM, len(g)), random_state=RANDOM_SEED))
                .reset_index(drop=True))
        print(f"      sampled rows: {len(df):,}")

        print("  [3/3] regressions ...")
        run_models(df, cfg, ec)
        del df; gc.collect()
    except Exception as e:
        print(f"  [FATAL] {country}: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    for c in ['USA', 'CHN']:
        run_country(c)
    print("\n" + "="*60)
    print("  FILES WRITTEN:")
    for p in WRITTEN:
        print("   -", p, "OK" if os.path.exists(p) else "MISSING!")
    if not WRITTEN:
        print("   (none — scroll up for the [ERROR]/[FATAL] message)")
    print("="*60)
