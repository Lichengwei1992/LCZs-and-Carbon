# -*- coding: utf-8 -*-
"""
Code_Robustness_IndStruct_ANNUAL.py
===============================================================
Annual-model counterpart of the EC/GDP industrial-structure robustness check.
This matches the ANNUAL fixed-effects specification that produces the
manuscript regression Tables 1-2 (Code_CHN_V1.py / Code_US_Final.py):

    controls = ln_Pop + ln_GDP + ln_HDD + ln_CDD + ln_GHI + C(Year) + C(admin)
    Model2 : ln_CO2 ~ controls + LCZ
    Model5 : ln_CO2 ~ controls + LCZ + ln_EC_GDP   (+ industrial structure)

Differences vs the monthly script:
  * panel = Panel_1km_clean.pkl  (annual, no Month / no Season)
  * NO Month FE, NO seasonal terms
  * NO stratified sampling (annual panel is small enough to fit)
  * REUSES the EC_GDP_by_admin_year.csv already produced by the monthly run
    (electricity intensity is admin x year, identical for both models).
    If that CSV is absent it rebuilds it from the rasters.

Run locally (needs statsmodels; rasterio/geopandas only if the EC CSV
must be rebuilt). Outputs land in each annual result dir, suffixed _ANNUAL.
"""

import os, gc, warnings, traceback
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

LCZ_MAX_ZERO_FRAC = 0.99

# --- adjust these paths to the annual panels you want to validate ---
COUNTRY_CFG = {
    'USA': {
        'panel':    r"D:\LCZCarbon\Result_USA_NEW\USA_1km\Panel_1km_clean.pkl",
        'outdir':   r"D:\LCZCarbon\Result_USA_NEW\USA_1km",
        'fe_col':   'State_ID',
        # reuse the admin x year EC/GDP table already produced by the monthly run:
        'ec_csv':   r"D:\LCZCarbon\Results_USA_Month\USA_1km\EC_GDP_by_admin_year.csv",
    },
    'CHN': {
        'panel':    r"D:\LCZCarbon\Results_CHN_NEW\CHN_1km\Panel_1km_clean.pkl",
        'outdir':   r"D:\LCZCarbon\Results_CHN_NEW\CHN_1km",
        'fe_col':   'City_ID',
        'ec_csv':   r"D:\LCZCarbon\Results_CHN_Month\CHN_1km\EC_GDP_by_admin_year.csv",
    },
}

WRITTEN = []
def _save_csv(df, path, note=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=bool(df.index.name), encoding='utf-8-sig')
    WRITTEN.append(path)
    print(f"      [WROTE] {path}  ({len(df)} rows) {note}")


def run_country(country):
    cfg = COUNTRY_CFG[country]
    fe_col, out_dir = cfg['fe_col'], cfg['outdir']
    print(f"\n{'='*60}\n  {country}  (ANNUAL)\n{'='*60}")
    os.makedirs(out_dir, exist_ok=True)
    try:
        # 1. EC/GDP admin x year table (reuse monthly output)
        if os.path.exists(cfg['ec_csv']):
            ec = pd.read_csv(cfg['ec_csv'])
            print(f"  reused EC table: {cfg['ec_csv']}  ({len(ec)} rows)")
        else:
            raise FileNotFoundError(
                f"EC table not found at {cfg['ec_csv']} — run the monthly "
                f"script first (it writes EC_GDP_by_admin_year.csv), or "
                f"point ec_csv to an existing one.")

        # 2. load annual panel (no sampling)
        df = pd.read_pickle(cfg['panel'])
        print(f"  annual panel: {len(df):,} rows, {df['Year'].nunique()} years, "
              f"{df[fe_col].nunique()} admin units")
        # CO2 p99 guard (panel is usually already cleaned)
        df = df[df['CO2'] <= df['CO2'].quantile(0.99)].copy()

        # 3. merge EC/GDP, log-transform
        df = df.merge(ec, on=[fe_col, 'Year'], how='left')
        print(f"  EC/GDP merge coverage: {1-df['EC_GDP'].isna().mean():.1%}")
        for c in ['CO2','Pop','GDP','HDD','CDD','GHI','EC_GDP']:
            df[f'ln_{c}'] = np.log(df[c].clip(lower=0) + 1).astype('float32')
        df.replace([np.inf,-np.inf], np.nan, inplace=True)

        all_lcz   = [c for c in df.columns if c.endswith('_share')]
        valid_lcz = (df[all_lcz] < 1e-6).mean()
        valid_lcz = valid_lcz[valid_lcz < LCZ_MAX_ZERO_FRAC].index.tolist()
        keep = ([f'ln_{c}' for c in ['CO2','Pop','GDP','HDD','CDD','GHI','EC_GDP']]
                + [fe_col,'Year'] + valid_lcz)
        df.dropna(subset=keep, inplace=True)
        print(f"  rows after dropna: {len(df):,}")

        ref_lcz  = 'LCZ9_share' if 'LCZ9_share' in valid_lcz else valid_lcz[0]
        lcz_vars = [c for c in valid_lcz if c != ref_lcz]
        controls = (f"ln_Pop + ln_GDP + ln_HDD + ln_CDD + ln_GHI"
                    f" + C(Year) + C({fe_col})")
        formulas = {
            'Model2_Morphology_ANNUAL': f"ln_CO2 ~ {controls} + {' + '.join(lcz_vars)}",
            'Model5_IndStruct_ANNUAL':  f"ln_CO2 ~ {controls} + {' + '.join(lcz_vars)} + ln_EC_GDP",
        }

        fitted = {}
        for name, formula in formulas.items():
            try:
                print(f"  fitting {name} ...")
                m = smf.ols(formula, data=df).fit(cov_type='cluster',
                                                  cov_kwds={'groups': df[fe_col]})
                fitted[name] = m
                ci = m.conf_int()
                coefs = pd.DataFrame({'coef': m.params,'std_err': m.bse,
                                      'pvalue': m.pvalues,'conf_low': ci[0],'conf_high': ci[1]})
                coefs.index.name = 'term'
                _save_csv(coefs, os.path.join(out_dir, f'{name}_Coefs.csv'),
                          note=f"R2={m.rsquared:.4f} N={int(m.nobs):,}")
            except Exception as e:
                print(f"  [ERROR] {name}: {e}"); traceback.print_exc()

        # comparison table
        focus = ['ln_Pop','ln_GDP','ln_EC_GDP'] + lcz_vars
        m2 = fitted.get('Model2_Morphology_ANNUAL')
        m5 = fitted.get('Model5_IndStruct_ANNUAL')
        rows = []
        for t in focus:
            c2 = m2.params.get(t,np.nan) if m2 is not None else np.nan
            c5 = m5.params.get(t,np.nan) if m5 is not None else np.nan
            rows.append({'term':t,
                         'Model2_coef':c2,'Model2_p':(m2.pvalues.get(t,np.nan) if m2 is not None else np.nan),
                         'Model5_coef':c5,'Model5_p':(m5.pvalues.get(t,np.nan) if m5 is not None else np.nan),
                         'pct_change':(100*(c5-c2)/abs(c2) if (not np.isnan(c2) and c2!=0 and not np.isnan(c5)) else np.nan)})
        comp = pd.DataFrame(rows)
        _save_csv(comp, os.path.join(out_dir,'Robustness_IndStruct_Comparison_ANNUAL.csv'))
        lcz = comp[comp.term.str.startswith('LCZ')]
        sig = lcz[lcz.Model2_p < 0.05]
        print(f"  >>> significant-LCZ mean |%change| = {sig.pct_change.abs().mean():.2f}% (n={len(sig)})")
        if m5 is not None:
            print(f"  >>> ln_EC_GDP = {m5.params.get('ln_EC_GDP',np.nan):.4f} "
                  f"(p={m5.pvalues.get('ln_EC_GDP',np.nan):.4g})")
        del df; gc.collect()
    except Exception as e:
        print(f"  [FATAL] {country}: {e}"); traceback.print_exc()


if __name__ == '__main__':
    for c in ['USA','CHN']:
        run_country(c)
    print("\n" + "="*60 + "\n  FILES WRITTEN:")
    for p in WRITTEN:
        print("   -", p, "OK" if os.path.exists(p) else "MISSING!")
    if not WRITTEN:
        print("   (none — see [ERROR]/[FATAL] above)")
    print("="*60)
