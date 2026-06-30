# -*- coding: utf-8 -*-
"""
Code_Diagnose_PopGDP.py
========================
诊断面板数据中 Pop 和 GDP 的逐年分布，判断是否存在跨年系统性异常。
同时直接读取 GDP Per Capita 栅格文件，验证原始栅格是否已修复。

直接在 PyCharm 运行，输出逐年统计表并保存 CSV。
运行时间：约 2-5 分钟（从大 pkl 中分层抽样）
"""

import pandas as pd
import numpy as np
import os

# ─── 路径配置 ───
USA_PANEL    = r"D:\LCZCarbon\Results_USA_Month\USA_1km\Panel_Monthly_clean.pkl"
CHN_PANEL    = r"D:\LCZCarbon\Results_CHN_Month\CHN_1km\Panel_Monthly_clean.pkl"
GDP_PC_USA   = r"D:\LCZCarbon\GDP Per Capita\USA"
GDP_PC_CHN   = r"D:\LCZCarbon\GDP Per Capita\CHN"
OUT_DIR      = r"D:\LCZCarbon\Results_DescStats_v2"
SAMPLE_PER_YEAR = 30_000   # 每年抽样行数

os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════
# Part 1: 从面板 pkl 诊断 Pop 和 GDP 分布
# ═══════════════════════════════════════════════════
all_results = []

for country, pkl_path in [('USA', USA_PANEL), ('CHN', CHN_PANEL)]:
    print(f"\n{'='*65}")
    print(f"  PANEL DIAGNOSIS  {country}: {pkl_path}")
    print(f"{'='*65}")

    df = pd.read_pickle(pkl_path)
    print(f"  Total rows: {len(df):,}")

    # 按年分层抽样（loop 写法，避免 pandas groupby.apply 的 KeyError bug）
    sampled_parts = []
    for yr in sorted(df['Year'].unique()):
        g = df[df['Year'] == yr]
        sampled_parts.append(
            g.sample(n=min(SAMPLE_PER_YEAR, len(g)), random_state=42)
        )
    df_s = pd.concat(sampled_parts, ignore_index=True)
    del df
    print(f"  Stratified sample: {len(df_s):,} rows\n")

    print(f"  {'Year':<6} {'Pop_median':>11} {'Pop_p99':>10} {'Pop_max':>10} "
          f"{'GDP_median':>11} {'GDP_p99':>10} {'GDP_max':>10}  Notes")
    print(f"  {'-'*82}")

    prev_pop = None
    prev_gdp = None

    for yr in sorted(df_s['Year'].unique()):
        sub = df_s[df_s['Year'] == yr]

        pop_med = sub['Pop'].median()
        pop_p99 = sub['Pop'].quantile(0.99)
        pop_max = sub['Pop'].max()
        gdp_med = sub['GDP'].median()
        gdp_p99 = sub['GDP'].quantile(0.99)
        gdp_max = sub['GDP'].max()

        notes = []
        if pop_med == 255:
            notes.append('Pop=255(uint8!)')
        elif prev_pop and abs(pop_med / (prev_pop + 1e-9) - 1) > 0.4:
            notes.append('Pop_ANOMALY')

        if prev_gdp and abs(gdp_med / (prev_gdp + 1e-9) - 1) > 0.4:
            notes.append('GDP_ANOMALY')

        note_str = ' '.join(notes)
        print(f"  {yr:<6} {pop_med:>11.1f} {pop_p99:>10.1f} {pop_max:>10.0f} "
              f"{gdp_med:>11.4f} {gdp_p99:>10.4f} {gdp_max:>10.2f}  {note_str}")

        all_results.append({
            'Country':    country,
            'Year':       yr,
            'Pop_median': pop_med,
            'Pop_p99':    pop_p99,
            'Pop_max':    pop_max,
            'GDP_median': gdp_med,
            'GDP_p99':    gdp_p99,
            'GDP_max':    gdp_max,
            'Notes':      note_str,
        })

        prev_pop = pop_med
        prev_gdp = gdp_med

# 保存面板诊断结果
df_out = pd.DataFrame(all_results)
out_path = os.path.join(OUT_DIR, "Diagnose_PopGDP_ByYear.csv")
df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f"\n  Saved: {out_path}")

# ═══════════════════════════════════════════════════
# Part 2: 直接读取 GDP Per Capita 栅格，验证是否已修复
# ═══════════════════════════════════════════════════
print(f"\n\n{'='*65}")
print("  GDP PER CAPITA RASTER DIAGNOSIS (direct TIF read)")
print(f"{'='*65}")

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("  [WARNING] rasterio not found – skipping TIF diagnosis.")
    print("  Install via:  pip install rasterio")

if HAS_RASTERIO:
    gdp_configs = [
        ('USA', GDP_PC_USA, 'USA'),
        ('CHN', GDP_PC_CHN, 'CHN'),
    ]
    gdp_raster_results = []

    for country, gdp_dir, suffix in gdp_configs:
        print(f"\n  {country}")
        print(f"  {'Year':<6} {'p50':>10} {'p95':>10} {'p99':>10}  Flag")
        print(f"  {'-'*50}")

        p50_list = []
        for yr in range(2000, 2020):
            path = os.path.join(gdp_dir, f"GDP_PC_{yr}_{suffix}.tif")
            if not os.path.exists(path):
                print(f"  {yr:<6}  MISSING")
                continue
            with rasterio.open(path) as src:
                arr = src.read(1).astype('float32').ravel()
            valid = arr[(arr > 0) & (arr < 1e15)]
            if len(valid) == 0:
                print(f"  {yr:<6}  NO VALID DATA")
                continue
            p50 = float(np.percentile(valid, 50))
            p95 = float(np.percentile(valid, 95))
            p99 = float(np.percentile(valid, 99))
            p50_list.append(p50)

            # Flag: if current year p50 is >5x the running median of all previous years
            if len(p50_list) >= 3:
                base = np.median(p50_list[:-1])
                flag = " *** ANOMALY ***" if p50 > base * 5 else ""
            else:
                flag = ""

            print(f"  {yr:<6} {p50:>10.5f} {p95:>10.4f} {p99:>10.4f} {flag}")
            gdp_raster_results.append({
                'Country': country, 'Year': yr,
                'GDP_PC_p50': p50, 'GDP_PC_p95': p95, 'GDP_PC_p99': p99,
            })

    # 保存栅格诊断结果
    df_gdp_raster = pd.DataFrame(gdp_raster_results)
    out_gdp = os.path.join(OUT_DIR, "Diagnose_GDP_PC_Raster.csv")
    df_gdp_raster.to_csv(out_gdp, index=False, encoding='utf-8-sig')
    print(f"\n  Saved: {out_gdp}")

# ═══════════════════════════════════════════════════
# Part 3: 汇总结论
# ═══════════════════════════════════════════════════
print(f"\n\n{'='*65}")
print("  SUMMARY")
print(f"{'='*65}")

for country in ['USA', 'CHN']:
    sub = df_out[df_out['Country'] == country]
    pop_capped   = (sub['Pop_median'] == 255).sum()
    pop_anomaly  = sub['Notes'].str.contains('Pop_ANOMALY').sum()
    gdp_anomaly  = sub['Notes'].str.contains('GDP_ANOMALY').sum()

    print(f"\n  {country}:")
    print(f"    Pop=255 (uint8 saturation) years : {pop_capped}/{len(sub)} "
          f"→ {'CRITICAL – must fix Pop rasters' if pop_capped > 0 else 'OK'}")
    print(f"    Pop year-on-year anomaly years   : {pop_anomaly}/{len(sub)}")
    print(f"    GDP year-on-year anomaly years   : {gdp_anomaly}/{len(sub)}")

print("""
  ─────────────────────────────────────────────────
  RECOMMENDED FIX ORDER
  ─────────────────────────────────────────────────
  1. Run Code_Clip_Population.py
       → regenerates USAPOP / CHNPOP as float32 from LandScan Global
       → fixes uint8 saturation AND year-specific anomalies

  2. Run Code_Fix_GDP_PerCapita.py
       → recomputes GDP Per Capita = GDP_total / pop (LandScan, correctly resampled)
       → fixes 2009/2018/2019 anomaly (~30–50× inflation)

  3. Re-run Code_US_Monthly.py  (rebuild USA Panel_Monthly_clean.pkl)
     Re-run Code_CHN_V1.py      (rebuild CHN Panel_Monthly_clean.pkl)

  4. Re-run Code_Monthly_Regression.py

  5. Re-run Code_Descriptive_Stats_v2.py  (RUN_FIGS = {1,2,3,4,5,6,7,8})
  ─────────────────────────────────────────────────
""")
