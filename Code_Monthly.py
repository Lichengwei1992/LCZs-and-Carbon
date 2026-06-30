# -*- coding: utf-8 -*-
"""
LCZ 与城市碳排放关系研究 - 月度版（美国 + 中国）
==================================================
先完成美国，再完成中国，共用同一套流水线代码。

  数据规律：
  [CO2]  月度文件：{prefix}_{YYMM:04d}.tif
         YYMM = (year-2000)*100 + month
         例：2001年3月 → 0103，放在 .../US/2001/ 或 .../CHN/2001/ 下
  [GHI]  月度独立文件（无需拼接）：
         USA: .../USA_GHI_Monthly/USA-{year}/USA_{year}_{month:02d}.tif
         CHN: .../CHN_GHI_Monthly/China-{year}/GHI_{year}_{month:02d}.tif
  [HDD/CDD] 年度多波段文件，按波段号读取对应月份（Band 1=1月…12=12月）
  [LCZ/Pop/GDP] 年度数据，同年12个月共用

  回归模型（1km，州/市 FE + Year FE + Month FE）：
  Model 1 – Baseline:   控制变量 + C(Year) + C(Month) + C(FE_ID)
  Model 2 – Morphology: + LCZ 主效应（基准: LCZ9）
  Model 3 – Seasonal:   + Season × LCZ 交互项
             Season: Winter(12,1,2) / Spring(3,4,5) / Summer(6,7,8) / Fall(9,10,11)
             基准季节: Spring

  输出路径：
    美国: D:/LCZCarbon/Results_USA_Month/USA_1km
    中国: D:/LCZCarbon/Results_CHN_Month/CHN_1km
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


# ================== 1. 国家配置 ==================

YEARS  = range(2000, 2020)
MONTHS = range(1, 13)

# 季节映射，基准为 Spring
SEASON_MAP = {
    12: 'Winter', 1: 'Winter',  2: 'Winter',
    3:  'Spring', 4: 'Spring',  5: 'Spring',
    6:  'Summer', 7: 'Summer',  8: 'Summer',
    9:  'Fall',  10: 'Fall',   11: 'Fall',
}
SEASON_BASE = 'Spring'   # 基准季节

CO2_PERCENTILE    = 99
LCZ_MAX_ZERO_FRAC = 0.99
MAX_ROWS_PER_YM   = None   # 每年月最大像元数，None = 不抽样

CONFIGS = {
    'USA': {
        # 月度 CO2：D:\LCZCarbon\US\{year}\USA_odiac2024_1km_excl_intl_{YYMM:04d}.tif
        'co2_base':    r"D:\LCZCarbon\US",
        'co2_prefix':  "USA_odiac2024_1km_excl_intl",
        # 年度 Pop / GDP
        'pop_dir':     r"D:\LCZCarbon\Population\USAPOP",
        'pop_tpl':     "POP{year}_USA.tif",
        'gdp_dir':     r"D:\LCZCarbon\GDP Per Capita\USA",
        'gdp_tpl':     "GDP_PC_{year}_USA.tif",
        # 年度 LCZ
        'lcz_dir':     r"E:\Data_All\LCZ_Carbon_Form\US\Final_LCZ_Maps",
        'lcz_tpl':     "TP_{year}.tif",
        # 年度多波段 HDD/CDD（Band=月份）
        'hdd_dir':     r"D:\LCZCarbon\HDD_CDD",
        'hdd_tpl':     "USA_HDD_{year}.tif",
        'cdd_tpl':     "USA_CDD_{year}.tif",
        # 月度 GHI：.../USA_GHI_Monthly/USA-{year}/USA_{year}_{month:02d}.tif
        'ghi_base':    r"D:\LCZCarbon\GHI\USA_GHI_Monthly",
        'ghi_subdir':  "USA-{year}",
        'ghi_tpl':     "USA_{year}_{month:02d}.tif",
        # 行政边界（州）
        'boundary_shp':r"D:\LCZCarbon\US_State_Boundaries\US_State_Boundaries.shp",
        'boundary_name_kw': ['NAME', 'STATE', 'STNAME', 'NAME_1', 'STATENAME'],
        'fe_col':      'State_ID',
        'fe_map_name': 'State_ID_Mapping.csv',
        # 建成区边界
        'urban_dir':   r"E:\Data_All\Urban Footprint\Cities_2000_2022",
        # 输出
        'output_base': r"D:\LCZCarbon\Results_USA_Month",
        'out_subdir':  'USA_1km',
        'chk_subdir':  'checkpoints_USA_monthly',
    },
    'CHN': {
        # 月度 CO2：D:\LCZCarbon\CHN\{year}\CHN_odiac2024_1km_excl_intl_{YYMM:04d}.tif
        'co2_base':    r"D:\LCZCarbon\CHN",
        'co2_prefix':  "CHN_odiac2024_1km_excl_intl",
        # 年度 Pop / GDP
        'pop_dir':     r"D:\LCZCarbon\Population\CHNPOP",
        'pop_tpl':     "POP{year}_CHN.tif",
        'gdp_dir':     r"D:\LCZCarbon\GDP Per Capita\CHN",
        'gdp_tpl':     "GDP_PC_{year}_CHN.tif",
        # 年度 LCZ
        'lcz_dir':     r"E:\Data_All\LCZ_Carbon_Form\China\LCZ_CHN",
        'lcz_tpl':     "CHN_LCZ_{year}.tif",
        # 年度多波段 HDD/CDD
        'hdd_dir':     r"D:\LCZCarbon\HDD_CDD",
        'hdd_tpl':     "China_HDD_{year}.tif",
        'cdd_tpl':     "China_CDD_{year}.tif",
        # 月度 GHI：.../CHN_GHI_Monthly/China-{year}/GHI_{year}_{month:02d}.tif
        'ghi_base':    r"D:\LCZCarbon\GHI\CHN_GHI_Monthly",
        'ghi_subdir':  "China-{year}",
        'ghi_tpl':     "GHI_{year}_{month:02d}.tif",
        # 行政边界（地级市）
        'boundary_shp':r"D:\LCZCarbon\ChinaBoundary\ChinaBoundaryNew.shp",
        'boundary_name_kw': ['NAME', 'CITY', '市', '名称', 'DIJI', 'PREFEC', 'DNAME', 'CNAME'],
        'fe_col':      'City_ID',
        'fe_map_name': 'City_ID_Mapping.csv',
        # 建成区边界
        'urban_dir':   r"E:\Data_All\Urban Footprint\Cities_2000_2022",
        # 输出
        'output_base': r"D:\LCZCarbon\Results_CHN_Month",
        'out_subdir':  'CHN_1km',
        'chk_subdir':  'checkpoints_CHN_monthly',
    },
}


# ================== 2. 工具函数 ==================

def co2_filename(prefix, year, month):
    """生成CO2月度文件名：YYMM = (year-2000)*100 + month"""
    yymm = (year - 2000) * 100 + month
    return f"{prefix}_{yymm:04d}.tif"


def load_and_resample(path, ref_meta, resampling=Resampling.bilinear):
    with rasterio.open(path) as src:
        dst = np.zeros((ref_meta['height'], ref_meta['width']), dtype='float32')
        reproject(
            source=rasterio.band(src, 1), destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=ref_meta['transform'], dst_crs=ref_meta['crs'],
            resampling=resampling
        )
        return dst


def read_climate_band(path, month, ref_meta):
    """从年度多波段 HDD/CDD 文件中读取指定月份波段（Band 1=1月…12=12月）"""
    with rasterio.open(path) as src:
        band_idx = min(month, src.count)   # 防越界
        data = src.read(band_idx).astype('float32')
        dst = np.zeros((ref_meta['height'], ref_meta['width']), dtype='float32')
        reproject(
            source=data, destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=ref_meta['transform'], dst_crs=ref_meta['crs'],
            resampling=Resampling.bilinear
        )
        return dst


def detect_lcz_encoding(lcz_path):
    with rasterio.open(lcz_path) as src:
        win = rasterio.windows.Window(0, 0, min(500, src.width), min(500, src.height))
        sample = src.read(1, window=win)
        vals = np.unique(sample[sample > 0])
    if any(101 <= v <= 107 for v in vals) and not any(11 <= v <= 17 for v in vals):
        print("    [LCZ编码] 101-107（替代编码）")
        return list(range(1, 11)) + list(range(101, 108))
    print("    [LCZ编码] 11-17（标准编码）")
    return list(range(1, 18))


def calculate_lcz_shares(lcz_path, ref_meta, all_classes):
    with rasterio.open(lcz_path) as src:
        lcz_data = src.read(1)
        src_tf, src_crs = src.transform, src.crs
    shares = {}
    for code in all_classes:
        binary = (lcz_data == code).astype('float32')
        dst = np.zeros((ref_meta['height'], ref_meta['width']), dtype='float32')
        reproject(
            source=binary, destination=dst,
            src_transform=src_tf, src_crs=src_crs,
            dst_transform=ref_meta['transform'], dst_crs=ref_meta['crs'],
            resampling=Resampling.average
        )
        shares[f'LCZ{code}_share'] = dst
        del binary
    del lcz_data
    gc.collect()
    return shares


def get_boundary_raster(shp_path, name_keywords, fe_col, map_name, ref_meta, out_dir):
    """栅格化行政边界，生成 FE_ID 数组，保存 ID 映射表"""
    print(f"  栅格化行政边界: {os.path.basename(shp_path)}")
    gdf = gpd.read_file(shp_path)
    if gdf.crs != ref_meta['crs']:
        gdf = gdf.to_crs(ref_meta['crs'])

    name_col = next(
        (c for c in gdf.columns
         if any(kw in c.upper() for kw in name_keywords) and c != 'geometry'),
        None
    )
    if not name_col:
        name_col = next(
            (c for c in gdf.columns if c != 'geometry' and gdf[c].dtype == object),
            None
        )
    if not name_col:
        raise ValueError(f"找不到名称列，现有列: {list(gdf.columns)}")
    print(f"    名称列: '{name_col}'，共 {len(gdf)} 个单元")

    gdf = gdf.sort_values(name_col).reset_index(drop=True)
    gdf['ID_Num'] = range(1, len(gdf) + 1)

    mapping = gdf[['ID_Num', name_col]].copy()
    mapping.columns = [fe_col, name_col]
    mapping.to_csv(os.path.join(out_dir, map_name), index=False, encoding='utf-8-sig')
    print(f"    映射表已保存: {map_name}")

    burned = features.rasterize(
        ((geom, val) for geom, val in zip(gdf.geometry, gdf['ID_Num'])),
        out_shape=(ref_meta['height'], ref_meta['width']),
        transform=ref_meta['transform'], fill=0, dtype='int16'
    )
    print(f"    有效像元: {(burned > 0).sum():,}")
    return burned


def get_urban_year_mask(urban_dir, year, ref_meta, study_bbox_geom):
    shp_path = os.path.join(urban_dir, f"Cities_{year}.shp")
    empty = np.zeros((ref_meta['height'], ref_meta['width']), dtype=bool)
    if not os.path.exists(shp_path):
        return empty
    gdf = gpd.read_file(shp_path)
    if len(gdf) == 0:
        return empty
    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:4326')
    bbox_shp = (
        gpd.GeoSeries([study_bbox_geom], crs=ref_meta['crs'])
        .to_crs(gdf.crs).iloc[0]
    )
    gdf = gdf[gdf.geometry.intersects(bbox_shp)].to_crs(ref_meta['crs'])
    if len(gdf) == 0:
        return empty
    burned = features.rasterize(
        ((geom, 1) for geom in gdf.geometry if geom is not None),
        out_shape=(ref_meta['height'], ref_meta['width']),
        transform=ref_meta['transform'], fill=0, dtype='uint8'
    )
    return burned.astype(bool)


# ================== 3. 回归函数 ==================

def run_monthly_regression(df_input, fe_col, out_dir, label):
    """
    三个月度回归模型（Month FE + Year FE + FE_ID，基准季节: Spring）
    Model 1: Baseline
    Model 2: + LCZ 主效应
    Model 3: + Season × LCZ 交互项
    """
    df = df_input.copy()
    print(f"\n[{label}] 回归准备，样本量: {len(df):,}")

    for col in ['CO2', 'Pop', 'GDP', 'HDD', 'CDD', 'GHI']:
        df[f'ln_{col}'] = np.log(df[col] + 1).astype('float32')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Season 列，以 Spring 为基准
    seasons = [s for s in ['Spring', 'Summer', 'Fall', 'Winter'] if s != SEASON_BASE]
    cat_order = [SEASON_BASE] + seasons
    df['Season'] = df['Month'].map(SEASON_MAP)
    df['Season'] = pd.Categorical(df['Season'], categories=cat_order, ordered=False)

    # LCZ 近零列筛选
    all_lcz   = [c for c in df.columns if '_share' in c]
    zero_frac = (df[all_lcz] < 1e-6).mean()
    means     = df[all_lcz].mean()
    prev      = (df[all_lcz] >= 1e-6).mean()
    valid_lcz = zero_frac[zero_frac < LCZ_MAX_ZERO_FRAC].index.tolist()
    dropped   = [c for c in all_lcz if c not in valid_lcz]

    print(f"  LCZ 保留 {len(valid_lcz)} 个，剔除 {len(dropped)} 个: "
          f"{[c.replace('_share','') for c in dropped]}")

    pd.DataFrame({
        'LCZ_col': all_lcz, 'mean_share': means.values,
        'prevalence': prev.values, 'zero_frac': zero_frac.values,
        'retained': [c in valid_lcz for c in all_lcz]
    }).to_csv(os.path.join(out_dir, 'LCZ_Coverage_Filter.csv'),
              index=False, encoding='utf-8-sig')

    keep = [f'ln_{c}' for c in ['CO2','Pop','GDP','HDD','CDD','GHI']] + \
           [fe_col, 'Year', 'Month', 'Season'] + valid_lcz
    df.dropna(subset=keep, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  清洗后: {len(df):,} 行")

    df[['CO2','Pop','GDP','HDD','CDD','GHI'] + valid_lcz].describe().to_csv(
        os.path.join(out_dir, 'Sample_Stats.csv'), encoding='utf-8-sig'
    )

    # 基准 LCZ
    ref_lcz = 'LCZ9_share' if 'LCZ9_share' in valid_lcz else valid_lcz[0]
    lcz_vars = [c for c in valid_lcz if c != ref_lcz]
    lcz_f    = " + ".join(lcz_vars)

    # Season × LCZ 交互（基准: Spring）
    season_inter = " + ".join(
        [f"C(Season, Treatment('{SEASON_BASE}')):{v}" for v in lcz_vars]
    )

    controls = (f"ln_Pop + ln_GDP + ln_HDD + ln_CDD + ln_GHI "
                f"+ C(Month) + C(Year) + C({fe_col})")

    models_def = {
        'Model1_Baseline':   f"ln_CO2 ~ {controls}",
        'Model2_Morphology': f"ln_CO2 ~ {controls} + {lcz_f}",
        'Model3_Seasonal':   f"ln_CO2 ~ {controls} + {lcz_f} + {season_inter}",
    }

    for name, formula in models_def.items():
        try:
            print(f"  Running {name}...")
            m = smf.ols(formula, data=df).fit(
                cov_type='cluster',
                cov_kwds={'groups': df[fe_col]}
            )

            with open(os.path.join(out_dir, f"{name}_Summary.txt"), 'w',
                      encoding='utf-8') as f:
                f.write(f"面板: 像元 × 年月 | FE: {fe_col} + Year + Month\n")
                f.write(f"LCZ 基准: {ref_lcz} | Season 基准: {SEASON_BASE}\n")
                f.write(f"剔除LCZ: {dropped}\n\n")
                f.write(m.summary().as_text())

            ci = m.conf_int()
            pd.DataFrame({
                'coef': m.params, 'std_err': m.bse,
                'pvalue': m.pvalues, 'conf_low': ci[0], 'conf_high': ci[1],
            }).to_csv(os.path.join(out_dir, f"{name}_Coefs.csv"),
                      encoding='utf-8-sig')

            # 单独提取 Month FE 系数（方便绘制季节曲线）
            month_idx = [p for p in m.params.index if 'C(Month)' in str(p)]
            if month_idx:
                pd.DataFrame({
                    'coef': m.params[month_idx],
                    'pvalue': m.pvalues[month_idx]
                }).to_csv(os.path.join(out_dir, f"{name}_MonthFE.csv"),
                          encoding='utf-8-sig')

            print(f"    R²={m.rsquared:.4f}, N={int(m.nobs):,}, "
                  f"条件数={m.condition_number:.2e}")

        except MemoryError:
            print(f"  {name}: 内存溢出，跳过。")
        except Exception as e:
            print(f"  {name}: 出错 → {e}")


# ================== 4. 单国处理主流程 ==================

def run_country(country, cfg):
    print("\n" + "=" * 65)
    print(f"  开始处理: {country}")
    print("=" * 65)

    out_dir = os.path.join(cfg['output_base'], cfg['out_subdir'])
    chk_dir = os.path.join(cfg['output_base'], cfg['chk_subdir'])
    for d in [out_dir, chk_dir]:
        os.makedirs(d, exist_ok=True)

    fe_col = cfg['fe_col']

    # --- 基准栅格（用2000年1月CO2确定 ref_meta）---
    ref_co2 = os.path.join(
        cfg['co2_base'], "2000",
        co2_filename(cfg['co2_prefix'], 2000, 1)
    )
    if not os.path.exists(ref_co2):
        raise FileNotFoundError(f"找不到基准CO2文件: {ref_co2}")
    with rasterio.open(ref_co2) as src:
        ref_meta = src.meta.copy()
    print(f"基准栅格: {ref_meta['width']} × {ref_meta['height']} px, "
          f"CRS={ref_meta['crs']}")

    bounds = array_bounds(ref_meta['height'], ref_meta['width'], ref_meta['transform'])
    study_bbox = box(*bounds)

    # --- 静态数据 ---
    print("\n[Phase 1] 静态数据...")
    fe_raster = get_boundary_raster(
        cfg['boundary_shp'], cfg['boundary_name_kw'],
        fe_col, cfg['fe_map_name'], ref_meta, out_dir
    )

    # --- 逐年月处理 ---
    print("\n[Phase 2] 逐年月处理...")
    lcz_classes = None
    # 年度缓存（同年12个月复用）
    lcz_cache = {}
    pop_cache = {}
    gdp_cache = {}
    urban_cache = {}

    for year in YEARS:
        f_lcz = os.path.join(cfg['lcz_dir'], cfg['lcz_tpl'].format(year=year))
        f_pop = os.path.join(cfg['pop_dir'], cfg['pop_tpl'].format(year=year))
        f_gdp = os.path.join(cfg['gdp_dir'], cfg['gdp_tpl'].format(year=year))

        # LCZ shares（年度，同年复用）
        if year not in lcz_cache:
            if not os.path.exists(f_lcz):
                print(f"  [{year}] 缺失 LCZ，跳过整年")
                for m in MONTHS:
                    chk = os.path.join(chk_dir, f"panel_{year}_{m:02d}.pkl")
                    if not os.path.exists(chk):
                        open(chk + '.skip', 'w').close()
                continue
            if lcz_classes is None:
                lcz_classes = detect_lcz_encoding(f_lcz)
            print(f"  [{year}] 计算 LCZ shares...")
            lcz_cache[year] = calculate_lcz_shares(f_lcz, ref_meta, lcz_classes)

        # Pop / GDP（年度，同年复用）
        if year not in pop_cache:
            pop_cache[year] = (load_and_resample(f_pop, ref_meta)
                               if os.path.exists(f_pop) else None)
            gdp_cache[year] = (load_and_resample(f_gdp, ref_meta)
                               if os.path.exists(f_gdp) else None)

        # 建成区掩膜（年度，同年复用）
        if year not in urban_cache:
            urban_cache[year] = get_urban_year_mask(
                cfg['urban_dir'], year, ref_meta, study_bbox
            )

        for month in MONTHS:
            chk = os.path.join(chk_dir, f"panel_{year}_{month:02d}.pkl")
            if os.path.exists(chk):
                print(f"  [断点] 跳过 {year}-{month:02d}")
                continue
            if os.path.exists(chk + '.skip'):
                continue

            # CO2 月度文件
            f_co2 = os.path.join(
                cfg['co2_base'], str(year),
                co2_filename(cfg['co2_prefix'], year, month)
            )
            # GHI 月度文件
            ghi_subdir = cfg['ghi_subdir'].format(year=year)
            f_ghi = os.path.join(
                cfg['ghi_base'], ghi_subdir,
                cfg['ghi_tpl'].format(year=year, month=month)
            )
            # HDD / CDD
            f_hdd = os.path.join(cfg['hdd_dir'], cfg['hdd_tpl'].format(year=year))
            f_cdd = os.path.join(cfg['hdd_dir'], cfg['cdd_tpl'].format(year=year))

            missing = [(p, lb) for p, lb in [
                (f_co2, 'CO2'), (f_ghi, 'GHI')
            ] if not os.path.exists(p)]
            if missing:
                for p, lb in missing:
                    print(f"  [{year}-{month:02d}] 缺失 {lb}: {p}，跳过")
                continue

            if pop_cache.get(year) is None or gdp_cache.get(year) is None:
                print(f"  [{year}-{month:02d}] Pop/GDP 缺失，跳过")
                continue

            print(f"  >>> {year}-{month:02d}")

            # 读取 CO2
            with rasterio.open(f_co2) as src:
                co2_data = src.read(1).astype('float32')
                nodata   = src.nodata

            co2_valid = (
                (co2_data != nodata) & (co2_data > 0)
                if nodata is not None else (co2_data > 0)
            )
            valid_mask = co2_valid & urban_cache[year]

            n_valid = valid_mask.sum()
            print(f"    有效像元: {n_valid:,}")
            if n_valid == 0:
                del co2_data; gc.collect()
                continue

            # 读取月度 GHI
            ghi_data = load_and_resample(f_ghi, ref_meta, Resampling.bilinear)

            # 读取月度 HDD/CDD（从年度多波段文件取对应波段）
            hdd_data = (read_climate_band(f_hdd, month, ref_meta)
                        if os.path.exists(f_hdd)
                        else np.zeros((ref_meta['height'], ref_meta['width']),
                                      dtype='float32'))
            cdd_data = (read_climate_band(f_cdd, month, ref_meta)
                        if os.path.exists(f_cdd)
                        else np.zeros((ref_meta['height'], ref_meta['width']),
                                      dtype='float32'))

            # 组装 DataFrame
            data = {
                'CO2':    co2_data[valid_mask],
                'Pop':    pop_cache[year][valid_mask],
                'GDP':    gdp_cache[year][valid_mask],
                'HDD':    hdd_data[valid_mask],
                'CDD':    cdd_data[valid_mask],
                'GHI':    ghi_data[valid_mask],
                fe_col:   fe_raster[valid_mask],
                'Year':   np.int16(year),
                'Month':  np.int8(month),
            }
            for k, arr in lcz_cache[year].items():
                data[k] = arr[valid_mask]

            df = pd.DataFrame(data)
            df = df[(df['Pop'] > 0) & (df['GDP'] > 0) & (df[fe_col] > 0)]

            if MAX_ROWS_PER_YM and len(df) > MAX_ROWS_PER_YM:
                df = df.sample(n=MAX_ROWS_PER_YM, random_state=42)

            for c in df.select_dtypes('float64').columns:
                df[c] = df[c].astype('float32')
            df['Year']   = df['Year'].astype('int16')
            df['Month']  = df['Month'].astype('int8')
            df[fe_col]   = df[fe_col].astype('int16')

            df.to_pickle(chk)
            print(f"    检查点: {os.path.basename(chk)}  ({len(df):,} 行)")

            del co2_data, ghi_data, hdd_data, cdd_data, df
            gc.collect()

        # 释放年度缓存
        for cache in [lcz_cache, pop_cache, gdp_cache, urban_cache]:
            cache.pop(year, None)
        gc.collect()

    # --- 合并面板 ---
    print(f"\n[Phase 3] 合并面板 ({country})...")
    pkls = sorted([
        os.path.join(chk_dir, f)
        for f in os.listdir(chk_dir)
        if f.endswith('.pkl')
    ])
    if not pkls:
        print(f"  [{country}] 无检查点文件，跳过回归。")
        return

    df_all = pd.concat([pd.read_pickle(p) for p in pkls], ignore_index=True)
    print(f"  合并: {len(df_all):,} 行，"
          f"{df_all['Year'].nunique()} 年 × {df_all['Month'].nunique()} 月")

    # CO2 异常值剔除
    co2_p99  = df_all['CO2'].quantile(CO2_PERCENTILE / 100)
    n_before = len(df_all)
    df_all   = df_all[df_all['CO2'] <= co2_p99].copy()
    print(f"  CO2 p99 阈值={co2_p99:.2f}，剔除 {n_before-len(df_all):,} 行，"
          f"剩余 {len(df_all):,} 行")

    df_all.to_pickle(os.path.join(out_dir, 'Panel_Monthly_clean.pkl'))

    # --- 回归 ---
    print(f"\n[Phase 4] 月度回归 ({country})...")
    run_monthly_regression(df_all, fe_col, out_dir, country)

    print(f"\n[{country}] 全部完成！结果: {out_dir}")


# ================== 5. 主程序 ==================

def main():
    print("=" * 65)
    print("LCZ 月度碳排放分析 — 美国 + 中国")
    print(f"Season 基准: {SEASON_BASE} | LCZ 基准: LCZ9")
    print("=" * 65)

    for country in ['USA', 'CHN']:
        run_country(country, CONFIGS[country])

    print("\n" + "=" * 65)
    print("两国全部完成！")
    print(f"  美国结果: {CONFIGS['USA']['output_base']}")
    print(f"  中国结果: {CONFIGS['CHN']['output_base']}")
    print("=" * 65)


if __name__ == '__main__':
    main()
