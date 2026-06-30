# -*- coding: utf-8 -*-
"""
LCZ 与城市碳排放关系研究 - 美国版 Final
==========================================
完整独立流水线（无需任何已有检查点），基于中国版 V1 改编：

  核心特性:
  [1] 动态年际建成区掩膜（Cities_2000.shp ~ Cities_2019.shp）
  [2] 无持久性筛选 —— 凡落入当年建成区的像元均纳入分析
  [3] MAUP 多尺度分析：1km / 5km / 10km
  [4] CO2 第 99 百分位异常值剔除
  [5] LCZ 近零列自动剔除（>99% 像元为零则剔除）
  [6] 断点续跑检查点机制（.pkl）
  [7] 州级固定效应（State FE + Year FE），州级聚类稳健 SE
  [8] 基准 LCZ 类别：LCZ9（稀疏建成）
  [9] 输出 State_ID_Mapping.csv 供地图绘制

  输出路径: D:/LCZCarbon/Result_USA_3  (USA_1km / USA_5km / USA_10km)

  ★ 运行前请核对 Section 1 中所有路径和文件名模板 ★
"""

import os
import gc
import math
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


# ================== 1. 路径配置与参数 ==================
# ★ 请根据实际文件路径修改以下内容 ★

base_paths = {
    'co2':       r"D:\LCZCarbon\CarbonSum\US",
    'pop':       r"D:\LCZCarbon\Population\USAPOP",
    'gdp':       r"D:\LCZCarbon\GDP Per Capita\USA",
    'lcz':       r"E:\Data_All\LCZ_Carbon_Form\US\Final_LCZ_Maps",
    'hdd_cdd':   r"D:\LCZCarbon\HDD_CDD",
    'ghi_east':  r"D:\LCZCarbon\GHI\USA_GISdata_LTAy_YearlyMonthlyTotals\eastern-hemisphere\GHI.tif",
    'ghi_west':  r"D:\LCZCarbon\GHI\USA_GISdata_LTAy_YearlyMonthlyTotals\western-hemisphere\GHI.tif",
    'state_shp': r"D:\LCZCarbon\US_State_Boundaries\US_State_Boundaries.shp",
    'urban_dir': r"E:\Data_All\Urban Footprint\Cities_2000_2022",
}

# 文件名模板（{year} 自动替换为年份，如 2000, 2001, ...）
FILE_TEMPLATES = {
    'co2': "CarbonSum_{year}_US.tif",
    'pop': "POP{year}_USA.tif",
    'gdp': "GDP_PC_{year}_USA.tif",
    'lcz': "TP_{year}.tif",
    'hdd': "USA_HDD_{year}.tif",
    'cdd': "USA_CDD_{year}.tif",
}

OUTPUT_BASE       = r"D:\LCZCarbon\Result_USA_NEW"
YEARS             = range(2000, 2020)
CO2_PERCENTILE    = 99       # CO2 异常值剔除阈值（百分位）
LCZ_MAX_ZERO_FRAC = 0.99     # LCZ 近零列剔除阈值（>99% 像元为零则剔除）
LCZ_NATURAL       = 'auto'   # LCZ 自然类型编码：'auto' / 'standard'(11-17) / 'alternative'(101-107)
MAX_ROWS_PER_YEAR = None     # 每年最大像元数；设为 None 则不抽样

SCALE_CONFIGS = {
    '1km':  {'factor': 1,  'out_dir': os.path.join(OUTPUT_BASE, 'USA_1km')},
    '5km':  {'factor': 5,  'out_dir': os.path.join(OUTPUT_BASE, 'USA_5km')},
    '10km': {'factor': 10, 'out_dir': os.path.join(OUTPUT_BASE, 'USA_10km')},
}

CHECKPOINT_DIR = os.path.join(OUTPUT_BASE, 'checkpoints_1km')

for d in [OUTPUT_BASE, CHECKPOINT_DIR] + [c['out_dir'] for c in SCALE_CONFIGS.values()]:
    os.makedirs(d, exist_ok=True)


# ================== 2. 工具函数 ==================

def check_file_exists(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[{label}] 文件不存在: {path}")


def load_and_resample(path, ref_meta, resampling=Resampling.bilinear):
    """读取单波段栅格并重采样到参考网格（输出 float32）"""
    with rasterio.open(path) as src:
        dst = np.zeros((ref_meta['height'], ref_meta['width']), dtype='float32')
        reproject(
            source=rasterio.band(src, 1), destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=ref_meta['transform'], dst_crs=ref_meta['crs'],
            resampling=resampling
        )
        return dst


def load_ghi(ghi_path_or_list, ref_meta):
    """
    加载 GHI 数据并重采样到参考网格。
    支持两种模式：
      - 单文件：ghi_path_or_list 为字符串路径
      - 多文件拼接：ghi_path_or_list 为文件路径列表（如东西半球两个文件）
        此时先用 rasterio.merge 拼接，再重采样
    """
    if isinstance(ghi_path_or_list, str):
        return load_and_resample(ghi_path_or_list, ref_meta, Resampling.bilinear)

    # 多文件拼接模式
    from rasterio.merge import merge as rio_merge
    srcs = [rasterio.open(p) for p in ghi_path_or_list]
    mosaic, mosaic_tf = rio_merge(srcs)
    for s in srcs:
        s.close()

    mosaic_data = mosaic[0].astype('float32')
    mosaic_crs  = srcs[0].crs

    dst = np.zeros((ref_meta['height'], ref_meta['width']), dtype='float32')
    reproject(
        source=mosaic_data, destination=dst,
        src_transform=mosaic_tf, src_crs=mosaic_crs,
        dst_transform=ref_meta['transform'], dst_crs=ref_meta['crs'],
        resampling=Resampling.bilinear
    )
    return dst


def process_climate_bands(path, ref_meta):
    """HDD/CDD 多波段（12个月）→ 先转 float32 再求年总值 → 双线性重采样"""
    with rasterio.open(path) as src:
        annual = src.read().astype('float32').sum(axis=0)  # 防 int16 溢出
        dst = np.zeros((ref_meta['height'], ref_meta['width']), dtype='float32')
        reproject(
            source=annual, destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=ref_meta['transform'], dst_crs=ref_meta['crs'],
            resampling=Resampling.bilinear
        )
        return dst


def detect_lcz_encoding(lcz_path):
    """自动检测 LCZ 自然类型编码（11-17 标准 或 101-107 替代编码）"""
    with rasterio.open(lcz_path) as src:
        win = rasterio.windows.Window(0, 0, min(500, src.width), min(500, src.height))
        sample = src.read(1, window=win)
        vals = np.unique(sample[sample > 0])
    if any(101 <= v <= 107 for v in vals) and not any(11 <= v <= 17 for v in vals):
        print("    [LCZ编码] 自然类型: 101-107（替代编码）")
        return list(range(1, 11)) + list(range(101, 108))
    else:
        print("    [LCZ编码] 自然类型: 11-17（标准编码）")
        return list(range(1, 18))


def calculate_lcz_shares(lcz_path, ref_meta, all_classes):
    """计算各 LCZ 类型在 1km 像元内的面积占比（Resampling.average = 面积比例）"""
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


def get_state_raster(shp_path, ref_meta, out_dir):
    """
    栅格化州边界，生成 State_ID 二维数组，保存 State_ID_Mapping.csv。
    自动识别州名列（NAME / STATE_NAME / NAME_1 等常见命名）。
    """
    print("  栅格化州边界...")
    states = gpd.read_file(shp_path)
    if states.crs != ref_meta['crs']:
        states = states.to_crs(ref_meta['crs'])

    # 自动识别州名列
    name_keywords = ['NAME', 'STATE', 'STNAME', 'NAME_1', 'STATENAME']
    name_col = next(
        (c for c in states.columns
         if any(kw in c.upper() for kw in name_keywords) and c != 'geometry'),
        None
    )
    if not name_col:
        name_col = next(
            (c for c in states.columns
             if c != 'geometry' and states[c].dtype == object),
            None
        )
    if not name_col:
        raise ValueError(
            f"无法在 SHP 属性表中找到州名列。现有列: {list(states.columns)}"
        )
    print(f"    使用州名列: '{name_col}'，共 {len(states)} 个州")

    states = states.sort_values(name_col).reset_index(drop=True)
    states['State_ID_Num'] = range(1, len(states) + 1)

    # 保存映射表
    mapping_df = states[['State_ID_Num', name_col]].copy()
    mapping_df.columns = ['State_ID', 'State_Name']
    map_path = os.path.join(out_dir, 'State_ID_Mapping.csv')
    mapping_df.to_csv(map_path, index=False, encoding='utf-8-sig')
    print(f"    State_ID 映射表已保存: {map_path}")

    # 栅格化
    burned = features.rasterize(
        ((geom, val) for geom, val in zip(states.geometry, states['State_ID_Num'])),
        out_shape=(ref_meta['height'], ref_meta['width']),
        transform=ref_meta['transform'],
        fill=0,
        dtype='int16'   # 美国 50 州 + DC，int16 完全足够
    )
    print(f"    州数量: {len(states)}, 有效像元: {(burned > 0).sum():,}")
    return burned


def get_urban_year_mask(urban_dir, year, ref_meta, study_bbox_geom):
    """返回指定年份的建成区二值掩膜（bool 数组）"""
    shp_path = os.path.join(urban_dir, f"Cities_{year}.shp")
    empty = np.zeros((ref_meta['height'], ref_meta['width']), dtype=bool)
    if not os.path.exists(shp_path):
        print(f"    警告: 找不到 Cities_{year}.shp，该年跳过")
        return empty

    gdf = gpd.read_file(shp_path)
    if len(gdf) == 0:
        return empty
    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:4326')

    bbox_in_shp_crs = (
        gpd.GeoSeries([study_bbox_geom], crs=ref_meta['crs'])
        .to_crs(gdf.crs).iloc[0]
    )
    gdf = gdf[gdf.geometry.intersects(bbox_in_shp_crs)].copy()
    if len(gdf) == 0:
        return empty
    gdf = gdf.to_crs(ref_meta['crs'])

    burned = features.rasterize(
        ((geom, 1) for geom in gdf.geometry if geom is not None),
        out_shape=(ref_meta['height'], ref_meta['width']),
        transform=ref_meta['transform'], fill=0, dtype='uint8'
    )
    return burned.astype(bool)


# ================== 3. MAUP 聚合函数 ==================

def aggregate_panel(df_1km, scale_factor, ref_width):
    """
    将 1km 面板聚合到 5km / 10km 尺度。
    规则: CO2/Pop → sum; GDP → 人口加权均值; HDD/CDD/GHI → 均值;
          LCZ_share → 均值; State_ID → 众数（面积最大的州）
    """
    df = df_1km.copy()
    n_cols_coarse = math.ceil(ref_width / scale_factor)
    df['cell_id'] = (
        (df['row_idx'] // scale_factor) * n_cols_coarse
        + (df['col_idx'] // scale_factor)
    )

    lcz_cols = [c for c in df.columns if '_share' in c]
    df['GDP_x_Pop'] = df['GDP'] * df['Pop']

    agg_dict = {
        'CO2':       'sum',
        'Pop':       'sum',
        'GDP_x_Pop': 'sum',
        'HDD':       'mean',
        'CDD':       'mean',
        'GHI':       'mean',
        'State_ID':  lambda x: int(x.value_counts().index[0]) if len(x) > 0 else 0,
    }
    for c in lcz_cols:
        agg_dict[c] = 'mean'

    grouped = df.groupby(['cell_id', 'Year']).agg(agg_dict).reset_index()
    grouped['GDP'] = (
        grouped['GDP_x_Pop'] / grouped['Pop'].replace(0, np.nan)
    ).astype('float32')
    grouped.drop(columns=['GDP_x_Pop', 'cell_id'], inplace=True)
    grouped = grouped[grouped['State_ID'] > 0].copy()

    for c in grouped.select_dtypes('float64').columns:
        grouped[c] = grouped[c].astype('float32')

    return grouped.reset_index(drop=True)


# ================== 4. 回归与结果输出 ==================

def run_regression_models(df_input, out_dir, scale_name):
    """
    四个递进 OLS 模型（State FE + Year FE，州级聚类稳健 SE）。
    基准 LCZ 类别：LCZ9（稀疏建成）。
    回归前自动剔除近零 LCZ 列（>LCZ_MAX_ZERO_FRAC 比例像元为零）。
    """
    df = df_input.copy()
    print(f"\n[{scale_name}] 回归准备，原始样本量: {len(df):,}")

    # 对数变换（+1 防零值取对数）
    for col in ['CO2', 'Pop', 'GDP', 'HDD', 'CDD', 'GHI']:
        df[f'ln_{col}'] = np.log(df[col] + 1).astype('float32')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # --- LCZ 近零列自动剔除 ---
    all_lcz_cols   = [c for c in df.columns if '_share' in c]
    lcz_zero_frac  = (df[all_lcz_cols] < 1e-6).mean()
    lcz_means      = df[all_lcz_cols].mean()
    lcz_prevalence = (df[all_lcz_cols] >= 1e-6).mean()

    valid_lcz   = lcz_zero_frac[lcz_zero_frac < LCZ_MAX_ZERO_FRAC].index.tolist()
    dropped_lcz = [c for c in all_lcz_cols if c not in valid_lcz]

    if dropped_lcz:
        print(f"  [LCZ筛选] 剔除近零列 ({len(dropped_lcz)}个): "
              f"{[c.replace('_share','') for c in dropped_lcz]}")
    print(f"  [LCZ筛选] 保留 {len(valid_lcz)} 个LCZ类型: "
          f"{[c.replace('_share','') for c in valid_lcz]}")

    # 保存 LCZ 筛选诊断表
    pd.DataFrame({
        'LCZ_col':    all_lcz_cols,
        'mean_share': lcz_means.values,
        'prevalence': lcz_prevalence.values,
        'zero_frac':  lcz_zero_frac.values,
        'retained':   [c in valid_lcz for c in all_lcz_cols]
    }).to_csv(os.path.join(out_dir, 'LCZ_Coverage_Filter.csv'),
              index=False, encoding='utf-8-sig')

    # 清洗缺失值
    keep_cols = (['ln_CO2', 'ln_Pop', 'ln_GDP', 'ln_HDD', 'ln_CDD', 'ln_GHI',
                  'State_ID', 'Year'] + valid_lcz)
    df.dropna(subset=keep_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  清洗后样本量: {len(df):,}")

    # --- 构建回归公式 ---
    controls = "ln_Pop + ln_GDP + ln_HDD + ln_CDD + ln_GHI + C(Year) + C(State_ID)"

    ref_lcz = 'LCZ9_share'
    if ref_lcz not in valid_lcz:
        ref_lcz = valid_lcz[0]
        print(f"  注意: LCZ9_share 不在保留列表，改用 {ref_lcz} 作为基准类别")

    lcz_vars  = [c for c in valid_lcz if c != ref_lcz]
    lcz_f     = " + ".join(lcz_vars)
    inter_pop = " + ".join([f"{v}:ln_Pop" for v in lcz_vars])
    inter_gdp = " + ".join([f"{v}:ln_GDP" for v in lcz_vars])
    inter_ghi = " + ".join([f"{v}:ln_GHI" for v in lcz_vars])

    models_def = {
        'Model1': f"ln_CO2 ~ {controls}",
        'Model2': f"ln_CO2 ~ {controls} + {lcz_f}",
        'Model3': f"ln_CO2 ~ {controls} + {lcz_f} + {inter_pop} + {inter_gdp}",
        'Model4': f"ln_CO2 ~ {controls} + {lcz_f} + {inter_pop} + {inter_gdp} + {inter_ghi}",
    }

    # 保存样本描述统计
    df[['CO2', 'Pop', 'GDP', 'HDD', 'CDD', 'GHI'] + valid_lcz].describe().to_csv(
        os.path.join(out_dir, 'Sample_Stats.csv'), encoding='utf-8-sig'
    )

    for name, formula in models_def.items():
        try:
            print(f"  Running {name}...")
            m = smf.ols(formula, data=df).fit(
                cov_type='cluster',
                cov_kwds={'groups': df['State_ID']}   # 州级聚类 SE
            )

            with open(os.path.join(out_dir, f"{name}_Summary.txt"), 'w',
                      encoding='utf-8') as f:
                f.write(f"LCZ 基准类别: {ref_lcz}\n")
                f.write(f"剔除近零LCZ: {dropped_lcz}\n\n")
                f.write(m.summary().as_text())

            ci = m.conf_int()
            pd.DataFrame({
                'coef':      m.params,
                'std_err':   m.bse,
                'pvalue':    m.pvalues,
                'conf_low':  ci[0],
                'conf_high': ci[1],
            }).to_csv(os.path.join(out_dir, f"{name}_Coefs.csv"),
                      encoding='utf-8-sig')

            print(f"    R²={m.rsquared:.4f}, Adj.R²={m.rsquared_adj:.4f}, "
                  f"N={int(m.nobs):,}, 条件数={m.condition_number:.2e}")

        except MemoryError:
            print(f"  {name}: 内存溢出，跳过。")
        except Exception as e:
            print(f"  {name}: 出错 → {e}")


# ================== 5. 主程序 ==================

def main():
    print("=" * 60)
    print("LCZ 与碳排放研究 - 美国 Final（无持久性筛选）")
    print("=" * 60)

    # --- 静态文件预检 ---
    check_file_exists(base_paths['ghi_east'],  'GHI East')
    check_file_exists(base_paths['ghi_west'],  'GHI West')
    check_file_exists(base_paths['state_shp'], 'State SHP')

    # --- 读取 CO2 基准栅格（获取 ref_meta）---
    ref_co2_name = FILE_TEMPLATES['co2'].format(year=2000)
    ref_co2_path = os.path.join(base_paths['co2'], ref_co2_name)
    check_file_exists(ref_co2_path, 'CO2-2000 基准')
    with rasterio.open(ref_co2_path) as src:
        ref_meta = src.meta.copy()
    print(f"\n基准栅格: {ref_meta['width']} × {ref_meta['height']} px @ 1km, "
          f"CRS={ref_meta['crs']}")

    # 研究区 bbox（用于裁剪全球建成区 SHP）
    bounds = array_bounds(ref_meta['height'], ref_meta['width'], ref_meta['transform'])
    study_bbox_geom = box(*bounds)

    # ---- Phase 1: 静态数据准备 ----
    print("\n[Phase 1] 静态数据准备...")

    # GHI（全研究期不变，仅加载一次）
    # 东西半球文件拼接后重采样到参考网格
    print("  拼接东西半球 GHI 并重采样（仅执行一次）...")
    ghi_resampled = load_ghi(
        [base_paths['ghi_west'], base_paths['ghi_east']], ref_meta
    )
    print(f"    GHI 重采样完成，有效值范围: "
          f"{ghi_resampled[ghi_resampled > 0].min():.1f} ~ "
          f"{ghi_resampled.max():.1f}")

    # 州边界栅格化
    state_raster = get_state_raster(base_paths['state_shp'], ref_meta, OUTPUT_BASE)

    # ---- Phase 2: 逐年处理 → 1km 检查点 ----
    print("\n[Phase 2] 逐年处理（1km 像元级）...")
    lcz_classes = None  # 首次检测后缓存

    for year in YEARS:
        chk = os.path.join(CHECKPOINT_DIR, f"panel_{year}.pkl")
        if os.path.exists(chk):
            print(f"  [断点续跑] 跳过 {year}")
            continue

        print(f"\n  >>> 年份: {year}")

        # 构建文件路径
        f_co2 = os.path.join(base_paths['co2'],     FILE_TEMPLATES['co2'].format(year=year))
        f_pop = os.path.join(base_paths['pop'],     FILE_TEMPLATES['pop'].format(year=year))
        f_gdp = os.path.join(base_paths['gdp'],     FILE_TEMPLATES['gdp'].format(year=year))
        f_lcz = os.path.join(base_paths['lcz'],     FILE_TEMPLATES['lcz'].format(year=year))
        f_hdd = os.path.join(base_paths['hdd_cdd'], FILE_TEMPLATES['hdd'].format(year=year))
        f_cdd = os.path.join(base_paths['hdd_cdd'], FILE_TEMPLATES['cdd'].format(year=year))

        # 文件完整性检查
        skip = False
        for fp, lb in [(f_co2,'CO2'),(f_pop,'POP'),(f_gdp,'GDP'),
                       (f_lcz,'LCZ'),(f_hdd,'HDD'),(f_cdd,'CDD')]:
            if not os.path.exists(fp):
                print(f"    缺失 [{lb}]: {fp}，跳过 {year}")
                skip = True; break
        if skip:
            continue

        # 读取 CO2
        with rasterio.open(f_co2) as src:
            co2_data = src.read(1).astype('float32')
            nodata   = src.nodata

        # 年度建成区掩膜（无持久性筛选，直接使用当年 SHP）
        urban_year_mask = get_urban_year_mask(
            base_paths['urban_dir'], year, ref_meta, study_bbox_geom
        )

        # 有效掩膜 = CO2有效 ∩ 当年建成区范围内
        co2_valid = (
            (co2_data != nodata) & (co2_data > 0)
            if nodata is not None else (co2_data > 0)
        )
        valid_mask = co2_valid & urban_year_mask

        n_valid = valid_mask.sum()
        print(f"    有效像元: {n_valid:,}")
        if n_valid == 0:
            print(f"    无有效像元，跳过")
            del co2_data, urban_year_mask; gc.collect()
            continue

        # LCZ 编码自动检测（仅第一年执行）
        if lcz_classes is None:
            if LCZ_NATURAL == 'auto':
                lcz_classes = detect_lcz_encoding(f_lcz)
            elif LCZ_NATURAL == 'alternative':
                lcz_classes = list(range(1, 11)) + list(range(101, 108))
            else:
                lcz_classes = list(range(1, 18))

        # 获取像元行列索引（供 MAUP 聚合）
        rows, cols = np.where(valid_mask)

        # 加载并重采样各变量
        print("    加载变量...")
        pop_data = load_and_resample(f_pop, ref_meta, Resampling.bilinear)
        gdp_data = load_and_resample(f_gdp, ref_meta, Resampling.bilinear)
        hdd_data = process_climate_bands(f_hdd, ref_meta)
        cdd_data = process_climate_bands(f_cdd, ref_meta)

        print("    计算 LCZ shares...")
        lcz_shares = calculate_lcz_shares(f_lcz, ref_meta, lcz_classes)

        # 组装 DataFrame
        data = {
            'CO2':      co2_data[valid_mask],
            'Pop':      pop_data[valid_mask],
            'GDP':      gdp_data[valid_mask],
            'HDD':      hdd_data[valid_mask],
            'CDD':      cdd_data[valid_mask],
            'GHI':      ghi_resampled[valid_mask],
            'State_ID': state_raster[valid_mask],
            'Year':     np.int16(year),
            'row_idx':  rows.astype('int32'),
            'col_idx':  cols.astype('int32'),
        }
        for k, arr in lcz_shares.items():
            data[k] = arr[valid_mask]

        df = pd.DataFrame(data)
        df = df[(df['Pop'] > 0) & (df['GDP'] > 0) & (df['State_ID'] > 0)]

        # 可选抽样
        if MAX_ROWS_PER_YEAR and len(df) > MAX_ROWS_PER_YEAR:
            df = df.sample(n=MAX_ROWS_PER_YEAR, random_state=42)
            print(f"    抽样后: {len(df):,} 行")

        # 类型压缩
        for c in df.select_dtypes('float64').columns:
            df[c] = df[c].astype('float32')
        df['Year']     = df['Year'].astype('int16')
        df['State_ID'] = df['State_ID'].astype('int16')

        df.to_pickle(chk)
        print(f"    检查点已保存: {chk}  ({len(df):,} 行)")

        del co2_data, pop_data, gdp_data, hdd_data, cdd_data
        del lcz_shares, urban_year_mask, df
        gc.collect()

    # ---- Phase 3: 合并 1km 面板 + CO2 异常值剔除 ----
    print("\n[Phase 3] 合并 1km 面板...")
    pkls = sorted([
        os.path.join(CHECKPOINT_DIR, f)
        for f in os.listdir(CHECKPOINT_DIR)
        if f.endswith('.pkl')
    ])
    if not pkls:
        raise ValueError("未找到任何检查点文件，请检查数据路径和建成区 SHP。")

    df_1km = pd.concat([pd.read_pickle(p) for p in pkls], ignore_index=True)
    print(f"  合并完成: {len(df_1km):,} 行 × {df_1km.shape[1]} 列，"
          f"覆盖 {df_1km['Year'].nunique()} 年")

    # CO2 异常值剔除
    print(f"\n[Phase 3b] CO2 异常值剔除（>{CO2_PERCENTILE} 百分位）...")
    co2_p99   = df_1km['CO2'].quantile(CO2_PERCENTILE / 100)
    n_before  = len(df_1km)
    df_1km    = df_1km[df_1km['CO2'] <= co2_p99].copy()
    n_removed = n_before - len(df_1km)
    print(f"  CO2 第{CO2_PERCENTILE}百分位阈值: {co2_p99:.4f}")
    print(f"  剔除像元: {n_removed:,} 行 ({n_removed/n_before*100:.2f}%)")
    print(f"  剩余样本: {len(df_1km):,} 行")

    ref_width = int(df_1km['col_idx'].max()) + 1
    print(f"  推断参考栅格宽度: {ref_width} px")

    df_1km.to_pickle(os.path.join(SCALE_CONFIGS['1km']['out_dir'], 'Panel_1km_clean.pkl'))

    # ---- Phase 4: MAUP 多尺度聚合 + 回归 ----
    print("\n[Phase 4] MAUP 多尺度回归...")

    for scale_name, cfg in SCALE_CONFIGS.items():
        factor  = cfg['factor']
        out_dir = cfg['out_dir']
        print(f"\n{'='*60}")
        print(f"尺度: {scale_name}  (聚合因子: {factor}×{factor})")

        if factor == 1:
            df_scale = df_1km.copy()
        else:
            print(f"  聚合 1km → {scale_name}...")
            df_scale = aggregate_panel(df_1km, factor, ref_width)
            df_scale.to_pickle(os.path.join(out_dir, f'Panel_{scale_name}_clean.pkl'))
            print(f"  {scale_name} 面板: {len(df_scale):,} 行")

        run_regression_models(df_scale, out_dir, scale_name)
        del df_scale
        gc.collect()

    print("\n" + "=" * 60)
    print("全部完成！结果目录:")
    for name, cfg in SCALE_CONFIGS.items():
        print(f"  {name}: {cfg['out_dir']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
