# -*- coding: utf-8 -*-
# Code_Fix_GDP_PerCapita.py
# =========================
# 用已裁剪好的人口栅格（D:/LCZCarbon/Population）重新计算 GDP Per Capita，
# 覆盖写入 D:/LCZCarbon/GDP Per Capita。
#
# 修复问题：2009、2018、2019 年 GDP Per Capita 数值异常偏大（~30-50x），
#           根因是原始计算时这 3 年所用的人口栅格值异常偏小（仅正常年份的 1-2%）。
#
# 流程：
#   1. 读取 Population/USAPOP/POP{year}_USA.tif（float32，已裁剪）
#   2. 通过 reproject 重采样到与 GDP_total 完全相同的格网（Resampling.average）
#   3. 计算 GDP_PC = GDP_total / resampled_pop（pop=0 处设为 nodata=0）
#   4. 写出 GDP Per Capita/USA/GDP_PC_{year}_USA.tif（float32, LZW 压缩）
#   同理处理 CHN。
#
# 运行时间：约 5-10 分钟（两国 x 20 年）
# 依赖：rasterio, numpy

import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# ─────────────────────────── 路径配置 ───────────────────────────

CONFIGS = [
    dict(
        country       = "USA",
        pop_dir       = r"D:\LCZCarbon\Population\USAPOP",
        pop_tpl       = "POP{year}_USA.tif",
        gdp_tot_dir   = r"D:\LCZCarbon\GDP\USAGDP",
        gdp_tot_tpl   = "{year}GDP_USA.tif",
        out_dir       = r"D:\LCZCarbon\GDP Per Capita\USA",
        out_tpl       = "GDP_PC_{year}_USA.tif",
    ),
    dict(
        country       = "CHN",
        pop_dir       = r"D:\LCZCarbon\Population\CHNPOP",
        pop_tpl       = "POP{year}_CHN.tif",
        gdp_tot_dir   = r"D:\LCZCarbon\GDP\CHNGDP",
        gdp_tot_tpl   = "{year}GDP_CHN.tif",
        out_dir       = r"D:\LCZCarbon\GDP Per Capita\CHN",
        out_tpl       = "GDP_PC_{year}_CHN.tif",
    ),
]

YEARS = list(range(2000, 2020))

# ──────────────────────────── 核心函数 ─────────────────────────────

def resample_pop_to_gdp_grid(pop_path, gdp_path):
    """
    将 pop_path 重采样到与 gdp_path 完全相同的格网（CRS、transform、shape）。
    使用 Resampling.average：将多个细格网像元平均到粗格网（与原始流程一致）。
    返回 2D float32 ndarray，shape = (gdp_height, gdp_width)。
    """
    with rasterio.open(gdp_path) as gdp_src:
        dst_crs       = gdp_src.crs
        dst_transform = gdp_src.transform
        dst_height    = gdp_src.height
        dst_width     = gdp_src.width

    with rasterio.open(pop_path) as pop_src:
        src_arr       = pop_src.read(1).astype("float32")
        src_transform = pop_src.transform
        src_crs       = pop_src.crs
        src_nodata    = pop_src.nodata if pop_src.nodata is not None else 0

    dst = np.zeros((dst_height, dst_width), dtype="float32")

    reproject(
        source        = src_arr,
        destination   = dst,
        src_transform = src_transform,
        src_crs       = src_crs,
        dst_transform = dst_transform,
        dst_crs       = dst_crs,
        resampling    = Resampling.average,
        src_nodata    = src_nodata,
        dst_nodata    = 0,
    )

    dst[dst < 0] = 0  # 消除重采样引入的负值伪像
    return dst, dst_transform, dst_crs, dst_height, dst_width


def compute_and_save_gdp_pc(gdp_path, pop_resampled,
                             out_path, dst_transform, dst_crs,
                             dst_height, dst_width):
    """
    GDP_PC = GDP_total / pop_resampled，保存为 float32 GeoTIFF（LZW 压缩）。
    pop = 0 的像元输出值为 0（nodata）。
    """
    with rasterio.open(gdp_path) as gsrc:
        gdp_arr = gsrc.read(1).astype("float32")

    if gdp_arr.shape != (dst_height, dst_width):
        raise ValueError(
            f"GDP shape {gdp_arr.shape} ≠ expected ({dst_height}, {dst_width})"
        )

    # pop >= 0.5 阈值：重采样后边缘像元可能有极小的正值（如 0.001），
    # 直接相除会得到天文数字；物理上 <0.5 人/像元视为无人口。
    pop_mask = pop_resampled >= 0.5
    with np.errstate(divide='ignore', invalid='ignore'):
        gdp_pc = np.where(
            pop_mask,
            gdp_arr / np.where(pop_mask, pop_resampled, 1.0),
            0.0
        ).astype("float32")

    # 清除残余 inf / nan（例如 GDP 本身异常像元）
    gdp_pc = np.nan_to_num(gdp_pc, nan=0.0, posinf=0.0, neginf=0.0)

    out_meta = dict(
        driver    = "GTiff",
        dtype     = "float32",
        count     = 1,
        height    = dst_height,
        width     = dst_width,
        crs       = dst_crs,
        transform = dst_transform,
        compress  = "lzw",
        nodata    = 0.0,
    )
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(gdp_pc, 1)

    return gdp_pc


# ──────────────────────────── 主流程 ───────────────────────────

def main():
    for cfg in CONFIGS:
        country = cfg["country"]
        os.makedirs(cfg["out_dir"], exist_ok=True)

        print(f"\n{'='*65}")
        print(f"  {country}  ({len(YEARS)} years)")
        print(f"{'='*65}")
        print(f"  {'Year':<6} {'Pop_med':>10} {'GDP_PC_p50':>12} {'GDP_PC_p99':>12}  Status")
        print(f"  {'-'*58}")

        for year in YEARS:
            pop_path = os.path.join(cfg["pop_dir"],     cfg["pop_tpl"].format(year=year))
            gdp_path = os.path.join(cfg["gdp_tot_dir"], cfg["gdp_tot_tpl"].format(year=year))
            out_path = os.path.join(cfg["out_dir"],     cfg["out_tpl"].format(year=year))

            for label, path in [("Pop", pop_path), ("GDP", gdp_path)]:
                if not os.path.exists(path):
                    print(f"  {year:<6}  SKIP – {label} file not found: {path}")
                    continue

            # ── 重采样人口到 GDP 格网 ──
            pop_rs, dst_tf, dst_crs, dst_h, dst_w = resample_pop_to_gdp_grid(
                pop_path, gdp_path
            )

            # ── 计算并保存 GDP Per Capita ──
            gdp_pc = compute_and_save_gdp_pc(
                gdp_path, pop_rs, out_path,
                dst_tf, dst_crs, dst_h, dst_w
            )

            # ── QC 打印 ──
            pop_valid = pop_rs[pop_rs > 0]
            pc_valid  = gdp_pc[gdp_pc > 0]
            pop_med   = float(np.median(pop_valid)) if len(pop_valid) > 0 else 0
            pc_p50    = float(np.percentile(pc_valid, 50))  if len(pc_valid) > 0 else 0
            pc_p99    = float(np.percentile(pc_valid, 99))  if len(pc_valid) > 0 else 0

            flag = " ← CHECK" if year in (2009, 2018, 2019) else ""
            print(f"  {year:<6} {pop_med:>10.2f} {pc_p50:>12.6f} {pc_p99:>12.4f}  OK{flag}")

    print(f"\n\n{'='*65}")
    print("  全部完成。GDP Per Capita 已覆盖更新。")
    print("  下一步：")
    print("    1. 重新运行 Code_US_Monthly.py  → 重建 USA Panel_Monthly_clean.pkl")
    print("    2. 重新运行 Code_CHN_V1.py       → 重建 CHN Panel_Monthly_clean.pkl")
    print("    3. 重新运行 Code_Monthly_Regression.py")
    print("    4. 重新运行 Code_Descriptive_Stats_v2.py（RUN_FIGS = {1,2,3,4,5,6,7,8}）")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
