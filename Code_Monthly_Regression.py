# -*- coding: utf-8 -*-
"""
月度面板回归 — 独立运行脚本（美国 + 中国）
============================================
直接读取已生成的 Panel_Monthly_clean.pkl，对大规模面板做分层随机抽样后跑回归。

  问题背景:
    美国面板约 2,933 万行 (3.26 GB)，中国约 2,649 万行 (2.95 GB)。
    OLS 构建设计矩阵时 (~130 列 × 3000 万行) 超出内存，回归被静默跳过。

  解决方案:
    按 Year × Month 分层抽样，每期最多保留 SAMPLE_PER_YM 行（默认 8000）。
    20年×12月 = 240 期 × 8000 = 192万行，统计效力充分，结果与全量回归近似一致。

  运行前确认：
    USA_PANEL  = 美国 Panel_Monthly_clean.pkl 路径
    CHN_PANEL  = 中国 Panel_Monthly_clean.pkl 路径
    USA_OUTDIR = 美国结果目录（与 Code_US_Monthly 输出目录相同）
    CHN_OUTDIR = 中国结果目录
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

# ================== 配置 ==================

USA_PANEL  = r"D:\LCZCarbon\Results_USA_Month\USA_1km\Panel_Monthly_clean.pkl"
CHN_PANEL  = r"D:\LCZCarbon\Results_CHN_Month\CHN_1km\Panel_Monthly_clean.pkl"

USA_OUTDIR = r"D:\LCZCarbon\Results_USA_Month\USA_1km"
CHN_OUTDIR = r"D:\LCZCarbon\Results_CHN_Month\CHN_1km"

SAMPLE_PER_YM     = 8000    # 每个 Year-Month 期最多抽样行数（总计约 192 万行）
RANDOM_SEED       = 42
CO2_PERCENTILE    = 99
LCZ_MAX_ZERO_FRAC = 0.99
SEASON_BASE       = 'Spring'

SEASON_MAP = {
    12: 'Winter', 1: 'Winter',  2: 'Winter',
    3:  'Spring', 4: 'Spring',  5: 'Spring',
    6:  'Summer', 7: 'Summer',  8: 'Summer',
    9:  'Fall',  10: 'Fall',   11: 'Fall',
}

COUNTRY_CFG = {
    'USA': {'panel': USA_PANEL, 'outdir': USA_OUTDIR, 'fe_col': 'State_ID'},
    'CHN': {'panel': CHN_PANEL, 'outdir': CHN_OUTDIR, 'fe_col': 'City_ID'},
}


# ================== 核心回归函数 ==================

def run_regression(df_input, fe_col, out_dir, label):
    """
    三个月度回归模型，基准：LCZ9 / Spring
    Model 1 – Baseline:   控制变量 + Month FE + Year FE + FE_ID
    Model 2 – Morphology: + LCZ 主效应
    Model 3 – Seasonal:   + Season × LCZ 交互项
    """
    df = df_input.copy()
    print(f"\n  [{label}] 样本量: {len(df):,} 行")

    # 对数变换
    for col in ['CO2', 'Pop', 'GDP', 'HDD', 'CDD', 'GHI']:
        df[f'ln_{col}'] = np.log(df[col].clip(lower=0) + 1).astype('float32')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Season 分类，Spring 为基准
    seasons    = [s for s in ['Spring', 'Summer', 'Fall', 'Winter'] if s != SEASON_BASE]
    cat_order  = [SEASON_BASE] + seasons
    df['Season'] = pd.Categorical(
        df['Month'].map(SEASON_MAP),
        categories=cat_order, ordered=False
    )

    # LCZ 近零列筛选
    all_lcz   = [c for c in df.columns if c.endswith('_share')]
    zero_frac = (df[all_lcz] < 1e-6).mean()
    valid_lcz = zero_frac[zero_frac < LCZ_MAX_ZERO_FRAC].index.tolist()
    dropped   = [c for c in all_lcz if c not in valid_lcz]
    print(f"    LCZ 保留 {len(valid_lcz)} 个，剔除: {[c.replace('_share','') for c in dropped]}")

    # 清洗 NaN
    keep = ([f'ln_{c}' for c in ['CO2','Pop','GDP','HDD','CDD','GHI']] +
            [fe_col, 'Year', 'Month', 'Season'] + valid_lcz)
    df.dropna(subset=keep, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"    清洗后: {len(df):,} 行")

    # 描述统计
    df[['CO2','Pop','GDP','HDD','CDD','GHI'] + valid_lcz].describe().to_csv(
        os.path.join(out_dir, 'Sample_Stats_Regression.csv'), encoding='utf-8-sig'
    )

    # 基准 LCZ (LCZ9)
    ref_lcz  = 'LCZ9_share' if 'LCZ9_share' in valid_lcz else valid_lcz[0]
    lcz_vars = [c for c in valid_lcz if c != ref_lcz]
    lcz_f    = ' + '.join(lcz_vars)

    # Season × LCZ 交互（基准: Spring）
    season_inter = ' + '.join(
        [f"C(Season, Treatment('{SEASON_BASE}')):{v}" for v in lcz_vars]
    )

    controls = (f"ln_Pop + ln_GDP + ln_HDD + ln_CDD + ln_GHI"
                f" + C(Month) + C(Year) + C({fe_col})")

    models_def = {
        'Model1_Baseline':   f"ln_CO2 ~ {controls}",
        'Model2_Morphology': f"ln_CO2 ~ {controls} + {lcz_f}",
        'Model3_Seasonal':   f"ln_CO2 ~ {controls} + {lcz_f} + {season_inter}",
    }

    for name, formula in models_def.items():
        print(f"\n  Running {name}...")
        try:
            m = smf.ols(formula, data=df).fit(
                cov_type='cluster',
                cov_kwds={'groups': df[fe_col]}
            )

            # 文字摘要
            with open(os.path.join(out_dir, f'{name}_Summary.txt'), 'w',
                      encoding='utf-8') as f:
                f.write(f"面板: 像元×年月 (分层抽样 {SAMPLE_PER_YM}/期) | "
                        f"FE: {fe_col} + Year + Month\n")
                f.write(f"LCZ 基准: {ref_lcz} | Season 基准: {SEASON_BASE}\n")
                f.write(f"剔除LCZ: {dropped}\n\n")
                f.write(m.summary().as_text())

            # 系数表
            ci = m.conf_int()
            coef_df = pd.DataFrame({
                'coef':      m.params,
                'std_err':   m.bse,
                'pvalue':    m.pvalues,
                'conf_low':  ci[0],
                'conf_high': ci[1],
            })
            coef_df.to_csv(os.path.join(out_dir, f'{name}_Coefs.csv'),
                           encoding='utf-8-sig')

            # Month FE 系数单独提取
            month_idx = [p for p in m.params.index if 'C(Month)' in str(p)]
            if month_idx:
                pd.DataFrame({
                    'coef':   m.params[month_idx],
                    'pvalue': m.pvalues[month_idx]
                }).to_csv(os.path.join(out_dir, f'{name}_MonthFE.csv'),
                          encoding='utf-8-sig')

            # Season×LCZ 交互系数单独提取（Model3）
            if 'Season' in formula:
                season_idx = [p for p in m.params.index if 'Season' in str(p)]
                if season_idx:
                    pd.DataFrame({
                        'coef':   m.params[season_idx],
                        'pvalue': m.pvalues[season_idx]
                    }).to_csv(os.path.join(out_dir, f'{name}_SeasonLCZ.csv'),
                              encoding='utf-8-sig')

            print(f"    R²={m.rsquared:.4f}  Adj-R²={m.rsquared_adj:.4f}"
                  f"  N={int(m.nobs):,}  条件数={m.condition_number:.2e}")

        except MemoryError:
            print(f"    MemoryError — 尝试减小 SAMPLE_PER_YM (当前={SAMPLE_PER_YM})")
        except Exception as e:
            print(f"    出错: {e}")


# ================== 主流程 ==================

def run_country_regression(country):
    cfg     = COUNTRY_CFG[country]
    pkl_path = cfg['panel']
    out_dir  = cfg['outdir']
    fe_col   = cfg['fe_col']

    print(f"\n{'='*60}")
    print(f"  {country} — 读取面板数据")
    print(f"{'='*60}")

    if not os.path.exists(pkl_path):
        print(f"  !! 找不到 Panel pkl: {pkl_path}")
        return

    print(f"  加载 {pkl_path} ...")
    df_all = pd.read_pickle(pkl_path)
    raw_n  = len(df_all)
    print(f"  原始行数: {raw_n:,}")

    # CO2 p99 再次确认（pkl 已经做过，但保留校验）
    co2_p99 = df_all['CO2'].quantile(CO2_PERCENTILE / 100)
    df_all  = df_all[df_all['CO2'] <= co2_p99].copy()
    print(f"  CO2 p99={co2_p99:.2f}，清洗后: {len(df_all):,} 行")

    # 按 Year × Month 分层抽样（用外部 Series 做分组键，避免 pandas 版本差异）
    print(f"  分层抽样 (每期最多 {SAMPLE_PER_YM} 行) ...")
    ym_key    = df_all['Year'].astype(str) + '_' + df_all['Month'].astype(str).str.zfill(2)
    n_periods = ym_key.nunique()

    sampled = (
        df_all.groupby(ym_key, group_keys=False)
              .apply(lambda g: g.sample(n=min(SAMPLE_PER_YM, len(g)),
                                        random_state=RANDOM_SEED))
              .reset_index(drop=True)
    )

    print(f"  抽样后: {len(sampled):,} 行 ({n_periods} 个年月期，"
          f"平均每期 {len(sampled)//n_periods:,} 行)")

    del df_all
    import gc; gc.collect()

    # 回归
    run_regression(sampled, fe_col, out_dir, country)
    print(f"\n  [{country}] 回归完成！结果目录: {out_dir}")


def main():
    print('='*60)
    print('月度面板回归（分层抽样版）— 美国 + 中国')
    print(f'每期抽样: {SAMPLE_PER_YM} 行 | Season基准: {SEASON_BASE} | LCZ基准: LCZ9')
    print('='*60)

    for country in ['USA', 'CHN']:
        run_country_regression(country)

    print('\n' + '='*60)
    print('两国回归全部完成！')
    print(f"  美国结果: {USA_OUTDIR}")
    print(f"  中国结果: {CHN_OUTDIR}")
    print('='*60)


if __name__ == '__main__':
    main()
