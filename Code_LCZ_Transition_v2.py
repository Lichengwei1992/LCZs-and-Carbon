# -*- coding: utf-8 -*-
# Code_LCZ_Transition_v2.py
# =============================================================
# LCZ 转变分析 — 优化版（两个独立模块）
#
# ── 模块 A：时段对比（层次一优化）──────────────────────────────
#   读取已有 Table_Trans1_Annual_Detail.csv，按突变节点分为
#   早期 / 晚期，生成 Fig 3 风格的对比图：
#     Fig_Trans3a_Period_Split.png  —— 4子图：USA/CHN × 早期/晚期
#     Fig_Trans3b_Period_Shift.png  —— 转变前后净贡献变化（哪些转变消失/新增）
#     Table_Trans4_Period_Split.csv —— 逐时段 Top 转变汇总
#
# ── 模块 B：LCZ 份额连续差分（层次二优化）──────────────────────
#   直接读取 Panel_Monthly_clean.pkl，对每个像元逐年计算
#   ΔLCZ_share 与 ΔCO₂，绕开 dominant LCZ 噪声：
#     Fig_Trans5_ShareDiff_Effect.png —— 每种 LCZ 份额增加时的 CO₂ 效应（条形）
#     Fig_Trans6_NetContrib_Type.png  —— 各 LCZ 类型净 CO₂ 贡献（频率×效应）
#     Fig_Trans7_Temporal_NetCO2.png  —— 逐年净 CO₂ 归因时序图（堆叠面积）
#     Table_Trans5_ShareDiff.csv      —— 每种 LCZ 类型的份额变化效应统计
#     Table_Trans6_Annual_NetCO2.csv  —— 逐年各 LCZ 类型净 CO₂ 归因
#
# 运行时间：模块 A <1 min；模块 B 约 5-15 min（取决于面板大小）
# =============================================================

import os
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy import stats

warnings.filterwarnings('ignore')

# ─────────────────────────── 路径 ────────────────────────────

USA_PANEL   = r"D:\LCZCarbon\Results_USA_Month\USA_1km\Panel_Monthly_clean.pkl"
CHN_PANEL   = r"D:\LCZCarbon\Results_CHN_Month\CHN_1km\Panel_Monthly_clean.pkl"
TRANS1_CSV  = r"D:\LCZCarbon\Results_DescStats_v2\Table_Trans1_Annual_Detail.csv"
OUT_DIR     = r"D:\LCZCarbon\Results_DescStats_v2"
os.makedirs(OUT_DIR, exist_ok=True)

# 突变节点（来自 Code_LCZ_CO2_Dynamics.py 的结构突变分析结果）
BREAK_YEARS = {'USA': 2009, 'CHN': 2013}

# 分层抽样：每 Year×Month 期最多保留行数（减少内存，模块 B 用）
SAMPLE_PER_YM = 6_000
RANDOM_SEED   = 42

# ─────────────────────── 样式常量 ────────────────────────────

matplotlib.rcParams.update({
    'font.family':        'Arial',
    'font.size':          14,
    # 四边完整轴框（Origin 风格）
    'axes.spines.top':    True,
    'axes.spines.right':  True,
    'axes.spines.left':   True,
    'axes.spines.bottom': True,
    'axes.linewidth':     1.0,
    # 刻度线
    'xtick.major.width':  1.0,
    'ytick.major.width':  1.0,
    'xtick.minor.width':  0.8,
    'ytick.minor.width':  0.8,
    'xtick.major.size':   5,
    'ytick.major.size':   5,
    'xtick.minor.size':   3,
    'ytick.minor.size':   3,
    'xtick.direction':    'in',
    'ytick.direction':    'in',
    # 无网格线，白色背景
    'axes.grid':          False,
    'legend.frameon':     True,
    'legend.framealpha':  0.9,
    'legend.edgecolor':   '#CCCCCC',
    'figure.dpi':         150,
    'savefig.dpi':        600,
    'savefig.bbox':       'tight',
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
})

FS_TITLE  = 18
FS_TICK   = 16
FS_LEGEND = 14
LW        = 1.0

USA_DARK   = '#1B6BAF'
USA_LIGHT  = '#87CEEB'
CHN_DARK   = '#C0392B'
CHN_LIGHT  = '#F4A6A6'
GRAY_LINE  = '#AAAAAA'

YEAR_TICKS = [2000, 2005, 2010, 2015, 2020]

# LCZ 标准标签（1-10 = Built，11-17 / A-G = Natural）
LCZ_SHORT = {
    1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'10',
    11:'A', 12:'B', 13:'C', 14:'D', 15:'E', 16:'F', 17:'G',
}
LCZ_FULL = {
    1:'Compact high-rise', 2:'Compact mid-rise', 3:'Compact low-rise',
    4:'Open high-rise',    5:'Open mid-rise',    6:'Open low-rise',
    7:'Lightweight low-rise', 8:'Large low-rise', 9:'Sparsely built',
    10:'Heavy industry',
    11:'Dense trees', 12:'Scattered trees', 13:'Bush/scrub',
    14:'Low plants',  15:'Bare rock/paved', 16:'Bare soil', 17:'Water',
}
# LCZ 1-10 = Built（含 10 Heavy industry）
# LCZ 11-17 = Natural（A-G）
BUILT_CODES   = set(range(1, 11))       # 1,2,3,4,5,6,7,8,9,10
NATURAL_CODES = set(range(11, 18))      # 11(A),12(B),...,17(G)
EXCLUDE_CODES = set()                   # 不排除任何类型


def style_ax(ax, xlabel='', ylabel='', title=''):
    """两轴框（左+下），刻度朝外，无网格，Arial字体"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(LW)
    ax.spines['bottom'].set_linewidth(LW)
    ax.tick_params(axis='both', labelsize=FS_TICK,
                   width=LW, length=5, direction='out',
                   top=False, right=False)
    ax.grid(False)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FS_TITLE, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS_TITLE, labelpad=8)
    if title:
        ax.set_title(title, fontsize=FS_TITLE, pad=10)


def _save_fig(fig, path_no_ext):
    """同时保存 PNG（600dpi）和 SVG（可编辑矢量）"""
    fig.savefig(path_no_ext + '.png', dpi=600, bbox_inches='tight')
    fig.savefig(path_no_ext + '.svg', bbox_inches='tight')


# ══════════════════════════════════════════════════════════════
# 模块 A：时段对比（基于已有 Table_Trans1_Annual_Detail.csv）
# ══════════════════════════════════════════════════════════════

def module_a_period_split(top_n=15):
    """
    按结构突变节点把转变数据拆为早期 / 晚期，
    生成两张 Fig 3 风格的对比图。
    """
    print("\n" + "=" * 60)
    print("  Module A: 时段对比分析")
    print("=" * 60)

    if not os.path.exists(TRANS1_CSV):
        print(f"  [ERROR] 找不到 {TRANS1_CSV}")
        print("  请先运行 Code_LCZ_Transition.py 生成 Table_Trans1_Annual_Detail.csv")
        return

    df = pd.read_csv(TRANS1_CSV, encoding='utf-8-sig')
    print(f"  读取 {TRANS1_CSV}，共 {len(df):,} 行")

    # ── 图 A1：4子图 — USA/CHN × 早期/晚期 Top-N 转变 ─────────────

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    for col_idx, country in enumerate(['USA', 'CHN']):
        brk = BREAK_YEARS[country]
        c_style = USA_DARK if country == 'USA' else CHN_DARK
        c_light  = USA_LIGHT if country == 'USA' else CHN_LIGHT

        for row_idx, (period_label, mask) in enumerate([
            (f'Early period  2000–{brk-1}',
             (df['Country'] == country) & (df['Year_from'] < brk)),
            (f'Late period  {brk}–2019',
             (df['Country'] == country) & (df['Year_from'] >= brk)),
        ]):
            ax = axes[row_idx][col_idx]
            sub = df[mask].copy()

            if sub.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=FS_TITLE)
                continue

            # 聚合：per-pixel 均值 = Σ(Delta_CO2_sum) / Σ(Count)
            agg = (sub.groupby(['Trans_label'])
                   .agg(
                       Net_CO2_sum = ('Delta_CO2_sum', 'sum'),
                       Count       = ('Count',         'sum'),
                   ).reset_index())
            agg['Mean_pp']  = agg['Net_CO2_sum'] / agg['Count'].clip(lower=1)
            agg['Mean_abs'] = agg['Mean_pp'].abs()

            # 各路径占该国家×时期所有转变像元年总数的百分比
            total_count = agg['Count'].sum()
            agg['Pct'] = agg['Count'] / total_count * 100

            # 过滤极小样本（Count < 10），避免噪声路径进入 Top N
            MIN_COUNT = 10
            agg_valid = agg[agg['Count'] >= MIN_COUNT].copy()

            # Top N 按 |per-pixel 均值| 排序
            top = agg_valid.nlargest(top_n, 'Mean_abs').sort_values('Mean_pp')

            colors   = [c_style if v > 0 else c_light for v in top['Mean_pp']]
            top_vals = top['Mean_pp']
            unit_str = 'ton C / km² / year'

            bars = ax.barh(range(len(top)), top_vals,
                           color=colors, height=0.65,
                           edgecolor='white', linewidth=0.4)
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top['Trans_label'], fontsize=FS_TICK - 2)
            ax.axvline(0, color='#333', linewidth=LW)

            # 数值标注：仅显示 per-pixel 均值
            vmax = top_vals.abs().max() if len(top_vals) > 0 else 1
            for bar, val in zip(bars, top_vals):
                offset = 0.02 * vmax
                ax.text(val + (offset if val >= 0 else -offset),
                        bar.get_y() + bar.get_height() / 2,
                        f'{val:.1f}',
                        va='center',
                        ha='left' if val >= 0 else 'right',
                        fontsize=FS_TICK - 5, color='#333')

            title_str = f'{country}  |  {period_label}\n(break year = {brk})'
            style_ax(ax,
                     xlabel=f'Mean ΔCO₂ per pixel  ({unit_str})',
                     title=title_str)

            # 图例
            leg_h = [
                Line2D([0],[0], color=c_style,  linewidth=6, label='Net increase'),
                Line2D([0],[0], color=c_light,  linewidth=6, label='Net decrease'),
            ]
            ax.legend(handles=leg_h, fontsize=FS_LEGEND - 2, loc='lower right')

    plt.suptitle(
        f'Top {top_n} LCZ Transitions by Mean ΔCO₂ per Pixel — Early vs. Late Period\n'
        f'(ranked by |mean ΔCO₂ / pixel|;  min. {MIN_COUNT} pixel-years required)',
        fontsize=FS_TITLE + 1, y=1.02)
    plt.tight_layout()
    out_a1 = os.path.join(OUT_DIR, 'Fig_Trans3a_Period_Split')
    _save_fig(fig, out_a1)
    plt.close(fig)
    print(f"  Saved: {out_a1}.png / .svg")

    # ── 图 A2：转变净贡献的时段变化（有/无某转变的差异）─────────────

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    for ax, country in zip(axes, ['USA', 'CHN']):
        brk = BREAK_YEARS[country]
        c_style = USA_DARK if country == 'USA' else CHN_DARK
        c_light  = USA_LIGHT if country == 'USA' else CHN_LIGHT

        sub = df[df['Country'] == country].copy()

        def _pp_mean(grp):
            """per-pixel 均值 = Σ(Delta_CO2_sum) / Σ(Count)"""
            return grp['Delta_CO2_sum'].sum() / max(grp['Count'].sum(), 1)

        early_pp = (sub[sub['Year_from'] < brk]
                    .groupby('Trans_label').apply(_pp_mean)
                    .rename('early'))
        late_pp  = (sub[sub['Year_from'] >= brk]
                    .groupby('Trans_label').apply(_pp_mean)
                    .rename('late'))
        both  = pd.concat([early_pp, late_pp], axis=1).fillna(0)
        both['shift'] = both['late'] - both['early']
        both['shift_abs'] = both['shift'].abs()

        # 按 |shift| 取 top_n
        top = both.nlargest(top_n, 'shift_abs').sort_values('shift')
        unit_str = 'ton C / km² / year'

        y_pos = np.arange(len(top))

        # 早期：空心条；晚期：实心条
        ax.barh(y_pos - 0.18, top['early'], height=0.33,
                color=c_light, edgecolor=c_style,
                linewidth=0.8, label=f'Early (2000–{brk-1})')
        ax.barh(y_pos + 0.18, top['late'],  height=0.33,
                color=c_style, edgecolor=c_style,
                linewidth=0.8, label=f'Late ({brk}–2019)')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top.index, fontsize=FS_TICK - 2)
        ax.axvline(0, color='#333', linewidth=LW)

        style_ax(ax,
                 xlabel=f'Mean ΔCO₂ per pixel  ({unit_str})',
                 title=f'{country}  —  Period Shift in Mean ΔCO₂ per Pixel\n'
                       f'(break year = {brk})')
        ax.legend(fontsize=FS_LEGEND - 2, loc='lower right')

    plt.tight_layout()
    out_a2 = os.path.join(OUT_DIR, 'Fig_Trans3b_Period_Shift')
    _save_fig(fig, out_a2)
    plt.close(fig)
    print(f"  Saved: {out_a2}.png / .svg")

    # ── 表格输出 ─────────────────────────────────────────────────

    rows_out = []
    for country in ['USA', 'CHN']:
        brk = BREAK_YEARS[country]
        for period, mask_fn in [('Early', lambda yr: yr < brk),
                                 ('Late',  lambda yr: yr >= brk)]:
            sub = df[(df['Country'] == country) & df['Year_from'].apply(mask_fn)]
            agg = (sub.groupby(['Trans_label', 'Category'])
                   .agg(Total_Count=('Count', 'sum'),
                        Net_CO2=('Delta_CO2_sum', 'sum'),
                        Mean_Delta_CO2=('Delta_CO2_mean', 'mean'),
                        N_years=('Year_from', 'nunique'))
                   .reset_index())
            # per-pixel 均值（ton C / km² / year）
            agg['Mean_pp'] = agg['Net_CO2'] / agg['Total_Count'].clip(lower=1)
            agg.insert(0, 'Period', period)
            agg.insert(0, 'Country', country)
            rows_out.append(agg)

    df_t4 = pd.concat(rows_out, ignore_index=True)
    df_t4.sort_values(['Country', 'Period', 'Mean_pp'], inplace=True)
    out_t4 = os.path.join(OUT_DIR, 'Table_Trans4_Period_Split.csv')
    df_t4.to_csv(out_t4, index=False, encoding='utf-8-sig')
    print(f"  Saved: {out_t4}  ({len(df_t4):,} 行)")

    print("  Module A 完成。")


# ══════════════════════════════════════════════════════════════
# 模块 B：LCZ 份额连续差分分析（直接读取 Panel pkl）
# ══════════════════════════════════════════════════════════════

def _detect_lcz_cols(df):
    """从面板列名识别 LCZ_share 列并标准化为 LCZ 1-17 编号"""
    raw = [c for c in df.columns if c.endswith('_share')]
    # 标准编码 LCZ1_share...LCZ17_share
    # 替代编码 LCZ101_share...LCZ107_share → 映射到 11-17
    mapping = {}
    for col in raw:
        try:
            code = int(col.replace('LCZ', '').replace('_share', ''))
        except ValueError:
            continue
        if 101 <= code <= 107:
            std_code = code - 90   # 101→11, 102→12, ...
        else:
            std_code = code
        if 1 <= std_code <= 17:
            mapping[col] = std_code
    return mapping   # {original_col_name: standard_code}


def _load_annual_panel(pkl_path, fe_col, sample_per_ym=SAMPLE_PER_YM):
    """
    加载面板 pkl，分层抽样后聚合为年度像元级数据。

    返回 DataFrame，列：Year, fe_col, pix_idx, CO2_annual,
                       LCZ{std}_share (1-17, 标准化), ...

    关键：pix_idx 是像元在 (Year, Month=1, fe_col) 分组内的
    行序号，由于每年读取的像元集合和顺序固定（来自同一持久建成区掩膜），
    同一 pix_idx 在不同年份代表同一地理位置像元。
    """
    print(f"    读取 {pkl_path} ...")
    df = pd.read_pickle(pkl_path)
    print(f"    原始行数: {len(df):,}")

    # 分层抽样（按 Year × Month）减少内存
    if sample_per_ym and sample_per_ym < len(df) // (len(df['Year'].unique()) * 12):
        ym_key = df['Year'].astype(str) + '_' + df['Month'].astype(str).str.zfill(2)
        df = (df.groupby(ym_key, group_keys=False)
                .apply(lambda g: g.sample(n=min(sample_per_ym, len(g)),
                                          random_state=RANDOM_SEED))
                .reset_index(drop=True))
        print(f"    抽样后: {len(df):,} 行")

    # 检测 LCZ 列
    lcz_map = _detect_lcz_cols(df)
    if not lcz_map:
        raise ValueError("未找到 LCZ_share 列，请检查面板结构")
    print(f"    识别到 {len(lcz_map)} 个 LCZ_share 列")

    # 在 Month=1 子集中按 (Year, fe_col) 分配像元索引
    df1 = df[df['Month'] == 1].copy()
    df1['pix_idx'] = df1.groupby(['Year', fe_col]).cumcount()

    # 年度 CO₂（12个月均值）：需要先分配 pix_idx 到全量数据
    # 方法：对每个 (Year, Month, fe_col) 组，用 cumcount 生成 pix_idx
    df['pix_idx'] = df.groupby(['Year', 'Month', fe_col]).cumcount()

    # 计算年度 CO₂ 均值
    co2_annual = (df.groupby(['Year', fe_col, 'pix_idx'])['CO2']
                  .mean()
                  .reset_index(name='CO2_annual'))

    # 提取 LCZ_share（Month=1 代表该年 LCZ 信息）
    lcz_orig_cols = list(lcz_map.keys())
    annual_lcz = df1[['Year', fe_col, 'pix_idx'] + lcz_orig_cols].copy()

    # 标准化列名
    rename_map = {col: f'LCZ{std}_share'
                  for col, std in lcz_map.items()}
    annual_lcz.rename(columns=rename_map, inplace=True)
    std_lcz_cols = [f'LCZ{std}_share' for std in sorted(set(lcz_map.values()))]

    # 合并
    annual = co2_annual.merge(annual_lcz, on=['Year', fe_col, 'pix_idx'], how='inner')
    del df, df1, co2_annual, annual_lcz
    gc.collect()
    print(f"    年度面板: {len(annual):,} 行  ({annual['Year'].nunique()} 年，"
          f"{annual[fe_col].nunique()} 个地区)")
    return annual, std_lcz_cols


def _compute_share_diffs(annual_df, fe_col, std_lcz_cols, min_years=5):
    """
    对每个 (fe_col, pix_idx) 像元，计算相邻年份的：
      - ΔCO₂ = CO₂_annual(t+1) - CO₂_annual(t)
      - ΔLCZᵢ_share  for each i

    返回差分 DataFrame。
    """
    print("    计算逐年差分 ...")
    annual_df = annual_df.sort_values(['fe_col' if 'fe_col' in annual_df.columns
                                       else fe_col, 'pix_idx', 'Year'])
    grp_cols = [fe_col, 'pix_idx']

    diff_cols = ['CO2_annual'] + std_lcz_cols
    diff_df   = annual_df.copy()

    for col in diff_cols:
        diff_df[f'd_{col}'] = diff_df.groupby(grp_cols)[col].diff()

    diff_df.dropna(subset=[f'd_{c}' for c in diff_cols], inplace=True)

    # 过滤：只保留在 ≥ min_years 年出现的像元
    pix_counts = diff_df.groupby(grp_cols)['Year'].count()
    valid_pix  = pix_counts[pix_counts >= min_years].index
    diff_df = diff_df.set_index(grp_cols).loc[valid_pix].reset_index()

    print(f"    差分行数: {len(diff_df):,}  "
          f"(有效像元 ≥{min_years}年: {len(valid_pix):,})")
    return diff_df


def _aggregate_by_lcz_change(diff_df, std_lcz_cols, share_threshold=0.02):
    """
    对每个 LCZ 类型 i，找出该类型份额变化 |ΔLCZᵢ| > threshold 的像元-年，
    分正向（增加）和负向（减少）统计：
      - 平均 ΔCO₂（CO₂ 效应）
      - 像元-年数（频率）
      - 净 CO₂ 贡献 = 平均 ΔCO₂ × 频率

    返回 DataFrame：LCZ_code, Direction, Count, Mean_dCO2,
                    Net_CO2, Mean_dShare, Std_dCO2, SE_dCO2
    """
    rows = []
    for col in std_lcz_cols:
        lcz_code = int(col.replace('LCZ', '').replace('_share', ''))
        d_col = f'd_{col}'
        if d_col not in diff_df.columns:
            continue

        for direction, sign in [('Gain', 1), ('Loss', -1)]:
            mask = diff_df[d_col] * sign > share_threshold
            sub  = diff_df.loc[mask, ['d_CO2_annual', d_col]].dropna()
            if len(sub) < 10:
                continue
            n       = len(sub)
            mean_co2 = sub['d_CO2_annual'].mean()
            std_co2  = sub['d_CO2_annual'].std()
            se_co2   = std_co2 / np.sqrt(n)
            mean_ds  = sub[d_col].mean() * sign   # 平均份额变化量（正数）
            net_co2  = mean_co2 * n

            # t检验显著性
            t_stat, p_val = stats.ttest_1samp(sub['d_CO2_annual'], 0)

            rows.append({
                'LCZ_code':   lcz_code,
                'LCZ_short':  LCZ_SHORT.get(lcz_code, str(lcz_code)),
                'LCZ_full':   LCZ_FULL.get(lcz_code, ''),
                'Type':       'Built' if lcz_code in BUILT_CODES else 'Natural',
                'Direction':  direction,
                'Count':      n,
                'Mean_dCO2':  mean_co2,
                'Std_dCO2':   std_co2,
                'SE_dCO2':    se_co2,
                'Net_CO2':    net_co2,
                'Mean_dShare': mean_ds,
                'T_stat':     t_stat,
                'P_value':    p_val,
            })

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return df_out
    df_out['Sig'] = df_out['P_value'].apply(
        lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else
                  ('*' if p < 0.05 else ('†' if p < 0.1 else 'ns'))))
    return df_out


def _annual_net_co2(diff_df, std_lcz_cols, share_threshold=0.02):
    """
    逐年：对每种 LCZ 类型，计算该年净 CO₂ 归因
    = Σ(ΔCO₂ × sign(ΔLCZᵢ)) for pixels with |ΔLCZᵢ| > threshold
    """
    rows = []
    for yr in sorted(diff_df['Year'].unique()):
        sub_yr = diff_df[diff_df['Year'] == yr]
        for col in std_lcz_cols:
            lcz_code = int(col.replace('LCZ', '').replace('_share', ''))
            d_col = f'd_{col}'
            if d_col not in sub_yr.columns:
                continue
            for direction, sign in [('Gain', 1), ('Loss', -1)]:
                mask = sub_yr[d_col] * sign > share_threshold
                sub = sub_yr.loc[mask, 'd_CO2_annual'].dropna()
                if len(sub) == 0:
                    continue
                rows.append({
                    'Year': yr,
                    'LCZ_code': lcz_code,
                    'LCZ_short': LCZ_SHORT.get(lcz_code, str(lcz_code)),
                    'Type': 'Built' if lcz_code in BUILT_CODES else 'Natural',
                    'Direction': direction,
                    'Count': len(sub),
                    'Net_CO2': sub.sum(),
                    'Mean_dCO2': sub.mean(),
                })
    return pd.DataFrame(rows)


def fig_share_effect(df_effect, country):
    """
    图 5：每种 LCZ 类型份额增加/减少时的 CO₂ 效应（水平条形图）
    x = 平均 ΔCO₂，y = LCZ 类型，颜色 = Gain/Loss，误差棒 = ±1SE
    上下两个子图：Gain（份额增加）和 Loss（份额减少）
    """
    c_dark  = USA_DARK  if country == 'USA' else CHN_DARK
    c_light = USA_LIGHT if country == 'USA' else CHN_LIGHT

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    for ax, direction, color in [
        (axes[0], 'Gain',  c_dark),
        (axes[1], 'Loss',  c_light),
    ]:
        sub = df_effect[df_effect['Direction'] == direction].copy()
        sub = sub.sort_values('Mean_dCO2')

        # 颜色：正 CO₂ 效应 = 深色，负 = 浅色
        bar_colors = [c_dark if v > 0 else c_light for v in sub['Mean_dCO2']]
        bars = ax.barh(sub['LCZ_short'], sub['Mean_dCO2'],
                       xerr=sub['SE_dCO2'],
                       color=bar_colors, height=0.65,
                       edgecolor='white', linewidth=0.3,
                       error_kw=dict(ecolor='#666', elinewidth=0.8, capsize=3))

        ax.axvline(0, color='#333', linewidth=LW)

        # 显著性标注
        for i, (_, row) in enumerate(sub.iterrows()):
            if row['Sig'] != 'ns':
                xpos = row['Mean_dCO2'] + (row['SE_dCO2'] + 0.5
                        if row['Mean_dCO2'] >= 0
                        else -row['SE_dCO2'] - 0.5)
                ax.text(xpos, i, row['Sig'],
                        va='center', ha='left' if row['Mean_dCO2'] >= 0 else 'right',
                        fontsize=FS_TICK - 4, color='#333')

        # Built/Natural 分隔线
        built_codes = [LCZ_SHORT[c] for c in range(1, 11) if LCZ_SHORT[c] in sub['LCZ_short'].values]
        nat_codes   = [LCZ_SHORT[c] for c in range(11, 18) if LCZ_SHORT[c] in sub['LCZ_short'].values]
        all_labels  = sub['LCZ_short'].tolist()

        dir_label = 'Share gain (+)' if direction == 'Gain' else 'Share loss (−)'
        style_ax(ax,
                 xlabel='Mean ΔCO₂  (ton C/cell/month)',
                 title=f'{direction}  —  {dir_label}')

    axes[0].set_ylabel('LCZ type', fontsize=FS_TITLE)
    fig.suptitle(f'{country}  —  CO₂ Effect per LCZ Share Change\n'
                 f'(left: when that LCZ type expands; right: when it shrinks)',
                 fontsize=FS_TITLE, y=1.01)
    plt.tight_layout()
    return fig


def fig_net_contrib(df_effect, country):
    """
    图 6：各 LCZ 类型净 CO₂ 贡献（频率 × 效应）—— Fig 3 同类型
    分 Gain / Loss 两列，正负颜色区分，按绝对值排序
    """
    c_dark  = USA_DARK  if country == 'USA' else CHN_DARK
    c_light = USA_LIGHT if country == 'USA' else CHN_LIGHT
    top_n = 14   # 每个 direction 显示前 N

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, direction in zip(axes, ['Gain', 'Loss']):
        sub = df_effect[df_effect['Direction'] == direction].copy()
        sub = sub.nlargest(top_n, 'Net_CO2' if direction == 'Gain' else 'Net_CO2')
        # 对 Loss 方向，Net_CO2 可能为负，按绝对值取 top_n
        sub['Net_abs'] = sub['Net_CO2'].abs()
        sub = sub.nlargest(top_n, 'Net_abs').sort_values('Net_CO2')

        scale_v = sub['Net_CO2'].abs().max()
        if scale_v > 1e6:
            vals = sub['Net_CO2'] / 1e6
            unit = '×10⁶ ton C/cell'
        elif scale_v > 1e3:
            vals = sub['Net_CO2'] / 1e3
            unit = '×10³ ton C/cell'
        else:
            vals = sub['Net_CO2']
            unit = 'ton C/cell'

        bar_colors = [c_dark if v > 0 else c_light for v in vals]
        bars = ax.barh(sub['LCZ_short'], vals,
                       color=bar_colors, height=0.65,
                       edgecolor='white', linewidth=0.4)

        # 标注
        vmax = vals.abs().max() if len(vals) > 0 else 1
        for bar, val, sig in zip(bars, vals, sub['Sig']):
            offset = 0.02 * vmax
            label  = f'{val:.2f}  {sig}'
            ax.text(val + (offset if val >= 0 else -offset),
                    bar.get_y() + bar.get_height() / 2,
                    label, va='center',
                    ha='left' if val >= 0 else 'right',
                    fontsize=FS_TICK - 5, color='#333')

        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub['LCZ_short'], fontsize=FS_TICK - 1)
        ax.axvline(0, color='#333', linewidth=LW)

        dir_label = 'Expanding LCZ types' if direction == 'Gain' else 'Shrinking LCZ types'
        style_ax(ax,
                 xlabel=f'Net CO₂ contribution  ({unit})',
                 title=f'{direction}  ─  {dir_label}')

    fig.suptitle(f'{country}  —  Net CO₂ Contribution by LCZ Share Change\n'
                 f'(Mean ΔCO₂ × pixel-year count;  Sig: †p<0.1 *p<0.05 **p<0.01 ***p<0.001)',
                 fontsize=FS_TITLE - 1, y=1.01)
    plt.tight_layout()
    return fig


def fig_temporal_net(df_annual, country):
    """
    图 7：逐年净 CO₂ 归因时序图（堆叠面积）
    仅展示绝对贡献 Top-8 的 LCZ 类型（Gain + Loss 合并）
    x = 年份，y = 净 CO₂（±），颜色 = LCZ 类型
    """
    c_dark  = USA_DARK  if country == 'USA' else CHN_DARK

    # 按 LCZ 类型合并 Gain + Loss
    annual_by_lcz = (df_annual.groupby(['Year', 'LCZ_code', 'LCZ_short', 'Type'])
                    ['Net_CO2'].sum().reset_index())

    # 取 |平均净 CO₂| Top-8 类型
    top_types = (annual_by_lcz.groupby('LCZ_code')['Net_CO2']
                 .apply(lambda x: x.abs().mean())
                 .nlargest(8).index.tolist())

    df_top = annual_by_lcz[annual_by_lcz['LCZ_code'].isin(top_types)].copy()
    pivot  = (df_top.pivot_table(index='Year', columns='LCZ_short',
                                  values='Net_CO2', aggfunc='sum')
              .fillna(0))

    # 颜色（区分 built/natural）
    cmap_built = plt.cm.Reds(np.linspace(0.4, 0.9, 10))
    cmap_nat   = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
    col_map    = {}
    bi, ni = 0, 0
    for code in range(1, 18):
        short = LCZ_SHORT.get(code, '')
        if short in pivot.columns:
            if code in BUILT_CODES:
                col_map[short] = cmap_built[bi % 10]; bi += 1
            else:
                col_map[short] = cmap_nat[ni % 8]; ni += 1

    fig, ax = plt.subplots(figsize=(10, 5.5))

    pos_cols = [c for c in pivot.columns if pivot[c].mean() >= 0]
    neg_cols = [c for c in pivot.columns if pivot[c].mean() < 0]

    # 堆叠正值
    bottoms = np.zeros(len(pivot))
    for col in pos_cols:
        vals = pivot[col].values
        ax.fill_between(pivot.index, bottoms, bottoms + vals,
                        color=col_map.get(col, '#999'),
                        alpha=0.75, label=col)
        ax.plot(pivot.index, bottoms + vals,
                color=col_map.get(col, '#999'), linewidth=0.5)
        bottoms += vals

    # 堆叠负值
    bottoms = np.zeros(len(pivot))
    for col in neg_cols:
        vals = pivot[col].values
        ax.fill_between(pivot.index, bottoms + vals, bottoms,
                        color=col_map.get(col, '#999'),
                        alpha=0.75, label=col)
        ax.plot(pivot.index, bottoms + vals,
                color=col_map.get(col, '#999'), linewidth=0.5)
        bottoms += vals

    ax.axhline(0, color='#333', linewidth=LW)
    for yr, lbl in {2008: '2008\nCrisis', 2015: 'Paris'}.items():
        ax.axvline(yr, color=GRAY_LINE, linewidth=0.8, linestyle=':')
        ax.text(yr, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] else 1,
                lbl, ha='center', fontsize=FS_LEGEND - 3, color='#999')

    ax.set_xticks(YEAR_TICKS)
    ax.set_xticklabels([str(y) for y in YEAR_TICKS], fontsize=FS_TICK)
    ax.set_xlim(1999.5, 2019.5)
    ax.legend(fontsize=FS_LEGEND - 3, loc='upper left',
              ncol=2, title='LCZ type', title_fontsize=FS_LEGEND - 3)

    style_ax(ax, xlabel='Year',
             ylabel='Net CO₂ attribution  (ton C/cell)',
             title=f'{country}  —  Annual Net CO₂ by LCZ Share Change\n'
                   f'(positive = that LCZ expanding increases CO₂;  top 8 types)')
    plt.tight_layout()
    return fig


def module_b_share_diff():
    """
    主函数：LCZ 份额连续差分分析（模块 B）
    """
    print("\n" + "=" * 60)
    print("  Module B: LCZ 份额连续差分分析")
    print("=" * 60)

    panels = {
        'USA': (USA_PANEL, 'State_ID'),
        'CHN': (CHN_PANEL, 'City_ID'),
    }

    all_effect_rows = []
    all_annual_rows = []

    for country, (pkl_path, fe_col) in panels.items():
        if not os.path.exists(pkl_path):
            print(f"  [WARN] 找不到 {pkl_path}，跳过 {country}")
            continue

        print(f"\n  {'─'*50}")
        print(f"  {country}")
        print(f"  {'─'*50}")

        # 1. 加载年度面板
        annual, std_lcz_cols = _load_annual_panel(pkl_path, fe_col)

        # 2. 计算差分
        diff_df = _compute_share_diffs(annual, fe_col, std_lcz_cols)
        del annual; gc.collect()

        # 3. 逐 LCZ 类型汇总效应
        print("    统计每种 LCZ 份额变化的 CO₂ 效应 ...")
        df_effect = _aggregate_by_lcz_change(diff_df, std_lcz_cols)
        df_effect.insert(0, 'Country', country)

        # 4. 逐年净 CO₂ 归因
        print("    计算逐年净 CO₂ 归因 ...")
        df_annual_net = _annual_net_co2(diff_df, std_lcz_cols)
        df_annual_net.insert(0, 'Country', country)
        del diff_df; gc.collect()

        all_effect_rows.append(df_effect)
        all_annual_rows.append(df_annual_net)

        # 5. 绘图（每个国家单独）
        print(f"    绘制 {country} 图表 ...")

        fig5 = fig_share_effect(df_effect, country)
        out5 = os.path.join(OUT_DIR, f'Fig_Trans5_{country}_ShareDiff_Effect')
        _save_fig(fig5, out5); plt.close(fig5)
        print(f"    Saved: {out5}.png / .svg")

        fig6 = fig_net_contrib(df_effect, country)
        out6 = os.path.join(OUT_DIR, f'Fig_Trans6_{country}_NetContrib')
        _save_fig(fig6, out6); plt.close(fig6)
        print(f"    Saved: {out6}.png / .svg")

        fig7 = fig_temporal_net(df_annual_net, country)
        out7 = os.path.join(OUT_DIR, f'Fig_Trans7_{country}_Temporal')
        _save_fig(fig7, out7); plt.close(fig7)
        print(f"    Saved: {out7}.png / .svg")

    # 6. 保存表格
    if all_effect_rows:
        df_t5 = pd.concat(all_effect_rows, ignore_index=True)
        df_t5.sort_values(['Country', 'LCZ_code', 'Direction'], inplace=True)
        out_t5 = os.path.join(OUT_DIR, 'Table_Trans5_ShareDiff_Effect.csv')
        df_t5.to_csv(out_t5, index=False, encoding='utf-8-sig')
        print(f"\n  Saved: {out_t5}  ({len(df_t5):,} 行)")

    if all_annual_rows:
        df_t6 = pd.concat(all_annual_rows, ignore_index=True)
        df_t6.sort_values(['Country', 'Year', 'LCZ_code'], inplace=True)
        out_t6 = os.path.join(OUT_DIR, 'Table_Trans6_Annual_NetCO2.csv')
        df_t6.to_csv(out_t6, index=False, encoding='utf-8-sig')
        print(f"  Saved: {out_t6}  ({len(df_t6):,} 行)")

    print("  Module B 完成。")


# ══════════════════════════════════════════════════════════════
# 模块 C：基于 Share 差分的四大类别分析
# ══════════════════════════════════════════════════════════════

CAT_ORDER  = ['Natural→Built', 'Built→Built', 'Built→Natural', 'Natural→Natural']
CAT_COLORS = {
    'Natural→Built':    '#C0392B',   # 深红：城镇化扩张
    'Built→Built':      '#E67E22',   # 橙：城市内部重构
    'Built→Natural':    '#27AE60',   # 绿：绿化/退建
    'Natural→Natural':  '#2980B9',   # 蓝：自然内部变化
}
CAT_LABELS = {
    'Natural→Built':    'Natural→Built (Urbanization)',
    'Built→Built':      'Built→Built (Densification)',
    'Built→Natural':    'Built→Natural (Greening)',
    'Natural→Natural':  'Natural→Natural (Natural change)',
}

YEARS_TICK_C = list(range(2000, 2020, 3))


def _classify_pixel_category(diff_df, std_lcz_cols, share_threshold=0.02):
    """
    对每个 (pixel, year) 差分记录，基于 Built_share 和 Natural_share 的变化
    判断属于四大类别之一。

    Built_share   = Σ LCZᵢ_share  for i in BUILT_CODES   (1-10)
    Natural_share = Σ LCZᵢ_share  for i in NATURAL_CODES (11-17)

    分类规则（三层判断，优先级依次递减）：

    Layer 1 — Built 总量净变化显著（|ΔBuilt| > thr）：
      ΔBuilt >  thr  且 ΔNatural < -thr  → Natural→Built  （自然转建成）
      ΔBuilt >  thr  且 ΔNatural ≥ -thr  → Built→Built    （建成内部扩张/密度增加）
      ΔBuilt < -thr  且 ΔNatural >  thr  → Built→Natural  （建成转自然/绿化）
      ΔBuilt < -thr  且 ΔNatural ≤  thr  → Built→Built    （建成内部收缩/重构）

    Layer 2 — Built 总量净变化不显著（|ΔBuilt| ≤ thr），
              但 Built 内部各 LCZ 之间发生了显著互换：
      d_built_internal = Σ |ΔLCZᵢ_share| for i in BUILT_CODES
      d_built_internal > thr              → Built→Built    （内部形态重构）

    Layer 3 — 以上均不满足 → Natural→Natural（自然覆盖内部变化或无显著变化）
    """
    built_cols   = [f'LCZ{c}_share' for c in sorted(BUILT_CODES)
                    if f'LCZ{c}_share' in diff_df.columns]
    natural_cols = [f'LCZ{c}_share' for c in sorted(NATURAL_CODES)
                    if f'LCZ{c}_share' in diff_df.columns]

    d_built_cols   = [f'd_{c}' for c in built_cols   if f'd_{c}' in diff_df.columns]
    d_natural_cols = [f'd_{c}' for c in natural_cols if f'd_{c}' in diff_df.columns]

    df = diff_df.copy()
    df['d_Built']           = df[d_built_cols].sum(axis=1)
    df['d_Natural']         = df[d_natural_cols].sum(axis=1)
    # Built 内部各 LCZ 绝对变化之和（捕捉内部互换，即使净变化为零）
    df['d_Built_internal']  = df[d_built_cols].abs().sum(axis=1)

    thr = share_threshold

    # Layer 1：Built 总量净变化显著
    cond_nb  = (df['d_Built'] >  thr) & (df['d_Natural'] < -thr)   # Natural→Built
    cond_bb1 = (df['d_Built'] >  thr) & (df['d_Natural'] >= -thr)  # Built→Built (扩张)
    cond_bn  = (df['d_Built'] < -thr) & (df['d_Natural'] >  thr)   # Built→Natural
    cond_bb2 = (df['d_Built'] < -thr) & (df['d_Natural'] <= thr)   # Built→Built (收缩重构)

    # Layer 2：Built 净变化不显著但内部互换显著
    cond_bb3 = (df['d_Built'].abs() <= thr) & (df['d_Built_internal'] > thr)  # Built→Built (内部重构)

    conditions = [cond_nb, cond_bb1, cond_bn, cond_bb2, cond_bb3]
    choices    = ['Natural→Built', 'Built→Built', 'Built→Natural',
                  'Built→Built',   'Built→Built']

    df['Category'] = np.select(conditions, choices, default='Natural→Natural')
    return df


def _category_annual_stats(cat_df):
    """
    按 (Year, Category) 汇总：
      - Pixel_Count：像元-年数
      - Net_CO2：ΔCO₂ 之和
      - Mean_Delta_CO2：ΔCO₂ 均值
    """
    grp = (cat_df.groupby(['Year', 'Category'])
           .agg(
               Pixel_Count=('d_CO2_annual', 'count'),
               Net_CO2=('d_CO2_annual', 'sum'),
               Mean_Delta_CO2=('d_CO2_annual', 'mean'),
           )
           .reset_index())
    return grp


def fig_category_combined(df_all, break_years):
    """
    4格组合图（2行×2列）：
      上排：Mean ΔCO₂ 折线（CHN / USA）
      下排：Net CO₂ 堆叠柱状（CHN / USA）
    """
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('LCZ Transition Category Analysis — China vs. USA\n'
                 '(Share-based classification: Built = LCZ 1–10, Natural = LCZ A–G)',
                 fontsize=14, fontweight='bold', y=1.02)

    countries = ['CHN', 'USA']

    # ── 上排：Mean ΔCO₂ 折线 ───────────────────────────────────
    for col_i, country in enumerate(countries):
        ax = axes[0, col_i]
        sub = df_all[df_all['Country'] == country]
        bk  = break_years[country]

        for cat in CAT_ORDER:
            d = sub[sub['Category'] == cat].sort_values('Year')
            if d.empty:
                continue
            ax.plot(d['Year'].astype(int), d['Mean_Delta_CO2'],
                    color=CAT_COLORS[cat], linewidth=2.2,
                    marker='o', markersize=4.5, label=CAT_LABELS[cat], zorder=3)

        ax.axvline(bk, color='#444', linewidth=1.4, linestyle='--', alpha=0.8, zorder=2)
        ax.axhline(0, color='black', linewidth=0.7, alpha=0.35, zorder=1)
        ylim = ax.get_ylim()
        ax.text(bk + 0.2, ylim[1] * 0.97 if ylim[1] != 0 else 1,
                f'Break\n{bk}', fontsize=8.5, color='#444', va='top', ha='left')

        ax.set_title(f'{country}  —  Mean ΔCO₂ per transition pixel\n'
                     f'(structural break = {bk})',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('Mean ΔCO₂  (ton C / cell / yr)', fontsize=10)
        ax.set_xticks(YEARS_TICK_C)
        ax.set_xticklabels([str(y) for y in YEARS_TICK_C], rotation=30, ha='right')
        ax.set_xlim(1999.5, 2019.2)
        ax.legend(fontsize=FS_LEGEND - 4, loc='upper left', framealpha=0.9,
                  handlelength=1.5, labelspacing=0.3, bbox_to_anchor=(0, 1))
        style_ax(ax)

    # ── 下排：Net CO₂ 堆叠柱状 ────────────────────────────────
    for col_i, country in enumerate(countries):
        ax = axes[1, col_i]
        sub = df_all[df_all['Country'] == country]
        bk  = break_years[country]
        years = sorted(sub['Year'].unique())

        pivot = (sub.pivot(index='Year', columns='Category', values='Net_CO2')
                 .reindex(index=years, columns=CAT_ORDER)
                 .fillna(0))

        scale = 1e3
        pos_data = pivot.clip(lower=0) / scale
        neg_data = pivot.clip(upper=0) / scale
        yr = np.array(years, dtype=int)

        pos_bottom = np.zeros(len(yr))
        for cat in CAT_ORDER:
            vals = pos_data[cat].values
            ax.bar(yr, vals, bottom=pos_bottom, color=CAT_COLORS[cat],
                   width=0.8, alpha=0.88, zorder=3)
            pos_bottom += vals

        neg_bottom = np.zeros(len(yr))
        for cat in CAT_ORDER:
            vals = neg_data[cat].values
            ax.bar(yr, vals, bottom=neg_bottom, color=CAT_COLORS[cat],
                   width=0.8, alpha=0.88, zorder=3)
            neg_bottom += vals

        ax.axvline(bk - 0.5, color='#333', linewidth=1.4, linestyle='--', alpha=0.8, zorder=4)
        ax.axhline(0, color='black', linewidth=0.9, zorder=4)
        ylim = ax.get_ylim()
        ax.text(bk - 0.3, ylim[1] * 0.97 if ylim[1] != 0 else 1,
                f'Break {bk}', fontsize=8.5, color='#333', va='top', ha='right')

        ax.set_title(f'{country}  —  Net CO₂ contribution\n'
                     f'(Mean ΔCO₂ × pixel count, structural break = {bk})',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('Net CO₂  (×10³ ton C / cell)', fontsize=10)
        ax.set_xticks(YEARS_TICK_C)
        ax.set_xticklabels([str(y) for y in YEARS_TICK_C], rotation=30, ha='right')
        ax.set_xlim(1999, 2019.5)
        style_ax(ax)

    # 全局图例
    handles = [mpatches.Patch(color=CAT_COLORS[c], alpha=0.88, label=CAT_LABELS[c])
               for c in CAT_ORDER]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               fontsize=9.5, framealpha=0.92,
               bbox_to_anchor=(0.5, -0.06),
               handlelength=1.8, labelspacing=0.4)

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    return fig


def module_c_category_analysis():
    """
    Module C：基于 share 差分的四大转变类别分析
    复用 Module B 的 _load_annual_panel 和 _compute_share_diffs，
    在差分数据基础上按 Built/Natural share 变化分类，生成：
      Fig_Trans_Category_Combined_v2.png
      Table_Trans_Category_ShareBased.csv
    """
    print("\n" + "=" * 60)
    print("  Module C: Share-based 四大类别分析")
    print("=" * 60)

    panels = {
        'CHN': (CHN_PANEL, 'City_ID'),
        'USA': (USA_PANEL, 'State_ID'),
    }

    all_rows = []

    for country, (pkl_path, fe_col) in panels.items():
        if not os.path.exists(pkl_path):
            print(f"  [WARN] 找不到 {pkl_path}，跳过 {country}")
            continue

        print(f"\n  {'─'*50}")
        print(f"  {country}")
        print(f"  {'─'*50}")

        # 加载 & 差分（复用 Module B 函数）
        annual, std_lcz_cols = _load_annual_panel(pkl_path, fe_col)
        diff_df = _compute_share_diffs(annual, fe_col, std_lcz_cols)
        del annual; gc.collect()

        # 分类
        print("    按 Built/Natural share 变化分类像元...")
        cat_df = _classify_pixel_category(diff_df, std_lcz_cols, share_threshold=0.02)
        del diff_df; gc.collect()

        # 年度统计
        stats_df = _category_annual_stats(cat_df)
        stats_df.insert(0, 'Country', country)
        all_rows.append(stats_df)
        del cat_df; gc.collect()
        print(f"    {country} 完成，年份数: {stats_df['Year'].nunique()}")

    if not all_rows:
        print("  [ERROR] 没有有效数据，跳过绘图")
        return

    df_all = pd.concat(all_rows, ignore_index=True)

    # 保存表格
    out_tbl = os.path.join(OUT_DIR, 'Table_Trans_Category_ShareBased.csv')
    df_all.to_csv(out_tbl, index=False, encoding='utf-8-sig')
    print(f"\n  Saved: {out_tbl}  ({len(df_all):,} 行)")

    # 绘图
    fig = fig_category_combined(df_all, BREAK_YEARS)
    out_fig = os.path.join(OUT_DIR, 'Fig_Trans_Category_Combined_v2')
    _save_fig(fig, out_fig)
    plt.close(fig)
    print(f"  Saved: {out_fig}.png / .svg")
    print("  Module C 完成。")


# ══════════════════════════════════════════════════════════════
# 模块 D：Share-based 替代图
#   D1 — LCZ 转变频率热力图（基于 Table_Trans1_Annual_Detail.csv）
#   D2 — 转变大类时序动态（基于 Table_Trans_Category_ShareBased.csv）
# ══════════════════════════════════════════════════════════════

CAT_COLORS = {
    'Natural→Natural': '#27AE60',
    'Natural→Built':   '#E74C3C',
    'Built→Built':     '#E67E22',
    'Built→Natural':   '#2980B9',
}
ALL_LCZ_CODES = list(range(1, 18))   # 1-10 Built, 11-17 Natural


def module_d1_frequency_heatmap():
    """
    基于 Table_Trans1_Annual_Detail.csv 构建 17×17 LCZ 转变频率热力图。
    使用 share-based 面板数据的转变计数（Count），不依赖 dominant LCZ。
    输出：Fig_Trans1_Frequency_Heatmap.png（覆盖旧图）
    """
    print("\n" + "=" * 60)
    print("  Module D1: LCZ 转变频率热力图（share-based）")
    print("=" * 60)

    if not os.path.exists(TRANS1_CSV):
        print(f"  [ERROR] 找不到 {TRANS1_CSV}")
        return

    df = pd.read_csv(TRANS1_CSV, encoding='utf-8-sig')
    tick_labels = [LCZ_SHORT[c] for c in ALL_LCZ_CODES]
    n = len(ALL_LCZ_CODES)
    divider_pos = 9.5   # Built(0-9) / Natural(10-16) 分隔线

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    for ax, country in zip(axes, ['USA', 'CHN']):
        sub = df[df['Country'] == country]

        # 构建 17×17 频率矩阵（行=from，列=to）
        freq = np.zeros((n, n), dtype=float)
        code_idx = {c: i for i, c in enumerate(ALL_LCZ_CODES)}

        agg = (sub.groupby(['LCZ_from', 'LCZ_to'])['Count'].sum().reset_index())
        for _, row in agg.iterrows():
            fi = code_idx.get(int(row['LCZ_from']))
            ti = code_idx.get(int(row['LCZ_to']))
            if fi is not None and ti is not None:
                freq[fi, ti] += row['Count']

        # 对角线=稳定（单独显示灰色）
        log_freq = np.log10(freq + 1)
        np.fill_diagonal(log_freq, np.nan)

        im = ax.imshow(log_freq, cmap='YlOrRd', aspect='auto',
                       origin='upper', vmin=0)

        # 对角线灰色填充
        for i in range(n):
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                       color='#CCCCCC', zorder=2))

        # Built/Natural 分隔线
        ax.axhline(divider_pos, color='#444', linewidth=1.0, linestyle='--')
        ax.axvline(divider_pos, color='#444', linewidth=1.0, linestyle='--')
        ax.text(divider_pos - 5, -1.2, 'Built',   ha='center',
                fontsize=FS_LEGEND, color='#444', fontweight='bold')
        ax.text(divider_pos + 3, -1.2, 'Natural', ha='center',
                fontsize=FS_LEGEND, color='#444', fontweight='bold')

        ax.set_xticks(range(n)); ax.set_xticklabels(tick_labels, fontsize=FS_TICK - 3)
        ax.set_yticks(range(n)); ax.set_yticklabels(tick_labels, fontsize=FS_TICK - 3)
        ax.set_xlabel('LCZ to (t+1)', fontsize=FS_TITLE - 2)
        ax.set_ylabel('LCZ from (t)',  fontsize=FS_TITLE - 2)
        ax.set_title(f'{country}  —  LCZ Transition Frequency\n'
                     f'(log₁₀ pixel-year count, 2001–2019, diagonal=stable)\n'
                     f'[share-based panel]',
                     fontsize=FS_TITLE - 3, pad=10)

        cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label('log₁₀ (pixel-year count)', fontsize=FS_LEGEND - 1)

        # 数值标注 Top 10%
        freq_flat = freq.copy(); np.fill_diagonal(freq_flat, 0)
        thresh = np.percentile(freq_flat[freq_flat > 0], 90)
        for i in range(n):
            for j in range(n):
                if i != j and freq_flat[i, j] >= thresh:
                    ax.text(j, i, f'{int(freq_flat[i,j]):,}',
                            ha='center', va='center',
                            fontsize=6.5, color='#1a1a1a', zorder=3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'Fig_Trans1_Frequency_Heatmap')
    _save_fig(fig, out)
    plt.close(fig)
    print(f"  Saved: {out}.png / .svg")


def module_d2_temporal_dynamics():
    """
    基于 Table_Trans_Category_ShareBased.csv 生成转变大类时序动态图。
    Y轴：各类别像元年数占当年总转变像元年数的百分比。
    输出：Fig_Trans4_Temporal_Dynamics.png（覆盖旧图）
    """
    print("\n" + "=" * 60)
    print("  Module D2: 转变大类时序动态（share-based）")
    print("=" * 60)

    cat_csv = os.path.join(OUT_DIR, 'Table_Trans_Category_ShareBased.csv')
    if not os.path.exists(cat_csv):
        print(f"  [ERROR] 找不到 {cat_csv}，请先运行 Module C")
        return

    df = pd.read_csv(cat_csv, encoding='utf-8-sig')

    # 每年各类别像元数 & 占比
    yr_total = df.groupby(['Country', 'Year'])['Pixel_Count'].sum().reset_index()
    yr_total.rename(columns={'Pixel_Count': 'Total'}, inplace=True)
    df = df.merge(yr_total, on=['Country', 'Year'])
    df['Pct'] = df['Pixel_Count'] / df['Total'] * 100

    cats = ['Natural→Natural', 'Natural→Built', 'Built→Built', 'Built→Natural']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, country in zip(axes, ['USA', 'CHN']):
        bk  = BREAK_YEARS[country]
        sub = df[df['Country'] == country]

        for cat in cats:
            cat_sub = sub[sub['Category'] == cat].sort_values('Year')
            col = CAT_COLORS.get(cat, '#888888')
            ax.plot(cat_sub['Year'], cat_sub['Pct'],
                    color=col, linewidth=LW + 0.5,
                    marker='o', markersize=4, label=cat)
            ax.fill_between(cat_sub['Year'], cat_sub['Pct'],
                            alpha=0.07, color=col)

        # 结构断点线
        ax.axvline(bk, color='#555', linewidth=1.0, linestyle='--', alpha=0.8)
        ylim = ax.get_ylim()
        ax.text(bk + 0.2, (ylim[1] if ylim[1] else 60) * 0.95,
                f'Break\n{bk}', fontsize=FS_LEGEND - 3,
                color='#555', va='top')

        # 事件线
        for yr, lbl in {2008: '2008\nCrisis', 2015: 'Paris'}.items():
            ax.axvline(yr, color='#BBBBBB', linewidth=0.7, linestyle=':')
            ax.text(yr, (ylim[1] if ylim[1] else 60) * 0.85,
                    lbl, ha='center', fontsize=FS_LEGEND - 4, color='#AAA')

        ax.set_xticks(YEAR_TICKS)
        ax.set_xticklabels([str(y) for y in YEAR_TICKS], fontsize=FS_TICK - 1)
        ax.set_xlim(1999, 2020)
        ax.set_ylim(0, None)
        ax.set_xlabel('Year', fontsize=FS_TITLE - 2)
        ax.set_ylabel('% of all transition pixel-years', fontsize=FS_TITLE - 2)
        ax.set_title(f'{country}  —  Transition Category Dynamics\n[share-based classification]',
                     fontsize=FS_TITLE - 2, pad=8)
        style_ax(ax)
        ax.legend(fontsize=FS_LEGEND - 2, loc='upper right',
                  bbox_to_anchor=(1, 1))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'Fig_Trans4_Temporal_Dynamics')
    _save_fig(fig, out)
    plt.close(fig)
    print(f"  Saved: {out}.png / .svg")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Code_LCZ_Transition_v2.py")
    print("  Module A: 时段对比  |  Module B: 份额差分")
    print("=" * 60)

    # ── 模块 A（需要 Table_Trans1_Annual_Detail.csv 已存在）
    module_a_period_split(top_n=15)

    # ── 模块 B（直接从 panel pkl 计算，无需缓存）
    module_b_share_diff()

    # ── 模块 C（share-based 四大类别分析）
    module_c_category_analysis()

    # ── 模块 D（share-based 替代图：热力图 + 时序动态）
    module_d1_frequency_heatmap()
    module_d2_temporal_dynamics()

    print("\n" + "=" * 60)
    print(f"  全部输出至: {OUT_DIR}")
    print("  Module A: Fig_Trans3a/3b  +  Table_Trans4")
    print("  Module B: Fig_Trans5/6/7 (USA+CHN)  +  Table_Trans5/6")
    print("  Module C: Fig_Trans_Category_Combined_v2  +  Table_Trans_Category_ShareBased")
    print("  Module D: Fig_Trans1_Frequency_Heatmap  +  Fig_Trans4_Temporal_Dynamics")
    print("=" * 60)
