# -*- coding: utf-8 -*-
# Code_FE_Decomposition.py
# =============================================================
# 基于 FE 回归系数的 CO₂ 变化贡献量分解（Kaya 框架对应）
#
# 分解逻辑：
#   FE 模型已估计：
#     ln_CO2 = β_Pop·ln_Pop + β_GDP·ln_GDP + Σᵢ βᵢ·LCZᵢ_share + FE + ε
#
#   对年际变化做一阶近似（链式法则）：
#     Δln_CO2 ≈ β_Pop·Δln_Pop + β_GDP·Δln_GDP + Σᵢ βᵢ·ΔLCZᵢ_share
#
#   将 Δln_CO2 ≈ ΔCO2/CO2，再乘以 CO2 基期水平，得绝对贡献量：
#     ΔCO2_hat ≈ CO2_t × [ β_Pop·Δln_Pop + β_GDP·Δln_GDP
#                          + Σᵢ βᵢ·ΔLCZᵢ_share ]
#
#   三效应（与 Kaya 框架对应）：
#     Scale     = CO2_t × β_Pop × Δln_Pop       （人口规模）
#     Affluence = CO2_t × β_GDP × Δln_GDP        （经济水平）
#     Morphology= CO2_t × Σᵢ βᵢ × ΔLCZᵢ_share  （城市形态/LCZ结构）
#
#   说明：
#     - 使用 Model2_Morphology 的系数（含 LCZ 主效应，不含 Season 交互）
#     - 年际差分在像元级别计算，再按国家聚合
#     - CO2_t 使用每像元的年均 CO₂
#
# 输出：
#   Table_FE_Decomp_Annual.csv
#   Fig_FE_Decomp_Stacked.png
#   Fig_FE_Decomp_Cumulative.png
#   Fig_FE_Decomp_Ternary.png
#   Fig_FE_Decomp_Period_Compare.png
#   Fig_FE_Decomp_vs_LMDI.png   （两套方法对比）
# =============================================================

import os, gc, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
warnings.filterwarnings('ignore')

# ────────────────────────── 路径 ─────────────────────────────
USA_PANEL   = r"D:\LCZCarbon\Results_USA_Month\USA_1km\Panel_Monthly_clean.pkl"
CHN_PANEL   = r"D:\LCZCarbon\Results_CHN_Month\CHN_1km\Panel_Monthly_clean.pkl"
USA_COEF    = r"D:\LCZCarbon\Results_USA_Month\USA_1km\Model2_Morphology_Coefs.csv"
CHN_COEF    = r"D:\LCZCarbon\Results_CHN_Month\CHN_1km\Model2_Morphology_Coefs.csv"
# LMDI 方法已放弃，相关文件已删除
OUT_DIR     = r"D:\LCZCarbon\Results_DescStats_v2"
os.makedirs(OUT_DIR, exist_ok=True)

BREAK_YEARS  = {'USA': 2009, 'CHN': 2013}
SAMPLE_PER_YM = 6_000
RANDOM_SEED   = 42

LCZ_SHORT = {
    1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',
    11:'A',12:'B',13:'C',14:'D',15:'E',16:'F',17:'G',
}

# ────────────────────────── 样式 ─────────────────────────────
matplotlib.rcParams.update({
    'font.family':        'Arial',
    'font.size':          14,
    # 四边完整轴框（Origin 风格）
    'axes.spines.top':    True,
    'axes.spines.right':  True,
    'axes.spines.left':   True,
    'axes.spines.bottom': True,
    'axes.linewidth':     1.0,
    # 刻度线朝内
    'xtick.major.width':  1.0,
    'ytick.major.width':  1.0,
    'xtick.major.size':   5,
    'ytick.major.size':   5,
    'xtick.direction':    'in',
    'ytick.direction':    'in',
    # 无网格，白色背景
    'axes.grid':          False,
    'legend.frameon':     True,
    'legend.framealpha':  0.9,
    'legend.edgecolor':   '#CCCCCC',
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'savefig.dpi':        600,
    'savefig.bbox':       'tight',
})

FS_TITLE  = 18
FS_TICK   = 16
FS_LEGEND = 14
LW_AX     = 1.0   # 轴线宽度

def _style_ax(ax):
    """两轴框（左+下），刻度朝外，无网格，Arial字体"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(LW_AX)
    ax.spines['bottom'].set_linewidth(LW_AX)
    ax.tick_params(axis='both', labelsize=FS_TICK,
                   width=LW_AX, length=5, direction='out',
                   top=False, right=False)
    ax.grid(False)


def _save_fig(fig, path_no_ext):
    """同时保存 PNG（600dpi）和 SVG（可编辑矢量）"""
    fig.savefig(path_no_ext + '.png', dpi=600, bbox_inches='tight')
    fig.savefig(path_no_ext + '.svg', bbox_inches='tight')


EFFECT_COLORS = {
    'Scale':              '#E74C3C',   # 红：人口规模
    'Affluence':          '#9B59B6',   # 紫：经济水平
    'Morphology_Built':   '#E67E22',   # 橙：建成形态变化
    'Morphology_Natural': '#27AE60',   # 绿：自然覆盖变化
    'Morphology':         '#2980B9',   # 蓝：合并（兼容旧图）
}
EFFECT_LABELS = {
    'Scale':              'Scale Effect (Population)',
    'Affluence':          'Affluence Effect (GDP)',
    'Morphology_Built':   'Built Morphology (LCZ 1–10 change)',
    'Morphology_Natural': 'Natural Morphology (LCZ A–G change)',
    'Morphology':         'Morphology Effect (LCZ structure)',
}

BUILT_CODES_FE   = set(range(1, 11))
NATURAL_CODES_FE = set(range(11, 18))
YEAR_TICKS = list(range(2000, 2020, 3))


# ══════════════════════════════════════════════════════════════
# Step 1：读取 FE 系数
# ══════════════════════════════════════════════════════════════

def load_fe_coefs(coef_path):
    """
    返回字典：
      {'ln_Pop': β_pop, 'ln_GDP': β_gdp,
       'LCZ1_share': β1, ..., 'LCZ17_share': β17}
    """
    df   = pd.read_csv(coef_path)
    df   = df.rename(columns={'Unnamed: 0': 'term'})
    coef = {}
    for _, row in df.iterrows():
        t = str(row['term']).strip()
        if t in ('ln_Pop', 'ln_GDP'):
            coef[t] = float(row['coef'])
        elif t.startswith('LCZ') and t.endswith('_share'):
            coef[t] = float(row['coef'])
    return coef


# ══════════════════════════════════════════════════════════════
# Step 2：构建年度像元级差分数据
# ══════════════════════════════════════════════════════════════

def load_annual_panel(pkl_path, fe_col, sample_per_ym=SAMPLE_PER_YM):
    """
    返回年度像元面板：Year, fe_col, pix_idx,
                      CO2_annual, ln_Pop_annual, ln_GDP_annual,
                      LCZ1_share … LCZ17_share
    """
    print(f"  读取 {pkl_path} ...")
    df = pd.read_pickle(pkl_path)
    print(f"  原始行数: {len(df):,}")

    n_ym = len(df['Year'].unique()) * 12
    if sample_per_ym and sample_per_ym < len(df) // n_ym:
        ym_key = (df['Year'].astype(str) + '_' +
                  df['Month'].astype(str).str.zfill(2))
        df = (df.groupby(ym_key, group_keys=False)
                .apply(lambda g: g.sample(n=min(sample_per_ym, len(g)),
                                          random_state=RANDOM_SEED))
                .reset_index(drop=True))
        print(f"  抽样后: {len(df):,} 行")

    # 标准化 LCZ 列名
    share_cols = [c for c in df.columns if c.endswith('_share')]
    lcz_map = {}
    for col in share_cols:
        try:
            code = int(col.replace('LCZ','').replace('_share',''))
        except ValueError:
            continue
        std = code - 90 if 101 <= code <= 107 else code
        if 1 <= std <= 17:
            lcz_map[col] = std
    df = df.rename(columns={c: f'LCZ{s}_share' for c,s in lcz_map.items()})
    std_lcz = [f'LCZ{s}_share' for s in sorted(set(lcz_map.values()))]

    # pix_idx
    df['pix_idx'] = df.groupby(['Year','Month',fe_col]).cumcount()

    # 年度 CO₂ 均值
    co2_ann = (df.groupby(['Year',fe_col,'pix_idx'])['CO2']
               .mean().reset_index(name='CO2_annual'))

    # Pop / GDP（Month=1）
    df1 = df[df['Month'] == 1].copy()
    df1['pix_idx'] = df1.groupby(['Year',fe_col]).cumcount()

    # 对数变换
    for col in ['Pop','GDP']:
        if col in df1.columns:
            df1[f'ln_{col}'] = np.log(df1[col].clip(lower=0) + 1)

    keep_cols = (['Year', fe_col, 'pix_idx'] +
                 [c for c in ['ln_Pop','ln_GDP'] if c in df1.columns] +
                 std_lcz)
    annual = co2_ann.merge(df1[keep_cols],
                           on=['Year',fe_col,'pix_idx'], how='inner')
    del df, df1, co2_ann; gc.collect()
    print(f"  年度面板: {len(annual):,} 行")
    return annual, std_lcz


def compute_fe_decomp(annual_df, fe_col, std_lcz, coef_dict, min_years=5):
    """
    逐像元逐年计算三效应绝对贡献量，再按国家-年聚合。

    三效应（绝对量近似）：
      Scale_it     = CO2_{t-1} × β_Pop × Δln_Pop_it
      Affluence_it = CO2_{t-1} × β_GDP × Δln_GDP_it
      Morphology_it= CO2_{t-1} × Σᵢ βᵢ × ΔLCZᵢ_share_it
    """
    print("  计算逐像元年际差分 ...")
    annual_df = annual_df.sort_values([fe_col,'pix_idx','Year'])
    grp = [fe_col,'pix_idx']

    diff_cols = ['CO2_annual','ln_Pop','ln_GDP'] + std_lcz
    diff_cols = [c for c in diff_cols if c in annual_df.columns]

    for col in diff_cols:
        annual_df[f'd_{col}'] = annual_df.groupby(grp)[col].diff()

    annual_df.dropna(subset=[f'd_{c}' for c in diff_cols], inplace=True)

    # 只保留出现 ≥ min_years 的像元
    pix_cnt = annual_df.groupby(grp)['Year'].count()
    valid   = pix_cnt[pix_cnt >= min_years].index
    annual_df = annual_df.set_index(grp).loc[valid].reset_index()
    print(f"  差分行数: {len(annual_df):,}")

    # CO2 基期水平（t-1）
    annual_df['CO2_base'] = annual_df['CO2_annual'] - annual_df['d_CO2_annual']

    # 四效应（Scale / Affluence / Morphology_Built / Morphology_Natural）
    beta_pop = coef_dict.get('ln_Pop', 0.0)
    beta_gdp = coef_dict.get('ln_GDP', 0.0)

    annual_df['Scale']     = (annual_df['CO2_base'] * beta_pop
                               * annual_df.get('d_ln_Pop', 0))
    annual_df['Affluence'] = (annual_df['CO2_base'] * beta_gdp
                               * annual_df.get('d_ln_GDP', 0))

    morph_built   = np.zeros(len(annual_df))
    morph_natural = np.zeros(len(annual_df))

    for col in std_lcz:
        code  = int(col.replace('LCZ','').replace('_share',''))
        key   = f'LCZ{code}_share'
        beta  = coef_dict.get(key, 0.0)
        d_col = f'd_{col}'
        if d_col not in annual_df.columns:
            continue
        contrib = annual_df['CO2_base'].values * beta * annual_df[d_col].values
        if code in BUILT_CODES_FE:
            morph_built   += contrib
        else:
            morph_natural += contrib

    annual_df['Morphology_Built']   = morph_built
    annual_df['Morphology_Natural'] = morph_natural
    annual_df['Morphology']         = morph_built + morph_natural  # 合并，兼容旧图

    # 国家年度聚合
    agg = (annual_df.groupby('Year')
           .agg(
               Delta_CO2         = ('d_CO2_annual',       'sum'),
               Scale             = ('Scale',              'sum'),
               Affluence         = ('Affluence',          'sum'),
               Morphology_Built  = ('Morphology_Built',   'sum'),
               Morphology_Natural= ('Morphology_Natural', 'sum'),
               Morphology        = ('Morphology',         'sum'),
               n_pixels          = ('d_CO2_annual',       'count'),   # 每年差分像元数
           ).reset_index())
    agg['Residual'] = (agg['Delta_CO2'] -
                       agg['Scale'] - agg['Affluence'] - agg['Morphology'])

    # ── per-pixel 标准化（ton C / km² / year）────────────────
    eff_cols = ['Delta_CO2','Scale','Affluence',
                'Morphology_Built','Morphology_Natural','Morphology','Residual']
    for col in eff_cols:
        agg[f'{col}_pp'] = agg[col] / agg['n_pixels']

    return agg


# ══════════════════════════════════════════════════════════════
# Step 3：可视化
# ══════════════════════════════════════════════════════════════

def _add_break(ax, bk):
    ax.axvline(bk - 0.5, color='#333', lw=1.3, ls='--', alpha=0.8, zorder=4)
    ylim = ax.get_ylim()
    top  = ylim[1] if ylim[1] != 0 else 1
    ax.text(bk - 0.4, top * 0.96, f'Break\n{bk}',
            fontsize=8.5, color='#333', va='top', ha='right')


def fig_stacked_fe(df_all):
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    for ax, country in zip(axes, ['CHN', 'USA']):
        sub = df_all[df_all['Country'] == country].sort_values('Year')
        bk  = BREAK_YEARS[country]
        yrs = sub['Year'].values
        sc  = 1e3

        pos_bot = np.zeros(len(sub))
        neg_bot = np.zeros(len(sub))
        for eff in ['Scale', 'Affluence', 'Morphology']:
            v  = sub[eff].values / sc
            pv = np.where(v > 0, v, 0)
            nv = np.where(v < 0, v, 0)
            ax.bar(yrs, pv, bottom=pos_bot, color=EFFECT_COLORS[eff],
                   width=0.75, alpha=0.88, label=EFFECT_LABELS[eff], zorder=3)
            ax.bar(yrs, nv, bottom=neg_bot, color=EFFECT_COLORS[eff],
                   width=0.75, alpha=0.88, zorder=3)
            pos_bot += pv;  neg_bot += nv

        ax.plot(yrs, sub['Delta_CO2'].values / sc,
                color='black', lw=1.8, marker='o', ms=4,
                label='Actual ΔCO₂', zorder=5)
        _add_break(ax, bk)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_title(f'{country}  —  FE-based Decomposition\n'
                     '(Scale / Affluence / Morphology)',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('CO₂ change (×10³ ton C/cell)', fontsize=11)
        ax.set_xticks(YEAR_TICKS)
        ax.set_xticklabels([str(y) for y in YEAR_TICKS], rotation=30, ha='right')
        ax.set_xlim(2000.5, 2019.5)
        _style_ax(ax)
        if country == 'CHN':
            ax.legend(fontsize=FS_LEGEND - 2, loc='upper left',
                      framealpha=0.9, ncol=1, bbox_to_anchor=(0, 1))
    plt.tight_layout()
    return fig


def fig_cumulative_fe(df_all):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, country in zip(axes, ['CHN', 'USA']):
        sub = df_all[df_all['Country'] == country].sort_values('Year')
        bk  = BREAK_YEARS[country]
        sc  = 1e3
        for eff in ['Scale', 'Affluence', 'Morphology']:
            ax.plot(sub['Year'], sub[eff].cumsum()/sc,
                    color=EFFECT_COLORS[eff], lw=2.2, marker='o', ms=4,
                    label=EFFECT_LABELS[eff])
        ax.plot(sub['Year'], sub['Delta_CO2'].cumsum()/sc,
                color='black', lw=1.5, ls='--', marker='s', ms=3.5,
                label='Actual ΔCO₂ (cumul.)')
        ax.axvline(bk, color='#333', lw=1.3, ls='--', alpha=0.75)
        ax.axhline(0, color='black', lw=0.7, alpha=0.4)
        ax.set_title(f'{country}  —  Cumulative FE Effects',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Cumulative CO₂ change (×10³ ton C/cell)', fontsize=10)
        ax.set_xticks(YEAR_TICKS)
        ax.set_xticklabels([str(y) for y in YEAR_TICKS], rotation=30, ha='right')
        ax.legend(fontsize=FS_LEGEND - 2, loc='upper left', framealpha=0.9)
        _style_ax(ax)
    plt.tight_layout()
    return fig


def _to_cartesian(s, c, i):
    tot = s + c + i + 1e-10
    x = 0.5 * (2*c + i) / tot
    y = (np.sqrt(3)/2) * i / tot
    return x, y


def fig_ternary_fe(df_all):
    fig, ax = plt.subplots(figsize=(9, 8))
    tri = plt.Polygon([[0,0],[1,0],[0.5,np.sqrt(3)/2]],
                      fill=False, edgecolor='#333', lw=1.5)
    ax.add_patch(tri)
    off = 0.06
    ax.text(0-off, 0-off, 'Scale\n(Population)', ha='center', va='top',
            fontsize=11, fontweight='bold', color=EFFECT_COLORS['Scale'])
    ax.text(1+off, 0-off, 'Affluence\n(GDP)',    ha='center', va='top',
            fontsize=11, fontweight='bold', color=EFFECT_COLORS['Affluence'])
    ax.text(0.5,  np.sqrt(3)/2+off, 'Morphology\n(LCZ structure)',
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            color=EFFECT_COLORS['Morphology'])
    for frac in [0.2, 0.4, 0.6, 0.8]:
        for pts in [
            [_to_cartesian(frac,1-frac,0), _to_cartesian(frac,0,1-frac)],
            [_to_cartesian(0,frac,1-frac), _to_cartesian(1-frac,frac,0)],
            [_to_cartesian(0,1-frac,frac), _to_cartesian(1-frac,0,frac)],
        ]:
            ax.plot([pts[0][0],pts[1][0]],[pts[0][1],pts[1][1]],
                    color='#CCC', lw=0.6, ls=':')
        # ── 刻度值标注 ──
        # Scale 轴（底边左侧，从左顶点沿左边向右）— 标注 Scale 占比
        sc_pt = _to_cartesian(frac, 1-frac, 0)        # Affluence=1-frac, Morph=0
        ax.text(sc_pt[0]-0.03, sc_pt[1]-0.015, f'{frac:.1f}',
                ha='right', va='top', fontsize=7, color='#888')
        # Affluence 轴（底边右侧）— 标注 Affluence 占比
        af_pt = _to_cartesian(0, frac, 1-frac)        # Scale=0, Morph=1-frac
        ax.text(af_pt[0]+0.03, af_pt[1]-0.015, f'{frac:.1f}',
                ha='left', va='top', fontsize=7, color='#888')
        # Morphology 轴（右边）— 标注 Morphology 占比
        mo_pt = _to_cartesian(1-frac, 0, frac)        # Scale=1-frac, Affluence=0
        ax.text(mo_pt[0]+0.03, mo_pt[1]+0.005, f'{frac:.1f}',
                ha='left', va='center', fontsize=7, color='#888')

    for country, cmap, marker in [('CHN', plt.cm.Reds,'o'),
                                   ('USA', plt.cm.Blues,'s')]:
        sub  = df_all[df_all['Country'] == country].sort_values('Year')
        yrs  = sub['Year'].values
        norm = Normalize(vmin=yrs.min(), vmax=yrs.max())
        S = sub['Scale'].abs().values
        C = sub['Affluence'].abs().values
        I = sub['Morphology'].abs().values
        xs, ys = _to_cartesian(S, C, I)
        for i in range(len(sub)):
            ax.scatter(xs[i], ys[i], color=cmap(norm(yrs[i])), s=90,
                       marker=marker, edgecolors='white', lw=0.6, zorder=5)
            if i < len(sub)-1:
                ax.annotate('', xy=(xs[i+1],ys[i+1]), xytext=(xs[i],ys[i]),
                            arrowprops=dict(arrowstyle='->',
                                           color=cmap(0.6), lw=0.8, alpha=0.6))
        ax.annotate(str(yrs[0]),  (xs[0], ys[0]),   fontsize=8,
                    color=cmap(0.5), ha='right')
        ax.annotate(str(yrs[-1]), (xs[-1], ys[-1]), fontsize=8,
                    color=cmap(0.8), ha='left')
        ax.scatter([],[], color=cmap(0.6), marker=marker, s=70,
                   label=country, edgecolors='white')

    ax.set_xlim(-0.18,1.18); ax.set_ylim(-0.18, np.sqrt(3)/2+0.22)
    ax.set_aspect('equal'); ax.axis('off')
    ax.legend(fontsize=FS_LEGEND - 2, loc='upper right', framealpha=0.9,
              bbox_to_anchor=(1.0, 0.98))
    ax.set_title('FE-based Decomposition — Ternary Plot\n'
                 '(Relative share of |Scale|, |Affluence|, |Morphology|;\n'
                 ' arrows = temporal trajectory 2001→2019)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    return fig


def fig_period_fe(df_all):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, country in zip(axes, ['CHN', 'USA']):
        sub   = df_all[df_all['Country'] == country]
        bk    = BREAK_YEARS[country]
        early = sub[sub['Year'] <  bk]
        late  = sub[sub['Year'] >= bk]
        effs  = ['Scale', 'Affluence', 'Morphology']
        x = np.arange(3);  w = 0.35;  sc = 1e3
        c_dark  = '#C0392B' if country == 'CHN' else '#1B6BAF'
        c_light = '#F4A6A6' if country == 'CHN' else '#87CEEB'
        e_m = [early[e].mean()/sc for e in effs]
        l_m = [late[e].mean() /sc for e in effs]
        b1 = ax.bar(x-w/2, e_m, width=w, color=c_dark,  alpha=0.88,
                    label=f'Early (2000–{bk-1})')
        b2 = ax.bar(x+w/2, l_m, width=w, color=c_light, alpha=0.88,
                    label=f'Late ({bk}–2019)')
        for bar in list(b1)+list(b2):
            h = bar.get_height()
            if abs(h) > 0.005:
                ax.text(bar.get_x()+bar.get_width()/2,
                        h+(0.05 if h>=0 else -0.05), f'{h:.2f}',
                        ha='center', va='bottom' if h>=0 else 'top',
                        fontsize=8, color='#333')
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(['Scale\n(Pop)','Affluence\n(GDP)',
                            'Morphology\n(LCZ)'], fontsize=10)
        ax.set_ylabel('Mean annual CO₂ change (×10³ ton C/cell)', fontsize=FS_TITLE - 2)
        ax.set_title(f'{country}  —  Early vs. Late (FE-based)\n'
                     f'(break = {bk})', fontsize=FS_TITLE, pad=10)
        ax.legend(fontsize=FS_LEGEND, framealpha=0.9, loc='upper left',
                  bbox_to_anchor=(0, 1))
        _style_ax(ax)
    plt.tight_layout()
    return fig


def fig_stacked_split(df_all):
    """
    新图：四效应堆叠柱（Scale / Affluence / Morphology_Built / Morphology_Natural）
    """
    EFFS = ['Scale', 'Affluence', 'Morphology_Built', 'Morphology_Natural']
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    for ax, country in zip(axes, ['CHN', 'USA']):
        sub = df_all[df_all['Country'] == country].sort_values('Year')
        bk  = BREAK_YEARS[country]
        yrs = sub['Year'].values
        sc  = 1e3

        pos_bot = np.zeros(len(sub))
        neg_bot = np.zeros(len(sub))
        for eff in EFFS:
            v  = sub[eff].values / sc
            pv = np.where(v > 0, v, 0)
            nv = np.where(v < 0, v, 0)
            ax.bar(yrs, pv, bottom=pos_bot, color=EFFECT_COLORS[eff],
                   width=0.75, alpha=0.88, label=EFFECT_LABELS[eff], zorder=3)
            ax.bar(yrs, nv, bottom=neg_bot, color=EFFECT_COLORS[eff],
                   width=0.75, alpha=0.88, zorder=3)
            pos_bot += pv;  neg_bot += nv

        ax.plot(yrs, sub['Delta_CO2'].values / sc,
                color='black', lw=1.8, marker='o', ms=4,
                label='Actual ΔCO₂', zorder=5)
        _add_break(ax, bk)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_title(f'{country}  —  FE Decomposition (4 Effects)\n'
                     'Scale / Affluence / Built Morphology / Natural Morphology',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('CO₂ change (×10³ ton C/cell)', fontsize=11)
        ax.set_xticks(YEAR_TICKS)
        ax.set_xticklabels([str(y) for y in YEAR_TICKS], rotation=30, ha='right')
        ax.set_xlim(2000.5, 2019.5)
        _style_ax(ax)
        if country == 'CHN':
            ax.legend(fontsize=FS_LEGEND - 2, loc='upper left',
                      framealpha=0.9, ncol=2, bbox_to_anchor=(0, 1))
    plt.tight_layout()
    return fig


def fig_period_split(df_all):
    """
    新图：早/晚期四效应对比柱状图（Built vs Natural Morphology分开）
    """
    EFFS = ['Scale', 'Affluence', 'Morphology_Built', 'Morphology_Natural']
    XLABELS = ['Scale\n(Pop)', 'Affluence\n(GDP)',
               'Built\nMorphology', 'Natural\nMorphology']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, country in zip(axes, ['CHN', 'USA']):
        sub   = df_all[df_all['Country'] == country]
        bk    = BREAK_YEARS[country]
        early = sub[sub['Year'] <  bk]
        late  = sub[sub['Year'] >= bk]
        x  = np.arange(len(EFFS));  w = 0.35;  sc = 1e3
        c_dark  = '#C0392B' if country == 'CHN' else '#1B6BAF'
        c_light = '#F4A6A6' if country == 'CHN' else '#87CEEB'

        e_m = [early[e].mean()/sc for e in EFFS]
        l_m = [late[e].mean() /sc for e in EFFS]

        b1 = ax.bar(x - w/2, e_m, width=w, color=c_dark,  alpha=0.88,
                    label=f'Early (2000–{bk-1})')
        b2 = ax.bar(x + w/2, l_m, width=w, color=c_light, alpha=0.88,
                    label=f'Late ({bk}–2019)')

        # 颜色区分 Built(橙)/Natural(绿) Morphology 柱
        for bars, vals in [(b1, e_m), (b2, l_m)]:
            for i, (bar, val) in enumerate(zip(bars, vals)):
                if EFFS[i] == 'Morphology_Built':
                    bar.set_facecolor(EFFECT_COLORS['Morphology_Built'])
                elif EFFS[i] == 'Morphology_Natural':
                    bar.set_facecolor(EFFECT_COLORS['Morphology_Natural'])
                if abs(val) > 0.005:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            val + (0.05 if val >= 0 else -0.05),
                            f'{val:.2f}',
                            ha='center', va='bottom' if val >= 0 else 'top',
                            fontsize=8, color='#333')

        ax.axhline(0, color='black', lw=0.8)
        # 分割线区分 Kaya因子 vs LCZ形态因子
        ax.axvline(1.5, color='#999', lw=0.8, ls='--', alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels(XLABELS, fontsize=FS_TICK)
        ax.set_ylabel('Mean annual CO₂ change (×10³ ton C/cell)', fontsize=FS_TITLE - 2)
        ax.set_title(f'{country}  —  4-Effect FE Decomposition\n'
                     f'Early vs. Late (break = {bk})',
                     fontsize=FS_TITLE, pad=10)
        ax.legend(fontsize=FS_LEGEND, framealpha=0.9, loc='upper left',
                  bbox_to_anchor=(0, 1))
        _style_ax(ax)

    plt.tight_layout()
    return fig


def fig_morph_decomp(df_all):
    """
    新图：Built vs Natural Morphology Effect 逐年对比折线（中美各一格）
    突出两个子效应方向对比
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, country in zip(axes, ['CHN', 'USA']):
        sub = df_all[df_all['Country'] == country].sort_values('Year')
        bk  = BREAK_YEARS[country]
        sc  = 1e3

        ax.plot(sub['Year'], sub['Morphology_Built']  /sc,
                color=EFFECT_COLORS['Morphology_Built'],   lw=2.2,
                marker='o', ms=5, label='Built Morphology (LCZ 1–10)')
        ax.plot(sub['Year'], sub['Morphology_Natural']/sc,
                color=EFFECT_COLORS['Morphology_Natural'], lw=2.2,
                marker='s', ms=5, label='Natural Morphology (LCZ A–G)')
        ax.plot(sub['Year'], sub['Morphology']        /sc,
                color='#555', lw=1.5, ls='--', marker='^', ms=4,
                label='Total Morphology (net)', alpha=0.7)

        ax.axvline(bk - 0.5, color='#333', lw=1.3, ls='--', alpha=0.8)
        ax.axhline(0, color='black', lw=0.8, alpha=0.5)
        ylim = ax.get_ylim()
        ax.text(bk - 0.4, ylim[1]*0.95 if ylim[1] else 1,
                f'Break\n{bk}', fontsize=8.5, color='#333', va='top', ha='right')

        ax.set_title(f'{country}  —  Built vs. Natural Morphology Effect\n'
                     '(FE-based, annual contribution to CO₂ change)',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('CO₂ change (×10³ ton C/cell)', fontsize=10)
        ax.set_xticks(YEAR_TICKS)
        ax.set_xticklabels([str(y) for y in YEAR_TICKS], rotation=30, ha='right')
        ax.set_xlim(1999.5, 2019.5)
        ax.legend(fontsize=FS_LEGEND - 2, framealpha=0.9, loc='upper left',
                  bbox_to_anchor=(0, 1))
        _style_ax(ax)

    fig.suptitle('LCZ Morphology Effect: Built vs. Natural Sub-components',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# Step 4：Per-pixel 标准化图 & 分期相对贡献率图
# ══════════════════════════════════════════════════════════════

def fig_period_pp(df_all):
    """
    Per-pixel 早/晚期对比图（四效应，单位 ton C / km² / year）
    每种效应 Early=深色 / Late=浅色，Morphology_Built/Natural 各有独立深浅色对
    """
    EFFS    = ['Scale_pp', 'Affluence_pp', 'Morphology_Built_pp', 'Morphology_Natural_pp']
    XLABELS = ['Scale\n(Pop)', 'Affluence\n(GDP)', 'Built\nMorphology', 'Natural\nMorphology']

    # 每种效应的 Early(深) / Late(浅) 颜色对
    EFF_COLORS_EARLY = {
        'Scale_pp':              '#C0392B',   # 深红
        'Affluence_pp':          '#7D3C98',   # 深紫
        'Morphology_Built_pp':   '#D35400',   # 深橙
        'Morphology_Natural_pp': '#1E8449',   # 深绿
    }
    EFF_COLORS_LATE = {
        'Scale_pp':              '#F1948A',   # 浅红
        'Affluence_pp':          '#C39BD3',   # 浅紫
        'Morphology_Built_pp':   '#F0B27A',   # 浅橙
        'Morphology_Natural_pp': '#82E0AA',   # 浅绿
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, country in zip(axes, ['CHN', 'USA']):
        sub   = df_all[df_all['Country'] == country]
        bk    = BREAK_YEARS[country]
        early = sub[sub['Year'] <  bk]
        late  = sub[sub['Year'] >= bk]
        x  = np.arange(len(EFFS));  w = 0.35

        e_m = [early[e].mean() for e in EFFS]
        l_m = [late[e].mean()  for e in EFFS]

        # 逐效应绘制，每种效应有自己的深/浅色
        for i, (eff, xl) in enumerate(zip(EFFS, XLABELS)):
            c_e = EFF_COLORS_EARLY[eff]
            c_l = EFF_COLORS_LATE[eff]
            b1 = ax.bar(x[i] - w/2, e_m[i], width=w, color=c_e, alpha=0.92,
                        label=f'Early (2000–{bk-1})' if i == 0 else '_nolegend_')
            b2 = ax.bar(x[i] + w/2, l_m[i], width=w, color=c_l, alpha=0.92,
                        label=f'Late ({bk}–2019)' if i == 0 else '_nolegend_')
            # 数值标注（bar 是 BarContainer，取 [0] 得到 Rectangle）
            for bc, val in [(b1, e_m[i]), (b2, l_m[i])]:
                if abs(val) > 1e-4:
                    rect = bc[0]
                    ax.text(rect.get_x() + rect.get_width()/2,
                            val + (0.003 if val >= 0 else -0.003),
                            f'{val:.3f}',
                            ha='center', va='bottom' if val >= 0 else 'top',
                            fontsize=FS_LEGEND - 2, color='#333')

        ax.axhline(0, color='black', lw=LW_AX)
        # 分隔线：社会经济 vs LCZ形态
        ax.axvline(1.5, color='#999', lw=0.8, ls='--', alpha=0.7)
        ax.text(0.5, ax.get_ylim()[1] if ax.get_ylim()[1] else 0.1,
                '← Socioeconomic', ha='center', va='bottom',
                fontsize=FS_LEGEND - 3, color='#888')
        ax.text(2.5, ax.get_ylim()[1] if ax.get_ylim()[1] else 0.1,
                'LCZ Morphology →', ha='center', va='bottom',
                fontsize=FS_LEGEND - 3, color='#888')

        ax.set_xticks(x)
        ax.set_xticklabels(XLABELS, fontsize=FS_TICK)
        ax.set_ylabel('Mean annual CO₂ change\n(ton C / km² / year)', fontsize=FS_TITLE - 2)
        ax.set_title(f'{country}  —  Per-pixel FE Decomposition\n'
                     f'Early vs. Late (break year = {bk})',
                     fontsize=FS_TITLE, pad=10)
        ax.legend(fontsize=FS_LEGEND, loc='upper left',
                  bbox_to_anchor=(0, 1), framealpha=0.9)
        _style_ax(ax)

    plt.tight_layout()
    return fig


def compute_period_share(df_all):
    """
    计算分期累计相对贡献率（%）。
    分母 = |累计 Delta_CO2|，避免逐年分母不稳定。
    效应列：Scale / Affluence / Morphology_Built / Morphology_Natural / Residual
    正值 = 增碳贡献，负值 = 减碳贡献（与 Delta_CO2 同号为正贡献）
    """
    EFFS = ['Scale', 'Affluence', 'Morphology_Built', 'Morphology_Natural', 'Residual']
    rows = []
    for country in ['CHN', 'USA']:
        sub = df_all[df_all['Country'] == country]
        bk  = BREAK_YEARS[country]
        for period, mask in [('Early', sub['Year'] < bk),
                              ('Late',  sub['Year'] >= bk)]:
            seg    = sub[mask]
            denom  = seg['Delta_CO2'].sum()
            denom_abs = abs(denom) if abs(denom) > 1 else 1   # 防止分母过小
            row = {'Country': country, 'Period': period,
                   'Delta_CO2_sum': denom}
            for eff in EFFS:
                row[f'{eff}_sum']   = seg[eff].sum()
                row[f'{eff}_pct']   = seg[eff].sum() / denom_abs * 100
            rows.append(row)
    return pd.DataFrame(rows)


def fig_share_period(share_df):
    """
    分期相对贡献率堆叠柱（正 + 负效应分开堆叠）
    每组4个柱：CHN Early / CHN Late / USA Early / USA Late
    """
    EFFS   = ['Scale', 'Affluence', 'Morphology_Built', 'Morphology_Natural', 'Residual']
    COLORS = [EFFECT_COLORS['Scale'], EFFECT_COLORS['Affluence'],
              EFFECT_COLORS['Morphology_Built'], EFFECT_COLORS['Morphology_Natural'],
              '#AAB7B8']
    LABELS = ['Scale (Pop)', 'Affluence (GDP)',
              'Built Morphology', 'Natural Morphology', 'Residual']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, country in zip(axes, ['CHN', 'USA']):
        sub = share_df[share_df['Country'] == country]
        periods = ['Early', 'Late']
        x = np.arange(len(periods));  w = 0.55

        pos_bot = np.zeros(len(periods))
        neg_bot = np.zeros(len(periods))

        for eff, col, lbl in zip(EFFS, COLORS, LABELS):
            vals = np.array([sub[sub['Period'] == p][f'{eff}_pct'].values[0]
                             for p in periods])
            pv = np.where(vals > 0, vals, 0)
            nv = np.where(vals < 0, vals, 0)
            ax.bar(x, pv, width=w, bottom=pos_bot, color=col,
                   alpha=0.88, label=lbl, zorder=3)
            ax.bar(x, nv, width=w, bottom=neg_bot, color=col,
                   alpha=0.88, zorder=3)
            # 标注非零值
            for i, (pvi, nvi) in enumerate(zip(pv, nv)):
                if abs(pvi) > 2:
                    ax.text(x[i], pos_bot[i] + pvi/2, f'{pvi:.1f}%',
                            ha='center', va='center', fontsize=8,
                            color='white', fontweight='bold')
                if abs(nvi) > 2:
                    ax.text(x[i], neg_bot[i] + nvi/2, f'{nvi:.1f}%',
                            ha='center', va='center', fontsize=8,
                            color='white', fontweight='bold')
            pos_bot += pv;  neg_bot += nv

        bk = BREAK_YEARS[country]
        xlabels = [f'Early\n(2000–{bk-1})', f'Late\n({bk}–2019)']
        ax.set_xticks(x);  ax.set_xticklabels(xlabels, fontsize=FS_TICK)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_ylabel('Contribution to ΔCO₂ (%)', fontsize=FS_TITLE - 2)
        ax.set_title(f'{country}  —  Relative Contribution by Effect\n'
                     '(% of total cumulative ΔCO₂ per period)',
                     fontsize=FS_TITLE, pad=10)
        if country == 'CHN':
            ax.legend(fontsize=FS_LEGEND - 2, loc='upper left',
                      framealpha=0.9, ncol=2, bbox_to_anchor=(0, 1))
        _style_ax(ax)

    plt.tight_layout()
    return fig


def fig_share_heatmap(share_df):
    """
    热力图：各效应×分期×国家的相对贡献率（%）
    便于论文中直观对比四格
    """
    EFFS   = ['Scale', 'Affluence', 'Morphology_Built', 'Morphology_Natural', 'Residual']
    LABELS = ['Scale\n(Pop)', 'Affluence\n(GDP)',
              'Built\nMorphology', 'Natural\nMorphology', 'Residual']

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, country in zip(axes, ['CHN', 'USA']):
        sub = share_df[share_df['Country'] == country]
        bk  = BREAK_YEARS[country]
        mat = np.array([
            [sub[sub['Period']=='Early'][f'{e}_pct'].values[0] for e in EFFS],
            [sub[sub['Period']=='Late' ][f'{e}_pct'].values[0] for e in EFFS],
        ])
        vmax = max(abs(mat).max(), 10)
        im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_xticks(range(len(EFFS)));  ax.set_xticklabels(LABELS, fontsize=9)
        ax.set_yticks([0,1])
        ax.set_yticklabels([f'Early (2000–{bk-1})', f'Late ({bk}–2019)'], fontsize=10)
        for i in range(2):
            for j in range(len(EFFS)):
                val = mat[i, j]
                ax.text(j, i, f'{val:.1f}%',
                        ha='center', va='center', fontsize=9,
                        color='white' if abs(val) > vmax*0.4 else '#333',
                        fontweight='bold')
        plt.colorbar(im, ax=ax, label='% of ΔCO₂', shrink=0.8)
        ax.set_title(f'{country}  —  Effect Share (%)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    return fig


def fig_vs_lmdi(df_fe, lmdi_csv):
    """
    对比图：FE分解 vs LMDI分解的 Morphology/Composition Effect（逐年）
    """
    if not os.path.exists(lmdi_csv):
        print("  [WARN] LMDI v2 CSV 未找到，跳过对比图")
        return None

    lmdi = pd.read_csv(lmdi_csv)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, country in zip(axes, ['CHN', 'USA']):
        fe   = df_fe[df_fe['Country'] == country].sort_values('Year')
        lm   = lmdi[lmdi['Country'] == country].sort_values('Year')
        bk   = BREAK_YEARS[country]
        sc   = 1e3
        c    = '#C0392B' if country == 'CHN' else '#1B6BAF'

        # FE Morphology
        ax.plot(fe['Year'], fe['Morphology']/sc,
                color=c, lw=2, marker='o', ms=5,
                label='FE: Morphology Effect')
        # LMDI Composition
        if 'Composition' in lm.columns:
            ax.plot(lm['Year'], lm['Composition']/sc,
                    color=c, lw=2, ls='--', marker='s', ms=5,
                    label='LMDI: Composition Effect')

        ax.axvline(bk-0.5, color='#333', lw=1.3, ls='--', alpha=0.8)
        ax.axhline(0, color='black', lw=0.7, alpha=0.4)
        ax.set_title(f'{country}  —  LCZ Structural Effect\n'
                     '(FE Morphology vs LMDI Composition)',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('CO₂ change (×10³ ton C/cell)', fontsize=10)
        ax.set_xticks(YEAR_TICKS)
        ax.set_xticklabels([str(y) for y in YEAR_TICKS], rotation=30, ha='right')
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(False)

    fig.suptitle('LCZ Structural Effect: FE-based vs LMDI-based',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  Code_FE_Decomposition.py  —  FE 系数贡献量分解")
    print("="*60)

    panels = {
        'CHN': (CHN_PANEL, 'City_ID',  CHN_COEF),
        'USA': (USA_PANEL, 'State_ID', USA_COEF),
    }

    all_rows = []
    for country, (pkl, fe_col, coef_path) in panels.items():
        if not os.path.exists(pkl):
            print(f"  [WARN] 找不到 {pkl}，跳过"); continue

        print(f"\n{'='*60}\n  {country}\n{'='*60}")

        coef_dict = load_fe_coefs(coef_path)
        print(f"  载入系数：β_Pop={coef_dict.get('ln_Pop',0):.4f}, "
              f"β_GDP={coef_dict.get('ln_GDP',0):.4f}, "
              f"LCZ系数数: {sum(1 for k in coef_dict if 'LCZ' in k)}")

        annual, std_lcz = load_annual_panel(pkl, fe_col)
        decomp = compute_fe_decomp(annual, fe_col, std_lcz, coef_dict)
        decomp.insert(0, 'Country', country)
        all_rows.append(decomp)
        del annual; gc.collect()

        out = os.path.join(OUT_DIR, f'Table_FE_Decomp_{country}.csv')
        decomp.to_csv(out, index=False, encoding='utf-8-sig')
        print(f"  Saved: {out}")

    df_all = pd.concat(all_rows, ignore_index=True)
    df_all.to_csv(os.path.join(OUT_DIR,'Table_FE_Decomp_Annual.csv'),
                  index=False, encoding='utf-8-sig')
    print(f"\n  Saved: Table_FE_Decomp_Annual.csv")

    # ── 绘图 ─────────────────────────────────────────────────
    for stem, fig in [
        ('Fig_FE_Decomp_Stacked',       fig_stacked_fe(df_all)),
        ('Fig_FE_Decomp_Cumulative',     fig_cumulative_fe(df_all)),
        ('Fig_FE_Decomp_Ternary',        fig_ternary_fe(df_all)),
        ('Fig_FE_Decomp_Period_Compare', fig_period_fe(df_all)),
    ]:
        _save_fig(fig, os.path.join(OUT_DIR, stem)); plt.close(fig)
        print(f"  Saved: {stem}.png / .svg")

    # ── 新增：4效应分解图（Morphology 拆分为 Built / Natural）─
    print("\n  生成 Built/Natural Morphology 拆分图 ...")
    for stem, fig in [
        ('Fig_FE_Decomp_4Effects_Stacked', fig_stacked_split(df_all)),
        ('Fig_FE_Decomp_4Effects_Period',  fig_period_split(df_all)),
        ('Fig_FE_Decomp_Morph_Decomp',     fig_morph_decomp(df_all)),
    ]:
        _save_fig(fig, os.path.join(OUT_DIR, stem)); plt.close(fig)
        print(f"  Saved: {stem}.png / .svg")

    # ── 新增：Per-pixel 标准化 + 相对贡献率图 ────────────────
    print("\n  生成 per-pixel 和相对贡献率图 ...")

    # 1. Per-pixel 早/晚期对比（ton C / km² / year）
    f_pp = fig_period_pp(df_all)
    _save_fig(f_pp, os.path.join(OUT_DIR, 'Fig_FE_Decomp_PerPixel_Period'))
    plt.close(f_pp)
    print("  Saved: Fig_FE_Decomp_PerPixel_Period.png / .svg")

    # 2. 计算分期相对贡献率
    share_df = compute_period_share(df_all)
    p_share = os.path.join(OUT_DIR, 'Table_FE_Decomp_PeriodShare.csv')
    share_df.to_csv(p_share, index=False, encoding='utf-8-sig')
    print(f"  Saved: {p_share}")
    print(share_df.to_string(index=False))

    # 3. 相对贡献率堆叠柱图
    f_share = fig_share_period(share_df)
    _save_fig(f_share, os.path.join(OUT_DIR, 'Fig_FE_Decomp_PeriodShare'))
    plt.close(f_share)
    print("  Saved: Fig_FE_Decomp_PeriodShare.png / .svg")

    # 4. 热力图
    f_heat = fig_share_heatmap(share_df)
    _save_fig(f_heat, os.path.join(OUT_DIR, 'Fig_FE_Decomp_ShareHeatmap'))
    plt.close(f_heat)
    print("  Saved: Fig_FE_Decomp_ShareHeatmap.png / .svg")

    # LMDI 对比图已移除（LMDI 方法已放弃）

    print("\n" + "="*60)
    print(f"  输出至: {OUT_DIR}")
    print("="*60)
