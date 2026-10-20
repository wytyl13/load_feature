# -*- coding: utf-8 -*-
"""
各特征与全社会用电量相关性统计图 (2026-09-14)
数据源: 神经网络MLP预测电量_训练测试预测.csv  (8特征MLP主口径, 1996-2025实际段)
口径: 双口径并列 —— ①水平值皮尔逊 r  ②同比增量(增速)皮尔逊 r
输出: 特征与电量相关性_20260914/ 下 PNG + CSV + XLSX(表格视图)
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE))
from project_paths import FORECAST_ROOT
OUTDIR = os.path.join(BASE, '特征与电量相关性_20260914')
os.makedirs(OUTDIR, exist_ok=True)

# ---------- 调色板(已过 validate_palette.js, light surface #fcfcfb) ----------
C_LEVEL, C_GROWTH = '#2a78d6', '#eb6834'          # 分类槽1 blue / 槽2 orange
SURFACE, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#898781'
GRID, BASELINE = '#e1e0d9', '#c3c2b7'

# ---------- 数据 ----------
df = pd.read_csv(os.path.join(str(FORECAST_ROOT), '神经网络MLP预测电量_训练测试预测.csv'), encoding='utf-8-sig')
FEATS = ['第一产业增加值', '第二产业增加值', '第三产业增加值', '人口',
         '全体居民人均可支配收入', '城镇居民人均可支配收入', '农村居民人均可支配收入']
col_of = {}
for c in df.columns:
    for f in FEATS:
        if c.startswith(f):
            col_of[f] = c
assert len(col_of) == len(FEATS), col_of

hist = df[df['分段'] != '预测集(未来预测)'].dropna(subset=['实际电量(亿kWh)']).copy()
hist = hist.sort_values('年份')
years = hist['年份'].values.astype(int)
y = hist['实际电量(亿kWh)'].values.astype(float)            # 全社会用电量 (亿kWh)

rows = []
for f in FEATS:
    x = hist[col_of[f]].values.astype(float)
    r_lv = float(np.corrcoef(x, y)[0, 1])                   # 水平值相关
    r_gr = float(np.corrcoef(np.diff(x), np.diff(y))[0, 1])  # 增速相关
    # 稳健性: 剔除1996-1999插补段(2000-2025)
    m2 = years >= 2000
    x2, y2 = x[m2], y[m2]
    r_lv2 = float(np.corrcoef(x2, y2)[0, 1])
    r_gr2 = float(np.corrcoef(np.diff(x2), np.diff(y2))[0, 1])
    rows.append(dict(特征=f, 水平值r=r_lv, 增速r=r_gr,
                     水平值r_2000后=r_lv2, 增速r_2000后=r_gr2))
R = pd.DataFrame(rows)

# 行序 = 增速 r 降序 (区分度最大的口径), 两个面板共用同一行序
order = R.sort_values('增速r', ascending=False).reset_index(drop=True)
print(order.round(4).to_string(index=False))

# ---------- 绘图 ----------
N = len(order)
YLBL = order['特征'].tolist()
V_LEVEL = order['水平值r'].values
V_GROW = order['增速r'].values
YL = np.arange(N)[::-1]                                     # 首行在顶部
XLIM = (0.0, 1.12)                                          # 两面板同轴(见说明)
BARH = 0.50

fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.0), dpi=200, sharey=True)
fig.patch.set_facecolor(SURFACE)

def style(ax, vals, color, title, sub):
    ax.set_facecolor(SURFACE)
    ax.set_xlim(*XLIM)
    ax.set_ylim(-0.75, N - 0.25)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xticklabels([f'{v:.1f}' for v in np.arange(0, 1.01, 0.2)],
                       fontsize=10, color=MUTED)
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(BASELINE)
    ax.tick_params(axis='y', length=0)
    if ax is axes[0]:                                       # sharey: 刻度与标签只在左面板设一次
        ax.set_yticks(YL)                                   # 必须先固定刻度, 否则标签错位
        ax.set_yticklabels(YLBL, fontsize=12, color=INK)
    ax.set_title(title, fontsize=14, color=INK, pad=26, loc='left', fontweight='bold')
    ax.text(0, 1.035, sub, transform=ax.transAxes, fontsize=11, color=INK2, va='bottom')
    for yy, v in zip(YL, vals):
        ax.text(min(v + 0.022, XLIM[1] - 0.005), yy, f'{v:.3f}',
                va='center', ha='left', fontsize=11, color=INK2)

def rounded_bars(ax, color):
    """4px 圆角数据端、锚定 0 基线 (FancyBboxPatch + mutation_aspect 校正为视觉正圆)"""
    fig.canvas.draw()
    bb = ax.get_window_extent()
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    sx, sy = bb.width / (x1 - x0), bb.height / (y1 - y0)     # px / 数据单位
    r_x = 4.0 / sx                                           # 4px 圆角
    for yy, v in zip(YL, (V_LEVEL if color == C_LEVEL else V_GROW)):
        try:
            p = FancyBboxPatch((0, yy - BARH / 2), v, BARH,
                               boxstyle=f'round,pad=0,rounding_size={r_x}',
                               mutation_aspect=sx / sy,
                               facecolor=color, edgecolor='none', zorder=3)
            ax.add_patch(p)
        except Exception:                                    # 退化: 平头矩形
            ax.barh(yy, v, height=BARH, color=color, zorder=3)

style(axes[0], V_LEVEL, C_LEVEL, '① 水平值相关系数 r',
      '各特征值与全社会用电量的相关（原始序列）')
style(axes[1], V_GROW, C_GROWTH, '② 增速相关系数 r',
      '各特征同比增量与全社会用电量同比增量的相关（Δ序列）')
rounded_bars(axes[0], C_LEVEL)
rounded_bars(axes[1], C_GROWTH)

fig.suptitle(f'各特征与全社会用电量的相关性（山东省 {years.min()}—{years.max()}，n={len(hist)}）',
             fontsize=17, color=INK, fontweight='bold', x=0.02, ha='left', y=0.985)
fig.text(0.02, 0.022,
         '口径：皮尔逊相关系数。① 用电量水平值 ~ 特征水平值；② 用电量同比增量 ~ 特征同比增量。\n'
         '注：1996—1999年「人口」「全体居民人均可支配收入」源表缺失、为线性插补值（该段增速恒为0）；'
         '剔除后（2000—2025）首位与末位排序不变（三产最高、人口最低）。\n'
         '数据源：神经网络MLP预测电量_训练测试预测.csv（8特征MLP主口径，实际值段）',
         fontsize=9, color=MUTED, va='bottom', linespacing=1.6)

fig.tight_layout(rect=[0, 0.135, 1, 0.945])
PNG = os.path.join(OUTDIR, '特征与电量相关性.png')
fig.savefig(PNG, dpi=200, facecolor=SURFACE)
plt.close(fig)
print('\n已输出:', PNG)

# ---------- 表格视图 (可访问性兜底 + 汇报备查) ----------
tbl = order.copy()
tbl.columns = ['特征', '水平值r(1996-2025)', '增速r(1996-2025)',
               '水平值r(2000-2025)', '增速r(2000-2025)']
for c in tbl.columns[1:]:
    tbl[c] = tbl[c].round(4)
tbl.to_csv(os.path.join(OUTDIR, '特征与电量相关性.csv'), index=False, encoding='utf-8-sig')
with pd.ExcelWriter(os.path.join(OUTDIR, '特征与电量相关性.xlsx'), engine='openpyxl') as w:
    tbl.to_excel(w, sheet_name='相关性', index=False)
print('已输出表格: 特征与电量相关性.csv / .xlsx')
