# -*- coding: utf-8 -*-
"""
各特征与用电量相关性占比 —— 饼图精简版 (2026-09-14)
口径: 增速口径相关系数 r 的平方(r²)归一化占比
      用增速口径的原因: 水平值口径7个特征 r 全在0.96~0.997, r²占比≈7等份, 饼图无意义
配色: 单色 (分类槽1 blue #2a78d6) —— 7个特征是无序类别, 扇区大小已承载全部信息
输出: 特征与电量相关性_20260914/特征与电量相关性_饼图.png
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE, '特征与电量相关性_20260914')
os.makedirs(OUTDIR, exist_ok=True)

C_SLICE, SURFACE, INK, MUTED = '#2a78d6', '#fcfcfb', '#0b0b0b', '#898781'

df = pd.read_csv(os.path.join(BASE, '神经网络MLP预测电量_训练测试预测.csv'), encoding='utf-8-sig')
FEATS = ['第一产业增加值', '第二产业增加值', '第三产业增加值', '人口',
         '全体居民人均可支配收入', '城镇居民人均可支配收入', '农村居民人均可支配收入']
col_of = {f: c for c in df.columns for f in FEATS if c.startswith(f)}
hist = df[df['分段'] != '预测集(未来预测)'].dropna(subset=['实际电量(亿kWh)']).sort_values('年份')
y = hist['实际电量(亿kWh)'].values.astype(float)

rec = []
for f in FEATS:
    x = hist[col_of[f]].values.astype(float)
    r = float(np.corrcoef(np.diff(x), np.diff(y))[0, 1])     # 增速口径
    rec.append(dict(特征=f, r=r, r2=r ** 2))
P = pd.DataFrame(rec)
P['占比%'] = P['r2'] / P['r2'].sum() * 100
P = P.sort_values('占比%', ascending=False).reset_index(drop=True)
print(P.round(4).to_string(index=False))

# ---------- 绘图 ----------
# 单色: 7个特征是无序类别, 扇区大小已承载全部信息, 颜色不再重复编码
# 彩色: 饼图为"相邻对"形式(每扇区只挨左右两个), 分类槽1-7过相邻对校验
CAT7 = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7']

def draw(colors, fname, note_extra=''):
    fig, ax = plt.subplots(figsize=(11.5, 7.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    vals = P['占比%'].values
    labels = [f'{n}\n{p:.1f}%' for n, p in zip(P['特征'], vals)]
    ax.pie(vals, labels=labels, startangle=90, counterclock=False, colors=colors,
           wedgeprops=dict(edgecolor=SURFACE, linewidth=2.0),      # 2px 表面间隙
           labeldistance=1.07, textprops=dict(fontsize=13, color=INK, linespacing=1.5))
    ax.set_title('各特征与用电量的相关性占比', fontsize=20, color=INK,
                 fontweight='bold', pad=30)
    ax.text(0.5, 0.965, f'山东省 1996—2025（增速口径，n={len(hist)}）',
            transform=ax.transAxes, ha='center', fontsize=13, color=MUTED)
    fig.text(0.5, 0.045,
             '占比 = 各特征与用电量增速的相关系数平方，在 7 个特征间的相对份额；'
             '相关系数平方反映该特征对用电量增速波动的解释力。\n'
             '数据源：神经网络MLP预测电量_训练测试预测.csv（8特征MLP主口径，实际值段）'
             + note_extra,
             ha='center', fontsize=9.5, color=MUTED, linespacing=1.7)
    fig.tight_layout(rect=[0, 0.09, 1, 0.90])
    p = os.path.join(OUTDIR, fname)
    fig.savefig(p, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print('已输出:', p)

draw([C_SLICE] * len(P), '特征与电量相关性_饼图_单色.png')
draw(CAT7, '特征与电量相关性_饼图_彩色.png')

P[['特征', 'r', 'r2', '占比%']].round(4).to_csv(
    os.path.join(OUTDIR, '特征与电量相关性_饼图.csv'), index=False, encoding='utf-8-sig')
