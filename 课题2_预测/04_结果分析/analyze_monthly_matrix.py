# -*- coding: utf-8 -*-
"""分析 分月8760矩阵 某年某月: 三条口径曲线特征(峰时刻/峰谷差/负荷率/形状差异)"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import itertools
import build_monthly_8760_matrix as m

YEAR, MO = 2020, 1
curves = m.curves_for(YEAR)

def daily_stats(ts, v):
    s = pd.Series(v, ts)
    grp = s.groupby(s.index.date)
    dmax = grp.max()
    dmin = grp.min()
    dmean = grp.mean()
    dmax_hour = s.groupby(s.index.date).idxmax().dt.hour
    pv = ((dmax - dmin) / dmax).mean() * 100
    lf = (dmean / dmax).mean() * 100
    wk = pd.Series(dmax.index).apply(lambda d: pd.Timestamp(d).dayofweek >= 5).values
    return dmax, dmin, dmean, dmax_hour, pv, lf, wk

print('=' * 72)
print(f'{YEAR}年{MO}月 三条口径曲线特征分析')
print('=' * 72)
info = {}
for (ts, vals), label, color, ls in curves:
    sel = ts.month == MO
    v = vals[sel]; t = ts[sel]
    dmax, dmin, dmean, dmax_hour, pv, lf, wk = daily_stats(t, v)
    eve = ((dmax_hour >= 17) & (dmax_hour <= 21)).mean() * 100
    morn = ((dmax_hour >= 8) & (dmax_hour <= 11)).mean() * 100
    peak_t = s_max = v.max()
    pt = t[np.argmax(v)]
    info[label.split(' ')[0] if '(' not in label else label] = dict(v=v, dmax=dmax, dmax_hour=dmax_hour)
    print(f'--- {label} ({color} {ls}) ---')
    print(f'  月最大 {v.max():,.0f} MW @ {pt:%m-%d %H:00}; 月最小 {v.min():,.0f} MW')
    print(f'  日峰时刻分布: 晚峰17-21点占{eve:.0f}% | 早峰8-11点占{morn:.0f}% | 峰出现在下午15-17点以外: {100-eve-morn:.0f}%')
    print(f'  平均日峰谷差率 {pv:.1f}% | 平均日负荷率 {lf:.1f}%')
    print(f'  周末日最大均值 {dmax.values[wk].mean():,.0f} MW vs 工作日 {dmax.values[~wk].mean():,.0f} MW (差 {(dmax.values[wk].mean()/dmax.values[~wk].mean()-1)*100:+.1f}%)')

print('=' * 72)
print('标幺后逐点相关性 (按 各自月最大 归一化)')
names = [label for _, label, _, _ in curves]
for (n1, a), (n2, b) in itertools.combinations(list(zip(names, [info[n] for n in names])), 2):
    u1 = a['v'] / a['v'].max(); u2 = b['v'] / b['v'].max()
    corr = np.corrcoef(u1, u2)[0, 1]
    mad = np.abs(u1 - u2).mean()
    print(f'  {n1} vs {n2}: 相关系数={corr:.4f}, 平均绝对差={mad:.4f}')
