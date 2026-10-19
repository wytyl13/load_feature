# -*- coding: utf-8 -*-
"""
ARIMA 滚动预测 2026-2035(10年): 一产/二产/三产增加值、人口
  - 逐年增速(%)
  - 两段总增速: 2026-2030段(基年2025->期末2030) 与 2031-2035段(2030->2035)
  - GDP总量 = 一产+二产+三产(ARIMA 分产业外推后求和)
输出: ARIMA增速预测_2026-2035.xlsx
数据源: ../2026-08-12/111.xlsx
代码阅读要点：本脚本只外推宏观特征及其增速，不直接生成负荷曲线；电量主链由 forecast_arima_ml.py 接收这些特征。
"""
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(BASE, '..', '2026-08-12', '111.xlsx'))
OUT = os.path.join(BASE, 'ARIMA增速预测_2026-2035.xlsx')

# ---------- 1. 读数据 ----------
raw = pd.read_excel(SRC, sheet_name=0, header=0)
years = [int(str(c)[:4]) for c in raw.columns[1:]]
df = raw.iloc[:8, 1:].T.copy()
df.columns = ['第一产业增加值', '第二产业增加值', '第三产业增加值', '人口',
              '电力消费量', '全体居民人均可支配收入', '城镇居民人均可支配收入', '农村居民人均可支配收入']
df.index = years
df = df.sort_index().interpolate(limit_direction='both')
print(f'样本: {df.index.min()}-{df.index.max()} 共{len(df)}年')

# ---------- 2. ARIMA 滚动外推 10 年 ----------
# 每次只预测下一年并把结果写回 hist，保证 2026-2035 的每个值都可追溯。
def determine_d(y, max_d=2):
    yt = np.asarray(y, dtype=float)
    d = 0
    while d < max_d:
        if adfuller(yt)[1] < 0.05:
            return d
        yt = np.diff(yt); d += 1
    return d

def auto_arima(y, p_=range(0, 4), q_=range(0, 4)):
    d = determine_d(y)
    best = None
    for p in p_:
        for q in q_:
            if p == 0 and q == 0:
                continue
            try:
                fit = ARIMA(y, order=(p, d, q)).fit(method_kwargs={'warn_convergence': False})
            except Exception:
                continue
            if best is None or fit.aic < best[0]:
                best = (fit.aic, (p, d, q), fit)
    return best

def safe_forecast(hist):
    """自动定阶预测1步; 若预测值异常(非有限/≤0/>上一年2.5倍)则回退简单模型, 再兜底年增2%"""
    last = float(hist[-1])
    for order in [None, (1, 1, 0), (0, 1, 1), (1, 1, 1)]:
        try:
            if order is None:
                aic, order2, fit = auto_arima(np.asarray(hist, float))
            else:
                order2 = order
                fit = ARIMA(hist, order=order).fit(method_kwargs={'warn_convergence': False})
            p = float(fit.forecast(1)[0])
            if np.isfinite(p) and 0 < p < last * 2.5:
                return p, order2
        except Exception:
            continue
    return last * 1.02, (1, 1, 0)   # 兜底: 年增2%

def rolling_forecast(data, n_out=10):
    hist = list(data)
    out, orders = [], []
    for _ in range(n_out):
        p, order = safe_forecast(hist)
        hist.append(p)
        out.append(p); orders.append(order)
    return out, orders

print('\n===== ARIMA 滚动预测 2026-2035 =====')
pred = {}   # 变量 -> {年份: 值}
for c in ['第一产业增加值', '第二产业增加值', '第三产业增加值', '人口']:
    vals, orders = rolling_forecast(df[c].values, 10)
    pred[c] = dict(zip(range(2026, 2036), vals))
    print(f'  {c}: 2025={df[c].values[-1]:,.1f} -> 2030={vals[4]:,.1f} -> 2035={vals[-1]:,.1f} (阶数{orders[0]})')

yrs = list(range(2026, 2036))
gdp_p = {y: sum(pred[c][y] for c in ['第一产业增加值', '第二产业增加值', '第三产业增加值']) for y in yrs}
pop_p = pred['人口']
# 含期初实际值的完整序列(2025 为基年, 用于增速计算)
gdp_full = {2025: sum(df[c].values[-1] for c in ['第一产业增加值', '第二产业增加值', '第三产业增加值']), **gdp_p}
pop_full = {2025: df['人口'].values[-1], **pop_p}

# ---------- 3. 逐年增速 ----------
def growth(series, base_years):
    """series: {year:value} 含期初; 逐年增速(当年/上年-1)"""
    return {y: series[y] / series[y-1] - 1 for y in base_years}

g_gdp_y = growth(gdp_full, yrs)       # 2026..2035 逐年
g_pop_y = growth(pop_full, yrs)
print('\n===== 逐年增速 =====')
print(f'  {"年份":<6}{"GDP总量":>12}{"GDP增速":>10}{"人口":>12}{"人口增速":>10}')
for y in yrs:
    print(f'  {y:<6}{gdp_p[y]:>12,.0f}{g_gdp_y[y]*100:>9.1f}%{pop_p[y]:>12,.0f}{g_pop_y[y]*100:>9.1f}%')

# ---------- 4. 两段总增速 ----------
# 总增速是期末/期初-1；年均复合增速还要按该段年数开方，二者不能混用。
def seg(series, s, e):
    """s..e 段总增速(基年 s-1 到期末 e)与 CAGR"""
    tot = series[e] / series[s-1] - 1
    cagr = (series[e] / series[s-1]) ** (1 / (e - s + 1)) - 1
    return tot, cagr

segments = [(2026, 2030, '2026-2030段(基年2025)'), (2031, 2035, '2031-2035段(基年2030)')]
pred_full = {c: {2025: df[c].values[-1], **pred[c]} for c in ['第一产业增加值', '第二产业增加值', '第三产业增加值']}
print('\n===== 两段总增速 =====')
rows = []
for s_, e_, name in segments:
    for label, ser in [('GDP总量', gdp_full), ('人口', pop_full)]:
        tot, cagr = seg(ser, s_, e_)
        print(f'  {name} {label}: 总增速 {tot*100:+.1f}%  ({ser[s_-1]:,.0f} -> {ser[e_]:,.0f}), 年均复合 {cagr*100:+.1f}%')
        rows.append({'段': name, '指标': label, '期初值': round(ser[s_-1], 1), '期末值': round(ser[e_], 1),
                     '总增速%': round(tot*100, 1), '年均复合增速%': round(cagr*100, 1)})
# 分产业两段总增速
for s_, e_, name in segments:
    for c in ['第一产业增加值', '第二产业增加值', '第三产业增加值']:
        tot, cagr = seg(pred_full[c], s_, e_)
        print(f'  {name} {c}: 总增速 {tot*100:+.1f}% (年均 {cagr*100:+.1f}%)')
        rows.append({'段': name, '指标': c, '期初值': round(pred_full[c][s_-1], 1), '期末值': round(pred_full[c][e_], 1),
                     '总增速%': round(tot*100, 1), '年均复合增速%': round(cagr*100, 1)})

# ---------- 5. 输出 xlsx ----------
with pd.ExcelWriter(OUT, engine='openpyxl') as w:
    pd.DataFrame({
        '年份': yrs,
        '第一产业增加值(亿元)': [round(pred['第一产业增加值'][y], 1) for y in yrs],
        '第二产业增加值(亿元)': [round(pred['第二产业增加值'][y], 1) for y in yrs],
        '第三产业增加值(亿元)': [round(pred['第三产业增加值'][y], 1) for y in yrs],
        'GDP总量(亿元)': [round(gdp_p[y], 1) for y in yrs],
        'GDP增速(%)': [round(g_gdp_y[y]*100, 1) for y in yrs],
        '人口(万人)': [round(pop_p[y], 1) for y in yrs],
        '人口增速(%)': [round(g_pop_y[y]*100, 1) for y in yrs],
    }).to_excel(w, sheet_name='逐年预测与增速', index=False)
    pd.DataFrame(rows).to_excel(w, sheet_name='两段总增速', index=False)
    # 增速汇总(横排: 指标 x 两段)
    seg30, seg35 = {r['指标']: r for r in rows if '2026-2030' in r['段']}, {r['指标']: r for r in rows if '2031-2035' in r['段']}
    order = ['GDP总量', '人口', '第一产业增加值', '第二产业增加值', '第三产业增加值']
    pd.DataFrame({
        '指标': order,
        '2026-2030段总增速(%)': [seg30[k]['总增速%'] for k in order],
        '2026-2030段年均复合(%)': [seg30[k]['年均复合增速%'] for k in order],
        '2031-2035段总增速(%)': [seg35[k]['总增速%'] for k in order],
        '2031-2035段年均复合(%)': [seg35[k]['年均复合增速%'] for k in order],
    }).to_excel(w, sheet_name='增速汇总', index=False)
print(f'\n已输出: {OUT}')
