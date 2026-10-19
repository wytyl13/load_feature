# -*- coding: utf-8 -*-
"""
ARIMA 滚动预测: 山东 GDP 总量(第一+第二+第三产业增加值)与人口 2026-2030
电量推算(主): 近10年(2016-2025) 人均电量~GDP 双对数回归, E_2030 = 人均电量×人口(ARIMA)
对照1: 全样本(2000-2025) ln(电量)~ln(GDP)+ln(人口) 双变量
对照2: 电力弹性法(2022-2025 弹性)
输出: ARIMA预测2026-2030.xlsx (供 build_new_8760.py 2030 锚点)
数据源: ../2026-08-12/111.xlsx (1996-2025 山东 三产增加值/人口/电力消费量)
滚动方式: 每步用截至当前年(含已预测年)序列, ADF定差分阶d, 固定d后用AIC网格选p,q
代码阅读要点：读取宏观序列后，先 ARIMA 外推 GDP 和人口，再用近十年双对数回归推算主电量；其余模型仅作对照。
"""
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(BASE, '..', '2026-08-12', '111.xlsx'))
OUT = os.path.join(BASE, 'ARIMA预测2026-2030.xlsx')

# ---------- 1. 读 111.xlsx ----------
df = pd.read_excel(SRC, sheet_name='Sheet1', header=None)
labels = df[0].tolist()
years = df.iloc[0, 1:].astype(str).str.replace('年', '').astype(int).tolist()  # 2025..1996
v = {lbl: df.iloc[i, 1:].astype(float).tolist() for i, lbl in enumerate(labels) if i >= 1}

def series_of(label):
    return {y: x for y, x in zip(years, v[label]) if not np.isnan(x)}

gdp1, gdp2, gdp3 = (series_of(x) for x in ['第一产业增加值(亿元)', '第二产业增加值(亿元)', '第三产业增加值(亿元)'])
gdp = {y: gdp1[y] + gdp2[y] + gdp3[y] for y in sorted(gdp1)}
pop = series_of('人口(万人)')            # 2000-2025
elec = series_of('电力消费量 (亿千瓦小时)')   # 1996-2025 全社会用电量
print(f'样本: GDP {min(gdp)}-{max(gdp)}({len(gdp)}点) 人口 {min(pop)}-{max(pop)}({len(pop)}点) 电量 {min(elec)}-{max(elec)}({len(elec)}点)')

# ---------- 2. ARIMA 滚动预测(ADF 定 d, 固定 d 后 AIC 选 p,q; 不同 d 的 AIC 不可比) ----------
# 每次只预测下一年并写回历史序列，便于逐年追踪，但会累积前期预测误差。
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

def rolling_forecast(data, n_out, label):
    hist = dict(sorted(data.items()))
    out = []
    for _ in range(n_out):
        y_arr = np.asarray(list(hist.values()), dtype=float)
        aic, order, fit = auto_arima(y_arr)
        out.append((max(hist) + 1, float(fit.forecast(1)[0]), order, round(aic, 1)))
        hist[max(hist) + 1] = out[-1][1]
    return out

print('\n===== ARIMA 滚动预测 2026-2030 =====')
gdp_pred = rolling_forecast(gdp, 5, 'GDP')
prev = None
for t, p, o, a in gdp_pred:
    note = f', 年增{(p/prev-1)*100:+.1f}%' if prev else ''
    print(f'  GDP {t}年: {p:,.1f} 亿元 (ARIMA{o}, AIC={a}{note})'); prev = p
pop_pred = rolling_forecast(pop, 5, '人口')
prev = None
for t, p, o, a in pop_pred:
    note = f', 年增{(p/prev-1)*100:+.1f}%' if prev else ''
    print(f'  人口 {t}年: {p:,.1f} 万人 (ARIMA{o}, AIC={a}{note})'); prev = p

g2030 = {t: g for t, g, _, _ in gdp_pred}[2030]
p2030 = {t: p for t, p, _, _ in pop_pred}[2030]
g_series = {**gdp, **{t: g for t, g, _, _ in gdp_pred}}
p_series = {**pop, **{t: p for t, p, _, _ in pop_pred}}

# ---------- 3. 主模型: 近10年人均电量~GDP 双对数回归 ----------
# 斜率 eW 是人均电量对 GDP 的弹性，人口规模由单独的 ARIMA 序列负责。
YW = list(range(2016, 2026))
lnEP = np.log([elec[y] / pop[y] for y in YW])
lnG = np.log([gdp[y] for y in YW])
Xw = sm.add_constant(lnG)
rW = sm.OLS(lnEP, Xw).fit()
eW = float(rW.params[1])
print('\n===== 主模型: ln(人均电量kWh) = a + b·ln(GDP), 2016-2025 =====')
print(f'  a={rW.params[0]:.3f}, b={eW:.3f}, R²={rW.rsquared:.4f}, p值={rW.pvalues[1]:.4f}')
ep25 = elec[2025] / pop[2025]
EP = {y: float(np.exp(rW.predict([1, np.log(g)]))[0]) for y, g in g_series.items()}
E_main = {y: EP[y] * p_series[y] for y in range(2026, 2031)}
e30 = E_main[2030]
print(f'  2030: 人均电量={EP[2030]*10000:,.0f} kWh/人 × 人口{p2030:,.0f}万 = 电量 {e30:,.1f} 亿kWh')

# ---------- 4. 对照1: 全样本 ln(电量)~ln(GDP)+ln(人口) ----------
ys = sorted(set(gdp) & set(pop) & set(elec))
lnE = np.log([elec[y] for y in ys]); lnP = np.log([pop[y] for y in ys])
Xf = sm.add_constant(np.column_stack([np.log([gdp[y] for y in ys]), lnP]))
rF = sm.OLS(lnE, Xf).fit()
E_alt1 = float(np.exp(rF.predict([1, np.log(g2030), np.log(p2030)]))[0])
print(f'\n===== 对照1: 全样本({ys[0]}-{ys[-1]}, n={len(ys)}) ln(电量)~ln(GDP)+ln(人口) =====')
print(f'  b={rF.params[1]:.3f}(p={rF.pvalues[1]:.3f}), c={rF.params[2]:.3f}(p={rF.pvalues[2]:.3f}), R²={rF.rsquared:.4f}')
print(f'  2030 电量 = {E_alt1:,.1f} 亿kWh (人口项不显著且方向不合理, 故仅作对照)')

# ---------- 5. 对照2: 弹性法(2022-2025) ----------
gy = np.array([gdp[y] for y in range(2022, 2026)]); ey = np.array([elec[y] for y in range(2022, 2026)])
e_ela = np.diff(np.log(ey)).sum() / np.diff(np.log(gy)).sum()   # 平均弹性
g_growth = (g2030 / gdp[2025]) ** (1 / 5) - 1
E_alt2 = elec[2025] * (1 + e_ela * g_growth) ** 5
print(f'\n===== 对照2: 弹性法 =====')
print(f'  2022-2025平均弹性 {e_ela:.3f}, ARIMA隐含GDP年增 {g_growth*100:.1f}% -> 2030电量 = {E_alt2:,.1f} 亿kWh')

# ---------- 6. 输出 xlsx ----------
# 同时输出主模型、对照模型和说明表，供后续 8760 脚本读取或人工复核。
with pd.ExcelWriter(OUT, engine='openpyxl') as w:
    pd.DataFrame({
        '年份': [t for t, _, _, _ in gdp_pred],
        'GDP总量(亿元)': [g for _, g, _, _ in gdp_pred],
        'GDP_ARIMA(p,d,q)': [str(o) for _, _, o, _ in gdp_pred],
        '人口(万人)': [p for _, p, _, _ in pop_pred],
        '人口_ARIMA(p,d,q)': [str(o) for _, _, o, _ in pop_pred],
        '人均电量(kWh/人)': [round(EP[t] * 10000, 1) for t, *_ in gdp_pred],
        '全社会用电量(亿kWh,主)': [round(E_main[t], 1) for t, *_ in gdp_pred],
        '对照1_全样本回归(亿kWh)': [round(np.exp(rF.predict([1, np.log(g_series[t]), np.log(p_series[t])]))[0], 1) for t, *_ in gdp_pred],
        '对照2_弹性法(亿kWh)': [round(elec[2025] * (1 + e_ela * g_growth) ** (t - 2025), 1) for t, *_ in gdp_pred],
    }).to_excel(w, sheet_name='ARIMA滚动预测', index=False)
    pd.DataFrame({
        '项目': ['样本区间', '主模型窗口', '主模型R²', '主模型人均电量弹性b', '2030GDP(ARIMA)', '2030人口(ARIMA)',
                '2030电量_主', '2030电量_对照1', '2030电量_对照2', 'NN预测2030电量'],
        '值': [f'1996-2025(电量)/2000-2025(人口)', f'2016-2025', round(rW.rsquared, 4), round(eW, 3),
               round(g2030, 1), round(p2030, 1), round(e30, 1), round(E_alt1, 1), round(E_alt2, 1), 11023.6],
    }).to_excel(w, sheet_name='模型说明', index=False)
print(f'\n已输出: {OUT}')
print(f'2030 电量锚点(主方法, 供build_new_8760.py): {e30:,.1f} 亿kWh')
