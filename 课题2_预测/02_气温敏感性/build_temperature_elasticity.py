# -*- coding: utf-8 -*-
"""
气温弹性估计。副本只保留原始负荷预测行业范围，不包含后续扩展行业。
气温弹性估计 (2026-08-25; 2026-09-03 标幺化+剔除增长; 2026-09-04 基温改数据拐点): CDD/HDD 度日法
数据: 山东16市等权ERA5(Open-Meteo)日均温(2020-2023, 全程室外温度) × 2022/2023实测10类行业日电量(2026-08-12/)
方法: 夏(6-9月) 日电量(÷同年同季均值标幺) ~ β0 + β1·CDD + 周末/节假日
      冬(11-2月) ~ β0 + β2·HDD18 + 周末/节假日
      增长被标幺分母吸收(2026-09-04领导拍板, 不再加2023年虚拟变量, 年FE版作对照)
基温(2026-09-05甲方拍板重做): 弃国标18/26与R²扫描固定阈值, 改全年温度-日电量V形拐点判定
      (build_base_kink.py: 2022/2023全年逐日对比电量升降, 剔除过年+节假日, 采用值=两年拐点算术平均):
      夏22.0°C / 冬12.75°C(总负荷口径), 运行时读 气温基温判定.json
另附基温扫描(夏20~32℃/冬5~20℃, 标幺口径R²)备查
输出: 气温弹性_2022_2023.csv + 气温基温扫描_2022_2023.csv + JSON(供2030气温情景)
"""
import os
import json
import numpy as np
import pandas as pd

# BASE 是脚本所在目录；原工程将温度 CSV 和结果放在此处，将历史 8760 放在相邻数据根目录。
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.normpath(os.path.join(BASE, '..', '2026-08-12'))
TEMP_CSV = os.path.join(BASE, 'temperature_daily_shandong.csv')

INDUSTRY_FILES = ['互联网和相关服务_MW.csv', '化学原料和化学制品制造业_MW.csv',
                  '总负荷去掉九个行业负荷.csv', '有色金属冶炼和压延加工业_MW.csv', '汽车制造业_MW.csv',
                  '电气机械和器材制造业_MW.csv', '非金属矿物制品业_MW.csv', '黑色金属冶炼和压延加工业_MW.csv']
SHORT = {'互联网和相关服务_MW.csv': '互联网',
         '化学原料和化学制品制造业_MW.csv': '化学', '总负荷去掉九个行业负荷.csv': '扣除(居民商业等)',
         '有色金属冶炼和压延加工业_MW.csv': '有色', '汽车制造业_MW.csv': '汽车',
         '电气机械和器材制造业_MW.csv': '电气机械', '非金属矿物制品业_MW.csv': '非金属',
         '黑色金属冶炼和压延加工业_MW.csv': '黑色'}

# 2022/2023 法定节假日(off=放假, 与 build_new_8760.py HOLIDAYS 一致)
HOLIDAYS = {
    2022: ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-31', '2022-02-01', '2022-02-02', '2022-02-03',
           '2022-02-04', '2022-02-05', '2022-02-06', '2022-04-03', '2022-04-04', '2022-04-05', '2022-04-30',
           '2022-05-01', '2022-05-02', '2022-05-03', '2022-05-04', '2022-06-03', '2022-06-04', '2022-06-05',
           '2022-09-10', '2022-09-11', '2022-09-12', '2022-10-01', '2022-10-02', '2022-10-03', '2022-10-04',
           '2022-10-05', '2022-10-06', '2022-10-07'],
    2023: ['2023-01-01', '2023-01-02', '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24', '2023-01-25',
           '2023-01-26', '2023-01-27', '2023-04-05', '2023-04-29', '2023-04-30', '2023-05-01', '2023-05-02',
           '2023-05-03', '2023-06-22', '2023-06-23', '2023-06-24', '2023-09-29', '2023-09-30', '2023-10-01',
           '2023-10-02', '2023-10-03', '2023-10-04', '2023-10-05', '2023-10-06']}


def readc(f):
    try:
        return pd.read_csv(f, encoding='utf-8-sig')
    except UnicodeDecodeError:
        return pd.read_csv(f, encoding='gbk')


def daily_energy(series_8760, dates):
    """8760 有名值(MW) -> 逐日电量(万kWh)"""
    # MW 是小时平均功率，逐小时求和得到 MWh，再乘 0.1 转成万 kWh。
    s = pd.Series(series_8760, index=dates)
    return s.resample('D').sum() * 0.1


def ols_lstsq(X, y):
    X = np.asarray(X, float); y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    n, p = X.shape
    resid = y - yhat
    dof = max(n - p, 1)
    sigma2 = float(resid @ resid / dof)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tstat = beta / se
    pval = 2 * (1 - _tdist_cdf(abs(tstat), dof))
    return beta, r2, pval, n


def _tdist_cdf(t, df):
    """Student-t CDF (不完全beta函数近似, 够用)"""
    x = df / (df + t * t)
    from scipy.special import betainc  # noqa: lazy
    return 0.5 * (1 + np.sign(t) * (1 - betainc(df / 2, 0.5, x)))


# ---------- 读数据 ----------
# 基温(2026-09-05 甲方拍板重做): 不再用国标18/26或R²扫描固定阈值, 改为 2022/2023 全年温度-日电量
# V形拐点判定(build_base_kink.py, 剔除过年+节假日, 基温取两年拐点算术平均值):
#   总负荷 夏拐点 22.0°C / 冬拐点 12.75°C (2022: 22.5/12.0, 2023: 21.5/13.5, 合并: 22.0/12.5)
_kink_json = os.path.join(BASE, '气温基温判定.json')
if os.path.exists(_kink_json):
    with open(_kink_json, encoding='utf-8') as f:
        _k = json.load(f)['adopted']
    BASE_SUMMER, BASE_WINTER = float(_k['base_summer']), float(_k['base_winter'])
    print(f'基温(读拐点判定JSON): 夏 {BASE_SUMMER}°C / 冬 {BASE_WINTER}°C')
else:
    BASE_SUMMER, BASE_WINTER = 23.0, 18.0   # 兜底: 拐点判定未跑时的旧值(不再推荐)
    print(f'基温(兜底旧值): 夏 {BASE_SUMMER}°C / 冬 {BASE_WINTER}°C')
temp = pd.read_csv(TEMP_CSV, encoding='utf-8-sig')
temp['date'] = pd.to_datetime(temp['date'])
CDD = np.maximum(0.0, temp['temp_mean'] - BASE_SUMMER)
HDD = np.maximum(0.0, BASE_WINTER - temp['temp_mean'])
temp['CDD'] = CDD; temp['HDD18'] = HDD

# 2022/2023 日期序列
dates = {}
series = {}
for year in (2022, 2023):
    idx = pd.date_range(f'{year}-01-01', f'{year}-12-31 23:00', freq='H')
    dates[year] = idx
    series[year] = {}
    for fn in INDUSTRY_FILES:
        p = os.path.join(DATA_ROOT, str(year), fn)
        v = readc(p)['值'].to_numpy(float)
        assert len(v) == 8760, f'{year} {fn} 长度 {len(v)}'
        series[year][fn] = daily_energy(v, idx)

# 合并两年日表
# 先按天建模，再按同年/同季均值标幺，避免把行业年增长直接误判为气温效应。
rows = []
for year in (2022, 2023):
    d = pd.DataFrame({'date': dates[year].normalize().unique()})
    d = d.merge(temp[['date', 'temp_mean', 'CDD', 'HDD18']], on='date', how='left')
    for fn in INDUSTRY_FILES:
        d[fn] = series[year][fn].values
    d['holiday'] = d['date'].dt.strftime('%Y-%m-%d').isin(set(HOLIDAYS[year]))
    d['weekend'] = d['date'].dt.dayofweek >= 5
    d['weekend_hol'] = (d['weekend'] | d['holiday']).astype(int)
    rows.append(d)
daily = pd.concat(rows, ignore_index=True)

assert daily['temp_mean'].notna().all(), '气温缺值'

# ---------- 回归: 夏(6-9月) 冬(11-2月, 跨年) ----------
# 2026-09-03 甲方要求剔除年际自然增长: 加入2023年虚拟变量(年固定效应), 2023相对2022的
# 整体增长被吸收, β 只由同年内温度逐日变化识别; 另报 年+月固定效应稳健版对照
SUMMER_MONTHS = [6, 7, 8, 9]
WINTER_MONTHS = [11, 12, 1, 2]


def fit_season(s, y_col, deg_col, year_fe=True, month_fe=False, perunit=False):
    """日电量 ~ 常数 + 度日 + 周末节假日 [+ 2023年虚拟 [+ 月固定效应]]; 返回度日项结果
    perunit=True: 标幺化口径(2026-09-04领导拍板) — y先除以同年同季平均日电量再回归,
    负荷增长被分母吸收, β 直接为 %/度日; 水平已标幺, 年虚拟变量不再需要(作对照另报)"""
    if perunit:
        yr = s['date'].dt.year
        y = (s[y_col] / s.groupby(yr)[y_col].transform('mean')).to_numpy(float)
    else:
        y = s[y_col].to_numpy(float)
    cols = [np.ones(len(s)), s[deg_col].to_numpy(float), s['weekend_hol'].to_numpy(float)]
    if year_fe:
        cols.append((s['date'].dt.year == 2023).to_numpy(float))
    if month_fe:
        for m in sorted(s['date'].dt.month.unique())[:-1]:
            cols.append((s['date'].dt.month == m).to_numpy(float))
    b, r2, p, n = ols_lstsq(np.column_stack(cols), y)
    beta = float(b[1])
    eps = beta * 100.0 if perunit else float(b[1] / s[y_col].mean() * 100.0)
    return dict(beta=beta, eps=eps, r2=r2, p=float(p[1]), n=n)


def reg_all(y_col):
    s = daily[daily['date'].dt.month.isin(SUMMER_MONTHS)]
    w = daily[daily['date'].dt.month.isin(WINTER_MONTHS)]
    return (fit_season(s, y_col, 'CDD', year_fe=False), fit_season(s, y_col, 'CDD', year_fe=True),
            fit_season(s, y_col, 'CDD', year_fe=True, month_fe=True),
            fit_season(w, y_col, 'HDD18', year_fe=False), fit_season(w, y_col, 'HDD18', year_fe=True),
            fit_season(w, y_col, 'HDD18', year_fe=True, month_fe=True),
            fit_season(s, y_col, 'CDD', perunit=True), fit_season(s, y_col, 'CDD', year_fe=True, perunit=True),
            fit_season(w, y_col, 'HDD18', perunit=True), fit_season(w, y_col, 'HDD18', year_fe=True, perunit=True))


# 总负荷 = 10类之和
for fn in INDUSTRY_FILES:
    daily['总负荷'] = daily.get('总负荷', 0) + daily[fn]

ALL_ROWS = [(fn, SHORT[fn]) for fn in INDUSTRY_FILES] + [('总负荷', '总负荷(10类之和)')]
result = []
for fn, name in ALL_ROWS:
    o_s, n_s, m_s, o_w, n_w, m_w, pu_s, pu_s2, pu_w, pu_w2 = reg_all(fn)
    # 标幺估计 → 绝对增量换算: β万 = 标幺斜率(%/度日)/100 × 该行业2022-23季均日电量(由旧绝对口径反推)
    M_s = n_s['beta'] / n_s['eps'] * 100.0 if n_s['eps'] else 0.0
    M_w = n_w['beta'] / n_w['eps'] * 100.0 if n_w['eps'] else 0.0
    result.append(dict(file=fn, 行业=name,
                       beta_wan_夏=round(pu_s['eps'] / 100.0 * M_s, 1),
                       beta_wan_冬=round(pu_w['eps'] / 100.0 * M_w, 1),
                       夏弹性_epCDD=round(n_s['eps'], 3), 冬弹性_epHDD=round(n_w['eps'], 3),
                       beta_夏=round(n_s['beta'], 1), p_夏=n_s['p'], R2_夏=round(n_s['r2'], 3),
                       beta_冬=round(n_w['beta'], 1), p_冬=n_w['p'], R2_冬=round(n_w['r2'], 3),
                       n_夏=n_s['n'], n_冬=n_w['n'],
                       夏弹性_未剔除增长=round(o_s['eps'], 3), 冬弹性_未剔除增长=round(o_w['eps'], 3),
                       beta_夏_未剔除增长=round(o_s['beta'], 1), beta_冬_未剔除增长=round(o_w['beta'], 1),
                       夏弹性_月FE稳健=round(m_s['eps'], 3), 冬弹性_月FE稳健=round(m_w['eps'], 3),
                       p_夏_月FE=m_s['p'], p_冬_月FE=m_w['p'],
                       # 标幺化口径(2026-09-04领导拍板, 主口径): 先除同年同季均值再回归, β即%/度日
                       夏弹性_标幺=round(pu_s['eps'], 3), 冬弹性_标幺=round(pu_w['eps'], 3),
                       p_夏_标幺=pu_s['p'], p_冬_标幺=pu_w['p'],
                       R2_夏_标幺=round(pu_s['r2'], 3), R2_冬_标幺=round(pu_w['r2'], 3),
                       夏弹性_标幺_年FE对照=round(pu_s2['eps'], 3), 冬弹性_标幺_年FE对照=round(pu_w2['eps'], 3)))
    print(f'{name:>12s}: 夏标幺 {pu_s["eps"]:+.3f}%/°C (p={pu_s["p"]:.2e}, R²={pu_s["r2"]:.3f}, 年FE对照 {pu_s2["eps"]:+.3f})'
          f' | 旧绝对口径 {n_s["eps"]:+.3f}%/°C (β={n_s["beta"]:+.1f}万kWh, p={n_s["p"]:.2e})')
    print(f'{"":12s}  冬标幺 {pu_w["eps"]:+.3f}%/°C (p={pu_w["p"]:.2e}, R²={pu_w["r2"]:.3f}, 年FE对照 {pu_w2["eps"]:+.3f})'
          f' | 旧绝对口径 {n_w["eps"]:+.3f}%/°C (β={n_w["beta"]:+.1f}万kWh, p={n_w["p"]:.2e})')

# ---------- 基温扫描(数据找最优基温, 2026-09-04领导要求): 夏20~32℃ / 冬5~20℃, 标幺化口径 R² ----------
scan_rows = []
for fn, name in ALL_ROWS:
    for kind, months, bases in [('CDD', SUMMER_MONTHS, range(20, 33)), ('HDD', WINTER_MONTHS, range(5, 21))]:
        s = daily[daily['date'].dt.month.isin(months)]
        for tb in bases:
            dd = np.maximum(0.0, s['temp_mean'] - tb) if kind == 'CDD' else np.maximum(0.0, tb - s['temp_mean'])
            if dd.max() <= 0 or int((dd > 0).sum()) < 30:   # 基温过高/过低无辨识度(如CDD32夏季全零), 跳过
                scan_rows.append(dict(行业=name, 类型=kind, 基温C=tb, R2=np.nan, beta=np.nan, p=np.nan))
                continue
            r = fit_season(s.assign(dd_t=dd), fn, 'dd_t', year_fe=False, perunit=True)
            scan_rows.append(dict(行业=name, 类型=kind, 基温C=tb, R2=round(r['r2'], 4),
                                  beta=round(r['beta'], 4), p=round(r['p'], 4)))   # beta=%/度日(标幺口径)
scan_df = pd.DataFrame(scan_rows)
scan_out = os.path.join(BASE, '气温基温扫描_2022_2023.csv')
scan_df.to_csv(scan_out, index=False, encoding='utf-8-sig')
print('\n基温扫描(总负荷, R²前3):')
for kind in ('CDD', 'HDD'):
    d = scan_df[(scan_df['行业'] == '总负荷(10类之和)') & (scan_df['类型'] == kind)].nlargest(3, 'R2')
    print('  ' + kind + ': ' + '  '.join(f"{int(r.基温C)}℃(R²={r.R2:.4f})" for r in d.itertuples()))
print('已输出:', scan_out)

df = pd.DataFrame(result)
out = os.path.join(BASE, '气温弹性_2022_2023.csv')
df.to_csv(out, index=False, encoding='utf-8-sig')
print('\n已输出:', out)

# ---------- 应用弹性: 显著筛选(p<0.05) + 物理修正(限电/趋势污染→0) ----------
# 规则: 夏/冬弹性仅当回归显著(p<0.05, 年固定效应口径)且无物理干扰时应用; 否则取 0
#   有色: 2022-08有序用电干扰(方向是限电所致) → 0
#   互联网: 冬弹性显著为负与物理方向相悖(该行业无采暖响应,
#     负号源于月度构成混杂), 保留旧覆盖冬→0; 夏按p值自动判定
APPLY_OVERRIDE = {
    '有色金属冶炼和压延加工业_MW.csv': (0.0, 0.0),
    '互联网和相关服务_MW.csv': (None, 0.0),
}
apply_result = []
for r in result:
    s = r['夏弹性_epCDD'] if r['p_夏'] < 0.05 else 0.0
    w = r['冬弹性_epHDD'] if r['p_冬'] < 0.05 else 0.0
    # 标幺化口径(主口径, 2026-09-04领导拍板): 同一p值规则+物理修正, β单位=%/度日
    sp = r['夏弹性_标幺'] if r['p_夏_标幺'] < 0.05 else 0.0
    wp = r['冬弹性_标幺'] if r['p_冬_标幺'] < 0.05 else 0.0
    # 万kWh/°C口径(2026-09-04用户要求显示/施加单位): 标幺估计×2022-23季均电量换算
    sw = r['beta_wan_夏'] if r['p_夏_标幺'] < 0.05 else 0.0
    ww = r['beta_wan_冬'] if r['p_冬_标幺'] < 0.05 else 0.0
    if r['file'] in APPLY_OVERRIDE:
        os_, ow_ = APPLY_OVERRIDE[r['file']]
        # None=不覆盖交p值规则; 显式数值=强制覆盖(0=置0)
        if os_ is not None:
            s = os_
            sp = 0.0 if os_ == 0.0 else sp
            sw = 0.0 if os_ == 0.0 else sw
        if ow_ is not None:
            w = ow_
            wp = 0.0 if ow_ == 0.0 else wp
            ww = 0.0 if ow_ == 0.0 else ww
    apply_result.append(dict(file=r['file'], 夏弹性应用=round(s, 4), 冬弹性应用=round(w, 4),
                             夏标幺应用=round(sp, 4), 冬标幺应用=round(wp, 4),
                             夏β万应用=round(sw, 1), 冬β万应用=round(ww, 1)))

# 存 JSON 供情景生成脚本用: {file: {summer, winter, summer_apply, winter_apply,
#                                    beta_summer, beta_winter, 及对照口径/基温}}
# beta = 回归绝对增量(万kWh/单位CDD·HDD, 年固定效应口径=已剔除年际自然增长), 应用弹性为0时同步置0
jout = os.path.join(BASE, '气温弹性_2022_2023.json')
payload = {}
for r in result:
    s = next(a['夏弹性应用'] for a in apply_result if a['file'] == r['file'])
    w = next(a['冬弹性应用'] for a in apply_result if a['file'] == r['file'])
    sp = next(a['夏标幺应用'] for a in apply_result if a['file'] == r['file'])
    wp = next(a['冬标幺应用'] for a in apply_result if a['file'] == r['file'])
    sw = next(a['夏β万应用'] for a in apply_result if a['file'] == r['file'])
    ww = next(a['冬β万应用'] for a in apply_result if a['file'] == r['file'])
    payload[r['file']] = {'summer': r['夏弹性_epCDD'], 'winter': r['冬弹性_epHDD'],
                          'summer_apply': s, 'winter_apply': w,
                          'beta_summer': r['beta_夏'] if s != 0 else 0.0,
                          'beta_winter': r['beta_冬'] if w != 0 else 0.0,
                          # 标幺化口径(主口径, 2026-09-04领导拍板): β单位=%/度日, 情景带按%缩放
                          'beta_pct_summer': sp, 'beta_pct_winter': wp,
                          'beta_pct_summer_raw': r['夏弹性_标幺'], 'beta_pct_winter_raw': r['冬弹性_标幺'],
                          'p_pct_summer': r['p_夏_标幺'], 'p_pct_winter': r['p_冬_标幺'],
                          # 万kWh/°C口径(2026-09-04用户要求): 标幺估计×2022-23季均电量换算, 绝对增量施加
                          'beta_wan_summer': sw, 'beta_wan_winter': ww,
                          'beta_summer_raw': r['beta_夏_未剔除增长'],      # 对照: 未剔除年际增长
                          'beta_winter_raw': r['beta_冬_未剔除增长'],
                          'eps_summer_raw': r['夏弹性_未剔除增长'],
                          'eps_winter_raw': r['冬弹性_未剔除增长'],
                          'eps_summer_monthfe': r['夏弹性_月FE稳健'],
                          'eps_winter_monthfe': r['冬弹性_月FE稳健'],
                          'base_summer': BASE_SUMMER, 'base_winter': BASE_WINTER}
with open(jout, 'w', encoding='utf-8') as f:
    json.dump(payload, f, ensure_ascii=False, indent=1)
print('已输出:', jout)
print('\n应用弹性(显著筛选+物理修正, 年固定效应口径=已剔除年际自然增长):')
for a in apply_result:
    print(f"  {a['file']}: 夏 {a['夏弹性应用']:+.3f} 冬 {a['冬弹性应用']:+.3f}")
