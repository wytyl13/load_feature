# -*- coding: utf-8 -*-
"""
负荷预测原始链路的基温拐点判定。副本范围不包含后续扩展行业。
基温拐点判定 (2026-09-05 甲方拍板重做底层逻辑, 替代"国标18/26°C"与"R²扫描23/18°C"固定阈值法):
  方法 = 把 2022/2023 全年温度和日电量拉出来对比升降, 找"电量开始变化的温度点" = 夏/冬基温
  ① 剔除 过年(春节法定假期前后各扩3天) + 全部法定节假日 (周末保留, 拟合中加周末哑变量)
  ② 归一化: 日电量 ÷ 同年全年日均电量 (吸收年际增长, 与标幺化拍板口径同思路)
  ③ 拟合: 标幺日电量 ~ a + b_hot·max(0, T−T_hot) + b_cold·max(0, T_cold−T) + 周末哑变量
     网格搜索 T_hot/T_cold (0.5°步长), 2022 / 2023 / 两年合并 各拟合一轮
  ④ 采用基温 = 2022 拐点与 2023 拐点的【算术平均值】(甲方明确: 不能取中位数, 要取平均值)
     两年合并拟合拐点作为参考一并列出
输出: 气温基温判定_2022_2023.csv + 气温基温判定.json(供 build_temperature_elasticity.py 读取)
      + 预测结果_2020-2030/气温情景图/基温拐点判定图.png
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.normpath(os.path.join(BASE, '..', '2026-08-12'))
OUTROOT = os.path.join(BASE, '预测结果_2020-2030')
FIGOUT = os.path.join(OUTROOT, '气温情景图')
os.makedirs(FIGOUT, exist_ok=True)

SERIES = {'总负荷': ['互联网和相关服务_MW.csv', '化学原料和化学制品制造业_MW.csv',
                    '总负荷去掉九个行业负荷.csv', '有色金属冶炼和压延加工业_MW.csv', '汽车制造业_MW.csv',
                    '电气机械和器材制造业_MW.csv', '非金属矿物制品业_MW.csv', '黑色金属冶炼和压延加工业_MW.csv'],
          '扣除(居民商业等)': ['总负荷去掉九个行业负荷.csv']}   # 总负荷拍板口径; 扣除(空调主体)作稳健性对照

# 2022/2023 法定节假日 (与 build_temperature_elasticity.py HOLIDAYS 一致)
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
# 过年扩展窗口: 春节法定假期 ±3 天 (返乡/复工负荷形态偏离, 比法定7天更宽)
SF_EXT = {2022: ('2022-01-28', '2022-02-09'), 2023: ('2023-01-18', '2023-01-30')}

MIN_DAYS = 20          # 拐点两侧最少样本天数(单侧), 不足则该格点无效
GRID_HOT = np.arange(18.0, 30.01, 0.5)    # 夏拐点搜索范围
GRID_COLD = np.arange(0.0, 18.01, 0.5)    # 冬拐点搜索范围


def readc(f):
    try:
        return pd.read_csv(f, encoding='utf-8-sig')
    except UnicodeDecodeError:
        return pd.read_csv(f, encoding='gbk')


# ---------- 数据: 16市日均温 + 逐日电量 ----------
# 温度是 16 市等权日均温；负荷先由小时聚合为日，再按同年平均值标幺。
temp = pd.read_csv(os.path.join(BASE, 'temperature_daily_shandong.csv'), encoding='utf-8-sig')
temp['date'] = pd.to_datetime(temp['date'])
temp = temp[temp['date'].dt.year.isin([2022, 2023])].copy()
daily = temp[['date', 'temp_mean']].copy()

for name, files in SERIES.items():
    tot = np.zeros(len(daily))
    for fn in files:
        for year in (2022, 2023):
            v = readc(os.path.join(DATA_ROOT, str(year), fn))['值'].to_numpy(float)
            assert len(v) == 8760, f'{year} {fn} 长度 {len(v)}'
            de = pd.Series(v.reshape(-1, 24).sum(axis=1) * 0.1).to_numpy()   # 万kWh/日
            tot[daily['date'].dt.year.to_numpy() == year] += de
    daily[name] = tot

# 标幺: ÷同年全年日均电量 (吸收年际增长)
for name in SERIES:
    yr = daily['date'].dt.year
    daily[f'{name}_pu'] = daily[name] / daily.groupby(yr)[name].transform('mean')

# 剔除过年+节假日
excl = np.zeros(len(daily), bool)
for year in (2022, 2023):
    d = daily['date'].dt.strftime('%Y-%m-%d')
    m = daily['date'].dt.year.to_numpy() == year
    hol = d.isin(set(HOLIDAYS[year])).to_numpy()
    s, e = pd.Timestamp(SF_EXT[year][0]), pd.Timestamp(SF_EXT[year][1])
    sf = ((daily['date'] >= s) & (daily['date'] <= e)).to_numpy()
    excl |= (hol | sf) & m
work = daily[~excl].copy()
print(f'剔除: 法定节假日+过年扩展窗口 共 {int(excl.sum())} 天 (2022 {int(excl[daily.date.dt.year==2022].sum())}/'
      f'2023 {int(excl[daily.date.dt.year==2023].sum())}), 建模样本 {len(work)} 天')


# ---------- V形分段线性拟合 ----------
# max(0,T-T_hot) 表示制冷侧超出基温的部分；max(0,T_cold-T) 表示制热侧低于基温的部分。
def vfit(T, y, wknd, t_hot, t_cold):
    X = np.column_stack([np.ones(len(T)),
                         np.maximum(0.0, T - t_hot),
                         np.maximum(0.0, t_cold - T),
                         wknd])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    return dict(r2=1 - ss_res / ss_tot, a=float(beta[0]), b_hot=float(beta[1]), b_cold=float(beta[2]))


def find_kinks(s, label):
    """s: work 子集, 含 temp_mean / {label}_pu / weekend; 返回 2022/2023/合并 三组拐点"""
    T = s['temp_mean'].to_numpy(float)
    W = s['weekend'].to_numpy(float)
    out = {}
    for scope, m in [('2022', s['date'].dt.year.to_numpy() == 2022),
                     ('2023', s['date'].dt.year.to_numpy() == 2023),
                     ('合并', np.ones(len(s), bool))]:
        Tm, Ym, Wm = T[m], s[f'{label}_pu'].to_numpy(float)[m], W[m]
        best = None
        for th in GRID_HOT:
            n_hot = int((Tm > th).sum())
            if n_hot < MIN_DAYS:
                continue
            for tc in GRID_COLD:
                if int((Tm < tc).sum()) < MIN_DAYS:
                    continue
                r = vfit(Tm, Ym, Wm, th, tc)
                if best is None or r['r2'] > best['r2']:
                    best = dict(base_summer=round(float(th), 2), base_winter=round(float(tc), 2),
                                a=round(r['a'], 4),
                                b_hot_pct=round(r['b_hot'] * 100, 3), b_cold_pct=round(r['b_cold'] * 100, 3),
                                r2=round(r['r2'], 4), n=int(m.sum()), n_hot=n_hot, n_cold=int((Tm < tc).sum()))
        out[scope] = best
        print(f'  [{label}|{scope}] 夏拐点 {best["base_summer"]}°C (上方{best["n_hot"]}天, 坡{best["b_hot_pct"]:+.2f}%/°C)  '
              f'冬拐点 {best["base_winter"]}°C (下方{best["n_cold"]}天, 坡{best["b_cold_pct"]:+.2f}%/°C)  R²={best["r2"]:.4f}')
    return out


work['weekend'] = (work['date'].dt.dayofweek >= 5).astype(float)
work['date'] = pd.to_datetime(work['date'])
result = {}
for label in SERIES:
    print(f'--- {label} 拐点搜索 ---')
    result[label] = find_kinks(work, label)

# ---------- 采用基温 = 2022与2023拐点的算术平均值 (甲方: 不取中位数, 取平均值) ----------
adopted = {}
for label in SERIES:
    r = result[label]
    bs = round((r['2022']['base_summer'] + r['2023']['base_summer']) / 2, 2)
    bw = round((r['2022']['base_winter'] + r['2023']['base_winter']) / 2, 2)
    adopted[label] = dict(base_summer=bs, base_winter=bw)
    print(f'>>> {label} 采用基温(两年平均): 夏 {bs}°C | 冬 {bw}°C')

# ---------- 输出 ----------
rows = []
for label in SERIES:
    for scope, r in result[label].items():
        rows.append(dict(序列=label, 范围=scope, 夏拐点C=r['base_summer'], 冬拐点C=r['base_winter'],
                         截距a=r['a'], 夏坡_PercC=r['b_hot_pct'], 冬坡_PercC=r['b_cold_pct'], R2=r['r2'],
                         n=r['n'], n_hot=r['n_hot'], n_cold=r['n_cold']))
    rows.append(dict(序列=label, 范围='采用(两年平均)', 夏拐点C=adopted[label]['base_summer'],
                     冬拐点C=adopted[label]['base_winter'], 夏坡_PercC=np.nan, 冬坡_PercC=np.nan,
                     R2=np.nan, n=np.nan, n_hot=np.nan, n_cold=np.nan))
df = pd.DataFrame(rows)
csv_out = os.path.join(BASE, '气温基温判定_2022_2023.csv')
df.to_csv(csv_out, index=False, encoding='utf-8-sig')
print('已输出:', csv_out)

json_out = os.path.join(BASE, '气温基温判定.json')
with open(json_out, 'w', encoding='utf-8') as f:
    json.dump({'adopted': {'base_summer': adopted['总负荷']['base_summer'],
                           'base_winter': adopted['总负荷']['base_winter'],
                           'rule': '2022/2023全年温度-日电量V形拐点两年算术平均(build_base_kink.py), 剔除过年+节假日',
                           'by_scope': result['总负荷'],
                           'by_scope_kouchu': result['扣除(居民商业等)']},
               'excluded_days': int(excl.sum())},
              f, ensure_ascii=False, indent=1)
print('已输出:', json_out)

# ---------- 图: 全年温度-日电量散点 + 分箱均值(平均) + V形拟合 + 拐点线 ----------
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
T_all = work['temp_mean'].to_numpy(float)
Y_all = work['总负荷_pu'].to_numpy(float)

# (a) 合并散点 + 1°C分箱均值 + 合并拟合V线 + 采用基温
ax = axes[0]
ax.scatter(T_all, Y_all, s=10, alpha=0.30, color='#95a5a6', label='2022/2023逐日(标幺,已剔节假日)')
bins = np.arange(-10.0, 33.01, 1.0)
g = work.assign(bin=pd.cut(work['temp_mean'], bins))
cnt = g.groupby('bin', observed=True)['总负荷_pu'].count()
bm = g.groupby('bin', observed=True).agg(t=('temp_mean', 'mean'), y=('总负荷_pu', 'mean'))
bm = bm[cnt.to_numpy() >= 4]
ax.plot(bm['t'], bm['y'], 'o-', color='#c0392b', lw=1.8, ms=4, label='1°C分箱均值(算术平均)')
p = result['总负荷']['合并']
tt = np.linspace(-10, 32, 200)
bc, bh = p['base_winter'], p['base_summer']
slope_c = result['总负荷']['合并']['b_cold_pct'] / 100
slope_h = result['总负荷']['合并']['b_hot_pct'] / 100
vy = p.get('a', 1.0) + np.maximum(0, tt - bh) * slope_h + np.maximum(0, bc - tt) * slope_c
ax.plot(tt, vy, '-', color='#2c3e50', lw=1.8, label=f'V形拟合(合并$R^2$={p["r2"]:.3f})')
ax.axvline(adopted['总负荷']['base_summer'], color='#d35400', ls='--', lw=1.6,
           label=f"夏基温{adopted['总负荷']['base_summer']}°C(两年平均)")
ax.axvline(adopted['总负荷']['base_winter'], color='#1f6fb2', ls='--', lw=1.6,
           label=f"冬基温{adopted['总负荷']['base_winter']}°C(两年平均)")
ax.set_title('全年温度-日电量对比：电量开始升降的温度点(基温拐点)', fontsize=12)
ax.set_xlabel('山东16市日均温 (°C)'); ax.set_ylabel('日电量 / 同年全年均值 (标幺)')
ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=9)

# (b) 分年分箱均值曲线 + 各年拐点 + 平均
ax = axes[1]
for year, color in [(2022, '#1f6fb2'), (2023, '#e67e22')]:
    gy = work[work['date'].dt.year == year].assign(bin=pd.cut(work.loc[work['date'].dt.year == year, 'temp_mean'], bins))
    c2 = gy.groupby('bin', observed=True)['总负荷_pu'].count()
    b2 = gy.groupby('bin', observed=True).agg(t=('temp_mean', 'mean'), y=('总负荷_pu', 'mean'))
    b2 = b2[c2.to_numpy() >= 4]
    ax.plot(b2['t'], b2['y'], 'o-', color=color, lw=1.5, ms=3.5, label=f'{year}分箱均值')
    r = result['总负荷'][str(year)]
    ax.axvline(r['base_summer'], color=color, ls=':', lw=1.4)
    ax.axvline(r['base_winter'], color=color, ls=':', lw=1.4)
ax.axvline(adopted['总负荷']['base_summer'], color='#d35400', ls='--', lw=1.6,
           label=f"夏基温{adopted['总负荷']['base_summer']}°C = 两年拐点平均")
ax.axvline(adopted['总负荷']['base_winter'], color='#1f6fb2', ls='--', lw=1.6,
           label=f"冬基温{adopted['总负荷']['base_winter']}°C = 两年拐点平均")
ax.set_title(f'2022 vs 2023 拐点对比：采用值取两年拐点算术平均(非中位数)', fontsize=12)
ax.set_xlabel('山东16市日均温 (°C)'); ax.set_ylabel('日电量 / 同年全年均值 (标幺)')
ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=9)

fig.suptitle('基温拐点判定：2022/2023全年温度-日电量V形拟合(已剔除过年与节假日, 标幺化口径)', fontsize=13.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig_out = os.path.join(FIGOUT, '基温拐点判定图.png')
fig.savefig(fig_out, dpi=150)
plt.close(fig)
print('已输出:', fig_out)
