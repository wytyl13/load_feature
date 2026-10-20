# -*- coding: utf-8 -*-
"""
基温扫描曲线 + 温度-负荷离散图。副本只保留原始负荷预测行业范围。
上排: R²-基温扫描曲线(夏CDD 20~32℃/冬HDD 5~20℃, 标幺化口径, 总负荷) — 数据找最优基温
下排: 温度-负荷离散图(2022/2023逐日, 标幺日电量, 灰点+0.5℃bin均值红线, 竖线=国标基温/数据最优基温)
输出: 预测结果_2020-2030/气温情景图/基温扫描与离散图.png

结构(S10 重构,行为不变): main() 编排——load_daily_frame 温度+总负荷日电量 →
load_season_frames 夏冬标幺 → 读弹性脚本扫描csv → plot_scan_figure 四联图。
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE))
from project_paths import FORECAST_ROOT, HOURLY_ROOT, TEMPERATURE_CSV
DATA_ROOT = str(HOURLY_ROOT)
TEMP_CSV = str(TEMPERATURE_CSV)
OUTROOT = str(FORECAST_ROOT)
FIGOUT = os.path.join(OUTROOT, '气温情景图')

INDUSTRY_FILES = ['互联网和相关服务_MW.csv', '化学原料和化学制品制造业_MW.csv',
                  '总负荷去掉九个行业负荷.csv', '有色金属冶炼和压延加工业_MW.csv', '汽车制造业_MW.csv',
                  '电气机械和器材制造业_MW.csv', '非金属矿物制品业_MW.csv', '黑色金属冶炼和压延加工业_MW.csv']
SUMMER = [6, 7, 8, 9]
WINTER = [11, 12, 1, 2]


# ---------- 数据: 16市日均温 + 总负荷日电量(10类之和) ----------
def load_daily_frame() -> pd.DataFrame:
    """装配 2022/2023 日表: 16市日均温 + 总负荷(10类之和)日电量。

    Returns:
        pd.DataFrame: 列=date/temp_mean/总负荷。
    """
    temp = pd.read_csv(TEMP_CSV, encoding='utf-8-sig')
    temp['date'] = pd.to_datetime(temp['date'])
    temp = temp[temp['date'].dt.year.isin([2022, 2023])].copy()

    daily = temp[['date', 'temp_mean']].copy()
    tot = np.zeros(len(daily))
    for fn in INDUSTRY_FILES:
        for year in (2022, 2023):
            p = os.path.join(DATA_ROOT, str(year), fn)
            try:
                v = pd.read_csv(p, encoding='utf-8-sig')['值'].to_numpy(float)
            except UnicodeDecodeError:
                v = pd.read_csv(p, encoding='gbk')['值'].to_numpy(float)
            assert len(v) == 8760
            de = pd.Series(v.reshape(-1, 24).sum(axis=1) * 0.1).to_numpy()   # 万kWh/日
            m = daily['date'].dt.year.to_numpy() == year
            tot[m] += de
    daily['总负荷'] = tot
    return daily


# 标幺: ÷同年同季平均日电量 (与主口径一致)
def perunit(s):
    yr = s['date'].dt.year
    return s['总负荷'] / s.groupby(yr)['总负荷'].transform('mean')


def load_season_frames(daily: pd.DataFrame) -> tuple:
    """切夏(6-9月)/冬(11-2月)子集并按同年同季均值标幺。

    Args:
        daily (pd.DataFrame): load_daily_frame 输出。

    Returns:
        tuple: (summer, winter) 各含 date/temp_mean/总负荷/pu。
    """
    summer = daily[daily['date'].dt.month.isin(SUMMER)].copy()
    winter = daily[daily['date'].dt.month.isin(WINTER)].copy()
    summer['pu'] = perunit(summer)
    winter['pu'] = perunit(winter)
    return summer, winter


# ---------- 扫描曲线: 读弹性脚本产出的扫描csv (总负荷, 标幺口径) ----------
def plot_scan_figure(summer: pd.DataFrame, winter: pd.DataFrame, scan: pd.DataFrame) -> None:
    """画基温扫描曲线(上排) + 温度-负荷离散图(下排) 四联图。"""
    z = scan[scan['行业'] == '总负荷(10类之和)']

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

    # (a) 夏 CDD基温扫描
    ax = axes[0][0]
    d = z[(z['类型'] == 'CDD')].dropna()
    ax.plot(d['基温C'], d['R2'], 'o-', color='#2c3e50', lw=1.6, ms=4)
    ax.axvline(26, color='#1f6fb2', ls='--', lw=1.4, label='国标26°C ($R^2$=0.626)')
    ax.axvline(23, color='#c0392b', ls='--', lw=1.4, label='数据最优23°C ($R^2$=0.772)')
    ax.set_title('夏季制冷基温扫描：$R^2$随基温变化（总负荷, 标幺口径）', fontsize=12)
    ax.set_xlabel('CDD基温 (°C)'); ax.set_ylabel(r'回归$R^2$')
    ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=10)

    # (b) 冬 HDD基温扫描
    ax = axes[0][1]
    d = z[(z['类型'] == 'HDD')].dropna()
    ax.plot(d['基温C'], d['R2'], 'o-', color='#2c3e50', lw=1.6, ms=4)
    ax.axvline(18, color='#c0392b', ls='--', lw=1.4, label='18°C (平台上, $R^2$=0.150)')
    ax.set_ylim(0.13, 0.16)
    ax.set_title('冬季采暖基温扫描：5~20°C全程平坦，数据分不出优劣', fontsize=12)
    ax.set_xlabel('HDD基温 (°C)'); ax.set_ylabel(r'回归$R^2$')
    ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=10)

    # (c) 夏 温度-负荷离散
    ax = axes[1][0]
    ax.scatter(summer['temp_mean'], summer['pu'], s=9, alpha=0.30, color='#95a5a6', label='2022/2023逐日(标幺)')
    bins = np.arange(17.0, 31.51, 0.5)
    g = summer.copy(); g['bin'] = pd.cut(g['temp_mean'], bins)
    cnt = g.groupby('bin', observed=True)['pu'].count()
    bm = g.groupby('bin', observed=True).agg(t=('temp_mean', 'mean'), y=('pu', 'mean'))
    bm = bm[cnt.to_numpy() >= 4]   # 样本<4天的分箱不画(端点噪声)
    ax.plot(bm['t'], bm['y'], 'o-', color='#c0392b', lw=1.8, ms=4, label='0.5°C分箱均值')
    ax.axvline(26, color='#1f6fb2', ls='--', lw=1.4, label='国标26°C')
    ax.axvline(23, color='#c0392b', ls=':', lw=1.4, label='数据最优23°C')
    ax.set_title('温度-负荷离散图·夏(6-9月)：23°C以上抬头明显，26°C前已起坡', fontsize=12)
    ax.set_xlabel('山东16市日均温 (°C)'); ax.set_ylabel('日电量 / 同年夏季均值 (标幺)')
    ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=9)

    # (d) 冬 温度-负荷离散
    ax = axes[1][1]
    ax.scatter(winter['temp_mean'], winter['pu'], s=9, alpha=0.30, color='#95a5a6', label='2022/2023逐日(标幺)')
    bins = np.arange(-10.0, 22.01, 1.0)
    g = winter.copy(); g['bin'] = pd.cut(g['temp_mean'], bins)
    cnt = g.groupby('bin', observed=True)['pu'].count()
    bm = g.groupby('bin', observed=True).agg(t=('temp_mean', 'mean'), y=('pu', 'mean'))
    bm = bm[cnt.to_numpy() >= 4]   # 样本<4天的分箱不画(端点噪声)
    ax.plot(bm['t'], bm['y'], 'o-', color='#c0392b', lw=1.8, ms=4, label='1°C分箱均值')
    ax.axvline(18, color='#c0392b', ls='--', lw=1.4, label='18°C(供热条例室温达标线)')
    ax.set_title('温度-负荷离散图·冬(11-2月)：整体平缓微升，集中供暖下无采暖激增', fontsize=12)
    ax.set_xlabel('山东16市日均温 (°C)'); ax.set_ylabel('日电量 / 同年冬季均值 (标幺)')
    ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=9)

    fig.suptitle('26°C/18°C 基温的数据验证：基温扫描曲线 + 温度-负荷离散图（总负荷, 标幺化口径, 2022/2023实测）', fontsize=13.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIGOUT, '基温扫描与离散图.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print('已输出:', out)


def main() -> None:
    """基温扫描与离散图主流程:日表→季标幺→读扫描csv→画图。"""
    os.makedirs(FIGOUT, exist_ok=True)
    daily = load_daily_frame()
    summer, winter = load_season_frames(daily)
    scan = pd.read_csv(os.path.join(BASE, '气温基温扫描_2022_2023.csv'), encoding='utf-8-sig')
    plot_scan_figure(summer, winter, scan)


if __name__ == '__main__':
    main()
