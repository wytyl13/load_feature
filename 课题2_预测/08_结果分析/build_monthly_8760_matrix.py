# -*- coding: utf-8 -*-
"""
分月8760小时曲线矩阵图: 每年12张(某年某月一张), 5年共60张
每张图 = 该年该月 3 条口径曲线(历史总负荷 / 扣除分行业 / 全系统总负荷)按月归一化叠加
- 历史总负荷: 2020-2023电网实测(17号 csv); 2030无实测→用预测总负荷(与全系统同源)
- 扣除分行业: 2020/2021 = 实测−9重点行业(生成, 负值截断, 见2026-08-19口径; 2026-08-27 v26 9行业化)
  2022/2023实测(12号) + 2030预测
- 全系统总负荷: 2020/2021 = 实测(不做总负荷预测) + 2022/2023实测(12号10类求和) + 2030预测
标幺基准 = 各曲线自己的月最大; 线型: 实测虚线, 合成/预测/计算值实线
输出: 预测结果_2020-2030/分月8760矩阵/{year}年{mo}月.png (60张)
用法: python build_monthly_8760_matrix.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, os.path.dirname(BASE))
from project_paths import FORECAST_ROOT, HOURLY_ROOT, TOTAL_LOAD_ROOT
DATA_ROOT = str(HOURLY_ROOT)   # 实测数据(2022/2023)
OUTROOT = str(FORECAST_ROOT)
OUT = os.path.join(OUTROOT, '分月8760矩阵')
os.makedirs(OUT, exist_ok=True)

INDUSTRY_FILES = ['互联网和相关服务_MW.csv', '充换电服务业_MW.csv', '化学原料和化学制品制造业_MW.csv',
                  '总负荷去掉九个行业负荷.csv', '有色金属冶炼和压延加工业_MW.csv', '汽车制造业_MW.csv',
                  '电气机械和器材制造业_MW.csv', '非金属矿物制品业_MW.csv', '黑色金属冶炼和压延加工业_MW.csv',
                  '数据中心_MW.csv']   # 2026-08-27 v26 第9个重点行业

def readc(f):
    try:
        return pd.read_csv(f, encoding='utf-8-sig')
    except UnicodeDecodeError:
        return pd.read_csv(f, encoding='gbk')

def load_hist(year):
    """电网实测总负荷: 17号 csv(2020删2/29)"""
    df = readc(os.path.join(str(TOTAL_LOAD_ROOT), '历史负荷数据%d.csv' % year))
    ts = pd.DatetimeIndex(pd.to_datetime(df['时间']))
    vals = df['值'].to_numpy(float)
    if year == 2020:
        keep = ~((ts.month == 2) & (ts.day == 29))
        ts, vals = ts[keep], vals[keep]
    return ts, vals

def load_rest(year):
    """扣除分行业: 2022=12号实测(未做预测); 2020/2021=实测−9行业(生成, 负值截断);
    2023/2030=预测合成(2023有预测)"""
    if year == 2022:
        df = readc(os.path.join(DATA_ROOT, '2022', '总负荷去掉九个行业负荷.csv'))
        return pd.DatetimeIndex(pd.to_datetime(df['时间'])), df['值'].to_numpy(float)
    df = readc(os.path.join(OUTROOT, str(year), '总负荷去掉九个行业负荷.csv'))
    return pd.DatetimeIndex(pd.to_datetime(df['时间'])), df['值'].to_numpy(float)

def load_sys(year):
    """全系统总负荷: 2022=12号10类求和(实测, 未做预测); 2020/2021=电网实测(不做总负荷预测);
    2023/2030=预测合成(2023有预测)"""
    if year in (2020, 2021):
        return load_hist(year)
    if year == 2022:
        vals = sum(readc(os.path.join(DATA_ROOT, '2022', fn))['值'].to_numpy(float)
                   for fn in INDUSTRY_FILES)
        ts = pd.date_range('2022-01-01 00:00', '2022-12-31 23:00', freq='h')
        return ts, vals
    df = readc(os.path.join(OUTROOT, str(year), '总负荷_合成.csv'))
    return pd.DatetimeIndex(pd.to_datetime(df['时间'])), df['值'].to_numpy(float)

def load_total(year):
    """总负荷: 2020-2023实测; 2030用预测(总负荷_合成.csv, 与全系统同源)"""
    if year == 2030:
        df = readc(os.path.join(OUTROOT, '2030', '总负荷_合成.csv'))
        return pd.DatetimeIndex(pd.to_datetime(df['时间'])), df['值'].to_numpy(float)
    return load_hist(year)

# 每条口径曲线: (加载器, 标签, 颜色, 线型)  实线=合成/预测/计算值, 虚线=实测
# 口径: 历史总负荷=电网实测(2030无实测用预测); 扣除分行业/全系统 =
#   未做总负荷预测的年份(2020/2021/2022)用实测口径(2020/2021扣除=实测−生成行业, 计算值实线),
#   做了预测的年份(2023/2030)用预测
def curves_for(year):
    tot = load_total(year)
    if year == 2022:   # 未做预测: 三列皆实测(9类求和 == 历史实测, 同源同值)
        return [(tot, '历史总负荷(实测)', '#1f6fb2', '--'),
                (load_rest(year), '扣除分行业(实测)', '#e67e22', '--'),
                (load_sys(year), '全系统总负荷(实测)', '#c0392b', '--')]
    if year in (2020, 2021):   # 不做总负荷预测: 总负荷=实测; 扣除分行业=实测−9重点行业(生成)
        return [(tot, '历史总负荷(实测)', '#1f6fb2', '--'),
                (load_rest(year), '扣除分行业(实测减生成行业)', '#e67e22', '-'),
                (load_sys(year), '全系统总负荷(实测)', '#c0392b', '--')]
    if year == 2030:   # 无实测: 总负荷用预测顶替(与全系统同源)
        return [(tot, '总负荷(预测)', '#1f6fb2', '-'),
                (load_rest(year), '扣除分行业(预测)', '#e67e22', '-'),
                (load_sys(year), '全系统总负荷(预测)', '#c0392b', '-')]
    # 2023: 历史实测 + 扣除/全系统预测(2023有预测)
    return [(tot, '历史总负荷(实测)', '#1f6fb2', '--'),
            (load_rest(year), '扣除分行业(预测)', '#e67e22', '-'),
            (load_sys(year), '全系统总负荷(预测)', '#c0392b', '-')]

YEARS = [2020, 2021, 2022, 2023, 2030]
month_names = ['1月', '2月', '3月', '4月', '5月', '6月',
               '7月', '8月', '9月', '10月', '11月', '12月']

for y in YEARS:
    curves = curves_for(y)
    for mo in range(1, 13):
        fig, ax = plt.subplots(figsize=(9, 4.2))
        for (ts, vals), label, color, ls in curves:
            sel = ts.month == mo
            v = vals[sel]
            t = ts[sel]
            x = (t.day.to_numpy() - 1) + t.hour.to_numpy() / 24
            ymax = v.max()
            ax.plot(x, v / ymax, color=color, ls=ls, lw=0.6,
                    label=f'{label} 月最大={ymax:,.0f}MW')
        days = int(ts[ts.month == mo].day.max())
        ticks = np.arange(1, days + 1, 2)
        if days not in ticks:
            ticks = np.append(ticks, days)
        ax.set_xticks(ticks - 1)
        ax.set_xticklabels([f'{int(d)}日' for d in ticks], fontsize=7)
        ax.set_xlim(-0.4, days)
        ax.set_ylabel('标幺值')
        ax.set_title(f'{y}年{month_names[mo-1]}8760小时负荷曲线(标幺, 基准=月最大)', fontsize=11.5)
        ax.grid(True, alpha=0.3)
        fig.subplots_adjust(bottom=0.22, top=0.89)
        fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3,
                   frameon=False, fontsize=8)
        fig.savefig(os.path.join(OUT, f'{y}年{month_names[mo-1]}.png'), dpi=150)
        plt.close(fig)
    print(f'{y} 年 12 张完成')
print(f'已输出 60 张分月8760矩阵图到 {OUT}')
