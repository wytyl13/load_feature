# -*- coding: utf-8 -*-
"""
2030 温度情景构造 (2026-08-25): 气候态分位法
数据: temperature_daily_shandong.csv (2020-2023 山东5城平均, ERA5)
方法: 逐日历日(2030年历, 跳过2/29)取 4年×前后7日窗口=60个观测的 p10/p50/p90
  => 低温年/正常年/高温年 三条 2030 逐日温度序列 (气候态情景, 具体某天温度不可预报)
输出: 2030_temperature_scenarios.csv (date, t_lo, t_mid, t_hi)
      + 月均温阈值表 (各月: 4年实际月均温 + 低/中/高情景月均温)

结构(S10 重构,行为不变): main() 编排——load_calendar_temp 读温剔2/29 →
build_scenarios 逐日历日分位 → 输出情景CSV → build_monthly_table 月均温阈值表并输出。
"""
import os
import sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE))
from project_paths import FORECAST_ROOT, TEMPERATURE_CSV
OUTROOT = str(FORECAST_ROOT)


def load_calendar_temp() -> pd.DataFrame:
    """读山东日均温并剔除 2/29(2030 非闰年)。

    Returns:
        pd.DataFrame: 列=date/temp_mean/md(月-日)。
    """
    temp = pd.read_csv(str(TEMPERATURE_CSV), encoding='utf-8-sig')
    temp['date'] = pd.to_datetime(temp['date'])
    temp['md'] = temp['date'].dt.strftime('%m-%d')   # 月-日 (跨年对齐用)
    return temp[temp['md'] != '02-29'].copy()


def build_scenarios(temp: pd.DataFrame) -> pd.DataFrame:
    """逐 2030 日历日取 ±7 日窗口观测的 p10/p50/p90。

    Args:
        temp (pd.DataFrame): load_calendar_temp 输出。

    Returns:
        pd.DataFrame: 列=date(2030)/t_lo/t_mid/t_hi。
    """
    dates2030 = pd.date_range('2030-01-01', '2030-12-31')
    rows = []
    for d in dates2030:
        md = d.strftime('%m-%d')
        d0 = pd.Timestamp(f'2030-{md}')
        # 前后7日窗口 (按年内日序, 跨年卷绕)
        doy = d0.dayofyear
        window_days = set()
        for k in range(-7, 8):
            dd = d0 + pd.Timedelta(days=k)
            window_days.add(dd.strftime('%m-%d'))
        obs = temp[temp['md'].isin(window_days)]['temp_mean'].to_numpy()
        if len(obs) < 20:
            print(f'警告 {md}: 窗口观测 {len(obs)}')
        rows.append((d, np.percentile(obs, 10), np.percentile(obs, 50), np.percentile(obs, 90)))
    return pd.DataFrame(rows, columns=['date', 't_lo', 't_mid', 't_hi'])


def build_monthly_table(temp: pd.DataFrame, scen: pd.DataFrame) -> pd.DataFrame:
    """月均温阈值表: 4年实际月均温 + 三情景月均温 + 4年极值。

    Args:
        temp (pd.DataFrame): load_calendar_temp 输出(已剔 2/29)。
        scen (pd.DataFrame): build_scenarios 输出。

    Returns:
        pd.DataFrame: 行=月, 列=各年/情景/极值月均温(保留1位)。
    """
    def monthly(vals, col):
        idx = pd.to_datetime(scen['date'].to_numpy())
        s = pd.Series(np.asarray(vals, float), index=idx)
        return s.groupby(s.index.month).mean()

    act = temp.copy()
    act['mon'] = act['date'].dt.month
    piv = act.pivot_table(index='mon', columns=act['date'].dt.year, values='temp_mean', aggfunc='mean')
    out = pd.DataFrame({
        '月': range(1, 13),
        '2020': piv[2020], '2021': piv[2021], '2022': piv[2022], '2023': piv[2023],
        '情景低温(月均温)': monthly(scen['t_lo'], 't_lo'),
        '情景正常(月均温)': monthly(scen['t_mid'], 't_mid'),
        '情景高温(月均温)': monthly(scen['t_hi'], 't_hi'),
        '4年最高月均温': piv.max(axis=1), '4年最低月均温': piv.min(axis=1),
    })
    for c in out.columns:
        if c != '月':
            out[c] = out[c].round(1)
    return out


def main() -> None:
    """2030 温度三情景构造主流程:读温→分位情景→月阈值表→两份输出。"""
    os.makedirs(OUTROOT, exist_ok=True)
    temp = load_calendar_temp()

    scen = build_scenarios(temp)
    scen['date'] = scen['date'].dt.strftime('%Y-%m-%d')
    scen.to_csv(os.path.join(OUTROOT, '2030_temperature_scenarios.csv'), index=False, encoding='utf-8-sig')
    print('已输出 2030_temperature_scenarios.csv, 行数', len(scen))

    out = build_monthly_table(temp, scen)
    out.to_csv(os.path.join(OUTROOT, '2030温度月阈值表.csv'), index=False, encoding='utf-8-sig')
    print('\n月均温阈值表 (°C):')
    print(out.to_string(index=False))
    print('\n情景 vs 正常: 高温月ΔT = 情景高温−情景正常')
    print((out['情景高温(月均温)'] - out['情景正常(月均温)']).round(1).to_string())


if __name__ == '__main__':
    main()
