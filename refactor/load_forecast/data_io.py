#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""三件 Excel 输入的读入(信任边界):111.xlsx / 行业电量简表.xlsx / 行业历史负荷.xlsx。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FEATURES = [
    "第一产业增加值",
    "第二产业增加值",
    "第三产业增加值",
    "人口",
    "全体居民人均可支配收入",
    "城镇居民人均可支配收入",
    "农村居民人均可支配收入",
]
TARGET = "电力消费量"

TOTAL_KEY = "全社会用电总计"
ENERGY_SHEET = "行业电量"
ENERGY_COLUMNS = [f"{year}年" for year in range(2020, 2026)]


def load_macro_table(path: Path) -> pd.DataFrame:
    """读取111.xlsx并整理为“年份×指标”的数值表。

    原始表是“指标在行、年份在列”；这里转置后统一为后续ARIMA和相关性
    计算所需的长时间序列格式，并对少量缺失值做线性插值。
    (与原包 02/03 段 load_macro 逐行一致,数值口径冻结。)
    """
    raw = pd.read_excel(path, sheet_name=0, header=0)
    years = [int(str(column)[:4]) for column in raw.columns[1:]]
    frame = raw.iloc[:8, 1:].T.copy()
    frame.columns = FEATURES[:4] + [TARGET] + FEATURES[4:]
    frame.index = years
    frame.index.name = "年份"
    return frame.sort_index().apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both")


def load_industry_energy(path: Path, industries: list[str]) -> pd.DataFrame:
    """读行业电量简表.xlsx → 以行业名为索引的 (10, 6) 电量表(万千瓦时, float64)。

    简表替代原 150 行分类表 + unique_row 定位;值已经生成工具 round-trip
    逐位断言与原表一致。行标签须恰好为 总计+9 行业(顺序不限,输出序由
    config.industries 派生)。
    """
    frame = pd.read_excel(path, sheet_name=ENERGY_SHEET, header=0)
    if list(frame.columns) != ["行业", *ENERGY_COLUMNS]:
        raise RuntimeError(f"行业电量简表列异常: {list(frame.columns)}")
    labels = list(frame["行业"].astype(str))
    expected = [TOTAL_KEY, *industries]
    if sorted(labels) != sorted(expected) or len(set(labels)) != len(labels):
        raise RuntimeError(f"行业电量简表行标签异常: {labels}(应为 {expected})")
    frame = frame.set_index("行业")
    frame.index = frame.index.astype(str)
    return frame.apply(pd.to_numeric, errors="raise")


def load_industry_loads(path: Path) -> dict[str, pd.DataFrame]:
    """读行业历史负荷.xlsx → {标准行业名: [时间(文本), 值(float)]}。

    按 sheet 在工作簿中的顺序迭代(该顺序承载原包 sorted 目录序,
    04 段审计行序由此派生);每 sheet 应为 2 年 8760×2=17520 行升序堆叠,
    逐位一致已由生成工具 round-trip 断言,年份切分与数值校验留给 04 段。
    """
    sheets = pd.read_excel(path, sheet_name=None, header=0)
    for name, sheet in sheets.items():
        if list(sheet.columns) != ["时间", "值"]:
            raise RuntimeError(f"负荷 sheet {name!r} 列异常: {list(sheet.columns)}(应为 [时间, 值])")
    return dict(sheets)
