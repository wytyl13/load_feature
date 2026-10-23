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
