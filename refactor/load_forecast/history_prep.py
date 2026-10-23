#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""04 段:历史负荷校验与标准化落盘(原包 04_历史负荷预处理/run_stage.py 移植)。

计算行逐行保持一致;差异仅一处:10 目录×2 CSV 改为 行业历史负荷.xlsx
单文件 10 sheet,行业名直接取 sheet 名(不再目录映射),行业迭代序=工作簿
sheet 序(已冻结为原 sorted 目录序,审计行序一致)。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_io import load_industry_loads


def validate_curve(frame: pd.DataFrame, label: str, year: int) -> pd.DataFrame:
    """校验一个行业一年8760点曲线，统一成“时间、值”两列并按时序排序。"""
    result = pd.DataFrame({
        "时间": pd.to_datetime(frame["时间"], errors="coerce"),
        "值": pd.to_numeric(frame["值"], errors="coerce"),
    }).dropna().sort_values("时间")
    expected = pd.date_range("{}-01-01 00:00:00".format(year), "{}-12-31 23:00:00".format(year), freq="h")
    if len(result) != len(expected):
        raise RuntimeError("{}: expected {} rows, got {}".format(label, len(expected), len(result)))
    if result["时间"].duplicated().any() or not np.array_equal(result["时间"].values, expected.values):
        raise RuntimeError("{}: timestamps are duplicated, missing or not hourly-continuous".format(label))
    if not np.isfinite(result["值"].values).all() or float(result["值"].max()) <= 0:
        raise RuntimeError("{}: values contain invalid numbers or no positive load".format(label))
    return result


def run_history_prep(load_workbook: Path, run_dir: Path, config: dict) -> Path:
    """校验20个历史序列并落盘为后续周法使用的标准目录结构。"""
    output = run_dir / "04_历史负荷预处理"
    prepared_root = output / "标准化行业历史负荷"
    prepared_root.mkdir(parents=True, exist_ok=True)
    years = [int(year) for year in config["history_years"]]
    audit_rows = []

    # 每个行业必须有2022和2023两年数据;sheet 名即标准行业名,
    # 后续阶段只使用标准名称。
    for industry, sheet in load_industry_loads(load_workbook).items():
        target_directory = prepared_root / industry
        target_directory.mkdir(parents=True, exist_ok=True)
        for year in years:
            part = sheet[sheet["时间"].astype(str).str.startswith(str(year))]
            curve = validate_curve(part, f"{industry}/{year}", year)
            curve.to_csv(target_directory / "历史负荷数据{}.csv".format(year), index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")
            values = curve["值"].values
            audit_rows.append({
                "行业": industry,
                "年份": year,
                "小时数": len(curve),
                "零值点数": int(np.sum(values == 0)),
                "零值占比_pct": float(np.mean(values == 0) * 100.0),
                "最小值": float(np.min(values)),
                "最大值": float(np.max(values)),
                "平均值": float(np.mean(values)),
                "最大负荷利用小时_h": float(np.sum(values) / np.max(values)),
            })
    pd.DataFrame(audit_rows).to_csv(output / "历史负荷质量检查.csv", index=False, encoding="utf-8-sig")
    print("[04] Historical curves validated and standardized: {} files".format(len(audit_rows)))
    return output
