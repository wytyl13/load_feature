#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/10/23
@Author  : weiyutao
@File    : make_simplified_inputs.py

一次性迁移工具:把原包 01_输入数据与配置 的三件输入转换为服务直传的两件简化输入。

产物(写入 --out-dir,默认 inputs/):
- 行业电量简表.xlsx:单 sheet"行业电量",10 行=全社会用电总计+9 行业(标准名),
  列=2020年..2025年,单位万千瓦时,值 float64 原样;
- 行业历史负荷.xlsx:10 sheet,sheet 名=标准行业名(扣除(总−9重点)→其他（扣除重点行业）),
  sheet 顺序冻结为原 sorted 目录序;每 sheet 两列[时间, 值],时间为文本,
  17520 行=2022 全年+2023 全年升序;
- 111.xlsx:直接拷贝,不转换。

写后立即读回,与源解析值 np.array_equal 逐位断言(float round-trip 铁证)。

用法:
    python make_simplified_inputs.py --source-root <原包>/01_输入数据与配置 [--out-dir inputs]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

TOTAL_KEY = "全社会用电总计"
ENERGY_SHEET = "行业电量"
ENERGY_COLUMNS = [f"{year}年" for year in range(2020, 2026)]
LOAD_TIME_COLUMN = "时间"
LOAD_VALUE_COLUMN = "值"


def unique_row(frame: pd.DataFrame, labels: pd.Series, key: str) -> pd.Series:
    """按行标签关键词唯一定位(与原包 03 段同逻辑),不唯一即报错。"""
    matches = [index for index, label in labels.items() if key in label]
    if len(matches) != 1:
        raise RuntimeError(f"分类表行定位 {key!r} 命中 {len(matches)} 行,应为 1")
    return frame.loc[matches[0]]


def build_energy_table(source_root: Path) -> pd.DataFrame:
    """分类表 → 行业电量简表(标准行业名 × 2020—2025)。"""
    config = json.loads((source_root / "config.json").read_text(encoding="utf-8"))
    workbook = pd.read_excel(
        source_root / "宏观与电量" / "山东2020-2025_全社会用电分类表（全口径）.xlsx",
        sheet_name=0, header=None,
    )
    labels = workbook.iloc[:, 0].fillna("").astype(str)
    rows = {TOTAL_KEY: unique_row(workbook, labels, TOTAL_KEY).iloc[1:7].astype(float).values}
    for industry, workbook_key in config["industry_workbook_mapping"].items():
        rows[industry] = unique_row(workbook, labels, workbook_key).iloc[1:7].astype(float).values
    table = pd.DataFrame(
        [[industry, *rows[industry]] for industry in [TOTAL_KEY, *config["industry_workbook_mapping"]]],
        columns=["行业", *ENERGY_COLUMNS],
    )
    return table


def build_load_sheets(source_root: Path) -> dict[str, pd.DataFrame]:
    """10 个行业目录 × 2 年 CSV → {标准行业名: (时间文本, 值) 两列 17520 行}。

    sheet 顺序冻结为原 sorted 目录序(扣除位原位替换标准名),由返回 dict 的插入序承载。
    """
    config = json.loads((source_root / "config.json").read_text(encoding="utf-8"))
    directory_mapping = config["source_directory_mapping"]
    years = [int(year) for year in config["history_years"]]
    source_directories = sorted(path for path in (source_root / "行业历史负荷").iterdir() if path.is_dir())
    if len(source_directories) != 10:
        raise RuntimeError(f"Expected 10 industry directories, got {len(source_directories)}")
    sheets: dict[str, pd.DataFrame] = {}
    for source_directory in source_directories:
        industry = directory_mapping.get(source_directory.name, source_directory.name)
        parts = []
        for year in years:
            frame = pd.read_csv(source_directory / f"历史负荷数据{year}.csv", encoding="utf-8-sig")
            parts.append(pd.DataFrame({
                LOAD_TIME_COLUMN: frame.iloc[:, 0].astype(str),
                LOAD_VALUE_COLUMN: pd.to_numeric(frame.iloc[:, 1], errors="raise"),
            }))
        sheet = pd.concat(parts, ignore_index=True)
        if len(sheet) != 8760 * len(years):
            raise RuntimeError(f"{industry}: 行数 {len(sheet)} != {8760 * len(years)}")
        sheets[industry] = sheet
    return sheets


def verify_round_trip(out_dir: Path, source_root: Path) -> None:
    """读回两件产物,与源解析值逐位比对(标签集/顺序/行数/数值全断言)。"""
    config = json.loads((source_root / "config.json").read_text(encoding="utf-8"))
    energy = pd.read_excel(out_dir / "行业电量简表.xlsx", sheet_name=ENERGY_SHEET, header=0)
    assert list(energy.columns) == ["行业", *ENERGY_COLUMNS], energy.columns
    assert list(energy["行业"]) == [TOTAL_KEY, *config["industry_workbook_mapping"]], list(energy["行业"])
    expected_energy = build_energy_table(source_root)
    assert np.array_equal(
        energy.iloc[:, 1:].to_numpy(dtype=float), expected_energy.iloc[:, 1:].to_numpy(dtype=float)
    ), "行业电量简表 round-trip 数值不一致"

    loads = pd.read_excel(out_dir / "行业历史负荷.xlsx", sheet_name=None, header=0)
    expected_sheets = build_load_sheets(source_root)
    assert list(loads.keys()) == list(expected_sheets.keys()), list(loads.keys())
    for name, sheet in loads.items():
        assert list(sheet.columns) == [LOAD_TIME_COLUMN, LOAD_VALUE_COLUMN], (name, sheet.columns)
        expected = expected_sheets[name]
        assert len(sheet) == len(expected), (name, len(sheet))
        assert (sheet[LOAD_TIME_COLUMN].astype(str) == expected[LOAD_TIME_COLUMN]).all(), name
        assert np.array_equal(
            sheet[LOAD_VALUE_COLUMN].to_numpy(dtype=float), expected[LOAD_VALUE_COLUMN].to_numpy(dtype=float)
        ), f"{name} 负荷 round-trip 数值不一致"
    print(f"round-trip 逐位断言通过:简表 {len(energy)} 行,负荷 {len(loads)} sheet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="原包 01_输入数据与配置")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "inputs")
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    energy_table = build_energy_table(args.source_root)
    with pd.ExcelWriter(out_dir / "行业电量简表.xlsx", engine="openpyxl") as writer:
        energy_table.to_excel(writer, sheet_name=ENERGY_SHEET, index=False)
    load_sheets = build_load_sheets(args.source_root)
    with pd.ExcelWriter(out_dir / "行业历史负荷.xlsx", engine="openpyxl") as writer:
        for industry, sheet in load_sheets.items():
            sheet.to_excel(writer, sheet_name=industry, index=False)
    shutil.copyfile(args.source_root / "宏观与电量" / "111.xlsx", out_dir / "111.xlsx")

    verify_round_trip(out_dir, args.source_root)
    print(f"简化输入已生成: {out_dir}")


if __name__ == "__main__":
    main()
