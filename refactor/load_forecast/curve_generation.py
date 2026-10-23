#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""05 段:周法生成 10 行业 2030 标幺 8760(原包 05_行业标幺8760生成/run_stage.py 移植)。

计算行逐行保持一致;差异仅三处:
- weekly_generator.py 同目录平铺直接 import(原为 importlib 动态加载,
  生成器本体整文件零修改,md5 与原包一致);
- 行业序从 config.industry_workbook_mapping 键序改为 config.industries
  (同一冻结序);
- 审计 generator 写逻辑名 "weekly_generator.py"(比对口径 D18 归一化同名)。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import weekly_generator as generator

RESIDUAL_KEY = "其他（扣除重点行业）"


def run_curve_generation(run_dir: Path, config: dict) -> Path:
    """按配置顺序生成10类行业的2030年8760点标幺曲线,写判据文件。

    每个行业先生成独立明细和独立CSV，再拼成软件接口所需的宽表：
    第一列为时间，后面每个行业一列标幺值，不含年份列。
    """
    source_root = run_dir / "04_历史负荷预处理" / "标准化行业历史负荷"
    output = run_dir / "05_行业标幺8760生成"
    detail_root = output / "生成明细"
    curve_root = output / "行业标幺8760"
    curve_root.mkdir(parents=True, exist_ok=True)
    target_year = int(config["target_year"])
    expected_time = pd.date_range("{}-01-01 00:00:00".format(target_year), periods=8760, freq="h")
    # 统一行业顺序：九个重点行业按config顺序，“其他”固定放最后，
    # 从而保证下游CSV列顺序稳定，便于软件接口映射。
    expected_industries = list(config["industries"]) + [RESIDUAL_KEY]
    source_industries = {path.name for path in source_root.iterdir() if path.is_dir()}
    if source_industries != set(expected_industries):
        raise RuntimeError("Industry directories do not match config: {}".format(sorted(source_industries)))
    wide = pd.DataFrame({"时间": expected_time.strftime("%Y-%m-%d %H:%M:%S")})
    audit_rows = []

    for industry in expected_industries:
        industry_directory = source_root / industry
        # weekly_generator只负责曲线形状；第06步才根据年电量换算MW。
        result = generator.generate(
            input_dir=industry_directory,
            output_dir=detail_root / industry,
            target_year=target_year,
            hist_years=[int(value) for value in config["history_years"]],
            typical_years=[int(value) for value in config["typical_years"]],
            holidays=list(config["holidays"]),
            pattern=config["holiday_pattern"],
            single_typical=bool(config["single_typical"]),
        )
        generated = pd.read_csv(result["hourly_csv"], encoding="utf-8-sig")
        values = pd.to_numeric(generated["全年小时标幺"], errors="raise").values.astype(float)
        if len(values) != 8760 or not np.isfinite(values).all() or float(values.max()) <= 0:
            raise RuntimeError("{}: invalid generated per-unit curve".format(industry))
        # 再归一化一次，保证每个行业的峰值标幺严格等于1。
        values = values / float(values.max())
        frame = pd.DataFrame({
            "时间": expected_time.strftime("%Y-%m-%d %H:%M:%S"),
            "年份": target_year,
            "行业": industry,
            "行业标幺值": values,
        })
        frame.to_csv(curve_root / "{}.csv".format(industry), index=False, encoding="utf-8-sig")
        wide["{}_标幺值".format(industry)] = values
        audit_rows.append({
            "行业": industry,
            "小时数": len(values),
            "最大标幺值": float(values.max()),
            "最小标幺值": float(values.min()),
            "最大负荷利用小时_h": float(values.sum()),
        })
    if len(audit_rows) != 10:
        raise RuntimeError("Expected 10 generated industries, got {}".format(len(audit_rows)))
    if len(wide) != 8760 or len(wide.columns) != 11 or "年份" in wide.columns:
        raise RuntimeError("The industry per-unit summary is not an 8760 x 11 wide table")
    if not np.isfinite(wide.iloc[:, 1:].values).all():
        raise RuntimeError("The industry per-unit summary contains null or infinite values")
    wide.to_csv(output / "2030行业标幺8760_汇总.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).to_csv(output / "行业标幺曲线指标.csv", index=False, encoding="utf-8-sig")
    audit = {
        "method": "week-day-hour",
        "generator": "weekly_generator.py",
        "parameters": {
            "target_year": target_year,
            "history_years": config["history_years"],
            "typical_years": config["typical_years"],
            "holidays": config["holidays"],
            "holiday_pattern": config["holiday_pattern"],
            "single_typical": config["single_typical"],
            "zero_guard_threshold": config["zero_guard_threshold"],
        },
        "summary_table": {
            "format": "wide",
            "rows": len(wide),
            "columns": len(wide.columns),
            "timestamp_columns": 1,
            "industry_per_unit_columns": len(wide.columns) - 1,
            "contains_year_column": "年份" in wide.columns,
            "no_null_or_infinite": bool(np.isfinite(wide.iloc[:, 1:].values).all()),
        },
        "excluded_ccd_script_called": False,
    }
    (output / "阶段审计.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[05] Weekly per-unit curves generated: 10 industries x 8760 hours")
    return output
