#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""包内默认配置(原包 01_输入数据与配置/config.json 收编)。

输入简化带来的键变更:
- 退役 source_directory_mapping/industry_workbook_mapping——行业历史负荷.xlsx
  的 sheet 名即标准行业名,行业电量简表的行名即标准行业名,不再需要映射;
- 新增 industries:9 行业标准名有序表,键序逐字冻结原 JSON industry_workbook_mapping
  的键序,03/05/06 的行列顺序全部由它派生。
"""

from __future__ import annotations

DEFAULT_CONFIG: dict = {
    "target_year": 2030,
    "history_years": [2022, 2023],
    "typical_years": [2022, 2023],
    "holidays": ["spring", "may", "national"],
    "holiday_pattern": "hybrid",
    "single_typical": True,
    "zero_guard_threshold": 0.05,
    "industries": [
        "居民生活",
        "充换电服务业",
        "化学原料和化学制品制造业",
        "有色金属冶炼和压延加工业",
        "汽车制造业",
        "电气机械和器材制造业",
        "非金属矿物制品业",
        "黑色金属冶炼和压延加工业",
        "数据中心",
    ],
    "ppt_adopted_share_model": {
        "居民生活": "linear",
        "充换电服务业": "exponential",
        "化学原料和化学制品制造业": "linear",
        "有色金属冶炼和压延加工业": "linear",
        "汽车制造业": "linear",
        "电气机械和器材制造业": "linear",
        "非金属矿物制品业": "linear",
        "黑色金属冶炼和压延加工业": "linear",
        "数据中心": "linear",
    },
}


def resolve_config(overrides: dict | None = None) -> dict:
    """默认配置合并覆盖项;未知键报错,防止拼写错误静默失效。"""
    config = dict(DEFAULT_CONFIG)
    for key, value in (overrides or {}).items():
        if key not in config:
            raise KeyError(f"未知配置键: {key}(合法键: {sorted(config)})")
        config[key] = value
    return config
