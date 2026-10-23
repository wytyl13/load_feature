#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/10/21 15:40
@Author  : weiyutao
@File    : paths.py

服务层路径常量与上传大小上限,全项目只在此定义。
"""

import os
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parents[1]      # service/ 目录
RUNS_DIR = SERVICE_DIR / "runs"                        # 运行工作区根目录
DB_PATH = RUNS_DIR / "service.sqlite3"                 # SQLite 数据库文件

# 课题2 集成版算法(refactor/load_forecast)的运行环境与入口。
# 依赖锁在 conda load_forecast 环境(DESIGN D18),与服务的 load_feature 环境不同,
# 因此服务不做进程内 import,一律 subprocess 调 CLI;解释器路径可用环境变量覆盖。
FORECAST_PYTHON = Path(
    os.environ.get("LOAD_FORECAST_PYTHON", r"C:\Users\weiyutao\.conda\envs\load_forecast\python.exe")
)
FORECAST_PIPELINE = SERVICE_DIR.parent / "refactor" / "load_forecast" / "pipeline.py"

# 上传文件大小上限(字节)
LOAD_WORKBOOK_MAX_BYTES = 200 * 1024 * 1024    # 用户曲线工作簿(样例约 6.5MB)
OFFICIAL_WORKBOOK_MAX_BYTES = 50 * 1024 * 1024  # 行业电量工作簿(样例约 43KB)
ANNUAL_ENERGY_MAX_BYTES = 20 * 1024 * 1024     # 年度电量模板(默认模板约 184KB)
MACRO_WORKBOOK_MAX_BYTES = 20 * 1024 * 1024    # 课题2 宏观工作簿 111.xlsx(样例约 16KB)
INDUSTRY_ENERGY_MAX_BYTES = 10 * 1024 * 1024   # 课题2 行业电量简表(样例约 10KB)
INDUSTRY_LOAD_MAX_BYTES = 50 * 1024 * 1024     # 课题2 行业历史负荷单文件 10 sheet(样例约 2MB)
