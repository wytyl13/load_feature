#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/10/21 15:40
@Author  : weiyutao
@File    : paths.py

服务层路径常量与上传大小上限,全项目只在此定义。
"""

from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parents[1]      # service/ 目录
RUNS_DIR = SERVICE_DIR / "runs"                        # 运行工作区根目录
DB_PATH = RUNS_DIR / "service.sqlite3"                 # SQLite 数据库文件

# 上传文件大小上限(字节)
LOAD_WORKBOOK_MAX_BYTES = 200 * 1024 * 1024    # 用户曲线工作簿(样例约 6.5MB)
OFFICIAL_WORKBOOK_MAX_BYTES = 50 * 1024 * 1024  # 行业电量工作簿(样例约 43KB)
ANNUAL_ENERGY_MAX_BYTES = 20 * 1024 * 1024     # 年度电量模板(默认模板约 184KB)
