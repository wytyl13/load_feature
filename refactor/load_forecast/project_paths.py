# -*- coding: utf-8 -*-
"""交付包统一路径配置。

所有脚本都以本文件所在目录为项目根目录，不再依赖原工程中的
``2026-08-12``、``存档`` 或 ``自己处理`` 目录。脚本可以从任意当前
工作目录启动，只要传入的 Python 环境已安装 requirements.txt 中的依赖。
"""
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
STATS_ROOT = PROJECT_ROOT / "02_统计与电量锚点"
HOURLY_ROOT = PROJECT_ROOT / "04_实测负荷8760"
TEMPERATURE_ROOT = PROJECT_ROOT / "03_气温敏感性"
TEMPERATURE_CSV = TEMPERATURE_ROOT / "temperature_daily_shandong.csv"
TEMPERATURE_XLSX = TEMPERATURE_ROOT / "山东16市逐日气温_ERA5_2020-2023.xlsx"
USER_CURVE_ROOT = PROJECT_ROOT / "05_用户级负荷曲线" / "补全后分行业数据"
GENERATOR_INPUT_ROOT = PROJECT_ROOT / "06_生成器输入"
TOTAL_LOAD_ROOT = HOURLY_ROOT / "总负荷"
FORECAST_ROOT = PROJECT_ROOT / "预测结果_2020-2030"
ENERGY_XLSX = STATS_ROOT / "111.xlsx"
CLASSIFICATION_XLSX = STATS_ROOT / "山东2020-2025_全社会用电分类表（全口径）.xlsx"


def ensure_project_dirs() -> None:
    """创建运行输出目录；输入目录缺失时由调用脚本给出明确错误。"""
    for path in (FORECAST_ROOT, TOTAL_LOAD_ROOT):
        path.mkdir(parents=True, exist_ok=True)

