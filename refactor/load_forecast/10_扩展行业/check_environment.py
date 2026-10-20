# -*- coding: utf-8 -*-
"""交付前的环境和关键输入检查。只读，不修改数据。"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
# 交付包内置必要输入；如需使用外部数据，可直接修改此处或使用复跑入口参数。
DATA = PROJECT


def check_package(name: str) -> str:
    try:
        module = importlib.import_module(name)
        return getattr(module, "__version__", "已导入")
    except Exception as exc:
        return f"失败: {type(exc).__name__}: {exc}"


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"工作目录: {PROJECT}")
    print("依赖:")
    failed = False
    for name in ("numpy", "pandas", "scipy", "statsmodels", "sklearn", "openpyxl", "matplotlib"):
        version = check_package(name)
        print(f"  {name}: {version}")
        if version.startswith("失败:"):
            failed = True
    paths = [
        DATA / "02_统计与电量锚点" / "111.xlsx",
        DATA / "02_统计与电量锚点" / "山东2020-2025_全社会用电分类表（全口径）.xlsx",
        DATA / "03_气温敏感性" / "temperature_daily_shandong.csv",
        DATA / "03_气温敏感性" / "山东16市逐日气温_ERA5_2020-2023.xlsx",
        DATA / "09_数据中心原始" / "202201.xlsx",
        DATA / "09_数据中心原始" / "202312.xlsx",
        DATA / "04_实测负荷8760" / "总负荷" / "历史负荷数据2022.csv",
        DATA / "04_实测负荷8760" / "总负荷" / "历史负荷数据2023.csv",
    ]
    print("关键输入:")
    for path in paths:
        ok = path.exists()
        print(f"  {'OK' if ok else '缺失'}: {path}")
        failed = failed or not ok
    for year in (2022, 2023):
        path = DATA / "04_实测负荷8760" / str(year) / "充换电服务业_MW.csv"
        if not path.exists():
            print(f"  缺失: {path}")
            failed = True
            continue
        import pandas as pd
        rows = None
        for encoding in ("utf-8-sig", "gbk", "utf-8"):
            try:
                rows = len(pd.read_csv(path, encoding=encoding))
                break
            except UnicodeDecodeError:
                continue
        if rows is None:
            print(f"  异常: 无法识别编码 {path}")
            failed = True
            continue
        print(f"  {'OK' if rows == 8760 else '异常'}: {path.name} {rows}行")
        failed = failed or rows != 8760
    for directory in (DATA / "05_用户级负荷曲线", DATA / "06_生成器输入"):
        n = sum(1 for p in directory.rglob("*.csv")) if directory.exists() else 0
        print(f"  {'OK' if n else '缺失'}: {directory} ({n} 个 CSV)")
        failed = failed or n == 0
    if failed:
        print("检查未通过，请先修复环境或输入路径。")
        return 1
    print("检查通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
