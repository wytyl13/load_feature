# -*- coding: utf-8 -*-
"""自包含完整运行入口；从任意当前工作目录启动均以本文件为根。"""
from __future__ import print_function
import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(python, script, *args):
    cmd = [python, str(ROOT / script)] + [str(x) for x in args]
    print("\n>>>", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.call(cmd, cwd=str(ROOT), env=env)
    if result:
        raise SystemExit("failed (exit %s): %s" % (result, script))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-arima", action="store_true")
    args = parser.parse_args()
    py = args.python
    run(py, "10_扩展行业/check_environment.py")
    if not args.skip_arima:
        run(py, "01_电量预测/forecast_arima_ml.py")
        run(py, "01_电量预测/build_nn_forecast_csv.py")
    for script in (
        "03_气温敏感性/build_base_kink.py",
        "03_气温敏感性/build_base_scan_fig.py",
        "03_气温敏感性/build_temperature_elasticity.py",
        "03_气温敏感性/build_temperature_years.py",
        "07_8760曲线/合成用户成行业标幺8760曲线.py",
        "07_8760曲线/批量生成2030行业8760标幺曲线_三假前当后周.py",
        "07_8760曲线/2030行业标幺曲线转有名值.py",
        "07_8760曲线/生成扣除重点行业剩余标幺曲线.py",
    ):
        run(py, script)
    for year in (2020, 2021, 2023, 2030):
        run(py, "10_扩展行业/build_new_8760_ccd_原始.py", year)
    run(py, "10_扩展行业/run_original_extensions.py", "--skip-arima")
    print("\nCOMPLETE: all package stages finished")


if __name__ == "__main__":
    main()
