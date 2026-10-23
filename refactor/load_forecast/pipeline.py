#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""课题2 软件集成版(20261022)重构:六步链唯一入口。

02 ARIMA宏观外推 → 03 MLP总电量+行业占比 → 04 历史负荷校验标准化
→ 05 周法标幺8760 → 06 MW换算与总负荷合成。

与原包差异:段由子进程改为同进程顺序调用(计算行逐段一致,见各段模块头);
输入由包内固定路径改为三件 Excel 显式传参;pipeline.log / run_manifest.json
形态保持原样(log 内容确定入判据,manifest 仅含时变的键不入 md5)。

入口:
    run_forecast(macro_workbook, industry_energy_workbook, industry_load_workbook,
                 output_dir, *, config_overrides=None) -> dict[str, Path]
    python pipeline.py --macro-xlsx … --energy-xlsx … --load-xlsx … --output-dir …
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import sys
from datetime import datetime
from pathlib import Path

from config import resolve_config
from curve_generation import run_curve_generation
from energy_forecast import run_energy_forecast
from history_prep import run_history_prep
from macro_forecast import run_macro_feature_forecast
from scaling_synthesis import run_scaling_synthesis

REQUIRED_MODULES = ["numpy", "pandas", "sklearn", "statsmodels", "openpyxl"]


def _validate_environment(workbooks: list[Path]) -> None:
    """检查依赖库与三件输入文件,报错给出可直接执行的修复命令。"""
    missing_modules = []
    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
        except ImportError:
            missing_modules.append(module)
    if missing_modules:
        raise RuntimeError(
            "Missing Python modules: {}.\nActive Python: {}\n"
            'Install with: "{}" -m pip -r "{}"'.format(
                ", ".join(missing_modules), sys.executable, sys.executable,
                Path(__file__).resolve().parent / "requirements.txt",
            )
        )
    missing = [str(path) for path in workbooks if not path.is_file()]
    if missing:
        raise RuntimeError("Missing input workbooks:\n" + "\n".join(missing))


def run_forecast(
    macro_workbook: Path,
    industry_energy_workbook: Path,
    industry_load_workbook: Path,
    output_dir: Path,
    *,
    config_overrides: dict | None = None,
) -> dict[str, Path]:
    """执行 02—06 全链,返回关键产物路径。

    output_dir 必须为空目录(或不存在),防止混入上一次运行的文件;
    任何一段抛异常即停止,manifest 记 failed 后原样上抛。
    """
    macro_workbook = Path(macro_workbook).resolve()
    industry_energy_workbook = Path(industry_energy_workbook).resolve()
    industry_load_workbook = Path(industry_load_workbook).resolve()
    _validate_environment([macro_workbook, industry_energy_workbook, industry_load_workbook])
    run_dir = Path(output_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("Requested output directory is not empty: {}".format(run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    config = resolve_config(config_overrides)

    stages = [
        ("02_宏观特征预测", lambda: run_macro_feature_forecast(macro_workbook, run_dir)),
        ("03_总电量与行业电量预测", lambda: run_energy_forecast(macro_workbook, industry_energy_workbook, run_dir, config)),
        ("04_历史负荷预处理", lambda: run_history_prep(industry_load_workbook, run_dir, config)),
        ("05_行业标幺8760生成", lambda: run_curve_generation(run_dir, config)),
        ("06_有名值换算与总负荷合成", lambda: run_scaling_synthesis(run_dir, config)),
    ]
    manifest = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "python": sys.executable,
        "inputs": {
            "macro_workbook": str(macro_workbook),
            "industry_energy_workbook": str(industry_energy_workbook),
            "industry_load_workbook": str(industry_load_workbook),
        },
        "dependency_versions": {
            module: getattr(importlib.import_module(module), "__version__", "unknown")
            for module in REQUIRED_MODULES
        },
        "stages": [],
    }
    print("Output directory: {}".format(run_dir))
    log_path = run_dir / "pipeline.log"
    with log_path.open("w", encoding="utf-8") as log:
        for stage_name, stage_call in stages:
            print("Running {} ...".format(stage_name))
            log.write("\n===== {} =====\n".format(stage_name))
            log.flush()
            started = datetime.now()
            buffer = io.StringIO()
            try:
                with contextlib.redirect_stdout(buffer):
                    stage_call()
            except Exception:
                output = buffer.getvalue()
                print(output, end="")
                log.write(output)
                manifest["stages"].append({
                    "name": stage_name,
                    "return_code": 1,
                    "elapsed_seconds": (datetime.now() - started).total_seconds(),
                })
                manifest["status"] = "failed"
                manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
                (run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
                raise
            output = buffer.getvalue()
            print(output, end="")
            log.write(output)
            log.flush()
            manifest["stages"].append({
                "name": stage_name,
                "return_code": 0,
                "elapsed_seconds": (datetime.now() - started).total_seconds(),
            })
    manifest["status"] = "passed"
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Pipeline completed successfully: {}".format(run_dir))
    return {
        "output_dir": run_dir,
        "log": log_path,
        "manifest": run_dir / "run_manifest.json",
        "industry_per_unit_summary": run_dir / "05_行业标幺8760生成" / "2030行业标幺8760_汇总.csv",
        "industry_mw_summary": run_dir / "06_有名值换算与总负荷合成" / "2030行业标幺及MW8760_汇总.csv",
        "total_load_curve": run_dir / "06_有名值换算与总负荷合成" / "2030总负荷_MW及标幺8760.csv",
        "total_metrics": run_dir / "06_有名值换算与总负荷合成" / "2030总负荷指标.csv",
        "final_validation": run_dir / "06_有名值换算与总负荷合成" / "最终校验.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete 2030 load forecasting pipeline")
    parser.add_argument("--macro-xlsx", type=Path, required=True, help="111.xlsx 宏观与电量")
    parser.add_argument("--energy-xlsx", type=Path, required=True, help="行业电量简表.xlsx")
    parser.add_argument("--load-xlsx", type=Path, required=True, help="行业历史负荷.xlsx")
    parser.add_argument("--output-dir", type=Path, required=True, help="空输出目录")
    args = parser.parse_args()
    run_forecast(args.macro_xlsx, args.energy_xlsx, args.load_xlsx, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
