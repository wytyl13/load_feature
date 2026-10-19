# -*- coding: utf-8 -*-
"""批量生成目标年行业 8760 标幺曲线。

本脚本复用 generate_annual_normalized_curve_v2_多假_工作双休.py 的核心算法,
只做输入/输出适配:

支持两类输入:
1. direct 目录:
   某目录/历史负荷数据2022.csv
   某目录/历史负荷数据2023.csv

2. year-subdirs 根目录:
   根目录/2022/居民生活_标幺8760.csv
   根目录/2023/居民生活_标幺8760.csv
   根目录/2022/{行业}_标幺8760.csv
   根目录/2023/{行业}_标幺8760.csv

对每个识别出的行业/目录, 脚本会临时整理成原生成器要求的:
   历史负荷数据{year}.csv, 字段为 时间,值
然后调用原生成器生成:
   {target_year}年全年小时标幺曲线.csv
   {target_year}年周峰值与假期修正.csv
   {target_year}年汇总说明.csv
   {target_year}年全年小时标幺曲线.svg

默认输入:
   ../行业标幺8760曲线/{2022,2023}/*.csv

默认输出:
   ../2030行业8760标幺曲线_批量/
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_SOURCE_ROOT = PROJECT_DIR / "行业标幺8760曲线"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "2030行业8760标幺曲线_批量"
DEFAULT_GENERATOR = (
    PROJECT_DIR
    / "存档"
    / "02_需求实现逻辑"
    / "04_标幺化8760生成"
    / "generate_annual_normalized_curve_v2_多假_工作双休.py"
)


@dataclass
class Case:
    name: str
    files_by_year: dict[int, Path]
    source_type: str


def parse_years(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def read_csv_auto(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"Cannot decode {path}")


def safe_name(text: str) -> str:
    text = re.sub(r'[<>:"/\\|?*]+', "_", text.strip())
    text = re.sub(r"\s+", "_", text)
    return text or "未命名"


def normalize_case_name(path: Path, year: int) -> str:
    name = path.stem
    name = name.replace(f"历史负荷数据{year}", "")
    name = name.replace("历史负荷数据", "")
    name = name.replace("_标幺8760", "")
    name = name.replace("标幺8760", "")
    name = re.sub(rf"[-_ ]*{year}[-_ ]*$", "", name)
    name = re.sub(r"[-_ ]+$", "", name)
    return safe_name(name)


def pick_time_value_columns(df: pd.DataFrame) -> tuple[str, str]:
    time_cols = [c for c in df.columns if "时间" in str(c)]
    time_col = str(time_cols[0] if time_cols else df.columns[0])

    priority = ["全年小时标幺", "值", "MW", "负荷", "load", "Load"]
    for key in priority:
        for col in df.columns:
            if col == time_col:
                continue
            if key in str(col):
                return time_col, str(col)

    skip = {time_col, "年份", "行业", "参与平均用户数", "周序", "周开始", "周结束", "周内日序", "小时", "类型", "日类型"}
    candidates = [str(col) for col in df.columns if str(col) not in skip]
    if not candidates:
        raise ValueError(f"无法识别数值列, 当前列: {list(df.columns)}")
    return time_col, candidates[0]


def write_generator_input(src: Path, dst: Path) -> None:
    df = read_csv_auto(src)
    time_col, value_col = pick_time_value_columns(df)
    out = pd.DataFrame(
        {
            "时间": pd.to_datetime(df[time_col]).dt.strftime("%Y-%m-%d %H:%M:%S"),
            "值": pd.to_numeric(df[value_col], errors="coerce"),
        }
    ).dropna(subset=["时间", "值"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False, encoding="utf-8-sig")


def discover_direct_cases(input_dirs: list[Path], years: list[int], fail_on_missing: bool) -> list[Case]:
    cases: list[Case] = []
    for directory in input_dirs:
        files = {year: directory / f"历史负荷数据{year}.csv" for year in years}
        missing = [str(path) for path in files.values() if not path.exists()]
        if missing:
            message = f"direct 输入目录缺少文件: {directory} -> {missing}"
            if fail_on_missing:
                raise FileNotFoundError(message)
            print(f"[skip] {message}")
            continue
        cases.append(Case(name=safe_name(directory.name), files_by_year=files, source_type="direct"))
    return cases


def discover_year_subdir_cases(source_roots: list[Path], years: list[int], fail_on_missing: bool) -> list[Case]:
    cases: list[Case] = []
    for root in source_roots:
        per_year: dict[int, dict[str, Path]] = {}
        for year in years:
            year_dir = root / str(year)
            if not year_dir.exists():
                message = f"缺少年份目录: {year_dir}"
                if fail_on_missing:
                    raise FileNotFoundError(message)
                print(f"[skip-root] {message}")
                per_year = {}
                break
            mapping: dict[str, Path] = {}
            for path in sorted(year_dir.glob("*.csv")):
                mapping[normalize_case_name(path, year)] = path
            per_year[year] = mapping
        if not per_year:
            continue

        common = set.intersection(*(set(mapping) for mapping in per_year.values()))
        all_names = set.union(*(set(mapping) for mapping in per_year.values()))
        for name in sorted(all_names - common):
            message = f"{root}: {name} 未同时具备年份 {years}"
            if fail_on_missing:
                raise FileNotFoundError(message)
            print(f"[skip] {message}")

        for name in sorted(common):
            files = {year: per_year[year][name] for year in years}
            cases.append(Case(name=name, files_by_year=files, source_type=f"year-subdirs:{root}"))
    return cases


def load_generator(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("annual_curve_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载生成器: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_flat_curve(src_csv: Path, dst_csv: Path, target_year: int, industry: str) -> None:
    df = read_csv_auto(src_csv)
    if "全年小时标幺" not in df.columns:
        raise ValueError(f"{src_csv} 缺少 全年小时标幺 列")
    out = pd.DataFrame(
        {
            "时间": pd.to_datetime(df["时间"]).dt.strftime("%Y-%m-%d %H:%M:%S"),
            "年份": target_year,
            "行业": industry,
            "全年小时标幺": pd.to_numeric(df["全年小时标幺"], errors="coerce"),
        }
    )
    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst_csv, index=False, encoding="utf-8-sig")


def run_case(
    module: Any,
    case: Case,
    staging_root: Path,
    output_root: Path,
    flat_root: Path,
    target_year: int,
    history_years: list[int],
    typical_years: list[int],
    holidays: list[str],
    holiday_pattern: str,
    single_typical: bool,
    clean_staging: bool,
) -> dict[str, object]:
    case_dir = staging_root / case.name
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    needed_years = sorted(set(history_years + typical_years))
    for year in needed_years:
        write_generator_input(case.files_by_year[year], case_dir / f"历史负荷数据{year}.csv")

    out_dir = output_root / case.name
    result = module.generate(
        input_dir=case_dir,
        output_dir=out_dir,
        target_year=target_year,
        hist_years=history_years,
        typical_years=typical_years,
        holidays=holidays,
        pattern=holiday_pattern,
        single_typical=single_typical,
    )
    hourly_csv = Path(result["hourly_csv"])
    flat_csv = flat_root / f"{case.name}_{target_year}标幺8760.csv"
    write_flat_curve(hourly_csv, flat_csv, target_year, case.name)

    if clean_staging:
        shutil.rmtree(case_dir)

    return {
        "名称": case.name,
        "输入类型": case.source_type,
        "目标年": target_year,
        "历史样本年": ",".join(str(y) for y in history_years),
        "典型日样本年": ",".join(str(y) for y in typical_years),
        "假期": ",".join(holidays),
        "假期周模式": holiday_pattern,
        "单典型日": single_typical,
        "小时点数": result["hours"],
        "利用小时数": result["utilization_hours"],
        "输出目录": str(out_dir),
        "全年小时标幺CSV": result["hourly_csv"],
        "扁平输出CSV": str(flat_csv),
        "源文件": "；".join(f"{year}:{path}" for year, path in sorted(case.files_by_year.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-roots",
        nargs="*",
        type=Path,
        default=[DEFAULT_SOURCE_ROOT],
        help="year-subdirs 结构根目录, 如 行业标幺8760曲线",
    )
    parser.add_argument(
        "--input-dirs",
        nargs="*",
        type=Path,
        default=[],
        help="direct 结构目录, 每个目录内有 历史负荷数据YYYY.csv",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="批量输出根目录")
    parser.add_argument("--generator", type=Path, default=DEFAULT_GENERATOR, help="原始生成器脚本路径")
    parser.add_argument("--target-year", type=int, default=2030, help="目标年")
    parser.add_argument("--history-years", default="2022,2023", help="周峰/日峰层历史年")
    parser.add_argument("--typical-years", default="2022,2023", help="24小时典型日来源年")
    parser.add_argument("--holidays", default="spring,may,national", help="假期列表: spring,may,national")
    parser.add_argument(
        "--holiday-pattern",
        default="hybrid",
        choices=["pm1", "pm0", "p01", "core", "span", "hybrid"],
        help="假期周模式; hybrid=春节前当后、五一当后、十一当后",
    )
    parser.add_argument("--single-typical", action="store_true", help="每周单条典型日曲线, 不分工作日/双休日")
    parser.add_argument("--fail-on-missing", action="store_true", help="缺少年份/行业文件时报错, 默认跳过")
    parser.add_argument("--keep-staging", action="store_true", help="保留临时整理目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history_years = parse_years(args.history_years)
    typical_years = parse_years(args.typical_years)
    needed_years = sorted(set(history_years + typical_years))
    holidays = [part.strip() for part in args.holidays.split(",") if part.strip()]

    module = load_generator(args.generator)
    cases = []
    cases.extend(discover_year_subdir_cases(args.source_roots, needed_years, args.fail_on_missing))
    cases.extend(discover_direct_cases(args.input_dirs, needed_years, args.fail_on_missing))
    if not cases:
        raise RuntimeError("没有发现可生成的输入对象")

    args.output_root.mkdir(parents=True, exist_ok=True)
    staging_root = args.output_root / "_整理后的生成器输入"
    flat_root = args.output_root / f"{args.target_year}行业标幺8760曲线"
    rows = []
    print(f"发现 {len(cases)} 个对象, 开始生成 {args.target_year} 年 8760 标幺曲线...")
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case.name}")
        rows.append(
            run_case(
                module=module,
                case=case,
                staging_root=staging_root,
                output_root=args.output_root,
                flat_root=flat_root,
                target_year=args.target_year,
                history_years=history_years,
                typical_years=typical_years,
                holidays=holidays,
                holiday_pattern=args.holiday_pattern,
                single_typical=args.single_typical,
                clean_staging=not args.keep_staging,
            )
        )

    summary_path = args.output_root / "批量生成汇总.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"完成: {args.output_root}")
    print(f"汇总: {summary_path}")
    print(f"扁平行业曲线目录: {flat_root}")


if __name__ == "__main__":
    main()
