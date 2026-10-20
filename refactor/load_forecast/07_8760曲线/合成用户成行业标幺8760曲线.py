# -*- coding: utf-8 -*-
"""合成用户级负荷为行业标幺 8760 曲线。

口径参考 build_new_8760_leapfix.py::load_industry_shape:
1. 同一行业内, 每个用户先按本用户有效小时年最大值标幺化;
2. 用户整日缺测或整日零负荷时, 该用户该日不参与行业平均;
3. 同一小时对有效用户做等权平均, 全部用户均缺时记 0;
4. 本副本只处理用户级行业，不包含后续扩展行业的单曲线分支。

默认输入:
    ../补全后分行业数据/{2022,2023}/*.csv
默认输出:
    ../行业标幺8760曲线/{year}/{行业}_标幺8760.csv
    ../行业标幺8760曲线/行业标幺8760合成汇总.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
import sys
sys.path.insert(0, str(PROJECT_DIR))
from project_paths import USER_CURVE_ROOT
DEFAULT_INPUT_ROOT = USER_CURVE_ROOT
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "行业标幺8760曲线"


INDUSTRY_PREFIX = {
    "居民生活": "居民生活-市中-用户",
    "化学原料和化学制品制造业": "第二产业-化学原料和化学制品制造业-用户",
    "有色金属冶炼和压延加工业": "第二产业-有色金属冶炼和压延加工业-用户",
    "汽车制造业": "第二产业-汽车制造业-用户",
    "电气机械和器材制造业": "第二产业-电气机械和器材制造业-用户",
    "非金属矿物制品业": "第二产业-非金属矿物制品业-用户",
    "黑色金属冶炼和压延加工业": "第二产业-黑色金属冶炼和压延加工业-用户",
}


@dataclass
class UserShape:
    name: str
    values: list[float | None]
    valid_hours: int
    excluded_zero_days: int
    year_max: float


def read_csv_auto(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    for encoding in ("utf-8-sig", "gbk", "utf-8"):
        try:
            with path.open("r", encoding=encoding, newline="") as file:
                reader = csv.DictReader(file)
                return list(reader), reader.fieldnames or []
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"Cannot decode {path}")


def find_columns(fields: list[str]) -> tuple[str, str]:
    time_col = "时间" if "时间" in fields else fields[0]
    value_col = "值" if "值" in fields else fields[1]
    return time_col, value_col


def full_hours(year: int) -> list[datetime]:
    start = datetime(year, 1, 1)
    return [start + timedelta(hours=i) for i in range(8760)]


def parse_time(value: str) -> datetime | None:
    value = (value or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return None


def parse_float(value: str) -> float | None:
    text = (value or "").strip().replace(",", "")
    if text in ("", "-", "NA", "N/A", "nan", "NaN", "null", "None"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_user_shape(path: Path, year: int, hours: list[datetime]) -> UserShape | None:
    """读取单个用户，剔除整日零负荷或缺测，再按该用户年最大值标幺化。"""
    rows, fields = read_csv_auto(path)
    time_col, value_col = find_columns(fields)
    raw: dict[datetime, float] = {}
    for row in rows:
        dt = parse_time(row.get(time_col, ""))
        value = parse_float(row.get(value_col, ""))
        if dt is None or value is None:
            continue
        raw[dt] = value

    values: list[float | None] = [raw.get(dt) for dt in hours]

    excluded_zero_days = 0
    for day_index in range(365):
        start = day_index * 24
        stop = start + 24
        day_values = [value for value in values[start:stop] if value is not None]
        if not day_values or max(day_values) <= 0:
            excluded_zero_days += 1
            for pos in range(start, stop):
                values[pos] = None

    valid_values = [value for value in values if value is not None]
    year_max = max(valid_values) if valid_values else 0.0
    if year_max <= 0:
        return None

    shape = [None if value is None else value / year_max for value in values]
    return UserShape(
        name=path.name,
        values=shape,
        valid_hours=sum(value is not None for value in shape),
        excluded_zero_days=excluded_zero_days,
        year_max=year_max,
    )


def fuse_user_industry(input_dir: Path, industry: str, prefix: str, year: int) -> tuple[list[dict[str, object]], dict] | None:
    """把同一行业用户曲线按小时等权平均，缺测用户不进入该小时分母。"""
    hours = full_hours(year)
    files = sorted(input_dir.glob(f"{prefix}*.csv"))
    if not files:
        return None

    shapes: list[UserShape] = []
    for path in files:
        shape = load_user_shape(path, year, hours)
        if shape is not None:
            shapes.append(shape)

    if not shapes:
        raise RuntimeError(f"{year} {industry}: all user files are invalid")

    # 先逐用户标幺，再横向平均，避免大用户的绝对容量支配行业形状。
    fused: list[float] = []
    valid_user_count: list[int] = []
    for pos in range(len(hours)):
        valid = [shape.values[pos] for shape in shapes if shape.values[pos] is not None]
        valid_user_count.append(len(valid))
        fused.append(sum(valid) / len(valid) if valid else 0.0)

    out_rows = [
        {
            "时间": hours.strftime("%Y-%m-%d %H:%M:%S"),
            "年份": year,
            "行业": industry,
            "全年小时标幺": value,
            "参与平均用户数": count,
        }
        for hours, value, count in zip(hours, fused, valid_user_count)
    ]

    summary = {
        "年份": year,
        "行业": industry,
        "输入文件数": len(files),
        "有效用户数": len(shapes),
        "用户文件": "、".join(s.name for s in shapes),
        "各用户有效小时": "、".join(f"{s.name}:{s.valid_hours}" for s in shapes),
        "各用户整日零负荷或缺测剔除天数": "、".join(f"{s.name}:{s.excluded_zero_days}" for s in shapes),
        "曲线最大值": max(fused),
        "曲线最小值": min(fused),
        "曲线平均值": sum(fused) / len(fused),
        "全用户缺失小时数": sum(count == 0 for count in valid_user_count),
    }
    return out_rows, summary


def fuse_single_curve(input_dir: Path, industry: str, pattern: str, year: int) -> tuple[list[dict[str, object]], dict] | None:
    hours = full_hours(year)
    files = sorted(input_dir.glob(pattern))
    if not files:
        return None
    if len(files) > 1:
        raise RuntimeError(f"{year} {industry}: matched multiple files: {files}")

    path = files[0]
    rows, fields = read_csv_auto(path)
    time_col, value_col = find_columns(fields)
    raw: dict[datetime, float] = {}
    for row in rows:
        dt = parse_time(row.get(time_col, ""))
        value = parse_float(row.get(value_col, ""))
        if dt is None or value is None:
            continue
        raw[dt] = value
    values: list[float | None] = [raw.get(dt) for dt in hours]
    valid = [value for value in values if value is not None]
    year_max = max(valid) if valid else 0.0
    if year_max <= 0:
        raise RuntimeError(f"{year} {industry}: max <= 0")
    fused = [0.0 if value is None else value / year_max for value in values]
    valid_count = [1 if value is not None else 0 for value in values]

    out_rows = [
        {
            "时间": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "年份": year,
            "行业": industry,
            "全年小时标幺": value,
            "参与平均用户数": count,
        }
        for dt, value, count in zip(hours, fused, valid_count)
    ]
    summary = {
        "年份": year,
        "行业": industry,
        "输入文件数": 1,
        "有效用户数": 1,
        "用户文件": path.name,
        "各用户有效小时": f"{path.name}:{sum(value is not None for value in values)}",
        "各用户整日零负荷或缺测剔除天数": "",
        "曲线最大值": max(fused),
        "曲线最小值": min(fused),
        "曲线平均值": sum(fused) / len(fused),
        "全用户缺失小时数": sum(value is None for value in values),
    }
    return out_rows, summary


def write_curve(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["时间", "年份", "行业", "全年小时标幺", "参与平均用户数"]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = row.copy()
            out["全年小时标幺"] = f"{float(out['全年小时标幺']):.10f}"
            writer.writerow(out)


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "年份",
        "行业",
        "输入文件数",
        "有效用户数",
        "用户文件",
        "各用户有效小时",
        "各用户整日零负荷或缺测剔除天数",
        "曲线最大值",
        "曲线最小值",
        "曲线平均值",
        "全用户缺失小时数",
        "输出文件",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            formatted = row.copy()
            for key in ("曲线最大值", "曲线最小值", "曲线平均值"):
                formatted[key] = f"{float(formatted[key]):.10f}"
            writer.writerow(formatted)


def run(input_root: Path, output_root: Path, years: list[int]) -> None:
    summary_rows: list[dict] = []
    for year in years:
        input_dir = input_root / str(year)
        if not input_dir.exists():
            raise FileNotFoundError(input_dir)

        for industry, prefix in INDUSTRY_PREFIX.items():
            result = fuse_user_industry(input_dir, industry, prefix, year)
            if result is None:
                continue
            out_df, summary = result
            out_file = output_root / str(year) / f"{industry}_标幺8760.csv"
            write_curve(out_file, out_df)
            summary["输出文件"] = str(out_file)
            summary_rows.append(summary)
            print(f"{year} {industry}: {summary['有效用户数']} users -> {out_file}")

    write_summary(output_root / "行业标幺8760合成汇总.csv", summary_rows)
    print(f"summary -> {output_root / '行业标幺8760合成汇总.csv'}")


def parse_years(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="合成用户级 CSV 为行业标幺 8760 曲线")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--years", default="2022,2023")
    args = parser.parse_args()
    run(args.input_root, args.output_root, parse_years(args.years))


if __name__ == "__main__":
    main()
