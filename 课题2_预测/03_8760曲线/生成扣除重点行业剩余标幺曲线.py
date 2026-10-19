# -*- coding: utf-8 -*-
"""生成“总负荷扣除重点行业后”的年度 8760 标幺曲线。

计算逻辑:
1. 读取每年总负荷曲线: 历史负荷数据{year}.csv;
2. 读取每个重点行业的年度标幺 8760 曲线;
3. 从山东全社会用电分类表读取该行业当年用电量;
4. 行业有名值(MW) = 行业标幺曲线 * 行业年电量(亿kWh) * 1e5 / 标幺曲线积分;
5. 剩余负荷 = 总负荷 - sum(重点行业有名值);
6. 剩余标幺 = 剩余负荷 / 剩余负荷年最大值。

默认输入:
    ../历史负荷数据2022.csv, ../历史负荷数据2023.csv
    ../行业标幺8760曲线/{year}/{行业}_标幺8760.csv
    ../存档/03_数据/02_统计与电量锚点/山东2020-2025_全社会用电分类表（全口径）.xlsx

默认输出:
    ../扣除重点行业剩余标幺曲线/{year}_扣除重点行业剩余_标幺8760.csv
    ../扣除重点行业剩余标幺曲线/扣除重点行业剩余计算汇总.csv
"""

from __future__ import annotations

import argparse
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DEFAULT_TOTAL_ROOT = PROJECT_DIR
DEFAULT_SHAPE_ROOT = PROJECT_DIR / "行业标幺8760曲线"
DEFAULT_ENERGY_XLSX = (
    PROJECT_DIR
    / "存档"
    / "03_数据"
    / "02_统计与电量锚点"
    / "山东2020-2025_全社会用电分类表（全口径）.xlsx"
)
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "扣除重点行业剩余标幺曲线"


INDUSTRY_ROWS = {
    "居民生活": "B、城乡居民生活用电合计",
    "化学原料和化学制品制造业": "14.化学原料和化学制品制造业",
    "有色金属冶炼和压延加工业": "20.有色金属冶炼和压延加工业",
    "汽车制造业": "24.汽车制造业",
    "电气机械和器材制造业": "26.电气机械和器材制造业",
    "非金属矿物制品业": "18.非金属矿物制品业",
    "黑色金属冶炼和压延加工业": "19.黑色金属冶炼和压延加工业",
}


@dataclass
class IndustryResult:
    year: int
    industry: str
    status: str
    energy_yikwh: float | None = None
    shape_file: str = ""
    pu_sum: float | None = None
    named_peak_mw: float | None = None
    named_energy_yikwh: float | None = None
    note: str = ""


def read_csv_auto(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"Cannot decode {path}")


def time_index(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 23:00:00", freq="h")


def pick_time_value_columns(df: pd.DataFrame) -> tuple[str, str]:
    time_candidates = [c for c in df.columns if "时间" in str(c)]
    time_col = time_candidates[0] if time_candidates else df.columns[0]

    value_candidates = [
        c
        for c in df.columns
        if c != time_col and ("全年小时标幺" in str(c) or str(c) in ("值", "负荷", "MW"))
    ]
    value_col = value_candidates[0] if value_candidates else [c for c in df.columns if c != time_col][0]
    return str(time_col), str(value_col)


def load_series(path: Path, year: int, value_col_hint: str | None = None) -> pd.Series:
    df = read_csv_auto(path)
    time_col, value_col = pick_time_value_columns(df)
    if value_col_hint and value_col_hint in df.columns:
        value_col = value_col_hint

    s = pd.Series(
        pd.to_numeric(df[value_col], errors="coerce").to_numpy(float),
        index=pd.DatetimeIndex(pd.to_datetime(df[time_col])),
        name=value_col,
    )
    s = s[~s.index.duplicated()].sort_index()
    s = s.reindex(time_index(year))
    if len(s) != 8760:
        raise RuntimeError(f"{path}: reindex 后不是 8760 点")
    if s.isna().any():
        missing = int(s.isna().sum())
        raise RuntimeError(f"{path}: 缺少 {missing} 个小时")
    return s.astype(float)


def read_energy_table(path: Path) -> tuple[pd.DataFrame, list[str]]:
    xl = pd.read_excel(path, header=None, skiprows=14).dropna(subset=[0])
    names = xl[0].astype(str).str.strip().tolist()
    return xl, names


def find_row(names: list[str], key: str) -> int:
    hits = [i for i, name in enumerate(names) if key in name]
    if not hits:
        raise KeyError(f"电量表中找不到行: {key}")
    return hits[0]


def energy_yikwh(xl: pd.DataFrame, names: list[str], row_key: str, year: int) -> float:
    if year < 2020 or year > 2025:
        raise ValueError("当前电量表只支持 2020-2025 年")
    col = 1 + (year - 2020)
    return float(xl.iloc[find_row(names, row_key), col]) / 1e4


def shape_path(shape_root: Path, year: int, industry: str) -> Path:
    return shape_root / str(year) / f"{industry}_标幺8760.csv"


def scale_shape_to_energy(shape: pd.Series, energy: float) -> pd.Series:
    """先把行业曲线峰值归一为 1，再按行业年电量定标为 MW。"""
    pu = shape.astype(float)
    max_value = float(pu.max())
    if max_value <= 0:
        raise RuntimeError(f"{shape.name or 'shape'}: 标幺曲线最大值 <= 0")
    pu = pu / max_value

    pu_sum = float(pu.sum())
    if pu_sum <= 0:
        raise RuntimeError(f"{shape.name or 'shape'}: 标幺曲线积分 <= 0")
    return pu * (energy * 1e5 / pu_sum)


def build_one_year(
    year: int,
    total_root: Path,
    shape_root: Path,
    xl: pd.DataFrame,
    names: list[str],
    output_root: Path,
    missing_industry: str,
    scale_total_to_excel: bool,
    clip_negative: bool,
) -> list[IndustryResult]:
    total_path = total_root / f"历史负荷数据{year}.csv"
    total = load_series(total_path, year)
    total_raw_energy = float(total.sum()) / 1e5

    table_total_energy = energy_yikwh(xl, names, "全社会用电总计", year)
    if scale_total_to_excel:
        total = total * (table_total_energy * 1e5 / float(total.sum()))

    industry_sum = pd.Series(0.0, index=total.index, name="重点行业合计MW")
    results: list[IndustryResult] = []

    for industry, row_key in INDUSTRY_ROWS.items():
        energy = energy_yikwh(xl, names, row_key, year)
        path = shape_path(shape_root, year, industry)
        if not path.exists():
            note = f"缺少行业标幺曲线: {path}"
            results.append(IndustryResult(year, industry, "缺少曲线", energy_yikwh=energy, note=note))
            if missing_industry == "fail":
                raise FileNotFoundError(note)
            continue

        shape = load_series(path, year, value_col_hint="全年小时标幺")
        named = scale_shape_to_energy(shape, energy)
        industry_sum = industry_sum.add(named, fill_value=0.0)
        results.append(
            IndustryResult(
                year=year,
                industry=industry,
                status="已扣除",
                energy_yikwh=energy,
                shape_file=str(path.relative_to(PROJECT_DIR)),
                pu_sum=float((shape / shape.max()).sum()),
                named_peak_mw=float(named.max()),
                named_energy_yikwh=float(named.sum()) / 1e5,
            )
        )

    residual = total - industry_sum
    negative_hours = int((residual < 0).sum())
    min_before_clip = float(residual.min())
    if clip_negative:
        residual = residual.clip(lower=0.0)

    peak = float(residual.max())
    if peak <= 0:
        raise RuntimeError(f"{year}: 剩余负荷最大值 <= 0, 无法标幺")
    residual_pu = residual / peak

    output_root.mkdir(parents=True, exist_ok=True)
    out_path = output_root / f"{year}_扣除重点行业剩余_标幺8760.csv"
    pd.DataFrame(
        {
            "时间": residual_pu.index.strftime("%Y-%m-%d %H:%M:%S"),
            "年份": year,
            "全年小时标幺": residual_pu.to_numpy(float),
        }
    ).to_csv(out_path, index=False, encoding="utf-8-sig")

    results.append(
        IndustryResult(
            year=year,
            industry="扣除重点行业剩余",
            status="输出",
            energy_yikwh=float(residual.sum()) / 1e5,
            shape_file=str(out_path.relative_to(PROJECT_DIR)),
            pu_sum=float(residual_pu.sum()),
            named_peak_mw=peak,
            named_energy_yikwh=float(residual.sum()) / 1e5,
            note=(
                f"总负荷原始电量={total_raw_energy:.4f}亿kWh;"
                f"山东表全社会={table_total_energy:.4f}亿kWh;"
                f"总负荷是否缩放={scale_total_to_excel};"
                f"负值小时数={negative_hours};"
                f"裁剪前最小值={min_before_clip:.4f}MW;"
                f"是否裁剪负值={clip_negative}"
            ),
        )
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[2022, 2023], help="要处理的年份")
    parser.add_argument("--total-root", type=Path, default=DEFAULT_TOTAL_ROOT, help="总负荷 CSV 所在目录")
    parser.add_argument("--shape-root", type=Path, default=DEFAULT_SHAPE_ROOT, help="行业标幺曲线根目录")
    parser.add_argument("--energy-xlsx", type=Path, default=DEFAULT_ENERGY_XLSX, help="行业电量 Excel")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="输出目录")
    parser.add_argument(
        "--missing-industry",
        choices=["skip", "fail"],
        default="skip",
        help="缺少某行业标幺曲线时跳过还是报错",
    )
    parser.add_argument(
        "--scale-total-to-excel",
        action="store_true",
        help="将总负荷曲线按形状缩放到山东表当年全社会用电量后再扣除",
    )
    parser.add_argument(
        "--clip-negative",
        action="store_true",
        help="剩余负荷出现负值时裁剪为 0 后再标幺",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xl, names = read_energy_table(args.energy_xlsx)

    all_results: list[IndustryResult] = []
    for year in args.years:
        print(f"开始处理 {year} ...")
        year_results = build_one_year(
            year=year,
            total_root=args.total_root,
            shape_root=args.shape_root,
            xl=xl,
            names=names,
            output_root=args.output_root,
            missing_industry=args.missing_industry,
            scale_total_to_excel=args.scale_total_to_excel,
            clip_negative=args.clip_negative,
        )
        all_results.extend(year_results)

    summary = pd.DataFrame([r.__dict__ for r in all_results])
    summary_path = args.output_root / "扣除重点行业剩余计算汇总.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"完成，输出目录: {args.output_root}")
    print(f"汇总文件: {summary_path}")


if __name__ == "__main__":
    main()
