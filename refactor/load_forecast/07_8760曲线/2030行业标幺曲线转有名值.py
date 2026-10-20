# -*- coding: utf-8 -*-
"""把 2030 各行业 8760 标幺曲线定标为 MW 有名值曲线。

输入:
    ../2030行业标幺8760曲线/*_2030标幺8760.csv
    ../存档/03_数据/02_统计与电量锚点/山东2020-2025_全社会用电分类表（全口径）.xlsx

默认口径:
    2030 全社会总电量 = 10,159.7 亿kWh
    重点行业占比 = 山东表 2020-2023 年占全社会用电量占比线性外推到 2030
    扣除重点行业剩余 = 总电量 - 本副本保留的重点行业电量

输出:
    ../2030行业有名值曲线/{行业}_2030有名值MW.csv
    ../2030行业有名值曲线/2030行业电量占比与定标汇总.csv
    ../2030行业有名值曲线/2030总负荷_合成.csv

定标公式:
    P_MW(t) = P_pu(t) / max(P_pu) * E_yi * 1e5 / sum(P_pu / max(P_pu))
其中 E_yi 为年电量(亿kWh)。这样保证 sum(P_MW) / 1e5 == E_yi。
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
from project_paths import CLASSIFICATION_XLSX

DEFAULT_SHAPE_DIR = PROJECT_DIR / "2030行业8760标幺曲线_三假前当后周" / "2030行业标幺8760曲线"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "2030行业有名值曲线"
DEFAULT_ENERGY_XLSX = CLASSIFICATION_XLSX

TOTAL_E_YI = 10159.7
INDUSTRY_ROW_KEYS = {
    "居民生活": "B、城乡居民生活用电合计",
    "化学原料和化学制品制造业": "化学原料和化学制品制造业",
    "有色金属冶炼和压延加工业": "有色金属冶炼和压延加工业",
    "汽车制造业": "汽车制造业",
    "电气机械和器材制造业": "电气机械和器材制造业",
    "非金属矿物制品业": "非金属矿物制品业",
    "黑色金属冶炼和压延加工业": "黑色金属冶炼和压延加工业",
}

RESIDUAL_NAME = "扣除重点行业剩余"


def read_csv_auto(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"Cannot decode {path}")


def safe_name(text: str) -> str:
    return re.sub(r'[<>:"/\\|?*]+', "_", text)


def find_row(names: list[str], key: str) -> int:
    hits = [i for i, name in enumerate(names) if key in name]
    if not hits:
        raise KeyError(f"山东表中找不到行: {key}")
    return hits[0]


def linear_extrapolate_2030_share(values: np.ndarray) -> float:
    years = np.array([2020, 2021, 2022, 2023], dtype=float)
    if np.all(values == values[0]):
        return float(values[0])
    k, b = np.polyfit(years, values.astype(float), 1)
    return max(0.0, float(k * 2030 + b))


def read_2030_energy(total_e_yi: float, energy_xlsx: Path) -> pd.DataFrame:
    xl = pd.read_excel(energy_xlsx, header=None, skiprows=14).dropna(subset=[0])
    names = xl[0].astype(str).str.strip().tolist()
    share_mat = xl.iloc[:, 8:14].to_numpy(float)
    cols_2020_2023 = np.array([0, 1, 2, 3])

    rows: list[dict[str, float | str]] = []
    for industry, key in INDUSTRY_ROW_KEYS.items():
        idx = find_row(names, key)
        hist_share = share_mat[idx, cols_2020_2023]
        share = linear_extrapolate_2030_share(hist_share)
        rows.append(
            {
                "行业": industry,
                "2030占比": share,
                "2030电量亿kWh": share * total_e_yi,
                "电量口径": "2020-2023占比线性外推",
            }
        )

    known_energy = sum(float(row["2030电量亿kWh"]) for row in rows)
    residual = total_e_yi - known_energy
    if residual < 0:
        raise RuntimeError(f"重点行业电量合计超过总电量: {known_energy:.4f} > {total_e_yi:.4f}")
    rows.append(
        {
            "行业": RESIDUAL_NAME,
            "2030占比": residual / total_e_yi,
            "2030电量亿kWh": residual,
            "电量口径": "总电量-9个重点行业电量",
        }
    )
    return pd.DataFrame(rows)


def shape_file_for(shape_dir: Path, industry: str) -> Path:
    candidates = [
        shape_dir / f"{industry}_2030标幺8760.csv",
        shape_dir / f"{industry}（互联网数据服务）_2030标幺8760.csv",
    ]
    for path in candidates:
        if path.exists():
            return path

    matches = sorted(shape_dir.glob(f"{industry}*2030标幺8760.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"找不到 {industry} 的2030标幺曲线: {shape_dir}")


def load_shape(path: Path) -> pd.Series:
    df = read_csv_auto(path)
    time_col = "时间" if "时间" in df.columns else df.columns[0]
    value_col = "全年小时标幺" if "全年小时标幺" in df.columns else df.columns[-1]
    s = pd.Series(
        pd.to_numeric(df[value_col], errors="coerce").to_numpy(float),
        index=pd.DatetimeIndex(pd.to_datetime(df[time_col])),
        name=path.stem,
    ).sort_index()
    s = s[~s.index.duplicated()]
    expected = pd.date_range("2030-01-01 00:00:00", "2030-12-31 23:00:00", freq="h")
    s = s.reindex(expected)
    if s.isna().any():
        raise RuntimeError(f"{path}: 缺少 {int(s.isna().sum())} 个小时或存在非数值")
    return s.astype(float)


def scale_to_energy(shape: pd.Series, energy_yi: float) -> pd.Series:
    """将形状曲线定标到年电量，返回 MW；sum(MW)/1e5 应等于亿 kWh。"""
    pu = shape / float(shape.max())
    if float(pu.max()) <= 0 or float(pu.sum()) <= 0:
        raise RuntimeError(f"{shape.name}: 标幺曲线最大值或积分 <= 0")
    return pu * (energy_yi * 1e5 / float(pu.sum()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape-dir", type=Path, default=DEFAULT_SHAPE_DIR, help="2030标幺曲线目录")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--energy-xlsx", type=Path, default=DEFAULT_ENERGY_XLSX, help="山东全社会用电分类表")
    parser.add_argument("--total-energy", type=float, default=TOTAL_E_YI, help="2030全社会总电量, 单位: 亿kWh")
    parser.add_argument("--fail-on-missing", action="store_true", help="缺少曲线时直接报错; 默认跳过并写入汇总")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    energy_df = read_2030_energy(args.total_energy, args.energy_xlsx)
    summary_rows: list[dict[str, object]] = []
    total_series: pd.Series | None = None

    for _, row in energy_df.iterrows():
        industry = str(row["行业"])
        energy_yi = float(row["2030电量亿kWh"])
        try:
            shape_path = shape_file_for(args.shape_dir, industry)
            shape = load_shape(shape_path)
            named = scale_to_energy(shape, energy_yi)
        except Exception as exc:
            if args.fail_on_missing:
                raise
            summary_rows.append(
                {
                    **row.to_dict(),
                    "状态": "跳过",
                    "标幺曲线文件": "",
                    "输出文件": "",
                    "年最大负荷MW": np.nan,
                    "年最小负荷MW": np.nan,
                    "年平均负荷MW": np.nan,
                    "校验电量亿kWh": np.nan,
                    "最大负荷利用小时h": np.nan,
                    "备注": str(exc),
                }
            )
            print(f"[skip] {industry}: {exc}")
            continue

        out_path = args.output_dir / f"{safe_name(industry)}_2030有名值MW.csv"
        pd.DataFrame(
            {
                "时间": named.index.strftime("%Y-%m-%d %H:%M:%S"),
                "年份": 2030,
                "行业": industry,
                "有名值MW": named.to_numpy(float),
            }
        ).to_csv(out_path, index=False, encoding="utf-8-sig")

        total_series = named if total_series is None else total_series.add(named, fill_value=0.0)
        summary_rows.append(
            {
                **row.to_dict(),
                "状态": "已输出",
                "标幺曲线文件": str(shape_path),
                "输出文件": str(out_path),
                "年最大负荷MW": float(named.max()),
                "年最小负荷MW": float(named.min()),
                "年平均负荷MW": float(named.mean()),
                "校验电量亿kWh": float(named.sum()) / 1e5,
                "最大负荷利用小时h": float(named.sum()) / float(named.max()),
                "备注": "",
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = args.output_dir / "2030行业电量占比与定标汇总.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    if total_series is not None:
        total_path = args.output_dir / "2030总负荷_合成.csv"
        pd.DataFrame(
            {
                "时间": total_series.index.strftime("%Y-%m-%d %H:%M:%S"),
                "年份": 2030,
                "有名值MW": total_series.to_numpy(float),
            }
        ).to_csv(total_path, index=False, encoding="utf-8-sig")
        print(f"总负荷合成: {total_path}")
        print(f"总电量校验: {total_series.sum()/1e5:.4f} 亿kWh")

    print(f"完成: {args.output_dir}")
    print(f"汇总: {summary_path}")


if __name__ == "__main__":
    main()
