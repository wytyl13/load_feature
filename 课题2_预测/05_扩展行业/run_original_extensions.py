# -*- coding: utf-8 -*-
"""用原始数据复跑充换电、数据中心和主预测链，并生成可审计的校验结果。

本脚本不会修改原始目录。所有结果写入本目录下的 ``验证结果_原始数据``，包括：

* 数据中心：原始 15 分钟 Excel -> 8760 小时 MW，并与存档 CSV 逐点比较；
* 充换电：调用原始周/日/小时生成器，分别复跑 2022+2023 和“仅 2023”两种口径；
* 主预测：调用整理版 ARIMA+MLP 源码。若当前环境只有旧版 statsmodels，使用兼容适配器，
  算法设定仍保持原脚本的 ADF 定阶、AIC 网格和 MLP(16,8)×5 种子集成。

默认数据来自 ``自己处理/存档/03_数据``，可用参数替换为其他数据副本。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
WORKSPACE = PROJECT.parent
ARCHIVE_DATA = WORKSPACE / "自己处理" / "存档" / "03_数据"
DEFAULT_HOURLY_ROOT = ARCHIVE_DATA / "03_实测负荷8760"
DEFAULT_DC_RAW_ROOT = ARCHIVE_DATA / "05_数据中心原始"
DEFAULT_GENERATOR_ROOT = ARCHIVE_DATA / "06_生成器输入目录"
DEFAULT_ENERGY_XLSX = ARCHIVE_DATA / "02_统计与电量锚点" / "111.xlsx"
DEFAULT_EXISTING_2030 = WORKSPACE / "自己处理" / "2030行业标幺8760曲线"
DEFAULT_ARCHIVE_SINGLE_2030 = WORKSPACE / "自己处理" / "存档" / "2030充换电服务业标幺曲线_仅2023_三假前当后周"
DEFAULT_OUT = HERE / "验证结果_原始数据"
INDUSTRY_8 = [
    "互联网和相关服务_MW.csv",
    "充换电服务业_MW.csv",
    "化学原料和化学制品制造业_MW.csv",
    "有色金属冶炼和压延加工业_MW.csv",
    "汽车制造业_MW.csv",
    "电气机械和器材制造业_MW.csv",
    "非金属矿物制品业_MW.csv",
    "黑色金属冶炼和压延加工业_MW.csv",
]


def read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, str(path))


def value_column(df: pd.DataFrame) -> str:
    for key in ("值", "全年小时标幺", "MW", "负荷"):
        for col in df.columns:
            if key in str(col):
                return str(col)
    candidates = [str(c) for c in df.columns if "时间" not in str(c)]
    if not candidates:
        raise ValueError(f"无法识别数值列: {list(df.columns)}")
    return candidates[0]


def load_hourly(path: Path, year: int) -> pd.Series:
    df = read_csv(path)
    time_col = next((str(c) for c in df.columns if "时间" in str(c)), str(df.columns[0]))
    idx = pd.DatetimeIndex(pd.to_datetime(df[time_col]))
    s = pd.Series(pd.to_numeric(df[value_column(df)], errors="coerce").to_numpy(float), index=idx)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    expected = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    return s.reindex(expected)


def series_summary(s: pd.Series, year: int, label: str) -> dict[str, object]:
    expected = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    valid = s.dropna()
    return {
        "年份": year,
        "对象": label,
        "点数": int(len(s)),
        "期望点数": int(len(expected)),
        "时间连续": bool(s.index.equals(expected)),
        "缺失小时数": int(s.isna().sum()),
        "零小时数": int((s.fillna(0) == 0).sum()),
        "最小MW": float(valid.min()) if len(valid) else None,
        "最大MW": float(valid.max()) if len(valid) else None,
        "平均MW": float(valid.mean()) if len(valid) else None,
        "电量亿kWh": float(valid.sum() / 1e5) if len(valid) else None,
    }


def load_month_total(path: Path, exclude: list[str]) -> pd.Series:
    df = pd.read_excel(path)
    if "时间" not in df.columns:
        raise ValueError(f"{path} 缺少 时间 列")
    ts = pd.to_datetime(df["时间"])
    num = df.drop(columns=["时间"]).replace("-", np.nan)
    num = num.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    drop_cols = [c for c in num.columns if any(key in str(c) for key in exclude)]
    num = num.drop(columns=drop_cols)
    s = pd.Series(num.sum(axis=1).to_numpy(float), index=ts)
    return s[~s.index.duplicated(keep="first")].sort_index()


def rebuild_data_center(raw_root: Path, hourly_root: Path, out_root: Path) -> list[dict[str, object]]:
    """重建 16 站数据中心小时曲线，并和存档 CSV 做逐点比较。"""
    exclude = ["玄铁站", "含光站", "飞天站"]
    rows: list[dict[str, object]] = []
    out_dc = out_root / "数据中心重建"
    out_dc.mkdir(parents=True, exist_ok=True)
    for year in (2022, 2023):
        files = sorted(raw_root.glob(f"{year}*.xlsx"))
        if len(files) != 12:
            raise FileNotFoundError(f"{year} 原始数据应为12个月，实际 {len(files)}: {raw_root}")
        hourly = pd.concat([load_month_total(path, exclude).resample("h", label="left", closed="left").mean() for path in files])
        hourly = hourly[~hourly.index.duplicated(keep="first")]
        idx = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
        hourly = hourly.reindex(idx)
        before_missing = int(hourly.isna().sum())
        filled = 0
        if before_missing:
            if year != 2023:
                raise ValueError(f"{year} 出现未预期缺测 {before_missing} 小时")
            jan = hourly[hourly.index.month == 1]
            avail = jan.dropna()
            profile = avail.groupby(avail.index.hour).mean()
            miss = jan[jan.isna()].index
            hourly.loc[miss] = profile.loc[miss.hour].to_numpy()
            filled = len(miss)
        hourly = hourly.round(4)
        dst = out_dc / str(year) / "数据中心_MW.csv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"时间": idx.strftime("%Y-%m-%d %H:%M:%S"), "值": hourly.to_numpy(float)}).to_csv(
            dst, index=False, encoding="utf-8-sig"
        )
        ref_path = hourly_root / str(year) / "数据中心_MW.csv"
        ref = load_hourly(ref_path, year)
        diff = hourly - ref
        valid = diff.dropna()
        rows.append({
            **series_summary(hourly, year, "数据中心_原始Excel重建"),
            "原始存档": str(ref_path),
            "原始缺测填充小时": filled,
            "存档缺失小时数": int(ref.isna().sum()),
            "逐点最大绝对差MW": float(valid.abs().max()) if len(valid) else None,
            "逐点平均绝对差MW": float(valid.abs().mean()) if len(valid) else None,
            "逐点相关系数": float(np.corrcoef(hourly.dropna(), ref.dropna())[0, 1]) if len(valid) else None,
            "重建文件": str(dst),
        })
    return rows


def validate_archived_hourly(hourly_root: Path) -> list[dict[str, object]]:
    """检查原始小时文件，并验证旧八行业剩余与新九行业剩余恒等式。"""
    rows: list[dict[str, object]] = []
    for year in (2022, 2023):
        year_dir = hourly_root / str(year)
        for filename in INDUSTRY_8 + ["数据中心_MW.csv", "总负荷去掉八个行业负荷.csv", "总负荷去掉九个行业负荷.csv"]:
            path = year_dir / filename
            if not path.exists():
                rows.append({"年份": year, "对象": filename, "状态": "缺失", "文件": str(path)})
                continue
            s = load_hourly(path, year)
            rows.append({**series_summary(s, year, filename), "状态": "已读取", "文件": str(path)})
        try:
            eight = sum((load_hourly(year_dir / fn, year) for fn in INDUSTRY_8), pd.Series(0.0, index=pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")))
            dc = load_hourly(year_dir / "数据中心_MW.csv", year)
            rest8 = load_hourly(year_dir / "总负荷去掉八个行业负荷.csv", year)
            rest9 = load_hourly(year_dir / "总负荷去掉九个行业负荷.csv", year)
            left = eight + rest8
            right = eight + dc + rest9
            delta = (left - right).dropna()
            rows.append({
                "年份": year,
                "对象": "八行业剩余+数据中心=九行业剩余恒等式",
                "状态": "通过" if float(delta.abs().max()) < 0.011 else "不通过",
                "逐点最大绝对差MW": float(delta.abs().max()),
                "逐点平均绝对差MW": float(delta.abs().mean()),
                "公式": "八行业合计+总负荷去掉八个 = 八行业合计+数据中心+总负荷去掉九个",
            })
        except FileNotFoundError as exc:
            rows.append({"年份": year, "对象": "八/九行业恒等式", "状态": "无法校验", "错误": str(exc)})
    return rows


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_generator(path: Path, name: str, zero_week_guard: bool = False):
    """加载原始生成器；充换电零值年份启用原工程已有的零周保护。"""
    if not zero_week_guard:
        return import_module(path, name)
    source = path.read_text(encoding="utf-8")
    old = "raw_week_norm[year][w_ - 1] / base_week_by_year[year][w_ - 1]"
    new = "(raw_week_norm[year][w_ - 1] / base_week_by_year[year][w_ - 1]) if base_week_by_year[year][w_ - 1] >= 0.05 else 1.0"
    if old not in source:
        raise RuntimeError(f"未找到原始生成器零周保护插入点: {path}")
    patched = source.replace(old, new)
    tmp = Path(tempfile.mkdtemp(prefix="original_generator_")) / path.name
    tmp.write_text(patched, encoding="utf-8")
    return import_module(tmp, name)


def stage_generator_input(src_dir: Path, dst_dir: Path, years: tuple[int, ...]) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for year in years:
        candidates = [src_dir / f"历史负荷数据{year}.csv", src_dir / f"数据中心-历史负荷数据{year}.csv"]
        source = next((path for path in candidates if path.exists()), None)
        if source is None:
            raise FileNotFoundError(f"{src_dir} 缺少 {year} 输入")
        shutil.copy2(source, dst_dir / f"历史负荷数据{year}.csv")


def read_curve(path: Path) -> np.ndarray:
    df = read_csv(path)
    return pd.to_numeric(df[value_column(df)], errors="coerce").to_numpy(float)


def curve_stats(values: np.ndarray, label: str) -> dict[str, object]:
    s = pd.Series(values, index=pd.date_range("2030-01-01", periods=len(values), freq="h"))
    # pandas 1.x 使用 M，pandas 2.x 的 ME 也表示月末；这里用跨版本写法。
    monthly = s.resample("M").mean()
    return {
        "对象": label,
        "点数": int(len(values)),
        "最大标幺": float(np.nanmax(values)),
        "最小标幺": float(np.nanmin(values)),
        "利用小时": float(np.nansum(values) / np.nanmax(values)),
        "峰值月份": int(s.idxmax().month),
        "11月均值": float(monthly.iloc[10]),
        "12月均值": float(monthly.iloc[11]),
        "零点数": int(np.sum(values == 0)),
    }


def generate_extension_curves(
    generator_root: Path,
    out_root: Path,
    existing_root: Path,
    archive_single_root: Path,
) -> list[dict[str, object]]:
    generator_path = HERE / "generate_annual_normalized_curve_v2_多假_工作双休_原始.py"
    curve_out = out_root / "2030标幺8760曲线"
    curve_out.mkdir(parents=True, exist_ok=True)
    specs = [("充换电服务业", "充换电服务业", "充换电服务业_2030标幺8760.csv"),
             ("数据中心", "数据中心", "数据中心_2030标幺8760.csv")]
    rows: list[dict[str, object]] = []
    for label, input_name, reference_name in specs:
        generator = import_generator(
            generator_path,
            f"original_hourly_generator_{input_name}",
            zero_week_guard=(input_name == "充换电服务业"),
        )
        source_dir = generator_root / input_name
        staged = out_root / "生成器输入" / input_name
        stage_generator_input(source_dir, staged, (2022, 2023))
        runs = [
            ("2022+2023_hybrid", [2022, 2023], [2022, 2023], "hybrid"),
            ("仅2023_hybrid", [2023], [2023], "hybrid"),
            ("仅2023_pm1", [2023], [2023], "pm1"),
        ]
        for tag, history, typical, holiday_pattern in runs:
            dest = curve_out / f"{label}_{tag}"
            result = generator.generate(
                input_dir=staged,
                output_dir=dest,
                target_year=2030,
                hist_years=history,
                typical_years=typical,
                holidays=["spring", "may", "national"],
                pattern=holiday_pattern,
                single_typical=True,
            )
            values = read_curve(Path(result["hourly_csv"]))
            row = curve_stats(values, f"{label}_{tag}")
            row.update({"生成文件": result["hourly_csv"], "训练年": ",".join(map(str, history)), "典型年": ",".join(map(str, typical)), "节假日模式": holiday_pattern})
            if tag == "仅2023_pm1":
                # 存档中的这个目录正是原始“仅2023”复跑结果，可用于判断生成器和输入是否命中。
                ref_path = archive_single_root / input_name / "2030年全年小时标幺曲线.csv"
            else:
                # 最终 2030 扁平目录采用后期用户级融合口径，仅作参考对照，不能视为同一训练口径。
                ref_path = existing_root / reference_name
            if ref_path.exists():
                ref = read_curve(ref_path)
                n = min(len(values), len(ref))
                row["与存档2030逐点最大差"] = float(np.abs(values[:n] - ref[:n]).max())
                row["与存档2030相关系数"] = float(np.corrcoef(values[:n], ref[:n])[0, 1])
                row["存档2030文件"] = str(ref_path)
            else:
                row["存档2030文件"] = "不存在，无法比对"
            rows.append(row)
    return rows


def run_arima_ml_compat(energy_xlsx: Path, out_root: Path) -> dict[str, object]:
    """运行原整理版 forecast_arima_ml.py；适配 statsmodels 0.10 的旧 ARIMA API。"""
    source = PROJECT / "01_电量预测" / "forecast_arima_ml.py"
    text = source.read_text(encoding="utf-8")
    old_import = "from statsmodels.tsa.arima.model import ARIMA"
    fallback = '''try:\n    from statsmodels.tsa.arima.model import ARIMA\nexcept ImportError:\n    from statsmodels.tsa.arima_model import ARIMA as _LegacyARIMA\n    class ARIMA:\n        def __init__(self, endog, order):\n            self._model = _LegacyARIMA(endog, order=order)\n        def fit(self, method_kwargs=None):\n            return self._model.fit(disp=0)'''
    if old_import not in text:
        raise RuntimeError("forecast_arima_ml.py 的 ARIMA 导入行已变化，未自动改写")
    text = text.replace(old_import, fallback)
    src_line = next(line for line in text.splitlines() if line.startswith("SRC = "))
    out_line = next(line for line in text.splitlines() if line.startswith("OUT = "))
    text = text.replace(src_line, f"SRC = r'{energy_xlsx}'")
    out_path = out_root / "ARIMA预测2026-2030_原始数据.xlsx"
    text = text.replace(out_line, f"OUT = r'{out_path}'")
    namespace = {"__file__": str(source), "__name__": "__main__"}
    try:
        exec(compile(text, str(source), "exec"), namespace)
    except Exception as exc:  # 保留失败原因到报告，不掩盖数据链问题
        return {"状态": "失败", "错误类型": type(exc).__name__, "错误": str(exc), "源脚本": str(source)}
    return {"状态": "成功", "输出": str(out_path), "源脚本": str(source)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hourly-root", type=Path, default=DEFAULT_HOURLY_ROOT)
    parser.add_argument("--dc-raw-root", type=Path, default=DEFAULT_DC_RAW_ROOT)
    parser.add_argument("--generator-root", type=Path, default=DEFAULT_GENERATOR_ROOT)
    parser.add_argument("--energy-xlsx", type=Path, default=DEFAULT_ENERGY_XLSX)
    parser.add_argument("--existing-2030-root", type=Path, default=DEFAULT_EXISTING_2030)
    parser.add_argument("--archive-single-2030-root", type=Path, default=DEFAULT_ARCHIVE_SINGLE_2030)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-arima", action="store_true")
    args = parser.parse_args()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)

    dc_rows = rebuild_data_center(args.dc_raw_root.resolve(), args.hourly_root.resolve(), out)
    archive_rows = validate_archived_hourly(args.hourly_root.resolve())
    curve_rows = generate_extension_curves(
        args.generator_root.resolve(),
        out,
        args.existing_2030_root.resolve(),
        args.archive_single_2030_root.resolve(),
    )
    status_path = out / "主预测运行状态.json"
    if args.skip_arima and status_path.exists():
        arima = json.loads(status_path.read_text(encoding="utf-8"))
    else:
        arima = {"状态": "跳过"} if args.skip_arima else run_arima_ml_compat(args.energy_xlsx.resolve(), out)

    summary = pd.DataFrame(archive_rows + dc_rows + curve_rows)
    summary.to_csv(out / "扩展行业与原始结果校验汇总.csv", index=False, encoding="utf-8-sig")
    (out / "主预测运行状态.json").write_text(json.dumps(arima, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成。结果目录: {out}")
    print(f"校验汇总: {out / '扩展行业与原始结果校验汇总.csv'}")
    print(f"主预测状态: {out / '主预测运行状态.json'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
