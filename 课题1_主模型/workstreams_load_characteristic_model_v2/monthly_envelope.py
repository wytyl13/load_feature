"""全省月度P50包络与历史日电量残差库。"""

from __future__ import annotations

import calendar

import numpy as np
import pandas as pd

from .calendar_utils import classify_date, season_of
from .quality import HOUR_COLS


PROVINCE_DAY_TYPE_FACTOR_SOURCE = "province_system_load_2023"
PROVINCE_DAY_TYPE_FACTOR_SCOPE = "province_shared_all_25_components"
PROVINCE_HOLIDAY_MIN_DAYS = 8


def _canonical_factor_day_type(value: object) -> str:
    text = str(value).strip()
    if text in {"工作日", "普通工作日", "调休工作日"}:
        return "工作日"
    if text in {"普通周末", "周末"}:
        return "普通周末"
    if text in {"法定节假日", "节假日"}:
        return "法定节假日"
    raise ValueError(f"存在未定义的全省日型：{text!r}")


def _complete_system_days(system_daily: pd.DataFrame) -> pd.DataFrame:
    missing = {"date", *HOUR_COLS} - set(system_daily.columns)
    if missing:
        raise ValueError(f"全省历史负荷缺少字段：{sorted(missing)}")
    frame = system_daily.copy()
    frame["date"] = pd.to_datetime(frame.date, errors="coerce").dt.normalize()
    if frame.date.isna().any():
        raise ValueError("全省历史负荷日期必须有效")
    values = frame[HOUR_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    valid = np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1) & (values.sum(axis=1) > 0)
    frame = frame.loc[valid].copy()
    frame["daily_energy_MWh"] = frame[HOUR_COLS].sum(axis=1)
    frame["year"] = frame.date.dt.year
    frame["month"] = frame.date.dt.month
    frame["season"] = frame.month.map(season_of)
    classified = frame.date.map(classify_date)
    frame["day_type"] = classified.map(lambda value: value[0])
    frame["day_group"] = classified.map(lambda value: value[1])
    return frame


def estimate_province_day_type_factors(
    system_history: pd.DataFrame,
    source_year: int = 2023,
) -> pd.DataFrame:
    """由全省系统负荷识别一套季节—日型共享倍率。

    倍率只使用 ``source_year`` 的完整日。工作日固定为1；若某季节
    法定节假日少于8天，则将周末与节假日合并计算。这套倍率不含
    ``industry`` 维度，必须统一用于24行业与修正组成。
    """

    source_year = int(source_year)
    if source_year != 2023:
        raise ValueError("正式全省共享日型倍率的source_year固定为2023")
    frame = _complete_system_days(system_history)
    frame = frame.loc[frame.year.eq(source_year)].copy()
    if frame.empty:
        raise ValueError(f"全省历史负荷缺少{source_year}年完整日")
    if frame.date.duplicated().any():
        duplicated = frame.loc[frame.date.duplicated(keep=False), "date"].min()
        raise ValueError(f"全省{source_year}年完整日日期重复：{duplicated:%Y-%m-%d}")
    frame["final_day_type"] = frame.day_type.map(_canonical_factor_day_type)

    rows: list[dict[str, object]] = []
    season_order = {value: index for index, value in enumerate(("春季", "夏季", "秋季", "冬季"))}
    for season, part in frame.groupby("season", sort=False):
        workday = part.loc[
            part.final_day_type.eq("工作日"), "daily_energy_MWh"
        ]
        if workday.empty:
            raise ValueError(f"{source_year}年全省{season}工作日基线缺失")
        workday_median = float(workday.median())
        if not np.isfinite(workday_median) or workday_median <= 0:
            raise ValueError(f"{source_year}年全省{season}工作日基线必须为正")

        holiday_count = int(part.final_day_type.eq("法定节假日").sum())
        merged = holiday_count < PROVINCE_HOLIDAY_MIN_DAYS
        base = {
            "season": str(season),
            "source_year": source_year,
            "factor_source": PROVINCE_DAY_TYPE_FACTOR_SOURCE,
            "factor_scope": PROVINCE_DAY_TYPE_FACTOR_SCOPE,
            "historical_shape_scope": "dispatch_proxy_not_hourly_deducted",
            "holiday_min_days": PROVINCE_HOLIDAY_MIN_DAYS,
            "holiday_valid_days": holiday_count,
            "holiday_merged_with_weekend": merged,
            "workday_median_energy_MWh": workday_median,
        }
        rows.append(
            {
                **base,
                "final_day_type": "工作日",
                "factor": 1.0,
                "sample_days": int(len(workday)),
                "median_daily_energy_MWh": workday_median,
                "fallback_status": "none",
            }
        )

        if merged:
            requested = ("普通周末", "法定节假日")
            nonwork = part.loc[
                part.final_day_type.isin(requested), "daily_energy_MWh"
            ]
            fallback = nonwork.empty
            nonwork_median = workday_median if fallback else float(nonwork.median())
            rows.append(
                {
                    **base,
                    "final_day_type": "合并非工作日",
                    "factor": nonwork_median / workday_median,
                    "sample_days": int(len(nonwork)),
                    "median_daily_energy_MWh": nonwork_median,
                    "fallback_status": (
                        "insufficient_days_fallback_1" if fallback else "none"
                    ),
                }
            )
        else:
            for day_type in ("普通周末", "法定节假日"):
                sample = part.loc[
                    part.final_day_type.eq(day_type), "daily_energy_MWh"
                ]
                fallback = sample.empty
                sample_median = workday_median if fallback else float(sample.median())
                rows.append(
                    {
                        **base,
                        "final_day_type": day_type,
                        "factor": sample_median / workday_median,
                        "sample_days": int(len(sample)),
                        "median_daily_energy_MWh": sample_median,
                        "fallback_status": (
                            "insufficient_days_fallback_1" if fallback else "none"
                        ),
                    }
                )

    result = pd.DataFrame.from_records(rows)
    if result.empty or (result.factor <= 0).any() or not np.isfinite(result.factor).all():
        raise ValueError("全省共享日型倍率必须为有限正数")
    result["_season_order"] = result.season.map(season_order)
    result["_day_order"] = result.final_day_type.map(
        {"工作日": 0, "普通周末": 1, "法定节假日": 2, "合并非工作日": 1}
    )
    return (
        result.sort_values(["_season_order", "_day_order"], kind="mergesort")
        .drop(columns=["_season_order", "_day_order"])
        .reset_index(drop=True)
    )


def build_monthly_envelope(system_daily: pd.DataFrame, coverage_threshold: float = 0.90) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _complete_system_days(system_daily)
    rows = []
    for (year, month), part in frame.groupby(["year", "month"]):
        expected_hours = calendar.monthrange(int(year), int(month))[1] * 24
        coverage = len(part) * 24 / expected_hours
        annual = frame.loc[frame.year == year]
        annual_mean = float(annual.daily_energy_MWh.sum() / (len(annual) * 24))
        month_mean = float(part.daily_energy_MWh.sum() / (len(part) * 24))
        rows.append({"history_year": int(year), "month": int(month), "season": season_of(int(month)), "valid_hours": len(part) * 24, "expected_hours": expected_hours, "coverage": coverage, "factor": month_mean / annual_mean if coverage >= coverage_threshold and annual_mean > 0 else np.nan})
    history = pd.DataFrame(rows)
    factors = []
    for month in range(1, 13):
        values = history.loc[(history.month == month) & history.factor.notna(), "factor"]
        if len(values) >= 3:
            factor = float(values.median())
            source = "同历月有效历史中位数"
            valid_years = len(values)
        else:
            season = season_of(month)
            fallback = history.loc[(history.season == season) & history.factor.notna(), "factor"]
            if fallback.empty:
                raise ValueError(f"月份{month}没有有效月度包络历史或季节回退")
            factor = float(fallback.median())
            source = "同季节回退"
            valid_years = len(values)
        factors.append({"month": month, "season": season_of(month), "factor_p50": factor, "valid_same_month_years": valid_years, "source": source})
    return history, pd.DataFrame(factors)


def target_month_shares(year: int, envelope_factors: pd.DataFrame) -> pd.DataFrame:
    lookup = envelope_factors.set_index("month").factor_p50
    rows = []
    for month in range(1, 13):
        hours = calendar.monthrange(int(year), month)[1] * 24
        rows.append({"year": int(year), "month": month, "hours": hours, "factor_p50": float(lookup.loc[month])})
    result = pd.DataFrame(rows)
    result["raw_weight"] = result.hours * result.factor_p50
    result["month_share"] = result.raw_weight / result.raw_weight.sum()
    if abs(float(result.month_share.sum()) - 1.0) > 1e-15:
        raise AssertionError("目标月电量份额未归一")
    return result


def build_interday_residual_library(system_daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _complete_system_days(system_daily)
    frame["residual_day_class"] = frame.day_type.map(
        {
            "普通工作日": "工作日",
            "调休工作日": "工作日",
            "普通周末": "普通周末",
            "法定节假日": "法定节假日",
        }
    )
    if frame.residual_day_class.isna().any():
        unknown = sorted(frame.loc[frame.residual_day_class.isna(), "day_type"].astype(str).unique())
        raise ValueError(f"存在未定义的历史日类型：{unknown}")
    baseline_rows = []
    for (year, month), month_part in frame.groupby(["year", "month"]):
        month_median = float(month_part.daily_energy_MWh.median())
        class_stats = month_part.groupby("residual_day_class").daily_energy_MWh.agg(["size", "median"])
        nonwork = month_part.loc[month_part.residual_day_class.isin(["普通周末", "法定节假日"]), "daily_energy_MWh"]
        for _, row in month_part.iterrows():
            day_class = str(row.residual_day_class)
            if day_class in class_stats.index and int(class_stats.loc[day_class, "size"]) >= 2:
                baseline = float(class_stats.loc[day_class, "median"])
                baseline_source = f"同年月-{day_class}中位数"
            elif day_class in {"普通周末", "法定节假日"} and len(nonwork) >= 2:
                baseline = float(nonwork.median())
                baseline_source = "同年月-非工作日合并中位数（样本不足回退）"
            else:
                baseline = month_median
                baseline_source = "同年月-全日中位数（样本不足回退）"
            baseline_rows.append(
                {
                    "date": row.date,
                    "history_year": int(year),
                    "month": int(month),
                    "day_type": row.day_type,
                    "day_group": row.day_group,
                    "residual_day_class": day_class,
                    "observed_daily_energy_MWh": float(row.daily_energy_MWh),
                    "baseline_daily_energy_MWh": baseline,
                    "residual_baseline_source": baseline_source,
                    "residual_scope": "历史年月内工作日/普通周末/法定节假日稳健基线",
                }
            )
    residuals = pd.DataFrame(baseline_rows).sort_values("date")
    residuals["log_ratio"] = np.log(residuals.observed_daily_energy_MWh / residuals.baseline_daily_energy_MWh)
    residuals["residual"] = residuals.log_ratio - residuals.groupby(["history_year", "month"]).log_ratio.transform("mean")
    residuals["baseline_fallback"] = residuals.residual_baseline_source.str.contains("回退")
    diagnostics = residuals.groupby(["history_year", "month"], as_index=False).agg(days=("date", "size"), fallback_days=("baseline_fallback", "sum"), residual_mean=("residual", "mean"), residual_std=("residual", "std"), min_residual=("residual", "min"), max_residual=("residual", "max"))
    return residuals.drop(columns="log_ratio"), diagnostics
