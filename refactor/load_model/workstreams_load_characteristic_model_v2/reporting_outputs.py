"""Pure reporting builders for normalized load-curve deliverables.

The model calculates energy and load metrics from absolute MW internally.  The
functions in this module deliberately expose only normalized curve ordinates,
plus scalar normalization bases needed for audit or reconstruction.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .calendar_utils import calendar_frame
from .result_tables import validate_run_manifest, validate_tables_manifest_contract


SEASONS = ("春季", "夏季", "秋季", "冬季")
FINAL_DAY_TYPES = ("工作日", "周末", "节假日", "周末及节假日（合并）")
HOLIDAY_MIN_DAYS = 8
NORMALIZATION_ATOL = 1e-12
REPORT_METHOD_STATEMENTS = (
    "行业形状采用用户日归一化后、同行业同日有效用户等权平均。",
    "2023行业medoid是未来形状唯一基准；2022只用于历史比较。",
    "行业未来形状始终先完成四时段份额调整，各时段内采用同比例迁移。",
    "日负荷率大于0.80且变异系数小于0.15的2023基准曲线判为平坦。",
    "只有非平坦且2022—2023峰谷差率绝对变化量位于6.5%—25%闭区间时，才执行峰谷差率第二阶段。",
    "门控未通过时，最终曲线等于四时段调整后的中间曲线。",
    "门控通过时，峰谷差率为最终优先目标，最终四时段份额只统计、不再恢复。",
    "晚峰与控制时段仍只作诊断，全省合成曲线不追加峰谷差率校正。",
    "2023全省季节—日型系数统一应用于全部25个组成。",
    "曲线电量为扣除厂用电、线损和抽蓄用电后的自然负荷口径。",
    "日内形状仍使用未逐时扣除上述三项的调度代理。",
)


def validate_reporting_contract(
    run_manifest: Mapping[str, object],
    tables_manifest: Mapping[str, object],
    *,
    require_parquet: bool = True,
) -> None:
    """Hard-gate report inputs against the final 3.5.0 method contract."""

    validate_run_manifest(
        run_manifest, require_success=True, require_parquet=require_parquet
    )
    validate_tables_manifest_contract(tables_manifest)


def load_reporting_contract(
    output_dir: str | Path,
    *,
    require_parquet: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read and validate both manifests before a report writes any artifact."""

    root = Path(output_dir).resolve()
    run_path = root / "run_metadata.json"
    tables_path = root / "数据表" / "tables_manifest.json"
    if not run_path.is_file():
        raise FileNotFoundError(run_path)
    if not tables_path.is_file():
        raise FileNotFoundError(tables_path)
    run_manifest = json.loads(run_path.read_text(encoding="utf-8-sig"))
    tables_manifest = json.loads(tables_path.read_text(encoding="utf-8-sig"))
    if not isinstance(run_manifest, dict) or not isinstance(tables_manifest, dict):
        raise RuntimeError("报告输入的运行清单与结果表清单必须是JSON对象")
    validate_reporting_contract(
        run_manifest, tables_manifest, require_parquet=require_parquet
    )
    return run_manifest, tables_manifest


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{label}缺少字段：{sorted(missing)}")


def _validated_hourly_long(
    hourly: pd.DataFrame,
    *,
    timestamp_col: str,
    year_col: str,
    scenario_col: str,
    trajectory_col: str,
    value_col: str,
    require_complete_years: bool,
) -> pd.DataFrame:
    context = [year_col, scenario_col, trajectory_col]
    required = [timestamp_col, *context, value_col]
    _require_columns(hourly, required, "完整逐时轨迹")
    result = hourly[required].copy()
    result[timestamp_col] = pd.to_datetime(result[timestamp_col], errors="coerce")
    if result[timestamp_col].isna().any():
        raise ValueError("完整逐时轨迹含无效时间戳")

    years = pd.to_numeric(result[year_col], errors="coerce")
    if (
        years.isna().any()
        or not np.isfinite(years.to_numpy(float)).all()
        or not np.equal(years, np.floor(years)).all()
    ):
        raise ValueError("完整逐时轨迹的year必须为整数")
    result[year_col] = years.astype(int)
    if not result[timestamp_col].dt.year.eq(result[year_col]).all():
        raise ValueError("完整逐时轨迹的year与时间戳年份不一致")

    values = pd.to_numeric(result[value_col], errors="coerce")
    if (
        values.isna().any()
        or not np.isfinite(values.to_numpy(float)).all()
        or (values < 0).any()
    ):
        raise ValueError("完整逐时轨迹的value必须为有限非负数")
    result[value_col] = values.astype(float)
    if result[scenario_col].isna().any() or result[trajectory_col].isna().any():
        raise ValueError("完整逐时轨迹的scenario/trajectory不能为空")
    result[scenario_col] = result[scenario_col].astype(str)
    result[trajectory_col] = result[trajectory_col].astype(str)

    unique_key = [*context, timestamp_col]
    if result.duplicated(unique_key).any():
        raise ValueError("完整逐时轨迹的year/scenario/trajectory/timestamp必须唯一")
    result = result.sort_values(unique_key, kind="mergesort").reset_index(drop=True)

    if require_complete_years:
        for keys, part in result.groupby(context, sort=False, observed=True):
            year = int(keys[0])
            expected = pd.date_range(
                f"{year}-01-01 00:00:00",
                f"{year}-12-31 23:00:00",
                freq="h",
            )
            actual = pd.DatetimeIndex(part[timestamp_col])
            if len(actual) != len(expected) or not actual.equals(expected):
                raise ValueError(
                    "完整逐时轨迹必须覆盖严格连续的完整年份："
                    f"{dict(zip(context, keys))}"
                )
    return result


def _calendar_dimension(years: Sequence[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    canonical = {
        "普通工作日": "工作日",
        "调休工作日": "工作日",
        "普通周末": "周末",
        "法定节假日": "节假日",
    }
    for year in years:
        frame = calendar_frame(int(year))[["date", "year", "season", "day_type"]].copy()
        frame["canonical_day_type"] = frame.day_type.map(canonical)
        if frame.canonical_day_type.isna().any():
            unknown = sorted(frame.loc[frame.canonical_day_type.isna(), "day_type"].unique())
            raise ValueError(f"日历含未知日型：{unknown}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_adaptive_typical_days(
    hourly: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    year_col: str = "year",
    scenario_col: str = "planning_scenario",
    trajectory_col: str = "trajectory",
    value_col: str = "value",
    holiday_min_days: int = HOLIDAY_MIN_DAYS,
) -> pd.DataFrame:
    """Build four-season adaptive typical-day curves from complete trajectories.

    Workdays always remain independent.  In each future year and season,
    weekends and statutory holidays are pooled before averaging when the
    number of distinct holiday dates is below ``holiday_min_days``.  Every
    returned 24-hour curve is energy-share normalized and therefore sums to
    one.
    """

    if int(holiday_min_days) <= 0:
        raise ValueError("holiday_min_days必须为正整数")
    frame = _validated_hourly_long(
        hourly,
        timestamp_col=timestamp_col,
        year_col=year_col,
        scenario_col=scenario_col,
        trajectory_col=trajectory_col,
        value_col=value_col,
        require_complete_years=True,
    )
    context = [year_col, scenario_col, trajectory_col]
    frame["date"] = frame[timestamp_col].dt.normalize()
    frame["hour"] = frame[timestamp_col].dt.hour
    calendar = _calendar_dimension(tuple(sorted(frame[year_col].unique())))
    calendar = calendar.rename(columns={"year": year_col})
    frame = frame.merge(
        calendar[[year_col, "date", "season", "canonical_day_type"]],
        on=[year_col, "date"],
        how="left",
        validate="many_to_one",
    )
    if frame[["season", "canonical_day_type"]].isna().any().any():
        raise ValueError("逐时轨迹存在无法映射到规范日历的日期")

    date_dimension = frame[
        [*context, "date", "season", "canonical_day_type"]
    ].drop_duplicates()
    if date_dimension.duplicated([*context, "date"]).any():
        raise AssertionError("同一轨迹日期出现多个日历分类")
    date_dimension["_is_holiday"] = date_dimension.canonical_day_type.eq("节假日")
    date_dimension["holiday_sample_days"] = (
        date_dimension.groupby([*context, "season"], observed=True)["_is_holiday"]
        .transform("sum")
        .astype(int)
    )
    date_dimension["holiday_merged_with_weekend"] = (
        date_dimension.holiday_sample_days < int(holiday_min_days)
    )
    merged_nonwork = (
        date_dimension.holiday_merged_with_weekend
        & date_dimension.canonical_day_type.isin(["周末", "节假日"])
    )
    date_dimension["final_day_type"] = date_dimension.canonical_day_type
    date_dimension.loc[merged_nonwork, "final_day_type"] = "周末及节假日（合并）"
    date_dimension["merge_reason"] = np.where(
        date_dimension.holiday_merged_with_weekend,
        date_dimension.holiday_sample_days.map(
            lambda value: (
                f"该年该季节法定节假日有效天数{int(value)}天，少于"
                f"{int(holiday_min_days)}天，与周末合并"
            )
        ),
        date_dimension.holiday_sample_days.map(
            lambda value: (
                f"该年该季节法定节假日有效天数{int(value)}天，达到"
                f"{int(holiday_min_days)}天门槛，周末与节假日分开"
            )
        ),
    )
    frame = frame.drop(columns=["season", "canonical_day_type"]).merge(
        date_dimension.drop(columns="_is_holiday"),
        on=[*context, "date"],
        how="left",
        validate="many_to_one",
    )

    cell = [*context, "season", "final_day_type"]
    result = (
        frame.groupby([*cell, "hour"], as_index=False, sort=False, observed=True)
        .agg(
            _typical_raw=(value_col, "mean"),
            sample_days=("date", "nunique"),
            holiday_sample_days=("holiday_sample_days", "first"),
            holiday_merged_with_weekend=("holiday_merged_with_weekend", "first"),
            merge_reason=("merge_reason", "first"),
        )
    )
    hours_per_cell = result.groupby(cell, observed=True).hour.nunique()
    if not hours_per_cell.eq(24).all():
        raise AssertionError("至少一个典型日单元未覆盖完整24小时")
    raw_totals = result.groupby(cell, observed=True)._typical_raw.transform("sum")
    if (
        not np.isfinite(raw_totals.to_numpy(float)).all()
        or (raw_totals <= 0).any()
    ):
        raise ValueError("至少一个典型日单元的24点总量非正")
    result["typical_day_pu"] = result._typical_raw / raw_totals
    result = result.drop(columns="_typical_raw")
    closure = result.groupby(cell, observed=True).typical_day_pu.sum()
    if not np.isclose(
        closure.to_numpy(float), 1.0, rtol=0.0, atol=NORMALIZATION_ATOL
    ).all():
        raise AssertionError("典型日标幺曲线24点和未闭合到1")

    season_order = {value: index for index, value in enumerate(SEASONS)}
    day_order = {value: index for index, value in enumerate(FINAL_DAY_TYPES)}
    result["_season_order"] = result.season.map(season_order)
    result["_day_order"] = result.final_day_type.map(day_order)
    if result[["_season_order", "_day_order"]].isna().any().any():
        raise AssertionError("典型日结果含未知季节或日型")
    result["normalization_method"] = "24_hour_sum_equals_1"
    ordered = [
        *context,
        "season",
        "final_day_type",
        "hour",
        "typical_day_pu",
        "sample_days",
        "holiday_sample_days",
        "holiday_merged_with_weekend",
        "merge_reason",
        "normalization_method",
    ]
    return (
        result.sort_values(
            [*context, "_season_order", "_day_order", "hour"], kind="mergesort"
        )[ordered]
        .reset_index(drop=True)
    )


def build_typical_day_load_factors(
    typical_days: pd.DataFrame,
    *,
    year_col: str = "year",
    scenario_col: str = "planning_scenario",
    trajectory_col: str = "trajectory",
    value_col: str = "typical_day_pu",
) -> pd.DataFrame:
    """Calculate each normalized typical day's load factor as mean / maximum."""

    cell = [year_col, scenario_col, trajectory_col, "season", "final_day_type"]
    required = [
        *cell,
        "hour",
        value_col,
        "sample_days",
        "holiday_sample_days",
        "holiday_merged_with_weekend",
        "merge_reason",
    ]
    _require_columns(typical_days, required, "典型日标幺曲线")
    frame = typical_days[required].copy()
    values = pd.to_numeric(frame[value_col], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(float)).all() or (values < 0).any():
        raise ValueError("典型日标幺值必须为有限非负数")
    frame[value_col] = values.astype(float)
    if frame.duplicated([*cell, "hour"]).any():
        raise ValueError("典型日曲线单元小时重复")

    rows: list[dict[str, object]] = []
    for keys, part in frame.groupby(cell, sort=False, observed=True):
        hours = set(pd.to_numeric(part.hour, errors="coerce").tolist())
        if hours != set(range(24)) or len(part) != 24:
            raise ValueError(f"典型日曲线必须恰有0..23时：{dict(zip(cell, keys))}")
        curve = part.sort_values("hour")[value_col].to_numpy(float)
        if not np.isclose(curve.sum(), 1.0, rtol=0.0, atol=NORMALIZATION_ATOL):
            raise ValueError(f"典型日曲线24点和必须为1：{dict(zip(cell, keys))}")
        maximum = float(curve.max())
        if maximum <= 0:
            raise ValueError("典型日最大标幺值必须为正")
        first = part.iloc[0]
        rows.append(
            {
                **dict(zip(cell, keys)),
                "sample_days": int(first.sample_days),
                "holiday_sample_days": int(first.holiday_sample_days),
                "holiday_merged_with_weekend": bool(
                    first.holiday_merged_with_weekend
                ),
                "merge_reason": str(first.merge_reason),
                "typical_day_mean_pu": float(curve.mean()),
                "typical_day_max_pu": maximum,
                "typical_day_load_factor": float(curve.mean() / maximum),
                "metric_definition": "mean(typical_day_pu)/max(typical_day_pu)",
            }
        )
    return pd.DataFrame.from_records(rows)


def normalize_hourly_trajectories(
    hourly: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    year_col: str = "year",
    scenario_col: str = "planning_scenario",
    trajectory_col: str = "trajectory",
    value_col: str = "value",
) -> pd.DataFrame:
    """Normalize each annual hourly trajectory by its own annual maximum."""

    frame = _validated_hourly_long(
        hourly,
        timestamp_col=timestamp_col,
        year_col=year_col,
        scenario_col=scenario_col,
        trajectory_col=trajectory_col,
        value_col=value_col,
        require_complete_years=False,
    )
    context = [year_col, scenario_col, trajectory_col]
    annual_peak = frame.groupby(context, observed=True)[value_col].transform("max")
    if (annual_peak <= 0).any():
        raise ValueError("年度逐时轨迹最大值必须为正")
    result = frame.drop(columns=value_col)
    result["load_pu"] = frame[value_col].to_numpy(float) / annual_peak.to_numpy(float)
    result["annual_peak_base_value"] = annual_peak.to_numpy(float)
    result["normalization_method"] = "annual_max_equals_1"
    maxima = result.groupby(context, observed=True).load_pu.max()
    if not np.isclose(maxima.to_numpy(float), 1.0, rtol=0.0, atol=NORMALIZATION_ATOL).all():
        raise AssertionError("年度逐时标幺曲线最大值未闭合到1")
    return result


def build_monthly_maximum_curves(
    hourly: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    year_col: str = "year",
    scenario_col: str = "planning_scenario",
    trajectory_col: str = "trajectory",
    value_col: str = "value",
) -> pd.DataFrame:
    """Build 12-point monthly-maximum curves normalized by annual maximum."""

    frame = _validated_hourly_long(
        hourly,
        timestamp_col=timestamp_col,
        year_col=year_col,
        scenario_col=scenario_col,
        trajectory_col=trajectory_col,
        value_col=value_col,
        require_complete_years=False,
    )
    context = [year_col, scenario_col, trajectory_col]
    frame["month"] = frame[timestamp_col].dt.month
    frame = frame.sort_values([*context, "month", timestamp_col], kind="mergesort")
    peak_indices = frame.groupby([*context, "month"], observed=True)[value_col].idxmax()
    monthly = frame.loc[
        peak_indices,
        [*context, "month", timestamp_col, value_col],
    ].copy()
    monthly = monthly.rename(columns={timestamp_col: "monthly_peak_timestamp"})
    annual_peak = (
        frame.groupby(context, as_index=False, observed=True)[value_col]
        .max()
        .rename(columns={value_col: "annual_peak_base_value"})
    )
    monthly = monthly.merge(annual_peak, on=context, how="left", validate="many_to_one")
    if (monthly.annual_peak_base_value <= 0).any():
        raise ValueError("月最大负荷曲线的年度最大值必须为正")
    monthly["monthly_max_pu"] = (
        monthly[value_col] / monthly.annual_peak_base_value
    )
    monthly = monthly.drop(columns=value_col)
    monthly["normalization_method"] = "monthly_max_over_same_trajectory_annual_max"
    for keys, part in monthly.groupby(context, sort=False, observed=True):
        if set(part.month.astype(int)) != set(range(1, 13)) or len(part) != 12:
            raise ValueError(
                "每条年度月最大负荷曲线必须恰有1..12月："
                f"{dict(zip(context, keys))}"
            )
        if not np.isclose(
            float(part.monthly_max_pu.max()), 1.0, rtol=0.0, atol=NORMALIZATION_ATOL
        ):
            raise AssertionError("12个月月最大负荷标幺曲线最大值未闭合到1")
    return monthly.sort_values([*context, "month"], kind="mergesort").reset_index(drop=True)


def extract_baseline_p50_workday_display(
    typical_days: pd.DataFrame,
    *,
    start_year: int = 2025,
    end_year: int = 2045,
    scenario: str = "基准",
    trajectory: str = "P50_real",
    year_col: str = "year",
    scenario_col: str = "planning_scenario",
    trajectory_col: str = "trajectory",
    value_col: str = "typical_day_pu",
) -> pd.DataFrame:
    """Extract the exact 2025--2045 baseline/P50/workday display contract."""

    required = [
        year_col,
        scenario_col,
        trajectory_col,
        "season",
        "final_day_type",
        "hour",
        value_col,
    ]
    _require_columns(typical_days, required, "典型日展示源表")
    if int(end_year) < int(start_year):
        raise ValueError("展示结束年份不得早于开始年份")
    selected = typical_days.loc[
        pd.to_numeric(typical_days[year_col], errors="coerce").between(
            int(start_year), int(end_year)
        )
        & typical_days[scenario_col].astype(str).eq(str(scenario))
        & typical_days[trajectory_col].astype(str).eq(str(trajectory))
        & typical_days.final_day_type.astype(str).eq("工作日")
    ].copy()
    key = [year_col, "season", "hour"]
    if selected.duplicated(key).any():
        raise ValueError("展示工作日曲线的year/season/hour必须唯一")
    expected = pd.MultiIndex.from_product(
        [range(int(start_year), int(end_year) + 1), SEASONS, range(24)],
        names=key,
    )
    actual = pd.MultiIndex.from_frame(selected[key])
    missing = expected.difference(actual)
    extra = actual.difference(expected)
    expected_rows = len(expected)
    if len(selected) != expected_rows or len(missing) or len(extra):
        raise ValueError(
            "展示工作日曲线覆盖不完整："
            f"expected_rows={expected_rows}, actual_rows={len(selected)}, "
            f"missing={list(missing[:5])}, extra={list(extra[:5])}"
        )
    closure = selected.groupby([year_col, "season"], observed=True)[value_col].sum()
    if not np.isclose(
        closure.to_numpy(float), 1.0, rtol=0.0, atol=NORMALIZATION_ATOL
    ).all():
        raise ValueError("展示工作日典型日曲线24点和必须为1")
    season_order = {value: index for index, value in enumerate(SEASONS)}
    selected["_season_order"] = selected.season.map(season_order)
    if selected._season_order.isna().any():
        raise ValueError("展示工作日曲线含未知季节")
    result = selected.sort_values(
        [year_col, "_season_order", "hour"], kind="mergesort"
    ).drop(columns="_season_order")
    if int(start_year) == 2025 and int(end_year) == 2045 and len(result) != 2016:
        raise AssertionError("2025—2045基准P50四季工作日展示长表必须恰有2016行")
    return result.reset_index(drop=True)


def select_m1_baseline_p50_annual_scale(
    annual_energy: pd.DataFrame,
    *,
    years: Iterable[int] | None = None,
    method_col: str = "method",
    scenario_col: str = "scenario",
    year_col: str = "year",
    p50_natural_col: str = "P50_natural_MWh",
) -> pd.DataFrame:
    """Select the M1 baseline annual scale from upstream P50 natural energy only.

    There is intentionally no quantile argument: P10/P90 fields are never
    eligible to supply the baseline annual scale.
    """

    _require_columns(
        annual_energy,
        [method_col, scenario_col, year_col, p50_natural_col],
        "上游年度电量",
    )
    frame = annual_energy.loc[
        annual_energy[method_col].astype(str).eq("M1")
        & annual_energy[scenario_col].astype(str).eq("基准"),
        [method_col, scenario_col, year_col, p50_natural_col],
    ].copy()
    requested = (
        None
        if years is None
        else tuple(sorted({int(value) for value in years}))
    )
    numeric_years = pd.to_numeric(frame[year_col], errors="coerce")
    if numeric_years.isna().any() or not np.equal(numeric_years, np.floor(numeric_years)).all():
        raise ValueError("M1基准年度电量的year必须为整数")
    frame[year_col] = numeric_years.astype(int)
    if requested is not None:
        frame = frame.loc[frame[year_col].isin(requested)].copy()
        missing = sorted(set(requested).difference(frame[year_col]))
        if missing:
            raise ValueError(f"M1基准P50自然负荷年度规模缺少年份：{missing}")
    if frame.empty:
        raise ValueError("上游年度电量没有M1基准情景P50自然负荷规模")
    if frame.duplicated(year_col).any():
        raise ValueError("M1基准P50自然负荷年度规模必须按year唯一")
    values = pd.to_numeric(frame[p50_natural_col], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(float)).all() or (values <= 0).any():
        raise ValueError("M1基准P50自然负荷年度规模必须为有限正数")
    result = pd.DataFrame(
        {
            "method": "M1",
            "planning_scenario": "基准",
            "year": frame[year_col].to_numpy(int),
            "annual_scale_MWh": values.to_numpy(float),
            "annual_energy_quantile": "P50",
            "annual_energy_basis": "P50_natural_total",
        }
    )
    return result.sort_values("year", kind="mergesort").reset_index(drop=True)


__all__ = [
    "FINAL_DAY_TYPES",
    "HOLIDAY_MIN_DAYS",
    "REPORT_METHOD_STATEMENTS",
    "SEASONS",
    "build_adaptive_typical_days",
    "build_monthly_maximum_curves",
    "build_typical_day_load_factors",
    "extract_baseline_p50_workday_display",
    "load_reporting_contract",
    "normalize_hourly_trajectories",
    "select_m1_baseline_p50_annual_scale",
    "validate_reporting_contract",
]
