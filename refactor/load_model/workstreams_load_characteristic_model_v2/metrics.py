"""全省负荷特性指标与季节×日型典型曲线。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar_utils import calendar_frame, classify_date, season_of
from .day_types import HOLIDAY, WEEKEND, WORKDAY
from .shape_transform import WINDOW_BOUNDS


WINDOWS = dict(zip(
    ("夜间", "上午", "午间", "晚间"),
    (range(start, stop) for start, stop in WINDOW_BOUNDS), strict=True,
))


def _validated_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "load_MW"}
    missing = required - set(hourly.columns)
    if missing:
        raise ValueError(f"逐时负荷缺少字段：{sorted(missing)}")
    frame = hourly.copy().sort_values("timestamp").reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame.timestamp)
    if frame.timestamp.duplicated().any() or not np.isfinite(frame.load_MW).all() or (frame.load_MW < 0).any():
        raise ValueError("逐时负荷含重复时间戳、空值或负值")
    expected = 8784 if pd.Timestamp(f"{frame.timestamp.dt.year.iloc[0]}-12-31").is_leap_year else 8760
    if frame.timestamp.dt.year.nunique() != 1 or len(frame) != expected:
        raise ValueError(f"逐时负荷应为单一完整年份{expected}小时，实际{len(frame)}")
    expected_index = pd.date_range(
        f"{int(frame.timestamp.dt.year.iloc[0])}-01-01",
        f"{int(frame.timestamp.dt.year.iloc[0])}-12-31 23:00",
        freq="h",
    )
    if not frame.timestamp.reset_index(drop=True).equals(pd.Series(expected_index, name="timestamp")):
        raise ValueError("逐时负荷时间戳必须是从元旦0时到年末23时的严格连续整点序列")
    frame["date"] = frame.timestamp.dt.normalize()
    frame["month"] = frame.timestamp.dt.month
    frame["hour"] = frame.timestamp.dt.hour
    return frame


def calculate_load_metrics(hourly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = _validated_hourly(hourly)
    daily = frame.groupby("date", as_index=False).agg(daily_energy_MWh=("load_MW", "sum"), daily_average_MW=("load_MW", "mean"), daily_max_MW=("load_MW", "max"), daily_min_MW=("load_MW", "min"))
    daily["daily_load_factor"] = daily.daily_average_MW / daily.daily_max_MW
    daily["month"] = daily.date.dt.month
    monthly_rows = []
    for month, part in frame.groupby("month", sort=True):
        peak_index = part.load_MW.idxmax()
        peak_timestamp = part.loc[peak_index, "timestamp"]
        peak_date = pd.Timestamp(peak_timestamp).normalize()
        peak_day = part.loc[part.date == peak_date]
        month_average = float(part.load_MW.mean())
        peak = float(part.load_MW.max())
        peak_day_average = float(peak_day.load_MW.mean())
        monthly_rows.append({"month": int(month), "monthly_energy_MWh": float(part.load_MW.sum()), "monthly_average_MW": month_average, "monthly_max_MW": peak, "monthly_peak_timestamp": peak_timestamp, "peak_day": peak_date, "peak_day_average_MW": peak_day_average, "peak_day_min_MW": float(peak_day.load_MW.min()), "peak_day_load_factor": peak_day_average / peak, "monthly_imbalance_rate": month_average / peak_day_average})
    monthly = pd.DataFrame(monthly_rows)
    annual_peak_index = frame.load_MW.idxmax()
    annual_energy = float(frame.load_MW.sum())
    annual_peak = float(frame.load_MW.max())
    year_hours = len(frame)
    annual = {"year": int(frame.timestamp.dt.year.iloc[0]), "annual_energy_MWh": annual_energy, "annual_max_MW": annual_peak, "annual_peak_timestamp": frame.loc[annual_peak_index, "timestamp"], "annual_min_MW": float(frame.load_MW.min()), "peak_valley_difference_MW": float(annual_peak - frame.load_MW.min()), "Tmax_h": annual_energy / annual_peak, "annual_load_factor": annual_energy / (annual_peak * year_hours), "year_hours": year_hours, "mean_daily_load_factor": float(daily.daily_load_factor.mean())}
    total_energy = annual_energy
    for label, hours in WINDOWS.items():
        annual[f"share_{label}"] = float(frame.loc[frame.hour.isin(list(hours)), "load_MW"].sum() / total_energy)
    return daily, monthly, pd.DataFrame([annual])


def season_daytype_typical_shapes(hourly: pd.DataFrame) -> pd.DataFrame:
    """按全部有效日的逐小时算术平均构造4季×2日型标幺曲线。

    每个单元先对同一小时的MW值做算术平均，再将24点除以其总和；因此
    ``normalized_load`` 在每个季节×日型内严格和为1。工作日和非工作日
    使用同一典型日定义，不对日峰值或日内总量另作加权。
    """

    frame = _validated_hourly(hourly)
    dates = frame[["date"]].drop_duplicates().sort_values("date").reset_index(drop=True)
    classifications = dates.date.map(classify_date)
    dates["day_group"] = classifications.map(lambda value: value[1])
    frame = frame.merge(dates, on="date", how="left", validate="many_to_one")
    frame["season"] = frame.month.map(season_of)
    expected_cells = {
        (season, day_group)
        for season in ("春季", "夏季", "秋季", "冬季")
        for day_group in ("工作日", "非工作日")
    }
    observed_cells = set(
        frame[["season", "day_group"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed_cells != expected_cells:
        raise ValueError(
            f"完整年份未覆盖全部4季×2日型：缺失{sorted(expected_cells - observed_cells)}"
        )

    result = (
        frame.groupby(["season", "day_group", "hour"], as_index=False, sort=False)
        .agg(
            typical_load_MW=("load_MW", "mean"),
            valid_days=("date", "nunique"),
        )
    )
    counts = result.groupby(["season", "day_group"], sort=False).hour.nunique()
    if not counts.eq(24).all():
        raise ValueError("至少一个季节×日型单元没有完整24小时，不能生成典型曲线")
    totals = result.groupby(["season", "day_group"]).typical_load_MW.transform("sum")
    if not np.isfinite(totals).all() or (totals <= 0).any():
        raise ValueError("至少一个季节×日型典型日的24点总量非正")
    result["normalized_load"] = result.typical_load_MW / totals
    closure = result.groupby(["season", "day_group"]).normalized_load.sum()
    if not np.isclose(closure.to_numpy(float), 1.0, rtol=0.0, atol=1e-12).all():
        raise AssertionError("季节×日型典型曲线归一化不闭合")
    season_order = {value: index for index, value in enumerate(("春季", "夏季", "秋季", "冬季"))}
    day_order = {value: index for index, value in enumerate(("工作日", "非工作日"))}
    result["_season_order"] = result.season.map(season_order)
    result["_day_order"] = result.day_group.map(day_order)
    return (
        result.sort_values(["_season_order", "_day_order", "hour"], kind="mergesort")
        .drop(columns=["_season_order", "_day_order"])
        .reset_index(drop=True)
    )


def historical_season_daytype_typical_shapes(
    hourly: pd.DataFrame,
    *,
    holiday_min_days: int = 8,
) -> pd.DataFrame:
    """构造历史全省4季×3规范日型的24点标幺典型日。

    输入必须是一个完整逐时年份。工作日由普通工作日和调休工作日组成；
    普通周末与法定节假日原则上分别提取。某季节法定节假日日期数少于
    ``holiday_min_days`` 时，先合并该季节全部普通周末和法定节假日样本，
    只计算一次典型曲线，再让“普通周末”和“法定节假日”两个规范标签
    引用完全相同的曲线。恰好达到门槛时不合并。

    ``sample_days`` 是对应规范标签自身的原始日期数；
    ``source_sample_days`` 是实际参与该曲线计算的日期数，因而合并时后者
    在周末和节假日两行相同。每个季节×规范日型的 ``normalized_load``
    24点严格和为1。
    """

    if int(holiday_min_days) < 1:
        raise ValueError("holiday_min_days必须为正整数")
    frame = _validated_hourly(hourly)
    year = int(frame.timestamp.dt.year.iloc[0])
    calendar = calendar_frame(year)[
        ["date", "season", "canonical_day_type"]
    ].copy()
    if calendar.date.duplicated().any():
        raise AssertionError("规范日历日期必须唯一")
    frame = frame.merge(calendar, on="date", how="left", validate="many_to_one")
    if frame[["season", "canonical_day_type"]].isna().any().any():
        raise ValueError("逐时年份存在无法映射到规范日历的日期")

    seasons = ("春季", "夏季", "秋季", "冬季")
    day_types = (WORKDAY, WEEKEND, HOLIDAY)
    rows: list[dict[str, object]] = []
    for season in seasons:
        season_part = frame.loc[frame.season.eq(season)].copy()
        raw_day_counts = (
            season_part.groupby("canonical_day_type", sort=False).date.nunique()
        )
        holiday_days = int(raw_day_counts.get(HOLIDAY, 0))
        merged = holiday_days < int(holiday_min_days)
        merge_reason = (
            "holiday_days_below_8"
            if merged
            else "holiday_days_at_least_8"
        )
        curve_cache: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray, int]] = {}

        for day_type in day_types:
            if day_type == WORKDAY:
                source_types = (WORKDAY,)
            elif merged:
                source_types = (WEEKEND, HOLIDAY)
            else:
                source_types = (day_type,)

            if source_types not in curve_cache:
                source = season_part.loc[
                    season_part.canonical_day_type.isin(source_types)
                ]
                source_sample_days = int(source.date.nunique())
                if source_sample_days == 0:
                    raise ValueError(
                        f"{year}/{season}/{day_type}没有可用于典型日的日期"
                    )
                curve = (
                    source.groupby("hour", sort=True).load_MW.mean().reindex(range(24))
                )
                if curve.isna().any():
                    raise ValueError(
                        f"{year}/{season}/{day_type}没有完整24小时，不能生成典型曲线"
                    )
                typical_load = curve.to_numpy(dtype=float, copy=True)
                total = float(typical_load.sum())
                if not np.isfinite(typical_load).all() or total <= 0:
                    raise ValueError(
                        f"{year}/{season}/{day_type}典型日含非有限值或24点总量非正"
                    )
                normalized = typical_load / total
                curve_cache[source_types] = (
                    typical_load,
                    normalized,
                    source_sample_days,
                )

            typical_load, normalized, source_sample_days = curve_cache[source_types]
            sample_days = int(raw_day_counts.get(day_type, 0))
            source_label = "|".join(source_types)
            for hour in range(24):
                rows.append(
                    {
                        "season": season,
                        "day_type": day_type,
                        "hour": hour,
                        "typical_load_MW": float(typical_load[hour]),
                        "normalized_load": float(normalized[hour]),
                        "sample_days": sample_days,
                        "source_sample_days": source_sample_days,
                        "valid_days": source_sample_days,
                        "source_day_types": source_label,
                        "holiday_days": holiday_days,
                        "holiday_min_days": int(holiday_min_days),
                        "holiday_merged_with_weekend": bool(merged),
                        "merge_reason": merge_reason,
                    }
                )

    result = pd.DataFrame(rows)
    expected_cells = {
        (season, day_type) for season in seasons for day_type in day_types
    }
    observed_cells = set(
        result[["season", "day_type"]].drop_duplicates().itertuples(
            index=False, name=None
        )
    )
    if observed_cells != expected_cells or len(result) != len(expected_cells) * 24:
        raise AssertionError("历史全省典型日必须完整覆盖4季×3规范日型×24小时")
    closure = result.groupby(["season", "day_type"]).normalized_load.sum()
    if not np.isclose(
        closure.to_numpy(dtype=float), 1.0, rtol=0.0, atol=1e-12
    ).all():
        raise AssertionError("历史全省4季×3规范日型曲线归一化不闭合")

    season_order = {value: index for index, value in enumerate(seasons)}
    day_order = {value: index for index, value in enumerate(day_types)}
    result["_season_order"] = result.season.map(season_order)
    result["_day_order"] = result.day_type.map(day_order)
    return (
        result.sort_values(
            ["_season_order", "_day_order", "hour"], kind="mergesort"
        )
        .drop(columns=["_season_order", "_day_order"])
        .reset_index(drop=True)
    )


def typical_workdays(hourly: pd.DataFrame) -> pd.DataFrame:
    """兼容既有工作日结果表，计算口径复用统一的季节×日型算术平均。"""

    result = season_daytype_typical_shapes(hourly)
    result = result.loc[result.day_group == "工作日"].copy()
    result = result.rename(columns={"normalized_load": "normalized_hour_share"})
    return result[["season", "hour", "typical_load_MW", "normalized_hour_share"]].reset_index(drop=True)
