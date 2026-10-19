"""快速验证用峰值月份诊断与代表轨迹选择。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .representative_paths import (
    _robust_distances,
    build_scenario_features,
)


def build_scenario_features_with_peak_month(
    multipliers: pd.DataFrame,
    baseline_daily_energy: pd.Series,
    baseline_daily_peak: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在既有场景特征上补充月峰值轮廓与全年峰值月份。"""

    monthly, annual = build_scenario_features(
        multipliers, baseline_daily_energy, baseline_daily_peak
    )
    peak = baseline_daily_peak.copy()
    peak.index = pd.to_datetime(peak.index).normalize()
    base = pd.DataFrame(
        {
            "date": peak.index,
            "baseline_peak": pd.to_numeric(peak, errors="coerce").to_numpy(float),
        }
    )
    data = multipliers[["scenario_id", "date", "daily_multiplier"]].copy()
    data["date"] = pd.to_datetime(data.date, errors="coerce").dt.normalize()
    data["daily_multiplier"] = pd.to_numeric(
        data.daily_multiplier, errors="coerce"
    )
    data = data.merge(base, on="date", how="left", validate="many_to_one")
    if data[["date", "daily_multiplier", "baseline_peak"]].isna().any().any():
        raise ValueError("峰值月份场景特征输入存在缺失")
    if (
        not np.isfinite(
            data[["daily_multiplier", "baseline_peak"]].to_numpy(float)
        ).all()
        or (data.daily_multiplier <= 0).any()
        or (data.baseline_peak <= 0).any()
    ):
        raise ValueError("峰值月份场景特征输入必须为有限正数")
    data["month"] = data.date.dt.month
    data["daily_peak"] = data.baseline_peak * data.daily_multiplier
    monthly_peak = (
        data.groupby(["scenario_id", "month"], as_index=False, sort=True)
        .daily_peak.max()
        .rename(columns={"daily_peak": "monthly_peak_MW"})
    )
    if monthly_peak.groupby("scenario_id").month.nunique().ne(12).any():
        raise ValueError("每个场景必须完整覆盖12个月")

    extras: list[dict[str, object]] = []
    for scenario_id, part in monthly_peak.groupby("scenario_id", sort=True):
        part = part.sort_values("month", kind="mergesort")
        values = part.monthly_peak_MW.to_numpy(float)
        months = part.month.to_numpy(int)
        peak_position = int(np.argmax(values))
        lookup = dict(zip(months, values, strict=True))
        extras.append(
            {
                "scenario_id": int(scenario_id),
                "annual_peak_month": int(months[peak_position]),
                "january_peak_ratio": float(lookup[1] / values[peak_position]),
                "july_june_peak_ratio": float(lookup[7] / lookup[6]),
                "july_above_june": bool(lookup[7] >= lookup[6]),
            }
        )
    extra = pd.DataFrame(extras)
    annual = annual.merge(extra, on="scenario_id", how="left", validate="one_to_one")
    if annual.isna().any().any():
        raise AssertionError("峰值月份特征合并后存在缺失")
    return monthly, annual


def peak_month_frequency(annual_stats: pd.DataFrame) -> pd.DataFrame:
    """统计各月份成为全年峰值月份的场景频率。"""

    required = {"scenario_id", "annual_peak_month"}
    missing = required - set(annual_stats.columns)
    if missing:
        raise ValueError(f"峰值月份频率输入缺少字段：{sorted(missing)}")
    data = annual_stats[list(required)].copy()
    data["annual_peak_month"] = pd.to_numeric(
        data.annual_peak_month, errors="coerce"
    )
    if (
        data.annual_peak_month.isna().any()
        or not np.equal(
            data.annual_peak_month, np.floor(data.annual_peak_month)
        ).all()
        or not data.annual_peak_month.between(1, 12).all()
    ):
        raise ValueError("annual_peak_month必须为1..12的整数")
    if data.scenario_id.duplicated().any():
        raise ValueError("峰值月份频率输入的scenario_id必须唯一")
    total = len(data)
    if total == 0:
        raise ValueError("峰值月份频率输入不能为空")
    result = (
        data.groupby("annual_peak_month", as_index=False)
        .agg(scenario_count=("scenario_id", "size"))
        .rename(columns={"annual_peak_month": "peak_month"})
        .sort_values("peak_month", kind="mergesort")
        .reset_index(drop=True)
    )
    result["scenario_share"] = result.scenario_count / total
    result["total_scenarios"] = total
    return result


def select_representative_scenarios_peak_banded(
    monthly_features: pd.DataFrame,
    annual_stats: pd.DataFrame,
) -> tuple[dict[str, int], pd.DataFrame]:
    """在年最大负荷分位带内选择特征距离最小的真实场景。"""

    required = {"scenario_id", "annual_peak_MW"}
    missing = required - set(annual_stats.columns)
    if missing:
        raise ValueError(f"年度场景统计缺少字段：{sorted(missing)}")
    if annual_stats.scenario_id.duplicated().any():
        raise ValueError("年度场景统计的scenario_id必须唯一")

    distances = _robust_distances(monthly_features)
    ranking = annual_stats.copy().merge(
        distances, on="scenario_id", how="left", validate="one_to_one"
    )
    ranking["distance"] = ranking.distance.fillna(0.0)
    ranking = ranking.sort_values(
        ["annual_peak_MW", "scenario_id"], kind="mergesort"
    ).reset_index(drop=True)
    n = len(ranking)
    if n == 0:
        raise ValueError("没有可选择的Bootstrap场景")
    ranking["peak_empirical_quantile"] = (np.arange(n) + 0.5) / n

    def choose_band(low: float, high: float, target: float) -> int:
        candidates = ranking.loc[
            ranking.peak_empirical_quantile.between(
                low, high, inclusive="both"
            )
        ].copy()
        if candidates.empty:
            candidates = (
                ranking.assign(
                    qdiff=(ranking.peak_empirical_quantile - target).abs()
                )
                .sort_values(["qdiff", "distance", "scenario_id"])
                .head(1)
            )
        return int(
            candidates.sort_values(["distance", "scenario_id"])
            .iloc[0]
            .scenario_id
        )

    selected = {
        "P10": choose_band(0.08, 0.12, 0.10),
        "P50": choose_band(0.45, 0.55, 0.50),
        "P90": choose_band(0.88, 0.92, 0.90),
    }
    ranking["representative_label"] = ""
    for label, scenario_id in selected.items():
        ranking.loc[
            ranking.scenario_id.eq(scenario_id), "representative_label"
        ] = label
    ranking["selection_method"] = "annual_peak_quantile_band_then_distance"
    return selected, ranking
