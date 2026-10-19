from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from workstreams_load_characteristic_model_v2.reporting_outputs import (
    REPORT_METHOD_STATEMENTS,
    SEASONS,
    build_adaptive_typical_days,
    build_monthly_maximum_curves,
    build_typical_day_load_factors,
    extract_baseline_p50_workday_display,
    normalize_hourly_trajectories,
    select_m1_baseline_p50_annual_scale,
    validate_reporting_contract,
)
from workstreams_load_characteristic_model_v2.result_tables import (
    RUN_MANIFEST_CONTRACT,
)


def _complete_year_hourly(year: int = 2025) -> pd.DataFrame:
    timestamps = pd.date_range(
        f"{year}-01-01 00:00:00",
        f"{year}-12-31 23:00:00",
        freq="h",
    )
    hours = timestamps.hour.to_numpy()
    # A positive, non-flat daily shape with a small seasonal perturbation.
    values = 1.0 + hours / 24.0 + timestamps.month.to_numpy() / 120.0
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "year": year,
            "planning_scenario": "基准",
            "trajectory": "P50_real",
            "value": values,
        }
    )


def test_adaptive_typical_days_use_distinct_holiday_dates_and_close_to_one() -> None:
    result = build_adaptive_typical_days(_complete_year_hourly(2025))
    cells = result[
        [
            "season",
            "final_day_type",
            "sample_days",
            "holiday_sample_days",
            "holiday_merged_with_weekend",
        ]
    ].drop_duplicates()

    spring = cells.loc[cells.season.eq("春季")]
    assert set(spring.final_day_type) == {"工作日", "周末", "节假日"}
    assert set(spring.holiday_sample_days) == {9}
    assert not spring.holiday_merged_with_weekend.any()

    # 2025 summer contains only two statutory-holiday dates, so weekends and
    # holidays must be pooled before the hourly average is taken.
    summer = cells.loc[cells.season.eq("夏季")]
    assert set(summer.final_day_type) == {"工作日", "周末及节假日（合并）"}
    assert set(summer.holiday_sample_days) == {2}
    assert summer.holiday_merged_with_weekend.all()
    merged = summer.loc[summer.final_day_type.eq("周末及节假日（合并）")].iloc[0]
    assert int(merged.sample_days) > int(merged.holiday_sample_days)

    # Exactly eight days is the independent-holiday side of the boundary.
    autumn = cells.loc[cells.season.eq("秋季")]
    assert set(autumn.holiday_sample_days) == {8}
    assert set(autumn.final_day_type) == {"工作日", "周末", "节假日"}

    key = ["year", "planning_scenario", "trajectory", "season", "final_day_type"]
    assert result.groupby(key).hour.nunique().eq(24).all()
    np.testing.assert_allclose(
        result.groupby(key).typical_day_pu.sum().to_numpy(float),
        1.0,
        rtol=0,
        atol=1e-12,
    )
    assert set(result.normalization_method) == {"24_hour_sum_equals_1"}


def test_adaptive_typical_days_require_complete_annual_trajectory() -> None:
    incomplete = _complete_year_hourly(2025).iloc[:-1]
    with pytest.raises(ValueError, match="完整年份"):
        build_adaptive_typical_days(incomplete)


def test_typical_day_load_factor_is_mean_over_max() -> None:
    curve = np.arange(1.0, 25.0)
    curve /= curve.sum()
    typical = pd.DataFrame(
        {
            "year": 2030,
            "planning_scenario": "基准",
            "trajectory": "P50_real",
            "season": "春季",
            "final_day_type": "工作日",
            "hour": range(24),
            "typical_day_pu": curve,
            "sample_days": 60,
            "holiday_sample_days": 8,
            "holiday_merged_with_weekend": False,
            "merge_reason": "达到8天门槛",
        }
    )
    result = build_typical_day_load_factors(typical)
    assert len(result) == 1
    assert result.loc[0, "typical_day_load_factor"] == pytest.approx(
        float(curve.mean() / curve.max())
    )
    assert result.loc[0, "typical_day_mean_pu"] == pytest.approx(1.0 / 24.0)


def test_annual_hourly_normalization_is_independent_by_trajectory() -> None:
    timestamps = pd.date_range("2030-01-01", periods=3, freq="h")
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "year": 2030,
                    "planning_scenario": "基准",
                    "trajectory": trajectory,
                    "value": values,
                }
            )
            for trajectory, values in (
                ("P10_real", [1.0, 2.0, 4.0]),
                ("P50_real", [10.0, 15.0, 20.0]),
            )
        ],
        ignore_index=True,
    )
    result = normalize_hourly_trajectories(frame)
    assert "value" not in result.columns
    maxima = result.groupby("trajectory").load_pu.max()
    np.testing.assert_allclose(maxima.to_numpy(float), 1.0)
    bases = result.groupby("trajectory").annual_peak_base_value.first()
    assert bases.to_dict() == {"P10_real": 4.0, "P50_real": 20.0}
    p50 = result.loc[result.trajectory.eq("P50_real"), "load_pu"].to_numpy(float)
    np.testing.assert_allclose(p50, [0.5, 0.75, 1.0])


def test_monthly_maximum_curve_has_twelve_points_and_same_annual_base() -> None:
    timestamps = pd.to_datetime([f"2030-{month:02d}-15 18:00" for month in range(1, 13)])
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "year": 2030,
            "planning_scenario": "基准",
            "trajectory": "P50_real",
            "value": np.arange(1.0, 13.0),
        }
    )
    result = build_monthly_maximum_curves(frame)
    assert len(result) == 12
    assert result.month.tolist() == list(range(1, 13))
    assert result.annual_peak_base_value.eq(12.0).all()
    np.testing.assert_allclose(result.monthly_max_pu, np.arange(1.0, 13.0) / 12.0)
    assert result.monthly_max_pu.max() == pytest.approx(1.0)
    assert "value" not in result.columns


def _display_typical_days() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    shape = np.arange(1.0, 25.0)
    shape /= shape.sum()
    for year in range(2025, 2046):
        for season in SEASONS:
            for hour, value in enumerate(shape):
                rows.append(
                    {
                        "year": year,
                        "planning_scenario": "基准",
                        "trajectory": "P50_real",
                        "season": season,
                        "final_day_type": "工作日",
                        "hour": hour,
                        "typical_day_pu": value,
                        "sample_days": 60,
                        "holiday_sample_days": 8,
                        "holiday_merged_with_weekend": False,
                        "merge_reason": "达到8天门槛",
                    }
                )
    rows.append(
        {
            "year": 2030,
            "planning_scenario": "低碳",
            "trajectory": "P50_real",
            "season": "春季",
            "final_day_type": "工作日",
            "hour": 0,
            "typical_day_pu": 1.0,
            "sample_days": 1,
            "holiday_sample_days": 8,
            "holiday_merged_with_weekend": False,
            "merge_reason": "非展示行",
        }
    )
    return pd.DataFrame(rows)


def test_display_extract_is_exact_2016_row_baseline_p50_workday_contract() -> None:
    result = extract_baseline_p50_workday_display(_display_typical_days())
    assert len(result) == 2016
    assert set(result.year) == set(range(2025, 2046))
    assert set(result.planning_scenario) == {"基准"}
    assert set(result.trajectory) == {"P50_real"}
    assert set(result.final_day_type) == {"工作日"}
    assert result.groupby(["year", "season"]).hour.nunique().eq(24).all()


def test_display_extract_rejects_a_missing_hour() -> None:
    source = _display_typical_days()
    missing = source.drop(
        source.index[
            source.planning_scenario.eq("基准")
            & source.year.eq(2025)
            & source.season.eq("春季")
            & source.hour.eq(0)
        ][0]
    )
    with pytest.raises(ValueError, match="覆盖不完整"):
        extract_baseline_p50_workday_display(missing)


def test_m1_baseline_annual_scale_uses_only_p50_natural_total() -> None:
    upstream = pd.DataFrame(
        [
            {
                "method": method,
                "scenario": scenario,
                "year": year,
                "P10_natural_MWh": 100.0 + year,
                "P50_natural_MWh": 200.0 + year,
                "P90_natural_MWh": 900.0 + year,
            }
            for method in ("M1", "方法A")
            for scenario in ("低碳", "基准", "高电气化")
            for year in (2030, 2031)
        ]
    )
    result = select_m1_baseline_p50_annual_scale(upstream, years=(2030, 2031))
    assert result.annual_scale_MWh.tolist() == [2230.0, 2231.0]
    assert set(result.method) == {"M1"}
    assert set(result.planning_scenario) == {"基准"}
    assert set(result.annual_energy_quantile) == {"P50"}
    assert set(result.annual_energy_basis) == {"P50_natural_total"}
    assert not result.annual_scale_MWh.isin(upstream.P10_natural_MWh).any()
    assert not result.annual_scale_MWh.isin(upstream.P90_natural_MWh).any()


def test_m1_baseline_scale_rejects_p10_p90_only_input() -> None:
    without_p50 = pd.DataFrame(
        [
            {
                "method": "M1",
                "scenario": "基准",
                "year": 2030,
                "P10_natural_MWh": 100.0,
                "P90_natural_MWh": 300.0,
            }
        ]
    )
    with pytest.raises(ValueError, match="P50_natural_MWh"):
        select_m1_baseline_p50_annual_scale(without_p50)


def test_report_method_statements_describe_the_double_gate() -> None:
    joined = "".join(REPORT_METHOD_STATEMENTS)
    assert "先完成四时段份额调整" in joined
    assert "日负荷率大于0.80" in joined
    assert "变异系数小于0.15" in joined
    assert "峰谷差率绝对变化量位于6.5%—25%闭区间" in joined
    assert "门控未通过时" in joined
    assert "最终曲线等于四时段" in joined
    assert "门控通过时" in joined and "最终优先目标" in joined


def test_reporting_contract_rejects_3_4_0_manifest() -> None:
    old_manifest = {
        "status": "success",
        "method": "M1",
        **RUN_MANIFEST_CONTRACT,
        "model_version": "3.4.0",
    }

    with pytest.raises(RuntimeError, match="model_version"):
        validate_reporting_contract(
            old_manifest,
            {},
            require_parquet=False,
        )
