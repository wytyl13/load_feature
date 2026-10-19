from __future__ import annotations

import numpy as np
import pandas as pd

from workstreams_load_characteristic_model_v2.baseline_shapes import build_baseline_shapes
from workstreams_load_characteristic_model_v2.parameter_pipeline import build_cell_quality
from workstreams_load_characteristic_model_v2.quality import SHAPE_COLS


def _curve(hour: int) -> np.ndarray:
    result = np.full(24, 1.0)
    result[hour] = 3.0
    return result / result.sum()


def _daily_row(
    date: pd.Timestamp,
    day_type: str,
    curve: np.ndarray,
) -> dict[str, object]:
    return {
        "department": "第二产业",
        "industry": "测试行业",
        "date": date,
        "year": int(date.year),
        "season": "冬季",
        "day_type_detail": day_type,
        "source_user_count": 2,
        "active_user_count": 2,
        "missing_user_count": 0,
        "partial_composition_day": False,
        "aggregation_method": "user_day_normalize_then_equal_weight",
        **dict(zip(SHAPE_COLS, curve, strict=True)),
    }


def _make_industry_daily(holiday_days_2022: int, holiday_days_2023: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year, holiday_days in ((2022, holiday_days_2022), (2023, holiday_days_2023)):
        start = pd.Timestamp(year=year, month=1, day=1)
        for offset in range(20):
            rows.append(_daily_row(start + pd.Timedelta(days=offset), "普通工作日", _curve(8)))
        for offset in range(8):
            rows.append(_daily_row(start + pd.Timedelta(days=30 + offset), "普通周末", _curve(12)))
        for offset in range(holiday_days):
            rows.append(_daily_row(start + pd.Timedelta(days=50 + offset), "法定节假日", _curve(20)))
    return pd.DataFrame(rows)


def test_holiday_exactly_eight_days_stays_separate() -> None:
    quality = build_cell_quality(_make_industry_daily(8, 8))
    cell = quality.query("industry == '测试行业' and season == '冬季'")
    assert set(cell.final_day_type) == {"工作日", "普通周末", "法定节假日"}
    assert not cell.holiday_merged_with_weekend.any()


def test_holiday_below_eight_days_is_merged_in_both_years() -> None:
    quality = build_cell_quality(_make_industry_daily(7, 12))
    cell = quality.query("industry == '测试行业' and season == '冬季'")
    assert set(cell.final_day_type) == {"工作日", "合并非工作日"}
    assert cell.holiday_merged_with_weekend.all()


def test_2023_medoid_is_an_actual_industry_day_and_is_baseline() -> None:
    dates = pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"])
    curves = [_curve(7), (_curve(7) + _curve(9)) / 2, _curve(9)]
    daily = pd.DataFrame(
        [_daily_row(date, "普通工作日", curve) for date, curve in zip(dates, curves, strict=True)]
    )
    quality = build_cell_quality(daily)

    shapes, _ = build_baseline_shapes(daily, quality)

    row = shapes.query("year == 2023 and final_day_type == '工作日'").iloc[0]
    source = daily.loc[daily.date.eq(row.medoid_date), SHAPE_COLS].iloc[0]
    assert np.allclose(row[SHAPE_COLS].to_numpy(float), source.to_numpy(float))
    assert bool(row.is_future_baseline)


def test_medoid_tie_is_resolved_by_earliest_date() -> None:
    dates = pd.to_datetime(["2023-01-05", "2023-01-03"])
    daily = pd.DataFrame(
        [
            _daily_row(dates[0], "普通工作日", _curve(7)),
            _daily_row(dates[1], "普通工作日", _curve(9)),
        ]
    )
    shapes, _ = build_baseline_shapes(daily, build_cell_quality(daily))
    row = shapes.query("year == 2023 and final_day_type == '工作日'").iloc[0]
    assert row.medoid_date == pd.Timestamp("2023-01-03")

