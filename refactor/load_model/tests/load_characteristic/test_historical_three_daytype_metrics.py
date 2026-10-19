from __future__ import annotations

import numpy as np
import pandas as pd

from workstreams_load_characteristic_model_v2.calendar_utils import calendar_frame
from workstreams_load_characteristic_model_v2.day_types import (
    HOLIDAY,
    WEEKEND,
    WORKDAY,
)
from workstreams_load_characteristic_model_v2.metrics import (
    historical_season_daytype_typical_shapes,
    season_daytype_typical_shapes,
)


def _hourly_2025() -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h")
    frame = pd.DataFrame({"timestamp": timestamps})
    frame["date"] = frame.timestamp.dt.normalize()
    frame["hour"] = frame.timestamp.dt.hour
    calendar = calendar_frame(2025)[["date", "canonical_day_type"]]
    frame = frame.merge(calendar, on="date", how="left", validate="many_to_one")

    workday = np.ones(24, dtype=float)
    workday[8] = 3.0
    weekend = np.ones(24, dtype=float)
    weekend[12] = 4.0
    holiday = np.ones(24, dtype=float)
    holiday[20] = 5.0
    shapes = {WORKDAY: workday, WEEKEND: weekend, HOLIDAY: holiday}
    frame["load_MW"] = [
        shapes[day_type][hour]
        for day_type, hour in zip(frame.canonical_day_type, frame.hour)
    ]
    return frame[["timestamp", "load_MW"]]


def test_three_daytype_typical_shapes_merge_below_8_and_keep_equal_8() -> None:
    result = historical_season_daytype_typical_shapes(_hourly_2025())

    assert len(result) == 4 * 3 * 24
    assert set(result.day_type) == {WORKDAY, WEEKEND, HOLIDAY}
    closure = result.groupby(["season", "day_type"]).normalized_load.sum()
    np.testing.assert_allclose(closure, 1.0, rtol=0.0, atol=1e-12)

    summer = result.loc[result.season.eq("夏季")]
    assert set(summer.holiday_days) == {2}
    assert summer.holiday_merged_with_weekend.all()
    summer_weekend = summer.loc[summer.day_type.eq(WEEKEND)].sort_values("hour")
    summer_holiday = summer.loc[summer.day_type.eq(HOLIDAY)].sort_values("hour")
    np.testing.assert_allclose(
        summer_weekend.normalized_load,
        summer_holiday.normalized_load,
        rtol=0.0,
        atol=0.0,
    )
    assert set(summer_weekend.source_sample_days) == set(
        summer_holiday.source_sample_days
    )
    assert set(summer_holiday.sample_days) == {2}

    # 2025年秋季恰有8个法定节假日，边界必须独立提取。
    autumn = result.loc[result.season.eq("秋季")]
    assert set(autumn.holiday_days) == {8}
    assert not autumn.holiday_merged_with_weekend.any()
    autumn_weekend = autumn.loc[autumn.day_type.eq(WEEKEND)].sort_values("hour")
    autumn_holiday = autumn.loc[autumn.day_type.eq(HOLIDAY)].sort_values("hour")
    assert not np.allclose(
        autumn_weekend.normalized_load, autumn_holiday.normalized_load
    )
    assert set(autumn_holiday.sample_days) == {8}
    assert set(autumn_holiday.source_sample_days) == {8}


def test_existing_two_group_typical_shape_contract_is_unchanged() -> None:
    result = season_daytype_typical_shapes(_hourly_2025())
    assert len(result) == 4 * 2 * 24
    assert set(result.day_group) == {"工作日", "非工作日"}
    closure = result.groupby(["season", "day_group"]).normalized_load.sum()
    np.testing.assert_allclose(closure, 1.0, rtol=0.0, atol=1e-12)

