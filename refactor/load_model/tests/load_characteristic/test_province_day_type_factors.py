from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from workstreams_load_characteristic_model_v2.annual_assembly import (
    allocate_component_daily_energy,
)
from workstreams_load_characteristic_model_v2.calendar_utils import calendar_frame
from workstreams_load_characteristic_model_v2.monthly_envelope import (
    estimate_province_day_type_factors,
)
from workstreams_load_characteristic_model_v2.quality import HOUR_COLS


def _history_row(date: str, daily_energy: float) -> dict[str, object]:
    return {
        "date": pd.Timestamp(date),
        **{column: daily_energy / 24.0 for column in HOUR_COLS},
    }


def test_province_day_type_factor_uses_2023_daily_energy_median_ratio() -> None:
    rows = [
        *[_history_row(date, 100.0) for date in ("2023-09-04", "2023-09-05", "2023-09-06")],
        *[_history_row(date, 80.0) for date in ("2023-09-09", "2023-09-10")],
        *[
            _history_row(str(date.date()), 70.0)
            for date in pd.date_range("2023-09-29", "2023-10-06", freq="D")
        ],
        # A different year must never influence the shared 2023 factors.
        _history_row("2022-09-05", 10000.0),
    ]

    factors = estimate_province_day_type_factors(pd.DataFrame(rows))
    autumn = factors.query("season == '秋季'").set_index("final_day_type")

    assert autumn.loc["工作日", "factor"] == pytest.approx(1.0)
    assert autumn.loc["普通周末", "factor"] == pytest.approx(0.8)
    assert autumn.loc["法定节假日", "factor"] == pytest.approx(0.7)
    assert not bool(autumn.holiday_merged_with_weekend.any())
    assert set(autumn.factor_scope) == {"province_shared_all_25_components"}


def test_province_holiday_below_eight_days_uses_combined_nonwork_factor() -> None:
    rows = [
        *[_history_row(str(date.date()), 100.0) for date in pd.date_range("2023-03-06", periods=20, freq="B")],
        *[_history_row(date, 80.0) for date in ("2023-03-04", "2023-03-05", "2023-03-11", "2023-03-12")],
        _history_row("2023-04-05", 60.0),
    ]

    factors = estimate_province_day_type_factors(pd.DataFrame(rows))
    spring = factors.query("season == '春季'")

    assert set(spring.final_day_type) == {"工作日", "合并非工作日"}
    assert spring.holiday_merged_with_weekend.all()
    combined = spring.set_index("final_day_type").loc["合并非工作日"]
    # Four weekend days at 80 and one holiday at 60 -> median remains 80.
    assert combined.factor == pytest.approx(0.8)
    assert int(combined.sample_days) == 5


def test_all_components_share_daily_multiplier_and_close_month_energy() -> None:
    calendar = calendar_frame(2030).query("month == 1").copy()
    factors = pd.DataFrame(
        [
            {
                "season": "冬季",
                "final_day_type": day_type,
                "factor": factor,
                "factor_scope": "province_shared_all_25_components",
                "source_year": 2023,
            }
            for day_type, factor in (
                ("工作日", 1.0),
                ("普通周末", 0.8),
                ("法定节假日", 0.7),
            )
        ]
    )
    month = pd.DataFrame(
        {
            "method": "M1",
            "planning_scenario": "基准",
            "year": 2030,
            "month": 1,
            "component": [f"组成{index:02d}" for index in range(25)],
            "month_energy_MWh": np.linspace(50.0, 150.0, 25),
        }
    )

    daily = allocate_component_daily_energy(month, calendar, factors)

    shares = daily.pivot(index="date", columns="component", values="daily_share")
    assert np.allclose(shares.to_numpy(), shares.iloc[:, [0]].to_numpy())
    actual = daily.groupby("component").daily_energy_MWh.sum().sort_index()
    target = month.set_index("component").month_energy_MWh.sort_index()
    assert np.allclose(actual, target, rtol=0.0, atol=1e-9)
    assert set(daily.factor_scope) == {"province_shared_all_25_components"}

