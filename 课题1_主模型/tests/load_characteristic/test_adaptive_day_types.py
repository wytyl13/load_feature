from __future__ import annotations

import pandas as pd

from workstreams_load_characteristic_model_v2.calendar_utils import (
    calendar_frame,
    classify_date,
)
from workstreams_load_characteristic_model_v2.day_types import (
    HOLIDAY,
    MERGED_NONWORKDAY,
    WEEKEND,
    WORKDAY,
)
from workstreams_load_characteristic_model_v2.representative_selection import (
    rank_representative_users,
)


def _quality_row(
    industry: str,
    user: str,
    year: int,
    season: str,
    day_type: str,
    valid_days: int,
    *,
    expected_days: int = 120,
    sample_energy: float = 1.0,
) -> dict[str, object]:
    return {
        "industry": industry,
        "user_id": user,
        "year": year,
        "season": season,
        "day_type": day_type,
        "valid_days": valid_days,
        "expected_days": expected_days,
        "sample_energy": sample_energy,
    }


def test_cross_year_official_holiday_is_classified_by_actual_date() -> None:
    detailed, group, source = classify_date(pd.Timestamp("2022-12-31"))
    assert (detailed, group, source) == (HOLIDAY, "非工作日", "已公布官方日历")

    row = calendar_frame(2022).loc[lambda frame: frame.date.eq("2022-12-31")].iloc[0]
    assert row.day_type == HOLIDAY
    assert row.canonical_day_type == HOLIDAY


def test_candidate_specific_merge_is_used_before_five_key_ranking() -> None:
    rows: list[dict[str, object]] = []
    for year in (2022, 2023):
        # 用户A有更多有效天数，但2022年节假日只有7天，因此只能形成
        # “工作日+合并非工作日”两个配对单元。
        rows.extend(
            [
                _quality_row("行业甲", "用户A", year, "春季", WORKDAY, 20),
                _quality_row("行业甲", "用户A", year, "春季", WEEKEND, 40),
                _quality_row(
                    "行业甲",
                    "用户A",
                    year,
                    "春季",
                    HOLIDAY,
                    7 if year == 2022 else 8,
                ),
                # 用户B恰好8天，节假日必须独立，形成三个配对单元。
                _quality_row("行业甲", "用户B", year, "春季", WORKDAY, 20),
                _quality_row("行业甲", "用户B", year, "春季", WEEKEND, 8),
                _quality_row("行业甲", "用户B", year, "春季", HOLIDAY, 8),
            ]
        )

    result = rank_representative_users(
        pd.DataFrame(rows), required_workday_seasons=("春季",)
    )

    ranking = result.ranking.set_index("user_id")
    assert ranking.loc["用户A", "paired_unit_count"] == 2
    assert ranking.loc["用户B", "paired_unit_count"] == 3
    assert result.selected.loc[0, "user_id"] == "用户B"

    decisions = result.candidate_day_type_decisions.set_index("user_id")
    assert bool(decisions.loc["用户A", "holiday_merged_with_weekend"])
    assert not bool(decisions.loc["用户B", "holiday_merged_with_weekend"])
    assert decisions.loc["用户A", "holiday_valid_days_2022"] == 7
    assert decisions.loc["用户B", "holiday_valid_days_2022"] == 8

    lock = result.industry_day_type_locks.iloc[0]
    assert lock.selected_user == "用户B"
    assert not bool(lock.holiday_merged_with_weekend)
    mapping = result.final_day_type_mapping.set_index("source_day_type")
    assert mapping.loc[WEEKEND, "final_day_type"] == WEEKEND
    assert mapping.loc[HOLIDAY, "final_day_type"] == HOLIDAY

    # 选定代表用户后，其独立口径锁定整个行业；用户A不再沿用自己的合并口径。
    a_cells = result.final_cell_quality.loc[
        result.final_cell_quality.user_id.eq("用户A")
    ]
    assert set(a_cells.final_day_type) == {WORKDAY, WEEKEND, HOLIDAY}


def test_selected_merge_lock_combines_weekend_and_holiday_valid_days() -> None:
    rows: list[dict[str, object]] = []
    for year, holiday_days in ((2022, 7), (2023, 8)):
        rows.extend(
            [
                _quality_row("行业乙", "唯一用户", year, "夏季", WORKDAY, 20),
                _quality_row("行业乙", "唯一用户", year, "夏季", WEEKEND, 9),
                _quality_row(
                    "行业乙", "唯一用户", year, "夏季", HOLIDAY, holiday_days
                ),
            ]
        )

    result = rank_representative_users(
        pd.DataFrame(rows), required_workday_seasons=("夏季",)
    )

    lock = result.industry_day_type_locks.iloc[0]
    assert bool(lock.holiday_merged_with_weekend)
    assert lock.merge_reason == "holiday_valid_days_below_8_in_either_year"

    mapping = result.final_day_type_mapping.set_index("source_day_type")
    assert mapping.loc[WEEKEND, "final_day_type"] == MERGED_NONWORKDAY
    assert mapping.loc[HOLIDAY, "final_day_type"] == MERGED_NONWORKDAY

    merged = result.final_cell_quality.loc[
        result.final_cell_quality.final_day_type.eq(MERGED_NONWORKDAY)
    ].set_index("year")
    assert merged.loc[2022, "valid_days"] == 16
    assert merged.loc[2023, "valid_days"] == 17


def test_legacy_two_group_input_remains_supported() -> None:
    rows: list[dict[str, object]] = []
    for year in (2022, 2023):
        rows.extend(
            [
                _quality_row("行业丙", "用户1", year, "秋季", "工作日", 20),
                _quality_row("行业丙", "用户1", year, "秋季", "非工作日", 8),
            ]
        )

    result = rank_representative_users(
        pd.DataFrame(rows), required_workday_seasons=("秋季",)
    )
    assert result.selected.loc[0, "paired_unit_count"] == 2
    assert bool(
        result.industry_day_type_locks.loc[
            0, "holiday_merged_with_weekend"
        ]
    )
    assert result.industry_day_type_locks.loc[0, "merge_reason"] == (
        "legacy_nonworkday_input"
    )

