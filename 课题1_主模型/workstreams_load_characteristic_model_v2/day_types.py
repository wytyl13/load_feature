"""Canonical and adaptive day-type rules shared by selection and assembly.

Historical observations start with three canonical classes: workday, ordinary
weekend and statutory holiday.  For each candidate user and season, holidays
remain independent only when both comparison years have at least eight valid
holiday dates.  Otherwise weekend and holiday observations are pooled before
any curve or quality statistic is calculated.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd


WORKDAY = "工作日"
WEEKEND = "普通周末"
HOLIDAY = "法定节假日"
MERGED_NONWORKDAY = "合并非工作日"
HOLIDAY_MIN_VALID_DAYS = 8

CANONICAL_SOURCE_DAY_TYPES = (WORKDAY, WEEKEND, HOLIDAY)
FINAL_DAY_TYPES = (WORKDAY, WEEKEND, HOLIDAY, MERGED_NONWORKDAY)

_WORKDAY_ALIASES = frozenset(
    {
        WORKDAY,
        "普通工作日",
        "调休工作日",
        "workday",
        "weekday",
        "wd",
        "working_day",
        "1",
        "true",
    }
)
_WEEKEND_ALIASES = frozenset(
    {WEEKEND, "周末", "周末日", "weekend", "ordinary_weekend"}
)
_HOLIDAY_ALIASES = frozenset(
    {HOLIDAY, "节假日", "法定假日", "holiday", "statutory_holiday"}
)
_MERGED_ALIASES = frozenset(
    {
        MERGED_NONWORKDAY,
        "非工作日",
        "周末及节假日",
        "周末及节假日（合并）",
        "合并周末节假日",
        "nonworkday",
        "non_workday",
        "0",
        "false",
    }
)


def _require_columns(
    frame: pd.DataFrame, columns: Iterable[str], label: str
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def canonical_day_type(value: object) -> str:
    """Map detailed, legacy and English labels to the canonical day types."""

    if isinstance(value, bool):
        return WORKDAY if value else MERGED_NONWORKDAY
    text = str(value).strip().lower()
    if text in _WORKDAY_ALIASES or (
        "工作日" in text and "非工作日" not in text
    ):
        return WORKDAY
    if text in _WEEKEND_ALIASES:
        return WEEKEND
    if text in _HOLIDAY_ALIASES:
        return HOLIDAY
    if text in _MERGED_ALIASES:
        return MERGED_NONWORKDAY
    raise ValueError(f"unknown day type: {value!r}")


def is_workday_type(value: object) -> bool:
    """Return whether a raw or canonical label belongs to the workday class."""

    return canonical_day_type(value) == WORKDAY


def _validated_years(years: Sequence[int]) -> tuple[int, int]:
    normalized = tuple(int(year) for year in years)
    if len(normalized) != 2 or normalized[0] == normalized[1]:
        raise ValueError("years must contain two distinct years")
    return normalized[0], normalized[1]


def build_candidate_day_type_decisions(
    cell_quality: pd.DataFrame,
    *,
    industry_col: str = "industry",
    user_col: str = "user_id",
    year_col: str = "year",
    season_col: str = "season",
    day_type_col: str = "day_type",
    valid_days_col: str = "valid_days",
    years: Sequence[int] = (2022, 2023),
    seasons: Iterable[object] | None = None,
    holiday_min_days: int = HOLIDAY_MIN_VALID_DAYS,
) -> pd.DataFrame:
    """Decide holiday merging independently for every candidate and season.

    A missing holiday row is treated as zero valid holiday dates.  Legacy
    ``非工作日`` input remains a merged non-workday contract because it cannot
    be split retrospectively.
    """

    first_year, second_year = _validated_years(years)
    if int(holiday_min_days) < 1:
        raise ValueError("holiday_min_days must be positive")
    required = (
        industry_col,
        user_col,
        year_col,
        season_col,
        day_type_col,
        valid_days_col,
    )
    _require_columns(cell_quality, required, "cell_quality")
    if cell_quality.empty:
        return pd.DataFrame()

    data = cell_quality.loc[:, required].copy()
    data[year_col] = pd.to_numeric(data[year_col], errors="raise").astype(int)
    data = data.loc[data[year_col].isin((first_year, second_year))].copy()
    if data.empty:
        raise ValueError(
            f"cell_quality contains neither requested year: {(first_year, second_year)}"
        )
    data[valid_days_col] = pd.to_numeric(
        data[valid_days_col], errors="coerce"
    )
    if data[valid_days_col].isna().any() or (data[valid_days_col] < 0).any():
        raise ValueError("valid day counts must be numeric, finite and non-negative")
    data["_source_day_type"] = data[day_type_col].map(canonical_day_type)

    candidate_keys = [industry_col, user_col]
    candidates = data[candidate_keys].drop_duplicates().reset_index(drop=True)
    if seasons is None:
        season_values = list(dict.fromkeys(data[season_col].tolist()))
    else:
        season_values = list(dict.fromkeys(seasons))
    if not season_values:
        raise ValueError("at least one season is required")
    season_frame = pd.DataFrame({season_col: season_values})
    universe = candidates.merge(season_frame, how="cross")

    grouped = (
        data.groupby(
            [*candidate_keys, season_col, year_col, "_source_day_type"],
            as_index=False,
            dropna=False,
            sort=False,
        )[valid_days_col]
        .sum()
    )
    observed = (
        grouped.groupby([*candidate_keys, season_col], sort=False)[
            "_source_day_type"
        ]
        .agg(lambda values: frozenset(values))
        .rename("_observed_day_types")
        .reset_index()
    )
    holiday = grouped.loc[grouped._source_day_type.eq(HOLIDAY)]
    holiday = (
        holiday.groupby(
            [*candidate_keys, season_col, year_col], as_index=False, sort=False
        )[valid_days_col]
        .sum()
        .pivot(
            index=[*candidate_keys, season_col],
            columns=year_col,
            values=valid_days_col,
        )
        .reset_index()
    )
    for year in (first_year, second_year):
        if year not in holiday:
            holiday[year] = 0.0
    holiday = holiday.rename(
        columns={
            first_year: f"holiday_valid_days_{first_year}",
            second_year: f"holiday_valid_days_{second_year}",
        }
    )

    result = universe.merge(
        observed, on=[*candidate_keys, season_col], how="left", validate="one_to_one"
    ).merge(
        holiday,
        on=[*candidate_keys, season_col],
        how="left",
        validate="one_to_one",
    )
    empty_types = frozenset()
    result["_observed_day_types"] = result["_observed_day_types"].map(
        lambda value: value if isinstance(value, frozenset) else empty_types
    )
    year_columns = [
        f"holiday_valid_days_{first_year}",
        f"holiday_valid_days_{second_year}",
    ]
    result[year_columns] = result[year_columns].fillna(0.0)

    detailed_flags: list[bool] = []
    legacy_flags: list[bool] = []
    for observed_types in result["_observed_day_types"]:
        detailed_flags.append(bool({WEEKEND, HOLIDAY}.intersection(observed_types)))
        legacy_flags.append(MERGED_NONWORKDAY in observed_types)
    result["_detailed_nonworkday_input"] = detailed_flags
    result["_legacy_nonworkday_input"] = legacy_flags
    mixed = result._detailed_nonworkday_input & result._legacy_nonworkday_input
    if mixed.any():
        bad = result.loc[mixed, candidate_keys + [season_col]].to_dict("records")
        raise ValueError(
            "detailed weekend/holiday rows cannot be mixed with legacy "
            f"non-workday rows in the same candidate season: {bad[:5]}"
        )

    below = result[year_columns].lt(float(holiday_min_days)).any(axis=1)
    result["holiday_merged_with_weekend"] = (
        result._legacy_nonworkday_input
        | (result._detailed_nonworkday_input & below)
    )
    result["merge_reason"] = "no_nonworkday_rows"
    result.loc[
        result._detailed_nonworkday_input & ~below, "merge_reason"
    ] = "holiday_valid_days_at_least_8_in_both_years"
    result.loc[
        result._detailed_nonworkday_input & below, "merge_reason"
    ] = "holiday_valid_days_below_8_in_either_year"
    result.loc[
        result._legacy_nonworkday_input, "merge_reason"
    ] = "legacy_nonworkday_input"
    result["day_type_contract"] = "workday_only"
    result.loc[result._detailed_nonworkday_input, "day_type_contract"] = (
        "adaptive_detailed_day_types"
    )
    result.loc[result._legacy_nonworkday_input, "day_type_contract"] = (
        "legacy_two_group_input"
    )
    result["holiday_min_days"] = int(holiday_min_days)
    result["holiday_valid_days_min"] = result[year_columns].min(axis=1)
    result = result.drop(
        columns=[
            "_observed_day_types",
            "_detailed_nonworkday_input",
            "_legacy_nonworkday_input",
        ]
    )
    for column in year_columns + ["holiday_valid_days_min"]:
        result[column] = result[column].round().astype(int)
    return result.sort_values(
        [*candidate_keys, season_col], kind="mergesort"
    ).reset_index(drop=True)


def _mapped_final_day_type(source: str, merged: bool) -> str:
    if source == WORKDAY:
        return WORKDAY
    if source == MERGED_NONWORKDAY:
        return MERGED_NONWORKDAY
    if source in {WEEKEND, HOLIDAY}:
        return MERGED_NONWORKDAY if merged else source
    raise ValueError(f"unknown canonical source day type: {source}")


def apply_day_type_decisions(
    cell_quality: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    decision_level: str,
    industry_col: str = "industry",
    user_col: str = "user_id",
    year_col: str = "year",
    season_col: str = "season",
    day_type_col: str = "day_type",
    sum_columns: Sequence[str] = (
        "valid_days",
        "expected_days",
        "sample_energy",
    ),
) -> pd.DataFrame:
    """Apply candidate- or industry-level locks and aggregate final cells."""

    if decision_level not in {"candidate", "industry"}:
        raise ValueError("decision_level must be 'candidate' or 'industry'")
    merge_keys = [industry_col, season_col]
    if decision_level == "candidate":
        merge_keys.insert(1, user_col)
    required = [industry_col, user_col, year_col, season_col, day_type_col, *sum_columns]
    _require_columns(cell_quality, required, "cell_quality")
    _require_columns(
        decisions,
        [*merge_keys, "holiday_merged_with_weekend", "merge_reason"],
        "day_type_decisions",
    )
    if decisions.duplicated(merge_keys).any():
        raise ValueError(f"day-type decisions must be unique by {merge_keys}")

    data = cell_quality.loc[:, required].copy()
    data["source_day_type"] = data[day_type_col].map(canonical_day_type)
    decision_columns = [
        *merge_keys,
        "holiday_merged_with_weekend",
        "merge_reason",
    ]
    for optional in (
        "selected_user",
        "day_type_contract",
        "holiday_min_days",
        "holiday_valid_days_min",
    ):
        if optional in decisions.columns:
            decision_columns.append(optional)
    data = data.merge(
        decisions[decision_columns],
        on=merge_keys,
        how="left",
        validate="many_to_one",
    )
    if data.holiday_merged_with_weekend.isna().any():
        missing = data.loc[
            data.holiday_merged_with_weekend.isna(), merge_keys
        ].drop_duplicates()
        raise ValueError(
            f"day-type decision missing for cells: {missing.to_dict('records')[:5]}"
        )
    impossible = (
        data.source_day_type.eq(MERGED_NONWORKDAY)
        & ~data.holiday_merged_with_weekend.astype(bool)
    )
    if impossible.any():
        bad = data.loc[impossible, [industry_col, user_col, season_col]].drop_duplicates()
        raise ValueError(
            "an independent industry lock cannot split legacy non-workday rows: "
            f"{bad.to_dict('records')[:5]}"
        )
    data["final_day_type"] = [
        _mapped_final_day_type(source, bool(merged))
        for source, merged in zip(
            data.source_day_type, data.holiday_merged_with_weekend
        )
    ]
    for column in sum_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[list(sum_columns)].isna().any().any():
        raise ValueError("final cell quality measures must be numeric and finite")

    group_columns = [
        industry_col,
        user_col,
        year_col,
        season_col,
        "final_day_type",
        "holiday_merged_with_weekend",
        "merge_reason",
    ]
    for optional in ("selected_user", "day_type_contract"):
        if optional in data.columns:
            group_columns.append(optional)
    aggregated = (
        data.groupby(group_columns, as_index=False, dropna=False, sort=False)[
            list(sum_columns)
        ]
        .sum()
    )
    source_types = (
        data.groupby(group_columns, as_index=False, dropna=False, sort=False)[
            "source_day_type"
        ]
        .agg(lambda values: "|".join(sorted(set(values))))
        .rename(columns={"source_day_type": "source_day_types"})
    )
    return aggregated.merge(
        source_types, on=group_columns, how="left", validate="one_to_one"
    ).sort_values(
        [industry_col, user_col, season_col, year_col, "final_day_type"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_industry_day_type_locks(
    candidate_decisions: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    industry_col: str = "industry",
    user_col: str = "user_id",
    season_col: str = "season",
) -> pd.DataFrame:
    """Lock every industry-season to its selected candidate's merge decision."""

    _require_columns(candidate_decisions, (industry_col, user_col, season_col), "candidate decisions")
    _require_columns(selected, (industry_col, user_col), "selected representatives")
    if selected[industry_col].duplicated().any():
        raise ValueError("selected representatives must be unique by industry")
    keys = [industry_col, user_col]
    locks = selected[keys].merge(
        candidate_decisions,
        on=keys,
        how="left",
        validate="one_to_many",
    )
    if locks[season_col].isna().any():
        raise ValueError("selected representative is missing day-type decisions")
    locks = locks.rename(columns={user_col: "selected_user"})
    ordered = [
        industry_col,
        season_col,
        "selected_user",
        *[
            column
            for column in locks.columns
            if column not in {industry_col, season_col, "selected_user"}
        ],
    ]
    return locks[ordered].sort_values(
        [industry_col, season_col], kind="mergesort"
    ).reset_index(drop=True)


def build_final_day_type_mapping(
    industry_locks: pd.DataFrame,
    *,
    industry_col: str = "industry",
    season_col: str = "season",
) -> pd.DataFrame:
    """Expand industry locks into an explicit source-to-final day-type map."""

    _require_columns(
        industry_locks,
        (
            industry_col,
            season_col,
            "holiday_merged_with_weekend",
            "merge_reason",
        ),
        "industry day-type locks",
    )
    rows: list[dict[str, object]] = []
    for lock in industry_locks.to_dict("records"):
        for source in (*CANONICAL_SOURCE_DAY_TYPES, MERGED_NONWORKDAY):
            rows.append(
                {
                    **lock,
                    "source_day_type": source,
                    "final_day_type": _mapped_final_day_type(
                        source, bool(lock["holiday_merged_with_weekend"])
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        [industry_col, season_col, "source_day_type"], kind="mergesort"
    ).reset_index(drop=True)


__all__ = [
    "CANONICAL_SOURCE_DAY_TYPES",
    "FINAL_DAY_TYPES",
    "HOLIDAY",
    "HOLIDAY_MIN_VALID_DAYS",
    "MERGED_NONWORKDAY",
    "WEEKEND",
    "WORKDAY",
    "apply_day_type_decisions",
    "build_candidate_day_type_decisions",
    "build_final_day_type_mapping",
    "build_industry_day_type_locks",
    "canonical_day_type",
    "is_workday_type",
]
