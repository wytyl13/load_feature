"""Representative-user ranking and fine-industry parameter trace.

The selection rules in this module are intentionally deterministic.  Sample
energy is used only as the fourth user-quality tie breaker.  Every traditional
fine industry keeps its own parameter estimate; official 2025 industry
electricity is retained only as an auditable weight input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from .day_types import (
    WORKDAY,
    apply_day_type_decisions,
    build_candidate_day_type_decisions,
    build_final_day_type_mapping,
    build_industry_day_type_locks,
    is_workday_type,
)


DEFAULT_SEASONS = ("春", "夏", "秋", "冬")


@dataclass(frozen=True)
class RepresentativeSelection:
    """Full user ranking plus the one selected row per fine industry."""

    ranking: pd.DataFrame
    selected: pd.DataFrame
    candidate_day_type_decisions: pd.DataFrame = field(default_factory=pd.DataFrame)
    industry_day_type_locks: pd.DataFrame = field(default_factory=pd.DataFrame)
    final_day_type_mapping: pd.DataFrame = field(default_factory=pd.DataFrame)
    final_cell_quality: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class IndustryParameterTrace:
    """Audit trace for the one-industry/one-parameter-set contract.

    No winner is selected across industries.  A row is active exactly when the
    same industry's paired history is valid.  Insufficient history means that
    the industry's own estimate is frozen at zero; it never causes borrowing
    from another industry.  The wrapper keeps that contract explicit.
    """

    trace: pd.DataFrame


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _is_workday(value: object) -> bool:
    return is_workday_type(value)


def _canonical_season(value: object) -> str:
    text = str(value).strip().lower()
    aliases = {
        "春": "春",
        "春季": "春",
        "spring": "春",
        "夏": "夏",
        "夏季": "夏",
        "summer": "夏",
        "秋": "秋",
        "秋季": "秋",
        "autumn": "秋",
        "fall": "秋",
        "冬": "冬",
        "冬季": "冬",
        "winter": "冬",
    }
    return aliases.get(text, str(value).strip())


def _display_season(value: object) -> str:
    return {
        "春": "春季",
        "夏": "夏季",
        "秋": "秋季",
        "冬": "冬季",
    }.get(str(value).strip(), str(value).strip())


def rank_representative_users(
    cell_quality: pd.DataFrame,
    *,
    industry_col: str = "industry",
    user_col: str = "user_id",
    year_col: str = "year",
    season_col: str = "season",
    day_type_col: str = "day_type",
    valid_days_col: str = "valid_days",
    expected_days_col: str = "expected_days",
    sample_energy_col: str = "sample_energy",
    years: tuple[int, int] = (2022, 2023),
    required_workday_seasons: Iterable[str] = DEFAULT_SEASONS,
    min_workday_days: int = 20,
    min_nonworkday_days: int = 8,
) -> RepresentativeSelection:
    """Rank users inside each fine industry using the approved five keys.

    ``cell_quality`` has one row per user/year/season/day-type cell. Duplicate
    rows are conservatively aggregated before ranking.  Detailed weekend and
    holiday rows are first mapped using each candidate's own two-year holiday
    coverage: either year below the non-workday threshold pools both source
    classes, while equality keeps holidays independent.  This candidate-level
    final grid feeds the approved five ranking keys.  After selection, the
    winner's decision is locked by industry and season and exposed, together
    with the source-to-final mapping and re-aggregated valid-day cells, on the
    returned ``RepresentativeSelection``.
    """

    if len(years) != 2 or years[0] == years[1]:
        raise ValueError("years must contain two distinct years")
    if min_workday_days < 1 or min_nonworkday_days < 1:
        raise ValueError("minimum valid-day thresholds must be positive")

    required = (
        industry_col,
        user_col,
        year_col,
        season_col,
        day_type_col,
        valid_days_col,
        expected_days_col,
        sample_energy_col,
    )
    _require_columns(cell_quality, required, "cell_quality")
    if cell_quality.empty:
        empty = pd.DataFrame()
        return RepresentativeSelection(empty, empty)

    data = cell_quality.loc[:, required].copy()
    data[industry_col] = data[industry_col].astype(str).str.strip()
    data["_user_sort"] = data[user_col].astype(str).str.strip()
    data[year_col] = pd.to_numeric(data[year_col], errors="raise").astype(int)
    data = data[data[year_col].isin(years)].copy()
    if data.empty:
        raise ValueError(f"cell_quality contains neither requested year: {years}")

    for column in (valid_days_col, expected_days_col, sample_energy_col):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[[valid_days_col, expected_days_col, sample_energy_col]].isna().any().any():
        raise ValueError("quality counts and sample energy must be numeric and finite")
    if (data[[valid_days_col, expected_days_col, sample_energy_col]] < 0).any().any():
        raise ValueError("quality counts and sample energy must be non-negative")

    data["_season"] = data[season_col].map(_canonical_season)
    required_season_values = [
        _canonical_season(value) for value in required_workday_seasons
    ]
    decision_seasons = list(
        dict.fromkeys([*required_season_values, *data["_season"].tolist()])
    )
    candidate_day_type_decisions = build_candidate_day_type_decisions(
        data,
        industry_col=industry_col,
        user_col=user_col,
        year_col=year_col,
        season_col="_season",
        day_type_col=day_type_col,
        valid_days_col=valid_days_col,
        years=years,
        seasons=decision_seasons,
        holiday_min_days=min_nonworkday_days,
    )
    cells = apply_day_type_decisions(
        data,
        candidate_day_type_decisions,
        decision_level="candidate",
        industry_col=industry_col,
        user_col=user_col,
        year_col=year_col,
        season_col="_season",
        day_type_col=day_type_col,
        sum_columns=(valid_days_col, expected_days_col, sample_energy_col),
    ).rename(columns={"final_day_type": "_day_type"})
    cells["_user_sort"] = cells[user_col].astype(str).str.strip()
    cells["_is_workday"] = cells["_day_type"].eq(WORKDAY)
    threshold = np.where(cells["_is_workday"], min_workday_days, min_nonworkday_days)
    cells["_qualified"] = cells[valid_days_col].to_numpy(dtype=float) >= threshold

    requested_years = set(int(year) for year in years)
    required_seasons = set(required_season_values)
    records: list[dict[str, object]] = []
    user_keys = [industry_col, user_col, "_user_sort"]
    for (industry, user, user_sort), user_cells in cells.groupby(
        user_keys, sort=False, dropna=False
    ):
        paired_units = 0
        paired_workday_seasons: set[str] = set()
        unit_keys = ["_season", "_day_type", "_is_workday"]
        for (season, _day_type, is_workday), unit in user_cells.groupby(
            unit_keys, sort=False, dropna=False
        ):
            qualified_years = set(
                unit.loc[unit["_qualified"], year_col].astype(int).tolist()
            )
            if requested_years.issubset(qualified_years):
                paired_units += 1
                if bool(is_workday):
                    paired_workday_seasons.add(str(season).strip())

        total_valid = float(user_cells[valid_days_col].sum())
        total_expected = float(user_cells[expected_days_col].sum())
        valid_rate = total_valid / total_expected if total_expected > 0 else 0.0
        sample_energy = float(user_cells[sample_energy_col].sum())
        available_years = set(user_cells[year_col].astype(int).tolist())
        has_two_years = requested_years.issubset(available_years)
        trend_eligible = required_seasons.issubset(paired_workday_seasons)
        if trend_eligible:
            trend_status = "short_history_exploratory"
        elif has_two_years:
            trend_status = "coverage_insufficient_frozen"
        else:
            trend_status = "single_year_frozen"

        records.append(
            {
                industry_col: industry,
                user_col: user,
                "_user_sort": user_sort,
                "paired_unit_count": int(paired_units),
                "paired_workday_season_count": int(len(paired_workday_seasons)),
                "total_valid_days": int(round(total_valid)),
                "valid_day_rate": float(valid_rate),
                "sample_energy": sample_energy,
                "has_two_years": bool(has_two_years),
                "trend_eligible": bool(trend_eligible),
                "trend_status": trend_status,
            }
        )

    ranking = pd.DataFrame.from_records(records)
    ranking = ranking.sort_values(
        [
            industry_col,
            "paired_unit_count",
            "total_valid_days",
            "valid_day_rate",
            "sample_energy",
            "_user_sort",
        ],
        ascending=[True, False, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking["rank_within_industry"] = (
        ranking.groupby(industry_col, sort=False).cumcount() + 1
    )
    ranking["selected"] = ranking["rank_within_industry"].eq(1)
    ranking["selection_reason"] = np.where(
        ranking["selected"],
        "highest_quality_by_approved_five_key_order",
        "lower_quality_rank",
    )
    ranking = ranking.drop(columns="_user_sort")
    selected = ranking.loc[ranking["selected"]].reset_index(drop=True).copy()

    internal_locks = build_industry_day_type_locks(
        candidate_day_type_decisions,
        selected,
        industry_col=industry_col,
        user_col=user_col,
        season_col="_season",
    )
    internal_mapping = build_final_day_type_mapping(
        internal_locks,
        industry_col=industry_col,
        season_col="_season",
    )
    final_cell_quality = apply_day_type_decisions(
        data,
        internal_locks,
        decision_level="industry",
        industry_col=industry_col,
        user_col=user_col,
        year_col=year_col,
        season_col="_season",
        day_type_col=day_type_col,
        sum_columns=(valid_days_col, expected_days_col, sample_energy_col),
    )

    def public_season(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.rename(columns={"_season": season_col}).copy()
        result[season_col] = result[season_col].map(_display_season)
        return result

    return RepresentativeSelection(
        ranking=ranking,
        selected=selected,
        candidate_day_type_decisions=public_season(candidate_day_type_decisions),
        industry_day_type_locks=public_season(internal_locks),
        final_day_type_mapping=public_season(internal_mapping),
        final_cell_quality=public_season(final_cell_quality),
    )


def build_industry_parameter_trace(
    candidates: pd.DataFrame,
    official_energy_2025: pd.DataFrame,
    *,
    industry_col: str = "industry",
    broad_sector_col: str = "broad_sector",
    energy_col: str = "official_energy_2025",
    valid_col: str = "parameter_valid",
) -> IndustryParameterTrace:
    """Validate and record that every fine industry uses its own parameters."""

    _require_columns(candidates, (industry_col, broad_sector_col), "candidates")
    _require_columns(official_energy_2025, (industry_col, energy_col), "official_energy_2025")
    if candidates.empty:
        raise ValueError("candidates must not be empty")

    candidate_data = candidates.copy()
    candidate_data[industry_col] = candidate_data[industry_col].astype(str).str.strip()
    candidate_data[broad_sector_col] = candidate_data[broad_sector_col].astype(str).str.strip()
    if candidate_data[industry_col].eq("").any() or candidate_data[broad_sector_col].eq("").any():
        raise ValueError("industry and broad_sector must be non-empty")
    if candidate_data[industry_col].duplicated().any():
        duplicate = sorted(candidate_data.loc[candidate_data[industry_col].duplicated(), industry_col].unique())
        raise ValueError(f"candidates contains duplicate industries: {duplicate}")
    if valid_col not in candidate_data:
        candidate_data[valid_col] = True
    valid_marker = candidate_data[valid_col]
    if pd.api.types.is_bool_dtype(valid_marker.dtype):
        candidate_data[valid_col] = valid_marker.fillna(False)
    else:
        candidate_data[valid_col] = (
            valid_marker.astype("string")
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes", "y", "是"})
        )
    count_columns = [
        column
        for column in ("estimated_cell_count", "frozen_cell_count")
        if column in candidate_data.columns
    ]
    if count_columns and len(count_columns) != 2:
        raise ValueError("cell parameter trace requires both estimated and frozen counts")
    optional_total = ["total_cell_count"] if "total_cell_count" in candidate_data.columns else []
    candidate_data = candidate_data[
        [industry_col, broad_sector_col, valid_col, *count_columns, *optional_total]
    ].copy()
    if count_columns:
        for column in count_columns:
            candidate_data[column] = pd.to_numeric(
                candidate_data[column], errors="coerce"
            )
        if optional_total:
            candidate_data["total_cell_count"] = pd.to_numeric(
                candidate_data["total_cell_count"], errors="coerce"
            )
            expected_total = candidate_data["total_cell_count"]
        else:
            expected_total = pd.Series(8, index=candidate_data.index, dtype=float)
        if (
            candidate_data[count_columns].isna().any().any()
            or (candidate_data[count_columns] < 0).any().any()
            or expected_total.isna().any()
            or (expected_total <= 0).any()
            or not candidate_data[count_columns].sum(axis=1).eq(expected_total).all()
        ):
            raise ValueError(
                "estimated and frozen cell counts must be non-negative and sum to total cells"
            )
        candidate_data[count_columns] = candidate_data[count_columns].astype(int)
        if optional_total:
            candidate_data["total_cell_count"] = candidate_data[
                "total_cell_count"
            ].astype(int)

    energy = official_energy_2025[[industry_col, energy_col]].copy()
    energy[industry_col] = energy[industry_col].astype(str).str.strip()
    if energy[industry_col].duplicated().any():
        duplicate = sorted(energy.loc[energy[industry_col].duplicated(), industry_col].unique())
        raise ValueError(f"official energy contains duplicate industries: {duplicate}")
    energy[energy_col] = pd.to_numeric(energy[energy_col], errors="coerce")

    trace = candidate_data.merge(energy, on=industry_col, how="left", validate="one_to_one")
    missing = trace.loc[trace[energy_col].isna(), industry_col].tolist()
    if missing:
        raise ValueError(f"valid parameter candidates missing official 2025 energy: {sorted(missing)}")
    invalid_energy = trace.loc[
        ~np.isfinite(trace[energy_col]) | (trace[energy_col] < 0),
        industry_col,
    ].tolist()
    if invalid_energy:
        raise ValueError(f"official 2025 energy must be finite and non-negative: {sorted(invalid_energy)}")

    trace["uses_own_parameters"] = True
    if count_columns:
        trace["parameter_status"] = np.select(
            [
                trace["frozen_cell_count"].eq(0),
                trace["estimated_cell_count"].eq(0),
            ],
            ["own_industry_all_cells_estimated", "own_industry_all_cells_frozen_zero"],
            default="own_industry_partial_cell_freeze",
        )
    else:
        trace["parameter_status"] = np.where(
            trace[valid_col],
            "own_industry_estimated",
            "own_industry_frozen_zero",
        )
    return IndustryParameterTrace(trace=trace.reset_index(drop=True))


__all__ = [
    "DEFAULT_SEASONS",
    "IndustryParameterTrace",
    "RepresentativeSelection",
    "build_industry_parameter_trace",
    "rank_representative_users",
]
