"""Auditable fine-industry shape judgement and parameter pipeline."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import (
    ANCHOR_REPAIR_ENABLED,
    FOUR_WINDOW_FINAL_SHARE_POLICY,
    LATE_PEAK_CONTROL_POLICY,
    PEAK_DAMPING_ENABLED,
    PEAK_VALLEY_EXTRAPOLATION_ENABLED,
    PEAK_VALLEY_GATE_POLICY,
    PEAK_VALLEY_RATE_TOLERANCE,
    PEAK_VALLEY_SUPPORTED_PHIS,
    SHAPE_EXTRAPOLATION_METHOD,
)
from .calendar_utils import calendar_frame
from .baseline_shapes import weighted_medoid
from .day_types import (
    HOLIDAY,
    MERGED_NONWORKDAY,
    WEEKEND,
    WORKDAY,
    canonical_day_type,
)
from .quality import SHAPE_COLS
from .representative_selection import (
    IndustryParameterTrace,
    RepresentativeSelection,
    build_industry_parameter_trace,
    rank_representative_users,
)
from .shape_parameters import (
    CONTROL_RATIO,
    SUPPORTED_HORIZONS,
    ShapeParameterEstimate,
    constrain_window_migration,
    extrapolated_increment,
    forecast_parameter_trajectory,
    forecast_peak_valley_target,
    identify_shape_parameters,
    judge_control_period_change,
    judge_evening_peak_position,
)
from .shape_transform import (
    apply_proportional_window_shares,
    apply_sequential_shape_transform,
    diagnose_shape,
    peak_valley_rate,
    window_shares,
)


SEASONS = ("春季", "夏季", "秋季", "冬季")
DAY_GROUPS = ("工作日", "非工作日")
ADAPTIVE_DAY_GROUPS = ("工作日", "周末", "节假日", "合并非工作日")
DAY_THRESHOLDS = {
    "工作日": 20,
    "非工作日": 8,
    "周末": 8,
    "节假日": 8,
    "合并非工作日": 8,
}


def _display_day_type(value: object) -> str:
    canonical = canonical_day_type(value)
    return {
        WORKDAY: "工作日",
        WEEKEND: "周末",
        HOLIDAY: "节假日",
        MERGED_NONWORKDAY: "合并非工作日",
    }[canonical]


def _build_user_cell_quality_legacy(
    records: pd.DataFrame, years: tuple[int, int] = (2022, 2023)
) -> pd.DataFrame:
    source_column = "day_type" if "day_type" in records.columns else "day_group"
    if source_column not in records.columns:
        raise ValueError("行业日数据缺少day_type/day_group")
    prepared = records.loc[records.year.isin(years)].copy()
    prepared["_source_day_type"] = prepared[source_column].map(canonical_day_type)
    legacy = set(prepared._source_day_type).issubset({WORKDAY, MERGED_NONWORKDAY})
    expected_rows = []
    for year in years:
        calendar = calendar_frame(year).copy()
        if legacy:
            calendar["_source_day_type"] = np.where(
                calendar.canonical_day_type.eq(WORKDAY), WORKDAY, MERGED_NONWORKDAY
            )
        else:
            calendar["_source_day_type"] = calendar.canonical_day_type
        expected = calendar.groupby(["season", "_source_day_type"]).size()
        for (season, day_type), count in expected.items():
            expected_rows.append(
                {
                    "year": year,
                    "season": season,
                    "day_type": day_type,
                    "expected_days": int(count),
                }
            )
    expected = pd.DataFrame(expected_rows)
    grouped = (
        prepared.groupby(
            ["industry", "user_key", "year", "season", "_source_day_type"],
            as_index=False,
        )
        .agg(valid_days=("date", "nunique"), sample_energy=("sample_energy", "sum"))
        .rename(columns={"user_key": "user_id", "_source_day_type": "day_type"})
    )
    users = (
        records[["industry", "user_key"]]
        .drop_duplicates()
        .rename(columns={"user_key": "user_id"})
    )
    grid = users.merge(
        pd.MultiIndex.from_product(
            [years, SEASONS, tuple(expected.day_type.unique())],
            names=["year", "season", "day_type"],
        ).to_frame(index=False),
        how="cross",
    )
    result = (
        grid.merge(
            grouped,
            on=["industry", "user_id", "year", "season", "day_type"],
            how="left",
        )
        .merge(
            expected,
            on=["year", "season", "day_type"],
            how="left",
            validate="many_to_one",
        )
    )
    result[["valid_days", "sample_energy"]] = result[["valid_days", "sample_energy"]].fillna(0.0)
    result["valid_days"] = result.valid_days.astype(int)
    return result


def _build_industry_cell_quality(
    industry_daily: pd.DataFrame,
    years: tuple[int, int] = (2022, 2023),
) -> pd.DataFrame:
    """Build adaptive day-type coverage from observed industry-days."""

    frame = industry_daily.copy()
    if "department" not in frame.columns and "broad_sector" in frame.columns:
        frame["department"] = frame["broad_sector"]
    source_column = next(
        (
            column
            for column in ("day_type_detail", "day_type", "final_day_type", "day_group")
            if column in frame.columns
        ),
        None,
    )
    required = {"industry", "year", "season", "date"}
    if source_column is None:
        required.add("day_type_detail")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"行业日数据缺少字段：{missing}")
    if len(years) != 2 or int(years[0]) == int(years[1]):
        raise ValueError("years必须包含两个不同年份")

    frame = frame.loc[frame.year.isin(years)].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["source_day_type"] = frame[source_column].map(canonical_day_type)
    duplicate_key = ["industry", "date"]
    if frame.duplicated(duplicate_key).any():
        duplicate = frame.loc[
            frame.duplicated(duplicate_key, keep=False), duplicate_key
        ].head(10)
        raise ValueError(f"行业日聚合记录重复：{duplicate.to_dict('records')}")

    if "department" in frame.columns:
        department_counts = frame.groupby("industry", observed=True).department.nunique()
        conflicts = department_counts.loc[department_counts.ne(1)]
        if not conflicts.empty:
            raise ValueError(f"行业存在多个产业大类：{conflicts.index.tolist()}")
        department_lookup = (
            frame.drop_duplicates("industry").set_index("industry").department.to_dict()
        )
    else:
        department_lookup = {}

    industries = tuple(sorted(frame.industry.astype(str).unique()))
    rows: list[dict[str, object]] = []
    for industry in industries:
        industry_part = frame.loc[frame.industry.astype(str).eq(industry)]
        for season in SEASONS:
            season_part = industry_part.loc[industry_part.season.astype(str).eq(season)]
            holiday_counts = {
                int(year): int(
                    season_part.loc[
                        season_part.year.eq(year)
                        & season_part.source_day_type.eq(HOLIDAY),
                        "date",
                    ].nunique()
                )
                for year in years
            }
            has_legacy_merged = bool(
                season_part.source_day_type.eq(MERGED_NONWORKDAY).any()
            )
            merge_holiday = bool(
                has_legacy_merged
                or any(count < DAY_THRESHOLDS["节假日"] for count in holiday_counts.values())
            )
            final_types = (
                (WORKDAY, MERGED_NONWORKDAY)
                if merge_holiday
                else (WORKDAY, WEEKEND, HOLIDAY)
            )
            merge_reason = (
                "legacy_nonworkday_input"
                if has_legacy_merged
                else (
                    "holiday_valid_days_below_8_in_either_year"
                    if merge_holiday
                    else "holiday_valid_days_at_least_8_in_both_years"
                )
            )

            counts: dict[tuple[int, str], int] = {}
            cell_parts: dict[tuple[int, str], pd.DataFrame] = {}
            for year in years:
                year_part = season_part.loc[season_part.year.eq(year)]
                for final_day_type in final_types:
                    if final_day_type == MERGED_NONWORKDAY:
                        member_types = {WEEKEND, HOLIDAY, MERGED_NONWORKDAY}
                    else:
                        member_types = {final_day_type}
                    cell_part = year_part.loc[
                        year_part.source_day_type.isin(member_types)
                    ]
                    key = (int(year), final_day_type)
                    counts[key] = int(cell_part.date.nunique())
                    cell_parts[key] = cell_part

            for final_day_type in final_types:
                threshold = DAY_THRESHOLDS[_display_day_type(final_day_type)]
                paired_eligible = all(
                    counts[(int(year), final_day_type)] >= threshold for year in years
                )
                for year in years:
                    part = cell_parts[(int(year), final_day_type)]
                    valid_days = counts[(int(year), final_day_type)]
                    partial_days = (
                        int(part.partial_composition_day.fillna(False).astype(bool).sum())
                        if "partial_composition_day" in part.columns
                        else 0
                    )
                    source_user_count = (
                        int(part.source_user_count.max())
                        if "source_user_count" in part.columns and not part.empty
                        else 0
                    )
                    min_active = (
                        int(part.active_user_count.min())
                        if "active_user_count" in part.columns and not part.empty
                        else 0
                    )
                    max_active = (
                        int(part.active_user_count.max())
                        if "active_user_count" in part.columns and not part.empty
                        else 0
                    )
                    rows.append(
                        {
                            "department": department_lookup.get(industry, ""),
                            "broad_sector": department_lookup.get(industry, ""),
                            "industry": industry,
                            "year": int(year),
                            "season": season,
                            "final_day_type": final_day_type,
                            "day_type": final_day_type,
                            "day_group": final_day_type,
                            "valid_days": valid_days,
                            "minimum_days": threshold,
                            "year_threshold_met": valid_days >= threshold,
                            "valid_days_2022": counts.get((2022, final_day_type), 0),
                            "valid_days_2023": counts.get((2023, final_day_type), 0),
                            "trend_eligible": paired_eligible,
                            "sample_sufficient": paired_eligible,
                            "trend_status": (
                                "paired_2022_2023_industry_days"
                                if paired_eligible
                                else "insufficient_history_no_extrapolation"
                            ),
                            "holiday_merged_with_weekend": merge_holiday,
                            "merge_reason": merge_reason,
                            "source_day_types": (
                                f"{WEEKEND}|{HOLIDAY}"
                                if final_day_type == MERGED_NONWORKDAY
                                else final_day_type
                            ),
                            "source_user_count": source_user_count,
                            "min_active_user_count": min_active,
                            "max_active_user_count": max_active,
                            "partial_composition_days": partial_days,
                            "partial_composition_rate": (
                                partial_days / valid_days if valid_days else 0.0
                            ),
                            # Composition changes are disclosed here and are
                            # deliberately absent from paired_eligible.
                            "user_composition_changed": bool(partial_days > 0),
                        }
                    )
    columns = [
        "department",
        "broad_sector",
        "industry",
        "year",
        "season",
        "final_day_type",
        "day_type",
        "day_group",
        "valid_days",
        "minimum_days",
        "year_threshold_met",
        "valid_days_2022",
        "valid_days_2023",
        "trend_eligible",
        "sample_sufficient",
        "trend_status",
        "holiday_merged_with_weekend",
        "merge_reason",
        "source_day_types",
        "source_user_count",
        "min_active_user_count",
        "max_active_user_count",
        "partial_composition_days",
        "partial_composition_rate",
        "user_composition_changed",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["industry", "season", "final_day_type", "year"], kind="mergesort"
    ).reset_index(drop=True)


def build_cell_quality(
    records: pd.DataFrame,
    years: tuple[int, int] = (2022, 2023),
) -> pd.DataFrame:
    """Build coverage at industry-day grain; retain the old user-grid API.

    ``aggregate_industry_daily`` deliberately removes ``user_key``.  Its
    presence therefore provides an unambiguous compatibility dispatch for
    older callers while the formal 3.5.0 path always takes the industry-day
    branch.
    """

    if "user_key" in records.columns:
        return _build_user_cell_quality_legacy(records, years=years)
    return _build_industry_cell_quality(records, years=years)


def _mean_cell_curves(records: pd.DataFrame, industry: str, user_id: str) -> pd.DataFrame:
    selected = records.loc[
        records.industry.eq(industry) & records.user_key.eq(user_id)
    ]
    rows: list[dict[str, object]] = []
    group_column = "final_day_type" if "final_day_type" in selected.columns else "day_group"
    for (year, season, day_group), part in selected.groupby(
        ["year", "season", group_column], sort=True
    ):
        curve = part[SHAPE_COLS].mean(axis=0).to_numpy(dtype=float, copy=True)
        curve /= curve.sum()
        rows.append(
            {
                "year": int(year),
                "season": season,
                "day_group": day_group,
                "valid_days": len(part),
                "curve": curve,
            }
        )
    return pd.DataFrame(rows)


def _apply_locked_day_types(
    records: pd.DataFrame, mapping: pd.DataFrame
) -> pd.DataFrame:
    required = {"industry", "season", "source_day_type", "final_day_type"}
    if not required.issubset(mapping.columns):
        raise ValueError(f"行业日型映射缺少字段：{sorted(required - set(mapping.columns))}")
    frame = records.copy()
    source_column = "day_type" if "day_type" in frame.columns else "day_group"
    frame["source_day_type"] = frame[source_column].map(canonical_day_type)
    keys = ["industry", "season", "source_day_type"]
    lookup = mapping[keys + ["final_day_type"]].drop_duplicates()
    if lookup.duplicated(keys).any():
        raise ValueError("行业日型映射键重复")
    frame = frame.merge(lookup, on=keys, how="left", validate="many_to_one")
    if frame.final_day_type.isna().any():
        missing = frame.loc[frame.final_day_type.isna(), keys].drop_duplicates()
        raise ValueError(f"行业日数据缺少最终日型映射：{missing.to_dict('records')[:10]}")
    frame["final_day_type"] = frame.final_day_type.map(_display_day_type)
    return frame


def _all_user_cell_curves(records: pd.DataFrame) -> pd.DataFrame:
    keys = ["industry", "year", "season", "final_day_type", "user_key"]
    rows: list[dict[str, object]] = []
    for key, part in records.groupby(keys, observed=True, sort=True):
        curve = part[SHAPE_COLS].mean(axis=0).to_numpy(dtype=float, copy=True)
        curve /= curve.sum()
        rows.append(
            {
                **dict(zip(keys, key)),
                "valid_days": int(part.date.nunique()),
                "sample_energy": float(part.sample_energy.sum()),
                "curve": curve,
            }
        )
    return pd.DataFrame(rows)


def _traditional_mapping(sample_mapping: pd.DataFrame) -> pd.DataFrame:
    required = {"industry", "broad_sector"}
    if not required.issubset(sample_mapping.columns):
        raise ValueError(f"样本映射缺少字段：{sorted(required - set(sample_mapping.columns))}")
    mapping = sample_mapping.copy()
    if "is_traditional" in mapping.columns:
        marker = mapping["is_traditional"]
        if pd.api.types.is_bool_dtype(marker.dtype):
            traditional = marker.fillna(False)
        else:
            traditional = (
                marker.astype("string")
                .str.strip()
                .str.lower()
                .isin({"true", "1", "yes", "y", "是"})
            )
        mapping = mapping.loc[traditional].copy()
    mapping = mapping[["industry", "broad_sector"]].drop_duplicates()
    if mapping.industry.duplicated().any():
        duplicate = sorted(mapping.loc[mapping.industry.duplicated(False), "industry"].unique())
        raise ValueError(f"样本行业存在多个产业大类：{duplicate}")
    if mapping.empty or mapping[["industry", "broad_sector"]].astype(str).eq("").any().any():
        raise ValueError("传统样本行业映射不能为空")
    return mapping


def _format_runs(runs: Sequence[Sequence[int]]) -> str:
    return ";".join("→".join(f"{int(hour):02d}" for hour in run) for run in runs)


def _evening_target_by_season(
    records: pd.DataFrame,
    representative: RepresentativeSelection,
) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, object]]]:
    detail_frames: list[pd.DataFrame] = []
    targets: dict[tuple[str, str], dict[str, object]] = {}
    for selected in representative.selected.itertuples(index=False):
        industry = str(selected.industry)
        user_id = str(selected.user_id)
        user_days = records.loc[
            records.industry.astype(str).eq(industry)
            & records.user_key.astype(str).eq(user_id)
        ]
        judgement, detail = judge_evening_peak_position(user_days, industry=industry)
        if detail.empty:
            continue
        detail = detail.assign(
            selected_representative_user=user_id,
            evening_judge_enabled=judgement.enabled,
            evening_judge_summary=judgement.reason,
        )
        detail_frames.append(detail)
        for hit in detail.loc[detail.month_triggered].itertuples(index=False):
            targets[(industry, str(hit.season))] = {
                "target_hour": int(hit.compare_typical_peak_hour),
                "reference_hour": int(hit.ref_peak_hour),
                "month": int(hit.month),
                "moved_share": float(hit.moved_share),
                "moved_mode_hour": int(hit.moved_mode_hour),
            }
    return (
        pd.concat(detail_frames, ignore_index=True, sort=False)
        if detail_frames
        else pd.DataFrame(),
        targets,
    )


def _identify_industry_parameters_legacy(
    records: pd.DataFrame,
    sample_mapping: pd.DataFrame,
    official_energy_2025: pd.DataFrame,
    *,
    control_ratio: float = CONTROL_RATIO,
) -> tuple[
    RepresentativeSelection,
    IndustryParameterTrace,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[tuple[str, str, str], ShapeParameterEstimate],
]:
    """Identify one independent parameter set for every industry shape cell."""

    mapping = _traditional_mapping(sample_mapping)
    mapped = records.merge(mapping, on="industry", how="left", validate="many_to_one")
    if mapped.broad_sector.isna().any():
        missing = sorted(mapped.loc[mapped.broad_sector.isna(), "industry"].unique())
        raise ValueError(f"样本行业未映射：{missing}")
    cell_quality = build_cell_quality(mapped)
    representative = rank_representative_users(
        cell_quality,
        required_workday_seasons=SEASONS,
        day_type_col="day_type",
    )

    estimates: dict[tuple[str, str, str], ShapeParameterEstimate] = {}
    history_rows: list[dict[str, object]] = []
    evening_frames: list[pd.DataFrame] = []
    judgement_rows: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    sector_lookup = mapping.set_index("industry").broad_sector.to_dict()

    for selected in representative.selected.itertuples(index=False):
        industry = str(selected.industry)
        user_id = str(selected.user_id)
        broad_sector = str(sector_lookup[industry])
        cells = _mean_cell_curves(mapped, industry, user_id)
        c22 = cells.loc[cells.year.eq(2022)].set_index(["season", "day_group"])
        c23 = cells.loc[cells.year.eq(2023)].set_index(["season", "day_group"])

        usable: list[tuple[str, str]] = []
        for season in SEASONS:
            for day_group in DAY_GROUPS:
                key = (season, day_group)
                threshold = 20 if day_group == "工作日" else 8
                if (
                    key in c22.index
                    and key in c23.index
                    and int(c22.loc[key].valid_days) >= threshold
                    and int(c23.loc[key].valid_days) >= threshold
                ):
                    usable.append(key)

        user_days = mapped.loc[
            mapped.industry.eq(industry) & mapped.user_key.eq(user_id)
        ]
        evening_judgement, evening_detail = judge_evening_peak_position(
            user_days, industry=industry
        )
        if not evening_detail.empty:
            evening_frames.append(
                evening_detail.assign(industry=industry, selected_user=user_id)
            )
        evening_seasons = set(evening_judgement.triggered_seasons)
        usable_keys = set(usable)
        estimated_cell_count = 0

        for season in SEASONS:
            for day_group in DAY_GROUPS:
                key = (season, day_group)
                has_22 = key in c22.index
                has_23 = key in c23.index
                base_days = int(c22.loc[key].valid_days) if has_22 else 0
                compare_days = int(c23.loc[key].valid_days) if has_23 else 0
                sufficient = key in usable_keys
                evening_hit = bool(sufficient and season in evening_seasons)
                if sufficient:
                    control = judge_control_period_change(
                        c22.loc[key].curve,
                        c23.loc[key].curve,
                        ratio=control_ratio,
                    )
                    control_hit = bool(control["control_triggered"])
                    change_rate = float(control["control_change_rate"])
                    control_hours = ",".join(
                        f"{hour}时" for hour in control["control_hours"]
                    )
                else:
                    control_hit = False
                    change_rate = float("nan")
                    control_hours = ""
                changed = bool(sufficient and (control_hit or evening_hit))
                if sufficient:
                    source_year = 2023 if changed else 2022
                elif has_23:
                    source_year = 2023
                elif has_22:
                    source_year = 2022
                else:
                    source_year = 2023

                if sufficient:
                    estimate = identify_shape_parameters(
                        np.asarray(c22.loc[key].curve, dtype=float),
                        np.asarray(c23.loc[key].curve, dtype=float),
                        weights=np.asarray(
                            [min(c22.loc[key].valid_days, c23.loc[key].valid_days)],
                            dtype=float,
                        ),
                        trend_eligible=True,
                    )
                    estimated_cell_count += 1
                elif has_23:
                    estimate = identify_shape_parameters(
                        np.asarray(c23.loc[key].curve, dtype=float),
                        None,
                        trend_eligible=False,
                    )
                elif has_22:
                    estimate = identify_shape_parameters(
                        np.asarray(c22.loc[key].curve, dtype=float),
                        None,
                        trend_eligible=False,
                    )
                else:
                    raise ValueError(
                        f"代表用户单元没有2022/2023形状：{industry}/{season}/{day_group}"
                    )
                estimate_key = (industry, season, day_group)
                estimates[estimate_key] = estimate

                if not sufficient:
                    reason = "样本不足，判别不可评估"
                elif control_hit and evening_hit:
                    reason = "控制时段变化率与晚峰后移均触发"
                elif control_hit:
                    reason = "控制时段负荷变化率超过10%"
                elif evening_hit:
                    reason = "晚峰后移触发"
                else:
                    reason = "两条判据均未触发，锚定2022但继续迁移"
                judgement_rows.append(
                    {
                        "industry": industry,
                        "broad_sector": broad_sector,
                        "selected_user": user_id,
                        "season": season,
                        "day_group": day_group,
                        "base_days": base_days,
                        "compare_days": compare_days,
                        "control_hours": control_hours,
                        "control_change_rate": change_rate,
                        "control_triggered": control_hit,
                        "evening_judge_enabled": evening_judgement.enabled,
                        "evening_judge_reason": evening_judgement.reason,
                        "evening_triggered": evening_hit,
                        "sample_sufficient": sufficient,
                        "trend_eligible": sufficient,
                        "trend_status": (
                            "short_history_exploratory"
                            if sufficient
                            else "insufficient_history_no_extrapolation"
                        ),
                        "judgement_status": (
                            "evaluable" if sufficient else "judgement_not_evaluable"
                        ),
                        "changed": changed,
                        "source_year": source_year,
                        "reason": reason,
                    }
                )

                shares_2022 = (
                    window_shares(np.asarray(c22.loc[key].curve, dtype=float))
                    if has_22
                    else np.full(4, np.nan)
                )
                shares_2023 = (
                    window_shares(np.asarray(c23.loc[key].curve, dtype=float))
                    if has_23
                    else np.full(4, np.nan)
                )
                window = estimate.window
                history_rows.append(
                    {
                        "industry": industry,
                        "broad_sector": broad_sector,
                        "selected_user": user_id,
                        "season": season,
                        "day_group": day_group,
                        "base_days": base_days,
                        "compare_days": compare_days,
                        "paired_cell": sufficient,
                        "trend_eligible": estimate.trend_eligible,
                        "trend_status": estimate.trend_status,
                        "night_share_2022": shares_2022[0],
                        "morning_share_2022": shares_2022[1],
                        "midday_share_2022": shares_2022[2],
                        "evening_share_2022": shares_2022[3],
                        "night_share_2023": shares_2023[0],
                        "morning_share_2023": shares_2023[1],
                        "midday_share_2023": shares_2023[2],
                        "evening_share_2023": shares_2023[3],
                        "b_night": window.b_observed[0],
                        "b_morning": window.b_observed[1],
                        "b_midday": window.b_observed[2],
                        "b_evening": window.b_observed[3],
                        "beta_a": estimate.beta_a,
                        "v_night": window.direction_v[0],
                        "v_morning": window.direction_v[1],
                        "v_midday": window.direction_v[2],
                        "v_evening": window.direction_v[3],
                    }
                )

        candidates.append(
            {
                "industry": industry,
                "broad_sector": broad_sector,
                "parameter_valid": estimated_cell_count == len(SEASONS) * len(DAY_GROUPS),
                "estimated_cell_count": estimated_cell_count,
                "frozen_cell_count": len(SEASONS) * len(DAY_GROUPS) - estimated_cell_count,
            }
        )

    parameter_trace = build_industry_parameter_trace(
        pd.DataFrame(candidates), official_energy_2025
    )
    history = pd.DataFrame(history_rows).merge(
        official_energy_2025[["industry", "official_energy_2025"]],
        on="industry",
        how="left",
        validate="many_to_one",
    )
    expected_cells = len(mapping) * len(SEASONS) * len(DAY_GROUPS)
    if len(history) != expected_cells or len(estimates) != expected_cells:
        raise AssertionError(
            f"行业单元参数数量错误：history={len(history)}, estimates={len(estimates)}, "
            f"expected={expected_cells}"
        )
    history_key = ["industry", "season", "day_group"]
    if history.duplicated(history_key).any():
        duplicate = history.loc[
            history.duplicated(history_key, keep=False), history_key
        ].to_dict("records")
        raise AssertionError(f"行业历史参数单元重复：{duplicate[:10]}")
    history = history.sort_values(history_key, kind="mergesort").reset_index(drop=True)
    evening = (
        pd.concat(evening_frames, ignore_index=True)
        if evening_frames
        else pd.DataFrame()
    )
    judgement = build_industry_cell_judgement(
        pd.DataFrame(judgement_rows), expected_industries=mapping.industry.tolist()
    )
    return representative, parameter_trace, history, evening, judgement, estimates


def identify_industry_parameters(
    records: pd.DataFrame,
    sample_mapping: pd.DataFrame,
    official_energy_2025: pd.DataFrame,
    *,
    control_ratio: float = CONTROL_RATIO,
) -> tuple[
    RepresentativeSelection,
    IndustryParameterTrace,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[tuple[str, str, str], ShapeParameterEstimate],
]:
    """Identify adaptive cells and build the adjusted-2022 stage-5 baseline.

    Representative-user selection and four-window migration remain separate
    from the baseline medoid.  The local control/evening transaction always
    starts from a 2022 medoid and may use only the same medoid source user's
    2023 curve as target evidence; it never substitutes the complete 2023
    curve and is not represented as a future extrapolation parameter.
    """

    # Compatibility-only 3.2 path.  Keeping this import local guarantees that
    # the formal 3.3 industry-day chain cannot accidentally execute local
    # control/evening shape adjustment.
    from .local_shape_adjustment import adjust_local_shape

    if not np.isclose(float(control_ratio), CONTROL_RATIO, rtol=0.0, atol=1e-12):
        raise ValueError(f"现行控制时段比例固定为{CONTROL_RATIO}")
    legacy_two_day_output = bool(
        "day_type" not in records.columns
        and "day_group" in records.columns
        and set(records.day_group.dropna().map(canonical_day_type)).issubset(
            {WORKDAY, MERGED_NONWORKDAY}
        )
    )
    mapping = _traditional_mapping(sample_mapping)
    mapped = records.merge(mapping, on="industry", how="left", validate="many_to_one")
    if mapped.broad_sector.isna().any():
        missing = sorted(mapped.loc[mapped.broad_sector.isna(), "industry"].unique())
        raise ValueError(f"样本行业未映射：{missing}")

    quality = build_cell_quality(mapped)
    representative = rank_representative_users(
        quality,
        required_workday_seasons=SEASONS,
        day_type_col="day_type",
    )
    if representative.final_day_type_mapping.empty:
        raise ValueError("代表用户选择没有生成行业最终日型映射")
    mapped_days = _apply_locked_day_types(
        mapped, representative.final_day_type_mapping
    )
    user_curves = _all_user_cell_curves(mapped_days)
    evening_detail, evening_targets = _evening_target_by_season(mapped, representative)

    final_quality = representative.final_cell_quality.copy()
    final_quality["final_day_type"] = final_quality.final_day_type.map(_display_day_type)
    counts_lookup = {
        (
            str(row.industry),
            str(row.user_id),
            str(row.season),
            str(row.final_day_type),
            int(row.year),
        ): int(row.valid_days)
        for row in final_quality.itertuples(index=False)
    }
    sector_lookup = mapping.set_index("industry").broad_sector.to_dict()
    selected_lookup = representative.selected.set_index("industry").user_id.astype(str).to_dict()
    locks = representative.industry_day_type_locks.copy()
    lock_lookup = {
        (str(row.industry), str(row.season)): row
        for row in locks.itertuples(index=False)
    }

    estimates: dict[tuple[str, str, str], ShapeParameterEstimate] = {}
    history_rows: list[dict[str, object]] = []
    judgement_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []

    for industry in mapping.industry.astype(str):
        broad_sector = str(sector_lookup[industry])
        representative_user = str(selected_lookup[industry])
        estimated_cell_count = 0
        total_cell_count = 0
        for season in SEASONS:
            lock = lock_lookup[(industry, season)]
            if bool(lock.holiday_merged_with_weekend):
                final_day_types = ("工作日", "合并非工作日")
            else:
                final_day_types = ("工作日", "周末", "节假日")
            evening_evidence = evening_targets.get((industry, season))
            evening_target_hour = (
                int(evening_evidence["target_hour"])
                if evening_evidence is not None
                else None
            )

            for day_group in final_day_types:
                output_day_group = (
                    "非工作日"
                    if legacy_two_day_output and day_group == "合并非工作日"
                    else day_group
                )
                total_cell_count += 1
                cell = user_curves.loc[
                    user_curves.industry.astype(str).eq(industry)
                    & user_curves.season.astype(str).eq(season)
                    & user_curves.final_day_type.astype(str).eq(day_group)
                ]
                anchor_2022 = cell.loc[cell.year.eq(2022)].reset_index(drop=True)
                anchor_year = 2022
                anchor_fallback = False
                if anchor_2022.empty:
                    anchor_2022 = cell.loc[cell.year.eq(2023)].reset_index(drop=True)
                    anchor_year = 2023
                    anchor_fallback = True
                if anchor_2022.empty:
                    raise ValueError(
                        f"行业基准曲线单元无2022/2023候选：{industry}/{season}/{day_group}"
                    )
                base_curve, medoid_index, medoid_objective = weighted_medoid(
                    np.vstack(anchor_2022.curve.to_list()),
                    anchor_2022.valid_days.to_numpy(dtype=float),
                )
                medoid_source = anchor_2022.iloc[medoid_index]
                medoid_source_user = str(medoid_source.user_key)
                same_source_2023 = cell.loc[
                    cell.year.eq(2023)
                    & cell.user_key.astype(str).eq(medoid_source_user)
                ]
                target_curve = (
                    None
                    if same_source_2023.empty
                    else np.asarray(same_source_2023.iloc[0].curve, dtype=float)
                )
                same_source_days = (
                    0 if same_source_2023.empty else int(same_source_2023.iloc[0].valid_days)
                )

                base_days = counts_lookup.get(
                    (industry, representative_user, season, day_group, 2022), 0
                )
                compare_days = counts_lookup.get(
                    (industry, representative_user, season, day_group, 2023), 0
                )
                threshold = DAY_THRESHOLDS[day_group]
                sufficient = bool(
                    base_days >= threshold
                    and compare_days >= threshold
                    and anchor_year == 2022
                    and target_curve is not None
                )

                local = adjust_local_shape(
                    base_curve,
                    target_curve,
                    evening_triggered=evening_evidence is not None,
                    evening_target_hour=evening_target_hour,
                    sample_evaluable=sufficient,
                )
                local_diag = local.diagnostics
                control_diag = local_diag.control
                evening_diag = local_diag.evening

                rep_cell = cell.loc[cell.user_key.astype(str).eq(representative_user)]
                rep22 = rep_cell.loc[rep_cell.year.eq(2022)]
                rep23 = rep_cell.loc[rep_cell.year.eq(2023)]
                if not rep22.empty and not rep23.empty and sufficient:
                    estimate = identify_shape_parameters(
                        np.asarray(rep22.iloc[0].curve, dtype=float),
                        np.asarray(rep23.iloc[0].curve, dtype=float),
                        weights=np.asarray([min(base_days, compare_days)], dtype=float),
                        trend_eligible=True,
                    )
                    estimated_cell_count += 1
                elif not rep22.empty:
                    estimate = identify_shape_parameters(
                        np.asarray(rep22.iloc[0].curve, dtype=float),
                        None,
                        trend_eligible=False,
                    )
                elif not rep23.empty:
                    estimate = identify_shape_parameters(
                        np.asarray(rep23.iloc[0].curve, dtype=float),
                        None,
                        trend_eligible=False,
                    )
                else:
                    estimate = identify_shape_parameters(
                        base_curve, None, trend_eligible=False
                    )
                estimate_key = (industry, season, output_day_group)
                estimates[estimate_key] = estimate

                shares_2022 = (
                    window_shares(np.asarray(rep22.iloc[0].curve, dtype=float))
                    if not rep22.empty
                    else np.full(4, np.nan)
                )
                shares_2023 = (
                    window_shares(np.asarray(rep23.iloc[0].curve, dtype=float))
                    if not rep23.empty
                    else np.full(4, np.nan)
                )
                window = estimate.window
                history_rows.append(
                    {
                        "industry": industry,
                        "broad_sector": broad_sector,
                        "selected_user": representative_user,
                        "season": season,
                        "day_group": output_day_group,
                        "holiday_merged_with_weekend": bool(
                            lock.holiday_merged_with_weekend
                        ),
                        "base_days": base_days,
                        "compare_days": compare_days,
                        "paired_cell": sufficient,
                        "trend_eligible": estimate.trend_eligible,
                        "trend_status": estimate.trend_status,
                        "night_share_2022": shares_2022[0],
                        "morning_share_2022": shares_2022[1],
                        "midday_share_2022": shares_2022[2],
                        "evening_share_2022": shares_2022[3],
                        "night_share_2023": shares_2023[0],
                        "morning_share_2023": shares_2023[1],
                        "midday_share_2023": shares_2023[2],
                        "evening_share_2023": shares_2023[3],
                        "b_night": window.b_observed[0],
                        "b_morning": window.b_observed[1],
                        "b_midday": window.b_observed[2],
                        "b_evening": window.b_observed[3],
                        "beta_a": estimate.beta_a,
                        "v_night": window.direction_v[0],
                        "v_morning": window.direction_v[1],
                        "v_midday": window.direction_v[2],
                        "v_evening": window.direction_v[3],
                    }
                )

                control_requested = bool(control_diag.triggered)
                evening_requested = bool(
                    evening_evidence is not None
                    and sufficient
                    and not local_diag.flatness.is_flat
                    and evening_diag.source_hour is not None
                    and evening_target_hour is not None
                    and int(evening_diag.source_hour) < int(evening_target_hour)
                )
                changed = bool(control_diag.applied or evening_diag.applied)
                if anchor_fallback:
                    reason = "2022基准缺失，回退2023且两项局部变化不执行"
                elif not sufficient:
                    reason = "样本不足或同源2023目标缺失，局部变化不可评估"
                elif local_diag.flatness.is_flat:
                    reason = "平坦曲线，仅执行四时段变化"
                elif changed:
                    reason = "2022主体上的局部变化已通过事务式校验并提交"
                elif control_requested or evening_requested:
                    reason = "变化请求已触发但候选未通过完整提交校验"
                else:
                    reason = "两项局部变化均未触发，仅执行四时段变化"

                judgement_rows.append(
                    {
                        "industry": industry,
                        "broad_sector": broad_sector,
                        "selected_user": representative_user,
                        "season": season,
                        "day_group": output_day_group,
                        "holiday_merged_with_weekend": bool(
                            lock.holiday_merged_with_weekend
                        ),
                        "merge_reason": str(lock.merge_reason),
                        "base_days": base_days,
                        "compare_days": compare_days,
                        "same_source_days_2023": same_source_days,
                        "sample_sufficient": sufficient,
                        "trend_eligible": sufficient,
                        "trend_status": (
                            "short_history_exploratory"
                            if sufficient
                            else "insufficient_history_no_extrapolation"
                        ),
                        "judgement_status": (
                            "evaluable" if sufficient else "judgement_not_evaluable"
                        ),
                        "source_year": anchor_year,
                        "anchor_fallback": anchor_fallback,
                        "medoid_source_user_2022": medoid_source_user,
                        "medoid_source_days_2022": int(medoid_source.valid_days),
                        "medoid_objective_l1": float(medoid_objective),
                        "load_factor_2022": local_diag.flatness.load_factor,
                        "cv_2022": local_diag.flatness.coefficient_of_variation,
                        "flat_curve": local_diag.flatness.is_flat,
                        "control_hours_2022": ",".join(
                            str(value) for value in control_diag.baseline_control.hours
                        ),
                        "control_hours_2023": ",".join(
                            str(value) for value in control_diag.target_hours
                        ),
                        "control_valid_runs_2022": _format_runs(
                            control_diag.baseline_control.valid_runs
                        ),
                        "control_valid_runs_2023": _format_runs(
                            control_diag.target_control.valid_runs
                            if control_diag.target_control is not None
                            else ()
                        ),
                        "peak_valley_rate_2022": control_diag.baseline_peak_valley_rate,
                        "peak_valley_rate_2023": control_diag.target_peak_valley_rate,
                        "peak_valley_abs_change": control_diag.absolute_peak_valley_change,
                        "control_duration_2022": control_diag.baseline_control.duration,
                        "control_duration_2023": (
                            control_diag.target_control.duration
                            if control_diag.target_control is not None
                            else np.nan
                        ),
                        "control_duration_abs_change": control_diag.absolute_duration_change,
                        "control_adjustment_requested": control_requested,
                        "control_adjustment_applied": control_diag.applied,
                        "control_status": control_diag.status,
                        "control_reason": control_diag.reason,
                        "evening_judge_enabled": industry
                        in {
                            "铁路运输业",
                            "航空运输业",
                            "电信、广播电视和卫星传输服务",
                            "互联网和相关服务",
                            "软件和信息技术服务业",
                            "学校",
                            "医院",
                            "居民生活",
                        },
                        "evening_formal_trigger": evening_evidence is not None,
                        "evening_trigger_month": (
                            int(evening_evidence["month"])
                            if evening_evidence is not None
                            else np.nan
                        ),
                        "evening_target_hour_2023": evening_target_hour,
                        "evening_adjustment_requested": evening_requested,
                        "evening_adjustment_applied": evening_diag.applied,
                        "evening_status": evening_diag.status,
                        "evening_reason": evening_diag.reason,
                        "local_adjustment_applied": local_diag.curve_changed,
                        "local_adjustment_status": local_diag.status,
                        "changed": changed,
                        "reason": reason,
                        **{
                            f"baseline_{column}": float(value)
                            for column, value in zip(SHAPE_COLS, local.curve, strict=True)
                        },
                    }
                )

        candidate_rows.append(
            {
                "industry": industry,
                "broad_sector": broad_sector,
                "parameter_valid": estimated_cell_count == total_cell_count,
                "estimated_cell_count": estimated_cell_count,
                "frozen_cell_count": total_cell_count - estimated_cell_count,
                "total_cell_count": total_cell_count,
            }
        )

    history = pd.DataFrame(history_rows).merge(
        official_energy_2025[["industry", "official_energy_2025"]],
        on="industry",
        how="left",
        validate="many_to_one",
    )
    judgement = build_industry_cell_judgement(
        pd.DataFrame(judgement_rows), expected_industries=mapping.industry.tolist()
    )
    expected_cells = len(judgement)
    if len(history) != expected_cells or len(estimates) != expected_cells:
        raise AssertionError(
            f"行业单元参数数量错误：history={len(history)}, estimates={len(estimates)}, "
            f"expected={expected_cells}"
        )
    parameter_trace = build_industry_parameter_trace(
        pd.DataFrame(candidate_rows), official_energy_2025
    )
    key = ["industry", "season", "day_group"]
    history = history.sort_values(key, kind="mergesort").reset_index(drop=True)
    return (
        representative,
        parameter_trace,
        history,
        evening_detail,
        judgement,
        estimates,
    )


def build_industry_cell_judgement(
    cell_judgement: pd.DataFrame,
    *,
    expected_industries: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Validate the complete industry/season/day-type judgement grid."""

    required = {
        "industry",
        "broad_sector",
        "season",
        "day_group",
        "sample_sufficient",
        "trend_eligible",
        "trend_status",
        "judgement_status",
        "evening_judge_enabled",
        "changed",
        "source_year",
        "reason",
    }
    if not required.issubset(cell_judgement.columns):
        raise ValueError(f"行业单元判别表缺少字段：{sorted(required - set(cell_judgement.columns))}")
    result = cell_judgement.copy()
    key = ["industry", "season", "day_group"]
    if result.duplicated(key).any():
        duplicate = result.loc[result.duplicated(key, keep=False), key].to_dict("records")
        raise ValueError(f"行业单元判别重复：{duplicate[:10]}")
    industries = (
        tuple(str(value) for value in expected_industries)
        if expected_industries is not None
        else tuple(sorted(result.industry.astype(str).unique()))
    )
    if len(industries) != len(set(industries)):
        raise ValueError("expected_industries包含重复值")
    observed_industries = set(result.industry.astype(str))
    if observed_industries != set(industries):
        raise ValueError(
            "行业单元判别行业集合不完整："
            f"missing={sorted(set(industries) - observed_industries)}, "
            f"extra={sorted(observed_industries - set(industries))}"
        )
    for industry in industries:
        part = result.loc[result.industry.astype(str).eq(industry)]
        for season in SEASONS:
            observed = set(
                part.loc[part.season.astype(str).eq(season), "day_group"].astype(str)
            )
            if observed not in (
                set(DAY_GROUPS),
                {"工作日", "合并非工作日"},
                {"工作日", "周末", "节假日"},
            ):
                raise ValueError(
                    f"行业单元判别日型组合非法：{industry}/{season}/{sorted(observed)}"
                )
    sector_count = result.groupby("industry").broad_sector.nunique()
    conflicts = sector_count.loc[sector_count.ne(1)].index.tolist()
    if conflicts:
        raise ValueError(f"行业判别存在多个产业大类：{conflicts}")
    return result.sort_values(key, kind="mergesort").reset_index(drop=True)


def _future_day_type_column(frame: pd.DataFrame, label: str) -> str:
    for column in ("final_day_type", "day_group"):
        if column in frame.columns:
            return column
    raise ValueError(f"{label}缺少final_day_type/day_group")


def _direction_from_history(row: pd.Series) -> np.ndarray:
    if "direction_v" in row.index:
        direction = np.asarray(row["direction_v"], dtype=float)
    else:
        required = ["v_night", "v_morning", "v_midday", "v_evening"]
        missing = [column for column in required if column not in row.index]
        if missing:
            raise ValueError(f"shape_history缺少迁移方向字段：{missing}")
        direction = row[required].to_numpy(dtype=float)
    if direction.shape != (4,) or not np.isfinite(direction).all():
        raise ValueError("shape_history迁移方向必须是有限四维向量")
    return direction


def _history_boolean(row: pd.Series, column: str) -> bool:
    """Read one mandatory in-memory shape-history boolean without truthy coercion."""

    value = row[column]
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"shape_history {column}必须为布尔值")
    return bool(value)


def _build_future_parameter_table_340(
    baseline_shapes: pd.DataFrame,
    shape_history: pd.DataFrame,
    years: Iterable[int],
) -> pd.DataFrame:
    """Generate 3.5.0 future shapes with the gated sequential transformation."""

    baseline = baseline_shapes.copy()
    baseline_day_type = _future_day_type_column(baseline, "行业基准形状")
    history = shape_history.copy()
    history_day_type = _future_day_type_column(history, "shape_history")
    required_baseline = {"industry", "season", *SHAPE_COLS}
    required_history = {
        "industry",
        "season",
        "beta_a",
        "trend_eligible",
        "peak_valley_rate_2023",
        "peak_valley_delta_signed",
        "flat_curve",
        "flat_borderline",
        "peak_valley_delta_abs",
        "peak_valley_delta_band",
        "peak_valley_evolution_enabled",
        "peak_valley_gate_reason",
        "peak_valley_gate_policy",
    }
    missing_baseline = sorted(required_baseline.difference(baseline.columns))
    missing_history = sorted(required_history.difference(history.columns))
    if missing_baseline:
        raise ValueError(f"行业基准形状缺少字段：{missing_baseline}")
    if missing_history:
        raise ValueError(f"shape_history缺少字段：{missing_history}")
    baseline["final_day_type"] = baseline[baseline_day_type].astype(str)
    history["final_day_type"] = history[history_day_type].astype(str)
    if "year" in baseline.columns:
        baseline = baseline.loc[baseline.year.astype(int).eq(2023)].copy()
    if "is_future_baseline" in baseline.columns:
        baseline = baseline.loc[baseline.is_future_baseline.astype(bool)].copy()
    key = ["industry", "season", "final_day_type"]
    if baseline.duplicated(key).any():
        raise ValueError("2023行业基准形状单元重复")
    if history.duplicated(key).any():
        raise ValueError("shape_history行业单元重复")
    missing_history_keys = baseline[key].merge(
        history[key], on=key, how="left", indicator=True
    )
    missing_history_keys = missing_history_keys.loc[
        missing_history_keys._merge.eq("left_only"), key
    ]
    if not missing_history_keys.empty:
        raise ValueError(
            "2023行业基准缺少自身shape_history："
            f"{missing_history_keys.to_dict('records')[:10]}"
        )
    year_values = tuple(int(year) for year in years)
    if not year_values:
        raise ValueError("未来参数年份不能为空")

    history_lookup = history.set_index(key, drop=False)
    rows: list[dict[str, object]] = []
    for base in baseline.sort_values(key, kind="mergesort").itertuples(index=False):
        industry = str(base.industry)
        season = str(base.season)
        final_day_type = str(base.final_day_type)
        history_row = history_lookup.loc[(industry, season, final_day_type)]
        if isinstance(history_row, pd.DataFrame):
            raise ValueError(f"shape_history行业单元重复：{industry}/{season}/{final_day_type}")
        direction = _direction_from_history(history_row)
        eligible = _history_boolean(history_row, "trend_eligible")
        flat_curve = _history_boolean(history_row, "flat_curve")
        flat_borderline = _history_boolean(history_row, "flat_borderline")
        peak_valley_delta_abs = float(history_row.peak_valley_delta_abs)
        if not np.isfinite(peak_valley_delta_abs) or peak_valley_delta_abs < 0.0:
            raise ValueError("shape_history peak_valley_delta_abs必须为有限非负值")
        peak_valley_delta_band = str(history_row.peak_valley_delta_band).strip()
        if not peak_valley_delta_band:
            raise ValueError("shape_history peak_valley_delta_band不能为空")
        gate_enabled = _history_boolean(
            history_row, "peak_valley_evolution_enabled"
        )
        gate_reason = str(history_row.peak_valley_gate_reason).strip()
        if gate_enabled != (gate_reason == "enabled"):
            raise ValueError(
                "峰谷差率门控布尔值与原因不一致："
                f"{industry}/{season}/{final_day_type}"
            )
        if gate_enabled and not eligible:
            raise ValueError(
                "历史不足单元不得启用峰谷差率演化："
                f"{industry}/{season}/{final_day_type}"
            )
        gate_policy = str(history_row.peak_valley_gate_policy).strip()
        if gate_policy != PEAK_VALLEY_GATE_POLICY:
            raise ValueError(
                "shape_history峰谷差率门控策略与当前方法不一致："
                f"{industry}/{season}/{final_day_type}"
            )
        beta_a = float(history_row.beta_a) if eligible else 0.0
        if not np.isfinite(beta_a) or beta_a < 0:
            raise ValueError("shape_history beta_a必须为有限非负值")
        if not eligible:
            direction = np.zeros(4, dtype=float)

        base_curve = np.asarray(
            [getattr(base, column) for column in SHAPE_COLS], dtype=float
        )
        base_shares = window_shares(base_curve)
        if "shares_2023" in history_row.index:
            recorded_shares = np.asarray(history_row.shares_2023, dtype=float)
            if (
                recorded_shares.shape == (4,)
                and np.isfinite(recorded_shares).all()
                and not np.allclose(
                    recorded_shares, base_shares, rtol=0.0, atol=1e-12
                )
            ):
                raise ValueError(
                    f"shape_history与2023基准份额不一致：{industry}/{season}/{final_day_type}"
                )
        baseline_diag = diagnose_shape(base_curve)
        recorded_peak_valley_rate = float(history_row.peak_valley_rate_2023)
        if not np.isfinite(recorded_peak_valley_rate):
            raise ValueError("shape_history peak_valley_rate_2023必须为有限值")
        base_peak_valley_rate = peak_valley_rate(base_curve)
        if not np.isclose(
            recorded_peak_valley_rate,
            base_peak_valley_rate,
            rtol=0.0,
            atol=PEAK_VALLEY_RATE_TOLERANCE,
        ):
            raise ValueError(
                "shape_history与2023基准峰谷差率不一致："
                f"{industry}/{season}/{final_day_type}"
            )
        peak_valley_delta = float(history_row.peak_valley_delta_signed)
        if not np.isfinite(peak_valley_delta):
            raise ValueError("shape_history peak_valley_delta_signed必须为有限值")
        if not eligible:
            peak_valley_delta = 0.0
        for horizon in SUPPORTED_HORIZONS:
            for year in year_values:
                requested = extrapolated_increment(beta_a, year, horizon)
                a_max, applied, boundary_hit = constrain_window_migration(
                    base_shares, direction, requested
                )
                requested_shares = base_shares + requested * direction
                applied_shares = base_shares + applied * direction
                for peak_valley_phi in PEAK_VALLEY_SUPPORTED_PHIS:
                    peak_valley_target = forecast_peak_valley_target(
                        rate_2023=recorded_peak_valley_rate,
                        delta_signed=peak_valley_delta,
                        year=year,
                        phi=peak_valley_phi,
                        trend_eligible=eligible,
                    )
                    transformed = apply_sequential_shape_transform(
                        base_curve,
                        applied_shares,
                        peak_valley_target.range_limited,
                        peak_valley_evolution_enabled=gate_enabled,
                        peak_valley_gate_reason=gate_reason,
                    )
                    intermediate_diag = diagnose_shape(
                        transformed.intermediate_shape
                    )
                    final_curve = transformed.final_shape
                    future_diag = diagnose_shape(final_curve)
                    common = {
                        "year": year,
                        "H": horizon,
                        "phi": peak_valley_phi,
                        "peak_valley_phi": peak_valley_phi,
                        "formal_H8": horizon == 8,
                        "formal_phi08": peak_valley_phi == 0.8,
                        "formal_H8_phi08": (
                            horizon == 8 and peak_valley_phi == 0.8
                        ),
                        "industry": industry,
                        "department": str(
                            getattr(
                                base,
                                "department",
                                getattr(base, "broad_sector", ""),
                            )
                        ),
                        "broad_sector": str(
                            getattr(
                                base,
                                "broad_sector",
                                getattr(base, "department", ""),
                            )
                        ),
                        "season": season,
                        "final_day_type": final_day_type,
                        "day_group": final_day_type,
                        "base_source_year": 2023,
                        "base_anchor_year": 2023,
                        "trend_eligible": eligible,
                        "flat_curve": flat_curve,
                        "flat_borderline": flat_borderline,
                        "peak_valley_delta_abs": peak_valley_delta_abs,
                        "peak_valley_delta_band": peak_valley_delta_band,
                        "peak_valley_evolution_enabled": gate_enabled,
                        "peak_valley_gate_reason": gate_reason,
                        "peak_valley_gate_policy": gate_policy,
                        "trend_status": str(
                            history_row.get(
                                "trend_status",
                                "paired_2022_2023_industry_medoids"
                                if eligible
                                else "insufficient_history_no_extrapolation",
                            )
                        ),
                        "fallback_status": str(
                            history_row.get(
                                "fallback_status",
                                getattr(base, "fallback_status", "none"),
                            )
                        ),
                        "cell_trend_frozen": not eligible,
                        "judgement_changed": False,
                        "beta_a": beta_a,
                        "direction_v": direction.copy(),
                        "a_requested": requested,
                        "a_max": a_max,
                        "a_max_unbounded": not np.isfinite(a_max),
                        "a_applied": applied,
                        "window_boundary_hit": boundary_hit,
                        "requested_share_night": requested_shares[0],
                        "requested_share_morning": requested_shares[1],
                        "requested_share_midday": requested_shares[2],
                        "requested_share_evening": requested_shares[3],
                        "applied_share_night": applied_shares[0],
                        "applied_share_morning": applied_shares[1],
                        "applied_share_midday": applied_shares[2],
                        "applied_share_evening": applied_shares[3],
                        # Compatibility aliases remain the first-stage targets.
                        "night_share": applied_shares[0],
                        "morning_share": applied_shares[1],
                        "midday_share": applied_shares[2],
                        "evening_share": applied_shares[3],
                        "peak_valley_cumulative_damping": peak_valley_target.cumulative_damping,
                        "peak_valley_target_requested": (
                            peak_valley_target.requested
                        ),
                        "peak_valley_target_range_limited": (
                            peak_valley_target.range_limited
                        ),
                        "peak_valley_target_applied": (
                            transformed.peak_valley.target_applied
                        ),
                        "peak_valley_target_clipped": (
                            peak_valley_target.range_clipped
                        ),
                        "peak_valley_adjustment_status": (
                            transformed.peak_valley.adjustment_status
                        ),
                        "peak_valley_rate_after_four_window": (
                            transformed.peak_valley.rate_before
                        ),
                        "peak_valley_lambda": (
                            transformed.peak_valley.lambda_value
                        ),
                        "peak_valley_rate_final": future_diag[
                            "peak_valley_rate"
                        ],
                        "four_window_share_intermediate_night": (
                            transformed.intermediate_window_shares[0]
                        ),
                        "four_window_share_intermediate_morning": (
                            transformed.intermediate_window_shares[1]
                        ),
                        "four_window_share_intermediate_midday": (
                            transformed.intermediate_window_shares[2]
                        ),
                        "four_window_share_intermediate_evening": (
                            transformed.intermediate_window_shares[3]
                        ),
                        "four_window_share_final_night": (
                            transformed.final_window_shares[0]
                        ),
                        "four_window_share_final_morning": (
                            transformed.final_window_shares[1]
                        ),
                        "four_window_share_final_midday": (
                            transformed.final_window_shares[2]
                        ),
                        "four_window_share_final_evening": (
                            transformed.final_window_shares[3]
                        ),
                        "four_window_share_deviation_night": (
                            transformed.final_window_share_deviation[0]
                        ),
                        "four_window_share_deviation_morning": (
                            transformed.final_window_share_deviation[1]
                        ),
                        "four_window_share_deviation_midday": (
                            transformed.final_window_share_deviation[2]
                        ),
                        "four_window_share_deviation_evening": (
                            transformed.final_window_share_deviation[3]
                        ),
                        "shape_extrapolation_method": SHAPE_EXTRAPOLATION_METHOD,
                        "four_window_final_share_policy": (
                            FOUR_WINDOW_FINAL_SHARE_POLICY
                        ),
                        "peak_valley_extrapolation_enabled": (
                            PEAK_VALLEY_EXTRAPOLATION_ENABLED
                        ),
                        "late_peak_control_policy": LATE_PEAK_CONTROL_POLICY,
                        "peak_damping_enabled": PEAK_DAMPING_ENABLED,
                        "anchor_repair_enabled": ANCHOR_REPAIR_ENABLED,
                        "future_peak_valley_rate": future_diag[
                            "peak_valley_rate"
                        ],
                        "future_load_factor": future_diag["load_factor"],
                        "future_cv": future_diag["cv"],
                        "future_evening_peak_hour": future_diag[
                            "evening_peak_hour"
                        ],
                        "future_raw_85pct_hours": future_diag[
                            "raw_85pct_hours"
                        ],
                        "future_effective_control_hours": future_diag[
                            "effective_control_hours"
                        ],
                        "evening_peak_changed_from_2023": (
                            future_diag["evening_peak_hour"]
                            != baseline_diag["evening_peak_hour"]
                        ),
                        "raw_85pct_set_changed_from_2023": (
                            set(future_diag["raw_85pct_hours"])
                            != set(baseline_diag["raw_85pct_hours"])
                        ),
                        "effective_control_set_changed_from_2023": (
                            set(future_diag["effective_control_hours"])
                            != set(baseline_diag["effective_control_hours"])
                        ),
                        "boundary_smoothing_max_abs_change": float(np.max(np.abs(transformed.intermediate_shape - transformed.proportional_shape))),
                        **{f"proportional_p{h:02d}": value for h, value in enumerate(transformed.proportional_shape)},
                        **{f"smoothed_p{h:02d}": value for h, value in enumerate(transformed.intermediate_shape)},
                        "intermediate_peak_valley_rate": intermediate_diag[
                            "peak_valley_rate"
                        ],
                        **dict(zip(SHAPE_COLS, final_curve, strict=True)),
                    }
                    rows.append(common)
    return pd.DataFrame(rows)


def build_future_parameter_table(
    baseline_shapes: pd.DataFrame,
    estimates: Mapping[tuple[str, str, str], ShapeParameterEstimate] | pd.DataFrame,
    years: Iterable[int],
) -> pd.DataFrame:
    """Build formal future shapes, retaining the old mapping call shape."""

    if isinstance(estimates, pd.DataFrame):
        return _build_future_parameter_table_340(baseline_shapes, estimates, years)
    return _build_future_parameter_table_legacy(baseline_shapes, estimates, years)


def _build_future_parameter_table_legacy(
    baseline_shapes: pd.DataFrame,
    estimates: Mapping[tuple[str, str, str], ShapeParameterEstimate],
    years: Iterable[int],
) -> pd.DataFrame:
    """Build per-industry future trajectories without cross-industry borrowing."""

    required = {
        "industry",
        "broad_sector",
        "season",
        "day_group",
        "anchor_year",
        "trend_frozen",
        "judgement_changed",
        *SHAPE_COLS,
    }
    if not required.issubset(baseline_shapes.columns):
        raise ValueError(f"行业基准形状缺少字段：{sorted(required - set(baseline_shapes.columns))}")
    key = ["industry", "season", "day_group"]
    if baseline_shapes.duplicated(key).any():
        raise ValueError("行业基准形状单元重复")
    year_values = tuple(int(year) for year in years)
    if not year_values:
        raise ValueError("未来参数年份不能为空")

    rows: list[dict[str, object]] = []
    for base in baseline_shapes.itertuples(index=False):
        industry = str(base.industry)
        season = str(base.season)
        day_group = str(base.day_group)
        estimate_key = (industry, season, day_group)
        if estimate_key not in estimates:
            raise ValueError(f"行业单元缺少自身形状参数：{industry}/{season}/{day_group}")
        estimate = estimates[estimate_key]
        cell_frozen = bool(base.trend_frozen)
        if not cell_frozen and not estimate.trend_eligible:
            raise ValueError(
                f"行业单元基准允许迁移但参数历史不足：{industry}/{season}/{day_group}"
            )
        if cell_frozen:
            cell_estimate = replace(
                estimate,
                beta_a=0.0,
                trend_eligible=False,
                trend_status="insufficient_history_no_extrapolation",
            )
        else:
            cell_estimate = estimate
        base_curve = np.asarray([getattr(base, column) for column in SHAPE_COLS], dtype=float)
        trajectories = forecast_parameter_trajectory(
            cell_estimate,
            year_values,
            base_window_shares=window_shares(base_curve),
        )
        for trajectory in trajectories.to_dict("records"):
            rows.append(
                {
                    **trajectory,
                    "industry": industry,
                    "broad_sector": str(base.broad_sector),
                    "season": season,
                    "day_group": day_group,
                    "base_anchor_year": int(base.anchor_year),
                    "judgement_changed": bool(base.judgement_changed),
                    "cell_trend_frozen": cell_frozen,
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "DAY_GROUPS",
    "SEASONS",
    "build_cell_quality",
    "build_future_parameter_table",
    "build_industry_cell_judgement",
    "identify_industry_parameters",
]
