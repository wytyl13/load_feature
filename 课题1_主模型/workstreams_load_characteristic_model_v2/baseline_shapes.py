"""Fine-industry baseline shapes and adaptive day-type energy factors."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .config import (
    BASE_MEDOID_YEAR,
    HISTORY_COMPARISON_YEAR,
    INDUSTRY_AGGREGATION_MODE,
    MEDOID_DISTANCE,
    MEDOID_TIE_BREAK,
    SHAPE_NUMERICAL_TOLERANCE,
    TYPICAL_CURVE_K,
)
from .day_types import HOLIDAY, MERGED_NONWORKDAY, WEEKEND, WORKDAY, canonical_day_type
from .quality import SHAPE_COLS


SEASONS = ("春季", "夏季", "秋季", "冬季")
LEGACY_DAY_GROUPS = ("工作日", "非工作日")
CANONICAL_DAY_GROUPS = ("工作日", "周末", "节假日")
MERGED_DAY_GROUP = "合并非工作日"
ALLOWED_DAY_GROUPS = frozenset((*LEGACY_DAY_GROUPS, *CANONICAL_DAY_GROUPS, MERGED_DAY_GROUP))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not keep.any():
        raise ValueError("加权中位数没有有效观测")
    order = np.argsort(values[keep], kind="mergesort")
    sorted_values = values[keep][order]
    sorted_weights = weights[keep][order]
    index = int(
        np.searchsorted(
            np.cumsum(sorted_weights), sorted_weights.sum() / 2, side="left"
        )
    )
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def weighted_medoid(curves: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, int, float]:
    curves = np.asarray(curves, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if curves.ndim != 2 or curves.shape[1] != 24 or len(curves) != len(weights):
        raise ValueError("weighted_medoid需要 n×24 曲线和 n 个权重")
    if (
        not np.isfinite(curves).all()
        or np.any(curves < 0)
        or np.any(curves.sum(axis=1) <= 0)
        or np.any(weights < 0)
        or weights.sum() <= 0
    ):
        raise ValueError("weighted_medoid输入非法")
    normalized = curves / curves.sum(axis=1, keepdims=True)
    distances = np.abs(normalized[:, None, :] - normalized[None, :, :]).sum(axis=2)
    objective = distances @ weights
    selected = int(
        np.flatnonzero(np.isclose(objective, objective.min(), rtol=0, atol=1e-15))[0]
    )
    curve = normalized[selected]
    return curve / curve.sum(), selected, float(objective[selected] / weights.sum())


def _prepare_records(records: pd.DataFrame) -> pd.DataFrame:
    frame = records.copy()
    if "broad_sector" not in frame.columns:
        if "department" not in frame.columns:
            raise ValueError("行业日数据缺少 broad_sector/department")
        frame = frame.rename(columns={"department": "broad_sector"})
    required = {
        "broad_sector",
        "industry",
        "user",
        "year",
        "season",
        *SHAPE_COLS,
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"行业日数据缺少字段：{sorted(required - set(frame.columns))}")
    sector_counts = frame.groupby("industry").broad_sector.nunique()
    conflicts = sector_counts.loc[sector_counts.ne(1)].index.tolist()
    if conflicts:
        raise ValueError(f"行业存在多个产业大类：{conflicts}")
    if "final_day_type" in frame.columns:
        frame["day_group"] = frame.final_day_type.astype(str)
    elif "day_group" not in frame.columns:
        raise ValueError("行业日数据缺少 day_group/final_day_type")
    return frame


def _user_centres(records: pd.DataFrame) -> pd.DataFrame:
    keys = ["broad_sector", "industry", "year", "season", "day_group", "user"]
    rows: list[dict[str, object]] = []
    for key, part in records.groupby(keys, observed=True):
        curve = part[SHAPE_COLS].mean(axis=0).to_numpy(dtype=float, copy=True)
        curve /= curve.sum()
        rows.append(
            {
                **dict(zip(keys, key)),
                "valid_days": len(part),
                **dict(zip(SHAPE_COLS, curve)),
            }
        )
    return pd.DataFrame(rows)


def _validate_cell_judgement(cell_judgement: pd.DataFrame) -> pd.DataFrame:
    required = {
        "industry",
        "broad_sector",
        "season",
        "day_group",
        "sample_sufficient",
        "trend_eligible",
        "trend_status",
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
    invalid_seasons = sorted(set(result.season.astype(str)).difference(SEASONS))
    invalid_day_groups = sorted(
        set(result.day_group.astype(str)).difference(ALLOWED_DAY_GROUPS)
    )
    if invalid_seasons or invalid_day_groups:
        raise ValueError(
            "行业单元判别含未知季节或日型："
            f"season={invalid_seasons}, day_group={invalid_day_groups}"
        )
    industries = sorted(result.industry.astype(str).unique())
    for industry in industries:
        industry_rows = result.loc[result.industry.astype(str).eq(industry)]
        for season in SEASONS:
            observed = set(
                industry_rows.loc[
                    industry_rows.season.astype(str).eq(season), "day_group"
                ].astype(str)
            )
            allowed_patterns = (
                set(LEGACY_DAY_GROUPS),
                {"工作日", MERGED_DAY_GROUP},
                set(CANONICAL_DAY_GROUPS),
            )
            if observed not in allowed_patterns:
                raise ValueError(
                    f"行业单元判别日型组合非法：{industry}/{season}/{sorted(observed)}"
                )
    return result.sort_values(key, kind="mergesort").reset_index(drop=True)


def _prepare_industry_daily(records: pd.DataFrame) -> pd.DataFrame:
    frame = records.copy()
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
    required = {"industry", "date", "year", "season", *SHAPE_COLS}
    missing = sorted(required.difference(frame.columns))
    if source_column is None:
        missing.append("day_type_detail")
    if missing:
        raise ValueError(f"行业日数据缺少字段：{sorted(set(missing))}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["source_day_type"] = frame[source_column].map(canonical_day_type)
    values = frame[SHAPE_COLS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("行业日数据含非有限值或负形状值")
    if len(values) and not np.allclose(
        values.sum(axis=1), 1.0, rtol=0.0, atol=SHAPE_NUMERICAL_TOLERANCE
    ):
        raise ValueError("行业日曲线未归一化")
    if frame.duplicated(["industry", "date"]).any():
        duplicate = frame.loc[
            frame.duplicated(["industry", "date"], keep=False),
            ["industry", "date"],
        ].head(10)
        raise ValueError(f"行业日记录重复：{duplicate.to_dict('records')}")
    return frame


def _build_industry_baseline_shapes(
    industry_daily: pd.DataFrame,
    cell_quality: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select an equal-date-weighted K=1 L1 medoid for each industry cell."""

    frame = _prepare_industry_daily(industry_daily)
    required_quality = {
        "industry",
        "year",
        "season",
        "final_day_type",
        "trend_eligible",
        "holiday_merged_with_weekend",
    }
    missing = sorted(required_quality.difference(cell_quality.columns))
    if missing:
        raise ValueError(f"行业日覆盖台账缺少字段：{missing}")
    quality = cell_quality.copy()
    quality["year"] = quality.year.astype(int)
    quality["final_day_type"] = quality.final_day_type.map(canonical_day_type)
    quality_key = ["industry", "year", "season", "final_day_type"]
    if quality.duplicated(quality_key).any():
        duplicate = quality.loc[
            quality.duplicated(quality_key, keep=False), quality_key
        ].head(10)
        raise ValueError(f"行业日覆盖台账键重复：{duplicate.to_dict('records')}")

    merge_flags = (
        quality.groupby(["industry", "season"], observed=True)[
            "holiday_merged_with_weekend"
        ]
        .agg(["min", "max"])
    )
    if not merge_flags.empty and (
        merge_flags["min"].astype(bool) != merge_flags["max"].astype(bool)
    ).any():
        raise ValueError("同一行业季节的节假日合并口径不一致")
    merge_lookup = {
        (str(industry), str(season)): bool(row["max"])
        for (industry, season), row in merge_flags.iterrows()
    }

    final_types: list[str] = []
    for row in frame.itertuples(index=False):
        source = str(row.source_day_type)
        merge = merge_lookup.get((str(row.industry), str(row.season)))
        if merge is None:
            raise ValueError(
                f"行业日缺少覆盖口径：{row.industry}/{row.season}"
            )
        if source == WORKDAY:
            final_types.append(WORKDAY)
        elif merge:
            final_types.append(MERGED_NONWORKDAY)
        elif source in {WEEKEND, HOLIDAY}:
            final_types.append(source)
        else:
            raise ValueError(
                f"未合并单元不能包含既有合并非工作日：{row.industry}/{row.season}"
            )
    frame["final_day_type"] = final_types

    outputs: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for quality_row in quality.sort_values(quality_key, kind="mergesort").itertuples(
        index=False
    ):
        industry = str(quality_row.industry)
        year = int(quality_row.year)
        season = str(quality_row.season)
        final_day_type = str(quality_row.final_day_type)
        candidates = frame.loc[
            frame.industry.astype(str).eq(industry)
            & frame.year.eq(year)
            & frame.season.astype(str).eq(season)
            & frame.final_day_type.eq(final_day_type)
        ].sort_values("date", kind="mergesort")
        if candidates.empty:
            status = (
                "baseline_missing_2023"
                if year == BASE_MEDOID_YEAR
                else (
                    "history_comparison_missing_2022"
                    if year == HISTORY_COMPARISON_YEAR
                    else "medoid_candidates_missing"
                )
            )
            diagnostics.append(
                {
                    "department": getattr(quality_row, "department", ""),
                    "broad_sector": getattr(quality_row, "broad_sector", ""),
                    "industry": industry,
                    "year": year,
                    "season": season,
                    "final_day_type": final_day_type,
                    "day_group": final_day_type,
                    "candidate_day_count": 0,
                    "medoid_date": pd.NaT,
                    "medoid_objective_l1": np.nan,
                    "medoid_status": status,
                    "trend_eligible": False,
                    "holiday_merged_with_weekend": bool(
                        quality_row.holiday_merged_with_weekend
                    ),
                    "source_user_count": int(
                        getattr(quality_row, "source_user_count", 0)
                    ),
                    "mean_active_user_count": np.nan,
                    "min_active_user_count": int(
                        getattr(quality_row, "min_active_user_count", 0)
                    ),
                    "max_active_user_count": int(
                        getattr(quality_row, "max_active_user_count", 0)
                    ),
                    "partial_composition_days": int(
                        getattr(quality_row, "partial_composition_days", 0)
                    ),
                }
            )
            continue

        curves = candidates[SHAPE_COLS].to_numpy(dtype=float)
        distances = np.abs(curves[:, None, :] - curves[None, :, :]).sum(axis=2)
        objective = distances.sum(axis=1)
        minimum = float(objective.min())
        selected_index = int(
            np.flatnonzero(
                np.isclose(objective, minimum, rtol=0.0, atol=1e-15)
            )[0]
        )
        selected = candidates.iloc[selected_index]
        curve = selected[SHAPE_COLS].to_numpy(dtype=float)
        trend_eligible = bool(quality_row.trend_eligible)
        department = (
            str(selected.department)
            if "department" in candidates.columns
            else str(getattr(quality_row, "department", ""))
        )
        output = {
            "department": department,
            "broad_sector": department,
            "industry": industry,
            "year": year,
            "season": season,
            "final_day_type": final_day_type,
            "day_group": final_day_type,
            "medoid_date": pd.Timestamp(selected.date),
            "sample_day_count": int(len(candidates)),
            "valid_days": int(len(candidates)),
            "medoid_objective_l1": minimum,
            "medoid_distance": MEDOID_DISTANCE,
            "typical_curve_k": TYPICAL_CURVE_K,
            "medoid_tie_break": MEDOID_TIE_BREAK,
            "candidate_weighting": "equal_industry_dates",
            "aggregation_method": str(
                selected.get("aggregation_method", INDUSTRY_AGGREGATION_MODE)
            ),
            "source_user_count": (
                int(candidates.source_user_count.max())
                if "source_user_count" in candidates.columns
                else int(getattr(quality_row, "source_user_count", 0))
            ),
            "mean_active_user_count": (
                float(candidates.active_user_count.mean())
                if "active_user_count" in candidates.columns
                else np.nan
            ),
            "min_active_user_count": (
                int(candidates.active_user_count.min())
                if "active_user_count" in candidates.columns
                else np.nan
            ),
            "max_active_user_count": (
                int(candidates.active_user_count.max())
                if "active_user_count" in candidates.columns
                else np.nan
            ),
            "partial_composition_days": (
                int(
                    candidates.partial_composition_day.fillna(False)
                    .astype(bool)
                    .sum()
                )
                if "partial_composition_day" in candidates.columns
                else 0
            ),
            "holiday_merged_with_weekend": bool(
                quality_row.holiday_merged_with_weekend
            ),
            "trend_eligible": trend_eligible,
            "trend_frozen": not trend_eligible,
            "trend_status": str(
                getattr(
                    quality_row,
                    "trend_status",
                    (
                        "paired_2022_2023_industry_days"
                        if trend_eligible
                        else "insufficient_history_no_extrapolation"
                    ),
                )
            ),
            "is_future_baseline": year == BASE_MEDOID_YEAR,
            "is_history_comparison": year == HISTORY_COMPARISON_YEAR,
            "base_source_year": BASE_MEDOID_YEAR,
            "anchor_year": year,
            "preferred_anchor_year": BASE_MEDOID_YEAR,
            "anchor_fallback": False,
            "judgement_changed": False,
            **dict(zip(SHAPE_COLS, curve, strict=True)),
        }
        outputs.append(output)
        diagnostics.append(
            {
                **{
                    key: output[key]
                    for key in (
                        "department",
                        "broad_sector",
                        "industry",
                        "year",
                        "season",
                        "final_day_type",
                        "day_group",
                        "medoid_date",
                        "medoid_objective_l1",
                        "trend_eligible",
                        "holiday_merged_with_weekend",
                    )
                },
                "candidate_day_count": int(len(candidates)),
                "medoid_status": "selected_actual_industry_day",
                "candidate_weighting": "equal_industry_dates",
                "source_user_count": output["source_user_count"],
                "mean_active_user_count": output["mean_active_user_count"],
                "min_active_user_count": output["min_active_user_count"],
                "max_active_user_count": output["max_active_user_count"],
                "partial_composition_days": output["partial_composition_days"],
            }
        )

    baseline_columns = [
        "department",
        "broad_sector",
        "industry",
        "year",
        "season",
        "final_day_type",
        "day_group",
        "medoid_date",
        "sample_day_count",
        "valid_days",
        "medoid_objective_l1",
        "medoid_distance",
        "typical_curve_k",
        "medoid_tie_break",
        "candidate_weighting",
        "aggregation_method",
        "source_user_count",
        "mean_active_user_count",
        "min_active_user_count",
        "max_active_user_count",
        "partial_composition_days",
        "holiday_merged_with_weekend",
        "trend_eligible",
        "trend_frozen",
        "trend_status",
        "is_future_baseline",
        "is_history_comparison",
        "base_source_year",
        "anchor_year",
        "preferred_anchor_year",
        "anchor_fallback",
        "judgement_changed",
        *SHAPE_COLS,
    ]
    baseline = pd.DataFrame(outputs, columns=baseline_columns)
    if not baseline.empty:
        baseline = baseline.sort_values(
            ["industry", "year", "season", "final_day_type"], kind="mergesort"
        ).reset_index(drop=True)
    diagnostic_frame = pd.DataFrame(diagnostics)
    if not diagnostic_frame.empty:
        diagnostic_frame = diagnostic_frame.sort_values(
            ["industry", "year", "season", "final_day_type"], kind="mergesort"
        ).reset_index(drop=True)
    return baseline, diagnostic_frame


def build_baseline_shapes(
    records: pd.DataFrame,
    cell_judgement: pd.DataFrame,
    *,
    anchor_year: int = BASE_MEDOID_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build 2022/2023 industry medoids, with a legacy-schema fallback."""

    if "year" in cell_judgement.columns and "final_day_type" in cell_judgement.columns:
        if anchor_year != BASE_MEDOID_YEAR:
            raise ValueError(f"正式行业基准年份固定为{BASE_MEDOID_YEAR}")
        return _build_industry_baseline_shapes(records, cell_judgement)
    return _build_baseline_shapes_legacy(
        records, cell_judgement, anchor_year=anchor_year
    )


def _build_baseline_shapes_legacy(
    records: pd.DataFrame,
    cell_judgement: pd.DataFrame,
    *,
    anchor_year: int = 2023,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one adjusted 2022 baseline for each industry/season/final-day-type.

    The current pipeline may attach ``baseline_p00`` ... ``baseline_p23`` to the
    judgement table.  Those values are the transactionally validated result of
    applying the control-period and evening-peak rules once to a 2022 medoid.
    They are consumed directly here.  The older medoid path is retained only so
    historical callers and focused unit tests remain readable; it never chooses
    2023 merely because a change was triggered.
    """

    del anchor_year  # anchors are determined by the approved binary judgement
    frame = _prepare_records(records)
    judgement = _validate_cell_judgement(cell_judgement)
    centres = _user_centres(frame)
    sector_lookup = frame.drop_duplicates("industry").set_index("industry").broad_sector

    outputs: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    baseline_columns = [f"baseline_{column}" for column in SHAPE_COLS]
    precomputed = all(column in judgement.columns for column in baseline_columns)
    for decision in judgement.itertuples(index=False):
        industry = str(decision.industry)
        broad_sector = str(decision.broad_sector)
        season = str(decision.season)
        day_group = str(decision.day_group)
        if industry not in sector_lookup:
            raise ValueError(f"行业基准曲线无日数据：{industry}")
        if str(sector_lookup.loc[industry]) != broad_sector:
            raise ValueError(f"行业产业大类冲突：{industry}")
        if precomputed:
            curve = np.asarray(
                [getattr(decision, column) for column in baseline_columns], dtype=float
            )
            if (
                curve.shape != (24,)
                or not np.isfinite(curve).all()
                or np.any(curve < 0)
                or not np.isclose(curve.sum(), 1.0, rtol=0.0, atol=1e-10)
            ):
                raise ValueError(
                    f"行业单元预计算基准曲线非法：{industry}/{season}/{day_group}"
                )
            used_year = int(decision.source_year)
            used_fallback = bool(getattr(decision, "anchor_fallback", used_year != 2022))
            trend_eligible = bool(decision.trend_eligible)
            frozen = bool(not trend_eligible or used_fallback)
            fallback_reason = (
                "2022年基准缺失，回退2023年且局部变化不执行"
                if used_fallback
                else ""
            )
            output = {
                "industry": industry,
                "broad_sector": broad_sector,
                "season": season,
                "day_group": day_group,
                "anchor_year": used_year,
                "preferred_anchor_year": 2022,
                "anchor_fallback": used_fallback,
                "anchor_fallback_reason": fallback_reason,
                "trend_eligible": not frozen,
                "trend_frozen": frozen,
                "trend_status": (
                    "short_history_exploratory"
                    if not frozen
                    else "insufficient_history_no_extrapolation"
                ),
                "judgement_changed": bool(decision.changed),
                "judgement_reason": str(decision.reason),
                "selected_user": str(decision.medoid_source_user_2022),
                "valid_days": int(getattr(decision, "medoid_source_days_2022", 0)),
                **dict(zip(SHAPE_COLS, curve)),
            }
            for optional in (
                "holiday_merged_with_weekend",
                "load_factor_2022",
                "cv_2022",
                "flat_curve",
                "control_adjustment_requested",
                "control_adjustment_applied",
                "control_status",
                "evening_adjustment_requested",
                "evening_adjustment_applied",
                "evening_status",
                "local_adjustment_applied",
                "local_adjustment_status",
                "same_source_days_2023",
            ):
                if hasattr(decision, optional):
                    output[optional] = getattr(decision, optional)
            outputs.append(output)
            diagnostics.append(
                {
                    "industry": industry,
                    "broad_sector": broad_sector,
                    "season": season,
                    "day_group": day_group,
                    "anchor_year": used_year,
                    "preferred_anchor_year": 2022,
                    "donor_users": np.nan,
                    "medoid_objective_l1": float(decision.medoid_objective_l1),
                    "trend_frozen": frozen,
                    "judgement_changed": bool(decision.changed),
                    "preferred_year_fallback": used_fallback,
                    "anchor_fallback_reason": fallback_reason,
                    "coverage_status": (
                        "paired_history_sufficient"
                        if not frozen
                        else "insufficient_or_anchor_fallback"
                    ),
                }
            )
            continue
        cell_all = centres.loc[
            centres.industry.eq(industry)
            & centres.season.eq(season)
            & centres.day_group.eq(day_group)
        ]
        if cell_all.empty:
            raise ValueError(f"行业基准曲线单元无数据：{industry}/{season}/{day_group}")

        changed = bool(decision.changed)
        sufficient = bool(decision.sample_sufficient)
        trend_eligible = bool(decision.trend_eligible)
        if sufficient != trend_eligible:
            raise ValueError(
                f"行业单元样本充分性与趋势资格冲突：{industry}/{season}/{day_group}"
            )
        preferred_year = int(decision.source_year)
        if preferred_year not in (2022, 2023):
            raise ValueError(
                f"行业单元锚点年非法：{industry}/{season}/{day_group}/{preferred_year}"
            )
        anchor = cell_all.loc[cell_all.year.eq(preferred_year)]
        used_year = preferred_year
        used_fallback = False
        if anchor.empty:
            available = sorted(
                (int(value) for value in cell_all.year.unique()),
                key=lambda year: (abs(year - preferred_year), -year),
            )
            if not available:
                raise ValueError(f"行业基准曲线无可用年份：{industry}/{season}/{day_group}")
            used_year = available[0]
            anchor = cell_all.loc[cell_all.year.eq(used_year)]
            used_fallback = True

        weights = anchor.valid_days.to_numpy(dtype=float)
        medoid_curve, medoid_index, objective = weighted_medoid(
            anchor[SHAPE_COLS].to_numpy(dtype=float), weights
        )
        selected = anchor.iloc[medoid_index]
        if precomputed:
            curve = np.asarray(
                [getattr(decision, column) for column in baseline_columns], dtype=float
            )
            if (
                curve.shape != (24,)
                or not np.isfinite(curve).all()
                or np.any(curve < 0)
                or not np.isclose(curve.sum(), 1.0, rtol=0.0, atol=1e-10)
            ):
                raise ValueError(
                    f"行业单元预计算基准曲线非法：{industry}/{season}/{day_group}"
                )
            if hasattr(decision, "medoid_source_user_2022"):
                source_user = str(decision.medoid_source_user_2022)
                source_match = anchor.loc[anchor.user.astype(str).eq(source_user)]
                if not source_match.empty:
                    selected = source_match.iloc[0]
            objective = float(
                getattr(decision, "medoid_objective_l1", objective)
            )
        else:
            curve = medoid_curve
        frozen = bool(not trend_eligible or used_fallback)
        fallback_reason = (
            f"首选{preferred_year}年无数据，回退{used_year}年"
            if used_fallback
            else ""
        )
        output = {
            "industry": industry,
            "broad_sector": broad_sector,
            "season": season,
            "day_group": day_group,
            "anchor_year": int(used_year),
            "preferred_anchor_year": preferred_year,
            "anchor_fallback": used_fallback,
            "anchor_fallback_reason": fallback_reason,
            "trend_eligible": not frozen,
            "trend_frozen": frozen,
            "trend_status": (
                "short_history_exploratory"
                if not frozen
                else "insufficient_history_no_extrapolation"
            ),
            "judgement_changed": changed,
            "judgement_reason": str(decision.reason),
            "selected_user": str(
                getattr(decision, "medoid_source_user_2022", selected.user)
            ),
            "valid_days": int(anchor.valid_days.sum()),
            **dict(zip(SHAPE_COLS, curve)),
        }
        for optional in (
            "holiday_merged_with_weekend",
            "load_factor_2022",
            "cv_2022",
            "flat_curve",
            "control_adjustment_requested",
            "control_adjustment_applied",
            "control_status",
            "evening_adjustment_requested",
            "evening_adjustment_applied",
            "evening_status",
            "local_adjustment_applied",
            "local_adjustment_status",
            "same_source_days_2023",
        ):
            if hasattr(decision, optional):
                output[optional] = getattr(decision, optional)
        outputs.append(output)
        diagnostics.append(
            {
                "industry": industry,
                "broad_sector": broad_sector,
                "season": season,
                "day_group": day_group,
                "anchor_year": int(used_year),
                "preferred_anchor_year": preferred_year,
                "donor_users": int(len(anchor)),
                "medoid_objective_l1": objective,
                "trend_frozen": frozen,
                "judgement_changed": changed,
                "preferred_year_fallback": used_fallback,
                "anchor_fallback_reason": fallback_reason,
                "coverage_status": (
                    "paired_history_sufficient"
                    if not frozen
                    else "insufficient_or_anchor_fallback"
                ),
            }
        )

    baseline = pd.DataFrame(outputs)
    diagnostics_frame = pd.DataFrame(diagnostics)
    expected_rows = len(judgement)
    if len(baseline) != expected_rows:
        raise AssertionError(f"行业基准曲线单元数错误：{len(baseline)} != {expected_rows}")
    return baseline, diagnostics_frame


def _canonical_day_type(value: object) -> str:
    text = str(value).strip()
    if text in {"工作日", "普通工作日", "调休工作日"}:
        return "工作日"
    if text in {"节假日", "法定节假日"}:
        return "节假日"
    if text in {"周末", "普通周末", "非工作日"}:
        return "周末"
    raise ValueError(f"未知历史日型：{text}")


def estimate_day_type_factors(
    records: pd.DataFrame,
    baseline_shapes: pd.DataFrame,
    *,
    anchor_year: int = 2023,
    expected_industries: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Estimate workday/weekend/holiday (or merged) daily-energy factors."""

    frame = _prepare_records(records)
    if "sample_energy" not in frame.columns:
        raise ValueError("行业日数据缺少字段：['sample_energy']")
    frame = frame.loc[frame.year.eq(anchor_year)].copy()
    required_shapes = {"industry", "season", "day_group"}
    if not required_shapes.issubset(baseline_shapes.columns):
        raise ValueError(
            f"行业基准形状缺少字段：{sorted(required_shapes - set(baseline_shapes.columns))}"
        )
    cells = baseline_shapes[["industry", "season", "day_group"]].drop_duplicates().copy()
    if cells.duplicated(["industry", "season", "day_group"]).any():
        raise ValueError("行业基准形状日型单元重复")

    if "final_day_type" not in records.columns:
        if "day_type" in frame.columns:
            frame["canonical_day_type"] = frame.day_type.map(_canonical_day_type)
        else:
            frame["canonical_day_type"] = frame.day_group.map(_canonical_day_type)
        merge_lookup = {
            (str(industry), str(season)): MERGED_DAY_GROUP in set(part.day_group.astype(str))
            for (industry, season), part in cells.groupby(["industry", "season"], sort=False)
        }
        frame["day_group"] = [
            (
                MERGED_DAY_GROUP
                if canonical != "工作日" and merge_lookup.get((str(industry), str(season)), False)
                else canonical
            )
            for industry, season, canonical in zip(
                frame.industry, frame.season, frame.canonical_day_type, strict=True
            )
        ]
    industries = (
        tuple(str(value) for value in expected_industries)
        if expected_industries is not None
        else tuple(sorted(frame.industry.astype(str).unique()))
    )
    sector_lookup = frame.drop_duplicates("industry").set_index("industry").broad_sector
    missing = sorted(set(industries).difference(sector_lookup.index.astype(str)))
    if missing:
        raise ValueError(f"非工作日系数缺少行业数据：{missing}")
    per_user = (
        frame.groupby(["industry", "season", "user", "day_group"], observed=True)
        .sample_energy.median()
        .unstack("day_group")
    )
    rows: list[dict[str, object]] = []
    for industry in industries:
        broad_sector = str(sector_lookup.loc[industry])
        for season in SEASONS:
            try:
                part = per_user.xs((industry, season), level=("industry", "season"))
            except KeyError:
                part = pd.DataFrame()
            cell_types = tuple(
                cells.loc[
                    cells.industry.astype(str).eq(industry)
                    & cells.season.astype(str).eq(season),
                    "day_group",
                ].astype(str)
            )
            if not cell_types:
                raise ValueError(f"日电量倍率缺少基准日型：{industry}/{season}")
            for day_group in cell_types:
                if day_group == "工作日":
                    factor, source = 1.0, "工作日基准"
                else:
                    ratios: list[float] = []
                    if not part.empty and {"工作日", day_group}.issubset(part.columns):
                        for _, row in part.iterrows():
                            workday = float(row["工作日"])
                            target = float(row[day_group])
                            if (
                                np.isfinite(workday)
                                and workday > 0
                                and np.isfinite(target)
                                and target > 0
                            ):
                                ratios.append(target / workday)
                    if ratios:
                        factor = weighted_median(
                            np.asarray(ratios, dtype=float),
                            np.ones(len(ratios), dtype=float),
                        )
                        source = "行业×季节×最终日型"
                    else:
                        factor, source = 1.0, "样本不足冻结为1"
                rows.append(
                    {
                        "industry": industry,
                        "broad_sector": broad_sector,
                        "season": season,
                        "day_group": day_group,
                        "day_energy_factor": float(factor),
                        "workday_factor": 1.0,
                        "nonwork_factor": (
                            float(factor) if day_group != "工作日" else np.nan
                        ),
                        "source": source,
                        "anchor_year": anchor_year,
                    }
                )
    return pd.DataFrame(rows)


def estimate_nonwork_factors(
    records: pd.DataFrame,
    *,
    anchor_year: int = 2023,
    expected_industries: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Backward-compatible two-day-type wrapper."""

    industries = (
        tuple(str(value) for value in expected_industries)
        if expected_industries is not None
        else tuple(sorted(records.industry.astype(str).unique()))
    )
    sectors = records.copy()
    if "broad_sector" not in sectors.columns:
        sectors["broad_sector"] = sectors.get("department", "")
    lookup = sectors.drop_duplicates("industry").set_index("industry").broad_sector
    baseline = pd.DataFrame(
        [
            {
                "industry": industry,
                "broad_sector": str(lookup.loc[industry]),
                "season": season,
                "day_group": day_group,
            }
            for industry in industries
            for season in SEASONS
            for day_group in LEGACY_DAY_GROUPS
        ]
    )
    legacy = records.copy()
    legacy["final_day_type"] = legacy.day_group.astype(str)
    result = estimate_day_type_factors(
        legacy,
        baseline,
        anchor_year=anchor_year,
        expected_industries=industries,
    )
    return result.loc[result.day_group.eq("工作日")].drop(columns=["day_group", "day_energy_factor", "nonwork_factor"]).merge(
        result.loc[result.day_group.eq("非工作日"), ["industry", "season", "nonwork_factor"]],
        on=["industry", "season"],
        how="left",
        validate="one_to_one",
    )


__all__ = [
    "build_baseline_shapes",
    "estimate_day_type_factors",
    "estimate_nonwork_factors",
    "weighted_median",
    "weighted_medoid",
]
