"""Fine-industry sequential shape identification and extrapolation.

The active model has one parameter set per traditional fine industry.  The
formal future-shape path first migrates four-window shares and then applies a
forecast peak-valley rate only when the per-cell flatness and historical-delta
gate is enabled.  Evening-peak and control-period routines remain diagnostic
only and do not create a continuous transformation parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import (
    BASE_MEDOID_YEAR,
    FLAT_BORDERLINE_BAND,
    FLAT_CV_THRESHOLD,
    FLAT_LOAD_FACTOR_THRESHOLD,
    HISTORY_COMPARISON_YEAR,
    PEAK_VALLEY_EVOLUTION_DELTA_LOWER_BOUND,
    PEAK_VALLEY_EVOLUTION_DELTA_UPPER_BOUND,
    PEAK_VALLEY_EXTRAPOLATION_ENABLED,
    PEAK_VALLEY_GATE_POLICY,
    PEAK_VALLEY_SUPPORTED_PHIS,
    SHAPE_NUMERICAL_TOLERANCE,
)
from .quality import SHAPE_COLS
from .shape_transform import (
    apply_proportional_window_shares,
    apply_sequential_shape_transform,
    diagnose_shape,
    normalize_shape,
    peak_valley_rate,
    window_shares,
)


S_MIN = 0.01
SUPPORTED_HORIZONS = (5, 8, 15)
EVENING_JUDGE_INDUSTRIES = frozenset(
    {
        "铁路运输业",
        "航空运输业",
        "电信、广播电视和卫星传输服务",
        "互联网和相关服务",
        "软件和信息技术服务业",
        "学校",
        "医院",
        "居民生活",
    }
)
EVENING_WINDOW_HOURS = tuple(range(17, 24))
MOVED_SHARE_THRESHOLD = 0.60
MIN_MONTH_DAYS = 5
CONTROL_RATIO = 0.85
CONTROL_CHANGE_THRESHOLD = 0.10


def _canonical_policy_boundary(value: float, boundary: float) -> float:
    """Snap binary round-off at one policy boundary to its exact value."""

    if np.isclose(
        value,
        boundary,
        rtol=0.0,
        atol=SHAPE_NUMERICAL_TOLERANCE,
    ):
        return float(boundary)
    return float(value)


def _canonical_peak_valley_delta(value: float) -> float:
    """Snap binary round-off at the two policy boundaries to exact values."""

    for boundary in (
        PEAK_VALLEY_EVOLUTION_DELTA_LOWER_BOUND,
        PEAK_VALLEY_EVOLUTION_DELTA_UPPER_BOUND,
    ):
        canonical = _canonical_policy_boundary(value, boundary)
        if canonical == boundary:
            return canonical
    return float(value)


def is_flat_curve(load_factor: float, cv: float) -> bool:
    """Return the strict 2023 flat-shape gate: LF > 0.80 and CV < 0.15."""

    try:
        lf = float(load_factor)
        coefficient = float(cv)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("load factor and CV must be finite scalars") from exc
    if not np.isfinite(lf) or not np.isfinite(coefficient):
        raise ValueError("load factor and CV must be finite")
    if lf <= 0.0 or coefficient < 0.0:
        raise ValueError("load factor must be positive and CV non-negative")
    lf = _canonical_policy_boundary(lf, FLAT_LOAD_FACTOR_THRESHOLD)
    coefficient = _canonical_policy_boundary(coefficient, FLAT_CV_THRESHOLD)
    return bool(
        lf > FLAT_LOAD_FACTOR_THRESHOLD
        and coefficient < FLAT_CV_THRESHOLD
    )


def classify_peak_valley_delta(delta_abs: float | None) -> str:
    """Classify absolute historical change against the formal gate band."""

    if delta_abs is None or pd.isna(delta_abs):
        return "insufficient_history"
    try:
        value = float(delta_abs)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "peak-valley delta_abs must be finite and non-negative"
        ) from exc
    if not np.isfinite(value) or value < 0:
        raise ValueError("peak-valley absolute change must be finite and non-negative")
    value = _canonical_peak_valley_delta(value)
    if value < PEAK_VALLEY_EVOLUTION_DELTA_LOWER_BOUND:
        return "below_lower_threshold"
    if value <= PEAK_VALLEY_EVOLUTION_DELTA_UPPER_BOUND:
        return "within_gate_band"
    return "above_upper_threshold"


def evaluate_peak_valley_evolution_gate(
    *,
    trend_eligible: bool,
    flat_curve: bool,
    delta_abs: float | None,
) -> tuple[bool, str]:
    """Apply the formal per-cell gate for peak-valley-rate evolution."""

    if not bool(trend_eligible):
        return False, "disabled_insufficient_history"
    if bool(flat_curve):
        return False, "disabled_flat_curve"
    if delta_abs is None:
        raise ValueError("eligible peak-valley gate requires delta_abs")
    try:
        value = float(delta_abs)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "peak-valley delta_abs must be finite and non-negative"
        ) from exc
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("peak-valley delta_abs must be finite and non-negative")
    value = _canonical_peak_valley_delta(value)
    band = classify_peak_valley_delta(value)
    if band == "below_lower_threshold":
        return False, "disabled_delta_below_lower_bound"
    if band == "above_upper_threshold":
        return False, "disabled_delta_above_upper_bound"
    return True, "enabled"


@dataclass(frozen=True)
class WindowMigrationEstimate:
    shares_2022: np.ndarray
    shares_2023: np.ndarray
    b_observed: np.ndarray
    beta_a_observed: float
    direction_v: np.ndarray


@dataclass(frozen=True)
class PeakValleyEstimate:
    """Observed directional peak-valley change between the two medoids."""

    rate_2022: float
    rate_2023: float
    delta_signed: float
    delta_abs: float
    diagnostic_band: str


@dataclass(frozen=True)
class PeakValleyTarget:
    """One target-year peak-valley forecast and its range protection."""

    year: int
    phi: float
    cumulative_damping: float
    requested: float
    range_limited: float
    range_clipped: bool
    trend_eligible: bool
    trend_status: str


@dataclass(frozen=True)
class EveningPeakJudgement:
    """Binary peak-position judgement retained for the change/no-change gate."""

    enabled: bool
    triggered_seasons: frozenset[str]
    triggered_month_count: int
    reason: str


@dataclass(frozen=True)
class ShapeParameterEstimate:
    """The active fine-industry parameter set."""

    window: WindowMigrationEstimate
    beta_a: float
    trend_eligible: bool
    trend_status: str


def _curve_matrix(curves: Sequence[np.ndarray] | np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(curves, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 24 or values.shape[0] == 0:
        raise ValueError(f"{label} must have shape (n, 24) with n >= 1")
    return np.vstack([normalize_shape(row) for row in values])


def _normalized_weights(weights: Sequence[float] | np.ndarray | None, n: int) -> np.ndarray:
    if weights is None:
        return np.full(n, 1.0 / n)
    values = np.asarray(weights, dtype=float)
    if values.shape != (n,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("weights must be a finite, non-negative n-vector")
    if values.sum() <= 0:
        raise ValueError("at least one weight must be positive")
    return values / values.sum()


def estimate_window_migration(
    curves_2022: Sequence[np.ndarray] | np.ndarray,
    curves_2023: Sequence[np.ndarray] | np.ndarray,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
    zero_threshold: float = 1e-8,
) -> WindowMigrationEstimate:
    """Estimate one industry's four-window migration from paired cells."""

    base = _curve_matrix(curves_2022, "curves_2022")
    target = _curve_matrix(curves_2023, "curves_2023")
    if base.shape != target.shape:
        raise ValueError("paired curve matrices must have identical shape")
    sample_weights = _normalized_weights(weights, len(base))
    shares_2022 = np.average(
        np.vstack([window_shares(row) for row in base]), axis=0, weights=sample_weights
    )
    shares_2023 = np.average(
        np.vstack([window_shares(row) for row in target]), axis=0, weights=sample_weights
    )
    difference = shares_2023 - shares_2022
    difference = difference - difference.mean()
    beta_a = float(0.5 * np.abs(difference).sum())
    if beta_a < zero_threshold:
        beta_a = 0.0
        direction = np.zeros(4, dtype=float)
    else:
        direction = difference / beta_a
    return WindowMigrationEstimate(
        shares_2022=shares_2022,
        shares_2023=shares_2023,
        b_observed=difference,
        beta_a_observed=beta_a,
        direction_v=direction,
    )


def estimate_peak_valley_change(
    curve_2022: np.ndarray,
    curve_2023: np.ndarray,
) -> PeakValleyEstimate:
    """Estimate the signed 2022-to-2023 peak-valley-rate change."""

    rate_2022 = peak_valley_rate(np.asarray(curve_2022, dtype=float))
    rate_2023 = peak_valley_rate(np.asarray(curve_2023, dtype=float))
    raw_delta_signed = float(rate_2023 - rate_2022)
    delta_abs = _canonical_peak_valley_delta(abs(raw_delta_signed))
    delta_signed = float(np.copysign(delta_abs, raw_delta_signed))
    return PeakValleyEstimate(
        rate_2022=rate_2022,
        rate_2023=rate_2023,
        delta_signed=delta_signed,
        delta_abs=delta_abs,
        diagnostic_band=classify_peak_valley_delta(delta_abs),
    )


def forecast_peak_valley_target(
    *,
    rate_2023: float,
    delta_signed: float,
    year: int,
    phi: float,
    trend_eligible: bool,
) -> PeakValleyTarget:
    """Extrapolate annual logit increments with damping from k=1 (PPT).

    Exact 0/1 rates are moved inward by 1e-12 only for the logarithm.
    The base year, no-history, no-change and zero-damping cases preserve
    the original baseline exactly. No historical rounding is performed.
    """
    if not np.isfinite(phi) or not 0.0 <= float(phi) < 1.0:
        raise ValueError("phi must be finite and in [0, 1)")
    if not np.isfinite(rate_2023) or not np.isfinite(delta_signed):
        raise ValueError("rate_2023 and delta_signed must be finite")
    if not np.isfinite(year) or not float(year).is_integer() or year < BASE_MEDOID_YEAR:
        raise ValueError("peak-valley forecast year must be an integer at or after the 2023 anchor")
    if not 0.0 <= float(rate_2023) <= 1.0:
        raise ValueError("rate_2023 must lie in [0, 1]")
    eligible = bool(trend_eligible)
    rate_2022 = float(rate_2023) - float(delta_signed)
    if eligible and not -1e-12 <= rate_2022 <= 1.0 + 1e-12:
        raise ValueError("implied rate_2022 must lie in [0, 1]")
    n = int(year) - BASE_MEDOID_YEAR
    # expm1 avoids cancellation for phi close to one.
    cumulative = (0.0 if float(phi) == 0.0 else
                  float(phi) * -np.expm1(n * np.log(float(phi))) / (1.0 - float(phi)))
    requested = float(rate_2023)
    if eligible and n and cumulative and float(delta_signed) != 0.0:
        r22, r23 = np.clip([rate_2022, rate_2023], 1e-12, 1.0 - 1e-12)
        z22, z23 = np.log([r22, r23]) - np.log1p(-np.array([r22, r23]))
        z = z23 + cumulative * (z23 - z22)
        # Stable inverse logit for either direction and long horizons.
        requested = float(1.0 / (1.0 + np.exp(-z)) if z >= 0.0 else
                          np.exp(z) / (1.0 + np.exp(z)))
    return PeakValleyTarget(
        year=int(year), phi=float(phi), cumulative_damping=float(cumulative),
        requested=requested, range_limited=requested, range_clipped=False,
        trend_eligible=eligible,
        trend_status=("paired_2022_2023_industry_medoids" if eligible else
                      "insufficient_history_no_extrapolation"),
    )


def _evening_peak_hour(curve: np.ndarray) -> int:
    values = normalize_shape(np.asarray(curve, dtype=float))
    window = values[list(EVENING_WINDOW_HOURS)]
    return EVENING_WINDOW_HOURS[int(np.argmax(window))]


def judge_evening_peak_position(
    day_records: pd.DataFrame,
    *,
    industry: str,
    base_year: int = 2022,
    compare_year: int = 2023,
    share_threshold: float = MOVED_SHARE_THRESHOLD,
    min_days: int = MIN_MONTH_DAYS,
) -> tuple[EveningPeakJudgement, pd.DataFrame]:
    """Judge evening-peak position movement without estimating a shift amount.

    For each season, a K=1 L1 medoid of all valid base-year days selects the
    only month that is judged.  The month triggers when at least 60% of
    compare-year daily peaks move later and their modal hour equals the
    compare-year monthly-typical peak hour.  Only industries on the explicit
    activity whitelist participate.
    """

    from .baseline_shapes import weighted_medoid
    from .quality import SHAPE_COLS

    industry_name = str(industry).strip()
    detail_columns = [
        "industry",
        "month",
        "season",
        "season_typical_month",
        "is_typical_day_month",
        "base_days",
        "compare_days",
        "ref_peak_hour",
        "moved_days",
        "moved_share",
        "moved_mode_hour",
        "compare_typical_peak_hour",
        "peak_position_consistent",
        "month_triggered",
        "reason",
    ]
    if industry_name not in EVENING_JUDGE_INDUSTRIES:
        return (
            EveningPeakJudgement(
                enabled=False,
                triggered_seasons=frozenset(),
                triggered_month_count=0,
                reason="该行业未启用晚峰判据",
            ),
            pd.DataFrame(columns=detail_columns),
        )

    season_of = {
        12: "冬季",
        1: "冬季",
        2: "冬季",
        3: "春季",
        4: "春季",
        5: "春季",
        6: "夏季",
        7: "夏季",
        8: "夏季",
        9: "秋季",
        10: "秋季",
        11: "秋季",
    }
    frame = day_records.copy()
    required = {"year", "month", *SHAPE_COLS}
    if not required.issubset(frame.columns):
        raise ValueError(f"晚峰判别输入缺少字段：{sorted(required - set(frame.columns))}")
    if "season" not in frame.columns:
        frame["season"] = frame.month.map(season_of)

    typical_month: dict[str, int] = {}
    for season in ("春季", "夏季", "秋季", "冬季"):
        season_days = frame.loc[frame.year.eq(base_year) & frame.season.eq(season)]
        if len(season_days):
            _, medoid_index, _ = weighted_medoid(
                season_days[SHAPE_COLS].to_numpy(dtype=float),
                np.ones(len(season_days), dtype=float),
            )
            typical_month[season] = int(season_days.month.iloc[medoid_index])

    rows: list[dict[str, object]] = []
    for month in range(1, 13):
        season = season_of[month]
        base_days = frame.loc[frame.year.eq(base_year) & frame.month.eq(month)]
        compare_days = frame.loc[frame.year.eq(compare_year) & frame.month.eq(month)]
        season_typical = typical_month.get(season)
        is_typical_month = season_typical == month
        row: dict[str, object] = {
            "industry": industry_name,
            "month": month,
            "season": season,
            "season_typical_month": season_typical if season_typical is not None else np.nan,
            "is_typical_day_month": bool(is_typical_month),
            "base_days": int(len(base_days)),
            "compare_days": int(len(compare_days)),
            "ref_peak_hour": np.nan,
            "moved_days": 0,
            "moved_share": np.nan,
            "moved_mode_hour": np.nan,
            "compare_typical_peak_hour": np.nan,
            "peak_position_consistent": False,
            "month_triggered": False,
            "reason": "样本不足",
        }
        if not is_typical_month:
            row["reason"] = (
                "非典型日所在月，不判断"
                if season_typical is not None
                else "基准年该季无有效日，无法提取典型日"
            )
        elif len(base_days) >= min_days and len(compare_days) >= min_days:
            base_typical = normalize_shape(
                base_days[SHAPE_COLS].mean(axis=0).to_numpy(dtype=float)
            )
            compare_typical = normalize_shape(
                compare_days[SHAPE_COLS].mean(axis=0).to_numpy(dtype=float)
            )
            ref_hour = _evening_peak_hour(base_typical)
            compare_peak = _evening_peak_hour(compare_typical)
            daily_peaks = compare_days[SHAPE_COLS].apply(
                lambda values: _evening_peak_hour(values.to_numpy(dtype=float)), axis=1
            )
            moved = daily_peaks[daily_peaks > ref_hour]
            moved_share = len(moved) / len(daily_peaks)
            moved_mode = int(moved.mode().iloc[0]) if not moved.empty else np.nan
            triggered = bool(
                moved_share >= share_threshold
                and not moved.empty
                and compare_peak == moved_mode
                and compare_peak > ref_hour
            )
            consistent = bool(not moved.empty and compare_peak == moved_mode)
            if triggered:
                reason = f"晚峰由{ref_hour}时后移至{compare_peak}时，后移日占比{moved_share:.0%}"
            elif moved_share < share_threshold:
                reason = "后移日占比不足60%"
            else:
                reason = "典型曲线峰位与后移日众数峰位不一致"
            row.update(
                ref_peak_hour=int(ref_hour),
                moved_days=int(len(moved)),
                moved_share=float(moved_share),
                moved_mode_hour=moved_mode,
                compare_typical_peak_hour=int(compare_peak),
                peak_position_consistent=consistent,
                month_triggered=triggered,
                reason=reason,
            )
        rows.append(row)

    detail = pd.DataFrame(rows, columns=detail_columns)
    triggered = detail.loc[detail.month_triggered]
    triggered_seasons = frozenset(triggered.season.astype(str))
    count = int(len(triggered))
    return (
        EveningPeakJudgement(
            enabled=True,
            triggered_seasons=triggered_seasons,
            triggered_month_count=count,
            reason=f"{count}个典型月份触发" if count else "无触发月",
        ),
        detail,
    )


def judge_control_period_change(
    base_curve: Sequence[float] | np.ndarray,
    compare_curve: Sequence[float] | np.ndarray,
    *,
    ratio: float = CONTROL_RATIO,
    threshold: float = CONTROL_CHANGE_THRESHOLD,
) -> dict[str, object]:
    """Judge whether load in the base curve's control hours changed materially."""

    base = normalize_shape(np.asarray(base_curve, dtype=float))
    compare = normalize_shape(np.asarray(compare_curve, dtype=float))
    control_hours = np.flatnonzero(base >= ratio * base.max())
    base_level = float(base[control_hours].mean())
    compare_level = float(compare[control_hours].mean())
    change_rate = (compare_level - base_level) / base_level if base_level > 0 else 0.0
    return {
        "control_hours": tuple(int(hour) for hour in control_hours),
        "control_level_base": base_level,
        "control_level_compare": compare_level,
        "control_change_rate": float(change_rate),
        "control_triggered": bool(abs(change_rate) > threshold),
    }


def _identify_shape_parameters_legacy(
    curves_2022: Sequence[np.ndarray] | np.ndarray,
    curves_2023: Sequence[np.ndarray] | np.ndarray | None,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
    trend_eligible: bool,
) -> ShapeParameterEstimate:
    """Identify the active four-window parameter set for one fine industry."""

    base = _curve_matrix(curves_2022, "curves_2022")
    if curves_2023 is None:
        shares = np.average(
            np.vstack([window_shares(row) for row in base]), axis=0
        )
        window = WindowMigrationEstimate(
            shares_2022=shares,
            shares_2023=shares.copy(),
            b_observed=np.zeros(4, dtype=float),
            beta_a_observed=0.0,
            direction_v=np.zeros(4, dtype=float),
        )
        return ShapeParameterEstimate(
            window=window,
            beta_a=0.0,
            trend_eligible=False,
            trend_status="insufficient_history_no_extrapolation",
        )

    target = _curve_matrix(curves_2023, "curves_2023")
    if base.shape != target.shape:
        raise ValueError("paired curve matrices must have identical shape")
    migration = estimate_window_migration(base, target, weights=weights)
    eligible = bool(trend_eligible)
    return ShapeParameterEstimate(
        window=migration,
        beta_a=float(migration.beta_a_observed) if eligible else 0.0,
        trend_eligible=eligible,
        trend_status=(
            "short_history_exploratory"
            if eligible
            else "insufficient_history_no_extrapolation"
        ),
    )


def _shape_history_day_type_column(frame: pd.DataFrame, label: str) -> str:
    for column in ("final_day_type", "day_group"):
        if column in frame.columns:
            return column
    raise ValueError(f"{label}缺少final_day_type/day_group")


def _identify_shape_history(
    baseline_shapes: pd.DataFrame,
    cell_quality: pd.DataFrame,
) -> pd.DataFrame:
    """Identify each industry's own 2022-to-2023 medoid migration."""

    baseline = baseline_shapes.copy()
    baseline_day_type = _shape_history_day_type_column(baseline, "行业medoid")
    quality = cell_quality.copy()
    quality_day_type = _shape_history_day_type_column(quality, "行业日覆盖台账")
    required_baseline = {"industry", "year", "season", *SHAPE_COLS}
    required_quality = {"industry", "year", "season", "trend_eligible"}
    missing_baseline = sorted(required_baseline.difference(baseline.columns))
    missing_quality = sorted(required_quality.difference(quality.columns))
    if missing_baseline:
        raise ValueError(f"行业medoid缺少字段：{missing_baseline}")
    if missing_quality:
        raise ValueError(f"行业日覆盖台账缺少字段：{missing_quality}")

    baseline["final_day_type"] = baseline[baseline_day_type].astype(str)
    quality["final_day_type"] = quality[quality_day_type].astype(str)
    baseline["year"] = baseline.year.astype(int)
    quality["year"] = quality.year.astype(int)
    cell_key = ["industry", "season", "final_day_type"]
    medoid_key = [*cell_key, "year"]
    if baseline.duplicated(medoid_key).any():
        duplicate = baseline.loc[
            baseline.duplicated(medoid_key, keep=False), medoid_key
        ].head(10)
        raise ValueError(f"行业medoid键重复：{duplicate.to_dict('records')}")

    quality_flags = quality.groupby(cell_key, observed=True).trend_eligible.agg(
        ["min", "max"]
    )
    if not quality_flags.empty and (
        quality_flags["min"].astype(bool) != quality_flags["max"].astype(bool)
    ).any():
        raise ValueError("行业单元跨年份trend_eligible不一致")
    quality_keys = quality[cell_key].drop_duplicates().sort_values(
        cell_key, kind="mergesort"
    )

    rows: list[dict[str, object]] = []
    for key_values in quality_keys.itertuples(index=False, name=None):
        industry, season, final_day_type = (str(value) for value in key_values)
        quality_part = quality.loc[
            quality.industry.astype(str).eq(industry)
            & quality.season.astype(str).eq(season)
            & quality.final_day_type.astype(str).eq(final_day_type)
        ]
        quality_by_year = quality_part.set_index("year")
        part = baseline.loc[
            baseline.industry.astype(str).eq(industry)
            & baseline.season.astype(str).eq(season)
            & baseline.final_day_type.astype(str).eq(final_day_type)
        ]
        row_2022 = part.loc[part.year.eq(HISTORY_COMPARISON_YEAR)]
        row_2023 = part.loc[part.year.eq(BASE_MEDOID_YEAR)]
        has_2022 = len(row_2022) == 1
        has_2023 = len(row_2023) == 1
        declared_eligible = bool(
            quality_flags.loc[(industry, season, final_day_type), "max"]
        )
        trend_eligible = bool(declared_eligible and has_2022 and has_2023)

        curve_2022 = (
            row_2022.iloc[0][SHAPE_COLS].to_numpy(dtype=float)
            if has_2022
            else None
        )
        curve_2023 = (
            row_2023.iloc[0][SHAPE_COLS].to_numpy(dtype=float)
            if has_2023
            else None
        )
        shares_2022 = (
            window_shares(curve_2022) if curve_2022 is not None else np.full(4, np.nan)
        )
        shares_2023 = (
            window_shares(curve_2023) if curve_2023 is not None else np.full(4, np.nan)
        )
        if has_2022 and has_2023:
            migration = estimate_window_migration(curve_2022, curve_2023)
            observed_b = migration.b_observed
            observed_beta = migration.beta_a_observed
        else:
            observed_b = np.full(4, np.nan)
            observed_beta = np.nan
        beta_a = float(observed_beta) if trend_eligible else 0.0
        direction = (
            observed_b / beta_a
            if trend_eligible and beta_a >= 1e-8
            else np.zeros(4, dtype=float)
        )

        diagnostic_2022 = diagnose_shape(curve_2022) if curve_2022 is not None else None
        diagnostic_2023 = diagnose_shape(curve_2023) if curve_2023 is not None else None
        pv_2022 = (
            float(diagnostic_2022["peak_valley_rate"])
            if diagnostic_2022 is not None
            else np.nan
        )
        pv_2023 = (
            float(diagnostic_2023["peak_valley_rate"])
            if diagnostic_2023 is not None
            else np.nan
        )
        if has_2022 and has_2023:
            peak_valley_estimate = estimate_peak_valley_change(
                curve_2022,
                curve_2023,
            )
            pv_2022 = peak_valley_estimate.rate_2022
            pv_2023 = peak_valley_estimate.rate_2023
            if trend_eligible:
                delta_signed = peak_valley_estimate.delta_signed
                delta_abs = peak_valley_estimate.delta_abs
                peak_valley_band = peak_valley_estimate.diagnostic_band
            else:
                delta_signed = 0.0
                delta_abs = 0.0
                peak_valley_band = classify_peak_valley_delta(None)
        else:
            # A missing paired history technically contributes no trend.  Keep
            # the availability-aware diagnostic label separate from that zero.
            delta_signed = 0.0
            delta_abs = 0.0
            peak_valley_band = classify_peak_valley_delta(None)
        load_factor_2023 = (
            _canonical_policy_boundary(
                float(diagnostic_2023["load_factor"]),
                FLAT_LOAD_FACTOR_THRESHOLD,
            )
            if diagnostic_2023 is not None
            else np.nan
        )
        cv_2023 = (
            _canonical_policy_boundary(
                float(diagnostic_2023["cv"]),
                FLAT_CV_THRESHOLD,
            )
            if diagnostic_2023 is not None
            else np.nan
        )
        flat_curve = bool(
            diagnostic_2023 is not None
            and is_flat_curve(load_factor_2023, cv_2023)
        )
        flat_borderline = bool(
            diagnostic_2023 is not None
            and (
                abs(load_factor_2023 - FLAT_LOAD_FACTOR_THRESHOLD)
                <= FLAT_BORDERLINE_BAND
                or abs(cv_2023 - FLAT_CV_THRESHOLD) <= FLAT_BORDERLINE_BAND
            )
        )
        source = row_2023.iloc[0] if has_2023 else (row_2022.iloc[0] if has_2022 else None)
        department = (
            str(
                source.get(
                    "department", source.get("broad_sector", "")
                )
            )
            if source is not None
            else ""
        )
        trend_status = (
            "paired_2022_2023_industry_medoids"
            if trend_eligible
            else "insufficient_history_no_extrapolation"
        )
        peak_valley_evolution_enabled, peak_valley_gate_reason = (
            evaluate_peak_valley_evolution_gate(
                trend_eligible=trend_eligible,
                flat_curve=flat_curve,
                delta_abs=delta_abs if trend_eligible else None,
            )
        )
        rows.append(
            {
                "department": department,
                "broad_sector": department,
                "industry": industry,
                "season": season,
                "final_day_type": final_day_type,
                "day_group": final_day_type,
                "base_source_year": BASE_MEDOID_YEAR,
                "history_comparison_year": HISTORY_COMPARISON_YEAR,
                "baseline_available_2023": has_2023,
                "history_available_2022": has_2022,
                "valid_industry_days_2022": (
                    int(quality_by_year.loc[2022, "valid_days"])
                    if "valid_days" in quality_by_year.columns
                    and 2022 in quality_by_year.index
                    else np.nan
                ),
                "valid_industry_days_2023": (
                    int(quality_by_year.loc[2023, "valid_days"])
                    if "valid_days" in quality_by_year.columns
                    and 2023 in quality_by_year.index
                    else np.nan
                ),
                "holiday_merged_with_weekend": bool(
                    quality_part.holiday_merged_with_weekend.astype(bool).iloc[0]
                )
                if "holiday_merged_with_weekend" in quality_part.columns
                and not quality_part.empty
                else False,
                "trend_eligible": trend_eligible,
                "trend_status": trend_status,
                "shares_2022": shares_2022,
                "shares_2023": shares_2023,
                "b_observed": observed_b,
                "beta_a_observed": observed_beta,
                "beta_a": beta_a,
                "direction_v": direction,
                "night_share_2022": shares_2022[0],
                "morning_share_2022": shares_2022[1],
                "midday_share_2022": shares_2022[2],
                "evening_share_2022": shares_2022[3],
                "night_share_2023": shares_2023[0],
                "morning_share_2023": shares_2023[1],
                "midday_share_2023": shares_2023[2],
                "evening_share_2023": shares_2023[3],
                "b_night": observed_b[0],
                "b_morning": observed_b[1],
                "b_midday": observed_b[2],
                "b_evening": observed_b[3],
                "v_night": direction[0],
                "v_morning": direction[1],
                "v_midday": direction[2],
                "v_evening": direction[3],
                "peak_valley_rate_2022": pv_2022,
                "peak_valley_rate_2023": pv_2023,
                "peak_valley_delta_signed": delta_signed,
                "peak_valley_delta_abs": delta_abs,
                "peak_valley_delta_band": peak_valley_band,
                "load_factor_2023": load_factor_2023,
                "cv_2023": cv_2023,
                "flat_curve": flat_curve,
                "flat_borderline": flat_borderline,
                "peak_valley_extrapolation_enabled": PEAK_VALLEY_EXTRAPOLATION_ENABLED,
                "peak_valley_evolution_enabled": peak_valley_evolution_enabled,
                "peak_valley_gate_reason": peak_valley_gate_reason,
                "peak_valley_gate_policy": PEAK_VALLEY_GATE_POLICY,
                "baseline_evening_peak_hour_2023": (
                    diagnostic_2023["evening_peak_hour"]
                    if diagnostic_2023 is not None
                    else np.nan
                ),
                "baseline_raw_85pct_hours_2023": (
                    diagnostic_2023["raw_85pct_hours"]
                    if diagnostic_2023 is not None
                    else ()
                ),
                "baseline_effective_control_hours_2023": (
                    diagnostic_2023["effective_control_hours"]
                    if diagnostic_2023 is not None
                    else ()
                ),
            }
        )
    return pd.DataFrame(rows)


def identify_shape_parameters(
    baseline_shapes: pd.DataFrame | Sequence[np.ndarray] | np.ndarray,
    cell_quality: pd.DataFrame | Sequence[np.ndarray] | np.ndarray | None,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
    trend_eligible: bool | None = None,
) -> pd.DataFrame | ShapeParameterEstimate:
    """Identify formal shape history, with legacy per-cell array support."""

    if isinstance(baseline_shapes, pd.DataFrame) and isinstance(
        cell_quality, pd.DataFrame
    ):
        if weights is not None or trend_eligible is not None:
            raise ValueError("DataFrame形状参数识别不接受用户权重或外部趋势开关")
        return _identify_shape_history(baseline_shapes, cell_quality)
    if trend_eligible is None:
        raise TypeError("legacy array calls must provide trend_eligible")
    return _identify_shape_parameters_legacy(
        baseline_shapes,
        cell_quality,
        weights=weights,
        trend_eligible=trend_eligible,
    )


def extrapolated_increment(
    beta: float,
    year: int | float,
    horizon: int,
    *,
    base_year: int = 2023,
) -> float:
    """Return beta*H*(1-exp(-(year-2023)/H)), anchored at zero in 2023."""

    if horizon not in SUPPORTED_HORIZONS:
        raise ValueError(f"horizon must be one of {SUPPORTED_HORIZONS}")
    if not np.isfinite(beta) or not np.isfinite(year):
        raise ValueError("beta and year must be finite")
    if year < base_year:
        raise ValueError("future extrapolation is defined only from the 2023 anchor")
    return float(beta * horizon * (1.0 - np.exp(-(float(year) - base_year) / horizon)))


def constrain_window_migration(
    base_shares: np.ndarray,
    direction: np.ndarray | None = None,
    requested: float | None = None,
    s_min: float = S_MIN,
    *,
    direction_v: np.ndarray | None = None,
    requested_a: float | None = None,
) -> tuple[float, float, bool]:
    """Return ``a_max``, applied migration and whether the boundary was hit."""

    shares = np.asarray(base_shares, dtype=float)
    if direction is not None and direction_v is not None:
        raise TypeError("provide direction or direction_v, not both")
    if requested is not None and requested_a is not None:
        raise TypeError("provide requested or requested_a, not both")
    selected_direction = direction if direction is not None else direction_v
    selected_requested = requested if requested is not None else requested_a
    if selected_direction is None or selected_requested is None:
        raise TypeError("direction and requested migration are required")
    direction_values = np.asarray(selected_direction, dtype=float)
    if shares.shape != (4,) or direction_values.shape != (4,):
        raise ValueError("base_shares and direction must be four-vectors")
    if not np.isfinite(shares).all() or not np.isfinite(direction_values).all():
        raise ValueError("window inputs must be finite")
    if not np.isclose(shares.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("base shares must sum to one")
    if np.any(shares < s_min - 1e-12):
        raise ValueError("base shares already violate S_min")
    if not np.isclose(direction_values.sum(), 0.0, atol=1e-12, rtol=0.0):
        raise ValueError("migration direction must sum to zero")
    requested_value = float(selected_requested)
    if not np.isfinite(requested_value) or requested_value < 0:
        raise ValueError("requested migration strength must be finite and non-negative")

    negative = direction_values < 0
    if negative.any():
        a_max = float(
            np.min(
                (shares[negative] - float(s_min))
                / (-direction_values[negative])
            )
        )
    else:
        a_max = float("inf")
    applied = min(requested_value, a_max)
    if applied < -1e-12:
        raise ValueError("四时段迁移可行幅度为负")
    applied = max(applied, 0.0)
    target = shares + applied * direction_values
    if np.any(target < s_min - 1e-12):
        raise RuntimeError("failed to enforce S_min")
    return a_max, applied, bool(requested_value > a_max)


def forecast_parameter_trajectory(
    estimate: ShapeParameterEstimate,
    years: Iterable[int],
    *,
    base_window_shares: np.ndarray,
    horizons: Iterable[int] = SUPPORTED_HORIZONS,
    s_min: float = S_MIN,
) -> pd.DataFrame:
    """Build the H=5/8/15 four-window trajectories; H=8 is formal."""

    year_values = tuple(int(year) for year in years)
    if not year_values:
        raise ValueError("future parameter years must not be empty")
    base_shares = np.asarray(base_window_shares, dtype=float)
    rows: list[dict[str, object]] = []
    for horizon in tuple(int(value) for value in horizons):
        if horizon not in SUPPORTED_HORIZONS:
            raise ValueError(f"unsupported horizon: {horizon}")
        for year in year_values:
            requested_a = extrapolated_increment(estimate.beta_a, year, horizon)
            a_max, used_a, boundary_hit = constrain_window_migration(
                base_shares,
                estimate.window.direction_v,
                requested_a,
                s_min=s_min,
            )
            requested_shares = (
                base_shares + requested_a * estimate.window.direction_v
            )
            target_shares = base_shares + used_a * estimate.window.direction_v
            if not np.isclose(target_shares.sum(), 1.0, rtol=0.0, atol=1e-12):
                raise RuntimeError("applied window shares do not sum to one")
            rows.append(
                {
                    "year": year,
                    "H": horizon,
                    "formal_H8": bool(horizon == 8),
                    "a_requested": requested_a,
                    "a_applied": used_a,
                    "a_max": a_max,
                    "a_max_unbounded": bool(not np.isfinite(a_max)),
                    "window_boundary_hit": boundary_hit,
                    "requested_share_night": requested_shares[0],
                    "requested_share_morning": requested_shares[1],
                    "requested_share_midday": requested_shares[2],
                    "requested_share_evening": requested_shares[3],
                    "applied_share_night": target_shares[0],
                    "applied_share_morning": target_shares[1],
                    "applied_share_midday": target_shares[2],
                    "applied_share_evening": target_shares[3],
                    # Stable aliases consumed by pre-3.3 assembly helpers.
                    "night_share": target_shares[0],
                    "morning_share": target_shares[1],
                    "midday_share": target_shares[2],
                    "evening_share": target_shares[3],
                    "trend_status": estimate.trend_status,
                }
            )
    return pd.DataFrame.from_records(rows)


def shape_for_trajectory_row(base_shape: np.ndarray, row: Mapping[str, object]) -> np.ndarray:
    """Apply one trajectory row and return the final second-stage shape."""

    target_shares = np.asarray(
        [
            row["applied_share_night"]
            if "applied_share_night" in row
            else row["night_share"],
            row["applied_share_morning"]
            if "applied_share_morning" in row
            else row["morning_share"],
            row["applied_share_midday"]
            if "applied_share_midday" in row
            else row["midday_share"],
            row["applied_share_evening"]
            if "applied_share_evening" in row
            else row["evening_share"],
        ],
        dtype=float,
    )
    if "peak_valley_target_range_limited" in row:
        target_peak_valley_rate = float(row["peak_valley_target_range_limited"])
    elif "peak_valley_target_applied" in row:
        target_peak_valley_rate = float(row["peak_valley_target_applied"])
    else:
        # Legacy trajectory rows predate the second-stage target.  Their
        # compatibility behavior is an identity peak-valley stage, not a
        # different active transformation path.
        intermediate = apply_proportional_window_shares(base_shape, target_shares)
        target_peak_valley_rate = peak_valley_rate(intermediate)
    gate_value = row.get("peak_valley_evolution_enabled", True)
    if not isinstance(gate_value, (bool, np.bool_)):
        raise ValueError("peak_valley_evolution_enabled must be boolean")
    peak_valley_evolution_enabled = bool(gate_value)
    peak_valley_gate_reason = str(
        row.get(
            "peak_valley_gate_reason",
            "enabled" if peak_valley_evolution_enabled else "disabled_unspecified",
        )
    )
    return apply_sequential_shape_transform(
        base_shape,
        target_shares,
        target_peak_valley_rate,
        peak_valley_evolution_enabled=peak_valley_evolution_enabled,
        peak_valley_gate_reason=peak_valley_gate_reason,
    ).final_shape


__all__ = [
    "CONTROL_CHANGE_THRESHOLD",
    "CONTROL_RATIO",
    "EVENING_JUDGE_INDUSTRIES",
    "EVENING_WINDOW_HOURS",
    "MIN_MONTH_DAYS",
    "MOVED_SHARE_THRESHOLD",
    "S_MIN",
    "SUPPORTED_HORIZONS",
    "EveningPeakJudgement",
    "PeakValleyEstimate",
    "PeakValleyTarget",
    "ShapeParameterEstimate",
    "WindowMigrationEstimate",
    "classify_peak_valley_delta",
    "constrain_window_migration",
    "estimate_window_migration",
    "estimate_peak_valley_change",
    "evaluate_peak_valley_evolution_gate",
    "extrapolated_increment",
    "forecast_parameter_trajectory",
    "forecast_peak_valley_target",
    "identify_shape_parameters",
    "is_flat_curve",
    "judge_control_period_change",
    "judge_evening_peak_position",
    "shape_for_trajectory_row",
]
