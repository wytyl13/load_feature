"""Baseline-only local shape adjustments for stage 5 of the load model.

The routines in this module deliberately operate on the normalized 2022
baseline.  A 2023 curve may supply *targets* for the control-period step, but
is never returned as a replacement curve.  Both the control-period adjustment
and the evening-peak adjustment are transactional: a failed validation returns
an unchanged copy of that step's input together with explicit diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


HOURS_PER_DAY = 24
EVENING_HOURS = tuple(range(17, 24))

FLAT_LOAD_FACTOR_THRESHOLD = 0.80
FLAT_CV_THRESHOLD = 0.15
CONTROL_RATIO = 0.85
PEAK_VALLEY_RATE_CHANGE_THRESHOLD = 0.065
CONTROL_DURATION_CHANGE_MIN = 2
MIN_CONTROL_PERIOD_HOURS = 3

NUMERIC_TOLERANCE = 1e-12
SUM_TOLERANCE = 1e-10
RATIO_TOLERANCE = 1e-10
CONTROL_MEMBER_MARGIN = 1e-5


@dataclass(frozen=True)
class FlatnessDiagnostics:
    """The two-factor flatness gate evaluated on a daily curve."""

    load_factor: float
    coefficient_of_variation: float
    is_flat: bool


@dataclass(frozen=True)
class ControlSetDiagnostics:
    """Raw and effective 85%-of-peak control-period identification."""

    threshold: float
    raw_hours: tuple[int, ...]
    raw_runs: tuple[tuple[int, ...], ...]
    valid_runs: tuple[tuple[int, ...], ...]
    short_runs: tuple[tuple[int, ...], ...]
    hours: tuple[int, ...]
    duration: int
    raw_full_day: bool


@dataclass(frozen=True)
class ControlAdjustmentDiagnostics:
    """Decision, transaction status, and invariants for control adjustment."""

    status: str
    reason: str
    triggered: bool
    peak_valley_triggered: bool
    duration_triggered: bool
    applied: bool
    baseline_peak_valley_rate: float
    target_peak_valley_rate: float | None
    absolute_peak_valley_change: float | None
    absolute_duration_change: int | None
    baseline_control: ControlSetDiagnostics
    target_control: ControlSetDiagnostics | None
    target_hours: tuple[int, ...]
    baseline_peak_hours: tuple[int, ...]
    baseline_valley_hours: tuple[int, ...]
    target_peak_level: float | None = None
    target_member_floor: float | None = None
    checks: tuple[tuple[str, bool], ...] = ()


@dataclass(frozen=True)
class EveningAdjustmentDiagnostics:
    """Decision, transfer details, and invariants for evening-peak movement."""

    status: str
    reason: str
    requested: bool
    applied: bool
    method: str | None
    source_hour: int | None
    target_hour: int | None
    shift_hours: int | None
    peak_tolerance: float | None
    near_peak_hours: tuple[int, ...] = ()
    plateau_hours: tuple[int, ...] = ()
    left_anchor: int | None = None
    right_anchor: int | None = None
    left_anchor_kind: str | None = None
    right_anchor_kind: str | None = None
    residual_support: tuple[int, ...] = ()
    movable_hours: tuple[int, ...] = ()
    moved_residual: float = 0.0
    net_removed_area: float = 0.0
    net_added_area: float = 0.0
    joint_control_constraint: bool = False
    checks: tuple[tuple[str, bool], ...] = ()


@dataclass(frozen=True)
class ControlAdjustmentResult:
    curve: np.ndarray
    diagnostics: ControlAdjustmentDiagnostics


@dataclass(frozen=True)
class EveningAdjustmentResult:
    curve: np.ndarray
    diagnostics: EveningAdjustmentDiagnostics


@dataclass(frozen=True)
class LocalShapeAdjustmentDiagnostics:
    """Combined diagnostics for the ordered control then evening transaction."""

    status: str
    sample_evaluable: bool
    flatness: FlatnessDiagnostics
    control: ControlAdjustmentDiagnostics
    evening: EveningAdjustmentDiagnostics
    curve_changed: bool


@dataclass(frozen=True)
class LocalShapeAdjustmentResult:
    curve: np.ndarray
    diagnostics: LocalShapeAdjustmentDiagnostics


def _as_curve(
    values: Sequence[float] | np.ndarray,
    *,
    name: str,
    require_unit_sum: bool,
) -> np.ndarray:
    """Return a defensive float copy after enforcing the public API contract."""

    curve = np.asarray(values, dtype=float)
    if curve.shape != (HOURS_PER_DAY,):
        raise ValueError(f"{name} must contain exactly 24 hourly values")
    if not np.all(np.isfinite(curve)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(curve < -NUMERIC_TOLERANCE):
        raise ValueError(f"{name} must be non-negative")
    curve = curve.copy()
    curve[curve < 0.0] = 0.0
    total = float(curve.sum())
    if total <= NUMERIC_TOLERANCE:
        raise ValueError(f"{name} must have positive daily energy")
    if require_unit_sum and not np.isclose(total, 1.0, atol=SUM_TOLERANCE, rtol=0.0):
        raise ValueError(f"{name} must be normalized to a daily sum of 1")
    return curve


def load_factor(curve: Sequence[float] | np.ndarray) -> float:
    """Daily load factor: mean divided by maximum."""

    values = _as_curve(curve, name="curve", require_unit_sum=False)
    return float(values.mean() / values.max())


def coefficient_of_variation(curve: Sequence[float] | np.ndarray) -> float:
    """Population coefficient of variation: standard deviation / mean."""

    values = _as_curve(curve, name="curve", require_unit_sum=False)
    return float(values.std(ddof=0) / values.mean())


def assess_flatness(curve: Sequence[float] | np.ndarray) -> FlatnessDiagnostics:
    """Apply the strict flat gate: LF > 0.8 and CV < 0.15."""

    lf = load_factor(curve)
    cv = coefficient_of_variation(curve)
    return FlatnessDiagnostics(
        load_factor=lf,
        coefficient_of_variation=cv,
        # Keep mathematically exact boundary values on the non-flat side even
        # when their binary representation lands a few ulps across a cutoff.
        is_flat=bool(
            lf > FLAT_LOAD_FACTOR_THRESHOLD + NUMERIC_TOLERANCE
            and cv < FLAT_CV_THRESHOLD - NUMERIC_TOLERANCE
        ),
    )


def peak_valley_rate(curve: Sequence[float] | np.ndarray) -> float:
    """Peak-valley difference rate, (max - min) / max."""

    values = _as_curve(curve, name="curve", require_unit_sum=False)
    maximum = float(values.max())
    return float((maximum - float(values.min())) / maximum)


def circular_runs(mask: Sequence[bool] | np.ndarray) -> tuple[tuple[int, ...], ...]:
    """Return maximal true runs on a 24-hour ring.

    A run crossing midnight is represented in circular order, for example
    ``(23, 0, 1)``.  The all-true mask is represented by one 24-hour run.
    """

    flags = np.asarray(mask, dtype=bool)
    if flags.shape != (HOURS_PER_DAY,):
        raise ValueError("mask must contain exactly 24 hourly flags")
    if not flags.any():
        return ()
    if flags.all():
        return (tuple(range(HOURS_PER_DAY)),)

    first_false = int(np.flatnonzero(~flags)[0])
    runs: list[tuple[int, ...]] = []
    current: list[int] = []
    for step in range(1, HOURS_PER_DAY + 1):
        hour = (first_false + step) % HOURS_PER_DAY
        if flags[hour]:
            current.append(hour)
        elif current:
            runs.append(tuple(current))
            current = []
    if current:
        runs.append(tuple(current))
    return tuple(runs)


def identify_control_periods(
    curve: Sequence[float] | np.ndarray,
    *,
    ratio: float = CONTROL_RATIO,
    minimum_hours: int = MIN_CONTROL_PERIOD_HOURS,
) -> ControlSetDiagnostics:
    """Identify effective control periods from circular 85%-threshold runs."""

    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    if minimum_hours < 1 or minimum_hours > HOURS_PER_DAY:
        raise ValueError("minimum_hours must be between 1 and 24")
    values = _as_curve(curve, name="curve", require_unit_sum=False)
    threshold = float(ratio * values.max())
    mask = values >= threshold - NUMERIC_TOLERANCE
    raw_runs = circular_runs(mask)
    valid_runs = tuple(run for run in raw_runs if len(run) >= minimum_hours)
    short_runs = tuple(run for run in raw_runs if len(run) < minimum_hours)
    hours = tuple(sorted(hour for run in valid_runs for hour in run))
    return ControlSetDiagnostics(
        threshold=threshold,
        raw_hours=tuple(int(hour) for hour in np.flatnonzero(mask)),
        raw_runs=raw_runs,
        valid_runs=valid_runs,
        short_runs=short_runs,
        hours=hours,
        duration=len(hours),
        raw_full_day=bool(mask.all()),
    )


def evening_peak_hour(curve: Sequence[float] | np.ndarray) -> int:
    """Earliest exact maximum in the absolute 17:00--23:00 window."""

    values = _as_curve(curve, name="curve", require_unit_sum=False)
    evening = values[np.asarray(EVENING_HOURS)]
    return EVENING_HOURS[int(np.argmax(evening))]


def _extreme_hours(values: np.ndarray, *, maximum: bool) -> tuple[int, ...]:
    extreme = float(values.max() if maximum else values.min())
    return tuple(
        int(hour)
        for hour in np.flatnonzero(np.abs(values - extreme) <= NUMERIC_TOLERANCE)
    )


def _check_map(items: Iterable[tuple[str, bool]]) -> tuple[tuple[str, bool], ...]:
    return tuple((name, bool(value)) for name, value in items)


def _noop_control_diagnostics(
    baseline: np.ndarray,
    *,
    status: str,
    reason: str,
    target: np.ndarray | None = None,
    triggered: bool = False,
    peak_valley_triggered: bool = False,
    duration_triggered: bool = False,
) -> ControlAdjustmentDiagnostics:
    baseline_control = identify_control_periods(baseline)
    target_control = identify_control_periods(target) if target is not None else None
    baseline_rate = peak_valley_rate(baseline)
    target_rate = peak_valley_rate(target) if target is not None else None
    return ControlAdjustmentDiagnostics(
        status=status,
        reason=reason,
        triggered=triggered,
        peak_valley_triggered=peak_valley_triggered,
        duration_triggered=duration_triggered,
        applied=False,
        baseline_peak_valley_rate=baseline_rate,
        target_peak_valley_rate=target_rate,
        absolute_peak_valley_change=(
            abs(target_rate - baseline_rate) if target_rate is not None else None
        ),
        absolute_duration_change=(
            abs(target_control.duration - baseline_control.duration)
            if target_control is not None
            else None
        ),
        baseline_control=baseline_control,
        target_control=target_control,
        target_hours=target_control.hours if target_control is not None else (),
        baseline_peak_hours=_extreme_hours(baseline, maximum=True),
        baseline_valley_hours=_extreme_hours(baseline, maximum=False),
    )


def _control_failure(
    baseline: np.ndarray,
    diagnostics: ControlAdjustmentDiagnostics,
    *,
    status: str,
    reason: str,
    target_peak_level: float | None = None,
    target_member_floor: float | None = None,
    checks: tuple[tuple[str, bool], ...] = (),
) -> ControlAdjustmentResult:
    return ControlAdjustmentResult(
        curve=baseline.copy(),
        diagnostics=ControlAdjustmentDiagnostics(
            status=status,
            reason=reason,
            triggered=diagnostics.triggered,
            peak_valley_triggered=diagnostics.peak_valley_triggered,
            duration_triggered=diagnostics.duration_triggered,
            applied=False,
            baseline_peak_valley_rate=diagnostics.baseline_peak_valley_rate,
            target_peak_valley_rate=diagnostics.target_peak_valley_rate,
            absolute_peak_valley_change=diagnostics.absolute_peak_valley_change,
            absolute_duration_change=diagnostics.absolute_duration_change,
            baseline_control=diagnostics.baseline_control,
            target_control=diagnostics.target_control,
            target_hours=diagnostics.target_hours,
            baseline_peak_hours=diagnostics.baseline_peak_hours,
            baseline_valley_hours=diagnostics.baseline_valley_hours,
            target_peak_level=target_peak_level,
            target_member_floor=target_member_floor,
            checks=checks,
        ),
    )


def adjust_control_period(
    baseline_2022: Sequence[float] | np.ndarray,
    target_2023: Sequence[float] | np.ndarray,
    *,
    apply_flat_gate: bool = True,
) -> ControlAdjustmentResult:
    """Transactionally stretch the target control hours on the 2022 baseline.

    The 2023 curve supplies only the target peak-valley rate and effective
    control-hour set.  It is never copied into the output.
    """

    baseline = _as_curve(
        baseline_2022, name="baseline_2022", require_unit_sum=True
    )
    target = _as_curve(target_2023, name="target_2023", require_unit_sum=True)

    if apply_flat_gate and assess_flatness(baseline).is_flat:
        diagnostics = _noop_control_diagnostics(
            baseline,
            status="flat_gate_skipped",
            reason="LF > 0.8 and CV < 0.15; local control adjustment is disabled",
            target=target,
        )
        return ControlAdjustmentResult(baseline.copy(), diagnostics)

    base_set = identify_control_periods(baseline)
    target_set = identify_control_periods(target)
    base_rate = peak_valley_rate(baseline)
    target_rate = peak_valley_rate(target)
    rate_change = abs(target_rate - base_rate)
    duration_change = abs(target_set.duration - base_set.duration)
    rate_trigger = bool(
        rate_change + NUMERIC_TOLERANCE >= PEAK_VALLEY_RATE_CHANGE_THRESHOLD
    )
    duration_trigger = bool(
        duration_change >= CONTROL_DURATION_CHANGE_MIN
    )
    triggered = bool(rate_trigger and duration_trigger)
    baseline_peaks = _extreme_hours(baseline, maximum=True)
    baseline_valleys = _extreme_hours(baseline, maximum=False)
    diagnostics = ControlAdjustmentDiagnostics(
        status="eligible" if triggered else "trigger_not_met",
        reason=(
            "both peak-valley-rate and effective-duration triggers are met"
            if triggered
            else "control adjustment requires |D23-D22| >= 0.065 and |H23-H22| >= 2"
        ),
        triggered=triggered,
        peak_valley_triggered=rate_trigger,
        duration_triggered=duration_trigger,
        applied=False,
        baseline_peak_valley_rate=base_rate,
        target_peak_valley_rate=target_rate,
        absolute_peak_valley_change=rate_change,
        absolute_duration_change=duration_change,
        baseline_control=base_set,
        target_control=target_set,
        target_hours=target_set.hours,
        baseline_peak_hours=baseline_peaks,
        baseline_valley_hours=baseline_valleys,
    )
    if not triggered:
        return ControlAdjustmentResult(baseline.copy(), diagnostics)

    if base_set.raw_full_day or target_set.raw_full_day:
        return _control_failure(
            baseline,
            diagnostics,
            status="inapplicable_raw_full_day",
            reason="the raw 85% threshold set covers all 24 hours",
        )
    if not target_set.hours:
        return _control_failure(
            baseline,
            diagnostics,
            status="inapplicable_empty_target_period",
            reason="the 2023 curve has no effective control run of at least 3 hours",
        )

    target_hours = set(target_set.hours)
    if not set(baseline_peaks).issubset(target_hours):
        return _control_failure(
            baseline,
            diagnostics,
            status="inapplicable_baseline_peak_outside_target",
            reason="at least one 2022 peak hour is outside the target control set",
        )
    if not set(baseline_valleys).isdisjoint(target_hours):
        return _control_failure(
            baseline,
            diagnostics,
            status="inapplicable_baseline_valley_inside_target",
            reason="at least one 2022 valley hour falls inside the target control set",
        )

    selected = np.asarray(target_set.hours, dtype=int)
    selected_values = baseline[selected]
    source_low = float(selected_values.min())
    source_high = float(selected_values.max())
    if source_high - source_low <= NUMERIC_TOLERANCE:
        return _control_failure(
            baseline,
            diagnostics,
            status="inapplicable_zero_selected_range",
            reason="2022 values in the target control set have no usable range",
        )
    if target_rate >= 1.0 - NUMERIC_TOLERANCE:
        return _control_failure(
            baseline,
            diagnostics,
            status="inapplicable_target_rate_near_one",
            reason="target peak-valley rate is too close to one",
        )

    baseline_minimum = float(baseline.min())
    target_peak_level = baseline_minimum / (1.0 - target_rate)
    target_member_floor = CONTROL_RATIO * target_peak_level + CONTROL_MEMBER_MARGIN
    if target_peak_level <= 0.0:
        return _control_failure(
            baseline,
            diagnostics,
            status="inapplicable_nonpositive_target_peak",
            reason="the target peak implied by the 2022 valley is not positive",
            target_peak_level=target_peak_level,
            target_member_floor=target_member_floor,
        )
    if not (
        baseline_minimum < target_member_floor <= target_peak_level
        and target_peak_level - target_member_floor > NUMERIC_TOLERANCE
    ):
        return _control_failure(
            baseline,
            diagnostics,
            status="inapplicable_target_levels",
            reason="the target member floor is not strictly between valley and peak",
            target_peak_level=target_peak_level,
            target_member_floor=target_member_floor,
        )

    outside = np.asarray(
        [hour for hour in range(HOURS_PER_DAY) if hour not in target_hours], dtype=int
    )
    if outside.size == 0 or not (
        float(baseline[outside].max())
        < CONTROL_RATIO * target_peak_level - NUMERIC_TOLERANCE
    ):
        return _control_failure(
            baseline,
            diagnostics,
            status="inapplicable_outside_threshold_conflict",
            reason="an hour outside the target set would enter the 85% control set",
            target_peak_level=target_peak_level,
            target_member_floor=target_member_floor,
        )

    candidate = baseline.copy()
    candidate[selected] = target_member_floor + (
        target_peak_level - target_member_floor
    ) * (selected_values - source_low) / (source_high - source_low)
    candidate /= float(candidate.sum())

    candidate_control = identify_control_periods(candidate)
    rounded_control = identify_control_periods(np.round(candidate, 5))
    candidate_peaks = _extreme_hours(candidate, maximum=True)
    candidate_valleys = _extreme_hours(candidate, maximum=False)

    positive_outside = outside[baseline[outside] > NUMERIC_TOLERANCE]
    if positive_outside.size:
        outside_ratios = candidate[positive_outside] / baseline[positive_outside]
        outside_common_scale = bool(
            np.max(np.abs(outside_ratios - outside_ratios[0]))
            <= RATIO_TOLERANCE * max(1.0, abs(float(outside_ratios[0])))
        )
    else:
        outside_common_scale = True
    outside_zero_preserved = bool(
        np.all(np.abs(candidate[outside[baseline[outside] <= NUMERIC_TOLERANCE]]) <= NUMERIC_TOLERANCE)
    )

    checks = _check_map(
        (
            ("finite", np.all(np.isfinite(candidate))),
            ("nonnegative", np.all(candidate >= -NUMERIC_TOLERANCE)),
            ("unit_sum", np.isclose(candidate.sum(), 1.0, atol=SUM_TOLERANCE, rtol=0.0)),
            (
                "target_peak_valley_rate",
                np.isclose(
                    peak_valley_rate(candidate),
                    target_rate,
                    atol=RATIO_TOLERANCE,
                    rtol=0.0,
                ),
            ),
            ("target_control_set", candidate_control.hours == target_set.hours),
            ("rounded_target_control_set", rounded_control.hours == target_set.hours),
            ("outside_common_scale", outside_common_scale),
            ("outside_zero_preserved", outside_zero_preserved),
            ("peak_hours_preserved", candidate_peaks == baseline_peaks),
            ("valley_hours_preserved", candidate_valleys == baseline_valleys),
        )
    )
    if not all(passed for _, passed in checks):
        failed = ", ".join(name for name, passed in checks if not passed)
        return _control_failure(
            baseline,
            diagnostics,
            status="transaction_validation_failed",
            reason=f"control candidate failed: {failed}",
            target_peak_level=target_peak_level,
            target_member_floor=target_member_floor,
            checks=checks,
        )

    return ControlAdjustmentResult(
        curve=candidate.copy(),
        diagnostics=ControlAdjustmentDiagnostics(
            status="applied",
            reason="control-period linear stretch and unified normalization committed",
            triggered=True,
            peak_valley_triggered=True,
            duration_triggered=True,
            applied=True,
            baseline_peak_valley_rate=base_rate,
            target_peak_valley_rate=target_rate,
            absolute_peak_valley_change=rate_change,
            absolute_duration_change=duration_change,
            baseline_control=base_set,
            target_control=target_set,
            target_hours=target_set.hours,
            baseline_peak_hours=baseline_peaks,
            baseline_valley_hours=baseline_valleys,
            target_peak_level=target_peak_level,
            target_member_floor=target_member_floor,
            checks=checks,
        ),
    )


def _noop_evening_diagnostics(
    *,
    status: str,
    reason: str,
    requested: bool,
    source_hour: int | None,
    target_hour: int | None,
    joint_control_constraint: bool = False,
) -> EveningAdjustmentDiagnostics:
    return EveningAdjustmentDiagnostics(
        status=status,
        reason=reason,
        requested=requested,
        applied=False,
        method=None,
        source_hour=source_hour,
        target_hour=target_hour,
        shift_hours=(
            target_hour - source_hour
            if source_hour is not None and target_hour is not None
            else None
        ),
        peak_tolerance=None,
        joint_control_constraint=joint_control_constraint,
    )


def _evening_failure(
    original: np.ndarray,
    diagnostics: EveningAdjustmentDiagnostics,
    *,
    status: str,
    reason: str,
    checks: tuple[tuple[str, bool], ...] = (),
) -> EveningAdjustmentResult:
    return EveningAdjustmentResult(
        curve=original.copy(),
        diagnostics=EveningAdjustmentDiagnostics(
            status=status,
            reason=reason,
            requested=diagnostics.requested,
            applied=False,
            method=diagnostics.method,
            source_hour=diagnostics.source_hour,
            target_hour=diagnostics.target_hour,
            shift_hours=diagnostics.shift_hours,
            peak_tolerance=diagnostics.peak_tolerance,
            near_peak_hours=diagnostics.near_peak_hours,
            plateau_hours=diagnostics.plateau_hours,
            left_anchor=diagnostics.left_anchor,
            right_anchor=diagnostics.right_anchor,
            left_anchor_kind=diagnostics.left_anchor_kind,
            right_anchor_kind=diagnostics.right_anchor_kind,
            residual_support=diagnostics.residual_support,
            movable_hours=diagnostics.movable_hours,
            moved_residual=diagnostics.moved_residual,
            net_removed_area=diagnostics.net_removed_area,
            net_added_area=diagnostics.net_added_area,
            joint_control_constraint=diagnostics.joint_control_constraint,
            checks=checks,
        ),
    )


def _validate_evening_candidate(
    original: np.ndarray,
    candidate: np.ndarray,
    *,
    target_hour: int,
    method: str,
    source_hour: int,
    moved_residual: float,
    enforce_control: bool,
    control_target_hours: tuple[int, ...],
    control_target_peak_valley_rate: float | None,
    control_valley_hours: tuple[int, ...],
) -> tuple[np.ndarray, tuple[tuple[str, bool], ...], float, float]:
    clean = candidate.copy()
    tiny_negative = (clean < 0.0) & (clean >= -NUMERIC_TOLERANCE)
    clean[tiny_negative] = 0.0
    outside_evening = np.asarray(
        [hour for hour in range(HOURS_PER_DAY) if hour not in EVENING_HOURS], dtype=int
    )
    delta = clean - original
    net_removed = float(np.maximum(-delta, 0.0).sum())
    net_added = float(np.maximum(delta, 0.0).sum())
    checks: list[tuple[str, bool]] = [
        ("finite", bool(np.all(np.isfinite(clean)))),
        ("nonnegative", bool(np.all(clean >= 0.0))),
        (
            "daily_energy_preserved",
            bool(np.isclose(clean.sum(), original.sum(), atol=SUM_TOLERANCE, rtol=0.0)),
        ),
        (
            "evening_energy_preserved",
            bool(
                np.isclose(
                    clean[np.asarray(EVENING_HOURS)].sum(),
                    original[np.asarray(EVENING_HOURS)].sum(),
                    atol=SUM_TOLERANCE,
                    rtol=0.0,
                )
            ),
        ),
        (
            "outside_evening_unchanged",
            bool(np.allclose(clean[outside_evening], original[outside_evening], atol=NUMERIC_TOLERANCE, rtol=0.0)),
        ),
        (
            "net_area_balanced",
            bool(np.isclose(net_removed, net_added, atol=SUM_TOLERANCE, rtol=0.0)),
        ),
        ("positive_transfer", bool(moved_residual > NUMERIC_TOLERANCE)),
        ("target_peak_reached", evening_peak_hour(clean) == target_hour),
    ]
    if method == "adjacent_swap":
        checks.append(("source_and_target_swapped", bool(
            np.isclose(clean[source_hour], original[target_hour], atol=NUMERIC_TOLERANCE, rtol=0.0)
            and np.isclose(clean[target_hour], original[source_hour], atol=NUMERIC_TOLERANCE, rtol=0.0)
        )))
    if enforce_control:
        if control_target_peak_valley_rate is None:
            raise ValueError(
                "control_target_peak_valley_rate is required when enforce_control=True"
            )
        control_hours = identify_control_periods(clean).hours
        rounded_control_hours = identify_control_periods(np.round(clean, 5)).hours
        checks.extend(
            (
                (
                    "joint_peak_valley_rate",
                    bool(
                        np.isclose(
                            peak_valley_rate(clean),
                            control_target_peak_valley_rate,
                            atol=RATIO_TOLERANCE,
                            rtol=0.0,
                        )
                    ),
                ),
                ("joint_control_set", control_hours == control_target_hours),
                (
                    "joint_rounded_control_set",
                    rounded_control_hours == control_target_hours,
                ),
                (
                    "joint_valley_hours",
                    _extreme_hours(clean, maximum=False) == control_valley_hours,
                ),
            )
        )
    return clean, _check_map(checks), net_removed, net_added


def relocate_evening_peak(
    curve: Sequence[float] | np.ndarray,
    target_hour: int | None,
    *,
    requested: bool = True,
    peak_tolerance: float | None = None,
    enforce_control: bool = False,
    control_target_hours: Sequence[int] = (),
    control_target_peak_valley_rate: float | None = None,
    control_valley_hours: Sequence[int] = (),
) -> EveningAdjustmentResult:
    """Transactionally move an evening peak to an absolute target hour.

    A one-hour move of a unique peak uses an exact swap.  Every other forward
    move uses the local peak-packet construction.  No post-hoc whole-curve
    normalization is performed.
    """

    original = _as_curve(curve, name="curve", require_unit_sum=True)
    source_hour = evening_peak_hour(original)
    target_hours_tuple = tuple(sorted(int(hour) for hour in control_target_hours))
    valley_hours_tuple = tuple(sorted(int(hour) for hour in control_valley_hours))

    if any(hour < 0 or hour >= HOURS_PER_DAY for hour in target_hours_tuple):
        raise ValueError("control_target_hours must contain valid absolute hours")
    if any(hour < 0 or hour >= HOURS_PER_DAY for hour in valley_hours_tuple):
        raise ValueError("control_valley_hours must contain valid absolute hours")
    if enforce_control and not target_hours_tuple:
        raise ValueError("control_target_hours is required when enforce_control=True")

    if not requested:
        diagnostics = _noop_evening_diagnostics(
            status="not_requested",
            reason="the evening-peak judgement did not request a move",
            requested=False,
            source_hour=source_hour,
            target_hour=target_hour if isinstance(target_hour, int) else None,
            joint_control_constraint=enforce_control,
        )
        return EveningAdjustmentResult(original.copy(), diagnostics)
    if target_hour is None:
        diagnostics = _noop_evening_diagnostics(
            status="target_detail_missing",
            reason="an absolute target peak hour was not supplied",
            requested=True,
            source_hour=source_hour,
            target_hour=None,
            joint_control_constraint=enforce_control,
        )
        return EveningAdjustmentResult(original.copy(), diagnostics)
    if isinstance(target_hour, bool) or not isinstance(target_hour, (int, np.integer)):
        diagnostics = _noop_evening_diagnostics(
            status="target_detail_invalid",
            reason="target hour must be an integer absolute hour",
            requested=True,
            source_hour=source_hour,
            target_hour=None,
            joint_control_constraint=enforce_control,
        )
        return EveningAdjustmentResult(original.copy(), diagnostics)
    target_hour = int(target_hour)
    if target_hour not in EVENING_HOURS:
        diagnostics = _noop_evening_diagnostics(
            status="target_outside_evening_window",
            reason="target hour must be in the absolute 17:00--23:00 window",
            requested=True,
            source_hour=source_hour,
            target_hour=target_hour,
            joint_control_constraint=enforce_control,
        )
        return EveningAdjustmentResult(original.copy(), diagnostics)
    if target_hour == source_hour:
        diagnostics = _noop_evening_diagnostics(
            status="already_at_target",
            reason="the current evening peak already occurs at the target hour",
            requested=True,
            source_hour=source_hour,
            target_hour=target_hour,
            joint_control_constraint=enforce_control,
        )
        return EveningAdjustmentResult(original.copy(), diagnostics)
    if source_hour > target_hour:
        diagnostics = _noop_evening_diagnostics(
            status="source_later_than_target_conflict",
            reason="the requested method only moves an evening peak later",
            requested=True,
            source_hour=source_hour,
            target_hour=target_hour,
            joint_control_constraint=enforce_control,
        )
        return EveningAdjustmentResult(original.copy(), diagnostics)

    evening_maximum = float(original[np.asarray(EVENING_HOURS)].max())
    tolerance = (
        max(NUMERIC_TOLERANCE, 1e-8 * evening_maximum)
        if peak_tolerance is None
        else float(peak_tolerance)
    )
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("peak_tolerance must be finite and positive")
    near_peak = tuple(
        hour
        for hour in EVENING_HOURS
        if abs(float(original[hour]) - evening_maximum) <= tolerance
    )
    shift = target_hour - source_hour

    if shift == 1 and len(near_peak) == 1:
        diagnostics = EveningAdjustmentDiagnostics(
            status="eligible",
            reason="a unique adjacent evening peak can be swapped exactly",
            requested=True,
            applied=False,
            method="adjacent_swap",
            source_hour=source_hour,
            target_hour=target_hour,
            shift_hours=shift,
            peak_tolerance=tolerance,
            near_peak_hours=near_peak,
            movable_hours=(source_hour,),
            moved_residual=abs(float(original[source_hour] - original[target_hour])),
            joint_control_constraint=enforce_control,
        )
        if enforce_control and not {
            source_hour,
            target_hour,
        }.issubset(set(target_hours_tuple)):
            return _evening_failure(
                original,
                diagnostics,
                status="joint_control_constraint_failed",
                reason="adjacent swap would leave the committed target control set",
            )
        candidate = original.copy()
        candidate[source_hour], candidate[target_hour] = (
            original[target_hour],
            original[source_hour],
        )
        clean, checks, net_removed, net_added = _validate_evening_candidate(
            original,
            candidate,
            target_hour=target_hour,
            method="adjacent_swap",
            source_hour=source_hour,
            moved_residual=diagnostics.moved_residual,
            enforce_control=enforce_control,
            control_target_hours=target_hours_tuple,
            control_target_peak_valley_rate=control_target_peak_valley_rate,
            control_valley_hours=valley_hours_tuple,
        )
        diagnostics = EveningAdjustmentDiagnostics(
            **{
                **diagnostics.__dict__,
                "net_removed_area": net_removed,
                "net_added_area": net_added,
                "checks": checks,
            }
        )
        if not all(passed for _, passed in checks):
            failed = ", ".join(name for name, passed in checks if not passed)
            return _evening_failure(
                original,
                diagnostics,
                status="transaction_validation_failed",
                reason=f"adjacent-swap candidate failed: {failed}",
                checks=checks,
            )
        return EveningAdjustmentResult(
            curve=clean.copy(),
            diagnostics=EveningAdjustmentDiagnostics(
                **{
                    **diagnostics.__dict__,
                    "status": "applied_adjacent_swap",
                    "reason": "unique adjacent evening peak swapped and committed",
                    "applied": True,
                }
            ),
        )

    near_peak_set = set(near_peak)
    plateau_left = source_hour
    plateau_right = source_hour
    while plateau_left - 1 in near_peak_set:
        plateau_left -= 1
    while plateau_right + 1 in near_peak_set:
        plateau_right += 1
    plateau = tuple(range(plateau_left, plateau_right + 1))

    local_valleys = tuple(
        hour
        for hour in range(1, HOURS_PER_DAY - 1)
        if original[hour] <= original[hour - 1] + tolerance
        and original[hour] <= original[hour + 1] + tolerance
    )
    left_candidates = [hour for hour in local_valleys if hour < plateau_left]
    right_candidates = [hour for hour in local_valleys if hour > plateau_right]
    left_anchor = max(left_candidates) if left_candidates else 0
    right_anchor = min(right_candidates) if right_candidates else HOURS_PER_DAY - 1
    left_kind = "local_valley" if left_candidates else "boundary_fallback"
    right_kind = "local_valley" if right_candidates else "boundary_fallback"

    diagnostics = EveningAdjustmentDiagnostics(
        status="eligible",
        reason="local peak packet is available for construction",
        requested=True,
        applied=False,
        method="peak_packet",
        source_hour=source_hour,
        target_hour=target_hour,
        shift_hours=shift,
        peak_tolerance=tolerance,
        near_peak_hours=near_peak,
        plateau_hours=plateau,
        left_anchor=left_anchor,
        right_anchor=right_anchor,
        left_anchor_kind=left_kind,
        right_anchor_kind=right_kind,
        joint_control_constraint=enforce_control,
    )
    if not (
        left_anchor < plateau_left <= source_hour <= plateau_right < right_anchor
    ):
        return _evening_failure(
            original,
            diagnostics,
            status="no_two_sided_anchors",
            reason="a strict left/right anchor pair could not be formed",
        )

    interval = np.arange(left_anchor, right_anchor + 1, dtype=int)
    span = right_anchor - left_anchor
    background = original[left_anchor] + (
        original[right_anchor] - original[left_anchor]
    ) * (interval - left_anchor) / span
    residual_values = np.maximum(original[interval] - background, 0.0)
    residual = np.zeros(HOURS_PER_DAY, dtype=float)
    residual[interval] = residual_values
    positive = residual > tolerance
    if not positive[source_hour]:
        return _evening_failure(
            original,
            diagnostics,
            status="empty_or_zero_peak_residual",
            reason="the source peak has no positive residual above its local background",
        )

    support_left = source_hour
    support_right = source_hour
    while support_left - 1 >= left_anchor and positive[support_left - 1]:
        support_left -= 1
    while support_right + 1 <= right_anchor and positive[support_right + 1]:
        support_right += 1
    support = tuple(range(support_left, support_right + 1))
    movable = tuple(
        hour
        for hour in support
        if hour in EVENING_HOURS and hour + shift in EVENING_HOURS
    )
    if enforce_control:
        allowed = set(target_hours_tuple)
        movable = tuple(
            hour
            for hour in movable
            if hour in allowed and hour + shift in allowed
        )
    moved_residual = float(residual[np.asarray(movable, dtype=int)].sum()) if movable else 0.0
    diagnostics = EveningAdjustmentDiagnostics(
        **{
            **diagnostics.__dict__,
            "residual_support": support,
            "movable_hours": movable,
            "moved_residual": moved_residual,
        }
    )
    if not movable or moved_residual <= NUMERIC_TOLERANCE:
        return _evening_failure(
            original,
            diagnostics,
            status="empty_movable_packet",
            reason=(
                "the peak packet has no transferable residual within the evening window"
                if not enforce_control
                else "the peak packet has no transfer allowed by the committed control set"
            ),
        )

    candidate = original.copy()
    for hour in movable:
        candidate[hour] -= residual[hour]
        candidate[hour + shift] += residual[hour]

    clean, checks, net_removed, net_added = _validate_evening_candidate(
        original,
        candidate,
        target_hour=target_hour,
        method="peak_packet",
        source_hour=source_hour,
        moved_residual=moved_residual,
        enforce_control=enforce_control,
        control_target_hours=target_hours_tuple,
        control_target_peak_valley_rate=control_target_peak_valley_rate,
        control_valley_hours=valley_hours_tuple,
    )
    diagnostics = EveningAdjustmentDiagnostics(
        **{
            **diagnostics.__dict__,
            "net_removed_area": net_removed,
            "net_added_area": net_added,
            "checks": checks,
        }
    )
    if not all(passed for _, passed in checks):
        failed_names = [name for name, passed in checks if not passed]
        status = (
            "target_not_achieved"
            if failed_names == ["target_peak_reached"]
            else "joint_control_validation_failed"
            if enforce_control
            and any(name.startswith("joint_") for name in failed_names)
            else "nonfinite_or_energy_check_failed"
        )
        return _evening_failure(
            original,
            diagnostics,
            status=status,
            reason=f"peak-packet candidate failed: {', '.join(failed_names)}",
            checks=checks,
        )

    full_packet = set(movable) == set(support)
    return EveningAdjustmentResult(
        curve=clean.copy(),
        diagnostics=EveningAdjustmentDiagnostics(
            **{
                **diagnostics.__dict__,
                "status": (
                    "applied_peak_packet_full"
                    if full_packet
                    else "applied_peak_packet_partial_boundary"
                ),
                "reason": "local peak packet moved and committed",
                "applied": True,
            }
        ),
    )


def adjust_local_shape(
    baseline_2022: Sequence[float] | np.ndarray,
    comparison_2023: Sequence[float] | np.ndarray | None = None,
    *,
    evening_triggered: bool = False,
    evening_target_hour: int | None = None,
    sample_evaluable: bool = True,
) -> LocalShapeAdjustmentResult:
    """Run the ordered, non-extrapolating stage-5 local adjustments.

    The order is control-period adjustment followed by evening-peak movement.
    A flat or unevaluable baseline receives neither local adjustment.  The
    caller remains responsible for the separate four-period transformation.
    """

    baseline = _as_curve(
        baseline_2022, name="baseline_2022", require_unit_sum=True
    )
    comparison = (
        _as_curve(comparison_2023, name="comparison_2023", require_unit_sum=True)
        if comparison_2023 is not None
        else None
    )
    flatness = assess_flatness(baseline)

    if not sample_evaluable:
        control_diag = _noop_control_diagnostics(
            baseline,
            status="sample_not_evaluable",
            reason="the sample did not pass the upstream evaluability gate",
            target=comparison,
        )
        evening_diag = _noop_evening_diagnostics(
            status="sample_not_evaluable",
            reason="the sample did not pass the upstream evaluability gate",
            requested=evening_triggered,
            source_hour=evening_peak_hour(baseline),
            target_hour=evening_target_hour,
        )
        diagnostics = LocalShapeAdjustmentDiagnostics(
            status="sample_not_evaluable",
            sample_evaluable=False,
            flatness=flatness,
            control=control_diag,
            evening=evening_diag,
            curve_changed=False,
        )
        return LocalShapeAdjustmentResult(baseline.copy(), diagnostics)

    if flatness.is_flat:
        control_diag = _noop_control_diagnostics(
            baseline,
            status="flat_gate_skipped",
            reason="LF > 0.8 and CV < 0.15; only the four-period change may run",
            target=comparison,
        )
        evening_diag = _noop_evening_diagnostics(
            status="flat_gate_skipped",
            reason="LF > 0.8 and CV < 0.15; only the four-period change may run",
            requested=evening_triggered,
            source_hour=evening_peak_hour(baseline),
            target_hour=evening_target_hour,
        )
        diagnostics = LocalShapeAdjustmentDiagnostics(
            status="flat_gate_skipped",
            sample_evaluable=True,
            flatness=flatness,
            control=control_diag,
            evening=evening_diag,
            curve_changed=False,
        )
        return LocalShapeAdjustmentResult(baseline.copy(), diagnostics)

    if comparison is None:
        control_result = ControlAdjustmentResult(
            baseline.copy(),
            _noop_control_diagnostics(
                baseline,
                status="target_detail_missing",
                reason="a same-source 2023 curve was not supplied for control targets",
            ),
        )
    else:
        control_result = adjust_control_period(
            baseline, comparison, apply_flat_gate=False
        )

    control_committed = control_result.diagnostics.applied
    if control_committed:
        control_rate = control_result.diagnostics.target_peak_valley_rate
        if control_rate is None:  # defensive; an applied control result always has it
            raise RuntimeError("applied control adjustment is missing its target rate")
        evening_result = relocate_evening_peak(
            control_result.curve,
            evening_target_hour,
            requested=evening_triggered,
            enforce_control=True,
            control_target_hours=control_result.diagnostics.target_hours,
            control_target_peak_valley_rate=control_rate,
            control_valley_hours=_extreme_hours(control_result.curve, maximum=False),
        )
    else:
        evening_result = relocate_evening_peak(
            control_result.curve,
            evening_target_hour,
            requested=evening_triggered,
        )

    control_applied = control_result.diagnostics.applied
    evening_applied = evening_result.diagnostics.applied
    if control_applied and evening_applied:
        status = "control_and_evening_applied"
    elif control_applied:
        status = "control_applied"
    elif evening_applied:
        status = "evening_applied"
    else:
        status = "unchanged"
    changed = not np.allclose(
        evening_result.curve, baseline, atol=NUMERIC_TOLERANCE, rtol=0.0
    )
    diagnostics = LocalShapeAdjustmentDiagnostics(
        status=status,
        sample_evaluable=True,
        flatness=flatness,
        control=control_result.diagnostics,
        evening=evening_result.diagnostics,
        curve_changed=bool(changed),
    )
    return LocalShapeAdjustmentResult(evening_result.curve.copy(), diagnostics)


__all__ = [
    "CONTROL_DURATION_CHANGE_MIN",
    "CONTROL_RATIO",
    "EVENING_HOURS",
    "FLAT_CV_THRESHOLD",
    "FLAT_LOAD_FACTOR_THRESHOLD",
    "MIN_CONTROL_PERIOD_HOURS",
    "PEAK_VALLEY_RATE_CHANGE_THRESHOLD",
    "ControlAdjustmentDiagnostics",
    "ControlAdjustmentResult",
    "ControlSetDiagnostics",
    "EveningAdjustmentDiagnostics",
    "EveningAdjustmentResult",
    "FlatnessDiagnostics",
    "LocalShapeAdjustmentDiagnostics",
    "LocalShapeAdjustmentResult",
    "adjust_control_period",
    "adjust_local_shape",
    "assess_flatness",
    "circular_runs",
    "coefficient_of_variation",
    "evening_peak_hour",
    "identify_control_periods",
    "load_factor",
    "peak_valley_rate",
    "relocate_evening_peak",
]
