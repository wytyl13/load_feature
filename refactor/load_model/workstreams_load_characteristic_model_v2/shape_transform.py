"""Energy-conserving 24-hour load-shape transformations for model 3.5.0.

The formal path applies proportional four-window shares, local boundary means
with window-energy correction, and then scales
all 24 hourly deviations around the daily mean only when the cell's
peak-valley evolution gate is enabled.  A skipped second stage preserves the
four-window result exactly; an applied stage may move those shares, which are
diagnostic after that point and are never repaired a third time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PEAK_VALLEY_RATE_TOLERANCE, SHAPE_NUMERICAL_TOLERANCE


HOURS_PER_DAY = 24
WINDOW_BOUNDS = ((0, 6), (6, 12), (12, 17), (17, 24))
WINDOW_NAMES = ("night_00_05", "morning_06_11", "midday_12_16", "evening_17_23")
BOUNDARY_SMOOTHING_HOURS = (5, 6, 11, 12, 16, 17)


@dataclass(frozen=True)
class PeakValleyAdjustmentResult:
    """Result and audit fields for the final peak-valley transformation."""

    final_shape: np.ndarray
    rate_before: float
    target_requested: float
    target_range_limited: float
    target_applied: float
    lambda_value: float
    target_clipped: bool
    adjustment_status: str


@dataclass(frozen=True)
class SequentialShapeTransformResult:
    """Intermediate and final outputs from the fixed two-stage shape contract."""

    intermediate_shape: np.ndarray
    final_shape: np.ndarray
    target_window_shares: np.ndarray
    intermediate_window_shares: np.ndarray
    final_window_shares: np.ndarray
    final_window_share_deviation: np.ndarray
    peak_valley: PeakValleyAdjustmentResult
    proportional_shape: np.ndarray


def smooth_window_boundaries(shape: np.ndarray) -> np.ndarray:
    """Simultaneous three-hour means at six boundary hours, then conserve shares.

    Only the six selected hours are averaged. A small proportional correction
    within each window restores its pre-smoothing energy before peak calibration.
    Midnight is deliberately not smoothed.
    """
    values = normalize_shape(shape)
    smoothed = values.copy()
    for hour in BOUNDARY_SMOOTHING_HOURS:
        smoothed[hour] = values[hour - 1:hour + 2].mean()
    return apply_proportional_window_shares(smoothed, window_shares(values))


def normalize_shape(shape: np.ndarray, *, atol: float = 1e-15) -> np.ndarray:
    """Return a finite, non-negative 24-point shape whose energy sums to one."""

    values = np.asarray(shape, dtype=float)
    if values.ndim != 1 or values.size != HOURS_PER_DAY:
        raise ValueError("a daily shape must be a one-dimensional 24-point array")
    if not np.isfinite(values).all():
        raise ValueError("daily shape contains non-finite values")
    if np.any(values < -atol):
        raise ValueError("daily shape contains negative values")
    values = np.maximum(values, 0.0).copy()
    total = float(values.sum())
    if total <= 0:
        raise ValueError("daily shape must have positive energy")
    return values / total


def load_factor(shape: np.ndarray) -> float:
    """Daily average load divided by daily maximum load."""

    values = normalize_shape(shape)
    return float(values.mean() / values.max())


def window_shares(shape: np.ndarray) -> np.ndarray:
    values = normalize_shape(shape)
    return np.asarray([values[start:stop].sum() for start, stop in WINDOW_BOUNDS])


def peak_valley_rate(shape: np.ndarray) -> float:
    """Return ``(maximum - minimum) / maximum`` for a valid daily shape."""

    values = normalize_shape(np.asarray(shape, dtype=float))
    maximum = float(values.max())
    minimum = float(values.min())
    return float((maximum - minimum) / maximum)


def apply_proportional_window_shares(
    base_shape: np.ndarray,
    target_shares: np.ndarray,
    *,
    atol: float = SHAPE_NUMERICAL_TOLERANCE,
) -> np.ndarray:
    """Hit four window shares exactly while preserving within-window ratios.

    This is the first formal 3.5.0 transform.  It deliberately has no
    smoothing, peak damping, control-period adjustment or repair stage.
    """

    base = normalize_shape(np.asarray(base_shape, dtype=float))
    target = np.asarray(target_shares, dtype=float)
    if target.shape != (4,) or not np.isfinite(target).all():
        raise ValueError("target_shares must contain four finite values")
    if np.any(target < -atol):
        raise ValueError("target window shares must be non-negative")
    target = np.maximum(target, 0.0)
    target_sum = float(target.sum())
    if abs(target_sum - 1.0) > atol:
        raise ValueError("target window shares must sum to one")
    target = target / target_sum

    current = window_shares(base)
    impossible = (current <= atol) & (target > atol)
    if impossible.any():
        names = [WINDOW_NAMES[index] for index in np.flatnonzero(impossible)]
        raise ValueError(f"cannot assign positive energy to empty source windows: {names}")

    result = np.zeros(HOURS_PER_DAY, dtype=float)
    for current_share, target_share, (start, stop) in zip(
        current, target, WINDOW_BOUNDS, strict=True
    ):
        if target_share <= atol:
            continue
        result[start:stop] = base[start:stop] * (target_share / current_share)

    total_error = abs(float(result.sum()) - 1.0)
    if total_error > atol:
        raise RuntimeError(
            f"proportional window transform failed energy conservation: {total_error}"
        )
    if total_error:
        result /= result.sum()
    if not np.isfinite(result).all() or np.any(result < -atol):
        raise RuntimeError("proportional window transform produced invalid values")
    if not np.allclose(window_shares(result), target, rtol=0.0, atol=atol):
        raise RuntimeError("proportional window transform missed target shares")

    # A ratio comparison by cross multiplication also handles zero-valued
    # hours without introducing divide-by-zero artifacts.
    for (start, stop), current_share, target_share in zip(
        WINDOW_BOUNDS, current, target, strict=True
    ):
        if current_share <= atol or target_share <= atol:
            continue
        expected = base[start:stop] * (target_share / current_share)
        if not np.allclose(result[start:stop], expected, rtol=0.0, atol=atol):
            raise RuntimeError("within-window hourly proportions were not preserved")
    return np.maximum(result, 0.0)


def apply_peak_valley_target(
    intermediate_shape: np.ndarray,
    target_rate: float,
    *,
    atol: float = SHAPE_NUMERICAL_TOLERANCE,
    rate_atol: float = PEAK_VALLEY_RATE_TOLERANCE,
) -> PeakValleyAdjustmentResult:
    """Scale deviations around the daily mean to hit a peak-valley target.

    The requested target is retained for audit and clipped to ``[0, 1]`` for
    application.  A strictly flat intermediate curve is never given an
    artificial peak: a positive target is recorded as unreachable and the
    applied target remains zero.
    """

    if not np.isfinite(atol) or atol <= 0:
        raise ValueError("atol must be a finite positive value")
    if not np.isfinite(rate_atol) or rate_atol <= 0:
        raise ValueError("rate_atol must be a finite positive value")
    values = normalize_shape(np.asarray(intermediate_shape, dtype=float), atol=atol)
    try:
        requested = float(target_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("target_rate must be a finite scalar") from exc
    if not np.isfinite(requested):
        raise ValueError("target_rate must be a finite scalar")

    limited = float(np.clip(requested, 0.0, 1.0))
    clipped = requested < 0.0 or requested > 1.0
    maximum = float(values.max())
    minimum = float(values.min())
    mean = float(values.mean())
    span = maximum - minimum
    rate_before = float(span / maximum)

    if span == 0.0:
        status = (
            "flat_intermediate_zero_target"
            if limited == 0.0
            else "flat_intermediate_target_unreachable"
        )
        return PeakValleyAdjustmentResult(
            final_shape=values.copy(),
            rate_before=rate_before,
            target_requested=requested,
            target_range_limited=limited,
            target_applied=0.0,
            lambda_value=1.0,
            target_clipped=clipped,
            adjustment_status=status,
        )

    if abs(limited - rate_before) <= rate_atol:
        return PeakValleyAdjustmentResult(
            final_shape=values.copy(),
            rate_before=rate_before,
            target_requested=requested,
            target_range_limited=limited,
            target_applied=limited,
            lambda_value=1.0,
            target_clipped=clipped,
            adjustment_status="target_already_met",
        )

    denominator = span - limited * (maximum - mean)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError("peak-valley closed-form denominator is not positive")
    lambda_value = float(limited * mean / denominator)
    if not np.isfinite(lambda_value) or lambda_value < 0.0:
        raise RuntimeError("peak-valley closed-form lambda is invalid")

    final = mean + lambda_value * (values - mean)
    if not np.isfinite(final).all():
        raise RuntimeError("peak-valley transform produced non-finite values")
    if np.any(final < -atol):
        raise RuntimeError("peak-valley transform produced negative values")
    final = final.copy()
    final[final < 0.0] = 0.0
    total = float(final.sum())
    if total <= 0.0 or abs(total - 1.0) > atol:
        raise RuntimeError("peak-valley transform failed energy conservation")
    if total != 1.0:
        final /= total

    rate_final = peak_valley_rate(final)
    if abs(rate_final - limited) > rate_atol:
        raise RuntimeError(
            "peak-valley transform missed target rate: "
            f"target={limited}, actual={rate_final}"
        )
    return PeakValleyAdjustmentResult(
        final_shape=final,
        rate_before=rate_before,
        target_requested=requested,
        target_range_limited=limited,
        target_applied=limited,
        lambda_value=lambda_value,
        target_clipped=clipped,
        adjustment_status="mean_centered_scaling_applied",
    )


def _skip_peak_valley_target(
    intermediate_shape: np.ndarray,
    target_rate: float,
    *,
    gate_reason: str,
    atol: float,
) -> PeakValleyAdjustmentResult:
    """Retain the theoretical target while preserving the first-stage shape."""

    values = np.asarray(intermediate_shape, dtype=float)
    # Validate through the shared normalization contract, but retain the exact
    # first-stage values: a closed gate must be an identity second stage, not
    # an additional floating-point renormalization.
    normalize_shape(values, atol=atol)
    try:
        requested = float(target_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("target_rate must be a finite scalar") from exc
    if not np.isfinite(requested):
        raise ValueError("target_rate must be a finite scalar")
    reason = str(gate_reason).strip()
    if not reason.startswith("disabled_"):
        raise ValueError("disabled peak-valley stage requires a disabled_ reason")
    limited = float(np.clip(requested, 0.0, 1.0))
    rate_before = peak_valley_rate(values)
    return PeakValleyAdjustmentResult(
        final_shape=values.copy(),
        rate_before=rate_before,
        target_requested=requested,
        target_range_limited=limited,
        target_applied=rate_before,
        lambda_value=1.0,
        target_clipped=bool(requested < 0.0 or requested > 1.0),
        adjustment_status=reason.replace("disabled_", "skipped_", 1),
    )


def apply_sequential_shape_transform(
    base_shape: np.ndarray,
    target_window_shares: np.ndarray,
    target_peak_valley_rate: float,
    *,
    peak_valley_evolution_enabled: bool = True,
    peak_valley_gate_reason: str = "enabled",
    shape_atol: float = SHAPE_NUMERICAL_TOLERANCE,
    rate_atol: float = PEAK_VALLEY_RATE_TOLERANCE,
) -> SequentialShapeTransformResult:
    """Apply four-window migration and the conditionally gated second stage."""

    intermediate = apply_proportional_window_shares(
        base_shape,
        target_window_shares,
        atol=shape_atol,
    )
    proportional = intermediate.copy()
    # No migration means no artificial step: preserve the 2023 anchor and
    # cells frozen for insufficient history exactly.
    if not np.allclose(window_shares(base_shape), target_window_shares,
                       rtol=0.0, atol=shape_atol):
        intermediate = smooth_window_boundaries(proportional)
    intermediate_shares = window_shares(intermediate)
    if peak_valley_evolution_enabled:
        if str(peak_valley_gate_reason) != "enabled":
            raise ValueError(
                "enabled peak-valley stage requires gate reason 'enabled'"
            )
        peak_valley = apply_peak_valley_target(
            intermediate,
            target_peak_valley_rate,
            atol=shape_atol,
            rate_atol=rate_atol,
        )
    else:
        peak_valley = _skip_peak_valley_target(
            intermediate,
            target_peak_valley_rate,
            gate_reason=peak_valley_gate_reason,
            atol=shape_atol,
        )
    final_shares = window_shares(peak_valley.final_shape)
    return SequentialShapeTransformResult(
        intermediate_shape=intermediate,
        final_shape=peak_valley.final_shape,
        target_window_shares=intermediate_shares.copy(),
        intermediate_window_shares=intermediate_shares,
        final_window_shares=final_shares,
        final_window_share_deviation=final_shares - intermediate_shares,
        peak_valley=peak_valley,
        proportional_shape=proportional,
    )


def _circular_true_runs(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != (HOURS_PER_DAY,):
        raise ValueError("control mask must contain 24 values")
    if not selected.any():
        return ()
    if selected.all():
        return (tuple(range(HOURS_PER_DAY)),)
    false_hour = int(np.flatnonzero(~selected)[0])
    start = (false_hour + 1) % HOURS_PER_DAY
    runs: list[tuple[int, ...]] = []
    active: list[int] = []
    for offset in range(HOURS_PER_DAY):
        hour = (start + offset) % HOURS_PER_DAY
        if selected[hour]:
            active.append(hour)
        elif active:
            runs.append(tuple(active))
            active = []
    if active:
        runs.append(tuple(active))
    return tuple(runs)


def diagnose_shape(
    shape: np.ndarray,
    *,
    control_ratio: float = 0.85,
    min_control_hours: int = 3,
) -> dict[str, object]:
    """Return post-transform diagnostics that never alter the input curve."""

    values = normalize_shape(np.asarray(shape, dtype=float))
    if not 0 < control_ratio <= 1:
        raise ValueError("control_ratio must lie in (0, 1]")
    if int(min_control_hours) <= 0:
        raise ValueError("min_control_hours must be positive")
    maximum = float(values.max())
    mean = float(values.mean())
    raw_mask = values >= control_ratio * maximum
    raw_hours = tuple(int(hour) for hour in np.flatnonzero(raw_mask))
    runs = _circular_true_runs(raw_mask)
    effective_runs = tuple(
        run for run in runs if len(run) >= int(min_control_hours)
    )
    effective_hours = tuple(sorted(hour for run in effective_runs for hour in run))
    evening = values[17:24]
    evening_peak_hour = 17 + int(np.argmax(evening))
    return {
        "peak_valley_rate": float((maximum - float(values.min())) / maximum),
        "load_factor": float(mean / maximum),
        "cv": float(np.std(values, ddof=0) / mean),
        "evening_peak_hour": evening_peak_hour,
        "raw_85pct_hours": raw_hours,
        "effective_control_hours": effective_hours,
        "effective_control_runs": effective_runs,
    }


def apply_window_shares(
    shape: np.ndarray,
    target_shares: np.ndarray,
    *,
    smooth_boundaries: bool = False,
    atol: float = SHAPE_NUMERICAL_TOLERANCE,
) -> np.ndarray:
    """Proportional window transform with optional local conservative smoothing."""

    result = apply_proportional_window_shares(shape, target_shares, atol=atol)
    if smooth_boundaries and not np.allclose(window_shares(shape), target_shares,
                                            rtol=0.0, atol=atol):
        result = smooth_window_boundaries(result)
    return result


__all__ = [
    "WINDOW_BOUNDS",
    "WINDOW_NAMES",
    "BOUNDARY_SMOOTHING_HOURS",
    "smooth_window_boundaries",
    "PeakValleyAdjustmentResult",
    "SequentialShapeTransformResult",
    "apply_peak_valley_target",
    "apply_sequential_shape_transform",
    "apply_window_shares",
    "apply_proportional_window_shares",
    "diagnose_shape",
    "load_factor",
    "normalize_shape",
    "peak_valley_rate",
    "window_shares",
]
