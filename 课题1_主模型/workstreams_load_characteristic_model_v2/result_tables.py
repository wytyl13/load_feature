"""Build and publish result tables under the current 3.5.0 contract.

The checks in this module sit at the last safe boundary before any result table
is written.  A stale shape chain must therefore fail here instead of producing
a plausible-looking stale manifest that a report could accidentally consume.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .config import (
    FLAT_CV_THRESHOLD,
    FLAT_LOAD_FACTOR_THRESHOLD,
    FOUR_WINDOW_FINAL_SHARE_POLICY,
    METHOD_REVISION,
    MODEL_VERSION,
    PEAK_VALLEY_EVOLUTION_DELTA_LOWER_BOUND,
    PEAK_VALLEY_EVOLUTION_DELTA_UPPER_BOUND,
    PEAK_VALLEY_EXTRAPOLATION_ENABLED,
    PEAK_VALLEY_FORMAL_PHI,
    PEAK_VALLEY_GATE_POLICY,
    PEAK_VALLEY_RATE_TOLERANCE,
    PEAK_VALLEY_SUPPORTED_PHIS,
    PEAK_VALLEY_TARGET_PRIORITY,
    SHAPE_EXTRAPOLATION_METHOD,
    SHAPE_NUMERICAL_TOLERANCE,
)
from .monthly_envelope import target_month_shares
from .quality import SHAPE_COLS
from .shape_parameters import (
    classify_peak_valley_delta,
    evaluate_peak_valley_evolution_gate,
)
from .shape_transform import peak_valley_rate


REPRESENTATIVE_TRACKS = ("P10", "P50", "P90")

RESULT_SCHEMA_VERSION = MODEL_VERSION
PROVINCE_SHARED_FACTOR_SCOPE = "province_shared_all_25_components"

# These tables are the minimum evidence needed to call a run a 3.5.0 result.
# The pipeline may publish additional current-method audit tables; the total
# number of result tables is intentionally not fixed.
REQUIRED_FORMAL_TABLES = frozenset(
    {
        "industry_daily_aggregate",
        "industry_aggregate_diagnostics",
        "cell_quality",
        "baseline_shapes",
        "baseline_diagnostics",
        "shape_diagnostics",
        "shape_history",
        "shape_future",
        "day_type_factors",
        "correction_component_shape",
        "correction_component_diagnostics",
        "component_daily_energy",
        "daily_energy_checks",
        "energy_checks",
    }
)

# Retired outputs must not be reachable from tables_manifest.json.  Diagnostic
# evidence about the future evening peak/control set remains allowed in the new
# shape tables; only the old adjustment-result products are retired.
FORBIDDEN_FORMAL_TABLES = frozenset(
    {
        "representative_selection",
        "parameter_source_trace",
        "group_annual_energy",
        "load_duration",
        "season_daytype_correction_component",
        "control_period_stretch_results",
        "evening_peak_move_results",
        "anchor_repair_results",
        "qp_results",
        "rho_fallback_results",
        "gamma_delta_results",
        "控制时段拉伸结果",
        "晚峰搬移结果",
    }
)

FORBIDDEN_SHAPE_FUTURE_COLUMNS = frozenset(
    {
        "gamma",
        "log_gamma_increment",
        "beta_delta_hours",
        "delta_continuous_hours",
        "delta_applied_hours",
        "centroid",
        "source_industry",
        "parameter_group",
        "application_run_group",
        "anchor_repair_applied",
        "repair_l2_distance",
        "rho",
    }
)
REQUIRED_SHAPE_FUTURE_COLUMNS = frozenset(
    {
        "industry",
        "season",
        "final_day_type",
        "year",
        "H",
        "phi",
        "peak_valley_phi",
        "base_source_year",
        "trend_eligible",
        "trend_status",
        "flat_curve",
        "flat_borderline",
        "peak_valley_delta_abs",
        "peak_valley_delta_band",
        "peak_valley_evolution_enabled",
        "peak_valley_gate_reason",
        "peak_valley_gate_policy",
        "a_requested",
        "a_max",
        "a_applied",
        "requested_share_night",
        "requested_share_morning",
        "requested_share_midday",
        "requested_share_evening",
        "applied_share_night",
        "applied_share_morning",
        "applied_share_midday",
        "applied_share_evening",
        "window_boundary_hit",
        "peak_valley_cumulative_damping",
        "peak_valley_target_requested",
        "peak_valley_target_range_limited",
        "peak_valley_target_applied",
        "peak_valley_target_clipped",
        "peak_valley_adjustment_status",
        "peak_valley_rate_after_four_window",
        "peak_valley_lambda",
        "peak_valley_rate_final",
        "four_window_share_intermediate_night",
        "four_window_share_intermediate_morning",
        "four_window_share_intermediate_midday",
        "four_window_share_intermediate_evening",
        "four_window_share_final_night",
        "four_window_share_final_morning",
        "four_window_share_final_midday",
        "four_window_share_final_evening",
        "four_window_share_deviation_night",
        "four_window_share_deviation_morning",
        "four_window_share_deviation_midday",
        "four_window_share_deviation_evening",
        "shape_extrapolation_method",
        "four_window_final_share_policy",
        "peak_valley_extrapolation_enabled",
        "future_peak_valley_rate",
        "future_evening_peak_hour",
        "future_raw_85pct_hours",
        "future_effective_control_hours",
    }
)
REQUIRED_DAY_TYPE_FACTOR_COLUMNS = frozenset(
    {
        "season",
        "final_day_type",
        "factor",
        "sample_days",
        "holiday_merged_with_weekend",
        "source_year",
        "factor_scope",
        "fallback_status",
    }
)

RUN_MANIFEST_CONTRACT: dict[str, object] = {
    "model_version": RESULT_SCHEMA_VERSION,
    "method_revision": METHOD_REVISION,
    "annual_energy_method": "M1",
    "shape_extrapolation_method": SHAPE_EXTRAPOLATION_METHOD,
    "four_window_final_share_policy": FOUR_WINDOW_FINAL_SHARE_POLICY,
    "peak_valley_target_priority": PEAK_VALLEY_TARGET_PRIORITY,
    "late_peak_control_policy": "diagnostic_only",
    "day_type_factor_scope": PROVINCE_SHARED_FACTOR_SCOPE,
    "peak_valley_extrapolation_enabled": PEAK_VALLEY_EXTRAPOLATION_ENABLED,
    "peak_valley_gate_policy": PEAK_VALLEY_GATE_POLICY,
    "peak_valley_delta_lower_bound": PEAK_VALLEY_EVOLUTION_DELTA_LOWER_BOUND,
    "peak_valley_delta_upper_bound": PEAK_VALLEY_EVOLUTION_DELTA_UPPER_BOUND,
    "flat_load_factor_threshold": FLAT_LOAD_FACTOR_THRESHOLD,
    "flat_cv_threshold": FLAT_CV_THRESHOLD,
    "formal_horizon": 8,
    "formal_peak_valley_phi": PEAK_VALLEY_FORMAL_PHI,
    "peak_damping_enabled": False,
    "anchor_repair_enabled": False,
}


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label}缺少字段：{sorted(missing)}")


def _boolean_column(
    frame: pd.DataFrame,
    column: str,
    *,
    label: str,
) -> np.ndarray:
    """Parse booleans from native values or an explicit textual vocabulary."""

    series = frame[column]
    if series.isna().any():
        raise ValueError(f"{label}.{column}必须为可解析布尔值")
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.to_numpy(dtype=bool)
    normalized = series.astype("string").str.strip().str.lower()
    allowed = {"true", "false", "1", "0", "yes", "no", "y", "n", "是", "否"}
    if not normalized.isin(allowed).all():
        raise ValueError(f"{label}.{column}必须为可解析布尔值")
    return normalized.isin({"true", "1", "yes", "y", "是"}).to_numpy(dtype=bool)


def _validated_peak_valley_gate(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Recompute every gate decision and its exact stable reason code."""

    label = "shape_future"
    trend_eligible = _boolean_column(frame, "trend_eligible", label=label)
    flat_curve = _boolean_column(frame, "flat_curve", label=label)
    flat_borderline = _boolean_column(frame, "flat_borderline", label=label)
    gate_enabled = _boolean_column(
        frame,
        "peak_valley_evolution_enabled",
        label=label,
    )
    try:
        delta_abs = pd.to_numeric(
            frame["peak_valley_delta_abs"], errors="raise"
        ).to_numpy(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "shape_future.peak_valley_delta_abs必须为非负有限数"
        ) from exc
    if not np.isfinite(delta_abs).all() or (delta_abs < 0.0).any():
        raise ValueError(
            "shape_future.peak_valley_delta_abs必须为非负有限数"
        )

    expected_enabled: list[bool] = []
    expected_reasons: list[str] = []
    expected_bands: list[str] = []
    for eligible, flat, delta in zip(
        trend_eligible,
        flat_curve,
        delta_abs,
        strict=True,
    ):
        expected_bands.append(
            classify_peak_valley_delta(float(delta) if eligible else None)
        )
        enabled, reason = evaluate_peak_valley_evolution_gate(
            trend_eligible=bool(eligible),
            flat_curve=bool(flat),
            delta_abs=float(delta) if eligible else None,
        )
        expected_enabled.append(enabled)
        expected_reasons.append(reason)

    expected_enabled_array = np.asarray(expected_enabled, dtype=bool)
    expected_reason_array = np.asarray(expected_reasons, dtype=object)
    expected_band_array = np.asarray(expected_bands, dtype=object)
    actual_reasons = frame["peak_valley_gate_reason"].astype(str).to_numpy()
    actual_bands = frame["peak_valley_delta_band"].astype(str).to_numpy()
    if not np.array_equal(gate_enabled, expected_enabled_array) or not np.array_equal(
        actual_reasons,
        expected_reason_array,
    ):
        raise ValueError(
            "shape_future峰谷差率门控结果或原因与历史充分性、平坦判定和变化区间不一致"
        )
    if not np.array_equal(actual_bands, expected_band_array):
        raise ValueError("shape_future.peak_valley_delta_band与历史充分性和变化量不一致")
    if set(frame["peak_valley_gate_policy"].astype(str)) != {
        PEAK_VALLEY_GATE_POLICY
    }:
        raise ValueError("shape_future峰谷差率门控策略标记不一致")
    evidence = pd.DataFrame(
        {
            "trend_eligible": trend_eligible,
            "flat_curve": flat_curve,
            "flat_borderline": flat_borderline,
            "peak_valley_delta_abs": delta_abs,
            "peak_valley_delta_band": actual_bands,
            "peak_valley_evolution_enabled": gate_enabled,
            "peak_valley_gate_reason": actual_reasons,
            "peak_valley_gate_policy": frame["peak_valley_gate_policy"]
            .astype(str)
            .to_numpy(),
        }
    )
    return gate_enabled, actual_reasons, evidence


def _validate_peak_valley_gate_stability(
    frame: pd.DataFrame,
    evidence: pd.DataFrame,
) -> None:
    """Require one history-derived gate record per cell across all futures."""

    cell_columns = ["industry", "season", "final_day_type"]
    audit = frame[cell_columns].reset_index(drop=True).copy()
    for column in evidence.columns:
        audit[column] = evidence[column].to_numpy()
    evidence_columns = list(evidence.columns)
    variability = audit.groupby(
        cell_columns,
        sort=False,
        dropna=False,
    )[evidence_columns].nunique(dropna=False)
    inconsistent = variability.gt(1).any(axis=1)
    if inconsistent.any():
        sample = list(variability.index[inconsistent][:10])
        raise ValueError(
            "shape_future同一行业/季节/日型的峰谷差率门控证据"
            f"跨year/H/phi必须恒定：{sample}"
        )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_run_manifest(
    manifest: Mapping[str, object],
    *,
    require_success: bool = True,
    require_parquet: bool = False,
) -> None:
    """Reject a run manifest that does not describe the final 3.5.0 method.

    ``method`` is retained as a compatibility/audit field, but
    ``annual_energy_method`` is the explicit 3.5.0 contract field.  Both must be
    M1 when ``method`` is present.  Formal report generation additionally asks
    for ``require_parquet=True``.
    """

    if not isinstance(manifest, Mapping):
        raise RuntimeError("结果口径不符：运行清单必须是JSON对象")
    if require_success and manifest.get("status") != "success":
        raise RuntimeError(
            f"结果口径不符：status={manifest.get('status')!r}，预期'success'"
        )
    for key, expected in RUN_MANIFEST_CONTRACT.items():
        actual = manifest.get(key)
        # ``False == 0`` in Python, so use identity for the three fixed
        # switches and reject numeric/string lookalikes in hand-edited JSON.
        matches = actual is expected if isinstance(expected, bool) else actual == expected
        if not matches:
            raise RuntimeError(f"结果口径不符：{key}={actual!r}，预期{expected!r}")
    if "method" in manifest and manifest.get("method") != "M1":
        raise RuntimeError(
            f"结果口径不符：method={manifest.get('method')!r}，预期'M1'"
        )
    if require_parquet:
        hourly = manifest.get("hourly_output")
        if not isinstance(hourly, Mapping):
            raise RuntimeError("正式发布阻断：hourly_output不是有效对象")
        output_format = str(hourly.get("format", "")).strip().lower()
        file_name = str(hourly.get("file", "")).strip()
        if output_format != "parquet" or Path(file_name).suffix.lower() != ".parquet":
            raise RuntimeError(
                "正式发布阻断：逐时结果必须成功写为Parquet，禁止CSV回退"
            )


def _validate_shape_future_schema(frame: pd.DataFrame) -> None:
    columns = set(frame.columns)
    retired = sorted(columns.intersection(FORBIDDEN_SHAPE_FUTURE_COLUMNS))
    if retired:
        raise ValueError(f"shape_future包含已退役字段：{retired}")
    missing = sorted(REQUIRED_SHAPE_FUTURE_COLUMNS.difference(columns))
    if missing:
        raise ValueError(f"shape_future缺少3.5.0字段：{missing}")
    if frame.empty:
        raise ValueError("shape_future不得为空")
    base_years = pd.to_numeric(frame["base_source_year"], errors="coerce")
    if base_years.isna().any() or set(base_years.astype(int)) != {2023}:
        raise ValueError("shape_future.base_source_year必须全部为2023")
    gate_enabled, gate_reasons, gate_evidence = _validated_peak_valley_gate(frame)

    key = ["industry", "season", "final_day_type", "year", "H", "phi"]
    if frame.duplicated(key).any():
        raise ValueError("shape_future单元键industry/season/day_type/year/H/phi重复")
    horizons = pd.to_numeric(frame["H"], errors="coerce").to_numpy(float)
    pv_horizons = pd.to_numeric(frame["phi"], errors="coerce").to_numpy(float)
    named_pv_horizons = pd.to_numeric(
        frame["peak_valley_phi"], errors="coerce"
    ).to_numpy(float)
    supported = {5, 8, 15}
    supported_phis = set(PEAK_VALLEY_SUPPORTED_PHIS)
    if (
        not np.isfinite(horizons).all()
        or not np.isfinite(pv_horizons).all()
        or not np.equal(horizons, np.floor(horizons)).all()
        or not set(horizons.astype(int)).issubset(supported)
        or not set(pv_horizons).issubset(supported_phis)
        or not np.array_equal(pv_horizons, named_pv_horizons)
    ):
        raise ValueError("shape_future的H/phi维度非法或含义不一致")
    sensitivity = frame[["industry", "season", "final_day_type", "year"]].copy()
    sensitivity["H"] = horizons.astype(int)
    sensitivity["phi"] = pv_horizons
    expected_pairs = {
        (horizon, peak_valley_phi)
        for horizon in supported
        for peak_valley_phi in supported_phis
    }
    for cell, group in sensitivity.groupby(
        ["industry", "season", "final_day_type", "year"],
        sort=False,
        dropna=False,
    ):
        observed_pairs = set(zip(group["H"], group["phi"], strict=True))
        if observed_pairs != expected_pairs:
            missing_pairs = sorted(expected_pairs.difference(observed_pairs))
            extra_pairs = sorted(observed_pairs.difference(expected_pairs))
            raise ValueError(
                "shape_future敏感性H×phi覆盖不完整："
                f"cell={cell!r}, missing={missing_pairs}, extra={extra_pairs}"
            )
    _validate_peak_valley_gate_stability(frame, gate_evidence)

    requested = pd.to_numeric(frame["a_requested"], errors="coerce").to_numpy(float)
    maximum = pd.to_numeric(frame["a_max"], errors="coerce").to_numpy(float)
    applied = pd.to_numeric(frame["a_applied"], errors="coerce").to_numpy(float)
    if (
        not np.isfinite(requested).all()
        or np.isnan(maximum).any()
        or not np.isfinite(applied).all()
    ):
        raise ValueError("shape_future的迁移幅度字段不得为空或无法解析")
    if np.isneginf(maximum).any() or (maximum < 0).any():
        raise ValueError("shape_future的a_max/a_applied取值非法")
    if (
        (requested < -1e-12).any()
        or (applied < -1e-12).any()
        or (applied > requested + 1e-12).any()
        or (applied > maximum + 1e-12).any()
    ):
        raise ValueError(
            "shape_future必须满足0<=a_applied<=min(a_requested,a_max)"
        )
    applied_share_columns = [
        "applied_share_night",
        "applied_share_morning",
        "applied_share_midday",
        "applied_share_evening",
    ]
    shares = frame[applied_share_columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    if (
        not np.isfinite(shares).all()
        or (shares < -SHAPE_NUMERICAL_TOLERANCE).any()
        or not np.isclose(
            shares.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=SHAPE_NUMERICAL_TOLERANCE,
        ).all()
    ):
        raise ValueError("shape_future四时段实际份额必须有限、非负且合计为1")

    names = ("night", "morning", "midday", "evening")
    intermediate = frame[
        [f"four_window_share_intermediate_{name}" for name in names]
    ].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    final = frame[
        [f"four_window_share_final_{name}" for name in names]
    ].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    deviation = frame[
        [f"four_window_share_deviation_{name}" for name in names]
    ].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if (
        not np.isfinite(intermediate).all()
        or np.any(intermediate < -SHAPE_NUMERICAL_TOLERANCE)
        or not np.allclose(
            intermediate,
            shares,
            rtol=0.0,
            atol=SHAPE_NUMERICAL_TOLERANCE,
        )
    ):
        raise ValueError("shape_future中间四时段份额未命中第一阶段应用目标")
    if (
        not np.isfinite(final).all()
        or np.any(final < -SHAPE_NUMERICAL_TOLERANCE)
        or not np.isclose(
            final.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=SHAPE_NUMERICAL_TOLERANCE,
        ).all()
        or not np.allclose(
            deviation,
            final - intermediate,
            rtol=0.0,
            atol=SHAPE_NUMERICAL_TOLERANCE,
        )
    ):
        raise ValueError("shape_future最终四时段份额或偏移字段非法")

    requested_pv = pd.to_numeric(
        frame["peak_valley_target_requested"], errors="coerce"
    ).to_numpy(float)
    range_limited_pv = pd.to_numeric(
        frame["peak_valley_target_range_limited"], errors="coerce"
    ).to_numpy(float)
    applied_pv = pd.to_numeric(
        frame["peak_valley_target_applied"], errors="coerce"
    ).to_numpy(float)
    lambda_values = pd.to_numeric(
        frame["peak_valley_lambda"], errors="coerce"
    ).to_numpy(float)
    final_rates = pd.to_numeric(
        frame["peak_valley_rate_final"], errors="coerce"
    ).to_numpy(float)
    future_rates = pd.to_numeric(
        frame["future_peak_valley_rate"], errors="coerce"
    ).to_numpy(float)
    after_four_window = pd.to_numeric(
        frame["peak_valley_rate_after_four_window"], errors="coerce"
    ).to_numpy(float)
    eta = pd.to_numeric(frame["peak_valley_cumulative_damping"], errors="coerce").to_numpy(float)
    status = frame["peak_valley_adjustment_status"].astype(str).to_numpy()
    flat_unreachable = status == "flat_intermediate_target_unreachable"
    if (
        not np.isfinite(requested_pv).all()
        or not np.isfinite(range_limited_pv).all()
        or not np.isfinite(applied_pv).all()
        or not np.isfinite(lambda_values).all()
        or not np.isfinite(final_rates).all()
        or not np.isfinite(future_rates).all()
        or not np.isfinite(after_four_window).all()
        or not np.isfinite(eta).all()
        or np.any(eta < 0.0)
        or np.any((range_limited_pv < 0.0) | (range_limited_pv > 1.0))
        or np.any((applied_pv < 0.0) | (applied_pv > 1.0))
        or not np.allclose(
            range_limited_pv,
            np.clip(requested_pv, 0.0, 1.0),
            rtol=0.0,
            atol=PEAK_VALLEY_RATE_TOLERANCE,
        )
        or not np.allclose(
            final_rates,
            applied_pv,
            rtol=0.0,
            atol=PEAK_VALLEY_RATE_TOLERANCE,
        )
        or not np.allclose(
            future_rates,
            final_rates,
            rtol=0.0,
            atol=PEAK_VALLEY_RATE_TOLERANCE,
        )
        or not np.allclose(
            applied_pv[gate_enabled & ~flat_unreachable],
            range_limited_pv[gate_enabled & ~flat_unreachable],
            rtol=0.0,
            atol=PEAK_VALLEY_RATE_TOLERANCE,
        )
    ):
        raise ValueError("shape_future峰谷差率目标或最终值非法")
    if flat_unreachable.any() and (
        not np.allclose(applied_pv[flat_unreachable], 0.0, rtol=0.0, atol=0.0)
        or not np.allclose(final_rates[flat_unreachable], 0.0, rtol=0.0, atol=0.0)
    ):
        raise ValueError("shape_future严格平坦例外处理不一致")

    disabled = ~gate_enabled
    expected_skipped_status = np.asarray(
        [reason.replace("disabled_", "skipped_", 1) for reason in gate_reasons],
        dtype=object,
    )
    if disabled.any() and (
        not np.allclose(
            lambda_values[disabled],
            1.0,
            rtol=0.0,
            atol=SHAPE_NUMERICAL_TOLERANCE,
        )
        or not np.allclose(
            applied_pv[disabled],
            after_four_window[disabled],
            rtol=0.0,
            atol=PEAK_VALLEY_RATE_TOLERANCE,
        )
        or not np.allclose(
            final_rates[disabled],
            after_four_window[disabled],
            rtol=0.0,
            atol=PEAK_VALLEY_RATE_TOLERANCE,
        )
        or not np.allclose(
            final[disabled],
            intermediate[disabled],
            rtol=0.0,
            atol=SHAPE_NUMERICAL_TOLERANCE,
        )
        or not np.allclose(
            deviation[disabled],
            0.0,
            rtol=0.0,
            atol=SHAPE_NUMERICAL_TOLERANCE,
        )
        or not np.array_equal(
            status[disabled],
            expected_skipped_status[disabled],
        )
    ):
        raise ValueError("峰谷差率门控关闭时必须仅保留四时段结果")

    enabled = _boolean_column(
        frame,
        "peak_valley_extrapolation_enabled",
        label="shape_future",
    )
    clipped = _boolean_column(
        frame,
        "peak_valley_target_clipped",
        label="shape_future",
    )
    expected_clipped = (requested_pv < 0.0) | (requested_pv > 1.0)
    if not enabled.all() or not np.array_equal(clipped, expected_clipped):
        raise ValueError("shape_future峰谷差率开关或范围保护标记不一致")
    if set(frame["shape_extrapolation_method"].astype(str)) != {
        SHAPE_EXTRAPOLATION_METHOD
    }:
        raise ValueError("shape_future形状外推方法标记不一致")
    if set(frame["four_window_final_share_policy"].astype(str)) != {
        FOUR_WINDOW_FINAL_SHARE_POLICY
    }:
        raise ValueError("shape_future最终四时段份额策略标记不一致")

    if set(SHAPE_COLS).issubset(frame.columns):
        curves = frame[list(SHAPE_COLS)].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        if (
            not np.isfinite(curves).all()
            or np.any(curves < -SHAPE_NUMERICAL_TOLERANCE)
            or not np.isclose(
                curves.sum(axis=1),
                1.0,
                rtol=0.0,
                atol=SHAPE_NUMERICAL_TOLERANCE,
            ).all()
        ):
            raise ValueError("shape_future最终24点曲线非法")
        actual_rates = np.asarray(
            [peak_valley_rate(curve) for curve in curves], dtype=float
        )
        actual_final_shares = np.column_stack(
            (
                curves[:, 0:6].sum(axis=1),
                curves[:, 6:12].sum(axis=1),
                curves[:, 12:17].sum(axis=1),
                curves[:, 17:24].sum(axis=1),
            )
        )
        if not np.allclose(
            actual_rates,
            applied_pv,
            rtol=0.0,
            atol=PEAK_VALLEY_RATE_TOLERANCE,
        ) or not np.allclose(
            actual_final_shares,
            final,
            rtol=0.0,
            atol=SHAPE_NUMERICAL_TOLERANCE,
        ):
            raise ValueError("shape_future最终24点曲线与审计字段不一致")


def _validate_day_type_factor_schema(frame: pd.DataFrame) -> None:
    if "industry" in frame.columns:
        raise ValueError("day_type_factors不得含industry：3.5.0只允许全省共享系数")
    missing = sorted(REQUIRED_DAY_TYPE_FACTOR_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"day_type_factors缺少3.5.0字段：{missing}")
    if frame.empty:
        raise ValueError("day_type_factors不得为空")
    scopes = set(frame["factor_scope"].dropna().astype(str))
    if scopes != {PROVINCE_SHARED_FACTOR_SCOPE}:
        raise ValueError(f"day_type_factors.factor_scope口径不符：{sorted(scopes)}")
    source_years = pd.to_numeric(frame["source_year"], errors="coerce")
    if source_years.isna().any() or set(source_years.astype(int)) != {2023}:
        raise ValueError("day_type_factors.source_year必须全部为2023")
    factors = pd.to_numeric(frame["factor"], errors="coerce").to_numpy(float)
    if not np.isfinite(factors).all() or (factors <= 0).any():
        raise ValueError("day_type_factors.factor必须为有限正数")
    if frame.duplicated(["season", "final_day_type"]).any():
        raise ValueError("day_type_factors季节/日型键重复")
    seasons = ("春季", "夏季", "秋季", "冬季")
    if set(frame["season"].astype(str)) != set(seasons):
        raise ValueError("day_type_factors必须覆盖春夏秋冬")
    split = {"工作日", "普通周末", "法定节假日"}
    merged = {"工作日", "合并非工作日"}
    for season in seasons:
        observed = set(
            frame.loc[frame["season"].astype(str).eq(season), "final_day_type"]
            .astype(str)
        )
        if observed not in (split, merged):
            raise ValueError(
                f"day_type_factors日型组合非法：{season}/{sorted(observed)}"
            )


def validate_result_tables_contract(tables: Mapping[str, pd.DataFrame]) -> None:
    """Validate the mandatory 3.5.0 table set before writing any file."""

    if not isinstance(tables, Mapping):
        raise TypeError("结果表集合必须是映射")
    names = set(tables)
    retired = sorted(names.intersection(FORBIDDEN_FORMAL_TABLES))
    if retired:
        raise ValueError(f"正式结果包含已退役表：{retired}")
    missing = sorted(REQUIRED_FORMAL_TABLES.difference(names))
    if missing:
        raise ValueError(f"正式结果缺少3.5.0核心表：{missing}")
    for name, frame in tables.items():
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"结果表{name}不是DataFrame")
    _validate_shape_future_schema(tables["shape_future"])
    _validate_day_type_factor_schema(tables["day_type_factors"])


def validate_tables_manifest_contract(manifest: Mapping[str, object]) -> None:
    """Validate names, schemas and version stamps without loading large CSVs."""

    if not isinstance(manifest, Mapping):
        raise RuntimeError("结果清单必须是JSON对象")
    names = set(manifest)
    retired = sorted(names.intersection(FORBIDDEN_FORMAL_TABLES))
    if retired:
        raise RuntimeError(f"结果清单包含已退役表：{retired}")
    missing = sorted(REQUIRED_FORMAL_TABLES.difference(names))
    if missing:
        raise RuntimeError(f"结果清单缺少3.5.0核心表：{missing}")
    for name, raw_entry in manifest.items():
        if not isinstance(raw_entry, Mapping):
            raise RuntimeError(f"结果清单{name}条目不是对象")
        if raw_entry.get("schema_version") != RESULT_SCHEMA_VERSION:
            raise RuntimeError(
                f"结果清单{name}.schema_version="
                f"{raw_entry.get('schema_version')!r}，预期{RESULT_SCHEMA_VERSION!r}"
            )
        file_name = raw_entry.get("file")
        columns = raw_entry.get("columns")
        if not isinstance(file_name, str) or not file_name:
            raise RuntimeError(f"结果清单{name}缺少有效file")
        if not isinstance(columns, list) or any(
            not isinstance(column, str) for column in columns
        ):
            raise RuntimeError(f"结果清单{name}缺少有效columns")

    shape_columns = set(manifest["shape_future"]["columns"])  # type: ignore[index]
    retired_shape = sorted(shape_columns.intersection(FORBIDDEN_SHAPE_FUTURE_COLUMNS))
    missing_shape = sorted(REQUIRED_SHAPE_FUTURE_COLUMNS.difference(shape_columns))
    if retired_shape:
        raise RuntimeError(f"shape_future清单包含已退役字段：{retired_shape}")
    if missing_shape:
        raise RuntimeError(f"shape_future清单缺少3.5.0字段：{missing_shape}")
    factor_columns = set(manifest["day_type_factors"]["columns"])  # type: ignore[index]
    if "industry" in factor_columns:
        raise RuntimeError("day_type_factors清单不得含industry")
    missing_factor = sorted(REQUIRED_DAY_TYPE_FACTOR_COLUMNS.difference(factor_columns))
    if missing_factor:
        raise RuntimeError(f"day_type_factors清单缺少3.5.0字段：{missing_factor}")


def build_industry_classification_table(
    sample_mapping: pd.DataFrame,
    sample_energy: pd.DataFrame,
    source_trace: pd.DataFrame,
) -> pd.DataFrame:
    """Combine all sample industries, including explicitly excluded new loads."""

    mapping_required = {
        "industry",
        "broad_sector",
        "is_traditional",
        "official_source_row",
        "official_label",
    }
    energy_required = {
        "industry",
        "official_energy_2025",
        "official_energy_basis",
        "unit",
    }
    _require_columns(sample_mapping, mapping_required, "样本行业映射")
    _require_columns(sample_energy, energy_required, "样本行业官方电量")
    if sample_mapping.industry.duplicated().any() or sample_energy.industry.duplicated().any():
        raise ValueError("行业分类输入必须按industry唯一")

    # official_label is already authoritative in sample_mapping.  Deliberately
    # do not merge a second copy, which would create official_label_x/_y.
    energy_columns = [
        "industry",
        "official_energy_2025",
        "official_energy_basis",
        "unit",
    ]
    result = sample_mapping.copy().merge(
        sample_energy[energy_columns],
        on="industry",
        how="left",
        validate="one_to_one",
    )
    if result[energy_columns[1:]].isna().any().any():
        missing = result.loc[result.official_energy_2025.isna(), "industry"].tolist()
        raise ValueError(f"行业分类缺少官方2025电量或单位：{missing}")

    trace_columns = ["industry", "parameter_valid", "uses_own_parameters", "parameter_status"]
    if source_trace.empty:
        trace = pd.DataFrame(columns=trace_columns)
    else:
        _require_columns(source_trace, set(trace_columns), "参数来源追溯")
        if source_trace.industry.duplicated().any():
            raise ValueError("参数来源追溯必须按industry唯一")
        trace = source_trace[trace_columns].copy()
    result = result.merge(trace, on="industry", how="left", validate="one_to_one")
    result["is_traditional"] = result.is_traditional.fillna(False).astype(bool)
    result["parameter_valid"] = result.parameter_valid.fillna(False).astype(bool)
    result["uses_own_parameters"] = result.uses_own_parameters.fillna(False).astype(bool)
    missing_traditional_trace = result.is_traditional & result.parameter_status.isna()
    result.loc[missing_traditional_trace, "parameter_status"] = "parameter_trace_unavailable"
    result["traditional_model_status"] = np.where(
        result.is_traditional,
        "纳入24细分行业形状模型",
        "留待后续新型负荷模块；当前电量包含在修正组成",
    )
    new_load = ~result.is_traditional
    result.loc[new_load, "parameter_valid"] = False
    result.loc[new_load, "uses_own_parameters"] = False
    result.loc[new_load, "parameter_status"] = "future_new_load_module_not_shaped_now"
    if "notes" not in result:
        result["notes"] = ""
    preferred = [
        "industry",
        "sample_industry",
        "broad_sector",
        "is_traditional",
        "traditional_model_status",
        "official_source_row",
        "official_label",
        "official_energy_2025",
        "official_energy_basis",
        "unit",
        "parameter_valid",
        "uses_own_parameters",
        "parameter_status",
        "notes",
    ]
    return result[[column for column in preferred if column in result]].reset_index(drop=True)


def build_target_month_energy_table(
    annual_energy: pd.DataFrame,
    envelope_factors: pd.DataFrame,
) -> pd.DataFrame:
    """Allocate upstream P10/P50/P90 natural-load energy to real months."""

    required = {
        "method",
        "scenario",
        "year",
        "P10_natural_MWh",
        "P50_natural_MWh",
        "P90_natural_MWh",
    }
    _require_columns(annual_energy, required, "年度电量")
    if annual_energy.duplicated(["method", "scenario", "year"]).any():
        raise ValueError("年度电量的method/scenario/year必须唯一")
    rows: list[dict[str, object]] = []
    for annual_row in annual_energy.itertuples(index=False):
        monthly = target_month_shares(int(annual_row.year), envelope_factors)
        for month_row in monthly.itertuples(index=False):
            record: dict[str, object] = {
                "method": str(annual_row.method),
                "planning_scenario": str(annual_row.scenario),
                "year": int(annual_row.year),
                "month": int(month_row.month),
                "hours": int(month_row.hours),
                "factor_p50": float(month_row.factor_p50),
                "month_share": float(month_row.month_share),
                "energy_scope": (
                    "natural_load_excluding_plant_aux_line_loss_"
                    "pumped_storage_charging"
                ),
                "annual_special_deduction_applied": True,
                "hourly_special_deduction_applied": False,
                "special_items_in_correction_component": False,
                "unit": "MWh",
            }
            for quantile in REPRESENTATIVE_TRACKS:
                annual_value = float(
                    getattr(annual_row, f"{quantile}_natural_MWh")
                )
                record[f"{quantile}_target_month_energy_MWh"] = (
                    annual_value * float(month_row.month_share)
                )
            rows.append(record)
    result = pd.DataFrame.from_records(rows)
    for keys, part in result.groupby(["method", "planning_scenario", "year"], sort=False):
        annual_row = annual_energy.loc[
            annual_energy.method.astype(str).eq(str(keys[0]))
            & annual_energy.scenario.astype(str).eq(str(keys[1]))
            & pd.to_numeric(annual_energy.year, errors="coerce").eq(int(keys[2]))
        ].iloc[0]
        for quantile in REPRESENTATIVE_TRACKS:
            actual = float(part[f"{quantile}_target_month_energy_MWh"].sum())
            expected = float(annual_row[f"{quantile}_natural_MWh"])
            if not np.isclose(actual, expected, rtol=1e-12, atol=1e-6):
                raise AssertionError(f"目标月电量不闭合：{keys}/{quantile}")
    return result


def _validated_monthly_peaks(
    frame: pd.DataFrame,
    *,
    label: str,
    representative: bool,
) -> pd.DataFrame:
    required = {
        "method",
        "planning_scenario",
        "year",
        "month",
        "monthly_max_MW",
        "monthly_peak_timestamp",
    }
    if representative:
        required.update({"representative_track", "bootstrap_scenario_id"})
    _require_columns(frame, required, label)
    result = frame.copy()
    for column in ("year", "month"):
        numeric = pd.to_numeric(result[column], errors="coerce")
        if (
            numeric.isna().any()
            or not np.isfinite(numeric.to_numpy(float)).all()
            or not np.equal(numeric, np.floor(numeric)).all()
        ):
            raise ValueError(f"{label}的{column}必须为整数")
        result[column] = numeric.astype(int)
    if not result.month.between(1, 12).all():
        raise ValueError(f"{label}的month必须为1..12")
    result["monthly_max_MW"] = pd.to_numeric(result.monthly_max_MW, errors="coerce")
    if (
        result.monthly_max_MW.isna().any()
        or not np.isfinite(result.monthly_max_MW.to_numpy(float)).all()
        or (result.monthly_max_MW < 0).any()
    ):
        raise ValueError(f"{label}的monthly_max_MW必须为有限非负数")
    result["monthly_peak_timestamp"] = pd.to_datetime(
        result.monthly_peak_timestamp, errors="coerce"
    )
    if result.monthly_peak_timestamp.isna().any():
        raise ValueError(f"{label}的monthly_peak_timestamp必须为有效时间")
    timestamp_matches = (
        result.monthly_peak_timestamp.dt.year.eq(result.year)
        & result.monthly_peak_timestamp.dt.month.eq(result.month)
    )
    if not timestamp_matches.all():
        raise ValueError(f"{label}的峰值时间必须落在对应year/month内")
    return result


def _validate_twelve_months(frame: pd.DataFrame, *, label: str) -> None:
    group_keys = ["method", "planning_scenario", "year"]
    expected = set(range(1, 13))
    for keys, part in frame.groupby(group_keys, sort=False, dropna=False):
        observed = set(part.month.astype(int))
        if observed != expected:
            raise ValueError(f"{label}每个方法/情景/年份必须恰有12个月：{keys}")


def build_annual_monthly_peak_curve(
    deterministic_monthly: pd.DataFrame,
    representative_monthly: pd.DataFrame,
) -> pd.DataFrame:
    """构造12个月月最大负荷年曲线，替代负荷持续曲线。

    P10/P50/P90列来自各自选中的完整真实Bootstrap代表情景；这里保留每个
    月峰值、峰值时刻和情景ID，不把12个月解释成彼此独立的月分位数。
    """

    deterministic = _validated_monthly_peaks(
        deterministic_monthly,
        label="确定性月最大负荷",
        representative=False,
    )
    representative = _validated_monthly_peaks(
        representative_monthly,
        label="真实代表情景月最大负荷",
        representative=True,
    )
    keys = ["method", "planning_scenario", "year", "month"]
    if deterministic.duplicated(keys).any():
        raise ValueError("确定性月最大负荷的method/planning_scenario/year/month必须唯一")
    if representative.duplicated([*keys, "representative_track"]).any():
        raise ValueError("真实代表情景月最大负荷的月份/代表轨迹必须唯一")
    unexpected = sorted(
        set(representative.representative_track.astype(str)).difference(
            REPRESENTATIVE_TRACKS
        )
    )
    if unexpected:
        raise ValueError(f"真实代表情景月最大负荷包含未知代表轨迹：{unexpected}")
    representative["representative_track"] = representative.representative_track.astype(str)
    coverage = representative.groupby(keys, sort=False).representative_track.nunique()
    if not coverage.eq(len(REPRESENTATIVE_TRACKS)).all():
        raise ValueError("每个月必须同时具有P10/P50/P90真实代表情景")
    if representative.bootstrap_scenario_id.isna().any():
        raise ValueError("真实代表情景月最大负荷缺少bootstrap_scenario_id")

    _validate_twelve_months(deterministic, label="确定性月最大负荷")
    for track in REPRESENTATIVE_TRACKS:
        _validate_twelve_months(
            representative.loc[representative.representative_track == track],
            label=f"{track}真实代表情景月最大负荷",
        )

    deterministic_contexts = set(
        deterministic[["method", "planning_scenario", "year"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    representative_contexts = set(
        representative[["method", "planning_scenario", "year"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if deterministic_contexts != representative_contexts:
        raise ValueError("确定性与真实代表情景月最大负荷的方法/情景/年份集合不一致")

    result = deterministic[keys + ["monthly_max_MW", "monthly_peak_timestamp"]].rename(
        columns={
            "monthly_max_MW": "deterministic_max_MW",
            "monthly_peak_timestamp": "deterministic_peak_timestamp",
        }
    )
    for track in REPRESENTATIVE_TRACKS:
        selected = representative.loc[
            representative.representative_track == track,
            keys
            + [
                "monthly_max_MW",
                "monthly_peak_timestamp",
                "bootstrap_scenario_id",
            ],
        ].rename(
            columns={
                "monthly_max_MW": f"{track}_real_max_MW",
                "monthly_peak_timestamp": f"{track}_real_peak_timestamp",
                "bootstrap_scenario_id": f"{track}_scenario_id",
            }
        )
        result = result.merge(selected, on=keys, how="left", validate="one_to_one")
    if result.isna().any().any():
        raise ValueError("12个月月最大负荷年曲线合并后存在缺失")
    result["curve_definition"] = (
        "12个月真实月最大负荷年曲线；P10/P50/P90为真实代表情景完整轨迹，"
        "不是逐月独立分位数"
    )
    result["unit"] = "MW"
    ordered = [
        *keys,
        "deterministic_max_MW",
        "deterministic_peak_timestamp",
        "P10_real_max_MW",
        "P10_real_peak_timestamp",
        "P10_scenario_id",
        "P50_real_max_MW",
        "P50_real_peak_timestamp",
        "P50_scenario_id",
        "P90_real_max_MW",
        "P90_real_peak_timestamp",
        "P90_scenario_id",
        "curve_definition",
        "unit",
    ]
    return result[ordered].sort_values(keys, kind="mergesort").reset_index(drop=True)


def save_result_tables(
    tables: Mapping[str, pd.DataFrame], output_dir: str | Path
) -> dict[str, dict[str, object]]:
    """Write a complete 3.5.0 table set and return its versioned manifest.

    Validation intentionally happens before the output directory is created so
    an old-method table collection cannot leave a partially published result.
    """

    validate_result_tables_contract(tables)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    previous_path = directory / "tables_manifest.json"
    previous: dict[str, dict[str, object]] = {}
    if previous_path.is_file():
        loaded = json.loads(previous_path.read_text(encoding="utf-8-sig"))
        if isinstance(loaded, dict):
            previous = loaded
    manifest: dict[str, dict[str, object]] = {}
    for name, frame in tables.items():
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"结果表{name}不是DataFrame")
        path = directory / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")
        digest = _sha256_path(path)
        manifest[name] = {
            "file": path.name,
            "format": "csv",
            "schema_version": RESULT_SCHEMA_VERSION,
            "rows": len(frame),
            "columns": list(frame.columns),
            "sha256": digest,
        }
    previous_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Only remove exact CSVs listed by the previous manifest and no longer
    # produced. This prevents old-contract tables from silently coexisting
    # with the new result set while keeping unrelated user files untouched.
    active_files = {str(entry["file"]) for entry in manifest.values()}
    for entry in previous.values():
        if not isinstance(entry, dict):
            continue
        filename = entry.get("file")
        if not isinstance(filename, str) or filename in active_files:
            continue
        stale = (directory / filename).resolve()
        if stale.parent == directory.resolve() and stale.suffix.lower() == ".csv" and stale.is_file():
            stale.unlink()
    return manifest


def annual_indicator_quantiles(scenario_metrics: pd.DataFrame) -> pd.DataFrame:
    identifiers = ["method", "planning_scenario", "year"]
    metrics = ["annual_energy_MWh", "annual_max_MW", "annual_min_MW", "peak_valley_difference_MW", "Tmax_h", "annual_load_factor", "mean_daily_load_factor"]
    rows = []
    for keys, part in scenario_metrics.groupby(identifiers, sort=True):
        for metric in metrics:
            values = pd.to_numeric(part[metric], errors="coerce").dropna()
            rows.append({**dict(zip(identifiers, keys)), "metric": metric, "P10": float(values.quantile(0.10)), "P50": float(values.quantile(0.50)), "P90": float(values.quantile(0.90)), "scenario_count": len(values), "result_type": "指标自身经验分位"})
    return pd.DataFrame(rows)


def monthly_indicator_quantiles(scenario_metrics: pd.DataFrame) -> pd.DataFrame:
    identifiers = ["method", "planning_scenario", "year", "month"]
    metrics = ["monthly_energy_MWh", "monthly_average_MW", "monthly_max_MW", "peak_day_average_MW", "peak_day_min_MW", "peak_day_load_factor", "monthly_imbalance_rate"]
    rows = []
    for keys, part in scenario_metrics.groupby(identifiers, sort=True):
        for metric in metrics:
            values = pd.to_numeric(part[metric], errors="coerce").dropna()
            rows.append({**dict(zip(identifiers, keys)), "metric": metric, "P10": float(values.quantile(0.10)), "P50": float(values.quantile(0.50)), "P90": float(values.quantile(0.90)), "scenario_count": len(values), "result_type": "指标自身经验分位"})
    return pd.DataFrame(rows)
