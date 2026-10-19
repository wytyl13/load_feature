"""以24个传统细分行业和1个修正组成装配全省逐时年负荷。"""

from __future__ import annotations

from itertools import product
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .annual_energy import ANNUAL_ENERGY_METHOD
from .calendar_utils import calendar_frame
from .config import (
    ANCHOR_REPAIR_ENABLED,
    FOUR_WINDOW_FINAL_SHARE_POLICY,
    LATE_PEAK_CONTROL_POLICY,
    PEAK_DAMPING_ENABLED,
    PEAK_VALLEY_EXTRAPOLATION_ENABLED,
    PEAK_VALLEY_FORMAL_PHI,
    PEAK_VALLEY_GATE_POLICY,
    PEAK_VALLEY_RATE_TOLERANCE,
    SHAPE_EXTRAPOLATION_METHOD,
    SHAPE_NUMERICAL_TOLERANCE,
)
from .correction_curve import (
    CORRECTION_SHAPE_STATUS,
    HISTORICAL_SHAPE_SCOPE,
    NATURAL_LOAD_SCOPE,
)
from .day_types import HOLIDAY, MERGED_NONWORKDAY, WEEKEND, WORKDAY, canonical_day_type
from .monthly_envelope import target_month_shares
from .quality import SHAPE_COLS
from .shape_parameters import (
    classify_peak_valley_delta,
    evaluate_peak_valley_evolution_gate,
    shape_for_trajectory_row,
)
from .shape_transform import peak_valley_rate


SEASONS = ("春季", "夏季", "秋季", "冬季")
DAY_GROUPS = ("工作日", "周末", "节假日")
MERGED_DAY_GROUP = MERGED_NONWORKDAY
ALLOWED_DAY_GROUPS = frozenset(
    (*DAY_GROUPS, WORKDAY, WEEKEND, HOLIDAY, MERGED_DAY_GROUP, "非工作日")
)
TRADITIONAL_COMPONENT_TYPE = "传统细分行业"
CORRECTION_COMPONENT_TYPE = "修正组成"
FORMAL_HORIZON = 8
FORMAL_PEAK_VALLEY_PHI = PEAK_VALLEY_FORMAL_PHI
EXPECTED_TRADITIONAL_COMPONENTS = 24
EXPECTED_COMPONENTS = 25
DEFAULT_CLOSURE_TOLERANCE = 1e-10
ANNUAL_ENERGY_QUANTILE_COLUMN = "annual_energy_quantile"
DETERMINISTIC_ANNUAL_ENERGY_QUANTILE = "P50"
ANNUAL_ENERGY_BASIS = "P50_natural_total"
TRAJECTORY_QUANTILE_BASIS = "annual_peak_bootstrap"
DAY_TYPE_FACTOR_SCOPE = "province_shared_all_25_components"


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label}缺少字段：{sorted(missing)}")


def _finite_numeric(frame: pd.DataFrame, column: str, *, label: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(float)).all():
        raise ValueError(f"{label}的{column}必须全部为有限数")
    return values.astype(float)


def _integer_numeric(frame: pd.DataFrame, column: str, *, label: str) -> pd.Series:
    values = _finite_numeric(frame, column, label=label)
    if not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{label}的{column}必须为整数")
    return values.astype(int)


def _boolean_column(
    frame: pd.DataFrame,
    column: str,
    *,
    label: str,
) -> np.ndarray:
    """Parse a complete boolean column without truthy-string coercion."""

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
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Recompute the per-cell gate and require its exact reason code."""

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
        raise ValueError(f"{label}.peak_valley_delta_abs必须为非负有限数") from exc
    if not np.isfinite(delta_abs).all() or (delta_abs < 0.0).any():
        raise ValueError(f"{label}.peak_valley_delta_abs必须为非负有限数")

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
            "峰谷差率门控结果或原因与历史充分性、平坦判定和变化区间不一致"
        )
    if not np.array_equal(actual_bands, expected_band_array):
        raise ValueError(f"{label}.peak_valley_delta_band与历史充分性和变化量不一致")
    if set(frame["peak_valley_gate_policy"].astype(str)) != {
        PEAK_VALLEY_GATE_POLICY
    }:
        raise ValueError("未来形状参数peak_valley_gate_policy不一致")
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
    *,
    label: str,
) -> None:
    """Require one history-derived gate record per cell across all futures."""

    cell_columns = ["industry", "season", "_final_day_type"]
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
            f"{label}同一行业/季节/日型的峰谷差率门控证据"
            f"跨year/H/phi必须恒定：{sample}"
        )


def _validate_cell_dimensions(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    result = frame.copy()
    if result[["season", "day_group"]].isna().any().any():
        raise ValueError(f"{label}的season/day_group不能为空")
    result["season"] = result.season.astype(str)
    result["day_group"] = result.day_group.astype(str)
    invalid_seasons = sorted(set(result.season).difference(SEASONS))
    invalid_day_groups = sorted(set(result.day_group).difference(ALLOWED_DAY_GROUPS))
    if invalid_seasons or invalid_day_groups:
        raise ValueError(
            f"{label}含未知季节或日型：season={invalid_seasons}, day_group={invalid_day_groups}"
        )
    return result


def _validate_complete_industry_cells(
    frame: pd.DataFrame,
    industries: Sequence[str],
    *,
    label: str,
) -> None:
    expected_industries = set(industries)
    observed_industries = set(frame.industry.astype(str))
    if observed_industries != expected_industries:
        raise ValueError(
            f"{label}行业集合不一致：missing={sorted(expected_industries - observed_industries)}, "
            f"extra={sorted(observed_industries - expected_industries)}"
        )
    for industry, part in frame.groupby("industry", sort=False):
        observed_cells = set(
            part[["season", "day_group"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        valid = True
        for season in SEASONS:
            observed_day_groups = {
                day_group for observed_season, day_group in observed_cells if observed_season == season
            }
            if observed_day_groups not in (
                set(DAY_GROUPS),
                {"工作日", MERGED_DAY_GROUP},
                {"工作日", "非工作日"},
            ):
                valid = False
                first = (season, ",".join(sorted(observed_day_groups)))
                break
        if not valid:
            raise ValueError(
                f"{label}单元不完整或日型组合非法：{industry}/{first[0]}/{first[1]}"
            )


def _validated_annual_components(
    component_annual_energy: pd.DataFrame,
    *,
    method: str,
    planning_scenario: str,
    year: int,
    tolerance: float,
) -> pd.DataFrame:
    required = {
        "method",
        "planning_scenario",
        "year",
        "component",
        "industry",
        "component_type",
        ANNUAL_ENERGY_QUANTILE_COLUMN,
        "electricity_share",
        "annual_energy_MWh",
    }
    _require_columns(component_annual_energy, required, "组成年度电量")
    frame = component_annual_energy.copy()
    frame["year"] = _integer_numeric(frame, "year", label="组成年度电量")
    frame["method"] = frame.method.astype(str)
    frame["planning_scenario"] = frame.planning_scenario.astype(str)
    frame[ANNUAL_ENERGY_QUANTILE_COLUMN] = (
        frame[ANNUAL_ENERGY_QUANTILE_COLUMN].astype(str).str.strip().str.upper()
    )
    invalid_quantiles = sorted(
        set(frame[ANNUAL_ENERGY_QUANTILE_COLUMN]).difference({"P10", "P50", "P90"})
    )
    if invalid_quantiles:
        raise ValueError(f"组成年度电量含未知分位：{invalid_quantiles}")
    selected = frame.loc[
        frame.method.eq(str(method))
        & frame.planning_scenario.eq(str(planning_scenario))
        & frame.year.eq(int(year))
        & frame[ANNUAL_ENERGY_QUANTILE_COLUMN].eq(
            DETERMINISTIC_ANNUAL_ENERGY_QUANTILE
        )
    ].copy()
    if selected.empty:
        raise ValueError(
            "组成年度电量缺少用于8760装配的P50自然负荷分位："
            f"{method}/{planning_scenario}/{year}"
        )
    if "annual_energy_basis" in selected:
        basis = set(selected.annual_energy_basis.astype(str))
        if basis != {ANNUAL_ENERGY_BASIS}:
            raise ValueError(
                "8760确定性装配只允许P50_natural_total："
                f"annual_energy_basis={sorted(basis)}"
            )
    if "annual_special_deduction_applied" in selected and not selected[
        "annual_special_deduction_applied"
    ].astype(bool).all():
        raise ValueError("组成年度电量必须已在上游完成三项扣除")
    if "special_items_in_correction_component" in selected and selected[
        "special_items_in_correction_component"
    ].astype(bool).any():
        raise ValueError("厂用电、线损和抽蓄用电不得进入组成电量")
    if len(selected) != EXPECTED_COMPONENTS:
        raise ValueError(
            f"组成年度电量必须恰有25项，实际{len(selected)}项：{method}/{planning_scenario}/{year}"
        )
    if selected[["component", "industry", "component_type"]].isna().any().any():
        raise ValueError("组成年度电量的component/industry/component_type不能为空")
    for column in ("component", "industry", "component_type"):
        selected[column] = selected[column].astype(str).str.strip()
    if (selected.component == "").any() or (selected.industry == "").any():
        raise ValueError("组成年度电量的component和industry不能为空")
    if selected.component.duplicated().any():
        raise ValueError("组成年度电量的component必须唯一")
    type_counts = selected.component_type.value_counts().to_dict()
    if type_counts != {
        TRADITIONAL_COMPONENT_TYPE: EXPECTED_TRADITIONAL_COMPONENTS,
        CORRECTION_COMPONENT_TYPE: 1,
    }:
        raise ValueError(
            "组成年度电量必须为24个传统细分行业和1个修正组成："
            f"actual={type_counts}"
        )
    traditional = selected.loc[
        selected.component_type.eq(TRADITIONAL_COMPONENT_TYPE)
    ]
    if traditional.industry.nunique() != EXPECTED_TRADITIONAL_COMPONENTS:
        raise ValueError("24个传统细分行业的industry必须一一唯一")
    correction = selected.loc[selected.component_type.eq(CORRECTION_COMPONENT_TYPE)]
    if str(correction.iloc[0].component) != CORRECTION_COMPONENT_TYPE:
        raise ValueError("第25项修正组成的component必须明确为“修正组成”")
    if str(correction.iloc[0].industry) != CORRECTION_COMPONENT_TYPE:
        raise ValueError("第25项修正组成的industry必须明确为“修正组成”")

    selected["electricity_share"] = _finite_numeric(
        selected, "electricity_share", label="组成年度电量"
    )
    selected["annual_energy_MWh"] = _finite_numeric(
        selected, "annual_energy_MWh", label="组成年度电量"
    )
    if (selected.electricity_share < 0).any() or (selected.annual_energy_MWh < 0).any():
        raise ValueError("组成年度电量的份额和电量必须非负")
    correction = selected.loc[selected.component_type.eq(CORRECTION_COMPONENT_TYPE)].iloc[0]
    if float(correction.electricity_share) <= 0 or float(correction.annual_energy_MWh) <= 0:
        raise ValueError("第25项修正组成的电量份额和年度电量必须为正")
    share_sum = float(selected.electricity_share.sum())
    if not np.isclose(share_sum, 1.0, rtol=0.0, atol=tolerance):
        raise ValueError(f"25个组成的electricity_share之和必须为1，实际{share_sum:.12g}")
    total_energy = float(selected.annual_energy_MWh.sum())
    if total_energy <= 0:
        raise ValueError("25个组成的年度总电量必须为正")
    implied = selected.annual_energy_MWh / total_energy
    if not np.allclose(
        implied.to_numpy(float),
        selected.electricity_share.to_numpy(float),
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError("annual_energy_MWh与electricity_share不一致")
    if "natural_total_energy_MWh" in selected:
        natural_total = _finite_numeric(
            selected, "natural_total_energy_MWh", label="组成年度电量"
        )
        if not np.isclose(
            natural_total.to_numpy(float),
            total_energy,
            rtol=tolerance,
            atol=tolerance,
        ).all():
            raise ValueError(
                "25个组成年度电量必须直接闭合到P50_natural_total"
            )
    return selected.sort_values(
        ["component_type", "component"], kind="mergesort"
    ).reset_index(drop=True)


def _validated_baseline_shapes(
    baseline_shapes: pd.DataFrame,
    industries: Sequence[str],
    *,
    tolerance: float,
) -> dict[tuple[str, str, str], Mapping[str, object]]:
    day_column = (
        "final_day_type" if "final_day_type" in baseline_shapes else "day_group"
    )
    required = {"industry", "season", day_column, *SHAPE_COLS}
    _require_columns(baseline_shapes, required, "行业基准形状")
    frame = baseline_shapes.copy()
    if "year" in frame:
        frame["year"] = _integer_numeric(frame, "year", label="行业基准形状")
        frame = frame.loc[frame.year.eq(2023)].copy()
    elif "is_future_baseline" in frame:
        marker = _boolean_column(frame, "is_future_baseline", label="行业基准形状")
        frame = frame.loc[marker].copy()
    if frame.empty:
        raise RuntimeError("2023行业工作日基准全部缺失")
    frame["industry"] = frame.industry.astype(str)
    frame["season"] = frame.season.astype(str)
    invalid_seasons = sorted(set(frame.season).difference(SEASONS))
    if invalid_seasons:
        raise ValueError(f"行业基准形状含未知季节：{invalid_seasons}")
    frame["_final_day_type"] = frame[day_column].map(canonical_day_type)
    key = ["industry", "season", "_final_day_type"]
    if frame.duplicated(key).any():
        raise ValueError("行业2023基准形状的industry/season/final_day_type必须唯一")
    expected_industries = set(str(value) for value in industries)
    observed_industries = set(frame.industry)
    if observed_industries != expected_industries:
        raise ValueError(
            "行业2023基准形状行业集合不一致："
            f"missing={sorted(expected_industries - observed_industries)}, "
            f"extra={sorted(observed_industries - expected_industries)}"
        )
    for industry in industries:
        for season in SEASONS:
            exists = frame.loc[
                frame.industry.eq(str(industry))
                & frame.season.eq(season)
                & frame._final_day_type.eq(WORKDAY)
            ]
            if len(exists) != 1:
                raise RuntimeError(
                    f"2023年{industry}/{season}工作日基准缺失或不唯一"
                )
    values = frame[SHAPE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("行业基准形状必须全部为有限数")
    if np.any(values < 0):
        raise ValueError("行业基准形状不能含负值")
    sums = values.sum(axis=1)
    if not np.isclose(sums, 1.0, rtol=0.0, atol=tolerance).all():
        row = frame.iloc[int(np.flatnonzero(~np.isclose(sums, 1.0, rtol=0.0, atol=tolerance))[0])]
        raise ValueError(
            f"行业基准形状24点和必须为1：{row.industry}/{row.season}/{row.day_group}"
        )
    trend_eligible = (
        _boolean_column(frame, "trend_eligible", label="行业基准形状")
        if "trend_eligible" in frame
        else np.ones(len(frame), dtype=bool)
    )
    records: dict[tuple[str, str, str], Mapping[str, object]] = {}
    for index, row in enumerate(frame.to_dict("records")):
        records[(str(row["industry"]), str(row["season"]), str(row["_final_day_type"]))] = {
            "shape": values[index].copy(),
            "trend_eligible": bool(trend_eligible[index]),
            "source_day_type": str(row["_final_day_type"]),
        }
    return records


def _validated_future_parameters(
    future_parameters: pd.DataFrame,
    industries: Sequence[str],
    *,
    year: int,
    horizon: int,
    peak_valley_phi: float,
    tolerance: float,
) -> dict[tuple[str, str, str], Mapping[str, object]]:
    legacy_share_columns = ["night_share", "morning_share", "midday_share", "evening_share"]
    applied_share_columns = [
        "applied_share_night",
        "applied_share_morning",
        "applied_share_midday",
        "applied_share_evening",
    ]
    day_column = (
        "final_day_type" if "final_day_type" in future_parameters else "day_group"
    )
    required = {
        "industry",
        "season",
        day_column,
        "year",
        "H",
        "phi",
        "peak_valley_phi",
    }
    sequential_columns = {
        "trend_eligible",
        "flat_curve",
        "flat_borderline",
        "peak_valley_delta_abs",
        "peak_valley_delta_band",
        "peak_valley_evolution_enabled",
        "peak_valley_gate_reason",
        "peak_valley_gate_policy",
        "peak_valley_target_requested",
        "peak_valley_target_range_limited",
        "peak_valley_target_applied",
        "peak_valley_target_clipped",
        "peak_valley_adjustment_status",
        "peak_valley_rate_after_four_window",
        "peak_valley_lambda",
        "peak_valley_rate_final",
        "shape_extrapolation_method",
        "four_window_final_share_policy",
        *{
            f"four_window_share_intermediate_{name}"
            for name in ("night", "morning", "midday", "evening")
        },
        *{
            f"four_window_share_final_{name}"
            for name in ("night", "morning", "midday", "evening")
        },
        *{
            f"four_window_share_deviation_{name}"
            for name in ("night", "morning", "midday", "evening")
        },
    }
    required.update(sequential_columns)
    required.update(applied_share_columns)
    if not set(SHAPE_COLS).issubset(future_parameters.columns):
        if set(applied_share_columns).issubset(future_parameters.columns):
            required.update(applied_share_columns)
        else:
            required.update(legacy_share_columns)
    _require_columns(future_parameters, required, "未来形状参数")
    frame = future_parameters.copy()
    frame["industry"] = frame.industry.astype(str)
    frame["season"] = frame.season.astype(str)
    frame["_final_day_type"] = frame[day_column].map(canonical_day_type)
    frame["year"] = _integer_numeric(frame, "year", label="未来形状参数")
    frame["H"] = _integer_numeric(frame, "H", label="未来形状参数")
    frame["phi"] = pd.to_numeric(frame["phi"], errors="raise")
    frame["peak_valley_phi"] = pd.to_numeric(frame["peak_valley_phi"], errors="raise")
    if not frame.phi.eq(frame.peak_valley_phi).all():
        raise ValueError("未来形状参数phi与peak_valley_phi不一致")
    _, _, gate_evidence = _validated_peak_valley_gate(
        frame,
        label="未来形状参数",
    )
    _validate_peak_valley_gate_stability(
        frame,
        gate_evidence,
        label="未来形状参数",
    )
    selected = frame.loc[
        frame.year.eq(int(year))
        & frame.H.eq(int(horizon))
        & frame.phi.eq(float(peak_valley_phi))
    ].copy()
    if selected.empty:
        raise ValueError(
            "未来形状参数缺少正式组合："
            f"year={year}, H={horizon}, phi={peak_valley_phi}"
        )
    if "base_source_year" in selected:
        source_years = _integer_numeric(
            selected, "base_source_year", label="未来形状参数"
        )
        if not source_years.eq(2023).all():
            raise ValueError("未来形状参数必须以2023 medoid为基准")
    key = ["industry", "season", "_final_day_type"]
    if selected.duplicated(key).any():
        raise ValueError(
            "未来形状参数的industry/season/final_day_type/year/H/phi必须唯一"
        )
    expected_industries = set(str(value) for value in industries)
    if set(selected.industry) != expected_industries:
        raise ValueError("未来形状参数行业集合不完整")
    if "formal_H8" in selected:
        marker = _boolean_column(selected, "formal_H8", label="未来形状参数")
        if not marker.all():
            raise ValueError("H=8未来形状参数必须全部标记formal_H8=True")
    if "formal_phi08" in selected:
        marker = _boolean_column(selected, "formal_phi08", label="未来形状参数")
        if not marker.all():
            raise ValueError("phi=0.8未来形状参数必须全部标记formal_phi08=True")
    gate_enabled, gate_reasons, _ = _validated_peak_valley_gate(
        selected,
        label="未来形状参数",
    )
    # Preserve the parsed value for the legacy no-direct-shape path, whose
    # transformation API intentionally accepts native booleans only.
    selected["peak_valley_evolution_enabled"] = gate_enabled
    records: dict[tuple[str, str, str], Mapping[str, object]] = {}
    direct_shapes = set(SHAPE_COLS).issubset(selected.columns)
    if direct_shapes:
        values = selected[SHAPE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        valid = (
            np.isfinite(values).all(axis=1)
            & (values >= 0).all(axis=1)
            & np.isclose(values.sum(axis=1), 1.0, rtol=0.0, atol=tolerance)
        )
        if not valid.all():
            raise ValueError("未来行业24点形状必须非负、有限且和为1")
        target_shares = selected[applied_share_columns].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        intermediate_columns = [
            f"four_window_share_intermediate_{name}"
            for name in ("night", "morning", "midday", "evening")
        ]
        final_columns = [
            f"four_window_share_final_{name}"
            for name in ("night", "morning", "midday", "evening")
        ]
        deviation_columns = [
            f"four_window_share_deviation_{name}"
            for name in ("night", "morning", "midday", "evening")
        ]
        intermediate_shares = selected[intermediate_columns].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        final_shares = selected[final_columns].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        deviations = selected[deviation_columns].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        actual_final_shares = np.column_stack(
            (
                values[:, 0:6].sum(axis=1),
                values[:, 6:12].sum(axis=1),
                values[:, 12:17].sum(axis=1),
                values[:, 17:24].sum(axis=1),
            )
        )
        if (
            not np.isfinite(target_shares).all()
            or np.any(target_shares < 0)
            or not np.isclose(
                target_shares.sum(axis=1), 1.0, rtol=0.0, atol=SHAPE_NUMERICAL_TOLERANCE
            ).all()
            or not np.allclose(
                intermediate_shares,
                target_shares,
                rtol=0.0,
                atol=SHAPE_NUMERICAL_TOLERANCE,
            )
        ):
            raise ValueError("中间曲线四时段份额未命中第一阶段应用目标")
        if (
            not np.isfinite(final_shares).all()
            or np.any(final_shares < 0)
            or not np.isclose(
                final_shares.sum(axis=1), 1.0, rtol=0.0, atol=SHAPE_NUMERICAL_TOLERANCE
            ).all()
            or not np.allclose(
                final_shares,
                actual_final_shares,
                rtol=0.0,
                atol=SHAPE_NUMERICAL_TOLERANCE,
            )
            or not np.allclose(
                deviations,
                final_shares - intermediate_shares,
                rtol=0.0,
                atol=SHAPE_NUMERICAL_TOLERANCE,
            )
        ):
            raise ValueError("最终四时段份额或偏移字段不一致")

        requested = selected.peak_valley_target_requested.apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        range_limited = selected.peak_valley_target_range_limited.apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        applied = selected.peak_valley_target_applied.apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        lambda_values = selected.peak_valley_lambda.apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        recorded_final_rates = selected.peak_valley_rate_final.apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        after_four_window = selected.peak_valley_rate_after_four_window.apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        actual_final_rates = np.asarray(
            [peak_valley_rate(row) for row in values],
            dtype=float,
        )
        if (
            not np.isfinite(requested).all()
            or not np.isfinite(range_limited).all()
            or not np.isfinite(applied).all()
            or not np.isfinite(lambda_values).all()
            or not np.isfinite(after_four_window).all()
            or np.any((range_limited < 0.0) | (range_limited > 1.0))
            or np.any((applied < 0.0) | (applied > 1.0))
            or not np.allclose(
                range_limited,
                np.clip(requested, 0.0, 1.0),
                rtol=0.0,
                atol=PEAK_VALLEY_RATE_TOLERANCE,
            )
            or not np.allclose(
                recorded_final_rates,
                actual_final_rates,
                rtol=0.0,
                atol=PEAK_VALLEY_RATE_TOLERANCE,
            )
            or not np.allclose(
                actual_final_rates,
                applied,
                rtol=0.0,
                atol=PEAK_VALLEY_RATE_TOLERANCE,
            )
        ):
            raise ValueError("最终峰谷差率未命中第二阶段应用目标")
        disabled = ~gate_enabled
        expected_skipped_status = np.asarray(
            [reason.replace("disabled_", "skipped_", 1) for reason in gate_reasons],
            dtype=object,
        )
        actual_status = selected.peak_valley_adjustment_status.astype(str).to_numpy()
        if disabled.any() and (
            not np.allclose(
                lambda_values[disabled],
                1.0,
                rtol=0.0,
                atol=SHAPE_NUMERICAL_TOLERANCE,
            )
            or not np.allclose(
                applied[disabled],
                after_four_window[disabled],
                rtol=0.0,
                atol=PEAK_VALLEY_RATE_TOLERANCE,
            )
            or not np.allclose(
                actual_final_rates[disabled],
                after_four_window[disabled],
                rtol=0.0,
                atol=PEAK_VALLEY_RATE_TOLERANCE,
            )
            or not np.allclose(
                final_shares[disabled],
                intermediate_shares[disabled],
                rtol=0.0,
                atol=SHAPE_NUMERICAL_TOLERANCE,
            )
            or not np.allclose(
                deviations[disabled],
                0.0,
                rtol=0.0,
                atol=SHAPE_NUMERICAL_TOLERANCE,
            )
            or not np.array_equal(
                actual_status[disabled],
                expected_skipped_status[disabled],
            )
        ):
            raise ValueError("峰谷差率门控关闭时必须仅保留四时段结果")
        flat_unreachable = selected.peak_valley_adjustment_status.astype(str).eq(
            "flat_intermediate_target_unreachable"
        ).to_numpy()
        if flat_unreachable.any() and (
            not np.allclose(applied[flat_unreachable], 0.0, atol=0.0, rtol=0.0)
            or not np.allclose(
                actual_final_rates[flat_unreachable], 0.0, atol=0.0, rtol=0.0
            )
        ):
            raise ValueError("严格平坦中间曲线的不可达目标处理不一致")
        if set(selected.shape_extrapolation_method.astype(str)) != {
            SHAPE_EXTRAPOLATION_METHOD
        }:
            raise ValueError("未来形状参数shape_extrapolation_method不一致")
        if set(selected.four_window_final_share_policy.astype(str)) != {
            FOUR_WINDOW_FINAL_SHARE_POLICY
        }:
            raise ValueError("未来形状参数four_window_final_share_policy不一致")
    else:
        source_columns = (
            applied_share_columns
            if set(applied_share_columns).issubset(selected.columns)
            else legacy_share_columns
        )
        for column in source_columns:
            selected[column] = _finite_numeric(selected, column, label="未来形状参数")
        shares = selected[source_columns].to_numpy(float)
        if np.any(shares < 0) or not np.isclose(
            shares.sum(axis=1), 1.0, rtol=0.0, atol=tolerance
        ).all():
            raise ValueError("未来形状参数的四时段份额必须非负且和为1")
        if source_columns is applied_share_columns:
            selected[legacy_share_columns] = selected[applied_share_columns].to_numpy(float)
    for index, row in enumerate(selected.to_dict("records")):
        record = dict(row)
        if direct_shapes:
            record["future_shape"] = values[index].copy()
        records[(str(row["industry"]), str(row["season"]), str(row["_final_day_type"]))] = record
    return records


def _validated_shared_day_type_factors(
    day_type_factors: pd.DataFrame,
    industries: Sequence[str] = (),
    *,
    required_seasons: Sequence[str] = SEASONS,
) -> dict[tuple[str, str], float]:
    """验证并展开全省共享日型倍率。

    旧调用方的行业表仅在各行业数值完全相同时可以折叠为共享表；
    不同行业的倍率会被拒绝，避免旧口径进入正式装配。
    """

    frame = day_type_factors.copy()
    _require_columns(frame, {"season"}, "全省共享日型倍率")
    if "factor_scope" in frame and set(frame.factor_scope.astype(str)) != {
        DAY_TYPE_FACTOR_SCOPE
    }:
        raise ValueError("日型倍率factor_scope必须为全省共享25组成")
    if "source_year" in frame:
        source_year = _integer_numeric(frame, "source_year", label="全省共享日型倍率")
        if not source_year.eq(2023).all():
            raise ValueError("全省共享日型倍率只能来自2023年")

    if "industry" in frame:
        expected = set(str(value) for value in industries)
        observed = set(frame.industry.astype(str))
        if expected and observed != expected:
            raise ValueError("兼容输入的行业日型倍率行业集合不完整")
        if "nonwork_factor" in frame:
            expanded: list[dict[str, object]] = []
            for row in frame.itertuples(index=False):
                expanded.extend(
                    [
                        {"industry": str(row.industry), "season": str(row.season), "_day_type": WORKDAY, "factor": 1.0},
                        {"industry": str(row.industry), "season": str(row.season), "_day_type": MERGED_NONWORKDAY, "factor": float(row.nonwork_factor)},
                    ]
                )
            frame = pd.DataFrame.from_records(expanded)
        else:
            day_column = "final_day_type" if "final_day_type" in frame else "day_group"
            factor_column = "factor" if "factor" in frame else "day_energy_factor"
            _require_columns(frame, {day_column, factor_column}, "兼容行业日型倍率")
            frame["_day_type"] = frame[day_column].map(canonical_day_type)
            frame["factor"] = frame[factor_column]
        frame["factor"] = pd.to_numeric(frame.factor, errors="coerce")
        spread = frame.groupby(["season", "_day_type"]).factor.agg(
            lambda values: float(values.max() - values.min())
        )
        if (spread > 1e-12).any():
            raise ValueError("旧行业日型倍率不一致，不得用于全省共享装配")
        frame = frame.groupby(["season", "_day_type"], as_index=False).factor.first()
    else:
        day_column = "final_day_type" if "final_day_type" in frame else "day_group"
        factor_column = "factor" if "factor" in frame else "day_energy_factor"
        _require_columns(frame, {day_column, factor_column}, "全省共享日型倍率")
        frame["_day_type"] = frame[day_column].map(canonical_day_type)
        frame["factor"] = frame[factor_column]

    frame["season"] = frame.season.astype(str)
    frame["factor"] = pd.to_numeric(frame.factor, errors="coerce")
    if frame.factor.isna().any() or not np.isfinite(frame.factor).all() or (frame.factor <= 0).any():
        raise ValueError("全省共享日型倍率必须为有限正数")
    if frame.duplicated(["season", "_day_type"]).any():
        raise ValueError("全省共享日型倍率season/final_day_type必须唯一")

    factors: dict[tuple[str, str], float] = {}
    for season in required_seasons:
        part = frame.loc[frame.season.eq(str(season))]
        work = part.loc[part._day_type.eq(WORKDAY), "factor"]
        if len(work) != 1 or not np.isclose(float(work.iloc[0]), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError(f"全省2023年{season}工作日倍率必须唯一且为1")
        factors[(str(season), WORKDAY)] = 1.0
        merged = part.loc[part._day_type.eq(MERGED_NONWORKDAY), "factor"]
        if not merged.empty:
            if len(merged) != 1:
                raise ValueError(f"全省{season}合并非工作日倍率不唯一")
            factors[(str(season), WEEKEND)] = float(merged.iloc[0])
            factors[(str(season), HOLIDAY)] = float(merged.iloc[0])
        else:
            for day_type in (WEEKEND, HOLIDAY):
                value = part.loc[part._day_type.eq(day_type), "factor"]
                if len(value) != 1:
                    raise ValueError(f"全省2023年{season}/{day_type}倍率缺失")
                factors[(str(season), day_type)] = float(value.iloc[0])
    return factors


def _validated_correction_shapes(
    correction_component: pd.DataFrame,
    *,
    tolerance: float,
) -> dict[tuple[str, str], np.ndarray]:
    required = {"season", "day_group", "hour", "correction_normalized_shape"}
    _require_columns(correction_component, required, "修正组成归一形状")
    frame = _validate_cell_dimensions(correction_component, label="修正组成归一形状")
    frame["_canonical_day_type"] = frame.day_group.map(canonical_day_type)
    if "special_items_in_correction_component" in frame and frame[
        "special_items_in_correction_component"
    ].astype(bool).any():
        raise ValueError("厂用电、线损和抽蓄用电不得进入修正组成")
    if "historical_shape_scope" in frame and set(
        frame.historical_shape_scope.astype(str)
    ) != {HISTORICAL_SHAPE_SCOPE}:
        raise ValueError("修正组成必须声明调度形状代理范围")
    if "correction_shape_status" in frame and set(
        frame.correction_shape_status.astype(str)
    ) != {CORRECTION_SHAPE_STATUS}:
        raise ValueError("修正组成必须声明由调度形状代理反演")
    frame["hour"] = _integer_numeric(frame, "hour", label="修正组成归一形状")
    if not frame.hour.between(0, 23).all():
        raise ValueError("修正组成归一形状的hour必须为0..23")
    key = ["season", "_canonical_day_type", "hour"]
    if frame.duplicated(key).any():
        raise ValueError("修正组成归一形状的season/day_group/hour必须唯一")
    observed = set(frame[key].itertuples(index=False, name=None))
    formal_expected = set(product(SEASONS, (WORKDAY, WEEKEND, HOLIDAY), range(24)))
    legacy_expected = set(
        product(SEASONS, (WORKDAY, MERGED_NONWORKDAY), range(24))
    )
    if observed not in (formal_expected, legacy_expected):
        raise ValueError(
            "修正组成归一形状必须完整覆盖4季×3规范日型×24小时，"
            "或旧版4季×2日型×24小时"
        )
    legacy_two_day = observed == legacy_expected
    frame["correction_normalized_shape"] = _finite_numeric(
        frame,
        "correction_normalized_shape",
        label="修正组成归一形状",
    )
    if (frame.correction_normalized_shape < -tolerance).any():
        raise ValueError("修正组成归一形状不能含超过容差的负值")
    sums = frame.groupby(["season", "_canonical_day_type"]).correction_normalized_shape.sum()
    if not np.isclose(
        sums.to_numpy(float), 1.0, rtol=0.0, atol=tolerance
    ).all():
        raise ValueError("修正组成每个季节日型的24点和必须为1")
    shapes: dict[tuple[str, str], np.ndarray] = {}
    for keys, part in frame.groupby(["season", "_canonical_day_type"], sort=False):
        values = part.sort_values("hour").correction_normalized_shape.to_numpy(float)
        # The inversion contract permits round-off in [-tolerance, 0).  Remove
        # only that numerical tail and re-close the 24-point shape before
        # multiplying by a large annual-energy value.
        values = np.maximum(values, 0.0)
        total = float(values.sum())
        if total <= 0:
            raise ValueError("修正组成归一形状清理数值尾差后不得为空")
        shapes[(str(keys[0]), str(keys[1]))] = values / total
    if legacy_two_day:
        for season in SEASONS:
            nonwork = shapes[(season, MERGED_NONWORKDAY)]
            shapes[(season, WEEKEND)] = nonwork.copy()
            shapes[(season, HOLIDAY)] = nonwork.copy()
    return shapes


def _validated_month_shares(year: int, envelope_factors: pd.DataFrame) -> pd.DataFrame:
    _require_columns(envelope_factors, {"month", "factor_p50"}, "月度包络")
    factors = envelope_factors.copy()
    factors["month"] = _integer_numeric(factors, "month", label="月度包络")
    factors["factor_p50"] = _finite_numeric(factors, "factor_p50", label="月度包络")
    if factors.month.duplicated().any() or set(factors.month) != set(range(1, 13)):
        raise ValueError("月度包络必须按1..12月各一行")
    if (factors.factor_p50 <= 0).any():
        raise ValueError("月度包络factor_p50必须为正")
    shares = target_month_shares(year, factors)
    if (
        not np.isfinite(shares.month_share.to_numpy(float)).all()
        or (shares.month_share <= 0).any()
        or not np.isclose(
            float(shares.month_share.sum()), 1.0, rtol=0.0, atol=1e-15
        )
    ):
        raise ValueError("月度包络生成的月份份额非法")
    return shares


def _validated_industry_shape_cache(
    industries: Sequence[str],
    baseline: Mapping[tuple[str, str, str], Mapping[str, object]],
    trajectories: Mapping[tuple[str, str, str], Mapping[str, object]],
    *,
    tolerance: float,
) -> tuple[
    dict[tuple[str, str, str], np.ndarray],
    dict[tuple[str, str, str], Mapping[str, object]],
]:
    shapes: dict[tuple[str, str, str], np.ndarray] = {}
    usage: dict[tuple[str, str, str], Mapping[str, object]] = {}
    expected_industries = set(str(value) for value in industries)
    if {key[0] for key in baseline} != expected_industries:
        raise ValueError("行业形状缓存的行业集合不完整")
    unexpected_parameters = set(trajectories).difference(baseline)
    if unexpected_parameters:
        raise ValueError(
            f"未来参数含无2023基准的单元：{sorted(unexpected_parameters)[:10]}"
        )
    for industry in sorted(expected_industries):
        for season in SEASONS:
            for calendar_day_type in (WORKDAY, WEEKEND, HOLIDAY):
                direct_key = (industry, season, calendar_day_type)
                merged_key = (industry, season, MERGED_NONWORKDAY)
                workday_key = (industry, season, WORKDAY)
                fallback = False
                if direct_key in baseline:
                    source_key = direct_key
                elif calendar_day_type in {WEEKEND, HOLIDAY} and merged_key in baseline:
                    source_key = merged_key
                elif calendar_day_type in {WEEKEND, HOLIDAY} and workday_key in baseline:
                    source_key = workday_key
                    fallback = True
                else:
                    raise RuntimeError(
                        f"2023年{industry}/{season}工作日基准缺失，无法回退{calendar_day_type}"
                    )

                baseline_record = baseline[source_key]
                base_values = np.asarray(baseline_record["shape"], dtype=float)
                trend_eligible = bool(
                    baseline_record.get("trend_eligible", True)
                ) and not fallback
                if trend_eligible:
                    if source_key not in trajectories:
                        raise ValueError(
                            "未来形状参数缺少可趋势单元："
                            f"{industry}/{season}/{source_key[2]}"
                        )
                    trajectory = trajectories[source_key]
                    try:
                        if "future_shape" in trajectory:
                            shape = np.asarray(trajectory["future_shape"], dtype=float)
                        else:
                            shape = shape_for_trajectory_row(base_values, trajectory)
                    except Exception as exc:
                        raise ValueError(
                            f"行业未来日内形状生成失败："
                            f"{industry}/{season}/{source_key[2]}: {exc}"
                        ) from exc
                else:
                    # 缺失非工作日时使用同季工作日2023原始基准，不施加趋势。
                    shape = base_values
                values = np.asarray(shape, dtype=float)
                if (
                    values.shape != (24,)
                    or not np.isfinite(values).all()
                    or np.any(values < 0)
                    or not np.isclose(float(values.sum()), 1.0, rtol=0.0, atol=tolerance)
                ):
                    raise ValueError(
                        "行业未来日内形状必须非负且24点和为1："
                        f"{industry}/{season}/{calendar_day_type}"
                    )
                output_key = (industry, season, calendar_day_type)
                shapes[output_key] = values.copy()
                usage[output_key] = {
                    "calendar_day_type": calendar_day_type,
                    "shape_source_day_type": source_key[2],
                    "shape_fallback_used": fallback,
                    "trend_eligible": trend_eligible,
                    "shape_source_year": 2023,
                }
    return shapes, usage


def _closure_record(
    *,
    method: str,
    planning_scenario: str,
    year: int,
    closure_scope: str,
    component: str,
    industry: str,
    component_type: str,
    month: int,
    target: float,
    actual: float,
    tolerance: float,
    date: pd.Timestamp | None = None,
    calendar_day_type: str | None = None,
    shape_source_day_type: str | None = None,
    shape_fallback_used: bool = False,
    trend_eligible: bool | None = None,
) -> dict[str, object]:
    absolute_error = abs(float(actual) - float(target))
    if target > 0:
        relative_error = absolute_error / float(target)
    else:
        relative_error = 0.0 if absolute_error == 0 else float("inf")
    passed = bool(relative_error <= tolerance and actual >= 0)
    if not passed:
        raise AssertionError(
            f"电量闭合或非负性失败：{closure_scope}/{component}/{year}/{month}, "
            f"target={target:.12g}, actual={actual:.12g}, error={relative_error:.12g}"
        )
    return {
        "method": str(method),
        "planning_scenario": str(planning_scenario),
        "year": int(year),
        "closure_scope": closure_scope,
        "component": component,
        "industry": industry,
        "component_type": component_type,
        ANNUAL_ENERGY_QUANTILE_COLUMN: DETERMINISTIC_ANNUAL_ENERGY_QUANTILE,
        "month": int(month),
        "date": pd.NaT if date is None else pd.Timestamp(date).normalize(),
        "calendar_day_type": calendar_day_type,
        "shape_source_day_type": shape_source_day_type,
        "shape_fallback_used": bool(shape_fallback_used),
        "trend_eligible": trend_eligible,
        "target_energy_MWh": float(target),
        "actual_energy_MWh": float(actual),
        "absolute_error_MWh": absolute_error,
        "relative_error": relative_error,
        "tolerance": float(tolerance),
        "passed": passed,
        "curve_energy_scope": NATURAL_LOAD_SCOPE,
        "annual_special_deduction_applied": True,
        "hourly_special_deduction_applied": False,
        "special_items_in_correction_component": False,
        "historical_shape_scope": HISTORICAL_SHAPE_SCOPE,
        "correction_shape_status": CORRECTION_SHAPE_STATUS,
        "annual_energy_basis": ANNUAL_ENERGY_BASIS,
        "trajectory_quantile_basis": TRAJECTORY_QUANTILE_BASIS,
    }


def _canonical_calendar_day_type(day: object) -> str:
    canonical = str(getattr(day, "canonical_day_type", "")).strip()
    if canonical:
        return canonical_day_type(canonical)
    detailed = str(getattr(day, "day_type", "")).strip()
    if detailed in {"普通工作日", "调休工作日", "工作日"}:
        return WORKDAY
    if detailed in {"法定节假日", "节假日"}:
        return HOLIDAY
    if detailed in {"普通周末", "周末"}:
        return WEEKEND
    legacy = str(getattr(day, "day_group", "")).strip()
    if legacy == "工作日":
        return WORKDAY
    if legacy == "非工作日":
        # Only legacy synthetic calendars lack the detailed label.  Weekend is
        # used there because old two-day-type industry shapes map both kinds to
        # the same final non-work cell.
        return WEEKEND
    raise ValueError(f"年度装配日历含未知日型：{detailed or legacy}")


def _industry_final_day_group(
    industry: str,
    season: str,
    canonical_day_group: str,
    available_shapes: Mapping[tuple[str, str, str], np.ndarray],
) -> str:
    if canonical_day_group == "工作日":
        return "工作日"
    for merged in (MERGED_DAY_GROUP, "非工作日"):
        if (industry, season, merged) in available_shapes:
            return merged
    if (industry, season, canonical_day_group) in available_shapes:
        return canonical_day_group
    raise KeyError(
        f"行业日型无法映射：{industry}/{season}/{canonical_day_group}"
    )


def allocate_component_daily_energy(
    component_month_energy: pd.DataFrame,
    target_calendar: pd.DataFrame,
    day_type_factors: pd.DataFrame,
    *,
    tolerance: float = DEFAULT_CLOSURE_TOLERANCE,
) -> pd.DataFrame:
    """用全省共享日型倍率将每个组成的月电量分配到日。"""

    _require_columns(
        component_month_energy,
        {"component", "year", "month", "month_energy_MWh"},
        "组成月度电量",
    )
    _require_columns(target_calendar, {"date"}, "目标日历")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance必须为有限正数")

    month_energy = component_month_energy.copy()
    month_energy["year"] = _integer_numeric(month_energy, "year", label="组成月度电量")
    month_energy["month"] = _integer_numeric(month_energy, "month", label="组成月度电量")
    month_energy["month_energy_MWh"] = _finite_numeric(
        month_energy, "month_energy_MWh", label="组成月度电量"
    )
    if (~month_energy.month.between(1, 12)).any() or (month_energy.month_energy_MWh < 0).any():
        raise ValueError("组成月度电量的month或电量越界")
    duplicate_key = [
        column
        for column in (
            "method",
            "planning_scenario",
            "annual_energy_quantile",
            "year",
            "month",
            "component",
        )
        if column in month_energy
    ]
    if month_energy.duplicated(duplicate_key).any():
        raise ValueError(f"组成月度电量的{duplicate_key}必须唯一")

    calendar = target_calendar.copy()
    calendar["date"] = pd.to_datetime(calendar.date, errors="coerce").dt.normalize()
    if calendar.date.isna().any() or calendar.date.duplicated().any():
        raise ValueError("目标日历日期必须有效且唯一")
    calendar["year"] = calendar.date.dt.year
    calendar["month"] = calendar.date.dt.month
    if "season" not in calendar:
        from .calendar_utils import season_of

        calendar["season"] = calendar.month.map(season_of)
    calendar["calendar_day_type"] = [
        _canonical_calendar_day_type(day)
        for day in calendar.itertuples(index=False)
    ]
    required_seasons = tuple(dict.fromkeys(calendar.season.astype(str)))
    factors = _validated_shared_day_type_factors(
        day_type_factors, required_seasons=required_seasons
    )

    rows: list[dict[str, object]] = []
    for monthly in month_energy.to_dict("records"):
        year = int(monthly["year"])
        month = int(monthly["month"])
        days = calendar.loc[
            calendar.year.eq(year) & calendar.month.eq(month)
        ].sort_values("date", kind="mergesort")
        if days.empty:
            raise ValueError(f"目标日历缺少{year}-{month:02d}")
        raw = np.asarray(
            [
                factors[(str(day.season), str(day.calendar_day_type))]
                for day in days.itertuples(index=False)
            ],
            dtype=float,
        )
        shares = raw / float(raw.sum())
        energies = float(monthly["month_energy_MWh"]) * shares
        for position, (_, day) in enumerate(days.iterrows()):
            rows.append(
                {
                    **monthly,
                    "date": pd.Timestamp(day.date),
                    "season": str(day.season),
                    "calendar_day_type": str(day.calendar_day_type),
                    "day_type_factor": float(raw[position]),
                    "daily_share": float(shares[position]),
                    "daily_energy_MWh": float(energies[position]),
                    "factor_scope": DAY_TYPE_FACTOR_SCOPE,
                    "day_type_factor_source_year": 2023,
                }
            )
    result = pd.DataFrame.from_records(rows)
    check_keys = duplicate_key
    actual = result.groupby(check_keys, as_index=False).daily_energy_MWh.sum()
    targets = month_energy[check_keys + ["month_energy_MWh"]]
    check = targets.merge(actual, on=check_keys, how="left", validate="one_to_one")
    scale = np.maximum(check.month_energy_MWh.abs().to_numpy(float), 1.0)
    if not (
        np.abs(check.daily_energy_MWh - check.month_energy_MWh).to_numpy(float)
        <= tolerance * scale
    ).all():
        raise AssertionError("全省共享日型倍率分配后组成月电量不闭合")
    return result.sort_values(check_keys + ["date"], kind="mergesort").reset_index(drop=True)


def assemble_province_year(
    year: int,
    planning_scenario: str,
    component_annual_energy: pd.DataFrame,
    baseline_shapes: pd.DataFrame,
    nonwork_factors: pd.DataFrame,
    future_parameters: pd.DataFrame,
    envelope_factors: pd.DataFrame,
    correction_component: pd.DataFrame,
    *,
    method: str = ANNUAL_ENERGY_METHOD,
    horizon: int = FORMAL_HORIZON,
    peak_valley_phi: float = FORMAL_PEAK_VALLEY_PHI,
    tolerance: float = DEFAULT_CLOSURE_TOLERANCE,
    include_daily_detail: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """以P50自然负荷电量逐组成分配月、日和24点，返回逐时曲线与闭合明细。

    24个传统行业使用2023 medoid，先按H=8外推四时段份额，再按
    phi=0.8外推峰谷差率并围绕全日均值缩放；
    第25个修正组成使用冻结的季节×规范日型 ``p_R``。全部25个组成
    共享同一月度包络和全省2023季节—日型倍率。
    """

    if int(horizon) != FORMAL_HORIZON:
        raise ValueError("正式年度装配只允许H=8")
    if float(peak_valley_phi) != FORMAL_PEAK_VALLEY_PHI:
        raise ValueError("正式年度装配只允许phi=0.8")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance必须为有限正数")
    year = int(year)
    annual = _validated_annual_components(
        component_annual_energy,
        method=method,
        planning_scenario=planning_scenario,
        year=year,
        tolerance=tolerance,
    )
    traditional = annual.loc[
        annual.component_type.eq(TRADITIONAL_COMPONENT_TYPE)
    ]
    industries = tuple(traditional.industry.astype(str))
    baseline = _validated_baseline_shapes(
        baseline_shapes, industries, tolerance=tolerance
    )
    trajectories = _validated_future_parameters(
        future_parameters,
        industries,
        year=year,
        horizon=horizon,
        peak_valley_phi=peak_valley_phi,
        tolerance=tolerance,
    )
    day_type_factors = _validated_shared_day_type_factors(
        nonwork_factors, industries, required_seasons=SEASONS
    )
    correction_shapes = _validated_correction_shapes(
        correction_component, tolerance=tolerance
    )
    month_shares = _validated_month_shares(year, envelope_factors).set_index("month")
    industry_shapes, industry_shape_usage = _validated_industry_shape_cache(
        industries,
        baseline,
        trajectories,
        tolerance=tolerance,
    )

    calendar = calendar_frame(year).sort_values("date", kind="mergesort").reset_index(drop=True)
    expected_days = 366 if pd.Timestamp(f"{year}-12-31").is_leap_year else 365
    if len(calendar) != expected_days or calendar.date.duplicated().any():
        raise ValueError(f"年度装配日历必须完整覆盖{expected_days}天")
    start_of_year = pd.Timestamp(f"{year}-01-01")
    timestamps = pd.date_range(start_of_year, f"{year}-12-31 23:00", freq="h")
    province = np.zeros(len(timestamps), dtype=float)
    province_daily_targets = np.zeros(expected_days, dtype=float)
    closure_rows: list[dict[str, object]] = []

    for component_row in annual.itertuples(index=False):
        component = str(component_row.component)
        industry = str(component_row.industry)
        component_type = str(component_row.component_type)
        annual_energy = float(component_row.annual_energy_MWh)
        component_hourly = np.zeros(len(timestamps), dtype=float)
        for month, month_days in calendar.groupby("month", sort=True):
            month = int(month)
            month_target = annual_energy * float(month_shares.loc[month, "month_share"])
            day_weights = np.asarray(
                [
                    day_type_factors[
                        (str(day.season), _canonical_calendar_day_type(day))
                    ]
                    for day in month_days.itertuples(index=False)
                ],
                dtype=float,
            )
            if not np.isfinite(day_weights).all() or np.any(day_weights <= 0):
                raise ValueError(f"日型电量倍率非法：{component}/{year}-{month:02d}")
            day_energies = month_target * day_weights / float(day_weights.sum())
            for day, daily_energy in zip(
                month_days.itertuples(index=False), day_energies, strict=True
            ):
                season = str(day.season)
                canonical_day_group = _canonical_calendar_day_type(day)
                if component_type == TRADITIONAL_COMPONENT_TYPE:
                    shape_key = (industry, season, canonical_day_group)
                    shape = industry_shapes[shape_key]
                    usage = industry_shape_usage[shape_key]
                else:
                    shape = correction_shapes[(season, canonical_day_group)]
                    usage = {
                        "calendar_day_type": canonical_day_group,
                        "shape_source_day_type": canonical_day_group,
                        "shape_fallback_used": False,
                        "trend_eligible": False,
                    }
                values = float(daily_energy) * shape
                if not np.isfinite(values).all() or np.any(values < 0):
                    raise AssertionError(
                        f"逐组成日内负荷出现非有限值或负值：{component}/{day.date}"
                    )
                start = (pd.Timestamp(day.date) - start_of_year).days * 24
                component_hourly[start : start + 24] = values
                province_daily_targets[start // 24] += float(daily_energy)
                if include_daily_detail:
                    closure_rows.append(
                        _closure_record(
                            method=method,
                            planning_scenario=planning_scenario,
                            year=year,
                            closure_scope="component_day",
                            component=component,
                            industry=industry,
                            component_type=component_type,
                            month=month,
                            date=pd.Timestamp(day.date),
                            target=float(daily_energy),
                            actual=float(values.sum()),
                            tolerance=tolerance,
                            calendar_day_type=str(usage["calendar_day_type"]),
                            shape_source_day_type=str(usage["shape_source_day_type"]),
                            shape_fallback_used=bool(usage["shape_fallback_used"]),
                            trend_eligible=bool(usage["trend_eligible"]),
                        )
                    )
            month_mask = timestamps.month == month
            month_actual = float(component_hourly[month_mask].sum())
            closure_rows.append(
                _closure_record(
                    method=method,
                    planning_scenario=planning_scenario,
                    year=year,
                    closure_scope="component_month",
                    component=component,
                    industry=industry,
                    component_type=component_type,
                    month=month,
                    target=month_target,
                    actual=month_actual,
                    tolerance=tolerance,
                )
            )
        component_actual = float(component_hourly.sum())
        closure_rows.append(
            _closure_record(
                method=method,
                planning_scenario=planning_scenario,
                year=year,
                closure_scope="component_annual",
                component=component,
                industry=industry,
                component_type=component_type,
                month=0,
                target=annual_energy,
                actual=component_actual,
                tolerance=tolerance,
            )
        )
        province += component_hourly

    if not np.isfinite(province).all() or np.any(province < 0):
        raise AssertionError("全省逐时曲线出现非有限值或负值")
    expected_hours = expected_days * 24
    if (
        len(timestamps) != expected_hours
        or timestamps.has_duplicates
        or not timestamps.is_monotonic_increasing
        or not np.all((timestamps[1:] - timestamps[:-1]) == pd.Timedelta(hours=1))
    ):
        raise AssertionError(f"全省逐时时间戳必须连续完整覆盖{expected_hours}小时")
    province_annual_target = float(annual.annual_energy_MWh.sum())
    if include_daily_detail:
        for day_position, date in enumerate(calendar.date):
            day_start = day_position * 24
            closure_rows.append(
                _closure_record(
                    method=method,
                    planning_scenario=planning_scenario,
                    year=year,
                    closure_scope="province_day",
                    component="全省",
                    industry="全省",
                    component_type="全省",
                    month=int(pd.Timestamp(date).month),
                    date=pd.Timestamp(date),
                    target=float(province_daily_targets[day_position]),
                    actual=float(province[day_start : day_start + 24].sum()),
                    tolerance=tolerance,
                )
            )
    for month in range(1, 13):
        month_target = province_annual_target * float(
            month_shares.loc[month, "month_share"]
        )
        month_actual = float(province[timestamps.month == month].sum())
        closure_rows.append(
            _closure_record(
                method=method,
                planning_scenario=planning_scenario,
                year=year,
                closure_scope="province_month",
                component="全省",
                industry="全省",
                component_type="全省",
                month=month,
                target=month_target,
                actual=month_actual,
                tolerance=tolerance,
            )
        )
    closure_rows.append(
        _closure_record(
            method=method,
            planning_scenario=planning_scenario,
            year=year,
            closure_scope="province_annual",
            component="全省",
            industry="全省",
            component_type="全省",
            month=0,
            target=province_annual_target,
            actual=float(province.sum()),
            tolerance=tolerance,
        )
    )

    hourly = pd.DataFrame(
        {
            "method": str(method),
            "planning_scenario": str(planning_scenario),
            "year": year,
            "timestamp": timestamps,
            "load_MW": province,
            "curve_energy_scope": NATURAL_LOAD_SCOPE,
            "annual_special_deduction_applied": True,
            "hourly_special_deduction_applied": False,
            "special_items_in_correction_component": False,
            "historical_shape_scope": HISTORICAL_SHAPE_SCOPE,
            "correction_shape_status": CORRECTION_SHAPE_STATUS,
            "annual_energy_basis": ANNUAL_ENERGY_BASIS,
            "trajectory_quantile_basis": TRAJECTORY_QUANTILE_BASIS,
            "new_load_independently_shaped": False,
            "new_load_in_correction_component": True,
            "external_new_load_overlay": False,
            "annual_energy_track": ANNUAL_ENERGY_BASIS,
            "day_type_factor_scope": DAY_TYPE_FACTOR_SCOPE,
            "shape_extrapolation_method": SHAPE_EXTRAPOLATION_METHOD,
            "peak_valley_extrapolation_enabled": PEAK_VALLEY_EXTRAPOLATION_ENABLED,
            "four_window_final_share_policy": FOUR_WINDOW_FINAL_SHARE_POLICY,
            "late_peak_control_policy": LATE_PEAK_CONTROL_POLICY,
            "peak_damping_enabled": PEAK_DAMPING_ENABLED,
            "anchor_repair_enabled": ANCHOR_REPAIR_ENABLED,
            ANNUAL_ENERGY_QUANTILE_COLUMN: DETERMINISTIC_ANNUAL_ENERGY_QUANTILE,
            "H": FORMAL_HORIZON,
            "phi": FORMAL_PEAK_VALLEY_PHI,
            "component_count": EXPECTED_COMPONENTS,
        }
    )
    closure = pd.DataFrame.from_records(closure_rows)
    closure["shape_extrapolation_method"] = SHAPE_EXTRAPOLATION_METHOD
    closure["peak_valley_extrapolation_enabled"] = PEAK_VALLEY_EXTRAPOLATION_ENABLED
    closure["four_window_final_share_policy"] = FOUR_WINDOW_FINAL_SHARE_POLICY
    closure["late_peak_control_policy"] = LATE_PEAK_CONTROL_POLICY
    closure["peak_damping_enabled"] = PEAK_DAMPING_ENABLED
    closure["anchor_repair_enabled"] = ANCHOR_REPAIR_ENABLED
    closure["H"] = FORMAL_HORIZON
    closure["phi"] = FORMAL_PEAK_VALLEY_PHI
    if {"run_group", "parameter_group"}.intersection(hourly.columns) or {
        "run_group",
        "parameter_group",
    }.intersection(closure.columns):
        raise AssertionError("25组成装配输出不得含旧运行组字段")
    return hourly, closure


__all__ = [
    "ANNUAL_ENERGY_BASIS",
    "ANNUAL_ENERGY_QUANTILE_COLUMN",
    "CORRECTION_COMPONENT_TYPE",
    "DEFAULT_CLOSURE_TOLERANCE",
    "DETERMINISTIC_ANNUAL_ENERGY_QUANTILE",
    "FORMAL_HORIZON",
    "FORMAL_PEAK_VALLEY_PHI",
    "TRAJECTORY_QUANTILE_BASIS",
    "TRADITIONAL_COMPONENT_TYPE",
    "allocate_component_daily_energy",
    "assemble_province_year",
]
