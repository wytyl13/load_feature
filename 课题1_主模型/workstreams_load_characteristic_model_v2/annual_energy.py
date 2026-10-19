"""年度全社会电量与24个细分行业/修正组成的电量分解。"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import (
    BROAD_SECTORS,
    CORRECTION_COMPONENT,
    INDUSTRY_WEIGHT_EVOLUTION_CSV,
    NATURAL_LOAD_CURVE_ENERGY_SCOPE,
    TRADITIONAL_COMPONENT_COUNT,
)


SCENARIO_ORDER = ("低碳", "基准", "高电气化")
# 现行正式年度电量主方法。方法A仍保留在上游输入中，只有调用者显式指定时使用。
ANNUAL_ENERGY_METHOD = "M1"
BASE_WEIGHT_YEAR = 2025
TARGET_WEIGHT_YEAR = 2040
QUANTILES = ("P10", "P50", "P90")
NATURAL_LOAD_SCOPE = NATURAL_LOAD_CURVE_ENERGY_SCOPE


def _read_upstream_payload(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("年度电量上游JSON顶层必须为对象")
    return payload


def _requested_years(years: Iterable[int] | None) -> tuple[int, ...] | None:
    return None if years is None else tuple(sorted({int(year) for year in years}))


def _validate_context_coverage(
    frame: pd.DataFrame,
    *,
    requested: tuple[int, ...] | None,
    scenario_column: str,
    label: str,
) -> None:
    if requested is None:
        return
    expected = {(scenario, year) for scenario in SCENARIO_ORDER for year in requested}
    actual = set(
        zip(frame[scenario_column].astype(str), pd.to_numeric(frame.year).astype(int))
    )
    absent = sorted(expected - actual)
    if absent:
        raise ValueError(f"{label}缺少规划情景/年份：{absent[:10]}")


def load_upstream_annual_energy(
    path: str | Path,
    years: Iterable[int] | None = None,
    method: str = ANNUAL_ENERGY_METHOD,
) -> pd.DataFrame:
    """读取上游毛电量与自然负荷电量，直接采用上游自然负荷结果。

    ``P*_natural_total`` 是年度组成和逐时装配的唯一电量目标。这里的
    ``full-natural`` 只用于形成可审计桥接字段，绝不再次按比例扣减。
    """

    payload = _read_upstream_payload(path)
    if "annual_electricity" not in payload:
        raise ValueError("年度电量上游JSON缺少 annual_electricity")
    frame = pd.DataFrame(payload["annual_electricity"])
    required = {"method", "scenario", "year"}
    for quantile in QUANTILES:
        required.update({quantile, f"{quantile}_natural_total"})
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "年度电量上游缺少full/natural必要字段："
            f"{sorted(missing)}"
        )
    frame = frame.loc[frame.method.eq(method)].copy()
    requested = _requested_years(years)
    if requested is not None:
        frame = frame.loc[frame.year.isin(requested)]
    if frame.empty:
        raise ValueError("年度电量上游筛选结果为空")
    if frame.duplicated(["scenario", "year"]).any():
        raise ValueError("年度电量上游 scenario/year 必须唯一")
    _validate_context_coverage(
        frame,
        requested=requested,
        scenario_column="scenario",
        label="年度电量",
    )
    for quantile in QUANTILES:
        full = pd.to_numeric(frame[quantile], errors="raise")
        natural = pd.to_numeric(
            frame[f"{quantile}_natural_total"], errors="raise"
        )
        if (~np.isfinite(full)).any() or (full <= 0).any():
            raise ValueError(f"年度全社会毛电量{quantile}必须有限且为正")
        if (~np.isfinite(natural)).any() or (natural <= 0).any():
            raise ValueError(f"年度natural load电量{quantile}必须有限且为正")
        deduction = full - natural
        if (deduction < -1e-10).any():
            raise ValueError(f"年度{quantile}自然负荷不得大于全社会毛电量")
        # 上游单位为亿千瓦时；1亿千瓦时 = 100000 MWh。
        frame[f"{quantile}_full_MWh"] = full * 1e5
        frame[f"{quantile}_natural_MWh"] = natural * 1e5
        frame[f"{quantile}_special_deduction_MWh"] = deduction * 1e5
        frame[f"{quantile}_deduction_ratio"] = deduction / full
        for sector in BROAD_SECTORS:
            source = f"{quantile}_{sector}"
            if source in frame:
                frame[f"{quantile}_full_{sector}_MWh"] = (
                    pd.to_numeric(frame[source], errors="raise") * 1e5
                )
            natural_source = f"{quantile}_natural_{sector}"
            if natural_source in frame:
                frame[f"{quantile}_natural_{sector}_MWh"] = (
                    pd.to_numeric(frame[natural_source], errors="raise") * 1e5
                )
    for prefix in ("", "natural_"):
        values = frame[
            [
                quantile if not prefix else f"{quantile}_{prefix}total"
                for quantile in QUANTILES
            ]
        ].to_numpy(float)
        if np.any(np.diff(values, axis=1) < -1e-10):
            label = "毛电量" if not prefix else "自然负荷电量"
            raise ValueError(f"年度{label}分位必须满足P10≤P50≤P90")
    frame["curve_energy_scope"] = NATURAL_LOAD_SCOPE
    frame["annual_special_deduction_applied"] = True
    frame["hourly_special_deduction_applied"] = False
    frame["special_items_in_correction_component"] = False
    frame["annual_energy_basis"] = "upstream_natural_total_no_second_deduction"
    frame["second_deduction_applied"] = False
    frame["natural_total_source_fields"] = "P10_natural_total|P50_natural_total|P90_natural_total"

    # 将长表桥接作为独立审计源核对，防止上游自然负荷字段被误读或重复扣减。
    bridge = load_upstream_scope_bridge(path, years=requested, method=method)
    annual_keys = frame[["method", "scenario", "year"]].copy()
    for quantile in QUANTILES:
        check = annual_keys.copy()
        check["quantile"] = quantile
        check["annual_full_MWh"] = frame[f"{quantile}_full_MWh"].to_numpy(float)
        check["annual_natural_MWh"] = frame[f"{quantile}_natural_MWh"].to_numpy(float)
        check = check.rename(columns={"scenario": "planning_scenario"})
        audit = check.merge(
            bridge[
                [
                    "method",
                    "planning_scenario",
                    "year",
                    "quantile",
                    "Q_full_MWh",
                    "Q_natural_MWh",
                ]
            ],
            on=["method", "planning_scenario", "year", "quantile"],
            how="left",
            validate="one_to_one",
        )
        if audit[["Q_full_MWh", "Q_natural_MWh"]].isna().any().any():
            raise ValueError(f"scope_bridge缺少{quantile}年度桥接记录")
        if not np.isclose(
            audit.annual_full_MWh, audit.Q_full_MWh, rtol=1e-10, atol=1e-2
        ).all() or not np.isclose(
            audit.annual_natural_MWh,
            audit.Q_natural_MWh,
            rtol=1e-10,
            atol=1e-2,
        ).all():
            raise ValueError(f"annual_electricity与scope_bridge的{quantile}口径不一致")
    explicit_names: dict[str, str] = {}
    for quantile in QUANTILES:
        explicit_names[quantile] = f"{quantile}_full_100m_kWh"
        explicit_names[f"{quantile}_natural_total"] = (
            f"{quantile}_natural_100m_kWh"
        )
        for sector in BROAD_SECTORS:
            explicit_names[f"{quantile}_{sector}"] = (
                f"{quantile}_full_{sector}_100m_kWh"
            )
            explicit_names[f"{quantile}_natural_{sector}"] = (
                f"{quantile}_natural_{sector}_100m_kWh"
            )
    explicit_names.update(
        {f"share_{sector}": f"full_share_{sector}" for sector in BROAD_SECTORS}
    )
    frame = frame.rename(
        columns={
            source: target
            for source, target in explicit_names.items()
            if source in frame.columns
        }
    )
    return frame.sort_values(["scenario", "year"], kind="mergesort").reset_index(drop=True)


def load_upstream_scope_bridge(
    path: str | Path,
    years: Iterable[int] | None = None,
    method: str = ANNUAL_ENERGY_METHOD,
) -> pd.DataFrame:
    """读取并验证上游毛电量—三项扣除—自然负荷桥接长表。"""

    payload = _read_upstream_payload(path)
    if "scope_bridge" not in payload:
        raise ValueError("年度电量上游JSON缺少 scope_bridge")
    frame = pd.DataFrame(payload["scope_bridge"])
    required = {
        "method",
        "scenario",
        "year",
        "quantile",
        "full_society_electricity",
        "special_deduction",
        "natural_load_electricity",
        "deduction_ratio",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"scope_bridge缺少字段：{sorted(missing)}")
    requested = _requested_years(years)
    frame = frame.loc[frame.method.astype(str).eq(str(method))].copy()
    if requested is not None:
        frame = frame.loc[pd.to_numeric(frame.year).isin(requested)].copy()
    if frame.empty:
        raise ValueError("scope_bridge筛选结果为空")
    frame = frame.rename(columns={"scenario": "planning_scenario"})
    keys = ["method", "planning_scenario", "year", "quantile"]
    if frame.duplicated(keys).any():
        raise ValueError("scope_bridge的method/scenario/year/quantile必须唯一")
    if not set(frame["quantile"].astype(str)).issubset(QUANTILES):
        raise ValueError("scope_bridge包含未知分位")
    _validate_context_coverage(
        frame,
        requested=requested,
        scenario_column="planning_scenario",
        label="scope_bridge",
    )
    if requested is not None:
        expected = {
            (scenario, year, quantile)
            for scenario in SCENARIO_ORDER
            for year in requested
            for quantile in QUANTILES
        }
        actual = set(
            zip(
                frame.planning_scenario.astype(str),
                pd.to_numeric(frame.year).astype(int),
                frame["quantile"].astype(str),
            )
        )
        absent = sorted(expected - actual)
        if absent:
            raise ValueError(f"scope_bridge缺少情景/年份/分位：{absent[:10]}")
    for source in (
        "full_society_electricity",
        "special_deduction",
        "natural_load_electricity",
        "deduction_ratio",
    ):
        frame[source] = pd.to_numeric(frame[source], errors="raise")
        if not np.isfinite(frame[source].to_numpy(float)).all():
            raise ValueError(f"scope_bridge的{source}必须有限")
    if (
        (frame.full_society_electricity <= 0).any()
        or (frame.natural_load_electricity <= 0).any()
        or (frame.special_deduction < 0).any()
        or (~frame.deduction_ratio.between(0, 1, inclusive="left")).any()
    ):
        raise ValueError("scope_bridge毛电量/自然负荷/扣除量或扣除比例越界")
    bridge_error = (
        frame.full_society_electricity
        - frame.special_deduction
        - frame.natural_load_electricity
    )
    if not np.isclose(bridge_error, 0.0, rtol=0.0, atol=1e-7).all():
        raise ValueError("scope_bridge不满足Q_full-D=Q_natural")
    ratio = frame.special_deduction / frame.full_society_electricity
    if not np.isclose(ratio, frame.deduction_ratio, rtol=1e-8, atol=1e-10).all():
        raise ValueError("scope_bridge扣除比例与扣除量不一致")
    scope_definition = payload.get("scope_definition", {})
    ratio_reference: float | None = None
    if isinstance(scope_definition, dict) and "deduction_ratio" in scope_definition:
        ratio_reference = float(scope_definition["deduction_ratio"])
        if not math.isfinite(ratio_reference) or not 0 <= ratio_reference < 1:
            raise ValueError("scope_definition.deduction_ratio越界")
        if not np.isclose(
            frame.deduction_ratio,
            ratio_reference,
            rtol=1e-8,
            atol=1e-10,
        ).all():
            raise ValueError("预测期scope_bridge未保持上游2025固定扣除比例")
    frame["Q_full_100m_kWh"] = frame.full_society_electricity
    frame["D_100m_kWh"] = frame.special_deduction
    frame["Q_natural_100m_kWh"] = frame.natural_load_electricity
    frame["Q_full_MWh"] = frame.Q_full_100m_kWh * 1e5
    frame["D_MWh"] = frame.D_100m_kWh * 1e5
    frame["Q_natural_MWh"] = frame.Q_natural_100m_kWh * 1e5
    frame["component_split_status"] = np.where(
        pd.to_numeric(frame.year).le(2025),
        "official_itemized_history_available",
        "aggregate_deduction_only_no_future_item_split",
    )
    frame["annual_special_deduction_applied"] = True
    frame["deduction_policy"] = (
        "upstream_fixed_2025_ratio"
        if ratio_reference is not None
        else "upstream_explicit_bridge_values"
    )
    frame["deduction_ratio_reference"] = (
        ratio_reference if ratio_reference is not None else np.nan
    )
    # The forecast bridge carries only aggregate D; never imply a future item split.
    frame["future_special_item_split_available"] = False
    frame["hourly_special_deduction_applied"] = False
    frame["special_items_in_correction_component"] = False
    frame["second_deduction_applied"] = False
    frame["curve_energy_scope"] = NATURAL_LOAD_SCOPE
    frame["source"] = str(Path(path).resolve())
    preferred = [
        "method",
        "planning_scenario",
        "year",
        "quantile",
        "Q_full_100m_kWh",
        "D_100m_kWh",
        "Q_natural_100m_kWh",
        "Q_full_MWh",
        "D_MWh",
        "Q_natural_MWh",
        "deduction_ratio",
        "deduction_policy",
        "deduction_ratio_reference",
        "component_split_status",
        "annual_special_deduction_applied",
        "future_special_item_split_available",
        "hourly_special_deduction_applied",
        "special_items_in_correction_component",
        "second_deduction_applied",
        "curve_energy_scope",
        "source",
    ]
    return frame[preferred].sort_values(keys, kind="mergesort").reset_index(drop=True)


def load_industry_weight_evolution(
    path: str | Path = INDUSTRY_WEIGHT_EVOLUTION_CSV,
) -> pd.DataFrame:
    """读取可编辑的2025—2040行业占比相对变化假设。"""

    source = Path(path)
    expected = {
        "industry",
        "base_year",
        "target_year",
        "target_multiplier",
        "rationale",
    }
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or ()) != expected:
            raise ValueError(f"行业权重演化配置字段不符：{reader.fieldnames}")
        rows = list(reader)
    frame = pd.DataFrame(rows)
    if frame.empty or frame.industry.eq("").any() or frame.industry.duplicated().any():
        raise ValueError("行业权重演化配置必须按industry非空且唯一")
    frame["base_year"] = pd.to_numeric(frame.base_year, errors="raise").astype(int)
    frame["target_year"] = pd.to_numeric(frame.target_year, errors="raise").astype(int)
    frame["target_multiplier"] = pd.to_numeric(
        frame.target_multiplier, errors="raise"
    ).astype(float)
    if set(frame.base_year) != {BASE_WEIGHT_YEAR} or set(frame.target_year) != {
        TARGET_WEIGHT_YEAR
    }:
        raise ValueError("当前行业权重配置必须统一使用2025基年和2040目标年")
    if (~np.isfinite(frame.target_multiplier)).any() or (
        frame.target_multiplier <= 0
    ).any():
        raise ValueError("行业2040目标倍数必须有限且为正")
    return frame.sort_values("industry", kind="mergesort").reset_index(drop=True)


def build_industry_annual_energy(
    annual_energy: pd.DataFrame,
    sample_industry_energy: pd.DataFrame,
    official_natural_total_2025: float,
    evolution: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """将各分位自然负荷年度电量分为24行业与第25项修正组成。

    24行业基准占比直接取各自官方2025电量除以2025自然负荷电量，不在
    一二三产或居民层面二次归一。未来占比按配置的2040相对倍数作指数平滑
    插值；2040以后固定在目标倍数。修正组成占比始终取剩余量。
    """

    required_annual = {
        "method",
        "scenario",
        "year",
        *(f"{quantile}_natural_MWh" for quantile in QUANTILES),
    }
    missing = required_annual - set(annual_energy.columns)
    if missing:
        raise ValueError(f"年度电量缺少字段：{sorted(missing)}")
    required_industry = {
        "industry",
        "broad_sector",
        "is_traditional",
        "official_energy_2025",
    }
    missing = required_industry - set(sample_industry_energy.columns)
    if missing:
        raise ValueError(f"细分行业电量缺少字段：{sorted(missing)}")
    if (
        not math.isfinite(float(official_natural_total_2025))
        or float(official_natural_total_2025) <= 0
    ):
        raise ValueError("官方2025自然负荷电量必须有限且为正")
    industries = sample_industry_energy.loc[
        sample_industry_energy.is_traditional.astype(bool),
        ["industry", "broad_sector", "official_energy_2025"],
    ].copy()
    if len(industries) != TRADITIONAL_COMPONENT_COUNT or not industries.industry.is_unique:
        raise ValueError(
            f"必须有{TRADITIONAL_COMPONENT_COUNT}个唯一传统细分行业，实际{len(industries)}"
        )
    if not set(industries.broad_sector).issubset(BROAD_SECTORS):
        raise ValueError("传统细分行业存在未知产业代理归属")
    industries["official_energy_2025"] = pd.to_numeric(
        industries.official_energy_2025, errors="raise"
    )
    if (industries.official_energy_2025 < 0).any():
        raise ValueError("细分行业官方2025电量不得为负")
    industries["base_share_2025"] = (
        industries.official_energy_2025 / float(official_natural_total_2025)
    )

    assumptions = load_industry_weight_evolution() if evolution is None else evolution.copy()
    required_evolution = {
        "industry",
        "base_year",
        "target_year",
        "target_multiplier",
        "rationale",
    }
    missing = required_evolution - set(assumptions.columns)
    if missing:
        raise ValueError(f"行业权重演化配置缺少字段：{sorted(missing)}")
    if assumptions.industry.duplicated().any():
        raise ValueError("行业权重演化配置industry必须唯一")
    absent = sorted(set(industries.industry) - set(assumptions.industry))
    extra = sorted(set(assumptions.industry) - set(industries.industry))
    if absent or extra:
        raise ValueError(f"行业权重演化配置与24行业不一致；missing={absent}, extra={extra}")
    industries = industries.merge(
        assumptions[list(required_evolution)], on="industry", how="left", validate="one_to_one"
    )

    records: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    span = TARGET_WEIGHT_YEAR - BASE_WEIGHT_YEAR
    for annual in annual_energy.itertuples(index=False):
        year = int(annual.year)
        if year < BASE_WEIGHT_YEAR:
            raise ValueError(f"行业未来权重仅定义于{BASE_WEIGHT_YEAR}年及以后：{year}")
        progress = min((year - BASE_WEIGHT_YEAR) / span, 1.0)
        for quantile in QUANTILES:
            total_MWh = float(getattr(annual, f"{quantile}_natural_MWh"))
            if not math.isfinite(total_MWh) or total_MWh <= 0:
                raise ValueError(
                    f"{annual.scenario}/{year}/{quantile}的natural_total必须为有限正数"
                )
            industry_rows: list[dict[str, object]] = []
            for industry in industries.itertuples(index=False):
                multiplier = float(industry.target_multiplier) ** progress
                share = float(industry.base_share_2025) * multiplier
                industry_rows.append(
                    {
                        "method": str(annual.method),
                        "planning_scenario": str(annual.scenario),
                        "year": year,
                        "annual_energy_quantile": quantile,
                        "annual_energy_basis": f"{quantile}_natural_total",
                        "natural_total_energy_MWh": total_MWh,
                        "component": str(industry.industry),
                        "industry": str(industry.industry),
                        "component_type": "传统细分行业",
                        "broad_sector": str(industry.broad_sector),
                        "base_share_2025": float(industry.base_share_2025),
                        "target_multiplier_2040": float(industry.target_multiplier),
                        "interpolation_progress": progress,
                        "electricity_share": share,
                        "annual_energy_MWh": total_MWh * share,
                        "weight_source": "官方2025行业电量/2025自然负荷电量 × 2025—2040指数平滑倍数",
                        "rationale": str(industry.rationale),
                        "curve_energy_scope": NATURAL_LOAD_SCOPE,
                        "annual_special_deduction_applied": True,
                        "hourly_special_deduction_applied": False,
                        "special_items_in_correction_component": False,
                        "second_deduction_applied": False,
                        "natural_total_source_field": f"{quantile}_natural_total",
                    }
                )
            industry_share = sum(
                float(row["electricity_share"]) for row in industry_rows
            )
            industry_energy_sum = sum(
                float(row["annual_energy_MWh"]) for row in industry_rows
            )
            correction_energy = total_MWh - industry_energy_sum
            correction_share = correction_energy / total_MWh
            if correction_share <= 0 or correction_share >= 1:
                raise ValueError(
                    f"修正组成占比越界：{annual.scenario}/{year}/{quantile}/"
                    f"{correction_share:.8f}"
                )
            records.extend(industry_rows)
            records.append(
                {
                    "method": str(annual.method),
                    "planning_scenario": str(annual.scenario),
                    "year": year,
                    "annual_energy_quantile": quantile,
                    "annual_energy_basis": f"{quantile}_natural_total",
                    "natural_total_energy_MWh": total_MWh,
                    "component": CORRECTION_COMPONENT,
                    "industry": CORRECTION_COMPONENT,
                    "component_type": "修正组成",
                    "broad_sector": CORRECTION_COMPONENT,
                    "base_share_2025": 1.0 - float(industries.base_share_2025.sum()),
                    "target_multiplier_2040": np.nan,
                    "interpolation_progress": progress,
                    "electricity_share": correction_share,
                    "annual_energy_MWh": correction_energy,
                    "weight_source": "1－24个细分行业自然负荷电量占比之和",
                    "rationale": (
                        "承接新型负荷及所有未被24行业显式覆盖的自然负荷电量；"
                        "厂用电、线损和抽蓄用电已在年度桥接中排除，不进入修正组成"
                    ),
                    "curve_energy_scope": NATURAL_LOAD_SCOPE,
                    "annual_special_deduction_applied": True,
                    "hourly_special_deduction_applied": False,
                    "special_items_in_correction_component": False,
                    "second_deduction_applied": False,
                    "natural_total_source_field": f"{quantile}_natural_total",
                }
            )
            actual_energy = industry_energy_sum + correction_energy
            checks.append(
                {
                    "method": str(annual.method),
                    "planning_scenario": str(annual.scenario),
                    "year": year,
                    "annual_energy_quantile": quantile,
                    "annual_energy_basis": f"{quantile}_natural_total",
                    "curve_energy_scope": NATURAL_LOAD_SCOPE,
                    "industry_share_sum": industry_share,
                    "correction_share": correction_share,
                    "component_share_sum": industry_share + correction_share,
                    "target_total_energy_MWh": total_MWh,
                    "component_total_energy_MWh": actual_energy,
                    "relative_error": abs(actual_energy / total_MWh - 1.0),
                    "special_deduction_energy_in_components_MWh": 0.0,
                    "annual_special_deduction_applied": True,
                    "hourly_special_deduction_applied": False,
                    "special_items_in_correction_component": False,
                    "second_deduction_applied": False,
                    "natural_total_source_field": f"{quantile}_natural_total",
                }
            )
    result = pd.DataFrame(records)
    closure = pd.DataFrame(checks)
    if float(closure.relative_error.max()) > 1e-12:
        raise AssertionError("24行业与修正组成年度电量未闭合")
    return result, closure


def build_proxy_sector_weights(component_energy: pd.DataFrame) -> pd.DataFrame:
    """汇总四类代理的派生份额；这些份额不反向参与电量分配。"""

    required = {
        "method",
        "planning_scenario",
        "year",
        "annual_energy_quantile",
        "component_type",
        "broad_sector",
        "electricity_share",
        "annual_energy_MWh",
    }
    missing = required - set(component_energy.columns)
    if missing:
        raise ValueError(f"组成电量表缺少字段：{sorted(missing)}")
    fine = component_energy.loc[component_energy.component_type.eq("传统细分行业")]
    result = (
        fine.groupby(
            [
                "method",
                "planning_scenario",
                "year",
                "annual_energy_quantile",
                "broad_sector",
            ],
            as_index=False,
        )[["electricity_share", "annual_energy_MWh"]]
        .sum()
        .rename(
            columns={
                "broad_sector": "proxy_sector",
                "electricity_share": "proxy_sector_share",
                "annual_energy_MWh": "proxy_sector_energy_MWh",
            }
        )
    )
    result["role"] = "由自然负荷口径下内部细分行业电量加权派生的代理；不是官方产业曲线或参数层"
    result["curve_energy_scope"] = NATURAL_LOAD_SCOPE
    result["special_items_in_correction_component"] = False
    return result


__all__ = [
    "ANNUAL_ENERGY_METHOD",
    "BASE_WEIGHT_YEAR",
    "SCENARIO_ORDER",
    "NATURAL_LOAD_SCOPE",
    "QUANTILES",
    "TARGET_WEIGHT_YEAR",
    "build_industry_annual_energy",
    "build_proxy_sector_weights",
    "load_industry_weight_evolution",
    "load_upstream_annual_energy",
    "load_upstream_scope_bridge",
]
