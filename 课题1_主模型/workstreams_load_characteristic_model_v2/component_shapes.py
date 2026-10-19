"""24个细分行业及四类代理曲线的标幺形状装配。"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .config import NATURAL_LOAD_CURVE_ENERGY_SCOPE
from .day_types import (
    HOLIDAY,
    MERGED_NONWORKDAY,
    WEEKEND,
    WORKDAY,
    canonical_day_type,
)
from .quality import SHAPE_COLS
from .shape_parameters import shape_for_trajectory_row


CELL_COLUMNS = ("season", "day_group")
SEASONS = ("春季", "夏季", "秋季", "冬季")
CANONICAL_DAY_GROUPS = (WORKDAY, WEEKEND, HOLIDAY)
MERGED_DAY_GROUP = "合并非工作日"
CONTEXT_COLUMNS = ("method", "planning_scenario", "year")
ANNUAL_ENERGY_QUANTILE_COLUMN = "annual_energy_quantile"
DETERMINISTIC_ANNUAL_ENERGY_QUANTILE = "P50"
NATURAL_LOAD_SCOPE = NATURAL_LOAD_CURVE_ENERGY_SCOPE


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label}缺少字段：{sorted(missing)}")


def _validate_shape(values: np.ndarray, label: str, tolerance: float = 1e-10) -> None:
    if values.shape != (24,) or not np.isfinite(values).all():
        raise ValueError(f"{label}必须为24个有限值")
    if (values < -tolerance).any() or not np.isclose(
        values.sum(), 1.0, rtol=0.0, atol=tolerance
    ):
        raise ValueError(f"{label}必须非负且24点和为1")


def _validate_adaptive_cells(frame: pd.DataFrame, *, label: str) -> None:
    """Validate workday plus either merged or split non-workday cells."""

    if frame[["industry", "season", "day_group"]].isna().any().any():
        raise ValueError(f"{label}的industry/season/day_group不能为空")
    invalid_seasons = sorted(set(frame.season.astype(str)).difference(SEASONS))
    if invalid_seasons:
        raise ValueError(f"{label}含未知季节：{invalid_seasons}")
    for industry, part in frame.groupby("industry", sort=False):
        for season in SEASONS:
            observed = set(
                part.loc[part.season.astype(str).eq(season), "day_group"]
                .map(canonical_day_type)
                .astype(str)
            )
            if observed not in (
                {WORKDAY, MERGED_NONWORKDAY},
                set(CANONICAL_DAY_GROUPS),
            ):
                raise ValueError(
                    f"{label}日型组合非法：{industry}/{season}/{sorted(observed)}"
                )


def _canonical_targets(day_group: str) -> tuple[str, ...]:
    value = canonical_day_type(day_group)
    if value == WORKDAY:
        return ("工作日",)
    if value == MERGED_NONWORKDAY:
        return ("周末", "节假日")
    if value == WEEKEND:
        return ("周末",)
    if value == HOLIDAY:
        return ("节假日",)
    raise ValueError(f"未知最终日型：{value}")


def build_base_year_industry_components(
    baseline_shapes: pd.DataFrame,
    industry_energy: pd.DataFrame,
    official_natural_total_energy: float,
) -> pd.DataFrame:
    """用官方自然负荷总量作分母，生成修正识别用的基准行业长表。"""

    _require_columns(
        baseline_shapes,
        {"industry", "broad_sector", *CELL_COLUMNS, *SHAPE_COLS},
        "行业基准形状",
    )
    _require_columns(industry_energy, {"industry", "official_energy"}, "基准年行业电量")
    if baseline_shapes.duplicated(["industry", *CELL_COLUMNS]).any():
        raise ValueError("行业基准形状industry/season/day_group必须唯一")
    _validate_adaptive_cells(baseline_shapes, label="行业基准形状")
    if industry_energy.industry.duplicated().any():
        raise ValueError("基准年行业电量industry必须唯一")
    total = float(official_natural_total_energy)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("基准年官方自然负荷电量必须有限且为正")
    energy = industry_energy[["industry", "official_energy"]].copy()
    energy["official_energy"] = pd.to_numeric(energy.official_energy, errors="raise")
    energy["share"] = energy.official_energy / total
    data = baseline_shapes.merge(energy[["industry", "share"]], on="industry", validate="many_to_one")
    if data.industry.nunique() != 24:
        raise ValueError("修正识别必须恰有24个传统细分行业")
    rows: list[dict[str, object]] = []
    for item in data.itertuples(index=False):
        values = np.asarray([getattr(item, column) for column in SHAPE_COLS], dtype=float)
        _validate_shape(values, f"{item.industry}/{item.season}/{item.day_group}")
        for canonical_day_group in _canonical_targets(str(item.day_group)):
            for hour, value in enumerate(values):
                rows.append(
                    {
                        "industry": str(item.industry),
                        "broad_sector": str(item.broad_sector),
                        "season": str(item.season),
                        "day_group": canonical_day_group,
                        "source_day_group": str(item.day_group),
                        "holiday_merged_with_weekend": canonical_day_type(
                            item.day_group
                        )
                        == MERGED_NONWORKDAY,
                        "hour": hour,
                        "share": float(item.share),
                        "normalized_load": float(value),
                        "weight_denominator_scope": NATURAL_LOAD_SCOPE,
                    }
                )
    result = pd.DataFrame(rows)
    key = ["industry", "season", "day_group", "hour"]
    expected_rows = sum(
        len(_canonical_targets(str(item.day_group))) * 24
        for item in data.itertuples(index=False)
    )
    if result.duplicated(key).any() or len(result) != expected_rows:
        raise AssertionError("修正识别行业形状展开后单元或行数不一致")
    return result


def build_future_industry_components(
    baseline_shapes: pd.DataFrame,
    future_parameters: pd.DataFrame,
    component_annual_energy: pd.DataFrame,
    *,
    horizon: int = 8,
    peak_valley_phi: float = 0.8,
) -> pd.DataFrame:
    """生成24行业未来长形状；8760形状装配只消费P50自然负荷权重。"""

    _require_columns(
        baseline_shapes,
        {"industry", "broad_sector", *CELL_COLUMNS, *SHAPE_COLS},
        "行业基准形状",
    )
    _require_columns(
        future_parameters,
        {
            "industry",
            *CELL_COLUMNS,
            "year",
            "H",
            "phi",
            "applied_share_night",
            "applied_share_morning",
            "applied_share_midday",
            "applied_share_evening",
            "peak_valley_target_range_limited",
            "peak_valley_evolution_enabled",
            "peak_valley_gate_reason",
        },
        "未来行业参数",
    )
    _require_columns(
        component_annual_energy,
        {
            *CONTEXT_COLUMNS,
            "industry",
            "component_type",
            "broad_sector",
            ANNUAL_ENERGY_QUANTILE_COLUMN,
            "electricity_share",
        },
        "组成年度电量",
    )
    base_key = ["industry", *CELL_COLUMNS]
    if baseline_shapes.duplicated(base_key).any():
        raise ValueError("行业基准形状单元重复")
    _validate_adaptive_cells(baseline_shapes, label="行业基准形状")
    formal = future_parameters.loc[
        pd.to_numeric(future_parameters.H).eq(int(horizon))
        & pd.to_numeric(future_parameters.phi).eq(float(peak_valley_phi))
    ].copy()
    if formal.empty or formal.duplicated(["industry", *CELL_COLUMNS, "year"]).any():
        raise ValueError(
            f"未来行业参数H={horizon}, phi={peak_valley_phi}为空或单元重复"
        )
    base = baseline_shapes.set_index(base_key)
    shape_rows: list[dict[str, object]] = []
    for trajectory in formal.itertuples(index=False):
        key = (str(trajectory.industry), str(trajectory.season), str(trajectory.day_group))
        if key not in base.index:
            raise ValueError(f"未来参数找不到基准形状：{key}")
        base_row = base.loc[key]
        if isinstance(base_row, pd.DataFrame):
            raise ValueError(f"行业基准形状单元重复：{key}")
        shape = shape_for_trajectory_row(
            base_row[SHAPE_COLS].to_numpy(dtype=float), trajectory._asdict()
        )
        _validate_shape(shape, f"{key}/{int(trajectory.year)}/H{horizon}")
        for canonical_day_group in _canonical_targets(key[2]):
            for hour, value in enumerate(shape):
                shape_rows.append(
                    {
                        "industry": key[0],
                        "season": key[1],
                        "day_group": canonical_day_group,
                        "source_day_group": key[2],
                        "holiday_merged_with_weekend": canonical_day_type(key[2])
                        == MERGED_NONWORKDAY,
                        "year": int(trajectory.year),
                        "hour": hour,
                        "normalized_load": float(value),
                        "H": int(horizon),
                        "phi": float(peak_valley_phi),
                    }
                )
    shapes = pd.DataFrame(shape_rows)
    fine = component_annual_energy.loc[
        component_annual_energy.component_type.eq("传统细分行业")
    ].copy()
    if fine.empty:
        raise ValueError("组成年度电量没有传统细分行业")
    fine[ANNUAL_ENERGY_QUANTILE_COLUMN] = (
        fine[ANNUAL_ENERGY_QUANTILE_COLUMN].astype(str).str.strip().str.upper()
    )
    invalid_quantiles = sorted(
        set(fine[ANNUAL_ENERGY_QUANTILE_COLUMN]).difference({"P10", "P50", "P90"})
    )
    if invalid_quantiles:
        raise ValueError(f"组成年度电量含未知分位：{invalid_quantiles}")
    fine = fine.loc[
        fine[ANNUAL_ENERGY_QUANTILE_COLUMN].eq(
            DETERMINISTIC_ANNUAL_ENERGY_QUANTILE
        )
    ].copy()
    if fine.empty:
        raise ValueError("组成年度电量缺少用于8760装配的P50自然负荷分位")
    if fine.duplicated([*CONTEXT_COLUMNS, "industry"]).any():
        raise ValueError("P50组成年度电量的情景/年份/行业必须唯一")
    counts = fine.groupby(list(CONTEXT_COLUMNS)).industry.nunique()
    if not counts.eq(24).all():
        raise ValueError("每个情景/年份必须恰有24个细分行业电量份额")
    output = fine[
        [
            *CONTEXT_COLUMNS,
            "industry",
            "broad_sector",
            ANNUAL_ENERGY_QUANTILE_COLUMN,
            "electricity_share",
        ]
    ].merge(shapes, on=["industry", "year"], how="left", validate="many_to_many")
    if output.normalized_load.isna().any():
        raise ValueError("未来行业形状与组成年度电量年份不匹配")
    output = output.rename(columns={"electricity_share": "share"})
    output["weight_denominator_scope"] = NATURAL_LOAD_SCOPE
    order = [
        *CONTEXT_COLUMNS,
        "industry",
        "broad_sector",
        ANNUAL_ENERGY_QUANTILE_COLUMN,
        *CELL_COLUMNS,
        "source_day_group",
        "holiday_merged_with_weekend",
        "hour",
        "share",
        "normalized_load",
        "H",
        "phi",
        "weight_denominator_scope",
    ]
    sort_columns = [*CONTEXT_COLUMNS, "industry", *CELL_COLUMNS, "hour"]
    return output[order].sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def build_proxy_sector_typical_shapes(
    future_industry_components: pd.DataFrame,
    *,
    context_columns: Sequence[str] = CONTEXT_COLUMNS,
) -> pd.DataFrame:
    """按内部行业自然负荷电量占比派生一二三产和居民代理曲线。"""

    context = list(context_columns)
    _require_columns(
        future_industry_components,
        {
            *context,
            "industry",
            "broad_sector",
            *CELL_COLUMNS,
            "hour",
            "share",
            "normalized_load",
        },
        "未来行业组成",
    )
    frame = future_industry_components.copy()
    frame["weighted_shape"] = frame.share * frame.normalized_load
    cell_keys = [*context, "broad_sector", *CELL_COLUMNS]
    unique_share = frame.drop_duplicates([*cell_keys, "industry"])
    sector_share = (
        unique_share.groupby(cell_keys, as_index=False, sort=False).share.sum()
        .rename(columns={"share": "proxy_sector_share"})
    )
    hourly = (
        frame.groupby([*cell_keys, "hour"], as_index=False, sort=False).weighted_shape.sum()
        .merge(sector_share, on=cell_keys, validate="many_to_one")
    )
    if (hourly.proxy_sector_share <= 0).any():
        raise ValueError("代理产业内部行业电量份额之和必须为正")
    hourly["proxy_normalized_load"] = (
        hourly.weighted_shape / hourly.proxy_sector_share
    )
    sums = hourly.groupby(cell_keys).proxy_normalized_load.sum()
    if not np.isclose(sums.to_numpy(float), 1.0, rtol=0, atol=1e-10).all():
        raise AssertionError("代理产业24点标幺曲线未闭合")
    hourly = hourly.rename(columns={"broad_sector": "proxy_sector"})
    hourly[ANNUAL_ENERGY_QUANTILE_COLUMN] = DETERMINISTIC_ANNUAL_ENERGY_QUANTILE
    hourly["weight_denominator_scope"] = NATURAL_LOAD_SCOPE
    hourly["role"] = "由内部细分行业按自然负荷电量占比加权派生；不作为参数或电量分配层"
    columns = [
        *context,
        "proxy_sector",
        ANNUAL_ENERGY_QUANTILE_COLUMN,
        *CELL_COLUMNS,
        "hour",
        "proxy_sector_share",
        "proxy_normalized_load",
        "weight_denominator_scope",
        "role",
    ]
    sort_columns = [*context, "proxy_sector", *CELL_COLUMNS, "hour"]
    return hourly[columns].sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


__all__ = [
    "ANNUAL_ENERGY_QUANTILE_COLUMN",
    "DETERMINISTIC_ANNUAL_ENERGY_QUANTILE",
    "NATURAL_LOAD_SCOPE",
    "build_base_year_industry_components",
    "build_future_industry_components",
    "build_proxy_sector_typical_shapes",
]
