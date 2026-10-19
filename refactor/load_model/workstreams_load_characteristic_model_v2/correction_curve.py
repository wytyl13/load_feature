"""季节×日型修正组成：24个传统行业之外的第25个电量组成。

本模块只处理标幺曲线与对应单元的电量份额，不再构造、缩放或映射8760点
绝对MW误差曲线。基准年从全省典型日中扣除24行业贡献，识别修正组成的
归一形状；未来年固定该形状，仅以未来24行业份额的余量重算修正组成份额。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np
import pandas as pd

from .config import NATURAL_LOAD_CURVE_ENERGY_SCOPE
from .day_types import HOLIDAY, MERGED_NONWORKDAY, WEEKEND, WORKDAY, canonical_day_type
from .quality import SHAPE_COLS

SEASONS = ("春季", "夏季", "秋季", "冬季")
DAY_GROUPS = ("工作日", "周末", "节假日")
LEGACY_DAY_GROUPS = ("工作日", "非工作日")
SUPPORTED_DAY_GROUPS = (*DAY_GROUPS, "非工作日")
HOURS = tuple(range(24))
EXPECTED_INDUSTRY_COUNT = 24
DEFAULT_TOLERANCE = 1e-12
DAY_ENERGY_FACTOR_STATUS = "fixed_1_not_identifiable_from_normalized_shapes"
NATURAL_LOAD_SCOPE = NATURAL_LOAD_CURVE_ENERGY_SCOPE
HISTORICAL_SHAPE_SCOPE = "dispatch_proxy_not_hourly_deducted"
CORRECTION_SHAPE_STATUS = "proxy_derived_from_dispatch_shape"
CORRECTION_SCOPE = (
    "自然负荷减24传统行业（三项排除电量不进入修正组成；"
    "新型负荷不另扣除并包含在修正项余量）"
)

CELL_COLUMNS = ["season", "day_group"]
HOUR_KEY = [*CELL_COLUMNS, "hour"]
PROVINCE_VALUE_COLUMN = "province_typical_share"
INDUSTRY_VALUE_COLUMN = "normalized_load"
CORRECTION_SHAPE_COLUMN = "correction_normalized_shape"


@dataclass(frozen=True)
class CorrectionArtifacts:
    """正式修正组成表及其摘要诊断；不含旧式8760 MW修正或缩放权重。"""

    component: pd.DataFrame
    diagnostics: dict[str, object]


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label}缺少字段：{sorted(missing)}")


def _prepare_dimensions(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    result = frame.copy()
    if result[CELL_COLUMNS].isna().any().any():
        raise ValueError(f"{label}的season/day_group不能为空")
    result["season"] = result.season.astype(str)
    result["day_group"] = result.day_group.astype(str)
    invalid_seasons = sorted(set(result.season).difference(SEASONS))
    invalid_day_groups = sorted(
        set(result.day_group).difference(SUPPORTED_DAY_GROUPS)
    )
    if invalid_seasons or invalid_day_groups:
        raise ValueError(
            f"{label}含未知季节或日型：season={invalid_seasons}, day_group={invalid_day_groups}"
        )
    numeric_hours = pd.to_numeric(result.hour, errors="coerce")
    if (
        numeric_hours.isna().any()
        or not np.isfinite(numeric_hours.to_numpy(float)).all()
        or not np.equal(numeric_hours, np.floor(numeric_hours)).all()
    ):
        raise ValueError(f"{label}的hour必须为0..23整数")
    result["hour"] = numeric_hours.astype(int)
    if not result.hour.between(0, 23).all():
        raise ValueError(f"{label}的hour必须为0..23整数")
    return result


def _as_finite_numeric(
    frame: pd.DataFrame,
    column: str,
    *,
    label: str,
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(float)).all():
        raise ValueError(f"{label}的{column}必须全部为有限数")
    return values.astype(float)


def _validate_profiles(
    frame: pd.DataFrame,
    profile_columns: Sequence[str],
    value_column: str,
    *,
    label: str,
    tolerance: float,
) -> None:
    profile_columns = list(profile_columns)
    counts = frame.groupby(profile_columns, sort=False).hour.nunique()
    if not counts.eq(24).all():
        first = counts.loc[~counts.eq(24)].index[0]
        raise ValueError(f"{label}的24点不完整：{profile_columns}={first}")
    sums = frame.groupby(profile_columns, sort=False)[value_column].sum()
    if not np.isclose(sums.to_numpy(float), 1.0, rtol=0.0, atol=tolerance).all():
        first = sums.loc[~np.isclose(sums, 1.0, rtol=0.0, atol=tolerance)].index[0]
        value = float(sums.loc[first])
        raise ValueError(f"{label}的24点和必须为1：{profile_columns}={first}, sum={value:.12g}")


def _validate_complete_grid(
    frame: pd.DataFrame,
    *,
    context_columns: Sequence[str] = (),
    entity_column: str | None = None,
    label: str,
) -> None:
    observed_day_groups = set(frame.day_group.astype(str))
    if "非工作日" in observed_day_groups:
        if not observed_day_groups.issubset(LEGACY_DAY_GROUPS):
            raise ValueError(f"{label}不得混用旧两日型与新三日型")
        day_groups = LEGACY_DAY_GROUPS
    else:
        day_groups = DAY_GROUPS
    expected = set(product(SEASONS, day_groups))
    grid_label = f"4季×{len(day_groups)}日型"
    unit_columns = [*context_columns]
    if entity_column is not None:
        unit_columns.append(entity_column)
    if not unit_columns:
        observed = set(frame[CELL_COLUMNS].drop_duplicates().itertuples(index=False, name=None))
        if observed != expected:
            raise ValueError(
                f"{label}必须覆盖{grid_label}，缺失={sorted(expected - observed)}"
            )
        return
    for keys, part in frame.groupby(unit_columns, sort=False, dropna=False):
        observed = set(part[CELL_COLUMNS].drop_duplicates().itertuples(index=False, name=None))
        if observed != expected:
            raise ValueError(
                f"{label}必须覆盖{grid_label}：{unit_columns}={keys}, "
                f"缺失={sorted(expected - observed)}"
            )


def _prepare_province(
    province_typical: pd.DataFrame,
    *,
    tolerance: float,
) -> pd.DataFrame:
    _require_columns(
        province_typical,
        {*HOUR_KEY, PROVINCE_VALUE_COLUMN},
        "全省季节日型典型曲线",
    )
    result = _prepare_dimensions(province_typical, label="全省季节日型典型曲线")
    if result.duplicated(HOUR_KEY).any():
        raise ValueError("全省季节日型典型曲线的season/day_group/hour必须唯一")
    result[PROVINCE_VALUE_COLUMN] = _as_finite_numeric(
        result, PROVINCE_VALUE_COLUMN, label="全省季节日型典型曲线"
    )
    if (result[PROVINCE_VALUE_COLUMN] < -tolerance).any():
        raise ValueError("全省季节日型典型曲线不能含负的标幺值")
    _validate_profiles(
        result,
        CELL_COLUMNS,
        PROVINCE_VALUE_COLUMN,
        label="全省季节日型典型曲线",
        tolerance=tolerance,
    )
    _validate_complete_grid(result, label="全省季节日型典型曲线")
    return result[HOUR_KEY + [PROVINCE_VALUE_COLUMN]].copy()


def _prepare_industries(
    industry_components: pd.DataFrame,
    *,
    context_columns: Sequence[str] = (),
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    context_columns = list(context_columns)
    required = {
        *context_columns,
        "industry",
        *HOUR_KEY,
        "share",
        INDUSTRY_VALUE_COLUMN,
    }
    _require_columns(industry_components, required, "24行业组成")
    result = _prepare_dimensions(industry_components, label="24行业组成")
    if result[[*context_columns, "industry"]].isna().any().any():
        raise ValueError("24行业组成的上下文字段和industry不能为空")
    result["industry"] = result.industry.astype(str)
    if (result.industry.str.strip() == "").any():
        raise ValueError("24行业组成的industry不能为空")
    key = [*context_columns, "industry", *HOUR_KEY]
    if result.duplicated(key).any():
        raise ValueError(f"24行业组成的{key}必须唯一")
    result["share"] = _as_finite_numeric(result, "share", label="24行业组成")
    result[INDUSTRY_VALUE_COLUMN] = _as_finite_numeric(
        result, INDUSTRY_VALUE_COLUMN, label="24行业组成"
    )
    if (result.share < 0.0).any():
        raise ValueError("24行业组成的share不能为负")
    if (result[INDUSTRY_VALUE_COLUMN] < -tolerance).any():
        raise ValueError("24行业组成的normalized_load不能含负值")

    profile_columns = [*context_columns, "industry", *CELL_COLUMNS]
    _validate_profiles(
        result,
        profile_columns,
        INDUSTRY_VALUE_COLUMN,
        label="24行业组成",
        tolerance=tolerance,
    )
    _validate_complete_grid(
        result,
        context_columns=context_columns,
        entity_column="industry",
        label="24行业组成",
    )

    identity_columns = [*context_columns, "industry", *CELL_COLUMNS]
    share_spread = result.groupby(identity_columns, sort=False).share.agg(
        lambda values: float(values.max() - values.min())
    )
    if (share_spread > tolerance).any():
        first = share_spread.loc[share_spread > tolerance].index[0]
        raise ValueError(f"同一行业季节日型的share必须在24小时保持不变：{first}")
    shares = result.groupby(identity_columns, as_index=False, sort=False).share.first()

    count_group = [*context_columns] if context_columns else None
    if count_group:
        industry_counts = shares.groupby(count_group, sort=False).industry.nunique()
        if not industry_counts.eq(EXPECTED_INDUSTRY_COUNT).all():
            first = industry_counts.loc[
                ~industry_counts.eq(EXPECTED_INDUSTRY_COUNT)
            ].index[0]
            raise ValueError(f"每个上下文必须恰有24个传统行业：context={first}")
    elif shares.industry.nunique() != EXPECTED_INDUSTRY_COUNT:
        raise ValueError(
            f"必须恰有24个传统行业，实际{shares.industry.nunique()}个"
        )

    weight_keys = [*context_columns, *CELL_COLUMNS]
    weight_sums = shares.groupby(weight_keys, as_index=False, sort=False).share.sum()
    weight_sums = weight_sums.rename(columns={"share": "industry_share_sum"})
    residual_share = 1.0 - weight_sums.industry_share_sum
    if (residual_share <= 0.0).any():
        first_index = residual_share.loc[residual_share <= 0.0].index[0]
        first = weight_sums.loc[first_index]
        cell = "/".join(str(first[column]) for column in weight_keys)
        raise ValueError(
            f"修正项电量份额必须为正：{cell}, omega_R={float(residual_share.loc[first_index]):.12g}"
        )
    return result, weight_sums


def _industry_hourly_contribution(
    industries: pd.DataFrame,
    *,
    context_columns: Sequence[str] = (),
) -> pd.DataFrame:
    result = industries.copy()
    result["industry_contribution_share"] = (
        result.share * result[INDUSTRY_VALUE_COLUMN]
    )
    keys = [*context_columns, *HOUR_KEY]
    return result.groupby(keys, as_index=False, sort=False).industry_contribution_share.sum()


def _sort_cells(frame: pd.DataFrame, context_columns: Sequence[str] = ()) -> pd.DataFrame:
    result = frame.copy()
    result["_season_order"] = result.season.map({value: index for index, value in enumerate(SEASONS)})
    result["_day_order"] = result.day_group.map(
        {"工作日": 0, "周末": 1, "节假日": 2, "非工作日": 1}
    )
    order = [*context_columns, "_season_order", "_day_order", "hour"]
    return (
        result.sort_values(order, kind="mergesort")
        .drop(columns=["_season_order", "_day_order"])
        .reset_index(drop=True)
    )


def _source_year_2023(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """从可能包含多年的表中严格取2023年基准。"""

    result = frame.copy()
    if "year" in result:
        year = pd.to_numeric(result.year, errors="coerce")
        if year.isna().any():
            raise ValueError(f"{label}的year必须为整数")
        result = result.loc[year.eq(2023)].copy()
    elif "source_year" in result:
        source_year = pd.to_numeric(result.source_year, errors="coerce")
        if source_year.isna().any():
            raise ValueError(f"{label}的source_year必须为整数")
        result = result.loc[source_year.eq(2023)].copy()
    elif "weight_year" in result:
        weight_year = pd.to_numeric(result.weight_year, errors="coerce")
        if weight_year.isna().any():
            raise ValueError(f"{label}的weight_year必须为整数")
        result = result.loc[weight_year.eq(2023)].copy()
    elif "is_future_baseline" in result:
        result = result.loc[result.is_future_baseline.astype(bool)].copy()
    if result.empty:
        raise ValueError(f"{label}缺少2023年基准")
    return result


def _day_type_column(frame: pd.DataFrame, *, label: str) -> str:
    for candidate in ("final_day_type", "day_type", "day_group"):
        if candidate in frame:
            return candidate
    raise ValueError(f"{label}缺少final_day_type/day_type/day_group")


def _province_proxy_for_correction(
    province_shape_2023: pd.DataFrame,
    *,
    tolerance: float,
) -> pd.DataFrame:
    province = _source_year_2023(province_shape_2023, label="2023全省形状代理")
    day_column = _day_type_column(province, label="2023全省形状代理")
    value_column = next(
        (
            column
            for column in (
                "province_shape_proxy",
                PROVINCE_VALUE_COLUMN,
                "normalized_load",
            )
            if column in province
        ),
        None,
    )
    if value_column is None:
        raise ValueError("全省2023形状代理缺少标幺值字段")
    _require_columns(province, {"season", "hour", day_column, value_column}, "2023全省形状代理")
    province["_day_type"] = province[day_column].map(canonical_day_type)
    province["_value"] = _as_finite_numeric(
        province, value_column, label="2023全省形状代理"
    )
    province["hour"] = pd.to_numeric(province.hour, errors="raise").astype(int)
    if (province._value < -tolerance).any():
        raise ValueError("全省2023形状代理不得为负")

    rows: list[dict[str, object]] = []
    for season in SEASONS:
        season_part = province.loc[province.season.astype(str).eq(season)]
        if season_part.empty:
            raise ValueError(f"全省2023形状代理缺少{season}")
        for target, output_group in (
            (WORKDAY, "工作日"),
            (WEEKEND, "周末"),
            (HOLIDAY, "节假日"),
        ):
            selected = season_part.loc[season_part._day_type.eq(target)]
            if selected.empty and target in {WEEKEND, HOLIDAY}:
                selected = season_part.loc[
                    season_part._day_type.eq(MERGED_NONWORKDAY)
                ]
            if selected.empty:
                raise ValueError(
                    f"全省2023形状代理缺少{season}/{target}"
                )
            if selected.hour.duplicated().any() or set(selected.hour) != set(HOURS):
                raise ValueError(
                    f"全省2023形状代理的24点不完整：{season}/{target}"
                )
            for _, row in selected.sort_values("hour").iterrows():
                rows.append(
                    {
                        "season": season,
                        "day_group": output_group,
                        "hour": int(row["hour"]),
                        PROVINCE_VALUE_COLUMN: float(row["_value"]),
                    }
                )
    return pd.DataFrame.from_records(rows)


def _industry_medoids_for_correction(
    industry_baseline_2023: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    tolerance: float,
) -> pd.DataFrame:
    baseline = _source_year_2023(
        industry_baseline_2023, label="2023行业聚合medoid"
    )
    if "is_future_baseline" in baseline and not baseline.is_future_baseline.astype(bool).all():
        raise ValueError("2023行业聚合medoid必须标记is_future_baseline=True")
    day_column = _day_type_column(baseline, label="2023行业聚合medoid")
    _require_columns(
        baseline,
        {"industry", "season", day_column, *SHAPE_COLS},
        "2023行业聚合medoid",
    )
    baseline["industry"] = baseline.industry.astype(str)
    baseline["_day_type"] = baseline[day_column].map(canonical_day_type)
    shape_values = baseline[SHAPE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(shape_values).all() or np.any(shape_values < -tolerance):
        raise ValueError("2023行业聚合medoid形状必须非负且有限")
    shape_sums = shape_values.sum(axis=1)
    if not np.isclose(shape_sums, 1.0, rtol=0.0, atol=tolerance).all():
        raise ValueError("2023行业聚合medoid每条24点和必须为1")

    expected_industries = set(weights.industry.astype(str))
    actual_industries = set(baseline.industry)
    if actual_industries != expected_industries:
        raise ValueError(
            "2023行业聚合medoid与修正识别权重行业集合不一致："
            f"missing={sorted(expected_industries - actual_industries)}, "
            f"extra={sorted(actual_industries - expected_industries)}"
        )

    rows: list[dict[str, object]] = []
    weight_lookup = weights.set_index("industry").share
    for industry in sorted(expected_industries):
        industry_part = baseline.loc[baseline.industry.eq(industry)]
        for season in SEASONS:
            season_part = industry_part.loc[industry_part.season.astype(str).eq(season)]
            workday = season_part.loc[season_part._day_type.eq(WORKDAY)]
            if len(workday) != 1:
                raise RuntimeError(
                    f"2023行业{industry}/{season}工作日基准缺失或不唯一"
                )
            for target, output_group in (
                (WORKDAY, "工作日"),
                (WEEKEND, "周末"),
                (HOLIDAY, "节假日"),
            ):
                selected = season_part.loc[season_part._day_type.eq(target)]
                if selected.empty and target in {WEEKEND, HOLIDAY}:
                    selected = season_part.loc[
                        season_part._day_type.eq(MERGED_NONWORKDAY)
                    ]
                if selected.empty:
                    selected = workday
                if len(selected) != 1:
                    raise ValueError(
                        f"2023行业聚合medoid单元不唯一：{industry}/{season}/{target}"
                    )
                values = selected.iloc[0][SHAPE_COLS].to_numpy(float)
                for hour, value in enumerate(values):
                    rows.append(
                        {
                            "industry": industry,
                            "season": season,
                            "day_group": output_group,
                            "hour": hour,
                            "share": float(weight_lookup.loc[industry]),
                            INDUSTRY_VALUE_COLUMN: float(value),
                        }
                    )
    return pd.DataFrame.from_records(rows)


def _correction_weights_2023(
    correction_weights_2023: pd.DataFrame,
) -> pd.DataFrame:
    weights = _source_year_2023(
        correction_weights_2023, label="2023修正组成识别权重"
    )
    _require_columns(weights, {"industry"}, "2023修正组成识别权重")
    share_column = next(
        (
            column
            for column in ("share", "electricity_share", "correction_weight", "weight")
            if column in weights
        ),
        None,
    )
    if share_column is None:
        raise ValueError(
            "2023修正组成识别权重缺少share/electricity_share字段"
        )
    result = weights[["industry", share_column]].rename(columns={share_column: "share"}).copy()
    result["industry"] = result.industry.astype(str).str.strip()
    result["share"] = pd.to_numeric(result.share, errors="coerce")
    if (
        len(result) != EXPECTED_INDUSTRY_COUNT
        or result.industry.eq("").any()
        or result.industry.duplicated().any()
    ):
        raise ValueError("修正组成识别权重必须恰好覆盖24个唯一行业")
    if result.share.isna().any() or not np.isfinite(result.share).all() or (result.share < 0).any():
        raise ValueError("修正组成识别权重必须为有限非负数")
    if float(result.share.sum()) >= 1.0:
        raise ValueError("修正组成权重必须为正：24行业权重合计不得达到1")
    return result


def build_correction_component_shape(
    province_shape_2023: pd.DataFrame,
    industry_baseline_2023: pd.DataFrame,
    correction_weights_2023: pd.DataFrame,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """用2023行业聚合medoid和自然负荷权重重新反演第25组成。

    行业权重不作内部归一化。行业或全省采用合并非工作日时，同一条
    形状同时映射到普通周末与法定节假日规范网格。
    """

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance必须为有限正数")
    weights = _correction_weights_2023(correction_weights_2023)
    province = _province_proxy_for_correction(
        province_shape_2023, tolerance=tolerance
    )
    industries = _industry_medoids_for_correction(
        industry_baseline_2023, weights, tolerance=tolerance
    )
    shape = build_season_daytype_correction_component(
        province, industries, tolerance=tolerance
    ).copy()
    # 日型电量倍率由全省2023共享系数在日分配阶段独立提供。
    # 新口径的修正组成形状不再携带“固定为1”的伪倍率。
    shape = shape.drop(
        columns=["correction_day_energy_factor", "day_energy_factor_status"],
        errors="ignore",
    )
    shape["industry_shape_source"] = "2023_industry_aggregate_medoid"
    shape["province_shape_year"] = 2023
    shape["industry_shape_year"] = 2023
    shape["correction_weight_year"] = 2023
    shape["final_day_type"] = shape.day_group.map(
        {"工作日": WORKDAY, "周末": WEEKEND, "节假日": HOLIDAY}
    )

    diagnostics = (
        shape.groupby(CELL_COLUMNS, as_index=False, sort=False)
        .agg(
            province_shape_sum=("province_shape_proxy", "sum"),
            C24_sum=("C24_contribution", "sum"),
            R_sum=("R_contribution", "sum"),
            R_min=("R_contribution", "min"),
            omega_R=("omega_R", "first"),
            correction_shape_sum=("correction_component_shape_proxy", "sum"),
            correction_shape_min=("correction_component_shape_proxy", "min"),
            reconstruction_max_abs_error=("reconstruction_max_abs_error", "max"),
        )
    )
    diagnostics["R_sum_error"] = np.abs(diagnostics.R_sum - diagnostics.omega_R)
    diagnostics["correction_shape_sum_error"] = np.abs(
        diagnostics.correction_shape_sum - 1.0
    )
    diagnostics["passed"] = (
        diagnostics.R_min.ge(-tolerance)
        & diagnostics.correction_shape_min.ge(-tolerance)
        & diagnostics.R_sum_error.le(tolerance)
        & diagnostics.correction_shape_sum_error.le(tolerance)
        & diagnostics.reconstruction_max_abs_error.le(tolerance)
    )
    diagnostics["industry_shape_source"] = "2023_industry_aggregate_medoid"
    diagnostics["historical_shape_scope"] = HISTORICAL_SHAPE_SCOPE
    diagnostics["correction_shape_status"] = CORRECTION_SHAPE_STATUS
    diagnostics["correction_weight_year"] = 2023
    if not diagnostics.passed.all():
        raise AssertionError("修正组成基准期复原或归一化校验失败")
    return _sort_cells(shape), _sort_cells(diagnostics.assign(hour=0)).drop(columns="hour")


def build_season_daytype_correction_component(
    province_typical: pd.DataFrame,
    industry_components: pd.DataFrame,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> pd.DataFrame:
    """识别基准年第25个电量组成。

    正式输入为4季×3规范日型；为兼容旧调用方，完整的
    ``工作日/非工作日`` 4季×2日型网格亦可原样计算与返回。

    输入约定：全省与每个行业的24点标幺曲线分别和为1；``share`` 是该
    季节×日型单元内行业电量占比，并在24小时保持不变。函数计算
    ``C24=Σ omega_i p_i``、``omega_R=1-Σomega_i``、
    ``R=p_province-C24``、``p_R=R/omega_R``。若任一 ``p_R`` 小于
    ``-tolerance``，立即报出季节、日型与小时，不做截断。

    仅凭这些标幺输入不能识别工作/非工作日总电量倍率，因此正式输出将
    ``correction_day_energy_factor`` 固定为1，并显式记录诊断状态。
    """

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance必须为有限正数")
    province = _prepare_province(province_typical, tolerance=tolerance)
    industries, weight_sums = _prepare_industries(
        industry_components, tolerance=tolerance
    )
    contribution = _industry_hourly_contribution(industries)
    result = province.merge(contribution, on=HOUR_KEY, how="left", validate="one_to_one")
    result = result.merge(weight_sums, on=CELL_COLUMNS, how="left", validate="many_to_one")
    if result[["industry_contribution_share", "industry_share_sum"]].isna().any().any():
        raise ValueError("全省典型曲线与24行业组成的季节×日型单元不一致")

    result["correction_component_share"] = 1.0 - result.industry_share_sum
    result["correction_contribution_share"] = (
        result[PROVINCE_VALUE_COLUMN] - result.industry_contribution_share
    )
    result[CORRECTION_SHAPE_COLUMN] = (
        result.correction_contribution_share / result.correction_component_share
    )

    bad = result.loc[result[CORRECTION_SHAPE_COLUMN] < -tolerance]
    if not bad.empty:
        row = _sort_cells(bad).iloc[0]
        raise ValueError(
            "修正组成归一形状出现超过容差的负值："
            f"season={row.season}/day_group={row.day_group}/hour={int(row.hour)}, "
            f"p_R={float(row[CORRECTION_SHAPE_COLUMN]):.12g}"
        )

    result["reconstructed_share"] = (
        result.industry_contribution_share + result.correction_contribution_share
    )
    result["_reconstruction_abs_error"] = np.abs(
        result.reconstructed_share - result[PROVINCE_VALUE_COLUMN]
    )
    cell_diagnostics = result.groupby(CELL_COLUMNS, as_index=False, sort=False).agg(
        province_shape_sum=(PROVINCE_VALUE_COLUMN, "sum"),
        industry_contribution_sum=("industry_contribution_share", "sum"),
        correction_contribution_sum=("correction_contribution_share", "sum"),
        correction_shape_sum=(CORRECTION_SHAPE_COLUMN, "sum"),
        correction_shape_min=(CORRECTION_SHAPE_COLUMN, "min"),
        reconstruction_max_abs_error=("_reconstruction_abs_error", "max"),
    )
    if not np.isclose(
        cell_diagnostics.correction_shape_sum.to_numpy(float),
        1.0,
        rtol=0.0,
        atol=tolerance,
    ).all():
        bad_cell = cell_diagnostics.loc[
            ~np.isclose(
                cell_diagnostics.correction_shape_sum,
                1.0,
                rtol=0.0,
                atol=tolerance,
            )
        ].iloc[0]
        raise ValueError(
            "修正项24点归一形状不闭合："
            f"{bad_cell.season}/{bad_cell.day_group}, "
            f"sum={float(bad_cell.correction_shape_sum):.12g}"
        )
    if (cell_diagnostics.reconstruction_max_abs_error > tolerance).any():
        raise AssertionError("24行业与修正项未能重构全省典型日")

    result = result.drop(columns=["_reconstruction_abs_error"]).merge(
        cell_diagnostics,
        on=CELL_COLUMNS,
        how="left",
        validate="many_to_one",
    )
    result["component"] = "correction_component"
    result["correction_scope"] = CORRECTION_SCOPE
    result["curve_energy_scope"] = NATURAL_LOAD_SCOPE
    result["historical_shape_scope"] = HISTORICAL_SHAPE_SCOPE
    result["correction_shape_status"] = CORRECTION_SHAPE_STATUS
    result["annual_special_deduction_applied"] = True
    result["hourly_special_deduction_applied"] = False
    result["special_items_in_correction_component"] = False
    # 保留内部计算列，同时提供方法文档中的显式“代理”字段名。
    result["province_shape_proxy"] = result[PROVINCE_VALUE_COLUMN]
    result["C24_contribution"] = result.industry_contribution_share
    result["R_contribution"] = result.correction_contribution_share
    result["correction_component_shape_proxy"] = result[
        CORRECTION_SHAPE_COLUMN
    ]
    result["omega_R"] = result.correction_component_share
    result["correction_day_energy_factor"] = 1.0
    result["day_energy_factor_status"] = DAY_ENERGY_FACTOR_STATUS
    result["industry_count"] = EXPECTED_INDUSTRY_COUNT
    ordered = [
        *HOUR_KEY,
        "component",
        "correction_scope",
        "curve_energy_scope",
        "historical_shape_scope",
        "correction_shape_status",
        "annual_special_deduction_applied",
        "hourly_special_deduction_applied",
        "special_items_in_correction_component",
        PROVINCE_VALUE_COLUMN,
        "province_shape_proxy",
        "industry_contribution_share",
        "C24_contribution",
        "correction_contribution_share",
        "R_contribution",
        "industry_share_sum",
        "correction_component_share",
        "omega_R",
        CORRECTION_SHAPE_COLUMN,
        "correction_component_shape_proxy",
        "correction_day_energy_factor",
        "day_energy_factor_status",
        "reconstructed_share",
        "province_shape_sum",
        "industry_contribution_sum",
        "correction_contribution_sum",
        "correction_shape_sum",
        "correction_shape_min",
        "reconstruction_max_abs_error",
        "industry_count",
    ]
    return _sort_cells(result)[ordered]


def _prepare_fixed_correction(
    correction_component: pd.DataFrame,
    *,
    tolerance: float,
) -> pd.DataFrame:
    _require_columns(
        correction_component,
        {*HOUR_KEY, CORRECTION_SHAPE_COLUMN},
        "固定修正组成形状",
    )
    result = _prepare_dimensions(correction_component, label="固定修正组成形状")
    if result.duplicated(HOUR_KEY).any():
        raise ValueError("固定修正组成形状的season/day_group/hour必须唯一")
    if "component" in result and set(result.component.astype(str)) != {
        "correction_component"
    }:
        raise ValueError("固定修正组成形状的component必须为correction_component")
    if "special_items_in_correction_component" in result:
        special_items_present = result[
            "special_items_in_correction_component"
        ].astype(bool)
        if special_items_present.any():
            raise ValueError("厂用电、线损和抽蓄用电不得进入固定修正组成")
    if "historical_shape_scope" in result and set(
        result.historical_shape_scope.astype(str)
    ) != {HISTORICAL_SHAPE_SCOPE}:
        raise ValueError("固定修正组成必须声明调度形状代理范围")
    if "correction_shape_status" in result and set(
        result.correction_shape_status.astype(str)
    ) != {CORRECTION_SHAPE_STATUS}:
        raise ValueError("固定修正组成必须声明由调度形状代理反演")
    result[CORRECTION_SHAPE_COLUMN] = _as_finite_numeric(
        result, CORRECTION_SHAPE_COLUMN, label="固定修正组成形状"
    )
    if (result[CORRECTION_SHAPE_COLUMN] < -tolerance).any():
        raise ValueError("固定修正组成形状不能含超过容差的负值")
    _validate_profiles(
        result,
        CELL_COLUMNS,
        CORRECTION_SHAPE_COLUMN,
        label="固定修正组成形状",
        tolerance=tolerance,
    )
    _validate_complete_grid(result, label="固定修正组成形状")
    columns = [*HOUR_KEY, CORRECTION_SHAPE_COLUMN]
    for optional in ("correction_day_energy_factor", "day_energy_factor_status"):
        if optional in result:
            columns.append(optional)
    return result[columns].copy()


def compose_with_fixed_correction_shape(
    industry_components: pd.DataFrame,
    correction_component: pd.DataFrame,
    *,
    context_columns: Sequence[str] = (),
    tolerance: float = DEFAULT_TOLERANCE,
) -> pd.DataFrame:
    """以冻结的 ``p_R`` 和未来余量 ``1-Σomega_i`` 合成总标幺曲线。

    ``context_columns`` 可传入 ``("method", "planning_scenario", "year")``
    等标识，一次装配多个未来单元。基准年的 ``correction_component_share``
    不会沿用；未来份额始终由当期24行业份额重新计算。
    """

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance必须为有限正数")
    context_columns = list(context_columns)
    industries, weight_sums = _prepare_industries(
        industry_components,
        context_columns=context_columns,
        tolerance=tolerance,
    )
    fixed = _prepare_fixed_correction(correction_component, tolerance=tolerance)
    contribution = _industry_hourly_contribution(
        industries, context_columns=context_columns
    )
    keys = [*context_columns, *HOUR_KEY]
    result = contribution.merge(fixed, on=HOUR_KEY, how="left", validate="many_to_one")
    result = result.merge(
        weight_sums,
        on=[*context_columns, *CELL_COLUMNS],
        how="left",
        validate="many_to_one",
    )
    if result[CORRECTION_SHAPE_COLUMN].isna().any():
        raise ValueError("未来24行业组成与固定修正形状的季节×日型单元不一致")
    result["correction_component_share"] = 1.0 - result.industry_share_sum
    result["correction_contribution_share"] = (
        result.correction_component_share * result[CORRECTION_SHAPE_COLUMN]
    )
    result["total_normalized_load"] = (
        result.industry_contribution_share + result.correction_contribution_share
    )
    if (result.total_normalized_load < -tolerance).any():
        raise ValueError("24行业与固定修正组成合成后出现负负荷")

    cell_keys = [*context_columns, *CELL_COLUMNS]
    closure = result.groupby(cell_keys, as_index=False, sort=False).agg(
        total_shape_sum=("total_normalized_load", "sum"),
        correction_shape_sum=(CORRECTION_SHAPE_COLUMN, "sum"),
        correction_shape_min=(CORRECTION_SHAPE_COLUMN, "min"),
    )
    if not np.isclose(
        closure.total_shape_sum.to_numpy(float),
        1.0,
        rtol=0.0,
        atol=tolerance,
    ).all():
        raise AssertionError("未来24行业与修正组成的季节日型电量不闭合")
    result = result.merge(
        closure,
        on=cell_keys,
        how="left",
        validate="many_to_one",
    )
    result["component"] = "correction_component"
    result["correction_scope"] = CORRECTION_SCOPE
    result["curve_energy_scope"] = NATURAL_LOAD_SCOPE
    result["historical_shape_scope"] = HISTORICAL_SHAPE_SCOPE
    result["correction_shape_status"] = CORRECTION_SHAPE_STATUS
    result["annual_special_deduction_applied"] = True
    result["hourly_special_deduction_applied"] = False
    result["special_items_in_correction_component"] = False
    result["C24_contribution"] = result.industry_contribution_share
    result["R_contribution"] = result.correction_contribution_share
    result["correction_component_shape_proxy"] = result[
        CORRECTION_SHAPE_COLUMN
    ]
    result["omega_R"] = result.correction_component_share
    result["industry_count"] = EXPECTED_INDUSTRY_COUNT
    ordered = [
        *keys,
        "component",
        "correction_scope",
        "curve_energy_scope",
        "historical_shape_scope",
        "correction_shape_status",
        "annual_special_deduction_applied",
        "hourly_special_deduction_applied",
        "special_items_in_correction_component",
        "industry_contribution_share",
        "C24_contribution",
        "industry_share_sum",
        "correction_component_share",
        "omega_R",
        CORRECTION_SHAPE_COLUMN,
        "correction_component_shape_proxy",
        "correction_contribution_share",
        "R_contribution",
        "total_normalized_load",
        "total_shape_sum",
        "correction_shape_sum",
        "correction_shape_min",
        "industry_count",
    ]
    for optional in ("correction_day_energy_factor", "day_energy_factor_status"):
        if optional in result:
            ordered.insert(-4, optional)
    return _sort_cells(result, context_columns=context_columns)[ordered]


def build_correction_artifacts(
    province_typical: pd.DataFrame,
    industry_components: pd.DataFrame,
    correction_weights_2023: pd.DataFrame | None = None,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> CorrectionArtifacts:
    """构建正式修正组成产物。

    传入 ``correction_weights_2023`` 时启用2023行业聚合medoid新接口；
    两参数调用仅保留给已组装好的旧long-form输入。
    """

    if correction_weights_2023 is None:
        component = build_season_daytype_correction_component(
            province_typical,
            industry_components,
            tolerance=tolerance,
        )
        diagnostic_frame: pd.DataFrame | None = None
    else:
        component, diagnostic_frame = build_correction_component_shape(
            province_typical,
            industry_components,
            correction_weights_2023,
            tolerance=tolerance,
        )
    diagnostics = {
        "component": "correction_component",
        "row_count": int(len(component)),
        "cell_count": int(component[CELL_COLUMNS].drop_duplicates().shape[0]),
        "industry_count": EXPECTED_INDUSTRY_COUNT,
        "max_reconstruction_abs_error": float(
            component.reconstruction_max_abs_error.max()
        ),
        "minimum_correction_normalized_shape": float(
            component.correction_shape_min.min()
        ),
        "day_energy_factor_status": (
            DAY_ENERGY_FACTOR_STATUS
            if correction_weights_2023 is None
            else "province_shared_all_25_components"
        ),
        "legacy_absolute_mw_correction": False,
        "new_load_deducted_before_residual": False,
        "special_items_in_correction_component": False,
        "annual_special_deduction_applied": True,
        "hourly_special_deduction_applied": False,
        "historical_shape_scope": HISTORICAL_SHAPE_SCOPE,
        "correction_shape_status": CORRECTION_SHAPE_STATUS,
        "curve_energy_scope": NATURAL_LOAD_SCOPE,
        "correction_scope": CORRECTION_SCOPE,
        "industry_shape_source": (
            "legacy_preassembled_industry_components"
            if correction_weights_2023 is None
            else "2023_industry_aggregate_medoid"
        ),
    }
    if diagnostic_frame is not None:
        diagnostics["all_cells_passed"] = bool(diagnostic_frame.passed.all())
    return CorrectionArtifacts(component=component, diagnostics=diagnostics)


__all__ = [
    "CorrectionArtifacts",
    "CORRECTION_SHAPE_STATUS",
    "CORRECTION_SCOPE",
    "DAY_ENERGY_FACTOR_STATUS",
    "HISTORICAL_SHAPE_SCOPE",
    "NATURAL_LOAD_SCOPE",
    "build_correction_artifacts",
    "build_correction_component_shape",
    "build_season_daytype_correction_component",
    "compose_with_fixed_correction_shape",
]
