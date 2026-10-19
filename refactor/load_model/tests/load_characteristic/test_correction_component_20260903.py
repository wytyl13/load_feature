from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from workstreams_load_characteristic_model_v2.correction_curve import (
    build_correction_component_shape,
)
from workstreams_load_characteristic_model_v2.quality import SHAPE_COLS


SEASONS = ("春季", "夏季", "秋季", "冬季")
DAY_TYPES = ("工作日", "普通周末", "法定节假日")


def _positive_case() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    industry_shape = np.arange(1.0, 25.0)
    industry_shape /= industry_shape.sum()
    correction_shape = industry_shape[::-1].copy()
    industry_share = 0.60
    province_shape = industry_share * industry_shape + (1.0 - industry_share) * correction_shape

    province = pd.DataFrame(
        [
            {
                "season": season,
                "day_type": day_type,
                "hour": hour,
                "province_shape_proxy": value,
            }
            for season in SEASONS
            for day_type in DAY_TYPES
            for hour, value in enumerate(province_shape)
        ]
    )
    industries = pd.DataFrame(
        [
            {
                "industry": f"行业{index + 1:02d}",
                "year": 2023,
                "season": season,
                "final_day_type": day_type,
                "is_future_baseline": True,
                **dict(zip(SHAPE_COLS, industry_shape, strict=True)),
            }
            for index in range(24)
            for season in SEASONS
            for day_type in DAY_TYPES
        ]
    )
    weights = pd.DataFrame(
        {
            "industry": [f"行业{index + 1:02d}" for index in range(24)],
            "share": industry_share / 24.0,
            "weight_year": 2023,
        }
    )
    return province, industries, weights


def test_correction_component_reconstructs_province_proxy() -> None:
    province, industries, weights = _positive_case()

    shape, diagnostics = build_correction_component_shape(
        province, industries, weights
    )
    reconstructed = (
        shape.C24_contribution.to_numpy(float)
        + shape.omega_R.to_numpy(float)
        * shape.correction_component_shape_proxy.to_numpy(float)
    )

    assert np.allclose(reconstructed, shape.province_shape_proxy, atol=1e-12)
    assert shape.R_contribution.min() >= -1e-12
    assert diagnostics.passed.all()
    assert set(shape.industry_shape_source) == {
        "2023_industry_aggregate_medoid"
    }
    assert set(shape.correction_weight_year) == {2023}
    assert np.allclose(shape.omega_R, 0.4)
    assert "correction_day_energy_factor" not in shape
    assert "day_energy_factor_status" not in shape


def test_negative_correction_residual_blocks_instead_of_clipping() -> None:
    province, industries, weights = _positive_case()
    target = (
        province.season.eq("春季")
        & province.day_type.eq("工作日")
        & province.hour.eq(0)
    )
    next_hour = (
        province.season.eq("春季")
        & province.day_type.eq("工作日")
        & province.hour.eq(1)
    )
    province.loc[target, "province_shape_proxy"] = 0.0001
    difference = 1.0 - province.loc[
        province.season.eq("春季") & province.day_type.eq("工作日"),
        "province_shape_proxy",
    ].sum()
    province.loc[next_hour, "province_shape_proxy"] += difference

    with pytest.raises(ValueError, match="修正组成.*负值"):
        build_correction_component_shape(province, industries, weights)
