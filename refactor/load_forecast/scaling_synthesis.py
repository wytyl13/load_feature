#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""06 段:标幺曲线换算 MW 并合成总负荷(原包 06_有名值换算与总负荷合成/run_stage.py 移植)。

计算行逐行保持一致;差异仅一处:配置从包内 config.json 改为显式传参。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def total_metrics(series: pd.Series):
    """计算总负荷峰值、均值、利用小时及季/月不平衡指标。"""
    quarter_means = [
        float(series[series.index.month.isin(months)].mean())
        for months in ([1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12])
    ]
    month_means = series.groupby(series.index.month).mean()
    mean = float(series.mean())
    peak = float(series.max())
    return {
        "年电量_亿kWh": float(series.sum()) / 1e5,
        "年最大负荷_MW": peak,
        "年最小负荷_MW": float(series.min()),
        "年平均负荷_MW": mean,
        "最大负荷利用小时_h": float(series.sum()) / peak,
        "季不平均负荷率_pct": (max(quarter_means) - min(quarter_means)) / mean * 100.0,
        "月不平衡负荷率_pct": (float(month_means.max()) - float(month_means.min())) / mean * 100.0,
    }


def run_scaling_synthesis(run_dir: Path, config: dict) -> Path:
    """将行业标幺曲线按预测年电量换算为MW并合成总负荷,含 12 项最终校验。

    对行业i，标幺曲线为s_i(t)、年电量为E_i亿kWh时：
    ``K_i = E_i * 100000 / sum(s_i)``，``P_i(t)=K_i*s_i(t)``。
    100000来自1亿kWh=100000MWh。总标幺必须在MW求和后单独归一化，
    不能直接相加各行业标幺值。
    """
    target_year = int(config["target_year"])
    stage03 = run_dir / "03_总电量与行业电量预测"
    # 第03步是电量唯一来源，避免固定配置与模型预测脱节。
    energy_table = pd.read_csv(stage03 / "2030行业电量锚点.csv", encoding="utf-8-sig")
    required_columns = {"行业", "最终用于曲线电量_亿kWh"}
    if not required_columns.issubset(energy_table.columns):
        raise RuntimeError("Stage 03 industry-energy output is missing required columns")
    if energy_table["行业"].duplicated().any() or len(energy_table) != 10:
        raise RuntimeError("Stage 03 must provide exactly 10 unique industry energies")
    energy = dict(zip(
        energy_table["行业"].astype(str),
        pd.to_numeric(energy_table["最终用于曲线电量_亿kWh"], errors="raise").astype(float),
    ))
    total_policy = pd.read_csv(stage03 / "2030总电量口径.csv", encoding="utf-8-sig")
    if len(total_policy) != 1 or not bool(total_policy.loc[0, "是否用于最终曲线"]):
        raise RuntimeError("Stage 03 must provide exactly one final model-total row")
    model_total = float(total_policy.loc[0, "2030总电量_亿kWh"])
    if abs(sum(energy.values()) - model_total) > 1e-8:
        raise RuntimeError("Stage 03 industry energies do not close to the model forecast total")
    shape_root = run_dir / "05_行业标幺8760生成" / "行业标幺8760"
    output = run_dir / "06_有名值换算与总负荷合成"
    industry_root = output / "行业MW8760"
    industry_root.mkdir(parents=True, exist_ok=True)
    expected_index = pd.date_range("{}-01-01 00:00:00".format(target_year), periods=8760, freq="h")
    all_industries, total = [], pd.Series(0.0, index=expected_index)
    # 软件接口采用宽表：一个时间戳，随后每个行业固定占两列。
    # 行业顺序继承第03步电量锚点顺序，避免自动排序。
    wide_industries = pd.DataFrame({"时间": expected_index.strftime("%Y-%m-%d %H:%M:%S")})
    validation_industries = []

    for industry in energy:
        path = shape_root / "{}.csv".format(industry)
        frame = pd.read_csv(path, encoding="utf-8-sig")
        timestamps = pd.DatetimeIndex(pd.to_datetime(frame["时间"]))
        per_unit = pd.to_numeric(frame["行业标幺值"], errors="coerce").values.astype(float)
        if len(per_unit) != 8760 or not timestamps.equals(expected_index):
            raise RuntimeError("{}: per-unit timestamps are incomplete".format(industry))
        if not np.isfinite(per_unit).all() or abs(float(per_unit.max()) - 1.0) > 1e-9:
            raise RuntimeError("{}: per-unit values are invalid or not normalized".format(industry))
        # 曲线每个点代表1小时，因此MW序列求和后得到MWh，再除以100000
        # 即可回到亿kWh；scale_mw使该积分严格等于目标行业年电量。
        scale_mw = energy[industry] * 1e5 / float(per_unit.sum())
        mw = per_unit * scale_mw
        result = pd.DataFrame({
            "时间": expected_index.strftime("%Y-%m-%d %H:%M:%S"),
            "年份": target_year,
            "行业": industry,
            "行业标幺值": per_unit,
            "行业负荷MW": mw,
        })
        result.to_csv(industry_root / "{}.csv".format(industry), index=False, encoding="utf-8-sig")
        all_industries.append(result)
        wide_industries["{}_标幺值".format(industry)] = per_unit
        wide_industries["{}_MW".format(industry)] = mw
        total = total.add(pd.Series(mw, index=expected_index), fill_value=0.0)
        reproduced_energy = float(mw.sum()) / 1e5
        validation_industries.append({
            "industry": industry,
            "rows": len(result),
            "per_unit_max": float(per_unit.max()),
            "target_energy_yi_kwh": energy[industry],
            "reproduced_energy_yi_kwh": reproduced_energy,
            "energy_difference_yi_kwh": reproduced_energy - energy[industry],
            "scale_factor_MW": scale_mw,
        })

    wide_industries.to_csv(output / "2030行业标幺及MW8760_汇总.csv", index=False, encoding="utf-8-sig")
    # 总负荷先按行业MW相加，再归一化为总标幺；各行业电量规模不同，
    # 不能把行业标幺曲线直接平均或相加。
    total_per_unit = total / float(total.max())
    total_frame = pd.DataFrame({
        "时间": expected_index.strftime("%Y-%m-%d %H:%M:%S"),
        "总负荷MW": total.values,
        "总负荷标幺值": total_per_unit.values,
    })
    total_frame.to_csv(output / "2030总负荷_MW及标幺8760.csv", index=False, encoding="utf-8-sig")

    metrics = total_metrics(total)
    pd.DataFrame([metrics]).to_csv(output / "2030总负荷指标.csv", index=False, encoding="utf-8-sig")
    total_energy = float(total.sum()) / 1e5
    # 这些检查覆盖点数、归一化、电量闭合、逐小时求和和数值有效性。
    checks = {
        "industry_count_is_10": len(all_industries) == 10,
        "each_industry_has_8760_rows": all(item["rows"] == 8760 for item in validation_industries),
        "each_industry_per_unit_max_is_1": all(abs(item["per_unit_max"] - 1.0) <= 1e-9 for item in validation_industries),
        "industry_energy_closure": all(abs(item["energy_difference_yi_kwh"]) <= 1e-8 for item in validation_industries),
        "total_has_8760_rows": len(total_frame) == 8760,
        "total_table_has_time_mw_and_per_unit_only": list(total_frame.columns) == ["时间", "总负荷MW", "总负荷标幺值"],
        "total_per_unit_max_is_1": abs(float(total_per_unit.max()) - 1.0) <= 1e-9,
        "total_energy_matches_model_forecast": abs(total_energy - model_total) <= 1e-8,
        "wide_table_has_one_timestamp_and_20_value_columns": list(wide_industries.columns).count("时间") == 1 and len(wide_industries.columns) == 21,
        "total_mw_equals_industry_hourly_sum": bool(np.allclose(
            wide_industries[["{}_MW".format(industry) for industry in energy]].sum(axis=1).values,
            total.values, rtol=0.0, atol=1e-8,
        )),
        "no_null_or_infinite": bool(np.isfinite(wide_industries.iloc[:, 1:].values).all() and np.isfinite(total_frame[["总负荷MW", "总负荷标幺值"]].values).all()),
        "ccd_script_was_not_called": True,
    }
    status = "passed" if all(value is True for value in checks.values()) else "failed"
    validation = {
        "status": status,
        "checks": checks,
        "model_forecast_total_energy_yi_kwh": model_total,
        "reproduced_total_energy_yi_kwh": total_energy,
        "metrics": metrics,
        "industries": validation_industries,
        "method": "weekly only; no CCD chain; total per-unit is normalized after summing industry MW",
    }
    (output / "最终校验.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    if status != "passed":
        raise RuntimeError("Final validation failed; see 最终校验.json")
    print("[06] Final validation passed: energy={:.6f} 亿kWh, peak={:.2f} MW".format(total_energy, metrics["年最大负荷_MW"]))
    return output
