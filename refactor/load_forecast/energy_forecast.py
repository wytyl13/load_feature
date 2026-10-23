#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""03 段:MLP 总电量预测与行业占比趋势外推(原包 03_总电量与行业电量预测/run_stage.py 移植)。

计算行逐行保持一致;差异仅两处:
- 行业电量从 150 行分类表 unique_row 定位改为行业电量简表按行名直取
  (值经 round-trip 逐位断言一致);
- 行业迭代序从 config.industry_workbook_mapping 键序改为 config.industries
  (同一冻结序)。
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from data_io import FEATURES, TARGET, TOTAL_KEY, load_industry_energy, load_macro_table

warnings.filterwarnings("ignore")

FEATURES4 = FEATURES[:4]
FEATURES7 = FEATURES


def fit_ensemble(x_train, y_train):
    """训练5个不同随机种子的MLP，并返回模型和特征标准化器。

    不同指标量纲差异很大，先用训练集拟合StandardScaler，再将同一个
    scaler用于回测和2030预测，避免数据泄漏。
    """
    scaler = StandardScaler().fit(x_train)
    transformed = scaler.transform(x_train)
    models = []
    for seed in range(5):
        model = MLPRegressor(
            hidden_layer_sizes=(16, 8), activation="relu", alpha=1e-3,
            max_iter=5000, random_state=seed, learning_rate_init=1e-3,
        )
        model.fit(transformed, y_train)
        models.append(model)
    return models, scaler


def predict_ensemble(models, scaler, values):
    """将输入标准化后取5个MLP预测值的算术平均。"""
    transformed = scaler.transform(values)
    return np.mean([model.predict(transformed) for model in models], axis=0)


def mape(actual, predicted):
    """计算总电量模型回测的MAPE。"""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100.0)


def r_squared(actual, predicted):
    """计算拟合优度R²；常量序列时返回1，避免除零。"""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    return 1.0 - float(np.sum((actual - predicted) ** 2)) / denominator if denominator else 1.0


def run_energy_forecast(macro_workbook: Path, energy_workbook: Path, run_dir: Path, config: dict) -> Path:
    """完成MLP总电量预测和行业占比趋势外推,写 6 个判据文件。

    第03步的关键接口是两个文件：
    ``2030总电量口径.csv`` 给出最终总量，
    ``2030行业电量锚点.csv`` 给出十个行业的最终曲线电量。
    第06步只读取这两个结果，不读取配置中的固定电量。
    """
    output = run_dir / "03_总电量与行业电量预测"
    output.mkdir(parents=True, exist_ok=True)
    macro = load_macro_table(macro_workbook)
    feature_forecast = pd.read_csv(run_dir / "02_宏观特征预测" / "ARIMA特征预测_2026-2030.csv", encoding="utf-8-sig")

    # 先用1996-2019训练、2020-2025测试，保留可审计的历史回测指标；
    # 随后再用1996-2025全样本训练最终模型。
    train_mask = macro.index <= 2019
    test_mask = macro.index >= 2020
    backtest_rows = []
    forecasts = {}
    for label, features in [("4特征", FEATURES4), ("8指标口径_7个输入特征", FEATURES7)]:
        back_models, back_scaler = fit_ensemble(macro.loc[train_mask, features].values, macro.loc[train_mask, TARGET].values)
        back_pred = predict_ensemble(back_models, back_scaler, macro.loc[test_mask, features].values)
        backtest_rows.append(
            {
                "模型": label,
                "训练区间": "1996-2019",
                "测试区间": "2020-2025",
                "MAPE_pct": mape(macro.loc[test_mask, TARGET].values, back_pred),
                "RMSE_亿kWh": float(np.sqrt(np.mean((back_pred - macro.loc[test_mask, TARGET].values) ** 2))),
            }
        )
        models, scaler = fit_ensemble(macro[features].values, macro[TARGET].values)
        forecasts[label] = predict_ensemble(models, scaler, feature_forecast[features].values)
    pd.DataFrame(backtest_rows).to_csv(output / "MLP总电量回测指标.csv", index=False, encoding="utf-8-sig")

    total_table = pd.DataFrame({
        "年份": feature_forecast["年份"].astype(int),
        "4特征MLP预测电量_亿kWh": forecasts["4特征"],
        "8指标口径MLP预测电量_亿kWh": forecasts["8指标口径_7个输入特征"],
    })
    total_table.to_csv(output / "MLP总电量预测_2026-2030.csv", index=False, encoding="utf-8-sig")
    model_2030 = float(total_table.loc[total_table["年份"] == 2030, "8指标口径MLP预测电量_亿kWh"].iloc[0])

    # 行业电量只用于拟合“行业占总量比例”的趋势，实际2030电量还要乘
    # 本次运行刚得到的MLP总电量，保证总量不会被固定值覆盖。
    energy = load_industry_energy(energy_workbook, config["industries"])
    years = np.arange(2020, 2026, dtype=float)
    total_history = energy.loc[TOTAL_KEY].to_numpy(dtype=float)

    curve_rows, selection_rows = [], []
    for industry in config["industries"]:
        values = energy.loc[industry].to_numpy(dtype=float)
        shares = values / total_history
        linear_coef = np.polyfit(years, shares, 1)
        linear_fit = np.polyval(linear_coef, years)
        exponential_coef = np.polyfit(years, np.log(shares), 1)
        exponential_fit = np.exp(np.polyval(exponential_coef, years))
        linear_r2 = r_squared(shares, linear_fit)
        exponential_r2 = r_squared(shares, exponential_fit)
        r2_winner = "linear" if linear_r2 >= exponential_r2 else "exponential"
        adopted = config["ppt_adopted_share_model"][industry]
        for year in range(2020, 2031):
            linear_share = float(np.polyval(linear_coef, year))
            exponential_share = float(np.exp(np.polyval(exponential_coef, year)))
            curve_rows.append({
                "行业": industry,
                "年份": year,
                "历史占比": float(shares[int(year - 2020)]) if year <= 2025 else np.nan,
                "线性预测占比": linear_share,
                "指数预测占比": exponential_share,
                "PPT采用模型": adopted,
                "PPT采用模型占比": linear_share if adopted == "linear" else exponential_share,
            })
        linear_share_2030 = float(np.polyval(linear_coef, 2030))
        exponential_share_2030 = float(np.exp(np.polyval(exponential_coef, 2030)))
        adopted_share = linear_share_2030 if adopted == "linear" else exponential_share_2030
        # PPT采用模型决定最终占比；R2最大模型仍保留在结果中用于审计对比。
        pred_linear = linear_share_2030 * model_2030
        pred_exp = exponential_share_2030 * model_2030
        adopted_energy = pred_linear if adopted == "linear" else pred_exp
        selection_rows.append({
            "行业": industry,
            "线性R2": linear_r2,
            "指数R2": exponential_r2,
            "R2最大模型": r2_winner,
            "PPT采用模型": adopted,
            "线性模型2030电量_亿kWh": pred_linear,
            "指数模型2030电量_亿kWh": pred_exp,
            "PPT采用模型2030电量_亿kWh": adopted_energy,
            "最终用于曲线占比_pct": adopted_share * 100.0,
            "最终用于曲线电量_亿kWh": adopted_energy,
        })

    pd.DataFrame(curve_rows).to_csv(output / "行业占比历史及两类预测曲线.csv", index=False, encoding="utf-8-sig")
    selection = pd.DataFrame(selection_rows)
    # 其他行业不是独立外推，而是总量扣除九个重点行业后的余额，
    # 因此十个行业电量之和严格等于MLP总量。
    residual = model_2030 - float(selection["最终用于曲线电量_亿kWh"].sum())
    if residual <= 0:
        raise RuntimeError("Forecasted key-industry energy exceeds the model total")
    selection = pd.concat([
        selection,
        pd.DataFrame([{
            "行业": "其他（扣除重点行业）",
            "线性R2": np.nan,
            "指数R2": np.nan,
            "R2最大模型": "residual",
            "PPT采用模型": "residual",
            "线性模型2030电量_亿kWh": np.nan,
            "指数模型2030电量_亿kWh": np.nan,
            "PPT采用模型2030电量_亿kWh": residual,
            "最终用于曲线占比_pct": residual / model_2030 * 100.0,
            "最终用于曲线电量_亿kWh": residual,
        }])
    ], ignore_index=True)
    selection.to_csv(output / "2030行业电量锚点.csv", index=False, encoding="utf-8-sig")

    total_policy = pd.DataFrame([
        {"口径": "8指标口径MLP预测值", "2030总电量_亿kWh": model_2030, "是否用于最终曲线": True},
    ])
    total_policy.to_csv(output / "2030总电量口径.csv", index=False, encoding="utf-8-sig")
    audit = {
        "mlp": "MLPRegressor(16,8), relu, alpha=1e-3, seeds=0..4 average",
        "model_2030_yi_kwh": model_2030,
        "final_curve_total_yi_kwh": model_2030,
        "forecast_industry_sum_yi_kwh": float(selection["最终用于曲线电量_亿kWh"].sum()),
        "final_curve_policy": "最终8760直接使用8指标口径MLP预测总电量；九个重点行业按预测占比分配，其他行业取余额",
    }
    (output / "阶段审计.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[03] Total and industry energy forecasting completed: final model2030={:.4f}".format(model_2030))
    return output
