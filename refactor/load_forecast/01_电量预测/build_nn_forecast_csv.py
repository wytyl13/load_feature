# -*- coding: utf-8 -*-
"""
神经网络(MLP)预测电量: 训练集/测试集/预测集一览 CSV (2026-09-04)
复现 forecast_arima_ml.py 两步法(ARIMA特征外推 -> MLP(16,8)x5种子集成), 只读原始数据,
不写任何 xlsx (避免覆盖 build_energy_forecast.py 追加到 ARIMA预测2026-2030.xlsx 的 9特征列)。
CSV 分段:
  训练集 1996-2019  样本内拟合值
  测试集 2020-2025  回测外推预测(与实际对比)
  预测集 2026-2030  全样本模型 + ARIMA外推特征 -> 未来电量
输出: 神经网络MLP预测电量_训练测试预测.csv
代码阅读要点：这是 forecast_arima_ml.py 的可审计 CSV 版本，把训练、回测和未来预测放入一张长表；模型结构和随机种子保持一致。

结构(S10 重构,行为不变): main() 编排——load_sample_frame 读取清洗 →
forecast_all_features ARIMA 外推 → run_mlp_models 回测/终测模型 → 组装 CSV 与自检。
"""
import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.stattools import adfuller
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arima_compat import ARIMA
from project_paths import ENERGY_XLSX, PROJECT_ROOT

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = str(ENERGY_XLSX)
ARIMA_XLSX = os.path.join(str(PROJECT_ROOT / '预测结果_2020-2030'), 'ARIMA预测2026-2030.xlsx')
OUT = os.path.join(str(PROJECT_ROOT / '预测结果_2020-2030'), '神经网络MLP预测电量_训练测试预测.csv')

FEAT4 = ['第一产业增加值', '第二产业增加值', '第三产业增加值', '人口']
NN_SEEDS = 5


# ---------- 1. 读取与清洗(与 forecast_arima_ml.py 完全一致) ----------
def load_sample_frame(src: str) -> pd.DataFrame:
    """读取 111.xlsx 宏观样本并按年索引插值补缺(与 forecast_arima_ml.py 一致)。

    Args:
        src (str): 111.xlsx 路径。

    Returns:
        pd.DataFrame: 行=年(1996-2025), 列=三产增加值/人口/电量/三种收入。
    """
    raw = pd.read_excel(src, sheet_name=0, header=0)
    years = [int(str(c)[:4]) for c in raw.columns[1:]]          # 2025..1996
    df = raw.iloc[:8, 1:].T.copy()
    df.columns = ['第一产业增加值', '第二产业增加值', '第三产业增加值', '人口',
                  '电力消费量', '全体居民人均可支配收入', '城镇居民人均可支配收入', '农村居民人均可支配收入']
    df.index = years
    df = df.sort_index().interpolate(limit_direction='both')    # 人口1996-1998线性插值
    print(f'样本: {df.index.min()}-{df.index.max()} 共{len(df)}年')
    return df


# ---------- 2. ARIMA 滚动预测特征 2026-2030 (与 forecast_arima_ml.py 完全一致) ----------
# 重复执行 ARIMA 而不是读取中间特征文件，最后再与既有 xlsx 做数值自检。
def determine_d(y, max_d=2):
    yt = np.asarray(y, dtype=float)
    d = 0
    while d < max_d:
        if adfuller(yt)[1] < 0.05:
            return d
        yt = np.diff(yt); d += 1
    return d

def auto_arima(y, p_=range(0, 4), q_=range(0, 4)):
    d = determine_d(y)
    best = None
    for p in p_:
        for q in q_:
            if p == 0 and q == 0:
                continue
            try:
                fit = ARIMA(y, order=(p, d, q)).fit(method_kwargs={'warn_convergence': False})
            except Exception:
                continue
            if best is None or fit.aic < best[0]:
                best = (fit.aic, (p, d, q), fit)
    return best

def rolling_forecast(data, n_out=5):
    hist = list(data)
    out, orders = [], []
    for _ in range(n_out):
        aic, order, fit = auto_arima(np.asarray(hist, float))
        hist.append(float(fit.forecast(1)[0]))
        out.append(hist[-1]); orders.append(order)
    return out, orders

def forecast_all_features(df: pd.DataFrame) -> tuple:
    """ARIMA 滚动外推 4 特征与 8 特征(含三种收入)至 2026-2030。

    Args:
        df (pd.DataFrame): 年度样本。

    Returns:
        tuple: (feat_pred 4特征, feat8_pred 8特征), 各为 特征名 -> {2026..2030: 值}。
    """
    FEAT8 = [c for c in df.columns if c != '电力消费量']
    feat_pred, feat8_pred = {}, {}
    for c in FEAT4:
        vals, _ = rolling_forecast(df[c].values)
        feat_pred[c] = dict(zip(range(2026, 2031), vals))
    print('4特征 ARIMA 外推完成')
    for c in FEAT8:
        if c in feat_pred:
            feat8_pred[c] = feat_pred[c]
            continue
        vals, _ = rolling_forecast(df[c].values)
        feat8_pred[c] = dict(zip(range(2026, 2031), vals))
    print('8特征(含三种收入) ARIMA 外推完成')
    return feat_pred, feat8_pred


# ---------- 3. MLP 集成 (与 forecast_arima_ml.py 完全一致) ----------
def fit_mlp_ensemble(Xtr, ytr):
    """训练 MLP(16,8)×NN_SEEDS 种子集成, 返回 (模型列表, scaler)"""
    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr)
    nns = []
    for s in range(NN_SEEDS):
        nn = MLPRegressor(hidden_layer_sizes=(16, 8), activation='relu', alpha=1e-3,
                          max_iter=5000, random_state=s, learning_rate_init=1e-3)
        nn.fit(Xtr_s, ytr)
        nns.append(nn)
    return nns, sc

def ens_predict(nns, sc, X):
    return np.mean([nn.predict(sc.transform(X)) for nn in nns], axis=0)

def mape(a, b):
    return float(np.mean(np.abs((np.array(a) - np.array(b)) / np.array(a))) * 100)

def run_mlp_models(df: pd.DataFrame, feat_pred: dict, feat8_pred: dict) -> tuple:
    """回测(训练1996-2019)与终测(全样本)两组 MLP 集成。

    Args:
        df (pd.DataFrame): 年度样本。
        feat_pred (dict): ARIMA 外推 4 特征。
        feat8_pred (dict): ARIMA 外推 8 特征。

    Returns:
        tuple: (p4_bt, p4_fit, p8_bt, p8_fit, p4_fc, p8_fc, yte4) ——
            回测外推/样本内拟合(4、8 特征)、终测未来预测(4、8 特征)、测试集实际电量。
    """
    FEAT8 = [c for c in df.columns if c != '电力消费量']
    train_mask, test_mask = df.index <= 2019, df.index > 2019
    yall = df['电力消费量'].values

    # 回测模型(训练1996-2019): 测试集外推预测 + 训练集样本内拟合
    Xtr4, ytr4 = df.loc[train_mask, FEAT4].values, df.loc[train_mask, '电力消费量'].values
    Xte4 = df.loc[test_mask, FEAT4].values
    nns4_bt, sc4_bt = fit_mlp_ensemble(Xtr4, ytr4)
    p4_bt, p4_fit = ens_predict(nns4_bt, sc4_bt, Xte4), ens_predict(nns4_bt, sc4_bt, Xtr4)
    nns8_bt, sc8_bt = fit_mlp_ensemble(df.loc[train_mask, FEAT8].values, ytr4)
    p8_bt, p8_fit = ens_predict(nns8_bt, sc8_bt, df.loc[test_mask, FEAT8].values), ens_predict(nns8_bt, sc8_bt, df.loc[train_mask, FEAT8].values)

    yte4 = df.loc[test_mask, "电力消费量"].values
    print(f'回测MAPE: 4特征={mape(yte4, p4_bt):.2f}% '
          f'8特征={mape(yte4, p8_bt):.2f}%  (模型说明sheet记录: 3.81% / 2.73%)')

    # 终测模型(全样本1996-2025): 输入 ARIMA 外推特征 -> 2026-2030
    Xnew4 = np.array([[feat_pred[c][y] for c in FEAT4] for y in range(2026, 2031)])
    nns4_f, sc4_f = fit_mlp_ensemble(df[FEAT4].values, yall)
    p4_fc = ens_predict(nns4_f, sc4_f, Xnew4)
    Xnew8 = np.array([[feat8_pred[c][y] for c in FEAT8] for y in range(2026, 2031)])
    nns8_f, sc8_f = fit_mlp_ensemble(df[FEAT8].values, yall)
    p8_fc = ens_predict(nns8_f, sc8_f, Xnew8)
    return p4_bt, p4_fit, p8_bt, p8_fit, p4_fc, p8_fc, yte4


def main() -> None:
    """组装 训练/测试/预测 三段长表 CSV 并与既有 xlsx 自检。"""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = load_sample_frame(SRC)
    feat_pred, feat8_pred = forecast_all_features(df)
    p4_bt, p4_fit, p8_bt, p8_fit, p4_fc, p8_fc, yte4 = run_mlp_models(df, feat_pred, feat8_pred)

    # ---------- 4. 组装 CSV ----------
    # 历史年份保留实际电量，未来年份留空，避免把预测误标为观测。
    seg_map = lambda y: ('训练集(样本内拟合)' if y <= 2019 else
                         '测试集(回测外推)' if y <= 2025 else '预测集(未来预测)')
    rows = []
    tr_idx = {y: k for k, y in enumerate(df.index[df.index <= 2019])}
    te_idx = {y: k for k, y in enumerate(df.index[df.index > 2019])}
    for y in df.index:                                          # 1996-2025 实际样本
        a = float(df.loc[y, '电力消费量'])
        tr = y in tr_idx
        p4 = p4_fit[tr_idx[y]] if tr else p4_bt[te_idx[y]]
        p8 = p8_fit[tr_idx[y]] if tr else p8_bt[te_idx[y]]
        rows.append({
            '分段': seg_map(y), '年份': y,
            **{f'{c}({u})': round(float(df.loc[y, c]), 1) for c, u in
               [('第一产业增加值', '亿元'), ('第二产业增加值', '亿元'), ('第三产业增加值', '亿元'),
                ('人口', '万人'), ('全体居民人均可支配收入', '元'),
                ('城镇居民人均可支配收入', '元'), ('农村居民人均可支配收入', '元')]},
            '特征来源': '实际值',
            '实际电量(亿kWh)': round(a, 1),
            '4特征MLP电量(亿kWh)': round(float(p4), 1),
            '8特征MLP电量(亿kWh)': round(float(p8), 1),
            '4特征误差(%)': round((p4 / a - 1) * 100, 2),
            '8特征误差(%)': round((p8 / a - 1) * 100, 2),
        })
    for j, y in enumerate(range(2026, 2031)):                   # 2026-2030 预测
        rows.append({
            '分段': seg_map(y), '年份': y,
            **{f'{c}({u})': round(float((feat_pred if c in feat_pred else feat8_pred)[c][y]), 1) for c, u in
               [('第一产业增加值', '亿元'), ('第二产业增加值', '亿元'), ('第三产业增加值', '亿元'),
                ('人口', '万人'), ('全体居民人均可支配收入', '元'),
                ('城镇居民人均可支配收入', '元'), ('农村居民人均可支配收入', '元')]},
            '特征来源': 'ARIMA外推',
            '实际电量(亿kWh)': None,
            '4特征MLP电量(亿kWh)': round(float(p4_fc[j]), 1),
            '8特征MLP电量(亿kWh)': round(float(p8_fc[j]), 1),
            '4特征误差(%)': None,
            '8特征误差(%)': None,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding='utf-8-sig')
    print(f'\n已输出: {OUT}  ({len(out)}行)')

    # ---------- 5. 自检: 复现值 vs 既有 xlsx (应仅差四舍五入) ----------
    if os.path.exists(ARIMA_XLSX):
        _r = pd.read_excel(ARIMA_XLSX, sheet_name='ARIMA特征与ML电量').set_index('年份')
        d4 = max(abs(p4_fc[j] - _r.loc[y, 'ML预测电量(亿kWh,4特征)']) for j, y in enumerate(range(2026, 2031)))
        d8 = max(abs(p8_fc[j] - _r.loc[y, 'ML预测电量(亿kWh,8特征对照)']) for j, y in enumerate(range(2026, 2031)))
        print(f'自检: 2026-2030 复现值 vs xlsx 最大偏差 4特征={d4:.3f} 8特征={d8:.3f} (亿kWh, <=0.05 为仅舍入差)')


if __name__ == '__main__':
    main()
