# -*- coding: utf-8 -*-
"""
两步法预测 2026-2030 全社会用电量(按用户口径):
  第1步 ARIMA 滚动预测 4 特征: 第一产业/第二产业/第三产业增加值、人口(2026-2030)
       (每步 ADF 定差分阶 d, 固定 d 后 AIC 网格选 p,q; 预测值滚入下步再拟合)
  第2步 机器学习(MLP 神经网络, 与 2026-08-12/nn_forecast_1996_2025.py 相同设定):
       - 回测: 训练1996-2019 -> 预测2020-2025(4特征 与 8特征对照 MAPE)
       - 终测: 全样本1996-2025训练 -> 输入 ARIMA 外推的 2026-2030 特征 -> 预测电量
输出: ARIMA预测2026-2030.xlsx (build_new_8760.py 2030 锚点读取 'ML预测电量(亿kWh)')
数据源: ../2026-08-12/111.xlsx
代码阅读要点：这是电量预测主实现，分为 ARIMA 特征外推和标准化 MLP 电量映射两步；回测与终测使用不同样本范围。

结构(S10 重构,行为不变): main() 编排——load_sample_frame 读取清洗 →
forecast_features_arima 特征外推 → run_backtest 回测 → run_final_and_write 终测与输出。
注意: 模块级 SRC = /OUT = 行前缀被 run_original_extensions.py:run_arima_ml_compat
按行首文本改写后 exec 复跑,不可改动行格式。
"""
import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from statsmodels.tsa.stattools import adfuller
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arima_compat import ARIMA
from project_paths import ENERGY_XLSX, PROJECT_ROOT

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = str(ENERGY_XLSX)
OUT = os.path.join(str(PROJECT_ROOT / '预测结果_2020-2030'), 'ARIMA预测2026-2030.xlsx')

FEAT4 = ['第一产业增加值', '第二产业增加值', '第三产业增加值', '人口']
NN_SEEDS = 5


# ---------- 1. 读取与清洗(同 nn_forecast: 插值补缺) ----------
def load_sample_frame(src: str) -> pd.DataFrame:
    """读取 111.xlsx 宏观样本并按年索引插值补缺。

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


# ---------- 2. ARIMA 滚动预测 4 特征 2026-2030 ----------
# 特征预测与电量预测解耦：先得到经济和人口条件，再由 MLP 输出电量。
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

def forecast_features_arima(df: pd.DataFrame) -> dict:
    """ARIMA 滚动外推 4 特征至 2026-2030。

    Args:
        df (pd.DataFrame): load_sample_frame 输出的年度样本。

    Returns:
        dict: 特征名 -> {2026..2030: 预测值}。
    """
    print('\n===== ARIMA 滚动预测 2026-2030: 一产/二产/三产/人口 =====')
    feat_pred = {}   # 特征名 -> {2026..2030: 值}
    for c in FEAT4:
        vals, orders = rolling_forecast(df[c].values)
        feat_pred[c] = dict(zip(range(2026, 2031), vals))
        seq = df[c].values
        print(f'  {c}: 2025实际={seq[-1]:,.1f} -> 2030预测={vals[-1]:,.1f} '
              f'(总增幅{(vals[-1]/seq[-1]-1)*100:+.1f}%, 年增{(vals[-1]/seq[-1])**0.2*100-100:+.1f}%, 阶数{orders}')
    g30 = df[FEAT4[-1]].values[-1]
    print(f'  (2030 GDP总量 = 一产+二产+三产 = {feat_pred["第一产业增加值"][2030]+feat_pred["第二产业增加值"][2030]+feat_pred["第三产业增加值"][2030]:,.1f} 亿元)')
    return feat_pred


# ---------- 3. ML 回测: 训练1996-2019, 测试2020-2025 (4特征 与 8特征对照) ----------
# 2020-2025 完全作为留出集，只在预测完成后计算 MAPE。
def mape(a, b):
    return float(np.mean(np.abs((np.array(a) - np.array(b)) / np.array(a))) * 100)

def fit_mlp_ensemble(Xtr, ytr):
    """训练 MLP(16,8)×NN_SEEDS 种子集成, 返回 (模型列表, scaler)"""
    # 不同特征量纲差异很大；缩放器只用训练集拟合，避免测试信息泄漏。
    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr)
    nns = []
    for s in range(NN_SEEDS):
        nn = MLPRegressor(hidden_layer_sizes=(16, 8), activation='relu', alpha=1e-3,
                          max_iter=5000, random_state=s, learning_rate_init=1e-3)
        nn.fit(Xtr_s, ytr)
        nns.append(nn)
    return nns, sc

def mlp_ensemble(Xtr, ytr, Xte):
    nns, sc = fit_mlp_ensemble(Xtr, ytr)
    return np.mean([nn.predict(sc.transform(Xte)) for nn in nns], axis=0), sc

def run_backtest(df: pd.DataFrame) -> tuple:
    """留出集回测: 4/8 特征 MLP 预测 2020-2025。

    Args:
        df (pd.DataFrame): 年度样本。

    Returns:
        tuple: (实测电量 yte4, 4特征预测 p4_bt, 8特征预测 p8_bt)。
    """
    train_mask, test_mask = df.index <= 2019, df.index > 2019
    Xtr4, ytr4 = df.loc[train_mask, FEAT4].values, df.loc[train_mask, '电力消费量'].values
    Xte4, yte4 = df.loc[test_mask, FEAT4].values, df.loc[test_mask, '电力消费量'].values
    p4_bt, sc4 = mlp_ensemble(Xtr4, ytr4, Xte4)
    FEAT8 = [c for c in df.columns if c != '电力消费量']
    p8_bt, _ = mlp_ensemble(df.loc[train_mask, FEAT8].values, ytr4, df.loc[test_mask, FEAT8].values)
    print(f'\n===== ML 回测(训练1996-2019, 测试2020-2025, MLP(16,8)×{NN_SEEDS}种子) =====')
    for t, a, p4, p8 in zip(df.index[test_mask], yte4, p4_bt, p8_bt):
        print(f'  {t}: 实际{a:,.1f}  4特征MLP={p4:,.1f}({(p4/a-1)*100:+.1f}%)  8特征MLP={p8:,.1f}({(p8/a-1)*100:+.1f}%)')
    print(f'  4特征 MLP MAPE(2020-2025) = {mape(yte4, p4_bt):.2f}%  (当时8特征=2.73%, 线性回归基线=6.07%)')
    print(f'  8特征 MLP MAPE(2020-2025) = {mape(yte4, p8_bt):.2f}%')
    return yte4, p4_bt, p8_bt


# ---------- 4. 终测(全样本)与 5. 输出 xlsx ----------
def run_final_and_write(df: pd.DataFrame, feat_pred: dict,
                        yte4: np.ndarray, p4_bt: np.ndarray, p8_bt: np.ndarray) -> None:
    """全样本终测(4/8 特征两口径)并写出 ARIMA预测2026-2030.xlsx。

    Args:
        df (pd.DataFrame): 年度样本。
        feat_pred (dict): ARIMA 外推的 4 特征。
        yte4 (np.ndarray): 回测实测电量(模型说明 sheet 引用)。
        p4_bt (np.ndarray): 回测 4 特征预测。
        p8_bt (np.ndarray): 回测 8 特征预测。

    Returns: None
    """
    # 十五五增速比较以 2025 实际电量为基准，不使用模型对 2025 的拟合值。
    # 4.1 4特征(用户口径): ARIMA外推 一产/二产/三产/人口
    Xall4, yall = df[FEAT4].values, df['电力消费量'].values
    Xnew4 = np.array([[feat_pred[c][y] for c in FEAT4] for y in range(2026, 2031)])
    E_ml = dict(zip(range(2026, 2031), mlp_ensemble(Xall4, yall, Xnew4)[0]))
    print('\n===== 终测(全样本1996-2025训练): 4特征 ARIMA外推 -> 电量 =====')
    prev_e = df.loc[2025, '电力消费量']
    for t in range(2026, 2031):
        print(f'  {t}: {E_ml[t]:,.1f} 亿kWh  (年增{(E_ml[t]/prev_e*100-100):+.1f}%)')
        prev_e = E_ml[t]

    # 4.2 8特征对照: ARIMA 再外推 三种收入 -> MLP
    FEAT8 = [c for c in df.columns if c != '电力消费量']
    print('\n===== 对照: 8特征(含三种收入) ARIMA外推 -> 电量 =====')
    feat8_pred = {}
    for c in FEAT8:
        vals, orders = rolling_forecast(df[c].values)
        feat8_pred[c] = dict(zip(range(2026, 2031), vals))
        print(f'  {c}: 2030={vals[-1]:,.1f} (年增{(vals[-1]/df[c].values[-1])**0.2*100-100:+.1f}%)')
    Xnew8 = np.array([[feat8_pred[c][y] for c in FEAT8] for y in range(2026, 2031)])
    nns8, sc8 = fit_mlp_ensemble(df[FEAT8].values, yall)          # 保留模型, 用于输出2025拟合值(基年)
    Xnew8_s = sc8.transform(Xnew8)
    E_ml8 = dict(zip(range(2026, 2031), np.mean([nn.predict(Xnew8_s) for nn in nns8], axis=0)))
    prev_e = df.loc[2025, '电力消费量']
    for t in range(2026, 2031):
        print(f'  {t}: {E_ml8[t]:,.1f} 亿kWh  (年增{(E_ml8[t]/prev_e*100-100):+.1f}%)')
        prev_e = E_ml8[t]
    print(f'  2030对比: 4特征={E_ml[2030]:,.1f}  8特征={E_ml8[2030]:,.1f}  NN原法=11023.6')
    # 2026-08-20 用户澄清(两次): 8,671亿=111.xlsx真实2025全社会电量, 是4特征与8特征共同的真实目标序列;
    # 8特征2030预测是对真实电量的预测, 基年同样用真实2025(8,671亿), 两口径同基年可直接对比。
    # (拟合值8494仅作模型偏差诊断参考, 不作基年——用它会把模型低估2%的偏差带进增速)
    i25 = df.index.get_loc(2025)
    Xall8_s = sc8.transform(df[FEAT8].values)
    E8_2025_fit = float(np.mean([nn.predict(Xall8_s)[i25] for nn in nns8], axis=0))
    E8_2025_act = float(df.loc[2025, '电力消费量'])
    r8_155 = (E_ml8[2030] / E8_2025_act) ** 0.2 - 1
    print(f'\n  [2026-08-20] 8特征 2025拟合值(模型诊断参考, 非基年) = {E8_2025_fit:,.1f} 亿kWh (实际 {E8_2025_act:,.1f})')
    print(f'  [2026-08-20] 8特征 2030 十五五年均增速(基年=2025实际 {E8_2025_act:,.1f} 亿kWh) = {r8_155*100:.2f}%')

    # ---------- 5. 输出 xlsx ----------
    with pd.ExcelWriter(OUT, engine='openpyxl') as w:
        pd.DataFrame({
            '年份': list(range(2026, 2031)),
            '第一产业增加值(亿元)': [feat_pred['第一产业增加值'][t] for t in range(2026, 2031)],
            '第二产业增加值(亿元)': [feat_pred['第二产业增加值'][t] for t in range(2026, 2031)],
            '第三产业增加值(亿元)': [feat_pred['第三产业增加值'][t] for t in range(2026, 2031)],
            '人口(万人)': [feat_pred['人口'][t] for t in range(2026, 2031)],
            'GDP总量(亿元)': [sum(feat_pred[c][t] for c in FEAT4[:3]) for t in range(2026, 2031)],
            'ML预测电量(亿kWh,4特征)': [round(E_ml[t], 1) for t in range(2026, 2031)],
            'ML预测电量(亿kWh,8特征对照)': [round(E_ml8[t], 1) for t in range(2026, 2031)],
        }).to_excel(w, sheet_name='ARIMA特征与ML电量', index=False)
        pd.DataFrame({
            '项目': ['第1步', '第2步', 'ML模型', '回测口径', '4特征MLP回测MAPE', '8特征MLP回测MAPE',
                    'NN原法2030电量', 'ML4特征预测2030电量', 'ML8特征预测2030电量',
                    '2030GDP总量(ARIMA)', '2030人口(ARIMA)',
                    '8特征MLP对2025的拟合值(模型诊断参考, 非基年)', '8特征2030十五五年均增速(%)'],
            '值': ['ARIMA滚动预测特征(一产/二产/三产/人口/收入)', 'MLP全样本训练后输入ARIMA外推特征',
                   'MLP(16,8), relu, α=1e-3, 5种子平均', '训练1996-2019, 测试2020-2025',
                   f'{mape(yte4, p4_bt):.2f}%', f'{mape(yte4, p8_bt):.2f}%',
                   11023.6, round(E_ml[2030], 1), round(E_ml8[2030], 1),
                   round(sum(feat_pred[c][2030] for c in FEAT4[:3]), 1), round(feat_pred['人口'][2030], 1),
                   round(E8_2025_fit, 1), round(r8_155 * 100, 2)],
        }).to_excel(w, sheet_name='模型说明', index=False)
    print(f'\n已输出: {OUT}')
    print(f'2030 电量锚点(ML预测, 供build_new_8760.py): {E_ml[2030]:,.1f} 亿kWh')


def main() -> None:
    """两步法电量预测主流程:读数→ARIMA外推→回测→终测→输出。"""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = load_sample_frame(SRC)
    feat_pred = forecast_features_arima(df)
    yte4, p4_bt, p8_bt = run_backtest(df)
    run_final_and_write(df, feat_pred, yte4, p4_bt, p8_bt)


if __name__ == '__main__':
    main()
