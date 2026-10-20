# -*- coding: utf-8 -*-
"""
新方法(2026-08-17): 按"月-工作日/周末(48h)/节假日"标幺曲线 生成目标年 8760
训练年 = 2022(22年生成23年, 其余年份同源)

方法:
1. 训练: 以2022年为基准, 每月按真实日历把每天归为 工作日(周一~周五, 含调休上班日)/
   周六/周日/节假日(法定放假日, 含落在周末的假期)
2. 每月三类标幺曲线:
   - 工作日24h曲线 = 当月工作日平均日曲线, 最大值归一为1(即每周5个工作日的典型曲线, max=1)
   - 周末48h曲线   = [周六平均24h; 周日平均24h], 曲线max=1, 再按周内比例缩放
     (周六/周日平均日峰值 ÷ 工作日平均日峰值, 保留一周内工作日/周末相对水平)
   - 节假日24h曲线 = 当月放假日平均日曲线, max=1, 再乘周内比例(节假日平均日峰值÷工作日)
   (训练月无放假日时, 节假日曲线用当月周日曲线兜底)
   另提取周内比例(每天峰值 ÷ 所在周工作日平均峰值)并构造周曲线(2026-08-24 拍板):
   每周 = 年初第N个7天块, 块内每天 = 该月该类型月典型曲线 × 该天周内比例, 拼接成7天曲线,
   ÷ 周典型日最大 -> 标幺周曲线(max=1); 周典型日最大 ÷ 年最大 -> 周相对水平(周/月水平差)
3. 训练年选择: 2023 = 仅2022; 2020/2021 = 融合2022+2023; 2030 = 融合2022+2023+20/21构造
   (2026-08-24 融合分到最小单位=周: 同周号(年初第N个7天块)跨年等权平均 周标幺曲线 与
   周相对水平; 各年该周无数据(整周零负荷)不参与, 全缺借最近周; 各年该周长度不一(年末块)
   截断到最短对齐)
4. 生成: 目标年逐日对齐真实日历, 每天24h = 该天周号的融合周标幺曲线(周内位置)×融合周水平
   => 月份天数天然=365/366, 闰年删2月29日凑8760
5. 有名值: 标幺×P_base, P_base = 年电量(万kWh)×10 / Σ(周标幺×周相对水平)
   (生成电量=锚定电量严格相等; 周水平融合后乘回, 保留月峰月谷不丢)
   (电量锚定与2026-08-12方法一致: 20-25年用山东表实际占比, 2030总量用NN预测锚点)
6. 输出与2026-08-12相同: {YEAR}/行业CSV+总负荷_合成.csv, {YEAR}_每月负荷特性指标.xlsx,
   典型日曲线{YEAR}/, 分行业典型日曲线{YEAR}/, (2023)预测vs实测对比.xlsx
7. **2026-08-19 口径修改(2020/2021)**: 不再预测系统总负荷——总负荷直接用实测
   (历史负荷数据（2019-2023）.csv); 扣除分行业 = 实测总负荷(8760) − 9个重点行业(生成, 8760)
   逐小时相减(22/23融合标幺只决定重点行业曲线), 负值截断到0;
   不输出 总负荷_合成.csv。2023/2030 保持原逻辑(总负荷=10类之和合成)
   8. **2026-08-27 v26 9行业化**: 数据中心(阿里8760)并入行业链; 2030电量=山东表2025
   "互联网数据服务"174,212.4万×1.054^5固定规则(不走占比外推); 扣除 = 总负荷去掉九个行业负荷
   9. **2026-09-03 用户级数据源**: 8个重点行业(居民生活/充换电/化工/有色/汽车/电气/非金属/黑色)
   改用 2026-08-17/{年}/ 用户级CSV训练: 行业内各用户曲线先各自÷本用户年最大标幺
   (用户量级不同, 直接相加会由大用户主导形状), 等权平均 -> 行业融合标幺曲线, 再走原周曲线
   训练(用户先标幺再平均, 其余步骤不变); 按时间戳对齐(部分用户文件缺小时, 如充换电2022仅
   10-12月), 缺失/整日零负荷的用户-时不参与平均(用户未投产≠行业无负荷), 全用户缺->0;
   数据中心(阿里8760)与扣除分行业(实测口径)无用户拆分, 仍读 2026-08-12
   10. **2026-09-05 互联网退出重点行业, 居民生活进入(第9席)**: 用户级=居民生活-市中-用户1/2
   (仅2022/2023); 电量锚=山东表"B、城乡居民生活用电合计"行(2020-2023实际电量, 2030走占比
   外推同规则); 互联网毛/净口径删除, 数据中心固定规则不变; 2023回测实测对比基准改用
   ACTUAL_FILES(旧10类实测清单含互联网——居民生活无全省实测8760)

用法: python build_new_8760.py 2023
"""
import sys
import os
import pandas as pd
import numpy as np

YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2023
SENSITIVITY = len(sys.argv) > 2 and sys.argv[2] == '--sens'   # 敏感性算例: 重点行业2025-2030增速极端化
SENS_MODE = (sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] in ('all', 'high', 'low', 'ind') else 'all')
# 敏感性板块(甲方2026-08-20要求分两板块): all=全部9行业 / high=2030最大负荷利用小时≥6000 / low=<6000;
# ind=单行业(2026-08-24, sys.argv[4]=山东表行业名, 只该行业增速极端化, 用于"特征与行业比重变化相关性"分析);
# 板块内行业增速极端化, 板块外行业电量增速保持原值不变
SENS_IND = sys.argv[4] if SENS_MODE == 'ind' else None
WKMAX = len(sys.argv) > 3 and sys.argv[3] == '--wkmax'        # 实验: 周基线 = 该周最大工作日峰值(替换工作日平均峰值)
BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE)
sys.path.insert(0, PROJECT_DIR)
from project_paths import CLASSIFICATION_XLSX, FORECAST_ROOT, HOURLY_ROOT, TOTAL_LOAD_ROOT, USER_CURVE_ROOT
DATA_ROOT = str(HOURLY_ROOT)                                      # 输入数据(2022/2023实测/行业表)
OUTROOT = os.environ.get('P8760_OUTROOT') or str(FORECAST_ROOT)   # 输出根(可重定向)
SRCROOT = OUTROOT                                                   # 2020/2021构造曲线读取根
SENS_SUFFIX = ({'all': '_sens', 'high': '_sens_high', 'low': '_sens_low', 'ind': '_sens_ind'}[SENS_MODE] if SENSITIVITY else '')
if SENSITIVITY and SENS_MODE == 'ind' and SENS_IND:
    SENS_SUFFIX += '_' + SENS_IND
OUT = os.path.join(OUTROOT, str(YEAR) + SENS_SUFFIX)
os.makedirs(OUT, exist_ok=True)

INDUSTRY_FILES = ['居民生活_MW.csv', '充换电服务业_MW.csv', '化学原料和化学制品制造业_MW.csv',
                  '总负荷去掉九个行业负荷.csv', '有色金属冶炼和压延加工业_MW.csv', '汽车制造业_MW.csv',
                  '电气机械和器材制造业_MW.csv', '非金属矿物制品业_MW.csv', '黑色金属冶炼和压延加工业_MW.csv',
                  '数据中心_MW.csv']   # 2026-09-05: 第9席 互联网→居民生活; 数据中心=阿里8760无用户拆分

# 2023回测实测对比基准(2026-08-12/{2023}/ 全省实测口径, 2026-09-05固定为旧10类清单):
# 居民生活无全省实测8760, 故实测合计仍按旧清单(含互联网)逐类相加, 与生成端行业划分无关
ACTUAL_FILES = ['互联网和相关服务_MW.csv', '充换电服务业_MW.csv', '化学原料和化学制品制造业_MW.csv',
                '总负荷去掉九个行业负荷.csv', '有色金属冶炼和压延加工业_MW.csv', '汽车制造业_MW.csv',
                '电气机械和器材制造业_MW.csv', '非金属矿物制品业_MW.csv', '黑色金属冶炼和压延加工业_MW.csv',
                '数据中心_MW.csv']

# 2026-09-03 用户级数据源(2026-08-17/{年}/, 与本脚本同目录): 行业CSV名 -> 用户文件名前缀;
# None = 无用户拆分(数据中心=阿里8760单曲线, 扣除分行业=实测口径), 仍读 DATA_ROOT(2026-08-12)
INDUSTRY_USER_PREFIX = {
    '居民生活_MW.csv': '居民生活-市中-用户',
    '充换电服务业_MW.csv': '第三产业-充换电服务业-用户',
    '化学原料和化学制品制造业_MW.csv': '第二产业-化学原料和化学制品制造业-用户',
    '有色金属冶炼和压延加工业_MW.csv': '第二产业-有色金属冶炼和压延加工业-用户',
    '汽车制造业_MW.csv': '第二产业-汽车制造业-用户',
    '电气机械和器材制造业_MW.csv': '第二产业-电气机械和器材制造业-用户',
    '非金属矿物制品业_MW.csv': '第二产业-非金属矿物制品业-用户',
    '黑色金属冶炼和压延加工业_MW.csv': '第二产业-黑色金属冶炼和压延加工业-用户',
    '数据中心_MW.csv': None,
    '总负荷去掉九个行业负荷.csv': None,
}

# ---------- 法定节假日(放假日 off, 调休上班日 work) ----------
HOLIDAYS = {
    2020: dict(off=['2020-01-01', '2020-01-24', '2020-01-25', '2020-01-26', '2020-01-27', '2020-01-28', '2020-01-29', '2020-01-30',
                    '2020-04-04', '2020-04-05', '2020-04-06',
                    '2020-05-01', '2020-05-02', '2020-05-03', '2020-05-04', '2020-05-05',
                    '2020-06-25', '2020-06-26', '2020-06-27',
                    '2020-10-01', '2020-10-02', '2020-10-03', '2020-10-04', '2020-10-05', '2020-10-06', '2020-10-07', '2020-10-08'],
               work=['2020-01-19', '2020-02-01', '2020-04-26', '2020-05-09', '2020-06-28', '2020-09-27', '2020-10-10']),
    2021: dict(off=['2021-01-01', '2021-01-02', '2021-01-03',
                    '2021-02-11', '2021-02-12', '2021-02-13', '2021-02-14', '2021-02-15', '2021-02-16', '2021-02-17',
                    '2021-04-03', '2021-04-04', '2021-04-05',
                    '2021-05-01', '2021-05-02', '2021-05-03', '2021-05-04', '2021-05-05',
                    '2021-06-12', '2021-06-13', '2021-06-14',
                    '2021-09-19', '2021-09-20', '2021-09-21',
                    '2021-10-01', '2021-10-02', '2021-10-03', '2021-10-04', '2021-10-05', '2021-10-06', '2021-10-07'],
               work=['2021-02-07', '2021-02-20', '2021-04-25', '2021-05-08', '2021-09-18', '2021-09-26', '2021-10-09']),
    2022: dict(off=['2022-01-01', '2022-01-02', '2022-01-03',
                    '2022-01-31', '2022-02-01', '2022-02-02', '2022-02-03', '2022-02-04', '2022-02-05', '2022-02-06',
                    '2022-04-03', '2022-04-04', '2022-04-05',
                    '2022-04-30', '2022-05-01', '2022-05-02', '2022-05-03', '2022-05-04',
                    '2022-06-03', '2022-06-04', '2022-06-05',
                    '2022-09-10', '2022-09-11', '2022-09-12',
                    '2022-10-01', '2022-10-02', '2022-10-03', '2022-10-04', '2022-10-05', '2022-10-06', '2022-10-07'],
               work=['2022-01-29', '2022-01-30', '2022-04-02', '2022-04-24', '2022-05-07', '2022-10-08', '2022-10-09']),
    2023: dict(off=['2023-01-01', '2023-01-02',
                    '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24', '2023-01-25', '2023-01-26', '2023-01-27',
                    '2023-04-05',
                    '2023-04-29', '2023-04-30', '2023-05-01', '2023-05-02', '2023-05-03',
                    '2023-06-22', '2023-06-23', '2023-06-24',
                    '2023-09-29', '2023-09-30', '2023-10-01', '2023-10-02', '2023-10-03', '2023-10-04', '2023-10-05', '2023-10-06'],
               work=['2023-01-28', '2023-01-29', '2023-04-23', '2023-05-06', '2023-06-25', '2023-10-07', '2023-10-08']),
    # 2030: 放假安排未公布, 春节(正月初一2030-02-03)/端午(五月初五2030-06-05)/中秋(八月十五2030-09-12)按农历推算估计, 调休忽略
    2030: dict(off=['2030-01-01', '2030-01-02',
                    '2030-02-03', '2030-02-04', '2030-02-05', '2030-02-06', '2030-02-07', '2030-02-08', '2030-02-09',
                    '2030-04-05', '2030-04-06', '2030-04-07',
                    '2030-05-01', '2030-05-02', '2030-05-03', '2030-05-04', '2030-05-05',
                    '2030-06-05', '2030-06-06', '2030-06-07',
                    '2030-09-12', '2030-09-13', '2030-09-14',
                    '2030-10-01', '2030-10-02', '2030-10-03', '2030-10-04', '2030-10-05', '2030-10-06', '2030-10-07'],
               work=[]),
}
assert YEAR in HOLIDAYS, f'缺少 {YEAR} 节假日表'
TRAIN_YEAR = 2022

def readc(f):
    try:
        return pd.read_csv(f, encoding='utf-8-sig')
    except UnicodeDecodeError:
        return pd.read_csv(f, encoding='gbk')

def load_year_curves(year):
    """读 DATA_ROOT(2026-08-12) {year}/ 下实测10类CSV -> {fn: 8760ndarray}
    仅用于 2023 回测的实测对比基准(actual), 训练改走 load_industry_shape;
    2026-09-05 起清单用 ACTUAL_FILES(居民生活无全省实测, 不随行业划分变)"""
    curves = {}
    for fn in ACTUAL_FILES:
        p = os.path.join(DATA_ROOT, str(year), fn)
        if not os.path.isfile(p):
            raise FileNotFoundError(f'未找到 {year} 年数据: {p}')
        curves[fn] = readc(p)['值'].to_numpy(float)
    return curves

def load_industry_shape(fn, year):
    """2026-09-03 用户级训练数据: 2026-08-17/{年}/ 各用户CSV -> 行业融合标幺曲线(8760, max<=1)。
    行业内各用户量级不同(大用户直接相加会主导形状), 故各用户先÷本用户年最大标幺, 再等权平均。
    - 按时间戳对齐(部分用户文件缺小时): 缺失或整日零负荷的用户-时不参与平均(用户未投产≠行业无负荷,
      与训练端"整日零负荷剔除"同语义); 全用户皆缺 -> 0(训练端按零日剔除/借月兜底)
    - 无用户拆分(数据中心=阿里8760单曲线, 扣除分行业=实测口径) 仍读 2026-08-12 单曲线"""
    prefix = INDUSTRY_USER_PREFIX[fn]
    if prefix is None:
        p = os.path.join(DATA_ROOT, str(year), fn)
        if not os.path.isfile(p):
            raise FileNotFoundError(f'未找到 {year} 年数据: {p}')
        return readc(p)['值'].to_numpy(float)
    d = os.path.join(str(USER_CURVE_ROOT), str(year))
    files = sorted(f for f in os.listdir(d) if f.startswith(prefix) and f.endswith('.csv'))
    if not files:
        raise FileNotFoundError(f'{year} 未找到用户文件: {os.path.join(d, prefix)}*')
    full = pd.date_range(f'{year}-01-01', f'{year}-12-31 23:00', freq='h')
    shapes, info = [], []
    for f in files:
        v = readc(os.path.join(d, f))
        s = pd.Series(v['值'].to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(v['时间'])))
        s = s[~s.index.duplicated()].reindex(full)
        day_max = s.groupby(s.index.date).transform('max')
        s[(day_max.isna()) | (day_max <= 0)] = np.nan   # 整日缺测/整日零负荷 -> 该用户该日不参与
        ymax = float(s.max())
        if not ymax > 0:
            print(f'  [warn] {f} 无有效数据, 剔除')
            continue
        shapes.append(s / ymax)
        info.append('%s:%dh' % (f.rsplit('-', 1)[-1].replace('.csv', ''), int(s.notna().sum())))
    if not shapes:
        raise RuntimeError(f'{year} {prefix}*: 所有用户文件无有效数据')
    fused = pd.concat(shapes, axis=1).mean(axis=1)      # 等权平均, NaN不参与(skipna)
    n_all = int(fused.isna().sum())
    if n_all:
        print(f'  [warn] {prefix}*: {n_all} 小时全用户缺, 记0')
    print(f'  {fn} <- {year} 用户融合({len(shapes)}用户: {" ".join(info)}), 融合峰={float(fused.max()):.3f}')
    return fused.fillna(0.0).to_numpy()

def clean_day_offset(d, jan1):
    """年内日序差(去掉2月29日)。数据文件已删2/29(8760行), 分块必须同步剔除,
    否则闰年(2020)3/1起的所有天块号/块内位置都会比平年大1, 融合时该位置分母多算一年 -> 被稀释。"""
    n = (d - jan1).days
    y = jan1.year
    if (y % 4 == 0 and y % 100 != 0) or y % 400 == 0:
        if d > pd.Timestamp(f'{y}-02-29').date():
            n -= 1
    return n

# ---------- 1. 训练: 2022 每月 工作日/周六/周日/节假日 标幺曲线 + 周内水平比例 ----------
def make_type_fn(year):
    off = {pd.Timestamp(d).date() for d in HOLIDAYS[year]['off']}
    work_adj = {pd.Timestamp(d).date() for d in HOLIDAYS[year]['work']}
    def typ(d):
        if d in off:
            return 'hol'
        if d in work_adj:
            return 'wd'
        return 'wd' if d.weekday() < 5 else ('sat' if d.weekday() == 5 else 'sun')
    return typ

def train_month_curves(vals, ts, typ, warn=[]):
    """按(月,类型)收集每天24h构造月典型曲线, 返回 (week_curves, week_levels):
    week_curves{w: 周标幺曲线(7×24, max=1)}, week_levels{w: 周相对水平=周典型日最大÷年最大}
    规则:
    - 整日零负荷的天(如充换电2022年9月前无服务)剔除, 不参与曲线形状与比例
    - 周内比例(实现'按一周的比例'): 每天峰值 ÷ 所在周(周一~周日)工作日平均峰值, 再按月平均,
      对偏斜分布稳健(避免'月平均工作日峰值'被少数异常天拉低导致比例>2)
    - 月内无有效数据(如充换电1-8月) -> 借用最近有数据月份的曲线与比例
    - 训练月无该类型(如3/7/8/11/12月无节假日) -> 用周日兜底"""
    days = {}
    for v, t in zip(vals, ts):
        days.setdefault(t.date(), np.zeros(24))[t.hour] = v
    valid = {d: arr for d, arr in days.items() if arr.max() > 0}   # 剔除整日零负荷
    month_ok = {m: any(d.month == m for d in valid) for m in range(1, 13)}

    # 周内比例: (d) -> day_peak / 该周工作日平均峰值
    week_wd_peak = {}
    for d in valid:
        wk = d - pd.Timedelta(days=d.weekday())   # 所在周周一
        if wk not in week_wd_peak:
            # 周基线 = 该周真实工作日(typ=='wd', 含调休上班日, 不含落在周一~周五的放假日)平均峰值
            wd_arrs = [a for (dd, a) in valid.items() if typ(dd) == 'wd' and dd - pd.Timedelta(days=dd.weekday()) == wk]
            peaks = [a.max() for a in wd_arrs]
            if WKMAX:   # 实验(2026-08-19): 周基线 = 该周最大工作日峰值, 周内比例全部 <=1
                week_wd_peak[wk] = float(max(peaks)) if peaks else 0.0
            else:
                week_wd_peak[wk] = float(np.mean(peaks)) if peaks else 0.0

    day_ratio = {}   # (d, type) -> 该天的周内比例
    for d, arr in valid.items():
        s = week_wd_peak[d - pd.Timedelta(days=d.weekday())]
        if s > 0:
            day_ratio[(d, typ(d))] = float(arr.max() / s)

    grouped = {}
    for d, arr in valid.items():
        grouped.setdefault((d.month, typ(d)), []).append(arr)

    ymax = float(vals.max())   # 标幺基准 = 该训练年8760最大(2026-08-19: 不用月最大, 保留月份水平,
    #                          # 即"标幺曲线×电量"的简单乘法; 月内归一化会丢失月水平使各月典型日=年最大)
    curves, ratios = {}, {}
    for (m, t), arrs in grouped.items():
        mean = np.mean(arrs, axis=0)
        curves[(m, t)] = mean / ymax
    for m in range(1, 13):
        if not month_ok[m]:
            continue
        ratios[(m, 'wd')] = 1.0
        for t in ['sat', 'sun', 'hol']:
            r = [v for (d, tt), v in day_ratio.items() if d.month == m and tt == t]
            if r:
                ratios[(m, t)] = float(np.mean(r))
            else:   # 该月无此类型 -> 周日兜底(无周日则工作日)
                src = 'sun' if (m, 'sun') in curves else 'wd'
                curves[(m, t)] = curves[(m, src)]
                r2 = [v for (d, tt), v in day_ratio.items() if d.month == m and tt == src]
                ratios[(m, t)] = float(np.mean(r2)) if r2 else 1.0
    # 无有效数据的月份: 借用最近有数据月份
    for m in range(1, 13):
        if month_ok[m]:
            continue
        k = min([mm for mm in range(1, 13) if month_ok[mm]], key=lambda mm: abs(mm - m))
        for t in ['wd', 'sat', 'sun', 'hol']:
            curves[(m, t)] = curves[(k, t)]
            ratios[(m, t)] = ratios[(k, t)]
        warn.append(f'    {m}月无有效数据, 借用{k}月曲线')

    # 周曲线(2026-08-24 用户拍板: 融合分到最小单位=周): 每周 = 年初第N个7天块,
    # 块内每天 = 该(月,类型)月典型曲线 × 该天周内比例, 按日历顺序拼接成7天曲线;
    # ÷ 周典型日最大 -> 标幺周曲线(周内形状, max=1); 周内实际最大 ÷ 年最大 -> 周相对水平
    # (周与周/月与月的水平差, 融合后生成时乘回, 避免"每周围最大归一化丢月峰月谷")
    jan1 = ts[0].normalize().date()
    weeks = {}
    for d, arr in valid.items():
        t = typ(d)
        w = clean_day_offset(d, jan1) // 7
        pos = clean_day_offset(d, jan1) % 7
        r = day_ratio.get((d, t), ratios[(d.month, t)])   # 该天周内比例, 缺失用月平均
        weeks.setdefault(w, {})[pos] = curves[(d.month, t)] * r
    week_curves, week_levels = {}, {}
    for w, posc in weeks.items():
        arr = np.zeros(7 * 24)
        for pos, c in posc.items():
            arr[pos * 24:(pos + 1) * 24] = c
        wmax = arr.max()
        if wmax <= 0:   # 整周零负荷(如充换电无服务期) -> 不输出, 融合时借周
            continue
        week_curves[w] = arr / wmax
        # 周水平 = 周典型日峰值÷年最大(arr=曲线×周内比例, 曲线已标幺到ymax, 故wmax即周水平);
        # 2026-08-24修复: 原代码 wmax/ymax 多除一次 -> 1e-5级, 与20/21构造(有名值0.5~1.0)量纲错配,
        # 融合时22/23周水平被完全淹没(2030年最大被2020构造的12/31周水平=1.0主导);
        # 用典型日峰值而非周实际最大: 2022年最大是单日尖峰, 实际峰值会把尖峰复制到2023(回测+9.3%),
        # 典型日峰值平滑尖峰, 2023回测-0.3%
        week_levels[w] = float(wmax)
    # 兜底: 零负荷周借最近有数据周
    n_weeks = (ts[-1].normalize().date() - jan1).days // 7 + 1
    have = set(week_curves)
    for w in range(n_weeks):
        if w in have:
            continue
        k = min(have, key=lambda x: abs(x - w))
        week_curves[w] = week_curves[k]
        week_levels[w] = week_levels[k]
        warn.append(f'    {w}周无有效数据(整周零负荷), 借用{k}周曲线')
    return week_curves, week_levels

def fuse_trainings(trainings):
    """融合多个训练年(2026-08-24 用户拍板: 融合分到最小单位=周):
    同周号(年初第N个7天块)跨年等权平均 周标幺曲线 与 周相对水平;
    各年该周无数据(整周零负荷被剔除)不参与; 某周所有训练年均无数据 -> 借最近有数据周;
    各年该周长度不一致(年末块)截断到最短对齐。
    trainings: [(week_curves, week_levels), ...]"""
    all_c = [t[0] for t in trainings]
    all_l = [t[1] for t in trainings]
    fused_curves, fused_levels = {}, {}
    for w in set().union(*[set(wc) for wc in all_c]):
        cs = [wc[w] for wc in all_c if w in wc]
        n = min(len(c) for c in cs)
        fused_curves[w] = np.mean([c[:n] for c in cs], axis=0)
        ls = [wl[w] for wl, wc in zip(all_l, all_c) if w in wc]
        fused_levels[w] = float(np.mean(ls))
    # 跨年周兜底(2026-08-24修复): 年末块(如w52)部分训练年只有1-2天(pos 0有值, 其余pos全0,
    # 如2022/2023的w52只有12/31, 2020闰年12/31却在pos 1) -> 全0块用该周pos 0填充,
    # 否则目标年12/31生成值为0: 2020/2021扣除行业12/31=纯实测(寒潮82,739.5篡夺年最大,
    # 2020构造周水平=1.0被融合进2030, 2030年最大被推到12/31)
    for w in list(fused_curves):
        c = fused_curves[w]
        n24 = len(c) // 24
        for pos in range(1, n24):
            if c[pos * 24:(pos + 1) * 24].max() == 0:
                c[pos * 24:(pos + 1) * 24] = c[:24]
    if fused_curves:   # 兜底: 全缺周借最近周
        for w in range(max(fused_curves) + 1):
            if w in fused_curves:
                continue
            k = min(fused_curves, key=lambda x: abs(x - w))
            fused_curves[w] = fused_curves[k]
            fused_levels[w] = fused_levels[k]
    return fused_curves, fused_levels

def train_year_industry(fn, year, warn):
    """训练单一年份某行业 -> (week_curves, week_levels)
    2026-09-03: 用户拆分行业读 2026-08-17 用户级数据(各用户自标幺等权融合的行业标幺曲线),
    数据中心/扣除分行业仍读 2026-08-12"""
    vals = load_industry_shape(fn, year)
    ts = pd.date_range(f'{year}-01-01 00:00', f'{year}-12-31 23:00', freq='h')
    return train_month_curves(vals, ts, make_type_fn(year), warn)

# ---------- 2. 生成目标年8760标幺(按真实日历逐周对齐) ----------
def build_year_perunit(year, week_curves, week_levels):
    """逐日对齐真实日历: 每天24h = 该天周号(年初第N个7天块)的融合周标幺曲线(块内位置)
    × 融合周相对水平。2026-08-24: 不再用逐日比例池循环, 每周用自己融合的周曲线。"""
    ts = pd.date_range(f'{year}-01-01 00:00', f'{year}-12-31 23:00', freq='h')
    if year % 4 == 0:   # 闰年删2月29日 -> 8760(与0812一致)
        keep = ~((ts.month == 2) & (ts.day == 29))
        ts = ts[keep]
    jan1 = pd.Timestamp(f'{year}-01-01').date()
    out = np.zeros(len(ts))
    for i, t in enumerate(ts):
        d = t.date()
        w = clean_day_offset(d, jan1) // 7
        c = week_curves.get(w)
        if c is None:   # 极端缺失兜底: 借最近周
            w2 = min(week_curves, key=lambda x: abs(x - w))
            c = week_curves[w2]
            w = w2
        pos = clean_day_offset(d, jan1) % 7
        out[i] = c[pos * 24 + t.hour] * week_levels[w]
    return out, ts

# ---------- 3. 目标年份电量(与0812一致: 20-25用山东表实际占比, 2030用NN锚点) ----------
xl = pd.read_excel(str(CLASSIFICATION_XLSX),
                   header=None, skiprows=14)
xl = xl.dropna(subset=[0])
name_list = xl[0].astype(str).str.strip().tolist()

def find_share(name):
    for k, v in share_rows.items():
        if name in k:
            return v
    raise KeyError(f'未找到行业: {name}')

if YEAR > 2023:
    # 仅2030: 占比按2020-2023四年逐年变化平均外推(线性最小二乘, 用户2026-08-19拍板;
    # 逐年比例外推会使低基数高增长行业(充换电)指数爆炸, 已弃用; 总量仍用ML锚点)
    hist_years = np.array([2020, 2021, 2022, 2023])
    share_mat = xl.iloc[:, 8:14].to_numpy(float)   # (行=行业, 列=2020..2025占比)
    e_mat = xl.iloc[:, 1:7].to_numpy(float)        # (行=行业, 列=2020..2025电量)
    cols = hist_years - 2020
    share_hist = {name: share_mat[j, cols] for j, name in enumerate(name_list)}
    def extrap(name):
        v = share_hist[name]
        if np.all(v == v[0]):   # 全0或全相等: 返回常数
            return v[0]
        k, b = np.polyfit(hist_years, v, 1)
        return max(0.0, k * YEAR + b)
    share_rows = {n: extrap(n) for n in name_list}
    _sy = os.environ.get('P8760_SHARE_YEAR', '')
    if _sy:
        _c = 8 + (int(_sy) - 2020)
        share_rows = dict(zip(name_list, xl.iloc[:, _c].to_numpy(float)))
        print(f'  >>> [占比改用 {_sy} 年实际值] 全部行业')
    _si = os.environ.get('P8760_SHARE23_INDS', '')      # 只把指定行业换成 2023 实际占比
    if _si:
        _c = 8 + 3
        _col23 = xl.iloc[:, _c].to_numpy(float)
        for _j, _n in enumerate(name_list):
            if any(k in _n for k in _si.split(',')):
                share_rows[_n] = float(_col23[_j])
                print(f'  >>> [占比换2023] {_n}  {float(_col23[_j]):.4%}')
    if YEAR == 2030:
        # 2030总量锚点(2026-09-04 拍板回退8特征单一口径): 8特征MLP(ARIMA特征外推)预测2030全社会用电量10,159.7亿kWh;
        # 无8特征对照列时回退ML预测电量(亿kWh)列, 再回退NN预测 11023.6亿kWh
        arima_f = os.path.join(str(FORECAST_ROOT), 'ARIMA预测2026-2030.xlsx')
        if os.path.isfile(arima_f):
            a = pd.read_excel(arima_f, sheet_name='ARIMA特征与ML电量')
            col = 'ML预测电量(亿kWh,8特征对照)' if 'ML预测电量(亿kWh,8特征对照)' in a.columns else 'ML预测电量(亿kWh)'
            tag = '8特征'
            TOTAL = float(a[a['年份'] == 2030][col].iloc[0]) * 10000   # 万kWh
            print(f'2030总量锚点: ML {tag}预测 {TOTAL/10000:,.1f} 亿kWh')
        else:
            nn = pd.read_excel(os.path.join(str(FORECAST_ROOT), 'NN预测2026-2030.xlsx'), sheet_name='2026-2030预测')
            TOTAL = float(nn[nn['年份'] == 2030]['预测电量(亿kWh)'].iloc[0]) * 10000   # 万kWh
    else:
        tot_hist = e_mat[name_list.index('全社会用电总计'), cols]
        TOTAL = tot_hist[0]
    ALL_IND = TOTAL   # 预测年直接锚定全社会用电量(10类之和=ML预测全社会; 2026-09-05起居民生活为独立重点行业, 扣除=全社会−10类)
    print(f'{YEAR}占比沿用{TRAIN_YEAR}: 全社会{TOTAL/10000:,.1f}亿kWh, 全行业占比{share_rows["A、全行业用电合计"]:.4f}(预测年已改为全社会口径)')
else:
    COL_E = 1 + (YEAR - 2020)   # 电量列(0列=行业名, 1-6=2020..2025)
    COL_S = 8 + (YEAR - 2020)   # 占比列(8-13=2020..2025)
    share_rows = dict(zip(name_list, xl[COL_S].to_numpy(float)))
    energy_rows = dict(zip(name_list, xl[COL_E].to_numpy(float)))
    TOTAL = energy_rows['全社会用电总计']        # 万kWh
    ALL_IND = energy_rows['A、全行业用电合计']    # 万kWh

ind_map = {  # CSV文件名 -> 山东表行业名
    '居民生活_MW.csv': 'B、城乡居民生活用电合计',   # 2026-09-05: 互联网退出, 居民生活进入(山东表B行)
    '充换电服务业_MW.csv': '充换电服务业',
    '化学原料和化学制品制造业_MW.csv': '化学原料和化学制品制造业',
    '有色金属冶炼和压延加工业_MW.csv': '有色金属冶炼和压延加工业',
    '汽车制造业_MW.csv': '汽车制造业',
    '电气机械和器材制造业_MW.csv': '电气机械和器材制造业',
    '非金属矿物制品业_MW.csv': '非金属矿物制品业',
    '黑色金属冶炼和压延加工业_MW.csv': '黑色金属冶炼和压延加工业',
    '数据中心_MW.csv': '互联网数据服务',   # 2026-08-27 v26: 山东表"其中：互联网数据服务"行(112行)
}
energy = {}   # 行业 -> 年电量(万kWh)
for fn, name in ind_map.items():
    energy[fn] = find_share(name) * TOTAL
if YEAR == 2030:
    # 数据中心2030固定规则(2026-08-27): 山东表2025"互联网数据服务"174,212.4万 × 2023-2025实际CAGR 1.054^5
    # 不走占比外推(2023口径跳变+583%会使外推爆炸到~175亿); 阿里8760只提供曲线形状
    energy['数据中心_MW.csv'] = 174212.4337 * 1.054 ** 5
# 2026-09-05: 互联网退出重点行业(毛/净口径与"互联网=毛−数据中心"收口一并删除);
# 数据中心保留独立行业, 固定规则电量锚不变
energy['总负荷去掉九个行业负荷.csv'] = ALL_IND - sum(energy.values())

def load_util_hours():
    """各行业2030最大负荷利用小时(h) — 读正常预测2030的 全年指标 xlsx(与最终结果表④块同源), 缺失时用回退表"""
    p = os.path.join(OUTROOT, '2030_每月负荷特性指标.xlsx')
    try:
        y = pd.read_excel(p, sheet_name='全年指标')
        d_all = dict(zip(y['行业'], y['最大负荷利用小时(h)']))   # 全表(含'数据中心', 过滤前取)
        y = y[y['行业'].isin(list(ind_map.values()))]
        d = dict(zip(y['行业'], y['最大负荷利用小时(h)']))
        if '数据中心' in d_all:   # v26: 指标表行业名=数据中心, 敏感性参数名=山东表'互联网数据服务'
            d['互联网数据服务'] = d_all['数据中心']
        if '居民生活' in d_all:   # 2026-09-05: 指标表行业名=居民生活, 敏感性参数名=山东表'B、城乡居民生活用电合计'
            d['B、城乡居民生活用电合计'] = d_all['居民生活']
        return d
    except Exception as e:
        print(f'  [warn] 读 {os.path.basename(p)} 失败({e}), 用回退利用小时表')
        return {'居民生活': 4500, 'B、城乡居民生活用电合计': 4500, '充换电服务业': 1373, '化学原料和化学制品制造业': 6717,
                '有色金属冶炼和压延加工业': 6875, '汽车制造业': 6408, '电气机械和器材制造业': 4710,
                '非金属矿物制品业': 3925, '黑色金属冶炼和压延加工业': 2319,
                '互联网数据服务': 8760}   # 数据中心: 24h连续运行, 回退表=8760(≥6000 → high板块)

if SENSITIVITY:
    # 敏感性算例(甲方2026-08-20要求分两板块): 板块内行业2025-2030增速极端化 — 增速>0的×2, 增速<0的设为全系统增速×2;
    # 板块划分依据=2030最大负荷利用小时(④分行业电量表): high=≥6000h, low=<6000h; 板块外行业增速保持原值不变。
    # 全系统总量(ML 8特征预测 10,159.7亿)不变, 扣除分行业吸收差额 => 纯看板块行业权重变化对负荷特性的影响
    e2025 = e_mat[:, 5]                     # 山东表2025年电量列
    i_tot = name_list.index('全社会用电总计')
    r_sys = (TOTAL / e2025[i_tot]) ** 0.2 - 1
    tmax = load_util_hours()                # 行业 -> 2030最大负荷利用小时(h)
    if SENS_MODE == 'ind':
        in_group = {fn: (ind_map[fn] == SENS_IND) for fn in ind_map}   # 单行业极端化(相关性分析用)
    else:
        in_group = {}
        for fn, name in ind_map.items():
            h = tmax.get(name, 6000 if SENS_MODE == 'high' else 0)
            in_group[fn] = (h >= 6000) if SENS_MODE == 'high' else (h < 6000)
    print(f'===== 敏感性算例[{SENS_MODE}{"-" + SENS_IND if SENS_IND else ""}]: 全系统2025-2030增速 r_sys={r_sys:.4f} =====')
    print('  板块内行业(增速极端化): ' + '、'.join(f'{ind_map[fn]}({tmax.get(ind_map[fn], 0):.0f}h)'
                                                 for fn in ind_map if in_group[fn]))
    print('  板块外行业(原增速不变): ' + '、'.join(ind_map[fn] for fn in ind_map if not in_group[fn]))
    for fn, name in ind_map.items():
        row_i = next(i for i, k in enumerate(name_list) if name in k)   # 模糊匹配表内行名
        e25 = e2025[row_i]
        r_orig = (energy[fn] / e25) ** 0.2 - 1
        # 2026-09-05: 互联网毛/净口径删除, 原"互联网r_orig用毛口径"特例一并删除
        if in_group[fn]:
            r_new = r_orig * 2 if r_orig > 0 else 2 * r_sys
            tag = '板块内(极端化)'
        else:
            r_new = r_orig
            tag = '板块外(不变)'
        energy[fn] = e25 * (1 + r_new) ** 5
        print(f'  {name}: r_orig={r_orig:.4f} -> r_new={r_new:.4f} [{tag}], 2030电量={energy[fn]:,.0f} 万kWh')
    energy['总负荷去掉九个行业负荷.csv'] = TOTAL - sum(energy[fn] for fn in ind_map)
    print(f'  扣除分行业(吸收差额): {energy["总负荷去掉九个行业负荷.csv"]:,.0f} 万kWh')
print(f'===== {YEAR}年 各行业电量(万kWh) =====')
for fn in INDUSTRY_FILES:
    print(f'  {fn}: {energy[fn]:,.0f}  (占比 {energy[fn]/ALL_IND:.4%})')

if YEAR in (2020, 2021, 2023):
    # 2026-08-19 口径统一(κ修正): 行业电量锚定用山东表"全口径"(占全社会~41%),
    # 而总负荷用"实测统调口径"(占全社会~75%)——口径错配使9行业/实测总负荷虚高到55%,
    # 凌晨瞬时占比顶破100%(2020年29h超限)。修正: 9行业电量 ×κ=实测总负荷电量/全社会电量,
    # 与实测总负荷同口径后 9行业/实测 = 41%, 凌晨瞬时 ≤75%, 数学上永不超限, 缩放逻辑删除。
    # 2023回测同样涉及山东表↔实测跨表换算, 必须κ(用户2026-08-19拍板: 所有跨表换算都K)。
    act0 = readc(os.path.join(str(TOTAL_LOAD_ROOT), '历史负荷数据%d.csv' % YEAR))
    a0 = pd.DatetimeIndex(pd.to_datetime(act0['时间'])); v0 = act0['值'].to_numpy(float)
    if YEAR == 2020:   # 闰年删2月29日 -> 8760(与生成表结构一致)
        keep = ~((a0.month == 2) & (a0.day == 29))
        a0, v0 = a0[keep], v0[keep]
    kappa = v0.sum() * 0.1 / TOTAL   # 实测总负荷电量(万kWh) / 山东表全社会(万kWh)
    print(f'[口径统一] {YEAR} 统调系数 κ = 实测总负荷({v0.sum()*0.1:,.0f}万kWh) / 全社会({TOTAL:,.0f}万kWh) = {kappa:.4f}')
    for fn in ind_map:
        energy[fn] *= kappa          # 9行业电量×κ, 与实测总负荷同口径(数据中心2020=14,629×κ≈1.10亿)
    energy['总负荷去掉九个行业负荷.csv'] = v0.sum() * 0.1 - sum(energy[fn] for fn in ind_map)
    print(f'  κ修正后9行业合计 {sum(energy[fn] for fn in ind_map):,.0f} 万kWh, '
          f'扣除分行业(=实测−9行业) {energy["总负荷去掉九个行业负荷.csv"]:,.0f} 万kWh')

def load_generated_year(fn, year):
    """读生成年(2020/2021)行业CSV(有名值MW), 重建周曲线+周水平(与 train_month_curves 同构):
    每周=年初第N个7天块内各天24h按日历拼接, ÷周典型日最大 -> 标幺周曲线; 周最大÷年最大 -> 周相对水平。
    2026-08-19 用户要求: 预测2030必须用2020-2023四年; 2026-08-24: 融合分到最小单位=周。"""
    df = pd.read_csv(os.path.join(SRCROOT, str(year), fn), encoding='utf-8-sig')
    ts = pd.DatetimeIndex(pd.to_datetime(df['时间']))
    vals = df['值'].to_numpy(float)
    ymax = float(vals.max())
    jan1 = pd.Timestamp(f'{year}-01-01').date()
    days = {}
    for v, t in zip(vals, ts):
        days.setdefault(t.date(), np.zeros(24))[t.hour] = v
    weeks = {}
    for d, arr in days.items():
        w = clean_day_offset(d, jan1) // 7
        pos = clean_day_offset(d, jan1) % 7
        weeks.setdefault(w, {})[pos] = arr
    week_curves, week_levels = {}, {}
    for w, posc in weeks.items():
        arr = np.zeros(7 * 24)
        for pos, c in posc.items():
            arr[pos * 24:(pos + 1) * 24] = c
        wmax = arr.max()
        if wmax <= 0:
            continue
        week_curves[w] = arr / wmax
        week_levels[w] = float(wmax / ymax)
    return week_curves, week_levels

# ---------- 4. 各行业: 训练曲线 -> 目标年8760标幺 -> 有名值 ----------
TRAIN_MODE = ('融合2022+2023' if YEAR in (2020, 2021)
              else ('融合2020-2023(22/23实测+20/21构造)' if YEAR == 2030 else '仅2022'))
print(f'===== 训练模式: {TRAIN_MODE} 各行业 周曲线(月典型曲线×周内比例拼接, 2026-08-24按周融合) =====')
series = {}   # 行业 -> (时间, 值)
for fn in INDUSTRY_FILES:
    warn = []
    if YEAR in (2020, 2021):
        t22 = train_year_industry(fn, 2022, warn)
        t23 = train_year_industry(fn, 2023, warn)
        week_curves, week_levels = fuse_trainings([t22, t23])
    elif YEAR == 2030:
        # 2026-08-19 用户要求: 预测2030必须用2020-2023四年数据; 22/23用实测训练周曲线,
        # 20/21为构造产物(形状源于22/23融合), 从生成CSV重建周曲线参与同周等权融合
        t22 = train_year_industry(fn, 2022, warn)
        t23 = train_year_industry(fn, 2023, warn)
        c20, l20 = load_generated_year(fn, 2020)
        c21, l21 = load_generated_year(fn, 2021)
        if os.environ.get('CCD_ONLY23', '') == '1' and fn == '充换电服务业_MW.csv':
            week_curves, week_levels = fuse_trainings([t23])
            print(f'  >>> [CCD only23] {fn}: 只用 2023 训练周曲线')
        else:
            w23f, l23f = fuse_trainings([t22, t23])
            week_curves, week_levels = fuse_trainings([(w23f, l23f), (c20, l20), (c21, l21)])
    else:
        week_curves, week_levels = train_year_industry(fn, 2022, warn)
    if fn == INDUSTRY_FILES[0]:
        print(f'  周曲线: {len(week_curves)} 周, 周相对水平 {min(week_levels.values()):.3f}~{max(week_levels.values()):.3f}')
    if warn:
        print(f'  [{fn}] 数据缺失处理:'); [print(w) for w in warn]
    perunit, ts = build_year_perunit(YEAR, week_curves, week_levels)
    p_base = energy[fn] * 10.0 / perunit.sum()      # MW
    vals = perunit * p_base
    series[fn] = (ts, vals)
    print(f'  {fn}: P_base={p_base:,.1f} MW, 年最大={vals.max():,.1f} MW, 年电量={vals.sum()*0.1:,.0f} 万kWh(目标{energy[fn]:,.0f})')

# ---------- 5. 输出 {YEAR}/ 文件夹CSV(与2022格式一致) ----------
for fn in INDUSTRY_FILES:
    ts_i, vals = series[fn]
    df = pd.DataFrame({'时间': ts_i.strftime('%Y-%m-%d %H:%M:%S'), '值': np.round(vals, 4)})
    df.to_csv(os.path.join(OUT, fn), index=False, encoding='utf-8-sig')
print(f'已输出 {len(INDUSTRY_FILES)} 个行业CSV到 {OUT}')

# ---------- 5.5 2020/2021: 不预测总负荷, 用实测; 扣除分行业 = 实测 − 9重点行业(生成) ----------
if YEAR in (2020, 2021):
    act = readc(os.path.join(str(TOTAL_LOAD_ROOT), '历史负荷数据%d.csv' % YEAR))
    act_ts = pd.DatetimeIndex(pd.to_datetime(act['时间']))
    act_vals = act['值'].to_numpy(float)
    if YEAR == 2020:   # 闰年删2月29日 -> 8760
        keep = ~((act_ts.month == 2) & (act_ts.day == 29))
        act_ts, act_vals = act_ts[keep], act_vals[keep]
    # 删除旧的预测总负荷产物与旧8行业版扣除(防残留误读)
    for old in ['总负荷_合成.csv', '总负荷去掉八个行业负荷.csv']:
        op = os.path.join(OUT, old)
        if os.path.isfile(op):
            os.remove(op); print(f'已删除: {op}')
    # 扣除分行业 = 实测总负荷(8760) − 9重点行业生成值(8760), 逐小时相减
    # (2026-08-19 口径统一κ修正后 9行业/实测=41% 凌晨瞬时≤75%, 数学上不超限, 无需缩放/截断)
    inds9 = [fn for fn in INDUSTRY_FILES if fn != '总负荷去掉九个行业负荷.csv']
    ind_sum = sum(series[fn][1] for fn in inds9)
    rest = act_vals - ind_sum
    nclip = int((rest < 0).sum())
    if nclip:
        print(f'  [提示] 扣除分行业负值 {nclip} 小时截断到 0 (占全年 {nclip/len(rest):.2%}, '
              f'截断电量 {rest[rest < 0].sum()*0.1:,.0f} 万kWh = 全年 {abs(rest[rest<0].sum()/act_vals.sum()):.3%})')
    rest = np.clip(rest, 0, None)
    series['总负荷去掉九个行业负荷.csv'] = (act_ts, rest)
    df_rest = pd.DataFrame({'时间': act_ts.strftime('%Y-%m-%d %H:%M:%S'), '值': np.round(rest, 4)})
    df_rest.to_csv(os.path.join(OUT, '总负荷去掉九个行业负荷.csv'), index=False, encoding='utf-8-sig')
    total = act_vals   # 总负荷 = 实测(不再 10 类之和合成)
    print(f'{YEAR} 实测总负荷: 年最大={total.max():,.1f} MW, 年电量={total.sum()*0.1:,.0f} 万kWh')
    print(f'{YEAR} 扣除分行业 = 实测 - 9重点行业(κ口径统一后 9行业/实测={sum(energy[fn] for fn in ind_map)/(v0.sum()*0.1):.1%}): 年最大={rest.max():,.1f} MW, '
          f'年电量={rest.sum()*0.1:,.0f} 万kWh (9行业之和={act_vals.sum()*0.1 - rest.sum()*0.1:,.0f})')
else:
    # ---------- 6. 总负荷 = 10类之和(9行业+扣除) ----------
    total = sum(vals for _, vals in series.values())
    df_total = pd.DataFrame({'时间': ts.strftime('%Y-%m-%d %H:%M:%S'), '值': np.round(total, 4)})
    df_total.to_csv(os.path.join(OUT, '总负荷_合成.csv'), index=False, encoding='utf-8-sig')
    print(f'{YEAR}合成总负荷: 年最大={total.max():,.1f} MW, 年电量={total.sum()*0.1:,.0f} 万kWh, 最大负荷利用小时={total.sum()/total.max():.0f} h')

# 若该年有实测数据, 加载用于对比
actual = None
actual_ts = None
if YEAR > TRAIN_YEAR and os.path.isdir(os.path.join(DATA_ROOT, str(YEAR))):
    act_curves = load_year_curves(YEAR)
    # 2026-09-05 修复: 实测合计按 ACTUAL_FILES(旧10类, 含实测扣除csv)逐类相加=实测总;
    # 不能用 INDUSTRY_FILES(居民生活无全省实测8760, 会 KeyError)
    actual = sum(act_curves[fn] for fn in ACTUAL_FILES)
    actual_ts = pd.DatetimeIndex(pd.to_datetime(readc(os.path.join(DATA_ROOT, str(YEAR), ACTUAL_FILES[0]))['时间']))
    print(f'已加载 {YEAR} 实测数据用于对比')

# ---------- 7. 每月/全年负荷特性指标 ----------
idx = pd.DatetimeIndex(ts)
m = idx.month

def monthly_metrics(vals):
    out = {}
    for mo in range(1, 13):
        v = vals[m == mo]
        out[mo] = dict(月最大=v.max(), 月最小=v.min(), 月平均=v.mean(),
                       峰谷差=v.max() - v.min())
    return out

def metrics_table(vals, label):
    """返回 (每月表12行, 全年表1行)"""
    mm = monthly_metrics(vals)
    ymax = vals.max()
    rows = []
    for mo in range(1, 13):
        d = mm[mo]
        rows.append({
            '行业': label, '月份': mo,
            '月最大负荷(MW)': round(d['月最大'], 1),
            '月最小负荷(MW)': round(d['月最小'], 1),
            '月平均负荷(MW)': round(d['月平均'], 1),
            '最大负荷率': round(d['月最大'] / ymax, 4),
            '最小负荷率': round(d['月最小'] / ymax, 4),
            '负荷率': round(d['月平均'] / d['月最大'], 4),
            '峰谷差率': round((d['月最大'] - d['月最小']) / d['月最大'], 4),
        })
    mo_max = np.array([mm[mo]['月最大'] for mo in range(1, 13)])
    q_max = np.array([vals[m.isin([3*q-2, 3*q-1, 3*q])].max() for q in range(1, 5)])
    year_row = {
        '行业': label,
        '年最大负荷(MW)': round(ymax, 1),
        '年最小负荷(MW)': round(vals.min(), 1),
        '年平均负荷(MW)': round(vals.mean(), 1),
        '年电量(万kWh)': round(vals.sum() * 0.1, 1),
        '最大负荷利用小时(h)': round(vals.sum() / ymax, 0),
        '月不平衡负荷率': round(mo_max.mean() / ymax, 4),
        '季不平衡负荷率': round(q_max.mean() / ymax, 4),
        '年负荷率': round(vals.sum() / (ymax * 8760), 4),
    }
    return pd.DataFrame(rows), pd.DataFrame([year_row])

month_tables, year_tables = [], []
for label, vals in [('总负荷', total)] + [(fn.replace('_MW.csv', ''), series[fn][1]) for fn in INDUSTRY_FILES]:
    mt, yt = metrics_table(vals, label)
    month_tables.append(mt)
    year_tables.append(yt)
mt = pd.concat(month_tables, ignore_index=True)
yt = pd.concat(year_tables, ignore_index=True)

XLSX = os.path.join(OUTROOT, f'{YEAR}{SENS_SUFFIX}_每月负荷特性指标.xlsx')
with pd.ExcelWriter(XLSX, engine='openpyxl') as w:
    mt.to_excel(w, sheet_name='每月指标', index=False)
    yt.to_excel(w, sheet_name='全年指标', index=False)

from openpyxl import load_workbook
wb = load_workbook(XLSX)
ws1 = wb['每月指标']; ws2 = wb['全年指标']
notes1 = [
    '指标说明(每月):',
    '  最大负荷率 = 月最大负荷 ÷ 年最大负荷',
    '  最小负荷率 = 月最小负荷 ÷ 年最大负荷',
    '  负荷率     = 月平均负荷 ÷ 月最大负荷',
    '  峰谷差率   = (月最大负荷 − 月最小负荷) ÷ 月最大负荷',
]
notes2 = [
    '指标说明(全年):',
    '  最大负荷利用小时数 = 年电量(万kWh)×10 ÷ 年最大负荷(MW)',
    '  月不平衡负荷率     = (Σ各月最大负荷 ÷ 12) ÷ 年最大负荷',
    '  季不平衡负荷率     = (Σ各季最大负荷 ÷ 4) ÷ 年最大负荷',
    '  年负荷率           = 年电量(万kWh)×10 ÷ (年最大负荷(MW) × 8760)',
]
for ws, notes in [(ws1, notes1), (ws2, notes2)]:
    start = ws.max_row + 2
    for j, t in enumerate(notes):
        ws.cell(row=start + j, column=1, value=t)
wb.save(XLSX)
print(f'指标表已输出: {XLSX}')

# ---------- 7.5 预测vs实测对比表(仅预测年且有实测) ----------
if actual is not None:
    mt_a, yt_a = metrics_table(actual, f'{YEAR}实测')
    p, a = yt.iloc[0], yt_a.iloc[0]
    cmp_year = pd.DataFrame({
        '指标': ['年最大负荷(MW)', '年电量(万kWh)', '最大负荷利用小时(h)', '月不平衡负荷率', '季不平衡负荷率', '年负荷率'],
        '预测': [p['年最大负荷(MW)'], p['年电量(万kWh)'], p['最大负荷利用小时(h)'],
                 p['月不平衡负荷率'], p['季不平衡负荷率'], p['年负荷率']],
        '实测': [a['年最大负荷(MW)'], a['年电量(万kWh)'], a['最大负荷利用小时(h)'],
                 a['月不平衡负荷率'], a['季不平衡负荷率'], a['年负荷率']],
    })
    cmp_year['偏差%'] = (cmp_year['预测'] / cmp_year['实测'] - 1) * 100
    pm = mt[mt['行业'] == '总负荷'].iloc[:12].reset_index(drop=True)
    am = mt_a[mt_a['行业'] == f'{YEAR}实测'].iloc[:12].reset_index(drop=True)
    cmp_month = pd.DataFrame({
        '月份': range(1, 13),
        '预测月最大(MW)': pm['月最大负荷(MW)'], '实测月最大(MW)': am['月最大负荷(MW)'],
        '预测负荷率': pm['负荷率'], '实测负荷率': am['负荷率'],
        '预测峰谷差率': pm['峰谷差率'], '实测峰谷差率': am['峰谷差率'],
    })
    cmp_month['月最大偏差%'] = (cmp_month['预测月最大(MW)'] / cmp_month['实测月最大(MW)'] - 1) * 100
    cmp_path = os.path.join(OUTROOT, f'{YEAR}_预测vs实测_对比.xlsx')
    with pd.ExcelWriter(cmp_path, engine='openpyxl') as w:
        cmp_year.to_excel(w, sheet_name='全年对比', index=False)
        cmp_month.to_excel(w, sheet_name='每月对比', index=False)
    print(f'预测vs实测对比已输出: {cmp_path}')
    print(cmp_year.round(2).to_string(index=False))

# ---------- 8. 每月典型日曲线图(月平均日曲线, 标幺) ----------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

month_names = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
fig_dir = os.path.join(OUTROOT, f'典型日曲线{YEAR}{SENS_SUFFIX}')
os.makedirs(fig_dir, exist_ok=True)

def typical_day(vals, mo, m_idx):
    v = vals[m_idx == mo].reshape(-1, 24)
    return v.mean(axis=0)

TAG = '实测' if YEAR in (2020, 2021) else '合成'
for mo in range(1, 13):
    td = typical_day(total, mo, m)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(24), td / td.max(), 'o-', lw=1.5, ms=4, color='C0', label=f'{YEAR}{TAG}')
    title = f'{YEAR}年{month_names[mo-1]}典型日负荷曲线(标幺, 基准=月最大)'
    if actual is not None:
        td_a = typical_day(actual, mo, actual_ts.month)
        ax.plot(range(24), td_a / td_a.max(), 's--', lw=1.2, ms=3.5, color='C3', label=f'{YEAR}实测')
        title += ' — 合成vs实测'
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel('小时'); ax.set_ylabel('标幺值'); ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, f'{YEAR}年{month_names[mo-1]}典型日曲线.png'), dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 6))
for mo in range(1, 13):
    td = typical_day(total, mo, m)
    ax.plot(range(24), td / total.max(), lw=1.0, color='C0', alpha=0.8)
if actual is not None:
    for mo in range(1, 13):
        td_a = typical_day(actual, mo, actual_ts.month)
        ax.plot(range(24), td_a / actual.max(), lw=1.0, ls='--', color='C3', alpha=0.7)
    ax.plot([], [], lw=1.2, color='C0', label='合成(实线)')
    ax.plot([], [], lw=1.2, ls='--', color='C3', label='实测(虚线)')
    for i, mo in enumerate(range(1, 13)):
        ax.plot([], [], color='none')
        ax.annotate(f'{i+1}月', xy=(22.5, 0.55 - i * 0.045), fontsize=7, color='0.35')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title(f'{YEAR}年逐月典型日负荷曲线(标幺, 基准=年最大) — 合成vs实测')
else:
    for mo in range(1, 13):
        td = typical_day(total, mo, m)
        ax.plot(range(24), td / total.max(), lw=1.2, label=month_names[mo-1])
    ax.set_title(f'{YEAR}年逐月典型日负荷曲线(标幺, 基准=年最大)')
    ax.legend(ncol=4, fontsize=8)
ax.set_xlabel('小时'); ax.set_ylabel('标幺值'); ax.set_xticks(range(0, 24, 2))
ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(fig_dir, f'{YEAR}年逐月典型日曲线合并图.png'), dpi=150); plt.close(fig)
tag = '含实测对比' if actual is not None else '无实测对比'
print(f'典型日曲线图已输出到 {fig_dir} ({tag})')

# ---------- 9. 分行业每月典型日曲线(每行业一张4x3子图网格) ----------
ind_fig_dir = os.path.join(OUTROOT, f'分行业典型日曲线{YEAR}{SENS_SUFFIX}')
os.makedirs(ind_fig_dir, exist_ok=True)
for fn in INDUSTRY_FILES:
    vals = series[fn][1]
    fig, axes = plt.subplots(3, 4, figsize=(15, 9))
    for i, mo in enumerate(range(1, 13)):
        ax = axes[i // 4][i % 4]
        td = typical_day(vals, mo, m)
        ax.plot(range(24), td / td.max(), 'o-', lw=1.2, ms=3, color='C0')
        ax.set_title(month_names[mo - 1], fontsize=10)
        ax.set_xticks(range(0, 24, 6))
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    fig.suptitle(f'{YEAR}年 {fn.replace("_MW.csv", "")} 逐月典型日曲线(标幺, 基准=月最大)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(ind_fig_dir, f'{fn.replace("_MW.csv", "")}.png'), dpi=150)
    plt.close(fig)
print(f'分行业典型日曲线图已输出到 {ind_fig_dir}')
print(f'===== {YEAR} 生成完成 =====')
