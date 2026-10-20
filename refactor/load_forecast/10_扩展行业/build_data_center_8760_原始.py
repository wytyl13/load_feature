# -*- coding: utf-8 -*-
"""
数据中心(阿里) 8760 处理 (2026-08-27)
输入: 20260616阿里巴巴2022.01-2023.12负荷数据/2022XX.xlsx, 2023XX.xlsx
      15分钟间隔, '-' = 该测量点无记录
处理: '-'→0, 逐行测量点求和 = 园区总负荷(15min), 每小时4点取平均 → 小时级
口径: 默认 16 站口径(2026-08-27 拍板, 与 2022 可比) = 剔除 飞天/玄铁/含光 3 站
      (2022 全年 3 站无记录, 2023-02 起才齐全; 19 站口径下 2023 约高 1 倍是口径
      变化非真实增长)。--19 开关可复跑旧 19 站口径。
输出: ../2026-08-12/{year}/数据中心_MW.csv (时间,值 与现有行业文件同格式, 4位小数)
      ../2026-08-12/{year}/总负荷去掉九个行业负荷.csv (旧扣除8 − 数据中心16站, 2026-08-27 新增)
已知数据缺陷(如实保留, 不修数据):
  1) 2022 全年 飞天/玄铁/含光 3 站无记录; 2023-01 玄铁/含光 仍无; 2023-02 起 19 站齐全
  2) 2023-01-01~01-12 11:00 缺测(约276h, 全年3.2%) -> 用 2023年1月已有时段的
     典型日曲线(逐小时均值)填充, 在交付说明中注明
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE)
sys.path.insert(0, PROJECT_DIR)
from project_paths import HOURLY_ROOT, TOTAL_LOAD_ROOT
SRC = os.path.join(PROJECT_DIR, '09_数据中心原始')
OUTROOT = str(HOURLY_ROOT)

# 16站口径 = 2022 有效口径: 3 站 2022 全年无记录(2023-02 起才齐全), 剔除后两年可比
EXCLUDE_16 = ['玄铁站', '含光站', '飞天站']
MODE_19 = '--19' in sys.argv          # --19 复跑旧 19 站口径


def readc(f, *paths):
    """utf-8-sig / gbk 容错读 CSV"""
    for enc in ('utf-8-sig', 'utf-8', 'gbk'):
        try:
            return pd.read_csv(os.path.join(*paths, f), encoding=enc)
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    raise FileNotFoundError(f'无法读取 {os.path.join(*paths, f)}')


def load_month_total(f):
    """单月文件 -> 15分钟总负荷 Series(时间索引); '-' 按 0 计"""
    df = pd.read_excel(os.path.join(SRC, f))
    ts = pd.to_datetime(df['时间'])
    num = df.drop(columns=['时间']).replace('-', np.nan)
    num = num.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    if not MODE_19:
        num = num.drop(columns=[c for c in num.columns
                                if any(s in c for s in EXCLUDE_16)])
    s = pd.Series(num.sum(axis=1).to_numpy(float), index=ts)
    s = s[~s.index.duplicated(keep='first')].sort_index()
    return s


def to_hourly(total15):
    """15min -> 小时(取该小时4点均值, 标小时起点)"""
    return total15.resample('h', label='left', closed='left').mean()


def year_8760(year):
    files = sorted(glob.glob(os.path.join(SRC, f'{year}*.xlsx')))
    assert len(files) == 12, f'{year} 月份文件数 {len(files)} != 12'
    hourly = pd.concat([to_hourly(load_month_total(os.path.basename(f))) for f in files])
    # 每月文件末行带下月 0:00 边界(2977=31×96+1), 去重保留首条
    hourly = hourly[~hourly.index.duplicated(keep='first')]
    idx = pd.date_range(f'{year}-01-01', f'{year}-12-31 23:00', freq='h')
    hourly = hourly.reindex(idx)
    return hourly


def fill_jan23_gap(h23):
    """2023-01 缺测时段用该月已有时段的典型日曲线(逐小时均值)填充"""
    jan = h23[h23.index.month == 1]
    miss = jan[jan.isna()].index
    if len(miss) == 0:                       # 空守卫(防 pandas3.x 空 groupby)
        return h23, 0
    avail = jan.dropna()
    prof = avail.groupby(avail.index.hour).mean()  # 0-23 时典型值
    h23.loc[miss] = prof.loc[miss.hour].to_numpy()
    return h23, len(miss)


def rebuild_rest9():
    """扣除9 = 旧扣除8 − 数据中心(16站), 2022/2023 各一; 恒等式断言 实测总−Σ9行业−扣除9≈0"""
    for year in (2022, 2023):
        old = readc('总负荷去掉八个行业负荷.csv', OUTROOT, str(year))
        dc = readc('数据中心_MW.csv', OUTROOT, str(year))
        t_old = pd.DatetimeIndex(pd.to_datetime(old['时间']))
        t_dc = pd.DatetimeIndex(pd.to_datetime(dc['时间']))
        assert t_old.equals(t_dc) and len(t_old) == 8760, f'{year} 时间轴不对齐'
        rest9 = (old['值'].to_numpy(float) - dc['值'].to_numpy(float)).round(4)
        # 恒等式: 实测总(0817\{year}\历史负荷数据（2019-2023）.csv) = Σ9行业 + 扣除9
        act = readc('历史负荷数据%s.csv' % year, str(TOTAL_LOAD_ROOT))
        act_s = pd.Series(act['值'].to_numpy(float), index=pd.to_datetime(act['时间']))
        act_s = act_s.sort_index().to_numpy(float)
        s9 = np.zeros(8760)
        for fn in ['互联网和相关服务_MW.csv', '充换电服务业_MW.csv', '化学原料和化学制品制造业_MW.csv',
                   '有色金属冶炼和压延加工业_MW.csv', '汽车制造业_MW.csv', '电气机械和器材制造业_MW.csv',
                   '非金属矿物制品业_MW.csv', '黑色金属冶炼和压延加工业_MW.csv', '数据中心_MW.csv']:
            s9 += readc(fn, OUTROOT, str(year))['值'].to_numpy(float)
        diff = np.abs(act_s - s9 - rest9).max()
        assert diff < 0.01, f'{year} 恒等式不成立 max|实测−Σ9−扣除9|={diff}'
        out = os.path.join(OUTROOT, str(year), '总负荷去掉九个行业负荷.csv')
        pd.DataFrame({'时间': dc['时间'], '值': rest9}).to_csv(out, index=False, encoding='utf-8-sig')
        print(f'{year}: 扣除9 已输出 {out}  年电量 {rest9.sum()*0.1/10000:.1f} 亿kWh  恒等式max|diff|={diff:.4f} MW')


def main():
    for year in (2022, 2023):
        h = year_8760(year)
        n_missing = int(h.isna().sum())
        if n_missing:
            h, n_fill = fill_jan23_gap(h) if year == 2023 else (h, 0)
            print(f'{year}: 缺测 {n_missing} h, 填充 {n_fill} h (典型日曲线)')
        assert h.notna().all() and len(h) == 8760
        h = h.round(4)
        outdir = os.path.join(OUTROOT, str(year))
        os.makedirs(outdir, exist_ok=True)
        out = os.path.join(outdir, '数据中心_MW.csv')
        # 2022 应不变(16站过滤对2022无副作用): 覆盖前断言 diff=0
        if year == 2022 and os.path.exists(out):
            old = pd.read_csv(out, encoding='utf-8-sig')['值'].to_numpy(float)
            # 历史 CSV 按 4 位小数保存，重建过程中可能出现 0.0001 MW 的舍入差。
            # 超过该量级才视为原始口径发生变化。
            delta = float(np.abs(old - h.to_numpy()).max())
            if delta > 1.0001e-4:
                raise AssertionError('2022 数据中心被意外改变! max diff=%.7f' % delta)
        pd.DataFrame({'时间': h.index.strftime('%Y-%m-%d %H:%M:%S'), '值': h.to_numpy()}).to_csv(
            out, index=False, encoding='utf-8-sig')
        tag = '19站' if MODE_19 else '16站'
        print(f'{year} [{tag}]: 已输出 {out}  年最大 {h.max():.1f} MW  年电量 {h.sum()*0.1/10000:.1f} 亿kWh  均值 {h.mean():.2f} MW')
    rebuild_rest9()


if __name__ == '__main__':
    main()
