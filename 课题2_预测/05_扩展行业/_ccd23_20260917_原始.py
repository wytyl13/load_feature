# -*- coding: utf-8 -*-
"""2026-09-17: 充换电 2030 标幺曲线 —— 弃用 2022, 只用 2023 生成。

动因: 2022 原始数据 1-8月 5,832 小时全 0、9-11月近 0、12月突跳到 0.3008 (100倍),
      22+23 融合把这条"年尾上翘"带进了主链 2030 曲线 (11月 0.259 → 12月 0.507, 翻倍)。

口径: 与 P14/P16 同 —— holidays=spring,may,national  pattern=hybrid  single_typical=True
      唯一改动: hist_years=[2023]  typical_years=[2023]

输入源已实证 = 20260910/_p15_work/输入/充换电服务业/  (22+23 复跑与主链逐点差 0.0000)
      注意: 不能用 20260916/03_数据/06_生成器输入目录/ 那份是 12号实测口径, 不是主链源。

输出: _ccd23_out_20260917/{周法,月法}/  (不覆盖 09-16 的 _task20260916/t2_out)
"""
from __future__ import annotations

import csv
import importlib.util
import io
import shutil
import sys
import tempfile
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
D10 = HERE / '20260910'
W16 = D10 / '_p16_work'
SRC_IN = D10 / '_p15_work' / '输入' / '充换电服务业'   # 主链真身(已实证)
Z = HERE / '20260916' / '02_需求实现逻辑' / '04_标幺化8760生成'
OUT = HERE / '_ccd23_out_20260917'

GUARD_OLD = """                raw_week_norm[year][w_ - 1] / base_week_by_year[year][w_ - 1]
                for year, w_ in srcs"""
GUARD_NEW = """                (raw_week_norm[year][w_ - 1] / base_week_by_year[year][w_ - 1])
                if base_week_by_year[year][w_ - 1] >= 0.05 else 1.0
                for year, w_ in srcs"""

IDX = pd.date_range('2030-01-01', '2030-12-31 23:00', freq='h')


def load_generator(fname: str, guard: bool):
    """guard=False → 用 09-16 之前的原版生成器(无 base_week 保护)。"""
    src = (Z / fname).read_text(encoding='utf-8')
    if guard:
        if GUARD_NEW not in src:
            src = src.replace(GUARD_OLD, GUARD_NEW)
    else:
        src = src.replace(GUARD_NEW, GUARD_OLD)
    t = Path(tempfile.mkdtemp(prefix='ccd23_')) / 'p.py'
    t.write_text(src, encoding='utf-8')
    spec = importlib.util.spec_from_file_location('g', t)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def read_pu(p: Path) -> np.ndarray:
    with p.open(encoding='utf-8-sig', newline='') as f:
        return np.array([float(r['全年小时标幺']) for r in csv.DictReader(f)])


def shape_stats(y: np.ndarray) -> dict:
    s = pd.Series(y, index=IDX)
    mo = s.resample('ME').mean()
    return dict(
        利用小时=y.sum() / y.max(),
        峰值月=int(s.idxmax().month),
        十一月=mo.iloc[10], 十二月=mo.iloc[11],
        年尾翘=mo.iloc[11] / mo.iloc[10],
        十二月电量占比=s[s.index.month == 12].sum() / y.sum(),
    )


# ---------- 0. 自检: 输入源必须能复现主链(22+23) ----------
print('=== 自检: 输入源复现主链? ===')
for method, fname in (('周法', 'generate_annual_normalized_curve_v2_多假_工作双休.py'),
                      ('月法', 'generate_annual_normalized_curve_monthly_top3_typical_v2_多假_工作双休.py')):
    m = load_generator(fname, guard=True)
    d = Path(tempfile.mkdtemp(prefix='ccd23chk_'))
    m.generate(input_dir=SRC_IN, output_dir=d, target_year=2030,
               typical_years=[2022, 2023], holidays=['spring', 'may', 'national'],
               hist_years=[2022, 2023], pattern='hybrid', single_typical=True)
    chk = read_pu(d / '2030年全年小时标幺曲线.csv')
    ref = read_pu(W16 / method / '充换电服务业' / '2030年全年小时标幺曲线.csv')
    ok = np.abs(chk - ref).max() < 1e-12
    print(f'  {method}: 22+23 复跑 vs 主链 最大逐点差={np.abs(chk - ref).max():.2e}'
          f'  → {"✓ 命中" if ok else "✗ 不符, 输入源不对"}')
    if not ok:
        sys.exit(1)
print()

# ---------- 1. 生成 ----------
tmp_in = Path(tempfile.mkdtemp(prefix='ccd23in_'))
shutil.copy2(SRC_IN / '历史负荷数据2023.csv', tmp_in / '历史负荷数据2023.csv')
print(f'临时输入目录(仅 2023): {tmp_in}\n')

GEN = (('周法', 'generate_annual_normalized_curve_v2_多假_工作双休.py'),
       ('月法', 'generate_annual_normalized_curve_monthly_top3_typical_v2_多假_工作双休.py'))

new_curves, guarded_curves = {}, {}
for method, fname in GEN:
    for guard, store in ((True, new_curves), (False, guarded_curves)):
        m = load_generator(fname, guard)
        d = OUT / method if guard else Path(tempfile.mkdtemp(prefix='ccd23ng_'))
        m.generate(input_dir=tmp_in, output_dir=d, target_year=2030,
                   typical_years=[2023], holidays=['spring', 'may', 'national'],
                   hist_years=[2023], pattern='hybrid', single_typical=True)
        store[method] = read_pu(d / '2030年全年小时标幺曲线.csv')

# guard 是否触发的实证: 带/不带保护跑出来的曲线是否逐点一致
print('=== base_week 保护(09-16 加的 patch)是否改变结果 ===')
for method, _ in GEN:
    a, b = new_curves[method], guarded_curves[method]
    print(f'  {method}: 带保护 vs 不带保护 最大逐点差={np.abs(a - b).max():.3e}'
          f'  → {"无影响(patch 是空操作)" if np.abs(a - b).max() < 1e-12 else "⚠ 有影响"}')

# ---------- 2. 与主链对比 ----------
print('\n=== 充换电 2030 标幺曲线: 主链(22+23) vs 新(仅23) ===')
rows = []
for method, _ in GEN:
    old = read_pu(W16 / method / '充换电服务业' / '2030年全年小时标幺曲线.csv')
    new = new_curves[method]
    so, sn = shape_stats(old), shape_stats(new)
    rows.append((method, so, sn))
    print(f'\n【{method}】')
    print(f'{"指标":<14}{"旧 22+23":>12}{"新 仅23":>12}{"变化":>14}')
    for k in ('利用小时', '峰值月', '十一月', '十二月', '年尾翘', '十二月电量占比'):
        ov, nv = so[k], sn[k]
        if k == '峰值月':
            print(f'{k:<14}{ov:>10}月{nv:>10}月{"翻到 %d 月" % nv:>14}')
        elif k == '年尾翘':
            print(f'{k:<14}{ov:>12.3f}{nv:>12.3f}{"−%.0f%%" % ((1 - nv / ov) * 100):>14}')
        elif k == '十二月电量占比':
            print(f'{k:<14}{ov:>11.2%}{nv:>12.2%}{"−%.1f pp" % ((ov - nv) * 100):>14}')
        else:
            print(f'{k:<14}{ov:>12.3f}{nv:>12.3f}{nv - ov:>+14.3f}')
    print(f'  逐点最大差={np.abs(new - old).max():.4f}   相关系数={np.corrcoef(new, old)[0, 1]:.4f}')

# ---------- 3. 月度均值并排 ----------
print('\n=== 月度均值并排 ===')
hdr = '  月      :' + ''.join(f'{i:>7d}' for i in range(1, 13))
for method, _ in GEN:
    old = read_pu(W16 / method / '充换电服务业' / '2030年全年小时标幺曲线.csv')
    new = new_curves[method]
    print(f'\n【{method}】')
    print(hdr)
    print('  旧22+23 :' + ''.join(f'{v:7.3f}' for v in pd.Series(old, index=IDX).resample('ME').mean()))
    print('  新仅23  :' + ''.join(f'{v:7.3f}' for v in pd.Series(new, index=IDX).resample('ME').mean()))

# 2023 原始形状作对照
raw = pd.read_csv(SRC_IN / '历史负荷数据2023.csv')
raw.columns = ['时间', '值']
raw = raw.set_index(pd.to_datetime(raw['时间']))['值']
print('  2023原始:' + ''.join(f'{v:7.3f}' for v in raw.resample('ME').mean()))

print(f'\n输出目录: {OUT}')
