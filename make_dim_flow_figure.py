#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NN 结构图风格的数据流转图:列=操作层,节点=圆圈,连线=数据流。

按八步流程编号:第一~三步形状线(琥珀),第四步电量线(蓝),历史全省支线(紫),
第五/六步汇合(红),第七/八步波动输出(绿)。中文字体雅黑,形状行纯 ASCII 等宽。
生成 算法流程图.png。
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

C_IN = ('#D6E4F7', '#2E5E9E')     # 输入
C_SHAPE = ('#FCEFD3', '#B07A1F')  # 形状线 一~三步
C_ELEC = ('#DBEEF4', '#1F6E7A')   # 电量线 四步
C_HIST = ('#EBE2F6', '#6C48A8')   # 历史全省支线
C_MERGE = ('#FAE3E5', '#A63A45')  # 汇合 五/六步
C_OUT = ('#DFF3D8', '#2F7A33')    # 波动输出 七/八步
INK, LINE = '#333333', '#AAAAAA'

fig, ax = plt.subplots(figsize=(16.5, 9.2), dpi=200)
ax.set_xlim(0, 120)
ax.set_ylim(70, 0)
ax.set_aspect('equal')
ax.axis('off')

XIN, X1, X2, X3, X4, X5, X6, X7, XO = 9, 22, 34.5, 47, 59.5, 72, 84.5, 97, 111


def node(x, y, text, palette, r=3.5, fs=7.0):
    ax.add_patch(Circle((x, y), r, fc=palette[0], ec=palette[1], lw=1.4, zorder=3))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            color=palette[1], fontweight='bold', zorder=4, linespacing=1.25)


def link(x0, y0, x1, y1):
    ax.plot([x0, x1], [y0, y1], color=LINE, lw=0.9, zorder=1)


def caption(x, y0, name, shape, note, color, fs_shape=7.6):
    ax.text(x, y0, name, ha='center', va='top', fontsize=8.6, color=color, fontweight='bold')
    ax.text(x, y0 + 3.2, shape, ha='center', va='top', fontsize=fs_shape,
            color=INK, fontfamily='monospace')
    ax.text(x, y0 + 6.6, note, ha='center', va='top', fontsize=7.2, color=INK)


ax.text(60, 4, '山东中长期负荷特性8760曲线 · 数据流转网络图', ha='center', va='center',
        fontsize=15, fontweight='bold', color=INK)

user, official, upstream, hist = (XIN, 11), (XIN, 24), (XIN, 31), (XIN, 41)
step1 = (X1, 11)
step2 = (X2, 11)
step3 = (X3, 13)
step4 = (X4, 26)
corr = (X5, 30)
typical = (X5, 41)
step6 = (X6, 22)
calenv = (X6, 41)
step7 = (X7, 22)
resid = (X7, 41)
step8 = (XO, 22)

link(*user, *step1)
link(*step1, *step2)
link(*step2, *step3)
link(*step3, *corr)
link(*official, *step4)
link(*upstream, *step4)
link(*step4, *corr)
link(*hist, *typical)
link(*typical, *corr)
link(*corr, *step6)
link(*step4, *step6)
link(*hist, *calenv)
link(*calenv, *step6)
link(*step6, *step7)
ax.plot([hist[0], hist[0], resid[0], resid[0]], [hist[1], 50, 50, resid[1] + 3.5],
        color=LINE, lw=0.9, zorder=1)  # 残差库支线绕行下方,避免穿过日型/月包络节点
link(*resid, *step7)
link(*step7, *step8)

node(*user, '用户日\n52x2年', C_IN, fs=6.6)
node(*official, '分类表\n135行', C_IN, fs=6.6)
node(*upstream, '上游\n3x3分位', C_IN, fs=6.6)
node(*hist, '全省\n5年', C_IN, fs=6.6)
node(*step1, '行业日\n形状', C_SHAPE)
node(*step2, 'medoid\n2023', C_SHAPE)
node(*step3, '未来\n行业形状', C_SHAPE, fs=6.6)
node(*step4, '25组成\n电量', C_ELEC)
node(*corr, '修正+总\n形状', C_MERGE, fs=6.6)
node(*typical, '全省典型\n12格', C_HIST, fs=6.6)
node(*step6, '装配\n8760', C_MERGE)
node(*calenv, '日型+月\n包络', C_HIST, fs=6.6)
node(*step7, 'Bootstrap\n20->3', C_OUT, fs=6.6)
node(*resid, '残差库\n1826日', C_HIST, fs=6.6)
node(*step8, '归一\npu', C_OUT)

ax.text(38, 7.6, '── 形状线', ha='center', va='center', fontsize=8, color=C_SHAPE[1], fontweight='bold')
ax.text(56, 22.5, '── 电量线', ha='center', va='center', fontsize=8, color=C_ELEC[1], fontweight='bold')
ax.text(20, 38.5, '── 历史全省线', ha='center', va='center', fontsize=8, color=C_HIST[1], fontweight='bold')

caption(XIN, 56, '输入', '(52,2,365,24)', '用户日|全省5年', C_IN[1], fs_shape=6.6)
ax.text(XIN, 65.5, '(5,365,24)MW|(135,6y)|(3,36y,3)', ha='center', va='top', fontsize=6.4,
        color=INK, fontfamily='monospace')
caption(X1, 56, '第一步 行业日', '(24,17386,24)', '用户日归一|等权平均', C_SHAPE[1])
caption(X2, 56, '第二步 medoid', '(24,2,10,24)', 'L1最早日|2023=基准', C_SHAPE[1])
caption(X3, 56, '第三步 未来形状', '(24,6,10,24)', 'a外推|变换链', C_SHAPE[1])
caption(X4, 56, '第四步 25组成', '(3,6,25)', 'P50份额拆分', C_ELEC[1])
caption(X5, 56, '第五步 修正总形状', '(3,6,12,24)', '余量修正|形状冻结', C_MERGE[1])
caption(X6, 56, '第六步 装配', '(3,6,8760)MW', '年月日时|逐级闭合', C_MERGE[1])
caption(X7, 56, '第七步 Bootstrap', '(3,6,20)', '块采样7|选3代表', C_OUT[1])
caption(XO, 56, '第八步 输出', 'load_pu', '每轨迹|年max=1', C_OUT[1])

ax.text(60, 69, '电量定年度·形状定日内·逐级闭合:第六步把未来形状按25组成电量分配到年/月/日/时,逐级闭合<=1e-10;修正组成承接余量,份额和=1;每轨迹年最大=1(残差库与日型/包络同出全省历史)',
        ha='center', va='center', fontsize=8.2, color=INK)

out = Path(__file__).resolve().parent / '算法流程图.png'
fig.savefig(out, bbox_inches='tight', facecolor='white')
print('saved:', out)
