# -*- coding: utf-8 -*-
"""气温—负荷关系 示意图 (docx【五】要求): 左=五步箭头链, 右=24h负荷曲线缩放示意"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_temp_page_work')
os.makedirs(OUT, exist_ok=True)
GREEN, BLUE, GREY = '#008279', '#1F6FB2', '#606060'

fig = plt.figure(figsize=(4.31, 2.43), dpi=200)
fig.patch.set_facecolor('#FFFFFF')

# ---- 左: 五步箭头链 ----
ax1 = fig.add_axes([0.01, 0.02, 0.40, 0.96]); ax1.axis('off')
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
steps = ['气温', 'CDD / HDD', 'β', '日电量变化', '24小时\n曲线缩放']
n = len(steps)
h, gap = 0.145, 0.055
y0 = (1 - (n * h + (n - 1) * gap)) / 2
for i, t in enumerate(steps):
    y = y0 + (n - 1 - i) * (h + gap)
    ax1.add_patch(FancyBboxPatch((0.08, y), 0.84, h, boxstyle='round,pad=0,rounding_size=0.05',
                                 facecolor='#E6F2F1', edgecolor=GREEN, linewidth=1.1))
    ax1.text(0.50, y + h / 2, t, ha='center', va='center', fontsize=9.5,
             color=GREEN, fontweight='bold', linespacing=1.15)
    if i < n - 1:
        ax1.add_patch(FancyArrowPatch((0.50, y - 0.004), (0.50, y - gap + 0.004),
                                      arrowstyle='-|>', mutation_scale=9, color=GREEN, linewidth=1.1))

# ---- 右: 24h 负荷曲线缩放示意 ----
ax2 = fig.add_axes([0.47, 0.20, 0.51, 0.62])
x = np.arange(0, 24)
shape = 0.62 + 0.20 * np.exp(-((x - 10) ** 2) / 8) + 0.36 * np.exp(-((x - 19) ** 2) / 5)
shape = shape / shape.max()
ax2.plot(x, shape, color=GREY, lw=2.0, label='正常温度')
ax2.plot(x, np.clip(shape * 1.13, 0, 1.16), color='#D62728', lw=1.8, ls='--', label='高温')
ax2.plot(x, shape * 0.88, color=BLUE, lw=1.8, ls='--', label='低温')
ax2.set_xticks([]); ax2.set_yticks([])
for s in ('top', 'right'):
    ax2.spines[s].set_visible(False)
for s in ('left', 'bottom'):
    ax2.spines[s].set_color('#BDBDBD')
ax2.set_xlabel('当天 24 小时', fontsize=8, color=GREY, labelpad=1)
ax2.set_ylabel('负荷', fontsize=8, color=GREY, labelpad=2)
ax2.legend(fontsize=7.5, frameon=False, loc='upper left', handlelength=1.4, borderpad=0.1)
ax2.text(0.5, -0.30, '高温→负荷上升　低温→负荷下降', transform=ax2.transAxes,
         ha='center', fontsize=8.5, color=BLUE, fontweight='bold')
ax2.set_title('按电量变化整体缩放当天曲线', fontsize=8.5, color=GREY, pad=3)

PNG = os.path.join(OUT, '气温负荷示意图.png')
fig.savefig(PNG, dpi=200, facecolor='#FFFFFF')
plt.close(fig)
print('已输出:', PNG, plt.imread(PNG).shape)
