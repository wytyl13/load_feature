# 山东负荷特性预测完整交付包

本目录是自包含运行版。代码、历史输入、温度数据、用户级曲线、数据中心原始 Excel 和充换电/数据中心生成器均已放在本目录内；运行时不需要原工程的 `自己处理`、`存档`、`2026-08-12` 或其他外部项目目录。

> 本副本为 S10 代码级重构版（`refactor/load_forecast/`）：7 个模块级流程脚本已函数化（`main()` 编排，`build_new_8760_ccd_原始.py` 拆为 main + 7 段函数），算法、CLI 与输出逐位未变——全新全链重跑后与原包交付零差异（判据 139 文件 + 107 图，D14 归一化口径）。运行方式与下文完全一致。

## 一、目录结构

```text
01_电量预测/             ARIMA 特征外推、MLP 电量预测、回测
02_统计与电量锚点/       111.xlsx 和山东行业电量分类表
03_气温敏感性/           温度 CSV/Excel、基温、CDD/HDD 弹性、2030 温度情景
04_实测负荷8760/         2022/2023 行业 8760 曲线及总负荷输入
05_用户级负荷曲线/       用户级行业曲线（2022、2023）
06_生成器输入/           周—日—小时生成器输入副本
07_8760曲线/             用户合成、2030 标幺曲线及有名值换算
08_结果分析/             相关性、月度矩阵和示意图
09_数据中心原始/         2022—2023 每月 15 分钟数据中心 Excel
10_扩展行业/             数据中心、充换电、统一复跑入口
project_paths.py         全部脚本共用的包内路径配置
arima_compat.py          新旧 statsmodels 的 ARIMA 兼容层
run_delivery.ps1         推荐的一键入口
```

压缩包内只保留一个 Markdown 文件，即本 `README.md`。温度文件为：

- `03_气温敏感性/temperature_daily_shandong.csv`：山东 16 市等权日均温，2020—2023；
- `03_气温敏感性/山东16市逐日气温_ERA5_2020-2023.xlsx`：温度原始/审计副本。

## 二、环境准备

需要 Python 3.7 及以上和 `requirements.txt` 中的依赖：numpy、pandas、scipy、statsmodels、scikit-learn、openpyxl、matplotlib。旧版 statsmodels 没有 `statsmodels.tsa.arima.model` 时，`arima_compat.py` 自动切换旧 API，并对奇异序列提供线性外推兜底。

```powershell
python -m pip install -r requirements.txt
```

检查包内所有关键输入：

```powershell
python 10_扩展行业/check_environment.py
```

## 三、推荐完整运行

```powershell
.\run_delivery.ps1 -Python "D:\App\Anaconda3\python.exe"
```

该入口会依次完成：

1. 数据中心 15 分钟 Excel 重建为小时 8760 曲线；
2. 2022/2023 行业和剩余负荷恒等式校验；
3. 充换电、数据中心 2030 标幺曲线生成；
4. 1996—2025 特征 ARIMA 外推、MLP 电量预测及留出集回测。

只运行扩展行业，不重新训练电量模型：

```powershell
python 10_扩展行业/run_original_extensions.py --skip-arima
```

所有结果写入 `10_扩展行业/验证结果_原始数据/`，不会覆盖包内输入。

## 四、各阶段单独运行

### 1. 电量预测

```powershell
python 01_电量预测/forecast_arima_ml.py
python 01_电量预测/build_nn_forecast_csv.py
python 01_电量预测/forecast_arima.py
python 01_电量预测/forecast_arima_growth.py
```

输出默认位于 `预测结果_2020-2030/`。

### 2. 气温敏感性

```powershell
python 03_气温敏感性/build_base_kink.py
python 03_气温敏感性/build_base_scan_fig.py
python 03_气温敏感性/build_temperature_elasticity.py
python 03_气温敏感性/build_temperature_years.py
```

顺序不能颠倒：基温判定 JSON 会被弹性脚本读取，温度情景随后用于 2030 分析。

### 3. 行业 8760 曲线

```powershell
python 07_8760曲线/合成用户成行业标幺8760曲线.py
python 07_8760曲线/批量生成2030行业8760标幺曲线_三假前当后周.py
python 07_8760曲线/2030行业标幺曲线转有名值.py
python 07_8760曲线/生成扣除重点行业剩余标幺曲线.py
```

核心换算为：

```text
P(MW,t) = pu(t) × E(亿kWh) × 100000 / Σpu(t)
```

因此生成的有名值曲线满足 `ΣP(MW)/100000 = 年电量(亿kWh)`。

### 4. 数据中心和充换电

```powershell
python 10_扩展行业/build_data_center_8760_原始.py
python 10_扩展行业/build_new_8760_ccd_原始.py 2030
python 10_扩展行业/_ccd23_20260917_原始.py
```

数据中心脚本使用包内 `09_数据中心原始/`，按 16 站口径处理；充换电脚本使用包内 `05_用户级负荷曲线/` 或 `06_生成器输入/`，不会再寻找历史临时目录。

## 五、验收标准

- `check_environment.py` 返回“检查通过”；
- 所有历史/预测小时曲线均为 8760 点、时间连续、无空值；
- 数据中心重建与包内历史曲线的最大差不超过保存精度；
- 八行业、数据中心、九行业剩余负荷恒等式通过；
- 2030 有名值年电量与对应锚点一致；
- ARIMA+MLP 主脚本返回退出码 0，并生成 `ARIMA预测2026-2030.xlsx`。

旧版模型会输出 statsmodels 的收敛提示，这些是模型诊断警告，不代表运行失败；以脚本退出码和输出文件为准。

## 六、输入数据说明

包内数据是当前复现口径的冻结副本。2022 年充换电曲线前期大量为零属于原始数据事实，不在交付版中自动平滑。2030 年法定节假日安排尚未正式发布时，生成器使用代码内的规划日期，并在汇总文件中保留节假日模式。

如需替换数据，优先替换相同相对路径的文件；不建议修改脚本中的路径。所有脚本的根路径集中在 `project_paths.py`，从任意当前工作目录启动都能定位到包内文件。
