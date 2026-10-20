# 负荷系统优化

## 1 背景

> 本章说清项目做什么、输入是什么、输出是什么。

本项目以两个已交付课题为模块做负荷系统优化。课题1《负荷特性主模型》输入历史全省及用户曲线、官方行业电量、上游年度电量 JSON 与三份 config,输出 2025—2060(常用 2030)三情景逐时 8760 负荷曲线与 68 张结果表。课题2《负荷特性预测》输入宏观特征、用电分类表、行业/用户小时曲线与数据中心 15 分钟原始数据,输出 2030 年行业级 8760 逐时曲线。两模块处理逻辑与口径约束详见 `DESIGN.md` §3—§5。

## 2 Quick Start

> 本章给环境安装与核心运行命令,均可直接复制执行。

环境(conda):

```bash
# 日常运行与课题2 全链路:直接用 anaconda base(Python 3.12.7,依赖已齐)
# 课题2 须加 PYTHONNOUSERSITE=1:user-site pandas 3.0.5 与 statsmodels 0.14.2 冲突(DESIGN D13)
C:/ProgramData/anaconda3/python.exe

# 课题1 基准复现与回归(逐位一致需锁定版本):
conda create -n load_feature python=3.13
C:/Users/weiyutao/.conda/envs/load_feature/python.exe -m pip install -r requirements.txt
```

课题1 回归与算例:

```bash
cd 课题1_主模型
C:/Users/weiyutao/.conda/envs/load_feature/python.exe run_tests.py            # 262 项回归
C:/Users/weiyutao/.conda/envs/load_feature/python.exe run_example.py --output-dir results/<目录>   # 快速算例(20 抽样)
C:/Users/weiyutao/.conda/envs/load_feature/python.exe run_formal_2025_2060.py --end-year 2030 --output-dir <新目录>  # 正式(1000 抽样)
```

课题2 环境检查与全链路(2026-10-20 自包含交付包,数据已内置):

```bash
cd 课题2_预测
PYTHONNOUSERSITE=1 C:/ProgramData/anaconda3/python.exe 10_扩展行业/check_environment.py   # 环境检查
PYTHONNOUSERSITE=1 C:/ProgramData/anaconda3/python.exe run_delivery.py                     # 一键全链路(--skip-arima 跳过电量训练)
```

## 3 架构

> 本章给模块划分与依赖方向(按项目实际裁剪)。

```
入口层   run_tests.py / run_example.py / run_formal_2025_2060.py   run_delivery.py / check_environment.py / run_original_extensions.py
           ↓                                                     ↓
模块层   课题1_主模型(workstreams_load_characteristic_model_v2)   课题2_预测(01/03/07/08/10 五个子链)
           ↓                                                     ↓
数据层   inputs/(随包) + 上游 JSON                              课题2_预测/(数据内置,02—06、09 目录)
           ↓                                                     ↓
结果层   results/(68 表 + 逐时 parquet + 校验 JSON)              预测结果_2020-2030/ 与曲线目录
```

| 模块 | 职责 | 编写规则 |
| --- | --- | --- |
| 课题1_主模型 | 电量→小时分配:典型日演化 + Bootstrap 不确定性 | `.claude/skills/` 通用规则 |
| 课题2_预测 | 原始统计→年度电量→8760 曲线全链路 | 同上 |
| baselines/ | 基准 md5 底册(重构/移植回归判据) | `algo-porting` §4 |

## 4 算法全流程

> 本章是端到端数据流主线:读数 → 计算 → 结果文件。

课题1 主链(依据 `DESIGN.md` §5.1):

| 步骤 | 输入 | 中间处理 | 输出 | 模块 |
| --- | --- | --- | --- | --- |
| 1 输入读取校验 | 六项输入(Excel/JSON/CSV) | 质量门控:24 点非负/总量正/非平线,剔除留诊断 | 有效用户日 | 主模型包 [1] |
| 2 典型日提取 | 有效用户日 | 归一 + 同行业同日等权平均,K=1 L1 medoid(2023 为未来基准) | 行业典型日形状 | 主模型包 [1] |
| 3 形状演化 | 典型日 + 权重演化 config | 四时段 H=8 指数饱和外推 + phi=0.8 Logit 阻尼峰谷调整(带门槛) | 目标年形状 | 主模型包 [1] |
| 4 年度装配 | 上游 JSON 自然负荷 P50 + 演化形状 | 24 行业 + 修正组成定标到年度电量 | 逐时 MW | 主模型包 [1] |
| 5 Bootstrap | 逐时曲线 | 每年每情景 1000 轨迹(块长 7,固定种子) | 轨迹与分位 | 主模型包 [1] |
| 6 输出与校验 | 全部结果 | 68 表 + 逐时 parquet + 展示子集 + 独立发布校验 | results/ | 主模型包 [1] |

课题2 主链(依据 `DESIGN.md` §4.3):01 电量预测(`01_电量预测/`,ARIMA+MLP)→ 02 气温敏感性(`03_气温敏感性/`,基温/CDD/HDD 三情景)→ 03 曲线生成(`07_8760曲线/`,行业标幺 × 目标年电量 → MW)→ 04 结果分析(`08_结果分析/`,只读)→ 05 扩展行业(`10_扩展行业/`,数据中心/充换电独立子链 + 恒等式校验)。逐步输入输出明细见 `课题2_预测/README.md`。

## 5 分实体明细

> 本章按模块给实体概览,明细见各自交付文档。

| 类型 | 模块 | 实体 |
| --- | --- | --- |
| 输入 | 课题1 | 副本山东负荷特性8760数据表、用电分类表、annual_energy_upstream.json、三份 config CSV |
| 输入 | 课题2 | 111.xlsx、用电分类表、行业小时曲线、用户级曲线、数据中心 15 分钟 Excel、生成器输入 |
| 计算单元 | 课题1 | 24 传统行业 + 1 修正组成(形状演化、年度装配、Bootstrap) |
| 计算单元 | 课题2 | 五个子链(电量、温度、曲线、分析、扩展行业;目录 01/03/07/08/10) |
| 约束 | 两模块 | 统一口径见 `DESIGN.md` §4(单位换算、标幺、自然负荷、timestamp 无时区) |

## 6 汇总输出

> 本章列输出文件与产出模块。

| 文件/目录 | 内容 | 产出模块 |
| --- | --- | --- |
| 课题1_主模型/results/<运行>/数据表/ | 68 张结果表(形状、装配、轨迹、分位、校验) | 课题1 |
| 课题1_主模型/results/<运行>/逐时曲线/ | 省级逐时标幺轨迹 parquet | 课题1 |
| 课题1_主模型/results/<运行>/正式调用_2030-2030/ | 平台调用子集 | 课题1 |
| 课题1_主模型/results/example_01/ | 开发验收算例基准(列集冻结,md5 见 baselines/) | 课题1 |
| 课题2_预测/预测结果_2020-2030/ | 电量预测 xlsx 与 2020/2021/2023/2030 行业 8760 | 课题2 |
| 课题2_预测/2030行业有名值曲线/ 等 | 2030 行业标幺/有名值/剩余曲线(含校验,见 DESIGN §4.4) | 课题2 |
| baselines/*.md5 | 基准与源码校验和底册 | 本项目 |

## 7 接口

> 本项目非服务项目,无服务端点;CLI 与 Python 入口见第 2 章命令。

## 8 Reference

> 本章是正文引用映射,只增不复用。

<!-- 入口 -->
[1] 课题1_主模型/workstreams_load_characteristic_model_v2/ 主模型包(pipeline.py 为组装入口)
[2] 课题1_主模型/run_tests.py 回归入口
[3] 课题2_预测/run_delivery.py 一键全链路入口
[4] 课题2_预测/10_扩展行业/check_environment.py 环境检查
[5] 课题2_预测/10_扩展行业/run_original_extensions.py 扩展行业参数化入口
