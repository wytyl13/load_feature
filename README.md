# 负荷系统优化

## 1 背景

> 本章说清项目做什么、输入是什么、输出是什么。

本项目以两个已交付课题为模块做负荷系统优化。课题1《负荷特性主模型》输入历史全省及用户曲线、官方行业电量、上游年度电量 JSON 与三份 config,输出 2025—2060(常用 2030)三情景逐时 8760 负荷曲线与 68 张结果表。课题2《负荷特性预测》输入宏观特征、用电分类表、行业/用户小时曲线与数据中心 15 分钟原始数据,输出 2030 年行业级 8760 逐时曲线。两模块处理逻辑详见 `DESIGN.md` §3/§4,统一口径见 §7。顶层提供服务层(FastAPI,`service/`),把课题1 重构版封装为"上传 Excel → 后台计算 → 轮询状态 → 下载指定情景 P50 逐时曲线"的 HTTP 接口(设计 `DESIGN.md` §5,端点见本文第 7 章)。

## 2 Quick Start

> 本章给环境安装与核心运行命令,均可直接复制执行。

环境安装(conda):

```bash
# 环境一:日常运行与课题2 全链路,直接用 anaconda base(Python 3.12.7,依赖已齐)
conda activate base

# 环境二:课题1 回归与服务层(逐位一致判据只在锁定版本内成立)
conda create -n load_feature python=3.13
conda activate load_feature
pip install -r requirements.txt
```

课题1 回归与算例(重构版 `refactor/load_model/`;原始交付 `课题1_主模型/` 留作参照):

```bash
conda activate load_feature
cd refactor/load_model
python run_tests.py                                                    # 262 项回归
python run_example.py --output-dir results/<目录>                      # 快速算例(20 抽样)
python run_formal_2025_2060.py --end-year 2030 --output-dir <新目录>   # 正式(1000 抽样)
```

课题2 环境检查与全链路(重构版 `refactor/load_forecast/`,数据已内置;原始交付 `课题2_预测/` 留作参照):

```bash
conda activate load_feature
$env:PYTHONNOUSERSITE = "1"   # 本会话生效;user-site pandas 3.0.5 与 statsmodels 0.14.2 冲突(DESIGN D13)
cd refactor/load_forecast
python 10_扩展行业/check_environment.py   # 环境检查
python run_delivery.py                     # 一键全链路(--skip-arima 跳过电量训练)
```

服务层启动(FastAPI,封装 refactor/load_model;前两行首次执行一次):

```bash
conda activate load_feature
pip install -r service/requirements.txt        # 服务依赖
pip install -e refactor/load_model --no-deps   # 算法包 editable 安装
cd service
python start.py --port 8000                   # 启动,默认 127.0.0.1:8000
```

## 3 架构

> 本章给模块划分与依赖方向(按项目实际裁剪)。

```
服务层   service/ start.py(FastAPI):上传 Excel → 后台调 run_pipeline → 轮询状态 → 下载结果 CSV
           ↓ 只调算法包唯一入口 run_pipeline(editable 安装)
入口层   run_tests.py / run_example.py / run_formal_2025_2060.py   run_delivery.py / check_environment.py / run_original_extensions.py
           ↓                                                     ↓
模块层   refactor/load_model(主模型包;课题1_主模型 留参照)     refactor/load_forecast(五子链;课题2_预测 留参照)
           ↓                                                     ↓
数据层   inputs/(随包) + 上游 JSON                              数据内置(02—06、09 目录)
           ↓                                                     ↓
结果层   results/(68 表 + 逐时 parquet + 校验 JSON)              预测结果_2020-2030/ 与曲线目录
```

| 模块         | 职责                                                           | 编写规则                     |
| ------------ | -------------------------------------------------------------- | ---------------------------- |
| 课题1_主模型 | 电量→小时分配:典型日演化 + Bootstrap 不确定性(留参照)         | `.claude/skills/` 通用规则 |
| 课题2_预测   | 原始统计→年度电量→8760 曲线全链路(留参照)                    | 同上                         |
| refactor/    | 课题1/课题2 代码级重构正典,行为与原始交付零差异(S8/S10)        | `algo-porting`             |
| service/     | HTTP 服务层:上传校验→模板转 JSON→后台算法→指定情景 P50 结果提取 | `service-api`              |
| baselines/   | 基准 md5 底册(重构/移植回归判据)                               | `algo-porting` §4         |

## 4 算法全流程

> 本章是端到端数据流主线:读数 → 计算 → 结果文件。

课题1 主链(依据 `DESIGN.md` §3.3):

| 步骤           | 输入                              | 中间处理                                                     | 输出           | 模块         |
| -------------- | --------------------------------- | ------------------------------------------------------------ | -------------- | ------------ |
| 1 输入读取校验 | 六项输入(Excel/JSON/CSV)          | 质量门控:24 点非负/总量正/非平线,剔除留诊断                  | 有效用户日     | 主模型包 [1] |
| 2 典型日提取   | 有效用户日                        | 归一 + 同行业同日等权平均,K=1 L1 medoid(2023 为未来基准)     | 行业典型日形状 | 主模型包 [1] |
| 3 形状演化     | 典型日 + 权重演化 config          | 四时段 H=8 指数饱和外推 + phi=0.8 Logit 阻尼峰谷调整(带门槛) | 目标年形状     | 主模型包 [1] |
| 4 年度装配     | 上游 JSON 自然负荷 P50 + 演化形状 | 24 行业 + 修正组成定标到年度电量                             | 逐时 MW        | 主模型包 [1] |
| 5 Bootstrap    | 逐时曲线                          | 每年每情景 1000 轨迹(块长 7,固定种子)                        | 轨迹与分位     | 主模型包 [1] |
| 6 输出与校验   | 全部结果                          | 68 表 + 逐时 parquet + 展示子集 + 独立发布校验               | results/       | 主模型包 [1] |

课题2 主链(依据 `DESIGN.md` §4.3):01 电量预测(`01_电量预测/`,ARIMA+MLP)→ 02 气温敏感性(`03_气温敏感性/`,基温/CDD/HDD 三情景)→ 03 曲线生成(`07_8760曲线/`,行业标幺 × 目标年电量 → MW)→ 04 结果分析(`08_结果分析/`,只读)→ 05 扩展行业(`10_扩展行业/`,数据中心/充换电独立子链 + 恒等式校验)。逐步输入输出明细见 `refactor/load_forecast/README.md`。

## 5 分实体明细

> 本章按模块给实体概览,明细见各自交付文档。

| 类型     | 模块     | 实体                                                                                    |
| -------- | -------- | --------------------------------------------------------------------------------------- |
| 输入     | 课题1    | 副本山东负荷特性8760数据表、用电分类表、annual_energy_upstream.json、三份 config CSV    |
| 输入     | 课题2    | 111.xlsx、用电分类表、行业小时曲线、用户级曲线、数据中心 15 分钟 Excel、生成器输入      |
| 计算单元 | 课题1    | 24 传统行业 + 1 修正组成(形状演化、年度装配、Bootstrap)                                 |
| 计算单元 | 课题2    | 五个子链(电量、温度、曲线、分析、扩展行业;目录 01/03/07/08/10)                          |
| 约束     | 两模块   | 统一口径见`DESIGN.md` §7(单位换算、标幺、自然负荷、timestamp 无时区)                 |
| 服务层   | service/ | 运行记录(runs 表)、上传文件(用户曲线/行业电量/年度电量模板)、结果 CSV;实体明细见第 7 章 |

## 6 汇总输出

> 本章列输出文件与产出模块。

| 文件/目录                                            | 内容                                                                  | 产出模块 |
| ---------------------------------------------------- | --------------------------------------------------------------------- | -------- |
| 课题1_主模型/results/<运行>/数据表/                  | 68 张结果表(形状、装配、轨迹、分位、校验)                             | 课题1    |
| 课题1_主模型/results/<运行>/逐时曲线/                | 省级逐时标幺轨迹 parquet                                              | 课题1    |
| 课题1_主模型/results/<运行>/正式调用_2030-2030/      | 平台调用子集                                                          | 课题1    |
| 课题1_主模型/results/example_01/                     | 开发验收算例基准(列集冻结,md5 见 baselines/)                          | 课题1    |
| 课题2_预测/预测结果_2020-2030/                       | 电量预测 xlsx 与 2020/2021/2023/2030 行业 8760                        | 课题2    |
| 课题2_预测/2030行业有名值曲线/ 等                    | 2030 行业标幺/有名值/剩余曲线(含校验,见 DESIGN §4.4)                 | 课题2    |
| baselines/*.md5                                      | 基准与源码校验和底册                                                  | 本项目   |
| service/runs/<run_id>/result/{情景}_p50_hourly.csv | 指定情景(默认基准)P50 逐时标幺曲线(timestamp, load_pu;2025—2030 共 52,584 行) | 服务层   |
| service/templates/annual_energy.xlsx                 | 年度电量模板(两张 sheet,列名=上游 JSON 键原样)                        | 服务层   |

注:`refactor/load_model/results/` 与 `refactor/load_forecast/` 同构产出上表课题1/课题2 文件(正典运行入口,见第 2 章);service 端结果在 `service/runs/<run_id>/` 下,不与课题1 结果混放。

## 7 接口

> 本章给服务层 HTTP 端点的传参与返回示例(启动命令见第 2 章;业务规则见 `DESIGN.md` §5)。
> 所有响应为统一壳 `{success, message, data, timestamp}`;计算任务全局串行,并发创建的运行排队执行(快速档约 9 分钟,正式档更长)。

### 7.1 创建运行 `POST /api/load-model/runs/create`

**传参**(multipart/form-data,文件以 binary 传):

```
load_workbook: binary      #必填;用户日曲线工作簿(.xlsx),必含"历史负荷数据（2019-2023）"sheet(括号为全角,与交付原表一致)
official_workbook: binary  #必填;全社会用电分类表(.xlsx,行业×2020—2025)
annual_energy: binary      #可选;年度电量模板(.xlsx,两张 sheet,列名=上游 JSON 键原样),不传用服务端内置默认
start_year: 2025           #可选,默认 2025;规划起始年(2025—2060)
end_year: 2030             #可选,默认 2030;规划结束年
sample_size: 20            #可选,默认 20;Bootstrap 抽样数,只允许 20(快速)/1000(正式)
scenario: 基准             #可选,默认 基准;规划情景 低碳/基准/高电气化,单算该情景并返回其 P50
```

```bash
curl -X POST http://127.0.0.1:8000/api/load-model/runs/create \
  -F "load_workbook=@副本山东负荷特性8760数据表.xlsx" \
  -F "official_workbook=@山东2020-2025_全社会用电分类表（全口径）.xlsx" \
  -F "annual_energy=@annual_energy.xlsx" \
  -F "start_year=2025" -F "end_year=2030" -F "sample_size=20" -F "scenario=基准"
```

**return**(创建即受理,计算转后台):

```json
{
  "success": true,    // 必返;true 成功/false 失败
  "message": "",      // 必返;失败为可直接展示的中文原因
  "data": {
    "job_id": "f71037c73c144c78a81dbc3b5e673f5d"  // 必返;任务标识,轮询与取结果的钥匙
  },
  "timestamp": "2026-10-21T15:51:20"  // 必返;响应时刻(ISO 格式,精确到秒)
}
```

### 7.2 状态轮询 `GET /api/load-model/runs/status?job_id=<任务标识>`

**传参**:查询参数 `job_id`(必填;创建运行返回的任务标识):

```bash
curl "http://127.0.0.1:8000/api/load-model/runs/status?job_id=f71037c73c144c78a81dbc3b5e673f5d"
```

**return**(succeeded 示例):

```json
{
  "success": true,    // 必返;true 成功/false 失败
  "message": "",      // 必返;失败为可直接展示的中文原因
  "data": {
    "job_id": "f71037c73c144c78a81dbc3b5e673f5d",  // 必返;任务标识
    "status": "succeeded",  // 必返;queued 排队/running 运行中/succeeded 成功/failed 失败
    "error": null     // 必返;失败原因,仅 failed 时非空
  },
  "timestamp": "2026-10-21T16:00:12"  // 必返;响应时刻(ISO 格式,精确到秒)
}
```

### 7.3 结果查看 `GET /api/load-model/runs/results?job_id=<任务标识>`

```bash
curl "http://127.0.0.1:8000/api/load-model/runs/results?job_id=f71037c73c144c78a81dbc3b5e673f5d"
```

**return**(views 展示 1 组,实际每年一组;每年曲线按本年最大标幺,建议前端分年渲染):

```json
{
  "success": true,    // 必返;true 成功/false 失败
  "message": "",      // 必返;失败为可直接展示的中文原因
  "data": {
    "views": [        // 必返;每年一组,一组一张图
      {
        "year": 2025, // 该组年份
        "sheet_name": "基准_p50_2025",  // 组名(=情景_文件名_年份)
        "index": ["2025-01-01 00:00:00", "2025-01-01 01:00:00"],  // 该年逐时时刻,与 data 等长
        "data": [     // 列数组(单曲线不分序列/参考线)
          {"column_name": "load_pu", "data": [0.505507618705083, 0.48492261178380186]}  // 标幺负荷,值与 CSV 逐位一致
        ]
      }
    ]
  },
  "timestamp": "2026-10-21T17:39:17"  // 必返;响应时刻(ISO 格式,精确到秒)
}
```

### 7.4 结果下载 `GET /api/load-model/runs/download?job_id=<任务标识>`

```bash
curl -o 基准_p50_hourly.csv "http://127.0.0.1:8000/api/load-model/runs/download?job_id=f71037c73c144c78a81dbc3b5e673f5d"
```

返回 CSV 文件(UTF-8),两列:`timestamp`(逐时时刻)与 `load_pu`(指定情景 P50 轨迹标幺负荷,年最大=1),文件名 `{情景}_p50_hourly.csv`;默认参数下 52,584 行(2025—2030,含闰年 2028 的 8784 行)。

### 7.5 错误映射

| HTTP 状态码 | 触发条件                                            | message 内容                              |
| ----------- | --------------------------------------------------- | ----------------------------------------- |
| 400         | 参数越界/文件扩展名或大小不符/模板缺 sheet 或含空值 | 中文,指明字段名或文件名                   |
| 404         | job_id 不存在                                       | 未找到运行记录:<job_id>                   |
| 409         | 状态非成功时请求结果                                | 当前状态与提示                            |
| 500         | 未捕获异常                                          | Internal Server Error: 原因(堆栈只进日志) |

## 8 Reference

> 本章是正文引用映射,只增不复用。

<!-- 入口 -->

[1] 课题1_主模型/workstreams_load_characteristic_model_v2/ 主模型包(pipeline.py 为组装入口)
[2] 课题1_主模型/run_tests.py 回归入口
[3] 课题2_预测/run_delivery.py 一键全链路入口
[4] 课题2_预测/10_扩展行业/check_environment.py 环境检查
[5] 课题2_预测/10_扩展行业/run_original_extensions.py 扩展行业参数化入口

<!-- 服务层 -->

[6] service/start.py 服务启动入口
[7] service/api/routers/run_router.py 服务路由(创建/轮询/结果查看/下载四端点)
[8] service/core/manager/run_manager.py 运行组织与算法包唯一调用方
