# 负荷系统优化

## 1 背景

> 本章说清项目做什么、输入是什么、输出是什么。

本项目以两个已交付课题为模块做负荷系统优化。课题1《负荷特性主模型》输入历史全省及用户曲线、官方行业电量、上游年度电量 JSON 与三份 config,输出 2025—2060(常用 2030)三情景逐时 8760 负荷曲线与 68 张结果表。课题2《负荷特性预测》输入宏观特征、用电分类表、行业/用户小时曲线与数据中心 15 分钟原始数据,输出 2030 年行业级 8760 逐时曲线。两模块处理逻辑详见 `DESIGN.md` §3/§4,统一口径见 §7。顶层提供服务层(FastAPI,`service/`),把课题1 重构版封装为"上传 Excel → 后台计算 → 轮询状态 → 下载指定情景 P50 逐时曲线"的 HTTP 接口(设计 `DESIGN.md` §5,端点见本文第 7 章)。

## 2 Quick Start

> 本章给环境安装与核心运行命令,均可直接复制执行。

环境安装(conda):

```bash
# 环境一:日常运行与课题2 旧交付包(课题2_预测)全链路,直接用 anaconda base(Python 3.12.7,依赖已齐)
conda activate base

# 环境二:课题1 回归与服务层(逐位一致判据只在锁定版本内成立)
conda create -n load_feature python=3.13
conda activate load_feature
pip install -r requirements.txt

# 环境三:课题2 集成版重构(refactor/load_forecast,锁定版本)
conda create -n load_forecast python=3.12
conda activate load_forecast
pip install -r refactor/load_forecast/requirements.txt
```

课题1 回归与算例(重构版 `refactor/load_model/`;原始交付 `课题1_主模型/` 留作参照):

```bash
conda activate load_feature
cd refactor/load_model
python run_tests.py                                                    # 262 项回归
python run_example.py --output-dir results/<目录>                      # 快速算例(20 抽样)
python run_formal_2025_2060.py --end-year 2030 --output-dir <新目录>   # 正式(1000 抽样)
```

课题2 全链路(集成版重构 `refactor/load_forecast/`,S12;原始交付 `负荷特性预测_软件集成版_20261022/` 留参照;旧交付 `课题2_预测/` 及其重构历史见 git/DESIGN §4.3—§4.5):

```bash
conda activate load_forecast
cd refactor/load_forecast
python pipeline.py --macro-xlsx inputs/111.xlsx \
    --energy-xlsx inputs/行业电量简表.xlsx \
    --load-xlsx inputs/行业历史负荷.xlsx \
    --output-dir outputs/<新目录>          # 六步链,2030 总电量 10159.7817 亿kWh
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
入口层   run_tests.py / run_example.py / run_formal_2025_2060.py   pipeline.py(--macro-xlsx/--energy-xlsx/--load-xlsx/--output-dir)
           ↓                                                     ↓
模块层   refactor/load_model(主模型包;课题1_主模型 留参照)     refactor/load_forecast(六步链;集成版 20261022 留参照)
           ↓                                                     ↓
数据层   inputs/(随包) + 上游 JSON                              三件 Excel(111/行业电量简表/行业历史负荷,输入简化见 DESIGN D18)
           ↓                                                     ↓
结果层   results/(68 表 + 逐时 parquet + 校验 JSON)              outputs/<run>/(02—06 段目录 + 标幺/MW 8760 汇总 + 最终校验)
```

| 模块                             | 职责                                                               | 编写规则                     |
| -------------------------------- | ------------------------------------------------------------------ | ---------------------------- |
| 课题1_主模型                     | 电量→小时分配:典型日演化 + Bootstrap 不确定性(留参照)             | `.claude/skills/` 通用规则 |
| 课题2_预测                       | 2026-10-20 自包含交付包:统计→年度电量→8760 曲线全链路(留参照)    | 同上                         |
| 负荷特性预测_软件集成版_20261022 | 课题2 软件集成正典,六步链(留参照,git 不入)                         | 同上                         |
| refactor/                        | 课题1/课题2 代码级重构正典,行为与原始交付零差异(S8/S12)            | `algo-porting`             |
| service/                         | HTTP 服务层:上传校验→模板转 JSON→后台算法→指定情景 P50 结果提取 | `service-api`              |
| baselines/                       | 基准 md5 底册(重构/移植回归判据)                                   | `algo-porting` §4         |

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

课题2 主链(软件集成版,依据 `DESIGN.md` §4.6):02 宏观特征预测(ARIMA 递归外推 2026—2030)→ 03 总电量与行业电量(MLP×5 种子集成 + 占比趋势外推)→ 04 历史负荷校验标准化 → 05 周法标幺 8760(周峰×日峰×典型日三层)→ 06 MW 换算与总负荷合成(电量闭合 12 项校验)。旧交付包五子链(电量/气温/曲线/分析/扩展)见 `DESIGN.md` §4.3。

## 5 分实体明细

> 本章按模块给实体概览,明细见各自交付文档。

| 类型     | 模块     | 实体                                                                                         |
| -------- | -------- | -------------------------------------------------------------------------------------------- |
| 输入     | 课题1    | 副本山东负荷特性8760数据表、用电分类表、annual_energy_upstream.json、三份 config CSV         |
| 输入     | 课题2    | 集成版三件:111.xlsx、行业电量简表(由分类表提取)、行业历史负荷单文件 10 sheet(由 10 目录合并) |
| 计算单元 | 课题1    | 24 传统行业 + 1 修正组成(形状演化、年度装配、Bootstrap)                                      |
| 计算单元 | 课题2    | 集成版六步链(宏观 ARIMA/MLP 电量/负荷校验/周法标幺/MW 合成;旧包五子链留参照)                 |
| 约束     | 两模块   | 统一口径见`DESIGN.md` §7(单位换算、标幺、自然负荷、timestamp 无时区)                      |
| 服务层   | service/ | 运行记录(runs 表)、上传文件(用户曲线/行业电量/年度电量模板)、结果 CSV;实体明细见第 7 章      |

## 6 汇总输出

> 本章列输出文件与产出模块。

| 文件/目录                                          | 内容                                                                                   | 产出模块 |
| -------------------------------------------------- | -------------------------------------------------------------------------------------- | -------- |
| 课题1_主模型/results/<运行>/数据表/                | 68 张结果表(形状、装配、轨迹、分位、校验)                                              | 课题1    |
| 课题1_主模型/results/<运行>/逐时曲线/              | 省级逐时标幺轨迹 parquet                                                               | 课题1    |
| 课题1_主模型/results/<运行>/正式调用_2030-2030/    | 平台调用子集                                                                           | 课题1    |
| 课题1_主模型/results/example_01/                   | 开发验收算例基准(列集冻结,md5 见 baselines/)                                           | 课题1    |
| 课题2_预测/预测结果_2020-2030/                     | 电量预测 xlsx 与 2020/2021/2023/2030 行业 8760(旧交付包,留参照)                        | 课题2    |
| refactor/load_forecast/outputs/<run></run>/        | 集成版重构输出:02—06 段目录、行业标幺/MW 8760 汇总、总负荷曲线、最终校验(S12)         | 课题2    |
| baselines/*.md5                                    | 基准与源码校验和底册                                                                   | 本项目   |
| service/runs/<run_id>/result/{情景}_p50_hourly.csv | 指定情景(默认基准)P50 逐时标幺曲线(timestamp, load_pu;2025—2030 共 52,584 行)         | 服务层   |
| service/runs/<run_id>/result/2030*.csv             | 课题2 两张结果:2030总负荷_MW及标幺8760 / 2030行业标幺及MW8760_汇总(与冻结基准逐位一致) | 服务层   |
| service/templates/annual_energy.xlsx               | 年度电量模板(两张 sheet,列名=上游 JSON 键原样)                                         | 服务层   |

注:`refactor/load_model/results/` 同构产出上表课题1 文件;`refactor/load_forecast/`(集成版)产出上表课题2 重构行;service 端结果在 `service/runs/<run_id>/` 下,不与课题1 结果混放。

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

**传参**:查询参数 `job_id`(可选;创建运行返回的任务标识)。不传则返回该课题最近 5 条历史任务状态列表(`data` 为数组,create_time 倒序,元素结构与单查相同,2026-10-23 增,课题2 同款见 §7.5):

```bash
curl "http://127.0.0.1:8000/api/load-model/runs/status?job_id=f71037c73c144c78a81dbc3b5e673f5d"
curl "http://127.0.0.1:8000/api/load-model/runs/status"   # → {"data": [{"job_id": …, "status": …, "error": …}, …最近 5 条]}
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

### 7.5 课题2 接口 `/api/load-forecast/runs`(2026-10-23,S13)

四端点与课题1 契约一致(job_id/英文状态名/统一壳),前缀不同;算法在 conda load_forecast 锁定环境以 subprocess 跑 `refactor/load_forecast/pipeline.py`(约 3 分钟),目标年 2030 固定,除三件文件外无其他参数。

**创建运行** `POST /api/load-forecast/runs/create` —— **传参**(multipart/form-data,文件以 binary 传,三件全必传):

```
macro_workbook: binary     #必填;宏观工作簿(.xlsx,即 111.xlsx),首表 1 行表头+8 行指标(1996—2025),首列为指标名
industry_energy: binary    #必填;行业电量简表(.xlsx),单 sheet"行业电量",列 [行业,2020年..2025年],10 行=全社会用电总计+9 行业(行序不限,标签须为标准名)
industry_load: binary      #必填;行业历史负荷(.xlsx),恰好 10 个 sheet:9 行业标准名+其他（扣除重点行业）,每 sheet 列 [时间,值],2022—2023 逐年 8760 行
```

9 行业标准名(冻结序):居民生活、充换电服务业、化学原料和化学制品制造业、有色金属冶炼和压延加工业、汽车制造业、电气机械和器材制造业、非金属矿物制品业、黑色金属冶炼和压延加工业、数据中心。上表结构只做快检(不符→400);深层校验(8760 连续性等)在算法内,不符→状态 failed 并回传原因。

```bash
curl -X POST "http://127.0.0.1:8000/api/load-forecast/runs/create" \
  -F "macro_workbook=@111.xlsx" \
  -F "industry_energy=@行业电量简表.xlsx" \
  -F "industry_load=@行业历史负荷.xlsx"
```

**return**(创建即受理,计算转后台;结构同 §7.1):

```json
{
  "success": true,
  "message": "",
  "data": {
    "job_id": "4073d44f9a044ff083e0510fa5498e51"  // 任务标识,轮询与取结果的钥匙
  },
  "timestamp": "2026-10-23T15:02:41"
}
```

**状态轮询** `GET /api/load-forecast/runs/status?job_id=<任务标识>` —— 返回结构与字段含义同 §7.2(`status`: queued/running/succeeded/failed;`error` 仅 failed 时非空,如"课题2 算法执行失败:RuntimeError: 居民生活/2022: expected 8760 rows, got 5(明细见 …/pipeline.log)")。**不传 job_id 返回课题2 最近 5 条历史任务状态列表**(create_time 倒序,元素结构同单查):

```bash
curl "http://127.0.0.1:8000/api/load-forecast/runs/status"
# → {"data": [{"job_id": "…", "status": "succeeded", "error": null}, {"job_id": "…", "status": "failed", "error": "课题2 算法执行失败:…"}, …最近 5 条]}
```

**结果查看** `GET /api/load-forecast/runs/results?job_id=<任务标识>` —— 返回 `views` 固定三组:总负荷一张、行业 MW 一张、行业标幺一张(行业两视图列名=行业名,列序=9 行业标准名+其他殿后;index 均 8760 个 2030 逐时时刻;JSON 值与结果 CSV 逐位一致;状态非 succeeded 时 409,见 §7.6):

```json
{
  "success": true,
  "message": "",
  "data": {
    "views": [
      {
        "year": 2030,
        "sheet_name": "2030总负荷",
        "index": ["2030-01-01 00:00:00", "2030-01-01 01:00:00"],
        "data": [
          {"column_name": "总负荷MW", "data": [95127.40453451488, 100069.75217169666]},
          {"column_name": "总负荷标幺值", "data": [0.5053690942660869, 0.5316255632738632]}
        ]
      },
      {
        "year": 2030,
        "sheet_name": "2030行业MW",
        "index": ["2030-01-01 00:00:00", "2030-01-01 01:00:00"],
        "data": [
          {"column_name": "居民生活", "data": [17434.097110906427]},
          {"column_name": "充换电服务业", "data": [957.0502950219634]}
          // …共 10 列,其余 8 行业标准名+其他（扣除重点行业）按冻结序
        ]
      },
      {
        "year": 2030,
        "sheet_name": "2030行业标幺",
        "index": ["2030-01-01 00:00:00", "2030-01-01 01:00:00"],
        "data": [
          {"column_name": "居民生活", "data": [0.33786634]}
          // …同上 10 列,值为各行业标幺
        ]
      }
    ]
  },
  "timestamp": "2026-10-23T15:06:03"
}
```

**结果下载** `GET /api/load-forecast/runs/download?job_id=<任务标识>&file=<total|industry>` —— `file=total`(默认)返回 `2030总负荷_MW及标幺8760.csv`(三列:时间、总负荷MW、总负荷标幺值);`file=industry` 返回 `2030行业标幺及MW8760_汇总.csv`(时间+10 行业×{标幺值,MW} 共 21 列,列名形如 `居民生活_标幺值`/`居民生活_MW`):

```bash
curl -o 总负荷.csv "http://127.0.0.1:8000/api/load-forecast/runs/download?job_id=<任务标识>&file=total"
curl -o 行业汇总.csv "http://127.0.0.1:8000/api/load-forecast/runs/download?job_id=<任务标识>&file=industry"
```

### 7.6 错误映射

课题1/课题2 接口共用:

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
[9] service/api/routers/forecast_router.py 课题2 服务路由(同四端点)
[10] service/core/manager/forecast_run_manager.py 课题2 运行组织(subprocess 调 load_forecast 环境)
