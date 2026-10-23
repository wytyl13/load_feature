# 课题2 负荷预测(软件集成版重构)

原包 `负荷特性预测_软件集成版_20261022` 六步链的重构版,计算行逐段一致,
全链 89 个判据文件与本机冻结基准逐位一致(DESIGN §4.6/D18)。

## 输入(三件 Excel,显式传参)

| 文件 | 内容 |
|---|---|
| 111.xlsx | 8 指标 × 1996–2025 年宏观与电量(原样使用) |
| 行业电量简表.xlsx | 单 sheet"行业电量",10 行=全社会用电总计+9 行业 × 2020–2025 年,万千瓦时(替代 150 行分类表) |
| 行业历史负荷.xlsx | 10 sheet=10 行业 × 17520 行 [时间, 值](2022+2023,替代 10 目录 × 2 CSV) |

## 结构

```
config.py          # 默认配置+resolve_config(原 config.json 收编,industries 有序表)
data_io.py         # 三件 Excel 读入+信任边界校验
macro_forecast.py  # 02 ARIMA 宏观特征外推
energy_forecast.py # 03 MLP 总电量+行业占比外推
history_prep.py    # 04 历史负荷校验标准化落盘
weekly_generator.py# 05 周法核心(整文件零修改,md5 同原包)
curve_generation.py# 05 编排
scaling_synthesis.py# 06 MW 换算与总负荷合成(12 项最终校验)
pipeline.py        # run_forecast() 唯一入口 + CLI
```

## 运行

```bash
# conda env: load_forecast(requirements.txt 锁定版)
python pipeline.py --macro-xlsx inputs/111.xlsx \
    --energy-xlsx inputs/行业电量简表.xlsx \
    --load-xlsx inputs/行业历史负荷.xlsx \
    --output-dir <空目录>
```

库内调用:`run_forecast(macro, energy, load, output_dir, *, config_overrides=None) -> dict[str, Path]`,
返回 key:output_dir / log / manifest / industry_per_unit_summary / industry_mw_summary /
total_load_curve / total_metrics / final_validation。

## 验证口径

`../common/compare_forecast_run.py compare <run_dir> ../../baselines/负荷预测集成版_全链_重构后.md5`;
2030 总电量 10159.7817 亿kWh(本机锁定环境;README 原文 10158.4246 为作者平台数值差异,见 DESIGN D18)。
