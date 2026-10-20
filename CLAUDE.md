# 负荷系统优化 · 开发指南

> 修改规则见 skill `doc-roles`：文档分工/环境约定段非用户明示不可改；
> 项目事实与当前进度段随项目状态更新。

## 文档分工

- 通用项目规则在 `.claude/skills/`（18 条），改任何文件前先读相关规则
- `DESIGN.md`：本项目唯一设计与规划文件（设计/规划/进度/决策）
- `README.md`：稳定的项目说明，收口时按 `readme-structure` 定稿

## 环境约定

- Python 用 conda 环境

## 项目事实与当前进度

- 模块:`课题1_主模型`(电量→逐时分配)、`课题2_预测`(2026-10-20 自包含交付包:统计→电量→8760 全链路,数据内置,git 只入代码);设计/规划/进度/决策一律见 DESIGN.md
- 环境:日常与课题2 用 anaconda base(Python 3.12.7);课题2 运行须加 `PYTHONNOUSERSITE=1`(user-site pandas 3.0.5 与 statsmodels 0.14.2 冲突,见 DESIGN D13);课题1 基准复现与回归用 conda 环境 load_feature(Python 3.13 + 交付锁定版,逐位一致判据只在锁定环境内成立)
- 进度(2026-10-20):S7/S8/S9 收口——课题1 重构版在 `refactor/load_model/`,原始交付 `课题1_主模型/` 留作参照;课题2 已换自包含包并全链路跑通(run_delivery.py,恒等式/重建比对通过),重构待启动(`refactor/load_forecast/`);待办 S5(优化方向确认)

