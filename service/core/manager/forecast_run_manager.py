#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/10/23
@Author  : weiyutao
@File    : forecast_run_manager.py

课题2 运行业务组织:三件上传校验与保存、subprocess 执行算法、结果提取与视图。
算法调用方式与课题1(进程内 import)不同:课题2 依赖锁定在 conda load_forecast
环境(DESIGN D18),服务经 subprocess 调 pipeline.py CLI,不做进程内 import。
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import shutil
import traceback
import uuid
from pathlib import Path

import openpyxl
import pandas as pd

from core.exceptions import ConflictError, NotFoundError
from core.paths import (
    FORECAST_PIPELINE,
    FORECAST_PYTHON,
    INDUSTRY_ENERGY_MAX_BYTES,
    INDUSTRY_LOAD_MAX_BYTES,
    MACRO_WORKBOOK_MAX_BYTES,
    RUNS_DIR,
)
from core.schema.run_dto import RunCreateDTO, RunStatusResp, RunUpdateDTO
from core.service.run_service import RunService
from core.table.run_record import RUN_STATUS_LABELS, RUN_STATUS_NAMES, RunRecord, RunStatus
from core.manager.run_manager import RunManager, UploadedFile

ENERGY_SHEET_NAME = "行业电量"                  # 行业电量简表唯一 sheet 名
ENERGY_COLUMNS = ["行业", *[f"{year}年" for year in range(2020, 2026)]]  # 简表冻结列集
TOTAL_KEY = "全社会用电总计"                     # 简表首行(总量)标签
# 9 行业标准名有序表与"其他"标签,拷自 refactor/load_forecast/config.py(冻结序,DESIGN D18);
# 服务与算法包跨 conda 环境,不能 import,只能拷贝常量。
FORECAST_INDUSTRIES = (
    "居民生活",
    "充换电服务业",
    "化学原料和化学制品制造业",
    "有色金属冶炼和压延加工业",
    "汽车制造业",
    "电气机械和器材制造业",
    "非金属矿物制品业",
    "黑色金属冶炼和压延加工业",
    "数据中心",
)
RESIDUAL_KEY = "其他（扣除重点行业）"
LOAD_COLUMNS = ["时间", "值"]                    # 负荷工作簿每 sheet 冻结列集
TARGET_YEAR = 2030                              # 目标年(config 冻结,不出接口参数)
TOTAL_RESULT_NAME = "2030总负荷_MW及标幺8760.csv"
INDUSTRY_RESULT_NAME = "2030行业标幺及MW8760_汇总.csv"
STAGE06_DIR = "06_有名值换算与总负荷合成"


class ForecastRunManager:
    """课题2 运行组织:登记上传 → 后台 subprocess 执行 → 提取总负荷与行业汇总。"""

    def __init__(self, run_service: RunService) -> None:
        self.logger = logging.getLogger("ForecastRunManager")
        self.run_service = run_service
        # ponytail: 课题2 内串行,与课题1 锁独立(两课题并行时 CPU 争用自担);要互斥再共用一把锁
        self._execute_lock = asyncio.Lock()

    async def create_run(
        self,
        macro_workbook: UploadedFile,
        industry_energy: UploadedFile,
        industry_load: UploadedFile,
    ) -> str:
        """校验并保存三件上传,登记运行记录(topic=load_forecast),返回 run_id。

        Args:
            macro_workbook (UploadedFile): 111.xlsx 宏观与电量工作簿。
            industry_energy (UploadedFile): 行业电量简表(单 sheet 10 行×2020—2025)。
            industry_load (UploadedFile): 行业历史负荷(10 sheet,每 sheet [时间, 值])。

        Returns:
            str: 运行标识 run_id。
        """
        try:
            RunManager._check_file(macro_workbook, MACRO_WORKBOOK_MAX_BYTES, "宏观工作簿")
            self._check_macro_workbook(macro_workbook)
            RunManager._check_file(industry_energy, INDUSTRY_ENERGY_MAX_BYTES, "行业电量简表")
            self._check_energy_workbook(industry_energy)
            RunManager._check_file(industry_load, INDUSTRY_LOAD_MAX_BYTES, "行业历史负荷")
            self._check_load_workbook(industry_load)
            if not FORECAST_PYTHON.is_file():
                raise ValueError(
                    f"课题2 算法环境缺失:{FORECAST_PYTHON}(可用环境变量 LOAD_FORECAST_PYTHON 指定)"
                )

            run_id = uuid.uuid4().hex
            input_dir = RUNS_DIR / run_id / "inputs"
            output_dir = RUNS_DIR / run_id / "outputs"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            (input_dir / "macro_workbook.xlsx").write_bytes(macro_workbook.content)
            (input_dir / "industry_energy.xlsx").write_bytes(industry_energy.content)
            (input_dir / "industry_load.xlsx").write_bytes(industry_load.content)

            await self.run_service.create(
                RunCreateDTO(
                    run_id=run_id,
                    topic="load_forecast",
                    params={
                        "filenames": {
                            "macro_workbook": macro_workbook.filename,
                            "industry_energy": industry_energy.filename,
                            "industry_load": industry_load.filename,
                        }
                    },
                    input_dir=str(input_dir),
                    output_dir=str(output_dir),
                )
            )
            self.logger.info("[ForecastRunManager]-%s- 运行已登记,输入目录 %s", run_id, input_dir)
            return run_id
        except ValueError as exc:
            self.logger.warning("[ForecastRunManager]-创建运行- 参数或文件校验未通过: %s", exc)
            raise
        except Exception as exc:
            self.logger.error("创建运行异常:\n%s", traceback.format_exc())
            raise RuntimeError(f"创建运行失败:{exc}") from exc

    async def execute_run(self, run_id: str) -> None:
        """后台 subprocess 执行算法并回写状态;由路由的 BackgroundTasks 调度。

        Args:
            run_id (str): 运行标识。

        Returns: None
        """
        record = await self.run_service.get(run_id)
        if record is None:
            self.logger.error("[ForecastRunManager]-%s- 运行记录不存在,后台任务退出", run_id)
            return

        async with self._execute_lock:
            await self.run_service.update(run_id, RunUpdateDTO(status=RunStatus.RUNNING))
            try:
                await self._run_pipeline(record)
                result_path = await asyncio.to_thread(self._extract_results, Path(record.output_dir))
                await self.run_service.update(
                    run_id,
                    RunUpdateDTO(status=RunStatus.SUCCEEDED, result_path=str(result_path)),
                )
                self.logger.info("[ForecastRunManager]-%s- 运行成功,结果 %s", run_id, result_path)
            except Exception as exc:
                self.logger.error(
                    "[ForecastRunManager]-%s- 算法执行失败:\n%s", run_id, traceback.format_exc()
                )
                await self.run_service.update(
                    run_id, RunUpdateDTO(status=RunStatus.FAILED, error_message=str(exc))
                )

    async def _run_pipeline(self, record: RunRecord) -> None:
        """subprocess 调 load_forecast 环境的 pipeline.py,失败抛 RuntimeError。"""
        input_dir = Path(record.input_dir)
        command = [
            str(FORECAST_PYTHON),
            str(FORECAST_PIPELINE),
            "--macro-xlsx", str(input_dir / "macro_workbook.xlsx"),
            "--energy-xlsx", str(input_dir / "industry_energy.xlsx"),
            "--load-xlsx", str(input_dir / "industry_load.xlsx"),
            "--output-dir", record.output_dir,
        ]
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(FORECAST_PIPELINE.parent),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await process.communicate()
        if stdout:
            self.logger.info("[ForecastRunManager]-%s- 算法输出:\n%s", record.run_id, stdout.decode("utf-8", errors="replace"))
        if process.returncode != 0:
            lines = [
                line for line in stderr.decode("utf-8", errors="replace").strip().splitlines()
                if line.strip()
            ]
            reason = lines[-1] if lines else f"退出码 {process.returncode}"
            raise RuntimeError(f"课题2 算法执行失败:{reason}(明细见 {record.output_dir}/pipeline.log)")

    async def get_status(self, run_id: str) -> RunStatusResp:
        """查询运行状态。

        Args:
            run_id (str): 运行标识。

        Returns:
            RunStatusResp: 状态响应结构。

        Raises:
            NotFoundError: 运行记录不存在。
        """
        record = await self._require_record(run_id)
        return RunStatusResp(
            job_id=record.run_id,
            status=RUN_STATUS_NAMES.get(record.status, "unknown"),
            error=record.error_message,
        )

    async def get_recent_status(self, limit: int = 5) -> list[RunStatusResp]:
        """查询课题2 最近若干条运行状态(历史任务,create_time 倒序)。

        Args:
            limit (int): 返回条数上限,默认 5。

        Returns:
            list[RunStatusResp]: 状态响应列表,最近在前。
        """
        records = await self.run_service.list_recent("load_forecast", limit)
        return [RunManager._status_resp(record) for record in records]

    async def get_result_path(self, run_id: str, which: str) -> Path:
        """返回结果 CSV 路径,供下载接口使用。

        Args:
            run_id (str): 运行标识。
            which (str): 结果文件选择:total 总负荷(默认)/industry 行业汇总。

        Returns:
            Path: 结果文件路径。

        Raises:
            NotFoundError: 运行记录不存在或结果文件缺失。
            ConflictError: 运行未成功,结果尚未生成。
        """
        record = await self._require_succeeded(run_id)
        name = INDUSTRY_RESULT_NAME if which == "industry" else TOTAL_RESULT_NAME
        path = Path(record.result_path).parent / name
        if not path.is_file():
            raise NotFoundError(f"运行 {run_id} 结果文件缺失:{path.name}")
        return path

    async def get_result_views(self, run_id: str) -> dict:
        """查询运行结果的 JSON 视图(总负荷+行业汇总两组),供前端绘图。

        Args:
            run_id (str): 运行标识。

        Returns:
            dict: {"views": [...]},两组 {year, sheet_name, index, data}。

        Raises:
            NotFoundError: 运行记录不存在。
            ConflictError: 运行未成功,结果尚未生成。
        """
        record = await self._require_succeeded(run_id)
        views = await asyncio.to_thread(
            self._build_result_views, Path(record.result_path).parent
        )
        return {"views": views}

    async def _require_succeeded(self, run_id: str) -> RunRecord:
        """取已成功运行的记录,不存在/未成功即抛错。"""
        record = await self._require_record(run_id)
        if record.status != RunStatus.SUCCEEDED or not record.result_path:
            raise ConflictError(
                f"运行 {run_id} 当前状态为 {RUN_STATUS_LABELS.get(record.status, '未知')},结果尚未生成"
            )
        return record

    async def _require_record(self, run_id: str) -> RunRecord:
        """取运行记录,不存在即抛 NotFoundError。"""
        record = await self.run_service.get(run_id)
        if record is None:
            raise NotFoundError(f"未找到运行记录:{run_id}")
        return record

    @staticmethod
    def _check_macro_workbook(uploaded: UploadedFile) -> None:
        """宏观工作簿快检:首表 1 表头行+8 指标行,首列为指标名。"""
        workbook = openpyxl.load_workbook(io.BytesIO(uploaded.content), read_only=True)
        try:
            sheet = workbook.worksheets[0]
            rows = list(sheet.iter_rows(min_row=1, max_row=9, max_col=1, values_only=True))
        finally:
            workbook.close()
        if len(rows) < 9 or any(row[0] in (None, "") for row in rows[1:]):
            raise ValueError(
                f"宏观工作簿 {uploaded.filename} 首表应为 1 行表头+8 行指标(首列为指标名),无法识别"
            )

    @staticmethod
    def _check_energy_workbook(uploaded: UploadedFile) -> None:
        """行业电量简表快检:sheet 名、冻结列集、10 行标签(总计+9 行业)。"""
        workbook = openpyxl.load_workbook(io.BytesIO(uploaded.content), read_only=True)
        try:
            if ENERGY_SHEET_NAME not in workbook.sheetnames:
                raise ValueError(f"行业电量简表 {uploaded.filename} 缺少 sheet:{ENERGY_SHEET_NAME!r}")
            sheet = workbook[ENERGY_SHEET_NAME]
            rows = list(sheet.iter_rows(min_row=1, max_row=11, values_only=True))
        finally:
            workbook.close()
        header = [str(cell) if cell is not None else "" for cell in rows[0]]
        if header != ENERGY_COLUMNS:
            raise ValueError(
                f"行业电量简表 {uploaded.filename} 表头应为 {ENERGY_COLUMNS},实际 {header}"
            )
        labels = [str(row[0]) if row and row[0] is not None else "" for row in rows[1:11]]
        if len(rows) < 11 or labels[0] != TOTAL_KEY or sorted(labels[1:]) != sorted(FORECAST_INDUSTRIES):
            raise ValueError(
                f"行业电量简表 {uploaded.filename} 应为 {TOTAL_KEY}+9 行业共 10 行,实际行标签 {labels}"
            )

    @staticmethod
    def _check_load_workbook(uploaded: UploadedFile) -> None:
        """行业历史负荷快检:恰好 10 个行业 sheet,每 sheet 列 [时间, 值]。"""
        workbook = openpyxl.load_workbook(io.BytesIO(uploaded.content), read_only=True)
        try:
            names = workbook.sheetnames
            if len(names) != 10:
                raise ValueError(f"行业历史负荷 {uploaded.filename} 应为 10 个行业 sheet,实际 {len(names)} 个")
            if set(names) != {*FORECAST_INDUSTRIES, RESIDUAL_KEY}:
                raise ValueError(
                    f"行业历史负荷 {uploaded.filename} sheet 名应为 9 行业标准名+{RESIDUAL_KEY},实际 {names}"
                )
            for name in names:
                header = next(workbook[name].iter_rows(min_row=1, max_row=1, values_only=True), None)
                columns = [str(cell) if cell is not None else "" for cell in (header or [])]
                if columns != LOAD_COLUMNS:
                    raise ValueError(
                        f"行业历史负荷 sheet {name!r} 列应为 {LOAD_COLUMNS},实际 {columns}"
                    )
        finally:
            workbook.close()

    @staticmethod
    def _extract_results(output_dir: Path) -> Path:
        """从算法输出提取两张结果 CSV 到 result/ 目录,返回总负荷结果路径。

        Args:
            output_dir (Path): 算法输出目录。

        Returns:
            Path: 总负荷结果 CSV 路径。

        Raises:
            ValueError: 输出缺少结果文件。
        """
        stage06 = output_dir / STAGE06_DIR
        sources = [stage06 / TOTAL_RESULT_NAME, stage06 / INDUSTRY_RESULT_NAME]
        missing = [path.name for path in sources if not path.is_file()]
        if missing:
            raise ValueError(f"算法输出缺少结果文件:{missing}")
        result_dir = output_dir.parent / "result"
        result_dir.mkdir(parents=True, exist_ok=True)
        for path in sources:
            shutil.copy2(path, result_dir / path.name)
        return result_dir / TOTAL_RESULT_NAME

    @staticmethod
    def _build_result_views(result_dir: Path) -> list[dict]:
        """两张结果 CSV 转绘图视图(round_trip 解析,JSON 值与 CSV 逐位一致)。

        行业汇总拆两张视图:MW 一张、标幺一张,列名取行业名本身(所在视图已表明量纲)。

        Args:
            result_dir (Path): result/ 目录(含两张结果 CSV)。

        Returns:
            list[dict]: 三组 {year, sheet_name, index, data}(总负荷/行业MW/行业标幺)。
        """
        total = pd.read_csv(
            result_dir / TOTAL_RESULT_NAME, dtype={"时间": str}, float_precision="round_trip",
            encoding="utf-8-sig",
        )
        industry = pd.read_csv(
            result_dir / INDUSTRY_RESULT_NAME, dtype={"时间": str}, float_precision="round_trip",
            encoding="utf-8-sig",
        )
        timestamps = total["时间"].tolist()
        assert industry["时间"].tolist() == timestamps

        def view(sheet_name: str, columns: list[tuple[str, pd.Series]]) -> dict:
            return {
                "year": TARGET_YEAR,
                "sheet_name": sheet_name,
                "index": timestamps,
                "data": [
                    {"column_name": name, "data": values.tolist()} for name, values in columns
                ],
            }

        mw_columns = [(c[: -len("_MW")], industry[c]) for c in industry.columns if c.endswith("_MW")]
        pu_columns = [
            (c[: -len("_标幺值")], industry[c]) for c in industry.columns if c.endswith("_标幺值")
        ]
        return [
            view(
                f"{TARGET_YEAR}总负荷",
                [("总负荷MW", total["总负荷MW"]), ("总负荷标幺值", total["总负荷标幺值"])],
            ),
            view(f"{TARGET_YEAR}行业MW", mw_columns),
            view(f"{TARGET_YEAR}行业标幺", pu_columns),
        ]
