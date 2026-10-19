# -*- coding: utf-8 -*-
"""运行并验收 3.5.0 自然负荷正式数据底座，不生成工作簿。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


FORMAL_YEARS = tuple(range(2025, 2061))
FORMAL_BOOTSTRAP_SCENARIOS = 1000


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "运行并保存2025年至指定截止年的底层结果（默认2030）；"
            "同步生成基准M1-P50与Bootstrap-P50四季工作日展示数据。"
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="全新输出目录；正式运行要求Parquet可用",
    )
    parser.add_argument(
        "--materialize-bootstrap-daily",
        action="store_true",
        help="另行物化Bootstrap逐日结果；默认不物化以控制体量",
    )
    parser.add_argument("--end-year", type=int, default=2030, help="结果截止年份，默认2030")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not 2030 <= args.end_year <= 2060:
        raise ValueError("end-year必须在2030—2060之间")
    years = tuple(range(2025, args.end_year + 1))
    output_dir = args.output_dir.resolve()
    if (output_dir / "run_metadata.json").exists() or (
        output_dir / "数据表" / "tables_manifest.json"
    ).exists():
        raise RuntimeError(
            "正式发布必须使用全新输出目录；旧口径目录只能作为历史审计材料"
        )

    # 延后导入，确保 --help 只展示固定口径而不会启动或初始化全量计算。
    from workstreams_load_characteristic_model_v2 import config
    from workstreams_load_characteristic_model_v2.formal_output_validation import (
        validate_formal_output,
    )
    from workstreams_load_characteristic_model_v2.pipeline import (
        run_model, _assert_current_method_contract,
    )

    # Use the same versioned method contract as the core pipeline.
    _assert_current_method_contract()

    outputs = run_model(
        years=years,
        n_scenarios=FORMAL_BOOTSTRAP_SCENARIOS,
        output_dir=output_dir,
        run_label=(
            f"2025—{args.end_year}正式底座（四时段局部平滑+Logit阻尼phi=0.8；"
            "基准年度规模固定采用M1 P50_natural_total；"
            "展示采用Bootstrap P50_real；24行业+修正组成）"
        ),
        method="M1",
        materialize_bootstrap_daily=args.materialize_bootstrap_daily,
        require_parquet=True,
    )
    validation = validate_formal_output(output_dir, base_years=years,
                                        study_years=range(2030, args.end_year + 1),
                                        display_years=range(2025, min(args.end_year, 2045) + 1))
    validation_path = output_dir / "正式发布校验.json"
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not validation["passed"]:
        failed = [
            item["name"]
            for item in validation.get("checks", [])
            if item.get("status") == "failed"
        ]
        raise RuntimeError(f"3.5.0正式发布校验未通过：{failed}")
    outputs = {**outputs, "release_validation": validation_path}
    print(
        json.dumps(
            {name: str(path) for name, path in outputs.items()},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
