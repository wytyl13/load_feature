"""运行小规模真实数据验收算例，或只校验已保存的结果。"""
from pathlib import Path
import argparse
import json
import time

from workstreams_load_characteristic_model_v2.pipeline import run_pipeline
from workstreams_load_characteristic_model_v2.formal_output_validation import validate_formal_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path, help='新结果目录；校验模式下为既有目录')
    parser.add_argument('--validate-only', action='store_true', help='只读校验，不写入结果目录')
    parser.add_argument('--bootstrap', type=int, default=20, help='每年份每情景抽样数，默认20；正式入口固定1000')
    args = parser.parse_args()
    started = time.monotonic()
    if not args.validate_only:
        run_pipeline(years=range(2025, 2031), output_dir=args.output_dir,
                     bootstrap_scenarios=args.bootstrap, require_parquet=True,
                     mode='delivery_example', run_label='开发验收算例：含闰年；不替代正式千场景运行')
    metadata = json.loads((args.output_dir / 'run_metadata.json').read_text(encoding='utf-8'))
    years = metadata['years']
    report = validate_formal_output(
        args.output_dir, base_years=years, study_years=[y for y in years if y >= 2030],
        display_years=[y for y in years if 2025 <= y <= 2045],
        scenarios=metadata['planning_scenarios'],
        expected_bootstrap=metadata['bootstrap_scenarios_per_context'])
    report['elapsed_seconds'] = round(time.monotonic() - started, 3)
    if not args.validate_only:
        (args.output_dir / 'sample_validation_report.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
