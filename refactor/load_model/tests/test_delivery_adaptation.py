from pathlib import Path
import importlib.util
import json
import shutil

import pytest
from workstreams_load_characteristic_model_v2 import config, pipeline
from workstreams_load_characteristic_model_v2 import formal_output_validation as validator

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_relocated_without_repository(tmp_path, monkeypatch):
    package = tmp_path / '甲方 解压目录' / 'workstreams_load_characteristic_model_v2'
    package.mkdir(parents=True)
    shutil.copy2(Path(config.__file__), package / 'config.py')
    monkeypatch.chdir(tmp_path)
    relocated = load_module('relocated_delivery_config', package / 'config.py')
    assert relocated.PROJECT_ROOT == package.parent
    assert relocated.DEFAULT_LOAD_WORKBOOK == package.parent / 'inputs' / config.LOAD_WORKBOOK_FILENAME
    assert relocated.ANNUAL_ENERGY_UPSTREAM == package.parent / 'inputs/annual_energy_upstream.json'


def test_all_six_runtime_inputs_exist():
    for path in [config.DEFAULT_LOAD_WORKBOOK, config.DEFAULT_OFFICIAL_WORKBOOK,
                 config.ANNUAL_ENERGY_UPSTREAM, config.SAMPLE_INDUSTRY_MAPPING_CSV,
                 config.OFFICIAL_2025_MAPPING_CSV, config.INDUSTRY_WEIGHT_EVOLUTION_CSV]:
        assert path.is_file(), path
        assert path.is_relative_to(ROOT)


def test_current_contract_still_rejects_method_drift(monkeypatch):
    monkeypatch.setattr(pipeline, 'METHOD_REVISION', 'old-method')
    with pytest.raises(RuntimeError, match='方法开关漂移'):
        pipeline._assert_current_method_contract()


def test_formal_entry_uses_current_contract(tmp_path, monkeypatch):
    runner = load_module('delivery_formal_entry', ROOT / 'run_formal_2025_2060.py')
    out = tmp_path / 'formal'
    calls = {}
    def run_model(**kwargs):
        calls.update(kwargs)
        out.mkdir()
        return {'output_dir': out}
    monkeypatch.setattr(pipeline, 'run_model', run_model)
    monkeypatch.setattr(validator, 'validate_formal_output', lambda *a, **k: {'passed': True})
    assert runner.main(['--end-year', '2030', '--output-dir', str(out)]) == 0
    assert calls['years'] == tuple(range(2025, 2031))
    assert calls['n_scenarios'] == 1000
    assert calls['require_parquet'] is True


def test_pipeline_rejects_occupied_output_before_reading_inputs(tmp_path, monkeypatch):
    out = tmp_path / 'existing'
    out.mkdir()
    sentinel = out / 'keep.txt'
    sentinel.write_text('original', encoding='utf-8')
    def should_not_read(**kwargs):
        raise AssertionError('read inputs too early')
    monkeypatch.setattr(pipeline, 'load_and_validate_inputs', should_not_read)
    with pytest.raises(FileExistsError, match='拒绝覆盖'):
        pipeline.run_pipeline(years=[2030], bootstrap_scenarios=20, output_dir=out)
    assert sentinel.read_text(encoding='utf-8') == 'original'
    assert list(out.iterdir()) == [sentinel]


def test_saved_hourly_reference_resolves_from_result_root(tmp_path, monkeypatch):
    root = tmp_path / 'relocated'
    (root / '数据表').mkdir(parents=True)
    (root / '逐时曲线').mkdir()
    hourly = root / '逐时曲线/test.parquet'
    hourly.write_bytes(b'location-only')
    (root / 'run_metadata.json').write_text(json.dumps({'hourly_output': {'file': '逐时曲线/test.parquet'}}), encoding='utf-8')
    (root / '数据表/tables_manifest.json').write_text('{}', encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    assert validator._ResultStore(root).hourly_path() == hourly
