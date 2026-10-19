"""Configuration for the versioned load-characteristic-model retrofit."""

from __future__ import annotations

from pathlib import Path


WORKSTREAM_DIR = Path(__file__).resolve().parent
# Delivery paths are anchored to this package, never to the source repository
# or the caller's working directory. Explicit run_pipeline paths take priority.
PROJECT_ROOT = WORKSTREAM_DIR.parent
LOAD_CHARACTERISTIC_DIR = PROJECT_ROOT
FINAL_OUTPUT_DIR = PROJECT_ROOT / "results"
INPUT_DIR = PROJECT_ROOT / "inputs"
CONFIG_DIR = WORKSTREAM_DIR / "config"

LOAD_WORKBOOK_FILENAME = "副本山东负荷特性8760数据表.xlsx"
OFFICIAL_WORKBOOK_FILENAME = "山东2020-2025_全社会用电分类表（全口径）.xlsx"

# 权威曲线源的逐字节快照随交付包存放；来源及SHA-256见交付来源清单。
DEFAULT_LOAD_WORKBOOK = INPUT_DIR / LOAD_WORKBOOK_FILENAME
DEFAULT_OFFICIAL_WORKBOOK = INPUT_DIR / OFFICIAL_WORKBOOK_FILENAME
ANNUAL_ENERGY_UPSTREAM = INPUT_DIR / "annual_energy_upstream.json"
HISTORY_ANALYSIS_INPUT = INPUT_DIR / "analysis_input.json"
SUPPLEMENTAL_FORECAST_INPUT = INPUT_DIR / "陕鲁用电量预测_基础数据与规划目标_第一阶段.xlsx"
SMOKE_OUTPUT_DIR = FINAL_OUTPUT_DIR / "冒烟验证_自然负荷_3.5.0"
SAMPLE_INDUSTRY_MAPPING_CSV = CONFIG_DIR / "sample_industry_mapping.csv"
OFFICIAL_2025_MAPPING_CSV = CONFIG_DIR / "official_2025_mapping.csv"
INDUSTRY_WEIGHT_EVOLUTION_CSV = CONFIG_DIR / "industry_weight_evolution.csv"

# 第一、第二、第三产业和居民生活只用于代理曲线汇总，不是参数层或电量分配层。
BROAD_SECTORS = ("第一产业", "第二产业", "第三产业", "居民生活")
CORRECTION_COMPONENT = "修正组成"
TRADITIONAL_COMPONENT_COUNT = 24

RESERVED_LOAD_SHEETS = ("用电结构", "历史负荷数据（2019-2023）")
OFFICIAL_TOTAL_LABEL = "全社会用电总计"
OFFICIAL_EXPECTED_UNIT = "万千瓦时"
OFFICIAL_RELATIVE_TOLERANCE = 1e-10

# 官方2020—2025全口径分类表中必须逐年、逐项读取的三项排除电量。
# 这些标签是源表精确标签，不允许模糊匹配或把三项合并到修正组成中。
OFFICIAL_PLANT_AUXILIARY_POWER_LABEL = "其中：电厂生产全部耗用电量"
OFFICIAL_LINE_LOSS_LABEL = "线路损失电量"
OFFICIAL_PUMPED_STORAGE_CHARGING_LABEL = "抽水蓄能抽水耗用电量"
OFFICIAL_EXCLUDED_ENERGY_LABELS = {
    "plant_auxiliary_power": OFFICIAL_PLANT_AUXILIARY_POWER_LABEL,
    "line_loss": OFFICIAL_LINE_LOSS_LABEL,
    "pumped_storage_charging": OFFICIAL_PUMPED_STORAGE_CHARGING_LABEL,
}
OFFICIAL_EXCLUDED_ENERGY_ACTUAL_YEARS = tuple(range(2020, 2026))

NATURAL_LOAD_CURVE_ENERGY_SCOPE = (
    "natural_load_excluding_plant_aux_line_loss_pumped_storage_charging"
)
HISTORICAL_SHAPE_SCOPE = "dispatch_proxy_not_hourly_deducted"

# These are mutually exclusive accounting scopes.  Their totals add to the
# official full-society total when residents are included.
OFFICIAL_DEPARTMENT_TOTAL_LABELS = {
    "农林牧渔业": "一、农、林、牧、渔业",
    "工业": "二、工业",
    "建筑业": "三、建筑业",
    "交通运输仓储邮政业": "四、交通运输、仓储和邮政业",
    "信息软件业": "五、信息传输、软件和信息技术服务业",
    "批发零售业": "六、批发和零售业",
    "住宿餐饮业": "七、住宿和餐饮业",
    "金融业": "八、金融业",
    "房地产业": "九、房地产业",
    "租赁商务服务业": "十、租赁和商务服务业",
    "公共服务管理": "十一、公共服务及管理组织",
    "居民生活": "B、城乡居民生活用电合计",
}

# Kept for auditability but deliberately disabled.  The old manual values are
# neither a runtime fallback nor an output source in the improved model.
LEGACY_MANUAL_PARAMETERS_ENABLED = False

# 2026-09-04 final method contract.  These values are deliberately constants,
# rather than user-facing switches: changing any one of them creates a
# different method and therefore requires a new model revision.
MODEL_VERSION = "3.5.0"
METHOD_REVISION = "2026-09-15-logit-damped-phi08"

INDUSTRY_AGGREGATION_MODE = "user_day_normalize_then_equal_weight"
TYPICAL_CURVE_K = 1
MEDOID_DISTANCE = "L1"
MEDOID_TIE_BREAK = "earliest_date"
BASE_MEDOID_YEAR = 2023
HISTORY_COMPARISON_YEAR = 2022

FLAT_LOAD_FACTOR_THRESHOLD = 0.80
FLAT_CV_THRESHOLD = 0.15
FLAT_BORDERLINE_BAND = 0.02
PEAK_VALLEY_EVOLUTION_DELTA_LOWER_BOUND = 0.065
PEAK_VALLEY_EVOLUTION_DELTA_UPPER_BOUND = 0.25
PEAK_VALLEY_GATE_POLICY = (
    "trend_eligible_and_nonflat_and_abs_delta_within_closed_interval_0.065_0.25"
)
PEAK_VALLEY_EXTRAPOLATION_ENABLED = True
PEAK_VALLEY_EXTRAPOLATION_METHOD = "logit_damped"
PEAK_VALLEY_TRANSFORM = "global_mean_centered_scaling"
PEAK_VALLEY_TARGET_PRIORITY = "final_when_gate_enabled_otherwise_four_window_final"
PEAK_VALLEY_SUPPORTED_PHIS = (0.8,)
PEAK_VALLEY_FORMAL_PHI = 0.8
PEAK_VALLEY_RATE_TOLERANCE = 1e-10

FOUR_WINDOW_EXTRAPOLATION_ENABLED = True
FOUR_WINDOW_TRANSFORM = "proportional_within_window"
SHAPE_EXTRAPOLATION_METHOD = "four_window_then_local_smoothing_then_gated_peak_valley_mean_scaling"
FOUR_WINDOW_FINAL_SHARE_POLICY = (
    "exact_if_peak_valley_skipped_diagnostic_if_applied"
)
LATE_PEAK_CONTROL_POLICY = "diagnostic_only"
PEAK_DAMPING_ENABLED = False
ANCHOR_REPAIR_ENABLED = False

DAY_TYPE_FACTOR_SOURCE = "province_system_load_2023"
DAY_TYPE_FACTOR_SCOPE = "province_shared_all_25_components"
SHAPE_NUMERICAL_TOLERANCE = 1e-12


__all__ = [
    "ANNUAL_ENERGY_UPSTREAM",
    "BROAD_SECTORS",
    "CONFIG_DIR",
    "CORRECTION_COMPONENT",
    "DEFAULT_LOAD_WORKBOOK",
    "DEFAULT_OFFICIAL_WORKBOOK",
    "FINAL_OUTPUT_DIR",
    "INPUT_DIR",
    "HISTORY_ANALYSIS_INPUT",
    "INDUSTRY_WEIGHT_EVOLUTION_CSV",
    "INDUSTRY_AGGREGATION_MODE",
    "LEGACY_MANUAL_PARAMETERS_ENABLED",
    "LOAD_CHARACTERISTIC_DIR",
    "LOAD_WORKBOOK_FILENAME",
    "OFFICIAL_2025_MAPPING_CSV",
    "OFFICIAL_DEPARTMENT_TOTAL_LABELS",
    "OFFICIAL_EXCLUDED_ENERGY_ACTUAL_YEARS",
    "OFFICIAL_EXCLUDED_ENERGY_LABELS",
    "OFFICIAL_EXPECTED_UNIT",
    "OFFICIAL_LINE_LOSS_LABEL",
    "OFFICIAL_PLANT_AUXILIARY_POWER_LABEL",
    "OFFICIAL_PUMPED_STORAGE_CHARGING_LABEL",
    "OFFICIAL_RELATIVE_TOLERANCE",
    "OFFICIAL_TOTAL_LABEL",
    "OFFICIAL_WORKBOOK_FILENAME",
    "HISTORICAL_SHAPE_SCOPE",
    "NATURAL_LOAD_CURVE_ENERGY_SCOPE",
    "PROJECT_ROOT",
    "RESERVED_LOAD_SHEETS",
    "SAMPLE_INDUSTRY_MAPPING_CSV",
    "SMOKE_OUTPUT_DIR",
    "SUPPLEMENTAL_FORECAST_INPUT",
    "TRADITIONAL_COMPONENT_COUNT",
    "WORKSTREAM_DIR",
    "ANCHOR_REPAIR_ENABLED",
    "BASE_MEDOID_YEAR",
    "DAY_TYPE_FACTOR_SCOPE",
    "DAY_TYPE_FACTOR_SOURCE",
    "FLAT_BORDERLINE_BAND",
    "FLAT_CV_THRESHOLD",
    "FLAT_LOAD_FACTOR_THRESHOLD",
    "FOUR_WINDOW_EXTRAPOLATION_ENABLED",
    "FOUR_WINDOW_FINAL_SHARE_POLICY",
    "FOUR_WINDOW_TRANSFORM",
    "HISTORY_COMPARISON_YEAR",
    "LATE_PEAK_CONTROL_POLICY",
    "MEDOID_DISTANCE",
    "MEDOID_TIE_BREAK",
    "METHOD_REVISION",
    "MODEL_VERSION",
    "PEAK_DAMPING_ENABLED",
    "PEAK_VALLEY_EVOLUTION_DELTA_LOWER_BOUND",
    "PEAK_VALLEY_EVOLUTION_DELTA_UPPER_BOUND",
    "PEAK_VALLEY_EXTRAPOLATION_ENABLED",
    "PEAK_VALLEY_EXTRAPOLATION_METHOD",
    "PEAK_VALLEY_FORMAL_PHI",
    "PEAK_VALLEY_GATE_POLICY",
    "PEAK_VALLEY_RATE_TOLERANCE",
    "PEAK_VALLEY_SUPPORTED_PHIS",
    "PEAK_VALLEY_TARGET_PRIORITY",
    "PEAK_VALLEY_TRANSFORM",
    "SHAPE_EXTRAPOLATION_METHOD",
    "SHAPE_NUMERICAL_TOLERANCE",
    "TYPICAL_CURVE_K",
]
