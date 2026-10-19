from workstreams_load_characteristic_model_v2 import config


def test_20260904_method_switches_are_fixed() -> None:
    assert config.PEAK_VALLEY_EXTRAPOLATION_ENABLED is True
    assert config.FOUR_WINDOW_EXTRAPOLATION_ENABLED is True
    assert config.FOUR_WINDOW_TRANSFORM == "proportional_within_window"
    assert config.PEAK_VALLEY_EXTRAPOLATION_METHOD == "logit_damped"
    assert config.PEAK_VALLEY_TRANSFORM == "global_mean_centered_scaling"
    assert (
        config.PEAK_VALLEY_TARGET_PRIORITY
        == "final_when_gate_enabled_otherwise_four_window_final"
    )
    assert config.PEAK_VALLEY_SUPPORTED_PHIS == (0.8,)
    assert config.PEAK_VALLEY_FORMAL_PHI == 0.8
    assert config.PEAK_VALLEY_RATE_TOLERANCE == 1e-10
    assert (
        config.SHAPE_EXTRAPOLATION_METHOD
        == "four_window_then_local_smoothing_then_gated_peak_valley_mean_scaling"
    )
    assert (
        config.FOUR_WINDOW_FINAL_SHARE_POLICY
        == "exact_if_peak_valley_skipped_diagnostic_if_applied"
    )
    assert config.PEAK_VALLEY_EVOLUTION_DELTA_LOWER_BOUND == 0.065
    assert config.PEAK_VALLEY_EVOLUTION_DELTA_UPPER_BOUND == 0.25
    assert config.FLAT_LOAD_FACTOR_THRESHOLD == 0.80
    assert config.FLAT_CV_THRESHOLD == 0.15
    assert config.PEAK_VALLEY_GATE_POLICY == (
        "trend_eligible_and_nonflat_and_abs_delta_within_closed_interval_0.065_0.25"
    )
    assert config.LATE_PEAK_CONTROL_POLICY == "diagnostic_only"
    assert config.PEAK_DAMPING_ENABLED is False
    assert config.ANCHOR_REPAIR_ENABLED is False
    assert config.DAY_TYPE_FACTOR_SOURCE == "province_system_load_2023"
    assert config.DAY_TYPE_FACTOR_SCOPE == "province_shared_all_25_components"
    assert config.MODEL_VERSION == "3.5.0"
    assert config.METHOD_REVISION == "2026-09-15-logit-damped-phi08"
