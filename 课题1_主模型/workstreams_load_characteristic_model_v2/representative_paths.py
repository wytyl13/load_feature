"""从完整真实场景中选择P10/P50/P90代表轨迹。"""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURES = ["cv", "acf1", "acf7", "max_mean", "min_mean"]


def _acf(values: np.ndarray, lag: int) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) <= lag or np.std(values) <= 1e-15:
        return 0.0
    return float(np.corrcoef(values[:-lag], values[lag:])[0, 1])


def build_scenario_features(
    multipliers: pd.DataFrame,
    baseline_daily_energy: pd.Series,
    baseline_daily_peak: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    energy = baseline_daily_energy.copy()
    peak = baseline_daily_peak.copy()
    energy.index = pd.to_datetime(energy.index).normalize()
    peak.index = pd.to_datetime(peak.index).normalize()
    base = pd.DataFrame({"date": energy.index, "baseline_energy": energy.values, "baseline_peak": peak.reindex(energy.index).values})
    data = multipliers[["scenario_id", "date", "daily_multiplier"]].merge(base, on="date", how="left", validate="many_to_one")
    if data[["baseline_energy", "baseline_peak"]].isna().any().any():
        raise ValueError("场景特征的日电量/日峰值基线缺失")
    data["daily_energy"] = data.baseline_energy * data.daily_multiplier
    data["daily_peak"] = data.baseline_peak * data.daily_multiplier
    data["month"] = pd.to_datetime(data.date).dt.month
    monthly_rows = []
    annual_rows = []
    for scenario_id, scenario in data.groupby("scenario_id", sort=True):
        for month, part in scenario.groupby("month", sort=True):
            values = part.sort_values("date").daily_energy.to_numpy(float)
            mean = float(values.mean())
            monthly_rows.append({"scenario_id": int(scenario_id), "month": int(month), "cv": float(values.std(ddof=0) / mean), "acf1": _acf(values, 1), "acf7": _acf(values, 7), "max_mean": float(values.max() / mean), "min_mean": float(values.min() / mean)})
        annual_rows.append({"scenario_id": int(scenario_id), "annual_peak_MW": float(scenario.daily_peak.max()), "annual_energy_MWh": float(scenario.daily_energy.sum())})
    return pd.DataFrame(monthly_rows), pd.DataFrame(annual_rows)


def _robust_distances(monthly_features: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for month in range(1, 13):
        part = monthly_features.loc[monthly_features.month == month].set_index("scenario_id")
        for feature in FEATURES:
            values = part[feature].astype(float)
            center = float(values.median())
            mad = float(np.median(np.abs(values - center)))
            if mad <= 0 or not np.isfinite(mad):
                continue
            pieces.append(((values - center) / (1.4826 * mad)).rename(f"m{month:02d}_{feature}"))
    if not pieces:
        ids = sorted(monthly_features.scenario_id.unique())
        return pd.DataFrame({"scenario_id": ids, "distance": np.zeros(len(ids))})
    matrix = pd.concat(pieces, axis=1).sort_index()
    distance = np.sqrt(np.square(matrix).sum(axis=1))
    return distance.rename("distance").reset_index()


def select_representative_scenarios(monthly_features: pd.DataFrame, annual_stats: pd.DataFrame) -> tuple[dict[str, int], pd.DataFrame]:
    distances = _robust_distances(monthly_features)
    ranking = annual_stats[["scenario_id", "annual_peak_MW"]].merge(distances, on="scenario_id", how="left")
    ranking["distance"] = ranking.distance.fillna(0.0)
    ranking = ranking.sort_values(["annual_peak_MW", "scenario_id"]).reset_index(drop=True)
    n = len(ranking)
    if n == 0:
        raise ValueError("没有可选择的Bootstrap场景")
    ranking["peak_empirical_quantile"] = (np.arange(n) + 0.5) / n
    p50 = int(ranking.sort_values(["distance", "scenario_id"]).iloc[0].scenario_id)

    def choose_band(low: float, high: float, target: float) -> int:
        candidates = ranking.loc[ranking.peak_empirical_quantile.between(low, high, inclusive="both")]
        if candidates.empty:
            candidates = ranking.assign(qdiff=(ranking.peak_empirical_quantile - target).abs()).sort_values(["qdiff", "scenario_id"]).head(1)
        return int(candidates.sort_values(["distance", "scenario_id"]).iloc[0].scenario_id)

    selected = {"P10": choose_band(0.08, 0.12, 0.10), "P50": p50, "P90": choose_band(0.88, 0.92, 0.90)}
    ranking["representative_label"] = ""
    for label, scenario_id in selected.items():
        ranking.loc[ranking.scenario_id == scenario_id, "representative_label"] = label
    return selected, ranking

