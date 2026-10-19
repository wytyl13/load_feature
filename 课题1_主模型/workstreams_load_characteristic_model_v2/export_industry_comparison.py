"""Recompute industry shapes and prepare the established 25-sheet workbook.

Run with --source-dir pointing to a previously extracted, separate-day-type
historical dataset (baseline_shapes.csv, cell_quality.csv, data.json).
Future values in the layout JSON are always replaced by this model's output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MODEL_VERSION, METHOD_REVISION, PEAK_VALLEY_FORMAL_PHI
from .parameter_pipeline import build_future_parameter_table
from .shape_parameters import identify_shape_parameters
from .result_tables import _validate_shape_future_schema
from .quality import SHAPE_COLS
from .shape_transform import window_shares


def prepare(source_dir: Path, output_dir: Path) -> dict:
    baseline = pd.read_csv(source_dir / "baseline_shapes.csv")
    quality = pd.read_csv(source_dir / "cell_quality.csv")
    keys = ["industry", "season", "final_day_type"]
    if len(baseline) != 576 or baseline.duplicated(keys + ["year"]).any():
        raise ValueError("Expected unique 24-industry, four-season, three-day-type historical curves")
    history = identify_shape_parameters(baseline, quality)
    future = build_future_parameter_table(baseline, history, [2030])
    _validate_shape_future_schema(future)
    formal = future.loc[future.H.eq(8) & future.phi.eq(PEAK_VALLEY_FORMAL_PHI)].copy()
    assert len(formal) == 288 and not formal.duplicated(keys).any()
    curves = formal[SHAPE_COLS].to_numpy(float)
    assert np.isfinite(curves).all() and curves.min() >= 0
    assert np.max(abs(curves.sum(axis=1) - 1)) < 1e-12
    groups = json.loads((source_dir / "data.json").read_text(encoding="utf-8"))
    b = baseline.set_index(keys + ["year"])
    f = formal.set_index(keys)
    history_error = 0.0
    for group in groups:
        for season in group["seasons"]:
            for row in season["rows"]:
                typ = "普通周末" if row["type"] == "周末" else row["type"]
                key = (group["name"], season["name"], typ)
                src = b.loc[(*key, row["year"])] if row["year"] != 2030 else f.loc[key]
                p = src[SHAPE_COLS].to_numpy(float)
                if row["year"] != 2030:
                    history_error = max(history_error, float(np.max(abs(p - row["p"]))))
                row["p"] = p.tolist()
                row["expected"] = [float(np.std(p)/np.mean(p)), float(np.mean(p)/max(p)),
                    float(min(p)/max(p)), float((max(p)-min(p))/max(p)),
                    int(np.argmax(p)), int(np.argmin(p)), *window_shares(p).tolist()]
    assert history_error < 1e-12
    old = pd.read_csv(source_dir / "shape_future.csv")
    old = old.loc[old.H.eq(8) & old.H_pv.eq(8)].set_index(keys)
    max_changes = np.max(abs(f[SHAPE_COLS].to_numpy(float) - old.loc[f.index, SHAPE_COLS].to_numpy(float)),axis=1)
    gate = formal.peak_valley_evolution_enabled.to_numpy(bool)
    assert np.max(max_changes[~gate], initial=0) < 1e-12
    school = f.loc[("学校", "冬季", "普通周末")]
    assert abs(school.future_peak_valley_rate - .7348577076736251) < 1e-12
    summary = {
        "model_version": MODEL_VERSION, "method_revision": METHOD_REVISION,
        "H": 8, "phi": PEAK_VALLEY_FORMAL_PHI, "historical_curves": 576,
        "future_curves": len(formal), "changed_future_curves": int((max_changes > 1e-12).sum()),
        "peak_valley_gate_enabled": int(gate.sum()), "historical_max_error": history_error,
        "school_winter_weekend_peak_valley": float(school.future_peak_valley_rate),
        "max_normalization_error": float(np.max(abs(curves.sum(axis=1)-1))),
        "sources": {name: {"path": str((source_dir/name).resolve()),
            "sha256": hashlib.sha256((source_dir/name).read_bytes()).hexdigest()}
            for name in ["baseline_shapes.csv", "cell_quality.csv", "data.json", "shape_future.csv"]},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in [("baseline_shapes",baseline),("cell_quality",quality),
                        ("shape_history",history),("shape_future",future)]:
        table.to_csv(output_dir/f"{name}.csv", index=False, encoding="utf-8-sig")
    (output_dir/"data.json").write_text(json.dumps(groups,ensure_ascii=False),encoding="utf-8")
    (output_dir/"run_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_dir,args.output_dir),ensure_ascii=False,indent=2))
