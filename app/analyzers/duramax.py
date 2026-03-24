from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.core.parser import calc_log_duration, choose_best_column, series_numeric
from app.core.trust import finalize_trust_buckets


CANONICAL_ALIASES: Dict[str, List[str]] = {
    "Time_sec": ["Time_sec", "Time", "Elapsed Time", "Offset", "Time (s)"],
    "RPM": ["Engine RPM", "RPM", "Engine Speed", "Engine Speed (SAE)"],
    "MAP_kPa": [
        "Intake Manifold Absolute Pressure",
        "Intake Manifold Absolute Pressure (SAE)",
        "MAP",
        "MAP (SAE)",
        "Boost/Vacuum",
    ],
    "Boost_kPa": ["Boost", "Boost Pressure", "Boost Pressure Desired", "Boost PSI", "Boost/Vacuum"],
    "BoostDesired_kPa": ["Desired Boost", "Boost Pressure Desired", "Boost Desired", "Desired MAP"],
    "APP_pct": ["Accelerator Pedal Position", "APP", "APP %", "Accelerator Position"],
    "Throttle_pct": ["Throttle Position", "ETC Throttle Position", "Throttle Position (%)"],
    "VSS_mph": ["Vehicle Speed", "MPH", "Speed", "Vehicle Speed (SAE)"],
    "RailPressure_kPa": [
        "Fuel Rail Pressure",
        "Fuel Rail Pressure Desired",
        "Fuel Rail Pressure (MPa)",
        "Fuel Rail Pressure (kPa)",
        "Rail Pressure",
    ],
    "RailPressureDesired_kPa": [
        "Desired Fuel Rail Pressure",
        "Fuel Rail Pressure Desired",
        "Rail Pressure Desired",
    ],
    "MainPulse_us": ["Main Injection Pulse", "Main Injection Duration", "Main PW", "Main Pulse Width"],
    "PilotPulse_us": ["Pilot Injection Pulse", "Pilot PW", "Pilot Pulse Width"],
    "SOI_deg": ["Main Injection Timing", "Injection Timing", "SOI", "Main SOI"],
    "MAF_gps": ["MAF Airflow", "Mass Air Flow", "MAF", "MAF (g/s)"],
    "IAT_C": ["Intake Air Temp", "IAT", "Intake Air Temperature"],
    "ECT_C": ["Engine Coolant Temp", "ECT", "Coolant Temp"],
    "VanePos_pct": ["Turbo Vane Position", "VGT Position", "Vane Position", "Vane Pos"],
    "VaneCmd_pct": ["Desired Vane Position", "Vane Position Desired", "VGT Desired"],
    "Lambda": ["Lambda", "Wideband Lambda", "External Wideband Lambda"],
}


ESSENTIAL_ANALYSIS_KEYS = ["RPM", "MAP_kPa"]
PRESSURE_CANONICALS = {"MAP_kPa", "Boost_kPa", "BoostDesired_kPa", "RailPressure_kPa", "RailPressureDesired_kPa"}

IDLE_RPM_MAX = 900.0
IDLE_APP_MAX = 8.0
IDLE_SPEED_MAX = 3.0
LOAD_APP_THRESHOLD = 55.0
LOAD_MAP_THRESHOLD = 130.0
BOOST_THRESHOLD_KPA = 15.0


def map_columns(df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    matched_raw_columns: Dict[str, str] = {}
    trust_buckets = {
        "confirmed_channels": [],
        "suspect_channels": [],
        "missing_channels": [],
        "invalid_channels": [],
        "uncertain_channels": [],
    }

    for canonical, aliases in CANONICAL_ALIASES.items():
        match = choose_best_column(list(df.columns), aliases)
        if match:
            matched_raw_columns[canonical] = match
            trust_buckets["confirmed_channels"].append(canonical)
        else:
            trust_buckets["missing_channels"].append(canonical)

    return matched_raw_columns, trust_buckets


def infer_pressure_mode_and_normalize(series: pd.Series, canonical: str) -> Tuple[pd.Series, str]:
    s = series.copy().astype(float)
    s = s.where(np.isfinite(s), np.nan)
    finite = s.dropna()
    if finite.empty:
        return s, "unknown"

    q95 = float(finite.quantile(0.95))

    if canonical in {"MAP_kPa", "Boost_kPa", "BoostDesired_kPa"}:
        if q95 <= 70:
            return s * 6.89475729, "psi_to_kPa"
        return s, "kPa"

    if canonical in {"RailPressure_kPa", "RailPressureDesired_kPa"}:
        if q95 <= 350:
            return s * 1000.0, "MPa_to_kPa"
        if q95 <= 3500:
            return s * 100.0, "bar_to_kPa"
        return s, "kPa"

    return s, "as_logged"


def is_flat(series: pd.Series, tol: float = 1e-6) -> bool:
    finite = series.dropna()
    if finite.empty:
        return False
    if finite.size < 20:
        return False
    return bool((finite.diff().abs().fillna(0) <= tol).mean() >= 0.98)


def build_numeric_frame(df: pd.DataFrame, matched: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, str]]:
    out = pd.DataFrame(index=df.index)
    invalid_reasons: Dict[str, str] = {}
    pressure_unit_modes: Dict[str, str] = {}

    for canonical, raw_col in matched.items():
        s = series_numeric(df[raw_col])

        if canonical in PRESSURE_CANONICALS:
            s, mode = infer_pressure_mode_and_normalize(s, canonical)
            pressure_unit_modes[canonical] = mode

        if canonical in {"RailPressure_kPa", "RailPressureDesired_kPa"} and is_flat(s):
            invalid_reasons[canonical] = "flatlined_rail_pressure_signal"
            out[canonical] = pd.Series(np.nan, index=df.index, dtype=float)
            continue

        out[canonical] = s.astype(float)

    return out, invalid_reasons, pressure_unit_modes


def determine_operating_mode(num: pd.DataFrame) -> Dict[str, Any]:
    mode: Dict[str, Any] = {
        "idle_detected": False,
        "high_load_detected": False,
        "boost_present": False,
        "max_map_kpa": None,
    }

    rpm = num["RPM"] if "RPM" in num else pd.Series(dtype=float)
    app = num["APP_pct"] if "APP_pct" in num else pd.Series(dtype=float)
    speed = num["VSS_mph"] if "VSS_mph" in num else pd.Series(dtype=float)
    map_kpa = num["MAP_kPa"] if "MAP_kPa" in num else pd.Series(dtype=float)

    if rpm.notna().any():
        mode["max_rpm"] = float(np.nanmax(rpm))

    if map_kpa.notna().any():
        mode["max_map_kpa"] = float(np.nanmax(map_kpa))
        mode["boost_present"] = bool(max(float(np.nanmax(map_kpa)) - 100.0, 0.0) >= BOOST_THRESHOLD_KPA)

    if rpm.notna().any() and app.notna().any():
        if speed.notna().any():
            idle_mask = (rpm <= IDLE_RPM_MAX) & (app <= IDLE_APP_MAX) & (speed <= IDLE_SPEED_MAX)
        else:
            idle_mask = (rpm <= IDLE_RPM_MAX) & (app <= IDLE_APP_MAX)
        mode["idle_detected"] = bool(idle_mask.fillna(False).any())

        if map_kpa.notna().any():
            mode["high_load_detected"] = bool(((app >= LOAD_APP_THRESHOLD) & (map_kpa >= LOAD_MAP_THRESHOLD)).fillna(False).any())
        else:
            mode["high_load_detected"] = bool((app >= LOAD_APP_THRESHOLD).fillna(False).any())

    return mode


def compute_tracking_diagnostics(num: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "boost_tracking": {"available": False, "max_error_kpa": None, "mean_error_kpa": None, "degraded": False},
        "rail_tracking": {"available": False, "max_error_kpa": None, "mean_error_kpa": None, "degraded": False},
        "vane_tracking": {"available": False, "max_error_pct": None, "mean_error_pct": None, "degraded": False},
    }

    if "Boost_kPa" in num and "BoostDesired_kPa" in num:
        actual = num["Boost_kPa"]
        desired = num["BoostDesired_kPa"]
        valid = actual.notna() & desired.notna()
        if valid.any():
            err = (desired[valid] - actual[valid]).abs()
            out["boost_tracking"] = {
                "available": True,
                "max_error_kpa": float(err.max()),
                "mean_error_kpa": float(err.mean()),
                "degraded": bool(float(err.mean()) > 25.0 or float(err.max()) > 60.0),
            }

    if "RailPressure_kPa" in num and "RailPressureDesired_kPa" in num:
        actual = num["RailPressure_kPa"]
        desired = num["RailPressureDesired_kPa"]
        valid = actual.notna() & desired.notna()
        if valid.any():
            err = (desired[valid] - actual[valid]).abs()
            out["rail_tracking"] = {
                "available": True,
                "max_error_kpa": float(err.max()),
                "mean_error_kpa": float(err.mean()),
                "degraded": bool(float(err.mean()) > 15000.0 or float(err.max()) > 35000.0),
            }

    if "VanePos_pct" in num and "VaneCmd_pct" in num:
        actual = num["VanePos_pct"]
        desired = num["VaneCmd_pct"]
        valid = actual.notna() & desired.notna()
        if valid.any():
            err = (desired[valid] - actual[valid]).abs()
            out["vane_tracking"] = {
                "available": True,
                "max_error_pct": float(err.max()),
                "mean_error_pct": float(err.mean()),
                "degraded": bool(float(err.mean()) > 10.0 or float(err.max()) > 20.0),
            }

    return out


def compute_rail_trust(num: pd.DataFrame, operating_mode: Dict[str, Any]) -> Tuple[bool, str, List[str], Dict[str, Any]]:
    uncertain: List[str] = []
    diagnostics: Dict[str, Any] = {"rail_channel_used": None, "checks": []}

    if "RailPressure_kPa" not in num or num["RailPressure_kPa"].dropna().size < 10:
        return False, "missing_rail_pressure_actual", uncertain, diagnostics

    rail = num["RailPressure_kPa"].dropna()
    diagnostics["rail_channel_used"] = "RailPressure_kPa"

    if float(rail.quantile(0.95)) < 12000.0:
        uncertain.append("RailPressure_kPa")
        diagnostics["checks"].append("rail_pressure_too_low_for_common_duramax_scale")
        return False, "rail_pressure_scale_or_channel_mismatch", uncertain, diagnostics

    if operating_mode.get("high_load_detected") and float(rail.quantile(0.75)) < 30000.0:
        uncertain.append("RailPressure_kPa")
        diagnostics["checks"].append("high_load_with_unusually_low_rail_pressure")
        return False, "high_load_low_rail_conflict", uncertain, diagnostics

    return True, "trusted", uncertain, diagnostics


def analyze_dataframe(df: pd.DataFrame, meta: Dict[str, Any]) -> Dict[str, Any]:
    matched_raw_columns, trust_buckets = map_columns(df)
    num, invalid_reasons, pressure_unit_modes = build_numeric_frame(df, matched_raw_columns)

    hard_stop_reasons: List[str] = []
    for key in ESSENTIAL_ANALYSIS_KEYS:
        if key not in matched_raw_columns:
            hard_stop_reasons.append(f"missing_{key}")
        elif key in num and num[key].dropna().size < 5:
            hard_stop_reasons.append(f"insufficient_{key}")

    uncertain: List[str] = []
    suspect: List[str] = []

    operating_mode = determine_operating_mode(num)
    tracking = compute_tracking_diagnostics(num)
    rail_trusted, rail_reason, rail_uncertain, rail_diag = compute_rail_trust(num, operating_mode)
    uncertain.extend(rail_uncertain)

    if tracking["boost_tracking"]["degraded"]:
        uncertain.extend(["Boost_kPa", "BoostDesired_kPa"])
    if tracking["rail_tracking"]["degraded"]:
        uncertain.extend(["RailPressure_kPa", "RailPressureDesired_kPa"])
    if tracking["vane_tracking"]["degraded"]:
        suspect.extend(["VanePos_pct", "VaneCmd_pct"])

    notes: List[str] = []
    if hard_stop_reasons:
        notes.append("Essential channels missing or insufficient; analysis is limited.")
    if invalid_reasons:
        notes.append("Flatlined rail-pressure channels marked invalid and excluded.")
    if tracking["boost_tracking"]["degraded"]:
        notes.append("Boost actual-vs-desired tracking error is high.")
    if tracking["rail_tracking"]["degraded"]:
        notes.append("Rail pressure actual-vs-desired tracking error is high.")
    if not rail_trusted:
        notes.append("Fueling guidance is limited because rail-pressure trust is not confirmed.")

    trust_buckets = finalize_trust_buckets(trust_buckets, invalid_reasons, uncertain, suspect)

    fueling_guidance: Dict[str, Any] = {
        "can_make_closed_loop_trim_based_suggestions": False,
        "can_make_wot_fueling_suggestions": bool(rail_trusted and not tracking["rail_tracking"]["degraded"]),
        "reason_wot_fueling_limited": None,
    }
    if not fueling_guidance["can_make_wot_fueling_suggestions"]:
        fueling_guidance["reason_wot_fueling_limited"] = (
            "No trustworthy rail-pressure actual for high-load fueling corrections."
        )

    summary = {
        "rail_pressure_trusted": bool(rail_trusted),
        "rail_pressure_reason": rail_reason,
        "idle_detected": bool(operating_mode.get("idle_detected", False)),
        "high_load_detected": bool(operating_mode.get("high_load_detected", False)),
        "boost_present": bool(operating_mode.get("boost_present", False)),
        "max_map_kpa": operating_mode.get("max_map_kpa"),
        "log_duration_sec": calc_log_duration(num),
    }

    return {
        "status": "ready" if not hard_stop_reasons else "limited",
        "filename": meta["filename"],
        "size_bytes": meta["size_bytes"],
        "row_count": meta["row_count"],
        "column_count": meta["column_count"],
        "header_row_index": meta["header_row_index"],
        "first_data_row_index": meta["first_data_row_index"],
        "log_duration_sec": summary["log_duration_sec"],
        "matched_raw_columns": matched_raw_columns,
        "pressure_unit_modes": pressure_unit_modes,
        "trust_buckets": trust_buckets,
        "invalid_channel_reasons": invalid_reasons,
        "hard_stop_reasons": hard_stop_reasons,
        "summary": summary,
        "operating_mode": operating_mode,
        "rail_diagnostics": rail_diag,
        "tracking_diagnostics": tracking,
        "fueling_guidance": fueling_guidance,
        "notes": notes,
    }


def validate_dataframe(df: pd.DataFrame, filename: str, mime_type: Optional[str], meta: Dict[str, Any]) -> Dict[str, Any]:
    matched_raw_columns, trust_buckets = map_columns(df)
    num, invalid_reasons, pressure_unit_modes = build_numeric_frame(df, matched_raw_columns)

    hard_stop_reasons: List[str] = []
    for key in ESSENTIAL_ANALYSIS_KEYS:
        if key not in matched_raw_columns:
            hard_stop_reasons.append(f"missing_{key}")
        elif key in num and num[key].dropna().size < 5:
            hard_stop_reasons.append(f"insufficient_{key}")

    trust_buckets = finalize_trust_buckets(trust_buckets, invalid_reasons, [], [])

    return {
        "status": "ready" if not hard_stop_reasons else "limited",
        "filename": filename,
        "content_type": mime_type or "text/csv",
        "size_bytes": meta["size_bytes"],
        "row_count": meta["row_count"],
        "column_count": meta["column_count"],
        "header_row_index": meta["header_row_index"],
        "first_data_row_index": meta["first_data_row_index"],
        "log_duration_sec": calc_log_duration(num),
        "matched_raw_columns": matched_raw_columns,
        "pressure_unit_modes": pressure_unit_modes,
        "trust_buckets": trust_buckets,
        "invalid_channel_reasons": invalid_reasons,
        "hard_stop_reasons": hard_stop_reasons,
    }
