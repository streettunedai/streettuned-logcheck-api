from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from app.core.parser import calc_log_duration, choose_best_column, series_numeric
from app.core.trust import finalize_trust_buckets


CANONICAL_ALIASES: Dict[str, List[str]] = {
    "Time_sec": ["Time_sec", "Time", "Elapsed Time", "Offset", "Time (s)"],
    "engine_rpm": ["Engine RPM", "RPM", "Engine Speed", "Engine Speed (SAE)"],
    "vehicle_speed": ["Vehicle Speed", "Vehicle Speed (SAE)", "Speed", "MPH"],
    "boost_psi": ["Boost", "Boost PSI", "Boost Pressure", "Turbo Boost", "Boost/Vacuum"],
    "map_kpa": ["MAP", "MAP (SAE)", "Intake Manifold Absolute Pressure", "Manifold Absolute Pressure"],
    "egt": ["EGT", "Exhaust Gas Temp", "Exhaust Gas Temperature", "Pyrometer"],
    "intake_air_temp": ["IAT", "Intake Air Temp", "Intake Air Temperature"],
    "coolant_temp": ["ECT", "Engine Coolant Temp", "Coolant Temp", "Coolant Temperature"],
    "throttle_position": ["Throttle Position", "TPS", "Throttle Position (%)"],
    "apps": ["APP", "APP %", "Accelerator Pedal Position", "Accelerator Position"],
    "fuel_pressure": ["Fuel Pressure", "Lift Pump Pressure", "Supply Pressure", "Low Side Fuel Pressure"],
    "rail_pressure_commanded": [
        "Rail Pressure Desired",
        "Desired Fuel Rail Pressure",
        "Fuel Rail Pressure Desired",
        "Commanded Rail Pressure",
    ],
    "rail_pressure_actual": ["Rail Pressure", "Fuel Rail Pressure", "Actual Rail Pressure", "Rail Pressure Actual"],
    "injection_quantity": ["Injection Quantity", "Main Injection Quantity", "Fuel Quantity", "mm3"],
    "pulse_width": ["Injector Pulse Width", "Injection Pulse Width", "Main PW", "Pulse Width"],
    "gear": ["Gear", "Current Gear", "Trans Gear"],
    "converter_slip": ["Converter Slip", "TCC Slip", "Torque Converter Slip"],
    "lockup": ["TCC Lockup", "Lockup", "Converter Clutch State"],
    "torque_nm": ["Torque", "Engine Torque", "Torque (SAE)", "Torque_Actual_Nm"],
    "cylinder_airmass": ["Cylinder Airmass", "Cylinder Air Mass", "Airmass per Cylinder"],
}

CUMMINS_PLATFORMS = {
    "cummins_12v_ve",
    "cummins_12v_ppump",
    "cummins_24v_vp44",
    "cummins_5_9_common_rail",
}

COMMON_REQUIRED = ["engine_rpm", "egt"]


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


def build_numeric_frame(df: pd.DataFrame, matched: Dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for canonical, raw_col in matched.items():
        out[canonical] = series_numeric(df[raw_col]).astype(float)

    if "map_kpa" in out and "boost_psi" not in out:
        out["boost_psi"] = ((out["map_kpa"] - 101.325).clip(lower=0.0)) / 6.89475729
    if "boost_psi" in out and "map_kpa" not in out:
        out["map_kpa"] = out["boost_psi"] * 6.89475729 + 101.325

    return out


def detect_cummins_platform(df: pd.DataFrame, matched: Dict[str, str], num: Optional[pd.DataFrame] = None) -> Tuple[Optional[str], Dict[str, Any]]:
    if not matched:
        matched, _ = map_columns(df)
    lower_cols = {c.strip().lower() for c in df.columns}
    num = num if num is not None else pd.DataFrame()

    has_boost = "boost_psi" in matched or "map_kpa" in matched
    has_egt = "egt" in matched
    has_fuel_pressure = "fuel_pressure" in matched
    has_rail = "rail_pressure_actual" in matched or "rail_pressure_commanded" in matched
    has_diesel_fueling = "injection_quantity" in matched or "pulse_width" in matched
    has_gm_markers = "torque_nm" in matched or "cylinder_airmass" in matched

    diesel_score = sum([has_boost, has_egt, has_fuel_pressure or has_rail, has_diesel_fueling])

    details = {
        "has_boost": has_boost,
        "has_egt": has_egt,
        "has_fuel_pressure": has_fuel_pressure,
        "has_rail": has_rail,
        "has_diesel_fueling": has_diesel_fueling,
        "has_gm_markers": has_gm_markers,
        "diesel_score": diesel_score,
    }

    if has_gm_markers or diesel_score < 3:
        return None, details

    if has_rail:
        return "cummins_5_9_common_rail", details

    vp44_hint = any("vp44" in c for c in lower_cols) or (has_fuel_pressure and ("apps" in matched or has_diesel_fueling))
    if vp44_hint:
        return "cummins_24v_vp44", details

    ppump_hint = any("p-pump" in c or "ppump" in c for c in lower_cols)
    if ppump_hint:
        return "cummins_12v_ppump", details

    return "cummins_12v_ve", details


def required_channels_for_platform(platform: str, matched: Dict[str, str]) -> List[str]:
    missing: List[str] = []
    for req in COMMON_REQUIRED:
        if req not in matched:
            missing.append(req)

    if "boost_psi" not in matched and "map_kpa" not in matched:
        missing.append("boost_psi_or_map_kpa")

    if platform in {"cummins_12v_ve", "cummins_12v_ppump"}:
        return missing
    if platform == "cummins_24v_vp44":
        if "fuel_pressure" not in matched:
            missing.append("fuel_pressure")
        return missing
    if platform == "cummins_5_9_common_rail":
        if "rail_pressure_commanded" not in matched:
            missing.append("rail_pressure_commanded")
        if "rail_pressure_actual" not in matched:
            missing.append("rail_pressure_actual")
        return missing
    return missing


def _load_mask(num: pd.DataFrame) -> pd.Series:
    if "apps" in num:
        return (num["apps"] >= 55.0).fillna(False)
    if "throttle_position" in num:
        return (num["throttle_position"] >= 55.0).fillna(False)
    if "boost_psi" in num:
        return (num["boost_psi"] >= 8.0).fillna(False)
    return pd.Series(False, index=num.index)


def analyze_dataframe(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    requested_platform: Optional[str] = None,
) -> Dict[str, Any]:
    matched_raw_columns, trust_buckets = map_columns(df)
    num = build_numeric_frame(df, matched_raw_columns)

    detected_platform, detection_details = detect_cummins_platform(df, matched_raw_columns, num)
    platform = requested_platform or detected_platform

    if platform not in CUMMINS_PLATFORMS:
        return {
            "status": "error",
            "error_type": "platform_detection_failed",
            "error": {
                "message": "Unable to classify this log as a supported Cummins platform.",
                "detection_details": detection_details,
            },
        }

    missing_required = required_channels_for_platform(platform, matched_raw_columns)
    if missing_required:
        return {
            "status": "error",
            "error_type": "missing_required_channels",
            "error": {
                "platform": platform,
                "required_missing": missing_required,
                "message": "Required Cummins channels are missing; analysis not executed.",
            },
            "matched_raw_columns": matched_raw_columns,
        }

    load_mask = _load_mask(num)
    findings: List[str] = []
    changes_required: List[str] = []
    verify_next: List[str] = []
    hard_stops: List[str] = []

    if "egt" in num and load_mask.any():
        egt_load = num.loc[load_mask, "egt"].dropna()
        if not egt_load.empty and float(egt_load.quantile(0.95)) > 1350.0:
            findings.append("high EGT under load")
            changes_required.append("reduce fueling in high-load cells until EGT stabilizes")
            verify_next.append("repeat pull with stable ambient and confirm EGT stays below target")

    if "boost_psi" in num and load_mask.any():
        boost_load = num.loc[load_mask, "boost_psi"].dropna()
        if not boost_load.empty and float(boost_load.quantile(0.8)) < 12.0:
            findings.append("low boost / slow spool")
            changes_required.append("inspect boost leaks, wastegate/VGT control, and turbine flow before fueling changes")
            hard_stops.append("do_not_add_fueling_with_low_boost")

    if "fuel_pressure" in num and platform == "cummins_24v_vp44" and load_mask.any():
        fp = num.loc[load_mask, "fuel_pressure"].dropna()
        if not fp.empty and float(fp.quantile(0.1)) < 10.0:
            findings.append("fuel supply issue (VP44)")
            changes_required.append("repair low lift-pump pressure before calibration changes")
            hard_stops.append("fuel_pressure_issue_must_be_fixed_first")

    if platform == "cummins_5_9_common_rail" and "rail_pressure_actual" in num and "rail_pressure_commanded" in num:
        valid = num["rail_pressure_actual"].notna() & num["rail_pressure_commanded"].notna()
        if valid.any():
            err = (num.loc[valid, "rail_pressure_commanded"] - num.loc[valid, "rail_pressure_actual"]).abs()
            if float(err.quantile(0.9)) > 15000.0:
                findings.append("rail pressure deviation (common rail)")
                changes_required.append("stabilize rail pressure tracking before increasing pulse or timing")
                verify_next.append("capture commanded vs actual rail pressure on identical load")

    if {"boost_psi", "injection_quantity"}.issubset(set(num.columns)) and load_mask.any():
        pre_spool = load_mask & (num["boost_psi"] < 5.0)
        if pre_spool.any() and float(num.loc[pre_spool, "injection_quantity"].quantile(0.8)) > 40.0:
            findings.append("overfueling before boost")
            changes_required.append("reduce low-boost fueling and reshape torque ramp")

    if "converter_slip" in num:
        slip = num.loc[load_mask, "converter_slip"].dropna() if load_mask.any() else num["converter_slip"].dropna()
        if not slip.empty and float(slip.quantile(0.9)) > 250.0:
            hard_stops.append("transmission_slip_detected_no_power_increase")
            findings.append("possible transmission slip")

    if not findings:
        findings.append("insufficient signal overlap for definitive diesel fault classification")
        verify_next.append("log boost, EGT, fueling, and pressure channels in the same pull")

    if "low boost / slow spool" in findings and "overfueling before boost" in findings:
        findings.append("possible turbo mismatch")
    if platform == "cummins_5_9_common_rail" and "rail pressure deviation (common rail)" in findings:
        findings.append("possible injector issue")

    trust_buckets = finalize_trust_buckets(trust_buckets, {}, [], [])

    return {
        "status": "ready",
        "platform": platform,
        "filename": meta["filename"],
        "log_duration_sec": calc_log_duration(num),
        "matched_raw_columns": matched_raw_columns,
        "trust_buckets": trust_buckets,
        "hard_stops": sorted(set(hard_stops)),
        "analysis": {
            "data_summary": "; ".join(findings),
            "root_cause": findings[0],
            "changes_required": "; ".join(sorted(set(changes_required))) if changes_required else "No change recommendation without fuller data.",
            "verify_next": "; ".join(sorted(set(verify_next))) if verify_next else "Capture repeat pull with required Cummins channels.",
        },
    }


def validate_dataframe(
    df: pd.DataFrame,
    filename: str,
    mime_type: Optional[str],
    meta: Dict[str, Any],
    requested_platform: Optional[str] = None,
) -> Dict[str, Any]:
    matched_raw_columns, trust_buckets = map_columns(df)
    num = build_numeric_frame(df, matched_raw_columns)
    detected_platform, detection_details = detect_cummins_platform(df, matched_raw_columns, num)

    platform = requested_platform or detected_platform
    if platform not in CUMMINS_PLATFORMS:
        return {
            "status": "error",
            "error_type": "platform_detection_failed",
            "error": {
                "message": "Unable to classify this log as a supported Cummins platform.",
                "detection_details": detection_details,
            },
        }

    missing_required = required_channels_for_platform(platform, matched_raw_columns)
    trust_buckets = finalize_trust_buckets(trust_buckets, {}, [], [])

    return {
        "status": "ready" if not missing_required else "error",
        "filename": filename,
        "content_type": mime_type or "text/csv",
        "size_bytes": meta["size_bytes"],
        "row_count": meta["row_count"],
        "column_count": meta["column_count"],
        "log_duration_sec": calc_log_duration(num),
        "platform": platform,
        "matched_raw_columns": matched_raw_columns,
        "trust_buckets": trust_buckets,
        "hard_stop_reasons": [f"missing_{m}" for m in missing_required],
        "error": None
        if not missing_required
        else {
            "message": "Required Cummins channels are missing; analysis blocked.",
            "required_missing": missing_required,
        },
    }
