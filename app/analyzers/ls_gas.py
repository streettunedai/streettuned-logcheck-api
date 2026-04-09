from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.core.parser import calc_log_duration, choose_best_column, safe_float, series_numeric
from app.core.trust import finalize_trust_buckets


CANONICAL_ALIASES: Dict[str, List[str]] = {
    "Time_sec": [
        "Time_sec",
        "Time",
        "Offset",
        "Time (s)",
        "Elapsed Time",
    ],
    "RPM": [
        "Engine RPM",
        "RPM",
        "Engine Speed",
        "Engine Speed (SAE)",
    ],
    "MAP_kPa": [
        "Intake Manifold Absolute Pressure (SAE)",
        "MAP (SAE)",
        "MAP",
        "Manifold Absolute Pressure",
        "Intake Manifold Absolute Pressure",
        "Boost Vacuum",
        "Boost/Vacuum",
    ],
    "Spark_deg": [
        "Timing Advance (SAE)",
        "Spark Advance",
        "Timing Advance",
        "Ignition Advance",
        "Spark",
        "Timing",
        "Ignition Timing",
    ],
    "KR_deg": [
        "Knock Retard",
        "Spark Retard",
        "KR",
    ],
    "TotalKR_deg": [
        "Total Knock Retard",
        "Total KR",
        "Knock Retard Total",
    ],
    "EQ_Cmd": [
        "Equivalence Ratio Commanded (SAE)",
        "Commanded EQ Ratio",
        "Commanded Lambda",
        "Lambda Commanded",
        "EQ Ratio Commanded",
        "EQ Cmd",
    ],
    "AFR_Cmd": [
        "Air-Fuel Ratio Commanded",
        "Commanded AFR",
        "AFR Commanded",
    ],
    "FuelSys1_Status": [
        "Fuel System #1 Status (SAE)",
        "Fuel Sys 1 Status",
        "Fuel System 1 Status",
    ],
    "TPS_pct": [
        "Absolute Throttle Position",
        "Throttle Position",
        "Throttle Angle",
        "TPS",
        "TPS %",
        "Throttle Position (%)",
        "Throttle Position Sensor",
    ],
    "Throttle_Actual_pct": [
        "Throttle Position",
        "Throttle Position (%)",
        "ETC Throttle Position",
        "Throttle Actual",
        "Throttle Blade Position",
        "Actual Throttle Position",
    ],
    "Throttle_Desired_pct": [
        "Desired Throttle Position",
        "Throttle Position Desired",
        "Throttle Desired",
    ],
    "Throttle_Commanded_pct": [
        "Commanded Throttle",
        "Throttle Commanded",
        "Commanded Throttle Position",
    ],
    "Pedal_pct": [
        "Accelerator Pedal Position",
        "APP",
        "APP %",
        "Accelerator Pedal Position D",
        "Accel Pedal Position",
        "Accelerator Position",
    ],
    "IAT_C": [
        "Intake Air Temp",
        "Intake Air Temperature",
        "IAT",
        "IAT (SAE)",
        "Intake Air Temp (SAE)",
    ],
    "ECT_C": [
        "Engine Coolant Temp",
        "ECT",
        "Coolant Temp",
        "Coolant Temperature",
        "Engine Coolant Temperature (SAE)",
    ],
    "STFT1_pct": [
        "Short Term FT B1",
        "STFT B1",
        "STFT1",
        "Short Term Fuel Trim Bank 1",
    ],
    "LTFT1_pct": [
        "Long Term FT B1",
        "LTFT B1",
        "LTFT1",
        "Long Term Fuel Trim Bank 1",
    ],
    "STFT2_pct": [
        "Short Term FT B2",
        "STFT B2",
        "STFT2",
        "Short Term Fuel Trim Bank 2",
    ],
    "LTFT2_pct": [
        "Long Term FT B2",
        "LTFT B2",
        "LTFT2",
        "Long Term Fuel Trim Bank 2",
    ],
    "O2S11_V": [
        "O2 Sensor Voltage B1S1",
        "O2 B1S1",
        "Bank 1 Sensor 1 O2",
        "O2S11",
    ],
    "O2S21_V": [
        "O2 Sensor Voltage B2S1",
        "O2 B2S1",
        "Bank 2 Sensor 1 O2",
        "O2S21",
    ],
    "WB_AFR": [
        "AEM Air/Fuel Ratio",
        "AEM AFR",
        "Wideband AFR",
        "AFR",
        "Lambda 1",
        "Lambda",
        "External Wideband AFR",
        "AEM UEGO AFR",
        "AEM UEGO Lambda",
    ],
    "WB_Lambda": [
        "AEM Lambda",
        "Lambda 1",
        "Lambda",
        "External Wideband Lambda",
        "Wideband Lambda",
        "AEM UEGO Lambda",
    ],
    "Boost_kPa": [
        "Boost",
        "Boost Pressure",
        "Boost/Vacuum",
        "Boost Vacuum",
        "Manifold Gauge Pressure",
    ],
    "FuelPressure_kPa": [
        "Fuel Pressure",
        "Fuel Rail Pressure",
        "Fuel PSI",
        "Fuel Pressure (PSI)",
        "Fuel Pressure (kPa)",
    ],
    "OilPressure_kPa": [
        "Oil Pressure",
        "Engine Oil Pressure",
        "Oil PSI",
        "Oil Pressure (PSI)",
        "Oil Pressure (kPa)",
    ],
    "VSS_mph": [
        "Vehicle Speed",
        "Speed",
        "MPH",
        "Vehicle Speed (SAE)",
    ],
    "Gear": [
        "Gear",
        "Trans Gear",
        "Current Gear",
    ],
    "InputSpeed_rpm": [
        "Input Speed",
        "Transmission Input Speed",
    ],
    "OutputSpeed_rpm": [
        "Output Speed",
        "Transmission Output Speed",
    ],
    "Slip_rpm": [
        "Slip",
        "TCC Slip",
        "Converter Slip",
    ],
    "DesiredTorque": [
        "Desired Torque",
        "Driver Desired Torque",
    ],
    "ActualTorque": [
        "Actual Torque",
        "Delivered Torque",
    ],
    "InjectorPW_ms": [
        "Injector Pulse Width",
        "Inj PW",
        "Pulse Width",
        "Injector PW",
    ],
    "MAF_Hz": [
        "MAF Frequency",
        "MAF Airflow Frequency",
        "MAF Hz",
    ],
}


ESSENTIAL_ANALYSIS_KEYS = ["RPM", "MAP_kPa", "TPS_pct"]
PRESSURE_CANONICALS = {"FuelPressure_kPa", "OilPressure_kPa", "MAP_kPa", "Boost_kPa"}

WOT_TPS_THRESHOLD = 75.0
IDLE_RPM_MAX = 1100.0
IDLE_TPS_MAX = 5.0
IDLE_SPEED_MAX = 3.0


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


def forward_fill_sparse(series: pd.Series, max_gap: int = 8) -> pd.Series:
    s = series.copy()
    non_na = s.notna().sum()
    if len(s) == 0:
        return s
    if 0 < non_na < len(s) * 0.7:
        return s.ffill(limit=max_gap)
    return s


def infer_pressure_mode_and_normalize(series: pd.Series, canonical: str) -> Tuple[pd.Series, str]:
    s = series.copy()
    s = s.where(np.isfinite(s), np.nan)

    finite = s.dropna()
    if finite.empty:
        return s, "unknown"

    q50 = float(finite.quantile(0.50))
    q95 = float(finite.quantile(0.95))

    if canonical == "MAP_kPa":
        if q95 <= 35:
            return s * 6.89475729, "psi_to_kPa"
        return s, "kPa"

    if canonical == "Boost_kPa":
        if q95 <= 60:
            return s * 6.89475729, "psi_to_kPa"
        return s, "kPa"

    if canonical in {"FuelPressure_kPa", "OilPressure_kPa"}:
        if q95 <= 250:
            return s * 6.89475729, "psi_to_kPa"
        return s, "kPa"

    if q50 <= 60:
        return s * 6.89475729, "psi_to_kPa"
    return s, "kPa"


def is_flat_zero_junk(series: pd.Series) -> bool:
    finite = series.dropna()
    if finite.empty:
        return False
    zero_ratio = float((finite.abs() < 1e-9).mean())
    return zero_ratio >= 0.95


def build_numeric_frame(df: pd.DataFrame, matched: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, str]]:
    out = pd.DataFrame(index=df.index)
    invalid_reasons: Dict[str, str] = {}
    pressure_unit_modes: Dict[str, str] = {}

    for canonical, raw_col in matched.items():
        s = series_numeric(df[raw_col])
        s = forward_fill_sparse(s)

        if canonical in PRESSURE_CANONICALS:
            s, mode = infer_pressure_mode_and_normalize(s, canonical)
            pressure_unit_modes[canonical] = mode

        if canonical in {"FuelPressure_kPa", "OilPressure_kPa"} and is_flat_zero_junk(s):
            invalid_reasons[canonical] = "flat_zero_sender_or_junk"
            out[canonical] = pd.Series(np.nan, index=df.index, dtype=float)
            continue

        out[canonical] = s.astype(float)

    return out, invalid_reasons, pressure_unit_modes


def determine_operating_mode(num: pd.DataFrame) -> Dict[str, Any]:
    mode: Dict[str, Any] = {
        "idle_detected": False,
        "wot_detected": False,
        "max_map_kpa": None,
        "boost_present": False,
    }

    rpm = num["RPM"] if "RPM" in num else pd.Series(dtype=float)
    tps = num["TPS_pct"] if "TPS_pct" in num else pd.Series(dtype=float)
    vss = num["VSS_mph"] if "VSS_mph" in num else pd.Series(dtype=float)
    map_kpa = num["MAP_kPa"] if "MAP_kPa" in num else pd.Series(dtype=float)

    if not rpm.empty:
        mode["max_rpm"] = float(np.nanmax(rpm)) if rpm.notna().any() else None
    if not map_kpa.empty and map_kpa.notna().any():
        mode["max_map_kpa"] = float(np.nanmax(map_kpa))
        mode["boost_present"] = bool(np.nanmax(map_kpa) > 105)

    if not rpm.empty and not tps.empty:
        if "VSS_mph" in num and vss.notna().any():
            idle_mask = (rpm < IDLE_RPM_MAX) & (tps <= IDLE_TPS_MAX) & (vss <= IDLE_SPEED_MAX)
        else:
            idle_mask = (rpm < IDLE_RPM_MAX) & (tps <= IDLE_TPS_MAX)
        mode["idle_detected"] = bool(idle_mask.fillna(False).any())

        wot_mask = tps >= WOT_TPS_THRESHOLD
        mode["wot_detected"] = bool(wot_mask.fillna(False).any())

    return mode


def summarize_kr_window(num: pd.DataFrame, effective: pd.Series, start: int, end: int) -> Dict[str, Any]:
    window = effective.loc[start:end]
    event = {
        "start_index": int(start),
        "end_index": int(end),
        "peak_kr_deg": float(window.max()) if window.notna().any() else None,
    }

    if "Time_sec" in num and num["Time_sec"].loc[start:end].notna().any():
        ts = num["Time_sec"].loc[start:end]
        event["start_time_sec"] = float(ts.iloc[0]) if safe_float(ts.iloc[0]) is not None else None
        event["end_time_sec"] = float(ts.iloc[-1]) if safe_float(ts.iloc[-1]) is not None else None

    for ch in ["RPM", "MAP_kPa", "TPS_pct", "Spark_deg"]:
        if ch in num and num[ch].loc[start:end].notna().any():
            arr = num[ch].loc[start:end]
            event[ch] = {
                "min": float(arr.min()),
                "max": float(arr.max()),
                "avg": float(arr.mean()),
            }

    return event


def extract_kr_events(num: pd.DataFrame) -> List[Dict[str, Any]]:
    if "KR_deg" not in num and "TotalKR_deg" not in num:
        return []

    kr = num["KR_deg"] if "KR_deg" in num else pd.Series(np.nan, index=num.index)
    tkr = num["TotalKR_deg"] if "TotalKR_deg" in num else pd.Series(np.nan, index=num.index)
    effective = pd.concat([kr, tkr], axis=1).max(axis=1, skipna=True)
    mask = effective.fillna(0) > 0.5

    events: List[Dict[str, Any]] = []
    active = False
    start = None

    for idx, is_on in mask.items():
        if is_on and not active:
            active = True
            start = idx
        elif not is_on and active:
            active = False
            end = idx - 1
            events.append(summarize_kr_window(num, effective, start, end))

    if active and start is not None:
        events.append(summarize_kr_window(num, effective, start, int(num.index.max())))

    return events


def compute_wideband_trust(num: pd.DataFrame) -> Tuple[bool, str, List[str], Dict[str, Any], Optional[pd.Series]]:
    uncertain: List[str] = []
    diagnostics: Dict[str, Any] = {
        "wideband_channel_used": None,
        "wideband_interpretation": None,
        "conflicts": [],
    }

    wb_series = None
    source = None

    if "WB_Lambda" in num and num["WB_Lambda"].dropna().size > 10:
        wb_series = num["WB_Lambda"].copy()
        source = "WB_Lambda"
    elif "WB_AFR" in num and num["WB_AFR"].dropna().size > 10:
        wb_series = num["WB_AFR"].copy()
        source = "WB_AFR"

    if wb_series is None:
        diagnostics["wideband_interpretation"] = "missing"
        return False, "no_external_wideband", uncertain, diagnostics, None

    finite = wb_series.dropna()
    if finite.empty:
        diagnostics["wideband_interpretation"] = "empty"
        uncertain.append(source)
        return False, "empty_external_wideband", uncertain, diagnostics, wb_series

    if source == "WB_AFR":
        med = float(finite.median())
        if 8.0 <= med <= 20.0:
            wb_series = wb_series / 14.7
            diagnostics["wideband_interpretation"] = "afr_converted_to_lambda"
        elif 0.45 <= med <= 1.5:
            diagnostics["wideband_interpretation"] = "already_lambda_scale"
        else:
            uncertain.append(source)
            diagnostics["wideband_interpretation"] = "unrecognized_scale"
            return False, "unrecognized_wideband_scale", uncertain, diagnostics, wb_series
    else:
        med = float(finite.median())
        if not (0.45 <= med <= 1.5):
            uncertain.append(source)
            diagnostics["wideband_interpretation"] = "lambda_out_of_range"
            return False, "wideband_out_of_range", uncertain, diagnostics, wb_series
        diagnostics["wideband_interpretation"] = "lambda"

    diagnostics["wideband_channel_used"] = source

    flat_ratio = float((finite.round(4).diff().fillna(0).abs() < 1e-9).mean())
    if flat_ratio > 0.98:
        uncertain.append(source)
        diagnostics["conflicts"].append("wideband_nearly_flatlined")
        return False, "wideband_flatlined", uncertain, diagnostics, wb_series

    if "MAP_kPa" in num and "EQ_Cmd" in num:
        map_max = float(num["MAP_kPa"].dropna().max()) if num["MAP_kPa"].dropna().size else None
        eq_max = float(num["EQ_Cmd"].dropna().max()) if num["EQ_Cmd"].dropna().size else None
        if map_max is not None and eq_max is not None:
            if map_max > 105 and eq_max <= 1.02:
                uncertain.extend(["MAP_kPa", "EQ_Cmd", source])
                diagnostics["conflicts"].append("boost_or_pe_conflict")
                return False, "map_pe_conflict", uncertain, diagnostics, wb_series

    return True, "trusted", uncertain, diagnostics, wb_series


def compute_throttle_diagnostics(num: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "has_desired": "Throttle_Desired_pct" in num and num["Throttle_Desired_pct"].dropna().size > 0,
        "has_actual": "Throttle_Actual_pct" in num and num["Throttle_Actual_pct"].dropna().size > 0,
        "has_commanded": "Throttle_Commanded_pct" in num and num["Throttle_Commanded_pct"].dropna().size > 0,
        "has_pedal": "Pedal_pct" in num and num["Pedal_pct"].dropna().size > 0,
        "mismatch_detected": False,
        "max_desired_minus_actual": None,
        "notes": [],
    }

    actual = num["Throttle_Actual_pct"] if "Throttle_Actual_pct" in num else None
    desired = num["Throttle_Desired_pct"] if "Throttle_Desired_pct" in num else None
    commanded = num["Throttle_Commanded_pct"] if "Throttle_Commanded_pct" in num else None
    pedal = num["Pedal_pct"] if "Pedal_pct" in num else None

    ref = desired if desired is not None and desired.dropna().size else commanded
    if actual is not None and ref is not None and actual.dropna().size and ref.dropna().size:
        delta = ref - actual
        out["max_desired_minus_actual"] = float(delta.abs().max())
        if delta.abs().max() > 10:
            out["mismatch_detected"] = True
            out["notes"].append("actual_throttle_deviates_from_desired_or_commanded")

    if pedal is not None and actual is not None and pedal.dropna().size and actual.dropna().size:
        pedal_high = pedal >= 80
        if pedal_high.any():
            high_window = actual[pedal_high]
            if high_window.dropna().size and float(high_window.max()) < 60:
                out["mismatch_detected"] = True
                out["notes"].append("high_pedal_with_low_actual_throttle")

    return out


def compute_map_boost_conflict(num: pd.DataFrame) -> Dict[str, Any]:
    result = {"conflict": False, "notes": []}

    if "MAP_kPa" not in num:
        return result

    map_kpa = num["MAP_kPa"].dropna()
    if map_kpa.empty:
        return result

    if "Boost_kPa" in num and num["Boost_kPa"].dropna().size:
        boost = num["Boost_kPa"].dropna()
        map_max = float(map_kpa.max())
        boost_max = float(boost.max())

        if boost_max > 20 and map_max < 103:
            result["conflict"] = True
            result["notes"].append("boost_channel_positive_but_map_not_showing_boost")

        if map_max > 120 and boost_max < 3:
            result["conflict"] = True
            result["notes"].append("map_shows_boost_but_boost_channel_does_not")

    return result


def avg_bank_trims(num: pd.DataFrame) -> Optional[pd.Series]:
    parts = []
    for ch in ["STFT1_pct", "LTFT1_pct", "STFT2_pct", "LTFT2_pct"]:
        if ch in num and num[ch].dropna().size:
            parts.append(num[ch])
    if not parts:
        return None
    stacked = pd.concat(parts, axis=1)
    return stacked.mean(axis=1, skipna=True)


def analyze_dataframe(df: pd.DataFrame, meta: Dict[str, Any], platform_hint: Optional[str] = None) -> Dict[str, Any]:
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
    analysis_scope = build_analysis_scope(matched_raw_columns, operating_mode)
    platform_details = detect_platform_guess(list(df.columns), platform_hint=platform_hint)
    kr_events = extract_kr_events(num)
    throttle_diag = compute_throttle_diagnostics(num)
    map_boost_conflict = compute_map_boost_conflict(num)

    if map_boost_conflict["conflict"]:
        uncertain.extend(["MAP_kPa", "Boost_kPa"])

    wideband_trusted, wideband_reason, wb_uncertain, wb_diag, wb_lambda = compute_wideband_trust(num)
    uncertain.extend(wb_uncertain)

    # Important prior bug fix retained:
    # summary.external_wideband_trusted must match final trust state after downgrades.
    if map_boost_conflict["conflict"] and wideband_trusted:
        wideband_trusted = False
        wideband_reason = "map_boost_conflict_downgraded"

    if "EQ_Cmd" in num and num["EQ_Cmd"].dropna().size:
        eq = num["EQ_Cmd"].dropna()
        if float(eq.max()) < 0.8 or float(eq.max()) > 1.8:
            uncertain.append("EQ_Cmd")

    trim_series = avg_bank_trims(num)
    trim_summary = None
    if trim_series is not None and trim_series.dropna().size:
        trim_summary = {
            "avg_trim_pct": float(trim_series.mean()),
            "min_trim_pct": float(trim_series.min()),
            "max_trim_pct": float(trim_series.max()),
        }

    fueling_guidance: Dict[str, Any] = {
        "can_make_closed_loop_trim_based_suggestions": trim_series is not None and trim_series.dropna().size > 0,
        "can_make_wot_fueling_suggestions": False,
        "reason_wot_fueling_limited": None,
    }

    if wb_lambda is not None and wideband_trusted:
        fueling_guidance["can_make_wot_fueling_suggestions"] = True
    else:
        fueling_guidance["can_make_wot_fueling_suggestions"] = False
        fueling_guidance["reason_wot_fueling_limited"] = (
            "No trustworthy external wideband actual available; do not use narrowbands as actual AFR."
        )

    notes: List[str] = []
    if hard_stop_reasons:
        notes.append("Essential channels missing or insufficient; analysis limited to confirmed data.")
    if invalid_reasons:
        notes.append("Flat-zero junk oil/fuel pressure channels marked invalid and excluded from hard-stop logic.")
    if throttle_diag["mismatch_detected"]:
        notes.append("Throttle desired/actual/commanded mismatch detected.")
    if map_boost_conflict["conflict"]:
        notes.append("MAP/boost conflict detected; labeled uncertain and edits should be limited.")
    if not wideband_trusted:
        notes.append("WOT fueling corrections should be limited because actual wideband trust is not confirmed.")

    trust_buckets = finalize_trust_buckets(trust_buckets, invalid_reasons, uncertain, suspect)

    summary = {
        "external_wideband_trusted": bool(wideband_trusted),
        "external_wideband_reason": wideband_reason,
        "idle_detected": bool(operating_mode.get("idle_detected", False)),
        "wot_detected": bool(operating_mode.get("wot_detected", False)),
        "boost_present": bool(operating_mode.get("boost_present", False)),
        "max_map_kpa": operating_mode.get("max_map_kpa"),
        "log_duration_sec": calc_log_duration(num),
        "kr_event_count": len(kr_events),
    }

    result = {
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
        "wideband_diagnostics": wb_diag,
        "throttle_diagnostics": throttle_diag,
        "map_boost_conflict": map_boost_conflict,
        "trim_summary": trim_summary,
        "kr_events": kr_events,
        "fueling_guidance": fueling_guidance,
        "notes": notes,
        "platform_guess": platform_details["platform_guess"],
        "platform_detection": platform_details,
        "analysis_scope": analysis_scope,
        "readable_data": {
            "confirmed": sorted(trust_buckets.get("confirmed_channels", [])),
            "missing_or_unreliable": sorted(
                set(trust_buckets.get("missing_channels", []))
                | set(trust_buckets.get("invalid_channels", []))
                | set(trust_buckets.get("uncertain_channels", []))
            ),
        },
        "conclusion_safety": {
            "safe_conclusions": [
                "Idle behavior can be reviewed safely." if analysis_scope["idle_review"] else None,
                "Cruise/load trend review is supported." if analysis_scope["cruise_review"] else None,
                "Knock trend review is supported." if analysis_scope["knock_review"] else None,
            ],
            "unsupported_conclusions": [
                "Numeric WOT fueling changes are not supported by this log." if not analysis_scope["wot_fueling_review"] else None,
                "Transmission correction conclusions are limited." if not analysis_scope["transmission_review"] else None,
            ],
        },
    }
    result["conclusion_safety"]["safe_conclusions"] = [x for x in result["conclusion_safety"]["safe_conclusions"] if x]
    result["conclusion_safety"]["unsupported_conclusions"] = [x for x in result["conclusion_safety"]["unsupported_conclusions"] if x]

    return result


def validate_dataframe(
    df: pd.DataFrame, filename: str, mime_type: Optional[str], meta: Dict[str, Any], platform_hint: Optional[str] = None
) -> Dict[str, Any]:
    matched_raw_columns, trust_buckets = map_columns(df)
    num, invalid_reasons, pressure_unit_modes = build_numeric_frame(df, matched_raw_columns)

    hard_stop_reasons: List[str] = []
    for key in ESSENTIAL_ANALYSIS_KEYS:
        if key not in matched_raw_columns:
            hard_stop_reasons.append(f"missing_{key}")
        elif key in num and num[key].dropna().size < 5:
            hard_stop_reasons.append(f"insufficient_{key}")

    operating_mode = determine_operating_mode(num)
    analysis_scope = build_analysis_scope(matched_raw_columns, operating_mode)
    platform_details = detect_platform_guess(list(df.columns), platform_hint=platform_hint)
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
        "platform_guess": platform_details["platform_guess"],
        "platform_detection": platform_details,
        "analysis_scope": analysis_scope,
        "capability_flags": analysis_scope,
    }
MOPAR_PLATFORM_MARKERS = {
    "Engine RPM",
    "Engine Speed",
    "Throttle Angle",
    "Ignition Advance",
    "Spark Retard",
    "Commanded Lambda",
    "Driver Desired Torque",
    "Current Gear",
}


def detect_platform_guess(columns: List[str], platform_hint: Optional[str] = None) -> Dict[str, Any]:
    normalized = {str(c).strip().lower() for c in columns}
    marker_hits = [m for m in MOPAR_PLATFORM_MARKERS if m.lower() in normalized]
    score = len(marker_hits)
    hint = (platform_hint or "").strip().lower()
    if hint == "mopar":
        score += 1
    guess = "mopar" if score >= 3 else None
    return {"platform_guess": guess, "mopar_score": score, "mopar_hits": marker_hits}


def build_analysis_scope(matched: Dict[str, str], operating_mode: Dict[str, Any]) -> Dict[str, bool]:
    has_rpm = "RPM" in matched
    has_map = "MAP_kPa" in matched
    has_tps = "TPS_pct" in matched or "Pedal_pct" in matched
    has_gear = "Gear" in matched or "InputSpeed_rpm" in matched or "OutputSpeed_rpm" in matched
    has_kr = "KR_deg" in matched or "TotalKR_deg" in matched
    has_wot_fuel = ("EQ_Cmd" in matched) and ("WB_AFR" in matched or "WB_Lambda" in matched)
    return {
        "basic_engine_review": has_rpm and has_map,
        "idle_review": has_rpm and has_tps and bool(operating_mode.get("idle_detected")),
        "cruise_review": has_rpm and has_tps,
        "shift_review": has_gear and has_tps,
        "knock_review": has_kr and has_rpm and has_map,
        "wot_fueling_review": has_wot_fuel and bool(operating_mode.get("wot_detected")),
        "transmission_review": has_gear,
    }
