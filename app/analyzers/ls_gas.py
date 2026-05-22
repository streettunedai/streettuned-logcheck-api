from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.core.parser import calc_log_duration, choose_best_column, safe_float, series_numeric
from app.core.trust import finalize_trust_buckets

CHANNEL_CONFIDENCE = {"CONFIRMED", "LIKELY", "SUSPECT", "MISSING"}

MAF_EXACT_ALIASES = {
    "maf hz", "maf frequency", "maf sensor frequency", "mass airflow sensor",
    "mass air flow sensor", "mass airflow", "mass air flow", "maf",
}
MAF_REJECT_TOKENS = {
    "dynamic airflow", "cylinder airmass", "volumetric efficiency airflow", "ve airflow",
    "air mass", "delivered airflow", "airflow estimate",
}
WIDEBAND_STRONG_ALIASES = {
    "wideband", "wideband afr", "wideband lambda", "wb afr", "wb lambda", "afr wideband",
    "lambda wideband", "air fuel ratio", "afr through eio", "lambda through eio",
}
WIDEBAND_BRAND_ALIASES = {"innovate", "aem", "plx", "ngk", "lc 1", "lm 1"}
WIDEBAND_ANALOG_ALIASES = {"mpvi pro input", "pro link input", "analog 1", "analog 2", "eio input 1", "eio input 2"}
WIDEBAND_REJECT_TOKENS = {
    "o2 b1", "o2 b2", "narrowband o2", "short term fuel trim", "long term fuel trim",
    "commanded afr", "commanded lambda", "eq ratio", "commanded eq ratio", "desired afr",
    "fuel target", "catalyst o2", "desired", "target", " error ",
}


def canonicalize_channel_name(name: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in str(name)).split())


def classify_unit(text: str) -> str:
    t = canonicalize_channel_name(text)
    if "hz" in t or "hertz" in t:
        return "hz"
    if "lambda" in t:
        return "lambda"
    if "afr" in t or "air fuel ratio" in t or t == "ratio":
        return "afr"
    if t in {"v", "volt", "volts", "voltage"} or " volt" in f" {t}":
        return "volts"
    if "g s" in t or "gps" in t:
        return "g/s"
    if "lb min" in t:
        return "lb/min"
    if "lb hr" in t or "lb h" in t:
        return "lb/hr"
    return "unknown"


def _is_dynamic(series: pd.Series) -> bool:
    finite = series.dropna()
    if finite.size < 8:
        return False
    return bool(float(finite.std()) > 1e-6 and float(finite.diff().abs().median()) > 0)


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
        "EQ Commanded",
        "Fuel Commanded EQ",
    ],
    "AFR_Cmd": [
        "Air-Fuel Ratio Commanded",
        "Commanded AFR",
        "AFR Commanded",
        "Desired AFR",
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
        "MPVI2.1 -> AEM 30-(03x0,2340,5130)",
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
    "MAF_lb_min": [
        "Airflow",
        "MAF Airflow Rate",
        "MAF Airflow",
        "Mass Airflow",
        "Mass Airflow Rate",
        "MAF",
    ],
    "CylAir_g": [
        "Cylinder Airmass",
        "Cylinder Air Mass",
        "Airmass",
        "Airmass per Cylinder",
    ],
    "TransTemp_C": [
        "Trans Temp",
        "Transmission Fluid Temp",
        "Transmission Temperature",
    ],
    "System_Volts": [
        "Voltage",
        "System Voltage",
        "Battery Voltage",
    ],
    "OilPressure_psi": [
        "Oil Pressure (PSI)",
        "Engine Oil Pressure (PSI)",
    ],
    "FuelPressure_psi": [
        "Fuel Pressure (PSI)",
        "Fuel PSI",
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
        if canonical in {"MAF_Hz", "WB_AFR", "WB_Lambda"}:
            continue
        match = choose_best_column(list(df.columns), aliases)
        if match:
            matched_raw_columns[canonical] = match
            trust_buckets["confirmed_channels"].append(canonical)
        else:
            trust_buckets["missing_channels"].append(canonical)

    maf_result = detect_maf_frequency(df)
    if maf_result["status"] in {"CONFIRMED", "LIKELY"} and maf_result["matched_channel"]:
        matched_raw_columns["MAF_Hz"] = maf_result["matched_channel"]
        trust_buckets["confirmed_channels"].append("MAF_Hz" if maf_result["status"] == "CONFIRMED" else "MAF_Hz")
    else:
        trust_buckets["missing_channels"].append("MAF_Hz" if maf_result["status"] == "MISSING" else "MAF_Hz")
        if maf_result["status"] == "SUSPECT":
            trust_buckets["suspect_channels"].append("MAF_Hz")
        else:
            trust_buckets["uncertain_channels"].append("MAF_Hz")

    wb_result = detect_wideband(df)
    if wb_result["status"] in {"CONFIRMED", "LIKELY"} and wb_result["matched_channel"]:
        matched_raw_columns[wb_result["target"]] = wb_result["matched_channel"]
        trust_buckets["confirmed_channels"].append(wb_result["target"] if wb_result["status"] == "CONFIRMED" else wb_result["target"])
    else:
        trust_buckets["missing_channels"].extend(["WB_AFR", "WB_Lambda"])
        if wb_result["status"] == "SUSPECT":
            trust_buckets["suspect_channels"].append("WB_AFR")

    return matched_raw_columns, trust_buckets


def detect_maf_frequency(df: pd.DataFrame) -> Dict[str, Any]:
    best = {"score": -1, "status": "MISSING", "matched_channel": None, "reason": "no acceptable MAF frequency channel"}
    for col in df.columns:
        c = canonicalize_channel_name(col)
        if any(tok in c for tok in MAF_REJECT_TOKENS):
            continue
        unit = classify_unit(col)
        vals = series_numeric(df[col])
        dynamic = _is_dynamic(vals)
        exact = c in MAF_EXACT_ALIASES
        strong = exact or "mass airflow" in c or "mass air flow" in c or ("maf" in c and unit == "hz")
        if not strong:
            continue
        if unit in {"g/s", "lb/min", "lb/hr"}:
            status, score, reason = "SUSPECT", 20, "name matches MAF alias but airflow-mass unit is not MAF frequency"
        elif unit == "hz" and dynamic:
            status, score, reason = "CONFIRMED", 100 + (10 if exact else 0), "MAF alias with Hz unit and dynamic numeric values"
        elif unit == "unknown" and dynamic:
            status, score, reason = "LIKELY", 70, "MAF alias with missing unit but frequency-like dynamic values"
        else:
            status, score, reason = "SUSPECT", 30, "MAF-like name but unit/behavior is conflicting or flatlined"
        if score > best["score"]:
            best = {"score": score, "status": status, "matched_channel": col, "reason": reason}
    return best


def detect_wideband(df: pd.DataFrame) -> Dict[str, Any]:
    best = {"score": -1, "status": "MISSING", "matched_channel": None, "target": "WB_AFR", "reason": "no acceptable wideband channel"}
    for col in df.columns:
        c = canonicalize_channel_name(col)
        if any(tok in f" {c} " for tok in WIDEBAND_REJECT_TOKENS):
            continue
        unit = classify_unit(col)
        vals = series_numeric(df[col])
        dynamic = _is_dynamic(vals)
        strong = c in WIDEBAND_STRONG_ALIASES or any(a in c for a in WIDEBAND_STRONG_ALIASES)
        brand = c in WIDEBAND_BRAND_ALIASES or any(a in c for a in WIDEBAND_BRAND_ALIASES)
        analog = c in WIDEBAND_ANALOG_ALIASES or any(a in c for a in WIDEBAND_ANALOG_ALIASES)
        if unit == "volts" and analog:
            status, score, reason = "SUSPECT", 25, "analog input is voltage-only with no AFR/lambda transform proof"
        elif (strong or analog) and unit in {"afr", "lambda"} and dynamic:
            status, score, reason = "CONFIRMED", 120 if strong else 110, "wideband/analog alias with AFR/lambda unit and plausible dynamic values"
        elif brand and unit in {"afr", "lambda"} and dynamic:
            status, score, reason = "LIKELY", 90, "brand-like alias supports wideband but requires stronger transform/name proof"
        elif strong and unit == "unknown" and dynamic:
            status, score, reason = "LIKELY", 80, "strong wideband alias with plausible values but unit missing"
        elif strong or analog or brand:
            status, score, reason = "SUSPECT", 35, "wideband-like name exists but behavior/unit is insufficient"
        else:
            continue
        target = "WB_Lambda" if unit == "lambda" else "WB_AFR"
        if score > best["score"]:
            best = {"score": score, "status": status, "matched_channel": col, "target": target, "reason": reason}
    return best


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

    for ch in ["RPM", "MAP_kPa", "TPS_pct", "Spark_deg", "IAT_C", "MAF_lb_min", "CylAir_g"]:
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


def build_channel_details(num: pd.DataFrame, matched: Dict[str, str], trust_buckets: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    details: List[Dict[str, Any]] = []
    confirmed = set(trust_buckets.get("confirmed_channels", []))
    uncertain = set(trust_buckets.get("uncertain_channels", []))
    suspect = set(trust_buckets.get("suspect_channels", []))
    missing = set(trust_buckets.get("missing_channels", []))
    for canonical in sorted(set(list(matched.keys()) + list(num.columns))):
        if canonical in num and num[canonical].dropna().size:
            s = num[canonical].dropna()
            details.append(
                {
                    "channel_name": canonical,
                    "unit": "kPa" if canonical.endswith("_kPa") else ("deg" if canonical.endswith("_deg") else None),
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "confidence": "CONFIRMED" if canonical in confirmed else ("SUSPECT" if canonical in suspect else ("LIKELY" if canonical in uncertain else "CONFIRMED")),
                    "reason": "matched_and_numeric",
                }
            )
        elif canonical in missing:
            details.append({"channel_name": canonical, "confidence": "MISSING", "reason": "not_found_or_not_usable"})
    return details


def build_segment_summary(num: pd.DataFrame) -> Dict[str, Any]:
    segments: List[str] = []
    if {"RPM", "TPS_pct"}.issubset(set(num.columns)):
        rpm = num["RPM"]
        tps = num["TPS_pct"]
        if ((rpm < IDLE_RPM_MAX) & (tps <= 10)).fillna(False).any():
            segments.append("idle_low_load")
        if ((tps >= 15) & (tps < 55)).fillna(False).any():
            segments.append("cruise_part_throttle")
        if (tps >= 75).fillna(False).any():
            segments.append("high_load_near_wot_pull")
    max_map = float(num["MAP_kPa"].dropna().max()) if "MAP_kPa" in num and num["MAP_kPa"].dropna().size else None
    return {
        "segments_found": segments,
        "max_map_kpa": max_map,
        "boost_review_supported": bool(max_map is not None and max_map > 105.0),
        "likely_na_from_map": bool(max_map is not None and max_map <= 105.0),
    }


def build_safety_edit_recommendation(kr_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    high_load_events = 0
    peak = 0.0
    for ev in kr_events:
        kr = float(ev.get("peak_kr_deg") or 0.0)
        tps_max = (((ev.get("TPS_pct") or {}).get("max")) if isinstance(ev.get("TPS_pct"), dict) else None) or 0.0
        map_max = (((ev.get("MAP_kPa") or {}).get("max")) if isinstance(ev.get("MAP_kPa"), dict) else None) or 0.0
        if kr > 3.0 and tps_max >= 70 and map_max >= 90:
            high_load_events += 1
            peak = max(peak, kr)
    if high_load_events >= 2:
        return {
            "kr_status": "HARD_STOP_PERFORMANCE_TUNING",
            "reason": f"repeated_high_load_kr_over_3deg_peak_{peak:.2f}",
            "safety_only_spark_edit": {"spark_delta_deg": -2.0, "rpm_range": [4800, 6400], "cyl_airmass_g_per_cyl_range": [0.70, 0.82]},
        }
    return {"kr_status": "NO_HARD_STOP_KR_PATTERN"}


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




def build_wideband_recovery_steps(
    wideband_trusted: bool,
    wideband_reason: str,
    invalid_reasons: Dict[str, str],
    wb_diag: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if wideband_trusted and "FuelPressure_psi" not in invalid_reasons:
        return None

    wb_channel = wb_diag.get("wideband_channel_used") or "WB channel"
    wb_state = wb_diag.get("wideband_interpretation") or "unknown"

    wb_name = wb_channel

    changes_required: List[str] = [
        "Open VCM Scanner and verify the wideband channel is in the Channels list, not only in a gauge/chart.",
        f"Confirm {wb_name} moves live with engine running and does not stay frozen.",
        "Validate analog/pro-link source assignment and transform math (lambda vs AFR conversion and voltage scaling).",
        "Save the scanner channel config before logging.",
        "Record a short native .hpl log first (idle, cruise, and one moderate pull) before any WOT hit.",
        "Re-open the saved .hpl and confirm the wideband channel exists in playback.",
    ]

    verify_next: List[str] = [
        "Channels list showing the wideband channel is selected for logging.",
        "Wideband transform/math parameters screen.",
        "MPVI Pro Link or analog input assignment screen.",
        "Saved log playback screenshot with wideband visible.",
    ]

    if "FuelPressure_psi" in invalid_reasons:
        changes_required.append("Fix fuel pressure source/transform separately; do not use flat-zero fuel pressure data for decisions.")

    summary = (
        f"Wideband visibility detected but trust is not confirmed (reason: {wideband_reason}; interpretation: {wb_state}). "
        "This is usually a scanner channel, export, or transform/input assignment issue rather than a tune issue."
    )

    return {
        "data_summary": summary,
        "root_cause": "scanner_config_or_channel_assignment",
        "changes_required": changes_required,
        "verify_next": verify_next,
    }
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


def _range_for(num: pd.DataFrame, channel: str) -> Optional[Dict[str, float]]:
    if channel not in num or num[channel].dropna().empty:
        return None
    s = num[channel].dropna()
    return {"min": float(s.min()), "max": float(s.max())}


def build_detailed_report_payload(
    num: pd.DataFrame,
    summary: Dict[str, Any],
    trust_buckets: Dict[str, List[str]],
    kr_events: List[Dict[str, Any]],
    per_bank_trim_summary: Dict[str, Any],
) -> Dict[str, Any]:
    confirmed_ranges = {
        "RPM": _range_for(num, "RPM"),
        "TPS_pct": _range_for(num, "TPS_pct"),
        "MAP_kPa": _range_for(num, "MAP_kPa"),
        "MAF_lb_min": _range_for(num, "MAF_lb_min"),
        "MAF_Hz": _range_for(num, "MAF_Hz"),
        "CylAir_g": _range_for(num, "CylAir_g"),
        "AFR_Cmd": _range_for(num, "AFR_Cmd"),
        "EQ_Cmd": _range_for(num, "EQ_Cmd"),
        "KR_deg": _range_for(num, "KR_deg") or _range_for(num, "TotalKR_deg"),
        "Spark_deg": _range_for(num, "Spark_deg"),
    }
    missing = sorted(
        set(trust_buckets.get("missing_channels", []))
        | set(trust_buckets.get("invalid_channels", []))
        | set(trust_buckets.get("uncertain_channels", []))
    )
    return {
        "report_version": "lslt_detailed_v1",
        "map_interpretation": {
            "max_map_kpa": summary.get("max_map_kpa"),
            "likely_na_from_map": bool((summary.get("max_map_kpa") or 0) <= 105),
            "boost_supported_by_log": bool((summary.get("max_map_kpa") or 0) > 105),
        },
        "commanded_fueling_detected": bool(summary.get("has_commanded_fueling_channel")),
        "confirmed_channel_ranges": confirmed_ranges,
        "missing_or_unreliable_channels": missing,
        "per_bank_trim_summary": per_bank_trim_summary,
        "kr_event_windows": kr_events,
        "recommended_safe_spark_action": build_safety_edit_recommendation(kr_events),
    }


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
    per_bank_trim_summary: Dict[str, Any] = {}
    for b in ("1", "2"):
        st = num[f"STFT{b}_pct"].dropna() if f"STFT{b}_pct" in num else pd.Series(dtype=float)
        lt = num[f"LTFT{b}_pct"].dropna() if f"LTFT{b}_pct" in num else pd.Series(dtype=float)
        if st.size or lt.size:
            per_bank_trim_summary[f"bank_{b}"] = {
                "stft_avg_pct": float(st.mean()) if st.size else None,
                "ltft_avg_pct": float(lt.mean()) if lt.size else None,
                "ltft_locked_zero": bool(lt.size and float((lt.abs() < 1e-9).mean()) > 0.95),
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
        "has_commanded_fueling_channel": bool(
            ("EQ_Cmd" in matched_raw_columns and num.get("EQ_Cmd", pd.Series(dtype=float)).dropna().size > 0)
            or ("AFR_Cmd" in matched_raw_columns and num.get("AFR_Cmd", pd.Series(dtype=float)).dropna().size > 0)
        ),
    }

    recovery = build_wideband_recovery_steps(
        wideband_trusted,
        wideband_reason,
        invalid_reasons,
        wb_diag,
    )

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
        "per_bank_trim_summary": per_bank_trim_summary,
        "kr_events": kr_events,
        "kr_safety_recommendation": build_safety_edit_recommendation(kr_events),
        "channel_details": build_channel_details(num, matched_raw_columns, trust_buckets),
        "segment_summary": build_segment_summary(num),
        "fueling_guidance": fueling_guidance,
        "notes": notes,
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
        "detailed_report": build_detailed_report_payload(
            num=num,
            summary=summary,
            trust_buckets=trust_buckets,
            kr_events=kr_events,
            per_bank_trim_summary=per_bank_trim_summary,
        ),
        "report_sections": build_report_sections(
            meta=meta,
            summary=summary,
            trust_buckets=trust_buckets,
            fueling_guidance=fueling_guidance,
            kr_events=kr_events,
        ),
    }
    result["conclusion_safety"]["safe_conclusions"] = [x for x in result["conclusion_safety"]["safe_conclusions"] if x]
    result["conclusion_safety"]["unsupported_conclusions"] = [x for x in result["conclusion_safety"]["unsupported_conclusions"] if x]
    if recovery:
        result["recovery_plan"] = recovery

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
        "analysis_scope": analysis_scope,
        "capability_flags": analysis_scope,
    }

def build_report_sections(
    meta: Dict[str, Any],
    summary: Dict[str, Any],
    trust_buckets: Dict[str, List[str]],
    fueling_guidance: Dict[str, Any],
    kr_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "what_i_received": {
            "filename": meta.get("filename"),
            "row_count": meta.get("row_count"),
            "column_count": meta.get("column_count"),
            "log_duration_sec": summary.get("log_duration_sec"),
        },
        "what_i_see": {
            "confirmed_channels": sorted(trust_buckets.get("confirmed_channels", [])),
            "missing_or_unreliable": sorted(
                set(trust_buckets.get("missing_channels", []))
                | set(trust_buckets.get("invalid_channels", []))
                | set(trust_buckets.get("uncertain_channels", []))
            ),
            "max_map_kpa": summary.get("max_map_kpa"),
            "kr_event_count": len(kr_events),
            "commanded_fueling_detected": bool(summary.get("has_commanded_fueling_channel")),
        },
        "edits": {
            "wot_fueling_allowed": bool(fueling_guidance.get("can_make_wot_fueling_suggestions")),
            "wot_fueling_reason": fueling_guidance.get("reason_wot_fueling_limited"),
            "safe_spark_action": build_safety_edit_recommendation(kr_events),
        },
        "do_not_touch": [
            "Do not make WOT fuel edits without trusted wideband actual.",
            "Do not add timing while repeated KR is present.",
            "Do not tune VE/MAF from suspect or conflicting MAP/boost data.",
        ],
        "next_log_plan": [
            "Log wideband AFR/lambda as a transformed channel.",
            "Log commanded AFR/EQ.",
            "Log RPM, TPS, MAP, KR, spark, MAF Hz, MAF airflow, and cylinder airmass.",
            "Re-log after safety spark change before any power optimization.",
        ],
    }


def build_analysis_scope(matched: Dict[str, str], operating_mode: Dict[str, Any]) -> Dict[str, bool]:
    has_rpm = "RPM" in matched
    has_map = "MAP_kPa" in matched
    has_tps = "TPS_pct" in matched or "Pedal_pct" in matched
    has_gear = "Gear" in matched or "InputSpeed_rpm" in matched or "OutputSpeed_rpm" in matched
    has_kr = "KR_deg" in matched or "TotalKR_deg" in matched
    has_commanded_fuel = ("EQ_Cmd" in matched) or ("AFR_Cmd" in matched)
    has_wot_fuel = has_commanded_fuel and ("WB_AFR" in matched or "WB_Lambda" in matched)
    return {
        "basic_engine_review": has_rpm and has_map,
        "idle_review": has_rpm and has_tps and bool(operating_mode.get("idle_detected")),
        "cruise_review": has_rpm and has_tps,
        "shift_review": has_gear and has_tps,
        "knock_review": has_kr and has_rpm and has_map,
        "wot_fueling_review": has_wot_fuel and bool(operating_mode.get("wot_detected")),
        "transmission_review": has_gear,
    }
