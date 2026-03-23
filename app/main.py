from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="StreetTunedAI Log Analyzer", version="1.0.1")


NUMERIC_FFILL_SEC = 0.20
STATUS_FFILL_SEC = 0.50

RAW_ALIAS_GROUPS: Dict[str, List[str]] = {
    "Time_sec": [
        "Time",
        "Time (s)",
        "Time [s]",
        "Offset",
        "Elapsed Time",
        "Timestamp",
        "Time Stamp",
    ],
    "RPM": [
        "Engine RPM",
        "Engine RPM (SAE)",
    ],
    "ECT": [
        "Engine Coolant Temp",
        "Engine Coolant Temp (SAE)",
        "Coolant Temp",
    ],
    "IAT": [
        "Intake Air Temp",
        "Intake Air Temp (SAE)",
        "Intake Air Temperature",
    ],
    "MAF_gps": [
        "Mass Airflow",
        "Mass Airflow (SAE)",
        "MAF Airflow Rate",
    ],
    "InjectorPW_B1_ms": [
        "Injector Pulse Width Avg. Bank 1",
        "Injector Pulse Width Bank 1",
        "Injector PW Avg Bank 1",
    ],
    "InjectorPW_B2_ms": [
        "Injector Pulse Width Avg. Bank 2",
        "Injector Pulse Width Bank 2",
        "Injector PW Avg Bank 2",
    ],
    "STFT_B1": [
        "Short Term Fuel Trim Bank 1",
        "Short Term Fuel Trim Bank 1 (SAE)",
    ],
    "STFT_B2": [
        "Short Term Fuel Trim Bank 2",
        "Short Term Fuel Trim Bank 2 (SAE)",
    ],
    "LTFT_B1": [
        "Long Term Fuel Trim Bank 1",
        "Long Term Fuel Trim Bank 1 (SAE)",
    ],
    "LTFT_B2": [
        "Long Term Fuel Trim Bank 2",
        "Long Term Fuel Trim Bank 2 (SAE)",
    ],
    "ThrottleDesired_pct": [
        "Throttle Desired Position",
    ],
    "ThrottleCommanded_pct": [
        "Commanded Throttle Actuator (SAE)",
        "Commanded Throttle Actuator",
    ],
    "ThrottleActual_pct": [
        "Throttle Position",
        "Throttle Position (SAE)",
        "Throttle Position Sensor",
        "Throttle Position Sensor 2",
    ],
    "MAP_kPa": [
        "Intake Manifold Absolute Pressure",
        "Intake Manifold Absolute Pressure (SAE)",
        "MAP",
        "MAP Sensor",
    ],
    "BoostRelatedPressure_kPa": [
        "Supercharger Inlet Pressure",
    ],
    "BoostRelatedVacuum_kPa": [
        "Supercharger Inlet Vacuum",
    ],
    "AFR_Act_External": [
        "MPVI2.1 -> AEM-4110",
        "AEM-4110",
        "MPVI2.1 AEM-4110",
    ],
    "FuelRailPressure": [
        "Fuel Rail Pressure",
        "Fuel Rail Pressure (SAE)",
    ],
    "FuelRailPressureRel": [
        "Fuel Rail Pressure (Relative)",
        "Fuel Rail Pressure (Relative) (SAE)",
    ],
    "Ethanol_pct": [
        "Ethanol Fuel %",
        "Ethanol Fuel % (SAE)",
    ],
    "PE_Status": [
        "Power Enrichment",
    ],
    "ClosedLoopActive": [
        "Closed Loop Active",
    ],
    "Battery_V": [
        "Control Module Voltage",
        "Control Module Voltage (SAE)",
    ],
    "Baro_kPa": [
        "Barometric Pressure",
        "Barometric Pressure (SAE)",
    ],
    "Spark_deg": [
        "Timing Advance",
        "Timing Advance (SAE)",
    ],
    "KR_deg": [
        "Knock Retard",
        "Total Knock Retard",
    ],
    "OilPressure": [
        "Oil Pressure",
        "Engine Oil Pressure",
        "Oil Pressure (SAE)",
    ],
    "FuelPressure": [
        "Fuel Pressure",
        "Fuel Pressure (SAE)",
        "Fuel Pump Pressure",
    ],
    "EqCmd": [
        "Equivalence Ratio Commanded",
        "Equivalence Ratio Commanded (SAE)",
    ],
    "AfrCmd": [
        "Air-Fuel Ratio Commanded",
        "Commanded AFR",
    ],
    "FuelSys1_Status": [
        "Fuel System #1 Status",
        "Fuel System #1 Status (SAE)",
    ],
}

TEXT_TRUE = {"on", "true", "yes", "enabled", "active", "closed loop", "power enrichment", "1"}
TEXT_FALSE = {"off", "false", "no", "disabled", "inactive", "open loop", "0"}


def clean_name(x: Any) -> str:
    s = "" if x is None else str(x)
    s = s.replace("\ufeff", "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def slug(x: str) -> str:
    s = clean_name(x).lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def try_float_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype(str).str.strip()
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace("%", "", regex=False)
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "null": np.nan})
    return pd.to_numeric(s, errors="coerce")


def detect_header_and_data_row(lines: List[str]) -> Tuple[int, int]:
    best_header = 0
    best_score = -1
    best_data = 1

    for i in range(min(len(lines), 60)):
        line = lines[i]
        if not line.strip():
            continue
        parts = [clean_name(p) for p in line.split(",")]
        nonempty = sum(1 for p in parts if p)
        score = 0

        if nonempty >= 4:
            score += nonempty

        joined = " | ".join(parts).lower()
        keywords = [
            "rpm",
            "throttle",
            "pressure",
            "fuel",
            "spark",
            "maf",
            "iat",
            "coolant",
            "knock",
            "equivalence",
            "barometric",
            "ethanol",
            "closed loop",
            "power enrichment",
            "aem",
            "mpvi2.1",
        ]
        score += sum(2 for k in keywords if k in joined)

        if "offset" in joined or "time" in joined:
            score += 4

        candidate_data = i + 1
        for j in range(i + 1, min(len(lines), i + 8)):
            row = [clean_name(p) for p in lines[j].split(",")]
            nums = sum(1 for p in row if re.fullmatch(r"-?\d+(\.\d+)?", p or ""))
            if nums >= max(2, len(row) // 4):
                candidate_data = j
                score += 5
                break

        if score > best_score:
            best_score = score
            best_header = i
            best_data = candidate_data

    return best_header, best_data


def load_csv_with_detected_header(raw: bytes) -> Tuple[pd.DataFrame, int, int]:
    text = raw.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    if not lines:
        raise HTTPException(status_code=400, detail="Empty file")

    header_row, first_data_row = detect_header_and_data_row(lines)
    df = pd.read_csv(
        io.StringIO(text),
        header=header_row,
        skip_blank_lines=False,
        dtype=str,
        engine="python",
        on_bad_lines="skip",
    )
    df.columns = [clean_name(c) for c in df.columns]
    df = df.dropna(how="all").reset_index(drop=True)

    if len(df) > 0:
        header_slugs = [slug(c) for c in df.columns]
        keep_rows = []
        for _, row in df.iterrows():
            vals = [slug(clean_name(v)) for v in row.tolist()]
            keep_rows.append(vals != header_slugs)
        df = df.loc[keep_rows].reset_index(drop=True)

    return df, header_row, first_data_row


def build_alias_lookup() -> Dict[str, List[str]]:
    lookup: Dict[str, List[str]] = {}
    for canon, names in RAW_ALIAS_GROUPS.items():
        lookup[canon] = list(dict.fromkeys(names))
    return lookup


ALIAS_LOOKUP = build_alias_lookup()


def select_matching_columns(columns: List[str], aliases: List[str]) -> List[str]:
    col_slug_map = {c: slug(c) for c in columns}
    alias_slugs = {slug(a) for a in aliases}
    out = []
    for col, col_s in col_slug_map.items():
        if col_s in alias_slugs:
            out.append(col)
    return out


def series_is_flatline_numeric(s: pd.Series, tolerance: float = 1e-9) -> bool:
    x = try_float_series(s).dropna()
    if len(x) < 5:
        return False
    return float(x.max() - x.min()) <= tolerance


def series_plausible_percent(s: pd.Series) -> bool:
    x = try_float_series(s).dropna()
    if len(x) < 5:
        return False
    return (x.between(-2, 105).mean() >= 0.90) and (x.max() - x.min() >= 0.5)


def series_plausible_afr(s: pd.Series, rpm: Optional[pd.Series] = None) -> bool:
    x = try_float_series(s).dropna()
    if len(x) < 5:
        return False
    if series_is_flatline_numeric(x):
        return False
    if x.between(7.0, 22.0).mean() < 0.90:
        return False
    if rpm is not None:
        r = try_float_series(rpm).reindex_like(s)
        running_mask = r.fillna(0) > 500
        if running_mask.any():
            xr = try_float_series(s)[running_mask].dropna()
            if len(xr) >= 5 and xr.between(8.0, 19.5).mean() < 0.90:
                return False
    return True


def series_plausible_map_kpa(s: pd.Series, rpm: Optional[pd.Series], baro: Optional[pd.Series]) -> Tuple[bool, str]:
    x = try_float_series(s)
    valid = x.dropna()
    if len(valid) < 5:
        return False, "insufficient_data"

    if valid.between(5, 315).mean() < 0.90:
        return False, "out_of_range"

    if rpm is not None:
        r = try_float_series(rpm).reindex_like(x).fillna(0)
        running = r > 500
        if running.any():
            xr = x[running].dropna()
            if len(xr) >= 5:
                if xr.between(10, 315).mean() < 0.90:
                    return False, "running_out_of_range"
                if baro is not None:
                    b = try_float_series(baro).reindex_like(x).ffill()
                    if len(b.dropna()) > 0:
                        if (xr < 15).mean() > 0.80:
                            return False, "implausibly_low_while_running"
                        if (xr > 220).mean() > 0.90 and (b.dropna().median() < 120):
                            return False, "implausibly_high_while_running"
    return True, "ok"


def fill_sparse_channels(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if time_col not in df.columns:
        return df

    out = df.copy()
    t = try_float_series(out[time_col])

    if t.isna().all():
        return out

    out[time_col] = t
    for col in out.columns:
        if col == time_col:
            continue

        raw = out[col]
        numeric = try_float_series(raw)
        numeric_valid = int(numeric.notna().sum())
        total_valid = int(raw.notna().sum())
        is_numeric_like = numeric_valid >= max(3, int(total_valid * 0.50)) if total_valid > 0 else False

        if is_numeric_like:
            last_idx = None
            last_val = None
            filled = []
            for idx, val in numeric.items():
                ti = t.loc[idx]
                if pd.notna(val):
                    last_idx = idx
                    last_val = val
                    filled.append(val)
                else:
                    if last_idx is not None and pd.notna(ti) and pd.notna(t.loc[last_idx]):
                        dt = float(ti - t.loc[last_idx])
                        filled.append(last_val if 0 <= dt <= NUMERIC_FFILL_SEC else np.nan)
                    else:
                        filled.append(np.nan)
            out[col] = pd.Series(filled, index=out.index)
        else:
            s = raw.astype(str).replace({"nan": np.nan, "None": np.nan, "": np.nan})
            last_idx = None
            last_val = None
            filled = []
            for idx, val in s.items():
                ti = t.loc[idx]
                if pd.notna(val):
                    last_idx = idx
                    last_val = val
                    filled.append(val)
                else:
                    if last_idx is not None and pd.notna(ti) and pd.notna(t.loc[last_idx]):
                        dt = float(ti - t.loc[last_idx])
                        filled.append(last_val if 0 <= dt <= STATUS_FFILL_SEC else np.nan)
                    else:
                        filled.append(np.nan)
            out[col] = pd.Series(filled, index=out.index)

    return out


def convert_status_to_bool(series: pd.Series) -> pd.Series:
    def _map(v: Any) -> Optional[bool]:
        if pd.isna(v):
            return np.nan
        s = clean_name(v).lower()
        if s in TEXT_TRUE:
            return True
        if s in TEXT_FALSE:
            return False
        try:
            return float(s) != 0.0
        except Exception:
            return np.nan

    return series.map(_map)


def choose_best_percent_column(df: pd.DataFrame, candidates: List[str]) -> Tuple[Optional[str], Optional[pd.Series], Optional[str]]:
    scored: List[Tuple[float, str, pd.Series]] = []
    reasons: Dict[str, str] = {}
    for col in candidates:
        s = try_float_series(df[col])
        valid = s.dropna()
        if len(valid) < 5:
            reasons[col] = "insufficient_data"
            continue
        if not series_plausible_percent(s):
            reasons[col] = "implausible_percent_scale"
            continue
        score = float(valid.notna().sum()) + float(valid.between(0, 100).mean()) * 100.0 - float(series_is_flatline_numeric(valid)) * 50.0
        scored.append((score, col, s))
        reasons[col] = "ok"

    if not scored:
        return None, None, None
    scored.sort(reverse=True, key=lambda x: x[0])
    _, col, series = scored[0]
    return col, series, reasons.get(col, "ok")


def choose_best_numeric_column(df: pd.DataFrame, candidates: List[str], plausibility_fn=None) -> Tuple[Optional[str], Optional[pd.Series], Optional[str]]:
    scored: List[Tuple[float, str, pd.Series]] = []
    reasons: Dict[str, str] = {}
    for col in candidates:
        s = try_float_series(df[col])
        valid = s.dropna()
        if len(valid) < 3:
            reasons[col] = "insufficient_data"
            continue
        if plausibility_fn is not None:
            ok, reason = plausibility_fn(s)
            if not ok:
                reasons[col] = reason
                continue
            reasons[col] = "ok"
        else:
            reasons[col] = "ok"
        score = float(valid.notna().sum()) - float(series_is_flatline_numeric(valid)) * 50.0
        scored.append((score, col, s))

    if not scored:
        return None, None, None
    scored.sort(reverse=True, key=lambda x: x[0])
    _, col, series = scored[0]
    return col, series, reasons.get(col, "ok")


@dataclass
class ChannelSelection:
    canonical: str
    selected_column: Optional[str] = None
    candidates: List[str] = field(default_factory=list)
    reason: str = "missing"
    trust: str = "missing"
    data: Optional[pd.Series] = None


def build_channel_map(df: pd.DataFrame) -> Dict[str, ChannelSelection]:
    cols = list(df.columns)
    selections: Dict[str, ChannelSelection] = {}

    for canonical, aliases in ALIAS_LOOKUP.items():
        candidates = select_matching_columns(cols, aliases)
        selections[canonical] = ChannelSelection(canonical=canonical, candidates=candidates)

    time_candidates = selections["Time_sec"].candidates
    time_col = None
    time_reason = "missing"
    if time_candidates:
        for col in time_candidates:
            s = try_float_series(df[col])
            if s.dropna().shape[0] >= 3:
                time_col = col
                time_reason = "ok"
                break
    selections["Time_sec"].selected_column = time_col
    selections["Time_sec"].reason = time_reason
    selections["Time_sec"].trust = "confirmed" if time_col else "missing"
    selections["Time_sec"].data = try_float_series(df[time_col]) if time_col else None

    def basic_numeric(canon: str):
        if not selections[canon].candidates:
            selections[canon].reason = "missing"
            selections[canon].trust = "missing"
            return
        col, series, reason = choose_best_numeric_column(df, selections[canon].candidates)
        selections[canon].selected_column = col
        selections[canon].data = series
        selections[canon].reason = reason or "missing"
        selections[canon].trust = "confirmed" if col else "missing"

    for canon in [
        "RPM",
        "ECT",
        "IAT",
        "MAF_gps",
        "InjectorPW_B1_ms",
        "InjectorPW_B2_ms",
        "STFT_B1",
        "STFT_B2",
        "LTFT_B1",
        "LTFT_B2",
        "FuelRailPressure",
        "FuelRailPressureRel",
        "Ethanol_pct",
        "Battery_V",
        "Baro_kPa",
        "Spark_deg",
        "KR_deg",
        "BoostRelatedPressure_kPa",
        "BoostRelatedVacuum_kPa",
        "ThrottleDesired_pct",
        "ThrottleCommanded_pct",
        "OilPressure",
        "FuelPressure",
        "EqCmd",
        "AfrCmd",
    ]:
        basic_numeric(canon)

    for canon in ["PE_Status", "ClosedLoopActive", "FuelSys1_Status"]:
        cands = selections[canon].candidates
        if not cands:
            selections[canon].trust = "missing"
            selections[canon].reason = "missing"
            continue
        best = cands[0]
        selections[canon].selected_column = best
        selections[canon].data = df[best]
        selections[canon].reason = "ok"
        selections[canon].trust = "confirmed"

    map_cands = selections["MAP_kPa"].candidates
    rpm_series = selections["RPM"].data
    baro_series = selections["Baro_kPa"].data
    if map_cands:
        best_score = -1e9
        best_col = None
        best_series = None
        best_reason = "missing"
        best_trust = "missing"
        for col in map_cands:
            s = try_float_series(df[col])
            ok, reason = series_plausible_map_kpa(s, rpm_series, baro_series)
            score = float(s.notna().sum())
            if not ok:
                score -= 1000.0
            if score > best_score:
                best_score = score
                best_col = col
                best_series = s
                best_reason = reason
                best_trust = "confirmed" if ok else "suspect"
        selections["MAP_kPa"].selected_column = best_col
        selections["MAP_kPa"].data = best_series
        selections["MAP_kPa"].reason = best_reason
        selections["MAP_kPa"].trust = best_trust
    else:
        selections["MAP_kPa"].reason = "missing"
        selections["MAP_kPa"].trust = "missing"

    afr_cands = selections["AFR_Act_External"].candidates
    if afr_cands:
        best_score = -1e9
        best_col = None
        best_series = None
        best_reason = "missing"
        best_trust = "missing"
        for col in afr_cands:
            s = try_float_series(df[col])
            ok = series_plausible_afr(s, rpm=rpm_series)
            score = float(s.notna().sum())
            if not ok:
                score -= 1000.0
            if score > best_score:
                best_score = score
                best_col = col
                best_series = s
                best_reason = "ok" if ok else "implausible_or_flatline"
                best_trust = "confirmed" if ok else "invalid"
        selections["AFR_Act_External"].selected_column = best_col
        selections["AFR_Act_External"].data = best_series
        selections["AFR_Act_External"].reason = best_reason
        selections["AFR_Act_External"].trust = best_trust
    else:
        selections["AFR_Act_External"].reason = "missing"
        selections["AFR_Act_External"].trust = "missing"

    actual_cands = selections["ThrottleActual_pct"].candidates
    if actual_cands:
        col, series, reason = choose_best_percent_column(df, actual_cands)
        selections["ThrottleActual_pct"].selected_column = col
        selections["ThrottleActual_pct"].data = series
        selections["ThrottleActual_pct"].reason = reason or "missing"
        selections["ThrottleActual_pct"].trust = "confirmed" if col else "suspect"
    else:
        selections["ThrottleActual_pct"].reason = "missing"
        selections["ThrottleActual_pct"].trust = "missing"

    return selections


def avg_bank(series1: Optional[pd.Series], series2: Optional[pd.Series]) -> Optional[pd.Series]:
    if series1 is None and series2 is None:
        return None
    if series1 is None:
        return try_float_series(series2)
    if series2 is None:
        return try_float_series(series1)
    a = try_float_series(series1)
    b = try_float_series(series2)
    return pd.concat([a, b], axis=1).mean(axis=1, skipna=True)


def classify_operating_mode(
    rpm: Optional[pd.Series],
    map_kpa: Optional[pd.Series],
    map_trust: str,
    baro_kpa: Optional[pd.Series],
    boost_pressure: Optional[pd.Series],
    boost_vacuum: Optional[pd.Series],
) -> Tuple[str, Dict[str, Any], List[str]]:
    evidence: Dict[str, Any] = {}
    uncertain_reasons: List[str] = []

    if rpm is None:
        return "uncertain", {"reason": "missing_rpm"}, ["missing_rpm"]

    r = try_float_series(rpm)
    running = r > 500
    if not running.any():
        return "unknown", {"reason": "not_running"}, []

    map_running = try_float_series(map_kpa)[running] if map_kpa is not None else pd.Series(dtype=float)
    baro_running = try_float_series(baro_kpa)[running] if baro_kpa is not None else pd.Series(dtype=float)
    bp_running = try_float_series(boost_pressure)[running] if boost_pressure is not None else pd.Series(dtype=float)
    bv_running = try_float_series(boost_vacuum)[running] if boost_vacuum is not None else pd.Series(dtype=float)

    map_med = float(map_running.median()) if len(map_running.dropna()) else None
    baro_med = float(baro_running.median()) if len(baro_running.dropna()) else None
    bp_med = float(bp_running.median()) if len(bp_running.dropna()) else None
    bv_med = float(bv_running.median()) if len(bv_running.dropna()) else None

    evidence["map_median_kpa"] = map_med
    evidence["baro_median_kpa"] = baro_med
    evidence["sc_inlet_pressure_median"] = bp_med
    evidence["sc_inlet_vacuum_median"] = bv_med
    evidence["map_trust"] = map_trust

    if map_trust != "confirmed":
        uncertain_reasons.append("map_suspect")

    delta = None
    if map_med is not None and baro_med is not None:
        delta = map_med - baro_med
        evidence["map_minus_baro_kpa"] = delta

    boost_signal_present = False
    if bp_med is not None and abs(bp_med) > 1.0:
        boost_signal_present = True
    if bv_med is not None and abs(bv_med) > 1.0:
        boost_signal_present = True

    if delta is None and not boost_signal_present:
        return "uncertain", evidence, ["insufficient_boost_evidence"]

    if delta is not None:
        if delta > 5:
            base_mode = "boost"
        elif delta < -5:
            base_mode = "vacuum_or_na"
        else:
            base_mode = "near_baro"
    else:
        base_mode = "unknown"

    if boost_signal_present and base_mode in {"vacuum_or_na", "near_baro"}:
        uncertain_reasons.append("map_conflicts_with_boost_related_channels")

    if uncertain_reasons:
        return "uncertain", evidence, uncertain_reasons

    if base_mode == "boost":
        return "boost", evidence, []
    if base_mode in {"vacuum_or_na", "near_baro"}:
        return "na", evidence, []
    return "uncertain", evidence, ["unable_to_classify"]


def detect_invalid_flat_zero_pressure(series: Optional[pd.Series], rpm: Optional[pd.Series]) -> bool:
    if series is None or rpm is None:
        return False
    s = try_float_series(series)
    r = try_float_series(rpm)
    running = r > 500
    if not running.any():
        return False
    x = s[running].dropna()
    if len(x) < 5:
        return False
    return float((x == 0).mean()) >= 0.95


def detect_idle_segments(time_s: Optional[pd.Series], rpm: Optional[pd.Series], throttle_actual: Optional[pd.Series]) -> List[Dict[str, Any]]:
    if time_s is None or rpm is None:
        return []
    t = try_float_series(time_s)
    r = try_float_series(rpm)
    thr = try_float_series(throttle_actual) if throttle_actual is not None else pd.Series(index=t.index, dtype=float)

    mask = (r.between(500, 1100)) & ((thr.isna()) | (thr <= 10))
    segments = []
    start_idx = None
    for idx, flag in mask.fillna(False).items():
        if flag and start_idx is None:
            start_idx = idx
        elif not flag and start_idx is not None:
            end_idx = idx - 1 if isinstance(idx, int) else idx
            t0 = t.loc[start_idx]
            t1 = t.loc[end_idx]
            if pd.notna(t0) and pd.notna(t1) and (t1 - t0) >= 1.0:
                rr = r.loc[start_idx:end_idx]
                segments.append({
                    "start_sec": float(t0),
                    "end_sec": float(t1),
                    "duration_sec": float(t1 - t0),
                    "avg_rpm": float(rr.mean(skipna=True)),
                })
            start_idx = None
    if start_idx is not None:
        end_idx = mask.index[-1]
        t0 = t.loc[start_idx]
        t1 = t.loc[end_idx]
        if pd.notna(t0) and pd.notna(t1) and (t1 - t0) >= 1.0:
            rr = r.loc[start_idx:end_idx]
            segments.append({
                "start_sec": float(t0),
                "end_sec": float(t1),
                "duration_sec": float(t1 - t0),
                "avg_rpm": float(rr.mean(skipna=True)),
            })
    return segments


def extract_kr_events(time_s: Optional[pd.Series], rpm: Optional[pd.Series], kr: Optional[pd.Series], spark: Optional[pd.Series]) -> List[Dict[str, Any]]:
    if time_s is None or kr is None:
        return []
    t = try_float_series(time_s)
    k = try_float_series(kr)
    r = try_float_series(rpm) if rpm is not None else pd.Series(index=t.index, dtype=float)
    sp = try_float_series(spark) if spark is not None else pd.Series(index=t.index, dtype=float)

    mask = k.fillna(0) > 0.0
    events = []
    active = False
    start = None

    for idx, flag in mask.items():
        if flag and not active:
            active = True
            start = idx
        elif not flag and active:
            end = idx - 1 if isinstance(idx, int) else idx
            events.append({
                "start_sec": float(t.loc[start]) if pd.notna(t.loc[start]) else None,
                "end_sec": float(t.loc[end]) if pd.notna(t.loc[end]) else None,
                "peak_kr_deg": float(k.loc[start:end].max(skipna=True)),
                "avg_rpm": float(r.loc[start:end].mean(skipna=True)) if len(r.loc[start:end].dropna()) else None,
                "avg_spark_deg": float(sp.loc[start:end].mean(skipna=True)) if len(sp.loc[start:end].dropna()) else None,
            })
            active = False
            start = None
    if active and start is not None:
        end = mask.index[-1]
        events.append({
            "start_sec": float(t.loc[start]) if pd.notna(t.loc[start]) else None,
            "end_sec": float(t.loc[end]) if pd.notna(t.loc[end]) else None,
            "peak_kr_deg": float(k.loc[start:end].max(skipna=True)),
            "avg_rpm": float(r.loc[start:end].mean(skipna=True)) if len(r.loc[start:end].dropna()) else None,
            "avg_spark_deg": float(sp.loc[start:end].mean(skipna=True)) if len(sp.loc[start:end].dropna()) else None,
        })
    return events


def compute_trim_summary(stft_b1: Optional[pd.Series], stft_b2: Optional[pd.Series], ltft_b1: Optional[pd.Series], ltft_b2: Optional[pd.Series]) -> Dict[str, Any]:
    stft_avg = avg_bank(stft_b1, stft_b2)
    ltft_avg = avg_bank(ltft_b1, ltft_b2)
    total = None
    if stft_avg is not None or ltft_avg is not None:
        a = try_float_series(stft_avg) if stft_avg is not None else pd.Series(dtype=float)
        b = try_float_series(ltft_avg) if ltft_avg is not None else pd.Series(dtype=float)
        total = pd.concat([a, b], axis=1).sum(axis=1, min_count=1)

    return {
        "stft_avg_mean": float(stft_avg.mean(skipna=True)) if stft_avg is not None and len(stft_avg.dropna()) else None,
        "ltft_avg_mean": float(ltft_avg.mean(skipna=True)) if ltft_avg is not None and len(ltft_avg.dropna()) else None,
        "combined_trim_mean": float(total.mean(skipna=True)) if total is not None and len(total.dropna()) else None,
    }


def select_analysis_channels(df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, ChannelSelection]]:
    selections = build_channel_map(df)
    time_col = selections["Time_sec"].selected_column
    if time_col:
        df_filled = fill_sparse_channels(df, time_col)
        selections = build_channel_map(df_filled)
    else:
        df_filled = df
    return {"df": df_filled}, selections


def build_trust_buckets(selections: Dict[str, ChannelSelection]) -> Dict[str, List[str]]:
    confirmed, suspect, missing, invalid, uncertain = [], [], [], [], []

    for canon, sel in selections.items():
        entry = f"{canon}:{sel.selected_column}" if sel.selected_column else canon
        if sel.trust == "confirmed":
            confirmed.append(entry)
        elif sel.trust == "suspect":
            suspect.append(entry)
        elif sel.trust == "invalid":
            invalid.append(entry)
        elif sel.trust == "uncertain":
            uncertain.append(entry)
        else:
            missing.append(entry)

    return {
        "confirmed_channels": sorted(confirmed),
        "suspect_channels": sorted(suspect),
        "missing_channels": sorted(missing),
        "invalid_channels": sorted(invalid),
        "uncertain_channels": sorted(uncertain),
    }


def analyze_df(df: pd.DataFrame, filename: str, header_row: int, first_data_row: int) -> Dict[str, Any]:
    context, selections = select_analysis_channels(df)
    dff = context["df"]

    time_s = selections["Time_sec"].data
    rpm = selections["RPM"].data
    ect = selections["ECT"].data
    iat = selections["IAT"].data
    maf = selections["MAF_gps"].data
    injpw_b1 = selections["InjectorPW_B1_ms"].data
    injpw_b2 = selections["InjectorPW_B2_ms"].data
    stft_b1 = selections["STFT_B1"].data
    stft_b2 = selections["STFT_B2"].data
    ltft_b1 = selections["LTFT_B1"].data
    ltft_b2 = selections["LTFT_B2"].data
    map_kpa = selections["MAP_kPa"].data
    baro = selections["Baro_kPa"].data
    pe = selections["PE_Status"].data
    cl = selections["ClosedLoopActive"].data
    batt = selections["Battery_V"].data
    spark = selections["Spark_deg"].data
    kr = selections["KR_deg"].data
    sc_inlet_p = selections["BoostRelatedPressure_kPa"].data
    sc_inlet_v = selections["BoostRelatedVacuum_kPa"].data
    afr_ext = selections["AFR_Act_External"].data
    throttle_des = selections["ThrottleDesired_pct"].data
    throttle_cmd = selections["ThrottleCommanded_pct"].data
    throttle_act = selections["ThrottleActual_pct"].data
    fuel_rail = selections["FuelRailPressure"].data
    fuel_rail_rel = selections["FuelRailPressureRel"].data
    ethanol = selections["Ethanol_pct"].data
    oil_pressure = selections["OilPressure"].data
    fuel_pressure = selections["FuelPressure"].data

    if time_s is not None:
        time_s = try_float_series(time_s)
    if rpm is not None:
        rpm = try_float_series(rpm)

    oil_invalid_flat_zero = detect_invalid_flat_zero_pressure(oil_pressure, rpm)
    fuel_invalid_flat_zero = detect_invalid_flat_zero_pressure(fuel_pressure, rpm)

    if oil_invalid_flat_zero:
        selections["OilPressure"].trust = "invalid"
        selections["OilPressure"].reason = "flat_zero_invalid_sender"
    if fuel_invalid_flat_zero:
        selections["FuelPressure"].trust = "invalid"
        selections["FuelPressure"].reason = "flat_zero_invalid_sender"

    operating_mode, op_evidence, op_uncertain = classify_operating_mode(
        rpm=rpm,
        map_kpa=map_kpa,
        map_trust=selections["MAP_kPa"].trust,
        baro_kpa=baro,
        boost_pressure=sc_inlet_p,
        boost_vacuum=sc_inlet_v,
    )

    if operating_mode == "uncertain":
        if selections["MAP_kPa"].trust != "invalid":
            selections["MAP_kPa"].trust = "uncertain"
        selections["MAP_kPa"].reason = ", ".join(op_uncertain) if op_uncertain else selections["MAP_kPa"].reason
        if selections["BoostRelatedPressure_kPa"].selected_column:
            selections["BoostRelatedPressure_kPa"].trust = "uncertain"
        if selections["BoostRelatedVacuum_kPa"].selected_column:
            selections["BoostRelatedVacuum_kPa"].trust = "uncertain"

    pe_bool = convert_status_to_bool(pe) if pe is not None else pd.Series(dtype=bool)
    cl_bool = convert_status_to_bool(cl) if cl is not None else pd.Series(dtype=bool)

    throttle_mismatch_pct_mean = None
    throttle_mismatch_pct_max = None
    if throttle_des is not None and throttle_act is not None:
        td = try_float_series(throttle_des)
        ta = try_float_series(throttle_act)
        delta = td - ta
        if len(delta.dropna()) > 0:
            throttle_mismatch_pct_mean = float(delta.abs().mean(skipna=True))
            throttle_mismatch_pct_max = float(delta.abs().max(skipna=True))

    trim_summary = compute_trim_summary(stft_b1, stft_b2, ltft_b1, ltft_b2)
    injpw_avg = avg_bank(injpw_b1, injpw_b2)

    idle_segments = detect_idle_segments(time_s, rpm, throttle_act)
    kr_events = extract_kr_events(time_s, rpm, kr, spark)

    wideband_trusted = selections["AFR_Act_External"].trust == "confirmed"
    pe_conflict = False
    wideband_conflict = False
    analysis_limits = []

    if operating_mode == "uncertain":
        analysis_limits.append("operating_mode_uncertain")

    if pe is not None and cl is not None and len(pe_bool) and len(cl_bool):
        both_true = (pe_bool == True) & (cl_bool == True)
        if len(both_true) and both_true.mean() > 0.25:
            pe_conflict = True
            analysis_limits.append("pe_closed_loop_conflict")

    if not wideband_trusted and selections["AFR_Act_External"].selected_column:
        wideband_conflict = True
        analysis_limits.append("external_wideband_untrusted")

    if operating_mode == "uncertain" or pe_conflict or wideband_conflict:
        if selections["AFR_Act_External"].selected_column and selections["AFR_Act_External"].trust != "invalid":
            selections["AFR_Act_External"].trust = "uncertain"

    hard_stops = []
    if rpm is None or len(rpm.dropna()) == 0:
        hard_stops.append("missing_rpm")
    if selections["Time_sec"].selected_column is None:
        hard_stops.append("missing_time")

    trust_buckets = build_trust_buckets(selections)

    matched_raw_columns = {
        canon: sel.selected_column
        for canon, sel in selections.items()
        if sel.selected_column is not None
    }
    raw_candidates = {
        canon: sel.candidates
        for canon, sel in selections.items()
        if sel.candidates
    }

    duration_sec = None
    if time_s is not None and len(time_s.dropna()) >= 2:
        ts = time_s.dropna()
        duration_sec = float(ts.iloc[-1] - ts.iloc[0])

    return {
        "status": "ready" if not hard_stops else "blocked",
        "filename": filename,
        "row_count": int(len(dff)),
        "column_count": int(len(dff.columns)),
        "header_row_index": int(header_row),
        "first_data_row_index": int(first_data_row),
        "log_duration_sec": duration_sec,
        "matched_raw_columns": matched_raw_columns,
        "raw_candidate_columns": raw_candidates,
        "operating_mode": operating_mode,
        "operating_mode_evidence": op_evidence,
        "analysis_limits": analysis_limits,
        "hard_stops": hard_stops,
        "channel_selection_detail": {
            canon: {
                "selected_column": sel.selected_column,
                "candidates": sel.candidates,
                "reason": sel.reason,
                "trust": sel.trust,
            }
            for canon, sel in selections.items()
        },
        **trust_buckets,
        "summary": {
            "rpm_mean": float(rpm.mean(skipna=True)) if rpm is not None and len(rpm.dropna()) else None,
            "ect_mean": float(try_float_series(ect).mean(skipna=True)) if ect is not None and len(try_float_series(ect).dropna()) else None,
            "iat_mean": float(try_float_series(iat).mean(skipna=True)) if iat is not None and len(try_float_series(iat).dropna()) else None,
            "maf_mean_gps": float(try_float_series(maf).mean(skipna=True)) if maf is not None and len(try_float_series(maf).dropna()) else None,
            "injpw_avg_mean_ms": float(injpw_avg.mean(skipna=True)) if injpw_avg is not None and len(injpw_avg.dropna()) else None,
            "fuel_rail_pressure_mean": float(try_float_series(fuel_rail).mean(skipna=True)) if fuel_rail is not None and len(try_float_series(fuel_rail).dropna()) else None,
            "fuel_rail_pressure_rel_mean": float(try_float_series(fuel_rail_rel).mean(skipna=True)) if fuel_rail_rel is not None and len(try_float_series(fuel_rail_rel).dropna()) else None,
            "ethanol_pct_mean": float(try_float_series(ethanol).mean(skipna=True)) if ethanol is not None and len(try_float_series(ethanol).dropna()) else None,
            "battery_v_mean": float(try_float_series(batt).mean(skipna=True)) if batt is not None and len(try_float_series(batt).dropna()) else None,
            "external_wideband_present": selections["AFR_Act_External"].selected_column is not None,
            "external_wideband_trusted": wideband_trusted,
            "throttle_desired_present": selections["ThrottleDesired_pct"].selected_column is not None,
            "throttle_actual_present": selections["ThrottleActual_pct"].selected_column is not None,
            "throttle_commanded_present": selections["ThrottleCommanded_pct"].selected_column is not None,
            "throttle_mismatch_pct_mean": throttle_mismatch_pct_mean,
            "throttle_mismatch_pct_max": throttle_mismatch_pct_max,
            "oil_pressure_flat_zero_invalid_ignored": oil_invalid_flat_zero,
            "fuel_pressure_flat_zero_invalid_ignored": fuel_invalid_flat_zero,
            **trim_summary,
        },
        "idle_detection": {
            "segments": idle_segments,
            "count": len(idle_segments),
        },
        "kr_events": {
            "events": kr_events,
            "count": len(kr_events),
        },
        "rules_applied": {
            "used_confirmed_data_only_for_numeric_recommendations": True,
            "limited_edits_when_map_wideband_pe_conflict": True,
            "no_wot_fueling_corrections_without_trustworthy_actual_wideband": True,
            "narrowbands_not_used_as_actual_afr": True,
            "flat_zero_junk_sender_oil_fuel_channels_do_not_trigger_hard_stop": True,
        },
    }


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "streettunedai-log-analyzer",
        "endpoints": ["/health", "/validate", "/analyze"],
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/validate")
async def validate(file: UploadFile = File(...)) -> JSONResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")

    try:
        df, header_row, first_data_row = load_csv_with_detected_header(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV parse failed: {e}")

    context, selections = select_analysis_channels(df)
    time_s = selections["Time_sec"].data

    duration = None
    if time_s is not None and len(try_float_series(time_s).dropna()) >= 2:
        ts = try_float_series(time_s).dropna()
        duration = float(ts.iloc[-1] - ts.iloc[0])

    matched_raw_columns = {
        canon: sel.selected_column
        for canon, sel in selections.items()
        if sel.selected_column is not None
    }

    payload = {
        "status": "ready",
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(raw),
        "row_count": int(len(context["df"])),
        "column_count": int(len(context["df"].columns)),
        "header_row_index": int(header_row),
        "first_data_row_index": int(first_data_row),
        "log_duration_sec": duration,
        "matched_raw_columns": matched_raw_columns,
        **build_trust_buckets(selections),
    }
    return JSONResponse(payload)


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> JSONResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")

    try:
        df, header_row, first_data_row = load_csv_with_detected_header(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV parse failed: {e}")

    result = analyze_df(df, file.filename or "upload.csv", header_row, first_data_row)
    return JSONResponse(result)