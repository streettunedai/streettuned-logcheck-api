from fastapi import FastAPI, File, UploadFile

app = FastAPI()


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


def parse_uploaded_csv(raw_bytes):
    import io
    import csv
    import pandas as pd

    encodings = ["utf-8-sig", "utf-8", "utf-16", "cp1252"]
    last_error = None

    header_tokens = [
        "engine rpm",
        "knock retard",
        "throttle position",
        "intake manifold absolute pressure",
        "equivalence ratio commanded",
        "mass airflow",
        "vehicle speed",
        "accelerator pedal position",
        "air-fuel ratio commanded",
        "ethanol fuel %",
        "dfco active",
        "power enrichment",
        "engine oil pressure",
        "intake air temp",
        "engine coolant temp",
        "timing advance",
        "fuel pressure",
        "injector pulse width",
        "offset",
    ]

    def looks_like_number(value):
        try:
            float(str(value).strip().replace(",", ""))
            return True
        except Exception:
            return False

    def row_score(row):
        text_cells = [str(x).strip() for x in row]
        joined = " | ".join(text_cells).lower()
        score = 0
        for token in header_tokens:
            if token in joined:
                score += 1
        return score

    for enc in encodings:
        try:
            text = raw_bytes.decode(enc, errors="replace")
            text = text.replace("\x00", "")

            sample = text[:5000]
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                sep = dialect.delimiter
            except Exception:
                sep = ","

            reader = csv.reader(io.StringIO(text), delimiter=sep)
            rows = [row for row in reader if row and any(str(cell).strip() for cell in row)]

            if not rows:
                return {
                    "status": "error",
                    "message": "CSV file was empty"
                }

            header_row_index = None
            best_score = 0

            for i, row in enumerate(rows[:80]):
                score = row_score(row)
                if score > best_score:
                    best_score = score
                    header_row_index = i

            if header_row_index is None or best_score == 0:
                return {
                    "status": "error",
                    "message": "Could not find HP Tuners header row"
                }

            channel_ids_row_index = header_row_index - 1 if header_row_index - 1 >= 0 else None

            units_row_index = None
            for i in range(header_row_index + 1, min(len(rows), header_row_index + 4)):
                row = rows[i]
                joined = " | ".join(str(x).strip().lower() for x in row)
                if any(unit in joined for unit in ["rpm", "kpa", "psi", "deg", "°", "mph", "ms", "hz", "v", "lb/min", "g/s", "%", "f", "lambda", "afr"]):
                    units_row_index = i
                    break

            header_row = [str(x).strip() for x in rows[header_row_index]]
            col_count = len(header_row)

            seen = {}
            clean_headers = []
            for idx, col in enumerate(header_row):
                name = col if col else f"unnamed_{idx}"
                if name in seen:
                    seen[name] += 1
                    name = f"{name}_{seen[name]}"
                else:
                    seen[name] = 0
                clean_headers.append(name)

            first_data_row_index = None
            for i in range(header_row_index + 1, min(len(rows), header_row_index + 15)):
                row = rows[i]
                padded = row + [""] * (col_count - len(row))
                padded = padded[:col_count]

                numeric_count = sum(1 for cell in padded if looks_like_number(cell))
                if numeric_count >= max(3, min(8, col_count // 4)):
                    first_data_row_index = i
                    break

            if first_data_row_index is None:
                return {
                    "status": "error",
                    "message": "Could not find first data row"
                }

            data_rows = []
            for row in rows[first_data_row_index:]:
                padded = [str(x).strip() for x in row] + [""] * (col_count - len(row))
                padded = padded[:col_count]
                data_rows.append(padded)

            df = pd.DataFrame(data_rows, columns=clean_headers)

            for col in df.columns:
                df[col] = df[col].replace("", pd.NA)

            df = df.dropna(axis=1, how="all")

            return {
                "status": "ready",
                "dataframe": df,
                "delimiter": sep,
                "channel_ids_row_index": channel_ids_row_index,
                "header_row_index": header_row_index,
                "units_row_index": units_row_index,
                "first_data_row_index": first_data_row_index,
            }

        except Exception as e:
            last_error = str(e)

    return {
        "status": "error",
        "message": f"Could not parse CSV: {last_error}"
    }


def canonical_alias_map():
    return {
        "Time_sec": [
            "Offset",
            "Time",
            "Time (s)",
            "Elapsed Time",
        ],
        "RPM": [
            "Engine RPM",
            "RPM",
        ],
        "MAP_kPa": [
            "Intake Manifold Absolute Pressure (SAE)",
            "Manifold Absolute Pressure",
            "MAP",
            "MAP Sensor",
            "MAP Pressure",
        ],
        "Spark_deg": [
            "Timing Advance (SAE)",
            "Timing Advance",
            "Spark Advance",
            "Spark",
        ],
        "KR_deg": [
            "Knock Retard",
            "KR",
        ],
        "TotalKR_deg": [
            "Total Knock Retard",
        ],
        "EQ_Cmd": [
            "Equivalence Ratio Commanded (SAE)",
            "Commanded EQ Ratio",
            "Commanded EQ",
        ],
        "AFR_Cmd": [
            "Air-Fuel Ratio Commanded",
            "Commanded AFR",
        ],
        "FuelSys1_Status": [
            "Fuel System #1 Status (SAE)",
            "Fuel System 1 Status",
        ],
        "STFT_B1": [
            "STFT Bank 1",
            "STFT B1",
            "Short Term FT B1",
            "Short Term Fuel Trim Bank 1",
        ],
        "STFT_B2": [
            "STFT Bank 2",
            "STFT B2",
            "Short Term FT B2",
            "Short Term Fuel Trim Bank 2",
        ],
        "LTFT_B1": [
            "LTFT Bank 1",
            "LTFT B1",
            "Long Term FT B1",
            "Long Term Fuel Trim Bank 1",
        ],
        "LTFT_B2": [
            "LTFT Bank 2",
            "LTFT B2",
            "Long Term FT B2",
            "Long Term Fuel Trim Bank 2",
        ],
        "TPS_pct": [
            "Throttle Position",
            "Throttle Position (%)",
            "Throttle Position (SAE)",
            "TPS",
        ],
        "APP_pct": [
            "Accelerator Pedal Position",
            "APP",
            "APP %",
            "Accelerator Pedal",
        ],
        "InjPW_ms": [
            "Injector Pulse Width",
            "Injector PW",
            "Avg. Injector Pulse Width",
            "Average Injector Pulse Width",
            "Injector Pulse Width Bank 1",
        ],
        "MAF_gps": [
            "Mass Airflow",
            "MAF",
            "MAF Airflow Rate (SAE)",
        ],
        "DynAir_gps": [
            "Dynamic Airflow",
            "Dyn Air",
            "Dynamic Cylinder Air",
        ],
        "IAT": [
            "Intake Air Temp",
            "IAT",
            "Intake Air Temperature",
        ],
        "ECT": [
            "Engine Coolant Temp",
            "ECT",
            "Coolant Temp",
        ],
        "VehicleSpeed": [
            "Vehicle Speed",
            "Speed",
            "Vehicle Speed (SAE)",
        ],
        "PE_Status": [
            "Power Enrichment",
            "PE",
        ],
        "FuelPressure": [
            "Fuel Pressure (SAE)",
            "Fuel Pressure",
        ],
        "OilPressure": [
            "Engine Oil Pressure",
            "Oil Pressure",
        ],
        "Ethanol_pct": [
            "Ethanol Fuel %",
            "Ethanol %",
            "Alcohol %",
        ],
        "DFCO_Status": [
            "DFCO Active",
            "DFCO",
        ],
        "O2_B1S1": [
            "O2 Voltage B1S1",
            "O2 B1S1",
        ],
        "O2_B2S1": [
            "O2 Voltage B2S1",
            "O2 B2S1",
        ],
        "AFR_Act": [
            "AFR Wideband",
            "Wideband AFR",
            "Air Fuel Ratio Wideband",
            "AFR Actual",
        ],
        "Lambda_Act": [
            "Lambda Wideband",
            "Wideband Lambda",
            "Lambda Actual",
            "EQ Ratio Actual",
            "Equivalence Ratio Actual",
        ],
    }


def normalize_columns(df):
    import pandas as pd

    alias_map = canonical_alias_map()
    normalized = {}
    matched_raw_columns = {}
    confirmed_channels = []
    missing_channels = []

    def find_match(candidates):
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        lowered = {str(col).strip().lower(): col for col in df.columns}
        for candidate in candidates:
            key = str(candidate).strip().lower()
            if key in lowered:
                return lowered[key]
        return None

    for canonical_name, candidates in alias_map.items():
        raw_col = find_match(candidates)
        if raw_col is None:
            missing_channels.append(canonical_name)
            continue

        s = pd.to_numeric(df[raw_col], errors="coerce")
        normalized[canonical_name] = s
        matched_raw_columns[canonical_name] = raw_col
        confirmed_channels.append(canonical_name)

    ndf = pd.DataFrame(normalized)

    invalid_channels = {}
    uncertain_channels = {}

    if "FuelPressure" in ndf.columns:
        s = pd.to_numeric(ndf["FuelPressure"], errors="coerce").dropna()
        if not s.empty and float(s.abs().max()) == 0.0:
            invalid_channels["FuelPressure"] = {
                "status": "invalid",
                "reason": "Fuel Pressure channel stayed at 0.0 for the whole file"
            }

    if "OilPressure" in ndf.columns:
        s = pd.to_numeric(ndf["OilPressure"], errors="coerce").dropna()
        if not s.empty and float(s.abs().max()) == 0.0:
            invalid_channels["OilPressure"] = {
                "status": "invalid",
                "reason": "Oil Pressure channel stayed at 0.0 for the whole file"
            }

    if "AFR_Act" not in ndf.columns and "Lambda_Act" not in ndf.columns:
        uncertain_channels["ActualFueling"] = {
            "status": "missing",
            "reason": "No confirmed actual wideband AFR/lambda channel"
        }

    if "O2_B1S1" in ndf.columns or "O2_B2S1" in ndf.columns:
        uncertain_channels["NarrowbandO2"] = {
            "status": "informational_only",
            "reason": "Narrowband O2 voltage is present but is not valid actual AFR/lambda for WOT fueling decisions"
        }

    return {
        "normalized_df": ndf,
        "matched_raw_columns": matched_raw_columns,
        "confirmed_channels": confirmed_channels,
        "missing_channels": missing_channels,
        "invalid_channels": invalid_channels,
        "uncertain_channels": uncertain_channels,
        "raw_columns": list(df.columns),
    }


def series_min_max(series):
    s = series.dropna()
    if s.empty:
        return None
    return {
        "min": float(s.min()),
        "max": float(s.max())
    }


def series_mean(series):
    s = series.dropna()
    if s.empty:
        return None
    return float(s.mean())


def series_max(series):
    s = series.dropna()
    if s.empty:
        return None
    return float(s.max())


def series_min(series):
    s = series.dropna()
    if s.empty:
        return None
    return float(s.min())


def normalize_map_series_to_kpa(series):
    s = series.dropna()
    if s.empty:
        return series, {
            "status": "empty",
            "note": "MAP channel found but no numeric data"
        }

    raw_min = float(s.min())
    raw_max = float(s.max())

    if raw_max <= 20:
        converted = series * 6.894757293168361
        converted_clean = converted.dropna()
        return converted, {
            "status": "converted_from_psi",
            "raw_min": raw_min,
            "raw_max": raw_max,
            "converted_min": float(converted_clean.min()) if not converted_clean.empty else None,
            "converted_max": float(converted_clean.max()) if not converted_clean.empty else None,
        }

    return series, {
        "status": "already_kpa",
        "raw_min": raw_min,
        "raw_max": raw_max,
    }


@app.post("/validate")
async def validate(file: UploadFile = File(...)):
    raw = await file.read()
    size_bytes = len(raw)

    parse = parse_uploaded_csv(raw)
    if parse.get("status") != "ready":
        return parse

    df = parse["dataframe"].copy()
    norm = normalize_columns(df)

    platform = "unknown"
    if "RPM" in norm["confirmed_channels"] and ("EQ_Cmd" in norm["confirmed_channels"] or "AFR_Cmd" in norm["confirmed_channels"]):
        platform = "ls_gas"

    return {
        "status": "ready",
        "platform": platform,
        "filename": file.filename,
        "size_bytes": size_bytes,
        "delimiter": parse.get("delimiter"),
        "channel_ids_row_index": parse.get("channel_ids_row_index"),
        "header_row_index": parse.get("header_row_index"),
        "units_row_index": parse.get("units_row_index"),
        "first_data_row_index": parse.get("first_data_row_index"),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "matched_raw_columns": norm["matched_raw_columns"],
        "confirmed_channels": norm["confirmed_channels"],
        "missing_channels": norm["missing_channels"],
        "invalid_channels": norm["invalid_channels"],
        "uncertain_channels": norm["uncertain_channels"],
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    raw = await file.read()
    size_bytes = len(raw)

    parse = parse_uploaded_csv(raw)
    if parse.get("status") != "ready":
        return parse

    df = parse["dataframe"].copy()
    norm = normalize_columns(df)
    ndf = norm["normalized_df"].copy()

    summary = {}

    map_info = {
        "status": "missing",
        "note": "No MAP column found"
    }

    if "MAP_kPa" in ndf.columns:
        normalized_map, map_diag = normalize_map_series_to_kpa(ndf["MAP_kPa"])
        ndf["MAP_kPa"] = normalized_map
        map_info = {
            "source_column": norm["matched_raw_columns"].get("MAP_kPa"),
            **map_diag
        }

    summary_fields = [
        ("RPM", "engine_rpm"),
        ("MAP_kPa", "map_kpa"),
        ("Spark_deg", "timing_advance"),
        ("KR_deg", "knock_retard"),
        ("TotalKR_deg", "total_knock_retard"),
        ("APP_pct", "accelerator_pedal_position"),
        ("TPS_pct", "throttle_position"),
        ("VehicleSpeed", "vehicle_speed"),
        ("IAT", "iat"),
        ("ECT", "coolant_temp"),
        ("Ethanol_pct", "ethanol_percent"),
        ("AFR_Cmd", "afr_commanded"),
        ("EQ_Cmd", "eq_ratio_commanded"),
        ("InjPW_ms", "injector_pulse_width"),
        ("MAF_gps", "maf_gps"),
        ("DynAir_gps", "dynamic_airflow"),
        ("FuelPressure", "fuel_pressure"),
        ("OilPressure", "oil_pressure"),
        ("Time_sec", "time_sec"),
    ]

    for canonical_name, response_name in summary_fields:
        if canonical_name in ndf.columns:
            stats = series_min_max(ndf[canonical_name])
            if stats:
                summary[response_name] = stats

    log_duration_sec = None
    if "Time_sec" in ndf.columns:
        t = ndf["Time_sec"].dropna()
        if not t.empty:
            log_duration_sec = float(t.max() - t.min())

    closed_loop = {}
    if all(col in ndf.columns for col in ["STFT_B1", "LTFT_B1"]):
        b1_total = ndf["STFT_B1"] + ndf["LTFT_B1"]
        mean_b1 = series_mean(b1_total)
        if mean_b1 is not None:
            closed_loop["bank1_mean_total_trim"] = mean_b1

    if all(col in ndf.columns for col in ["STFT_B2", "LTFT_B2"]):
        b2_total = ndf["STFT_B2"] + ndf["LTFT_B2"]
        mean_b2 = series_mean(b2_total)
        if mean_b2 is not None:
            closed_loop["bank2_mean_total_trim"] = mean_b2

    if all(col in ndf.columns for col in ["STFT_B1", "LTFT_B1", "STFT_B2", "LTFT_B2"]):
        b1_total = ndf["STFT_B1"] + ndf["LTFT_B1"]
        b2_total = ndf["STFT_B2"] + ndf["LTFT_B2"]
        diff = (b1_total - b2_total).abs()
        max_imbalance = series_max(diff)
        if max_imbalance is not None:
            closed_loop["max_bank_to_bank_imbalance"] = max_imbalance

    idle = {}
    if "RPM" in ndf.columns:
        rpm_mean = series_mean(ndf["RPM"])
        rpm_stats = series_min_max(ndf["RPM"])
        if rpm_mean is not None:
            idle["rpm_mean_full_log"] = rpm_mean
        if rpm_stats is not None:
            idle["rpm_min_full_log"] = rpm_stats["min"]
            idle["rpm_max_full_log"] = rpm_stats["max"]

    injector_estimate = {}
    if "InjPW_ms" in ndf.columns and "RPM" in ndf.columns:
        pw_max = series_max(ndf["InjPW_ms"])
        injector_estimate["injpw_max_ms"] = pw_max
        duty_est = (ndf["InjPW_ms"] * ndf["RPM"]) / 1200.0
        duty_max = series_max(duty_est)
        if duty_max is not None:
            injector_estimate["estimated_duty_cycle_percent"] = duty_max

    hard_stops = []

    kr_max = series_max(ndf["KR_deg"]) if "KR_deg" in ndf.columns else None
    if kr_max is not None and kr_max > 3.0:
        hard_stops.append({
            "type": "knock_retard",
            "status": "tripped",
            "value": kr_max,
            "threshold": 3.0,
            "message": "KR exceeded safe threshold. Stop calibration changes and diagnose cause."
        })

    ect_max = series_max(ndf["ECT"]) if "ECT" in ndf.columns else None
    if ect_max is not None and ect_max > 240.0:
        hard_stops.append({
            "type": "coolant_temp",
            "status": "tripped",
            "value": ect_max,
            "threshold": 240.0,
            "message": "ECT exceeded safe threshold. Stop calibration changes and fix thermal issue."
        })

    oil_min = series_min(ndf["OilPressure"]) if "OilPressure" in ndf.columns else None
    oil_invalid = "OilPressure" in norm["invalid_channels"]
    if oil_min is not None and not oil_invalid and oil_min < 10.0:
        hard_stops.append({
            "type": "oil_pressure",
            "status": "tripped",
            "value": oil_min,
            "threshold": 10.0,
            "message": "Oil pressure dropped into unsafe range. Stop calibration changes and inspect engine/mechanical health."
        })

    operating_mode = {
        "map_based_classification": None,
        "note": None
    }

    map_max = summary.get("map_kpa", {}).get("max")
    if map_max is not None:
        if map_max <= 100:
            operating_mode["map_based_classification"] = "na"
            operating_mode["note"] = "MAP stayed at or below 100 kPa."
        elif map_max <= 105:
            operating_mode["map_based_classification"] = "verify"
            operating_mode["note"] = "MAP was between 100 and 105 kPa. Verify sensor scaling and actual boost state."
        else:
            operating_mode["map_based_classification"] = "boost"
            operating_mode["note"] = "MAP exceeded 105 kPa."

    recommendations = []

    has_actual_wideband = "AFR_Act" in ndf.columns or "Lambda_Act" in ndf.columns
    fuel_pressure_invalid = "FuelPressure" in norm["invalid_channels"]

    if hard_stops:
        recommendations.append("Hard stop triggered. Do not make calibration changes until the cause is diagnosed.")
    else:
        if map_max is not None and map_max <= 100:
            recommendations.append("Log appears naturally aspirated based on MAP. Use NA workflow, not boost workflow.")
        if kr_max is not None and kr_max > 0.5:
            recommendations.append("KR is present. Remove timing in the affected area and verify fueling, IAT, and mechanical noise.")

    if not has_actual_wideband:
        recommendations.append("No confirmed actual wideband AFR/lambda channel. Do not make WOT fueling corrections from this log alone.")

    if fuel_pressure_invalid:
        recommendations.append("Fuel pressure channel is invalid in this file. Do not use it for load or WOT fuel system decisions.")

    if oil_invalid:
        recommendations.append("Oil pressure channel is invalid in this file. Report it, but do not use it as a hard-stop trigger.")

    analysis_notes = []

    if closed_loop:
        analysis_notes.append("Closed-loop trims are available for part-throttle analysis.")
    if not has_actual_wideband:
        analysis_notes.append("Narrowband O2 voltage does not count as actual AFR/lambda.")
    if "PE_Status" in ndf.columns:
        analysis_notes.append("PE status channel is present.")
    if "FuelPressure" in norm["invalid_channels"]:
        analysis_notes.append("Fuel Pressure (SAE) stayed at 0.0 and was marked invalid.")
    if "OilPressure" in norm["invalid_channels"]:
        analysis_notes.append("Oil Pressure stayed at 0.0 and was marked invalid.")

    return {
        "status": "ready",
        "filename": file.filename,
        "size_bytes": size_bytes,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "header_row_index": parse.get("header_row_index"),
        "first_data_row_index": parse.get("first_data_row_index"),
        "log_duration_sec": log_duration_sec,
        "matched_raw_columns": norm["matched_raw_columns"],
        "confirmed_channels": norm["confirmed_channels"],
        "missing_channels": norm["missing_channels"],
        "invalid_channels": norm["invalid_channels"],
        "uncertain_channels": norm["uncertain_channels"],
        "summary": summary,
        "closed_loop": closed_loop,
        "idle": idle,
        "injector_estimate": injector_estimate,
        "unit_sanity": {
            "map": map_info
        },
        "hard_stops": hard_stops,
        "operating_mode": operating_mode,
        "analysis_notes": analysis_notes,
        "recommendations": recommendations
    }