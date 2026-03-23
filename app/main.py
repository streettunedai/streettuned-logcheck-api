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
            for i in range(header_row_index + 1, min(len(rows), header_row_index + 12)):
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
                "header_row_index": header_row_index,
                "first_data_row_index": first_data_row_index,
            }

        except Exception as e:
            last_error = str(e)

    return {
        "status": "error",
        "message": f"Could not parse CSV: {last_error}"
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    import pandas as pd

    raw = await file.read()
    size_bytes = len(raw)

    parse = parse_uploaded_csv(raw)
    if parse.get("status") != "ready":
        return parse

    df = parse["dataframe"].copy()

    def safe_min_max(series):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return None
        return {
            "min": float(s.min()),
            "max": float(s.max())
        }

    def max_numeric(col_name):
        if col_name not in df.columns:
            return None
        s = pd.to_numeric(df[col_name], errors="coerce").dropna()
        if s.empty:
            return None
        return float(s.max())

    def min_numeric(col_name):
        if col_name not in df.columns:
            return None
        s = pd.to_numeric(df[col_name], errors="coerce").dropna()
        if s.empty:
            return None
        return float(s.min())

    def normalize_map_to_kpa(local_df):
        possible_map_cols = [
            "Intake Manifold Absolute Pressure (SAE)",
            "Manifold Absolute Pressure",
            "MAP",
        ]

        map_col = next((c for c in possible_map_cols if c in local_df.columns), None)

        if not map_col:
            return None, {
                "status": "missing",
                "note": "No MAP column found"
            }

        s = pd.to_numeric(local_df[map_col], errors="coerce").dropna()

        if s.empty:
            return None, {
                "status": "empty",
                "note": "MAP column found but no numeric data"
            }

        raw_min = float(s.min())
        raw_max = float(s.max())

        if raw_max <= 20:
            converted = pd.to_numeric(local_df[map_col], errors="coerce") * 6.894757
            converted_clean = converted.dropna()
            return converted, {
                "status": "converted_from_psi",
                "source_column": map_col,
                "raw_min": raw_min,
                "raw_max": raw_max,
                "converted_min": float(converted_clean.min()) if not converted_clean.empty else None,
                "converted_max": float(converted_clean.max()) if not converted_clean.empty else None,
            }

        return pd.to_numeric(local_df[map_col], errors="coerce"), {
            "status": "already_kpa",
            "source_column": map_col,
            "raw_min": raw_min,
            "raw_max": raw_max,
        }

    summary = {}

    if "Engine RPM" in df.columns:
        stats = safe_min_max(df["Engine RPM"])
        if stats:
            summary["engine_rpm"] = stats

    map_series, map_info = normalize_map_to_kpa(df)
    if map_series is not None:
        stats = safe_min_max(map_series)
        if stats:
            summary["map_kpa"] = stats

    if "Knock Retard" in df.columns:
        stats = safe_min_max(df["Knock Retard"])
        if stats:
            summary["knock_retard"] = stats

    if "Total Knock Retard" in df.columns:
        stats = safe_min_max(df["Total Knock Retard"])
        if stats:
            summary["total_knock_retard"] = stats

    if "Accelerator Pedal Position" in df.columns:
        stats = safe_min_max(df["Accelerator Pedal Position"])
        if stats:
            summary["accelerator_pedal_position"] = stats

    if "Throttle Position" in df.columns:
        stats = safe_min_max(df["Throttle Position"])
        if stats:
            summary["throttle_position"] = stats

    if "Vehicle Speed" in df.columns:
        stats = safe_min_max(df["Vehicle Speed"])
        if stats:
            summary["vehicle_speed"] = stats

    if "Intake Air Temp" in df.columns:
        stats = safe_min_max(df["Intake Air Temp"])
        if stats:
            summary["iat"] = stats
    elif "IAT" in df.columns:
        stats = safe_min_max(df["IAT"])
        if stats:
            summary["iat"] = stats

    if "Engine Coolant Temp" in df.columns:
        stats = safe_min_max(df["Engine Coolant Temp"])
        if stats:
            summary["coolant_temp"] = stats
    elif "ECT" in df.columns:
        stats = safe_min_max(df["ECT"])
        if stats:
            summary["coolant_temp"] = stats

    if "Ethanol Fuel %" in df.columns:
        stats = safe_min_max(df["Ethanol Fuel %"])
        if stats:
            summary["ethanol_percent"] = stats

    if "Air-Fuel Ratio Commanded" in df.columns:
        stats = safe_min_max(df["Air-Fuel Ratio Commanded"])
        if stats:
            summary["afr_commanded"] = stats

    if "Equivalence Ratio Commanded (SAE)" in df.columns:
        stats = safe_min_max(df["Equivalence Ratio Commanded (SAE)"])
        if stats:
            summary["eq_ratio_commanded"] = stats

    hard_stops = []

    kr_max = max_numeric("Knock Retard")
    if kr_max is not None and kr_max > 3.0:
        hard_stops.append({
            "type": "knock_retard",
            "status": "tripped",
            "value": kr_max,
            "threshold": 3.0,
            "message": "KR exceeded safe threshold. Stop calibration changes and diagnose cause."
        })

    ect_max = None
    if "Engine Coolant Temp" in df.columns:
        ect_max = max_numeric("Engine Coolant Temp")
    elif "ECT" in df.columns:
        ect_max = max_numeric("ECT")

    if ect_max is not None and ect_max > 240.0:
        hard_stops.append({
            "type": "coolant_temp",
            "status": "tripped",
            "value": ect_max,
            "threshold": 240.0,
            "message": "ECT exceeded safe threshold. Stop calibration changes and fix thermal issue."
        })

    oil_min = min_numeric("Engine Oil Pressure")
    if oil_min is not None and oil_min < 10.0:
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

    map_max = None
    if "map_kpa" in summary:
        map_max = summary["map_kpa"]["max"]

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

    if hard_stops:
        recommendations.append("Hard stop triggered. Do not make calibration changes until the cause is diagnosed.")
    else:
        if map_max is not None and map_max <= 100:
            recommendations.append("Log appears naturally aspirated based on MAP. Use NA workflow, not boost workflow.")
        if kr_max is not None and kr_max > 0.5:
            recommendations.append("KR is present. Remove timing in the affected area and verify fueling, IAT, and mechanical noise.")
        if kr_max is not None and kr_max > 3.0:
            recommendations.append("KR exceeds 3 degrees. Stop calibration changes and inspect cause before additional pulls.")

    return {
        "status": "ready",
        "filename": file.filename,
        "size_bytes": size_bytes,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "header_row_index": parse.get("header_row_index"),
        "first_data_row_index": parse.get("first_data_row_index"),
        "summary": summary,
        "unit_sanity": {
            "map": map_info
        },
        "hard_stops": hard_stops,
        "operating_mode": operating_mode,
        "recommendations": recommendations
    }