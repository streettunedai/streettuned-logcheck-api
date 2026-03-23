from fastapi import FastAPI, File, UploadFile

app = FastAPI()

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

    return {
        "status": "ready",
        "filename": file.filename,
        "size_bytes": size_bytes,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "summary": summary,
        "unit_sanity": {
            "map": map_info
        }
    }