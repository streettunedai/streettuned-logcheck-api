from fastapi import FastAPI, UploadFile, File
import csv
import io

app = FastAPI()


@app.get("/")
def root():
    return {"message": "StreetTunedAI LogCheck API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


def next_nonempty_csv_row(lines, start_index):
    for i in range(start_index, len(lines)):
        line = lines[i].strip()
        if line and "," in line and "[" not in line:
            return i, [c.strip() for c in line.split(",")]
    return None, []


def parse_hp_tuners_csv(text):
    lines = text.splitlines()

    channel_info_index = None
    channel_data_index = None

    for i, line in enumerate(lines):
        if "[Channel Information]" in line:
            channel_info_index = i
        if "[Channel Data]" in line:
            channel_data_index = i

    if channel_info_index is None:
        return {"ok": False, "message": "Missing [Channel Information] section"}

    ids_index, ids_row = next_nonempty_csv_row(lines, channel_info_index + 1)
    names_index, names_row = next_nonempty_csv_row(
        lines,
        ids_index + 1 if ids_index is not None else channel_info_index + 1
    )
    units_index, units_row = next_nonempty_csv_row(
        lines,
        names_index + 1 if names_index is not None else channel_info_index + 1
    )

    if names_index is None or not names_row:
        return {"ok": False, "message": "Could not find channel names row"}

    if channel_data_index is None:
        return {"ok": False, "message": "Missing [Channel Data] section"}

    first_data_index, first_data_row = next_nonempty_csv_row(lines, channel_data_index + 1)

    if first_data_index is None:
        return {"ok": False, "message": "No data rows found"}

    data_text = "\n".join(lines[first_data_index:])
    reader = csv.reader(io.StringIO(data_text))
    parsed_rows = list(reader)

    return {
        "ok": True,
        "channel_ids_row_index": ids_index,
        "header_row_index": names_index,
        "units_row_index": units_index,
        "first_data_row_index": first_data_index,
        "columns": names_row,
        "rows": parsed_rows,
    }


def safe_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text == "---":
        return None
    try:
        return float(text)
    except Exception:
        return None


def get_series(rows, columns, target_name):
    if target_name not in columns:
        return []
    idx = columns.index(target_name)
    values = []
    for row in rows:
        if idx < len(row):
            v = safe_float(row[idx])
            if v is not None:
                values.append(v)
    return values


def min_max(values):
    if not values:
        return {"min": None, "max": None}
    return {"min": min(values), "max": max(values)}


@app.post("/validate")
async def validate(file: UploadFile = File(...)):
    contents = await file.read()
    size_bytes = len(contents)

    try:
        text = contents.decode("utf-8-sig", errors="replace")
    except Exception:
        return {
            "status": "error",
            "platform": "unknown",
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": size_bytes,
            "readable": False,
            "header_found": False,
            "header_row_index": None,
            "row_count": 0,
            "column_count": 0,
            "columns": [],
            "message": "Could not decode uploaded file"
        }

    parsed = parse_hp_tuners_csv(text)

    if not parsed["ok"]:
        return {
            "status": "ok",
            "platform": "unknown",
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": size_bytes,
            "readable": False,
            "header_found": False,
            "header_row_index": None,
            "row_count": 0,
            "column_count": 0,
            "columns": [],
            "message": parsed["message"]
        }

    columns = parsed["columns"]
    rows = parsed["rows"]
    lower_cols = [c.lower() for c in columns]

    platform = "unknown"

    ls_markers = [
        "rpm", "map", "maf", "spark", "iat", "ect",
        "stft", "ltft", "injector", "knock", "wideband",
        "air-fuel ratio", "equivalence ratio", "throttle"
    ]
    diesel_markers = [
        "rail pressure", "desired rail", "main injection",
        "pilot injection", "vane position", "soi", "mm3",
        "fuel rail pressure", "boost pressure desired"
    ]

    ls_hits = sum(1 for marker in ls_markers if any(marker in col for col in lower_cols))
    diesel_hits = sum(1 for marker in diesel_markers if any(marker in col for col in lower_cols))

    if diesel_hits > ls_hits and diesel_hits >= 2:
        platform = "diesel"
    elif ls_hits >= 2:
        platform = "ls_gas"

    return {
        "status": "ready",
        "platform": platform,
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": size_bytes,
        "readable": True,
        "header_found": True,
        "header_row_index": parsed["header_row_index"],
        "channel_ids_row_index": parsed["channel_ids_row_index"],
        "units_row_index": parsed["units_row_index"],
        "first_data_row_index": parsed["first_data_row_index"],
        "row_count": len(rows),
        "column_count": len(columns),
        "columns": columns
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    size_bytes = len(contents)

    try:
        text = contents.decode("utf-8-sig", errors="replace")
    except Exception:
        return {
            "status": "error",
            "message": "Could not decode uploaded file"
        }

    parsed = parse_hp_tuners_csv(text)

    if not parsed["ok"]:
        return {
            "status": "error",
            "message": parsed["message"]
        }

    columns = parsed["columns"]
    rows = parsed["rows"]

    rpm = get_series(rows, columns, "Engine RPM")
    map_kpa = get_series(rows, columns, "Intake Manifold Absolute Pressure (SAE)")
    knock = get_series(rows, columns, "Knock Retard")
    total_knock = get_series(rows, columns, "Total Knock Retard")
    pedal = get_series(rows, columns, "Accelerator Pedal Position")
    throttle = get_series(rows, columns, "Throttle Position")
    vehicle_speed = get_series(rows, columns, "Vehicle Speed (SAE)")
    iat = get_series(rows, columns, "Intake Air Temp (SAE)")
    coolant = get_series(rows, columns, "Engine Coolant Temp")
    ethanol = get_series(rows, columns, "Ethanol Fuel % (SAE)")
    afr_cmd = get_series(rows, columns, "Air-Fuel Ratio Commanded")
    eq_cmd = get_series(rows, columns, "Equivalence Ratio Commanded (SAE)")

    return {
        "status": "ready",
        "filename": file.filename,
        "size_bytes": size_bytes,
        "row_count": len(rows),
        "column_count": len(columns),
        "summary": {
            "engine_rpm": min_max(rpm),
            "map_kpa": min_max(map_kpa),
            "knock_retard": min_max(knock),
            "total_knock_retard": min_max(total_knock),
            "accelerator_pedal_position": min_max(pedal),
            "throttle_position": min_max(throttle),
            "vehicle_speed": min_max(vehicle_speed),
            "iat": min_max(iat),
            "coolant_temp": min_max(coolant),
            "ethanol_percent": min_max(ethanol),
            "afr_commanded": min_max(afr_cmd),
            "eq_ratio_commanded": min_max(eq_cmd),
        }
    }