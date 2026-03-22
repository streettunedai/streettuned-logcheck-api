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
        if line and "," in line and not line.startswith("["):
            return i, [c.strip() for c in line.split(",")]
    return None, []


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

    lines = text.splitlines()

    channel_info_index = None
    channel_data_index = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "[Channel Information]":
            channel_info_index = i
        elif stripped == "[Channel Data]":
            channel_data_index = i

    if channel_info_index is None:
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
            "message": "Missing [Channel Information] section"
        }

    ids_index, ids_row = next_nonempty_csv_row(lines, channel_info_index + 1)
    names_index, names_row = next_nonempty_csv_row(lines, (ids_index + 1) if ids_index is not None else channel_info_index + 1)
    units_index, units_row = next_nonempty_csv_row(lines, (names_index + 1) if names_index is not None else channel_info_index + 1)

    if names_index is None or not names_row:
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
            "message": "Could not find channel names row"
        }

    columns = names_row
    header_index = names_index

    if channel_data_index is None:
        return {
            "status": "ok",
            "platform": "unknown",
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": size_bytes,
            "readable": False,
            "header_found": True,
            "header_row_index": header_index,
            "row_count": 0,
            "column_count": len(columns),
            "columns": columns,
            "message": "Missing [Channel Data] section"
        }

    first_data_index, first_data_row = next_nonempty_csv_row(lines, channel_data_index + 1)

    if first_data_index is None:
        return {
            "status": "ok",
            "platform": "unknown",
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": size_bytes,
            "readable": False,
            "header_found": True,
            "header_row_index": header_index,
            "row_count": 0,
            "column_count": len(columns),
            "columns": columns,
            "message": "No data rows found"
        }

    data_text = "\n".join(lines[first_data_index:])
    reader = csv.reader(io.StringIO(data_text))
    parsed_rows = list(reader)
    row_count = len(parsed_rows)

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
        "header_row_index": header_index,
        "channel_ids_row_index": ids_index,
        "units_row_index": units_index,
        "first_data_row_index": first_data_index,
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns
    }