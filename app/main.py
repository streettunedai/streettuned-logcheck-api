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

def is_mostly_numeric(values):
    non_empty = [v.strip() for v in values if v.strip()]
    if not non_empty:
        return False

    numeric_count = 0
    for v in non_empty:
        try:
            float(v)
            numeric_count += 1
        except ValueError:
            pass

    return numeric_count / len(non_empty) >= 0.7

def is_likely_header(values):
    non_empty = [v.strip() for v in values if v.strip()]
    if len(non_empty) < 3:
        return False

    if is_mostly_numeric(non_empty):
        return False

    textish_count = 0
    for v in non_empty:
        has_alpha = any(ch.isalpha() for ch in v)
        if has_alpha:
            textish_count += 1

    return textish_count / len(non_empty) >= 0.4

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
            "row_count": 0,
            "column_count": 0,
            "columns": [],
            "message": "Could not decode uploaded file"
        }

    lines = text.splitlines()

    header_index = None
    columns = []

    for i, line in enumerate(lines[:80]):
        if "," not in line:
            continue

        possible_cols = [c.strip() for c in line.split(",")]

        if is_likely_header(possible_cols):
            header_index = i
            columns = possible_cols
            break

    if header_index is None:
        return {
            "status": "ok",
            "platform": "unknown",
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": size_bytes,
            "readable": False,
            "header_found": False,
            "row_count": 0,
            "column_count": 0,
            "columns": [],
            "message": "No usable text-based CSV header row found"
        }

    data_text = "\n".join(lines[header_index:])
    reader = csv.reader(io.StringIO(data_text))
    parsed_rows = list(reader)

    if len(parsed_rows) < 2:
        return {
            "status": "ok",
            "platform": "unknown",
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": size_bytes,
            "readable": False,
            "header_found": True,
            "row_count": 0,
            "column_count": len(columns),
            "columns": columns,
            "message": "Header found but no data rows found"
        }

    data_rows = parsed_rows[1:]
    row_count = len(data_rows)
    lower_cols = [c.lower() for c in columns]

    platform = "unknown"

    ls_markers = [
        "rpm", "map", "maf", "spark", "iat", "ect",
        "stft", "ltft", "injector", "knock", "wideband",
        "throttle", "commanded eq", "lambda", "afr"
    ]
    diesel_markers = [
        "rail pressure", "desired rail", "main injection",
        "pilot injection", "vane position", "soi", "mm3",
        "fuel rail pressure", "boost pressure desired",
        "desired boost", "actual boost"
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
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns
    }