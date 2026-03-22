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
        test = v.replace(".", "", 1).replace("-", "", 1)
        if test.isdigit():
            numeric_count += 1

    return numeric_count / len(non_empty) >= 0.8


def score_header(values):
    non_empty = [v.strip() for v in values if v.strip()]
    if len(non_empty) < 3:
        return -999

    score = 0

    # penalize rows that are mostly numeric IDs
    if is_mostly_numeric(non_empty):
        score -= 100

    # reward rows that look like real channel names
    for v in non_empty:
        lower = v.lower()

        if any(ch.isalpha() for ch in v):
            score += 2

        if " " in v or "_" in v or "/" in v or "(" in v or ")" in v:
            score += 2

        if lower in {
            "time", "rpm", "map", "maf", "spark", "iat", "ect", "tps",
            "lambda", "wideband", "boost", "knock", "stft", "ltft"
        }:
            score += 8

        if any(term in lower for term in [
            "pressure", "temp", "temperature", "advance", "injector",
            "spark", "fuel", "throttle", "pedal", "commanded", "desired",
            "airflow", "eq ratio", "lambda", "knock", "boost", "speed"
        ]):
            score += 4

    return score


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

    header_index = None
    columns = []
    best_score = -999

    for i, line in enumerate(lines[:50]):
        if "," not in line:
            continue

        possible_cols = [c.strip() for c in line.split(",")]
        current_score = score_header(possible_cols)

        if current_score > best_score:
            best_score = current_score
            header_index = i
            columns = possible_cols

    if header_index is None or best_score < 0:
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
            "message": "No usable CSV header row found"
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
            "header_row_index": header_index,
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
        "lambda", "throttle", "pedal"
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
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns
    }