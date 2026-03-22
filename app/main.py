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


def parse_csv_upload(contents: bytes):
    size_bytes = len(contents)

    try:
        text = contents.decode("utf-8-sig", errors="replace")
    except Exception:
        return {
            "ok": False,
            "size_bytes": size_bytes,
            "message": "Could not decode uploaded file"
        }

    lines = text.splitlines()

    header_row = None
    header_index = None
    columns = []

    # find first likely header row within first 50 lines
    for i, line in enumerate(lines[:50]):
        if "," in line:
            possible_cols = [c.strip() for c in line.split(",")]
            non_empty = [c for c in possible_cols if c]
            if len(non_empty) >= 3:
                header_row = line
                header_index = i
                columns = possible_cols
                break

    if header_row is None:
        return {
            "ok": False,
            "size_bytes": size_bytes,
            "message": "No usable CSV header row found"
        }

    data_text = "\n".join(lines[header_index:])
    reader = csv.reader(io.StringIO(data_text))
    parsed_rows = list(reader)

    if len(parsed_rows) < 2:
        return {
            "ok": False,
            "size_bytes": size_bytes,
            "header_found": True,
            "header_row_index": header_index,
            "columns": columns,
            "message": "Header found but no data rows found"
        }

    data_rows = parsed_rows[1:]
    row_count = len(data_rows)
    lower_cols = [c.lower() for c in columns]

    # platform guess
    platform = "unknown"

    ls_markers = [
        "rpm", "map", "maf", "spark", "iat", "ect",
        "stft", "ltft", "injector", "knock", "equivalence ratio"
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

    has_rpm = any("rpm" in col for col in lower_cols)
    has_map = any("map" in col or "manifold absolute pressure" in col for col in lower_cols)
    has_maf = any("maf" in col or "mass airflow" in col for col in lower_cols)
    has_spark = any("spark" in col or "timing advance" in col for col in lower_cols)
    has_kr = any("knock retard" in col for col in lower_cols)
    has_wideband = any(
        "wideband" in col or
        "lambda" in col or
        "equivalence ratio" in col or
        "air-fuel ratio" in col
        for col in lower_cols
    )

    looks_like_hptuners = any(
        "sae" in col or
        "fuel trim cell" in col or
        "power enrichment" in col or
        "dfco active" in col
        for col in lower_cols
    )

    if has_rpm and (has_map or has_maf) and has_spark:
        recommended_next_step = "ready_for_analysis"
    else:
        recommended_next_step = "missing_key_channels"

    return {
        "ok": True,
        "size_bytes": size_bytes,
        "header_found": True,
        "header_row_index": header_index,
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns,
        "platform": platform,
        "has_rpm": has_rpm,
        "has_map": has_map,
        "has_maf": has_maf,
        "has_spark": has_spark,
        "has_kr": has_kr,
        "has_wideband": has_wideband,
        "looks_like_hptuners": looks_like_hptuners,
        "recommended_next_step": recommended_next_step,
        "lower_cols": lower_cols,
    }


@app.post("/validate")
async def validate(file: UploadFile = File(...)):
    contents = await file.read()
    parsed = parse_csv_upload(contents)

    if not parsed["ok"]:
        return {
            "status": "error",
            "platform": "unknown",
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": parsed.get("size_bytes", len(contents)),
            "readable": False,
            "header_found": parsed.get("header_found", False),
            "header_row_index": parsed.get("header_row_index"),
            "row_count": parsed.get("row_count", 0),
            "column_count": len(parsed.get("columns", [])),
            "columns": parsed.get("columns", []),
            "message": parsed.get("message", "File could not be parsed")
        }

    return {
        "status": "ready",
        "platform": parsed["platform"],
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": parsed["size_bytes"],
        "readable": True,
        "header_found": parsed["header_found"],
        "header_row_index": parsed["header_row_index"],
        "row_count": parsed["row_count"],
        "column_count": parsed["column_count"],
        "columns": parsed["columns"],
        "has_rpm": parsed["has_rpm"],
        "has_map": parsed["has_map"],
        "has_maf": parsed["has_maf"],
        "has_spark": parsed["has_spark"],
        "has_kr": parsed["has_kr"],
        "has_wideband": parsed["has_wideband"],
        "looks_like_hptuners": parsed["looks_like_hptuners"],
        "recommended_next_step": parsed["recommended_next_step"]
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    parsed = parse_csv_upload(contents)

    if not parsed["ok"]:
        return {
            "status": "error",
            "platform": "unknown",
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": parsed.get("size_bytes", len(contents)),
            "readable": False,
            "message": parsed.get("message", "File could not be parsed")
        }

    columns = parsed["columns"]
    lower_cols = parsed["lower_cols"]

    missing_for_review = []
    if not parsed["has_rpm"]:
        missing_for_review.append("RPM")
    if not parsed["has_map"]:
        missing_for_review.append("MAP")
    if not parsed["has_maf"]:
        missing_for_review.append("MAF")
    if not parsed["has_spark"]:
        missing_for_review.append("Spark")
    if not parsed["has_kr"]:
        missing_for_review.append("Knock Retard")
    if not parsed["has_wideband"]:
        missing_for_review.append("Wideband or Lambda")

    has_pe = any("power enrichment" in col for col in lower_cols)
    has_dfco = any("dfco active" in col for col in lower_cols)
    has_iat = any("iat" in col or "intake air temp" in col for col in lower_cols)
    has_ect = any("ect" in col or "engine coolant temp" in col for col in lower_cols)
    has_stft = any("short term fuel trim" in col or "stft" in col for col in lower_cols)
    has_ltft = any("long term fuel trim" in col or "ltft" in col for col in lower_cols)
    has_throttle = any("throttle" in col for col in lower_cols)
    has_pedal = any("accelerator pedal" in col or "pedal position" in col for col in lower_cols)

    likely_usable_for_real_review = (
        parsed["has_rpm"] and
        (parsed["has_map"] or parsed["has_maf"]) and
        parsed["has_spark"] and
        parsed["has_kr"]
    )

    return {
        "status": "analyzed",
        "platform": parsed["platform"],
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": parsed["size_bytes"],
        "readable": True,
        "looks_like_hptuners": parsed["looks_like_hptuners"],
        "row_count": parsed["row_count"],
        "column_count": parsed["column_count"],
        "has_rpm": parsed["has_rpm"],
        "has_map": parsed["has_map"],
        "has_maf": parsed["has_maf"],
        "has_spark": parsed["has_spark"],
        "has_kr": parsed["has_kr"],
        "has_wideband": parsed["has_wideband"],
        "has_pe": has_pe,
        "has_dfco": has_dfco,
        "has_iat": has_iat,
        "has_ect": has_ect,
        "has_stft": has_stft,
        "has_ltft": has_ltft,
        "has_throttle": has_throttle,
        "has_pedal": has_pedal,
        "likely_usable_for_real_review": likely_usable_for_real_review,
        "missing_for_review": missing_for_review,
        "recommended_next_step": (
            "proceed_to_log_summary" if likely_usable_for_real_review else "log_missing_channels"
        ),
        "detected_columns_sample": columns[:20]
    }