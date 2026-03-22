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

    def is_mostly_numeric(values):
        cleaned = [v.strip() for v in values if v.strip()]
        if not cleaned:
            return False

        numeric_count = 0
        for v in cleaned:
            test = v.replace(".", "", 1).replace("-", "", 1)
            if test.isdigit():
                numeric_count += 1

        return (numeric_count / len(cleaned)) >= 0.7

    def looks_like_real_header(values):
        cleaned = [v.strip() for v in values if v.strip()]
        if len(cleaned) < 3:
            return False

        if is_mostly_numeric(cleaned):
            return False

        alpha_count = sum(1 for v in cleaned if any(ch.isalpha() for ch in v))
        return alpha_count >= 3

    # find first likely real header row within first 80 lines
    for i, line in enumerate(lines[:80]):
        if "," in line:
            possible_cols = [c.strip() for c in line.split(",")]
            if looks_like_real_header(possible_cols):
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