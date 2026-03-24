from __future__ import annotations

import csv
import io
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import HTTPException


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float, np.number)):
        try:
            if pd.isna(x):
                return None
        except Exception:
            pass
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "n/a", "na", "--"}:
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None


def series_numeric(raw: pd.Series) -> pd.Series:
    return pd.to_numeric(raw.astype(str).str.strip().str.replace(",", "", regex=False), errors="coerce")


def clean_column_name(col: str) -> str:
    return re.sub(r"\s+", " ", str(col).strip()).replace("\ufeff", "")


def decode_bytes(content: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            return content.decode(enc)
        except Exception:
            continue
    return content.decode("utf-8", errors="replace")


def detect_delimiter(sample_text: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample_text[:8192], delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except Exception:
        for delim in [",", "\t", ";", "|"]:
            if delim in sample_text:
                return delim
        return ","


def looks_numeric_row(values: List[str]) -> float:
    cleaned = [str(v).strip() for v in values if str(v).strip() != ""]
    if not cleaned:
        return 0.0
    numeric = sum(1 for v in cleaned if safe_float(v) is not None)
    return numeric / max(len(cleaned), 1)


def alias_hit_score(values: List[str], canonical_aliases: Dict[str, List[str]]) -> int:
    vals = {clean_column_name(v).lower() for v in values if str(v).strip()}
    hits = 0
    for alias_list in canonical_aliases.values():
        for alias in alias_list:
            if clean_column_name(alias).lower() in vals:
                hits += 1
                break
    return hits


def detect_header_and_data_rows(lines: List[List[str]], canonical_aliases: Dict[str, List[str]]) -> Tuple[int, int]:
    best_header_idx = 0
    best_score = -1.0

    max_scan = min(len(lines), 80)
    for i in range(max_scan):
        row = [clean_column_name(v) for v in lines[i]]
        nonempty = [v for v in row if v]
        if len(nonempty) < 3:
            continue

        hits = alias_hit_score(nonempty, canonical_aliases)
        unique_ratio = len(set(nonempty)) / max(len(nonempty), 1)
        next_numeric = 0.0
        for j in range(i + 1, min(i + 6, len(lines))):
            next_numeric = max(next_numeric, looks_numeric_row(lines[j]))

        score = hits * 10 + unique_ratio * 2 + next_numeric * 8
        if score > best_score:
            best_score = score
            best_header_idx = i

    first_data_idx = best_header_idx + 1
    for i in range(best_header_idx + 1, min(best_header_idx + 20, len(lines))):
        if looks_numeric_row(lines[i]) >= 0.35:
            first_data_idx = i
            break

    return best_header_idx, first_data_idx


def read_raw_lines(text: str, delimiter: str) -> List[List[str]]:
    f = io.StringIO(text)
    return list(csv.reader(f, delimiter=delimiter))


def parse_csv_bytes(
    content: bytes,
    filename: str = "upload.csv",
    canonical_aliases: Optional[Dict[str, List[str]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    text = decode_bytes(content)
    delimiter = detect_delimiter(text)
    raw_lines = read_raw_lines(text, delimiter)

    if not raw_lines:
        raise HTTPException(status_code=400, detail="CSV appears empty.")

    alias_map = canonical_aliases or {}
    header_row_index, first_data_row_index = detect_header_and_data_rows(raw_lines, alias_map)

    df = pd.read_csv(
        io.BytesIO(content),
        sep=delimiter,
        engine="python",
        skiprows=header_row_index,
        header=0,
        dtype=str,
        on_bad_lines="skip",
    )

    df.columns = [clean_column_name(c) for c in df.columns]
    df = df.loc[:, [c != "" and not str(c).startswith("Unnamed:") for c in df.columns]].copy()

    pre_rows_to_drop = max(first_data_row_index - (header_row_index + 1), 0)
    if pre_rows_to_drop > 0 and len(df) > pre_rows_to_drop:
        df = df.iloc[pre_rows_to_drop:].reset_index(drop=True)

    df = df.dropna(axis=0, how="all").reset_index(drop=True)

    meta = {
        "filename": filename,
        "delimiter": delimiter,
        "header_row_index": header_row_index,
        "first_data_row_index": first_data_row_index,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "raw_text_preview": text[:500],
        "size_bytes": len(content),
    }
    return df, meta


def choose_best_column(columns: List[str], aliases: List[str]) -> Optional[str]:
    normalized_cols = {clean_column_name(c).lower(): c for c in columns}

    for alias in aliases:
        key = clean_column_name(alias).lower()
        if key in normalized_cols:
            return normalized_cols[key]

    alias_norms = [clean_column_name(a).lower() for a in aliases]
    for col in columns:
        c = clean_column_name(col).lower()
        if c in alias_norms:
            return col

    for alias in aliases:
        a = clean_column_name(alias).lower()
        for col in columns:
            c = clean_column_name(col).lower()
            if c == a:
                return col

    for alias in aliases:
        a = clean_column_name(alias).lower()
        for col in columns:
            c = clean_column_name(col).lower()
            if a in c or c in a:
                return col

    return None


def calc_log_duration(num: pd.DataFrame) -> Optional[float]:
    if "Time_sec" in num and num["Time_sec"].dropna().size > 1:
        s = num["Time_sec"].dropna()
        return float(s.iloc[-1] - s.iloc[0])
    return None
