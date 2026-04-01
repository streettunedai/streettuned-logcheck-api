from __future__ import annotations

from typing import Any, Dict, Tuple

from fastapi import FastAPI, HTTPException, Request

from app.analyzers.duramax import (
    CANONICAL_ALIASES as DURAMAX_CANONICAL_ALIASES,
    analyze_dataframe as duramax_analyze_dataframe,
    validate_dataframe as duramax_validate_dataframe,
)
from app.analyzers.ls_gas import CANONICAL_ALIASES, analyze_dataframe, validate_dataframe
from app.analyzers.cummins import (
    CANONICAL_ALIASES as CUMMINS_CANONICAL_ALIASES,
    analyze_dataframe as cummins_analyze_dataframe,
    detect_cummins_platform,
    validate_dataframe as cummins_validate_dataframe,
)
from app.core.intake import extract_input_payload
from app.core.parser import parse_csv_bytes


app = FastAPI(title="StreetTunedAI LogCheck API", version="1.2.0")


def resolve_platform(request: Request) -> str:
    platform = (request.query_params.get("platform") or "auto").strip().lower()
    if platform in {"ls", "ls_gas"}:
        return "ls"
    if platform == "duramax":
        return "duramax"
    if platform in {"cummins", "cummins_12v_ve", "cummins_12v_ppump", "cummins_24v_vp44", "cummins_5_9_common_rail", "auto"}:
        return platform
    raise HTTPException(status_code=400, detail=f"Unsupported platform '{platform}'.")


def get_platform_handlers(platform: str) -> Tuple[Dict[str, Any], Any, Any]:
    if platform == "ls":
        return CANONICAL_ALIASES, validate_dataframe, analyze_dataframe
    if platform == "duramax":
        return DURAMAX_CANONICAL_ALIASES, duramax_validate_dataframe, duramax_analyze_dataframe
    if platform in {"cummins", "cummins_12v_ve", "cummins_12v_ppump", "cummins_24v_vp44", "cummins_5_9_common_rail"}:
        return CUMMINS_CANONICAL_ALIASES, cummins_validate_dataframe, cummins_analyze_dataframe
    raise HTTPException(status_code=400, detail=f"Unsupported platform '{platform}'.")


@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "name": "StreetTunedAI LogCheck API",
        "status": "online",
        "endpoints": ["/", "/health", "/validate", "/analyze"],
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "OK"}


@app.post("/validate")
async def validate(request: Request) -> Dict[str, Any]:
    platform = resolve_platform(request)
    content, filename, mime_type = await extract_input_payload(request)

    if platform == "auto":
        df, meta = parse_csv_bytes(content, filename=filename, canonical_aliases=CUMMINS_CANONICAL_ALIASES)
        detected, _details = detect_cummins_platform(df, {})
        if detected:
            return cummins_validate_dataframe(df, filename, mime_type, meta, requested_platform=detected)
        df, meta = parse_csv_bytes(content, filename=filename, canonical_aliases=CANONICAL_ALIASES)
        return validate_dataframe(df, filename, mime_type, meta)

    canonical_aliases, platform_validate, _platform_analyze = get_platform_handlers(platform)
    df, meta = parse_csv_bytes(content, filename=filename, canonical_aliases=canonical_aliases)
    if platform.startswith("cummins") or platform == "cummins":
        requested = None if platform == "cummins" else platform
        return platform_validate(df, filename, mime_type, meta, requested_platform=requested)
    return platform_validate(df, filename, mime_type, meta)


@app.post("/analyze")
async def analyze(request: Request) -> Dict[str, Any]:
    platform = resolve_platform(request)
    content, filename, _mime_type = await extract_input_payload(request)

    if platform == "auto":
        df, meta = parse_csv_bytes(content, filename=filename, canonical_aliases=CUMMINS_CANONICAL_ALIASES)
        detected, _details = detect_cummins_platform(df, {})
        if detected:
            return cummins_analyze_dataframe(df, meta, requested_platform=detected)
        df, meta = parse_csv_bytes(content, filename=filename, canonical_aliases=CANONICAL_ALIASES)
        return analyze_dataframe(df, meta)

    canonical_aliases, _platform_validate, platform_analyze = get_platform_handlers(platform)
    df, meta = parse_csv_bytes(content, filename=filename, canonical_aliases=canonical_aliases)
    if platform.startswith("cummins") or platform == "cummins":
        requested = None if platform == "cummins" else platform
        return platform_analyze(df, meta, requested_platform=requested)
    return platform_analyze(df, meta)
