from __future__ import annotations

from typing import Any, Dict, Tuple

from fastapi import FastAPI, HTTPException, Request

from app.analyzers.ls_gas import CANONICAL_ALIASES, analyze_dataframe, validate_dataframe
from app.core.intake import extract_input_payload
from app.core.parser import parse_csv_bytes


app = FastAPI(title="StreetTunedAI LogCheck API", version="1.2.0")


def resolve_platform(request: Request) -> str:
    platform = (request.query_params.get("platform") or "ls").strip().lower()
    if platform in {"ls", "ls_gas"}:
        return "ls"
    if platform == "duramax":
        raise HTTPException(status_code=501, detail="Duramax analyzer is not enabled yet.")
    raise HTTPException(status_code=400, detail=f"Unsupported platform '{platform}'.")


def get_platform_handlers(platform: str) -> Tuple[Dict[str, Any], Any, Any]:
    if platform == "ls":
        return CANONICAL_ALIASES, validate_dataframe, analyze_dataframe
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
    canonical_aliases, platform_validate, _platform_analyze = get_platform_handlers(platform)
    content, filename, mime_type = await extract_input_payload(request)
    df, meta = parse_csv_bytes(content, filename=filename, canonical_aliases=canonical_aliases)
    return platform_validate(df, filename, mime_type, meta)


@app.post("/analyze")
async def analyze(request: Request) -> Dict[str, Any]:
    platform = resolve_platform(request)
    canonical_aliases, _platform_validate, platform_analyze = get_platform_handlers(platform)
    content, filename, _mime_type = await extract_input_payload(request)
    df, meta = parse_csv_bytes(content, filename=filename, canonical_aliases=canonical_aliases)
    return platform_analyze(df, meta)
