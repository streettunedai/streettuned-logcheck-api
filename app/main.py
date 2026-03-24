from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, Request

from app.analyzers.ls_gas import CANONICAL_ALIASES, analyze_dataframe, validate_dataframe
from app.core.intake import extract_input_payload
from app.core.parser import parse_csv_bytes


app = FastAPI(title="StreetTunedAI LogCheck API", version="1.2.0")


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
    content, filename, mime_type = await extract_input_payload(request)
    df, meta = parse_csv_bytes(content, filename=filename, canonical_aliases=CANONICAL_ALIASES)
    return validate_dataframe(df, filename, mime_type, meta)


@app.post("/analyze")
async def analyze(request: Request) -> Dict[str, Any]:
    content, filename, _mime_type = await extract_input_payload(request)
    df, meta = parse_csv_bytes(content, filename=filename, canonical_aliases=CANONICAL_ALIASES)
    return analyze_dataframe(df, meta)
