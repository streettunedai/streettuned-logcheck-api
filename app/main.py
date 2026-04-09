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
from app.core.intake import IntakeError, extract_input_payload
from app.core.parser import parse_csv_bytes


app = FastAPI(title="StreetTunedAI LogCheck API", version="1.2.0")


def resolve_platform(request: Request, platform_hint: str | None = None) -> str:
    platform = (request.query_params.get("platform") or "auto").strip().lower()
    if platform in {"ls", "ls_gas"}:
        return "ls"
    if platform == "mopar":
        return "ls"
    if platform == "duramax":
        return "duramax"
    if platform in {"cummins", "cummins_12v_ve", "cummins_12v_ppump", "cummins_24v_vp44", "cummins_5_9_common_rail", "auto"}:
        return platform
    if platform == "auto" and (platform_hint or "").strip().lower() == "mopar":
        return "auto"
    raise HTTPException(status_code=400, detail=f"Unsupported platform '{platform}'.")


def get_platform_handlers(platform: str) -> Tuple[Dict[str, Any], Any, Any]:
    if platform == "ls":
        return CANONICAL_ALIASES, validate_dataframe, analyze_dataframe
    if platform == "duramax":
        return DURAMAX_CANONICAL_ALIASES, duramax_validate_dataframe, duramax_analyze_dataframe
    if platform in {"cummins", "cummins_12v_ve", "cummins_12v_ppump", "cummins_24v_vp44", "cummins_5_9_common_rail"}:
        return CUMMINS_CANONICAL_ALIASES, cummins_validate_dataframe, cummins_analyze_dataframe
    raise HTTPException(status_code=400, detail=f"Unsupported platform '{platform}'.")


def user_error(stage: str, error_code: str, message: str, platform_guess: str | None = None) -> Dict[str, Any]:
    return {
        "ok": False,
        "stage": stage,
        "error_code": error_code,
        "message": message,
        "platform_guess": platform_guess,
    }


def normalize_platform_result(stage: str, result: Dict[str, Any]) -> Dict[str, Any]:
    if result.get("status") == "error":
        message = (
            result.get("error", {}) or {}
        ).get("message") or result.get("analysis", {}).get("data_summary") or "Request could not be completed."
        error_code = "platform_uncertain" if result.get("error_type") == "platform_detection_failed" else "insufficient_channels"
        return user_error(stage, error_code, message, result.get("platform"))
    return result


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
    try:
        payload = await extract_input_payload(request)
    except IntakeError as e:
        return user_error("validate", e.error_code, e.message)

    platform = resolve_platform(request, payload.platform_hint)
    content, filename, mime_type = payload.content, payload.filename, payload.mime_type

    try:
        if platform == "auto":
            df, meta = parse_csv_bytes(content, filename=filename, canonical_aliases=CUMMINS_CANONICAL_ALIASES)
            detected, _details = detect_cummins_platform(df, {})
            if detected:
                out = cummins_validate_dataframe(df, filename, mime_type, meta, requested_platform=detected)
                out = normalize_platform_result("validate", out)
                out["ok"] = out.get("status") != "error" and out.get("ok", True)
                return out
            df, meta = parse_csv_bytes(content, filename=filename, canonical_aliases=CANONICAL_ALIASES)
            out = validate_dataframe(df, filename, mime_type, meta, platform_hint=payload.platform_hint)
            out["ok"] = True
            return out

        canonical_aliases, platform_validate, _platform_analyze = get_platform_handlers(platform)
        df, meta = parse_csv_bytes(content, filename=filename, canonical_aliases=canonical_aliases)
        if platform.startswith("cummins") or platform == "cummins":
            requested = None if platform == "cummins" else platform
            out = platform_validate(df, filename, mime_type, meta, requested_platform=requested)
            out = normalize_platform_result("validate", out)
            out["ok"] = out.get("status") != "error" and out.get("ok", True)
            return out
        if platform == "ls":
            out = platform_validate(df, filename, mime_type, meta, platform_hint=payload.platform_hint)
        else:
            out = platform_validate(df, filename, mime_type, meta)
        out["ok"] = True
        return out
    except HTTPException:
        return user_error("validate", "unreadable_file", "Attached file could not be parsed.")
    except Exception:
        return user_error("validate", "unreadable_file", "Attached file could not be parsed.")


@app.post("/analyze")
async def analyze(request: Request) -> Dict[str, Any]:
    try:
        payload = await extract_input_payload(request)
    except IntakeError as e:
        return user_error("analyze", e.error_code, e.message)

    platform = resolve_platform(request, payload.platform_hint)
    content, filename, _mime_type = payload.content, payload.filename, payload.mime_type

    try:
        if platform == "auto":
            df, meta = parse_csv_bytes(content, filename=filename, canonical_aliases=CUMMINS_CANONICAL_ALIASES)
            detected, _details = detect_cummins_platform(df, {})
            if detected:
                out = cummins_analyze_dataframe(df, meta, requested_platform=detected)
                out = normalize_platform_result("analyze", out)
                out["ok"] = out.get("status") != "error" and out.get("ok", True)
                return out
            df, meta = parse_csv_bytes(content, filename=filename, canonical_aliases=CANONICAL_ALIASES)
            out = analyze_dataframe(df, meta, platform_hint=payload.platform_hint)
            out["ok"] = True
            return out

        canonical_aliases, _platform_validate, platform_analyze = get_platform_handlers(platform)
        df, meta = parse_csv_bytes(content, filename=filename, canonical_aliases=canonical_aliases)
        if platform.startswith("cummins") or platform == "cummins":
            requested = None if platform == "cummins" else platform
            out = platform_analyze(df, meta, requested_platform=requested)
            out = normalize_platform_result("analyze", out)
            out["ok"] = out.get("status") != "error" and out.get("ok", True)
            return out
        if platform == "ls":
            out = platform_analyze(df, meta, platform_hint=payload.platform_hint)
        else:
            out = platform_analyze(df, meta)
        out["ok"] = True
        return out
    except HTTPException:
        return user_error("analyze", "unreadable_file", "Attached file could not be parsed.")
    except Exception:
        return user_error("analyze", "unreadable_file", "Attached file could not be parsed.")
