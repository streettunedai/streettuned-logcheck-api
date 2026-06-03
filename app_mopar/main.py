from __future__ import annotations

import csv
import io
import os
import re
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse

import httpx
from fastapi import Body, FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(
    title="streettuned-mopar-logcheck-api",
    version="0.1.0",
    description="Standalone Mopar-only HP Tuners log validation and analysis API.",
)


class ActionLogRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    openaiFileIdRefs: Any = None
    file: Any = None
    files: Any = None
    file_path: str | None = None
    filePath: str | None = None
    url: str | None = None
    download_link: str | None = None
    platform_hint: str | None = Field(default="mopar")
    question: str | None = None
    vehicle_context: dict[str, Any] | None = None


class CapabilityFlags(BaseModel):
    basic_engine_review: bool = False
    idle_review: bool = False
    cruise_review: bool = False
    shift_review: bool = False
    knock_review: bool = False
    wot_fueling_review: bool = False
    transmission_review: bool = False


class PlatformGuess(BaseModel):
    platform: Literal["mopar"] | None = None
    confidence: Literal["confirmed", "hinted", "uncertain", "incompatible"] = "uncertain"
    reasons: list[str] = Field(default_factory=list)


class MoparResponse(BaseModel):
    ok: bool
    stage: Literal["validate", "analyze"]
    status: Literal["PASS", "PARTIAL_PASS", "FAIL"] = "FAIL"
    error_code: str | None = None
    message: str
    platform: Literal["mopar"] | None = "mopar"
    platform_guess: PlatformGuess | None = None
    rows: int | None = None
    columns: list[str] = Field(default_factory=list)
    detected_channels: dict[str, str] = Field(default_factory=dict)
    readable_scope: CapabilityFlags = Field(default_factory=CapabilityFlags)
    confirmed_readable_data: list[str] = Field(default_factory=list)
    missing_or_unreliable_data: list[str] = Field(default_factory=list)
    safe_conclusions: list[str] = Field(default_factory=list)
    unsupported_conclusions: list[str] = Field(default_factory=list)
    response: str | None = None
    debug: dict[str, Any] | None = None


class IntakeError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class ParseError(Exception):
    pass


ID_KEYS = ("id", "file_id", "fileId", "openaiFileId", "openai_file_id")
URL_KEYS = ("download_link", "downloadLink", "signed_url", "signedUrl", "file_url", "fileUrl", "url", "href")
PATH_KEYS = ("file_path", "filePath", "path")

ALIASES: dict[str, tuple[str, ...]] = {
    "time": (r"\btime\b", r"timestamp"),
    "rpm": (r"\brpm\b", r"\bengine\s+rpm\b", r"\bengine\s+speed\b"),
    "vehicle_speed": (r"\bvehicle\s+speed\b", r"\bvss\b", r"\bmph\b"),
    "map": (r"\bmap\b", r"\bmanifold\s+absolute\s+pressure\b", r"\bintake\s+manifold\s+pressure\b"),
    "tps": (r"\btps\b", r"\bthrottle\s+position\b", r"\bthrottle\s+angle\b", r"\bactual\s+throttle\b"),
    "pedal": (r"\baccelerator\s+pedal\b", r"\bpedal\s+position\b", r"\bapp\b"),
    "engine_load": (r"\bengine\s+load\b", r"\bcalculated\s+load\b", r"\bload\b"),
    "iat": (r"\biat\b", r"\bintake\s+air\s+temp"),
    "ect": (r"\bect\b", r"\bcoolant\s+temp", r"\bengine\s+coolant\s+temp"),
    "spark_advance": (r"\bspark\s+advance\b", r"\btiming\s+advance\b", r"\bignition\s+advance\b"),
    "knock_retard": (r"\bknock\s+retard\b", r"\bkr\b", r"\bspark\s+retard\b"),
    "stft_b1": (r"\bstft\s*(?:b|bank)?\s*1\b", r"\bshort\s+term\s+fuel\s+trim\s*(?:b|bank)?\s*1\b"),
    "stft_b2": (r"\bstft\s*(?:b|bank)?\s*2\b", r"\bshort\s+term\s+fuel\s+trim\s*(?:b|bank)?\s*2\b"),
    "ltft_b1": (r"\bltft\s*(?:b|bank)?\s*1\b", r"\blong\s+term\s+fuel\s+trim\s*(?:b|bank)?\s*1\b"),
    "ltft_b2": (r"\bltft\s*(?:b|bank)?\s*2\b", r"\blong\s+term\s+fuel\s+trim\s*(?:b|bank)?\s*2\b"),
    "commanded_eq": (r"\bcommanded\s+eq\b", r"\beq\s+ratio\b", r"\bequivalence\s+ratio\b"),
    "commanded_lambda": (r"\bcommanded\s+lambda\b", r"\btarget\s+lambda\b"),
    "wideband_afr": (r"\bwideband\s+afr\b", r"\bwb\s+afr\b", r"\bmeasured\s+afr\b"),
    "wideband_lambda": (r"\bwideband\s+lambda\b", r"\bwb\s+lambda\b", r"\bmeasured\s+lambda\b"),
    "current_gear": (r"\bcurrent\s+gear\b", r"\btrans\s+gear\b", r"\bgear\b"),
    "input_speed": (r"\binput\s+speed\b", r"\bturbine\s+speed\b"),
    "output_speed": (r"\boutput\s+speed\b", r"\boss\b"),
    "slip_speed": (r"\bslip\s+speed\b", r"\bconverter\s+slip\b", r"\btcc\s+slip\b"),
    "desired_torque": (r"\bdesired\s+torque\b", r"\bdriver\s+desired\s+torque\b", r"\brequested\s+torque\b"),
    "actual_torque": (r"\bactual\s+torque\b", r"\bdelivered\s+torque\b", r"\bestimated\s+torque\b"),
}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    invalid_json = any(error.get("type") == "json_invalid" for error in exc.errors())
    body = fail(
        "validate",
        "invalid_json" if invalid_json else "missing_file",
        "Invalid JSON request body." if invalid_json else "No file was attached.",
        debug={"validation_errors": exc.errors()},
    )
    return JSONResponse(status_code=400 if invalid_json else 200, content=body.model_dump(mode="json"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "streettuned-mopar-logcheck-api", "version": "0.1.0"}


@app.post("/validate", response_model=MoparResponse, operation_id="validateMoparLog", openapi_extra={"x-openai-isConsequential": False})
async def validate(payload: ActionLogRequest | None = Body(default=None)) -> MoparResponse:
    return await handle("validate", payload or ActionLogRequest())


@app.post("/analyze", response_model=MoparResponse, operation_id="analyzeMoparLog", openapi_extra={"x-openai-isConsequential": False})
async def analyze(payload: ActionLogRequest | None = Body(default=None)) -> MoparResponse:
    return await handle("analyze", payload or ActionLogRequest())


@app.post("/upload/validate", response_model=MoparResponse)
async def upload_validate(file: UploadFile = File(...), platform_hint: str | None = Form(default="mopar")) -> MoparResponse:
    return review(await file.read(), file.filename or "upload.csv", platform_hint, "validate")


@app.post("/upload/analyze", response_model=MoparResponse)
async def upload_analyze(file: UploadFile = File(...), platform_hint: str | None = Form(default="mopar")) -> MoparResponse:
    return review(await file.read(), file.filename or "upload.csv", platform_hint, "analyze")


async def handle(stage: Literal["validate", "analyze"], payload: ActionLogRequest) -> MoparResponse:
    try:
        content, name = await resolve_upload(payload)
    except IntakeError as exc:
        return fail(stage, exc.code, exc.message)
    return review(content, name, payload.platform_hint or "mopar", stage)


async def resolve_upload(payload: ActionLogRequest) -> tuple[bytes, str]:
    candidates: list[tuple[str, str | None, str]] = []
    add_candidates(candidates, payload.file_path)
    add_candidates(candidates, payload.filePath)
    add_candidates(candidates, payload.download_link)
    add_candidates(candidates, payload.url)
    add_candidates(candidates, payload.openaiFileIdRefs)
    add_candidates(candidates, payload.file)
    add_candidates(candidates, payload.files)
    for key, value in (payload.model_extra or {}).items():
        if "file" in key.lower() or key in {"url", "path", "download_link"}:
            add_candidates(candidates, value)
    if not candidates:
        raise IntakeError("missing_file", "No file was attached.")

    last_error: IntakeError | None = None
    for value, name, kind in candidates:
        try:
            if kind == "url":
                return await download_url(value, name)
            if kind == "file_id":
                return await download_file_id(value, name)
            return read_local(value, name)
        except IntakeError as exc:
            last_error = exc
    raise last_error or IntakeError("file_download_failed", "Unable to read uploaded file.")


def add_candidates(candidates: list[tuple[str, str | None, str]], value: Any, name: str | None = None) -> None:
    if value is None:
        return
    if isinstance(value, BaseModel):
        value = value.model_dump(exclude_none=True)
    if isinstance(value, list):
        for item in value:
            add_candidates(candidates, item, name)
        return
    if isinstance(value, dict):
        file_name = value.get("name") if isinstance(value.get("name"), str) else name
        for key in URL_KEYS:
            if isinstance(value.get(key), str) and value[key].strip():
                candidates.append((value[key].strip(), file_name, "url"))
                return
        for key in PATH_KEYS:
            if isinstance(value.get(key), str) and value[key].strip():
                candidates.append((value[key].strip(), file_name, "path"))
                return
        for key in ID_KEYS:
            if isinstance(value.get(key), str) and value[key].strip():
                candidates.append((value[key].strip(), file_name, "file_id"))
                return
        return
    if isinstance(value, str) and value.strip():
        text = value.strip()
        kind = "url" if urlparse(text).scheme.lower() in {"http", "https"} else "file_id" if text.startswith("file-") else "path"
        candidates.append((text, name, kind))


def read_local(path_text: str, name: str | None) -> tuple[bytes, str]:
    parsed = urlparse(path_text)
    if parsed.scheme.lower() == "file":
        raw = unquote(parsed.path)
        if os.name == "nt" and raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
            raw = raw[1:]
        path = Path(raw).expanduser()
    else:
        path = Path(path_text).expanduser()
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise IntakeError("file_download_failed", f"Could not read uploaded file: {exc}") from exc
    if not data:
        raise IntakeError("unreadable_file", "The attached file was empty.")
    return data, name or path.name or "upload.csv"


async def download_file_id(file_id: str, name: str | None) -> tuple[bytes, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise IntakeError("file_download_failed", "The uploaded file reference did not include a download link, and OPENAI_API_KEY is not configured.")
    return await download_url(f"https://api.openai.com/v1/files/{file_id}/content", name or f"{file_id}.csv", {"Authorization": f"Bearer {api_key}"})


async def download_url(url: str, name: str | None, headers: dict[str, str] | None = None) -> tuple[bytes, str]:
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True, headers=headers) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise IntakeError("file_download_failed", f"Could not download uploaded file: {exc}") from exc
    if not response.content:
        raise IntakeError("unreadable_file", "The downloaded file was empty.")
    return response.content, name or Path(unquote(urlparse(url).path)).name or "upload.csv"


def review(content: bytes, source_name: str, platform_hint: str | None, stage: Literal["validate", "analyze"]) -> MoparResponse:
    try:
        rows, headers, meta = parse_log(content)
    except ParseError:
        return fail(stage, "unreadable_file", "The attached file could not be parsed as a readable log.")

    channels = detect_channels(headers, rows)
    guess = guess_platform(source_name, headers, channels, platform_hint)
    segments = segment(rows, channels)
    scope = capabilities(channels, segments)
    detected = {key: data["raw_name"] for key, data in channels.items()}
    confirmed = [f"{len(rows)} rows parsed from {len(headers)} columns."] + [f"{key} confirmed from '{name}'." for key, name in detected.items()]
    missing = missing_list(channels, scope)
    safe = safe_list(channels, scope, segments, guess)
    unsupported = unsupported_list(scope, segments, guess)

    summary = summary_text(rows, headers, channels, scope, segments, guess)
    if guess.confidence in {"uncertain", "incompatible"}:
        response_text = format_sections(summary, "Readable data exists, but Mopar platform coverage is not confirmed. I cannot confirm this with current data.", "No Mopar tuning changes are supported until platform coverage is confirmed.", "Send platform_hint='mopar' with year, model, engine, PCM, and transmission, or upload a log with Mopar-specific channels.")
        return fail(stage, "platform_uncertain", "Readable data exists, but Mopar platform coverage is not confirmed.", guess, len(rows), headers, detected, scope, confirmed, missing, safe, unsupported, response_text, meta)

    if not scope.basic_engine_review:
        response_text = format_sections(summary, "The file has readable rows, but not enough confirmed Mopar engine channels for review.", "No tuning changes are supported from this log.", "Log Engine RPM plus MAP, throttle position, pedal position, ECT, IAT, spark advance, and knock retard.")
        return fail(stage, "insufficient_channels", "Readable data exists, but not enough Mopar engine channels were confirmed.", guess, len(rows), headers, detected, scope, confirmed, missing, safe, unsupported, response_text, meta)

    if stage == "validate":
        root = "Validation only. The log is readable as Mopar-compatible data, but no tuning diagnosis was calculated."
        changes = "Validation only. No edits calculated."
    else:
        root = root_cause(channels, segments, scope)
        changes = changes_required(channels, segments, scope)
    verify = verify_next(channels, scope, segments)
    response_text = format_sections(summary, root, changes, verify)
    status = "PASS" if scope.knock_review or scope.wot_fueling_review or scope.transmission_review or scope.cruise_review else "PARTIAL_PASS"
    return MoparResponse(ok=True, stage=stage, status=status, message="Mopar log data is readable. Conclusions are limited to confirmed channels.", platform="mopar", platform_guess=guess, rows=len(rows), columns=headers, detected_channels=detected, readable_scope=scope, confirmed_readable_data=confirmed, missing_or_unreliable_data=missing, safe_conclusions=safe, unsupported_conclusions=unsupported, response=response_text, debug={"parser": meta, "segments": {k: len(v) for k, v in segments.items()}})


def fail(stage: Literal["validate", "analyze"], code: str, message: str, guess: PlatformGuess | None = None, rows: int | None = None, columns: list[str] | None = None, detected: dict[str, str] | None = None, scope: CapabilityFlags | None = None, confirmed: list[str] | None = None, missing: list[str] | None = None, safe: list[str] | None = None, unsupported: list[str] | None = None, response: str | None = None, debug: dict[str, Any] | None = None) -> MoparResponse:
    return MoparResponse(ok=False, stage=stage, status="FAIL", error_code=code, message=message, platform=None if code == "missing_file" else "mopar", platform_guess=guess, rows=rows, columns=columns or [], detected_channels=detected or {}, readable_scope=scope or CapabilityFlags(), confirmed_readable_data=confirmed or [], missing_or_unreliable_data=missing or [], safe_conclusions=safe or [], unsupported_conclusions=unsupported or [], response=response, debug=debug)


def parse_log(content: bytes) -> tuple[list[dict[str, str]], list[str], dict[str, Any]]:
    if not content or not content.strip():
        raise ParseError("empty")
    text = decode(content).replace("\x00", "")
    best = None
    for delim in delimiters(text):
        parsed_rows = list(csv.reader(io.StringIO(text), delimiter=delim))
        header_index, score = header_row(parsed_rows)
        if header_index is None:
            continue
        if best is None or score > best[0]:
            best = (score, delim, parsed_rows, header_index)
    if best is None:
        raise ParseError("no header")
    _, delim, parsed_rows, header_index = best
    headers = unique([cell.strip() or f"column_{i}" for i, cell in enumerate(parsed_rows[header_index], 1)])
    data_rows: list[dict[str, str]] = []
    uneven = 0
    for row in parsed_rows[header_index + 1:]:
        if not any(cell.strip() for cell in row):
            continue
        if len(row) != len(headers):
            uneven += 1
        padded = list(row[:len(headers)]) + [""] * max(0, len(headers) - len(row))
        data_rows.append({h: v.strip() for h, v in zip(headers, padded)})
    if not data_rows:
        raise ParseError("no rows")
    return data_rows, headers, {"delimiter": delim, "metadata_rows": header_index, "uneven_rows": uneven}


def decode(content: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            pass
    return content.decode("latin-1", errors="replace")


def delimiters(text: str) -> list[str]:
    sample = "\n".join(text.splitlines()[:20])
    out: list[str] = []
    try:
        out.append(csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter)
    except csv.Error:
        pass
    return out + [d for d in [",", "\t", ";", "|"] if d not in out]


def header_row(rows: list[list[str]]) -> tuple[int | None, int]:
    words = ("rpm", "engine speed", "map", "manifold", "throttle", "pedal", "spark", "knock", "lambda", "afr", "gear", "torque")
    best_i, best_score = None, -999
    for i, row in enumerate(rows[:40]):
        cells = [c.strip() for c in row if c.strip()]
        if len(cells) < 2:
            continue
        joined = " ".join(cells).lower()
        hits = sum(1 for w in words if w in joined)
        numeric = sum(1 for c in cells if number(c) is not None)
        score = len(cells) + hits * 5 - numeric * 2
        if score > best_score:
            best_i, best_score = i, score
    return best_i, best_score


def unique(headers: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    out: list[str] = []
    for h in headers:
        counts[h] = counts.get(h, 0) + 1
        out.append(h if counts[h] == 1 else f"{h}_{counts[h]}")
    return out


def detect_channels(headers: list[str], rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for key, patterns in ALIASES.items():
        for header in headers:
            norm = clean(header)
            if key == "rpm" and "vehicle speed" in norm:
                continue
            if key == "vehicle_speed" and ("engine speed" in norm or "rpm" in norm):
                continue
            if any(re.search(pattern, norm) for pattern in patterns):
                found[key] = {"raw_name": split_unit(header)[0], "unit": split_unit(header)[1], "values": [number(row.get(header)) for row in rows]}
                break
    return found


def clean(value: str) -> str:
    value = value.lower().replace("_", " ")
    value = re.sub(r"[^a-z0-9%./+\- ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def split_unit(header: str) -> tuple[str, str | None]:
    match = re.search(r"^(.*?)[\s]*(?:\[([^\]]+)\]|\(([^)]+)\))\s*$", header.strip())
    if not match:
        return header.strip(), None
    return match.group(1).strip() or header.strip(), (match.group(2) or match.group(3) or None)


def number(value: Any) -> float | None:
    text = "" if value is None else str(value).strip().replace(",", "")
    if not text or text.lower() in {"na", "n/a", "nan", "null", "none", "--"}:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def guess_platform(source: str, headers: list[str], channels: dict[str, Any], hint: str | None) -> PlatformGuess:
    text = " ".join([source, *headers]).lower()
    score, reasons = 0, []
    if (hint or "").lower() in {"mopar", "dodge", "chrysler", "jeep", "hemi", "ngc", "gpec"}:
        score += 4; reasons.append(f"platform_hint='{hint}' supplied.")
    for marker in ("mopar", "dodge", "chrysler", "jeep", "hemi", "ngc", "gpec", "srt8", "5.7", "6.1", "nag1", "545rfe"):
        if marker in text:
            score += 2; reasons.append(f"Found Mopar marker '{marker}'.")
    for marker in ("spark retard", "throttle angle", "current gear", "desired torque", "actual torque"):
        if marker in text:
            score += 1; reasons.append(f"Found Mopar-compatible channel marker '{marker}'.")
    bad = [m for m in ("gen4 ls", "gen 4 ls", "gen5 lt", "gen 5 lt", "ecoboost", "duramax", "powerstroke", "cummins") if m in text]
    if bad and score < 4:
        return PlatformGuess(platform=None, confidence="incompatible", reasons=[f"Found non-Mopar marker '{m}'." for m in bad])
    if score >= 6:
        return PlatformGuess(platform="mopar", confidence="confirmed", reasons=dedupe(reasons))
    if score >= 4:
        return PlatformGuess(platform="mopar", confidence="hinted", reasons=dedupe(reasons))
    return PlatformGuess(platform=None, confidence="uncertain", reasons=["No Mopar-specific markers found."])


def segment(rows: list[dict[str, str]], ch: dict[str, dict[str, Any]]) -> dict[str, list[int]]:
    seg = {"running": [], "idle": [], "cruise": [], "wot": [], "shift": []}
    last_gear = None
    for i in range(len(rows)):
        rpm, speed, tps, pedal, mapv, gear = val(ch, "rpm", i), val(ch, "vehicle_speed", i), val(ch, "tps", i), val(ch, "pedal", i), map_kpa(ch.get("map"), i), val(ch, "current_gear", i)
        if rpm is None or rpm <= 400:
            continue
        seg["running"].append(i)
        if (speed is None or speed < 5) and (tps is None or tps < 8):
            seg["idle"].append(i)
        if speed is not None and speed > 25 and tps is not None and 5 <= tps <= 40:
            seg["cruise"].append(i)
        if (tps is not None and tps >= 70) or (pedal is not None and pedal >= 70) or (mapv is not None and mapv >= 85):
            seg["wot"].append(i)
        if gear is not None and last_gear is not None and gear != last_gear:
            seg["shift"].append(i)
        if gear is not None:
            last_gear = gear
    return seg


def capabilities(ch: dict[str, Any], seg: dict[str, list[int]]) -> CapabilityFlags:
    has_load = "map" in ch or "engine_load" in ch
    has_throttle = "tps" in ch or "pedal" in ch
    has_wb = "wideband_afr" in ch or "wideband_lambda" in ch
    has_cmd = "commanded_eq" in ch or "commanded_lambda" in ch
    has_trans = any(k in ch for k in ("current_gear", "input_speed", "output_speed", "slip_speed"))
    return CapabilityFlags(
        basic_engine_review="rpm" in ch and (has_load or has_throttle),
        idle_review="rpm" in ch and bool(seg["idle"]) and any(k in ch for k in ("ect", "iat", "stft_b1", "ltft_b1", "map", "tps")),
        cruise_review="rpm" in ch and bool(seg["cruise"]) and has_load,
        shift_review=bool(seg["shift"]) and has_trans,
        knock_review="knock_retard" in ch and "spark_advance" in ch and has_load,
        wot_fueling_review=bool(seg["wot"]) and has_wb and has_cmd and has_load,
        transmission_review=has_trans and ("rpm" in ch or "vehicle_speed" in ch),
    )


def summary_text(rows: list[dict[str, str]], headers: list[str], ch: dict[str, Any], scope: CapabilityFlags, seg: dict[str, list[int]], guess: PlatformGuess) -> str:
    scopes = [k for k, v in scope.model_dump().items() if v]
    names = ", ".join(f"{k}: {v['raw_name']}" for k, v in ch.items()) or "none confirmed"
    return f"Rows parsed: {len(rows)}. Platform read: {guess.confidence} Mopar. Readable scope: {', '.join(scopes) if scopes else 'none'}. Operating regions: running rows {len(seg['running'])}, idle rows {len(seg['idle'])}, cruise rows {len(seg['cruise'])}, WOT/load rows {len(seg['wot'])}, shift rows {len(seg['shift'])}. Confirmed channels: {names}."


def root_cause(ch: dict[str, Any], seg: dict[str, list[int]], scope: CapabilityFlags) -> str:
    trims = trim_findings(ch, seg)
    knocks = knock_findings(ch, seg)
    if trims:
        return " ".join(trims)
    if knocks:
        return " ".join(knocks)
    return "No single root cause is confirmed by the readable data. The current log supports limited Mopar engine review only."


def changes_required(ch: dict[str, Any], seg: dict[str, list[int]], scope: CapabilityFlags) -> str:
    lines = []
    if not scope.wot_fueling_review:
        lines.append("Do not make a WOT change yet. WOT fueling changes require commanded lambda/EQ plus measured wideband AFR/lambda.")
    if "knock_retard" in ch and not scope.knock_review:
        lines.append("Do not add timing yet. Knock data is not supported by enough load/spark context.")
    if trim_findings(ch, seg):
        lines.append("Closed-loop fuel or airflow direction may need review in the matching idle/cruise area, but no exact table or percentage change is supported by this data alone.")
    return " ".join(lines) if lines else "No numeric fueling, spark, torque, airflow, or transmission changes are supported by this log."


def verify_next(ch: dict[str, Any], scope: CapabilityFlags, seg: dict[str, list[int]]) -> str:
    need = []
    for key, label in (("rpm", "Engine RPM"), ("map", "MAP"), ("tps", "Throttle Position or Throttle Angle"), ("pedal", "Accelerator Pedal Position"), ("spark_advance", "Spark Advance"), ("knock_retard", "Knock Retard or Spark Retard")):
        if key not in ch:
            need.append(label)
    if not scope.wot_fueling_review:
        need.append("Commanded Lambda/EQ and measured wideband AFR/Lambda for WOT")
    if not scope.transmission_review:
        need.append("Current Gear, Input Speed, Output Speed, and Slip Speed for transmission review")
    text = "Log " + ", ".join(dedupe(need)) + "."
    if not seg["cruise"]:
        text += " Add a steady cruise section."
    if not seg["idle"]:
        text += " Add warm idle data."
    return text


def missing_list(ch: dict[str, Any], scope: CapabilityFlags) -> list[str]:
    out = []
    for key, label in (("rpm", "Engine RPM"), ("map", "MAP / Manifold Absolute Pressure"), ("tps", "Throttle Position / Throttle Angle"), ("pedal", "Accelerator Pedal Position"), ("spark_advance", "Spark Advance / Timing Advance"), ("knock_retard", "Knock Retard / Spark Retard"), ("current_gear", "Current Gear"), ("input_speed", "Input Speed"), ("output_speed", "Output Speed"), ("slip_speed", "Slip Speed"), ("desired_torque", "Desired Torque"), ("actual_torque", "Actual Torque")):
        if key not in ch:
            out.append(label)
    if not ("commanded_lambda" in ch or "commanded_eq" in ch):
        out.append("Commanded Lambda or EQ Ratio")
    if not ("wideband_lambda" in ch or "wideband_afr" in ch):
        out.append("Wideband AFR or Lambda")
    if not scope.wot_fueling_review:
        out.append("WOT fueling review not fully supported.")
    if not scope.transmission_review:
        out.append("Transmission review not fully supported.")
    return dedupe(out)


def safe_list(ch: dict[str, Any], scope: CapabilityFlags, seg: dict[str, list[int]], guess: PlatformGuess) -> list[str]:
    out = []
    if guess.confidence in {"confirmed", "hinted"}: out.append("Mopar-compatible review is allowed within confirmed channels.")
    for attr, label in (("basic_engine_review", "Basic engine review is supported."), ("idle_review", "Idle review is supported."), ("cruise_review", "Cruise review is supported."), ("knock_review", "Spark/knock review is supported."), ("wot_fueling_review", "WOT fueling review is supported because commanded fueling and measured wideband are present."), ("transmission_review", "Transmission review is supported within available gear/speed/slip data.")):
        if getattr(scope, attr): out.append(label)
    return dedupe(out + trim_findings(ch, seg) + knock_findings(ch, seg))


def unsupported_list(scope: CapabilityFlags, seg: dict[str, list[int]], guess: PlatformGuess) -> list[str]:
    out = []
    if guess.confidence == "uncertain": out.append("Mopar platform coverage is unsupported until vehicle/platform evidence is confirmed.")
    if not scope.wot_fueling_review: out.append("Numeric WOT fueling changes are unsupported. I cannot confirm this with current data.")
    if not scope.knock_review: out.append("Spark/timing changes are unsupported without confirmed spark, knock, and load context.")
    if not scope.transmission_review: out.append("Transmission changes are unsupported without gear, input/output speed, and slip context.")
    if not seg["cruise"]: out.append("Cruise airflow/fuel-trim conclusions are limited because no steady cruise section was confirmed.")
    return dedupe(out)


def trim_findings(ch: dict[str, Any], seg: dict[str, list[int]]) -> list[str]:
    idx = seg["idle"] + seg["cruise"]
    if not idx: return []
    banks = []
    for bank, s_key, l_key in (("bank 1", "stft_b1", "ltft_b1"), ("bank 2", "stft_b2", "ltft_b2")):
        st, lt = avg(ch, s_key, idx), avg(ch, l_key, idx)
        if st is not None or lt is not None:
            banks.append((bank, (st or 0) + (lt or 0)))
    out = []
    for bank, total in banks:
        if total >= 10: out.append(f"Closed-loop trims show a lean correction trend on {bank} around {total:.1f}% in readable idle/cruise data.")
        elif total <= -10: out.append(f"Closed-loop trims show a rich correction trend on {bank} around {total:.1f}% in readable idle/cruise data.")
    if len(banks) == 2 and abs(banks[0][1] - banks[1][1]) >= 8:
        out.append("Bank-to-bank trim difference is large enough to check for bank-specific sensor, exhaust, vacuum, or mechanical issues before tuning.")
    return out


def knock_findings(ch: dict[str, Any], seg: dict[str, list[int]]) -> list[str]:
    if "knock_retard" not in ch: return []
    idx = seg["running"] or list(range(len(ch["knock_retard"]["values"])))
    vals = [val(ch, "knock_retard", i) for i in idx]
    vals = [v for v in vals if v is not None]
    if not vals: return []
    peak = max(vals)
    if peak >= 3: return [f"Knock retard is present up to {peak:.1f} degrees. Do not add timing yet."]
    if peak > 0: return [f"Small knock retard is present up to {peak:.1f} degrees; verify repeatability before changing spark."]
    return ["No knock retard was visible in the readable running rows."]


def format_sections(data: str, root: str, changes: str, verify: str) -> str:
    return f"[DATA SUMMARY]\n{data}\n\n[ROOT CAUSE]\n{root}\n\n[CHANGES REQUIRED]\n{changes}\n\n[VERIFY NEXT]\n{verify}"


def val(ch: dict[str, Any], key: str, i: int) -> float | None:
    if key not in ch or i >= len(ch[key]["values"]): return None
    return ch[key]["values"][i]


def avg(ch: dict[str, Any], key: str, idx: list[int]) -> float | None:
    vals = [val(ch, key, i) for i in idx]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def map_kpa(channel: dict[str, Any] | None, i: int) -> float | None:
    if not channel or i >= len(channel["values"]): return None
    value = channel["values"][i]
    if value is None: return None
    unit = (channel.get("unit") or "").lower()
    if "psi" in unit: return value * 6.89476
    if "bar" in unit and "kpa" not in unit: return value * 100
    return value


def dedupe(items: list[str]) -> list[str]:
    out, seen = [], set()
    for item in items:
        if item not in seen:
            seen.add(item); out.append(item)
    return out
