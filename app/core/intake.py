from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import HTTPException, Request


class IntakeError(Exception):
    def __init__(self, *, error_code: str, message: str, status_code: int = 200) -> None:
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


@dataclass
class OpenAIFileRef:
    id: str
    download_link: str
    name: Optional[str] = None
    mime_type: Optional[str] = None


@dataclass
class InputPayload:
    content: bytes
    filename: str
    mime_type: Optional[str]
    platform_hint: Optional[str] = None
    file_id: Optional[str] = None


FILE_ID_KEYS = ("id", "file_id", "openaiFileId", "openai_file_id")
DOWNLOAD_LINK_KEYS = ("download_link", "downloadLink", "url", "signed_url")


def _extract_file_ref(item: Any) -> Optional[OpenAIFileRef]:
    if isinstance(item, str):
        value = item.strip()
        if not value:
            return None
        return OpenAIFileRef(id=value, download_link="")

    if not isinstance(item, dict):
        return None

    file_id = next((str(item[k]).strip() for k in FILE_ID_KEYS if item.get(k)), "")
    if not file_id:
        return None

    download_link = next((str(item[k]).strip() for k in DOWNLOAD_LINK_KEYS if item.get(k)), "")
    return OpenAIFileRef(
        id=file_id,
        download_link=download_link,
        name=item.get("name") or item.get("filename"),
        mime_type=item.get("mime_type") or item.get("content_type"),
    )


async def download_openai_file_ref(file_ref: OpenAIFileRef) -> Tuple[bytes, str, Optional[str]]:
    if not file_ref.download_link:
        raise IntakeError(
            error_code="file_download_failed",
            message=f"Unable to download file '{file_ref.id}' because no download link was provided.",
        )
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.get(file_ref.download_link)
            response.raise_for_status()
            content_type = response.headers.get("content-type")
            return response.content, (file_ref.name or f"{file_ref.id}.csv"), content_type
    except IntakeError:
        raise
    except Exception:
        raise IntakeError(
            error_code="file_download_failed",
            message=f"Failed to download attached file '{file_ref.id}'.",
        )


async def extract_input_payload(request: Request) -> InputPayload:
    content_type = request.headers.get("content-type", "").lower()

    if "application/json" in content_type:
        try:
            payload = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {str(e)}")

        refs = payload.get("openaiFileIdRefs", [])
        if not isinstance(refs, list) or not refs:
            raise IntakeError(error_code="missing_file", message="No file was attached.")

        normalized_refs = [_extract_file_ref(item) for item in refs]
        usable_ref = next((ref for ref in normalized_refs if ref and ref.id), None)
        if not usable_ref:
            raise IntakeError(error_code="missing_file", message="No file was attached.")

        content, filename, mime_type = await download_openai_file_ref(usable_ref)
        return InputPayload(
            content=content,
            filename=filename,
            mime_type=mime_type,
            platform_hint=(payload.get("platform_hint") or payload.get("platformHint") or None),
            file_id=usable_ref.id,
        )

    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("file")
        if upload is None:
            raise IntakeError(error_code="missing_file", message="No file was attached.")
        content = await upload.read()
        if not content:
            raise IntakeError(error_code="unreadable_file", message="Uploaded file is empty.")
        return InputPayload(
            content=content,
            filename=getattr(upload, "filename", "upload.csv") or "upload.csv",
            mime_type=getattr(upload, "content_type", None),
            platform_hint=form.get("platform_hint") or form.get("platformHint"),
        )

    raise IntakeError(
        error_code="missing_file",
        message="Unsupported content type. Use application/json with openaiFileIdRefs or multipart/form-data with file.",
    )
