from __future__ import annotations

from typing import List, Optional, Tuple

import httpx
from fastapi import HTTPException, Request
from pydantic import BaseModel, Field


class OpenAIFileRef(BaseModel):
    id: str
    download_link: str
    name: Optional[str] = None
    mime_type: Optional[str] = None


class OpenAIFileRefsRequest(BaseModel):
    openaiFileIdRefs: List[OpenAIFileRef] = Field(default_factory=list)


async def download_openai_file_ref(file_ref: OpenAIFileRef) -> Tuple[bytes, str, Optional[str]]:
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.get(file_ref.download_link)
            response.raise_for_status()
            content_type = response.headers.get("content-type")
            return response.content, (file_ref.name or f"{file_ref.id}.csv"), content_type
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download file ref {file_ref.id}: {str(e)}")


async def extract_input_payload(request: Request) -> Tuple[bytes, str, Optional[str]]:
    content_type = request.headers.get("content-type", "").lower()

    if "application/json" in content_type:
        try:
            payload = OpenAIFileRefsRequest.model_validate(await request.json())
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {str(e)}")

        if not payload.openaiFileIdRefs:
            raise HTTPException(status_code=400, detail="No openaiFileIdRefs provided.")

        file_ref = payload.openaiFileIdRefs[0]
        return await download_openai_file_ref(file_ref)

    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("file")
        if upload is None:
            raise HTTPException(status_code=400, detail="Multipart request missing file field.")
        content = await upload.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        return content, getattr(upload, "filename", "upload.csv") or "upload.csv", getattr(upload, "content_type", None)

    raise HTTPException(
        status_code=415,
        detail="Unsupported content type. Use application/json with openaiFileIdRefs or multipart/form-data with file.",
    )
