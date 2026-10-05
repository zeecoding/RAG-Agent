"""Supabase Object Storage client for questionnaires."""

import logging
from typing import Optional

import httpx
from fastapi import HTTPException

from app.config import settings

logger = logging.getLogger(__name__)

DOCX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _get_headers() -> dict[str, str]:
    if not settings.supabase_service_role_key:
        raise HTTPException(
            status_code=500,
            detail="SUPABASE_SERVICE_ROLE_KEY is not configured.",
        )
    return {
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
        "apikey": settings.supabase_service_role_key,
    }


async def upload_questionnaire_file(
    organization_id: str,
    questionnaire_id: str,
    file_bytes: bytes,
    filename: str = "original.docx",
) -> str:
    """Uploads raw original questionnaire bytes to Supabase Storage.

    Storage path format: questionnaires/{organization_id}/{questionnaire_id}/original.docx
    Returns the storage path.
    """
    if not settings.supabase_url:
        raise HTTPException(
            status_code=500,
            detail="SUPABASE_URL is not configured.",
        )

    clean_filename = "original.docx" if filename.lower().endswith(".docx") else filename
    relative_path = f"{organization_id}/{questionnaire_id}/{clean_filename}"
    bucket = settings.supabase_storage_bucket or "questionnaires"
    upload_url = f"{settings.supabase_url.rstrip('/')}/storage/v1/object/{bucket}/{relative_path}"

    headers = _get_headers()
    headers["x-upsert"] = "true"
    headers["Content-Type"] = DOCX_MIME_TYPE

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                upload_url,
                headers=headers,
                content=file_bytes,
            )

        if response.status_code not in (200, 201):
            logger.error("Supabase storage upload failed: %d %s", response.status_code, response.text)
            raise HTTPException(
                status_code=502,
                detail=f"Failed to upload questionnaire to storage: {response.text}",
            )

        full_storage_path = f"{bucket}/{relative_path}"
        logger.info("Uploaded questionnaire file to %s", full_storage_path)
        return full_storage_path

    except httpx.RequestError as exc:
        logger.error("Network error during Supabase storage upload: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Storage service unavailable: {str(exc)}",
        ) from exc


async def download_questionnaire_file(storage_path: str) -> bytes:
    """Downloads raw questionnaire bytes from Supabase Storage given a storage path.

    Accepts paths like 'questionnaires/org_123/quest_456/original.docx' or
    'org_123/quest_456/original.docx'.
    """
    if not settings.supabase_url:
        raise HTTPException(
            status_code=500,
            detail="SUPABASE_URL is not configured.",
        )

    bucket = settings.supabase_storage_bucket or "questionnaires"

    # Normalize path so bucket name is not duplicated
    path = storage_path.strip("/")
    if path.startswith(f"{bucket}/"):
        path = path[len(bucket) + 1 :]

    download_url = f"{settings.supabase_url.rstrip('/')}/storage/v1/object/{bucket}/{path}"
    headers = _get_headers()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(download_url, headers=headers)

        if response.status_code == 404:
            logger.error("Questionnaire template not found in storage: %s", download_url)
            raise HTTPException(
                status_code=404,
                detail=f"Questionnaire file not found in storage: {storage_path}",
            )
        if response.status_code != 200:
            logger.error("Supabase storage download failed: %d %s", response.status_code, response.text)
            raise HTTPException(
                status_code=502,
                detail=f"Failed to download questionnaire from storage: {response.text}",
            )

        return response.content

    except httpx.RequestError as exc:
        logger.error("Network error during Supabase storage download: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Storage service unavailable: {str(exc)}",
        ) from exc
