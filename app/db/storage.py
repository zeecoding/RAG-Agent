"""Supabase Object Storage client for questionnaires.

All helpers use raw httpx rather than the supabase-py SDK so that the module
stays async-native and has no extra dependency beyond what is already present.
"""

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


def _bucket() -> str:
    return settings.supabase_storage_bucket or "questionnaires"


def _normalize_path(storage_path: str) -> str:
    """Strip the bucket prefix from a storage path if present.

    Accepts both ``questionnaires/org/quest/file.docx`` and
    ``org/quest/file.docx``; always returns the bare relative path.
    """
    bucket = _bucket()
    path = storage_path.strip("/")
    if path.startswith(f"{bucket}/"):
        path = path[len(bucket) + 1:]
    return path


async def upload_questionnaire_file(
    organization_id: str,
    questionnaire_id: str,
    file_bytes: bytes,
    filename: str = "original.docx",
) -> str:
    """Uploads raw original questionnaire bytes to Supabase Storage.

    Storage path format: questionnaires/{organization_id}/{questionnaire_id}/original.docx
    Returns the full storage path (bucket-prefixed).
    """
    if not settings.supabase_url:
        raise HTTPException(
            status_code=500,
            detail="SUPABASE_URL is not configured.",
        )

    clean_filename = "original.docx" if filename.lower().endswith(".docx") else filename
    relative_path = f"{organization_id}/{questionnaire_id}/{clean_filename}"
    bucket = _bucket()
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

    bucket = _bucket()
    path = _normalize_path(storage_path)
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


# Alias used across questionnaires.py
download_file_bytes = download_questionnaire_file


async def upload_completed_docx(storage_path: str, file_bytes: bytes) -> str:
    """Uploads or overwrites (upserts) the mutated completed DOCX into the questionnaires bucket."""
    if not settings.supabase_url:
        raise HTTPException(
            status_code=500,
            detail="SUPABASE_URL is not configured.",
        )

    bucket = _bucket()
    relative_path = _normalize_path(storage_path)
    upload_url = f"{settings.supabase_url.rstrip('/')}/storage/v1/object/{bucket}/{relative_path}"

    headers = _get_headers()
    headers["x-upsert"] = "true"
    headers["Content-Type"] = DOCX_MIME_TYPE

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                upload_url,
                headers=headers,
                content=file_bytes,
            )

        if response.status_code not in (200, 201):
            logger.error(
                "Supabase storage upload (completed DOCX) failed: %d %s",
                response.status_code,
                response.text,
            )
            raise HTTPException(
                status_code=502,
                detail=f"Failed to upload completed DOCX to storage: {response.text}",
            )

        return storage_path

    except httpx.RequestError as exc:
        logger.error("Network error uploading completed DOCX: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Storage service unavailable: {str(exc)}",
        ) from exc


async def get_docx_signed_url(storage_path: str, expires_in: int = 7200) -> str:
    """Generates a temporary signed URL valid for expires_in seconds."""
    if not settings.supabase_url:
        raise HTTPException(
            status_code=500,
            detail="SUPABASE_URL is not configured.",
        )

    bucket = _bucket()
    relative_path = _normalize_path(storage_path)
    sign_url = f"{settings.supabase_url.rstrip('/')}/storage/v1/object/sign/{bucket}/{relative_path}"
    headers = _get_headers()
    headers["Content-Type"] = "application/json"

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                sign_url,
                headers=headers,
                json={"expiresIn": expires_in},
            )

        if response.status_code not in (200, 201):
            logger.error(
                "Supabase signed URL generation failed: %d %s",
                response.status_code,
                response.text,
            )
            raise HTTPException(
                status_code=502,
                detail=f"Failed to generate signed URL: {response.text}",
            )

        data = response.json()
        signed_url = data.get("signedURL") or data.get("signedUrl")
        if not signed_url:
            raise HTTPException(
                status_code=502,
                detail=f"Signed URL missing from Supabase response: {data}",
            )

        if signed_url.startswith("/"):
            signed_url = f"{settings.supabase_url.rstrip('/')}{signed_url}"

        return signed_url

    except httpx.RequestError as exc:
        logger.error("Network error generating signed URL: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Storage service unavailable: {str(exc)}",
        ) from exc


def _kb_bucket() -> str:
    return getattr(settings, "supabase_kb_bucket", None) or "kb_docs"


async def upload_kb_document_file(
    organization_id: str,
    filename: str,
    file_bytes: bytes,
    content_type: str = "application/octet-stream",
) -> str:
    """Uploads raw organization knowledge base document bytes to Supabase Storage in the private kb_docs bucket.

    Storage path format: kb_docs/{organization_id}/{filename}
    Returns the full storage path (e.g. kb_docs/{organization_id}/{filename}).
    """
    if not settings.supabase_url or not settings.supabase_service_role_key:
        logger.warning("Supabase storage not configured; skipping kb_docs upload for %s", filename)
        return ""

    bucket = _kb_bucket()
    relative_path = f"{organization_id}/{filename}"
    upload_url = f"{settings.supabase_url.rstrip('/')}/storage/v1/object/{bucket}/{relative_path}"

    headers = _get_headers()
    headers["x-upsert"] = "true"
    headers["Content-Type"] = content_type

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                upload_url,
                headers=headers,
                content=file_bytes,
            )

        if response.status_code not in (200, 201):
            logger.warning(
                "Supabase kb_docs upload warning (%d): %s",
                response.status_code,
                response.text,
            )
            return ""

        full_storage_path = f"{bucket}/{relative_path}"
        logger.info("Successfully uploaded KB document to %s", full_storage_path)
        return full_storage_path

    except Exception as exc:
        logger.warning("Network or client error during Supabase kb_docs upload: %s", exc)
        return ""


async def create_signed_kb_url(
    organization_id: str,
    filename: str,
    expires_in: int = 3600,
) -> Optional[str]:
    """Generates a time-limited signed URL to download or view a private kb_docs file."""
    if not settings.supabase_url or not settings.supabase_service_role_key:
        return None

    bucket = _kb_bucket()
    relative_path = f"{organization_id}/{filename}"
    sign_url = f"{settings.supabase_url.rstrip('/')}/storage/v1/object/sign/{bucket}/{relative_path}"
    headers = _get_headers()
    headers["Content-Type"] = "application/json"

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                sign_url,
                headers=headers,
                json={"expiresIn": expires_in},
            )

        if response.status_code not in (200, 201):
            return None

        data = response.json()
        signed_url = data.get("signedURL") or data.get("signedUrl")
        if not signed_url:
            return None

        if signed_url.startswith("/"):
            signed_url = f"{settings.supabase_url.rstrip('/')}{signed_url}"

        return signed_url
    except Exception as exc:
        logger.warning("Could not generate signed kb_docs URL: %s", exc)
        return None
