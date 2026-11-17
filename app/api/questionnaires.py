"""Questionnaire ingestion, AI answer generation, and in-place DOCX export routes."""

import asyncio
import io
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import jwt
from cuid2 import cuid_wrapper
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from pydantic import BaseModel, Field

from app.agent.graph import run_agent
from app.auth.dependencies import AuthenticatedUser, require_responder_or_admin, require_any_org_member
from app.config import settings
from app.db.pool import get_pool
from app.db.storage import (
    DOCX_MIME_TYPE,
    download_file_bytes,
    get_docx_signed_url,
    upload_completed_docx,
    upload_questionnaire_file,
)
from app.exporter.docx_engine import (
    extract_csv_questions,
    extract_docx_questions,
    extract_xlsx_questions,
    locate_and_sync_docx_anchors,
    write_csv_answers_in_place,
    write_docx_answers_in_place,
    write_single_answer_in_place,
    write_xlsx_answers_in_place,
)

XLSX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
CSV_MIME_TYPE = "text/csv; charset=utf-8"


logger = logging.getLogger(__name__)

cuid_generator = cuid_wrapper()

_question_answering_semaphore = asyncio.Semaphore(1)

router = APIRouter(prefix="/questionnaires", tags=["Questionnaires"])

DEFAULT_ONLYOFFICE_SECRET = "tC7kCr4RfQc2h2imtHeYy2W1iYERIPz6"


def _get_onlyoffice_jwt_secret() -> str:
    """Helper to retrieve the JWT secret for ONLYOFFICE signing."""
    return (
        getattr(settings, "onlyoffice_jwt_secret", None)
        or os.getenv("ONLYOFFICE_JWT_SECRET")
        or DEFAULT_ONLYOFFICE_SECRET
    )


# ── Response Schemas ──────────────────────────────────────────────────────────

class ExtractedQuestionResponse(BaseModel):
    id: str
    question: str
    coordinates: dict[str, Any]
    status: str
    draft_answer: Optional[str] = None
    confidence: Optional[int] = None


class QuestionnaireSummaryResponse(BaseModel):
    questionnaire_id: str
    title: str
    filename: str
    storage_path: str
    status: str
    question_count: int
    questions: list[ExtractedQuestionResponse] = []


class AnswerAllResponse(BaseModel):
    questionnaire_id: str
    total_questions: int
    answered_count: int
    status: str


# ── Internal Helper: Build & Persist Completed DOCX ─────────────────────────

async def _generate_and_persist_completed_docx(
    questionnaire_id: str,
    org_id: str,
) -> tuple[str, str]:
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=500, detail="Database pool unavailable.")

    async with pool.acquire() as conn:
        q_row = await conn.fetchrow(
            """
            SELECT q.*, p.organization_id as proj_org_id
            FROM questionnaires q
            LEFT JOIN projects p ON q.project_id = p.id
            WHERE q.id = $1
            """,
            questionnaire_id,
        )
        if not q_row:
            raise HTTPException(status_code=404, detail="Questionnaire not found.")

        q_dict = dict(q_row)
        original_path = q_dict.get("file_url") or q_dict.get("storage_path")
        if not original_path:
            raise HTTPException(
                status_code=404,
                detail="Questionnaire template file path not recorded.",
            )

        questions = await conn.fetch(
            """
            SELECT id, question, coordinates, draft_answer, status
            FROM questions
            WHERE questionnaire_id = $1
            ORDER BY created_at ASC
            """,
            questionnaire_id,
        )

        completed_path = q_dict.get("completed_file_path")
        has_manual = bool(q_dict.get("has_manual_edits"))

    # If completed file already exists with manual edits, preserve it as base; otherwise use original template
    base_path = completed_path if (completed_path and has_manual) else original_path
    try:
        base_bytes = await download_file_bytes(base_path)
    except Exception:
        base_bytes = await download_file_bytes(original_path)

    answered_items: list[dict] = []
    for q in questions:
        q_d = dict(q)
        coords = q_d.get("coordinates")
        if isinstance(coords, str):
            try:
                coords = json.loads(coords)
            except Exception:
                coords = {}
        answered_items.append({
            "question": q_d.get("question", ""),
            "coordinates": coords or {},
            "answer": q_d.get("draft_answer") or "",
        })

    file_name = q_dict.get("file_name") or "questionnaire.docx"
    file_ext = (q_dict.get("file_type") or (file_name.rsplit(".", 1)[-1] if "." in file_name else "docx")).lower()

    if file_ext in ("xlsx", "xls"):
        completed_bytes = write_xlsx_answers_in_place(base_bytes, answered_items)
        ext = "xlsx"
    elif file_ext == "csv":
        completed_bytes = write_csv_answers_in_place(base_bytes, answered_items)
        ext = "csv"
    else:
        completed_bytes = write_docx_answers_in_place(base_bytes, answered_items)
        ext = "docx"

    effective_org_id = org_id or q_dict.get("organization_id") or q_dict.get("proj_org_id") or "default_org"
    storage_path = f"{effective_org_id}/{questionnaire_id}/completed_{questionnaire_id}.{ext}"
    await upload_completed_docx(storage_path, completed_bytes)

    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE questionnaires
            SET completed_file_path       = $1,
                completed_file_updated_at = now(),
                has_manual_edits          = FALSE,
                updated_at                = now()
            WHERE id = $2
            """,
            storage_path,
            questionnaire_id,
        )

    signed_url = await get_docx_signed_url(storage_path)
    return storage_path, signed_url


# ── 1. Upload & Ingestion Endpoint ───────────────────────────────────────────

@router.post("/upload", response_model=QuestionnaireSummaryResponse)
async def upload_questionnaire(
    request: Request,
    file: UploadFile = File(..., description="Target .docx questionnaire file"),
    title: Optional[str] = Form(None, description="Optional title for the questionnaire"),
    project_id: Optional[str] = Form(None, description="Optional associated project ID"),
):
    try:
        user_id = request.headers.get("X-User-Id")
        user_org = request.headers.get("X-Organization-Id")

        pool = get_pool()
        if pool is None:
            raise HTTPException(status_code=500, detail="Database pool unavailable.")

        async with pool.acquire() as conn:
            if project_id:
                proj_row = await conn.fetchrow("SELECT id, organization_id FROM projects WHERE id = $1", project_id)
                if proj_row:
                    user_org = user_org or proj_row["organization_id"]
            else:
                proj_row = await conn.fetchrow("SELECT id, organization_id FROM projects LIMIT 1")
                if proj_row:
                    project_id = proj_row["id"]
                    user_org = user_org or proj_row["organization_id"]

            valid_user_id = None
            if user_id:
                member_row = await conn.fetchrow(
                    "SELECT id FROM members WHERE id = $1 OR supabase_user_id = $1 LIMIT 1",
                    user_id
                )
                if member_row:
                    valid_user_id = member_row["id"]

            if not valid_user_id:
                first_member = await conn.fetchrow("SELECT id FROM members LIMIT 1")
                valid_user_id = first_member["id"] if first_member else None

        filename = file.filename or "questionnaire.docx"
        file_ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "docx"
        if file_ext not in ("docx", "doc", "xlsx", "xls", "csv"):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Supported formats: Microsoft Word (.docx), Excel (.xlsx), CSV (.csv).",
            )

        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        try:
            if file_ext in ("xlsx", "xls"):
                extracted_items, template_bytes = extract_xlsx_questions(file_bytes)
            elif file_ext == "csv":
                extracted_items, template_bytes = extract_csv_questions(file_bytes)
            else:
                extracted_items, template_bytes = extract_docx_questions(file_bytes)
        except Exception as exc:
            logger.error("Failed to parse %s file %s: %s", file_ext.upper(), filename, exc)
            raise HTTPException(
                status_code=422,
                detail=f"Unable to parse .{file_ext} structure: {str(exc)}",
            ) from exc

        questionnaire_id = cuid_generator()
        doc_title = title.strip() if title and title.strip() else filename

        storage_path = await upload_questionnaire_file(
            organization_id=user_org or "default_org",
            questionnaire_id=questionnaire_id,
            file_bytes=template_bytes,
            filename=filename,
        )

        created_questions: list[ExtractedQuestionResponse] = []

        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO questionnaires (
                        id, organization_id, file_name,
                        file_url, status, question_count,
                        project_id, uploaded_by_id, file_size, file_type,
                        created_at, updated_at
                    )
                    VALUES (
                        $1, $2, $3,
                        $4, 'PROCESSING'::"QuestionnaireStatus", $5,
                        $6, $7, $8, $9,
                        now(), now()
                    )
                    """,
                    questionnaire_id,
                    user_org,
                    filename,
                    storage_path,
                    len(extracted_items),
                    project_id,
                    valid_user_id,
                    len(file_bytes),
                    file_ext,
                )

                for idx, item in enumerate(extracted_items):
                    q_id = cuid_generator()
                    coords_dict = item.coordinates.model_dump()
                    coords_dict["order_index"] = idx
                    coords_json = json.dumps(coords_dict)

                    await conn.execute(
                        """
                        INSERT INTO questions (
                            id, question, coordinates, status, questionnaire_id,
                            organization_id, source_file, created_by_id,
                            created_at, updated_at
                        )
                        VALUES ($1, $2, $3::jsonb, 'TODO'::"QuestionStatus", $4, $5, $6, $7, now(), now())
                        """,
                        q_id,
                        item.question_text,
                        coords_json,
                        questionnaire_id,
                        user_org,
                        filename,
                        valid_user_id,
                    )

                    created_questions.append(
                        ExtractedQuestionResponse(
                            id=q_id,
                            question=item.question_text,
                            coordinates=coords_dict,
                            status="TODO",
                        )
                    )

        return QuestionnaireSummaryResponse(
            questionnaire_id=questionnaire_id,
            title=doc_title,
            filename=filename,
            storage_path=storage_path,
            status="PROCESSING",
            question_count=len(created_questions),
            questions=created_questions,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in upload_questionnaire: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── 2. Retrieve Questionnaire Details ────────────────────────────────────────

@router.get("/{questionnaire_id}")
async def get_questionnaire(
    questionnaire_id: str,
    user: AuthenticatedUser = Depends(require_responder_or_admin),
):
    """Retrieves metadata and all questions for a specific questionnaire."""
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=500, detail="Database pool unavailable.")

    try:
        async with pool.acquire() as conn:
            q_row = await conn.fetchrow(
                """
                SELECT q.id, q.file_name, q.file_url, q.completed_file_path, q.status, q.project_id,
                       p.name AS project_name, p.client AS project_client
                FROM questionnaires q
                LEFT JOIN projects p ON q.project_id = p.id
                WHERE q.id = $1
                """,
                questionnaire_id,
            )
            if not q_row:
                raise HTTPException(status_code=404, detail="Questionnaire not found.")

            questions = await conn.fetch(
                """
                SELECT q.id, q.question, q.coordinates, q.status, q.draft_answer, q.confidence,
                       q.assigned_to_id, m.name AS assigned_to_name
                FROM questions q
                LEFT JOIN members m ON q.assigned_to_id = m.id
                WHERE q.questionnaire_id = $1
                ORDER BY COALESCE((q.coordinates->>'order_index')::int, 999999) ASC, q.created_at ASC, q.id ASC
                """,
                questionnaire_id,
            )

        q_dict = dict(q_row)
        parsed_questions = []
        for q in questions:
            q_d = dict(q)
            coords = q_d.get("coordinates")
            if isinstance(coords, str):
                try:
                    coords = json.loads(coords)
                except Exception:
                    coords = {}
            elif not isinstance(coords, dict):
                coords = {}

            sources = coords.get("sources", [])
            parsed_questions.append({
                "id": str(q_d.get("id")),
                "question": str(q_d.get("question") or ""),
                "coordinates": coords,
                "status": str(q_d.get("status") or "TODO"),
                "draft_answer": q_d.get("draft_answer"),
                "confidence": q_d.get("confidence"),
                "assigned_to_id": q_d.get("assigned_to_id"),
                "assigned_to_name": q_d.get("assigned_to_name"),
                "sources": sources,
            })

        fn = q_dict.get("file_name") or "questionnaire.docx"
        st_path = q_dict.get("file_url") or q_dict.get("completed_file_path") or ""

        return {
            "questionnaire_id": str(q_dict.get("id")),
            "title": str(fn),
            "filename": str(fn),
            "storage_path": str(st_path),
            "status": str(q_dict.get("status") or "PROCESSED"),
            "project_id": str(q_dict.get("project_id") or ""),
            "project_name": str(q_dict.get("project_name") or "Compliance RFP"),
            "project_client": str(q_dict.get("project_client") or ""),
            "question_count": len(parsed_questions),
            "questions": parsed_questions,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error in get_questionnaire: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── 3. Document Freshness Status ──────────────────────────────────────────────

@router.get("/{questionnaire_id}/document-status")
async def get_document_status(
    questionnaire_id: str,
    user: AuthenticatedUser = Depends(require_responder_or_admin),
):
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=500, detail="Database pool unavailable.")

    try:
        async with pool.acquire() as conn:
            q_row = await conn.fetchrow(
                "SELECT * FROM questionnaires WHERE id = $1",
                questionnaire_id,
            )
            if not q_row:
                raise HTTPException(status_code=404, detail="Questionnaire not found.")

            q_dict = dict(q_row)

            latest_question_update = await conn.fetchval(
                "SELECT MAX(updated_at) FROM questions WHERE questionnaire_id = $1",
                questionnaire_id,
            )

        completed_at = q_dict.get("completed_file_updated_at")
        file_path = q_dict.get("completed_file_path") or q_dict.get("file_url")
        has_file = bool(file_path)

        is_stale = True
        if has_file and completed_at and latest_question_update:
            is_stale = latest_question_update > completed_at
        elif has_file and completed_at and not latest_question_update:
            is_stale = False

        return {
            "questionnaire_id": questionnaire_id,
            "has_completed_document": has_file,
            "completed_file_updated_at": completed_at.isoformat() if completed_at else None,
            "latest_question_updated_at": latest_question_update.isoformat() if latest_question_update else None,
            "is_stale": is_stale,
            "has_manual_edits": bool(q_dict.get("has_manual_edits", False)),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error in get_document_status: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── 4. Individual Question AI Answering ───────────────────────────────────────

async def _sync_single_answer_to_file(
    questionnaire_id: str,
    organization_id: str,
    question_text: str,
    coords: dict,
    answer_text: str,
    pool: Any,
) -> dict:
    """Surgically writes an updated answer into the existing questionnaire document file in Supabase Storage.
    Supports Microsoft Word (.docx), Excel (.xlsx/.xls), and CSV (.csv).
    """
    async with pool.acquire() as conn:
        q_meta = await conn.fetchrow(
            "SELECT completed_file_path, file_url, file_type, file_name FROM questionnaires WHERE id = $1",
            questionnaire_id,
        )
    if not q_meta:
        return coords

    file_ext = (
        q_meta.get("file_type")
        or (q_meta["file_name"].rsplit(".", 1)[-1] if "." in q_meta["file_name"] else "docx")
    ).lower()

    clean_ext = "xlsx" if file_ext in ("xlsx", "xls") else ("csv" if file_ext == "csv" else "docx")
    storage_path = q_meta.get("completed_file_path")
    if not storage_path:
        storage_path = f"{organization_id}/{questionnaire_id}/completed_{questionnaire_id}.{clean_ext}"

    try:
        base_bytes = None
        if q_meta.get("completed_file_path"):
            try:
                base_bytes = await download_file_bytes(q_meta["completed_file_path"])
            except Exception:
                pass
        if not base_bytes and q_meta.get("file_url"):
            base_bytes = await download_file_bytes(q_meta["file_url"])

        if base_bytes:
            new_bytes = None
            if clean_ext == "docx":
                new_bytes, updated_coords = write_single_answer_in_place(
                    base_bytes,
                    question_text,
                    coords,
                    answer_text,
                )
                if updated_coords:
                    coords.update(updated_coords)
            elif clean_ext == "xlsx":
                answered_item = {
                    "question": question_text,
                    "coordinates": coords,
                    "draft_answer": answer_text,
                    "answer": answer_text,
                }
                new_bytes = write_xlsx_answers_in_place(base_bytes, [answered_item])
            elif clean_ext == "csv":
                answered_item = {
                    "question": question_text,
                    "coordinates": coords,
                    "draft_answer": answer_text,
                    "answer": answer_text,
                }
                new_bytes = write_csv_answers_in_place(base_bytes, [answered_item])

            if new_bytes:
                await upload_completed_docx(storage_path, new_bytes)
                async with pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE questionnaires
                        SET completed_file_path = $1,
                            completed_file_updated_at = now()
                        WHERE id = $2
                        """,
                        storage_path,
                        questionnaire_id,
                    )
    except Exception as doc_err:
        logger.warning("Could not sync answer to %s template: %s", clean_ext, doc_err)

    return coords


async def _sync_answer_to_library(
    pool,
    question_text: str,
    answer_text: str,
    confidence: int | None,
    questionnaire_id: str,
    organization_id: str,
    source_question_id: str | None = None,
    user_id: str | None = None,
    author_name: str | None = None,
):
    """Synchronizes an approved answer into the decoupled answer_library_items and answer_library_versions tables."""
    if not pool or not question_text or not (answer_text or "").strip():
        return
    try:
        async with pool.acquire() as conn:
            q_meta = await conn.fetchrow(
                """
                SELECT q.file_name, p.name AS project_name
                FROM questionnaires q
                LEFT JOIN projects p ON q.project_id = p.id
                WHERE q.id = $1
                """,
                questionnaire_id,
            )
            rfp_name = q_meta["file_name"] if q_meta and q_meta["file_name"] else "Institutional Library"
            project_name = q_meta["project_name"] if q_meta and q_meta["project_name"] else "Institutional Memory"
            source_file = rfp_name

            existing = None
            if source_question_id:
                existing = await conn.fetchrow(
                    "SELECT id FROM answer_library_items WHERE source_question_id = $1 AND organization_id = $2",
                    source_question_id,
                    organization_id,
                )
            if not existing:
                existing = await conn.fetchrow(
                    "SELECT id FROM answer_library_items WHERE question = $1 AND organization_id = $2",
                    question_text,
                    organization_id,
                )

            if existing:
                item_id = existing["id"]
                await conn.execute(
                    """
                    UPDATE answer_library_items
                    SET draft_answer = $1, status = 'APPROVED'::\"QuestionStatus\", confidence = $2,
                        source_file = $3, rfp_name = $4, project_name = $5, updated_at = now()
                    WHERE id = $6
                    """,
                    answer_text,
                    confidence,
                    source_file,
                    rfp_name,
                    project_name,
                    item_id,
                )
                total_versions = await conn.fetchval(
                    "SELECT COUNT(*) FROM answer_library_versions WHERE library_item_id = $1",
                    item_id,
                ) or 0
                v_id = cuid_generator()
                await conn.execute(
                    """
                    INSERT INTO answer_library_versions (
                        id, library_item_id, answer_text, version_number,
                        author_name, confidence, change_summary, created_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, now())
                    """,
                    v_id,
                    item_id,
                    answer_text,
                    total_versions + 1,
                    author_name or "ComplianceAI Engine",
                    confidence,
                    "Updated approved revision via ComplianceAI",
                )
            else:
                item_id = cuid_generator()
                await conn.execute(
                    """
                    INSERT INTO answer_library_items (
                        id, question, draft_answer, status, confidence,
                        source_file, rfp_name, project_name, source_question_id,
                        organization_id, created_by_id, created_at, updated_at
                    )
                    VALUES ($1, $2, $3, 'APPROVED'::\"QuestionStatus\", $4, $5, $6, $7, $8, $9, $10, now(), now())
                    """,
                    item_id,
                    question_text,
                    answer_text,
                    confidence,
                    source_file,
                    rfp_name,
                    project_name,
                    source_question_id,
                    organization_id,
                    user_id,
                )
                v_id = cuid_generator()
                await conn.execute(
                    """
                    INSERT INTO answer_library_versions (
                        id, library_item_id, answer_text, version_number,
                        author_name, confidence, change_summary, created_at
                    )
                    VALUES ($1, $2, $3, 1, $4, $5, 'Initial approved AI generated answer', now())
                    """,
                    v_id,
                    item_id,
                    answer_text,
                    author_name or "ComplianceAI Engine",
                    confidence,
                )
    except Exception as e:
        logger.warning("Could not sync answer to library: %s", e)


@router.post("/{questionnaire_id}/questions/{question_id}/answer")
async def answer_single_question(
    questionnaire_id: str,
    question_id: str,
    user: AuthenticatedUser = Depends(require_responder_or_admin),
):
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=500, detail="Database pool unavailable.")

    async with pool.acquire() as conn:
        question_row = await conn.fetchrow(
            "SELECT id, question, coordinates, draft_answer, status, assigned_to_id FROM questions WHERE id = $1 AND questionnaire_id = $2",
            question_id,
            questionnaire_id,
        )
        if not question_row:
            raise HTTPException(status_code=404, detail="Question not found.")

        if question_row["assigned_to_id"] and user.role == "RESPONDER":
            raise HTTPException(
                status_code=403,
                detail="This question has been assigned to a team member and cannot be generated or regenerated by the responder.",
            )

    async with _question_answering_semaphore:
        try:
            agent_res = await run_agent(
                question=question_row["question"],
                organization_id=user.organization_id,
            )
            draft = agent_res.get("draft", "")
            conf_score = int(round(agent_res.get("confidence_score", 0.0)))
            conf_level = agent_res.get("confidence_level", "red")
            status = "APPROVED" if conf_level in ("green", "amber") else "IN_REVIEW"
        except Exception as exc:
            logger.exception("Agent failed to answer question %s: %s", question_id, exc)
            raise HTTPException(status_code=500, detail=f"AI generation failed: {str(exc)}")

    # Extract reference sources from RAG chunks
    sources = []
    raw_chunks = agent_res.get("chunks", [])
    if raw_chunks:
        doc_ids = list({getattr(c, "document_id", None) for c in raw_chunks if getattr(c, "document_id", None)})
        doc_map = {}
        if doc_ids:
            try:
                async with pool.acquire() as conn:
                    doc_rows = await conn.fetch(
                        "SELECT id, filename FROM rag_documents WHERE id = ANY($1::text[])",
                        doc_ids
                    )
                    doc_map = {r["id"]: r["filename"] for r in doc_rows}
            except Exception as e:
                logger.warning("Could not fetch document filenames: %s", e)

        for c in raw_chunks[:4]:
            c_doc_id = getattr(c, "document_id", "")
            if c_doc_id == "institutional_library" or str(c_doc_id).startswith("lib_"):
                doc_name = "Verified Institutional Answer Library"
            else:
                doc_name = doc_map.get(c_doc_id, "Compliance Knowledge Base")
            heading = getattr(c, "heading_path", "") or "Policy Section"
            content_excerpt = (getattr(c, "content", "") or "")[:3000].strip()
            score = round(float(getattr(c, "combined_score", 0.0) or 0.0) * 100, 1)
            sources.append({
                "document_id": getattr(c, "document_id", ""),
                "document_name": doc_name,
                "heading": heading,
                "excerpt": content_excerpt,
                "score": score,
            })

    coords = question_row["coordinates"]
    if isinstance(coords, str):
        try:
            coords = json.loads(coords)
        except Exception:
            coords = {}
    elif not isinstance(coords, dict):
        coords = {}

    coords["sources"] = sources

    # ── Auto-sync into completed file in storage and answer library if approved ──
    if status == "APPROVED":
        coords = await _sync_single_answer_to_file(
            questionnaire_id=questionnaire_id,
            organization_id=user.organization_id,
            question_text=question_row["question"],
            coords=coords,
            answer_text=draft,
            pool=pool,
        )
        await _sync_answer_to_library(
            pool=pool,
            question_text=question_row["question"],
            answer_text=draft,
            confidence=conf_score,
            questionnaire_id=questionnaire_id,
            organization_id=user.organization_id,
            source_question_id=question_id,
            user_id=user.id,
            author_name=user.name,
        )

    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE questions
            SET draft_answer = $1, confidence = $2, status = $3::"QuestionStatus", coordinates = $4::jsonb, updated_at = now()
            WHERE id = $5
            """,
            draft,
            conf_score,
            status,
            json.dumps(coords),
            question_id,
        )

    return {
        "id": question_id,
        "question": question_row["question"],
        "coordinates": coords,
        "status": status,
        "draft_answer": draft,
        "confidence": conf_score,
        "sources": sources,
    }


class SyncQuestionDocRequest(BaseModel):
    answer: str


@router.post("/{questionnaire_id}/questions/{question_id}/sync-doc")
async def sync_question_to_doc(
    questionnaire_id: str,
    question_id: str,
    req: SyncQuestionDocRequest,
    user: AuthenticatedUser = Depends(require_any_org_member),
):
    """Surgically writes an updated answer into the existing completed document in-place.
    Preserves all manually added table rows, paragraphs, and styles.
    """
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=500, detail="Database pool unavailable.")

    async with pool.acquire() as conn:
        question_row = await conn.fetchrow(
            "SELECT id, question, coordinates, draft_answer, assigned_to_id FROM questions WHERE id = $1 AND questionnaire_id = $2",
            question_id,
            questionnaire_id,
        )
        if not question_row:
            raise HTTPException(status_code=404, detail="Question not found.")

        if question_row["assigned_to_id"] and user.role == "RESPONDER":
            raise HTTPException(
                status_code=403,
                detail="This question has been assigned to a team member and cannot be edited by the responder.",
            )

        q_meta = await conn.fetchrow(
            "SELECT completed_file_path, file_url, file_type, file_name FROM questionnaires WHERE id = $1",
            questionnaire_id,
        )

    coords = question_row["coordinates"]
    if isinstance(coords, str):
        try:
            coords = json.loads(coords)
        except Exception:
            coords = {}
    elif not isinstance(coords, dict):
        coords = {}

    coords = await _sync_single_answer_to_file(
        questionnaire_id=questionnaire_id,
        organization_id=user.organization_id,
        question_text=question_row["question"],
        coords=coords,
        answer_text=req.answer,
        pool=pool,
    )

    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE questions
            SET draft_answer = $1,
                coordinates = $2::jsonb,
                updated_at = now()
            WHERE id = $3
            """,
            req.answer,
            json.dumps(coords),
            question_id,
        )

    return {"status": "success", "coordinates": coords, "answer": req.answer}


# ── 5. Batch AI Answering ─────────────────────────────────────────────────────

@router.post("/{questionnaire_id}/answer-all")
async def answer_all_questions(
    questionnaire_id: str,
    user: AuthenticatedUser = Depends(require_responder_or_admin),
):
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=500, detail="Database pool unavailable.")

    async with pool.acquire() as conn:
        questions = await conn.fetch(
            "SELECT id, question, coordinates FROM questions WHERE questionnaire_id = $1 ORDER BY COALESCE((coordinates->>'order_index')::int, 999999) ASC, created_at ASC, id ASC",
            questionnaire_id,
        )

    if not questions:
        raise HTTPException(status_code=400, detail="No questions found for this questionnaire.")

    answered_count = 0
    results = []
    async with _question_answering_semaphore:
        for idx, q in enumerate(questions):
            if idx > 0:
                await asyncio.sleep(2.0)
            try:
                agent_res = await run_agent(
                    question=q["question"],
                    organization_id=user.organization_id,
                )
                draft = agent_res.get("draft", "")
                conf_score = int(round(agent_res.get("confidence_score", 0.0)))
                conf_level = agent_res.get("confidence_level", "red")
                status = "APPROVED" if conf_level in ("green", "amber") else "IN_REVIEW"

                # Extract reference sources from RAG chunks
                sources = []
                raw_chunks = agent_res.get("chunks", [])
                if raw_chunks:
                    doc_ids = list({getattr(c, "document_id", None) for c in raw_chunks if getattr(c, "document_id", None)})
                    doc_map = {}
                    if doc_ids:
                        try:
                            async with pool.acquire() as conn:
                                doc_rows = await conn.fetch(
                                    "SELECT id, filename FROM rag_documents WHERE id = ANY($1::text[])",
                                    doc_ids,
                                )
                                doc_map = {r["id"]: r["filename"] for r in doc_rows}
                        except Exception as e:
                            logger.warning("Could not fetch document filenames: %s", e)

                    for c in raw_chunks[:4]:
                        c_doc_id = getattr(c, "document_id", "")
                        if c_doc_id == "institutional_library" or str(c_doc_id).startswith("lib_"):
                            doc_name = "Verified Institutional Answer Library"
                        else:
                            doc_name = doc_map.get(c_doc_id, "Compliance Knowledge Base")
                        heading = getattr(c, "heading_path", "") or "Policy Section"
                        content_excerpt = (getattr(c, "content", "") or "")[:3000].strip()
                        score = round(float(getattr(c, "combined_score", 0.0) or 0.0) * 100, 1)
                        sources.append({
                            "document_id": getattr(c, "document_id", ""),
                            "document_name": doc_name,
                            "heading": heading,
                            "excerpt": content_excerpt,
                            "score": score,
                        })

                coords = q["coordinates"]
                if isinstance(coords, str):
                    try:
                        coords = json.loads(coords)
                    except Exception:
                        coords = {}
                elif not isinstance(coords, dict):
                    coords = {}

                coords["sources"] = sources

                # Auto-sync into completed file in storage and answer library if approved
                if status == "APPROVED":
                    try:
                        coords = await _sync_single_answer_to_file(
                            questionnaire_id=questionnaire_id,
                            organization_id=user.organization_id,
                            question_text=q["question"],
                            coords=coords,
                            answer_text=draft,
                            pool=pool,
                        )
                    except Exception as sync_err:
                        logger.warning("Failed to sync approved answer to document for question %s: %s", q["id"], sync_err)

                    try:
                        await _sync_answer_to_library(
                            pool=pool,
                            question_text=q["question"],
                            answer_text=draft,
                            confidence=conf_score,
                            questionnaire_id=questionnaire_id,
                            organization_id=user.organization_id,
                            source_question_id=q["id"],
                            user_id=user.id,
                            author_name=user.name,
                        )
                    except Exception as lib_err:
                        logger.warning("Failed to sync approved answer to library for question %s: %s", q["id"], lib_err)

                async with pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE questions
                        SET draft_answer = $1, confidence = $2, status = $3::"QuestionStatus", coordinates = $4::jsonb, updated_at = now()
                        WHERE id = $5
                        """,
                        draft,
                        conf_score,
                        status,
                        json.dumps(coords),
                        q["id"],
                    )

                results.append({
                    "id": q["id"],
                    "question_id": q["id"],
                    "question": q["question"],
                    "coordinates": coords,
                    "status": status,
                    "draft_answer": draft,
                    "confidence": conf_score,
                    "confidence_level": conf_level,
                    "sources": sources,
                    "references": sources,
                })
                answered_count += 1
            except Exception as e:
                logger.error("Failed to answer question %s: %s", q["id"], e)

    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE questionnaires SET status = 'PROCESSED'::\"QuestionnaireStatus\", updated_at = now() WHERE id = $1",
            questionnaire_id,
        )

    return {
        "questionnaire_id": questionnaire_id,
        "total_questions": len(questions),
        "answered_count": answered_count,
        "status": "PROCESSED",
        "answers": results,
        "references": results,
        "results": results,
    }


# ── 6. Build Document ────────────────────────────────────────────────────────

@router.post("/{questionnaire_id}/build-document")
async def build_document(
    questionnaire_id: str,
    user: AuthenticatedUser = Depends(require_responder_or_admin),
):
    storage_path, signed_url = await _generate_and_persist_completed_docx(
        questionnaire_id, user.organization_id
    )
    return {
        "status": "success",
        "message": "Completed questionnaire built and stored successfully.",
        "storage_path": storage_path,
        "download_url": signed_url,
    }


# ── 7. Direct Binary Stream for ONLYOFFICE ───────────────────────────────────

# ── 7. Direct Binary Stream for ONLYOFFICE ───────────────────────────────────

@router.get("/{questionnaire_id}/onlyoffice-file")
async def get_onlyoffice_file(questionnaire_id: str):
    """Direct binary endpoint for ONLYOFFICE Document Server.
    Pulls bytes from private Supabase Storage and streams them directly.
    """
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=500, detail="Database pool unavailable.")

    async with pool.acquire() as conn:
        q_row = await conn.fetchrow(
            """
            SELECT id, completed_file_path, file_url, project_id
            FROM questionnaires
            WHERE id = $1
            """,
            questionnaire_id,
        )

    if not q_row:
        raise HTTPException(status_code=404, detail="Questionnaire not found.")

    # In your schema, the path is stored in completed_file_path or file_url
    storage_path = (
        q_row.get("completed_file_path")
        or q_row.get("file_url")
    )

    if not storage_path:
        raise HTTPException(status_code=404, detail="Document file path not recorded.")

    # Strip any bucket prefix or full URL artifacts to ensure a clean relative path
    clean_path = storage_path.strip("/")
    if clean_path.startswith("questionnaires/"):
        clean_path = clean_path[len("questionnaires/"):]

    file_bytes = await download_file_bytes(clean_path)

    ext = (clean_path.rsplit('.', 1)[-1] if '.' in clean_path else 'docx').lower()
    if ext in ('xlsx', 'xls'):
        mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    elif ext == 'csv':
        mime = 'text/csv; charset=utf-8'
    elif ext in ('pptx', 'ppt'):
        mime = 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
    else:
        mime = DOCX_MIME_TYPE

    return Response(
        content=file_bytes,
        media_type=mime,
        headers={
            "Content-Disposition": f'inline; filename="completed_{questionnaire_id}.{ext}"',
            "Content-Type": mime,
        },
    )

# ── 8. ONLYOFFICE Editor Configuration (With JWT Signing) ────────────────────

@router.get("/{questionnaire_id}/editor-config")
async def get_editor_config(
    questionnaire_id: str,
    user: AuthenticatedUser = Depends(require_responder_or_admin),
):
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=500, detail="Database pool unavailable.")

    async with pool.acquire() as conn:
        q_row = await conn.fetchrow(
            "SELECT * FROM questionnaires WHERE id = $1",
            questionnaire_id,
        )
        if not q_row:
            raise HTTPException(status_code=404, detail="Questionnaire not found.")

    q_dict = dict(q_row)
    storage_path = q_dict.get("completed_file_path")
    completed_at = q_dict.get("completed_file_updated_at")

    if not storage_path:
        org_id = q_dict.get("organization_id") or user.organization_id
        storage_path, _ = await _generate_and_persist_completed_docx(
            questionnaire_id, org_id
        )
        async with pool.acquire() as conn:
            completed_at = await conn.fetchval(
                "SELECT completed_file_updated_at FROM questionnaires WHERE id = $1",
                questionnaire_id,
            )

    ts = int(completed_at.timestamp()) if completed_at else int(time.time())
    rev_key = f"{questionnaire_id}_{ts}"

    # Use host.docker.internal so the container can reach port 8000 on Windows host
    docker_host_base = getattr(settings, "api_base_url", None) or os.getenv("API_BASE_URL") or "http://host.docker.internal:8000"
    document_stream_url = f"{docker_host_base}/api/v1/questionnaires/{questionnaire_id}/onlyoffice-file"
    callback_url = f"{docker_host_base}/api/v1/questionnaires/{questionnaire_id}/onlyoffice-callback"

    ts = int(completed_at.timestamp()) if completed_at else int(time.time())
    rev_key = f"v3_{questionnaire_id}_{ts}"

    file_name = q_dict.get("file_name") or f"completed_{questionnaire_id}.docx"
    file_ext = (q_dict.get("file_type") or (file_name.rsplit(".", 1)[-1] if "." in file_name else "docx")).lower()

    if file_ext in ("xlsx", "xls", "csv"):
        doc_type = "cell"
        editor_ext = "xlsx" if file_ext in ("xlsx", "xls") else "csv"
    elif file_ext in ("pptx", "ppt"):
        doc_type = "slide"
        editor_ext = "pptx"
    else:
        doc_type = "word"
        editor_ext = "docx"

    config_payload = {
        "documentType": doc_type,
        "width": "100%",
        "height": "100%",
        "document": {
            "fileType": editor_ext,
            "key": rev_key,
            "title": f"completed_{questionnaire_id}.{editor_ext}",
            "url": document_stream_url,
            "permissions": {
                "download": True,
                "edit": True,
                "print": True,
                "review": False,
                "chat": False,
                "comment": True,
            },
        },
        "editorConfig": {
            "mode": "edit",
            "callbackUrl": callback_url,
            "user": {
                "id": user.id,
                "name": getattr(user, "name", None) or user.email,
            },
            "customization": {
                "compactHeader": True,
                "compactToolbar": True,
                "toolbarHideFileName": True,
                "about": False,
                "feedback": False,
                "help": False,
                "hideRightMenu": True,
                "features": {
                    "spellcheck": True,
                },
            },
        },
    }

    # Generate and sign JWT token using HS256 with the secret
    jwt_secret = _get_onlyoffice_jwt_secret()
    token = jwt.encode(config_payload, jwt_secret, algorithm="HS256")
    config_payload["token"] = token

    return config_payload


# ── 9. ONLYOFFICE Save Callback (With JWT Decoding) ──────────────────────────

@router.post("/{questionnaire_id}/onlyoffice-callback")
async def onlyoffice_callback(
    questionnaire_id: str,
    request: Request,
):
    try:
        body = await request.json()
    except Exception:
        return {"error": 1}

    # If JWT is enabled in ONLYOFFICE, extract signed payload from token
    if "token" in body:
        jwt_secret = _get_onlyoffice_jwt_secret()
        try:
            decoded = jwt.decode(body["token"], jwt_secret, algorithms=["HS256"])
            body = decoded.get("payload", decoded)
        except Exception as e:
            logger.error("Failed to decode ONLYOFFICE callback JWT: %s", e)
            return {"error": 1}

    status = body.get("status")
    download_url = body.get("url")

    if status not in (2, 6):
        return {"error": 0}

    if not download_url:
        return {"error": 1}

    pool = get_pool()
    if pool is None:
        return {"error": 1}

    async with pool.acquire() as conn:
        q_row = await conn.fetchrow(
            """
            SELECT q.*, p.organization_id as proj_org_id 
            FROM questionnaires q 
            LEFT JOIN projects p ON q.project_id = p.id 
            WHERE q.id = $1
            """,
            questionnaire_id,
        )
        if not q_row:
            return {"error": 1}
        q_d = dict(q_row)
        org_id = q_d.get("organization_id") or q_d.get("proj_org_id") or "default_org"

    try:
        fetch_url = download_url
        internal_oo = getattr(settings, "onlyoffice_internal_url", "http://127.0.0.1:8085")
        if "http://localhost/" in fetch_url or "http://127.0.0.1/" in fetch_url:
            fetch_url = fetch_url.replace("http://localhost/", f"{internal_oo}/").replace("http://127.0.0.1/", f"{internal_oo}/")
        elif "http://localhost:80/" in fetch_url or "http://127.0.0.1:80/" in fetch_url:
            fetch_url = fetch_url.replace("http://localhost:80/", f"{internal_oo}/").replace("http://127.0.0.1:80/", f"{internal_oo}/")

        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.get(fetch_url)
        if res.status_code != 200:
            return {"error": 1}

        file_name = q_d.get("file_name") or f"completed_{questionnaire_id}.docx"
        file_ext = (q_d.get("file_type") or (file_name.rsplit(".", 1)[-1] if "." in file_name else "docx")).lower()
        editor_ext = "xlsx" if file_ext in ("xlsx", "xls") else ("csv" if file_ext == "csv" else "docx")
        storage_path = f"{org_id}/{questionnaire_id}/completed_{questionnaire_id}.{editor_ext}"
        await upload_completed_docx(storage_path, res.content)

        # ── Reverse Sync: Sync document edits back into questions table ──
        if editor_ext == "docx":
            try:
                async with pool.acquire() as conn:
                    db_questions = await conn.fetch(
                        """
                        SELECT id, question, coordinates, draft_answer
                        FROM questions
                        WHERE questionnaire_id = $1
                        """,
                        questionnaire_id,
                    )
                q_list = [dict(q) for q in db_questions]
                sync_results = locate_and_sync_docx_anchors(res.content, q_list)

                async with pool.acquire() as conn:
                    for item in sync_results:
                        if item.get("has_changes"):
                            await conn.execute(
                                """
                                UPDATE questions
                                SET draft_answer = $1,
                                    coordinates  = $2::jsonb,
                                    updated_at   = now()
                                WHERE id = $3
                                """,
                                item["new_answer"],
                                json.dumps(item["new_coordinates"]),
                                item["id"],
                            )
                logger.info(
                    "Reverse sync completed for questionnaire %s: synced %d questions",
                    questionnaire_id,
                    len(sync_results),
                )
            except Exception as sync_exc:
                logger.warning("Reverse sync failed for questionnaire %s: %s", questionnaire_id, sync_exc)

        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE questionnaires
                SET completed_file_updated_at = now(),
                    has_manual_edits          = TRUE,
                    updated_at                = now()
                WHERE id = $1
                """,
                questionnaire_id,
            )
        return {"error": 0}
    except Exception as exc:
        logger.exception("Error processing ONLYOFFICE save: %s", exc)
        return {"error": 1}


# ── 10. Export Attachment (DOCX & PDF) ─────────────────────────────────────────

def _generate_questionnaire_pdf(title: str, project_name: str, client_name: str, questions: list) -> bytes:
    from io import BytesIO
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#05603A'),
    )
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#4A5568'),
    )
    q_num_style = ParagraphStyle(
        'QNum',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#1A202C'),
    )
    ans_style = ParagraphStyle(
        'QAns',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#2D3748'),
    )
    badge_style = ParagraphStyle(
        'Badge',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#05603A'),
    )

    story = []
    story.append(Paragraph("<b>ComplianceAi · Verified Response Export</b>", meta_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(title or "Compliance Questionnaire", title_style))
    story.append(Spacer(1, 4))
    meta_text = f"Project: <b>{project_name or 'Enterprise RFP'}</b>"
    if client_name:
        meta_text += f" · Client: <b>{client_name}</b>"
    meta_text += f" · Total Questions: <b>{len(questions)}</b>"
    story.append(Paragraph(meta_text, meta_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E0'), spaceAfter=14))

    for idx, q in enumerate(questions, 1):
        q_text = (q.get("question") or "Question").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        raw_ans = q.get("draft_answer") or "No answer provided yet."
        a_text = raw_ans.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
        status = q.get("status") or "TODO"
        conf = q.get("confidence")
        conf_str = f" · {conf}% Conf" if conf is not None else ""

        q_p = Paragraph(f"<b>Q{idx}. {q_text}</b>", q_num_style)
        stat_p = Paragraph(f"<font color='#05603A'><b>[{status}{conf_str}]</b></font>", badge_style)
        ans_p = Paragraph(a_text, ans_style)

        table_data = [
            [q_p, stat_p],
            [ans_p, ""],
        ]
        t = Table(table_data, colWidths=[440, 90])
        t.setStyle(TableStyle([
            ('SPAN', (0, 1), (1, 1)),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ]))
        story.append(t)
        story.append(Spacer(1, 8))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


@router.get("/{questionnaire_id}/export")
async def export_questionnaire(
    questionnaire_id: str,
    format: Optional[str] = Query(None, description="Optional export format override: 'docx', 'xlsx', 'csv', or 'pdf'"),
    user: AuthenticatedUser = Depends(require_responder_or_admin),
):
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=500, detail="Database pool unavailable.")

    async with pool.acquire() as conn:
        q_row = await conn.fetchrow(
            """
            SELECT q.id, q.file_name, q.file_url, q.completed_file_path, q.status, q.file_type,
                   p.name AS project_name, p.client AS project_client,
                   q.organization_id
            FROM questionnaires q
            LEFT JOIN projects p ON q.project_id = p.id
            WHERE q.id = $1
            """,
            questionnaire_id,
        )
        if not q_row:
            raise HTTPException(status_code=404, detail="Questionnaire not found.")

        questions = await conn.fetch(
            "SELECT id, question, coordinates, draft_answer, status, confidence FROM questions WHERE questionnaire_id = $1 ORDER BY created_at ASC",
            questionnaire_id,
        )

    file_name = q_row["file_name"] or f"questionnaire_{questionnaire_id}.docx"
    file_ext = (q_row["file_type"] or (file_name.rsplit(".", 1)[-1] if "." in file_name else "docx")).lower()
    base_name = file_name.rsplit(".", 1)[0]

    # Format defaults strictly to the original uploaded format unless explicit override
    clean_format = format.lower().strip() if format else file_ext

    # Optional PDF export if explicitly requested
    if clean_format == "pdf":
        pdf_bytes = _generate_questionnaire_pdf(
            title=file_name,
            project_name=q_row["project_name"] or "",
            client_name=q_row["project_client"] or "",
            questions=[dict(q) for q in questions],
        )
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{base_name}_verified.pdf"',
                "Access-Control-Expose-Headers": "Content-Disposition",
            },
        )

    # Excel (.xlsx) export
    if clean_format in ("xlsx", "xls"):
        orig_bytes = None
        if q_row["file_url"]:
            try:
                orig_bytes = await download_file_bytes(q_row["file_url"])
            except Exception as exc:
                logger.warning("Could not fetch original xlsx from storage: %s", exc)

        answered_items = []
        for q in questions:
            q_d = dict(q)
            coords = q_d.get("coordinates")
            if isinstance(coords, str):
                try:
                    coords = json.loads(coords)
                except Exception:
                    coords = {}
            answered_items.append({
                "coordinates": coords or {},
                "answer": q_d.get("draft_answer") or "",
                "question": q_d.get("question") or "",
            })

        file_bytes = write_xlsx_answers_in_place(orig_bytes or b"", answered_items)
        return Response(
            content=file_bytes,
            media_type=XLSX_MIME_TYPE,
            headers={
                "Content-Disposition": f'attachment; filename="{base_name}_completed.xlsx"',
                "Access-Control-Expose-Headers": "Content-Disposition",
            },
        )

    # CSV (.csv) export
    if clean_format == "csv":
        orig_bytes = None
        if q_row["file_url"]:
            try:
                orig_bytes = await download_file_bytes(q_row["file_url"])
            except Exception as exc:
                logger.warning("Could not fetch original csv from storage: %s", exc)

        answered_items = []
        for q in questions:
            q_d = dict(q)
            coords = q_d.get("coordinates")
            if isinstance(coords, str):
                try:
                    coords = json.loads(coords)
                except Exception:
                    coords = {}
            answered_items.append({
                "coordinates": coords or {},
                "answer": q_d.get("draft_answer") or "",
                "question": q_d.get("question") or "",
            })

        file_bytes = write_csv_answers_in_place(orig_bytes or b"", answered_items)
        return Response(
            content=file_bytes,
            media_type=CSV_MIME_TYPE,
            headers={
                "Content-Disposition": f'attachment; filename="{base_name}_completed.csv"',
                "Access-Control-Expose-Headers": "Content-Disposition",
            },
        )

    # Word (.docx) export (format preserving)
    if not q_row["completed_file_path"]:
        storage_path, _ = await _generate_and_persist_completed_docx(
            questionnaire_id, user.organization_id
        )
    else:
        storage_path = q_row["completed_file_path"]

    file_bytes = await download_file_bytes(storage_path)
    return Response(
        content=file_bytes,
        media_type=DOCX_MIME_TYPE,
        headers={
            "Content-Disposition": f'attachment; filename="{base_name}_completed.docx"',
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )
