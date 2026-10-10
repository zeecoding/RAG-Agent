"""Questionnaire ingestion, AI answer generation, and in-place DOCX export routes.

New in this revision:
  - _generate_and_persist_completed_docx() — internal helper that builds the
    completed DOCX, uploads it to Supabase Storage, and records the path +
    timestamp in the questionnaires table.
  - GET  /{id}/document-status        — freshness + manual-edit status check.
  - POST /{id}/build-document         — explicit rebuild trigger.
  - POST /{id}/onlyoffice-callback    — ONLYOFFICE Document Server save webhook;
                                        persists user styling edits back to Storage.
  - GET  /{id}/editor-config          — ONLYOFFICE JSON config (auto-builds if needed).
  - GET  /{id}/export                 — serves the stored DOCX (builds on demand if
                                        not yet persisted).  Replaces the old POST
                                        version which streamed without persisting.
"""

import asyncio
import io
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from cuid2 import cuid_wrapper
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.graph import run_agent
from app.auth.dependencies import AuthenticatedUser, require_responder_or_admin
from app.config import settings
from app.db.pool import get_pool
from app.db.storage import (
    DOCX_MIME_TYPE,
    download_file_bytes,
    download_questionnaire_file,
    get_docx_signed_url,
    upload_completed_docx,
    upload_questionnaire_file,
)
from app.exporter.docx_engine import extract_docx_questions, write_docx_answers_in_place
from app.exporter.schemas import AnsweredItem

logger = logging.getLogger(__name__)

cuid_generator = cuid_wrapper()

# Global concurrency queue: limit concurrent LLM answer generation to prevent 429 rate limit errors
_question_answering_semaphore = asyncio.Semaphore(1)

router = APIRouter(prefix="/questionnaires", tags=["Questionnaires"])


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
    """Build the completed DOCX in memory, upload to Supabase Storage, update
    the questionnaires row with the path and timestamp, and return
    ``(storage_path, signed_url)``.

    The signed URL is valid for 2 hours and is suitable for both ONLYOFFICE
    document loading and direct browser download.
    """
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=500, detail="Database pool unavailable.")

    # ── 1. Fetch questionnaire template path and answered questions ───────────
    async with pool.acquire() as conn:
        q_row = await conn.fetchrow(
            """
            SELECT storage_path, file_url
            FROM questionnaires
            WHERE id = $1 AND organization_id = $2
            """,
            questionnaire_id,
            org_id,
        )
        if not q_row:
            raise HTTPException(status_code=404, detail="Questionnaire not found.")

        original_path = q_row["storage_path"] or q_row["file_url"]
        if not original_path:
            raise HTTPException(
                status_code=404,
                detail="Questionnaire template storage path not recorded.",
            )

        questions = await conn.fetch(
            """
            SELECT id, question, coordinates, draft_answer, status
            FROM questions
            WHERE questionnaire_id = $1 AND organization_id = $2
            ORDER BY created_at ASC
            """,
            questionnaire_id,
            org_id,
        )

    # ── 2. Download original template bytes from Supabase ────────────────────
    original_bytes = await download_file_bytes(original_path)

    # ── 3. Build answered-items payload ──────────────────────────────────────
    answered_items: list[dict] = []
    for q in questions:
        coords = q["coordinates"]
        if isinstance(coords, str):
            try:
                coords = json.loads(coords)
            except Exception:
                coords = {}
        answered_items.append({
            "coordinates": coords,
            "answer": q["draft_answer"] or "",
        })

    logger.info(
        "Building completed DOCX for questionnaire %s: %d answered items.",
        questionnaire_id,
        len(answered_items),
    )

    # ── 4. Mutate DOCX in memory ──────────────────────────────────────────────
    completed_bytes = write_docx_answers_in_place(original_bytes, answered_items)

    # ── 5. Upload to Supabase bucket ──────────────────────────────────────────
    # Bare relative path (no bucket prefix) — stored in the DB for re-use.
    storage_path = f"{org_id}/{questionnaire_id}/completed_{questionnaire_id}.docx"
    await upload_completed_docx(storage_path, completed_bytes)

    # ── 6. Record path + timestamp in questionnaires table ───────────────────
    # Also reset has_manual_edits so the flag accurately reflects the current
    # file: a fresh automated build contains only agent answers, no human styling.
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

    # ── 7. Generate signed URL (2-hour window) ───────────────────────────────
    signed_url = await get_docx_signed_url(storage_path)
    logger.info("Completed DOCX persisted at %s for questionnaire %s.", storage_path, questionnaire_id)
    return storage_path, signed_url


# ── 1. Upload & Ingestion Endpoint ───────────────────────────────────────────

@router.post("/upload", response_model=QuestionnaireSummaryResponse)
async def upload_questionnaire(
    file: UploadFile = File(..., description="Target .docx questionnaire file"),
    title: Optional[str] = Form(None, description="Optional title for the questionnaire"),
    project_id: Optional[str] = Form(None, description="Optional associated project ID"),
    user: AuthenticatedUser = Depends(require_responder_or_admin),
):
    """Uploads a .docx questionnaire template, coordinates extraction of questions
    and their exact write-back positions, uploads raw binary to Supabase Storage,
    and records everything transactionally in Neon Postgres.
    """
    filename = file.filename or "questionnaire.docx"
    if not filename.lower().endswith(".docx"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Only Microsoft Word (.docx) files are supported.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 1. Parse questions and coordinates; also obtain the (possibly mutated) template bytes
    try:
        extracted_items, template_bytes = extract_docx_questions(file_bytes)
    except Exception as exc:
        logger.error("Failed to parse DOCX file %s: %s", filename, exc)
        raise HTTPException(
            status_code=422,
            detail=f"Unable to parse .docx structure: {str(exc)}",
        ) from exc

    questionnaire_id = cuid_generator()
    doc_title = title.strip() if title and title.strip() else filename

    # 2. Upload the template to Supabase Storage.
    # Use `template_bytes` (returned by extract_docx_questions) rather than
    # the raw `file_bytes`.  For table-based documents they are identical; for
    # paragraph-style documents `template_bytes` contains the injected blank
    # answer-slot paragraphs whose indices match the stored coordinates.
    try:
        storage_path = await upload_questionnaire_file(
            organization_id=user.organization_id,
            questionnaire_id=questionnaire_id,
            file_bytes=template_bytes,
            filename=filename,
        )
    except Exception as exc:
        logger.exception("Failed to upload questionnaire binary to storage: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Storage upload failed: {str(exc)}",
        ) from exc


    # 3. Atomic Database Insertion into Neon
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=500, detail="Database pool unavailable.")

    created_questions: list[ExtractedQuestionResponse] = []

    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                # If project_id wasn't provided, use user's default/first project if available
                target_project_id = project_id
                if not target_project_id:
                    proj_row = await conn.fetchrow(
                        "SELECT id FROM projects WHERE organization_id = $1 LIMIT 1",
                        user.organization_id,
                    )
                    if proj_row:
                        target_project_id = proj_row["id"]

                await conn.execute(
                    """
                    INSERT INTO questionnaires (
                        id, organization_id, title, filename, file_name,
                        storage_path, file_url, status, question_count,
                        project_id, uploaded_by_id, file_size, file_type,
                        created_at, updated_at
                    )
                    VALUES (
                        $1, $2, $3, $4, $5,
                        $6, $7, $8::\"QuestionnaireStatus\", $9,
                        $10, $11, $12, $13,
                        now(), now()
                    )
                    """,
                    questionnaire_id,
                    user.organization_id,
                    doc_title,
                    filename,
                    filename,
                    storage_path,
                    storage_path,
                    "PROCESSING",
                    len(extracted_items),
                    target_project_id,
                    user.id,
                    len(file_bytes),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )

                # Batch insert extracted questions with coordinates
                for item in extracted_items:
                    q_id = cuid_generator()
                    coords_dict = item.coordinates.model_dump()
                    coords_json = json.dumps(coords_dict)

                    await conn.execute(
                        """
                        INSERT INTO questions (
                            id, question, coordinates, status, questionnaire_id,
                            organization_id, source_file, created_by_id,
                            created_at, updated_at
                        )
                        VALUES ($1, $2, $3::jsonb, $4::\"QuestionStatus\", $5, $6, $7, $8, now(), now())
                        """,
                        q_id,
                        item.question_text,
                        coords_json,
                        "TODO",
                        questionnaire_id,
                        user.organization_id,
                        filename,
                        user.id,
                    )

                    created_questions.append(
                        ExtractedQuestionResponse(
                            id=q_id,
                            question=item.question_text,
                            coordinates=coords_dict,
                            status="TODO",
                        )
                    )
    except Exception as exc:
        logger.exception("Failed to persist questionnaire %s in database: %s", questionnaire_id, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Database error saving questionnaire: {str(exc)}",
        ) from exc

    return QuestionnaireSummaryResponse(
        questionnaire_id=questionnaire_id,
        title=doc_title,
        filename=filename,
        storage_path=storage_path,
        status="PROCESSING",
        question_count=len(created_questions),
        questions=created_questions,
    )


# ── 2. Individual Question AI Answering Route ────────────────────────────────

@router.post("/{questionnaire_id}/questions/{question_id}/answer", response_model=ExtractedQuestionResponse)
async def answer_single_question(
    questionnaire_id: str,
    question_id: str,
    user: AuthenticatedUser = Depends(require_responder_or_admin),
):
    """Executes the LangGraph RAG agent (run_agent) for a single question with
    queue-controlled rate limiting, saving draft answer and confidence score.
    """
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=500, detail="Database pool unavailable.")

    async with pool.acquire() as conn:
        # Verify questionnaire ownership
        q_row = await conn.fetchrow(
            """
            SELECT id FROM questionnaires
            WHERE id = $1 AND organization_id = $2
            """,
            questionnaire_id,
            user.organization_id,
        )
        if not q_row:
            raise HTTPException(status_code=404, detail="Questionnaire not found.")

        # Fetch target question
        question_row = await conn.fetchrow(
            """
            SELECT id, question, coordinates, draft_answer, status, confidence
            FROM questions
            WHERE id = $1 AND questionnaire_id = $2 AND organization_id = $3
            """,
            question_id,
            questionnaire_id,
            user.organization_id,
        )
        if not question_row:
            raise HTTPException(status_code=404, detail="Question not found.")

    # Execute under concurrency queue to prevent 429 rate limit errors
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
            raise HTTPException(
                status_code=500,
                detail=f"AI answer generation failed: {str(exc)}",
            ) from exc

    # Persist updated question answer in Neon
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE questions
            SET draft_answer = $1, confidence = $2, status = $3::\"QuestionStatus\", updated_at = now()
            WHERE id = $4
            """,
            draft,
            conf_score,
            status,
            question_id,
        )

        # Update questionnaire status if all questions now have draft answers
        unanswered = await conn.fetchval(
            """
            SELECT COUNT(*) FROM questions
            WHERE questionnaire_id = $1 AND (draft_answer IS NULL OR draft_answer = '')
            """,
            questionnaire_id,
        )
        if unanswered == 0:
            await conn.execute(
                """
                UPDATE questionnaires
                SET status = 'PROCESSED'::"QuestionnaireStatus", updated_at = now()
                WHERE id = $1
                """,
                questionnaire_id,
            )

    coords = question_row["coordinates"]
    if isinstance(coords, str):
        try:
            coords = json.loads(coords)
        except Exception:
            coords = {}

    return ExtractedQuestionResponse(
        id=question_id,
        question=question_row["question"],
        coordinates=coords or {},
        status=status,
        draft_answer=draft,
        confidence=conf_score,
    )


# ── 3. Automated Bulk AI Answering Route ──────────────────────────────────────

@router.post("/{questionnaire_id}/answer-all", response_model=AnswerAllResponse)
async def answer_all_questions(
    questionnaire_id: str,
    user: AuthenticatedUser = Depends(require_responder_or_admin),
):
    """Executes the LangGraph RAG agent (run_agent) across all extracted questions
    in the questionnaire sequentially with queue pacing to prevent 429 rate limit errors.

    The DB connection is released before the LLM loop begins so that multi-minute
    inference work does not starve the asyncpg connection pool. Each question's result
    is persisted in its own short-lived acquire/release cycle.
    """
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=500, detail="Database pool unavailable.")

    # ── Phase 1: Fetch ─────────────────────────────────────────────────────────
    # Acquire a connection only for the initial read, then release immediately.
    async with pool.acquire() as conn:
        # Verify questionnaire ownership
        q_row = await conn.fetchrow(
            """
            SELECT id, status FROM questionnaires
            WHERE id = $1 AND organization_id = $2
            """,
            questionnaire_id,
            user.organization_id,
        )
        if not q_row:
            raise HTTPException(status_code=404, detail="Questionnaire not found.")

        # Fetch questions with coordinates
        questions = await conn.fetch(
            """
            SELECT id, question, coordinates, draft_answer
            FROM questions
            WHERE questionnaire_id = $1 AND organization_id = $2
            ORDER BY created_at ASC
            """,
            questionnaire_id,
            user.organization_id,
        )

    # Connection is now released — the pool is not starved during the LLM loop.

    if not questions:
        raise HTTPException(
            status_code=400,
            detail="No questions found for this questionnaire.",
        )

    # ── Phase 2: LLM loop with per-question DB updates ────────────────────────
    answered_count = 0
    async with _question_answering_semaphore:
        for idx, q in enumerate(questions):
            # Pacing delay between questions to stay safely under Groq RPM/TPM limits
            if idx > 0:
                await asyncio.sleep(2.0)

            # Execute RAG agent pipeline with individual resilience
            try:
                agent_res = await run_agent(
                    question=q["question"],
                    organization_id=user.organization_id,
                )
                draft = agent_res.get("draft", "")
                conf_score = int(round(agent_res.get("confidence_score", 0.0)))
                conf_level = agent_res.get("confidence_level", "red")
                status = "APPROVED" if conf_level in ("green", "amber") else "IN_REVIEW"

                # Short-lived acquire per question — does not hold the connection
                # idle during the preceding or following LLM calls.
                async with pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE questions
                        SET draft_answer = $1, confidence = $2, status = $3::\"QuestionStatus\", updated_at = now()
                        WHERE id = $4
                        """,
                        draft,
                        conf_score,
                        status,
                        q["id"],
                    )
                answered_count += 1
            except Exception as e:
                logger.error("Failed to answer question %s in batch: %s", q["id"], e)
                if "429" in str(e) or "rate limit" in str(e).lower():
                    logger.warning("Rate limit hit during batch answering. Cooling down for 6s...")
                    await asyncio.sleep(6.0)

    # ── Phase 3: Final questionnaire status update ────────────────────────────
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE questionnaires
            SET status = 'PROCESSED'::"QuestionnaireStatus", updated_at = now()
            WHERE id = $1
            """,
            questionnaire_id,
        )

    return AnswerAllResponse(
        questionnaire_id=questionnaire_id,
        total_questions=len(questions),
        answered_count=answered_count,
        status="PROCESSED",
    )


# ── 4. Document Freshness Status ──────────────────────────────────────────────

@router.get("/{questionnaire_id}/document-status")
async def get_document_status(
    questionnaire_id: str,
    user: AuthenticatedUser = Depends(require_responder_or_admin),
):
    """Checks whether the saved completed DOCX is stale compared to the latest
    question updates, and whether the stored document contains manual ONLYOFFICE
    edits that would be overwritten by a fresh automated rebuild.

    Returns a JSON payload the Next.js frontend uses to:
      - Decide whether to show a "Rebuild document" prompt.
      - Warn the user if clicking "Rebuild" would destroy styling edits.
    """
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=500, detail="Database pool unavailable.")

    async with pool.acquire() as conn:
        q_row = await conn.fetchrow(
            """
            SELECT completed_file_path,
                   completed_file_updated_at,
                   has_manual_edits
            FROM questionnaires
            WHERE id = $1 AND organization_id = $2
            """,
            questionnaire_id,
            user.organization_id,
        )
        if not q_row:
            raise HTTPException(status_code=404, detail="Questionnaire not found.")

        # Latest modification across all child questions
        latest_question_update = await conn.fetchval(
            """
            SELECT MAX(updated_at)
            FROM questions
            WHERE questionnaire_id = $1 AND organization_id = $2
            """,
            questionnaire_id,
            user.organization_id,
        )

    completed_at: Optional[datetime] = q_row["completed_file_updated_at"]
    has_file = bool(q_row["completed_file_path"])

    # Staleness logic:
    #  - No completed file at all              → stale (needs first build)
    #  - File exists, questions updated after  → stale
    #  - File exists, no question updates      → fresh
    #  - File exists, questions all predated   → fresh
    is_stale = True
    if has_file and completed_at and latest_question_update:
        is_stale = latest_question_update > completed_at
    elif has_file and completed_at and not latest_question_update:
        is_stale = False

    return {
        "questionnaire_id": questionnaire_id,
        "has_completed_document": has_file,
        "completed_file_updated_at": completed_at,
        "latest_question_updated_at": latest_question_update,
        "is_stale": is_stale,
        # True when the stored DOCX contains styling/formatting changes saved
        # through ONLYOFFICE — warn the user before a rebuild erases them.
        "has_manual_edits": bool(q_row["has_manual_edits"]),
    }


# ── 5. Build / Re-Build Endpoint ──────────────────────────────────────────────

@router.post("/{questionnaire_id}/build-document")
async def build_document(
    questionnaire_id: str,
    user: AuthenticatedUser = Depends(require_responder_or_admin),
):
    """Builds (or re-builds) the completed DOCX, persists it in Supabase Storage,
    and updates the freshness timestamp.

    Triggered by the Next.js frontend when the user clicks 'Build / Update Doc'
    or when the editor launch detects a stale document.
    """
    storage_path, signed_url = await _generate_and_persist_completed_docx(
        questionnaire_id, user.organization_id
    )
    return {
        "status": "success",
        "message": "Completed questionnaire built and stored successfully.",
        "storage_path": storage_path,
        "download_url": signed_url,
    }


# ── 6. ONLYOFFICE Save Callback ───────────────────────────────────────────────

@router.post("/{questionnaire_id}/onlyoffice-callback")
async def onlyoffice_callback(
    questionnaire_id: str,
    request: Request,
):
    """Webhook invoked by ONLYOFFICE Document Server whenever a document is
    saved or closed.

    ONLYOFFICE status codes (subset used here):
      1  — Document is being edited; no action needed.
      2  — All editors closed; document is ready for saving.  ``body.url``
           holds a temporary download link valid for ~15 minutes.
      4  — Closed with no changes; no action needed.
      6  — Force-save triggered (e.g. auto-save interval or API call).
           ``body.url`` holds a temporary download link.

    For status 2 and 6 this endpoint:
      1. Downloads the updated `.docx` bytes from the Document Server URL.
      2. Upserts those bytes into Supabase Storage at the same path as the
         agent-generated document (overwriting the previous version).
      3. Sets ``has_manual_edits = TRUE`` and updates ``completed_file_updated_at``
         in the ``questionnaires`` table.

    No user auth dependency — this request originates from the ONLYOFFICE
    Docker container (or cloud server), not from a browser session.  Callers
    must ensure ONLYOFFICE is configured with the correct JWT secret when
    operating in a non-trusted network.

    Always returns ``{"error": 0}`` on success (required by ONLYOFFICE to
    confirm receipt).  Returns ``{"error": 1}`` on any recoverable failure so
    ONLYOFFICE can retry.
    """
    try:
        body = await request.json()
    except Exception:
        logger.warning("ONLYOFFICE callback: received non-JSON body for questionnaire %s", questionnaire_id)
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    status = body.get("status")
    download_url = body.get("url")

    logger.info(
        "ONLYOFFICE callback for questionnaire %s: status=%s, has_url=%s",
        questionnaire_id,
        status,
        bool(download_url),
    )

    # Status 1 (being edited) and 4 (closed, no changes) require no action.
    if status not in (2, 6):
        return {"error": 0}

    # Status 2 / 6: the document is ready; ``url`` is a temporary signed link
    # to the updated binary on the ONLYOFFICE Document Server.
    if not download_url:
        logger.warning(
            "ONLYOFFICE callback status %d received for questionnaire %s but no download URL supplied",
            status,
            questionnaire_id,
        )
        return {"error": 1}

    pool = get_pool()
    if pool is None:
        logger.error("ONLYOFFICE callback: database pool unavailable for questionnaire %s", questionnaire_id)
        return {"error": 1, "message": "Database unavailable"}

    # Resolve the organization so we can build the correct storage path.
    async with pool.acquire() as conn:
        q_row = await conn.fetchrow(
            "SELECT organization_id FROM questionnaires WHERE id = $1",
            questionnaire_id,
        )
        if not q_row:
            logger.error(
                "ONLYOFFICE callback: questionnaire %s not found in database", questionnaire_id
            )
            return {"error": 1, "message": "Questionnaire not found"}
        org_id: str = q_row["organization_id"]

    try:
        # ── Step 1: Download the edited DOCX from ONLYOFFICE ─────────────────
        # The URL is a temporary link served by the Document Server itself.
        # Use a dedicated AsyncClient (not the shared one) because this call
        # happens in a background webhook context.
        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.get(download_url)

        if res.status_code != 200:
            logger.error(
                "ONLYOFFICE callback: failed to download document from Document Server "
                "(status %d) for questionnaire %s",
                res.status_code,
                questionnaire_id,
            )
            return {"error": 1}

        edited_bytes = res.content
        logger.info(
            "ONLYOFFICE callback: downloaded %d bytes from Document Server for questionnaire %s",
            len(edited_bytes),
            questionnaire_id,
        )

        # ── Step 2: Upsert into Supabase Storage ─────────────────────────────
        # Use the same bare-path convention as _generate_and_persist_completed_docx:
        # no bucket prefix, upload_completed_docx() handles that internally.
        storage_path = f"{org_id}/{questionnaire_id}/completed_{questionnaire_id}.docx"
        await upload_completed_docx(storage_path, edited_bytes)

        # ── Step 3: Record timestamp + manual-edit flag in Postgres ──────────
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

        logger.info(
            "ONLYOFFICE callback: manual edits persisted for questionnaire %s (status=%d)",
            questionnaire_id,
            status,
        )

    except Exception as exc:
        logger.exception(
            "ONLYOFFICE callback: unexpected error processing save for questionnaire %s: %s",
            questionnaire_id,
            exc,
        )
        # Return error=1 so ONLYOFFICE will retry the callback.
        return {"error": 1}

    # ONLYOFFICE requires exactly {"error": 0} to confirm the callback was handled.
    return {"error": 0}


# ── 7. ONLYOFFICE Editor Configuration ───────────────────────────────────────

@router.get("/{questionnaire_id}/editor-config")
async def get_editor_config(
    questionnaire_id: str,
    user: AuthenticatedUser = Depends(require_responder_or_admin),
):
    """Returns the JSON configuration block for ``@onlyoffice/document-editor-react``.

    Auto-builds the completed DOCX if it has not been generated yet.
    The ``document.key`` revision key changes whenever the document is rebuilt,
    which forces ONLYOFFICE to reload the file from the new signed URL.
    """
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=500, detail="Database pool unavailable.")

    async with pool.acquire() as conn:
        q_row = await conn.fetchrow(
            """
            SELECT completed_file_path, completed_file_updated_at
            FROM questionnaires
            WHERE id = $1 AND organization_id = $2
            """,
            questionnaire_id,
            user.organization_id,
        )
        if not q_row:
            raise HTTPException(status_code=404, detail="Questionnaire not found.")

    storage_path: Optional[str] = q_row["completed_file_path"]
    completed_at: Optional[datetime] = q_row["completed_file_updated_at"]

    # Auto-build if document has never been generated
    if not storage_path:
        storage_path, _ = await _generate_and_persist_completed_docx(
            questionnaire_id, user.organization_id
        )
        # Refresh the timestamp after auto-build
        pool = get_pool()
        async with pool.acquire() as conn:
            completed_at = await conn.fetchval(
                "SELECT completed_file_updated_at FROM questionnaires WHERE id = $1",
                questionnaire_id,
            )

    signed_url = await get_docx_signed_url(storage_path)

    # Revision key: changes every time the document is rebuilt so ONLYOFFICE
    # knows to discard its cached version.
    ts = int(completed_at.timestamp()) if completed_at else int(time.time())
    rev_key = f"{questionnaire_id}_{ts}"

    callback_url = (
        f"{settings.api_base_url.rstrip('/')}"
        f"/api/v1/questionnaires/{questionnaire_id}/onlyoffice-callback"
    )

    return {
        "documentType": "word",
        "document": {
            "fileType": "docx",
            "key": rev_key,
            "title": f"completed_{questionnaire_id}.docx",
            "url": signed_url,
        },
        "editorConfig": {
            "mode": "edit",
            "callbackUrl": callback_url,
            "user": {
                "id": user.id,
                "name": user.email,
            },
        },
    }


# ── 8. Export / Download Endpoint ─────────────────────────────────────────────

@router.get("/{questionnaire_id}/export")
async def export_questionnaire(
    questionnaire_id: str,
    user: AuthenticatedUser = Depends(require_responder_or_admin),
):
    """Serves the completed DOCX as an attachment download.

    If the document has already been built and persisted in Supabase Storage,
    it is streamed directly from there.  If no persisted copy exists yet, the
    document is built on-demand and then stored before being served (so the
    next call is instant).

    This replaces the old POST /export endpoint that returned an ephemeral
    in-memory stream without persisting.
    """
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=500, detail="Database pool unavailable.")

    async with pool.acquire() as conn:
        q_row = await conn.fetchrow(
            """
            SELECT completed_file_path
            FROM questionnaires
            WHERE id = $1 AND organization_id = $2
            """,
            questionnaire_id,
            user.organization_id,
        )

    if not q_row:
        raise HTTPException(status_code=404, detail="Questionnaire not found.")

    storage_path: Optional[str] = q_row["completed_file_path"]

    # Build on demand if no persisted copy exists
    if not storage_path:
        storage_path, _ = await _generate_and_persist_completed_docx(
            questionnaire_id, user.organization_id
        )

    file_bytes = await download_file_bytes(storage_path)
    return Response(
        content=file_bytes,
        media_type=DOCX_MIME_TYPE,
        headers={
            "Content-Disposition": f'attachment; filename="completed_{questionnaire_id}.docx"',
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )


# ── 9. Retrieve Questionnaire Details ────────────────────────────────────────

@router.get("/{questionnaire_id}", response_model=QuestionnaireSummaryResponse)
async def get_questionnaire(
    questionnaire_id: str,
    user: AuthenticatedUser = Depends(require_responder_or_admin),
):
    """Retrieves metadata and all questions for a specific questionnaire."""
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=500, detail="Database pool unavailable.")

    async with pool.acquire() as conn:
        q_row = await conn.fetchrow(
            """
            SELECT id, title, filename, file_name, storage_path, file_url, status
            FROM questionnaires
            WHERE id = $1 AND organization_id = $2
            """,
            questionnaire_id,
            user.organization_id,
        )
        if not q_row:
            raise HTTPException(status_code=404, detail="Questionnaire not found.")

        questions = await conn.fetch(
            """
            SELECT id, question, coordinates, status, draft_answer, confidence
            FROM questions
            WHERE questionnaire_id = $1 AND organization_id = $2
            ORDER BY created_at ASC
            """,
            questionnaire_id,
            user.organization_id,
        )

    parsed_questions = []
    for q in questions:
        coords = q["coordinates"]
        if isinstance(coords, str):
            coords = json.loads(coords)
        parsed_questions.append(
            ExtractedQuestionResponse(
                id=q["id"],
                question=q["question"],
                coordinates=coords or {},
                status=q["status"] or "TODO",
                draft_answer=q["draft_answer"],
                confidence=q["confidence"],
            )
        )

    return QuestionnaireSummaryResponse(
        questionnaire_id=q_row["id"],
        title=q_row["title"] or q_row["filename"] or "Untitled Questionnaire",
        filename=q_row["filename"] or q_row["file_name"] or "questionnaire.docx",
        storage_path=q_row["storage_path"] or q_row["file_url"] or "",
        status=q_row["status"] or "PROCESSED",
        question_count=len(parsed_questions),
        questions=parsed_questions,
    )
