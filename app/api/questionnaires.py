"""Questionnaire ingestion, AI answer generation, and in-place DOCX export routes."""

import asyncio
import io
import json
import logging
from typing import Any, Optional

from cuid2 import cuid_wrapper
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.graph import run_agent
from app.auth.dependencies import AuthenticatedUser, require_responder_or_admin
from app.db.pool import get_pool
from app.db.storage import (
    DOCX_MIME_TYPE,
    download_questionnaire_file,
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
                        $6, $7, $8::"QuestionnaireStatus", $9,
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
                        VALUES ($1, $2, $3::jsonb, $4::"QuestionStatus", $5, $6, $7, $8, now(), now())
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
            SET draft_answer = $1, confidence = $2, status = $3::"QuestionStatus", updated_at = now()
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
                        SET draft_answer = $1, confidence = $2, status = $3::"QuestionStatus", updated_at = now()
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



# ── 3. Format-Preserving In-Place Export Route ─────────────────────────────────

@router.post("/{questionnaire_id}/export")
async def export_questionnaire(
    questionnaire_id: str,
    user: AuthenticatedUser = Depends(require_responder_or_admin),
):
    """Retrieves original .docx from Supabase Storage, applies approved answers
    into designated coordinates with run-level precision, and returns the modified
    Word document via StreamingResponse.
    """
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=500, detail="Database pool unavailable.")

    async with pool.acquire() as conn:
        q_row = await conn.fetchrow(
            """
            SELECT id, title, filename, file_name, storage_path, file_url
            FROM questionnaires
            WHERE id = $1 AND organization_id = $2
            """,
            questionnaire_id,
            user.organization_id,
        )
        if not q_row:
            raise HTTPException(status_code=404, detail="Questionnaire not found.")

        storage_path = q_row["storage_path"] or q_row["file_url"]
        if not storage_path:
            raise HTTPException(
                status_code=404,
                detail="Original questionnaire storage path not recorded.",
            )

        filename = q_row["filename"] or q_row["file_name"] or "questionnaire.docx"

        # Fetch questions that have valid coordinates and non-empty answers
        question_rows = await conn.fetch(
            """
            SELECT id, question, coordinates, draft_answer
            FROM questions
            WHERE questionnaire_id = $1 AND organization_id = $2
              AND coordinates IS NOT NULL
              AND draft_answer IS NOT NULL AND draft_answer != ''
            """,
            questionnaire_id,
            user.organization_id,
        )

    # 1. Download original .docx from Supabase Storage
    original_bytes = await download_questionnaire_file(storage_path)

    # 2. Build answered items payload
    answered_items: list[dict] = []
    for row in question_rows:
        coords = row["coordinates"]
        if isinstance(coords, str):
            coords = json.loads(coords)
        answered_items.append({
            "coordinates": coords,
            "answer": row["draft_answer"],
        })

    logger.info(
        "Exporting questionnaire %s: mutating %d answered coordinates in-place",
        questionnaire_id,
        len(answered_items),
    )

    # 3. Surgical in-place run mutation
    mutated_bytes = write_docx_answers_in_place(original_bytes, answered_items)

    clean_filename = filename if filename.startswith("completed_") else f"completed_{filename}"

    return StreamingResponse(
        io.BytesIO(mutated_bytes),
        media_type=DOCX_MIME_TYPE,
        headers={
            "Content-Disposition": f'attachment; filename="{clean_filename}"',
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )


# ── 4. Retrieve Questionnaire Details ────────────────────────────────────────

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
