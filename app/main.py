import asyncio
import hashlib
import json
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.agent.graph import run_agent
from app.config import settings
from app.db.pool import close_pool, get_pool, init_pool
from app.rag.chunker import chunk_sections
from app.rag.embeddings import embed, embed_batch
from app.rag.enrichment import enrich_document_metadata
from app.rag.groq_client import init_groq, close_groq
from app.rag.guardrails import (
    extract_email_domains_from_text,
    scan_content_for_injection_patterns,
    scan_document_intelligently,
)
from app.rag.parser import parse_file
from app.rag.retrieval import mark_chunks_used

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_pool()
    await init_groq()
    yield
    await close_groq()
    await close_pool()


app = FastAPI(title="RFP Compliance Agent — RAG Core", lifespan=lifespan)

# CORS — allow the upload UI and any local frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Upload UI ─────────────────────────────────────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.get("/upload-ui", response_class=HTMLResponse)
async def upload_ui():
    """Serve the upload and testing interface."""
    html_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="Upload UI not found")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# ── File Upload (Parse → Chunk → Embed → Store) ──────────────────────
ALLOWED_EXTENSIONS = set(settings.allowed_extensions.split(","))


def _table_to_embedding_text(chunk) -> str:
    """Convert a table chunk into a richer natural-language representation for embedding."""
    lines = [line.strip() for line in chunk.content.strip().split("\n") if line.strip()]
    if len(lines) < 3:
        prefix = f"Section: {chunk.heading_path}\n" if chunk.heading_path else ""
        return f"{prefix}{chunk.content}"

    # Parse header row
    header_cells = [c.strip() for c in lines[0].split("|") if c.strip()]
    if not header_cells:
        prefix = f"Section: {chunk.heading_path}\n" if chunk.heading_path else ""
        return f"{prefix}{chunk.content}"

    row_sentences = []
    # Skip separator line (line 1), iterate data rows
    for line in lines[2:]:
        cells = [c.strip() for c in line.split("|")]
        if line.startswith("|") and cells:
            cells = cells[1:]
        if line.endswith("|") and cells:
            cells = cells[:-1]

        if not any(cells):
            continue

        row_parts = []
        for h, val in zip(header_cells, cells):
            if val:
                row_parts.append(f"{h}: {val}")

        if row_parts:
            row_sentences.append(", ".join(row_parts) + ".")

    heading_prefix = f"Section: {chunk.heading_path}.\n" if chunk.heading_path else ""
    natural_prose = " ".join(row_sentences)

    return f"{heading_prefix}{chunk.content}\n\nTable Summary / Rows:\n{natural_prose}".strip()


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    organization_id: str = Form(...),
):
    """Upload a document file, parse it, chunk it, embed chunks, and store
    everything in the RAG knowledge base. This is the full ingestion pipeline."""
    start_time = time.time()

    # Validate file extension
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Validate file size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {size_mb:.1f}MB. Max: {settings.max_upload_size_mb}MB",
        )

    # Compute SHA-256 hash of file content for duplicate detection
    content_hash = hashlib.sha256(content).hexdigest()

    # Validate organization exists
    pool = get_pool()
    async with pool.acquire() as conn:
        org = await conn.fetchrow(
            "SELECT id FROM organizations WHERE id = $1", organization_id
        )
    if org is None:
        raise HTTPException(status_code=404, detail="Organization not found")

    # Reject duplicate uploads
    async with pool.acquire() as conn:
        dup = await conn.fetchrow(
            """
            SELECT id, filename FROM rag_documents
            WHERE organization_id = $1
              AND is_archived = false
              AND (content_hash = $2 OR filename = $3)
            """,
            organization_id,
            content_hash,
            filename,
        )
    if dup is not None:
        dup_name = dup["filename"]
        if dup_name == filename:
            detail = f"A document named '{filename}' already exists in this organization's knowledge base."
        else:
            detail = f"This file's content is identical to an already-uploaded document ('{dup_name}')."
        raise HTTPException(status_code=409, detail=detail)

    # Save to temp file for parsing
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 1. Parse the file
        logger.info(f"Parsing {filename} ({size_mb:.1f}MB)...")
        sections = await asyncio.to_thread(parse_file, tmp_path, filename)
        if not sections:
            raise HTTPException(
                status_code=422,
                detail="No content could be extracted from this file. It may be empty or a scanned image.",
            )
        logger.info(f"Parsed {len(sections)} sections from {filename}")

        # Ingestion Document Poisoning Guardrail (Option a):
        # Run deterministic regex triage over full text before chunking or embedding.
        # If triggered, escalate to intelligent LLM scanner on targeted windows
        # around the suspicious matches to verify if it represents an actual attack payload.
        text_sections = [s for s in sections if s.content_type == "text"]
        full_text = "\n".join(s.content for s in text_sections)

        if full_text and scan_content_for_injection_patterns(full_text):
            logger.warning(
                "Suspicious prompt injection patterns detected in '%s' (org=%s). "
                "Escalating to intelligent scanner...",
                filename,
                organization_id,
            )
            is_injected, verdict = await scan_document_intelligently(full_text)
            if is_injected:
                risk_cat = verdict.risk_category if verdict else "instruction_override"
                reason = verdict.explanation if verdict else "Suspicious prompt injection payload detected"
                logger.warning(
                    "DOCUMENT_INJECTION_BLOCKED: '%s' rejected for org '%s'. category=%s reason=%s",
                    filename,
                    organization_id,
                    risk_cat,
                    reason,
                )

                # Persist audit record in rag_guardrail_events (question_id is NULL for ingestion events)
                try:
                    async with pool.acquire() as conn:
                        await conn.execute(
                            """
                            INSERT INTO rag_guardrail_events
                                (organization_id, question_id, event_type, payload)
                            VALUES ($1, NULL, $2, $3::jsonb)
                            """,
                            organization_id,
                            "DOCUMENT_INJECTION_BLOCKED",
                            json.dumps({
                                "filename": filename,
                                "risk_category": risk_cat,
                                "reason": reason,
                                "content_hash": content_hash,
                            }),
                        )
                except Exception as db_err:
                    logger.warning("Could not persist DOCUMENT_INJECTION_BLOCKED event to database: %s", db_err)

                raise HTTPException(
                    status_code=422,
                    detail=f"Document '{filename}' was rejected: Prompt injection payload detected ({risk_cat}).",
                )

        # 2. Chunk the sections
        chunks = await asyncio.to_thread(
            chunk_sections,
            sections,
            settings.chunk_size,
            settings.chunk_overlap,
        )
        logger.info(f"Created {len(chunks)} chunks from {filename}")

        # Determine source_type from extension
        source_type_map = {
            ".pdf": "policy",
            ".docx": "policy",
            ".xlsx": "spreadsheet",
            ".csv": "spreadsheet",
            ".txt": "text",
            ".md": "text",
        }
        source_type = source_type_map.get(ext, "other")

        # 3. Enrich metadata
        first_text = "\n".join(s.content for s in text_sections[:5])
        last_text = "\n".join(s.content for s in text_sections[-5:])
        enrichment = await enrich_document_metadata(filename, first_text, last_text, full_text=full_text)
        logger.info(f"Enrichment for {filename}: {enrichment}")

        effective_date_obj = None
        if enrichment["effective_date"]:
            try:
                effective_date_obj = datetime.strptime(enrichment["effective_date"], "%Y-%m-%d").date()
            except ValueError:
                logger.warning(f"Could not parse effective_date '{enrichment['effective_date']}' for {filename}")

        # 4. Embed all chunks in batch
        logger.info(f"Embedding {len(chunks)} chunks...")
        embedding_texts = []
        for c in chunks:
            if c.content_type == "table":
                embedding_texts.append(_table_to_embedding_text(c))
            else:
                prefix = f"Section: {c.heading_path}\n" if c.heading_path else ""
                embedding_texts.append(f"{prefix}{c.content}" if prefix else c.content)

        embeddings = await embed_batch(embedding_texts)

        # 5. Register document and store chunks in a single transaction
        async with pool.acquire() as conn:
            async with conn.transaction():
                doc_row = await conn.fetchrow(
                    """
                    INSERT INTO rag_documents
                        (organization_id, filename, source_type, category, tags,
                         effective_date, supersedes_label, content_hash)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING id
                    """,
                    organization_id,
                    filename,
                    source_type,
                    enrichment["category"],
                    enrichment["tags"],
                    effective_date_obj,
                    enrichment["supersedes_label"],
                    content_hash,
                )
                document_id = doc_row["id"]

                # 6. Store all chunks
                records = []
                for chunk, embedding in zip(chunks, embeddings):
                    meta_with_type = {**chunk.metadata, "content_type": chunk.content_type}
                    records.append((
                        organization_id,
                        document_id,
                        chunk.chunk_index,
                        chunk.content,
                        chunk.heading_path,
                        json.dumps(meta_with_type),
                        embedding,
                    ))

                await conn.executemany(
                    """
                    INSERT INTO rag_chunks
                        (organization_id, document_id, chunk_index, content,
                         heading_path, metadata, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
                    """,
                    records,
                )

                # 7. Extract corporate email domains and track tenant domain identity
                doc_domains = extract_email_domains_from_text(full_text)
                if doc_domains:
                    # A. Track document-domain association (cascades on document deletion)
                    doc_domain_records = [(organization_id, document_id, d) for d in doc_domains]
                    try:
                        await conn.executemany(
                            """
                            INSERT INTO rag_document_domains
                                (organization_id, document_id, domain)
                            VALUES ($1, $2, $3)
                            ON CONFLICT (document_id, domain) DO NOTHING
                            """,
                            doc_domain_records,
                        )
                    except Exception as doc_dom_err:
                        logger.debug("rag_document_domains insert skipped: %s", doc_dom_err)

                    # B. Batch upsert into rag_tenant_domains (eliminating N+1 sequential executes)
                    tenant_domain_records = [(organization_id, d) for d in doc_domains]
                    await conn.executemany(
                        """
                        INSERT INTO rag_tenant_domains
                            (organization_id, domain, times_seen, last_seen_at)
                        VALUES ($1, $2, 1, now())
                        ON CONFLICT (organization_id, domain)
                        DO UPDATE SET
                            times_seen = rag_tenant_domains.times_seen + 1,
                            last_seen_at = now()
                        """,
                        tenant_domain_records,
                    )
                    logger.info(
                        "Tracked %d corporate domain(s) in batch from %s: %s",
                        len(doc_domains),
                        filename,
                        doc_domains,
                    )

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(f"✅ Ingested {filename}: {len(chunks)} chunks in {elapsed_ms}ms")

        return {
            "document_id": document_id,
            "filename": filename,
            "chunks_created": len(chunks),
            "sections_parsed": len(sections),
            "processing_time_ms": elapsed_ms,
        }

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Document Management ───────────────────────────────────────────────
@app.get("/documents")
async def list_documents(organization_id: str):
    """List all RAG documents for an organization with chunk counts."""
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT d.id, d.filename, d.source_type, d.category, d.tags, d.created_at,
                   d.effective_date, d.supersedes_label, d.is_archived,
                   COUNT(c.id) AS chunks_count
            FROM rag_documents d
            LEFT JOIN rag_chunks c ON c.document_id = d.id
            WHERE d.organization_id = $1 AND d.is_archived = false
            GROUP BY d.id
            ORDER BY d.created_at DESC
            """,
            organization_id,
        )
    return [
        {
            "id": str(r["id"]),
            "filename": r["filename"],
            "source_type": r["source_type"],
            "category": r["category"],
            "tags": r["tags"] or [],
            "effective_date": r["effective_date"].isoformat() if r["effective_date"] else None,
            "supersedes_label": r["supersedes_label"],
            "chunks_count": r["chunks_count"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        }
        for r in rows
    ]


# ── KNOWN SECURITY GAP / ARCHITECTURE NOTE ───────────────────────────
# /documents/{document_id}/chunks currently returns raw rag_chunks.content
# with no sanitization at all. Anyone with the organization_id can read
# unredacted source text regardless of prompt-level or output-level guardrails.
# We intentionally do not silently patch this endpoint in Phase 2:
# 1. Admin and audit inspection workflows often require verifying raw chunk storage.
# 2. Applying context-stage or output-stage allowlisting here requires formal
#    scoping of read-access permissions and role boundaries (e.g. tenant admin
#    vs standard user) to avoid breaking legitimate data verification access.
# ─────────────────────────────────────────────────────────────────────
@app.get("/documents/{document_id}/chunks")
async def get_document_chunks(document_id: str, organization_id: str):
    """Get all chunks for a specific document."""
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, chunk_index, content, heading_path, metadata,
                   times_used, created_at
            FROM rag_chunks
            WHERE document_id = $1 AND organization_id = $2
            ORDER BY chunk_index
            """,
            document_id,
            organization_id,
        )
    if not rows:
        raise HTTPException(status_code=404, detail="Document not found or has no chunks")

    return [
        {
            "id": str(r["id"]),
            "chunk_index": r["chunk_index"],
            "content": r["content"],
            "heading_path": r["heading_path"],
            "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
            "times_used": r["times_used"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        }
        for r in rows
    ]


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str, organization_id: str):
    """Delete a document and all its chunks (cascade)."""
    pool = get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM rag_documents WHERE id = $1 AND organization_id = $2",
            document_id,
            organization_id,
        )
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Document not found for this organization")
    return {"status": "deleted", "document_id": document_id}


# ── Ask Schemas (Structured Citations & Safety Events) ────────────────
class SourceChunkMetadata(BaseModel):
    id: str
    document_id: str | None = None
    heading_path: str | None = None
    effective_date: str | None = None
    content_type: str = "text"
    content: str
    combined_score: float = 0.0


class AskTestRequest(BaseModel):
    question: str
    organization_id: str


class AskTestResponse(BaseModel):
    question: str
    answer: str
    confidence_score: float
    confidence_level: str
    confidence_breakdown: dict[str, Any] = Field(default_factory=dict)
    is_refusal: bool = False
    attempts: int = 1
    source_chunks: list[SourceChunkMetadata] = Field(default_factory=list)
    initial_validation_passed: bool | None = None
    guardrail_events: list[dict[str, Any]] = Field(default_factory=list)


class AskRequest(BaseModel):
    question_id: str          # Real questions.id from Prisma
    organization_id: str      # Required for org-scoped retrieval


class AskResponse(BaseModel):
    answer: str
    confidence_score: float
    confidence_level: str
    confidence_breakdown: dict[str, Any] = Field(default_factory=dict)
    is_refusal: bool = False
    attempts: int = 1
    source_chunk_ids: list[str] = Field(default_factory=list)
    initial_validation_passed: bool | None = None
    guardrail_events: list[dict[str, Any]] = Field(default_factory=list)


# ── Ask Endpoints ─────────────────────────────────────────────────────
@app.post("/ask/test", response_model=AskTestResponse)
async def ask_test(req: AskTestRequest):
    """Standalone ask endpoint for testing and benchmarking.
    Runs the full LangGraph agent pipeline and returns the draft answer
    along with confidence scoring, guardrail audit logs, and structured citation chunks.
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        org = await conn.fetchrow(
            "SELECT id FROM organizations WHERE id = $1", req.organization_id
        )
    if org is None:
        raise HTTPException(status_code=404, detail="Organization not found")

    result = await run_agent(req.question, organization_id=req.organization_id)
    chunks = result.get("chunks", [])

    # Mark chunks used only if retrieval executed (not blocked by input guardrail)
    if chunks:
        await mark_chunks_used([c.id for c in chunks])

    source_chunks = [
        SourceChunkMetadata(
            id=c.id,
            document_id=getattr(c, "document_id", None),
            heading_path=c.heading_path,
            effective_date=getattr(c, "effective_date", None) or "Undated",
            content_type=c.content_type,
            # Provide more room for markdown tables so table columns are preserved
            content=c.content[:2000] if c.content_type == "table" else c.content[:600],
            combined_score=round(getattr(c, "combined_score", 0.0), 4),
        )
        for c in chunks
    ]

    return AskTestResponse(
        question=req.question,
        answer=result.get("draft", ""),
        confidence_score=result.get("confidence_score", 0.0),
        confidence_level=result.get("confidence_level", "red"),
        confidence_breakdown=result.get("confidence_breakdown", {}),
        is_refusal=result.get("is_refusal", False),
        attempts=result.get("attempts", 1),
        source_chunks=source_chunks,
        initial_validation_passed=result.get("initial_validation_passed", True),
        guardrail_events=result.get("guardrail_events", []),
    )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """Runs the full agent pipeline for an existing RFP question row, then updates
    the questions record (draft_answer, confidence, status, updated_at) and logs
    history to rag_answer_attempts.
    """
    pool = get_pool()

    async with pool.acquire() as conn:
        question_row = await conn.fetchrow(
            "SELECT question FROM questions WHERE id = $1 AND organization_id = $2",
            req.question_id,
            req.organization_id,
        )
    if question_row is None:
        raise HTTPException(status_code=404, detail="Question not found for this organization")

    result = await run_agent(question_row["question"], organization_id=req.organization_id)
    chunks = result.get("chunks", [])
    chunk_ids = [c.id for c in chunks]

    if chunk_ids:
        await mark_chunks_used(chunk_ids)

    # Route status based on safety flags and confidence
    if result.get("injection_detected"):
        new_status = "FLAGGED"
    elif result.get("confidence_level") == "green":
        new_status = "IN_REVIEW"
    else:
        new_status = "IN_PROGRESS"

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                UPDATE questions
                SET draft_answer = $1, confidence = $2, status = $3, updated_at = now()
                WHERE id = $4 AND organization_id = $5
                """,
                result.get("draft", ""),
                int(round(result.get("confidence_score", 0.0))),
                new_status,
                req.question_id,
                req.organization_id,
            )
            await conn.execute(
                """
                INSERT INTO rag_answer_attempts
                    (question_id, answer_text, confidence_score, confidence_level,
                     source_chunk_ids, attempts)
                VALUES ($1, $2, $3, $4, $5::text[], $6)
                """,
                req.question_id,
                result.get("draft", ""),
                result.get("confidence_score", 0.0),
                result.get("confidence_level", "red"),
                chunk_ids,
                result.get("attempts", 1),
            )

            guardrail_events = result.get("guardrail_events", [])
            if guardrail_events:
                await conn.executemany(
                    """
                    INSERT INTO rag_guardrail_events
                        (organization_id, question_id, event_type, payload)
                    VALUES ($1, $2, $3, $4::jsonb)
                    """,
                    [
                        (
                            req.organization_id,
                            req.question_id,
                            ev.get("type", "UNKNOWN"),
                            json.dumps(ev),
                        )
                        for ev in guardrail_events
                    ],
                )

    return AskResponse(
        answer=result.get("draft", ""),
        confidence_score=result.get("confidence_score", 0.0),
        confidence_level=result.get("confidence_level", "red"),
        confidence_breakdown=result.get("confidence_breakdown", {}),
        is_refusal=result.get("is_refusal", False),
        attempts=result.get("attempts", 1),
        source_chunk_ids=chunk_ids,
        initial_validation_passed=result.get("initial_validation_passed", True),
        guardrail_events=result.get("guardrail_events", []),
    )


# ── Legacy Ingest Endpoints (kept for backward compat) ────────────────
class IngestChunkRequest(BaseModel):
    document_id: str
    organization_id: str
    chunk_index: int
    content: str
    heading_path: str | None = None
    metadata: dict = {}


@app.post("/ingest/chunk")
async def ingest_chunk(req: IngestChunkRequest):
    """Embeds and stores a single pre-chunked piece of text into the RAG knowledge base."""
    embedding = await embed(req.content)
    metadata_json = json.dumps(req.metadata)
    pool = get_pool()
    async with pool.acquire() as conn:
        doc = await conn.fetchrow(
            "SELECT id FROM rag_documents WHERE id = $1 AND organization_id = $2",
            req.document_id,
            req.organization_id,
        )
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found for this organization")

        row = await conn.fetchrow(
            """
            INSERT INTO rag_chunks
                (organization_id, document_id, chunk_index, content, heading_path, metadata, embedding)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
            RETURNING id
            """,
            req.organization_id,
            req.document_id,
            req.chunk_index,
            req.content,
            req.heading_path,
            metadata_json,
            embedding,
        )
    return {"chunk_id": row["id"]}


class CreateDocumentRequest(BaseModel):
    organization_id: str
    filename: str
    source_type: str
    category: str | None = None
    tags: list[str] = []


@app.post("/ingest/document")
async def create_document(req: CreateDocumentRequest):
    """Registers a source document before its chunks are ingested."""
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO rag_documents (organization_id, filename, source_type, category, tags)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
            """,
            req.organization_id,
            req.filename,
            req.source_type,
            req.category,
            req.tags,
        )
    return {"document_id": row["id"]}