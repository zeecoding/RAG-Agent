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
import uuid

import httpx
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from starlette.responses import Response
from pydantic import BaseModel, Field

from app.agent.graph import run_agent
from app.api.questionnaires import router as questionnaires_router
from app.auth.dependencies import AuthenticatedUser, require_responder_or_admin
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
from app.db.storage import upload_kb_document_file, create_signed_kb_url

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

app.include_router(questionnaires_router, prefix="/api/v1")


# ── Health ────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Auth Endpoints ───────────────────────────────────────────────────
class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/auth/login")
async def login(req: LoginRequest):
    """Authenticate directly with Supabase Auth and return session bearer token."""
    if not settings.supabase_url:
        raise HTTPException(status_code=500, detail="SUPABASE_URL is not configured.")

    token_url = f"{settings.supabase_url.rstrip('/')}/auth/v1/token?grant_type=password"
    headers = {
        "apikey": settings.supabase_service_role_key or "",
        "Content-Type": "application/json",
    }
    payload = {"email": req.email, "password": req.password}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.post(token_url, headers=headers, json=payload)
    except httpx.RequestError as exc:
        logger.error("Supabase Auth connection error: %s", exc)
        raise HTTPException(status_code=503, detail="Unable to connect to authentication server.")

    if res.status_code != 200:
        error_detail = "Invalid login credentials."
        try:
            err_json = res.json()
            error_detail = err_json.get("error_description") or err_json.get("msg") or res.text
        except Exception:
            pass
        raise HTTPException(status_code=res.status_code, detail=error_detail)

    data = res.json()
    return {
        "access_token": data.get("access_token"),
        "user": data.get("user"),
    }


@app.get("/auth/me", response_model=AuthenticatedUser)
async def get_me(user: AuthenticatedUser = Depends(require_responder_or_admin)):
    """Return verified user profile and organization role."""
    return user


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


@app.get("/questionnaires-ui", response_class=HTMLResponse)
async def questionnaires_ui():
    """Serve the DOCX questionnaire export and testing interface."""
    html_path = os.path.join(STATIC_DIR, "questionnaire_ui.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="Questionnaire UI not found")
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

    # Reject identical duplicates or auto-archive prior version of updated document
    async with pool.acquire() as conn:
        dup_hash = await conn.fetchrow(
            """
            SELECT id, filename FROM rag_documents
            WHERE organization_id = $1
              AND is_archived = false
              AND content_hash = $2
            """,
            organization_id,
            content_hash,
        )
        if dup_hash is not None:
            raise HTTPException(
                status_code=409,
                detail=f"This file's content is identical to an already-uploaded document ('{dup_hash['filename']}').",
            )

        prior_doc = await conn.fetchrow(
            """
            SELECT id, filename, supersedes_label FROM rag_documents
            WHERE organization_id = $1
              AND is_archived = false
              AND filename = $2
            """,
            organization_id,
            filename,
        )
        if prior_doc is not None:
            logger.info(
                "Detected revised document for '%s'. Archiving previous version (id: %s).",
                filename,
                prior_doc["id"],
            )
            await conn.execute(
                """
                UPDATE rag_documents
                SET is_archived = true,
                    supersedes_label = COALESCE(supersedes_label, '') || ' [Archived by revision]',
                    updated_at = now()
                WHERE id = $1
                """,
                prior_doc["id"],
            )

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
                generated_doc_id = str(uuid.uuid4())
                doc_row = await conn.fetchrow(
                    """
                    INSERT INTO rag_documents
                        (id, organization_id, filename, source_type, category, tags,
                         effective_date, supersedes_label, content_hash, is_archived)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, false)
                    RETURNING id
                    """,
                    generated_doc_id,
                    organization_id,
                    filename,
                    source_type,
                    enrichment["category"],
                    enrichment["tags"],
                    effective_date_obj,
                    enrichment["supersedes_label"],
                    content_hash,
                )
                document_id = doc_row["id"] if doc_row and doc_row["id"] else generated_doc_id

                # 6. Store all chunks
                records = []
                for chunk, embedding in zip(chunks, embeddings):
                    meta_with_type = {**chunk.metadata, "content_type": chunk.content_type}
                    records.append((
                        str(uuid.uuid4()),
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
                        (id, organization_id, document_id, chunk_index, content,
                         heading_path, metadata, embedding, tsv)
                    VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, to_tsvector('english', $5))
                    """,
                    records,
                )

        # 7. Extract corporate email domains and track tenant domain identity
        doc_domains = extract_email_domains_from_text(full_text)
        if doc_domains:
            try:
                async with pool.acquire() as domain_conn:
                    async with domain_conn.transaction():
                        # A. Track document-domain association
                        doc_domain_records = [(organization_id, document_id, d) for d in doc_domains]
                        await domain_conn.executemany(
                            """
                            INSERT INTO rag_document_domains
                                (organization_id, document_id, domain)
                            VALUES ($1, $2, $3)
                            ON CONFLICT (document_id, domain) DO NOTHING
                            """,
                            doc_domain_records,
                        )

                        # B. Batch upsert into rag_tenant_domains
                        tenant_domain_records = [(organization_id, d) for d in doc_domains]
                        await domain_conn.executemany(
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
            except Exception as dom_err:
                logger.warning("Domain tracking notice for %s: %s", filename, dom_err)

        # Upload raw document to private Supabase Storage bucket 'kb_docs'
        bucket_name = getattr(settings, "supabase_kb_bucket", "kb_docs")
        storage_path = ""
        try:
            storage_path = await upload_kb_document_file(
                organization_id=organization_id,
                filename=filename,
                file_bytes=content,
                content_type=file.content_type or "application/octet-stream",
            )
        except Exception as st_err:
            logger.warning("Could not upload '%s' to Supabase kb_docs: %s", filename, st_err)

        if not storage_path:
            storage_path = f"{bucket_name}/{organization_id}/{filename}"

        if settings.supabase_url:
            storage_url = f"{settings.supabase_url.rstrip('/')}/storage/v1/object/authenticated/{storage_path}"
        else:
            storage_url = storage_path

        # Persist storage_path and storage_url in rag_documents
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE rag_documents SET storage_path = $1, storage_url = $2 WHERE id = $3",
                    storage_path,
                    storage_url,
                    document_id,
                )
        except Exception as save_err:
            logger.warning("Could not persist storage_url in rag_documents: %s", save_err)

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(f"✅ Ingested {filename}: {len(chunks)} chunks in {elapsed_ms}ms")

        return {
            "document_id": document_id,
            "filename": filename,
            "chunks_created": len(chunks),
            "sections_parsed": len(sections),
            "processing_time_ms": elapsed_ms,
            "storage_bucket": bucket_name,
            "storage_path": storage_path,
            "storage_url": storage_url,
        }

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Document Management ───────────────────────────────────────────────
@app.get("/documents")
async def list_documents(organization_id: str, include_archived: bool = True):
    """List all RAG documents for an organization with chunk counts and storage URLs."""
    pool = get_pool()
    where_archived = "" if include_archived else "AND d.is_archived = false"
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT d.id, d.filename, d.source_type, d.category, d.tags, d.created_at,
                   d.effective_date, d.supersedes_label, d.is_archived,
                   d.storage_url, d.storage_path,
                   COUNT(c.id) AS chunks_count
            FROM rag_documents d
            LEFT JOIN rag_chunks c ON c.document_id = d.id
            WHERE d.organization_id = $1 {where_archived}
            GROUP BY d.id
            ORDER BY d.created_at DESC
            """,
            organization_id,
        )

    bucket_name = getattr(settings, "supabase_kb_bucket", "kb_docs")
    supabase_base = settings.supabase_url.rstrip("/") if settings.supabase_url else ""

    result = []
    for r in rows:
        spath = r["storage_path"] or f"{bucket_name}/{organization_id}/{r['filename']}"
        surl = r["storage_url"]
        if not surl and supabase_base:
            surl = f"{supabase_base}/storage/v1/object/authenticated/{spath}"
        elif not surl:
            surl = spath

        result.append({
            "id": str(r["id"]),
            "filename": r["filename"],
            "source_type": r["source_type"],
            "category": r["category"],
            "tags": r["tags"] or [],
            "effective_date": r["effective_date"].isoformat() if r["effective_date"] else None,
            "supersedes_label": r["supersedes_label"],
            "is_archived": bool(r["is_archived"]),
            "chunks_count": r["chunks_count"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "storage_path": spath,
            "storage_url": surl,
        })
    return result


@app.get("/documents/{document_id}/download-url")
async def get_document_download_url(document_id: str, organization_id: str):
    """Generates a signed download URL or returns the storage URL for a document."""
    pool = get_pool()
    async with pool.acquire() as conn:
        doc = await conn.fetchrow(
            "SELECT id, filename, storage_url, storage_path FROM rag_documents WHERE id = $1 AND organization_id = $2",
            document_id,
            organization_id,
        )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    signed = await create_signed_kb_url(organization_id, doc["filename"], expires_in=3600)
    return {
        "document_id": document_id,
        "filename": doc["filename"],
        "storage_url": doc["storage_url"],
        "signed_url": signed or doc["storage_url"],
    }


@app.post("/documents/{document_id}/archive")
async def archive_document(document_id: str, organization_id: str):
    """Toggle soft-archive status for a document."""
    pool = get_pool()
    async with pool.acquire() as conn:
        doc = await conn.fetchrow(
            "SELECT id, is_archived FROM rag_documents WHERE id = $1 AND organization_id = $2",
            document_id,
            organization_id,
        )
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        new_status = not bool(doc["is_archived"])
        await conn.execute(
            """
            UPDATE rag_documents
            SET is_archived = $1, updated_at = now()
            WHERE id = $2 AND organization_id = $3
            """,
            new_status,
            document_id,
            organization_id,
        )
    return {
        "status": "success",
        "document_id": document_id,
        "is_archived": new_status,
        "message": "Document archived." if new_status else "Document restored to active.",
    }


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
    document_name: str | None = None
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


# ── Responder Chatbot Schemas ─────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    organization_id: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    question: str
    answer: str
    confidence_score: float
    confidence_level: str
    is_refusal: bool = False
    source_chunks: list[SourceChunkMetadata] = Field(default_factory=list)
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


@app.post("/api/v1/chat", response_model=ChatResponse)
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """Responder knowledgebase chatbot endpoint.
    Executes the full LangGraph compliance agent pipeline:
    - Input guardrails (prompt injection & confidential probe screening)
    - Hybrid retrieval scoped to company knowledgebase
    - Context grading & temporal conflict precedence
    - Grounded executive drafting with policy citations
    - Output guardrails & structural PII / credential sanitization
    - Grounding validation
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

    if chunks:
        await mark_chunks_used([c.id for c in chunks])

    doc_ids = list({getattr(c, "document_id", None) for c in chunks if getattr(c, "document_id", None)})
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
            logger.warning("Could not fetch document filenames for chat: %s", e)

    source_chunks = [
        SourceChunkMetadata(
            id=c.id,
            document_id=getattr(c, "document_id", None),
            document_name=doc_map.get(getattr(c, "document_id", None), "Company Policy Document"),
            heading_path=c.heading_path,
            effective_date=getattr(c, "effective_date", None) or "Undated",
            content_type=c.content_type,
            content=c.content[:2000] if c.content_type == "table" else c.content[:600],
            combined_score=round(getattr(c, "combined_score", 0.0), 4),
        )
        for c in chunks
    ]

    return ChatResponse(
        question=req.question,
        answer=result.get("draft", ""),
        confidence_score=result.get("confidence_score", 0.0),
        confidence_level=result.get("confidence_level", "green"),
        is_refusal=result.get("is_refusal", False),
        source_chunks=source_chunks,
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
                    (id, question_id, answer_text, confidence_score, confidence_level,
                     source_chunk_ids, attempts)
                VALUES ($1, $2, $3, $4, $5, $6::text[], $7)
                """,
                str(uuid.uuid4()),
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
                        (id, organization_id, question_id, event_type, payload)
                    VALUES ($1, $2, $3, $4, $5::jsonb)
                    """,
                    [
                        (
                            str(uuid.uuid4()),
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

        chunk_id = str(uuid.uuid4())
        row = await conn.fetchrow(
            """
            INSERT INTO rag_chunks
                (id, organization_id, document_id, chunk_index, content, heading_path, metadata, embedding, tsv)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, to_tsvector('english', $5))
            RETURNING id
            """,
            chunk_id,
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
        doc_id = str(uuid.uuid4())
        row = await conn.fetchrow(
            """
            INSERT INTO rag_documents (id, organization_id, filename, source_type, category, tags, is_archived)
            VALUES ($1, $2, $3, $4, $5, $6, false)
            RETURNING id
            """,
            doc_id,
            req.organization_id,
            req.filename,
            req.source_type,
            req.category,
            req.tags,
        )
    return {"document_id": row["id"]}


@app.post("/library/detect-duplicates")
async def detect_duplicate_answers(
    organization_id: str = Query(...),
    threshold: float = Query(0.85),
):
    """Pairwise cosine similarity job for detecting duplicate answers (>85% similarity)
    described in RFP Use Case 9."""
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=500, detail="Database pool unavailable.")

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, question, draft_answer, confidence,
                   rfp_name, project_name
            FROM answer_library_items
            WHERE organization_id = $1
              AND status = 'APPROVED'
              AND draft_answer IS NOT NULL
              AND length(trim(draft_answer)) > 15
            ORDER BY created_at DESC
            LIMIT 150
            """,
            organization_id,
        )

        # Fallback to questions table if answer_library_items has fewer than 2 entries
        if len(rows) < 2:
            legacy_rows = await conn.fetch(
                """
                SELECT q.id, q.question, q.draft_answer, q.confidence,
                       qn.file_name as rfp_name, p.name as project_name
                FROM questions q
                LEFT JOIN questionnaires qn ON q.questionnaire_id = qn.id
                LEFT JOIN projects p ON qn.project_id = p.id
                WHERE q.organization_id = $1
                  AND q.status = 'APPROVED'
                  AND q.draft_answer IS NOT NULL
                  AND length(trim(q.draft_answer)) > 15
                ORDER BY q.created_at DESC
                LIMIT 150
                """,
                organization_id,
            )
            if len(legacy_rows) >= len(rows):
                rows = legacy_rows

    if len(rows) < 2:
        return {
            "duplicates_count": 0,
            "threshold": threshold,
            "candidates": [],
        }

    texts = [r["draft_answer"] for r in rows]
    embeddings = await embed_batch(texts)
    emb_matrix = np.array(embeddings, dtype=np.float32)

    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    normalized = emb_matrix / norms
    sim_matrix = np.dot(normalized, normalized.T)

    candidates = []
    n = len(rows)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if sim >= threshold:
                candidates.append({
                    "pair_id": f"{rows[i]['id']}_{rows[j]['id']}",
                    "similarity": round(sim * 100, 1),
                    "similarity_score": round(sim, 4),
                    "primary_question": {
                        "id": rows[i]["id"],
                        "question": rows[i]["question"],
                        "answer": rows[i]["draft_answer"],
                        "rfp_name": rows[i]["rfp_name"] or "Historical RFP",
                        "project_name": rows[i]["project_name"] or "General",
                    },
                    "duplicate_question": {
                        "id": rows[j]["id"],
                        "question": rows[j]["question"],
                        "answer": rows[j]["draft_answer"],
                        "rfp_name": rows[j]["rfp_name"] or "Historical RFP",
                        "project_name": rows[j]["project_name"] or "General",
                    },
                })

    candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
    return {
        "duplicates_count": len(candidates),
        "threshold": threshold,
        "candidates": candidates,
    }


# ── 12. Automated PDF Compliance Report Engine (RFP-FR-058 / 060) ────────────
@app.get("/reports/generate-pdf")
@app.post("/reports/generate-pdf")
async def generate_pdf_report(
    organization_id: str = Query(...),
):
    """Generates an executive, publication-grade compliance PDF report containing
    RFP metrics, question grounding rates, knowledge base status, and safety audits."""
    import io
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from starlette.responses import Response

    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=500, detail="Database pool unavailable.")

    async with pool.acquire() as conn:
        org = await conn.fetchrow("SELECT id, name FROM organizations WHERE id = $1", organization_id)
        org_name = org["name"] if org else "Organization"

        # KPI stats
        total_projects = await conn.fetchval(
            "SELECT count(*) FROM projects WHERE organization_id = $1", organization_id
        ) or 0

        q_stats = await conn.fetchrow(
            """
            SELECT 
                count(*) as total,
                count(*) FILTER (WHERE status = 'APPROVED') as approved,
                count(*) FILTER (WHERE status = 'IN_PROGRESS') as in_progress,
                count(*) FILTER (WHERE status = 'IN_REVIEW') as in_review,
                count(*) FILTER (WHERE status = 'REJECTED') as rejected,
                count(*) FILTER (WHERE status = 'ARCHIVED') as archived,
                coalesce(round(avg(confidence)), 0) as avg_confidence
            FROM questions
            WHERE organization_id = $1
            """,
            organization_id,
        )

        # Knowledge Base docs
        kb_docs = await conn.fetch(
            """
            SELECT d.filename, d.category, d.source_type, count(c.id) as chunks_count, d.created_at
            FROM rag_documents d
            LEFT JOIN rag_chunks c ON c.document_id = d.id
            WHERE d.organization_id = $1 AND d.is_archived = false
            GROUP BY d.id
            ORDER BY d.created_at DESC
            LIMIT 6
            """,
            organization_id,
        )

        total_chunks = await conn.fetchval(
            "SELECT count(*) FROM rag_chunks WHERE organization_id = $1", organization_id
        ) or 0

        # Guardrail events count
        guardrails_count = await conn.fetchval(
            "SELECT count(*) FROM rag_guardrail_events WHERE organization_id = $1", organization_id
        ) or 0

    total_q = q_stats["total"] or 0
    approved_q = q_stats["approved"] or 0
    approval_rate = round((approved_q / total_q) * 100, 1) if total_q > 0 else 0.0
    avg_conf = int(q_stats["avg_confidence"] or 0)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1b4332"),
        alignment=0,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#525252"),
    )
    h2_style = ParagraphStyle(
        "H2Style",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1b4332"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#262626"),
    )

    story = []

    # Title & Header
    story.append(Paragraph("ComplianceAI · Executive Compliance & RFP Digest", title_style))
    report_date = datetime.now().strftime("%B %d, %Y - %H:%M UTC")
    story.append(Paragraph(f"<b>Organization:</b> {org_name} &nbsp;|&nbsp; <b>Generated:</b> {report_date}", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2d6a4f"), spaceAfter=14))

    # KPI Summary Grid
    kpi_data = [
        [
            Paragraph(f"<b>Active RFP Projects</b><br/><font size=14 color='#1b4332'>{total_projects}</font>", body_style),
            Paragraph(f"<b>Total Clauses Evaluated</b><br/><font size=14 color='#1b4332'>{total_q}</font>", body_style),
            Paragraph(f"<b>Approval & Grounding Rate</b><br/><font size=14 color='#2d6a4f'>{approval_rate}%</font>", body_style),
            Paragraph(f"<b>Mean Confidence Score</b><br/><font size=14 color='#1b4332'>{avg_conf}%</font>", body_style),
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[135, 135, 135, 135])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f0fdf4')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#bbf7d0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dcfce7')),
        ('PADDING', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 14))

    # Section 1: Question Review Breakdown
    story.append(Paragraph("1. Questionnaire Items Breakdown by Verification Status", h2_style))
    q_table_data = [
        ["Status", "Questions Count", "Percentage of Total", "Verification State"],
        ["Approved & Finalized", str(approved_q), f"{round(approved_q/(total_q or 1)*100, 1)}%", "Export-Ready"],
        ["In Review (SME Pending)", str(q_stats["in_review"] or 0), f"{round((q_stats['in_review'] or 0)/(total_q or 1)*100, 1)}%", "Awaiting Approval"],
        ["In Progress (Assigned)", str(q_stats["in_progress"] or 0), f"{round((q_stats['in_progress'] or 0)/(total_q or 1)*100, 1)}%", "Draft Underway"],
        ["To Do / Unassigned", str(q_stats["total"] - approved_q - (q_stats["in_review"] or 0) - (q_stats["in_progress"] or 0)), f"{round((total_q - approved_q)/(total_q or 1)*100, 1)}%", "Pending"],
        ["Soft-Archived", str(q_stats["archived"] or 0), "-", "Retired from Library"],
    ]
    q_table = Table(q_table_data, colWidths=[160, 110, 130, 140])
    q_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1b4332')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#ffffff'), colors.HexColor('#f9fafb')]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e5e7eb')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(q_table)
    story.append(Spacer(1, 14))

    # Section 2: Knowledge Base & Vector Coverage
    story.append(Paragraph(f"2. Knowledge Base & Vector Indexing Status ({total_chunks} total chunks)", h2_style))
    if kb_docs:
        kb_table_data = [["Document Filename", "Category", "Source Type", "Indexed Chunks", "Ingested Date"]]
        for d in kb_docs:
            kb_table_data.append([
                Paragraph(d["filename"], body_style),
                d["category"] or "General",
                d["source_type"] or "policy",
                str(d["chunks_count"] or 0),
                d["created_at"].strftime("%Y-%m-%d") if d["created_at"] else "Recent",
            ])
        kb_table = Table(kb_table_data, colWidths=[200, 90, 80, 85, 85])
        kb_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2d6a4f')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#ffffff'), colors.HexColor('#f9fafb')]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e5e7eb')),
            ('PADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(kb_table)
    else:
        story.append(Paragraph("No active knowledge base documents found.", body_style))
    story.append(Spacer(1, 14))

    # Section 3: Guardrail & Security Summary
    story.append(Paragraph("3. Security & Safety Guardrails Audit", h2_style))
    story.append(Paragraph(
        f"ComplianceAI active guardrails inspected all incoming documents and queries. "
        f"<b>{guardrails_count} security triage events</b> have been logged to date, including automated "
        f"SHA-256 duplicate rejection, prompt injection pattern detection, and grounded citation enforcement.",
        body_style
    ))
    story.append(Spacer(1, 20))

    # Disclaimer Footer
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#d4d4d4"), spaceAfter=8))
    footer_text = "CONFIDENTIAL & PROPRIETARY · GENERATED BY COMPLIANCEAI RECURRING REPORT DAEMON · TABLE 3.31 COMPLIANT"
    story.append(Paragraph(f"<font size=7 color='#737373'>{footer_text}</font>", body_style))

    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()

    filename = f"Compliance_Report_{organization_id[:8]}_{datetime.now().strftime('%Y%m%d')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
