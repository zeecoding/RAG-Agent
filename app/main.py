import asyncio
import json
import hashlib
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.agent.graph import run_agent
from app.config import settings
from app.db.pool import close_pool, get_pool, init_pool
from app.rag.chunker import chunk_sections
from app.rag.embeddings import embed, embed_batch
from app.rag.enrichment import enrich_document_metadata
from app.rag.groq_client import init_groq, close_groq
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
# allow_credentials=True is intentionally NOT set: this API uses no cookies/
# auth, and browsers reject wildcard origin + credentials together anyway.
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
    """Convert a table chunk into a richer natural-language representation for embedding.

    Raw markdown tables (| col1 | col2 |) embed poorly against natural-language queries
    because embedding models are trained on prose, not pipe-delimited syntax.
    This generates natural-language context sentences from table rows and appends them
    to the raw markdown table (along with section heading context).

    The raw markdown table is still stored in rag_chunks.content for the LLM to see;
    this enriched text is only used for computing the embedding vector.
    """
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

    # Reject duplicate uploads — same content or same filename within the
    # same organization. Catches both exact re-uploads (same bytes) and
    # filename collisions that would confuse the knowledge base.
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

        # 3. Enrich metadata (category, tags, effective_date, supersedes_label)
        # from the first/last few text sections — footers/last pages carry
        # effective-date and version info as often as headers do.
        #
        # IMPORTANT: DOCX parsing produces one section PER PARAGRAPH, not
        # one per document region — a real metadata block (e.g. "Doc ID /
        # Version" on one line, "Issuing Authority" on the next) can easily
        # span 2-3 paragraphs. Taking only text_sections[-1] silently drops
        # everything but the very last paragraph, which is exactly the bug
        # that missed the Effective Date line in this test. Grab the last
        # (and first) 5 sections instead of just one.
        text_sections = [s for s in sections if s.content_type == "text"]
        first_text = "\n".join(s.content for s in text_sections[:5])
        last_text = "\n".join(s.content for s in text_sections[-5:])
        full_text = "\n".join(s.content for s in text_sections)
        enrichment = await enrich_document_metadata(filename, first_text, last_text, full_text=full_text)
        logger.info(f"Enrichment for {filename}: {enrichment}")

        # asyncpg requires an actual date object for a DATE column — it
        # will NOT parse a "YYYY-MM-DD" string for you, even with an
        # explicit ::date cast in the query. enrichment.py's own
        # validation already confirmed the string is well-formed, so this
        # parse is expected to succeed; the try/except is just a guard
        # against never letting an enrichment quirk fail the whole upload.
        effective_date_obj = None
        if enrichment["effective_date"]:
            try:
                effective_date_obj = datetime.strptime(enrichment["effective_date"], "%Y-%m-%d").date()
            except ValueError:
                logger.warning(f"Could not parse effective_date '{enrichment['effective_date']}' for {filename}")

        # 4. Embed all chunks in batch (CPU-bound local inference, done before acquiring DB connection)
        # Tables get natural-language context generated from columns/rows,
        # appended to the raw table and section heading, so semantic search
        # can find them against natural language questions.
        logger.info(f"Embedding {len(chunks)} chunks...")
        embedding_texts = []
        for c in chunks:
            if c.content_type == "table":
                embedding_texts.append(_table_to_embedding_text(c))
            else:
                prefix = f"Section: {c.heading_path}\n" if c.heading_path else ""
                embedding_texts.append(f"{prefix}{c.content}" if prefix else c.content)

        embeddings = await embed_batch(embedding_texts)

        # 5. Register document and store chunks in a single transaction (atomic)
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

                # Execute a single bulk insert
                await conn.executemany(
                    """
                    INSERT INTO rag_chunks
                        (organization_id, document_id, chunk_index, content,
                         heading_path, metadata, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
                    """,
                    records,
                )

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"✅ Ingested {filename}: {len(chunks)} chunks in {elapsed_ms}ms"
        )

        return {
            "document_id": document_id,
            "filename": filename,
            "chunks_created": len(chunks),
            "sections_parsed": len(sections),
            "processing_time_ms": elapsed_ms,
        }

    finally:
        # Cleanup temp file
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
            # Surfaced as a suggestion for the admin to act on — never
            # auto-archives anything. See enrichment.py for why.
            "supersedes_label": r["supersedes_label"],
            "chunks_count": r["chunks_count"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        }
        for r in rows
    ]


@app.get("/documents/{document_id}/chunks")
async def get_document_chunks(document_id: str, organization_id: str):
    """Get all chunks for a specific document. organization_id is required
    and enforced — without it, any org could read any other org's chunks
    by guessing a document_id."""
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
    """Delete a document and all its chunks (cascade). organization_id is
    required and enforced — without it, any org could delete any other
    org's document by guessing a document_id."""
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


# ── Ask (Standalone Test — no question_id required) ───────────────────
class AskTestRequest(BaseModel):
    question: str
    organization_id: str


class AskTestResponse(BaseModel):
    answer: str
    confidence_score: float
    confidence_level: str
    confidence_breakdown: dict
    attempts: int
    source_chunks: list[dict]


@app.post("/ask/test", response_model=AskTestResponse)
async def ask_test(req: AskTestRequest):
    """Standalone ask endpoint for testing — no question_id needed.
    Runs the full LangGraph agent loop and returns the answer with
    confidence scoring and source chunks."""
    # Verify org exists
    pool = get_pool()
    async with pool.acquire() as conn:
        org = await conn.fetchrow(
            "SELECT id FROM organizations WHERE id = $1", req.organization_id
        )
    if org is None:
        raise HTTPException(status_code=404, detail="Organization not found")

    result = await run_agent(req.question, organization_id=req.organization_id)
    chunks = result.get("chunks", [])
    await mark_chunks_used([c.id for c in chunks])

    source_chunks = [
        {
            "id": c.id,
            # Tables get more room so their data isn't cut off
            "content": c.content[:2000] if c.content_type == "table" else c.content[:500],
            "heading_path": c.heading_path,
            "content_type": c.content_type,
            "combined_score": round(c.combined_score, 4),
        }
        for c in chunks
    ]

    return AskTestResponse(
        answer=result["draft"],
        confidence_score=result["confidence_score"],
        confidence_level=result["confidence_level"],
        confidence_breakdown=result["confidence_breakdown"],
        attempts=result["attempts"],
        source_chunks=source_chunks,
    )


# ── Ask (Original — writes back to questions table) ───────────────────
class AskRequest(BaseModel):
    question_id: str          # the real questions.id from Prisma — required now,
                               # since the answer is written back onto this row
    organization_id: str      # required for org-scoped retrieval — never optional


class AskResponse(BaseModel):
    answer: str
    confidence_score: float
    confidence_level: str
    confidence_breakdown: dict
    attempts: int
    source_chunk_ids: list[str]


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """Runs the full agent loop for an existing question, then writes the
    result directly onto the real `questions` row (draft_answer, confidence,
    status) — the same row your Next.js app already reads from. No separate
    answers table; rag_answer_attempts below is history only."""
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
    chunk_ids = [c.id for c in result.get("chunks", [])]
    await mark_chunks_used(chunk_ids)

    new_status = "IN_REVIEW" if result["confidence_level"] == "green" else "IN_PROGRESS"

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                UPDATE questions
                SET draft_answer = $1, confidence = $2, status = $3, updated_at = now()
                WHERE id = $4 AND organization_id = $5
                """,
                result["draft"],
                int(result["confidence_score"]),
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
                result["draft"],
                result["confidence_score"],
                result["confidence_level"],
                chunk_ids,
                result["attempts"],
            )

    return AskResponse(
        answer=result["draft"],
        confidence_score=result["confidence_score"],
        confidence_level=result["confidence_level"],
        confidence_breakdown=result["confidence_breakdown"],
        attempts=result["attempts"],
        source_chunk_ids=chunk_ids,
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
    """Embeds and stores a single pre-chunked piece of text into the RAG
    knowledge base (rag_chunks) — separate from `questionnaires`, which is
    the incoming RFP being answered, not the source material the agent
    retrieves from."""
    embedding = await embed(req.content)
    metadata_json = json.dumps(req.metadata)
    pool = get_pool()
    async with pool.acquire() as conn:
        # Confirm the document exists and belongs to this org before
        # attaching a chunk to it — cheap check, prevents cross-org writes.
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
    """Registers a source document before its chunks are ingested. Call
    this once per uploaded policy/past-answer file, then /ingest/chunk
    for each chunk produced from it."""
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
