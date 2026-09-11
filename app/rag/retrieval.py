import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime

from app.db.pool import get_pool
from app.rag.embeddings import embed


@dataclass
class RetrievedChunk:
    id: str
    document_id: str
    content: str
    heading_path: str | None
    vector_score: float
    keyword_score: float
    combined_score: float
    content_type: str = "text"
    created_at: str | None = None
    times_used: int = 0
    effective_date: str | None = None


def sanitize_query(query: str) -> str:
    """Strip unsafe control characters (:, ;, *, \\0, |) while preserving
    alphanumeric text, spaces, hyphens, and standard double quotes.
    """
    cleaned = re.sub(r"[:;*\x00|]", " ", query)
    return " ".join(cleaned.split())


# Reciprocal Rank Fusion constant — standard default from the original RRF
# paper (Cormack, Clarke & Buettcher 2009). Controls how quickly lower-ranked
# results' contributions decay. k=60 is broadly suitable; no per-corpus
# tuning needed in practice.
RRF_K = 60


def _calculate_recency_multiplier(effective_date_val) -> float:
    """Calculates up to a 25% rank boost for active documents effective within
    the last 365 days, decaying exponentially toward 1.0 (neutral) for older documents.
    """
    if not effective_date_val:
        return 1.0

    if isinstance(effective_date_val, str):
        try:
            doc_date = datetime.strptime(effective_date_val, "%Y-%m-%d").date()
        except ValueError:
            return 1.0
    elif isinstance(effective_date_val, date):
        doc_date = effective_date_val
    else:
        return 1.0

    days_old = max(0, (date.today() - doc_date).days)
    boost = 0.25 * math.exp(-days_old / 365.0)
    return 1.0 + boost


# Hybrid search: rank chunks by Reciprocal Rank Fusion (RRF) of
#  - vector similarity rank (ordered by cosine distance to query embedding)
#  - keyword relevance rank (ordered by ts_rank against a websearch_to_tsquery)
#
# RRF combines RANKS, not raw scores, so the different scales of cosine
# similarity (~0-1, compressed near top for BGE) and ts_rank (unbounded)
# don't cause the arbitrary weighting problems that a linear blend had.
#
# organization_id is filtered on BOTH CTEs — this is the line that stops
# one organization's RAG agent from ever retrieving another organization's
# confidential policy/answer documents. Never remove this filter.
#
# is_archived = false is joined in the same way — this is the line that
# stops an outdated/superseded document from ever being retrieved
# alongside its current replacement. Without it, two conflicting policy
# documents (e.g. an old and new PTO policy) can both surface for the
# same question, and the LLM has no reliable way to know which is current.
_HYBRID_SQL = """
WITH vector_ranked AS (
    SELECT c.id, c.document_id, c.content, c.heading_path, c.created_at, c.times_used,
           c.metadata, d.effective_date,
           1 - (c.embedding <=> $1::vector) AS vector_score,
           ROW_NUMBER() OVER (ORDER BY c.embedding <=> $1::vector) AS vector_rank
    FROM rag_chunks c
    JOIN rag_documents d ON d.id = c.document_id
    WHERE c.organization_id = $4 AND d.is_archived = false
    ORDER BY c.embedding <=> $1::vector
    LIMIT $2
),
keyword_ranked AS (
    SELECT c.id, c.document_id, c.content, c.heading_path, c.created_at, c.times_used,
           c.metadata, d.effective_date,
           ts_rank(c.tsv, websearch_to_tsquery('english', $3)) AS keyword_score,
           ROW_NUMBER() OVER (ORDER BY ts_rank(c.tsv, websearch_to_tsquery('english', $3)) DESC) AS keyword_rank
    FROM rag_chunks c
    JOIN rag_documents d ON d.id = c.document_id
    WHERE c.organization_id = $4 AND d.is_archived = false
      AND c.tsv @@ websearch_to_tsquery('english', $3)
    ORDER BY keyword_score DESC
    LIMIT $2
)
SELECT
    COALESCE(v.id, k.id) AS id,
    COALESCE(v.document_id, k.document_id) AS document_id,
    COALESCE(v.content, k.content) AS content,
    COALESCE(v.heading_path, k.heading_path) AS heading_path,
    COALESCE(v.created_at, k.created_at) AS created_at,
    COALESCE(v.times_used, k.times_used) AS times_used,
    COALESCE(v.metadata, k.metadata) AS metadata,
    COALESCE(v.effective_date, k.effective_date) AS effective_date,
    COALESCE(v.vector_score, 0) AS vector_score,
    COALESCE(k.keyword_score, 0) AS keyword_score,
    v.vector_rank,
    k.keyword_rank
FROM vector_ranked v
FULL OUTER JOIN keyword_ranked k ON v.id = k.id
"""


async def hybrid_search(
    query: str, organization_id: str, top_k: int | None = None
) -> list[RetrievedChunk]:
    from app.config import settings
    top_k = top_k or settings.top_k_chunks
    pool = get_pool()
    sanitized_query = sanitize_query(query)
    query_embedding = await embed(query)

    async with pool.acquire() as conn:
        rows = await conn.fetch(_HYBRID_SQL, query_embedding, top_k, sanitized_query, organization_id)

    results: list[RetrievedChunk] = []
    for r in rows:
        # Reciprocal Rank Fusion: combine rank positions, not raw scores.
        # A chunk absent from one list gets no contribution from that list.
        rrf_score = 0.0
        if r["vector_rank"] is not None:
            rrf_score += 1.0 / (RRF_K + r["vector_rank"])
        if r["keyword_rank"] is not None:
            rrf_score += 1.0 / (RRF_K + r["keyword_rank"])

        # Temporal boost: up to 25% higher RRF for recent docs
        recency_mult = _calculate_recency_multiplier(r["effective_date"])
        final_score = rrf_score * recency_mult

        # Parse content_type from chunk metadata (set by chunker at ingestion)
        chunk_meta = r["metadata"]
        if isinstance(chunk_meta, str):
            try:
                chunk_meta = json.loads(chunk_meta)
            except (json.JSONDecodeError, TypeError):
                chunk_meta = {}
        elif chunk_meta is None:
            chunk_meta = {}
        content_type = chunk_meta.get("content_type", "text") if isinstance(chunk_meta, dict) else "text"

        effective_date_str = r["effective_date"].isoformat() if r["effective_date"] else None

        results.append(
            RetrievedChunk(
                id=str(r["id"]),
                document_id=str(r["document_id"]),
                content=r["content"],
                heading_path=r["heading_path"],
                vector_score=float(r["vector_score"]),
                keyword_score=float(r["keyword_score"]),
                combined_score=final_score,
                content_type=content_type,
                created_at=r["created_at"].isoformat() if r["created_at"] else None,
                times_used=r["times_used"] or 0,
                effective_date=effective_date_str,
            )
        )

    results.sort(key=lambda c: c.combined_score, reverse=True)
    return results[:top_k]


async def mark_chunks_used(chunk_ids: list[str]):
    if not chunk_ids:
        return
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE rag_chunks
            SET times_used = times_used + 1, last_used_at = now()
            WHERE id = ANY($1::text[])
            """,
            chunk_ids,
        )
