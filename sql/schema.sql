-- =====================================================================
-- RAG knowledge-base tables — additive migration, runs alongside your
-- existing Prisma-managed schema. Does NOT touch organizations, members,
-- questions, questionnaires, or any table Prisma already owns.
--
-- IDs here use TEXT to match Prisma's cuid() ids (not uuid) so foreign
-- keys to organizations/questions actually line up.
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ---------------------------------------------------------------------
-- documents: source knowledge base (past answers, policies, audit
-- reports) that the agent retrieves from — separate from `questionnaires`,
-- which is the incoming RFP being answered.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rag_documents (
    id              TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    organization_id TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    filename        TEXT NOT NULL,
    source_type     TEXT NOT NULL,              -- 'policy' | 'past_answer' | 'audit_report' | 'spec'
    category        TEXT,
    tags            TEXT[] DEFAULT '{}',
    is_archived     BOOLEAN DEFAULT FALSE,
    content_hash    TEXT,                          -- SHA-256 of raw file bytes, for duplicate detection
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rag_documents_org ON rag_documents(organization_id);
CREATE INDEX IF NOT EXISTS idx_rag_documents_hash ON rag_documents(organization_id, content_hash);

-- ---------------------------------------------------------------------
-- chunks: structure-aware chunks with embedding + full-text search.
-- organization_id is denormalized here (not just via document_id join)
-- so every retrieval query can filter by org directly — this is the line
-- that prevents one org's RAG agent from ever retrieving another org's
-- confidential documents. Never query this table without that filter.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rag_chunks (
    id              TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    organization_id TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    document_id     TEXT NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
    chunk_index     INT NOT NULL,
    content         TEXT NOT NULL,
    heading_path    TEXT,
    metadata        JSONB DEFAULT '{}'::jsonb,
    embedding       vector(768),                 -- 768 = BAAI/bge-base-en-v1.5
    tsv             tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    times_used      INT DEFAULT 0,
    last_used_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding
    ON rag_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_tsv ON rag_chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_org ON rag_chunks(organization_id);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_document ON rag_chunks(document_id);

-- ---------------------------------------------------------------------
-- rag_answer_attempts: OPTIONAL audit trail of what the agent generated
-- and why (confidence breakdown, which chunks it used), kept separate
-- from `questions.draft_answer`/`questions.confidence` (the live values
-- your app already reads/writes). This table is history, not the source
-- of truth — the agent still writes the final draft back onto the real
-- `questions` row so your existing UI needs zero changes.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rag_answer_attempts (
    id                  TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    question_id         TEXT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    answer_text         TEXT,
    confidence_score    NUMERIC(5,2),
    confidence_level    TEXT,
    source_chunk_ids    TEXT[] DEFAULT '{}',
    attempts            INT DEFAULT 1,
    created_at          TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rag_answer_attempts_question ON rag_answer_attempts(question_id);
