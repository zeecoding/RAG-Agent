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

-- ---------------------------------------------------------------------
-- rag_guardrail_events: Audit log for security & compliance guardrail
-- events (PII redactions, prompt injections, document poisoning).
-- question_id is nullable because ingestion-stage events (e.g. document
-- upload rejections) have no associated question.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rag_guardrail_events (
    id              TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    organization_id TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    question_id     TEXT REFERENCES questions(id) ON DELETE CASCADE,
    event_type      TEXT NOT NULL,
    payload         JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rag_guardrail_events_org_type_time
    ON rag_guardrail_events(organization_id, event_type, created_at DESC);

-- ---------------------------------------------------------------------
-- rag_tenant_domains: Inferred corporate email domains per tenant organization.
-- Populated during document ingestion by extracting email domains from
-- uploaded text. A domain is considered tenant-verified once seen across
-- 2+ separate documents (times_seen >= 2).
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rag_tenant_domains (
    organization_id TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    domain          TEXT NOT NULL,
    times_seen      INT DEFAULT 1,
    last_seen_at    TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (organization_id, domain)
);

CREATE INDEX IF NOT EXISTS idx_rag_tenant_domains_org_verified
    ON rag_tenant_domains(organization_id, domain) WHERE times_seen >= 2;

-- ---------------------------------------------------------------------
-- rag_document_domains: Document-to-domain mapping with ON DELETE CASCADE.
-- Prevents counter drift when documents are deleted or re-uploaded.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rag_document_domains (
    organization_id TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    document_id     TEXT NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
    domain          TEXT NOT NULL,
    PRIMARY KEY (document_id, domain)
);

CREATE INDEX IF NOT EXISTS idx_rag_document_domains_org_domain
    ON rag_document_domains(organization_id, domain);

-- ---------------------------------------------------------------------
-- questionnaires & coordinate tracking: Format-preserving questionnaire
-- exporter integration. Additive alterations to existing Prisma schema.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS questionnaires (
    id              TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    organization_id TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    title           TEXT NOT NULL,
    filename        TEXT NOT NULL,
    storage_path    TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'IN_PROGRESS',
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

-- Add missing columns to questionnaires if table was created by Prisma
ALTER TABLE questionnaires ADD COLUMN IF NOT EXISTS organization_id TEXT REFERENCES organizations(id) ON DELETE CASCADE;
ALTER TABLE questionnaires ADD COLUMN IF NOT EXISTS title TEXT;
ALTER TABLE questionnaires ADD COLUMN IF NOT EXISTS filename TEXT;
ALTER TABLE questionnaires ADD COLUMN IF NOT EXISTS storage_path TEXT;

-- Relax Prisma not-null constraints for standalone questionnaire imports
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'questionnaires' AND column_name = 'project_id' AND is_nullable = 'NO') THEN
        ALTER TABLE questionnaires ALTER COLUMN project_id DROP NOT NULL;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'questionnaires' AND column_name = 'file_name' AND is_nullable = 'NO') THEN
        ALTER TABLE questionnaires ALTER COLUMN file_name DROP NOT NULL;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'questionnaires' AND column_name = 'file_url' AND is_nullable = 'NO') THEN
        ALTER TABLE questionnaires ALTER COLUMN file_url DROP NOT NULL;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'questionnaires' AND column_name = 'file_size' AND is_nullable = 'NO') THEN
        ALTER TABLE questionnaires ALTER COLUMN file_size DROP NOT NULL;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'questionnaires' AND column_name = 'file_type' AND is_nullable = 'NO') THEN
        ALTER TABLE questionnaires ALTER COLUMN file_type DROP NOT NULL;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'questionnaires' AND column_name = 'question_count' AND is_nullable = 'NO') THEN
        ALTER TABLE questionnaires ALTER COLUMN question_count DROP NOT NULL;
    END IF;
END $$;

-- Ensure questions table has coordinate tracking
ALTER TABLE questions ADD COLUMN IF NOT EXISTS coordinates JSONB;
ALTER TABLE questions ADD COLUMN IF NOT EXISTS questionnaire_id TEXT REFERENCES questionnaires(id) ON DELETE CASCADE;

-- Ensure enums include required statuses (safe no-op if enum or value already exists)
DO $$
BEGIN
    BEGIN
        ALTER TYPE "QuestionStatus" ADD VALUE IF NOT EXISTS 'DRAFT';
    EXCEPTION WHEN undefined_object THEN NULL;
    END;
    BEGIN
        ALTER TYPE "QuestionnaireStatus" ADD VALUE IF NOT EXISTS 'IN_PROGRESS';
    EXCEPTION WHEN undefined_object THEN NULL;
    END;
END $$;

ALTER TABLE questionnaires ALTER COLUMN updated_at SET DEFAULT now();
ALTER TABLE questions ALTER COLUMN updated_at SET DEFAULT now();

CREATE INDEX IF NOT EXISTS idx_questionnaires_org ON questionnaires(organization_id);
CREATE INDEX IF NOT EXISTS idx_questions_questionnaire ON questions(questionnaire_id);

-- ---------------------------------------------------------------------
-- Completed-document persistence & freshness tracking
-- (migration_004 — additive, safe to re-run)
--
-- completed_file_path       – bare relative storage path inside the
--                             questionnaires bucket, e.g.
--                             {org_id}/{questionnaire_id}/completed_{questionnaire_id}.docx
--                             NULL until the first build is triggered.
--
-- completed_file_updated_at – timestamp of the last successful build.
--                             NULL until first build.
--                             The Next.js frontend compares this against
--                             MAX(questions.updated_at) to determine whether
--                             the stored DOCX is stale and needs rebuilding.
-- ---------------------------------------------------------------------
ALTER TABLE questionnaires
    ADD COLUMN IF NOT EXISTS completed_file_path       TEXT,
    ADD COLUMN IF NOT EXISTS completed_file_updated_at TIMESTAMPTZ;

-- Index to speed up freshness checks (document-status endpoint) which filter
-- by id + organization_id and read only the two new columns.
CREATE INDEX IF NOT EXISTS idx_questionnaires_completed_freshness
    ON questionnaires(organization_id, id)
    WHERE completed_file_path IS NOT NULL;

-- ---------------------------------------------------------------------
-- Manual-edit tracking for ONLYOFFICE human polish workflow
-- (migration_005 — additive, safe to re-run)
--
-- has_manual_edits – set to TRUE by the /onlyoffice-callback endpoint
--                    whenever the ONLYOFFICE Document Server reports a
--                    successful save (status 2 or 6).
--
--                    Exposed through GET /document-status so the Next.js
--                    frontend can display:
--                      "This document has manual styling edits. Rebuilding
--                       will overwrite them."
--                    before the user triggers POST /build-document.
--
--                    Reset to FALSE each time _generate_and_persist_completed_docx
--                    runs a fresh automated build from agent answers.
-- ---------------------------------------------------------------------
ALTER TABLE questionnaires
    ADD COLUMN IF NOT EXISTS has_manual_edits BOOLEAN DEFAULT FALSE;
