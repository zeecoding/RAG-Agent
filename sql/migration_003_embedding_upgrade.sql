-- Migration 003: Embedding model upgrade + duplicate detection
--
-- 1. Upgrades embedding column from vector(384) to vector(768) for
--    BAAI/bge-base-en-v1.5 (was all-MiniLM-L6-v2).
-- 2. Adds content_hash column to rag_documents for duplicate upload
--    prevention (SHA-256 of raw file bytes).
--
-- IMPORTANT: This migration deletes ALL existing embeddings because
-- 384-dim and 768-dim vectors are incompatible. Re-upload all documents
-- after running this migration. Run once against your Neon branch.

-- Step 1: Delete all existing chunks (embeddings are incompatible)
DELETE FROM rag_chunks;

-- Step 2: Change embedding column dimension
ALTER TABLE rag_chunks ALTER COLUMN embedding TYPE vector(768);

-- Step 3: Rebuild the HNSW index for the new dimension
DROP INDEX IF EXISTS idx_rag_chunks_embedding;
CREATE INDEX idx_rag_chunks_embedding
    ON rag_chunks USING hnsw (embedding vector_cosine_ops);

-- Step 4: Add content_hash column for duplicate detection
ALTER TABLE rag_documents
    ADD COLUMN IF NOT EXISTS content_hash TEXT;

CREATE INDEX IF NOT EXISTS idx_rag_documents_hash
    ON rag_documents(organization_id, content_hash);
