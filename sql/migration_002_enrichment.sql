-- Adds the two fields metadata enrichment needs to store beyond the
-- existing category/tags columns. Run once against your Neon branch.

ALTER TABLE rag_documents
    ADD COLUMN IF NOT EXISTS effective_date DATE,
    ADD COLUMN IF NOT EXISTS supersedes_label TEXT;

CREATE INDEX IF NOT EXISTS idx_rag_documents_effective_date ON rag_documents(effective_date);
