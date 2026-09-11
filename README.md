# RFP Compliance Assistant — RAG Core Engine

A production-hardened, multi-tenant Retrieval-Augmented Generation (RAG) backend engineered for automated compliance, security questionnaire completion, and Request for Proposal (RFP) response generation[cite: 16, 20].

The system uses a **LangGraph-orchestrated dual-model pipeline** (20B generator / 120B validator) combined with **PostgreSQL + pgvector hybrid retrieval** (Reciprocal Rank Fusion) and an **evidence-calibrated 7-factor confidence scoring engine**

---

## System Architecture

                              [ User / API Request ]
                                         |
                                         v
                              +---------------------+
                              |    node_retrieve    |
                              | (Hybrid RRF Search) |
                              +----------+----------+
                                         | Chunks (Top K=4)
                                         v
                              +---------------------+
                              | node_grade_context  |  <--- 20B Model (Reading Comprehension)
                              | (Source Discrepancy)|
                              +----------+----------+
                                         |
                                         v
                       +-----------------------------------+
                       |            node_draft             |  <--- 20B Model (Draft Generation)
                       | (Context-Constrained Synthesis)   |
                       +-----------------+-----------------+
                                         | Draft Answer
                                         v
                              +---------------------+
                              |    node_validate    |  <--- 120B Model (Strict JSON Schema)
                              |  (Grounding & Roles)|
                              +----------+----------+
                                         |
                   +---------------------+---------------------+
                   | (Failed / Attempts < 3)                   | (Passed OR Attempts >= 3)
                   v                                           v
         [ Loop back to node_draft ]                  +------------------+
         (Anti-apology directive)                     |    node_score    |
                                                      | (7-Signal Math)  |
                                                      +--------+---------+
                                                               |
                                                   [ Database Persistence ]
                                                   - questions.draft_answer
                                                   - questions.confidence
                                                   - rag_answer_attempts

---

## Core Capabilities & Engineering Hardening

### 1. Dual-Model Architecture & Token Governance
* **Draft Generation & Pre-Grading (`openai/gpt-oss-20b`):** Handles high-throughput reading comprehension, internal contradiction detection, and initial synthesis while preserving API quota and latency budgets.
* **Strict Validation & Role Auditing (`openai/gpt-oss-120b`):** Enforces grounding, detects false-premise rationalization, and flags direction mismatches via strict structured JSON outputs (`ValidationResult`).
* **Anti-Apology Retry Bounds:** On failed validation retries, prompt constraints prevent conversational apology loops (`"I apologize for the confusion..."`) from corrupting subsequent attempts.

### 2. Multi-Tenant Hybrid Retrieval (RRF)
* **Reciprocal Rank Fusion (RRF, $k=60$):** Eliminates arbitrary score-normalization heuristics by combining ranks from dense vector cosine similarity and full-text keyword search:
  $$\text{RRF Score} = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
* **Tenant Isolation & Lifecycle Enforcement:** Every retrieval query enforces `c.organization_id = $4` and `d.is_archived = false`, preventing cross-tenant leakage and isolating superseded documentation.
* **Websearch Sanitization:** Employs `websearch_to_tsquery` alongside regex control-character sanitization, safely preserving exact double quotes and hyphenated compliance identifiers (e.g., `SOP-IT-2026-04`) without query parsing faults.

### 3. Structural Chunking & Non-Blocking Ingestion
* **Table Semantics Preservation:** Markdown tables are preserved as whole units[cite: 3, 17]. At ingestion, tables generate synthetic natural-language context sentences from column-row mappings for dense vector indexing, while the pristine markdown table is stored for inference context.
* **Heading Path Lineage:** Maintains document hierarchy (`H1 > H2 > H3`) alongside text chunks for semantic disambiguation.
* **Async Event Loop Isolation:** All CPU-bound parsing (`pdfplumber`, `python-docx`, `openpyxl`) and recursive text chunking run inside worker threads via `asyncio.to_thread`, preventing event loop starvation during concurrent `/health` pings.
* **ReDoS Hardening:** Collapses consecutive table-of-contents leader dots (`\.{3,}`) prior to regex punctuation boundary splitting.

---

## 7-Factor Confidence Formulation

Confidence is derived mathematically using weighted signals rather than qualitative LLM self-scoring:

$$\text{Confidence} = 100 \times \sum_{i} (w_i \cdot s_i)$$

| Metric | Weight ($w_i$) | Evaluation Mechanism |
| :--- | :---: | :--- |
| **Keyword Match** | `0.12` | Token intersection between question terms and retrieved context. |
| **Semantic Similarity** | `0.22` | Mean cosine similarity of top-3 retrieved vector chunks ($1 - \text{cosine\_distance}$). |
| **Source Quality** | `0.08` | Linear temporal decay based on document `effective_date` (1-year half-life)[cite: 4]. |
| **Completeness** | `0.08` | Word count penalty band ($<8$ words: 0.2, $8\text{--}20$: 0.6, $20\text{--}150$: 1.0, $>150$: 0.8). |
| **Source Agreement** | `0.18` | Binary flag triggered by `node_grade_context` (0.3 if internal contradiction exists, 1.0 otherwise)[cite: 4]. |
| **Answer Relevance** | `0.18` | Dual-tier refusal detection (0.2 if refusal flagged, 1.0 otherwise)[cite: 4]. |
| **Direction Mismatch** | `0.14` | Entity relationship audit (0.3 if client/vendor roles reversed, 1.0 otherwise). |

### Classification Tiers & Refusal Logic
* **Green ($\ge 80$):** Fully verified, grounded, and aligned.Auto-promoted to `IN_REVIEW`.
* **Yellow ($50\text{--}79$):** Requires human validation. Routes to `IN_PROGRESS`.
  * **Explicit Refusals:** Clamped to **$50.0\text{--}58.0$** by design. Honest admissions of missing knowledge will never score Green[cite: 4].
* **Red ($< 50$):** Severe grounding failure, multi-party conflict, or validation exhaustion.

### Dual-Tier Refusal Fallback Architecture
1. **Primary Evaluation:** 120B model sets `is_refusal: bool`. Differentiates epistemic refusals (*"The context does not state..."*) from factual system constraints (*"The internal reporting stack does not have automatic failover..."*).
2. **Deterministic Regex Fallback:**
   * **Tier 1 (Strong):** Matches explicit context absences anywhere in text (supports straight `'` and curly `’` apostrophes).
   * **Tier 2 (Weak):** Matches generic negation (`"does not have"`, `"is not permitted"`), requiring co-occurrence with self-reference words (`"documents"`, `"materials"`, `"context"`) to prevent penalizing factual negative assertions.

---

## Database Architecture

The data layer integrates directly with the Prisma-managed application schema via PostgreSQL and `pgvector`:

+-------------------------------------------------------------+
|                      rag_documents                          |
+-------------------------------------------------------------+
| id (TEXT, cuid/uuid)            | Primary Key               |
| organization_id (TEXT)          | Foreign Key (Tenant)      |
| filename (TEXT)                 | Original Upload Name      |
| source_type (TEXT)              | policy, past_answer, etc. |
| category (TEXT)                 | Security, Compliance, etc.|
| tags (TEXT[])                   | Normalized Keywords       |
| is_archived (BOOLEAN)           | Soft-Delete Flag          |
| content_hash (TEXT)             | SHA-256 (Deduplication)   |
| effective_date (DATE)           | Temporal Quality Anchor   |
| supersedes_label (TEXT)         | Version Replacement Tag   |
+-------------------------------------------------------------+
| 1
|
| N
+-------------------------------------------------------------+
|                       rag_chunks                            |
+-------------------------------------------------------------+
| id (TEXT, cuid/uuid)            | Primary Key               |
| organization_id (TEXT)          | Denormalized Tenant Index |
| document_id (TEXT)              | Foreign Key (Cascade)     |
| chunk_index (INT)               | Sequential Order          |
| content (TEXT)                  | Clean Extracted Text/Table|
| heading_path (TEXT)             | Lineage Tree String       |
| metadata (JSONB)                | Page, Row Count, Flags    |
| embedding (vector(768))         | BAAI/bge-base-en-v1.5     |
| tsv (tsvector)                  | Stored to_tsvector        |
+-------------------------------------------------------------+


---

## Technical History & Engineering Findings

| Milestone | Defect Identified | Root Cause | Engineering Resolution |
| :--- | :--- | :--- | :--- |
| **V1** | Cross-tenant leakage & stale retrieval. | Missing `organization_id` filters on chunk lookups; `is_archived` column unjoined in CTEs. | Enforced strict tenant parameters across all routes; joined `rag_documents.is_archived = false` into hybrid search. |
| **V1-V2** | Inflated confidence on conflicting sources. | Scoring signals evaluated lexical/semantic density without evaluating factual agreement. | Added `node_grade_context` and the `source_agreement` signal, penalizing contradictory sources. |
| **V2** | Green confidence unreachable. | `all-MiniLM-L6-v2` cosine similarities compressed at ~0.75 for technical text. | Migrated to `BAAI/bge-base-en-v1.5` (768 dimensions), implementing asymmetric query-prefix handling. |
| **V3** | Valid refusals scoring Green. | Groq models output curly apostrophes (`don’t`); regex checked ASCII only (`don't`). | Upgraded regex to `[\'\u2019]`; established 120B structured validation as the primary check. |
| **V4** | False premise adoption & role reversal. | Drafter rationalized false metrics (e.g., 10s failover); confused customer payment terms with vendor obligations. | Created `direction_mismatch` signal and negative validation rules for ungrounded premises. |
| **V5** | Silent failure on negative factual claims. | `_STRONG_REFUSAL_PATTERNS` contained `"does not have"`, penalizing documented limitations as refusals. | Moved ambiguous phrases to Tier 2 with a `_SELF_REFERENCE_RE` gate. |
| **V6** | Event-loop blocking & tsquery crashes. | Synchronous file parsing stalled the async loop; special characters broke `plainto_tsquery`. | Offloaded chunking/parsing to `asyncio.to_thread`; migrated to `websearch_to_tsquery`. |

---

## Getting Started

### Prerequisites
* Python 3.11+
* PostgreSQL 15+ with `pgvector` and `pg_trgm` extensions enabled[cite: 14]
* Groq Cloud API Key (configured for `openai/gpt-oss-20b` and `openai/gpt-oss-120b`)
### Local Installation
1. Clone the repository and configure your virtual environment:
   ```bash
   git clone [https://github.com/zeecoding/RAG-Agent.git](https://github.com/zeecoding/RAG-Agent.git)
   cd RAG-Agent
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
Configure environment variables (.env):

Ini, TOML
NEON_DATABASE_URL=postgresql://user:pass@ep-xyz.neon.tech/neondb?sslmode=require
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_DRAFT_MODEL=openai/gpt-oss-20b
GROQ_VALIDATE_MODEL=openai/gpt-oss-120b
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIM=768
TOP_K_CHUNKS=4
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_UPLOAD_SIZE_MB=50
Run migrations against your database:

Bash
psql $NEON_DATABASE_URL -f sql/schema.sql
psql $NEON_DATABASE_URL -f sql/migration_002_enrichment.sql
psql $NEON_DATABASE_URL -f sql/migration_003_embedding_upgrade.sql
Start the FastAPI development server:

Bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload