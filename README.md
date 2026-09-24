# RFP Compliance Assistant — RAG Core Engine

A production-hardened, multi-tenant Retrieval-Augmented Generation (RAG) backend engineered for automated compliance, security questionnaire completion, and Request for Proposal (RFP) response generation[cite: 16].

The system uses a **LangGraph-orchestrated dual-model pipeline** (20B generator / 120B validator) combined with **PostgreSQL + pgvector hybrid retrieval** (Reciprocal Rank Fusion) and an **evidence-calibrated 7-factor confidence scoring engine**[cite: 16].

---

## System Architecture

```text
[ User / API Request ]
         |
         v
+---------------------+
|    node_retrieve    |
| (Hybrid RRF Search) |
+----------+----------+
           | Chunks (Top K=6)
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
(State: initial_validation_passed)           |    node_score    |
                                            | (7-Signal Math)  |
                                            +--------+---------+
                                                     |
                                               [ Database Persistence ]
                                               - questions.draft_answer
                                               - questions.confidence (Rounded Int)
                                               - rag_answer_attempts
```

---

## Core Capabilities & Engineering Hardening

### 1. Dual-Model Architecture & Token Governance

- **Draft Generation & Pre-Grading (`openai/gpt-oss-20b`):** Handles high-throughput reading comprehension, internal contradiction detection, and initial synthesis while preserving API quota and latency budgets.
- **Strict Validation & Role Auditing (`openai/gpt-oss-120b`):** Enforces grounding, detects false-premise rationalization, and flags direction mismatches via strict structured JSON outputs (`ValidationResult`)[cite: 7].
- **Loop State Preservation:** Tracks `initial_validation_passed` separately from transient retry states to measure initial draft accuracy versus post-correction convergence[cite: 15, 16].

### 2. Multi-Tenant Hybrid Retrieval (RRF)

- **Reciprocal Rank Fusion (RRF, $k=60$):** Combines ranks from dense vector cosine similarity and full-text keyword search:

  $$\text{RRF Score} = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

- **Tenant Isolation & Lifecycle Enforcement:** Every retrieval query enforces `c.organization_id = $4` and `d.is_archived = false`, preventing cross-tenant leakage and isolating superseded documentation.
- **Websearch Sanitization:** Employs `websearch_to_tsquery` alongside regex control-character sanitization, safely preserving exact double quotes and hyphenated compliance identifiers (e.g., `SOP-IT-2026-04`) without query parsing faults.

### 3. Structural Chunking & Non-Blocking Ingestion

- **Table Semantics Preservation:** Markdown tables are preserved as whole units[cite: 3]. At ingestion, tables generate synthetic natural-language context sentences from column-row mappings for dense vector indexing, while the pristine markdown table is stored for inference context.
- **Heading Path Lineage:** Maintains document hierarchy (`H1 > H2 > H3`) alongside text chunks for semantic disambiguation.
- **Async Event Loop Isolation:** All CPU-bound parsing (`pdfplumber`, `python-docx`, `openpyxl`) and recursive text chunking run inside worker threads via `asyncio.to_thread`, preventing event loop starvation.
- **ReDoS Hardening:** Collapses consecutive table-of-contents leader dots (`\.{3,}`) prior to regex punctuation boundary splitting.

---

## 7-Factor Confidence Formulation

Confidence is derived mathematically using weighted signals rather than qualitative LLM self-scoring:

$$\text{Confidence} = 100 \times \sum_{i} (w_i \cdot s_i)$$

| Metric | Weight ($w_i$) | Evaluation Mechanism |
| :--- | :---: | :--- |
| **Keyword Match** | `0.12` | Token intersection ratio between question terms and retrieved context. |
| **Semantic Similarity** | `0.22` | Mean cosine similarity of top-3 retrieved vector chunks ($1 - \text{cosine\_distance}$). |
| **Source Quality (Recency under Conflict)** | `0.08` | Returns `1.0` for uncontested policies. Under active conflict, applies a 5-year decay curve ($1.0 - \text{age}/1825$, floor $0.40$)[cite: 15]. |
| **Completeness** | `0.08` | Length scoring ($<2$ words: 0.3, $2\text{--}150$ words: 1.0, $>150$ words: 0.85) to avoid penalizing concise legal answers[cite: 15]. |
| **Source Agreement** | `0.18` | Flagged by `node_grade_context` (0.3 if internal contradiction exists, 1.0 otherwise)[cite: 15]. |
| **Answer Relevance** | `0.18` | Dual-gate refusal detection (0.2 if refusal confirmed, 1.0 otherwise)[cite: 15]. |
| **Direction Mismatch** | `0.14` | Entity relationship audit (0.3 if client/vendor roles reversed, 1.0 otherwise)[cite: 15]. |

### Classification Tiers & Verification Floor

- **Green ($\ge 80$):** Fully verified, grounded, and aligned. Auto-promoted to `IN_REVIEW`.
- **Yellow ($50\text{--}79$):** Requires human validation. Routes to `IN_PROGRESS`.
  - **Verified Refusals:** Clamped to **$50.0\text{--}58.0$** by design[cite: 15]. Honest admissions of missing knowledge will never score Green.
  - **Temporal & Metric Conflicts:** Unresolved discrepancies hold answers safely in the **$70.0\text{--}76.0$** band for compliance sign-off[cite: 11].
- **Red ($< 50$):** Severe grounding failure, multi-party conflict, or validation rejection (`VALIDATION_PENALTY = 0.40`)[cite: 15].

### Defense-in-Depth Refusal Architecture (The Consensus Gate)

1. **Primary Authority:** The 120B QA validator sets `is_refusal: bool`[cite: 15].
2. **Secondary Regex & Grounding Fallback:** If the LLM misses a refusal, regex patterns fire only if the answer shares $<25\%$ of key nouns with the question[cite: 15]. This protects valid premise rejections (e.g., rejecting an incorrect SLA) from false-positive refusal clamps[cite: 15].

---

## 4-Dimensional Benchmark Evaluation (50-Question Battery)

The system is evaluated across four independent operational dimensions via `tests/run_evaluation_matrix.py`[cite: 9, 13]:

| Metric Dimension | Target | Production Benchmark Result |
| :--- | :---: | :---: |
| **Final Grounding Pass Rate** | $100\%$ | **100.0% (50/50)** |
| **Initial Draft Pass Rate** | $>90\%$ | **92.0% (46/50)** (Avg 1.08 attempts)[cite: 6] |
| **Behavioral Mode Alignment** | $>85\%$ | **88.0% (44/50)** |
| **Confidence Calibration Rate** | $>90\%$ | **96.0% (48/50)** |
| **Retrieval Hit@6** | $>65\%$ | **70.0% (35/50)** |

---

## Database Architecture

The data layer integrates directly with the Prisma-managed application schema via PostgreSQL and `pgvector`:

```text
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
```

### Technical History & Engineering Findings

| Milestone | Defect Identified | Root Cause | Engineering Resolution |
| :--- | :--- | :--- | :--- |
| **V1** | Cross-tenant leakage & stale retrieval. | Missing `organization_id` filters on chunk lookups; `is_archived` column unjoined in CTEs. | Enforced strict tenant parameters across all routes; joined `rag_documents.is_archived = false` into hybrid search. |
| **V2** | Inflated confidence on conflicting sources. | Scoring signals evaluated lexical/semantic density without evaluating factual agreement. | Added `node_grade_context` and the `source_agreement` signal, penalizing contradictory sources. |
| **V3** | Valid refusals scoring Green. | Groq models output curly apostrophes (`don’t`); regex checked ASCII only (`don't`). | Upgraded regex to `[\'\u2019]`; established 120B structured validation as primary authority[cite: 4, 15]. |
| **V4** | False premise adoption & role reversal. | Drafter rationalized false metrics (e.g., 10s failover); confused customer payment terms with vendor obligations. | Created `direction_mismatch` signal and negative validation rules for ungrounded premises[cite: 15]. |
| **V5** | Premise rejections clamped to Yellow. | Aggressive refusal regex fired on negative assertion phrasing ("does not specify"). | Implemented the Consensus Gate: regex only clamps when question-answer term overlap is $<25\%$[cite: 15]. |
| **V6** | Uncontested policies penalized by age. | 365-day source quality decay unfairly depressed legacy, unrevoked policies. | Gated age decay strictly on `conflict_detected=True` over a 5-year graceful window[cite: 15]. |
| **V7** | Float truncation mismatch in DB writes. | `int(result["confidence_score"])` truncated $79.99$ to $79$, desyncing numeric queries from Green status. | Fixed the persistence layer to store normalized numeric confidence values consistently. |

---

## Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL 15+ with `pgvector` and `pg_trgm` extensions enabled
- Groq Cloud API key configured for `openai/gpt-oss-20b` and `openai/gpt-oss-120b`

### Local Installation

1. Clone the repository and configure your virtual environment:

```bash
git clone https://github.com/zeecoding/RAG-Agent.git
cd RAG-Agent
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Configure environment variables in a `.env` file:

```ini
NEON_DATABASE_URL=postgresql://user:pass@ep-xyz.neon.tech/neondb?sslmode=require
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_DRAFT_MODEL=openai/gpt-oss-20b
GROQ_VALIDATE_MODEL=openai/gpt-oss-120b
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIM=768
TOP_K_CHUNKS=6
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_UPLOAD_SIZE_MB=50
```

3. Run migrations against your database:

```bash
psql "$NEON_DATABASE_URL" -f sql/schema.sql
psql "$NEON_DATABASE_URL" -f sql/migration_002_enrichment.sql
psql "$NEON_DATABASE_URL" -f sql/migration_003_embedding_upgrade.sql
```

4. Start the FastAPI development server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
