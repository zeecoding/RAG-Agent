# backend — Intelligent RFP & Compliance Agent (RAG Core)

Lives alongside `frontend/` in the same repo. This service owns retrieval,
the LangGraph agent, and confidence scoring — it's separate from the Next.js
app, which keeps its own real-time/chat features untouched.

## 1. Set up Neon

```bash
psql "$NEON_DATABASE_URL" -f sql/schema.sql
```

## 2. Get a Groq API key

console.groq.com/keys — free, no card. Groq is a *hosted* API (you call
their endpoint), not something you run yourself — no GPU or server needed.

## 3. Install & run

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # fill in NEON_DATABASE_URL and GROQ_API_KEY
uvicorn app.main:app --reload
```

First run will download the `all-MiniLM-L6-v2` embedding model (~90MB) —
that happens once and is cached locally after.

## Endpoints

- `GET /health`
- `POST /ingest/chunk` — embeds (locally, CPU) and stores one chunk
- `POST /ask` — retrieve → draft (Groq) → validate (Groq) → refine loop → 
  confidence score. Persists to `answers` if you pass a `question_id`.

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "Do you encrypt data at rest?"}'
```

## Credentials needed (only two)

| Var | Where to get it |
|---|---|
| `NEON_DATABASE_URL` | Neon dashboard → pooled connection string |
| `GROQ_API_KEY` | console.groq.com/keys |

Nothing else — Supabase auth/realtime stays entirely on the `frontend/` side.

## Not yet built

- Module 1 ingestion pipeline (parsing + structure-aware chunking)
- Format-preserving file writer (Module 4)
- LLM-judge evaluation harness (precision/recall/latency)
