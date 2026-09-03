"""Metadata enrichment: one extra Groq call per uploaded document (not per
chunk, not per query) to populate rag_documents.category, .tags,
.effective_date, and .supersedes_label.

Design rules, deliberately:
  - Reads BOTH the first and last extracted text sections, since effective
    dates and version/supersession notices are as likely to sit in a footer
    or final page as in a header.
  - The model is explicitly told to output null rather than infer a date
    or supersession it doesn't literally see — a fabricated date is worse
    than a missing one, since it would lend false authority to a document.
  - category is constrained to a fixed vocabulary so it stays consistent
    and filterable across every document ever uploaded, rather than
    drifting ("HR" vs "Human Resources" vs "hr_policy") over time.
  - A failure here (bad JSON, API error) must never block the upload —
    the document still ingests with empty metadata, and enrichment can be
    retried later. Never let a nice-to-have step fail a core one.
  - This only fills descriptive metadata. It never sets is_archived —
    that stays a manual admin decision (see supersedes_label below, which
    is surfaced to the admin as a suggestion, not acted on automatically).
"""
import json
import logging
import re
from datetime import datetime

from app.config import settings
from app.rag.groq_client import get_groq_client

logger = logging.getLogger(__name__)

ALLOWED_CATEGORIES = [
    "Security", "Compliance", "HR", "Legal", "Financial",
    "Technical", "Operational", "Other",
]

ENRICHMENT_SYSTEM_PROMPT = f"""You extract structured metadata from a document excerpt for a knowledge base.

Rules — follow exactly:
- Only extract information that is explicitly and literally stated in the text.
- If a field is not stated, output null for it. NEVER infer, guess, or estimate a date, version, or supersession — a wrong guess is worse than admitting it's unknown.
- "category" MUST be exactly one of: {", ".join(ALLOWED_CATEGORIES)}. Pick the closest match; use "Other" only if none fit.
- "tags" is a list of 3-6 short lowercase keyword tags (e.g. "pto", "encryption", "refund-policy") describing the document's topic. No duplicates.
- "effective_date" is an ISO date (YYYY-MM-DD) if an explicit effective/issue date is stated, else null. If only a month/year is given (e.g. "March 2025"), use the first day of that month.
- "supersedes_label" is the exact identifier of a prior document/policy version this one states it replaces (e.g. "HR-302-v2024"), else null.

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{"category": "...", "tags": ["...", "..."], "effective_date": "YYYY-MM-DD" or null, "supersedes_label": "..." or null}}
"""


def _default_result() -> dict:
    return {"category": None, "tags": [], "effective_date": None, "supersedes_label": None}


# Deterministic, no-LLM-call safety net — runs ONLY when the LLM extraction
# comes back with no date, so it never adds cost on the common path where
# the LLM already found it. This is the "scan the overall document" fallback:
# unlike the LLM step (which only sees the first/last few sections to keep
# token cost down), this regex runs over the full concatenated text.
_DATE_PATTERNS = [
    # "Effective Date: March 1, 2025" or "Effective: March 1, 2025"
    r"[Ee]ffective\s*(?:[Dd]ate)?\s*:?\s*([A-Za-z]+\.?\s+\d{1,2},?\s+\d{4})",
    # "Effective Date: 2025-03-01"
    r"[Ee]ffective\s*(?:[Dd]ate)?\s*:?\s*(\d{4}-\d{2}-\d{2})",
    # Slash-separated dates (dd/mm/yyyy vs mm/dd/yyyy) deliberately omitted:
    # they are locale-ambiguous and this regex fallback has no way to know
    # which locale a given document uses. Safer to return None (effective_date
    # stays null — a legitimate, honest outcome the enrichment design already
    # handles) than to silently store a wrong date that could mis-rank retrieval.
    # "Effective: January 2025" or "Effective Date: March 2025" (Month YYYY only)
    r"[Ee]ffective\s*(?:[Dd]ate)?\s*:?\s*([A-Za-z]+\.?\s+\d{4})",
    # "Effective: 15 March 2025" or "Effective Date: 1 January 2024" (dd Month YYYY)
    r"[Ee]ffective\s*(?:[Dd]ate)?\s*:?\s*(\d{1,2}\s+[A-Za-z]+\.?\s+\d{4})",
]
_DATE_FORMATS = [
    "%B %d %Y", "%b %d %Y", "%Y-%m-%d",
    "%B %Y", "%b %Y",              # Month YYYY (maps to 1st of month)
    "%d %B %Y", "%d %b %Y",        # dd Month YYYY
]


def _regex_fallback_date(full_text: str) -> str | None:
    if not full_text:
        return None
    for pattern in _DATE_PATTERNS:
        match = re.search(pattern, full_text)
        if not match:
            continue
        raw = match.group(1).replace(",", "").strip()
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
    return None


def _validate(result: dict) -> dict:
    """Defensively sanitize whatever the model returned — never trust LLM
    JSON output blindly, even with an explicit schema in the prompt."""
    out = _default_result()

    category = result.get("category")
    if isinstance(category, str) and category in ALLOWED_CATEGORIES:
        out["category"] = category

    tags = result.get("tags")
    if isinstance(tags, list):
        cleaned = []
        for t in tags:
            if isinstance(t, str) and t.strip():
                cleaned.append(t.strip().lower()[:50])
        out["tags"] = list(dict.fromkeys(cleaned))[:6]  # dedupe, cap at 6

    effective_date = result.get("effective_date")
    if isinstance(effective_date, str) and len(effective_date) == 10:
        # Cheap format sanity check (YYYY-MM-DD) — real validation happens
        # at the DB layer when this is cast to a DATE column.
        parts = effective_date.split("-")
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            out["effective_date"] = effective_date

    supersedes = result.get("supersedes_label")
    if isinstance(supersedes, str) and supersedes.strip():
        out["supersedes_label"] = supersedes.strip()[:100]

    return out


async def enrich_document_metadata(
    filename: str, first_text: str, last_text: str, full_text: str | None = None
) -> dict:
    """Returns a dict with category, tags, effective_date, supersedes_label.
    Always returns a usable (possibly empty) result — never raises."""
    excerpt = (
        f"Filename: {filename}\n\n"
        f"--- Start of document ---\n{first_text[:2000]}\n\n"
        f"--- End of document ---\n{last_text[-1500:]}"
    )

    result = _default_result()
    try:
        response = await get_groq_client().chat(
            ENRICHMENT_SYSTEM_PROMPT,
            excerpt,
            model=settings.groq_draft_model,  # same cheap model as drafting — this is a light task
            temperature=0.0,
        )
        parsed = json.loads(response.strip())
        result = _validate(parsed)
    except json.JSONDecodeError:
        logger.warning(f"Enrichment returned non-JSON for {filename}, using empty metadata")
    except Exception as e:
        logger.warning(f"Enrichment failed for {filename}: {e} — using empty metadata")

    # Fallback only fires when the LLM step found nothing — no added cost
    # on the common path, and it's a whole-document scan rather than being
    # limited to the first/last excerpt sent to the LLM.
    if not result["effective_date"] and full_text:
        fallback_date = _regex_fallback_date(full_text)
        if fallback_date:
            logger.info(f"Regex fallback found effective_date for {filename}: {fallback_date}")
            result["effective_date"] = fallback_date

    return result
