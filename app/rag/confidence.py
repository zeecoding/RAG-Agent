import re
from datetime import datetime, timezone

from app.rag.retrieval import RetrievedChunk

KEYWORD_WEIGHT = 0.12
SEMANTIC_WEIGHT = 0.22
SOURCE_QUALITY_WEIGHT = 0.08
COMPLETENESS_WEIGHT = 0.08
SOURCE_AGREEMENT_WEIGHT = 0.18
ANSWER_RELEVANCE_WEIGHT = 0.18
DIRECTION_MISMATCH_WEIGHT = 0.14


def _keyword_match_score(question: str, chunks: list[RetrievedChunk]) -> float:
    """Rough proxy: fraction of significant question terms present in retrieved content."""
    if not chunks:
        return 0.0
    terms = {t.lower() for t in re.findall(r"[A-Za-z0-9\-]{3,}", question)}
    if not terms:
        return 0.5
    joined = " ".join(c.content.lower() for c in chunks)
    hits = sum(1 for t in terms if t in joined)
    return min(hits / len(terms), 1.0)


def _semantic_score(chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0
    # vector_score is already 1 - cosine_distance, i.e. similarity in [0,1]
    top = sorted((c.vector_score for c in chunks), reverse=True)[:3]
    return sum(top) / len(top)


def _source_quality_score(chunk_metadata: list[dict]) -> float:
    """Source quality based on document recency. Expects each dict to
    (optionally) have 'updated_at' (ISO string).

    times_used is intentionally NOT included here — it tracks how often
    a chunk was *retrieved*, not how *trustworthy* it is. Including it
    caused repeated questions to gradually inflate confidence scores:
    ask the same question twice and the chunks' times_used goes up,
    making the second answer score higher for no good reason. The
    times_used column is still tracked in the DB for admin analytics,
    just not used in confidence scoring."""
    if not chunk_metadata:
        return 0.5
    scores = []
    now = datetime.now(timezone.utc)
    for m in chunk_metadata:
        recency = 0.5
        updated_at = m.get("updated_at")
        if updated_at:
            try:
                dt = datetime.fromisoformat(updated_at)
                age_days = (now - dt).days
                recency = max(0.0, 1 - age_days / 365)  # linear decay over 1 year
            except ValueError:
                pass
        scores.append(recency)
    return sum(scores) / len(scores)


def _completeness_score(answer_text: str) -> float:
    length = len(answer_text.split())
    if length < 8:
        return 0.2
    if length < 20:
        return 0.6
    if length <= 150:
        return 1.0
    return 0.8  # overly long answers lose a little


def _source_agreement_score(conflict_detected: bool) -> float:
    """The signal Q5 exposed as missing: an answer can score well on
    keyword/semantic/completeness while still hedging across two sources
    that disagree with each other. This is detected by the validate step
    (same LLM call, no added cost) and penalized here directly, regardless
    of how well-written the resulting answer is."""
    return 0.3 if conflict_detected else 1.0


def _direction_mismatch_score(direction_mismatch: bool | None) -> float:
    """Penalizes answers where a policy or clause was cited for the wrong party/role
    (e.g. Customer-pays-Company clause cited for Company-pays-Vendor question).
    Uses a 0.3 penalty similar to source_agreement."""
    return 0.3 if direction_mismatch is True else 1.0


# Phrases that indicate the agent could NOT answer from the knowledge base.
# These are checked case-insensitively against the full answer text.
# A match means the agent admitted it doesn't have the information — the
# answer is a refusal, not a substantive response, and confidence should
# reflect that (low, not green/high-yellow).
# [\'\u2019]? matches a straight apostrophe ('), a curly one (\u2019), or
# neither — NOT just '?, which only ever matched the straight ASCII
# version. This is the fallback path only (see _answer_relevance_score);
# fixing it here is defense in depth, not a substitute for the LLM-based
# primary check.
_APOS = "[\'\u2019]?"

# Tier 1: unambiguous — safe to match anywhere in the answer text.
_STRONG_REFUSAL_PATTERNS = [
    rf"(?:don{_APOS}t|do not|doesn{_APOS}t|does not)\s+(?:contain|mention|cover|see|find)",
    r"(?:no|not any|no specific)\s+(?:information|mention|details?|data|reference|coverage)",
    r"(?:no|without)\s+mention\s+of",
    rf"(?:can{_APOS}t|cannot|unable to)\s+(?:confirm|find|locate|identify|determine|verify|describe)",
    rf"(?:don{_APOS}t|do not)\s+(?:appear|seem)\s+to",
    r"(?:no|not)\s+(?:seeing|aware of)",
    rf"(?:i{_APOS}m not seeing|i{_APOS}m not aware)",
    r"not\s+(?:contain)\s+any\s+information",
    r"(?:absent|missing)\s+from\s+(?:the\s+)?(?:documents?|sources?|context|provided|materials)",
    r"provided\s+(?:materials?|documents?|sources?|context)\s+(?:don|do|does)",
]

# Tier 2: ambiguous on their own (also match legitimate restriction
# language like "access is not available"). Only counts as a refusal
# signal if the answer ALSO self-references the source material nearby.
_WEAK_REFUSAL_PATTERNS = [
    r"not\s+(?:mentioned|covered|addressed|included|found|available|aware)",
    rf"(?:isn{_APOS}t|is not)\s+(?:mentioned|covered|addressed|included|available)",
    rf"(?:don{_APOS}t|do not|doesn{_APOS}t|does not)\s+(?:have|include)",
]
_SELF_REFERENCE_RE = re.compile(
    r"(?:materials?|documents?|sources?|context|provided|information)", re.IGNORECASE
)


def _answer_relevance_score(answer_text: str, is_refusal: bool | None = None) -> float:
    if is_refusal is True:
        return 0.2
    lower = answer_text.lower()
    for pattern in _STRONG_REFUSAL_PATTERNS:
        if re.search(pattern, lower):
            return 0.2
    for pattern in _WEAK_REFUSAL_PATTERNS:
        if re.search(pattern, lower) and _SELF_REFERENCE_RE.search(lower):
            return 0.2
    return 1.0


def score_answer(
    question: str,
    answer_text: str,
    chunks: list[RetrievedChunk],
    chunk_metadata: list[dict] | None = None,
    conflict_detected: bool = False,
    is_refusal: bool | None = None,
    direction_mismatch: bool | None = None,
) -> tuple[float, str, dict]:
    """Returns (confidence_score 0-100, level 'green'|'yellow'|'red', breakdown dict)."""
    keyword = _keyword_match_score(question, chunks)
    semantic = _semantic_score(chunks)
    source_quality = _source_quality_score(chunk_metadata or [])
    completeness = _completeness_score(answer_text)
    agreement = _source_agreement_score(conflict_detected)
    relevance = _answer_relevance_score(answer_text, is_refusal=is_refusal)
    direction = _direction_mismatch_score(direction_mismatch)

    score = (
        KEYWORD_WEIGHT * keyword
        + SEMANTIC_WEIGHT * semantic
        + SOURCE_QUALITY_WEIGHT * source_quality
        + COMPLETENESS_WEIGHT * completeness
        + SOURCE_AGREEMENT_WEIGHT * agreement
        + ANSWER_RELEVANCE_WEIGHT * relevance
        + DIRECTION_MISMATCH_WEIGHT * direction
    ) * 100

    # If the answer is a detected refusal (relevance == 0.2), map the score
    # into the 50-58 range (Yellow low). Refusals are honest "I don't know"
    # responses and belong in Yellow low (50-59) by design, never Green or High Yellow.
    if relevance < 0.5:
        score = min(58.0, max(50.0, score * 0.85))

    if score >= 80:
        level = "green"
    elif score >= 50:
        level = "yellow"
    else:
        level = "red"

    breakdown = {
        "keyword_match": round(keyword, 3),
        "semantic_similarity": round(semantic, 3),
        "source_quality": round(source_quality, 3),
        "completeness": round(completeness, 3),
        "source_agreement": round(agreement, 3),
        "answer_relevance": round(relevance, 3),
        "direction_mismatch": round(direction, 3),
    }

    return round(score, 2), level, breakdown
