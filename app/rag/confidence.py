import logging
import re
from datetime import datetime, timezone

from app.rag.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

# Enterprise 7-signal composite weights (Sum = 1.00)
KEYWORD_WEIGHT = 0.10
SEMANTIC_WEIGHT = 0.24
SOURCE_QUALITY_WEIGHT = 0.08
COMPLETENESS_WEIGHT = 0.08
SOURCE_AGREEMENT_WEIGHT = 0.18
ANSWER_RELEVANCE_WEIGHT = 0.18
DIRECTION_MISMATCH_WEIGHT = 0.14
VALIDATION_PENALTY = 0.40  # Cuts score to <50% (Red) when ungrounded


def _keyword_match_score(question: str, chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0
    terms = {t.lower() for t in re.findall(r"[A-Za-z0-9\-]{3,}", question)}
    if not terms:
        return 0.5
    joined = " ".join(c.content.lower() for c in chunks)
    hits = sum(1 for t in terms if t in joined)
    ratio = hits / len(terms)
    # A 65% keyword overlap against dense compliance text indicates strong term recall
    return min(1.0, ratio / 0.65)


def _semantic_score(chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0
    top = sorted((c.vector_score for c in chunks), reverse=True)[:3]
    raw_avg = sum(top) / len(top)
    # pgvector cosine similarity on bge-base typically spans 0.40 to 0.75 for domain queries
    # Normalizing with base 0.35 and span 0.35 maps 0.61 -> ~0.74, and 0.70+ -> 1.0
    return min(1.0, max(0.0, (raw_avg - 0.35) / 0.35))


def _source_quality_score(chunk_metadata: list[dict], conflict_detected: bool = False) -> float:
    if not chunk_metadata:
        return 0.85
    
    # If no contradiction exists between sources, unrevoked policies are 100% authoritative
    if not conflict_detected:
        return 1.0

    scores = []
    now = datetime.now(timezone.utc)
    for m in chunk_metadata:
        recency = 0.70
        updated_at = m.get("updated_at")
        if updated_at:
            try:
                dt = datetime.fromisoformat(updated_at)
                age_days = (now - dt).days
                recency = max(0.40, 1.0 - (age_days / 1825))  # 5-year decay under active conflict
            except ValueError:
                pass
        scores.append(recency)
    return sum(scores) / len(scores)


def _completeness_score(answer_text: str) -> float:
    length = len(answer_text.split())
    if length < 2:
        return 0.3
    if length <= 150:
        return 1.0
    return 0.85


def _source_agreement_score(conflict_detected: bool) -> float:
    return 0.3 if conflict_detected else 1.0


def _direction_mismatch_score(direction_mismatch: bool | None) -> float:
    return 0.3 if direction_mismatch is True else 1.0


_APOS = "[\'\u2019]?"

_STRONG_REFUSAL_PATTERNS = [
    rf"(?:don{_APOS}t|do not|doesn{_APOS}t|does not)\s+(?:mention|cover|see|find|specify|state|detail|prescribe|define)",
    r"(?:no|not any|no specific)\s+(?:information|mention|details?|data|reference|coverage)",
    r"(?:no|without)\s+mention\s+of",
    rf"(?:can{_APOS}t|cannot|unable to)\s+(?:confirm|find|locate|identify|determine|verify|describe)",
    rf"(?:don{_APOS}t|do not)\s+(?:appear|seem)\s+to",
    r"(?:no|not)\s+(?:seeing|aware of)",
    rf"(?:i{_APOS}m not seeing|i{_APOS}m not aware)",
    r"(?:absent|missing)\s+from\s+(?:the\s+)?(?:documents?|sources?|context|provided|materials?|agreements?|policies)",
    r"provided\s+(?:materials?|documents?|sources?|context|agreements?|policies)\s+(?:don|do|does|did)",
]

_WEAK_REFUSAL_PATTERNS = [
    r"not\s+(?:mentioned|covered|addressed|found|available|aware|specified|detailed|stated)",
    rf"(?:isn{_APOS}t|is not)\s+(?:mentioned|covered|addressed|available|specified|detailed|stated)",
]

_SELF_REFERENCE_RE = re.compile(
    r"(?:materials?|documents?|sources?|context|provided|information|agreements?|policies|contracts?)",
    re.IGNORECASE,
)


def _answer_relevance_score(
    question: str,
    answer_text: str,
    is_refusal: bool | None = None,
) -> float:
    # 1. Primary Authority: LLM Structured Output Validator
    if is_refusal is True:
        return 0.2

    # 2. Secondary Gate: Regex + Grounding Overlap (Defense in Depth)
    lower = answer_text.lower()
    regex_hit = any(re.search(p, lower) for p in _STRONG_REFUSAL_PATTERNS)
    if not regex_hit:
        regex_hit = any(
            re.search(p, lower) and _SELF_REFERENCE_RE.search(lower)
            for p in _WEAK_REFUSAL_PATTERNS
        )

    if regex_hit:
        # Check if the answer quotes/discusses the question's specific entities.
        # An active premise rejection reuses the question's nouns.
        # A lazy refusal shares zero context.
        q_terms = {t.lower() for t in re.findall(r"[A-Za-z0-9\-]{4,}", question)}
        ans_terms = {t.lower() for t in re.findall(r"[A-Za-z0-9\-]{4,}", answer_text)}
        overlap = len(q_terms & ans_terms) / max(len(q_terms), 1)

        if overlap < 0.25:
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
    validation_passed: bool = True,
    retrieval_confidence: float | None = None,
) -> tuple[float, str, dict]:
    keyword = _keyword_match_score(question, chunks)
    semantic = _semantic_score(chunks)
    source_quality = _source_quality_score(chunk_metadata or [], conflict_detected=conflict_detected)
    completeness = _completeness_score(answer_text)
    agreement = _source_agreement_score(conflict_detected)
    relevance = _answer_relevance_score(question, answer_text, is_refusal=is_refusal)
    direction = _direction_mismatch_score(direction_mismatch)

    raw_score = (
        KEYWORD_WEIGHT * keyword
        + SEMANTIC_WEIGHT * semantic
        + SOURCE_QUALITY_WEIGHT * source_quality
        + COMPLETENESS_WEIGHT * completeness
        + SOURCE_AGREEMENT_WEIGHT * agreement
        + ANSWER_RELEVANCE_WEIGHT * relevance
        + DIRECTION_MISMATCH_WEIGHT * direction
    ) * 100

    if not validation_passed:
        score = raw_score * VALIDATION_PENALTY
    else:
        score = raw_score

    # Clamp verified refusals into the Yellow band
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
        "validation_passed": validation_passed,
        "retrieval_confidence": round(retrieval_confidence, 4) if retrieval_confidence is not None else None,
    }

    return round(score, 2), level, breakdown