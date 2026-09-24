import logging
import re
from datetime import datetime, timezone

from app.rag.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

KEYWORD_WEIGHT = 0.12
SEMANTIC_WEIGHT = 0.22
SOURCE_QUALITY_WEIGHT = 0.08
COMPLETENESS_WEIGHT = 0.08
SOURCE_AGREEMENT_WEIGHT = 0.18
ANSWER_RELEVANCE_WEIGHT = 0.18
DIRECTION_MISMATCH_WEIGHT = 0.14
VALIDATION_PENALTY = 0.40  # Reduces confidence to <50% (Red) when ungrounded

# Weak-retrieval hard gate thresholds.
# Calibrated against RAG-Evaluation-V4.json: all well-grounded green answers
# had keyword_match >= 0.6 and semantic_similarity >= 0.75. The Q21 failure
# case (weak retrieval, defensible answer, false-green) was at 0.462 / 0.623.
# Both thresholds must be breached simultaneously — a single weak signal may
# just reflect unusual vocabulary or embedding compression, not a true
# retrieval gap.
WEAK_RETRIEVAL_KEYWORD_THRESHOLD = 0.55
WEAK_RETRIEVAL_SEMANTIC_THRESHOLD = 0.70
WEAK_RETRIEVAL_SCORE_CAP = 79.0  # Max score when gate fires (top of yellow)


def _keyword_match_score(question: str, chunks: list[RetrievedChunk]) -> float:
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
    top = sorted((c.vector_score for c in chunks), reverse=True)[:3]
    return sum(top) / len(top)


def _source_quality_score(chunk_metadata: list[dict], conflict_detected: bool = False) -> float:
    """Enterprise Source Quality:
    - If no conflict exists between sources, age is irrelevant: unrevoked policies are 100% authoritative.
    - If a conflict IS detected, favor newer documents over older ones.
    - If updated_at is missing, default to neutral authority (0.85), not a failure penalty.
    """
    if not chunk_metadata:
        return 0.85
    
    # When all retrieved sources agree, legacy documentation remains fully authoritative
    if not conflict_detected:
        return 1.0

    # Under conflict, penalize older chunks to prioritize newer amendments
    scores = []
    now = datetime.now(timezone.utc)
    for m in chunk_metadata:
        recency = 0.70
        updated_at = m.get("updated_at")
        if updated_at:
            try:
                dt = datetime.fromisoformat(updated_at)
                age_days = (now - dt).days
                recency = max(0.40, 1.0 - (age_days / 1825))  # 5-year graceful window
            except ValueError:
                pass
        scores.append(recency)
    return sum(scores) / len(scores)


def _completeness_score(answer_text: str) -> float:
    length = len(answer_text.split())
    # Do not penalize concise answers (e.g., "25 Business Days.")
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

# Epistemic refusals only. Removed "contain" and "provisions" to protect legal negatives.
# Added "define" and "prescribe".
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
    # Primary Gate: LLM Structured Output Validator is the sole authority
    # for refusal classification. It reads the actual retrieved context and
    # can distinguish "information genuinely absent" from "confidently
    # rejecting a false premise using facts it found."
    if is_refusal is True:
        return 0.2

    # Diagnostic layer: regex patterns detect refusal-like phrasing for
    # observability. When the regex fires but is_refusal is False, we log
    # a warning for drift detection but do NOT override the score — the
    # 120B validator has strictly more information than a regex can access.
    lower = answer_text.lower()
    regex_hit = any(re.search(p, lower) for p in _STRONG_REFUSAL_PATTERNS)
    if not regex_hit:
        regex_hit = any(
            re.search(p, lower) and _SELF_REFERENCE_RE.search(lower)
            for p in _WEAK_REFUSAL_PATTERNS
        )

    if regex_hit and is_refusal is False:
        logger.warning(
            "Regex refusal pattern detected but is_refusal=False from validator; "
            "deferring to validator. Answer excerpt: %s",
            answer_text[:120],
        )

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
    """Score a drafted answer on a 0-100 scale from seven weighted signals.

    Args:
        retrieval_confidence: Mean RRF combined_score of retrieved chunks.
            This is a better signal of retrieval quality than the per-signal
            proxies (keyword_match, semantic_similarity) because it encodes
            how well each chunk matched BOTH vector and keyword search
            relative to the entire corpus, including the temporal recency
            boost. The per-signal proxies reconstruct partial views from
            raw scores that sit on different scales. Threading the actual
            RRF score avoids this information loss.
    """
    keyword = _keyword_match_score(question, chunks)
    semantic = _semantic_score(chunks)
    source_quality = _source_quality_score(chunk_metadata or [], conflict_detected=conflict_detected)
    completeness = _completeness_score(answer_text)
    agreement = _source_agreement_score(conflict_detected)
    
    # Pass question and answer directly into relevance validation
    relevance = _answer_relevance_score(
        question=question,
        answer_text=answer_text,
        is_refusal=is_refusal,
    )
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

    if relevance < 0.5:
        score = min(58.0, max(50.0, score * 0.85))

    # ── Weak-Retrieval Hard Gate ─────────────────────────────────────────
    # When BOTH retrieval-evidence signals are below their calibrated
    # thresholds, the five default-to-1.0 signals cannot be trusted to
    # reflect genuine confidence — they merely reflect the absence of a
    # detected problem, not the presence of strong evidence. Cap the score
    # at the top of yellow to force human review.
    #
    # This gate is intentionally an explicit, named block (not a silent
    # weight adjustment) so it is visible in code review and independently
    # testable. See WEAK_RETRIEVAL_* constants for threshold rationale.
    weak_retrieval_gate_fired = False
    if keyword < WEAK_RETRIEVAL_KEYWORD_THRESHOLD and semantic < WEAK_RETRIEVAL_SEMANTIC_THRESHOLD:
        if score > WEAK_RETRIEVAL_SCORE_CAP:
            logger.info(
                "Weak-retrieval gate fired: keyword=%.3f (<%s), semantic=%.3f (<%s). "
                "Capping score from %.2f to %.1f (yellow).",
                keyword, WEAK_RETRIEVAL_KEYWORD_THRESHOLD,
                semantic, WEAK_RETRIEVAL_SEMANTIC_THRESHOLD,
                score, WEAK_RETRIEVAL_SCORE_CAP,
            )
            score = WEAK_RETRIEVAL_SCORE_CAP
        weak_retrieval_gate_fired = True

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
        "weak_retrieval_gate_fired": weak_retrieval_gate_fired,
    }

    return round(score, 2), level, breakdown