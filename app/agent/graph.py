"""LangGraph implementation of the Answer Generation Agent (FYP Module 3).

Mirrors the architecture diagram:
  Planner -> retrieve (hybrid search) -> draft -> validate
  -> Conditional Router: pass (done) / fail (loop back to draft, up to N times)
  -> confidence scoring -> human validation queue if not green
"""
import logging
from typing import TypedDict

from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from app.config import settings
from app.rag.confidence import score_answer
from app.rag.groq_client import get_groq_client
from app.rag.retrieval import RetrievedChunk, hybrid_search

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    question: str
    organization_id: str
    chunks: list[RetrievedChunk]
    draft: str
    validation_passed: bool
    validation_feedback: str
    attempts: int
    confidence_score: float
    confidence_level: str
    confidence_breakdown: dict
    source_conflict_detected: bool
    conflict_description: str
    is_refusal: bool
    direction_mismatch: bool


SYSTEM_DRAFT = (
    "You are a Senior Enterprise Compliance Director authoring authoritative, clear responses "
    "to vendor security questionnaires and RFPs on behalf of our company.\n\n"
    "Executive Tone & Writing Style:\n"
    "- Write with direct, confident executive phrasing; eliminate filler preambles like "
    "'Based on the provided documents', 'According to the context', or 'As stated in the materials'.\n"
    "- Use natural punctuation, cadence, and sentence structures—incorporate em dashes (—) and semicolons "
    "where helpful to synthesize complex operational or legal standards.\n"
    "- State commitments plainly; do not hedge unless the documentation specifies a procedural exception.\n"
    "- Do NOT use markdown styling such as bolding (**text**) or italics (*text*); produce clean, natural prose.\n"
    "- CITATION RESTRICTION (CRITICAL): Absolutely never mention 'Source 1', 'Source 2', 'Sources 1 and 2', "
    "or bracketed tags like '[Effective Date: ...]' anywhere in your text. Source tracking is managed out-of-band "
    "by the platform. Refer to standards by their policy topic (e.g., 'Under current data protection procedures...' "
    "or 'Current organizational policy sets...') rather than index labels.\n\n"
    "Temporal Precedence & Conflict Resolution Rules (CRITICAL):\n"
    "- Each context passage is tagged with an [Effective Date: YYYY-MM-DD]. Treat newer dates as superseding older dates.\n"
    "- Direct Conflict / Override: If an older document and a newer document contradict each other regarding the same metric, "
    "SLA, timeline, or policy rule, the NEWER document always takes complete precedence. Adopt the figure from the newer document "
    "and treat the older figure as superseded. Do NOT blend them into an artificial compromise or range.\n"
    "- Non-Conflicting Fallback: If a specific detail, technical spec, or step is NOT mentioned in the newer document, but is "
    "clearly defined in an older document and has not been explicitly revoked, you MUST use that information from the older document. "
    "Newer documents do not wipe out older granular rules unless they directly contradict or replace them.\n\n"
    "Perspective & Party Role Boundaries:\n"
    "- 'Our company' is the entity on whose behalf you are responding.\n"
    "- Customer Terms (Customer -> Company) govern client obligations to us. They do NOT describe company obligations to vendors.\n"
    "- Vendor Terms (Company -> Vendor) govern supplier obligations. NEVER confuse the two directions.\n\n"
    "Factual Guardrails:\n"
    "- Answer EXCLUSIVELY from provided context. If an item is absent across both new and older documents, state clearly that "
    "it is not addressed without apologetic filler.\n"
    "- Reject false premises in questions (e.g., claiming an unverified failover window) and assert the real documented figures.\n"
    "- Synthesize table cell data naturally into prose sentences rather than dumping raw markdown pipes."
)


class ContextGradeResult(BaseModel):
    conflict_detected: bool = Field(
        description="True if the retrieved sources disagree with each other "
        "on a specific fact relevant to the question"
    )
    conflict_description: str = Field(
        description="Short description of the conflicting values if "
        "conflict_detected is True, empty string otherwise"
    )


SYSTEM_GRADE_CONTEXT = (
    "You are reviewing retrieved source material BEFORE any answer is "
    "written. Given a question and the retrieved context (which may include "
    "TABLE data), determine only one thing: do any of the sources disagree "
    "with each other on a specific fact relevant to the question (e.g. two "
    "different dollar amounts, dates, or time windows for what appears to "
    "be the same underlying policy)? Do not evaluate any answer — none "
    "exists yet. Just grade whether the source material itself is internally "
    "consistent for this question."
)


class ValidationResult(BaseModel):
    """Structured output schema for the validate step. All fields are required
    (Groq's strict structured-outputs mode requires this)."""
    direction_mismatch: bool = Field(
        description="True if a policy/clause was applied to the wrong party or "
        "reversed relationship relative to the question (e.g. Customer-pays-Company "
        "clause cited for Company-pays-Vendor question)"
    )
    verdict_passed: bool = Field(
        description="True only if the draft is fully supported by context, directly "
        "answers the question, AND direction_mismatch is False"
    )
    verdict_reason: str = Field(
        description="Short reason if verdict_passed is False, empty string otherwise"
    )
    is_refusal: bool = Field(
        description="True if the draft states, implies, or hedges that the information "
        "is not in the context, not mentioned, not covered, unknown, or unable to be confirmed"
    )


SYSTEM_VALIDATE = (
    "You are a strict QA reviewer for compliance questionnaire answers.\n\n"
    "Evaluate the draft answer against the provided context and question. "
    "Your response will be parsed as structured JSON — focus on accuracy, not formatting.\n\n"
    "Evaluation criteria:\n\n"
    "1. direction_mismatch: Check the direction of the relationship and parties:\n"
    "   - In customer contracts, 'Customer' is a client paying our company. A clause where "
    "'Customer pays Company late' (e.g. 1.5% interest) governs customers, NOT our company paying vendors.\n"
    "   - If the question asks about 'company paying a vendor' or 'vendor invoice payment', "
    "but the draft cites a clause where 'Customer shall pay...', or equates the company to "
    "'Customer', this is a direction mismatch.\n"
    "   - Set true if a clause/policy is applied to the wrong party or reversed relationship.\n\n"
    "2. verdict_passed:\n"
    "   - If direction_mismatch is true, verdict_passed MUST be false.\n"
    "   - Is the draft fully supported by the context and does it directly answer the question "
    "accurately? If numbers/facts are cited, do they match the context?\n"
    "   - Set false if the draft adopts or rationalizes false premises from the question (e.g., agreeing with a metric not supported by the context).\n"
    "   - Set false if the draft relies on speculative assumptions, unstated mappings, or hypothetical conditions ('If X is treated as...') not explicitly stated in the context.\n"
    "   - Set true only if the draft passes all checks.\n\n"
    "3. verdict_reason: If verdict_passed is false, provide a short reason. "
    "If direction_mismatch is true, mention the direction mismatch specifically. "
    "Empty string if verdict_passed is true.\n\n"
    "4. is_refusal: Does the draft state, imply, or hedge that the information is NOT in the "
    "context, not mentioned, not covered, unknown, or unable to be confirmed? Set true if so."
)


async def node_retrieve(state: AgentState) -> AgentState:
    chunks = await hybrid_search(state["question"], organization_id=state["organization_id"])
    logger.info(f"node_retrieve: found {len(chunks)} chunks")
    return {"chunks": chunks}


async def node_grade_context(state: AgentState) -> AgentState:
    context = _build_context(state.get("chunks", []))
    user_prompt = f"Context:\n{context}\n\nQuestion: {state['question']}"
    raw = await get_groq_client().chat_structured(
        system=SYSTEM_GRADE_CONTEXT,
        user=user_prompt,
        model=settings.groq_draft_model,  # 20B model for reading comprehension, preserving 120B TPM quota
        schema=ContextGradeResult.model_json_schema(),
        schema_name="ContextGradeResult",
        temperature=0.0,
    )
    result = ContextGradeResult(**raw)
    logger.info(f"node_grade_context: conflict_detected={result.conflict_detected}")
    return {
        "source_conflict_detected": result.conflict_detected,
        "conflict_description": result.conflict_description,
    }


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks):
        date_str = getattr(c, "effective_date", None) or "Undated"
        section_str = c.heading_path or "General"
        if c.content_type == 'table':
            parts.append(f"[Source {i+1} | Effective Date: {date_str} | TABLE from: {section_str}]\n{c.content}")
        else:
            parts.append(f"[Source {i+1} | Effective Date: {date_str} | Section: {section_str}]\n{c.content}")
    return "\n\n---\n\n".join(parts)


async def node_draft(state: AgentState) -> AgentState:
    attempt_num = state.get("attempts", 0) + 1
    logger.info(f"node_draft: attempt {attempt_num}")
    context = _build_context(state.get("chunks", []))
    feedback = state.get("validation_feedback", "")

    conflict_note = ""
    if state.get("source_conflict_detected"):
        conflict_note = (
            f"\n\nNOTE: The retrieved sources show divergent information: "
            f"{state.get('conflict_description', '')}. Follow the Temporal Precedence "
            f"rules: if the discrepancy stems from an older document versus a newer document, "
            f"adopt the newer document's metric as authoritative. If the dates are identical or "
            f"the hierarchy is ambiguous, explicitly state both figures and identify the discrepancy."
        )
    user_prompt = f"Context:\n{context}\n\nQuestion: {state['question']}{conflict_note}"
    if feedback:
        user_prompt += f"\n\nYour previous attempt was rejected: {feedback}\nRevise accordingly."

    answer = await get_groq_client().chat(SYSTEM_DRAFT, user_prompt, model=settings.groq_draft_model)
    return {
        "draft": answer,
        "attempts": attempt_num,
    }


async def node_validate(state: AgentState) -> AgentState:
    context = _build_context(state.get("chunks", []))
    user_prompt = (
        f"Context:\n{context}\n\nQuestion: {state['question']}\n\n"
        f"Draft answer: {state['draft']}"
    )

    # Use structured outputs — Groq's strict JSON schema mode guarantees
    # the response conforms to ValidationResult's schema. No more fragile
    # line-by-line string parsing that breaks on markdown formatting.
    raw = await get_groq_client().chat_structured(
        system=SYSTEM_VALIDATE,
        user=user_prompt,
        model=settings.groq_validate_model,
        schema=ValidationResult.model_json_schema(),
        schema_name="ValidationResult",
        temperature=0.0,
    )
    result = ValidationResult(**raw)

    passed = result.verdict_passed and not result.direction_mismatch

    logger.info(
        f"node_validate: verdict passed={passed}, "
        f"is_refusal={result.is_refusal}, direction_mismatch={result.direction_mismatch}"
    )
    feedback = ""
    if not passed:
        if result.direction_mismatch and result.verdict_passed:
            # Model said pass but direction is wrong — override with clear feedback
            feedback = (
                "FAIL: direction mismatch — you cited a Customer-pays-Company clause "
                "for a question asking about Company paying a Vendor. If the context "
                "does not cover vendor payment terms, state that it is not covered."
            )
        else:
            feedback = result.verdict_reason or "FAIL: validation did not pass"

    return {
        "validation_passed": passed,
        "validation_feedback": feedback,
        "is_refusal": result.is_refusal,
        "direction_mismatch": result.direction_mismatch,
    }


def route_after_validate(state: AgentState) -> str:
    if state.get("validation_passed"):
        return "score"
    if state.get("attempts", 0) >= settings.max_refine_attempts:
        # Give up refining — score whatever we have so low confidence
        # routes it to human review instead of looping forever.
        return "score"
    return "draft"


async def node_score(state: AgentState) -> AgentState:
    chunks = state.get("chunks", [])
    chunk_metadata = [
        {"updated_at": c.created_at, "times_used": c.times_used} for c in chunks
    ]
    conflict_detected = state.get("source_conflict_detected", False)
    is_refusal = state.get("is_refusal")
    direction_mismatch = state.get("direction_mismatch")
    score, level, breakdown = score_answer(
        state["question"], state["draft"], chunks,
        chunk_metadata=chunk_metadata, conflict_detected=conflict_detected,
        is_refusal=is_refusal, direction_mismatch=direction_mismatch,
    )
    logger.info(f"node_score: score={score}, level={level}, breakdown={breakdown}")
    return {"confidence_score": score, "confidence_level": level, "confidence_breakdown": breakdown}


def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("grade_context", node_grade_context)
    graph.add_node("generate_draft", node_draft)
    graph.add_node("validate", node_validate)
    graph.add_node("score", node_score)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_context")
    graph.add_edge("grade_context", "generate_draft")
    graph.add_edge("generate_draft", "validate")
    graph.add_conditional_edges(
        "validate", route_after_validate, {"draft": "generate_draft", "score": "score"}
    )
    graph.add_edge("score", END)

    return graph.compile()


agent_app = build_agent_graph()


async def run_agent(question: str, organization_id: str) -> AgentState:
    result = await agent_app.ainvoke(
        {"question": question, "organization_id": organization_id, "attempts": 0}
    )
    return result
