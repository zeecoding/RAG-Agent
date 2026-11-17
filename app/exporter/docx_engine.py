import csv
import json
"""OpenXML (.docx) Format-Preserving Exporter & In-Place Mutation Engine.

Performs surgical, coordinate-targeted writes into in-memory docx buffers without
destroying surrounding styles, XML tags (w:tcPr, w:pPr), headers, footers, margins,
or embedded media/logos.
"""

import io
import logging
import re
from typing import Any

import docx
from docx.oxml import OxmlElement
from docx.shared import Pt, RGBColor

from app.exporter.schemas import (
    AnsweredItem,
    DocxParagraphCoordinate,
    DocxTableCoordinate,
    ExtractedQuestionItem,
)

logger = logging.getLogger(__name__)

QUESTION_COLUMN_TOKENS = (
    "requirement",
    "question",
    "control",
    "description",
    "item",
    "audit item",
    "prompt",
    "security standard",
    "topic",
    "rfp question",
    "inquiry",
    "criteria",
    "assessment",
    "control inquiry",
    "check",
    "clause",
    "specification",
    "policy",
)

RESPONSE_COLUMN_TOKENS = (
    "vendor response",
    "response",
    "comments",
    "evidence",
    "answer",
    "vendor comment",
    "supplier response",
    "compliance response",
    "findings",
    "provider response",
    "organization response",
    "reply",
    "vendor answer",
    "status",
    "result",
    "notes",
    "confirmation",
)

# Explicit prompt patterns: matches Q1:, Question 1:, Requirement 1:, Item 1:, Prompt 1:, etc.
EXPLICIT_PROMPT_RE = re.compile(
    r"^(?:Question|Requirement|Prompt|Item|Control|Inquiry|Q)\s*\d+[:.]?",
    re.IGNORECASE,
)

# Prefix cleaner to strip "Question 2:", "Q1:", "Requirement 1:", "1. ", etc. so only the question is saved
QUESTION_PREFIX_STRIP_RE = re.compile(
    r"^(?:(?:Question|Requirement|Prompt|Item|Control|Inquiry|Q)\s*\d+[:.\-\s]+|\d+[\.\)]\s+)",
    re.IGNORECASE,
)

def strip_question_prefix(text: str) -> str:
    """Strips leading 'Question 2:', 'Q1:', '1. ', etc. before the question prompt."""
    if not text:
        return text
    lines = text.split("\n", 1)
    first_line = QUESTION_PREFIX_STRIP_RE.sub("", lines[0].strip()).strip()
    if not first_line:
        first_line = lines[0].strip()
    if len(lines) > 1:
        return first_line + "\n" + lines[1]
    return first_line

# Numbered question prompt pattern: e.g. "1. What is...", "2. Describe...", "3) Confirm..."
NUMBERED_QUESTION_RE = re.compile(
    r"^\d+[\.\)]\s+(?:what|how|within|describe|confirm|explain|detail|provide|verify|list|state|are|is|do|does|can|could|should|must|which|who|where|when|why|identify|outline|summarize|specify)\b",
    re.IGNORECASE,
)

# Section headers to explicitly exclude: e.g. "1. Vendor Assessment Instructions & Scope"
SECTION_HEADER_RE = re.compile(
    r"^\d+[\.\)]\s+(?:Part\s+[A-Z]|Section\s+\d+|Vendor\s+Assessment|Instructions|Scope|Appendix|Overview|Controls|Schedule|Summary|Introduction|Table\s+of\s+Contents)\b",
    re.IGNORECASE,
)

IGNORED_STARTS = (
    "instructions:",
    "instruction:",
    "figure ",
    "document control:",
    "confidentiality notice:",
    "disclaimer:",
    "table of contents:",
    "note:",
    "guidance & explanation:",
    "guidance:",
    "explanation:",
)

ANSWER_LABEL_RE = re.compile(
    r"^(?:answer|response|vendor\s+response|provider\s+response|comments?|compliance\s+response|a)\s*:\s*$",
    re.IGNORECASE,
)

_PLACEHOLDER_TOKENS = (
    "[enter",
    "[awaiting",
    "[pending",
    "[response",
    "[answer",
    "[tbd",
    "[n/a",
    "[vendor",
    "vendor response",
    "[your answer",
)

GUIDANCE_PREFIXES = (
    "guidance & explanation:",
    "guidance:",
    "explanation:",
    "instructions:",
    "instruction:",
    "note:",
    "helpful hint",
)

DEFAULT_FONT_NAME = "Calibri"
DEFAULT_FONT_SIZE_PT = 9.5
DEFAULT_COLOR_RGB = (0x0F, 0x17, 0x2A)  # Slate-900

BOLD_MD_REGEX = re.compile(r"\*\*(.*?)\*\*")


def _is_answer_placeholder(text: str) -> bool:
    """Returns True if text is a standalone placeholder or answer label line."""
    lower = text.lower().strip()
    if not lower:
        return False
    if ANSWER_LABEL_RE.match(lower):
        return True
    if any(tok in lower for tok in _PLACEHOLDER_TOKENS):
        return True
    if lower.startswith("[") and any(w in lower for w in ("response", "answer", "vendor", "comment")):
        return True
    if set(lower) <= {"_", "-", ".", " "}:
        return True
    return False


def _has_trailing_placeholder(text: str) -> bool:
    """Returns True if text ends with an inline placeholder like '[Vendor Response: ]'."""
    lower = text.lower().strip()
    return any(lower.endswith(tok) for tok in _PLACEHOLDER_TOKENS) or bool(
        re.search(r"\[vendor\s+response:?\s*\]$", lower)
    )


def _strip_trailing_placeholder(text: str) -> str:
    """Removes trailing '[Vendor Response: ]' or similar tags from paragraph text."""
    return re.sub(
        r"\[(?:vendor\s+response|enter\s+vendor\s+response|response|answer):?\s*\]$",
        "",
        text,
        flags=re.IGNORECASE,
    ).rstrip()


def _is_guidance_paragraph(text: str) -> bool:
    """Returns True if text is an instructional guidance block or details explaining how to answer."""
    lower = text.lower().strip()
    return (
        any(lower.startswith(pfx) for pfx in GUIDANCE_PREFIXES)
        or (lower.startswith("(") and lower.endswith(")"))
    )


def _find_table_header_columns(table) -> tuple[int, int, int] | None:
    max_scan_rows = min(3, len(table.rows))

    for r_idx in range(max_scan_rows):
        row = table.rows[r_idx]
        cells = row.cells
        if len(cells) < 2:
            continue

        q_col_candidate: int | None = None
        resp_col_candidate: int | None = None

        for c_idx, cell in enumerate(cells):
            cell_text = cell.text.strip().lower()
            if not cell_text:
                continue

            if any(t in cell_text for t in RESPONSE_COLUMN_TOKENS):
                if resp_col_candidate is None:
                    resp_col_candidate = c_idx
            elif any(t in cell_text for t in QUESTION_COLUMN_TOKENS):
                if q_col_candidate is None:
                    q_col_candidate = c_idx

        if (
            q_col_candidate is not None
            and resp_col_candidate is not None
            and q_col_candidate != resp_col_candidate
        ):
            return r_idx, q_col_candidate, resp_col_candidate

    # Dynamic heuristic fallback: doesn't hardcode column numbers
    if len(table.rows) >= 2:
        num_cols = len(table.columns)
        if num_cols == 2:
            return 0, 0, 1
        elif num_cols >= 3:
            # First column is usually ID (e.g. Q1), middle column is Question, last column is Response
            return 0, 1, num_cols - 1

    return None


def extract_docx_questions(file_bytes: bytes) -> tuple[list[ExtractedQuestionItem], bytes]:
    """Extracts questions and precise coordinate positions from a .docx file in natural document reading order."""
    if not file_bytes:
        return [], file_bytes

    doc = docx.Document(io.BytesIO(file_bytes))
    extracted: list[ExtractedQuestionItem] = []

    paragraphs_map = {p._p: (idx, p) for idx, p in enumerate(doc.paragraphs)}
    tables_map = {t._tbl: (idx, t) for idx, t in enumerate(doc.tables)}
    processed_p_indices: set[int] = set()

    # Traverse document body elements in exact reading flow order
    for child in doc.element.body:
        if child in paragraphs_map:
            p_idx, p = paragraphs_map[child]
            if p_idx in processed_p_indices:
                continue

            p_text = p.text.strip()
            if not p_text:
                continue

            p_lower = p_text.lower()
            if any(p_lower.startswith(prefix) for prefix in IGNORED_STARTS):
                continue

            if SECTION_HEADER_RE.search(p_text):
                continue

            is_explicit = bool(EXPLICIT_PROMPT_RE.match(p_text))
            is_numbered_q = bool(NUMBERED_QUESTION_RE.match(p_text))
            is_question_mark = p_text.endswith("?") and len(p_text) >= 15

            # Also check if subsequent paragraph is an Answer: or Response: placeholder label
            next_is_answer = False
            if p_idx + 1 < len(doc.paragraphs):
                next_text = doc.paragraphs[p_idx + 1].text.strip()
                if _is_answer_placeholder(next_text):
                    next_is_answer = True

            if is_explicit or is_numbered_q or (is_question_mark and not p_lower.startswith("figure")) or (next_is_answer and len(p_text) >= 15):
                full_question = p_text
                scan_idx = p_idx + 1

                # Capture any subsequent detail/guidance paragraphs explaining how to answer
                while scan_idx < len(doc.paragraphs):
                    cand_text = doc.paragraphs[scan_idx].text.strip()
                    if _is_answer_placeholder(cand_text):
                        break
                    if _is_guidance_paragraph(cand_text):
                        full_question += "\n" + cand_text
                        processed_p_indices.add(scan_idx)
                        scan_idx += 1
                    else:
                        break

                has_placeholder = False
                target_idx = scan_idx
                if scan_idx < len(doc.paragraphs):
                    cand_text = doc.paragraphs[scan_idx].text.strip()
                    if _is_answer_placeholder(cand_text):
                        has_placeholder = True
                        processed_p_indices.add(scan_idx)

                coord = DocxParagraphCoordinate(
                    target_type="paragraph",
                    paragraph_idx=target_idx if has_placeholder else (scan_idx - 1),
                    insert_after=not has_placeholder,
                )

                extracted.append(
                    ExtractedQuestionItem(
                        question_text=strip_question_prefix(full_question),
                        coordinates=coord,
                    )
                )
                processed_p_indices.add(p_idx)

        elif child in tables_map:
            t_idx, table = tables_map[child]
            header_info = _find_table_header_columns(table)
            if not header_info:
                continue

            header_row_idx, q_col_idx, resp_col_idx = header_info

            for r_idx in range(header_row_idx + 1, len(table.rows)):
                row = table.rows[r_idx]
                if q_col_idx >= len(row.cells) or resp_col_idx >= len(row.cells):
                    continue

                q_cell = row.cells[q_col_idx]
                resp_cell = row.cells[resp_col_idx]

                if q_cell._tc == resp_cell._tc:
                    continue

                q_text = q_cell.text.strip()
                if not q_text or set(q_text) <= {"-", "_", " ", "\t", "\n"}:
                    continue

                extracted.append(
                    ExtractedQuestionItem(
                        question_text=strip_question_prefix(q_text),
                        coordinates=DocxTableCoordinate(
                            target_type="table_cell",
                            table_idx=t_idx,
                            row_idx=r_idx,
                            target_col_idx=resp_col_idx,
                        ),
                    )
                )

    _buf = io.BytesIO()
    doc.save(_buf)
    template_bytes = _buf.getvalue()

    return extracted, template_bytes


def _clean_markdown_text(text: str) -> str:
    cleaned = re.sub(r"(?m)^\s*[-*•]\s+", "", text)
    cleaned = cleaned.replace("  \n", "\n").replace("\r\n", "\n")
    return cleaned


def _normalize_match_text(text: str) -> str:
    if not text:
        return ""
    # Strip common leading markers like "Q1.", "1.", "Requirement 1:", whitespace
    t = re.sub(r"^(?:q\d+[:.]|requirement\s+\d+[:.]|\d+[\.\)]\s+)", "", text.strip(), flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def _text_matches(candidate: str, target: str) -> bool:
    c_norm = _normalize_match_text(candidate)
    t_norm = _normalize_match_text(target)
    if not c_norm or not t_norm:
        return False
    if c_norm == t_norm:
        return True
    if len(t_norm) >= 15 and (t_norm in c_norm or c_norm in t_norm):
        return True
    words_c = set(re.findall(r"\w+", c_norm))
    words_t = set(re.findall(r"\w+", t_norm))
    if words_c and words_t:
        overlap = len(words_c & words_t) / max(len(words_c), len(words_t))
        if overlap >= 0.70:
            return True
    return False


def _clear_paragraph_runs_xml(paragraph) -> None:
    """Completely strips all <w:r> run elements from the paragraph XML tree."""
    p_elem = paragraph._p
    for child in list(p_elem):
        if child.tag.endswith("r"):
            p_elem.remove(child)


def _clear_cell_completely(cell) -> Any:
    """Wipes all text, runs, paragraphs, hyperlinks, and content controls (sdt)
    from a table cell, preserving only its cell properties (w:tcPr).
    """
    tc_elem = cell._tc
    for child in list(tc_elem):
        if not child.tag.endswith("tcPr"):
            tc_elem.remove(child)

    return cell.add_paragraph()


def _write_paragraph_runs(
    paragraph,
    text: str,
    font_name: str = DEFAULT_FONT_NAME,
    font_size_pt: float = DEFAULT_FONT_SIZE_PT,
    color_rgb: tuple[int, int, int] = DEFAULT_COLOR_RGB,
    convert_bold_markdown: bool = True,
) -> None:
    _clear_paragraph_runs_xml(paragraph)

    sanitized_text = _clean_markdown_text(text)

    if not convert_bold_markdown or "**" not in sanitized_text:
        clean_plain = sanitized_text.replace("*", "")
        r = paragraph.add_run(clean_plain)
        r.font.name = font_name
        r.font.size = Pt(font_size_pt)
        r.font.color.rgb = RGBColor(*color_rgb)
        r.font.bold = False
        r.font.italic = False
        return

    parts = BOLD_MD_REGEX.split(sanitized_text)
    for idx, part in enumerate(parts):
        if not part:
            continue
        is_bold = (idx % 2 == 1)
        r = paragraph.add_run(part)
        r.font.name = font_name
        r.font.size = Pt(font_size_pt)
        r.font.color.rgb = RGBColor(*color_rgb)
        r.font.bold = is_bold
        r.font.italic = False


def _write_cell_in_place(
    cell,
    text: str,
    font_name: str = DEFAULT_FONT_NAME,
    font_size_pt: float = DEFAULT_FONT_SIZE_PT,
    color_rgb: tuple[int, int, int] = DEFAULT_COLOR_RGB,
) -> None:
    """Completely empties the target cell before writing the answer runs."""
    p = _clear_cell_completely(cell)
    _write_paragraph_runs(p, text, font_name, font_size_pt, color_rgb)


def write_docx_answers_in_place(
    original_bytes: bytes,
    answered_items: list[dict | AnsweredItem],
    font_name: str = DEFAULT_FONT_NAME,
    font_size_pt: float = DEFAULT_FONT_SIZE_PT,
    color_rgb: tuple[int, int, int] = DEFAULT_COLOR_RGB,
) -> bytes:
    """Performs non-destructive in-place writes without modifying table column XML structures."""
    if not original_bytes:
        return b""

    doc = docx.Document(io.BytesIO(original_bytes))
    orig_paragraphs = list(doc.paragraphs)

    for item in answered_items:
        if isinstance(item, AnsweredItem):
            coords = item.coordinates
            ans_text = item.answer
            q_text = getattr(item, "question_text", getattr(item, "question", ""))
        elif isinstance(item, dict):
            coords = item.get("coordinates") or item
            ans_text = item.get("answer") or item.get("draft_answer") or item.get("answer_text", "")
            q_text = item.get("question") or item.get("question_text", "")
        else:
            coords = getattr(item, "coordinates", item)
            ans_text = getattr(item, "answer", getattr(item, "draft_answer", ""))
            q_text = getattr(item, "question", getattr(item, "question_text", ""))

        if not coords:
            continue

        target_type = coords.get("target_type") if isinstance(coords, dict) else getattr(coords, "target_type", None)

        if target_type == "table_cell":
            table_idx = coords["table_idx"] if isinstance(coords, dict) else coords.table_idx
            row_idx = coords["row_idx"] if isinstance(coords, dict) else coords.row_idx
            target_col_idx = coords["target_col_idx"] if isinstance(coords, dict) else coords.target_col_idx

            if 0 <= table_idx < len(doc.tables):
                table = doc.tables[table_idx]
                target_row = None

                # 1. Dynamic Anchor Check: If q_text provided, check if row has shifted
                if q_text:
                    # Check current row_idx first
                    if 0 <= row_idx < len(table.rows):
                        for cell in table.rows[row_idx].cells:
                            if _text_matches(cell.text, q_text):
                                target_row = table.rows[row_idx]
                                break

                    # If shifted (e.g. user added rows above), scan table rows dynamically
                    if target_row is None:
                        for r_i, r in enumerate(table.rows):
                            for cell in r.cells:
                                if _text_matches(cell.text, q_text):
                                    target_row = r
                                    row_idx = r_i
                                    if isinstance(coords, dict):
                                        coords["row_idx"] = r_i
                                    break
                            if target_row is not None:
                                break

                # Fallback to direct row index if anchor search didn't run or didn't find match
                if target_row is None and 0 <= row_idx < len(table.rows):
                    target_row = table.rows[row_idx]

                if target_row is not None:
                    if 0 <= target_col_idx < len(target_row.cells):
                        cell = target_row.cells[target_col_idx]
                        _write_cell_in_place(cell, ans_text, font_name, font_size_pt, color_rgb)
                    else:
                        logger.warning(
                            "target_col_idx %d out of bounds for table %d, row %d",
                            target_col_idx,
                            table_idx,
                            row_idx,
                        )
                else:
                    logger.warning("row_idx %d out of bounds for table %d", row_idx, table_idx)
            else:
                logger.warning("table_idx %d out of bounds", table_idx)

        elif target_type == "paragraph":
            paragraph_idx = coords.get("paragraph_idx") if isinstance(coords, dict) else coords.paragraph_idx
            insert_after = (
                coords.get("insert_after", False)
                if isinstance(coords, dict)
                else getattr(coords, "insert_after", False)
            )

            # Step 1: Attempt to locate question paragraph by matching text
            question_p = None
            question_p_idx = None

            if q_text:
                # Check original index first if within bounds
                if 0 <= paragraph_idx < len(doc.paragraphs) and _text_matches(doc.paragraphs[paragraph_idx].text, q_text):
                    question_p = doc.paragraphs[paragraph_idx]
                    question_p_idx = paragraph_idx
                else:
                    # Scan document paragraphs for match with q_text
                    for p_i, p in enumerate(doc.paragraphs):
                        if _text_matches(p.text, q_text):
                            question_p = p
                            question_p_idx = p_i
                            if isinstance(coords, dict):
                                coords["paragraph_idx"] = p_i
                            break

            # Step 2: If question paragraph was found, NEVER OVERWRITE IT!
            # Locate placeholder paragraph or insert directly after question/guidance.
            if question_p is not None and question_p_idx is not None:
                scan_i = question_p_idx + 1
                # Skip any guidance/instruction paragraphs explaining the question
                while scan_i < len(doc.paragraphs) and _is_guidance_paragraph(doc.paragraphs[scan_i].text):
                    scan_i += 1

                # Check if scan_i points to an answer placeholder paragraph (e.g. "Answer:", "Response:")
                placeholder_p = None
                if scan_i < len(doc.paragraphs):
                    cand_text = doc.paragraphs[scan_i].text.strip()
                    if _is_answer_placeholder(cand_text) or re.match(r"^(?:answer|response|vendor response|a)\s*:\s*", cand_text, re.IGNORECASE):
                        placeholder_p = doc.paragraphs[scan_i]

                if placeholder_p is not None:
                    # Write into placeholder paragraph, preserving label (e.g. "Answer:")
                    orig_label_match = re.match(
                        r"^((?:Answer|Response|Vendor Response|Provider Response|Compliance Response|A)\s*:\s*)",
                        placeholder_p.text.strip(),
                        re.IGNORECASE,
                    )
                    label = orig_label_match.group(1).rstrip() if orig_label_match else "Answer:"
                    if ans_text.lower().startswith(label.lower()):
                        final_text = ans_text
                    else:
                        final_text = f"{label} {ans_text}"
                    _write_paragraph_runs(placeholder_p, final_text, font_name, font_size_pt, color_rgb)
                else:
                    # No placeholder paragraph: insert after last question/guidance paragraph
                    last_content_p = doc.paragraphs[scan_i - 1]
                    new_p = doc.add_paragraph()
                    last_content_p._p.addnext(new_p._p)
                    final_text = ans_text if ans_text.lower().startswith("answer:") else f"Answer: {ans_text}"
                    _write_paragraph_runs(new_p, final_text, font_name, font_size_pt, color_rgb)

            # Step 3: Fallback if question text matching did not locate a question paragraph
            elif 0 <= paragraph_idx < len(doc.paragraphs):
                target_p = doc.paragraphs[paragraph_idx]
                cand_text = target_p.text.strip()
                # If target_p itself is a placeholder paragraph (e.g. "Answer:"), write into it
                if _is_answer_placeholder(cand_text) or re.match(r"^(?:answer|response|vendor response|a)\s*:\s*", cand_text, re.IGNORECASE):
                    orig_label_match = re.match(
                        r"^((?:Answer|Response|Vendor Response|Provider Response|Compliance Response|A)\s*:\s*)",
                        cand_text,
                        re.IGNORECASE,
                    )
                    label = orig_label_match.group(1).rstrip() if orig_label_match else "Answer:"
                    final_text = ans_text if ans_text.lower().startswith(label.lower()) else f"{label} {ans_text}"
                    _write_paragraph_runs(target_p, final_text, font_name, font_size_pt, color_rgb)
                else:
                    # target_p is not a placeholder — do not overwrite! Insert new paragraph after it
                    new_p = doc.add_paragraph()
                    target_p._p.addnext(new_p._p)
                    final_text = ans_text if ans_text.lower().startswith("answer:") else f"Answer: {ans_text}"
                    _write_paragraph_runs(new_p, final_text, font_name, font_size_pt, color_rgb)
            else:
                logger.warning("paragraph_idx %d out of bounds (doc has %d paragraphs)", paragraph_idx, len(doc.paragraphs))

    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()


def locate_and_sync_docx_anchors(
    file_bytes: bytes,
    questions: list[dict],
) -> list[dict]:
    """Scans an updated docx binary (e.g. saved from ONLYOFFICE) against known questions.

    Dynamically matches each question text across tables and paragraphs to:
    1. Recalculate its current coordinates (row_idx, paragraph_idx) if rows or paragraphs were inserted.
    2. Extract the updated response text from the target cell or paragraph.
    Returns a list of updated question dictionaries with 'new_answer', 'new_coordinates', and 'has_changes'.
    """
    if not file_bytes:
        return []

    doc = docx.Document(io.BytesIO(file_bytes))
    updated_questions: list[dict] = []

    # Map table paragraphs to avoid double-matching outside
    table_p_elements = {
        p._p
        for table in doc.tables
        for row in table.rows
        for cell in row.cells
        for p in cell.paragraphs
    }

    for q in questions:
        q_id = q.get("id")
        q_text = q.get("question") or q.get("question_text") or ""
        old_coords = q.get("coordinates") or {}
        old_answer = (q.get("draft_answer") or "").strip()

        if isinstance(old_coords, str):
            try:
                old_coords = json.loads(old_coords)
            except Exception:
                old_coords = {}

        target_type = old_coords.get("target_type")
        matched = False
        new_coords = dict(old_coords)
        new_answer = old_answer

        # 1. Search in Tables
        if target_type == "table_cell" or not target_type:
            table_idx = old_coords.get("table_idx", 0)
            target_col_idx = old_coords.get("target_col_idx", 1)

            # Check specified table first
            tables_to_check = [table_idx] if 0 <= table_idx < len(doc.tables) else list(range(len(doc.tables)))
            for t_i in tables_to_check:
                t = doc.tables[t_i]
                for r_i, row in enumerate(t.rows):
                    row_cells = row.cells
                    # Check if any cell in row matches question text
                    for c_i, cell in enumerate(row_cells):
                        if _text_matches(cell.text, q_text):
                            matched = True
                            # Find response column
                            resp_col = target_col_idx
                            if resp_col >= len(row_cells) or resp_col == c_i:
                                resp_col = (c_i + 1) if (c_i + 1) < len(row_cells) else c_i

                            extracted_text = row_cells[resp_col].text.strip()
                            new_coords = {
                                "target_type": "table_cell",
                                "table_idx": t_i,
                                "row_idx": r_i,
                                "target_col_idx": resp_col,
                            }
                            # Only overwrite answer if cell doesn't just contain a placeholder
                            if extracted_text and not _is_answer_placeholder(extracted_text):
                                new_answer = extracted_text
                            break
                    if matched:
                        break
                if matched:
                    break

        # 2. Search in Paragraphs if not matched in tables
        if not matched:
            for p_i, p in enumerate(doc.paragraphs):
                if p._p in table_p_elements:
                    continue
                if _text_matches(p.text, q_text):
                    matched = True
                    # Check next paragraph for answer
                    resp_text = ""
                    if p_i + 1 < len(doc.paragraphs) and doc.paragraphs[p_i + 1]._p not in table_p_elements:
                        cand_text = doc.paragraphs[p_i + 1].text.strip()
                        if cand_text and not _is_answer_placeholder(cand_text):
                            resp_text = cand_text

                    new_coords = {
                        "target_type": "paragraph",
                        "paragraph_idx": p_i,
                        "insert_after": True,
                    }
                    if resp_text:
                        new_answer = resp_text
                    break

        has_answer_change = (new_answer != old_answer) and bool(new_answer)
        has_coord_change = (new_coords != old_coords)

        updated_questions.append({
            "id": q_id,
            "question": q_text,
            "old_answer": old_answer,
            "new_answer": new_answer,
            "old_coordinates": old_coords,
            "new_coordinates": new_coords,
            "has_changes": has_answer_change or has_coord_change,
            "matched": matched,
        })

    return updated_questions


def write_single_answer_in_place(
    file_bytes: bytes,
    question_text: str,
    coordinates: Any,
    new_answer: str,
    font_name: str = DEFAULT_FONT_NAME,
    font_size_pt: float = DEFAULT_FONT_SIZE_PT,
    color_rgb: tuple[int, int, int] = DEFAULT_COLOR_RGB,
) -> tuple[bytes, dict]:
    """Mutates a single question answer in an existing completed docx buffer.

    Locates the question anchor dynamically to handle prior row/paragraph insertions.
    Returns (updated_bytes, updated_coordinates).
    """
    if isinstance(coordinates, str):
        try:
            coords_dict = json.loads(coordinates)
        except Exception:
            coords_dict = {}
    elif isinstance(coordinates, dict):
        coords_dict = dict(coordinates)
    else:
        coords_dict = {}

    item = {
        "question": question_text,
        "coordinates": coords_dict,
        "draft_answer": new_answer,
    }
    updated_bytes = write_docx_answers_in_place(
        file_bytes,
        [item],
        font_name=font_name,
        font_size_pt=font_size_pt,
        color_rgb=color_rgb,
    )
    return updated_bytes, item["coordinates"]


def extract_xlsx_questions(file_bytes: bytes) -> tuple[list[ExtractedQuestionItem], bytes]:
    """Extract questions and coordinates from an Excel (.xlsx) spreadsheet."""
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
    extracted: list[ExtractedQuestionItem] = []

    for sheet_idx, sheet_name in enumerate(wb.sheetnames):
        sheet = wb[sheet_name]
        rows = list(sheet.iter_rows(values_only=False))
        if not rows:
            continue

        header_row_idx = None
        q_col_idx = None
        resp_col_idx = None

        # Inspect first 10 rows for header
        for r_idx, row in enumerate(rows[:10]):
            for c_idx, cell in enumerate(row):
                val = str(cell.value or "").strip().lower()
                if any(tok in val for tok in QUESTION_COLUMN_TOKENS):
                    q_col_idx = c_idx
                    header_row_idx = r_idx
                if any(tok in val for tok in RESPONSE_COLUMN_TOKENS):
                    resp_col_idx = c_idx
            if q_col_idx is not None:
                if resp_col_idx is None:
                    resp_col_idx = q_col_idx + 1
                break

        if q_col_idx is None:
            q_col_idx = 0
            resp_col_idx = 1
            start_row = 0
        else:
            start_row = (header_row_idx or 0) + 1

        for r_idx in range(start_row, len(rows)):
            row = rows[r_idx]
            if q_col_idx < len(row):
                q_val = str(row[q_col_idx].value or "").strip()
                if q_val and len(q_val) > 3 and not set(q_val) <= {"-", "_", " ", "\t", "\n"}:
                    extracted.append(
                        ExtractedQuestionItem(
                            question_text=strip_question_prefix(q_val),
                            coordinates=DocxTableCoordinate(
                                target_type="table_cell",
                                table_idx=sheet_idx,
                                row_idx=r_idx,
                                target_col_idx=resp_col_idx,
                            ),
                        )
                    )

    return extracted, file_bytes


def extract_csv_questions(file_bytes: bytes) -> tuple[list[ExtractedQuestionItem], bytes]:
    """Extract questions and coordinates from a CSV file."""
    text = file_bytes.decode("utf-8-sig", errors="replace")
    reader = list(csv.reader(io.StringIO(text)))
    extracted: list[ExtractedQuestionItem] = []
    if not reader:
        return extracted, file_bytes

    header_row_idx = None
    q_col_idx = None
    resp_col_idx = None

    for r_idx, row in enumerate(reader[:10]):
        for c_idx, cell in enumerate(row):
            val = cell.strip().lower()
            if any(tok in val for tok in QUESTION_COLUMN_TOKENS):
                q_col_idx = c_idx
                header_row_idx = r_idx
            if any(tok in val for tok in RESPONSE_COLUMN_TOKENS):
                resp_col_idx = c_idx
        if q_col_idx is not None:
            if resp_col_idx is None:
                resp_col_idx = q_col_idx + 1
            break

    if q_col_idx is None:
        q_col_idx = 0
        resp_col_idx = 1
        start_row = 0
    else:
        start_row = (header_row_idx or 0) + 1

    for r_idx in range(start_row, len(reader)):
        row = reader[r_idx]
        if q_col_idx < len(row):
            q_val = row[q_col_idx].strip()
            if q_val and len(q_val) > 3 and not set(q_val) <= {"-", "_", " ", "\t", "\n"}:
                extracted.append(
                    ExtractedQuestionItem(
                        question_text=strip_question_prefix(q_val),
                        coordinates=DocxTableCoordinate(
                            target_type="table_cell",
                            table_idx=0,
                            row_idx=r_idx,
                            target_col_idx=resp_col_idx,
                        ),
                    )
                )

    return extracted, file_bytes


def write_xlsx_answers_in_place(original_bytes: bytes, answered_items: list[dict]) -> bytes:
    """Write verified answers into an Excel (.xlsx) file, preserving original styling."""
    import openpyxl
    try:
        wb = openpyxl.load_workbook(io.BytesIO(original_bytes))
    except Exception as exc:
        logger.warning("Could not load original xlsx: %s. Creating new.", exc)
        wb = None

    if wb is None:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Responses"
        ws.append(["#", "Question", "ComplianceAi Answer"])
        for idx, item in enumerate(answered_items, 1):
            ans = item.get("answer") or item.get("draft_answer") or ""
            ws.append([idx, item.get("question") or "", ans])
        buf = io.BytesIO()
        wb.save(buf)
        return buf.getvalue()

    sheet_names = wb.sheetnames
    for item in answered_items:
        coords = item.get("coordinates") or {}
        if isinstance(coords, str):
            try:
                coords = json.loads(coords)
            except Exception:
                coords = {}

        ans_text = item.get("answer") or item.get("draft_answer") or ""
        table_idx = coords.get("table_idx", 0)
        row_idx = coords.get("row_idx")
        target_col_idx = coords.get("target_col_idx")

        sheet = wb[sheet_names[table_idx]] if 0 <= table_idx < len(sheet_names) else wb.active

        if row_idx is not None and target_col_idx is not None:
            # 0-indexed in schema, 1-indexed in openpyxl
            cell = sheet.cell(row=int(row_idx) + 1, column=int(target_col_idx) + 1)
            cell.value = ans_text
        else:
            q_text = (item.get("question") or "").strip().lower()
            if q_text:
                for row in sheet.iter_rows():
                    for cell in row:
                        if cell.value and q_text in str(cell.value).lower():
                            sheet.cell(row=cell.row, column=cell.column + 1, value=ans_text)
                            break

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def write_csv_answers_in_place(original_bytes: bytes, answered_items: list[dict]) -> bytes:
    """Write verified answers into a CSV file."""
    try:
        text = original_bytes.decode("utf-8-sig", errors="replace")
        rows = list(csv.reader(io.StringIO(text)))
    except Exception as exc:
        logger.warning("Could not load original CSV: %s. Creating new.", exc)
        rows = []

    if not rows:
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["#", "Question", "Answer"])
        for idx, item in enumerate(answered_items, 1):
            ans = item.get("answer") or item.get("draft_answer") or ""
            writer.writerow([idx, item.get("question") or "", ans])
        return out.getvalue().encode("utf-8-sig")

    for item in answered_items:
        coords = item.get("coordinates") or {}
        if isinstance(coords, str):
            try:
                coords = json.loads(coords)
            except Exception:
                coords = {}

        ans_text = item.get("answer") or item.get("draft_answer") or ""
        row_idx = coords.get("row_idx")
        target_col_idx = coords.get("target_col_idx")

        if row_idx is not None and 0 <= int(row_idx) < len(rows):
            target_row = rows[int(row_idx)]
            c_idx = int(target_col_idx) if target_col_idx is not None else len(target_row)
            while len(target_row) <= c_idx:
                target_row.append("")
            target_row[c_idx] = ans_text
        else:
            q_text = (item.get("question") or "").strip().lower()
            if q_text:
                for r in rows:
                    for i_col, val in enumerate(r):
                        if q_text in val.lower():
                            if i_col + 1 < len(r):
                                r[i_col + 1] = ans_text
                            else:
                                r.append(ans_text)
                            break

    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerows(rows)
    return out.getvalue().encode("utf-8-sig")
