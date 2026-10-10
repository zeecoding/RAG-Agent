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
)

PROMPT_PATTERNS = [
    re.compile(r"^(?:Q\d+[:.]|Requirement\s+\d+[:.]|\d+[\.\)]\s+)", re.IGNORECASE),
]

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

IGNORED_BANNER_TOKENS = (
    "instructions:",
    "confidentiality notice",
    "confidential",
    "all rights reserved",
    "proprietary",
    "table of contents",
    "scope:",
    "overview:",
    "guidance:",
    "disclaimer:",
)

DEFAULT_FONT_NAME = "Calibri"
DEFAULT_FONT_SIZE_PT = 9.5
DEFAULT_COLOR_RGB = (0x0F, 0x17, 0x2A)  # Slate-900

BOLD_MD_REGEX = re.compile(r"\*\*(.*?)\*\*")


def _is_answer_placeholder(text: str) -> bool:
    """Returns True if text is a standalone placeholder line."""
    lower = text.lower().strip()
    if any(tok in lower for tok in _PLACEHOLDER_TOKENS):
        return True
    if lower.startswith("[") and any(w in lower for w in ("response", "answer", "vendor", "comment")):
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
    """Returns True if text is an instructional guidance block."""
    lower = text.lower().strip()
    return any(lower.startswith(pfx) for pfx in GUIDANCE_PREFIXES)


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

    return None


def extract_docx_questions(file_bytes: bytes) -> tuple[list[ExtractedQuestionItem], bytes]:
    """Extracts questions and precise coordinate positions from a .docx file."""
    if not file_bytes:
        return [], file_bytes

    doc = docx.Document(io.BytesIO(file_bytes))
    extracted: list[ExtractedQuestionItem] = []

    # 1. Process all tables
    for t_idx, table in enumerate(doc.tables):
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
                    question_text=q_text,
                    coordinates=DocxTableCoordinate(
                        target_type="table_cell",
                        table_idx=t_idx,
                        row_idx=r_idx,
                        target_col_idx=resp_col_idx,
                    ),
                )
            )

    # 2. Extract paragraph prompts (outside of tables)
    table_p_elements = {
        p._p
        for table in doc.tables
        for row in table.rows
        for cell in row.cells
        for p in cell.paragraphs
    }

    paragraphs = doc.paragraphs
    for p_idx, p in enumerate(paragraphs):
        if p._p in table_p_elements:
            continue

        p_text = p.text.strip()
        if not p_text:
            continue

        p_lower = p_text.lower()
        if any(banner in p_lower for banner in IGNORED_BANNER_TOKENS):
            continue

        is_prompt = any(pattern.search(p_text) for pattern in PROMPT_PATTERNS)
        is_question = p_text.endswith("?") and len(p_text) >= 10

        if is_prompt or is_question:
            scan_idx = p_idx + 1
            while scan_idx < len(paragraphs) and _is_guidance_paragraph(paragraphs[scan_idx].text):
                g_text = paragraphs[scan_idx].text
                if _has_trailing_placeholder(g_text):
                    cleaned_g = _strip_trailing_placeholder(g_text)
                    _clear_paragraph_runs_xml(paragraphs[scan_idx])
                    paragraphs[scan_idx].add_run(cleaned_g)
                scan_idx += 1

            has_placeholder = False
            if scan_idx < len(paragraphs):
                cand_text = paragraphs[scan_idx].text.strip()
                if _is_answer_placeholder(cand_text):
                    has_placeholder = True

            if has_placeholder:
                coord = DocxParagraphCoordinate(
                    target_type="paragraph",
                    paragraph_idx=scan_idx,
                    insert_after=False,
                )
            else:
                target_anchor = scan_idx - 1 if scan_idx > p_idx else p_idx
                coord = DocxParagraphCoordinate(
                    target_type="paragraph",
                    paragraph_idx=target_anchor,
                    insert_after=True,
                )

            extracted.append(
                ExtractedQuestionItem(
                    question_text=p_text,
                    coordinates=coord,
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
        elif isinstance(item, dict):
            coords = item.get("coordinates") or item
            ans_text = item.get("answer") or item.get("draft_answer") or item.get("answer_text", "")
        else:
            coords = getattr(item, "coordinates", item)
            ans_text = getattr(item, "answer", getattr(item, "draft_answer", ""))

        if not coords:
            continue

        target_type = coords.get("target_type") if isinstance(coords, dict) else getattr(coords, "target_type", None)

        if target_type == "table_cell":
            table_idx = coords["table_idx"] if isinstance(coords, dict) else coords.table_idx
            row_idx = coords["row_idx"] if isinstance(coords, dict) else coords.row_idx
            target_col_idx = coords["target_col_idx"] if isinstance(coords, dict) else coords.target_col_idx

            if 0 <= table_idx < len(doc.tables):
                table = doc.tables[table_idx]
                if 0 <= row_idx < len(table.rows):
                    row = table.rows[row_idx]
                    if 0 <= target_col_idx < len(row.cells):
                        cell = row.cells[target_col_idx]
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

            if 0 <= paragraph_idx < len(orig_paragraphs):
                if insert_after:
                    ref_p = orig_paragraphs[paragraph_idx]
                    new_p = doc.add_paragraph()
                    ref_p._p.addnext(new_p._p)
                    _write_paragraph_runs(new_p, ans_text, font_name, font_size_pt, color_rgb)
                else:
                    target_p = orig_paragraphs[paragraph_idx]
                    _write_paragraph_runs(target_p, ans_text, font_name, font_size_pt, color_rgb)
            else:
                logger.warning("paragraph_idx %d out of bounds", paragraph_idx)

    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()