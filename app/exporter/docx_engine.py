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

# Token sets for identifying question and response columns in questionnaire tables
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

# Text patterns for non-table prompt-style document fallback
PROMPT_PATTERNS = [
    re.compile(r"^(?:Q\d+[:.]|Requirement\s+\d+[:.]|\d+[\.\)]\s+)", re.IGNORECASE),
]

# Tokens that indicate a blank answer slot / placeholder line in paragraph-style docs
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


def _is_answer_placeholder(text: str) -> bool:
    """Returns True if `text` looks like a blank answer-slot placeholder rather than real content.

    Used by the paragraph fallback extractor to distinguish a genuine answer slot
    (e.g. "[Enter Vendor Response Here]" or "[Vendor Response: ]") from a neighbouring question or body text,
    so the write-back coordinate targets the placeholder line — not the question itself.
    """
    lower = text.lower().strip()
    if any(tok in lower for tok in _PLACEHOLDER_TOKENS):
        return True
    if lower.startswith("[") and any(w in lower for w in ("response", "answer", "vendor", "comment")):
        return True
    return False


def _insert_blank_paragraph_after(paragraph) -> None:
    """Splices an empty ``w:p`` element directly after *paragraph* in the document body.

    Uses lxml's ``addnext()`` for O(1) in-place XML insertion.  The calling loop
    tracks a cumulative ``insertion_offset`` so that original paragraph indices
    remain stable across multiple insertions within the same scan pass.

    The inserted node carries no runs, style, or properties — it is a genuine
    blank line that ``write_docx_answers_in_place`` will write the AI answer into.
    """
    new_p = OxmlElement("w:p")
    paragraph._p.addnext(new_p)


# Ignored banner tokens in fallback paragraph extraction
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

# Executive typography standard
DEFAULT_FONT_NAME = "Calibri"
DEFAULT_FONT_SIZE_PT = 9.5
DEFAULT_COLOR_RGB = (0x0F, 0x17, 0x2A)  # Slate-900


def _find_table_header_columns(table) -> tuple[int, int, int] | None:
    """Inspects candidate header rows in a table to locate question and response columns.

    Scans the first up to 3 rows (in case row 0 is a title or merged banner).
    Returns (header_row_idx, question_col_idx, response_col_idx) if found, else None.
    """
    max_scan_rows = min(3, len(table.rows))

    for r_idx in range(max_scan_rows):
        row = table.rows[r_idx]
        cells = row.cells
        if len(cells) < 2:
            continue

        q_col_candidate: int | None = None
        resp_col_candidate: int | None = None

        # Prioritize exact/strong matches first
        for c_idx, cell in enumerate(cells):
            cell_text = cell.text.strip().lower()
            if not cell_text:
                continue

            # Check response candidates — keep first match only (first-match-wins)
            if any(t in cell_text for t in RESPONSE_COLUMN_TOKENS):
                if resp_col_candidate is None:
                    resp_col_candidate = c_idx

            # Check question candidates — keep first match only (first-match-wins)
            elif any(t in cell_text for t in QUESTION_COLUMN_TOKENS):
                if q_col_candidate is None:
                    q_col_candidate = c_idx

        # If we have both and they point to distinct columns, we found our header
        if (
            q_col_candidate is not None
            and resp_col_candidate is not None
            and q_col_candidate != resp_col_candidate
        ):
            return r_idx, q_col_candidate, resp_col_candidate

    return None


def extract_docx_questions(file_bytes: bytes) -> tuple[list[ExtractedQuestionItem], bytes]:
    """Extracts questions and their exact coordinate write-back positions from a .docx file.

    Returns a tuple of:
      - ``items``:  list of :class:`ExtractedQuestionItem` with coordinates.
      - ``template_bytes``:  the (possibly mutated) document bytes to use as the
        template for :func:`write_docx_answers_in_place`.

        For **table-based** documents, ``template_bytes`` is identical to the
        input ``file_bytes`` — no paragraphs are modified.

        For **paragraph-fallback** documents, the two-phase injector may splice
        blank answer-slot ``w:p`` elements into the in-memory XML tree.
        ``template_bytes`` reflects those insertions.  Callers **must** pass
        ``template_bytes`` (not the original ``file_bytes``) to
        :func:`write_docx_answers_in_place`; otherwise the injected coordinates
        will be out-of-bounds on a fresh 3-paragraph document.

    1. Inspects all tables in the document.
       Detects header rows matching Question and Response column tokens.
       Extracts questions row-by-row and maps exact (table_idx, row_idx, target_col_idx).
    2. If no table questions are found, falls back to paragraph-style extraction
       for prompt-style documents ('Q1.', 'Requirement 1:', or trailing '?'),
       while ignoring document preamble, instructions, and confidentiality notices.
    """
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
            # Bounds check in case of irregular rows
            if q_col_idx >= len(row.cells) or resp_col_idx >= len(row.cells):
                continue

            q_cell = row.cells[q_col_idx]
            resp_cell = row.cells[resp_col_idx]

            # Detect merged section divider rows (where question and response cell point to the same XML element)
            if q_cell._tc == resp_cell._tc:
                continue

            q_text = q_cell.text.strip()
            if not q_text:
                continue

            # Ignore empty spacer rows or rows with only dashes/whitespace
            if set(q_text) <= {"-", "_", " ", "\t", "\n"}:
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

    # 2. Extract paragraph-level prompts (outside of tables)
    # Collect paragraph XML elements already inside table cells to prevent duplication
    table_p_elements = {
        p._p
        for table in doc.tables
        for row in table.rows
        for cell in row.cells
        for p in cell.paragraphs
    }

    paragraphs = doc.paragraphs
    for p_idx, p in enumerate(paragraphs):
        # Skip paragraphs that belong to tables
        if p._p in table_p_elements:
            continue

        p_text = p.text.strip()
        if not p_text:
            continue

        p_lower = p_text.lower()
        if any(banner in p_lower for banner in IGNORED_BANNER_TOKENS):
            continue

        # Check prompt pattern or interrogative sentence
        is_prompt = any(pattern.search(p_text) for pattern in PROMPT_PATTERNS)
        is_question = p_text.endswith("?") and len(p_text) >= 10

        if is_prompt or is_question:
            # Check if immediately followed by an answer placeholder
            has_placeholder = False
            if p_idx + 1 < len(paragraphs):
                next_text = paragraphs[p_idx + 1].text.strip()
                if _is_answer_placeholder(next_text):
                    has_placeholder = True

            if has_placeholder:
                coord = DocxParagraphCoordinate(
                    target_type="paragraph",
                    paragraph_idx=p_idx + 1,
                    insert_after=False,
                )
            else:
                coord = DocxParagraphCoordinate(
                    target_type="paragraph",
                    paragraph_idx=p_idx,
                    insert_after=True,
                )

            extracted.append(
                ExtractedQuestionItem(
                    question_text=p_text,
                    coordinates=coord,
                )
            )

    # Serialise the (possibly mutated) document back to bytes.
    _buf = io.BytesIO()
    doc.save(_buf)
    template_bytes = _buf.getvalue()

    return extracted, template_bytes




# Regex to detect bold markdown spans: **text**
BOLD_MD_REGEX = re.compile(r"\*\*(.*?)\*\*")


def _clean_markdown_text(text: str) -> str:
    """Cleans up residual Markdown list tokens, stray dashes, and spacing issues."""
    # Remove leading markdown bullet dashes or list hyphens at line beginnings
    cleaned = re.sub(r"(?m)^\s*[-*•]\s+", "", text)
    # Normalize double spaces or markdown linebreaks
    cleaned = cleaned.replace("  \n", "\n").replace("\r\n", "\n")
    return cleaned


def _write_paragraph_runs(
    paragraph,
    text: str,
    font_name: str = DEFAULT_FONT_NAME,
    font_size_pt: float = DEFAULT_FONT_SIZE_PT,
    color_rgb: tuple[int, int, int] = DEFAULT_COLOR_RGB,
    convert_bold_markdown: bool = True,
) -> None:
    """Writes text into paragraph runs without deleting the paragraph XML node (w:p).
    If convert_bold_markdown is True, parses **word** into native Word bold runs,
    preventing raw asterisks from appearing in the exported document.
    """
    # 1. Clear existing runs
    for r in paragraph.runs:
        r.text = ""

    # 2. Clean bullet hyphens and stray markers
    sanitized_text = _clean_markdown_text(text)

    # 3. If no markdown asterisks present, write as a single clean run
    if not convert_bold_markdown or "**" not in sanitized_text:
        # Strip any stray single asterisks
        clean_plain = sanitized_text.replace("*", "")
        r = paragraph.runs[0] if paragraph.runs else paragraph.add_run()
        r.text = clean_plain
        r.font.name = font_name
        r.font.size = Pt(font_size_pt)
        r.font.color.rgb = RGBColor(*color_rgb)
        r.font.bold = False
        r.font.italic = False
        return

    # 4. If markdown bold spans exist, split and convert to native Word formatting
    # E.g. "Text before **Bold Subhead** Text after"
    parts = BOLD_MD_REGEX.split(sanitized_text)
    # Regex split alternates: [normal_text, bold_text, normal_text, bold_text, ...]

    for idx, part in enumerate(parts):
        if not part:
            continue

        # Odd indices are the captured bold groups
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
    """Surgically writes text into a table cell at the run level, preserving w:tcPr and cell properties."""
    paragraphs = cell.paragraphs
    if not paragraphs:
        p = cell.add_paragraph()
    else:
        p = paragraphs[0]
        # Clear extra paragraphs in the cell while preserving node structure
        for extra_p in paragraphs[1:]:
            for r in extra_p.runs:
                r.text = ""

    _write_paragraph_runs(p, text, font_name, font_size_pt, color_rgb)


def _write_paragraph_in_place(
    paragraph,
    text: str,
    font_name: str = DEFAULT_FONT_NAME,
    font_size_pt: float = DEFAULT_FONT_SIZE_PT,
    color_rgb: tuple[int, int, int] = DEFAULT_COLOR_RGB,
) -> None:
    """Surgically writes text into a paragraph at the run level, preserving w:pPr."""
    _write_paragraph_runs(paragraph, text, font_name, font_size_pt, color_rgb)


def write_docx_answers_in_place(
    original_bytes: bytes,
    answered_items: list[dict | AnsweredItem],
    font_name: str = DEFAULT_FONT_NAME,
    font_size_pt: float = DEFAULT_FONT_SIZE_PT,
    color_rgb: tuple[int, int, int] = DEFAULT_COLOR_RGB,
) -> bytes:
    """Performs non-destructive in-place writes into designated coordinates in a .docx document.

    Preserves 100% of headers, footers, logos, cell backgrounds, borders, and margins.
    Only mutates text inside the targeted coordinate runs.

    Args:
        original_bytes: Raw bytes of the original .docx template.
        answered_items: Sequence of answered items (Pydantic models or plain dicts).
        font_name: Font family for injected answer text. Defaults to Calibri.
        font_size_pt: Font size in points for injected answer text. Defaults to 9.5pt.
        color_rgb: RGB colour tuple for injected answer text. Defaults to slate-900 (#0F172A).
    """
    if not original_bytes:
        return b""

    doc = docx.Document(io.BytesIO(original_bytes))
    orig_paragraphs = list(doc.paragraphs)

    for item in answered_items:
        # Extract coordinates and answer text from dict or Pydantic model
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
                        logger.warning("target_col_idx %d out of bounds for table %d, row %d", target_col_idx, table_idx, row_idx)
                else:
                    logger.warning("row_idx %d out of bounds for table %d", row_idx, table_idx)
            else:
                logger.warning("table_idx %d out of bounds (doc has %d tables)", table_idx, len(doc.tables))

        elif target_type == "paragraph":
            paragraph_idx = coords.get("paragraph_idx") if isinstance(coords, dict) else coords.paragraph_idx
            insert_after = coords.get("insert_after", False) if isinstance(coords, dict) else getattr(coords, "insert_after", False)

            if 0 <= paragraph_idx < len(orig_paragraphs):
                if insert_after:
                    # No placeholder existed in the template; inject a new sibling paragraph beneath the prompt
                    ref_p = orig_paragraphs[paragraph_idx]
                    new_p = doc.add_paragraph()
                    ref_p._p.addnext(new_p._p)
                    _write_paragraph_runs(new_p, ans_text, font_name, font_size_pt, color_rgb)
                else:
                    # A placeholder exists at this coordinate; overwrite and replace its text in place
                    target_p = orig_paragraphs[paragraph_idx]
                    _write_paragraph_runs(target_p, ans_text, font_name, font_size_pt, color_rgb)
            else:
                logger.warning("paragraph_idx %d out of bounds (doc has %d paragraphs)", paragraph_idx, len(orig_paragraphs))

    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()
