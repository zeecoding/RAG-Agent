"""Pydantic schemas for Word (.docx) coordinate-based question extraction and in-place mutation."""

from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field


class DocxTableCoordinate(BaseModel):
    """Pinpoints an exact table cell where an answer should be written."""
    target_type: Literal["table_cell"] = "table_cell"
    table_idx: int = Field(..., description="0-indexed position of the table within the document.")
    row_idx: int = Field(..., description="0-indexed row position within the table.")
    target_col_idx: int = Field(..., description="0-indexed column position of the response cell.")


class DocxParagraphCoordinate(BaseModel):
    """Pinpoints an exact paragraph where an answer should be written (fallback format)."""
    target_type: Literal["paragraph"] = "paragraph"
    paragraph_idx: int = Field(..., description="0-indexed position of the paragraph within the document.")
    insert_after: bool = Field(
        default=False,
        description="Whether to inject a new sibling paragraph beneath the prompt (True) or overwrite the placeholder in place (False).",
    )


DocxCoordinate = Annotated[
    Union[DocxTableCoordinate, DocxParagraphCoordinate],
    Field(discriminator="target_type"),
]


class ExtractedQuestionItem(BaseModel):
    """Question item extracted from a DOCX template along with its exact write-back coordinates."""
    question_text: str = Field(..., description="Sanitized question or requirement text.")
    coordinates: DocxCoordinate = Field(..., description="Exact coordinate mapping for writing the response.")


class AnsweredItem(BaseModel):
    """An approved or drafted answer to be written back into the template."""
    coordinates: DocxCoordinate = Field(..., description="Target coordinate in the document.")
    answer: str = Field(..., description="The answer text to inject.")
