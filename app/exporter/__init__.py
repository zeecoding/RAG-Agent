"""Word (.docx) format-preserving questionnaire extraction and in-place mutation package."""

from app.exporter.schemas import (
    AnsweredItem,
    DocxCoordinate,
    DocxParagraphCoordinate,
    DocxTableCoordinate,
    ExtractedQuestionItem,
)

__all__ = [
    "DocxTableCoordinate",
    "DocxParagraphCoordinate",
    "DocxCoordinate",
    "ExtractedQuestionItem",
    "AnsweredItem",
]
