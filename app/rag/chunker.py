import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from app.rag.parser import ParsedSection

logger = logging.getLogger(__name__)

@dataclass
class PreparedChunk:
    content: str
    chunk_index: int
    heading_path: Optional[str]
    content_type: str  # 'text' | 'table'
    metadata: Dict[str, Any]

_ABBREVIATIONS = {
    "e.g.", "i.e.", "etc.", "vs.", "no.", "corp.", "inc.", "ltd.",
    "dr.", "mr.", "mrs.", "ms.", "jr.", "sr.", "u.s.", "u.k.",
}


def split_text_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences using punctuation boundaries, with a merge
    pass that avoids splitting immediately after common abbreviations
    (e.g., i.e., Corp., SOC 2 Type II.) — no new pip dependency.
    """
    text = re.sub(r'\.{3,}', '... ', text)
    raw_splits = re.split(r'(?<=[.!?])\s+', text.strip())
    merged: List[str] = []
    for part in raw_splits:
        if merged:
            prev_words = merged[-1].split()
            last_word = prev_words[-1].lower() if prev_words else ""
            last_word_raw = prev_words[-1] if prev_words else ""
            # Merge back if the previous fragment ends in a known abbreviation
            # or a short uppercase-letter-plus-period pattern (e.g. "II.", "A.")
            if last_word in _ABBREVIATIONS or re.match(r'^[A-Z]{1,3}\.$', last_word_raw):
                merged[-1] = merged[-1] + " " + part
                continue
        merged.append(part)
    return [s for s in merged if s.strip()]

def chunk_sections(sections: List[ParsedSection], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[PreparedChunk]:
    """
    Structure-aware recursive chunker.
    
    Rules:
    1. Text chunks are split by sentence boundaries, respecting section/heading boundaries.
    2. Table chunks are NEVER split, regardless of size.
    3. Overlap is applied between consecutive text chunks within the same section.
    """
    chunks = []
    current_chunk_idx = 0

    for section in sections:
        if section.content_type == 'table':
            chunks.append(PreparedChunk(
                content=section.content,
                chunk_index=current_chunk_idx,
                heading_path=section.heading_path,
                content_type='table',
                metadata={
                    'page_number': section.page_number,
                    **section.metadata
                }
            ))
            current_chunk_idx += 1
            
        elif section.content_type == 'text':
            sentences = split_text_into_sentences(section.content)
            
            current_chunk_text = ""
            
            for sentence in sentences:
                if len(current_chunk_text) + len(sentence) + 1 <= chunk_size:
                    if current_chunk_text:
                        current_chunk_text += " " + sentence
                    else:
                        current_chunk_text = sentence
                else:
                    if current_chunk_text:
                        chunks.append(PreparedChunk(
                            content=current_chunk_text.strip(),
                            chunk_index=current_chunk_idx,
                            heading_path=section.heading_path,
                            content_type='text',
                            metadata={
                                'page_number': section.page_number,
                                **section.metadata
                            }
                        ))
                        current_chunk_idx += 1
                        
                        # Apply overlap by taking the end of the previous chunk
                        overlap_text = ""
                        if chunk_overlap > 0:
                            overlap_text = current_chunk_text[-chunk_overlap:]
                            # Trim to first space to avoid partial words
                            space_idx = overlap_text.find(' ')
                            if space_idx != -1:
                                overlap_text = overlap_text[space_idx:].strip()
                                
                        current_chunk_text = overlap_text + (" " if overlap_text else "") + sentence
                    else:
                        # Sentence is larger than chunk_size, keep it atomic
                        chunks.append(PreparedChunk(
                            content=sentence.strip(),
                            chunk_index=current_chunk_idx,
                            heading_path=section.heading_path,
                            content_type='text',
                            metadata={
                                'page_number': section.page_number,
                                **section.metadata
                            }
                        ))
                        current_chunk_idx += 1
                        current_chunk_text = ""
                        
            # Remaining text
            if current_chunk_text.strip():
                chunks.append(PreparedChunk(
                    content=current_chunk_text.strip(),
                    chunk_index=current_chunk_idx,
                    heading_path=section.heading_path,
                    content_type='text',
                    metadata={
                        'page_number': section.page_number,
                        **section.metadata
                    }
                ))
                current_chunk_idx += 1
                
        else:
            # Fallback for any other content types
            chunks.append(PreparedChunk(
                content=section.content,
                chunk_index=current_chunk_idx,
                heading_path=section.heading_path,
                content_type=section.content_type,
                metadata={
                    'page_number': section.page_number,
                    **section.metadata
                }
            ))
            current_chunk_idx += 1

    return chunks
