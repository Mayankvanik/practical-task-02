"""
ingestion/text_extractor.py
────────────────────────────
Extracts text from each section of the PDF using PyMuPDF, then splits it
into overlapping chunks that respect section boundaries.

Attaches section metadata to every chunk for rich Qdrant payload.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from ingestion.structure_parser import SectionNode

logger = logging.getLogger(__name__)

# Simple word-count approximation of tokens (1 token ≈ 0.75 words for English)
_WORDS_PER_TOKEN = 0.75


@dataclass
class TextChunk:
    """One text chunk from the document, ready for embedding + storage."""
    chunk_id: str
    chunk_type: str  # "text"
    text: str
    chapter_num: Optional[int]
    chapter_title: str
    section_id: str
    section_title: str
    parent_section_id: Optional[str]
    heading_hierarchy: list[str]
    page_numbers: list[int]
    cross_references: list[str] = field(default_factory=list)
    acronyms_used: list[str] = field(default_factory=list)


# ── Cross-reference detection ─────────────────────────────────────────────────
_XREF_RE = re.compile(
    r"\bsee\s+(?:Section|Chapter|Figure|Table|Appendix|step)\s+"
    r"([\d.]+|[A-Z](?:\.\d+)*)",
    re.IGNORECASE,
)
_ACRONYM_RE = re.compile(r"\b([A-Z]{2,8})\b")


def _extract_cross_references(text: str) -> list[str]:
    """Find all cross-references in the text (e.g. 'see Section 4.2')."""
    return list(dict.fromkeys(m.group(1) for m in _XREF_RE.finditer(text)))


def _extract_acronyms(text: str, known_acronyms: set[str]) -> list[str]:
    """Return known acronyms found in the text."""
    found = {m.group(1) for m in _ACRONYM_RE.finditer(text)}
    return sorted(found & known_acronyms)


def _words(text: str) -> int:
    return len(text.split())


def _split_into_chunks(
    text: str,
    max_tokens: int = 600,
    overlap_tokens: int = 80,
) -> list[str]:
    """
    Split text into chunks by sentence boundaries, respecting max_tokens.
    Adds an overlap window of ~overlap_tokens between consecutive chunks.
    """
    # Split on sentence endings while keeping the delimiter
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current_words: list[str] = []
    current_count = 0
    max_words = int(max_tokens / _WORDS_PER_TOKEN)
    overlap_words = int(overlap_tokens / _WORDS_PER_TOKEN)

    for sent in sentences:
        sent_words = sent.split()
        if current_count + len(sent_words) > max_words:
            if current_words:
                chunks.append(" ".join(current_words))
                # Keep tail for overlap
                current_words = current_words[-overlap_words:]
                current_count = len(current_words)
        current_words.extend(sent_words)
        current_count += len(sent_words)

    if current_words:
        chunks.append(" ".join(current_words))

    return [c for c in chunks if c.strip()]


def extract_section_text(
    doc: fitz.Document,
    node: SectionNode,
    max_tokens: int = 600,
    overlap_tokens: int = 80,
    known_acronyms: set[str] | None = None,
    all_nodes: list[SectionNode] | None = None,
) -> list[TextChunk]:
    """
    Extract and chunk the text for one SectionNode.
    Returns a list of TextChunk (one or more per section based on length).
    """
    known_acronyms = known_acronyms or set()
    page_start = max(0, node.page_start)
    page_end = min(node.page_end or page_start, len(doc) - 1)

    # ── Collect raw text across pages ─────────────────────────────────────────
    full_text_parts: list[str] = []
    page_nums: list[int] = []
    for pg in range(page_start, page_end + 1):
        page_text = doc[pg].get_text("text")
        if page_text.strip():
            full_text_parts.append(page_text)
            page_nums.append(pg + 1)  # store 1-indexed page number

    raw_text = "\n".join(full_text_parts)
    node.raw_text = raw_text  # cache for cross-reference linking

    if not raw_text.strip():
        return []

    # ── Determine hierarchy labels ────────────────────────────────────────────
    parts = node.section_id.split(".")
    chapter_num: Optional[int] = None
    try:
        chapter_num = int(parts[0]) if parts[0].isdigit() else None
    except ValueError:
        pass

    heading_hierarchy = _build_heading_hierarchy(node, all_nodes or [])

    # ── Split into chunks ─────────────────────────────────────────────────────
    raw_chunks = _split_into_chunks(raw_text, max_tokens, overlap_tokens)
    result: list[TextChunk] = []
    for idx, chunk_text in enumerate(raw_chunks):
        chunk = TextChunk(
            chunk_id=f"{node.section_id}_t{idx}",
            chunk_type="text",
            text=chunk_text,
            chapter_num=chapter_num,
            chapter_title=_find_chapter_title(node, all_nodes or []),
            section_id=node.section_id,
            section_title=node.title,
            parent_section_id=node.parent_id,
            heading_hierarchy=heading_hierarchy,
            page_numbers=page_nums,
            cross_references=_extract_cross_references(chunk_text),
            acronyms_used=_extract_acronyms(chunk_text, known_acronyms),
        )
        result.append(chunk)

    return result


def _build_heading_hierarchy(
    node: SectionNode, all_nodes: list[SectionNode]
) -> list[str]:
    """Build the heading breadcrumb trail for a node."""
    node_map = {n.section_id: n for n in all_nodes}
    trail: list[str] = []
    current_id = node.section_id
    while current_id:
        n = node_map.get(current_id)
        if not n:
            break
        trail.insert(0, n.title)
        current_id = n.parent_id or ""
    return trail


def _find_chapter_title(node: SectionNode, all_nodes: list[SectionNode]) -> str:
    """Walk up to find the L1 chapter title."""
    node_map = {n.section_id: n for n in all_nodes}
    current_id = node.section_id
    while current_id:
        n = node_map.get(current_id)
        if not n:
            break
        if n.level == 1:
            return n.title
        current_id = n.parent_id or ""
    return node.title


def extract_all_text_chunks(
    pdf_path: Path,
    nodes: list[SectionNode],
    max_tokens: int = 600,
    overlap_tokens: int = 80,
    known_acronyms: set[str] | None = None,
) -> list[TextChunk]:
    """
    Iterate all section nodes and extract text chunks.
    Returns a flat list of all TextChunk objects.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    all_chunks: list[TextChunk] = []

    known_acronyms = known_acronyms or set()

    for node in nodes:
        try:
            chunks = extract_section_text(
                doc,
                node,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                known_acronyms=known_acronyms,
                all_nodes=nodes,
            )
            all_chunks.extend(chunks)
        except Exception as e:
            logger.warning(f"Failed to extract text for {node.section_id}: {e}")

    doc.close()
    logger.info(f"Text extraction complete: {len(all_chunks)} text chunks from {len(nodes)} sections.")
    return all_chunks
