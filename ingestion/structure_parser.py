"""
ingestion/structure_parser.py
─────────────────────────────
Parses the PDF's Table of Contents / section hierarchy using PyMuPDF's built-in
outline-tree reader and a regex fallback for heading detection.

Produces a list of SectionNode objects that describe the document's hierarchy:
    Part → Chapter → Section → Subsection (up to 4 levels deep)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Regex patterns for section headings found in the NASA handbook
# Matches: "6.3.2.1  Some Title", "6.3  Section Name", "Appendix A  ..."
_SECTION_RE = re.compile(
    r"^(\d+(?:\.\d+){0,3}|Appendix\s+[A-Z])\s{1,6}([A-Z][\w\s/,&:()'-]{3,80})$",
    re.MULTILINE,
)


@dataclass
class SectionNode:
    """Represents one node in the document hierarchy."""
    section_id: str          # e.g. "6.3.2.1" or "Appendix B"
    title: str
    level: int               # 1 = chapter, 2 = section, 3 = subsection, 4 = sub-sub
    page_start: int          # 0-indexed page number in the PDF
    page_end: Optional[int]  # filled in after parsing all sections
    parent_id: Optional[str] = None
    children: list[str] = field(default_factory=list)  # child section_ids

    # Populated later by text_extractor
    raw_text: str = ""

    def depth_label(self) -> str:
        """User-friendly level label."""
        return {1: "chapter", 2: "section", 3: "subsection", 4: "paragraph"}.get(
            self.level, "section"
        )


def _level_from_id(section_id: str) -> int:
    """Infer hierarchy level from the section numbering (e.g. '6.3.2.1' → 4)."""
    if section_id.lower().startswith("appendix"):
        return 1
    return len(section_id.split("."))


def parse_structure(pdf_path: Path) -> list[SectionNode]:
    """
    Primary: use PyMuPDF's built-in outline (TOC).
    Fallback: scan page text with regex for section headings.

    Returns a flat list of SectionNode (sorted by page_start).
    Sets page_end for each node after the list is built.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))

    nodes: list[SectionNode] = []

    # ── 1. Try embedded TOC ──────────────────────────────────────────────────
    toc = doc.get_toc(simple=False)  # [[level, title, page, xref_dict], ...]
    if toc:
        logger.info(f"Using embedded TOC — {len(toc)} entries found.")
        for entry in toc:
            lvl, title, page = entry[0], entry[1], entry[2] - 1  # convert to 0-indexed
            # Build a section_id from the level + title prefix heuristic
            # For NASA handbook the TOC has numbered entries like "6.3.2"
            num_match = re.match(r"^(\d+(?:\.\d+){0,3}|Appendix\s*[A-Z])", title.strip())
            section_id = num_match.group(1).strip() if num_match else f"L{lvl}_{len(nodes)}"
            clean_title = title.strip()

            nodes.append(
                SectionNode(
                    section_id=section_id,
                    title=clean_title,
                    level=min(lvl, 4),
                    page_start=max(0, page),
                    page_end=None,
                )
            )
    else:
        # ── 2. Regex fallback ────────────────────────────────────────────────
        logger.warning("No embedded TOC found. Falling back to regex heading detection.")
        for pg_idx, page in enumerate(doc):
            text = page.get_text("text")
            for m in _SECTION_RE.finditer(text):
                section_id = m.group(1).strip()
                title_text = m.group(2).strip()
                lvl = _level_from_id(section_id)
                nodes.append(
                    SectionNode(
                        section_id=section_id,
                        title=f"{section_id} {title_text}",
                        level=min(lvl, 4),
                        page_start=pg_idx,
                        page_end=None,
                    )
                )

    # ── 3. Set page_end for each node ────────────────────────────────────────
    for i, node in enumerate(nodes):
        if i + 1 < len(nodes):
            node.page_end = nodes[i + 1].page_start - 1
        else:
            node.page_end = len(doc) - 1

    # ── 4. Wire parent–child relationships ───────────────────────────────────
    id_map: dict[str, SectionNode] = {n.section_id: n for n in nodes}
    for node in nodes:
        parts = node.section_id.split(".")
        if len(parts) > 1:
            parent_id = ".".join(parts[:-1])
            if parent_id in id_map:
                node.parent_id = parent_id
                id_map[parent_id].children.append(node.section_id)

    doc.close()
    logger.info(
        f"Structure parsed: {len(nodes)} nodes, "
        f"depth up to {max((n.level for n in nodes), default=1)}"
    )
    return nodes


def section_map(nodes: list[SectionNode]) -> dict[str, SectionNode]:
    """Return {section_id: SectionNode} for quick lookup."""
    return {n.section_id: n for n in nodes}
