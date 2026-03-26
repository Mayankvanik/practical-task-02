"""
ingestion/table_extractor.py
─────────────────────────────
Extracts tables from all PDF pages using pdfplumber.

Multi-page table merging:
- Detects consecutive pages with tables of the same column count.
- Propagates headers from the first page to continuation pages.
- Converts each table to Markdown + stores as JSON payload.

Produces TableChunk objects for Qdrant storage.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber

logger = logging.getLogger(__name__)


@dataclass
class TableChunk:
    """One table (possibly merged from multiple pages) stored as a chunk."""
    chunk_id: str
    chunk_type: str = "table"
    text: str = ""          # Markdown representation of the table
    table_json: str = ""    # JSON string of the raw cell data
    caption: str = ""
    page_numbers: list[int] = field(default_factory=list)
    section_id: str = ""
    section_title: str = ""
    chapter_num: Optional[int] = None
    chapter_title: str = ""
    heading_hierarchy: list[str] = field(default_factory=list)
    cross_references: list[str] = field(default_factory=list)
    acronyms_used: list[str] = field(default_factory=list)


def _rows_to_markdown(rows: list[list]) -> str:
    """Convert a list-of-rows (lists) to a Markdown table string."""
    if not rows:
        return ""
    # Normalise cells: replace None with empty string
    clean = [[str(c or "").replace("\n", " ").strip() for c in row] for row in rows]
    if not clean:
        return ""
    header = clean[0]
    sep = ["---"] * len(header)
    body = clean[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in body:
        # Pad row if fewer columns than header
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(padded[: len(header)]) + " |")
    return "\n".join(lines)


def _chapter_from_section(section_id: str) -> Optional[int]:
    try:
        return int(section_id.split(".")[0])
    except (ValueError, IndexError):
        return None


def extract_tables(
    pdf_path: Path,
    tables_dir: Path,
    section_nodes=None,
) -> list[TableChunk]:
    """
    Extract all tables from the PDF.
    Attempts to merge multi-page tables.
    Returns a list of TableChunk objects.
    """
    pdf_path = Path(pdf_path)
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Build page→section lookup
    page_to_section: dict[int, str] = {}
    if section_nodes:
        for node in section_nodes:
            for pg in range(node.page_start, (node.page_end or node.page_start) + 1):
                if pg not in page_to_section:
                    page_to_section[pg] = node.section_id

    table_chunks: list[TableChunk] = []
    table_counter = 0

    # Holder for multi-page merge candidates
    pending_rows: list[list] = []
    pending_pages: list[int] = []
    pending_ncols: int = 0
    pending_section: str = ""

    def _flush_pending():
        nonlocal pending_rows, pending_pages, pending_ncols, pending_section, table_counter
        if not pending_rows:
            return
        md = _rows_to_markdown(pending_rows)
        section_id = pending_section
        caption = f"Table {table_counter + 1} (pages {pending_pages[0]}–{pending_pages[-1]})"
        chunk = TableChunk(
            chunk_id=f"table_{table_counter:04d}",
            text=md,
            table_json=json.dumps(pending_rows, ensure_ascii=False),
            caption=caption,
            page_numbers=list(pending_pages),
            section_id=section_id,
            chapter_num=_chapter_from_section(section_id),
        )
        table_chunks.append(chunk)
        # Save markdown to file
        md_path = tables_dir / f"table_{table_counter:04d}.md"
        md_path.write_text(f"# {caption}\n\n{md}\n", encoding="utf-8")

        table_counter += 1
        pending_rows = []
        pending_pages = []
        pending_ncols = 0
        pending_section = ""

    with pdfplumber.open(str(pdf_path)) as pdf:
        prev_had_table = False

        for pg_idx, page in enumerate(pdf.pages):
            page_num = pg_idx + 1
            section_id = page_to_section.get(pg_idx, "")

            try:
                tables = page.extract_tables()
            except Exception as e:
                logger.debug(f"pdfplumber error on page {page_num}: {e}")
                tables = []

            if not tables:
                if prev_had_table:
                    _flush_pending()
                prev_had_table = False
                continue

            for raw_table in tables:
                if not raw_table or len(raw_table) < 2:
                    continue
                ncols = len(raw_table[0]) if raw_table else 0

                # Multi-page merge: same column count as pending
                if prev_had_table and pending_ncols == ncols:
                    # Skip header row on continuation (assume first row = header)
                    pending_rows.extend(raw_table[1:])
                    if page_num not in pending_pages:
                        pending_pages.append(page_num)
                else:
                    # Start new table
                    _flush_pending()
                    pending_rows = list(raw_table)
                    pending_pages = [page_num]
                    pending_ncols = ncols
                    pending_section = section_id

            prev_had_table = bool(tables)

        # Flush any remaining pending table
        _flush_pending()

    logger.info(
        f"Table extraction complete: {len(table_chunks)} tables "
        f"saved to {tables_dir}"
    )
    return table_chunks
