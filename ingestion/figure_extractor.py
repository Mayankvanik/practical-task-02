"""
ingestion/figure_extractor.py
──────────────────────────────
Extracts figures, charts, and diagrams from a PDF using PyMuPDF.

Strategy:
1.  Find figure captions via regex ("Figure 6-3 …", "Fig. 4 …").
2.  For each caption, render the surrounding page region as a high-res PNG.
3.  Also extract embedded raster images that are large enough to be diagrams.
4.  Save each figure as a PNG to FIGURES_DIR.
5.  Return FigureChunk objects carrying figure metadata for Qdrant storage.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Matches captions like:
#   "Figure 2-3. Title of figure"
#   "Figure 6.7:  Some title"
#   "Fig. 3 Title"
#   "FIGURE 4-2 — Title"
_CAPTION_RE = re.compile(
    r"(Fig(?:ure)?\.?\s*\d+[-.]?\d*\s*[.:\-—]?\s*[^\n]{3,120})",
    re.IGNORECASE,
)

# Min area (px²) for an embedded image to be considered a diagram (not a logo)
_MIN_IMAGE_AREA = 40_000  # ~200×200 px


@dataclass
class FigureChunk:
    """Metadata + reference for one extracted figure."""
    chunk_id: str
    chunk_type: str = "figure"
    caption: str = ""
    figure_ref: str = ""        # e.g. "Figure 6-3"
    figure_path: str = ""       # relative path to PNG
    page_number: int = 0        # 1-indexed
    section_id: str = ""
    section_title: str = ""
    chapter_num: Optional[int] = None
    chapter_title: str = ""
    heading_hierarchy: list[str] = field(default_factory=list)
    # The caption text is used as the "text" for embedding
    text: str = ""
    cross_references: list[str] = field(default_factory=list)
    acronyms_used: list[str] = field(default_factory=list)


def _chapter_from_section(section_id: str) -> Optional[int]:
    try:
        return int(section_id.split(".")[0])
    except (ValueError, IndexError):
        return None


def extract_figures(
    pdf_path: Path,
    figures_dir: Path,
    section_nodes=None,
    dpi: int = 150,
) -> list[FigureChunk]:
    """
    Extract all figures/diagrams from the PDF.

    Returns a list of FigureChunk objects (one per figure).
    """
    pdf_path = Path(pdf_path)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    all_figures: list[FigureChunk] = []
    fig_counter = 0

    # Build quick page→section lookup
    page_to_section: dict[int, str] = {}
    if section_nodes:
        for node in section_nodes:
            for pg in range(node.page_start, (node.page_end or node.page_start) + 1):
                if pg not in page_to_section:
                    page_to_section[pg] = node.section_id

    for pg_idx, page in enumerate(doc):
        text = page.get_text("text")
        page_rect = page.rect
        section_id = page_to_section.get(pg_idx, "")
        chapter_num = _chapter_from_section(section_id)

        # ── 1. Caption-driven figure extraction ─────────────────────────────
        for m in _CAPTION_RE.finditer(text):
            caption_text = m.group(1).strip()
            fig_ref_match = re.match(r"(Fig(?:ure)?\.?\s*[\d.-]+)", caption_text, re.IGNORECASE)
            fig_ref = fig_ref_match.group(1).strip() if fig_ref_match else f"Fig_{fig_counter}"

            # Try to find the caption bbox on the page to crop around it
            caption_rects = page.search_for(caption_text[:40])  # search first 40 chars
            if caption_rects:
                cap_rect = caption_rects[0]
                # Expand clip upward to capture the figure above the caption
                clip = fitz.Rect(
                    page_rect.x0,
                    max(page_rect.y0, cap_rect.y0 - 300),  # 300pt above caption
                    page_rect.x1,
                    cap_rect.y1 + 10,
                )
            else:
                # Fallback: crop the upper half of the page
                clip = fitz.Rect(
                    page_rect.x0,
                    page_rect.y0,
                    page_rect.x1,
                    page_rect.y1 * 0.6,
                )

            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)

            fig_name = f"figure_{pg_idx + 1:04d}_{fig_counter:03d}.png"
            fig_path = figures_dir / fig_name
            pix.save(str(fig_path))

            chunk = FigureChunk(
                chunk_id=f"fig_{pg_idx + 1}_{fig_counter}",
                caption=caption_text,
                figure_ref=fig_ref,
                figure_path=str(fig_path.as_posix()),
                page_number=pg_idx + 1,
                section_id=section_id,
                chapter_num=chapter_num,
                text=f"[{fig_ref}] {caption_text}",
            )
            all_figures.append(chunk)
            fig_counter += 1

        # ── 2. Embedded raster image extraction ──────────────────────────────
        img_list = page.get_images(full=True)
        for img_idx, img_info in enumerate(img_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                width, height = base_image["width"], base_image["height"]

                if width * height < _MIN_IMAGE_AREA:
                    continue  # skip tiny decorative images

                ext = base_image["ext"]
                fig_name = f"image_{pg_idx + 1:04d}_{img_idx:03d}.{ext}"
                fig_path = figures_dir / fig_name
                with open(str(fig_path), "wb") as f:
                    f.write(img_bytes)

                chunk = FigureChunk(
                    chunk_id=f"img_{pg_idx + 1}_{img_idx}",
                    caption=f"Embedded image on page {pg_idx + 1}",
                    figure_ref=f"Image p{pg_idx + 1}.{img_idx}",
                    figure_path=str(fig_path.as_posix()),
                    page_number=pg_idx + 1,
                    section_id=section_id,
                    chapter_num=chapter_num,
                    text=f"[Image p{pg_idx + 1}.{img_idx}] Embedded diagram/figure on page {pg_idx + 1}",
                )
                all_figures.append(chunk)

            except Exception as e:
                logger.debug(f"Could not extract image xref {xref} on page {pg_idx+1}: {e}")

    doc.close()
    logger.info(
        f"Figure extraction complete: {len(all_figures)} figures saved to {figures_dir}"
    )
    return all_figures
