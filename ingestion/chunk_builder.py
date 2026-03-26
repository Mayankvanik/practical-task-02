"""
ingestion/chunk_builder.py
───────────────────────────
Combines all chunk types (text, table, figure) into a unified list of dicts
ready for Qdrant upsert.

Each dict maps directly to a Qdrant point payload + text for embedding.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Union

from ingestion.figure_extractor import FigureChunk
from ingestion.table_extractor import TableChunk
from ingestion.text_extractor import TextChunk

logger = logging.getLogger(__name__)

AnyChunk = Union[TextChunk, TableChunk, FigureChunk]


def build_unified_chunks(
    text_chunks: list[TextChunk],
    table_chunks: list[TableChunk],
    figure_chunks: list[FigureChunk],
) -> list[dict]:
    """
    Merge all chunk types into a flat list of dicts.
    Each dict has the keys:
        - chunk_id   (str) — unique ID
        - text       (str) — the text to embed
        - payload    (dict) — all metadata for Qdrant payload
    """
    unified: list[dict] = []

    for tc in text_chunks:
        d = asdict(tc)
        unified.append(
            {
                "chunk_id": tc.chunk_id,
                "text": tc.text,
                "payload": {k: v for k, v in d.items() if k != "text"},
            }
        )

    for tb in table_chunks:
        d = asdict(tb)
        text_for_embed = f"{tb.caption}\n\n{tb.text}"
        unified.append(
            {
                "chunk_id": tb.chunk_id,
                "text": text_for_embed,
                "payload": {k: v for k, v in d.items() if k not in ("text",)},
            }
        )

    for fg in figure_chunks:
        d = asdict(fg)
        text_for_embed = fg.text  # caption-based text
        unified.append(
            {
                "chunk_id": fg.chunk_id,
                "text": text_for_embed,
                "payload": {k: v for k, v in d.items() if k not in ("text",)},
            }
        )

    logger.info(
        f"Unified chunks: {len(text_chunks)} text + {len(table_chunks)} tables "
        f"+ {len(figure_chunks)} figures = {len(unified)} total"
    )
    return unified


def save_chunks_to_disk(chunks: list[dict], output_dir: Path) -> None:
    """
    Serialize all chunks as individual JSON files for debugging / inspection.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        out_file = output_dir / f"{chunk['chunk_id']}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(chunk, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(chunks)} chunk JSON files to {output_dir}")
