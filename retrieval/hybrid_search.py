"""
retrieval/hybrid_search.py
───────────────────────────
Performs hybrid search (dense + sparse with RRF fusion) over the Qdrant
collection.

Also supports:
  - Payload-based filtering (by chapter, chunk_type, section)
  - Multi-hop cross-reference expansion
"""
from __future__ import annotations

import logging
from typing import Optional

from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchValue,
    Prefetch,
    SparseVector,
)

import config
from storage.embedder import embed_query
from storage.qdrant_manager import get_client

logger = logging.getLogger(__name__)


def search(
    query: str,
    top_k: int = 10,
    filter_chunk_type: Optional[str] = None,
    filter_chapter: Optional[int] = None,
    filter_section_ids: Optional[list[str]] = None,
    prefetch_k: int = 30,
) -> list[dict]:
    """
    Run hybrid search (dense + sparse RRF) and return top_k results.

    Each result dict contains:
        id, score, chunk_id, text, chunk_type, section_id,
        chapter_num, page_numbers, figure_path (if applicable), …
    """
    client = get_client()

    # ── Embed query ───────────────────────────────────────────────────────────
    emb = embed_query(query)

    # ── Build optional payload filter ────────────────────────────────────────
    must_conditions = []
    if filter_chunk_type:
        must_conditions.append(
            FieldCondition(key="chunk_type", match=MatchValue(value=filter_chunk_type))
        )
    if filter_chapter is not None:
        must_conditions.append(
            FieldCondition(key="chapter_num", match=MatchValue(value=filter_chapter))
        )
    if filter_section_ids:
        must_conditions.append(
            FieldCondition(key="section_id", match=MatchAny(any=filter_section_ids))
        )

    qdrant_filter = Filter(must=must_conditions) if must_conditions else None

    # ── Hybrid query with RRF ─────────────────────────────────────────────────
    results = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        prefetch=[
            Prefetch(
                query=emb.dense,
                using="dense",
                limit=prefetch_k,
                filter=qdrant_filter,
            ),
            Prefetch(
                query=SparseVector(
                    indices=emb.sparse_indices,
                    values=emb.sparse_values,
                ),
                using="sparse",
                limit=prefetch_k,
                filter=qdrant_filter,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )

    hits = []
    for point in results.points:
        payload = point.payload or {}
        hits.append(
            {
                "id": point.id,
                "score": point.score,
                "chunk_id": payload.get("chunk_id", ""),
                "text": payload.get("text", ""),
                "chunk_type": payload.get("chunk_type", "text"),
                "section_id": payload.get("section_id", ""),
                "section_title": payload.get("section_title", ""),
                "chapter_num": payload.get("chapter_num"),
                "chapter_title": payload.get("chapter_title", ""),
                "page_numbers": payload.get("page_numbers", []),
                "heading_hierarchy": payload.get("heading_hierarchy", []),
                "figure_path": payload.get("figure_path", ""),
                "caption": payload.get("caption", ""),
                "cross_references": payload.get("cross_references", []),
                "figure_ref": payload.get("figure_ref", ""),
                "table_json": payload.get("table_json", ""),
            }
        )

    logger.info(f"Hybrid search returned {len(hits)} results for: {query[:60]!r}")
    return hits


def search_with_cross_refs(
    query: str,
    top_k: int = 8,
    expand_refs: bool = True,
    **kwargs,
) -> list[dict]:
    """
    Search with optional cross-reference expansion (multi-hop):
    1. Run primary search.
    2. Collect all cross_references from primary hits.
    3. Re-query for those referenced sections.
    4. Merge and deduplicate results.
    """
    primary_hits = search(query, top_k=top_k, **kwargs)

    if not expand_refs:
        return primary_hits

    # Collect unique cross-reference section IDs from primary hits
    ref_ids: set[str] = set()
    for hit in primary_hits:
        for ref in hit.get("cross_references", []):
            ref_ids.add(ref)

    if not ref_ids:
        return primary_hits

    logger.info(f"Expanding {len(ref_ids)} cross-references: {ref_ids}")

    # Fetch referenced sections directly via payload filter
    additional_hits = search(
        query,
        top_k=len(ref_ids) * 2,
        filter_section_ids=list(ref_ids),
    )

    # Merge, dedup by chunk_id, primary hits first
    seen_ids: set[str] = {h["chunk_id"] for h in primary_hits}
    for hit in additional_hits:
        if hit["chunk_id"] not in seen_ids:
            hit["_via_cross_ref"] = True  # tag for UI
            primary_hits.append(hit)
            seen_ids.add(hit["chunk_id"])

    return primary_hits
