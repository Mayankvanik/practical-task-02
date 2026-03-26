"""
storage/indexer.py
───────────────────
Upserts all chunks into the Qdrant Cloud collection in batches.

Each Qdrant point has:
  - id       : deterministic integer hash of chunk_id
  - vectors  : {"dense": [...], "sparse": {"indices": [...], "values": [...]}}
  - payload  : all chunk metadata fields
"""
from __future__ import annotations

import hashlib
import logging

from qdrant_client.models import PointStruct, SparseVector

import config
from storage.embedder import EmbeddingPair, embed_batch
from storage.qdrant_manager import get_client

logger = logging.getLogger(__name__)

_UPSERT_BATCH = 50  # number of points per upsert call


def _chunk_id_to_int(chunk_id: str) -> int:
    """Convert a string chunk_id to a stable unsigned integer for Qdrant point ID."""
    return int(hashlib.md5(chunk_id.encode()).hexdigest(), 16) % (2**63)


def index_chunks(
    chunks: list[dict],
    embed_batch_size: int = 50,
) -> int:
    """
    Embed and upsert all chunks into Qdrant.

    Args:
        chunks: list of dicts each containing 'chunk_id', 'text', 'payload'
        embed_batch_size: how many chunks to embed at once

    Returns:
        Total number of points upserted.
    """
    client = get_client()
    collection_name = config.QDRANT_COLLECTION
    total_upserted = 0

    logger.info(f"Indexing {len(chunks)} chunks into '{collection_name}' …")

    for batch_start in range(0, len(chunks), embed_batch_size):
        batch = chunks[batch_start : batch_start + embed_batch_size]
        texts = [c["text"] for c in batch]

        # ── Embed ─────────────────────────────────────────────────────────────
        try:
            pairs: list[EmbeddingPair] = embed_batch(texts)
        except Exception as e:
            logger.error(f"Embedding failed for batch starting at {batch_start}: {e}")
            continue

        # ── Build PointStructs ────────────────────────────────────────────────
        points: list[PointStruct] = []
        for chunk, emb in zip(batch, pairs):
            point_id = _chunk_id_to_int(chunk["chunk_id"])
            point = PointStruct(
                id=point_id,
                vector={
                    "dense": emb.dense,
                    "sparse": SparseVector(
                        indices=emb.sparse_indices,
                        values=emb.sparse_values,
                    ),
                },
                payload={
                    **chunk["payload"],
                    "chunk_id": chunk["chunk_id"],
                    # Store the text in payload too, so we can retrieve it at query time
                    "text": chunk["text"],
                },
            )
            points.append(point)

        # ── Upsert to Qdrant ──────────────────────────────────────────────────
        try:
            client.upsert(collection_name=collection_name, points=points)
            total_upserted += len(points)
            logger.info(
                f"  Upserted batch {batch_start // embed_batch_size + 1}: "
                f"{len(points)} points  (total: {total_upserted})"
            )
        except Exception as e:
            logger.error(f"Qdrant upsert failed for batch at {batch_start}: {e}")

    logger.info(
        f"Indexing complete. {total_upserted}/{len(chunks)} chunks stored in Qdrant."
    )
    return total_upserted


def get_collection_stats() -> dict:
    """Return info about the current collection (point count, vector config)."""
    client = get_client()
    info = client.get_collection(config.QDRANT_COLLECTION)
    return {
        "points_count": info.points_count,
        "indexed_vectors_count": info.indexed_vectors_count,
        "status": str(info.status),
        "collection": config.QDRANT_COLLECTION,
    }
