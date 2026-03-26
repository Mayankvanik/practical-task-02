"""
storage/qdrant_manager.py
──────────────────────────
Manages the Qdrant Cloud collection:
- Creates (or recreates) the collection with dense + sparse named vectors.
- Creates payload indexes on key metadata fields for fast filtering.
"""
from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PayloadSchemaType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
    VectorsConfig,
)

import config

logger = logging.getLogger(__name__)

# Singleton client
_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """Return (and cache) the Qdrant cloud client."""
    global _client
    if _client is None:
        _client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            timeout=60,
        )
        logger.info(f"Connected to Qdrant at {config.QDRANT_URL}")
    return _client


def create_collection(recreate: bool = False) -> None:
    """
    Create the Qdrant collection with:
      - 'dense'  : cosine-distance dense vectors (OpenAI embeddings)
      - 'sparse' : sparse vectors for BM25/keyword hybrid search

    Adds payload indexes on fields used for filtering.
    """
    client = get_client()
    collection_name = config.QDRANT_COLLECTION

    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        if recreate:
            logger.warning(f"Deleting existing collection '{collection_name}' for recreation.")
            client.delete_collection(collection_name)
        else:
            logger.info(f"Collection '{collection_name}' already exists — skipping creation.")
            return

    logger.info(f"Creating collection '{collection_name}' …")

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(
                size=config.EMBEDDING_DIM,
                distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        },
    )

    # ── Payload indexes for efficient filtering ───────────────────────────────
    index_fields = {
        "chunk_type":   PayloadSchemaType.KEYWORD,
        "section_id":   PayloadSchemaType.KEYWORD,
        "chapter_num":  PayloadSchemaType.INTEGER,
        "page_numbers": PayloadSchemaType.INTEGER,
    }
    for field, schema in index_fields.items():
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=schema,
        )
        logger.debug(f"  Created payload index on '{field}'")

    logger.info(f"Collection '{collection_name}' created successfully.")
