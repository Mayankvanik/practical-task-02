"""
storage/embedder.py
────────────────────
Generates dense embeddings (via OpenAI) and sparse vectors (BM25-style via
fastembed) for each chunk's text.

Dense embeddings   → semantic / conceptual similarity
Sparse vectors     → keyword / exact-term matching (like BM25)

The two are combined in Qdrant with Reciprocal Rank Fusion (RRF) for
hybrid search.
"""
from __future__ import annotations

import logging
from typing import NamedTuple

import config

logger = logging.getLogger(__name__)


class EmbeddingPair(NamedTuple):
    dense: list[float]          # dense vector
    sparse_indices: list[int]   # sparse vector (CSR format)
    sparse_values: list[float]


# ── Dense embedder (OpenAI) ───────────────────────────────────────────────────
def _get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=config.OPENAI_API_KEY)


def embed_dense_batch(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """
    Embed a list of texts using OpenAI embeddings.
    Returns a list of float vectors.
    """
    client = _get_openai_client()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Clean texts: replace newlines (OpenAI recommendation)
        batch = [t.replace("\n", " ").strip() or "empty" for t in batch]
        resp = client.embeddings.create(model=config.EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([item.embedding for item in resp.data])
        logger.debug(f"  Embedded batch {i // batch_size + 1}: {len(batch)} texts")

    return all_embeddings


# ── Sparse embedder (fastembed BM25) ──────────────────────────────────────────
_sparse_model = None


def _get_sparse_model():
    global _sparse_model
    if _sparse_model is None:
        try:
            from fastembed import SparseTextEmbedding
            _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
            logger.info("Sparse (BM25) model loaded via fastembed.")
        except Exception as e:
            logger.error(f"Failed to load sparse model: {e}")
            raise
    return _sparse_model


def embed_sparse_batch(texts: list[str]) -> list[tuple[list[int], list[float]]]:
    """
    Generate sparse vectors for a list of texts using fastembed BM25.
    Returns list of (indices, values) tuples.
    """
    model = _get_sparse_model()
    results = []
    for embedding in model.embed(texts, batch_size=32):
        indices = embedding.indices.tolist()
        values = embedding.values.tolist()
        results.append((indices, values))
    return results


def embed_batch(texts: list[str]) -> list[EmbeddingPair]:
    """
    Generate both dense and sparse embeddings for a list of texts.
    Returns a list of EmbeddingPair named tuples.
    """
    logger.info(f"Embedding {len(texts)} texts (dense + sparse) …")
    dense_vecs = embed_dense_batch(texts)
    sparse_vecs = embed_sparse_batch(texts)

    pairs: list[EmbeddingPair] = []
    for dense, (s_idx, s_val) in zip(dense_vecs, sparse_vecs):
        pairs.append(EmbeddingPair(
            dense=dense,
            sparse_indices=s_idx,
            sparse_values=s_val,
        ))
    return pairs


def embed_query(query: str) -> EmbeddingPair:
    """Embed a single query string (for retrieval)."""
    pairs = embed_batch([query])
    return pairs[0]
