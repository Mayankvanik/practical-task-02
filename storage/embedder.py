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


# ── Dense embedder (SentenceTransformers) ───────────────────────────────────────────────────
_dense_model = None

def _get_dense_model():
    global _dense_model
    if _dense_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading dense model {config.EMBEDDING_MODEL} on CPU...")
            # Force CPU to avoid 'meta tensor' error with partial CUDA installs
            _dense_model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")
            logger.info(f"Dense model loaded successfully on device: {_dense_model.device}")
        except Exception as e:
            logger.error(f"Failed to load dense model: {e}")
            raise
    return _dense_model


def embed_dense_batch(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """
    Embed a list of texts using local SentenceTransformers.
    """
    if not texts:
        return []

    model = _get_dense_model()
    # clean texts
    clean_texts = [str(t).replace("\n", " ").strip() or "empty" for t in texts]
    
    embeddings = model.encode(clean_texts, batch_size=batch_size, show_progress_bar=False)
    return [emb.tolist() for emb in embeddings]


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
    # Many local models (like bge or mixedbread-ai) need a specific prompt prefix for queries to match passages.
    prefix = ""
    if "bge" in config.EMBEDDING_MODEL.lower() or "mxbai" in config.EMBEDDING_MODEL.lower():
        prefix = "Represent this sentence for searching relevant passages: "
    
    dense_query = prefix + query
    logger.info(f"Embedding search query: {dense_query[:50]}...")
    
    # We call dense and sparse embedding directly rather than embed_batch to apply prefix ONLY to dense
    dense_vecs = embed_dense_batch([dense_query], batch_size=1)
    sparse_vecs = embed_sparse_batch([query])  # Sparse doesn't need the prompt
    
    return EmbeddingPair(
        dense=dense_vecs[0],
        sparse_indices=sparse_vecs[0][0],
        sparse_values=sparse_vecs[0][1]
    )
