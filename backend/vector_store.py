"""
FAISS CPU Vector Store for Reading Passage Evidence Locator.
Maintains an in-memory FAISS IndexFlatIP (Inner Product = Cosine Similarity for normalized vectors)
and stores associated passage metadata for instant retrieval.
Supports multiple uploaded PDFs.
"""

import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS CPU vector store supporting multi-document indexing and similarity search."""

    def __init__(self, dimension: int = 384):
        """
        Args:
            dimension: Dimension of the embedding vectors (default 384 for all-MiniLM-L6-v2).
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: List[Dict[str, Any]] = []

    def add_documents(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]):
        """
        Adds document chunks and their corresponding embeddings to the index.

        Args:
            embeddings: 2D numpy array of shape (N, dimension) of dtype float32.
            chunks: List of N metadata dictionaries matching the embeddings.
        """
        if len(chunks) == 0:
            logger.warning("No chunks provided to add to vector store.")
            return

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Mismatch: embeddings count ({embeddings.shape[0]}) != chunks count ({len(chunks)})"
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}"
            )

        # Ensure float32 format
        emb_f32 = embeddings.astype(np.float32)

        # Add to FAISS index
        self.index.add(emb_f32)
        # Store metadata
        self.metadata.extend(chunks)

        logger.info(f"Added {len(chunks)} chunks to FAISS index. Total indexed: {self.total_chunks}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Performs inner product (cosine similarity) search for the query embedding.

        Args:
            query_embedding: 2D numpy array of shape (1, dimension).
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            List of tuples: [(chunk_metadata, similarity_score), ...] sorted descending by score.
        """
        if self.total_chunks == 0:
            logger.warning("Search called on empty vector store.")
            return []

        # Ensure 2D float32 array
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_f32 = query_embedding.astype(np.float32)

        k = min(top_k, self.total_chunks)
        scores, indices = self.index.search(query_f32, k)

        results: List[Tuple[Dict[str, Any], float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                results.append((self.metadata[idx], float(score)))

        return results

    def clear(self):
        """Resets the vector index and metadata."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        logger.info("Cleared FAISS vector store.")

    @property
    def total_chunks(self) -> int:
        """Returns total number of chunks currently indexed."""
        return self.index.ntotal if self.index is not None else 0

    def get_indexed_documents(self) -> List[str]:
        """Returns unique list of document names currently in the store."""
        seen = set()
        docs = []
        for meta in self.metadata:
            d_name = meta.get("document_name", "unknown")
            if d_name not in seen:
                seen.add(d_name)
                docs.append(d_name)
        return docs
