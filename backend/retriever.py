"""
Retriever module for Reading Passage Evidence Locator.
Retrieves the most relevant passages for a question, ranks them by similarity,
applies configurable evidence thresholds, and determines evidence sufficiency.
"""

import logging
from typing import List, Dict, Any
from backend.vector_store import FAISSVectorStore
from backend.embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class EvidenceRetriever:
    """Retrieves top-k evidence passages and filters by similarity score threshold."""

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedding_manager: EmbeddingManager,
        similarity_threshold: float = 0.35
    ):
        """
        Args:
            vector_store: FAISS vector store instance.
            embedding_manager: Embedding manager instance.
            similarity_threshold: Minimum cosine similarity score required for relevance.
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.similarity_threshold = similarity_threshold

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k relevant passages for a given user query.

        Args:
            query: The question asked by user.
            top_k: Maximum number of passages to return.
            threshold: Optional override for the similarity threshold.

        Returns:
            List of evidence dictionaries ranked by similarity score:
            [
                {
                    "chunk_id": int,
                    "document_name": str,
                    "page_number": int,
                    "text": str,
                    "similarity_score": float
                },
                ...
            ]
        """
        if not query or not query.strip():
            return []

        applied_threshold = threshold if threshold is not None else self.similarity_threshold

        # Encode query
        query_emb = self.embedding_manager.get_query_embedding(query)

        # Search FAISS index
        search_results = self.vector_store.search(query_emb, top_k=top_k)

        ranked_evidence: List[Dict[str, Any]] = []
        for chunk_meta, score in search_results:
            if score >= applied_threshold:
                evidence_item = {
                    "chunk_id": chunk_meta.get("chunk_id"),
                    "document_name": chunk_meta.get("document_name"),
                    "page_number": chunk_meta.get("page_number"),
                    "text": chunk_meta.get("text"),
                    "similarity_score": round(float(score), 4)
                }
                ranked_evidence.append(evidence_item)

        logger.info(
            f"Retrieved {len(ranked_evidence)} passages (out of {len(search_results)} candidates) "
            f"with score >= {applied_threshold} for query: '{query[:50]}...'"
        )
        return ranked_evidence

    def is_evidence_sufficient(
        self,
        retrieved_passages: List[Dict[str, Any]],
        threshold: float = None
    ) -> bool:
        """
        Checks whether the retrieved passages contain sufficient semantic evidence.
        If no passages pass the threshold or the passages are empty, returns False.

        Args:
            retrieved_passages: List of retrieved evidence passages.
            threshold: Optional threshold override.

        Returns:
            bool: True if sufficient evidence is present, False otherwise.
        """
        if not retrieved_passages:
            return False

        applied_threshold = threshold if threshold is not None else self.similarity_threshold
        # Check if the highest ranked passage meets the threshold
        top_score = max((p.get("similarity_score", 0.0) for p in retrieved_passages), default=0.0)
        return top_score >= applied_threshold
