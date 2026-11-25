"""
Embedding manager module for Reading Passage Evidence Locator.
Wraps Sentence Transformers with CPU execution and singleton caching.
Ensures the model is loaded only once in memory across requests.
"""

import threading
import logging
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Thread-safe Singleton class for generating normalized sentence embeddings.
    Caches the embedding model in memory to prevent repeated overhead.
    """
    _instance = None
    _lock = threading.Lock()
    _model = None
    _model_name = None

    def __new__(cls, model_name: str = "all-MiniLM-L6-v2"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EmbeddingManager, cls).__new__(cls)
                cls._instance._init_model(model_name)
        return cls._instance

    def _init_model(self, model_name: str):
        """Initializes the Sentence Transformer model on CPU."""
        self._model_name = model_name
        logger.info(f"Loading Sentence Transformer model '{model_name}' on CPU...")
        try:
            self._model = SentenceTransformer(model_name, device="cpu")
            if hasattr(self._model, "get_embedding_dimension"):
                self._dimension = self._model.get_embedding_dimension()
            else:
                self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Loaded '{model_name}' successfully. Embedding dimension: {self._dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model '{model_name}': {e}")
            raise RuntimeError(f"Could not initialize embedding model '{model_name}': {e}")

    @property
    def dimension(self) -> int:
        """Returns embedding vector dimension (384 for all-MiniLM-L6-v2)."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Returns the loaded model name."""
        return self._model_name

    def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generates L2-normalized embeddings for one or more text inputs.

        Args:
            texts: Single string or list of strings.

        Returns:
            np.ndarray: 2D array of shape (N, dimension) with dtype float32.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        # Generate embeddings on CPU with normalization
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
            device="cpu"
        )

        return embeddings.astype(np.float32)

    def get_query_embedding(self, query: str) -> np.ndarray:
        """
        Generates normalized embedding for a single user query.

        Args:
            query: User question string.

        Returns:
            np.ndarray: 2D array of shape (1, dimension).
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")
        return self.get_embeddings([query.strip()])
