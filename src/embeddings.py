"""Sentence embedding module wrapping sentence-transformers with batching and normalization."""
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL_NAME
from src.utils import logger, timed_execution


class EmbeddingEngine:
    """Extracts dense sentence embeddings using MiniLM with L2 normalization."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        logger.info(f"Loading embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded successfully (Dimension: {self.dim}).")

    @timed_execution
    def embed_texts(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode a list of text strings into numpy embedding matrix.

        Args:
            texts: List of text strings.
            normalize: If True, apply L2 normalization for exact cosine similarity.
            show_progress_bar: Whether to display sentence-transformers progress bar.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim) with float32 dtype.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        # Sanitize text
        cleaned_texts = [str(t) if t is not None and len(str(t).strip()) > 0 else " " for t in texts]

        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """Encode a single query string into shape (1, embedding_dim)."""
        emb = self.embed_texts([query], normalize=normalize, show_progress_bar=False)
        return emb.reshape(1, -1)
