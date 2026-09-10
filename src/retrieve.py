"""RAG Retriever module querying FAISS index for top-K historical Apple Support exemplars."""
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import faiss
import numpy as np

from src.config import FAISS_INDEX_PATH, FAISS_METADATA_PATH, TOP_K_RETRIEVAL
from src.embeddings import EmbeddingEngine
from src.utils import logger


class RAGRetriever:
    """Semantic vector search retriever querying historical Apple Support Q&A pairs."""

    def __init__(
        self,
        index_path: Union[str, Path] = FAISS_INDEX_PATH,
        metadata_path: Union[str, Path] = FAISS_METADATA_PATH,
        embedding_engine: Optional[EmbeddingEngine] = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError(
                f"FAISS index or metadata not found at {self.index_path} / {self.metadata_path}. "
                "Please run `python run_pipeline.py` or `python -m src.build_index` first."
            )

        logger.info(f"Loading FAISS index from {self.index_path}...")
        self.index = faiss.read_index(str(self.index_path))

        logger.info(f"Loading metadata from {self.metadata_path}...")
        with open(self.metadata_path, "rb") as f:
            self.metadata: List[Dict[str, str]] = pickle.load(f)

        self.embedding_engine = embedding_engine or EmbeddingEngine()
        logger.info(f"RAGRetriever initialized with {self.index.ntotal} indexed vectors.")

    def search(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        """Search top-K most similar historical customer queries.

        Args:
            query: Inbound customer tweet text.
            k: Number of nearest neighbors to retrieve.

        Returns:
            List of dicts containing similarity score, tweet, apple_reply, and intent.
        """
        if self.index.ntotal == 0:
            return []

        k = min(k, self.index.ntotal)
        query_emb = self.embedding_engine.embed_query(query, normalize=True)
        scores, indices = self.index.search(query_emb, k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            results.append(
                {
                    "similarity": float(score),
                    "tweet": meta["tweet"],
                    "apple_reply": meta["apple_reply"],
                    "intent": meta["intent"],
                }
            )
        return results

    @staticmethod
    def format_context(retrieved_examples: List[Dict[str, Any]]) -> str:
        """Format retrieved examples into structured prompt context block."""
        if not retrieved_examples:
            return "No historical examples found."

        formatted_blocks = []
        for i, ex in enumerate(retrieved_examples, 1):
            block = (
                f"[Example {i}] (Intent: {ex.get('intent', 'unknown')}, Relevance: {ex.get('similarity', 0.0):.2f})\n"
                f"Customer: {ex['tweet']}\n"
                f"Apple Support Reply: {ex['apple_reply']}"
            )
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)
