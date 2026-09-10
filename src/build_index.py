"""Build and serialize FAISS vector index from historical Apple Support conversation corpus."""
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import faiss
import numpy as np
import pandas as pd

from src.config import (
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    MODELS_DIR,
    TRAIN_CSV_PATH,
    ensure_directories_exist,
)
from src.embeddings import EmbeddingEngine
from src.utils import logger, timed_execution


@timed_execution
def build_faiss_index(
    train_csv: Path = TRAIN_CSV_PATH,
    index_path: Path = FAISS_INDEX_PATH,
    metadata_path: Path = FAISS_METADATA_PATH,
    embedding_engine: Optional[EmbeddingEngine] = None,
) -> tuple:
    """Build FAISS IndexFlatIP (Cosine similarity) over customer tweets and persist index + metadata.

    Args:
        train_csv: Path to train.csv containing 'tweet', 'gold_reply', 'intent'.
        index_path: Output path for FAISS binary index.
        metadata_path: Output path for pickle metadata mapping.
        embedding_engine: Optional pre-instantiated EmbeddingEngine.

    Returns:
        Tuple of (faiss.IndexFlatIP, list of metadata dicts)
    """
    ensure_directories_exist()
    if not train_csv.exists():
        raise FileNotFoundError(f"Training dataset not found at {train_csv}")

    logger.info(f"Loading conversation corpus for indexing from {train_csv}...")
    df = pd.read_csv(train_csv)
    logger.info(f"Corpus size: {len(df)} Q&A pairs.")

    engine = embedding_engine or EmbeddingEngine()

    logger.info("Generating L2-normalized embeddings for vector index...")
    embeddings = engine.embed_texts(df["tweet"].tolist(), normalize=True, show_progress_bar=True)

    dim = embeddings.shape[1]
    logger.info(f"Building FAISS IndexFlatIP with dimension {dim} for {len(embeddings)} vectors...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    metadata: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        metadata.append(
            {
                "tweet": str(row["tweet"]),
                "apple_reply": str(row["gold_reply"]),
                "intent": str(row.get("intent", "general_inquiry")),
            }
        )

    # Persist index and metadata
    faiss.write_index(index, str(index_path))
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    logger.info(f"Successfully saved FAISS index to {index_path}")
    logger.info(f"Successfully saved metadata ({len(metadata)} records) to {metadata_path}")

    return index, metadata


if __name__ == "__main__":
    build_faiss_index()
