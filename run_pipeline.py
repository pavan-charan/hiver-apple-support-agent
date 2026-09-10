"""End-to-end training, indexing, and evaluation orchestration CLI for Apple Support AI Agent."""
import argparse
import sys
from pathlib import Path

from src.build_index import build_faiss_index
from src.config import (
    GOLDEN_200_CSV_PATH,
    RAW_TWCS_PATH,
    TRAIN_CSV_PATH,
    VAL_CSV_PATH,
    ensure_directories_exist,
)
from src.create_golden_dataset import create_golden_files
from src.embeddings import EmbeddingEngine
from src.evaluate import evaluate_agent
from src.preprocess import preprocess_dataset
from src.train_classifier import train_intent_classifier
from src.utils import logger, seed_everything, timed_execution


@timed_execution
def main():
    parser = argparse.ArgumentParser(description="Run complete Apple Support AI Agent pipeline.")
    parser.add_argument("--raw-path", type=str, default=str(RAW_TWCS_PATH), help="Path to raw twcs.csv")
    parser.add_argument("--force-preprocess", action="store_true", help="Force re-running preprocessing from raw CSV")
    parser.add_argument("--sample-size", type=int, default=10000, help="Number of threads to sample from raw data")
    parser.add_argument("--judge", action="store_true", help="Run LLM-as-a-judge during evaluation")
    args = parser.parse_args()

    seed_everything(42)
    ensure_directories_exist()

    logger.info("=========================================================")
    logger.info("  Starting Apple Support AI Agent End-to-End Pipeline   ")
    logger.info("=========================================================")

    # Ensure golden datasets exist
    if args.force_preprocess or not GOLDEN_200_CSV_PATH.exists():
        logger.info("Creating curated golden benchmark dataset...")
        create_golden_files()

    raw_csv_path = Path(args.raw_path)

    # Step 1: Preprocessing
    if args.force_preprocess or not TRAIN_CSV_PATH.exists():
        if raw_csv_path.exists():
            logger.info(f"Step 1/4: Preprocessing raw dataset from {raw_csv_path}...")
            preprocess_dataset(raw_path=raw_csv_path, sample_size=args.sample_size)
        else:
            if not TRAIN_CSV_PATH.exists():
                logger.error(
                    f"\n[ERROR] Raw Kaggle dataset not found at '{raw_csv_path}'.\n"
                    "Please download 'Customer Support on Twitter' from Kaggle and place 'twcs.csv' into 'data/raw/twcs.csv'.\n"
                    "If you only wish to test inference or evaluation, pre-processed train.csv or golden_200.csv is required."
                )
                sys.exit(1)
            else:
                logger.info("Step 1/4: Using existing processed train.csv & validation.csv.")
    else:
        logger.info("Step 1/4: Found existing processed train.csv & validation.csv. Skipping preprocess.")

    # Initialize shared embedding engine
    logger.info("Initializing SentenceTransformer embedding engine (all-MiniLM-L6-v2)...")
    embedding_engine = EmbeddingEngine()

    # Step 2: Train Intent Classifier
    logger.info("Step 2/4: Training Logistic Regression Intent Classifier...")
    intent_classifier, train_metrics = train_intent_classifier(
        train_csv=TRAIN_CSV_PATH,
        val_csv=VAL_CSV_PATH,
        save_models=True,
    )

    # Step 3: Build FAISS Vector Index
    logger.info("Step 3/4: Building FAISS Vector Index for RAG retrieval...")
    build_faiss_index(
        train_csv=TRAIN_CSV_PATH,
        embedding_engine=embedding_engine,
    )

    # Step 4: Evaluate Pipeline on Golden Benchmark
    logger.info("Step 4/4: Evaluating Full Agent on Golden Benchmark...")
    metrics = evaluate_agent(
        dataset_path=GOLDEN_200_CSV_PATH,
        run_judge=args.judge,
    )

    logger.info("=========================================================")
    logger.info("  Pipeline Completed Successfully in < 15 Minutes!       ")
    logger.info("  Results saved to: results/metrics.json & predictions.csv")
    logger.info("=========================================================")


if __name__ == "__main__":
    main()
