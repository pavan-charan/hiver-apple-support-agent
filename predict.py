"""Batch prediction CLI on unseen customer tweets using pre-trained AppleSupportAgent."""
import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config import PREDICTIONS_CSV_PATH, TOP_K_RETRIEVAL
from src.inference import AppleSupportAgent
from src.utils import logger, timed_execution


@timed_execution
def predict_on_file(
    input_path: Path,
    output_path: Path,
    k: int = TOP_K_RETRIEVAL,
) -> pd.DataFrame:
    """Load unseen customer tweets, run inference through AppleSupportAgent, and save output CSV.

    Args:
        input_path: Path to input CSV containing 'tweet' or 'text' column.
        output_path: Destination path for output CSV.
        k: Number of RAG exemplars to retrieve.

    Returns:
        pd.DataFrame containing prediction results.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    logger.info(f"Loading input data from {input_path}...")
    df = pd.read_csv(input_path)

    # Detect text column name
    text_col = None
    for col in ["tweet", "text", "customer_message", "query", "message"]:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        raise ValueError(
            f"Input CSV at {input_path} must contain a 'tweet' or 'text' column. "
            f"Found columns: {list(df.columns)}"
        )

    tweets = df[text_col].astype(str).tolist()
    logger.info(f"Loaded {len(tweets)} tweets for inference.")

    # Initialize agent (loads pre-trained weights and index without retraining)
    agent = AppleSupportAgent()

    logger.info(f"Running inference through AppleSupportAgent (k={k})...")
    results = agent.predict_batch(tweets, k=k)

    # Structure into required output format
    output_df = pd.DataFrame(
        [
            {
                "tweet": r["tweet"],
                "intent": r["intent"],
                "confidence": r["confidence"],
                "generated_reply": r["generated_reply"],
                "decision": r["decision"],
                "reason": r["reason"],
            }
            for r in results
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(output_df)} predictions to {output_path}")

    # Display preview table
    logger.info("\nSample Prediction Output:\n" + output_df.head(3).to_string())
    return output_df


def main():
    parser = argparse.ArgumentParser(description="Predict intents, replies, and routing on unseen Apple tweets.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV containing customer tweets (must have 'tweet' or 'text' column)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PREDICTIONS_CSV_PATH),
        help="Path to output CSV for structured predictions",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=TOP_K_RETRIEVAL,
        help="Number of nearest historical exemplars to retrieve",
    )
    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)

    try:
        predict_on_file(input_path=input_file, output_path=output_file, k=args.k)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
