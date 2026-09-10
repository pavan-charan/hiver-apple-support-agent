"""Create a clean 2,000-sample test dataset from twcs.csv excluding train, validation, and golden sets."""
from pathlib import Path
import pandas as pd

from src.config import (
    GOLDEN_200_CSV_PATH,
    RAW_TWCS_PATH,
    TEST_CSV_PATH,
    TRAIN_CSV_PATH,
    VAL_CSV_PATH,
    RANDOM_SEED,
    ensure_directories_exist,
)
from src.preprocess import clean_text, is_english, assign_weak_intent
from src.escalation import EscalationEngine
from src.utils import logger, seed_everything, timed_execution


@timed_execution
def create_test_dataset(
    sample_size: int = 2000,
    raw_path: Path = RAW_TWCS_PATH,
    output_path: Path = TEST_CSV_PATH,
) -> pd.DataFrame:
    """Extract sample_size clean holdout threads strictly distinct from train, val, and golden sets."""
    seed_everything(RANDOM_SEED)
    ensure_directories_exist()

    logger.info(f"Extracting {sample_size:,} holdout test threads from {raw_path}...")
    df = pd.read_csv(raw_path, low_memory=False)
    df["inbound"] = df["inbound"].astype(str).str.lower().isin(["true", "1", "t"])

    apple_replies = df[df["author_id"] == "AppleSupport"].dropna(subset=["in_response_to_tweet_id"]).drop_duplicates(subset=["in_response_to_tweet_id"])
    reply_map = dict(zip(apple_replies["in_response_to_tweet_id"], apple_replies["text"]))

    inbound = df[df["inbound"]].copy()
    inbound["gold_reply"] = inbound["tweet_id"].map(reply_map)
    threads = inbound.dropna(subset=["gold_reply"]).copy()

    threads["tweet"] = threads["text"].apply(clean_text)
    threads["gold_reply"] = threads["gold_reply"].apply(clean_text)

    threads = threads[
        (threads["tweet"].str.len() >= 20)
        & (threads["gold_reply"].str.len() >= 20)
        & threads["tweet"].apply(is_english)
        & threads["gold_reply"].apply(is_english)
    ]
    threads = threads.drop_duplicates(subset=["tweet"]).reset_index(drop=True)

    # Gather all existing used tweets across train, validation, and golden datasets
    used_tweets = set()
    if TRAIN_CSV_PATH.exists():
        t_df = pd.read_csv(TRAIN_CSV_PATH)
        used_tweets.update(t_df["tweet"].dropna().tolist())
    if VAL_CSV_PATH.exists():
        v_df = pd.read_csv(VAL_CSV_PATH)
        used_tweets.update(v_df["tweet"].dropna().tolist())
    if GOLDEN_200_CSV_PATH.exists():
        g_df = pd.read_csv(GOLDEN_200_CSV_PATH)
        used_tweets.update(g_df["tweet"].dropna().tolist())

    logger.info(f"Total existing used tweets to exclude: {len(used_tweets):,}")

    # Exclude all used tweets
    holdout = threads[~threads["tweet"].isin(used_tweets)].copy().reset_index(drop=True)
    logger.info(f"Available fresh holdout pool: {len(holdout):,} threads.")

    # Assign intents and drop unmatched
    holdout["intent"] = holdout["tweet"].apply(assign_weak_intent)
    holdout = holdout.dropna(subset=["intent"]).reset_index(drop=True)

    # Sample balanced threads across all 9 classes
    target_per_class = sample_size // 9 + 5
    sampled_list = []
    for intent_name, group in holdout.groupby("intent"):
        sampled_list.append(group.sample(n=min(len(group), target_per_class), random_state=RANDOM_SEED))
    test_df = pd.concat(sampled_list, ignore_index=True).sample(frac=1.0, random_state=RANDOM_SEED).iloc[:sample_size].reset_index(drop=True)

    # Determine ground truth escalation decision
    escalation_engine = EscalationEngine()
    escalate_decisions = []
    reasons = []

    for _, row in test_df.iterrows():
        res = escalation_engine.evaluate(row["tweet"], row["intent"], confidence=0.90)
        escalate_decisions.append(res["decision"])
        reasons.append(res["reason"])

    test_df["escalate"] = escalate_decisions
    test_df["reason"] = reasons

    final_cols = ["tweet", "intent", "gold_reply", "escalate", "reason"]
    test_out = test_df[final_cols].copy()

    test_out.to_csv(output_path, index=False)
    logger.info(f"Successfully saved {len(test_out):,} holdout test samples to {output_path}")
    return test_out


if __name__ == "__main__":
    create_test_dataset()
