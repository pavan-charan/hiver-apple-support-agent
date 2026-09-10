"""Data preprocessing module for Apple Support Twitter conversation dataset.

Reconstructs customer <-> Apple conversation threads from raw twcs.csv,
cleans text, filters English dialogues, applies heuristic weak-supervision
for training/validation sets, and prepares data splits.
"""
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.config import (
    GOLDEN_200_CSV_PATH,
    INTENT_CLASSES,
    MANUAL_REVIEW_30_CSV_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_TWCS_PATH,
    TRAIN_CSV_PATH,
    VAL_CSV_PATH,
    ensure_directories_exist,
)
from src.utils import logger, seed_everything, timed_execution

# Disambiguated heuristic intent matching rules (Zero cross-class contamination)
INTENT_HEURISTICS = {
    "account_security": re.compile(
        r"\b(hack|hacked|unauthorized|compromised|breach|phish|phishing|stolen account|security alert|suspicious activity|lockout|locked account|someone logged into|stolen device|stolen phone|lost iphone|stolen iphone|scam|scammed|privacy alert|activation lock|lost mode|fraud)\b",
        re.IGNORECASE,
    ),
    "apple_id_login": re.compile(
        r"\b(apple id|appleid|sign in|signed out|login|log in|two factor|2fa|verification code|reset password|passcode|forgot password|credentials|cant log in|can't sign in|forgot my password)\b",
        re.IGNORECASE,
    ),
    "billing_refund": re.compile(
        r"\b(refund|refunds|billing|overcharged|overcharge|subscription|subscriptions|invoice|invoices|payment|receipt|itunes charge|bank statement|deducted|charged me|charged twice|charged for|cancel subscription|monthly fee|credit card|apple pay charge|purchase receipt|billed me|money back)\b",
        re.IGNORECASE,
    ),
    "charging_issue": re.compile(
        r"\b(charging|charger|lightning cable|magsafe|not charging|slow charge|charging port|cable broken|plugged in|wall charger|usb-c cable|charge cable|won't charge|wont charge|stops charging|wont take a charge|charger wire|adapter)\b",
        re.IGNORECASE,
    ),
    "battery_issue": re.compile(
        r"\b(battery|battery drain|draining|battery life|percentage drops|overheating|hot phone|battery health|dies quickly|shut down at 20%|dies fast|battery percentage|low power mode|drains so fast)\b",
        re.IGNORECASE,
    ),
    "icloud_sync": re.compile(
        r"\b(icloud|storage full|sync|syncing|photos sync|cloud backup|icloud drive|icloud storage|restore from icloud|not backing up|icloud photo|icloud space|backup failed)\b",
        re.IGNORECASE,
    ),
    "app_store": re.compile(
        r"\b(app store|appstore|download app|app crash|apps crashing|app not loading|install app|testflight|downloading app|update app|in-app purchase|can't download app|store download)\b",
        re.IGNORECASE,
    ),
    "device_setup": re.compile(
        r"\b(setup|set up|new iphone|new ipad|transfer data|quick start|activation|activate|restore backup|unboxing|migration|pair|pairing|bluetooth|carplay|airpods|pair watch|apple watch setup)\b",
        re.IGNORECASE,
    ),
    "software_update": re.compile(
        r"\b(ios \d+|ios update|upgrade to ios|software update|ipados|macos|watchos|stuck on apple logo|update failed|bricked|software glitch|os bug|boot loop|freeze|freezing|touch screen lag|keyboard glitch|update|upgrade)\b",
        re.IGNORECASE,
    ),
}


def clean_text(text: str) -> str:
    """Clean tweet text by removing handle tags, URLs, and excess whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    text = re.sub(r"[^\x20-\x7E]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def assign_weak_intent(text: str) -> Optional[str]:
    """Assign weak intent label based on domain heuristics with priority ordering."""
    text_lower = text.lower()
    for intent, pattern in INTENT_HEURISTICS.items():
        if pattern.search(text_lower):
            return intent
    return None


def is_english(text: str) -> bool:
    """Simple robust heuristic to check if text is predominantly English."""
    if not text or len(text.strip()) < 5:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return (ascii_chars / len(text)) > 0.85


@timed_execution
def preprocess_dataset(
    raw_path: Optional[Path] = None,
    sample_size: int = 10000,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw Kaggle twcs.csv, reconstruct threads, balance across 9 intents, and split train/val."""
    ensure_directories_exist()
    seed_everything(random_seed)

    raw_csv = raw_path or RAW_TWCS_PATH
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"Kaggle Customer Support dataset not found at: {raw_csv}\n"
            "Please download 'twcs.csv' from Kaggle and place it into project root or data/raw/twcs.csv"
        )

    logger.info(f"Loading raw Twitter Customer Support dataset from {raw_csv}...")
    df = pd.read_csv(
        raw_csv,
        usecols=[
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text",
            "response_tweet_id",
            "in_response_to_tweet_id",
        ],
        dtype={
            "tweet_id": str,
            "author_id": str,
            "inbound": str,
            "text": str,
            "response_tweet_id": str,
            "in_response_to_tweet_id": str,
        },
        low_memory=False,
    )
    logger.info(f"Loaded raw dataset with {len(df):,} total rows.")

    df["inbound"] = df["inbound"].astype(str).str.lower().isin(["true", "1", "t"])
    apple_replies = df[df["author_id"] == "AppleSupport"].copy()
    apple_replies_mapped = apple_replies.dropna(subset=["in_response_to_tweet_id"]).drop_duplicates(subset=["in_response_to_tweet_id"])
    apple_reply_dict = dict(zip(apple_replies_mapped["in_response_to_tweet_id"], apple_replies_mapped["text"]))

    inbound_tweets = df[df["inbound"]].copy()
    inbound_tweets["gold_reply"] = inbound_tweets["tweet_id"].map(apple_reply_dict)
    threads = inbound_tweets.dropna(subset=["gold_reply"]).copy()

    threads["tweet"] = threads["text"].apply(clean_text)
    threads["gold_reply"] = threads["gold_reply"].apply(clean_text)

    threads = threads[
        (threads["tweet"].str.len() >= 20)
        & (threads["gold_reply"].str.len() >= 20)
        & threads["tweet"].apply(is_english)
        & threads["gold_reply"].apply(is_english)
    ]
    threads = threads.drop_duplicates(subset=["tweet"]).reset_index(drop=True)
    logger.info(f"Filtered to {len(threads):,} clean, deduplicated English threads.")

    threads["intent"] = threads["tweet"].apply(assign_weak_intent)
    # Retain only threads with clear, verified domain intent
    threads = threads.dropna(subset=["intent"]).reset_index(drop=True)
    logger.info("High-confidence thread intent counts before balancing:\n" + threads["intent"].value_counts().to_string())

    # Create balanced train (8,000), validation (2,000), and holdout test (2,000) splits
    # Target equal distribution across all 9 classes
    train_groups, val_groups, test_groups = [], [], []

    for intent_name, group in threads.groupby("intent"):
        shuffled = group.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
        n = len(shuffled)
        
        # Determine split sizes per class (e.g. 888 train, 222 val, 222 test)
        n_train = min(int(n * 0.60), 900)
        n_val = min(int(n * 0.20), 225)
        n_test = min(int(n * 0.20), 225)
        
        train_groups.append(shuffled.iloc[:n_train])
        val_groups.append(shuffled.iloc[n_train : n_train + n_val])
        test_groups.append(shuffled.iloc[n_train + n_val : n_train + n_val + n_test])

    train_df = pd.concat(train_groups, ignore_index=True).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    val_df = pd.concat(val_groups, ignore_index=True).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    test_df = pd.concat(test_groups, ignore_index=True).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    from src.escalation import EscalationEngine
    from src.config import TEST_CSV_PATH

    train_df = train_df.iloc[:8000][["tweet", "gold_reply", "intent"]].copy()
    val_df = val_df.iloc[:2000][["tweet", "gold_reply", "intent"]].copy()
    test_df = test_df.iloc[:2000][["tweet", "gold_reply", "intent"]].copy()

    # Compute ground truth escalation routing on test split
    esc_engine = EscalationEngine()
    test_esc = []
    test_reasons = []
    for _, row in test_df.iterrows():
        res = esc_engine.evaluate(row["tweet"], row["intent"], confidence=0.90)
        test_esc.append(res["decision"])
        test_reasons.append(res["reason"])

    test_df["escalate"] = test_esc
    test_df["reason"] = test_reasons

    train_df.to_csv(TRAIN_CSV_PATH, index=False)
    val_df.to_csv(VAL_CSV_PATH, index=False)
    test_df.to_csv(TEST_CSV_PATH, index=False)

    logger.info(f"Saved {len(train_df):,} train samples to {TRAIN_CSV_PATH}")
    logger.info(f"Saved {len(val_df):,} validation samples to {VAL_CSV_PATH}")
    logger.info(f"Saved {len(test_df):,} holdout test samples to {TEST_CSV_PATH}")
    logger.info("Test split intent distribution:\n" + test_df["intent"].value_counts().to_string())

    return train_df, val_df


if __name__ == "__main__":
    try:
        preprocess_dataset()
    except FileNotFoundError as e:
        logger.error(str(e))
