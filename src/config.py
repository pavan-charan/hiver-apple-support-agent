"""Central configuration module for Apple Support AI Agent."""
import os
from pathlib import Path
from typing import List

# Base Paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
GOLDEN_DATA_DIR: Path = DATA_DIR / "golden"

MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# Data File Paths
RAW_TWCS_PATH: Path = RAW_DATA_DIR / "twcs.csv"
if not RAW_TWCS_PATH.exists() and (PROJECT_ROOT / "twcs.csv").exists():
    RAW_TWCS_PATH = PROJECT_ROOT / "twcs.csv"

TRAIN_CSV_PATH: Path = PROCESSED_DATA_DIR / "train.csv"
VAL_CSV_PATH: Path = PROCESSED_DATA_DIR / "validation.csv"
TEST_CSV_PATH: Path = PROCESSED_DATA_DIR / "test.csv"
GOLDEN_200_CSV_PATH: Path = GOLDEN_DATA_DIR / "golden_200.csv"
MANUAL_REVIEW_30_CSV_PATH: Path = GOLDEN_DATA_DIR / "manual_review_30.csv"

# Model File Paths
INTENT_CLASSIFIER_PATH: Path = MODELS_DIR / "intent_classifier.pkl"
LABEL_ENCODER_PATH: Path = MODELS_DIR / "label_encoder.pkl"
FAISS_INDEX_PATH: Path = MODELS_DIR / "faiss_index.bin"
FAISS_METADATA_PATH: Path = MODELS_DIR / "faiss_metadata.pkl"

# Results File Paths
METRICS_JSON_PATH: Path = RESULTS_DIR / "metrics.json"
PREDICTIONS_CSV_PATH: Path = RESULTS_DIR / "predictions.csv"
TEST_METRICS_JSON_PATH: Path = RESULTS_DIR / "test_metrics.json"
TEST_PREDICTIONS_CSV_PATH: Path = RESULTS_DIR / "test_predictions.csv"
CONFUSION_MATRIX_PATH: Path = RESULTS_DIR / "confusion_matrix.png"

# Reproducibility Seed
RANDOM_SEED: int = 42

# 9 Canonical Intent Classes
INTENT_CLASSES: List[str] = [
    "apple_id_login",
    "icloud_sync",
    "app_store",
    "battery_issue",
    "charging_issue",
    "software_update",
    "device_setup",
    "billing_refund",
    "account_security",
]

# Embedding Configuration
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384
EMBEDDING_BATCH_SIZE: int = 64

# Retrieval Configuration
TOP_K_RETRIEVAL: int = 5

# Escalation Engine Configuration
CONFIDENCE_THRESHOLD_AUTO: float = 0.85
CONFIDENCE_THRESHOLD_ESCALATE: float = 0.60

ESCALATE_INTENTS: List[str] = [
    "account_security",
    "billing_refund",
]

ESCALATE_KEYWORDS: List[str] = [
    "lost",
    "stolen",
    "fraud",
    "unauthorized",
    "hacked",
    "compromised",
    "scam",
    "bank",
    "chargeback",
    "lawsuit",
    "lawyer",
    "stole",
    "breach",
    "phishing",
    "stolen phone",
    "stolen ipad",
    "stolen macbook",
    "stolen watch",
]

# OpenAI & Reply Generation Configuration
OPENAI_MODEL_NAME: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_REPLY_WORDS: int = 70


def ensure_directories_exist() -> None:
    """Ensure all required directories exist."""
    for directory in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        GOLDEN_DATA_DIR,
        MODELS_DIR,
        RESULTS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
