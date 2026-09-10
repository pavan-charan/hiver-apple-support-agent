"""Intent classifier training and serialization module using Logistic Regression on MiniLM embeddings."""
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

from src.config import (
    INTENT_CLASSES,
    INTENT_CLASSIFIER_PATH,
    LABEL_ENCODER_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    TRAIN_CSV_PATH,
    VAL_CSV_PATH,
    ensure_directories_exist,
)
from src.embeddings import EmbeddingEngine
from src.utils import logger, seed_everything, timed_execution


class IntentClassifier:
    """Logistic Regression Intent Classifier over dense MiniLM embeddings."""

    def __init__(
        self,
        classifier: Optional[LogisticRegression] = None,
        label_encoder: Optional[LabelEncoder] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
    ) -> None:
        self.classifier = classifier
        self.label_encoder = label_encoder
        self.embedding_engine = embedding_engine

    @classmethod
    def load(
        cls,
        classifier_path: Union[str, Path] = INTENT_CLASSIFIER_PATH,
        label_encoder_path: Union[str, Path] = LABEL_ENCODER_PATH,
        embedding_engine: Optional[EmbeddingEngine] = None,
    ) -> "IntentClassifier":
        """Load pre-trained classifier and label encoder from disk."""
        c_path = Path(classifier_path)
        le_path = Path(label_encoder_path)

        if not c_path.exists() or not le_path.exists():
            raise FileNotFoundError(
                f"Model artifacts not found at {c_path} or {le_path}. "
                "Please run `python run_pipeline.py` first to train the classifier."
            )

        with open(c_path, "rb") as f:
            classifier = pickle.load(f)
        with open(le_path, "rb") as f:
            label_encoder = pickle.load(f)

        engine = embedding_engine or EmbeddingEngine()
        logger.info(f"Loaded IntentClassifier with classes: {label_encoder.classes_.tolist()}")
        return cls(classifier=classifier, label_encoder=label_encoder, embedding_engine=engine)

    def save(
        self,
        classifier_path: Union[str, Path] = INTENT_CLASSIFIER_PATH,
        label_encoder_path: Union[str, Path] = LABEL_ENCODER_PATH,
    ) -> None:
        """Save classifier and label encoder to disk."""
        ensure_directories_exist()
        with open(classifier_path, "wb") as f:
            pickle.dump(self.classifier, f)
        with open(label_encoder_path, "wb") as f:
            pickle.dump(self.label_encoder, f)
        logger.info(f"Saved classifier to {classifier_path} and label encoder to {label_encoder_path}")

    def predict_single(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """Predict intent and confidence for a single input text string.

        Returns:
            Tuple of (predicted_intent, confidence_score, all_class_probabilities)
        """
        if self.embedding_engine is None:
            self.embedding_engine = EmbeddingEngine()

        emb = self.embedding_engine.embed_query(text)
        probs = self.classifier.predict_proba(emb)[0]
        max_idx = int(np.argmax(probs))
        confidence = float(probs[max_idx])
        predicted_intent = str(self.label_encoder.inverse_transform([max_idx])[0])

        all_probs = {
            cls_name: float(probs[idx])
            for idx, cls_name in enumerate(self.label_encoder.classes_)
        }
        return predicted_intent, confidence, all_probs

    def predict_batch(self, texts: List[str]) -> Tuple[List[str], List[float]]:
        """Batch predict intents and confidences for a list of texts."""
        if self.embedding_engine is None:
            self.embedding_engine = EmbeddingEngine()

        embs = self.embedding_engine.embed_texts(texts, show_progress_bar=False)
        probs = self.classifier.predict_proba(embs)
        max_indices = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1).tolist()
        predicted_intents = self.label_encoder.inverse_transform(max_indices).tolist()
        return predicted_intents, confidences


@timed_execution
def train_intent_classifier(
    train_csv: Path = TRAIN_CSV_PATH,
    val_csv: Path = VAL_CSV_PATH,
    save_models: bool = True,
) -> Tuple[IntentClassifier, Dict[str, float]]:
    """Train Logistic Regression intent classifier on embedded training dataset.

    Args:
        train_csv: Path to train.csv containing 'tweet' and 'intent'.
        val_csv: Path to validation.csv.
        save_models: Whether to serialize model artifacts to models/ directory.

    Returns:
        Tuple of (trained IntentClassifier instance, evaluation metrics dict)
    """
    seed_everything(RANDOM_SEED)
    ensure_directories_exist()

    if not train_csv.exists():
        raise FileNotFoundError(f"Training dataset not found at {train_csv}")

    logger.info(f"Loading training data from {train_csv}...")
    train_df = pd.read_csv(train_csv)
    logger.info(f"Train dataset: {len(train_df)} rows.")

    val_df = pd.read_csv(val_csv) if val_csv.exists() else None

    embedding_engine = EmbeddingEngine()

    logger.info("Computing embeddings for training corpus...")
    X_train = embedding_engine.embed_texts(train_df["tweet"].tolist(), show_progress_bar=True)
    
    label_encoder = LabelEncoder()
    # Fit encoder ensuring all canonical classes are included
    label_encoder.fit(INTENT_CLASSES)
    y_train = label_encoder.transform(train_df["intent"])

    logger.info(f"Training Logistic Regression (C=1.0, max_iter=1000, balanced weights)...")
    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)

    # Validation evaluation
    metrics: Dict[str, float] = {}
    if val_df is not None and len(val_df) > 0:
        logger.info(f"Evaluating classifier on validation set ({len(val_df)} rows)...")
        X_val = embedding_engine.embed_texts(val_df["tweet"].tolist(), show_progress_bar=False)
        y_val = label_encoder.transform(val_df["intent"])
        y_pred = clf.predict(X_val)

        acc = float(accuracy_score(y_val, y_pred))
        macro_f1 = float(f1_score(y_val, y_pred, average="macro"))
        metrics["val_accuracy"] = round(acc, 4)
        metrics["val_macro_f1"] = round(macro_f1, 4)
        logger.info(f"Validation Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}")
        logger.info("\n" + classification_report(y_val, y_pred, target_names=label_encoder.classes_))

    intent_classifier = IntentClassifier(
        classifier=clf,
        label_encoder=label_encoder,
        embedding_engine=embedding_engine,
    )

    if save_models:
        intent_classifier.save()

    return intent_classifier, metrics


if __name__ == "__main__":
    train_intent_classifier()
