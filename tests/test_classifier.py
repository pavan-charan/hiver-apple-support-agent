"""Unit tests for IntentClassifier training, inference, and serialization."""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import unittest
import numpy as np
from src.config import INTENT_CLASSES, INTENT_CLASSIFIER_PATH, LABEL_ENCODER_PATH
from src.embeddings import EmbeddingEngine
from src.train_classifier import IntentClassifier, train_intent_classifier


class TestClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier, cls.metrics = train_intent_classifier()

    def test_classifier_prediction_output_structure(self):
        text = "My iPhone battery is draining extremely fast after updating."
        intent, conf, all_probs = self.classifier.predict_single(text)

        self.assertIsInstance(intent, str)
        self.assertIn(intent, INTENT_CLASSES)
        self.assertTrue(0.0 <= conf <= 1.0)
        self.assertIsInstance(all_probs, dict)
        self.assertEqual(len(all_probs), len(INTENT_CLASSES))
        self.assertAlmostEqual(sum(all_probs.values()), 1.0, places=2)

    def test_classifier_batch_prediction(self):
        texts = [
            "How do I reset my Apple ID password?",
            "My phone is not charging with lightning cable.",
        ]
        intents, confs = self.classifier.predict_batch(texts)
        self.assertEqual(len(intents), 2)
        self.assertEqual(len(confs), 2)
        for intent, conf in zip(intents, confs):
            self.assertIn(intent, INTENT_CLASSES)
            self.assertTrue(0.0 <= conf <= 1.0)

    def test_classifier_serialization(self):
        c_path = PROJECT_ROOT / "models" / "temp_test_clf.pkl"
        le_path = PROJECT_ROOT / "models" / "temp_test_le.pkl"
        self.classifier.save(classifier_path=c_path, label_encoder_path=le_path)

        self.assertTrue(c_path.exists())
        self.assertTrue(le_path.exists())

        loaded = IntentClassifier.load(
            classifier_path=c_path,
            label_encoder_path=le_path,
            embedding_engine=self.classifier.embedding_engine,
        )
        self.assertIsNotNone(loaded)
        intent, conf, _ = loaded.predict_single("I need a refund for duplicate subscription")
        self.assertIn(intent, INTENT_CLASSES)

        # Cleanup
        if c_path.exists():
            c_path.unlink()
        if le_path.exists():
            le_path.unlink()


if __name__ == "__main__":
    unittest.main()
