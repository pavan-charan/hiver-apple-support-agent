"""Standard unittest-compatible test runner for Apple Support AI Agent."""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import unittest
from src.config import INTENT_CLASSES
from src.escalation import EscalationEngine
from src.inference import AppleSupportAgent
from src.retrieve import RAGRetriever
from src.train_classifier import IntentClassifier, train_intent_classifier
from src.build_index import build_faiss_index


class TestEscalationEngine(unittest.TestCase):
    def setUp(self):
        self.engine = EscalationEngine()

    def test_escalate_critical_intent(self):
        res = self.engine.evaluate("Help with my account", "account_security", 0.95)
        self.assertEqual(res["decision"], "ESCALATE")
        self.assertIn("account_security", res["reason"])

    def test_escalate_billing(self):
        res = self.engine.evaluate("Refund my money", "billing_refund", 0.90)
        self.assertEqual(res["decision"], "ESCALATE")

    def test_escalate_stolen_keyword(self):
        res = self.engine.evaluate("My iPhone was stolen at the gym", "battery_issue", 0.92)
        self.assertEqual(res["decision"], "ESCALATE")
        self.assertIn("stolen", res["reason"].lower())

    def test_escalate_low_confidence(self):
        res = self.engine.evaluate("Unknown issue", "software_update", 0.45)
        self.assertEqual(res["decision"], "ESCALATE")

    def test_auto_handle_high_confidence(self):
        res = self.engine.evaluate("My battery health is 75%", "battery_issue", 0.94)
        self.assertEqual(res["decision"], "AUTO")


class TestPipelineAndAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.config import FAISS_INDEX_PATH, INTENT_CLASSIFIER_PATH
        if not INTENT_CLASSIFIER_PATH.exists():
            train_intent_classifier()
        if not FAISS_INDEX_PATH.exists():
            build_faiss_index()
        cls.agent = AppleSupportAgent()

    def test_classifier_prediction(self):
        intent, conf, all_probs = self.agent.classifier.predict_single("My battery drains in 1 hour.")
        self.assertIn(intent, INTENT_CLASSES)
        self.assertTrue(0.0 <= conf <= 1.0)
        self.assertEqual(len(all_probs), len(INTENT_CLASSES))

    def test_retriever_top_k(self):
        results = self.agent.retriever.search("How to sync photos to iCloud?", k=5)
        self.assertEqual(len(results), 5)
        self.assertIn("apple_reply", results[0])
        self.assertIn("similarity", results[0])

    def test_agent_predict_contract(self):
        res = self.agent.predict("My iPad screen is frozen during update.", k=3)
        self.assertIn("tweet", res)
        self.assertIn("intent", res)
        self.assertIn("confidence", res)
        self.assertIn("generated_reply", res)
        self.assertIn("decision", res)
        self.assertIn("reason", res)
        self.assertIn(res["decision"], ["AUTO", "ESCALATE"])
        self.assertTrue(len(res["generated_reply"].split()) <= 70)


if __name__ == "__main__":
    unittest.main(verbosity=2)
