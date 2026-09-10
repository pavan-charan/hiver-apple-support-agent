"""Unit tests for the unified AppleSupportAgent inference interface."""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import unittest
from src.config import INTENT_CLASSES
from src.inference import AppleSupportAgent


class TestAppleSupportAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = AppleSupportAgent()

    def test_agent_single_prediction_contract(self):
        query = "My battery drains in 2 hours and device gets hot."
        res = self.agent.predict(query, k=3)

        self.assertIsInstance(res, dict)
        self.assertEqual(res["tweet"], query)
        self.assertIn(res["intent"], INTENT_CLASSES)
        self.assertTrue(0.0 <= res["confidence"] <= 1.0)
        self.assertIsInstance(res["generated_reply"], str)
        self.assertTrue(len(res["generated_reply"]) > 0)
        self.assertTrue(len(res["generated_reply"].split()) <= 70)
        self.assertIn(res["decision"], ["AUTO", "ESCALATE"])
        self.assertIsInstance(res["reason"], str)
        self.assertTrue(len(res["reason"]) > 0)
        self.assertIsInstance(res["retrieved_context"], list)
        self.assertEqual(len(res["retrieved_context"]), 3)

    def test_agent_escalation_routing_security(self):
        query = "Someone stole my phone and hacked my Apple ID password!"
        res = self.agent.predict(query)

        self.assertEqual(res["decision"], "ESCALATE")
        self.assertTrue(
            any(term in res["reason"].lower() for term in ["stolen", "hack", "account_security", "specialist"])
        )

    def test_agent_batch_prediction(self):
        queries = [
            "How do I update to iOS 17?",
            "I was overcharged on my iTunes receipt.",
        ]
        results = self.agent.predict_batch(queries, k=2)

        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIn("intent", r)
            self.assertIn("decision", r)
            self.assertIn("generated_reply", r)


if __name__ == "__main__":
    unittest.main()
