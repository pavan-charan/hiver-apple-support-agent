"""Unit tests for EscalationEngine decision rules and rationale generation."""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import unittest
from src.escalation import EscalationEngine


class TestEscalation(unittest.TestCase):
    def setUp(self):
        self.engine = EscalationEngine()

    def test_escalate_critical_intent_account_security(self):
        res = self.engine.evaluate(
            text="Can you help me with my account?",
            intent="account_security",
            confidence=0.95,
        )
        self.assertEqual(res["decision"], "ESCALATE")
        self.assertIn("account_security", res["reason"])

    def test_escalate_critical_intent_billing_refund(self):
        res = self.engine.evaluate(
            text="I need assistance with an invoice.",
            intent="billing_refund",
            confidence=0.92,
        )
        self.assertEqual(res["decision"], "ESCALATE")
        self.assertIn("billing_refund", res["reason"])

    def test_escalate_sensitive_keyword_stolen(self):
        res = self.engine.evaluate(
            text="My iPhone was stolen at the gym yesterday!",
            intent="battery_issue",
            confidence=0.90,
        )
        self.assertEqual(res["decision"], "ESCALATE")
        self.assertIn("stolen", res["reason"].lower())

    def test_escalate_sensitive_keyword_fraud(self):
        res = self.engine.evaluate(
            text="I suspect there is fraud on my Apple Card.",
            intent="app_store",
            confidence=0.88,
        )
        self.assertEqual(res["decision"], "ESCALATE")
        self.assertIn("fraud", res["reason"].lower())

    def test_escalate_low_confidence(self):
        res = self.engine.evaluate(
            text="The thing on my screen looks weird.",
            intent="software_update",
            confidence=0.45,
        )
        self.assertEqual(res["decision"], "ESCALATE")
        self.assertTrue("0.45" in res["reason"] or "Low intent classifier confidence" in res["reason"])

    def test_auto_handle_high_confidence(self):
        res = self.engine.evaluate(
            text="My iPhone battery health is at 76%.",
            intent="battery_issue",
            confidence=0.92,
        )
        self.assertEqual(res["decision"], "AUTO")
        self.assertTrue("battery issue" in res["reason"] or "High confidence" in res["reason"])

    def test_auto_handle_moderate_confidence_standard_intent(self):
        res = self.engine.evaluate(
            text="How do I setup Quick Start for my new phone?",
            intent="device_setup",
            confidence=0.75,
        )
        self.assertEqual(res["decision"], "AUTO")


if __name__ == "__main__":
    unittest.main()
