"""Hybrid rule + confidence escalation decision engine for Apple Support tickets."""
import re
from typing import Dict, List, Optional

from src.config import (
    CONFIDENCE_THRESHOLD_AUTO,
    CONFIDENCE_THRESHOLD_ESCALATE,
    ESCALATE_INTENTS,
    ESCALATE_KEYWORDS,
)
from src.utils import logger


class EscalationEngine:
    """Evaluates customer messages, predicted intent, and confidence to make deterministic routing decisions."""

    def __init__(
        self,
        escalate_intents: Optional[List[str]] = None,
        escalate_keywords: Optional[List[str]] = None,
        auto_threshold: float = CONFIDENCE_THRESHOLD_AUTO,
        escalate_threshold: float = CONFIDENCE_THRESHOLD_ESCALATE,
    ) -> None:
        self.escalate_intents = set(escalate_intents or ESCALATE_INTENTS)
        keywords = escalate_keywords or ESCALATE_KEYWORDS
        pattern_str = r"\b(" + "|".join(re.escape(kw) for kw in keywords) + r")\b"
        self.keyword_regex = re.compile(pattern_str, re.IGNORECASE)
        self.auto_threshold = auto_threshold
        self.escalate_threshold = escalate_threshold

    def evaluate(self, text: str, intent: str, confidence: float) -> Dict[str, str]:
        """Determine routing decision and provide clear human-interpretable rationale.

        Decision Rules:
        1. Sensitive Intents: If intent is account_security or billing_refund -> ESCALATE
        2. High-Risk Keywords: If text mentions stolen/lost/fraud/hacked/unauthorized -> ESCALATE
        3. Model Uncertainty: If confidence < 0.60 -> ESCALATE
        4. High-Confidence Auto-Handle: If confidence >= 0.85 and no triggers -> AUTO
        5. Moderate Confidence: If 0.60 <= confidence < 0.85 and safe standard intent -> AUTO

        Returns:
            Dict with 'decision' ('AUTO' or 'ESCALATE') and 'reason'.
        """
        # Rule 1: High-risk intent check
        if intent in self.escalate_intents:
            return {
                "decision": "ESCALATE",
                "reason": f"High-risk intent '{intent}' requires specialist intervention",
            }

        # Rule 2: Keyword match for lost/stolen/fraud/breach
        kw_match = self.keyword_regex.search(text)
        if kw_match:
            matched_term = kw_match.group(0)
            return {
                "decision": "ESCALATE",
                "reason": f"Sensitive keyword '{matched_term}' detected in customer message",
            }

        # Rule 3: Low model confidence threshold (< 0.60)
        if confidence < self.escalate_threshold:
            return {
                "decision": "ESCALATE",
                "reason": f"Low intent classifier confidence ({confidence:.2f} < {self.escalate_threshold})",
            }

        # Rule 4: High confidence auto-handle (>= 0.85)
        if confidence >= self.auto_threshold:
            readable_intent = intent.replace("_", " ")
            return {
                "decision": "AUTO",
                "reason": f"High confidence {readable_intent} ({confidence:.2f}) with standard resolution path",
            }

        # Rule 5: Moderate confidence (0.60 <= confidence < 0.85)
        readable_intent = intent.replace("_", " ")
        return {
            "decision": "AUTO",
            "reason": f"Standard {readable_intent} query ({confidence:.2f}) routed to automated guidance",
        }
