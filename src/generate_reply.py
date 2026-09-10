"""Reply generation module grounded in retrieved historical Apple Support examples using OpenAI SDK."""
import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.config import MAX_REPLY_WORDS, OPENAI_MODEL_NAME
from src.utils import logger

SYSTEM_PROMPT = """You are an official Apple Support AI Agent responding to customer support inquiries on Twitter/X.

Your replies MUST adhere strictly to the following rules:
1. Tone: Empathetic, calm, professional, concise, and helpful.
2. Grounding: Rely EXCLUSIVELY on the provided historical Apple Support examples and verified Apple support procedures.
3. Policy Integrity: NEVER hallucinate, invent unreleased policies, or promise unauthorized hardware replacements or refunds.
4. Word Limit: Strictly MAXIMUM 70 words. Keep it crisp and directly actionable.
5. Formatting: Output ONLY the direct customer-facing reply. Do not include greetings like "Dear customer" or meta commentary.
"""

USER_PROMPT_TEMPLATE = """Predicted Intent: {intent}

Customer Message:
"{tweet}"

Historical Apple Support Reference Examples (Grounded Context):
{context}

Draft an official Apple Support reply for the customer message that addresses their issue directly based on the reference context."""


class ReplyGenerator:
    """Generates customer-facing replies grounded in retrieved historical support threads."""

    def __init__(
        self,
        model_name: str = OPENAI_MODEL_NAME,
        api_key: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client: Optional[OpenAI] = None

        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
                logger.info(f"OpenAI client initialized with model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}. Falling back to grounded retrieval synthesis.")
        else:
            logger.info("OPENAI_API_KEY not set. Using grounded retrieval fallback engine.")

    def generate(
        self,
        tweet: str,
        intent: str,
        retrieved_examples: List[Dict[str, Any]],
        context_str: Optional[str] = None,
    ) -> str:
        """Generate grounded Apple Support response.

        Args:
            tweet: Inbound customer tweet.
            intent: Predicted intent category.
            retrieved_examples: Top-K retrieved exemplars from FAISS.
            context_str: Optional pre-formatted context string.

        Returns:
            Customer-facing reply string under 70 words.
        """
        if not context_str and retrieved_examples:
            formatted_examples = []
            for i, ex in enumerate(retrieved_examples, 1):
                formatted_examples.append(
                    f"Example {i}:\nQ: {ex['tweet']}\nA: {ex['apple_reply']}"
                )
            context_str = "\n\n".join(formatted_examples)
        elif not context_str:
            context_str = "No specific reference examples available."

        if self.client:
            try:
                user_content = USER_PROMPT_TEMPLATE.format(
                    intent=intent,
                    tweet=tweet,
                    context=context_str,
                )
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.3,
                    max_tokens=150,
                )
                reply = response.choices[0].message.content.strip()
                return self._enforce_word_limit(reply, MAX_REPLY_WORDS)
            except Exception as e:
                logger.warning(f"OpenAI API call failed: {e}. Utilizing grounded retrieval fallback.")

        # Offline grounded fallback: return highest-scoring retrieved historical resolution
        if retrieved_examples and len(retrieved_examples) > 0:
            best_reply = retrieved_examples[0]["apple_reply"]
            return self._enforce_word_limit(best_reply, MAX_REPLY_WORDS)

        return "We're here to help. Please restart your device, check Settings for available updates, and let us know if the issue persists."

    @staticmethod
    def _enforce_word_limit(text: str, max_words: int = MAX_REPLY_WORDS) -> str:
        """Enforce strict word limit."""
        words = text.split()
        if len(words) <= max_words:
            return text
        truncated = " ".join(words[:max_words])
        # Clean trailing punctuation
        truncated = re.sub(r"[,;:\-\s]+$", "", truncated)
        if not truncated.endswith((".", "!", "?")):
            truncated += "."
        return truncated
