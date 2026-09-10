"""Unified AppleSupportAgent inference module orchestrating embedding, classification, RAG retrieval, escalation, and reply generation."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.config import (
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    INTENT_CLASSIFIER_PATH,
    LABEL_ENCODER_PATH,
    OPENAI_MODEL_NAME,
    TOP_K_RETRIEVAL,
)
from src.embeddings import EmbeddingEngine
from src.escalation import EscalationEngine
from src.generate_reply import ReplyGenerator
from src.retrieve import RAGRetriever
from src.train_classifier import IntentClassifier
from src.utils import logger


class AppleSupportAgent:
    """Production-grade Apple Support Agent orchestrating end-to-end multi-stage inference."""

    def __init__(
        self,
        classifier_path: Union[str, Path] = INTENT_CLASSIFIER_PATH,
        label_encoder_path: Union[str, Path] = LABEL_ENCODER_PATH,
        faiss_index_path: Union[str, Path] = FAISS_INDEX_PATH,
        faiss_metadata_path: Union[str, Path] = FAISS_METADATA_PATH,
        openai_model: str = OPENAI_MODEL_NAME,
        api_key: Optional[str] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
    ) -> None:
        logger.info("Initializing AppleSupportAgent components...")
        self.embedding_engine = embedding_engine or EmbeddingEngine()

        self.classifier = IntentClassifier.load(
            classifier_path=classifier_path,
            label_encoder_path=label_encoder_path,
            embedding_engine=self.embedding_engine,
        )

        self.retriever = RAGRetriever(
            index_path=faiss_index_path,
            metadata_path=faiss_metadata_path,
            embedding_engine=self.embedding_engine,
        )

        self.escalation_engine = EscalationEngine()

        self.reply_generator = ReplyGenerator(
            model_name=openai_model,
            api_key=api_key,
        )

        logger.info("AppleSupportAgent initialized and ready for inference.")

    def predict(self, text: str, k: int = TOP_K_RETRIEVAL) -> Dict[str, Any]:
        """Perform full multi-stage inference for a single customer query.

        Args:
            text: Customer tweet / query string.
            k: Number of historical RAG exemplars to retrieve.

        Returns:
            Structured dictionary with intent, confidence, generated_reply, decision, reason, and retrieved_context.
        """
        clean_query = text.strip() if isinstance(text, str) else ""

        # 1. Intent Classification & Confidence
        intent, confidence, _ = self.classifier.predict_single(clean_query)

        # 2. Semantic RAG Retrieval
        retrieved_examples = self.retriever.search(clean_query, k=k)

        # 3. Escalation Decision & Reasoning
        escalation = self.escalation_engine.evaluate(clean_query, intent, confidence)

        # 4. Grounded Reply Generation
        generated_reply = self.reply_generator.generate(
            tweet=clean_query,
            intent=intent,
            retrieved_examples=retrieved_examples,
        )

        return {
            "tweet": clean_query,
            "intent": intent,
            "confidence": round(float(confidence), 4),
            "generated_reply": generated_reply,
            "decision": escalation["decision"],
            "reason": escalation["reason"],
            "retrieved_context": retrieved_examples,
        }

    def predict_batch(self, texts: List[str], k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        """Run batch inference over a list of texts."""
        return [self.predict(t, k=k) for t in texts]
