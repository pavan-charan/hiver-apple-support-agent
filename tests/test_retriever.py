"""Unit tests for FAISS indexing and RAGRetriever query search."""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import unittest
from src.build_index import build_faiss_index
from src.retrieve import RAGRetriever


class TestRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build_faiss_index()
        cls.retriever = RAGRetriever()

    def test_retriever_search_returns_top_k(self):
        query = "How to restore iPhone from iCloud backup?"
        results = self.retriever.search(query, k=5)

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 5)
        for item in results:
            self.assertIn("similarity", item)
            self.assertIn("tweet", item)
            self.assertIn("apple_reply", item)
            self.assertIn("intent", item)
            self.assertIsInstance(item["similarity"], float)
            self.assertTrue(len(item["apple_reply"]) > 0)

    def test_retriever_format_context(self):
        query = "My battery health dropped to 75%"
        results = self.retriever.search(query, k=3)
        context = self.retriever.format_context(results)

        self.assertIsInstance(context, str)
        self.assertIn("Example 1", context)
        self.assertIn("Apple Support Reply", context)


if __name__ == "__main__":
    unittest.main()
