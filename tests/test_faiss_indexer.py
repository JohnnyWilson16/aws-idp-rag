"""
Unit tests for FAISS vector indexer, cosine similarity ranking, and search.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/idp_pipeline")))
from faiss_indexer import (
    create_empty_index,
    add_chunks_to_index,
    search_index,
)


class TestFAISSIndexer(unittest.TestCase):

    def setUp(self):
        self.dim = 384
        self.index = create_empty_index(self.dim)
        self.metadata_list = []

        # Create two distinct test chunks and artificial orthogonal vectors
        self.chunks = [
            {
                "chunk_id": "doc1_p1_c0",
                "doc_name": "agreement.pdf",
                "page_number": 1,
                "text": "The liability cap is $500,000 USD.",
            },
            {
                "chunk_id": "doc2_p2_c0",
                "doc_name": "invoice.pdf",
                "page_number": 2,
                "text": "Payment terms are Net 30 days.",
            },
        ]

        # Orthogonal unit vectors
        v1 = np.zeros(self.dim, dtype=np.float32)
        v1[0] = 1.0  # Align with feature 0

        v2 = np.zeros(self.dim, dtype=np.float32)
        v2[1] = 1.0  # Align with feature 1

        self.embeddings = np.vstack([v1, v2])

    def test_add_and_search(self):
        """Verify vector addition and cosine similarity nearest neighbor retrieval."""
        idx, meta = add_chunks_to_index(
            self.index, self.metadata_list, self.chunks, self.embeddings
        )

        self.assertEqual(idx.ntotal, 2)
        self.assertEqual(len(meta), 2)

        # Query aligned with v1 (agreement.pdf)
        query_v1 = np.zeros(self.dim, dtype=np.float32)
        query_v1[0] = 1.0

        results = search_index(idx, meta, query_v1, top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["doc_name"], "agreement.pdf")
        self.assertEqual(results[0]["page_number"], 1)
        self.assertAlmostEqual(results[0]["score"], 1.0, places=2)

        # Query aligned with v2 (invoice.pdf)
        query_v2 = np.zeros(self.dim, dtype=np.float32)
        query_v2[1] = 1.0

        results2 = search_index(idx, meta, query_v2, top_k=1)
        self.assertEqual(results2[0]["doc_name"], "invoice.pdf")
        self.assertEqual(results2[0]["page_number"], 2)

    def test_bedrock_1024_dimension_indexing(self):
        """Verify that index dynamically adapts to 1024-dimensional Bedrock Titan vectors."""
        dim_1024 = 1024
        chunks = [
            {"chunk_id": "c1", "doc_name": "contract.pdf", "page_number": 1, "text": "Clause 1"},
            {"chunk_id": "c2", "doc_name": "contract.pdf", "page_number": 2, "text": "Clause 2"},
        ]
        rng = np.random.RandomState(42)
        raw_vecs = rng.randn(2, dim_1024).astype(np.float32)

        idx, meta = add_chunks_to_index(None, [], chunks, raw_vecs)
        self.assertEqual(idx.d, 1024)
        self.assertEqual(idx.ntotal, 2)

        query = raw_vecs[0]
        results = search_index(idx, meta, query, top_k=1)
        self.assertEqual(results[0]["chunk_id"], "c1")
        self.assertAlmostEqual(results[0]["score"], 1.0, places=2)

    def test_dimension_adaptation(self):
        """Verify that vectors adapt to target dimension without crashing FAISS search."""
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/query_api")))
        from embedding_client import _adapt_dimension

        # 384-d vector adapted to 1024-d index
        vec_384 = np.ones(384, dtype=np.float32)
        adapted_1024 = _adapt_dimension(vec_384, target_dim=1024)
        self.assertEqual(adapted_1024.shape[0], 1024)
        self.assertAlmostEqual(np.linalg.norm(adapted_1024), 1.0, places=4)

        # 1024-d vector adapted to 384-d index
        vec_1024 = np.ones(1024, dtype=np.float32)
        adapted_384 = _adapt_dimension(vec_1024, target_dim=384)
        self.assertEqual(adapted_384.shape[0], 384)
        self.assertAlmostEqual(np.linalg.norm(adapted_384), 1.0, places=4)

    def test_hybrid_retrieval(self):
        """Verify Dense (FAISS) + Sparse (Lexical) Hybrid Search with exact document matching."""
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/query_api")))
        from app import _retrieve_top_k

        idx, meta = add_chunks_to_index(
            self.index, self.metadata_list, self.chunks, self.embeddings
        )

        # Lexical keyword match on "Net 30 days" should rank invoice.pdf top
        results = _retrieve_top_k(
            index=idx,
            metadata_list=meta,
            query_vector=None,
            query_text="What are the Net 30 payment terms?",
            top_k=1,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["doc_name"], "invoice.pdf")
        self.assertEqual(results[0]["page_number"], 2)


if __name__ == "__main__":
    unittest.main()

