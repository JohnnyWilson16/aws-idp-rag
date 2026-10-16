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


if __name__ == "__main__":
    unittest.main()
