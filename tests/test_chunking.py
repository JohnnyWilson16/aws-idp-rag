"""
Unit tests for document chunking with strict page-level metadata preservation.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/idp_pipeline")))
from textract_utils import chunk_document_pages


class TestDocumentChunking(unittest.TestCase):

    def setUp(self):
        self.doc_name = "test_contract.pdf"
        self.sample_pages_map = {
            1: {
                "page_number": 1,
                "lines": [
                    "PARTIES AND EFFECTIVE DATE",
                    "This agreement is made between Acme Corp and Beta LLC.",
                    "Term of agreement is 24 months.",
                ],
                "full_text": (
                    "PARTIES AND EFFECTIVE DATE\n"
                    "This agreement is made between Acme Corp and Beta LLC.\n"
                    "Term of agreement is 24 months."
                ),
            },
            2: {
                "page_number": 2,
                "lines": [
                    "SLA AND GUARANTEES",
                    "Provider guarantees 99.9% monthly uptime.",
                    "Latency target is less than 200 milliseconds.",
                ],
                "full_text": (
                    "SLA AND GUARANTEES\n"
                    "Provider guarantees 99.9% monthly uptime.\n"
                    "Latency target is less than 200 milliseconds."
                ),
            },
            3: {
                "page_number": 3,
                "lines": [
                    "LIMITATION OF LIABILITY",
                    "In no event shall aggregate liability exceed $500,000 USD.",
                    "Neither party is liable for indirect or consequential damages.",
                ],
                "full_text": (
                    "LIMITATION OF LIABILITY\n"
                    "In no event shall aggregate liability exceed $500,000 USD.\n"
                    "Neither party is liable for indirect or consequential damages."
                ),
            },
        }

    def test_chunk_page_preservation(self):
        """Verify that every generated chunk preserves the correct page number and document name."""
        chunks = chunk_document_pages(
            pages_map=self.sample_pages_map,
            doc_name=self.doc_name,
            max_chunk_chars=1000,
            overlap_chars=100,
        )

        self.assertEqual(len(chunks), 3)

        # Page 1 verification
        self.assertEqual(chunks[0]["doc_name"], self.doc_name)
        self.assertEqual(chunks[0]["page_number"], 1)
        self.assertIn("Acme Corp", chunks[0]["text"])

        # Page 2 verification
        self.assertEqual(chunks[1]["doc_name"], self.doc_name)
        self.assertEqual(chunks[1]["page_number"], 2)
        self.assertIn("99.9%", chunks[1]["text"])

        # Page 3 verification
        self.assertEqual(chunks[2]["doc_name"], self.doc_name)
        self.assertEqual(chunks[2]["page_number"], 3)
        self.assertIn("$500,000", chunks[2]["text"])

    def test_multi_chunk_per_page(self):
        """Verify that large text on a single page is split into multiple chunks with identical page attribution."""
        long_text = "Important clause content. " * 100  # ~2700 chars
        long_page_map = {
            1: {
                "page_number": 1,
                "lines": [long_text],
                "full_text": long_text,
            }
        }

        chunks = chunk_document_pages(
            pages_map=long_page_map,
            doc_name="long_doc.pdf",
            max_chunk_chars=1000,
            overlap_chars=100,
        )

        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertEqual(c["page_number"], 1)
            self.assertEqual(c["doc_name"], "long_doc.pdf")
            self.assertLessEqual(len(c["text"]), 1100)


if __name__ == "__main__":
    unittest.main()
