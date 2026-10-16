"""
Unit tests for Textract block parsing and OCR hierarchy reconstruction.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/idp_pipeline")))
from textract_utils import parse_textract_blocks


class TestTextractParsing(unittest.TestCase):

    def setUp(self):
        # Simulated raw blocks returned by Amazon Textract
        self.mock_blocks = [
            {"BlockType": "PAGE", "Page": 1},
            {
                "BlockType": "LINE",
                "Page": 1,
                "Text": "INVOICE #INV-2026-001",
                "Confidence": 99.8,
            },
            {
                "BlockType": "LINE",
                "Page": 1,
                "Text": "Total Amount: $4,500.00",
                "Confidence": 98.5,
            },
            {"BlockType": "PAGE", "Page": 2},
            {
                "BlockType": "LINE",
                "Page": 2,
                "Text": "REMITTANCE INSTRUCTIONS",
                "Confidence": 99.1,
            },
            {
                "BlockType": "LINE",
                "Page": 2,
                "Text": "Bank: Chase Manhattan Bank",
                "Confidence": 97.4,
            },
            {
                "BlockType": "WORD",
                "Page": 2,
                "Text": "Bank:",
                "Confidence": 99.0,
            },
        ]

    def test_parse_textract_blocks(self):
        """Verify that LINE blocks are grouped by page with confidence and full text."""
        pages_map = parse_textract_blocks(self.mock_blocks)

        self.assertEqual(len(pages_map), 2)
        self.assertIn(1, pages_map)
        self.assertIn(2, pages_map)

        # Page 1 check
        p1 = pages_map[1]
        self.assertEqual(p1["page_number"], 1)
        self.assertEqual(len(p1["lines"]), 2)
        self.assertIn("INVOICE #INV-2026-001", p1["lines"])
        self.assertIn("Total Amount: $4,500.00", p1["lines"])
        self.assertAlmostEqual(p1["average_confidence"], 99.15, places=1)

        # Page 2 check
        p2 = pages_map[2]
        self.assertEqual(p2["page_number"], 2)
        self.assertEqual(len(p2["lines"]), 2)
        self.assertIn("REMITTANCE INSTRUCTIONS", p2["lines"])


if __name__ == "__main__":
    unittest.main()
