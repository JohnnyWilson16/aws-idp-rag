"""
Comprehensive Evaluation & Test Runner for desktop/testdata_aws_rag.
Processes all 7 sample documents (including multi-page scanned PDFs),
generates page-preserving vector embeddings, indexes them into FAISS,
syncs to the live AWS Cloud S3 & DynamoDB backend, and queries the live API Gateway.
"""

import io
import json
import os
import sys
import time
from typing import Any, Dict, List
import boto3
import numpy as np
import requests
from pypdf import PdfReader
from PIL import Image

try:
    from ocrmac import ocrmac
    OCRMAC_AVAILABLE = True
except ImportError:
    OCRMAC_AVAILABLE = False

# Add repo root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.idp_pipeline.textract_utils import chunk_document_pages
from src.idp_pipeline.faiss_indexer import (
    create_empty_index,
    load_index_from_s3,
    save_index_to_s3,
    add_chunks_to_index,
)
from src.idp_pipeline.embedding_utils import get_batch_embeddings

API_GATEWAY_URL = os.environ.get(
    "API_GATEWAY_URL",
    "https://lq2w2n8jae.execute-api.us-east-1.amazonaws.com/dev",
)
S3_BUCKET = os.environ.get("DOCUMENT_BUCKET_NAME", "aws-idp-rag-storage-099256073383-us-east-1-dev")
DYNAMODB_TABLE = os.environ.get("DYNAMODB_TABLE_NAME", "DocumentMetadata-dev")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

s3_client = boto3.client("s3", region_name=AWS_REGION)
dynamodb_resource = boto3.resource("dynamodb", region_name=AWS_REGION)
metadata_table = dynamodb_resource.Table(DYNAMODB_TABLE)


def extract_pages(pdf_path: str) -> Dict[str, Dict[str, Any]]:
    """Extract pages using text streams or ocrmac for scanned pages."""
    reader = PdfReader(pdf_path)
    pages_map = {}

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        text = (page.extract_text() or "").strip()

        if not text and len(page.images) > 0 and OCRMAC_AVAILABLE:
            # Scanned image OCR via Apple Vision
            img_data = page.images[0].data
            img = Image.open(io.BytesIO(img_data))
            annotations = ocrmac.OCR(img).recognize()
            lines = [{"text": a[0], "confidence": float(a[1]) * 100} for a in annotations]
            text = "\n".join([a[0] for a in annotations])
        else:
            lines = [{"text": l, "confidence": 100.0} for l in text.splitlines() if l.strip()]

        pages_map[str(page_num)] = {
            "page_number": page_num,
            "lines": lines,
            "full_text": text,
            "line_count": len(lines),
            "confidence_avg": 100.0,
        }

    return pages_map


def process_dataset(data_dir: str):
    print("=" * 80)
    print(f"📂 INGESTING SAMPLE TEST DATA FROM: {data_dir}")
    print("=" * 80)

    pdf_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pdf")])
    if not pdf_files:
        print(f"No PDF files found in {data_dir}")
        return

    # Initialize fresh index to ensure clean state with latest embeddings
    index = create_empty_index(dim=1024)
    metadata_list = []
    all_chunks: List[Dict[str, Any]] = []

    for filename in pdf_files:
        filepath = os.path.join(data_dir, filename)
        print(f"\n--> Processing '{filename}'...")
        pages_map = extract_pages(filepath)
        page_count = len(pages_map)
        total_chars = sum(len(p["full_text"]) for p in pages_map.values())
        print(f"    Extracted {page_count} pages ({total_chars} total characters)")

        # Chunk document
        chunks = chunk_document_pages(
            pages_map=pages_map,
            doc_name=filename,
            max_chunk_chars=1800,
            overlap_chars=200,
        )
        print(f"    Created {len(chunks)} page-preserving chunks")
        all_chunks.extend(chunks)

        # Upload raw PDF to S3 uploads
        with open(filepath, "rb") as f:
            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=f"uploads/{filename}",
                Body=f.read(),
                ContentType="application/pdf",
            )

        # Upload extracted JSON to S3 extracted/
        extracted_payload = {
            "document_id": filename,
            "filename": filename,
            "page_count": page_count,
            "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pages": pages_map,
        }
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=f"extracted/{filename}.json",
            Body=json.dumps(extracted_payload, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

        # Update DynamoDB status to COMPLETED
        metadata_table.put_item(
            Item={
                "DocumentId": filename,
                "filename": filename,
                "s3_bucket": S3_BUCKET,
                "s3_key": f"uploads/{filename}",
                "status": "COMPLETED",
                "stage": "INDEXED",
                "page_count": page_count,
                "chunk_count": len(chunks),
                "extracted_s3_key": f"extracted/{filename}.json",
                "vector_store_s3_key": "vector_store/index.faiss",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )

    print(f"\n--> Computing embeddings for all {len(all_chunks)} chunks...")
    chunk_texts = [f"Document: {c['doc_name']} Page: {c['page_number']}\n{c['text']}" for c in all_chunks]
    embeddings = get_batch_embeddings(chunk_texts, provider="bedrock")
    index, metadata_list = add_chunks_to_index(index, metadata_list, all_chunks, embeddings)

    # Persist updated FAISS index to S3
    save_index_to_s3(index, metadata_list, bucket_name=S3_BUCKET, s3_client=s3_client)
    print(f"✅ Successfully indexed {index.ntotal} total vectors into S3: s3://{S3_BUCKET}/vector_store/")


def run_test_queries():
    test_queries = [
        {
            "category": "Bank Statement (Digital)",
            "expected_doc": "sample-bank-statement.pdf",
            "question": "What is the opening balance and account holder on sample-bank-statement.pdf?",
        },
        {
            "category": "Invoice (Digital)",
            "expected_doc": "sample-invoice.pdf",
            "question": "What is the invoice number, total amount due, and vendor name on sample-invoice.pdf?",
        },
        {
            "category": "Receipt (Digital)",
            "expected_doc": "sample-receipt.pdf",
            "question": "What was ordered and what was the total bill at Corner Cafe in sample-receipt.pdf?",
        },
        {
            "category": "Rental Agreement (Scanned)",
            "expected_doc": "sample-rental-agreement-scan.pdf",
            "question": "Who is the licensor and licensee, and what is the rental amount in sample-rental-agreement-scan.pdf?",
        },
        {
            "category": "Scanned Letter (Scanned)",
            "expected_doc": "sample-scanned-letter.pdf",
            "question": "What is the subject of the letter from Brightwater Housing Society in sample-scanned-letter.pdf?",
        },
        {
            "category": "Notice of Planned Maintenance (Scanned)",
            "expected_doc": "sample-scanned-page.pdf",
            "question": "What is the notice of planned maintenance about in sample-scanned-page.pdf?",
        },
        {
            "category": "Annual Library Report (Scanned Multi-Page)",
            "expected_doc": "sample-scanned-report-3-pages.pdf",
            "question": "How many new items did Larkfield Community Library purchase according to sample-scanned-report-3-pages.pdf?",
        },
    ]

    print("\n" + "=" * 80)
    print("🔍 TESTING LIVE AWS RAG API WITH QUERIES ACROSS ALL TEST DOCUMENTS")
    print(f"   Target Endpoint: {API_GATEWAY_URL}/query")
    print("=" * 80)

    passed = 0
    for idx, q in enumerate(test_queries, 1):
        print(f"\n❓ [Test {idx}/{len(test_queries)} - {q['category']}]")
        print(f"   Question: \"{q['question']}\"")

        payload = {"question": q["question"], "top_k": 2}
        try:
            resp = requests.post(f"{API_GATEWAY_URL}/query", json=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                print("   💬 Answer:")
                answer_lines = data.get("answer", "").strip().splitlines()
                for line in answer_lines[:5]:
                    print(f"      {line}")
                if len(answer_lines) > 5:
                    print("      ...")

                sources = data.get("sources", [])
                if sources:
                    top_source = sources[0]
                    is_match = top_source.get("doc_name") == q["expected_doc"]
                    if is_match:
                        passed += 1
                        status = "✅ MATCH"
                    else:
                        status = "⚠️ DIFFERENT DOC"
                    print(f"   📌 Primary Citation: [{top_source.get('doc_name')} - Page {top_source.get('page_number')}] (Score: {top_source.get('relevance_score'):.4f}) -> {status}")
            else:
                print(f"   ❌ HTTP Error {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"   ❌ Query Failed: {e}")

    print("\n" + "=" * 80)
    print(f"🏁 RESULT: {passed}/{len(test_queries)} TESTS RETRIEVED EXACT TARGET DOCUMENT ON CLOUD API")
    print("=" * 80)


if __name__ == "__main__":
    test_dir = os.path.expanduser("~/Desktop/testdata_aws_rag")
    process_dataset(test_dir)
    time.sleep(2)
    run_test_queries()
