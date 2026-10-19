"""
Re-index all 11 documents with 384-dimensional Open-Source embeddings
(sentence-transformers/all-MiniLM-L6-v2) to Amazon S3 FAISS vector store.
Strictly Bedrock Prohibition Compliant.
"""

import io
import json
import os
import sys
import boto3
import numpy as np

# Ensure repo root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src", "idp_pipeline"))

from textract_utils import chunk_document_pages
from faiss_indexer import (
    create_empty_index,
    save_index_to_s3,
    add_chunks_to_index,
)
from embedding_utils import get_batch_embeddings

S3_BUCKET = "aws-idp-rag-storage-099256073383-us-east-1-dev"
DYNAMODB_TABLE = "DocumentMetadata-dev"
REGION = "us-east-1"

s3_client = boto3.client("s3", region_name=REGION)
dynamodb = boto3.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(DYNAMODB_TABLE)


def reindex_all():
    print("=" * 80)
    print("🚀 RE-INDEXING ALL DOCUMENTS WITH 384-DIMENSIONAL OPEN-SOURCE EMBEDDINGS")
    print(f"   Target S3 Bucket: {S3_BUCKET}")
    print("=" * 80)

    # 1. List all extracted OCR files in S3
    resp = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix="extracted/")
    items = resp.get("Contents", [])
    if not items:
        print("No extracted documents found in S3!")
        return

    # Initialize clean 384-d index
    index = create_empty_index(dim=384)
    metadata_list = []

    print(f"\nFound {len(items)} extracted JSON files in S3 extracted/ prefix.\n")

    for item in sorted(items, key=lambda x: x["Key"]):
        key = item["Key"]
        if not key.endswith(".json"):
            continue

        doc_name = key.replace("extracted/", "").replace(".json", "")
        print(f"--> Processing extracted document: {doc_name}...")

        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        payload = json.loads(obj["Body"].read().decode("utf-8"))
        pages_map = payload.get("pages", {})

        chunks = chunk_document_pages(
            pages_map=pages_map,
            doc_name=doc_name,
            max_chunk_chars=2000,
            overlap_chars=200,
        )

        chunk_texts = [
            f"Document: {c.get('doc_name', '')} Page: {c.get('page_number', 1)}\n{c['text']}"
            for c in chunks
        ]

        if not chunk_texts:
            print(f"    ⚠️ No text in {doc_name}")
            continue

        # Generate 384-d embeddings using open-source provider
        embeddings = get_batch_embeddings(
            chunk_texts,
            provider="huggingface",
            model_id="sentence-transformers/all-MiniLM-L6-v2",
        )

        # Add to FAISS index
        index, metadata_list = add_chunks_to_index(
            index=index,
            metadata_list=metadata_list,
            new_chunks=chunks,
            new_embeddings=embeddings,
        )

        # Update DynamoDB with exact chunk count and COMPLETED status
        table.update_item(
            Key={"DocumentId": doc_name},
            UpdateExpression="SET #st = :st, stage = :stage, chunk_count = :cc, page_count = :pc, vector_store_s3_key = :vkey",
            ExpressionAttributeNames={"#st": "status"},
            ExpressionAttributeValues={
                ":st": "COMPLETED",
                ":stage": "INDEXED",
                ":cc": str(len(chunks)),
                ":pc": str(len(pages_map)),
                ":vkey": "vector_store/index.faiss",
            },
        )
        print(f"    ✅ Indexed {len(chunks)} chunks across {len(pages_map)} pages.")

    # Save 384-d index to S3
    print(f"\nSaving updated 384-d FAISS index ({index.ntotal} vectors) to S3...")
    save_index_to_s3(index, metadata_list, S3_BUCKET, s3_client=s3_client)
    print("🎉 RE-INDEXING COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    reindex_all()
