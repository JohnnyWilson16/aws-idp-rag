"""
Rebuild the S3 vector store from the OCR output in extracted/*.json.

Use it to switch embedding models or to migrate an old deployment
(vector_store/index.faiss + metadata.pkl) to vector_store/store.npz.
The OCR step does not run again.

    MODEL_PROVIDER=bedrock python scripts/reindex_store.py --bucket <DocumentStorageBucketName> --table <DocumentMetadataTableName>
"""

import argparse
import json
import os
import sys

import boto3

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "idp_pipeline"))

from embedding_utils import embedding_manifest, get_batch_embeddings  # noqa: E402
from faiss_indexer import STORE_S3_KEY, add_chunks_to_index, serialize_store  # noqa: E402
from textract_utils import chunk_document_pages  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--table", help="DynamoDB table to update chunk counts in (optional)")
    args = parser.parse_args()

    s3 = boto3.client("s3")
    table = boto3.resource("dynamodb").Table(args.table) if args.table else None
    manifest = embedding_manifest()
    print(f"Rebuilding s3://{args.bucket}/{STORE_S3_KEY} with {manifest}")

    index, metadata = None, []
    keys = [
        o["Key"]
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=args.bucket, Prefix="extracted/")
        for o in page.get("Contents", [])
        if o["Key"].endswith(".json")
    ]
    for key in sorted(keys):
        payload = json.loads(s3.get_object(Bucket=args.bucket, Key=key)["Body"].read())
        doc_name = payload.get("filename") or os.path.basename(key)[: -len(".json")]
        chunks = chunk_document_pages(payload.get("pages", {}), doc_name=doc_name, max_chunk_chars=2000, overlap_chars=200)
        if not chunks:
            print(f"  skip {doc_name}: no text")
            continue
        texts = [f"Document: {c['doc_name']} Page: {c['page_number']}\n{c['text']}" for c in chunks]
        embeddings = get_batch_embeddings(texts, provider=manifest["embedding_provider"])
        index, metadata = add_chunks_to_index(index, metadata, chunks, embeddings, replace_doc=doc_name)
        if table:
            table.update_item(
                Key={"DocumentId": payload.get("document_id", doc_name)},
                UpdateExpression=(
                    "SET #st = :s, stage = :g, chunk_count = :c, vector_store_s3_key = :k REMOVE error_message"
                ),
                ExpressionAttributeNames={"#st": "status"},
                ExpressionAttributeValues={":s": "COMPLETED", ":g": "INDEXED", ":c": len(chunks), ":k": STORE_S3_KEY},
            )
        print(f"  {doc_name}: {len(chunks)} chunks")

    if index is None:
        sys.exit("No extracted documents found.")
    # ponytail: unconditional overwrite; run it while no uploads are in flight
    s3.put_object(Bucket=args.bucket, Key=STORE_S3_KEY, Body=serialize_store(index, metadata, manifest))
    print(f"Done: {index.ntotal} vectors")


if __name__ == "__main__":
    main()
