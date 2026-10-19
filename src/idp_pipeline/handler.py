"""
AWS Lambda Handler for Intelligent Document Processing (IDP) Pipeline.

Orchestrated by AWS Step Functions to perform:
1. Start Textract Asynchronous OCR & Initialize DynamoDB record
2. Poll Textract Job Status
3. Extract text preserving page numbers, store raw JSON in S3
4. Chunk text, compute embeddings, and upsert into S3-backed FAISS vector store
5. Update DynamoDB status to COMPLETED or FAILED
"""

from datetime import datetime, timezone
import json
import logging
import os
import urllib.parse
from typing import Any, Dict
import boto3

from textract_utils import (
    start_document_text_detection,
    check_document_text_detection_status,
    fetch_all_textract_blocks,
    parse_textract_blocks,
    chunk_document_pages,
)
from embedding_utils import get_batch_embeddings
from faiss_indexer import (
    load_index_from_s3,
    save_index_to_s3,
    add_chunks_to_index,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME", "DocumentMetadataTable")
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(DYNAMODB_TABLE_NAME)
s3_client = boto3.client("s3")


def _now_iso() -> str:
    """Current UTC ISO 8601 timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _sanitize_doc_id(s3_key: str) -> str:
    """Derive clean document ID from S3 key."""
    basename = os.path.basename(s3_key)
    safe_name = urllib.parse.unquote_plus(basename)
    return safe_name.replace(" ", "_")


def _extract_pages_fallback(s3_bucket: str, s3_key: str) -> Dict[str, Dict[str, Any]]:
    """
    Fallback parser using pypdf if Textract is unavailable or unsubscribed.
    """
    import io
    from pypdf import PdfReader

    logger.info(f"Using pypdf fallback extraction for s3://{s3_bucket}/{s3_key}")
    obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
    pdf_bytes = io.BytesIO(obj["Body"].read())
    reader = PdfReader(pdf_bytes)
    pages_map: Dict[str, Dict[str, Any]] = {}
    for idx, page in enumerate(reader.pages):
        page_num = str(idx + 1)
        text = page.extract_text() or ""
        lines = [{"text": l, "confidence": 100.0} for l in text.splitlines() if l.strip()]
        pages_map[page_num] = {
            "page_number": idx + 1,
            "lines": lines,
            "full_text": text.strip(),
            "line_count": len(lines),
            "confidence_avg": 100.0,
        }
    return pages_map


def handle_start(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 1: Receive document upload event, register in DynamoDB,
    and start asynchronous Textract OCR (with pypdf fallback).
    """
    s3_bucket = event.get("s3_bucket")
    s3_key = event.get("s3_key")

    # If triggered directly by S3 Event / EventBridge
    if not s3_bucket or not s3_key:
        if "detail" in event and "bucket" in event["detail"]:
            s3_bucket = event["detail"]["bucket"]["name"]
            s3_key = urllib.parse.unquote_plus(event["detail"]["object"]["key"])
        elif "Records" in event and len(event["Records"]) > 0:
            s3_record = event["Records"][0]["s3"]
            s3_bucket = s3_record["bucket"]["name"]
            s3_key = urllib.parse.unquote_plus(s3_record["object"]["key"])

    if not s3_bucket or not s3_key:
        raise ValueError(f"Missing S3 bucket or key in event: {json.dumps(event)}")

    filename = os.path.basename(s3_key)
    doc_id = event.get("document_id") or _sanitize_doc_id(s3_key)
    now = _now_iso()

    logger.info(f"Registering document {doc_id} in DynamoDB table {DYNAMODB_TABLE_NAME}")
    table.put_item(
        Item={
            "DocumentId": doc_id,
            "filename": filename,
            "s3_bucket": s3_bucket,
            "s3_key": s3_key,
            "status": "PROCESSING",
            "stage": "OCR_SUBMITTED",
            "created_at": now,
            "updated_at": now,
        }
    )

    # Attempt asynchronous Textract job, falling back to pypdf on permission/subscription failure
    job_id = None
    try:
        job_id = start_document_text_detection(s3_bucket, s3_key)
        table.update_item(
            Key={"DocumentId": doc_id},
            UpdateExpression="SET job_id = :jid, updated_at = :u",
            ExpressionAttributeValues={":jid": job_id, ":u": _now_iso()},
        )
    except Exception as e:
        logger.warning(f"Textract start failed ({e}). Falling back to native PDF extraction.")
        pages_map = _extract_pages_fallback(s3_bucket, s3_key)
        extracted_s3_key = f"extracted/{doc_id}.json"
        extracted_payload = {
            "document_id": doc_id,
            "filename": filename,
            "page_count": len(pages_map),
            "extracted_at": _now_iso(),
            "pages": pages_map,
            "source": "native_pdf_fallback",
        }
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=extracted_s3_key,
            Body=json.dumps(extracted_payload, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        job_id = f"FALLBACK_{doc_id}"
        table.update_item(
            Key={"DocumentId": doc_id},
            UpdateExpression="SET job_id = :jid, stage = :st, updated_at = :u",
            ExpressionAttributeValues={":jid": job_id, ":st": "FALLBACK_PARSED", ":u": _now_iso()},
        )


    return {
        "document_id": doc_id,
        "s3_bucket": s3_bucket,
        "s3_key": s3_key,
        "filename": filename,
        "job_id": job_id,
        "status": "PROCESSING",
    }


def handle_check_status(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 2: Check Textract job status.
    """
    job_id = event.get("job_id")
    if not job_id:
        raise ValueError("Missing job_id in check_status event")

    if job_id.startswith("FALLBACK_"):
        result = dict(event)
        result["job_status"] = "SUCCEEDED"
        result["status_message"] = "Extracted via resilient fallback PDF parser"
        return result

    status_data = check_document_text_detection_status(job_id)
    result = dict(event)
    result["job_status"] = status_data["job_status"]
    result["status_message"] = status_data["status_message"]
    return result


def handle_process_and_index(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 3: Fetch OCR blocks, store extracted JSON in S3, chunk text,
    compute embeddings, upsert into S3 FAISS index, and mark COMPLETED.
    """
    job_id = event["job_id"]
    doc_id = event["document_id"]
    s3_bucket = event["s3_bucket"]
    filename = event.get("filename", doc_id)

    logger.info(f"Processing output for document {doc_id} (job {job_id})")

    extracted_s3_key = f"extracted/{doc_id}.json"
    if job_id.startswith("FALLBACK_"):
        logger.info(f"Loading pre-extracted fallback pages from s3://{s3_bucket}/{extracted_s3_key}")
        obj = s3_client.get_object(Bucket=s3_bucket, Key=extracted_s3_key)
        payload = json.loads(obj["Body"].read().decode("utf-8"))
        pages_map = payload.get("pages", {})
        page_count = len(pages_map)
    else:
        # 1. Fetch all Textract blocks with pagination
        blocks = fetch_all_textract_blocks(job_id)
        pages_map = parse_textract_blocks(blocks)
        page_count = len(pages_map)

        # 2. Store raw extracted OCR output as JSON in S3
        extracted_payload = {
            "document_id": doc_id,
            "filename": filename,
            "page_count": page_count,
            "extracted_at": _now_iso(),
            "pages": pages_map,
        }
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=extracted_s3_key,
            Body=json.dumps(extracted_payload, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info(f"Saved raw extracted OCR to s3://{s3_bucket}/{extracted_s3_key}")

    # 3. Chunk text across pages preserving metadata
    chunks = chunk_document_pages(
        pages_map=pages_map,
        doc_name=filename,
        max_chunk_chars=2000,  # ~500 tokens
        overlap_chars=200,     # ~50 tokens
    )
    chunk_count = len(chunks)

    # 4. Generate embeddings via Amazon Bedrock (or Hugging Face fallback)
    provider = event.get("provider") or os.environ.get("MODEL_PROVIDER", "bedrock").lower()
    model_id = event.get("embedding_model") or os.environ.get("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    chunk_texts = [c["text"] for c in chunks]

    if chunk_texts:
        logger.info(f"Generating embeddings for {len(chunk_texts)} chunks using provider: {provider} ({model_id})...")
        embeddings = get_batch_embeddings(
            chunk_texts,
            provider=provider,
            model_id=model_id,
            token=hf_token,
        )

        # 5. Load or create FAISS index from S3 and append new vectors
        index, metadata_list = load_index_from_s3(s3_bucket)
        index, metadata_list = add_chunks_to_index(
            index=index,
            metadata_list=metadata_list,
            new_chunks=chunks,
            new_embeddings=embeddings,
        )

        # 6. Save updated index and metadata back to S3
        save_index_to_s3(index, metadata_list, s3_bucket)
    else:
        logger.warning(f"No text extracted from document {doc_id}")

    # 7. Update DynamoDB status to COMPLETED
    now = _now_iso()
    table.update_item(
        Key={"DocumentId": doc_id},
        UpdateExpression=(
            "SET #st = :st, stage = :sg, page_count = :pc, chunk_count = :cc, "
            "extracted_s3_key = :ek, vector_store_s3_key = :vk, updated_at = :u"
        ),
        ExpressionAttributeNames={"#st": "status"},
        ExpressionAttributeValues={
            ":st": "COMPLETED",
            ":sg": "INDEXED",
            ":pc": page_count,
            ":cc": chunk_count,
            ":ek": extracted_s3_key,
            ":vk": "vector_store/index.faiss",
            ":u": now,
        },
    )

    logger.info(f"Document {doc_id} successfully indexed! Pages: {page_count}, Chunks: {chunk_count}")

    return {
        "document_id": doc_id,
        "status": "COMPLETED",
        "page_count": page_count,
        "chunk_count": chunk_count,
        "extracted_s3_key": extracted_s3_key,
    }


def handle_failure(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 4: Record error details in DynamoDB and mark FAILED.
    """
    exec_input = event.get("execution_input", {})
    doc_id = (
        event.get("document_id")
        or exec_input.get("document_id")
        or (exec_input.get("detail", {}).get("object", {}).get("key") and _sanitize_doc_id(exec_input["detail"]["object"]["key"]))
        or "UNKNOWN_DOC"
    )
    error = event.get("error") or event.get("Cause") or "Unspecified processing error"

    logger.error(f"Processing failed for document {doc_id}: {error}")

    try:
        table.update_item(
            Key={"DocumentId": doc_id},
            UpdateExpression="SET #st = :st, error_message = :err, updated_at = :u",
            ExpressionAttributeNames={"#st": "status"},
            ExpressionAttributeValues={
                ":st": "FAILED",
                ":err": str(error),
                ":u": _now_iso(),
            },
        )
    except Exception as e:
        logger.warning(f"Could not record failure in DynamoDB: {e}")

    return {
        "document_id": doc_id,
        "status": "FAILED",
        "error": str(error),
    }


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Step Functions task router.
    """
    logger.info(f"IDP Lambda invoked with event: {json.dumps(event, default=str)}")
    action = event.get("action")

    # If action is omitted, determine from context or S3 payload
    if not action:
        if "s3_bucket" in event or "detail" in event or "Records" in event:
            action = "start"
        elif "job_status" in event or "job_id" in event:
            action = "check_status"

    if action == "start":
        return handle_start(event)
    elif action == "check_status":
        return handle_check_status(event)
    elif action == "process_and_index":
        return handle_process_and_index(event)
    elif action == "handle_failure":
        return handle_failure(event)
    else:
        raise ValueError(f"Unrecognized action: '{action}'. Expected start|check_status|process_and_index|handle_failure")
