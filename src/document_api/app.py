"""
AWS Lambda Handler for Document Management & Presigned Upload API.

Endpoints:
- GET /documents: List all documents and processing states from DynamoDB
- POST /upload-url: Generate S3 presigned PUT URL for direct client-to-S3 document upload
- GET /documents/{id}: Get detailed status of a specific document
"""

from datetime import datetime, timezone
import json
import logging
import os
import urllib.parse
from typing import Any, Dict
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME", "DocumentMetadataTable")
S3_BUCKET_NAME = os.environ.get("DOCUMENT_BUCKET_NAME", "")
URL_EXPIRATION_SECONDS = int(os.environ.get("URL_EXPIRATION_SECONDS", "900"))

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(DYNAMODB_TABLE_NAME)
s3_client = boto3.client("s3")


def _cors_headers() -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type,Authorization,X-Amz-Date,X-Api-Key",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    }


def _respond(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": _cors_headers(),
        "body": json.dumps(body, default=str),
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def handle_list_documents() -> Dict[str, Any]:
    """Scan DynamoDB to retrieve all uploaded documents and their processing status."""
    try:
        response = table.scan()
        items = response.get("Items", [])

        # Sort documents by created_at descending
        items.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return _respond(200, {"documents": items, "count": len(items)})
    except Exception as e:
        logger.error(f"Failed to scan DynamoDB: {e}")
        return _respond(500, {"error": f"Failed to retrieve documents: {str(e)}"})


def handle_get_document(doc_id: str) -> Dict[str, Any]:
    """Get single document metadata by DocumentId."""
    try:
        response = table.get_item(Key={"DocumentId": doc_id})
        item = response.get("Item")
        if not item:
            return _respond(404, {"error": f"Document '{doc_id}' not found"})
        return _respond(200, {"document": item})
    except Exception as e:
        logger.error(f"Failed to get item {doc_id}: {e}")
        return _respond(500, {"error": str(e)})


def handle_generate_upload_url(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate an Amazon S3 presigned PUT URL allowing the client (Streamlit)
    to upload large PDFs/images directly to S3 without sending raw bytes through API Gateway.
    """
    filename = body.get("filename")
    if not filename:
        return _respond(400, {"error": "Field 'filename' is required"})

    clean_filename = os.path.basename(filename).replace(" ", "_")
    content_type = body.get("content_type", "application/pdf")
    s3_key = f"uploads/{clean_filename}"
    doc_id = clean_filename
    now = _now_iso()

    try:
        presigned_url = s3_client.generate_presigned_url(
            ClientMethod="put_object",
            Params={
                "Bucket": S3_BUCKET_NAME,
                "Key": s3_key,
                "ContentType": content_type,
            },
            ExpiresIn=URL_EXPIRATION_SECONDS,
        )

        # Pre-register entry in DynamoDB
        table.put_item(
            Item={
                "DocumentId": doc_id,
                "filename": clean_filename,
                "s3_bucket": S3_BUCKET_NAME,
                "s3_key": s3_key,
                "status": "INITIALIZED",
                "stage": "WAITING_FOR_UPLOAD",
                "content_type": content_type,
                "created_at": now,
                "updated_at": now,
            }
        )

        return _respond(
            200,
            {
                "document_id": doc_id,
                "s3_key": s3_key,
                "s3_bucket": S3_BUCKET_NAME,
                "upload_url": presigned_url,
                "expires_in_seconds": URL_EXPIRATION_SECONDS,
            },
        )
    except Exception as e:
        logger.error(f"Failed to generate presigned URL: {e}")
        return _respond(500, {"error": f"Could not generate upload URL: {str(e)}"})


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Route document API requests."""
    logger.info(f"Document API invoked: {json.dumps(event, default=str)}")

    http_method = (
        event.get("httpMethod")
        or event.get("requestContext", {}).get("http", {}).get("method")
        or "GET"
    )

    if http_method == "OPTIONS":
        return _respond(200, {"message": "CORS preflight OK"})

    path = event.get("path") or event.get("rawPath") or ""

    if http_method == "GET":
        path_parameters = event.get("pathParameters") or {}
        doc_id = path_parameters.get("document_id")
        if doc_id:
            return handle_get_document(doc_id)
        return handle_list_documents()

    elif http_method == "POST":
        body_raw = event.get("body", "{}")
        body = json.loads(body_raw) if isinstance(body_raw, str) else (body_raw or {})
        return handle_generate_upload_url(body)

    return _respond(405, {"error": f"Method {http_method} not allowed"})
