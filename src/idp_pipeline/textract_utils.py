"""
Amazon Textract Utilities for Multi-Page Asynchronous Document OCR and Chunking.

Handles:
- Starting asynchronous Textract document text detection for PDFs / images.
- Polling and paginating through complete Textract result blocks.
- Preserving page numbers, line hierarchies, and document structure.
- Chunking extracted text with metadata (document name, page number, chunk ID).
"""

import logging
from typing import Any, Dict, List, Optional
import boto3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_textract_client(region_name: Optional[str] = None) -> Any:
    """Instantiate or reuse a boto3 Textract client."""
    return boto3.client("textract", region_name=region_name)


def start_document_text_detection(
    s3_bucket: str,
    s3_key: str,
    client: Optional[Any] = None,
) -> str:
    """
    Initiate asynchronous Textract text detection for an S3 document.

    Args:
        s3_bucket: Name of the S3 bucket where document is stored.
        s3_key: S3 object key (e.g. 'uploads/contract.pdf').
        client: Optional boto3 textract client.

    Returns:
        JobId string for tracking processing status.
    """
    textract = client or get_textract_client()
    logger.info(f"Starting async Textract job for s3://{s3_bucket}/{s3_key}")

    response = textract.start_document_text_detection(
        DocumentLocation={
            "S3Object": {
                "Bucket": s3_bucket,
                "Name": s3_key,
            }
        }
    )
    job_id = response.get("JobId")
    if not job_id:
        raise ValueError(f"Textract failed to return JobId for s3://{s3_bucket}/{s3_key}")

    logger.info(f"Textract job started successfully with JobId: {job_id}")
    return job_id


def check_document_text_detection_status(
    job_id: str,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Check the current status of an asynchronous Textract job.

    Args:
        job_id: The Textract JobId.
        client: Optional boto3 textract client.

    Returns:
        Dict with 'JobStatus' ('IN_PROGRESS', 'SUCCEEDED', 'FAILED') and status message.
    """
    textract = client or get_textract_client()
    response = textract.get_document_text_detection(JobId=job_id, MaxResults=1)
    status = response.get("JobStatus", "UNKNOWN")
    status_message = response.get("StatusMessage", "")

    logger.info(f"Job {job_id} status: {status} ({status_message})")
    return {
        "job_id": job_id,
        "job_status": status,
        "status_message": status_message,
    }


def fetch_all_textract_blocks(
    job_id: str,
    client: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch all blocks across all pages from a completed Textract job using pagination.

    Args:
        job_id: The completed Textract JobId.
        client: Optional boto3 textract client.

    Returns:
        Complete list of Textract Block dictionaries.
    """
    textract = client or get_textract_client()
    all_blocks: List[Dict[str, Any]] = []
    next_token: Optional[str] = None

    while True:
        kwargs: Dict[str, Any] = {"JobId": job_id}
        if next_token:
            kwargs["NextToken"] = next_token

        response = textract.get_document_text_detection(**kwargs)
        blocks = response.get("Blocks", [])
        all_blocks.extend(blocks)

        next_token = response.get("NextToken")
        if not next_token:
            break

    logger.info(f"Retrieved {len(all_blocks)} total blocks for Textract job {job_id}")
    return all_blocks


def parse_textract_blocks(blocks: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Parse Textract blocks into organized pages with extracted lines and confidence.

    Args:
        blocks: Raw blocks from Textract.

    Returns:
        Dictionary keyed by page_number (1-indexed):
        {
            1: {
                "page_number": 1,
                "lines": ["LINE 1", "LINE 2", ...],
                "full_text": "LINE 1\\nLINE 2...",
                "average_confidence": 98.4
            },
            ...
        }
    """
    pages_map: Dict[int, Dict[str, Any]] = {}
    confidences: Dict[int, List[float]] = {}

    for block in blocks:
        block_type = block.get("BlockType")
        if block_type == "LINE":
            page_num = block.get("Page", 1)
            text = block.get("Text", "").strip()
            confidence = block.get("Confidence", 100.0)

            if not text:
                continue

            if page_num not in pages_map:
                pages_map[page_num] = {
                    "page_number": page_num,
                    "lines": [],
                    "full_text": "",
                    "average_confidence": 100.0,
                }
                confidences[page_num] = []

            pages_map[page_num]["lines"].append(text)
            confidences[page_num].append(confidence)

    # Compile full text and compute average confidence per page
    for page_num, page_data in pages_map.items():
        page_data["full_text"] = "\n".join(page_data["lines"])
        if confidences.get(page_num):
            page_data["average_confidence"] = round(
                sum(confidences[page_num]) / len(confidences[page_num]), 2
            )

    return dict(sorted(pages_map.items()))


def chunk_document_pages(
    pages_map: Dict[int, Dict[str, Any]],
    doc_name: str,
    max_chunk_chars: int = 2000,
    overlap_chars: int = 200,
) -> List[Dict[str, Any]]:
    """
    Split extracted text across pages into chunks with metadata attribution.

    Each chunk retains the document name, source page number, and unique chunk ID.
    Approximate rule: 1 token ~= 4 chars, so 500 tokens ~= 2000 chars, 50 tokens ~= 200 chars.

    Args:
        pages_map: Dict returned by parse_textract_blocks.
        doc_name: Basename of the source document (e.g. 'contract.pdf').
        max_chunk_chars: Target maximum character length per chunk.
        overlap_chars: Overlapping characters between consecutive chunks on a page.

    Returns:
        List of chunk dictionaries ready for embedding generation:
        [
            {
                "chunk_id": "contract.pdf_p1_c0",
                "doc_name": "contract.pdf",
                "page_number": 1,
                "text": "...",
                "char_length": 1850,
                "token_estimate": 462
            },
            ...
        ]
    """
    chunks: List[Dict[str, Any]] = []

    for page_num_key, page_data in pages_map.items():
        try:
            page_num = int(page_num_key)
        except (ValueError, TypeError):
            page_num = 1

        if isinstance(page_data, str):
            page_text = page_data
            lines = [{"text": l} for l in page_text.splitlines() if l.strip()]
        else:
            lines = page_data.get("lines", [])
            page_text = page_data.get("full_text", "")

        if not lines or not page_text.strip():
            continue

        # If page text is within chunk size limit, keep as single chunk
        if len(page_text) <= max_chunk_chars:
            chunks.append({
                "chunk_id": f"{doc_name}_p{page_num}_c0",
                "doc_name": doc_name,
                "page_number": page_num,
                "text": page_text.strip(),
                "char_length": len(page_text.strip()),
                "token_estimate": len(page_text.strip()) // 4,
            })
            continue

        # Otherwise, perform sliding-window chunking respecting paragraph/sentence boundaries
        start = 0
        chunk_idx = 0
        text_len = len(page_text)

        while start < text_len:
            end = min(start + max_chunk_chars, text_len)

            # If not at the end of the text, try to break at a newline or period
            if end < text_len:
                boundary = page_text.rfind("\n", start + overlap_chars, end)
                if boundary == -1:
                    boundary = page_text.rfind(". ", start + overlap_chars, end)
                if boundary != -1 and boundary > start:
                    end = boundary + 1

            chunk_content = page_text[start:end].strip()
            if chunk_content:
                chunks.append({
                    "chunk_id": f"{doc_name}_p{page_num}_c{chunk_idx}",
                    "doc_name": doc_name,
                    "page_number": page_num,
                    "text": chunk_content,
                    "char_length": len(chunk_content),
                    "token_estimate": len(chunk_content) // 4,
                })
                chunk_idx += 1

            if end >= text_len:
                break

            start = max(end - overlap_chars, start + 1)

    logger.info(f"Generated {len(chunks)} chunks from {len(pages_map)} pages of {doc_name}")
    return chunks
