"""
AWS Lambda Handler for Query & RAG Retrieval API.

Performs:
1. Loads/caches S3-backed FAISS vector store into Lambda /tmp.
2. Embeds user query using all-MiniLM-L6-v2.
3. Retrieves Top-K relevant document chunks with page-level metadata.
4. Constructs strictly grounded context prompt to prevent hallucinations.
5. Invokes Hugging Face Serverless Inference API for LLM text generation.
6. Returns verified answer and structured source citations.
"""

import json
import logging
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple
import faiss
import numpy as np
import requests
import boto3
from botocore.exceptions import ClientError

from embedding_client import get_query_embedding

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Environment variables
S3_BUCKET_NAME = os.environ.get("DOCUMENT_BUCKET_NAME", "")
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")
FALLBACK_LLM_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]
VECTOR_DIM = 384
INDEX_S3_KEY = "vector_store/index.faiss"
METADATA_S3_KEY = "vector_store/metadata.pkl"

s3_client = boto3.client("s3")

# Global in-memory cache across warm Lambda invocations
_CACHED_INDEX: Optional[faiss.IndexFlatIP] = None
_CACHED_METADATA: Optional[List[Dict[str, Any]]] = None
_CACHED_ETAG: Optional[str] = None


def _build_cors_headers() -> Dict[str, str]:
    """CORS headers for web and Streamlit clients."""
    return {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type,Authorization,X-Amz-Date,X-Api-Key",
        "Access-Control-Allow-Methods": "POST,OPTIONS",
    }


def _respond(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    """Standard API Gateway JSON response helper."""
    return {
        "statusCode": status_code,
        "headers": _build_cors_headers(),
        "body": json.dumps(body),
    }


def _load_cached_vector_store(
    bucket_name: str,
    local_dir: str = "/tmp",
) -> Tuple[Optional[faiss.IndexFlatIP], List[Dict[str, Any]]]:
    """
    Load FAISS index and metadata from S3, caching in Lambda /tmp and memory.
    Checks S3 ETag to only re-download when updated.
    """
    global _CACHED_INDEX, _CACHED_METADATA, _CACHED_ETAG

    if not bucket_name:
        logger.error("DOCUMENT_BUCKET_NAME environment variable is empty")
        return None, []

    local_index = os.path.join(local_dir, "query_index.faiss")
    local_meta = os.path.join(local_dir, "query_metadata.pkl")

    try:
        # Check current S3 object ETag/LastModified
        head = s3_client.head_object(Bucket=bucket_name, Key=INDEX_S3_KEY)
        s3_etag = head.get("ETag", "")

        # Reuse warm memory cache if ETag matches
        if _CACHED_INDEX is not None and _CACHED_ETAG == s3_etag:
            return _CACHED_INDEX, _CACHED_METADATA or []

        logger.info(f"Downloading vector store from s3://{bucket_name}/{INDEX_S3_KEY}")
        s3_client.download_file(bucket_name, INDEX_S3_KEY, local_index)
        s3_client.download_file(bucket_name, METADATA_S3_KEY, local_meta)

        _CACHED_INDEX = faiss.read_index(local_index)
        with open(local_meta, "rb") as f:
            _CACHED_METADATA = pickle.load(f)
        _CACHED_ETAG = s3_etag

        logger.info(f"Vector store loaded successfully: {_CACHED_INDEX.ntotal} vectors")
        return _CACHED_INDEX, _CACHED_METADATA

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code in ("404", "NoSuchKey", "NotFound"):
            logger.warning("Vector store does not exist in S3 yet.")
            return None, []
        logger.error(f"S3 download error: {e}")
        return None, []
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        return None, []


def _retrieve_top_k(
    index: faiss.IndexFlatIP,
    metadata_list: List[Dict[str, Any]],
    query_vector: np.ndarray,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Retrieve top-K matching chunks using cosine similarity."""
    if index is None or index.ntotal == 0 or not metadata_list:
        return []

    q_vec = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
    norm = np.linalg.norm(q_vec)
    if norm > 0:
        q_vec = q_vec / norm

    k = min(top_k, index.ntotal)
    distances, indices = index.search(q_vec, k)

    results: List[Dict[str, Any]] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata_list):
            continue
        meta = metadata_list[idx]
        results.append({
            "doc_name": meta.get("doc_name", "unknown_document"),
            "page_number": meta.get("page_number", 1),
            "snippet": meta.get("text", ""),
            "relevance_score": round(float(dist), 4),
        })

    return results


def _call_huggingface_llm(
    prompt_system: str,
    prompt_user: str,
    model_id: str,
    hf_token: Optional[str] = None,
    max_tokens: int = 512,
) -> Optional[str]:
    """
    Call Hugging Face Serverless Inference API for chat or text generation.
    Supports chat completion API and falls back to standard inference pipeline.
    """
    api_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN", "")
    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token.strip()}"

    models_to_try = [model_id] + [m for m in FALLBACK_LLM_MODELS if m != model_id]

    for current_model in models_to_try:
        # 1. Try Chat Completions API
        chat_urls = [
            f"https://router.huggingface.co/hf-inference/models/{current_model}/v1/chat/completions",
            f"https://api-inference.huggingface.co/models/{current_model}/v1/chat/completions",
        ]
        chat_payload = {
            "model": current_model,
            "messages": [
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "top_p": 0.9,
        }

        for chat_url in chat_urls:
            try:
                res = requests.post(chat_url, headers=headers, json=chat_payload, timeout=25)
                if res.status_code == 200:
                    data = res.json()
                    choices = data.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "").strip()
            except Exception as e:
                logger.warning(f"Chat completions error with {chat_url}: {e}")

        # 2. Try Standard Text Generation Endpoint
        gen_urls = [
            f"https://router.huggingface.co/hf-inference/models/{current_model}",
            f"https://api-inference.huggingface.co/models/{current_model}",
        ]
        combined_prompt = f"<|system|>\n{prompt_system}\n<|user|>\n{prompt_user}\n<|assistant|>\n"
        gen_payload = {
            "inputs": combined_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.1,
                "return_full_text": False,
            },
        }

        for gen_url in gen_urls:
            try:
                res = requests.post(gen_url, headers=headers, json=gen_payload, timeout=25)
                if res.status_code == 200:
                    data = res.json()
                    if isinstance(data, list) and len(data) > 0:
                        return data[0].get("generated_text", "").strip()
                    elif isinstance(data, dict):
                        return data.get("generated_text", "").strip()
            except Exception as e:
                logger.warning(f"Standard generation error with {gen_url}: {e}")

    return None


def _generate_extractive_fallback(question: str, sources: List[Dict[str, Any]]) -> str:
    """
    Extractive synthesis fallback when external LLM APIs are offline or token is absent.
    Provides verifiable, hallucination-free answers directly from matched excerpts.
    """
    if not sources:
        return "I cannot find the answer to this question in the provided documents."

    lines = [
        "Based on the relevant excerpts retrieved from the processed documents:",
        "",
    ]
    for idx, s in enumerate(sources, start=1):
        doc_name = s.get("doc_name", "document")
        page_num = s.get("page_number", 1)
        raw_text = s.get("snippet") or s.get("text", "")
        lines.append(f"• **[Document: {doc_name}, Page: {page_num}]**:")
        # Provide concise snippet
        clean_snip = " ".join(raw_text.split()[:80])
        lines.append(f'  "{clean_snip}..."')
        lines.append("")

    lines.append(
        "*(Note: Generated via extractive retrieval verification based directly on OCR output.)*"
    )
    return "\n".join(lines)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler for POST /query.
    """
    logger.info(f"Query API invoked: {event.get('httpMethod', '')} {event.get('path', '')}")

    # Handle CORS OPTIONS preflight
    http_method = event.get("httpMethod") or event.get("requestContext", {}).get("http", {}).get("method")
    if http_method == "OPTIONS":
        return _respond(200, {"message": "CORS preflight OK"})

    # Parse JSON body
    body_raw = event.get("body")
    if not body_raw:
        return _respond(400, {"error": "Missing request body"})

    try:
        body = json.loads(body_raw) if isinstance(body_raw, str) else body_raw
    except Exception:
        return _respond(400, {"error": "Invalid JSON in request body"})

    question = body.get("question", "").strip()
    if not question:
        return _respond(400, {"error": "Field 'question' is required"})

    top_k = min(int(body.get("top_k", 3)), 10)
    user_token = body.get("hf_token")
    requested_model = body.get("model", DEFAULT_LLM_MODEL)
    bucket_name = body.get("s3_bucket") or S3_BUCKET_NAME

    # 1. Load vector store
    index, metadata_list = _load_cached_vector_store(bucket_name)
    if index is None or index.ntotal == 0:
        return _respond(
            200,
            {
                "answer": "No indexed documents found in S3 vector store. Please upload documents first via the IDP workflow.",
                "sources": [],
                "chunks_retrieved": 0,
                "model_used": "none",
            },
        )

    # 2. Embed user question
    query_vector = get_query_embedding(question, token=user_token)

    # 3. Retrieve Top-K chunks
    sources = _retrieve_top_k(index, metadata_list, query_vector, top_k=top_k)

    if not sources:
        return _respond(
            200,
            {
                "answer": "I cannot find any relevant excerpts in the provided documents to answer this question.",
                "sources": [],
                "chunks_retrieved": 0,
                "model_used": "none",
            },
        )

    # 4. Formulate strictly grounded RAG prompt
    context_blocks = []
    for idx, s in enumerate(sources, start=1):
        context_blocks.append(
            f"[Excerpt {idx}] Document: {s['doc_name']} | Page: {s['page_number']}\n{s['snippet']}"
        )
    formatted_context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an expert AI assistant performing Intelligent Document Retrieval.\n"
        "Answer the user's question ONLY using the provided document excerpts below.\n"
        "If the answer cannot be found in the excerpts, state clearly: "
        "'I cannot find the answer to this question in the provided documents.'\n"
        "DO NOT speculate, assume, or hallucinate information not present in the excerpts.\n"
        "Always cite the specific Document Name and Page Number for every claim made."
    )

    user_prompt = (
        f"--- CONTEXT EXCERPTS ---\n"
        f"{formatted_context}\n\n"
        f"--- USER QUESTION ---\n"
        f"{question}\n\n"
        f"--- INSTRUCTIONS FOR ANSWER ---\n"
        f"1. Answer strictly using facts from the excerpts above.\n"
        f"2. Cite your sources with document name and page number (e.g., '[Document: example.pdf, Page: 2]').\n"
        f"3. Keep your response direct, professional, and clear."
    )

    # 5. Call Hugging Face Serverless LLM
    answer = _call_huggingface_llm(
        prompt_system=system_prompt,
        prompt_user=user_prompt,
        model_id=requested_model,
        hf_token=user_token,
        max_tokens=512,
    )

    # If external API unavailable, use structured extractive fallback
    if not answer:
        logger.info("External LLM did not respond. Using extractive citation fallback.")
        answer = _generate_extractive_fallback(question, sources)

    return _respond(
        200,
        {
            "question": question,
            "answer": answer,
            "sources": sources,
            "chunks_retrieved": len(sources),
            "model_used": requested_model,
        },
    )
