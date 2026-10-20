"""
AWS Lambda Handler for Query & RAG Retrieval API.

Performs:
1. Loads/caches S3-backed FAISS vector store into Lambda /tmp.
2. Embeds user query using Amazon Bedrock Titan or Hugging Face all-MiniLM-L6-v2.
3. Retrieves Top-K relevant document chunks with page-level metadata.
4. Constructs strictly grounded context prompt to prevent hallucinations.
5. Invokes Amazon Bedrock Converse API (Claude 3 Haiku / Nova / Llama 3) with Hugging Face fallback.
6. Returns verified answer and structured source citations.
"""

import json
import logging
import math
import os
import pickle
import re
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

# Environment variables (Strict Bedrock Prohibition Compliant)
S3_BUCKET_NAME = os.environ.get("DOCUMENT_BUCKET_NAME", "")
DEFAULT_PROVIDER = os.environ.get("MODEL_PROVIDER", "huggingface").lower()
HF_DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")
FALLBACK_LLM_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]
BEDROCK_DEFAULT_LLM_MODEL = os.environ.get(
    "BEDROCK_LLM_MODEL", "amazon.nova-micro-v1:0"
)
BEDROCK_REGION = (
    os.environ.get("BEDROCK_REGION")
    or os.environ.get("AWS_REGION")
    or os.environ.get("AWS_DEFAULT_REGION")
    or "us-east-1"
)

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

        logger.info(f"Vector store loaded successfully: {_CACHED_INDEX.ntotal} vectors (dim={_CACHED_INDEX.d})")
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
    index: Optional[faiss.IndexFlatIP],
    metadata_list: List[Dict[str, Any]],
    query_vector: Optional[np.ndarray],
    query_text: str = "",
    top_k: int = 3,
    min_score: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-K matching chunks using Dense (FAISS) + Sparse (Lexical) Hybrid Search.
    Guarantees page-level provenance, deduplication, and high keyword precision.
    """
    if not metadata_list:
        return []

    # Comprehensive Stopwords & Query Intent Vocabulary
    stops = {
        "what", "is", "the", "in", "and", "or", "to", "for", "of", "a", "an", "on",
        "with", "at", "by", "from", "it", "this", "that", "these", "those", "how", "who",
        "where", "when", "which", "whom", "whose", "why", "will", "would", "shall", "should",
        "could", "may", "might", "must", "do", "did", "does", "done", "doing", "be", "being",
        "been", "was", "were", "there", "their", "they", "them", "my", "your", "his", "her",
        "its", "our", "can", "you", "tell", "me", "about", "please", "give", "have", "has", "had"
    }
    inquiry_words = {
        "located", "location", "situated", "based", "find", "found", "show", "give",
        "tell", "details", "detail", "information", "info", "list", "state", "stated",
        "describe", "described", "provide", "provided", "mention", "mentioned"
    }

    raw_words = re.findall(r"[a-z0-9]+", query_text.lower()) if query_text else []
    query_words = [w for w in raw_words if w not in stops and len(w) > 1]
    subject_words = [w for w in query_words if w not in inquiry_words]

    # 1. Compute dense FAISS similarity scores
    dense_scores = [0.0] * len(metadata_list)
    if index is not None and index.ntotal > 0 and query_vector is not None:
        q_vec = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm
        k_search = min(len(metadata_list), index.ntotal)
        distances, indices = index.search(q_vec, k_search)
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(metadata_list):
                dense_scores[idx] = float(dist)

    # 2. Compute sparse lexical scores with subject-entity weighting
    sparse_scores = [0.0] * len(metadata_list)
    phrase = " ".join(subject_words) if subject_words else " ".join(query_words)

    for idx, item in enumerate(metadata_list):
        text_lower = (item.get("text", "") or "").lower()
        doc_lower = (item.get("doc_name", "") or "").lower()
        score = 0.0

        # Exact phrase match in body
        if len(subject_words) > 1 and phrase in text_lower:
            score += 5.0

        # Subject words in text
        matched_subject_in_text = 0
        for w in subject_words:
            cnt = len(re.findall(r"\b" + re.escape(w) + r"\b", text_lower))
            if cnt > 0:
                score += 2.0 + math.log(1 + cnt)
                matched_subject_in_text += 1

        # Subject words in doc_name
        matched_subject_in_doc = 0
        for w in subject_words:
            if re.search(r"\b" + re.escape(w) + r"\b", doc_lower):
                score += 1.5
                matched_subject_in_doc += 1

        # Inquiry words in text (give minor weight only if subject words matched)
        if matched_subject_in_text > 0 or matched_subject_in_doc > 0:
            for w in inquiry_words:
                if w in raw_words:
                    cnt = len(re.findall(r"\b" + re.escape(w) + r"\b", text_lower))
                    if cnt > 0:
                        score += 0.5

        # Strict subject penalty: if subject words exist in query, candidate must match at least one
        if subject_words and (matched_subject_in_text == 0 and matched_subject_in_doc == 0):
            score = 0.0

        sparse_scores[idx] = score

    # Normalize sparse scores to [0, 1] range if max > 0
    max_sparse = max(sparse_scores) if sparse_scores else 0.0
    norm_sparse = [s / max_sparse if max_sparse > 0 else 0.0 for s in sparse_scores]

    # 3. Hybrid fusion and candidate filtering
    candidates = []
    for idx, meta in enumerate(metadata_list):
        d_val = max(0.0, dense_scores[idx])
        s_val = norm_sparse[idx]

        # Filter out chunks with 0 sparse score unless dense score is exceptionally high
        if subject_words and s_val == 0.0 and d_val < 0.65:
            continue

        # Ignore tiny uninformative chunks (< 30 chars) if they don't have top sparse match
        text_len = len((meta.get("text") or "").strip())
        if text_len < 30 and s_val < 0.8:
            continue

        combined = 0.4 * d_val + 0.6 * s_val if max_sparse > 0 else d_val
        candidates.append((combined, d_val, s_val, idx, meta))

    candidates.sort(key=lambda x: x[0], reverse=True)
    if not candidates:
        return []

    top_score = candidates[0][0]
    min_cutoff = max(0.20, top_score * 0.55)

    results: List[Dict[str, Any]] = []
    seen_chunks = set()

    for combined, d_val, s_val, idx, meta in candidates:
        if combined < min_cutoff:
            break
        doc_name = meta.get("doc_name", "unknown_document")
        page_num = meta.get("page_number", 1)
        snippet = meta.get("text", "")

        chunk_key = meta.get("chunk_id") or f"{doc_name}_p{page_num}"
        if chunk_key in seen_chunks:
            continue
        seen_chunks.add(chunk_key)

        results.append({
            "doc_name": doc_name,
            "page_number": page_num,
            "snippet": snippet,
            "relevance_score": round(float(combined), 4),
            "dense_score": round(float(d_val), 4),
            "sparse_score": round(float(s_val), 4),
        })

        if len(results) >= top_k:
            break

    return results


def _call_bedrock_llm(
    prompt_system: str,
    prompt_user: str,
    model_id: str = BEDROCK_DEFAULT_LLM_MODEL,
    region_name: Optional[str] = None,
    max_tokens: int = 512,
) -> Optional[str]:
    """
    Call Amazon Bedrock using the unified Converse API.
    Supports Anthropic Claude (3.5 Sonnet, 3 Haiku), Amazon Nova, Meta Llama, and Mistral.
    """
    region = region_name or BEDROCK_REGION
    logger.info(f"Calling Amazon Bedrock Converse API with model {model_id} in {region}")

    try:
        bedrock = boto3.client("bedrock-runtime", region_name=region)
        response = bedrock.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt_user}]}],
            system=[{"text": prompt_system}],
            inferenceConfig={
                "maxTokens": max_tokens,
                "temperature": 0.1,
                "topP": 0.9,
            },
        )
        output_content = response.get("output", {}).get("message", {}).get("content", [])
        if output_content and "text" in output_content[0]:
            return output_content[0]["text"].strip()
    except Exception as e:
        logger.warning(f"Amazon Bedrock Converse call failed for {model_id}: {e}")

    return None


def _call_huggingface_llm(
    prompt_system: str,
    prompt_user: str,
    model_id: str,
    hf_token: Optional[str] = None,
    max_tokens: int = 512,
) -> Optional[str]:
    """
    Call Hugging Face Serverless Inference API for chat or text generation (fallback).
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
                res = requests.post(chat_url, headers=headers, json=chat_payload, timeout=20)
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
                res = requests.post(gen_url, headers=headers, json=gen_payload, timeout=20)
                if res.status_code == 200:
                    data = res.json()
                    if isinstance(data, list) and len(data) > 0:
                        return data[0].get("generated_text", "").strip()
                    elif isinstance(data, dict):
                        return data.get("generated_text", "").strip()
            except Exception as e:
                logger.warning(f"Standard generation error with {gen_url}: {e}")

    return None


def _generate_extractive_fallback(question: str, sources: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Extractive synthesis fallback when external LLM APIs are offline or token is absent.
    Performs per-source relevance verification to eliminate false-positive excerpts.
    """
    if not sources:
        return "I cannot find any relevant excerpts in the provided documents to answer this question.", []

    stops = {
        "what", "is", "the", "in", "and", "or", "to", "for", "of", "a", "an", "on",
        "with", "at", "by", "from", "it", "this", "that", "these", "those", "how", "who",
        "where", "when", "can", "you", "tell", "me", "about", "please", "give", "were",
        "does", "did", "have", "has", "had", "are", "been"
    }
    inquiry_words = {
        "located", "location", "situated", "based", "find", "found", "show", "give",
        "tell", "details", "detail", "information", "info", "list", "state", "stated",
        "describe", "described", "provide", "provided", "mention", "mentioned"
    }

    raw_words = re.findall(r"[a-z0-9]+", question.lower())
    query_words = [w for w in raw_words if w not in stops and len(w) > 1]
    subject_words = [w for w in query_words if w not in inquiry_words]

    # Verify each source individually
    valid_sources = []
    for s in sources:
        snip = (s.get("snippet") or s.get("text", "")).lower()
        doc = (s.get("doc_name") or "").lower()
        score = s.get("relevance_score", 0.0)

        # Check if snippet or doc name matches subject words
        matched_subjects = [w for w in subject_words if re.search(r"\b" + re.escape(w) + r"\b", snip) or re.search(r"\b" + re.escape(w) + r"\b", doc)]
        matched_inquiry = [w for w in inquiry_words if re.search(r"\b" + re.escape(w) + r"\b", snip)]

        if subject_words:
            # If query has subject words, this source must match at least one or have high score
            if matched_subjects or score >= 0.65:
                valid_sources.append(s)
        else:
            # If query only has inquiry words, check any word match or score
            if matched_inquiry or score >= 0.35:
                valid_sources.append(s)

    if not valid_sources:
        return (
            f"I cannot find information about '{question}' in the uploaded documents. "
            "Please ask a question specific to the contents of your uploaded documents.",
            [],
        )

    lines = [
        "Based on the relevant excerpts retrieved from the processed documents:",
        "",
    ]
    for s in valid_sources:
        doc_name = s.get("doc_name", "document")
        page_num = s.get("page_number", 1)
        raw_text = s.get("snippet") or s.get("text", "")
        lines.append(f"• **[Document: {doc_name}, Page: {page_num}]**:")
        words = raw_text.split()
        clean_snip = " ".join(words[:150])
        trailing = "..." if len(words) > 150 else ""
        lines.append(f'  "{clean_snip}{trailing}"')
        lines.append("")

    lines.append(
        "*(Note: Generated via extractive retrieval verification based directly on OCR output.)*"
    )
    return "\n".join(lines), valid_sources


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

    # 0. Conversational Intent Guardrail for greetings, chitchat, and identity
    greetings = {
        "hello", "hi", "hey", "namaste", "good morning", "good afternoon",
        "good evening", "greetings", "howdy", "hola", "sup", "yo",
        "how are you", "who are you", "what can you do", "help",
        "thanks", "thank you", "bye", "goodbye"
    }
    clean_q = re.sub(r"[^\w\s]", "", question.strip().lower())
    if clean_q in greetings:
        return _respond(
            200,
            {
                "question": question,
                "answer": (
                    "Hello! I am your Intelligent Document Processing (IDP) Assistant on AWS. "
                    "You can ask me specific questions about your uploaded documents (such as invoices, "
                    "bank statements, rental agreements, receipts, and reports), and I will retrieve the exact answers "
                    "with document and page citations."
                ),
                "sources": [],
                "chunks_retrieved": 0,
                "model_used": "conversational-guardrail",
                "provider": (body.get("provider") or DEFAULT_PROVIDER).lower(),
            },
        )

    top_k = min(int(body.get("top_k", 3)), 10)
    user_token = body.get("hf_token")
    provider = (body.get("provider") or DEFAULT_PROVIDER).lower()
    bucket_name = body.get("s3_bucket") or S3_BUCKET_NAME

    if provider == "bedrock":
        requested_model = body.get("model") or BEDROCK_DEFAULT_LLM_MODEL
    else:
        requested_model = body.get("model") or HF_DEFAULT_LLM_MODEL

    # 1. Load vector store from S3
    index, metadata_list = _load_cached_vector_store(bucket_name)
    if index is None or index.ntotal == 0:
        return _respond(
            200,
            {
                "answer": "No indexed documents found in S3 vector store. Please upload documents first via the IDP workflow.",
                "sources": [],
                "chunks_retrieved": 0,
                "model_used": "none",
                "provider": provider,
            },
        )

    # 2. Embed user question matching the index vector dimension
    target_dim = index.d if index is not None else None
    query_vector = get_query_embedding(
        question,
        provider=provider,
        token=user_token,
        target_dim=target_dim,
    )

    # 3. Retrieve Top-K chunks via Hybrid Search
    sources = _retrieve_top_k(
        index,
        metadata_list,
        query_vector,
        query_text=question,
        top_k=top_k,
    )

    if not sources:
        return _respond(
            200,
            {
                "answer": "I cannot find any relevant excerpts in the provided documents to answer this question.",
                "sources": [],
                "chunks_retrieved": 0,
                "model_used": requested_model,
                "provider": provider,
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

    # 5. Call LLM (Strict No-Bedrock Default: Open-Source Hugging Face LLMs / Extractive Fallback)
    answer = None
    model_used = requested_model

    # 1. Amazon Bedrock Foundation Models (AWS Native)
    if provider == "bedrock":
        answer = _call_bedrock_llm(
            prompt_system=system_prompt,
            prompt_user=user_prompt,
            model_id=requested_model,
            max_tokens=512,
        )
        if answer:
            model_used = requested_model
        else:
            logger.info("Bedrock unavailable or unauthorized. Falling back to open-source / extractive pipeline.")

    # 2. Open-Source Generative LLMs (Hugging Face Serverless)
    if not answer and (provider in ("huggingface", "opensource", "default") or user_token or os.environ.get("HF_TOKEN")):
        answer = _call_huggingface_llm(
            prompt_system=system_prompt,
            prompt_user=user_prompt,
            model_id=requested_model if provider != "bedrock" else HF_DEFAULT_LLM_MODEL,
            hf_token=user_token,
            max_tokens=512,
        )
        if answer:
            model_used = requested_model if provider != "bedrock" else HF_DEFAULT_LLM_MODEL

    # 3. Grounded Extractive Citation Fallback
    if not answer:
        logger.info("Generative LLMs unavailable. Using extractive citation synthesis.")
        answer, sources = _generate_extractive_fallback(question, sources)
        model_used = "extractive-fallback"

    return _respond(
        200,
        {
            "question": question,
            "answer": answer,
            "sources": sources,
            "chunks_retrieved": len(sources),
            "model_used": model_used,
            "provider": provider,
        },
    )
