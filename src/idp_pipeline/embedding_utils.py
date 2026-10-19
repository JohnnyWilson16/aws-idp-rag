"""
Embedding Generator supporting Amazon Bedrock (Titan V2/V1) and Hugging Face.

Features:
- Native Amazon Bedrock Titan Text Embeddings V2 (1024-d, 512-d) and V1 (1536-d).
- Hugging Face Serverless Inference API (all-MiniLM-L6-v2, 384-d).
- L2-normalization for exact cosine similarity in FAISS IndexFlatIP.
- Resilient fallback: Bedrock -> Hugging Face -> Deterministic Local Hashing.
"""

import hashlib
import json
import logging
import os
import re
import time
from typing import List, Optional, Union
import numpy as np
import requests
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Provider & Model configurations
DEFAULT_PROVIDER = os.environ.get("MODEL_PROVIDER", "bedrock").lower()
BEDROCK_EMBEDDING_MODEL = os.environ.get(
    "BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0"
)
BEDROCK_REGION = (
    os.environ.get("BEDROCK_REGION")
    or os.environ.get("AWS_REGION")
    or os.environ.get("AWS_DEFAULT_REGION")
    or "us-east-1"
)
BEDROCK_DEFAULT_DIM = 1024

HF_EMBEDDING_MODEL_ID = os.environ.get(
    "EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2"
)
HF_EMBEDDING_DIM = 384
HF_ROUTER_URL = f"https://router.huggingface.co/hf-inference/models/{HF_EMBEDDING_MODEL_ID}"
HF_LEGACY_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_EMBEDDING_MODEL_ID}"


def _get_hf_headers(token: Optional[str] = None) -> dict:
    """Construct HTTP authorization headers for Hugging Face Inference API."""
    api_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN", "")
    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token.strip()}"
    return headers


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    """L2-normalize vector or batch of vectors so dot-product equals cosine similarity."""
    if vec.ndim == 1:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vec / norms


def _offline_fallback_embedding(text: str, dim: int = BEDROCK_DEFAULT_DIM) -> np.ndarray:
    """
    Generate a deterministic positive term-frequency hashed embedding vector.
    Maps words, bigrams, and character n-grams to guarantee positive cosine similarity
    between queries and relevant document excerpts.
    """
    vec = np.zeros(dim, dtype=np.float32)
    stops = {
        "what", "is", "the", "in", "and", "or", "to", "for", "of", "a", "an", "on",
        "with", "at", "by", "from", "it", "this", "that", "these", "those", "how", "who"
    }
    raw = re.findall(r"[a-z0-9]+", text.lower())
    words = [w for w in raw if w not in stops and len(w) > 1]

    for w in words:
        h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 2.0

    # Bigrams for phrase preservation
    for i in range(len(words) - 1):
        bigram = f"{words[i]}_{words[i+1]}"
        h = int(hashlib.md5(bigram.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.5

    clean = "".join(words)
    for i in range(len(clean) - 2):
        gram = clean[i : i + 3]
        h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 0.2

    return _normalize_vector(vec)


def get_bedrock_embedding(
    text: str,
    model_id: str = BEDROCK_EMBEDDING_MODEL,
    dimensions: int = BEDROCK_DEFAULT_DIM,
    region_name: Optional[str] = None,
) -> np.ndarray:
    """
    Generate embeddings using Amazon Bedrock (Titan Embeddings V2 or V1).
    """
    clean_text = text.replace("\n", " ").strip()
    if not clean_text:
        return np.zeros(dimensions, dtype=np.float32)

    region = region_name or BEDROCK_REGION
    client = boto3.client("bedrock-runtime", region_name=region)

    if "titan-embed-text-v2" in model_id:
        payload = {
            "inputText": clean_text,
            "dimensions": dimensions,
            "normalize": True,
        }
    else:
        payload = {"inputText": clean_text}

    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(payload),
        contentType="application/json",
        accept="application/json",
    )
    resp_body = json.loads(response["body"].read())
    vec = np.array(resp_body["embedding"], dtype=np.float32)
    return _normalize_vector(vec)


def get_hf_embedding(
    text: str,
    token: Optional[str] = None,
    max_retries: int = 2,
) -> np.ndarray:
    """
    Generate normalized embeddings using Hugging Face Serverless API.
    """
    clean_text = text.replace("\n", " ").strip()
    if not clean_text:
        return np.zeros(HF_EMBEDDING_DIM, dtype=np.float32)

    headers = _get_hf_headers(token)
    endpoints = [HF_ROUTER_URL, HF_LEGACY_URL]

    for attempt in range(max_retries):
        for url in endpoints:
            try:
                res = requests.post(
                    url,
                    headers=headers,
                    json={"inputs": clean_text, "options": {"wait_for_model": True}},
                    timeout=15,
                )
                if res.status_code == 200:
                    data = res.json()
                    vec = np.array(data, dtype=np.float32)
                    if vec.ndim == 2:
                        vec = np.mean(vec, axis=0)
                    elif vec.ndim > 2:
                        vec = vec.flatten()[:HF_EMBEDDING_DIM]
                    if vec.shape[0] == HF_EMBEDDING_DIM:
                        return _normalize_vector(vec)
            except Exception as e:
                logger.warning(f"HF embedding error: {e}")
        time.sleep(1)

    return _offline_fallback_embedding(clean_text, dim=HF_EMBEDDING_DIM)


def get_embedding(
    text: str,
    provider: Optional[str] = None,
    model_id: Optional[str] = None,
    token: Optional[str] = None,
    region_name: Optional[str] = None,
) -> np.ndarray:
    """
    Unified embedding entrypoint.
    Prioritizes Amazon Bedrock (if configured), falling back to Hugging Face or offline hashing.
    """
    chosen_provider = (provider or DEFAULT_PROVIDER).lower()

    if chosen_provider == "bedrock":
        try:
            m_id = model_id or BEDROCK_EMBEDDING_MODEL
            return get_bedrock_embedding(text, model_id=m_id, region_name=region_name)
        except Exception as e:
            logger.warning(f"Amazon Bedrock embedding failed ({e}), falling back to Hugging Face / offline.")

    # Hugging Face or local fallback
    if token or os.environ.get("HF_TOKEN") or chosen_provider == "huggingface":
        try:
            return get_hf_embedding(text, token=token)
        except Exception as e:
            logger.warning(f"HF embedding failed ({e}), using offline fallback.")

    target_dim = BEDROCK_DEFAULT_DIM if chosen_provider == "bedrock" else HF_EMBEDDING_DIM
    return _offline_fallback_embedding(text, dim=target_dim)


def get_batch_embeddings(
    texts: List[str],
    provider: Optional[str] = None,
    model_id: Optional[str] = None,
    token: Optional[str] = None,
    batch_size: int = 8,
    region_name: Optional[str] = None,
) -> np.ndarray:
    """
    Generate normalized embeddings for a list of texts in batches.
    """
    all_embeddings: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for t in batch:
            emb = get_embedding(
                t,
                provider=provider,
                model_id=model_id,
                token=token,
                region_name=region_name,
            )
            all_embeddings.append(emb)

    return np.vstack(all_embeddings).astype(np.float32)
