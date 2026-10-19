"""
Embedding Client for Query API Lambda.
Generates normalized vector embeddings using Amazon Bedrock Titan or Hugging Face.
"""

import hashlib
import json
import logging
import os
import re
import time
from typing import Optional
import numpy as np
import requests
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_PROVIDER = os.environ.get("MODEL_PROVIDER", "huggingface").lower()
HF_EMBEDDING_MODEL_ID = os.environ.get(
    "EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2"
)
HF_EMBEDDING_DIM = int(os.environ.get("VECTOR_DIM", "384"))
HF_ROUTER_URL = f"https://router.huggingface.co/hf-inference/models/{HF_EMBEDDING_MODEL_ID}"
HF_LEGACY_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_EMBEDDING_MODEL_ID}"

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


def _normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize vector to unit length."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _adapt_dimension(vec: np.ndarray, target_dim: Optional[int]) -> np.ndarray:
    """
    Guarantees mathematical dimension alignment with FAISS index.
    If target_dim is provided and differs from vector length:
    - Truncates if vec.shape[0] > target_dim
    - Zero-pads if vec.shape[0] < target_dim
    Then re-normalizes to preserve cosine similarity.
    """
    if not target_dim or vec.shape[0] == target_dim:
        return _normalize(vec)
    if vec.shape[0] > target_dim:
        adapted = vec[:target_dim]
    else:
        adapted = np.pad(vec, (0, target_dim - vec.shape[0]), mode="constant")
    return _normalize(adapted)


def _offline_fallback(text: str, dim: int = HF_EMBEDDING_DIM) -> np.ndarray:
    """
    Generate deterministic positive term-frequency hashed vector for offline queries.
    Matches feature hashing logic from indexing for high keyword cosine overlap.
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

    return _normalize(vec)


def _get_bedrock_query_embedding(
    text: str,
    model_id: str = BEDROCK_EMBEDDING_MODEL,
    dimensions: int = BEDROCK_DEFAULT_DIM,
    region_name: Optional[str] = None,
) -> np.ndarray:
    """Invoke Bedrock Titan Embedding for query."""
    region = region_name or BEDROCK_REGION
    client = boto3.client("bedrock-runtime", region_name=region)

    if "titan-embed-text-v2" in model_id:
        payload = {
            "inputText": text,
            "dimensions": dimensions,
            "normalize": True,
        }
    else:
        payload = {"inputText": text}

    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(payload),
        contentType="application/json",
        accept="application/json",
    )
    resp_body = json.loads(response["body"].read())
    return _normalize(np.array(resp_body["embedding"], dtype=np.float32))


def get_query_embedding(
    text: str,
    provider: Optional[str] = None,
    model_id: Optional[str] = None,
    token: Optional[str] = None,
    region_name: Optional[str] = None,
    target_dim: Optional[int] = None,
) -> np.ndarray:
    """
    Generate normalized embedding for user query.
    Strict No-Bedrock Default: Open-Source Hugging Face / Sentence-Transformers (384-d),
    with automatic vector dimension adaptation matching index.d.
    """
    clean_text = text.replace("\n", " ").strip()
    chosen_provider = (provider or DEFAULT_PROVIDER).lower()
    expected_dim = target_dim or HF_EMBEDDING_DIM

    # 1. Primary Open-Source Path (Hugging Face Serverless)
    if chosen_provider in ("huggingface", "opensource", "default"):
        api_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN", "")
        if api_token:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_token.strip()}",
            }
            try:
                res = requests.post(
                    HF_ROUTER_URL,
                    headers=headers,
                    json={"inputs": clean_text, "options": {"wait_for_model": True}},
                    timeout=5,
                )
                if res.status_code == 200:
                    data = res.json()
                    vec = np.array(data, dtype=np.float32)
                    if vec.ndim == 2:
                        vec = np.mean(vec, axis=0)
                    elif vec.ndim > 2:
                        vec = vec.flatten()[:HF_EMBEDDING_DIM]
                    if vec.shape[0] == HF_EMBEDDING_DIM:
                        return _adapt_dimension(vec, target_dim=expected_dim)
            except Exception as e:
                logger.warning(f"HF query embedding call error: {e}")

        return _adapt_dimension(_offline_fallback(clean_text, dim=HF_EMBEDDING_DIM), target_dim=expected_dim)

    # 2. Legacy Bedrock Path (only if explicitly specified)
    if chosen_provider == "bedrock":
        try:
            m_id = model_id or BEDROCK_EMBEDDING_MODEL
            dim = target_dim or (1536 if "titan-embed-text-v1" in m_id else BEDROCK_DEFAULT_DIM)
            vec = _get_bedrock_query_embedding(clean_text, model_id=m_id, dimensions=dim, region_name=region_name)
            return _adapt_dimension(vec, target_dim=expected_dim)
        except Exception as e:
            logger.warning(f"Bedrock query embedding error ({e}), trying fallback.")
            return _adapt_dimension(_offline_fallback(clean_text, dim=HF_EMBEDDING_DIM), target_dim=expected_dim)

    return _adapt_dimension(_offline_fallback(clean_text, dim=expected_dim), target_dim=expected_dim)
