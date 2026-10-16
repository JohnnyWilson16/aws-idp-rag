"""
Embedding Client for Query API Lambda.
Generates 384-dimensional normalized vector embeddings using Hugging Face Serverless API.
"""

import hashlib
import logging
import os
import time
from typing import Optional
import numpy as np
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EMBEDDING_MODEL_ID = os.environ.get(
    "EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIM = 384
HF_ROUTER_URL = f"https://router.huggingface.co/hf-inference/models/{EMBEDDING_MODEL_ID}"
HF_LEGACY_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL_ID}"


def _normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize vector to unit length."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _offline_fallback(text: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """
    Generate deterministic term-frequency hashed vector for offline queries.
    Matches feature hashing logic from indexing for high keyword cosine overlap.
    """
    vec = np.zeros(dim, dtype=np.float32)
    words = [w.strip(".,;:!?()[]{}\"'") for w in text.lower().split()]
    for w in words:
        if not w:
            continue
        h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if ((h >> 16) & 1) else -1.0
        vec[idx] += sign

    clean = text.lower().replace(" ", "")
    for i in range(len(clean) - 2):
        gram = clean[i : i + 3]
        h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 0.3

    return _normalize(vec)


def get_query_embedding(
    text: str,
    token: Optional[str] = None,
    max_retries: int = 2,
) -> np.ndarray:
    """
    Generate normalized 384-dimensional embedding for the user query.

    Args:
        text: Query string.
        token: Optional Hugging Face API token.
        max_retries: Retry count for cold starts.

    Returns:
        np.ndarray of shape (384,) float32.
    """
    clean_text = text.replace("\n", " ").strip()
    if not clean_text:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    api_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN", "")
    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token.strip()}"

    for attempt in range(max_retries):
        for url in [HF_ROUTER_URL, HF_LEGACY_URL]:
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
                        vec = vec.flatten()[:EMBEDDING_DIM]

                    if vec.shape[0] == EMBEDDING_DIM:
                        return _normalize(vec)

                elif res.status_code == 503:
                    time.sleep(2)
                    continue
                elif res.status_code in (401, 403):
                    logger.warning("HF Token missing or invalid. Using offline vector.")
                    return _offline_fallback(clean_text)

            except Exception as e:
                logger.warning(f"Embedding request error to {url}: {e}")

    logger.warning("Embedding service unavailable, using local fallback.")
    return _offline_fallback(clean_text)
