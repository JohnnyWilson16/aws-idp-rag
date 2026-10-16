"""
Embedding Generator using Hugging Face Serverless Inference API and all-MiniLM-L6-v2.

Features:
- Free tier serverless embedding generation for 384-dimensional vectors.
- L2-normalization for exact cosine similarity search in FAISS IndexFlatIP.
- Batch processing with exponential backoff retry for cold starts.
- Offline deterministic fallback for local testing and unit tests.
"""

import hashlib
import logging
import os
import time
from typing import List, Optional, Union
import numpy as np
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Default model configuration
EMBEDDING_MODEL_ID = os.environ.get(
    "EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIM = 384

# Hugging Face Router & Legacy Inference URLs
HF_ROUTER_URL = f"https://router.huggingface.co/hf-inference/models/{EMBEDDING_MODEL_ID}"
HF_LEGACY_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL_ID}"


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


def _offline_fallback_embedding(text: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """
    Generate a deterministic term-frequency hashed embedding vector for offline testing.
    Maps words and character tri-grams to 384 dimensions so shared keywords produce high cosine similarity.
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

    # Add character n-grams for subword matching
    clean = text.lower().replace(" ", "")
    for i in range(len(clean) - 2):
        gram = clean[i : i + 3]
        h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 0.3

    return _normalize_vector(vec)


def get_embedding(
    text: str,
    token: Optional[str] = None,
    max_retries: int = 3,
) -> np.ndarray:
    """
    Generate a 384-dimensional normalized embedding vector for a single text.

    Args:
        text: String input to embed.
        token: Optional Hugging Face API token.
        max_retries: Number of retries on 503 model loading response.

    Returns:
        np.ndarray of shape (384,) float32.
    """
    clean_text = text.replace("\n", " ").strip()
    if not clean_text:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    headers = _get_hf_headers(token)

    # Try router URL first, fallback to legacy URL
    endpoints = [HF_ROUTER_URL, HF_LEGACY_URL]

    for attempt in range(max_retries):
        for url in endpoints:
            try:
                payload = {
                    "inputs": clean_text,
                    "options": {"wait_for_model": True},
                }
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=20,
                )

                if response.status_code == 200:
                    data = response.json()
                    vec = np.array(data, dtype=np.float32)
                    # If model returns nested [1, 384] or [seq_len, 384], reduce to mean
                    if vec.ndim == 2:
                        vec = np.mean(vec, axis=0)
                    elif vec.ndim > 2:
                        vec = vec.flatten()[:EMBEDDING_DIM]

                    if vec.shape[0] == EMBEDDING_DIM:
                        return _normalize_vector(vec)
                    else:
                        logger.warning(f"Unexpected embedding dimension: {vec.shape}")

                elif response.status_code == 503:
                    data = response.json()
                    wait_time = data.get("estimated_time", 10)
                    logger.warning(
                        f"HF model {EMBEDDING_MODEL_ID} is loading. Waiting {wait_time}s..."
                    )
                    time.sleep(min(wait_time, 15))
                    continue

                elif response.status_code in (401, 403):
                    logger.warning(
                        f"HF API returned {response.status_code}: Token missing or invalid. Using offline fallback."
                    )
                    return _offline_fallback_embedding(clean_text)

            except Exception as e:
                logger.warning(f"Error querying {url}: {e}")

        time.sleep(2 ** attempt)

    logger.warning("Falling back to local deterministic embedding generator.")
    return _offline_fallback_embedding(clean_text)


def get_batch_embeddings(
    texts: List[str],
    token: Optional[str] = None,
    batch_size: int = 8,
) -> np.ndarray:
    """
    Generate normalized embeddings for a list of texts in batches.

    Args:
        texts: List of text strings.
        token: Optional Hugging Face API token.
        batch_size: Number of texts per batch.

    Returns:
        np.ndarray of shape (N, 384) float32.
    """
    all_embeddings: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for t in batch:
            emb = get_embedding(t, token=token)
            all_embeddings.append(emb)

    return np.vstack(all_embeddings).astype(np.float32)
