"""
Embedding providers shared by the indexing and query Lambdas.

This file exists twice (src/idp_pipeline/embedding_utils.py and
src/query_api/embedding_client.py) because each Lambda is packaged on its own.
tests/test_embeddings.py fails if the two copies drift apart.

Providers:
- bedrock     Amazon Titan Text Embeddings V2 (256, 512 or 1024 dims)
- huggingface sentence-transformers/all-MiniLM-L6-v2 via HF Inference (384 dims)
- local-hash  Deterministic hashed bag-of-words. No network. For offline demos and tests.

A provider failure raises EmbeddingError. Nothing falls back silently to another
model: vectors from different models are not comparable, so a silent fallback
corrupts the index.
"""

import hashlib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import boto3
import numpy as np
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BEDROCK_EMBEDDING_MODEL = os.environ.get("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
HF_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
HF_EMBEDDING_DIM = 384
LOCAL_HASH_DIM = 384
TITAN_V2_DIMS = (256, 512, 1024)
REGION = os.environ.get("BEDROCK_REGION") or os.environ.get("AWS_REGION") or "us-east-1"

_PROVIDER_ALIASES = {
    "bedrock": "bedrock",
    "huggingface": "huggingface",
    "opensource": "huggingface",
    "local-hash": "local-hash",
    "local": "local-hash",
    "local-extractive": "local-hash",
}

_STOPWORDS = {
    "what", "is", "the", "in", "and", "or", "to", "for", "of", "a", "an", "on",
    "with", "at", "by", "from", "it", "this", "that", "these", "those", "how", "who",
}


class EmbeddingError(RuntimeError):
    """The configured embedding provider could not produce a vector."""


def resolve_provider(provider: Optional[str] = None) -> str:
    name = (provider or os.environ.get("MODEL_PROVIDER") or "bedrock").lower()
    if name not in _PROVIDER_ALIASES:
        raise ValueError(f"Unknown embedding provider: {name}")
    return _PROVIDER_ALIASES[name]


def embedding_manifest(provider: Optional[str] = None) -> Dict[str, Any]:
    """Identify the vector space. The store refuses to mix vectors from different manifests."""
    p = resolve_provider(provider)
    if p == "bedrock":
        dim = int(os.environ.get("VECTOR_DIM", "1024"))
        return {
            "embedding_provider": p,
            "embedding_model": BEDROCK_EMBEDDING_MODEL,
            "dim": dim if dim in TITAN_V2_DIMS else 1024,
        }
    if p == "huggingface":
        return {"embedding_provider": p, "embedding_model": HF_EMBEDDING_MODEL, "dim": HF_EMBEDDING_DIM}
    return {"embedding_provider": p, "embedding_model": "hashed-bow-v1", "dim": LOCAL_HASH_DIM}


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return (vec / norm if norm > 0 else vec).astype(np.float32)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def local_hash_embedding(text: str, dim: int = LOCAL_HASH_DIM) -> np.ndarray:
    """Hashed unigrams + bigrams. Lexical only, but deterministic and free."""
    vec = np.zeros(dim, dtype=np.float32)
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS and len(w) > 1]
    for w in words:
        vec[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 2.0
    for a, b in zip(words, words[1:]):
        vec[int(hashlib.md5(f"{a}_{b}".encode()).hexdigest(), 16) % dim] += 1.5
    return _normalize(vec)


def _bedrock_embedding(client: Any, text: str, manifest: Dict[str, Any]) -> np.ndarray:
    response = client.invoke_model(
        modelId=manifest["embedding_model"],
        body=json.dumps({"inputText": text, "dimensions": manifest["dim"], "normalize": True}),
        contentType="application/json",
        accept="application/json",
    )
    return _normalize(np.array(json.loads(response["body"].read())["embedding"], dtype=np.float32))


def _hf_embeddings(texts: List[str], token: Optional[str], retries: int = 3) -> List[np.ndarray]:
    """One HTTP call per batch. The feature-extraction endpoint accepts a list of inputs."""
    api_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not api_token:
        raise EmbeddingError("Hugging Face provider needs HF_TOKEN")
    url = f"https://router.huggingface.co/hf-inference/models/{HF_EMBEDDING_MODEL}/pipeline/feature-extraction"
    headers = {"Authorization": f"Bearer {api_token.strip()}"}
    last_error = ""
    for attempt in range(retries):
        try:
            res = requests.post(url, headers=headers, json={"inputs": texts}, timeout=30)
            if res.status_code == 200:
                arr = np.array(res.json(), dtype=np.float32)
                if arr.ndim == 3:  # token-level output: mean-pool
                    arr = arr.mean(axis=1)
                if arr.shape != (len(texts), HF_EMBEDDING_DIM):
                    raise EmbeddingError(f"Unexpected HF embedding shape {arr.shape}")
                return [_normalize(v) for v in arr]
            last_error = f"HTTP {res.status_code}: {res.text[:200]}"
        except requests.RequestException as e:
            last_error = str(e)
        time.sleep(2**attempt)
    raise EmbeddingError(f"Hugging Face embedding failed: {last_error}")


def get_batch_embeddings(
    texts: List[str],
    provider: Optional[str] = None,
    token: Optional[str] = None,
    batch_size: int = 16,
    max_workers: int = 8,
) -> np.ndarray:
    """Embed many texts. Returns an (n, dim) float32 array of unit vectors."""
    manifest = embedding_manifest(provider)
    p = manifest["embedding_provider"]
    texts = [_clean(t) or " " for t in texts]

    if p == "local-hash":
        vectors = [local_hash_embedding(t, manifest["dim"]) for t in texts]
    elif p == "huggingface":
        vectors = []
        for i in range(0, len(texts), batch_size):
            vectors.extend(_hf_embeddings(texts[i : i + batch_size], token))
    else:
        # Titan has no batch API: run requests concurrently (I/O bound).
        client = boto3.client("bedrock-runtime", region_name=REGION)
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                vectors = list(pool.map(lambda t: _bedrock_embedding(client, t, manifest), texts))
        except Exception as e:
            raise EmbeddingError(f"Bedrock embedding failed: {e}") from e

    if not vectors:
        return np.zeros((0, manifest["dim"]), dtype=np.float32)
    return np.vstack(vectors).astype(np.float32)


def get_embedding(text: str, provider: Optional[str] = None, token: Optional[str] = None) -> np.ndarray:
    return get_batch_embeddings([text], provider=provider, token=token)[0]
