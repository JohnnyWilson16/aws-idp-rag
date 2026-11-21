"""
S3-backed FAISS vector store.

The whole store (FAISS index + chunk metadata + embedding manifest) lives in ONE
S3 object, so a reader never sees an index and metadata from different writes.
Writers use S3 conditional writes (If-Match / If-None-Match) and retry on
conflict, so two documents indexed at the same time cannot overwrite each other.

The object is an .npz file loaded with allow_pickle=False: no pickle is ever
deserialized from S3.
"""

import io
import json
import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import boto3
import faiss
import numpy as np
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

STORE_S3_KEY = "vector_store/store.npz"
MAX_WRITE_ATTEMPTS = 6

Manifest = Dict[str, Any]  # {"embedding_provider", "embedding_model", "dim"}


class ManifestMismatchError(ValueError):
    """The store was built with a different embedding model than the new vectors."""


def create_empty_index(dim: int) -> faiss.IndexFlatIP:
    """Exact inner-product index. With L2-normalized vectors, IP equals cosine similarity."""
    return faiss.IndexFlatIP(dim)


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def serialize_store(index: faiss.Index, metadata: List[Dict[str, Any]], manifest: Manifest) -> bytes:
    buf = io.BytesIO()
    np.savez(
        buf,
        index=faiss.serialize_index(index),
        metadata=np.frombuffer(json.dumps(metadata).encode("utf-8"), dtype=np.uint8),
        manifest=np.frombuffer(json.dumps(manifest).encode("utf-8"), dtype=np.uint8),
    )
    return buf.getvalue()


def deserialize_store(data: bytes) -> Tuple[faiss.Index, List[Dict[str, Any]], Manifest]:
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        index = faiss.deserialize_index(npz["index"])
        metadata = json.loads(npz["metadata"].tobytes().decode("utf-8"))
        manifest = json.loads(npz["manifest"].tobytes().decode("utf-8"))
    if index.ntotal != len(metadata):
        raise ValueError(f"Corrupt store: {index.ntotal} vectors but {len(metadata)} metadata rows")
    return index, metadata, manifest


def load_store(
    bucket_name: str, s3_client: Optional[Any] = None
) -> Tuple[Optional[faiss.Index], List[Dict[str, Any]], Optional[Manifest], Optional[str]]:
    """Return (index, metadata, manifest, etag). All None/empty if the store does not exist yet."""
    s3 = s3_client or boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=STORE_S3_KEY)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return None, [], None, None
        raise
    index, metadata, manifest = deserialize_store(obj["Body"].read())
    return index, metadata, manifest, obj["ETag"]


def add_chunks_to_index(
    index: Optional[faiss.Index],
    metadata_list: List[Dict[str, Any]],
    new_chunks: List[Dict[str, Any]],
    new_embeddings: np.ndarray,
    replace_doc: Optional[str] = None,
) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """
    Add vectors and metadata. If replace_doc is set, first drop every existing
    chunk of that document so a re-upload does not create duplicates.
    """
    vectors = _normalize_rows(new_embeddings)
    if len(new_chunks) != vectors.shape[0]:
        raise ValueError(f"{len(new_chunks)} chunks but {vectors.shape[0]} embeddings")
    dim = vectors.shape[1]

    if index is None:
        index = create_empty_index(dim)
    elif index.d != dim:
        raise ManifestMismatchError(f"Index dimension is {index.d}, new vectors are {dim}. Re-index the store.")

    if replace_doc is not None and index.ntotal:
        keep = [i for i, m in enumerate(metadata_list) if m.get("doc_name") != replace_doc]
        if len(keep) < index.ntotal:
            kept_vectors = index.reconstruct_n(0, index.ntotal)[keep]
            index = create_empty_index(dim)
            if len(keep):
                index.add(kept_vectors)
            metadata_list = [metadata_list[i] for i in keep]
            logger.info(f"Removed previous chunks of {replace_doc}; {len(keep)} vectors kept")

    index.add(vectors)
    metadata_list = list(metadata_list) + list(new_chunks)
    return index, metadata_list


def upsert_document(
    bucket_name: str,
    doc_name: str,
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    manifest: Manifest,
    s3_client: Optional[Any] = None,
) -> int:
    """
    Read-modify-write the store with optimistic concurrency.
    Returns the total vector count after the write.
    """
    s3 = s3_client or boto3.client("s3")
    for attempt in range(1, MAX_WRITE_ATTEMPTS + 1):
        index, metadata, current_manifest, etag = load_store(bucket_name, s3)
        if current_manifest is not None and current_manifest != manifest:
            raise ManifestMismatchError(
                f"Store was built with {current_manifest}, new vectors use {manifest}. "
                "Run scripts/reindex_store.py to rebuild it with one embedding model."
            )
        index, metadata = add_chunks_to_index(index, metadata, chunks, embeddings, replace_doc=doc_name)

        condition = {"IfMatch": etag} if etag else {"IfNoneMatch": "*"}
        try:
            s3.put_object(
                Bucket=bucket_name,
                Key=STORE_S3_KEY,
                Body=serialize_store(index, metadata, manifest),
                ContentType="application/octet-stream",
                **condition,
            )
            logger.info(f"Store saved: {index.ntotal} vectors (attempt {attempt})")
            return index.ntotal
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code not in ("PreconditionFailed", "ConditionalRequestConflict"):
                raise
            logger.warning(f"Concurrent store write detected ({code}); retrying ({attempt}/{MAX_WRITE_ATTEMPTS})")
            time.sleep(min(2**attempt, 20) * random.uniform(0.5, 1.0))
    raise RuntimeError(f"Could not write vector store after {MAX_WRITE_ATTEMPTS} attempts")


def search_index(
    index: faiss.Index,
    metadata_list: List[Dict[str, Any]],
    query_vector: np.ndarray,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Dense-only top-k search. Used by the local CLI runner."""
    if index is None or index.ntotal == 0:
        return []
    q_vec = _normalize_rows(query_vector.reshape(1, -1))
    distances, indices = index.search(q_vec, min(top_k, index.ntotal))
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(metadata_list):
            meta = metadata_list[idx]
            results.append({
                "score": round(float(dist), 4),
                "chunk_id": meta.get("chunk_id", f"chunk_{idx}"),
                "doc_name": meta.get("doc_name", "unknown"),
                "page_number": meta.get("page_number", 1),
                "text": meta.get("text", ""),
            })
    return results
