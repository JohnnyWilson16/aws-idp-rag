"""
FAISS Vector Index Manager for S3-backed Serverless Storage and Retrieval.

Features:
- Inner Product (cosine similarity on L2-normalized vectors) FAISS Flat index.
- Serialization of binary index and chunk metadata to Amazon S3.
- In-memory / /tmp caching for low-latency AWS Lambda execution.
- Vector upsert, deduplication, and Top-K similarity search.
"""

import io
import json
import logging
import os
import pickle
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import faiss
import numpy as np
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

VECTOR_DIM = int(os.environ.get("VECTOR_DIM", "384"))
INDEX_S3_KEY = "vector_store/index.faiss"
METADATA_S3_KEY = "vector_store/metadata.pkl"


def get_s3_client(region_name: Optional[str] = None) -> Any:
    """Instantiate or reuse boto3 S3 client."""
    return boto3.client("s3", region_name=region_name)


def create_empty_index(dim: int = VECTOR_DIM) -> faiss.IndexFlatIP:
    """Create a new FAISS Flat Inner Product index for dense vectors."""
    return faiss.IndexFlatIP(dim)


def load_index_from_s3(
    bucket_name: str,
    s3_client: Optional[Any] = None,
    local_cache_dir: str = "/tmp",
) -> Tuple[faiss.IndexFlatIP, List[Dict[str, Any]]]:
    """
    Download and deserialize the FAISS index and metadata from S3.
    If the index does not exist yet in S3, initializes a new empty index and metadata list.

    Args:
        bucket_name: Target S3 bucket name.
        s3_client: Optional boto3 S3 client.
        local_cache_dir: Local path (e.g. /tmp in Lambda) for temporary storage.

    Returns:
        Tuple of (faiss_index, metadata_list)
    """
    s3 = s3_client or get_s3_client()
    local_index_path = os.path.join(local_cache_dir, "index.faiss")
    local_meta_path = os.path.join(local_cache_dir, "metadata.pkl")

    try:
        logger.info(f"Checking for existing vector store in s3://{bucket_name}/{INDEX_S3_KEY}")
        s3.download_file(bucket_name, INDEX_S3_KEY, local_index_path)
        s3.download_file(bucket_name, METADATA_S3_KEY, local_meta_path)

        index = faiss.read_index(local_index_path)
        with open(local_meta_path, "rb") as f:
            metadata_list = pickle.load(f)

        logger.info(f"Loaded existing index with {index.ntotal} vectors and {len(metadata_list)} metadata entries")
        return index, metadata_list

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code in ("404", "NoSuchKey", "NotFound"):
            logger.info("No existing index found in S3. Creating a new empty index.")
            return create_empty_index(VECTOR_DIM), []
        logger.error(f"Failed to load index from S3: {e}")
        raise


def save_index_to_s3(
    index: faiss.IndexFlatIP,
    metadata_list: List[Dict[str, Any]],
    bucket_name: str,
    s3_client: Optional[Any] = None,
    local_cache_dir: str = "/tmp",
) -> None:
    """
    Serialize the FAISS index and metadata and upload them to Amazon S3.

    Args:
        index: FAISS index object.
        metadata_list: List of metadata dictionaries corresponding to vectors.
        bucket_name: Target S3 bucket name.
        s3_client: Optional boto3 S3 client.
        local_cache_dir: Local path for temporary files.
    """
    s3 = s3_client or get_s3_client()
    local_index_path = os.path.join(local_cache_dir, "index.faiss")
    local_meta_path = os.path.join(local_cache_dir, "metadata.pkl")

    # Write binary index and pickle metadata
    faiss.write_index(index, local_index_path)
    with open(local_meta_path, "wb") as f:
        pickle.dump(metadata_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"Uploading updated index ({index.ntotal} vectors) to s3://{bucket_name}/{INDEX_S3_KEY}")
    s3.upload_file(local_index_path, bucket_name, INDEX_S3_KEY)
    s3.upload_file(local_meta_path, bucket_name, METADATA_S3_KEY)
    logger.info("Vector store successfully saved to S3.")


def add_chunks_to_index(
    index: faiss.IndexFlatIP,
    metadata_list: List[Dict[str, Any]],
    new_chunks: List[Dict[str, Any]],
    new_embeddings: np.ndarray,
) -> Tuple[faiss.IndexFlatIP, List[Dict[str, Any]]]:
    """
    Add newly generated embeddings and metadata to an existing or fresh index.

    Args:
        index: Target FAISS index.
        metadata_list: Existing metadata list.
        new_chunks: List of chunk dictionaries.
        new_embeddings: np.ndarray of shape (N, 384) float32.

    Returns:
        Tuple of updated (index, metadata_list)
    """
    if len(new_chunks) == 0:
        return index, metadata_list

    # Ensure float32 format
    embeddings = np.ascontiguousarray(new_embeddings, dtype=np.float32)
    dim = embeddings.shape[1]

    # Reinitialize if index dimension does not match incoming vectors
    if index is None or index.d != dim:
        logger.info(f"Adapting FAISS index dimension to match vectors: {dim}")
        index = create_empty_index(dim)
        metadata_list = []

    # Normalize vectors to unit length for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_embeddings = embeddings / norms

    # Add to FAISS index
    index.add(normalized_embeddings)
    metadata_list.extend(new_chunks)

    logger.info(f"Index now contains {index.ntotal} vectors after adding {len(new_chunks)} chunks")
    return index, metadata_list


def search_index(
    index: faiss.IndexFlatIP,
    metadata_list: List[Dict[str, Any]],
    query_vector: np.ndarray,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Search index for top-k closest chunks to query vector using cosine similarity.

    Args:
        index: FAISS index object.
        metadata_list: List of metadata dicts corresponding to vector IDs.
        query_vector: Query embedding np.ndarray of shape (384,).
        top_k: Number of nearest neighbors to retrieve.

    Returns:
        List of matching result dictionaries sorted by similarity score:
        [
            {
                "score": 0.892,
                "chunk_id": "contract.pdf_p2_c1",
                "doc_name": "contract.pdf",
                "page_number": 2,
                "text": "..."
            },
            ...
        ]
    """
    if index.ntotal == 0 or len(metadata_list) == 0:
        logger.warning("Search called on empty index")
        return []

    # Normalize query vector
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
            "score": round(float(dist), 4),
            "chunk_id": meta.get("chunk_id", f"chunk_{idx}"),
            "doc_name": meta.get("doc_name", "unknown"),
            "page_number": meta.get("page_number", 1),
            "text": meta.get("text", ""),
        })

    return results
