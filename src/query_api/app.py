"""
Query API Lambda (POST /query).

1. Load the S3 vector store once per warm container. Reload only when its ETag changes.
2. Embed the question with the SAME model that built the index (read from the store manifest).
3. Hybrid retrieval: dense cosine (FAISS) + sparse BM25, fused 50/50.
4. Generate a grounded answer with Amazon Bedrock (Converse API) or Hugging Face.
5. If no LLM is reachable, return an extractive answer built only from retrieved sentences.
"""

import io
import json
import logging
import math
import os
import re
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import boto3
import faiss
import numpy as np
import requests
from botocore.exceptions import ClientError
from embedding_client import EmbeddingError, get_embedding

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

S3_BUCKET_NAME = os.environ.get("DOCUMENT_BUCKET_NAME", "")
DEFAULT_PROVIDER = os.environ.get("MODEL_PROVIDER", "bedrock").lower()
HF_DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")
HF_FALLBACK_LLM_MODELS = ["Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"]
BEDROCK_DEFAULT_LLM_MODEL = os.environ.get("BEDROCK_LLM_MODEL", "us.amazon.nova-micro-v1:0")
BEDROCK_FALLBACK_MODELS = ["us.amazon.nova-micro-v1:0", "us.amazon.nova-lite-v1:0"]
BEDROCK_REGION = os.environ.get("BEDROCK_REGION") or os.environ.get("AWS_REGION") or "us-east-1"

STORE_S3_KEY = "vector_store/store.npz"
MAX_QUESTION_CHARS = 2000
MAX_TOP_K = 10

# Retrieval tuning
HYBRID_ALPHA = 0.5        # weight of dense score; (1 - alpha) goes to BM25
DENSE_ONLY_MIN = 0.5      # a chunk with no keyword match needs at least this cosine score
MIN_SCORE = 0.2           # absolute floor for the fused score
RELATIVE_CUTOFF = 0.5     # drop chunks scoring below 50% of the best chunk
DENSE_CANDIDATES = 200    # dense neighbours fetched from FAISS per query
BM25_K1, BM25_B = 1.5, 0.75

NOT_FOUND = "I cannot find the answer to this question in the provided documents."

STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can", "could",
    "did", "do", "does", "for", "from", "give", "had", "has", "have", "how", "i", "in", "is",
    "it", "its", "me", "my", "of", "on", "or", "our", "please", "shall", "should", "show",
    "tell", "that", "the", "their", "them", "there", "these", "they", "this", "those", "to",
    "was", "we", "were", "what", "when", "where", "which", "who", "whom", "whose", "why",
    "will", "with", "would", "you", "your",
}
GREETINGS = {
    "hello", "hi", "hey", "namaste", "good morning", "good afternoon", "good evening",
    "greetings", "how are you", "who are you", "what can you do", "help", "thanks",
    "thank you", "bye", "goodbye",
}

s3_client = boto3.client("s3")

# Warm-container cache. Keyed by the S3 ETag of the store object.
_STORE: Dict[str, Any] = {}


def _respond(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _tokenize(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 and w not in STOPWORDS]


def _chunk_tokens(meta: Dict[str, Any]) -> List[str]:
    # The file name is part of the searchable text, so "invoice" matches invoice.pdf.
    return _tokenize(f"{meta.get('doc_name', '')} {meta.get('text', '')}")


def _load_store(bucket_name: str) -> Dict[str, Any]:
    """Return the cached store, downloading it only when the S3 object changed."""
    try:
        etag = s3_client.head_object(Bucket=bucket_name, Key=STORE_S3_KEY)["ETag"]
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return {}
        raise
    if _STORE.get("etag") == etag:
        return _STORE

    data = s3_client.get_object(Bucket=bucket_name, Key=STORE_S3_KEY)["Body"].read()
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        index = faiss.deserialize_index(npz["index"])
        metadata = json.loads(npz["metadata"].tobytes().decode("utf-8"))
        manifest = json.loads(npz["manifest"].tobytes().decode("utf-8"))

    _STORE.clear()
    _STORE.update(etag=etag, index=index, metadata=metadata, manifest=manifest, bm25=BM25.from_metadata(metadata))
    logger.info(f"Vector store loaded: {index.ntotal} vectors, manifest={manifest}")
    return _STORE


class BM25:
    """Okapi BM25 over an inverted index built once per store version."""

    def __init__(self, doc_tokens: List[List[str]]):
        self.n = len(doc_tokens)
        doc_len = np.array([len(t) for t in doc_tokens], dtype=np.float32)
        avgdl = float(doc_len.mean()) if self.n else 1.0
        self.norm = BM25_K1 * (1 - BM25_B + BM25_B * doc_len / max(avgdl, 1e-9))
        postings: Dict[str, Tuple[List[int], List[int]]] = defaultdict(lambda: ([], []))
        for i, toks in enumerate(doc_tokens):
            for term, count in Counter(toks).items():
                postings[term][0].append(i)
                postings[term][1].append(count)
        self.postings = {t: (np.array(ids), np.array(tf, dtype=np.float32)) for t, (ids, tf) in postings.items()}

    @classmethod
    def from_metadata(cls, metadata: List[Dict[str, Any]]) -> "BM25":
        return cls([_chunk_tokens(m) for m in metadata])

    def scores(self, query_terms: List[str]) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        for term in set(query_terms):
            if term in self.postings:
                ids, tf = self.postings[term]
                idf = math.log(1 + (self.n - len(ids) + 0.5) / (len(ids) + 0.5))
                out[ids] += idf * tf * (BM25_K1 + 1) / (tf + self.norm[ids])
        return out


def _bm25_scores(query_terms: List[str], doc_tokens: List[List[str]]) -> List[float]:
    return BM25(doc_tokens).scores(query_terms).tolist()


def _retrieve_top_k(
    index: Optional[faiss.Index],
    metadata_list: List[Dict[str, Any]],
    query_vector: Optional[np.ndarray],
    query_text: str,
    top_k: int = 3,
    bm25: Optional[BM25] = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval. Fused score = alpha * cosine + (1 - alpha) * BM25 / max(BM25).
    If query_vector is None (embedding provider down), it runs BM25 only.
    """
    n = len(metadata_list)
    if not n:
        return []
    bm25 = bm25 or BM25.from_metadata(metadata_list)

    dense = np.zeros(n, dtype=np.float32)
    use_dense = query_vector is not None and index is not None and index.ntotal > 0
    if use_dense:
        q = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        if q.shape[1] != index.d:
            raise ValueError(f"Query vector has {q.shape[1]} dims, index has {index.d}")
        q /= np.linalg.norm(q) or 1.0
        # Exact search, top DENSE_CANDIDATES only. Keyword hits outside that list get dense = 0.
        # ponytail: IndexFlatIP is exact O(n); switch to IndexHNSWFlat past ~100k chunks
        distances, indices = index.search(q, min(DENSE_CANDIDATES, index.ntotal))
        valid = (indices[0] >= 0) & (indices[0] < n)
        dense[indices[0][valid]] = np.maximum(distances[0][valid], 0.0)

    sparse = bm25.scores(_tokenize(query_text))
    if sparse.max(initial=0.0) > 0:
        sparse /= sparse.max()

    fused = HYBRID_ALPHA * dense + (1 - HYBRID_ALPHA) * sparse if use_dense else sparse
    eligible = np.flatnonzero((sparse > 0) | (dense >= DENSE_ONLY_MIN))
    if not len(eligible):
        return []
    ranked = eligible[np.argsort(-fused[eligible], kind="stable")]
    cutoff = max(MIN_SCORE, float(fused[ranked[0]]) * RELATIVE_CUTOFF)

    results, seen = [], set()
    for i in ranked:
        if fused[i] < cutoff or len(results) >= top_k:
            break
        meta = metadata_list[i]
        key = meta.get("chunk_id") or f"{meta.get('doc_name')}_{i}"
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "doc_name": meta.get("doc_name", "unknown_document"),
            "page_number": meta.get("page_number", 1),
            "snippet": meta.get("text", ""),
            "relevance_score": round(float(fused[i]), 4),
            "dense_score": round(float(dense[i]), 4),
            "sparse_score": round(float(sparse[i]), 4),
        })
    return results


def _extractive_answer(question: str, sources: List[Dict[str, Any]]) -> str:
    """
    No-LLM fallback: return the source sentences that share the most query terms,
    each with its citation. It never writes new text, so it cannot hallucinate.
    """
    terms = set(_tokenize(question))
    scored = []
    for rank, src in enumerate(sources):
        text = re.sub(r"\s+", " ", src.get("snippet", "")).strip()
        for sent in re.split(r"(?<=[.!?])\s+", text):
            overlap = len(terms.intersection(_tokenize(sent)))
            if overlap and len(sent) > 20:
                scored.append((overlap, -rank, sent, src))
    if not scored:
        return NOT_FOUND
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    lines = [f"- {sent} [Document: {src['doc_name']}, Page: {src['page_number']}]" for _, _, sent, src in scored[:3]]
    return "Relevant excerpts from your documents:\n" + "\n".join(lines)


def _call_bedrock_llm(prompt_system: str, prompt_user: str, model_id: str, max_tokens: int = 512) -> Optional[str]:
    """Bedrock Converse API. Tries the requested model, then cheap Nova fallbacks."""
    candidates = [model_id]
    if not model_id.startswith(("us.", "eu.", "apac.", "global.")):
        candidates.append(f"us.{model_id}")  # cross-region inference profile
    candidates += [m for m in BEDROCK_FALLBACK_MODELS if m not in candidates]

    bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    for mid in candidates:
        try:
            response = bedrock.converse(
                modelId=mid,
                system=[{"text": prompt_system}],
                messages=[{"role": "user", "content": [{"text": prompt_user}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": 0.1},
            )
            content = response.get("output", {}).get("message", {}).get("content", [])
            if content and "text" in content[0]:
                return content[0]["text"].strip()
        except Exception as e:
            logger.warning(f"Bedrock Converse failed for {mid}: {e}")
    return None


def _call_huggingface_llm(
    prompt_system: str, prompt_user: str, model_id: str, hf_token: Optional[str] = None, max_tokens: int = 512
) -> Optional[str]:
    """Hugging Face router, OpenAI-compatible chat completions."""
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        return None
    for model in [model_id] + [m for m in HF_FALLBACK_LLM_MODELS if m != model_id]:
        try:
            res = requests.post(
                "https://router.huggingface.co/v1/chat/completions",
                headers={"Authorization": f"Bearer {token.strip()}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": prompt_system},
                        {"role": "user", "content": prompt_user},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.1,
                },
                timeout=20,
            )
            if res.status_code == 200:
                choices = res.json().get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "").strip()
            logger.warning(f"HF chat failed for {model}: HTTP {res.status_code}")
        except requests.RequestException as e:
            logger.warning(f"HF chat error for {model}: {e}")
    return None


SYSTEM_PROMPT = (
    "You answer questions about the user's documents.\n"
    "Use ONLY the document excerpts provided. Do not use outside knowledge.\n"
    f"If the excerpts do not contain the answer, reply exactly: '{NOT_FOUND}'\n"
    "Cite every claim as [Document: <name>, Page: <number>]."
)


def _build_user_prompt(question: str, sources: List[Dict[str, Any]]) -> str:
    blocks = [
        f"[Excerpt {i}] Document: {s['doc_name']} | Page: {s['page_number']}\n{s['snippet']}"
        for i, s in enumerate(sources, start=1)
    ]
    return "--- EXCERPTS ---\n" + "\n\n".join(blocks) + f"\n\n--- QUESTION ---\n{question}"


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        raw = event.get("body") or "{}"
        body = json.loads(raw) if isinstance(raw, str) else raw
        question = str(body.get("question", "")).strip()
        top_k = max(1, min(int(body.get("top_k", 3)), MAX_TOP_K))
    except (ValueError, TypeError):
        return _respond(400, {"error": "Body must be JSON with a 'question' string and an optional integer 'top_k'"})
    if not question:
        return _respond(400, {"error": "Field 'question' is required"})
    if len(question) > MAX_QUESTION_CHARS:
        return _respond(400, {"error": f"'question' is longer than {MAX_QUESTION_CHARS} characters"})

    provider = (body.get("provider") or DEFAULT_PROVIDER).lower()
    user_token = body.get("hf_token")
    default_model = BEDROCK_DEFAULT_LLM_MODEL if provider == "bedrock" else HF_DEFAULT_LLM_MODEL
    requested_model = body.get("model") or body.get("model_id") or default_model
    base = {"question": question, "provider": provider}

    if re.sub(r"[^\w\s]", "", question.lower()) in GREETINGS:
        return _respond(200, {
            **base,
            "answer": "Hi! Ask me a question about your uploaded documents. I answer with document and page citations.",
            "sources": [], "chunks_retrieved": 0, "model_used": "greeting",
        })

    store = _load_store(S3_BUCKET_NAME)
    if not store or store["index"].ntotal == 0:
        return _respond(200, {
            **base, "answer": "No documents are indexed yet. Upload a document first.",
            "sources": [], "chunks_retrieved": 0, "model_used": "none",
        })

    # Embed with the model that built the index, whatever LLM provider the user picked.
    retrieval_mode = "hybrid"
    t0 = time.perf_counter()
    try:
        query_vector = get_embedding(question, provider=store["manifest"]["embedding_provider"], token=user_token)
    except (EmbeddingError, ValueError) as e:
        logger.warning(f"Query embedding failed, using BM25 only: {e}")
        query_vector, retrieval_mode = None, "sparse-only"
    t1 = time.perf_counter()
    sources = _retrieve_top_k(
        store["index"], store["metadata"], query_vector, question, top_k=top_k, bm25=store["bm25"]
    )
    t2 = time.perf_counter()

    answer, model_used = None, requested_model
    if sources:
        user_prompt = _build_user_prompt(question, sources)
        if provider == "bedrock":
            answer = _call_bedrock_llm(SYSTEM_PROMPT, user_prompt, requested_model)
        if not answer and (provider != "bedrock" or user_token or os.environ.get("HF_TOKEN")):
            hf_model = requested_model if provider != "bedrock" else HF_DEFAULT_LLM_MODEL
            answer = _call_huggingface_llm(SYSTEM_PROMPT, user_prompt, hf_model, hf_token=user_token)
            model_used = hf_model
        if not answer:
            answer, model_used = _extractive_answer(question, sources), "extractive-fallback"
    else:
        answer, model_used = NOT_FOUND, "none"
    t3 = time.perf_counter()

    timings = {
        "embed_ms": round((t1 - t0) * 1000, 2),
        "retrieve_ms": round((t2 - t1) * 1000, 2),
        "generate_ms": round((t3 - t2) * 1000, 2),
        "total_ms": round((t3 - started) * 1000, 2),
    }
    # One structured line per query: CloudWatch Logs Insights can chart these fields.
    logger.info(json.dumps({
        "event": "query", "provider": provider, "model_used": model_used,
        "retrieval_mode": retrieval_mode, "chunks": len(sources), **timings,
    }))
    return _respond(200, {
        **base, "answer": answer, "sources": sources, "chunks_retrieved": len(sources),
        "model_used": model_used, "retrieval_mode": retrieval_mode, "timings_ms": timings,
    })
