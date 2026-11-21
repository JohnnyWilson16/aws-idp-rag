"""
Measure in-Lambda hybrid retrieval latency (FAISS exact search + BM25 + fusion)
on synthetic chunks. Embedding and LLM calls are not included.

    python scripts/bench_retrieval.py
"""

import os
import statistics
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "idp_pipeline"))
sys.path.insert(0, os.path.join(ROOT, "src", "query_api"))
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

import app  # noqa: E402
from faiss_indexer import add_chunks_to_index  # noqa: E402

rng = np.random.default_rng(0)
VOCAB = [f"term{i}" for i in range(5000)]

for n_chunks, dim in [(500, 1024), (5000, 1024), (20000, 384)]:
    metadata = [
        {"chunk_id": str(i), "doc_name": f"doc{i // 20}.pdf", "page_number": i % 20 + 1,
         "text": " ".join(rng.choice(VOCAB, 300))}
        for i in range(n_chunks)
    ]
    index, metadata = add_chunks_to_index(None, [], metadata, rng.standard_normal((n_chunks, dim)).astype(np.float32))
    bm25 = app.BM25.from_metadata(metadata)

    timings = []
    for _ in range(50):
        q_vec = rng.standard_normal(dim).astype(np.float32)
        question = " ".join(rng.choice(VOCAB, 6))
        t0 = time.perf_counter()
        app._retrieve_top_k(index, metadata, q_vec, question, top_k=3, bm25=bm25)
        timings.append((time.perf_counter() - t0) * 1000)
    timings.sort()
    print(f"{n_chunks:>6} chunks x {dim:>4}d: p50 {statistics.median(timings):6.2f} ms   "
          f"p95 {timings[int(len(timings) * 0.95) - 1]:6.2f} ms")
