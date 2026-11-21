"""Vector store concurrency, BM25 hybrid retrieval, and grounding behavior."""

import filecmp
import os

import app
import faiss_indexer
import numpy as np
import pytest
from botocore.exceptions import ClientError
from embedding_utils import get_batch_embeddings
from faiss_indexer import ManifestMismatchError, load_store, upsert_document

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST = {"embedding_provider": "local-hash", "embedding_model": "hashed-bow-v1", "dim": 384}


class FakeS3:
    """In-memory S3 with ETags and conditional PutObject (If-Match / If-None-Match)."""

    def __init__(self):
        self.objects = {}
        self.version = 0
        self.before_put = None  # hook to simulate a concurrent writer

    def _err(self, code):
        return ClientError({"Error": {"Code": code}}, "S3")

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise self._err("NoSuchKey")
        body, etag = self.objects[Key]
        return {"Body": type("B", (), {"read": lambda _s: body})(), "ETag": etag}

    def put_object(self, Bucket, Key, Body, IfMatch=None, IfNoneMatch=None, **_):
        if self.before_put:
            hook, self.before_put = self.before_put, None
            hook()
        current = self.objects.get(Key)
        if IfNoneMatch == "*" and current is not None:
            raise self._err("PreconditionFailed")
        if IfMatch is not None and (current is None or current[1] != IfMatch):
            raise self._err("PreconditionFailed")
        self.version += 1
        self.objects[Key] = (Body, f'"v{self.version}"')


def _chunks(doc, texts):
    return [{"chunk_id": f"{doc}_{i}", "doc_name": doc, "page_number": i + 1, "text": t} for i, t in enumerate(texts)]


def _embed(texts):
    return get_batch_embeddings(texts, provider="local-hash")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(faiss_indexer.time, "sleep", lambda _s: None)


def test_store_roundtrip_and_reupload_replaces_old_chunks():
    s3 = FakeS3()
    a = _chunks("a.pdf", ["alpha clause one", "alpha clause two"])
    upsert_document("b", "a.pdf", a, _embed([c["text"] for c in a]), MANIFEST, s3)
    b = _chunks("b.pdf", ["beta invoice total"])
    upsert_document("b", "b.pdf", b, _embed([c["text"] for c in b]), MANIFEST, s3)

    a2 = _chunks("a.pdf", ["alpha revised"])
    total = upsert_document("b", "a.pdf", a2, _embed([c["text"] for c in a2]), MANIFEST, s3)

    index, meta, manifest, _ = load_store("b", s3)
    assert total == index.ntotal == len(meta) == 2
    assert sorted(m["text"] for m in meta) == ["alpha revised", "beta invoice total"]
    assert manifest == MANIFEST


def test_concurrent_writer_is_not_lost():
    s3 = FakeS3()
    a = _chunks("a.pdf", ["alpha"])
    b = _chunks("b.pdf", ["beta"])
    # While writer A is between read and write, writer B commits first.
    s3.before_put = lambda: upsert_document("b", "b.pdf", b, _embed(["beta"]), MANIFEST, s3)
    upsert_document("b", "a.pdf", a, _embed(["alpha"]), MANIFEST, s3)

    _, meta, _, _ = load_store("b", s3)
    assert sorted(m["doc_name"] for m in meta) == ["a.pdf", "b.pdf"]


def test_mixed_embedding_models_are_rejected():
    s3 = FakeS3()
    a = _chunks("a.pdf", ["alpha"])
    upsert_document("b", "a.pdf", a, _embed(["alpha"]), MANIFEST, s3)
    titan = {"embedding_provider": "bedrock", "embedding_model": "amazon.titan-embed-text-v2:0", "dim": 1024}
    with pytest.raises(ManifestMismatchError):
        upsert_document("b", "c.pdf", _chunks("c.pdf", ["x"]), np.ones((1, 1024), np.float32), titan, s3)


def test_bm25_prefers_rare_term_and_shorter_doc():
    docs = [["invoice", "total", "due"], ["invoice", "invoice", "payment"], ["liability", "cap"]]
    scores = app._bm25_scores(["liability"], docs)
    assert scores[2] > 0 and scores[0] == scores[1] == 0
    scores = app._bm25_scores(["invoice", "due"], docs)
    assert scores[0] > scores[1] > scores[2] == 0


def test_hybrid_retrieval_and_refusal():
    texts = [
        "Aggregate liability shall not exceed $500,000 USD.",
        "Payment terms are Net 30 days. Late fees are 1.5% per month.",
        "The provider guarantees 99.9% monthly uptime.",
    ]
    meta = _chunks("msa.pdf", texts)
    index, meta = faiss_indexer.add_chunks_to_index(None, [], meta, _embed(texts))

    q = "What is the liability cap?"
    results = app._retrieve_top_k(index, meta, _embed([q])[0], q, top_k=3)
    assert results[0]["page_number"] == 1
    assert results[0]["sparse_score"] == 1.0

    q = "What is the capital of Mars?"
    assert app._retrieve_top_k(index, meta, _embed([q])[0], q) == []


def test_extractive_answer_only_quotes_sources():
    src = [{"doc_name": "msa.pdf", "page_number": 3, "snippet": "Intro text here. Payment terms are Net 30 days from invoice."}]
    answer = app._extractive_answer("What are the payment terms?", src)
    assert "Payment terms are Net 30 days from invoice." in answer
    assert "[Document: msa.pdf, Page: 3]" in answer
    assert app._extractive_answer("unrelated zebra", src) == app.NOT_FOUND


def test_embedding_module_copies_are_identical():
    assert filecmp.cmp(
        os.path.join(ROOT, "src", "idp_pipeline", "embedding_utils.py"),
        os.path.join(ROOT, "src", "query_api", "embedding_client.py"),
        shallow=False,
    ), "Copy src/idp_pipeline/embedding_utils.py to src/query_api/embedding_client.py"


def test_query_handler_validates_input():
    assert app.lambda_handler({"body": "not json"}, None)["statusCode"] == 400
    assert app.lambda_handler({"body": '{"question": ""}'}, None)["statusCode"] == 400
    assert app.lambda_handler({"body": '{"question": "' + "x" * 3000 + '"}'}, None)["statusCode"] == 400


def test_index_handler_writes_store_and_marks_completed(monkeypatch):
    import json as _json

    import handler

    s3 = FakeS3()
    pages = {"pages": {"1": {"page_number": 1, "full_text": "Payment terms are Net 30 days.",
                             "lines": [{"text": "Payment terms are Net 30 days."}]}}}
    s3.objects["extracted/doc.pdf.json"] = (_json.dumps(pages).encode(), '"x"')
    updates = []
    monkeypatch.setattr(handler, "s3_client", s3)
    monkeypatch.setattr(handler, "table", type("T", (), {"update_item": lambda _s, **kw: updates.append(kw)})())
    monkeypatch.setenv("MODEL_PROVIDER", "local-hash")

    out = handler.handle_process_and_index(
        {"job_id": "FALLBACK_1", "document_id": "doc.pdf", "s3_bucket": "b", "filename": "doc.pdf"}
    )
    assert out["status"] == "COMPLETED" and out["chunk_count"] == 1
    _, meta, manifest, _ = load_store("b", s3)
    assert meta[0]["doc_name"] == "doc.pdf" and manifest["embedding_provider"] == "local-hash"
    assert updates[-1]["ExpressionAttributeValues"][":st"] == "COMPLETED"
