"""
Run the IDP + RAG pipeline locally on PDFs, with no deployed stack.

It uses the same chunking, embedding, indexing, retrieval and answer code as
the Lambdas. Only OCR differs: it reads the PDF text layer with pypdf.

    python scripts/local_pipeline_runner.py --provider local      # offline, no credentials
    python scripts/local_pipeline_runner.py --provider bedrock    # needs AWS credentials + Bedrock access
    python scripts/local_pipeline_runner.py --provider huggingface --hf-token hf_...
"""

import argparse
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "idp_pipeline"))
sys.path.insert(0, os.path.join(ROOT, "src", "query_api"))
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

import app  # noqa: E402  (query Lambda)
from embedding_utils import get_batch_embeddings  # noqa: E402
from faiss_indexer import add_chunks_to_index  # noqa: E402
from pypdf import PdfReader  # noqa: E402
from textract_utils import chunk_document_pages  # noqa: E402

SAMPLES = [os.path.join(ROOT, "samples", f) for f in ("master_service_agreement.pdf", "cloud_services_invoice.pdf")]
QUESTIONS = [
    "What is the aggregate liability cap in the Master Services Agreement?",
    "What are the payment terms and late fees on invoices?",
    "What uptime percentage does the service level agreement commit to?",
    "What is the total amount due on the invoice?",
    "What is the capital of Mars?",
]


def extract_pages(pdf_path: str) -> dict:
    pages = {}
    for n, page in enumerate(PdfReader(pdf_path).pages, start=1):
        lines = [ln.strip() for ln in (page.extract_text() or "").splitlines() if ln.strip()]
        pages[n] = {"page_number": n, "lines": lines, "full_text": "\n".join(lines)}
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", choices=["local", "bedrock", "huggingface"], default="local")
    parser.add_argument("--model", help="LLM model ID (bedrock or huggingface only)")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("pdfs", nargs="*", default=SAMPLES)
    args = parser.parse_args()

    index, metadata = None, []
    for path in args.pdfs:
        doc_name = os.path.basename(path)
        chunks = chunk_document_pages(extract_pages(path), doc_name=doc_name, max_chunk_chars=2000, overlap_chars=200)
        texts = [f"Document: {c['doc_name']} Page: {c['page_number']}\n{c['text']}" for c in chunks]
        embeddings = get_batch_embeddings(texts, provider=args.provider, token=args.hf_token)
        index, metadata = add_chunks_to_index(index, metadata, chunks, embeddings, replace_doc=doc_name)
        print(f"Indexed {doc_name}: {len(chunks)} chunks")
    print(f"Index: {index.ntotal} vectors, dim={index.d}\n")

    bm25 = app.BM25.from_metadata(metadata)
    for question in QUESTIONS:
        q_vec = get_batch_embeddings([question], provider=args.provider, token=args.hf_token)[0]
        t0 = time.perf_counter()
        sources = app._retrieve_top_k(index, metadata, q_vec, question, top_k=3, bm25=bm25)
        retrieve_ms = (time.perf_counter() - t0) * 1000

        answer = None
        if sources and args.provider != "local":
            prompt = app._build_user_prompt(question, sources)
            if args.provider == "bedrock":
                answer = app._call_bedrock_llm(app.SYSTEM_PROMPT, prompt, args.model or app.BEDROCK_DEFAULT_LLM_MODEL)
            else:
                answer = app._call_huggingface_llm(
                    app.SYSTEM_PROMPT, prompt, args.model or app.HF_DEFAULT_LLM_MODEL, hf_token=args.hf_token
                )
        if not answer:
            answer = app._extractive_answer(question, sources) if sources else app.NOT_FOUND

        print(f"Q: {question}   (retrieval {retrieve_ms:.2f} ms)")
        for s in sources:
            print(f"   [{s['doc_name']} p{s['page_number']}] fused={s['relevance_score']} "
                  f"dense={s['dense_score']} bm25={s['sparse_score']}")
        print(f"A: {answer}\n")


if __name__ == "__main__":
    main()
