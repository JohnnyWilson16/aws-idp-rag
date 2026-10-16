"""
Local End-to-End Pipeline Runner & Verification CLI.

Demonstrates the full IDP + RAG workflow locally:
1. Ingests local PDF documents (e.g. samples/master_service_agreement.pdf).
2. Extracts text preserving exact page numbers.
3. Chunks text with sliding window and metadata attribution.
4. Generates 384-dimensional embeddings (sentence-transformers/all-MiniLM-L6-v2).
5. Builds and queries a FAISS IndexFlatIP vector store.
6. Formulates strictly grounded prompt and retrieves answers with verified page citations.
"""

import argparse
import os
import sys
from typing import Dict, List
import faiss
import numpy as np
from pypdf import PdfReader

# Add src directories to Python module path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/idp_pipeline")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/query_api")))

from textract_utils import chunk_document_pages
from embedding_utils import get_batch_embeddings, get_embedding
from faiss_indexer import add_chunks_to_index, create_empty_index, search_index
from app import _call_huggingface_llm, _generate_extractive_fallback


def extract_pages_from_pdf(pdf_path: str) -> Dict[int, Dict]:
    """Extract text from local PDF mapping to 1-indexed pages."""
    reader = PdfReader(pdf_path)
    pages_map = {}
    for idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        pages_map[idx] = {
            "page_number": idx,
            "lines": lines,
            "full_text": "\n".join(lines),
            "average_confidence": 100.0,
        }
    return pages_map


def run_pipeline(
    pdf_paths: List[str],
    queries: List[str],
    hf_token: str = "",
    model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
):
    """Execute end-to-end local IDP and RAG verification."""
    print("=" * 70)
    print("⚡ AWS SERVERLESS IDP + RAG LOCAL PIPELINE VERIFICATION")
    print("=" * 70)

    # 1. Document Ingestion & Chunking
    index = create_empty_index(384)
    metadata_list = []
    total_pages = 0

    for path in pdf_paths:
        if not os.path.exists(path):
            print(f"Warning: File not found: {path}")
            continue

        doc_name = os.path.basename(path)
        print(f"\n[1/4] Processing Document: {doc_name}")
        pages_map = extract_pages_from_pdf(path)
        total_pages += len(pages_map)
        print(f"      -> Extracted {len(pages_map)} pages.")

        chunks = chunk_document_pages(
            pages_map=pages_map,
            doc_name=doc_name,
            max_chunk_chars=1800,
            overlap_chars=200,
        )
        print(f"      -> Created {len(chunks)} text chunks with page citations.")

        # 2. Embedding Generation
        print(f"[2/4] Generating 384-d embeddings (sentence-transformers/all-MiniLM-L6-v2)...")
        texts = [c["text"] for c in chunks]
        embeddings = get_batch_embeddings(texts, token=hf_token)

        # 3. Vector Indexing
        print(f"[3/4] Indexing vectors into FAISS IndexFlatIP (Cosine Similarity)...")
        index, metadata_list = add_chunks_to_index(index, metadata_list, chunks, embeddings)

    print(f"\n✅ Indexing complete: {index.ntotal} total vectors across {total_pages} pages.\n")

    # 4. Query & RAG Retrieval
    print("=" * 70)
    print("🔍 TESTING RAG RETRIEVAL & GROUNDED CITATIONS")
    print("=" * 70)

    for q_idx, question in enumerate(queries, start=1):
        print(f"\n❓ Question {q_idx}: '{question}'")
        q_vec = get_embedding(question, token=hf_token)
        matches = search_index(index, metadata_list, q_vec, top_k=3)

        print(f"   Retrieved {len(matches)} matching chunks:")
        for m_idx, m in enumerate(matches, start=1):
            print(
                f"   • Excerpt {m_idx}: [{m['doc_name']} - Page {m['page_number']}] "
                f"(Relevance Score: {m['score']:.4f})"
            )

        # Build prompt
        context_blocks = [
            f"[Excerpt {i}] Document: {m['doc_name']} | Page: {m['page_number']}\n{m['text']}"
            for i, m in enumerate(matches, start=1)
        ]
        context_str = "\n\n".join(context_blocks)

        system_prompt = (
            "You are an expert AI assistant performing Intelligent Document Retrieval.\n"
            "Answer the user's question ONLY using the provided document excerpts below.\n"
            "If the answer cannot be found in the excerpts, state clearly: "
            "'I cannot find the answer to this question in the provided documents.'\n"
            "Always cite the specific Document Name and Page Number for every claim made."
        )
        user_prompt = (
            f"--- CONTEXT EXCERPTS ---\n{context_str}\n\n"
            f"--- USER QUESTION ---\n{question}\n\n"
            f"--- INSTRUCTIONS FOR ANSWER ---\n"
            f"1. Base your answer purely on the context excerpts.\n"
            f"2. Cite document names and page numbers in brackets.\n"
            f"3. Keep the answer clear and concise."
        )

        answer = _call_huggingface_llm(
            prompt_system=system_prompt,
            prompt_user=user_prompt,
            model_id=model_id,
            hf_token=hf_token,
            max_tokens=400,
        )

        if not answer:
            answer = _generate_extractive_fallback(question, matches)

        print("\n💬 Grounded Assistant Response:")
        print("-" * 50)
        print(answer)
        print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run local RAG pipeline verification")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""), help="Hugging Face API Token")
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Hugging Face model ID",
    )
    args = parser.parse_args()

    sample_docs = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../samples/master_service_agreement.pdf")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../samples/cloud_services_invoice.pdf")),
    ]

    sample_questions = [
        "What is the aggregate liability cap in the Master Services Agreement?",
        "What are the payment terms and late fees on invoices?",
        "What uptime percentage does the service level agreement commit to?",
        "What is the total amount due on invoice INV-2026-8891?",
    ]

    run_pipeline(
        pdf_paths=sample_docs,
        queries=sample_questions,
        hf_token=args.hf_token,
        model_id=args.model,
    )
