"""
Streamlit Web Application for AWS Serverless IDP + RAG.

Features:
- Dual AI Provider Support: Amazon Bedrock (Claude 3 Haiku, Nova, Titan Embeddings V2) & Hugging Face Serverless.
- Document Upload (PDF / PNG / JPG) via S3 Presigned URLs.
- Real-time Document Processing Status Tracker (DynamoDB polling).
- Conversational RAG Chat Interface with Strict Context Grounding.
- Collapsible Source Attribution Expanders with Document Name & Page Citations.
- System Architecture & Free-Tier Cost Transparency Dashboard.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional
import requests
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="AWS Serverless IDP + RAG Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished interface
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge-completed {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-processing {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-failed {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .source-box {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am your **AWS Serverless Intelligent Document Assistant** powered by **Amazon Bedrock**.\n\n"
                "Upload PDFs or scanned images in the sidebar to process them through Amazon Textract "
                "OCR and S3-backed vector indexing. Then ask any question, and I will answer with "
                "strict context grounding and page citations."
            ),
            "sources": [],
        }
    ]

if "api_url" not in st.session_state:
    st.session_state.api_url = os.environ.get(
        "API_GATEWAY_URL",
        "https://lq2w2n8jae.execute-api.us-east-1.amazonaws.com/dev",
    )

if "hf_token" not in st.session_state:
    st.session_state.hf_token = os.environ.get("HF_TOKEN", "")


# Helper functions to interact with API Gateway
def fetch_documents(api_url: str) -> List[Dict[str, Any]]:
    """Fetch list of processed documents from API Gateway GET /documents."""
    try:
        url = f"{api_url.rstrip('/')}/documents"
        res = requests.get(url, timeout=8)
        if res.status_code == 200:
            return res.json().get("documents", [])
    except Exception as e:
        st.sidebar.caption(f"Connection notice: {e}")
    return []


def request_upload_url(api_url: str, filename: str, content_type: str) -> Optional[Dict[str, Any]]:
    """Request presigned upload URL from POST /upload-url."""
    try:
        url = f"{api_url.rstrip('/')}/upload-url"
        payload = {"filename": filename, "content_type": content_type}
        res = requests.post(url, json=payload, timeout=8)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        st.error(f"Error obtaining upload URL: {e}")
    return None


def upload_file_to_s3(upload_url: str, file_bytes: bytes, content_type: str) -> bool:
    """Upload raw document bytes directly to S3 via presigned PUT URL."""
    try:
        headers = {"Content-Type": content_type}
        res = requests.put(upload_url, data=file_bytes, headers=headers, timeout=60)
        return res.status_code in (200, 204)
    except Exception as e:
        st.error(f"Failed to upload to S3: {e}")
        return False


def query_rag_assistant(
    api_url: str,
    question: str,
    top_k: int,
    model: str,
    provider: str = "bedrock",
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Send user query to API Gateway POST /query."""
    try:
        url = f"{api_url.rstrip('/')}/query"
        payload = {
            "question": question,
            "top_k": top_k,
            "model": model,
            "provider": provider,
            "hf_token": hf_token,
        }
        res = requests.post(url, json=payload, timeout=45)
        if res.status_code == 200:
            return res.json()
        else:
            return {
                "answer": f"API returned error status {res.status_code}: {res.text}",
                "sources": [],
            }
    except Exception as e:
        return {
            "answer": f"Unable to reach RAG Query API at {api_url}: {e}",
            "sources": [],
        }


# ==============================================================================
# SIDEBAR: DOCUMENT INGESTION & CONFIGURATION
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ System Settings")
    api_gateway_input = st.text_input(
        "API Gateway Endpoint URL",
        value=st.session_state.api_url,
        help="Base URL of deployed AWS SAM HTTP API (e.g. https://xyz.execute-api.us-east-1.amazonaws.com/dev)",
    )
    st.session_state.api_url = api_gateway_input

    provider_choice = st.selectbox(
        "AI Model Provider",
        options=["Amazon Bedrock", "Hugging Face Serverless"],
        index=0,
        help="Amazon Bedrock provides managed foundation models (Claude 3.5, Nova, Titan Embeddings).",
    )

    if provider_choice == "Amazon Bedrock":
        provider_key = "bedrock"
        selected_model = st.selectbox(
            "Bedrock LLM Model",
            options=[
                "amazon.nova-micro-v1:0",
                "amazon.nova-lite-v1:0",
                "meta.llama3-8b-instruct-v1:0",
                "meta.llama3-1-8b-instruct-v1:0",
                "anthropic.claude-3-5-sonnet-20240620-v1:0",
                "amazon.titan-text-express-v1",
            ],
            index=0,
            help="Unified Bedrock Converse API models.",
        )
        st.caption("Embedding Engine: `amazon.titan-embed-text-v2:0` (1024-d)")
    else:
        provider_key = "huggingface"
        selected_model = st.selectbox(
            "Hugging Face LLM Model",
            options=[
                "meta-llama/Llama-3.2-3B-Instruct",
                "Qwen/Qwen2.5-7B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.3",
            ],
            index=0,
        )
        hf_token_input = st.text_input(
            "Hugging Face Token (Optional)",
            value=st.session_state.hf_token,
            type="password",
            help="Used for serverless inference on open LLMs (free at huggingface.co/settings/tokens)",
        )
        st.session_state.hf_token = hf_token_input
        st.caption("Embedding Engine: `all-MiniLM-L6-v2` (384-d)")

    top_k_val = st.slider("Retrieval Depth (Top-K Chunks)", min_value=1, max_value=5, value=3)

    st.markdown("---")
    st.markdown("### 📄 Document Ingestion (IDP)")
    uploaded_file = st.file_uploader(
        "Upload PDF or Scanned Image",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Triggers async Textract OCR, chunking, and FAISS vector indexing via AWS Step Functions",
    )

    if uploaded_file is not None:
        if st.button("🚀 Upload & Trigger Ingestion", use_container_width=True, type="primary"):
            with st.spinner("Requesting S3 Presigned URL..."):
                file_bytes = uploaded_file.getvalue()
                c_type = uploaded_file.type or "application/pdf"
                url_data = request_upload_url(
                    api_url=st.session_state.api_url,
                    filename=uploaded_file.name,
                    content_type=c_type,
                )

                if url_data and "upload_url" in url_data:
                    upload_target = url_data["upload_url"]
                    st.info(f"Uploading {uploaded_file.name} to Amazon S3...")
                    success = upload_file_to_s3(upload_target, file_bytes, c_type)
                    if success:
                        st.success(f"Uploaded successfully! S3 event triggered Step Functions processing.")
                        st.rerun()
                    else:
                        st.error("Failed to upload document bytes to S3.")
                else:
                    st.warning(
                        "Could not generate upload URL. Check your API Gateway endpoint or use manual S3 upload."
                    )

    st.markdown("---")
    col_docs, col_refresh = st.columns([3, 1])
    with col_docs:
        st.markdown("### 📑 Processed Documents")
    with col_refresh:
        if st.button("🔄", help="Refresh document list"):
            st.rerun()

    documents = fetch_documents(st.session_state.api_url)

    if documents:
        for doc in documents:
            doc_id = doc.get("DocumentId", "doc")
            fname = doc.get("filename", doc_id)
            status = doc.get("status", "UNKNOWN")
            page_count = doc.get("page_count", "-")
            chunk_count = doc.get("chunk_count", "-")

            badge_class = "badge-processing"
            if status == "COMPLETED":
                badge_class = "badge-completed"
            elif status == "FAILED":
                badge_class = "badge-failed"

            with st.container():
                st.markdown(
                    f"**{fname}** <span class='{badge_class}'>{status}</span>",
                    unsafe_allow_html=True,
                )
                st.caption(f"Pages: {page_count} | Chunks: {chunk_count}")
                st.divider()
    else:
        st.info("No documents indexed yet. Upload a document to start.")


# ==============================================================================
# MAIN VIEW: CHAT & ARCHITECTURE TABS
# ==============================================================================
st.markdown("<div class='main-header'>⚡ Serverless IDP + RAG Assistant</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>End-to-End Intelligent Document Processing on AWS (Amazon Bedrock • Textract • Step Functions • S3 FAISS)</div>",
    unsafe_allow_html=True,
)

tab_chat, tab_arch = st.tabs(["💬 Interactive RAG Assistant", "🏗️ Architecture & Cost Breakdown"])

with tab_chat:
    # Quick sample question chips
    st.markdown("**Quick Test Prompts:**")
    col1, col2, col3 = st.columns(3)
    sample_q = None
    with col1:
        if st.button("📌 What is the liability cap in the contract?"):
            sample_q = "What is the liability cap in the contract?"
    with col2:
        if st.button("💳 What are the payment terms and late fees?"):
            sample_q = "What are the payment terms and late fees?"
    with col3:
        if st.button("⏱️ What uptime SLA guarantees are specified?"):
            sample_q = "What uptime SLA guarantees are specified?"

    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 Verified Sources ({len(msg['sources'])} citations)"):
                    for idx, src in enumerate(msg["sources"], start=1):
                        st.markdown(
                            f"**Citation {idx}:** `{src.get('doc_name')}` — **Page {src.get('page_number')}** "
                            f"(Relevance Score: `{src.get('relevance_score', 0):.2f}`)"
                        )
                        st.markdown(
                            f"<div class='source-box'>{src.get('snippet')}</div>",
                            unsafe_allow_html=True,
                        )

    # Chat input handling
    prompt = st.chat_input("Ask a question about the uploaded documents...") or sample_q

    if prompt:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Query backend
        with st.chat_message("assistant"):
            with st.spinner(f"Searching FAISS index & invoking {provider_choice} ({selected_model})..."):
                response_data = query_rag_assistant(
                    api_url=st.session_state.api_url,
                    question=prompt,
                    top_k=top_k_val,
                    model=selected_model,
                    provider=provider_key,
                    hf_token=st.session_state.hf_token,
                )

                answer_text = response_data.get("answer", "No response generated.")
                sources = response_data.get("sources", [])

                st.markdown(answer_text)

                if sources:
                    with st.expander(f"📚 Verified Sources ({len(sources)} citations)"):
                        for idx, src in enumerate(sources, start=1):
                            st.markdown(
                                f"**Citation {idx}:** `{src.get('doc_name')}` — **Page {src.get('page_number')}** "
                                f"(Relevance Score: `{src.get('relevance_score', 0):.2f}`)"
                            )
                            st.markdown(
                                f"<div class='source-box'>{src.get('snippet')}</div>",
                                unsafe_allow_html=True,
                            )

                # Persist message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources,
                })


with tab_arch:
    st.markdown("### 🏛️ Complete System Architecture")
    st.markdown(
        """
        This solution follows the Intelligent Document Processing (IDP) lifecycle:
        **Document -> OCR -> S3 Storage -> Indexing -> User Query -> AI-Generated Response**
        powered by **Amazon Bedrock** (Claude 3 Haiku, Nova, Titan Embeddings) with Hugging Face Serverless fallback.
        """
    )

    st.markdown(
        """
```mermaid
flowchart TD
    subgraph INGESTION["1. Document Ingestion & IDP Pipeline"]
        A[User Uploads PDF/Image] -->|S3 Presigned URL| B[(Amazon S3 /uploads)]
        B -->|S3 Event / EventBridge| C[AWS Step Functions State Machine]
        C -->|1. Start OCR| D[Amazon Textract Async]
        D -->|2. Poll Status| C
        C -->|3. Paginated Text Extraction| E[Lambda: IDP Processor]
        E -->|Store Raw OCR JSON| F[(Amazon S3 /extracted)]
        E -->|Update Status| G[(Amazon DynamoDB)]
        E -->|Chunk 500 tokens / 50 overlap| H[Amazon Bedrock: Titan Text Embeddings V2]
        H -->|Generate 1024-d Vectors| I[FAISS IndexFlatIP Cosine Index]
        I -->|Save .faiss & .pkl| J[(Amazon S3 /vector_store)]
    end

    subgraph RETRIEVAL["2. Serverless RAG Query Pipeline"]
        K[User Query via Streamlit UI] -->|POST /query| L[Amazon API Gateway]
        L --> M[Lambda: RAG Query Function]
        M -->|Load/Cache Index| J
        M -->|Embed Query| H
        M -->|Top-K Vector Search| N[Retrieve Top-3 Chunks with Page Citations]
        N -->|Strictly Grounded Prompt| O[Amazon Bedrock: Claude 3 Haiku / Nova / Llama 3]
        O -->|Grounded Answer + Citations| M
        M -->|Return Answer + Sources| K
    end
```
        """
    )

    st.markdown("---")
    st.markdown("### 💰 AWS Cost & Model Pricing Breakdown")
    st.markdown(
        """
| AWS / Cloud Service | Free Tier / Standard Pricing | System Usage in This Project | Monthly Cost |
| :--- | :--- | :--- | :--- |
| **Amazon Bedrock (Titan Embeddings V2)** | $0.02 per 1,000,000 input tokens | 1024-d embeddings for document chunks & queries | **<$0.01** |
| **Amazon Bedrock (Claude 3 Haiku)** | $0.00025 / 1K in, $0.00125 / 1K out | Grounded prompt synthesis and citation formatting | **<$0.05** |
| **Amazon S3** | 5 GB Standard Storage, 20,000 GETs, 2,000 PUTs | Document uploads, raw OCR JSON, FAISS index artifacts | **$0.00** |
| **AWS Lambda** | 1,000,000 requests/month, 3.2M sec compute | IDP processor, Document API, Query API | **$0.00** |
| **Amazon DynamoDB** | 25 GB Storage, 25 RCU / 25 WCU | Document processing states, page counts, chunk tracking | **$0.00** |
| **AWS Step Functions** | 4,000 state transitions / month | OCR polling and pipeline orchestration | **$0.00** |
| **Amazon Textract** | 1,000 pages / month (DetectDocumentText) | Multi-page PDF & scanned image asynchronous OCR | **$0.00** |
| **Amazon API Gateway** | 1,000,000 HTTP API calls / month | Lightweight REST API routes (/query, /documents) | **$0.00** |
| **TOTAL ESTIMATED BILL** | | | **<$0.10 / month** |
        """
    )

    st.markdown("---")
    st.markdown("### 🛡️ Design Decisions & Trade-Offs")
    st.markdown(
        """
        1. **Native Amazon Bedrock Integration**: Utilizes industry-standard Foundation Models (`amazon.titan-embed-text-v2:0` for 1024-d embeddings and `anthropic.claude-3-haiku-20240307-v1:0` via the unified Bedrock Converse API).
        2. **Multi-Provider Fallback**: If Bedrock access is not yet enabled or during offline local development, the system seamlessly falls back to Hugging Face Serverless APIs or local deterministic hashing.
        3. **FAISS in S3 vs. Amazon OpenSearch Serverless**: OpenSearch Serverless charges an active base cost of 4 OCUs ($0.24/OCU/hr ≈ $700/month!). Using FAISS Flat indexes persisted in S3 and cached in Lambda `/tmp` costs $0.00 and provides sub-4ms retrieval for small-to-medium enterprise document collections.
        4. **Asynchronous Textract with Step Functions**: Multi-page PDFs cannot be processed synchronously due to Textract payload limits. Step Functions provides resilient, serverless polling and error handling with zero idle compute cost.
        5. **Hallucination Prevention**: Prompts explicitly forbid speculative reasoning and mandate source citation format `[Document: ..., Page: ...]`.
        """
    )
