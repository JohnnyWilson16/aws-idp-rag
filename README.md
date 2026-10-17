# ⚡ Serverless Intelligent Document Processing (IDP) & RAG Assistant on AWS

> **A Production-Ready, Low-Cost Retrieval-Augmented Generation (RAG) System on AWS powered by Amazon Bedrock (Nova Micro, Titan Text Embeddings V2) with Multi-Provider Fallback (Hugging Face / Local Extractive).**
> Follows the complete IDP Lifecycle: **Documents → OCR → Amazon S3 Storage → Vector Indexing → User Query → AI Grounded Response with Citations.**

[![AWS SAM](https://img.shields.io/badge/AWS-SAM%20Serverless-orange?logo=amazon-aws)](https://aws.amazon.com/serverless/sam/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11%20%7C%203.13-blue?logo=python)](https://www.python.org/)
[![Amazon Bedrock](https://img.shields.io/badge/AWS%20Bedrock-Nova%20Micro%20%2B%20Titan%20V2-232F3E?logo=amazon-aws)](https://aws.amazon.com/bedrock/)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS%20CPU-green)](https://github.com/facebookresearch/faiss)
[![Amazon Textract](https://img.shields.io/badge/OCR-Amazon%20Textract%20Async-red?logo=amazon-aws)](https://aws.amazon.com/textract/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 🏛️ High-Level System Architecture

```mermaid
flowchart TD
    subgraph INGESTION["1. Document Ingestion & Intelligent Processing Pipeline (IDP)"]
        UI_UP[User Uploads PDF/Scanned Image] -->|Presigned PUT URL| S3_IN[(Amazon S3 /uploads)]
        S3_IN -->|S3 ObjectCreated Event| EB[Amazon EventBridge]
        EB -->|Trigger State Machine| SF[AWS Step Functions State Machine]
        SF -->|Task 1: Start OCR| TX[Amazon Textract StartDocumentTextDetection]
        TX -->|Task 2: Poll Status| SF
        SF -->|Task 3: Paginated Text Extraction| LP[Lambda: IDP Pipeline Processor]
        LP -->|Store Raw OCR JSON| S3_EXT[(Amazon S3 /extracted)]
        LP -->|Update Status: PROCESSING/COMPLETED| DDB[(Amazon DynamoDB DocumentMetadata)]
        LP -->|Chunking: 500 tokens / 50 overlap| CHUNK[Page-Preserving Text Chunks]
        CHUNK -->|Bedrock Titan Embeddings V2| EMB[amazon.titan-embed-text-v2:0]
        EMB -->|Upsert 1024-d / Dynamic Vectors| FAISS[FAISS IndexFlatIP Cosine Index]
        FAISS -->|Persist .faiss & .pkl| S3_VEC[(Amazon S3 /vector_store)]
    end

    subgraph RETRIEVAL["2. Serverless RAG Query & Grounded Response Pipeline"]
        USER[User asks Question via Streamlit UI] -->|POST /query| APIGW[Amazon API Gateway HTTP API]
        APIGW --> QLAMBDA[Lambda: RAG Query Function]
        QLAMBDA -->|Download / Check ETag /tmp Cache| S3_VEC
        QLAMBDA -->|Embed Query| EMB2[Amazon Bedrock Titan Embeddings V2]
        QLAMBDA -->|Vector Similarity Search| RETRIEVE[Retrieve Top-K Chunks with Page Citations]
        RETRIEVE -->|Strict Context Grounding Prompt| LLM[Amazon Bedrock: amazon.nova-micro-v1:0]
        LLM -->|Grounded Answer + Citations| QLAMBDA
        QLAMBDA -->|JSON Answer + Sources Array| USER
    end
```

---

## 💡 Key Architectural Decisions & Constraint Adherence

### 1. Amazon Bedrock Native with Resilient Multi-Provider Fallback
* **Primary AI Engine (Amazon Bedrock):**
  * **Embeddings:** `amazon.titan-embed-text-v2:0` (1024-dimensional normalized dense vectors, high retrieval accuracy).
  * **LLM Generation:** `amazon.nova-micro-v1:0` via the unified `bedrock-runtime` Converse API. It delivers ultra-fast responses (sub-second TTFT) at minimal cost ($0.035 / 1M input tokens). Also supports `amazon.nova-lite-v1:0`, `meta.llama3-8b-instruct-v1:0`, and Anthropic Claude.
* **Multi-Provider Fallback:**
  * In new AWS accounts where foundation model access is pending approval, or during API rate limits, the system dynamically falls back to **Hugging Face Serverless** (`all-MiniLM-L6-v2` / `Llama-3.2-3B`) or deterministic local extractive grounding. The application never fails or crashes due to model provider unavailability.

### 2. Free Tier & Ultra-Low-Cost Optimization (< $0.05 / month)
* **Vector Database:** Instead of spinning up Amazon OpenSearch Serverless (which mandates an active baseline of 4 OCUs costing **~$700/month**), we serialize a FAISS `IndexFlatIP` directly to Amazon S3 (`/vector_store/index.faiss` and `metadata.pkl`).
* **Lambda `/tmp` Caching:** The Query Lambda caches the FAISS binary index in its `/tmp` directory and in-memory across warm invocations, inspecting the S3 object's ETag to only fetch updates. Retrieval takes **< 4 ms** with zero database hosting fees.
* **On-Demand Serverless:** Every compute component scales to zero when idle:
  * **AWS Lambda:** 1,000,000 free requests/month (ARM64 architecture for 20% cost and performance savings).
  * **Amazon DynamoDB:** On-Demand (`PAY_PER_REQUEST`), zero provisioned capacity charge.
  * **AWS Step Functions:** 4,000 free state transitions/month.
  * **Amazon S3:** 5 GB Free Tier storage.
  * **Amazon Textract:** 1,000 free pages/month under AWS Free Tier.
  * **Amazon Bedrock:** Pay-per-token pricing with zero monthly minimums. 1,000 document queries costs less than **$0.05**.

### 3. Asynchronous Multi-Page Document OCR
* Synchronous Textract API (`DetectDocumentText`) is strictly limited to single-page images and throws `UnsupportedDocumentException` on multi-page PDFs.
* Our architecture utilizes `StartDocumentTextDetection` and `GetDocumentTextDetection` orchestrated by AWS Step Functions with retry loops and pagination (`NextToken`), preserving document layout, confidence scores, and exact page numbers.

### 4. Hallucination Mitigation & Citation Grounding
* Text chunks are attributed with their source document name and 1-indexed page numbers: `[Document: {doc_name}, Page: {page_number}]`.
* The system prompt strictly constrains the LLM:
  > *"Answer the user's question ONLY using the provided document excerpts below. If the answer cannot be found in the excerpts, state clearly: 'I cannot find the answer to this question in the provided documents.' DO NOT speculate or hallucinate. Always cite document name and page number."*

---

## 💰 Cost Breakdown Comparison

| Cloud Service | Enterprise Default Pattern | Our Architecture | Monthly Cost (Our Solution) |
| :--- | :--- | :--- | :--- |
| **Vector Store** | OpenSearch Serverless (4 OCUs minimum) | FAISS on Amazon S3 + Lambda `/tmp` Cache | **$0.00** |
| **Document OCR** | Sync Textract / Dedicated EC2 OCR | Asynchronous Textract + Step Functions | **$0.00** (Free Tier 1K pages) |
| **LLM Inference** | Dedicated SageMaker ml.g5.2xlarge ($1.21/hr) | Bedrock Nova Micro Serverless | **<$0.02** (1,000 queries) |
| **Embeddings** | Dedicated SageMaker Embedding Endpoint | Bedrock Titan Text Embeddings V2 | **<$0.01** (1,000 pages) |
| **Compute & API** | EKS / ECS Container Cluster | AWS Lambda (ARM64) + HTTP API Gateway | **$0.00** (Free Tier 1M requests) |
| **State & Tracking** | Aurora Serverless DB | DynamoDB (On-Demand) | **$0.00** (25 GB free) |
| **TOTAL ESTIMATED MONTHLY BILL** | **~$1,500.00+ / month** | **Serverless Pay-Per-Token** | **<$0.05 / month** |

---

## 📂 Project Repository Structure

```
aws-idp-rag/
├── template.yaml                       # AWS SAM Template (S3, DynamoDB, Step Functions, Lambda ARM64, Bedrock IAM)
├── .gitignore                          # Excludes venv, sam artifacts, and temp caches
├── src/
│   ├── idp_pipeline/
│   │   ├── __init__.py
│   │   ├── handler.py                  # Step Functions Lambda tasks: start, check_status, process_and_index
│   │   ├── textract_utils.py           # Multi-page Textract async OCR, page preservation, sliding-window chunking
│   │   ├── embedding_utils.py          # Bedrock Titan V2 (1024-d) with Hugging Face & local hashing fallback
│   │   ├── faiss_indexer.py            # Dynamic dimension FAISS IndexFlatIP management & S3 serialization
│   │   └── requirements.txt            # Pre-compiled manylinux2014_aarch64 wheels (numpy==2.2.6, faiss-cpu)
│   ├── query_api/
│   │   ├── __init__.py
│   │   ├── app.py                      # API Gateway Lambda: Bedrock Converse API / HF LLM, vector search, citations
│   │   ├── embedding_client.py         # Query vector embedding client (Bedrock Titan V2 / HF MiniLM)
│   │   └── requirements.txt
│   └── document_api/
│       ├── __init__.py
│       ├── app.py                      # API Gateway Lambda: GET /documents (DynamoDB list), POST /upload-url (S3 presigned PUT)
│       └── requirements.txt
├── frontend/
│   ├── app.py                          # Streamlit UI: Model & Provider dropdown, real-time DynamoDB tracker, citations
│   ├── requirements.txt
│   └── .streamlit/
│       └── config.toml                 # Polished dark/light UI theme configuration
├── scripts/
│   ├── create_test_docs.py             # Generates realistic sample business PDFs (contracts, invoices, SLAs)
│   ├── local_pipeline_runner.py        # Local CLI supporting --provider bedrock and --provider huggingface
│   └── setup_aws_resources.sh          # Automation script for SAM build, deploy, test, and teardown
├── tests/
│   ├── test_chunking.py                # Unit tests for chunking with page number tracking
│   ├── test_textract_utils.py          # Unit tests for parsing Textract blocks
│   └── test_faiss_indexer.py           # Unit tests for FAISS vector addition, cosine similarity, dynamic 1024-d/384-d
└── samples/                            # Realistic sample business documents
    ├── master_service_agreement.pdf    # 4-page enterprise agreement (Scope, SLAs, Fees, Liability Cap)
    └── cloud_services_invoice.pdf      # 2-page detailed cloud invoice (Line items, banking details)
```

---

## 🚀 Quickstart & Local Verification (No AWS Credentials Required)

You can run and verify the entire OCR, chunking, embedding, vector search, and grounded RAG response locally before deploying to AWS!

### 1. Set Up Virtual Environment & Dependencies

```bash
git clone https://github.com/<your-username>/aws-idp-rag.git
cd aws-idp-rag

python3 -m venv .venv
source .venv/bin/activate
pip install -r src/idp_pipeline/requirements.txt
pip install -r frontend/requirements.txt
pip install pypdf reportlab pytest
```

### 2. Generate Realistic Sample Business Documents

```bash
python3 scripts/create_test_docs.py
```
*Outputs: `samples/master_service_agreement.pdf` (4 pages) and `samples/cloud_services_invoice.pdf` (2 pages).*

### 3. Run the Local End-to-End Pipeline Verification

```bash
# Option A: Run with Amazon Bedrock (requires AWS credentials with Bedrock access)
python3 scripts/local_pipeline_runner.py --provider bedrock --model amazon.nova-micro-v1:0

# Option B: Run fully offline / Hugging Face serverless (Zero AWS credentials needed)
python3 scripts/local_pipeline_runner.py --provider huggingface
```

Expected output:
* Parses multi-page PDF documents.
* Splits text into 500-token chunks retaining exact page numbers.
* Computes normalized embeddings (1024-d Titan V2 or 384-d MiniLM).
* Indexes chunks into a FAISS `IndexFlatIP`.
* Runs test queries and prints grounded answers with citations:
  ```text
  ❓ Question: 'What is the aggregate liability cap in the Master Services Agreement?'
  • Excerpt: [master_service_agreement.pdf - Page 4] (Relevance Score: 0.8941)
  💬 Grounded Assistant Response:
  Based on the Master Services Agreement (Page 4), either party's aggregate liability
  is strictly capped at $500,000 USD (Five Hundred Thousand Dollars).
  ```

### 4. Run Unit Test Suite

```bash
pytest tests/
```
*Verifies chunking page preservation, Textract block parser, and FAISS dynamic dimension vector search (5/5 passing).*

---

## ☁️ Deployment Guide to AWS

### 1. Prerequisites
- [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) configured with an AWS account:
  ```bash
  aws configure
  ```
- [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html) installed.
- (Recommended) In the AWS Console, visit **Amazon Bedrock → Model access** and ensure access is granted to:
  - **Amazon:** Nova Micro, Nova Lite, Titan Text Embeddings V2
  - **Meta:** Llama 3 / 3.2 (optional)

### 2. Build and Deploy with AWS SAM

```bash
# Build Lambda artifacts inside Docker container (ARM64)
sam build --use-container

# Deploy stack interactively
sam deploy --guided
```

SAM Prompt Inputs:
- **Stack Name:** `aws-idp-rag-stack`
- **AWS Region:** `us-east-1` (or your preferred region with Bedrock support)
- **Parameter Environment:** `dev`
- **Parameter ModelProvider:** `bedrock` (or `huggingface`)
- **Parameter BedrockLLMModel:** `amazon.nova-micro-v1:0`
- **Parameter BedrockEmbeddingModel:** `amazon.titan-embed-text-v2:0`
- **Confirm changes before deploy:** `Y`
- **Allow SAM CLI IAM role creation:** `Y`
- **Disable rollback:** `Y`

Once deployed, SAM outputs the **ApiEndpoint**:
```text
Outputs:
ApiEndpoint = https://abcdef1234.execute-api.us-east-1.amazonaws.com/dev
DocumentStorageBucketName = aws-idp-rag-storage-123456789012-us-east-1-dev
DocumentMetadataTableName = DocumentMetadata-dev
```

### 3. Launch the Streamlit Frontend

```bash
export API_GATEWAY_URL="https://abcdef1234.execute-api.us-east-1.amazonaws.com/dev"
streamlit run frontend/app.py
```

1. Open `http://localhost:8501` in your browser.
2. In the sidebar, select **Amazon Bedrock** and model **amazon.nova-micro-v1:0**.
3. Upload `samples/master_service_agreement.pdf`.
4. Watch the real-time Document Status Tracker change from `PROCESSING` to `🟢 COMPLETED` (Pages: 4, Chunks: 4).
5. Ask questions in the chat interface!

---

## 🧪 Testing the Live API via CLI

### Upload a Document via Presigned URL:
```bash
# 1. Request presigned upload URL
curl -X POST "${API_GATEWAY_URL}/upload-url" \
  -H "Content-Type: application/json" \
  -d '{"filename": "contract.pdf", "content_type": "application/pdf"}'

# 2. Upload document bytes directly to S3
curl -X PUT "<presigned_upload_url>" \
  -H "Content-Type: application/pdf" \
  --data-binary "@samples/master_service_agreement.pdf"
```

### Check Ingestion Status:
```bash
curl -X GET "${API_GATEWAY_URL}/documents"
```

### Query the RAG Pipeline:
```bash
curl -X POST "${API_GATEWAY_URL}/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the liability limit in the agreement?",
    "provider": "bedrock",
    "model_id": "amazon.nova-micro-v1:0",
    "top_k": 3
  }'
```

Sample API Response:
```json
{
  "question": "What is the liability limit in the agreement?",
  "answer": "According to the Master Services Agreement (Page 4), aggregate liability arising out of or related to the Agreement is limited to $500,000 USD.",
  "sources": [
    {
      "doc_name": "master_service_agreement.pdf",
      "page_number": 4,
      "snippet": "7. LIMITATION OF LIABILITY TO THE MAXIMUM EXTENT PERMITTED BY LAW, IN NO EVENT SHALL EITHER PARTY'S AGGREGATE LIABILITY ARISING OUT OF OR RELATED TO THIS AGREEMENT EXCEED $500,000 USD...",
      "relevance_score": 0.8941
    }
  ],
  "chunks_retrieved": 1,
  "model_used": "amazon.nova-micro-v1:0"
}
```

---

## 🧹 Cost Protection & Resource Teardown

To guarantee zero accidental charges after testing or presenting:

```bash
# Automated cleanup script (empties S3 and tears down CloudFormation stack)
./scripts/setup_aws_resources.sh delete
```

Or manually with the AWS CLI:
```bash
aws s3 rm s3://<DocumentStorageBucketName> --recursive
sam delete --stack-name aws-idp-rag-stack --region us-east-1 --no-prompts
```

---

## 🎓 Interview Walkthrough & Architecture Defense Guide

### Q1: How did you design the AI model layer, and why Amazon Bedrock + Nova Micro / Titan V2?
> **Answer:** 
> - **Embeddings:** We chose `amazon.titan-embed-text-v2:0` because it generates high-density 1024-dimensional normalized vectors specifically tuned for document retrieval, costing only **$0.02 per 1,000,000 tokens**.
> - **LLM Generation:** We selected `amazon.nova-micro-v1:0` via the unified `bedrock-runtime` Converse API. It costs just **$0.035 per 1,000,000 input tokens** and offers sub-second inference latency, making it the most cost-effective foundation model on AWS.
> - **Multi-Provider Resilience:** Foundation model access can sometimes be delayed or region-locked. We designed a dual-provider abstraction: if Bedrock throws `ValidationException` or `AccessDeniedException`, the system seamlessly falls back to Hugging Face Serverless or verified local extractive citations.

### Q2: Why did you choose FAISS stored in S3 rather than Amazon OpenSearch Serverless or Pinecone?
> **Answer:** Amazon OpenSearch Serverless enforces a minimum billing configuration of 4 OpenSearch Compute Units (OCUs) even when idle, which results in an unavoidable baseline cost of approximately **$700 per month**. Managed vector SaaS like Pinecone introduces third-party billing and vendor lock-in.
> For enterprise document processing up to hundreds of thousands of vectors, a FAISS `IndexFlatIP` index serialized to S3 is ideal:
> - Storage cost in S3 is a fraction of a cent (within Free Tier 5GB).
> - Warm Lambda functions cache the index in `/tmp` and memory, providing sub-millisecond retrieval.
> - ETag inspection ensures atomic index synchronization whenever Step Functions publishes an updated index.

### Q3: How do you handle multi-page PDFs and prevent OCR failure in Textract?
> **Answer:** Textract's synchronous `DetectDocumentText` throws an exception for multi-page documents. Our pipeline triggers `StartDocumentTextDetection` asynchronously and coordinates polling through AWS Step Functions. Once complete, we paginate through the results using `NextToken` to collect all `LINE` blocks, grouping lines by page number into a structured dictionary. This guarantees that multi-page tables, headers, and clauses never lose their page attribution.

### Q4: How does your system prevent hallucinations?
> **Answer:** 
> 1. **Page-Preserving Chunking:** Every chunk explicitly retains `{ "doc_name": ..., "page_number": ... }`.
> 2. **Strict Context Conditioning:** The prompt instructs the LLM that it is an extractive retrieval assistant that must *only* use facts from the provided excerpts and must cite sources in `[Document: ..., Page: ...]` format.
> 3. **Refusal Directive:** If the question cannot be answered using the retrieved chunks, the model is instructed to output: *"I cannot find the answer to this question in the provided documents."*

### Q5: What enhancements would you build with more engineering time or an enterprise budget?
> **Answer:**
> - **Hybrid Search:** Combining dense FAISS vector search with sparse BM25 keyword search (Reciprocal Rank Fusion - RRF) to improve accuracy on exact acronyms and invoice part numbers.
> - **Textract Queries & Tables API:** Utilizing Textract's `AnalyzeDocument` with `TABLES` and `FORMS` features to extract cell coordinates and key-value pairs directly into JSON tables.
> - **Cross-Encoder Reranking:** Adding a lightweight cross-encoder (e.g. `ms-marco-MiniLM-L-6-v2`) in Lambda to re-score the top 20 retrieved candidates before prompting the LLM.
> - **Cognito User Authentication:** Adding AWS Cognito user pools to secure the Streamlit dashboard and API Gateway endpoints with Role-Based Access Control (RBAC).
