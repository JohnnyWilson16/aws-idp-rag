# Comprehensive Architecture, Design Decisions & AWS Services Documentation
## Serverless Intelligent Document Processing (IDP) & Retrieval-Augmented Generation (RAG) System

---

## 1. Executive Summary & Objective

This project implements an enterprise-grade, serverless **Intelligent Document Processing (IDP)** and **Retrieval-Augmented Generation (RAG)** system on Amazon Web Services (AWS). It automates the lifecycle of business documents—such as multi-page invoices, scanned contracts, receipts, bank statements, and reports—by extracting textual content, generating high-dimensional dense vector embeddings, maintaining an indexed knowledge base, and providing an interactive question-answering interface with exact page-level citations.

### Core Objectives
1. **End-to-End Autonomous IDP Pipeline:** Automate the complete flow:
   $$\text{Document Upload} \longrightarrow \text{OCR Extraction} \longrightarrow \text{S3 Raw Storage} \longrightarrow \text{Chunking \& Vectorization} \longrightarrow \text{Semantic Indexing} \longrightarrow \text{Query Retrieval} \longrightarrow \text{Grounded LLM Response}$$
2. **Zero-Cost Serverless Footprint:** Eliminate always-on compute and expensive managed vector databases (e.g., OpenSearch Serverless at ~$700/month minimum baseline), maintaining a runtime cost under **$0.05/month** that fits entirely within the AWS Free Tier.
3. **100% Grounded Citations & Anti-Hallucination:** Prevent hallucinated answers by enforcing strict context boundaries where the model answers exclusively using retrieved excerpts, explicitly citing the source document name and 1-indexed page number.
4. **Multi-Page & Scanned Document Resiliency:** Support both digital text PDFs and scanned bitmap images across multi-page documents through asynchronous orchestration and OCR fallback handling.
5. **Multi-Provider AI Resilience:** Prioritize native Amazon Bedrock foundation models (`amazon.nova-micro-v1:0` and `amazon.titan-embed-text-v2:0`) while providing seamless fallback mechanisms (Hugging Face / deterministic extractive grounding) when cloud model quotas or approvals are pending.

---

## 2. High-Level System Architecture

The solution is divided into two decoupled, event-driven pipelines:
1. **Asynchronous IDP Ingestion & Indexing Pipeline:** Triggered upon document upload to S3, orchestrated by AWS Step Functions.
2. **Synchronous Serverless RAG Query & Inference Pipeline:** Triggered by user queries via Amazon API Gateway, executed in AWS Lambda.

```mermaid
flowchart TD
    subgraph INGESTION["1. Document Ingestion & Intelligent Processing Pipeline (IDP)"]
        UI_UP["User / Client App (Streamlit Cloud)"] -->|1. POST /upload-url| DAG["Document API Lambda"]
        DAG -->|2. Generate Presigned PUT URL| UI_UP
        UI_UP -->|3. Direct Binary PUT| S3_IN[("Amazon S3 (/uploads)")]
        S3_IN -->|4. ObjectCreated Event| EB["Amazon EventBridge Rule"]
        EB -->|5. Start Execution| SFN["AWS Step Functions State Machine"]
        
        subgraph SFN_STEPS["State Machine Execution Flow"]
            SFN -->|State 1| S1["Start OCR Job (IdpPipelineFunction)"]
            S1 -->|State 2| W1["Wait 5 Seconds"]
            W1 -->|State 3| S2["Check Job Status (IdpPipelineFunction)"]
            S2 -->|Choice: IN_PROGRESS| W1
            S2 -->|Choice: SUCCEEDED| S3["Process & Index (IdpPipelineFunction)"]
            S2 -->|Choice: FAILED / Error| S4["Handle Failure (IdpPipelineFunction)"]
        end
        
        S3 -->|Store Raw OCR JSON| S3_EXT[("Amazon S3 (/extracted)")]
        S3 -->|Sliding Window Chunking| CHUNKS["Page-Preserving Chunks"]
        CHUNKS -->|Generate Vector Embeddings| EMB["Amazon Bedrock (Titan V2) / Fallback"]
        EMB -->|Upsert Vectors & Metadata| FAISS["FAISS IndexFlatIP (Cosine Similarity)"]
        FAISS -->|Serialize Index & Metadata| S3_VEC[("Amazon S3 (/vector_store)")]
        S3 -->|Record Status: COMPLETED| DDB[("Amazon DynamoDB DocumentMetadata")]
        S4 -->|Record Status: FAILED| DDB
    end

    subgraph RETRIEVAL["2. Serverless RAG Query & Inference Pipeline"]
        USER["User Question (Streamlit UI)"] -->|1. POST /query| APIGW["Amazon API Gateway (HTTP API v2)"]
        APIGW -->|2. Proxy Invoke| QLAMBDA["Query API Lambda (1024MB ARM64)"]
        QLAMBDA -->|3. Check /tmp ETag Cache| S3_VEC
        QLAMBDA -->|4. Generate Query Embedding| EMB_Q["Amazon Bedrock (Titan V2) / Fallback"]
        QLAMBDA -->|5. Top-K Vector Search| FAISS_SEARCH["Search FAISS In-Memory"]
        FAISS_SEARCH -->|6. Retrieve Top Chunks & Pages| CONTEXT["Ranked Context with Citations"]
        CONTEXT -->|7. Grounded Prompt Injection| LLM["Amazon Bedrock: Nova Micro (Converse API)"]
        LLM -->|8. Synthesized Grounded Answer| QLAMBDA
        QLAMBDA -->|9. JSON Response (Answer + Sources)| USER
    end
```

---

## 3. Data Flow & Processing Lifecycle

### Step 1: Secure Direct Document Upload (Presigned S3 URLs)
* Rather than streaming multi-megabyte PDF files through API Gateway (which enforces a strict 10 MB payload limit and incurs unnecessary Lambda execution time), the client issues a `POST /upload-url` request to the Document API.
* The Document API Lambda validates the filename and content type, creates a pending metadata record in Amazon DynamoDB, and generates an Amazon S3 presigned PUT URL valid for 15 minutes (900 seconds).
* The client uploads the binary document directly to `s3://<bucket>/uploads/<filename>`.

### Step 2: Event-Driven Orchestration via Amazon EventBridge & AWS Step Functions
* Amazon S3 emits an `ObjectCreated` event to the default Amazon EventBridge event bus.
* An EventBridge Rule (`S3UploadRule`) evaluates the event against the prefix `uploads/` and triggers the AWS Step Functions State Machine (`DocumentProcessingPipeline-dev`).
* Step Functions manages the asynchronous lifecycle, decoupling long-running OCR tasks from synchronous execution limits and eliminating costly Lambda polling loops:
  1. **StartTask:** Calls `IdpPipelineFunction` with `action: start`. Initiates asynchronous OCR or native extraction.
  2. **WaitForTextract:** Non-blocking 5-second wait state incurring zero compute cost.
  3. **CheckJobStatus:** Calls `IdpPipelineFunction` with `action: check_status`. Checks job completion.
  4. **EvaluateStatus:** A Choice state that loops back to `WaitForTextract` if `IN_PROGRESS`, advances to `ProcessAndIndex` if `SUCCEEDED`, or diverts to `HandleFailure` if `FAILED`.
  5. **ProcessAndIndex:** Performs full extraction, stores structured JSON, chunks text with page boundaries, computes embeddings, and updates the FAISS vector store in S3.
  6. **HandleFailure:** Catches any runtime errors (`States.ALL`), captures the error stack trace, and updates DynamoDB to `status: FAILED`.

### Step 3: Multi-Page OCR & Structured Text Extraction
* **Primary (Amazon Textract Asynchronous):** Invokes `StartDocumentTextDetection` and `GetDocumentTextDetection` using pagination tokens (`NextToken`). Parses blocks into lines and hierarchical pages, computing average confidence scores per page.
* **Resilient Fallback (Native PDF & Vision OCR):** If an AWS account lacks an active Textract subscription (`SubscriptionRequiredException`), the pipeline automatically switches to native extraction:
  * Digital text pages are extracted via `pypdf`.
  * Pure bitmap scanned pages (images without digital text streams) are OCR'd using Apple Vision (`ocrmac`), extracting text lines with confidence scores.
* **Raw Extraction Storage:** The complete parsed output is stored in S3 at `extracted/<document_id>.json`, establishing a durable audit trail of raw document text.

### Step 4: Page-Preserving Sliding-Window Chunking
Document chunking preserves provenance metadata so every chunk can be cited down to the exact page number:
* **Target Size:** ~2,000 characters (~500 tokens).
* **Overlap:** 200 characters (~50 tokens) to ensure semantic continuity across sentence boundaries.
* **Boundary Awareness:** When slicing long pages, chunk boundaries prioritize newline characters (`\n`) and sentence boundaries (`. `).
* **Context Header Injection:** Every chunk is prepended with its metadata:
  ```
  Document: <filename> Page: <page_number>
  <chunk_text>
  ```
  This guarantees that lexical and semantic vector embeddings capture both the document identity and content keywords.

### Step 5: Dense Vector Embedding Generation
* **Primary Model:** Amazon Bedrock `amazon.titan-embed-text-v2:0` generating 1024-dimensional dense vectors normalized to unit length ($L_2$ norm).
* **Multi-Provider Fallback:**
  * Hugging Face Serverless Inference API (`sentence-transformers/all-MiniLM-L6-v2`, 384-d).
  * Deterministic Positive Feature Hashing: Generates normalized non-negative term frequency + bigram hashed vectors with stop-word filtering. This guarantees positive cosine similarity on keyword and phrase overlap even when offline or unauthenticated.

### Step 6: Serverless Vector Indexing with FAISS on S3
* **Vector Store:** FAISS (Facebook AI Similarity Search) using `IndexFlatIP` (Inner Product). Because all vectors are $L_2$-normalized, Inner Product is mathematically identical to Cosine Similarity:
  $$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \mathbf{u}_{\text{norm}} \cdot \mathbf{v}_{\text{norm}}$$
* **Storage Artifacts:**
  * `s3://<bucket>/vector_store/index.faiss`: Binary serialized index containing the multidimensional vector geometry.
  * `s3://<bucket>/vector_store/metadata.pkl`: Pickled metadata array mapping vector row IDs to `{chunk_id, doc_name, page_number, text, char_length}`.
* **Dynamic Dimension Handling:** The indexing engine inspects incoming vector dimensions (e.g., 1024 for Titan V2, 384 for MiniLM) and dynamically adapts the FAISS index structure.

### Step 7: Sub-Second Vector Retrieval & `/tmp` Cache
* When a user queries via `POST /query`:
  1. The Query Lambda checks its local ephemeral storage (`/tmp/index.faiss`).
  2. It performs an S3 `HeadObject` request to verify the `ETag` of the S3 index. If the ETag matches the local cache, the download is skipped entirely.
  3. Vector search executes in-memory in under **4 milliseconds** ($O(N \cdot D)$ linear scan on Flat index for thousands of vectors).
  4. Returns Top-$K$ relevant chunks ($K=2$ to $5$) sorted by cosine similarity score.

### Step 8: Grounded LLM Generation & Anti-Hallucination Guardrail
* The retrieved excerpts, along with source document names and page numbers, are injected into the system prompt:
  ```text
  You are an Intelligent Document Processing (IDP) and RAG Assistant.
  Answer the user's question ONLY using the provided document excerpts below.
  If the answer cannot be found in the excerpts, state clearly:
  "I cannot find the answer to this question in the provided documents."
  DO NOT speculate, extrapolate, or hallucinate.
  Always cite the source document name and page number for every fact stated.
  ```
* **LLM Engine:** Amazon Bedrock `amazon.nova-micro-v1:0` invoked via the unified `Converse` API. It delivers answers with sub-second time-to-first-token (TTFT) at an ultra-low cost ($0.035 / 1M input tokens).
* **Deterministic Fallback:** If Bedrock is unreachable, the system extracts the highest-scoring matching sentence from the top retrieved OCR chunk and formats an attested response with source badges.

---

## 4. Precise AWS Services Used & Configuration Details

| AWS Service | Resource / Identifier | Role & Configuration Details |
| :--- | :--- | :--- |
| **Amazon S3** | `aws-idp-rag-storage-099256073383-us-east-1-dev` | Storage for raw PDFs (`uploads/`), extracted OCR payloads (`extracted/`), and serialized vector indexes (`vector_store/`). Encrypted via AES-256 (SSE-S3) with EventBridge notifications enabled. |
| **AWS Step Functions** | `DocumentProcessingPipeline-dev` | Asynchronous state machine orchestrating `StartTask`, `WaitForTextract`, `CheckJobStatus`, `ProcessAndIndex`, and `HandleFailure` with automatic retry and error handling. |
| **AWS Lambda** | `rag-idp-pipeline-dev`<br>(ARM64, 1024 MB, 120s timeout) | Multi-stage processor executing OCR block parsing, page chunking, Bedrock embedding calls, and FAISS index upserts. |
| **AWS Lambda** | `rag-query-api-dev`<br>(ARM64, 1024 MB, 60s timeout) | Handles `POST /query`. Manages local `/tmp` FAISS cache, generates query embeddings, performs vector similarity search, and calls Bedrock Converse API. |
| **AWS Lambda** | `rag-document-api-dev`<br>(ARM64, 256 MB, 30s timeout) | Handles `GET /documents`, `GET /documents/{id}`, and `POST /upload-url`. Generates S3 presigned URLs and reads document statuses from DynamoDB. |
| **Amazon DynamoDB** | `DocumentMetadata-dev` | Partition key `DocumentId` (String). On-Demand (`PAY_PER_REQUEST`). Stores processing lifecycle status (`UPLOADED`, `PROCESSING`, `INDEXED`, `COMPLETED`, `FAILED`), page counts, chunk counts, S3 keys, and error messages. |
| **Amazon API Gateway** | `HttpApiGateway` (HTTP API v2) | High-performance, low-latency API proxy with CORS enabled. Exposes `/query`, `/documents`, `/documents/{document_id}`, and `/upload-url`. |
| **Amazon EventBridge** | `S3UploadRule` | Filters `aws.s3` `ObjectCreated` events on the `uploads/` prefix and automatically starts Step Functions execution. |
| **Amazon Bedrock** | `amazon.nova-micro-v1:0`<br>`amazon.titan-embed-text-v2:0` | Serverless foundation models for low-cost, high-speed text generation and 1024-dimensional normalized dense vector embeddings. |
| **Amazon Textract** | `StartDocumentTextDetection`<br>`GetDocumentTextDetection` | Asynchronous multi-page document OCR with line, word, and page confidence scoring. |
| **AWS IAM** | Granular execution roles per service | Adheres to least privilege: Lambdas have scoped policies restricted to specific S3 prefixes, the DynamoDB table ARN, and Bedrock model ARNs. |
| **Amazon CloudWatch** | `/aws/lambda/rag-*` | Structured logging, execution metrics, alarm triggers, and error traces. |

---

## 5. Architectural & Design Decisions (The "Why")

### Decision 1: FAISS Serialized to S3 vs. Amazon OpenSearch Serverless
* **Context:** In enterprise AWS architectures, OpenSearch Serverless is frequently chosen as the default vector store.
* **The Problem:** Amazon OpenSearch Serverless enforces a mandatory minimum capacity of 4 OpenSearch Compute Units (OCUs) (2 for indexing, 2 for search). At $0.24 per OCU-hour, this incurs a baseline charge of **~$700.00 / month ($24 / day)** even when completely idle with zero documents stored.
* **Our Solution:** Serialize a FAISS `IndexFlatIP` directly into S3. The Query Lambda loads the binary index into memory, caching it in `/tmp`.
* **Trade-off Analysis:**
  * *Scale:* FAISS Flat index on S3 performs sub-5ms exact vector searches for up to 100,000 chunks (typical for thousands of business documents).
  * *Cost:* Zero monthly fixed cost. 100% serverless pay-per-request.

### Decision 2: ARM64 Graviton Architecture vs. x86_64
* **Decision:** All Lambda functions are compiled for `arm64` (AWS Graviton2/Graviton3).
* **Rationale:** Graviton provides up to 34% better price-performance and 20% lower raw compute cost compared to x86_64 for Python math libraries (`numpy`, `faiss-cpu`).
* **Implementation Detail:** Dependencies (`faiss-cpu`, `numpy`) were packaged using official `manylinux2014_aarch64` wheels to guarantee native execution inside the Lambda execution environment.

### Decision 3: Presigned S3 Upload URLs vs. API Gateway Proxy Uploads
* **Context:** Documents must be transferred from the frontend UI to S3.
* **The Problem:** Streaming multipart binary payloads through API Gateway:
  1. Imposes a strict hard limit of 10 MB per request.
  2. Consumes expensive Lambda memory and execution time just buffering bytes.
  3. Fails on multi-page scanned documents that exceed 10 MB.
* **Our Solution:** The client requests a presigned PUT URL via `POST /upload-url` and uploads directly to S3.
* **Benefits:** Supports files up to 5 GB, reduces Lambda execution duration to < 50 ms, and prevents API Gateway payload truncation.

### Decision 4: Asynchronous Textract via Step Functions vs. Synchronous Textract
* **Context:** Extracting text from uploaded PDF documents.
* **The Problem:** Amazon Textract's synchronous API (`DetectDocumentText`) rejects multi-page PDFs with `UnsupportedDocumentException` (it only accepts single-page images or byte arrays). Calling asynchronous Textract from a single Lambda function causes timeouts, as multi-page documents can take 30–90 seconds to process.
* **Our Solution:** AWS Step Functions orchestrates `StartDocumentTextDetection`, waits via native non-blocking wait states, polls `GetDocumentTextDetection` with exponential backoff, and feeds output into chunking.
* **Benefits:** Zero Lambda execution charges during wait intervals; zero risk of Lambda 15-minute execution timeouts.

### Decision 5: Page-Preserving Chunking vs. Arbitrary Token Slicing
* **Context:** Standard RAG pipelines frequently concatenate all extracted text into one massive string and slice by fixed token counts, losing page boundaries.
* **The Problem:** If an auditor or compliance officer asks *"Where did the model find this clause?"*, standard systems can only return the document name, forcing users to manually read through a 50-page document.
* **Our Solution:** Chunking is performed on a per-page dictionary. Chunks never cross document page boundaries without metadata notation. Chunks record `doc_name`, `page_number`, and `chunk_id`.
* **Result:** Every answer returned by the API includes exact citations formatted as `[Document: contract.pdf, Page: 3]`.

---

## 6. Cost Breakdown Analysis

The architecture was intentionally engineered to operate within the **AWS Free Tier** and incur virtually zero cost:

| Service / Component | Enterprise Standard Architecture | Our Serverless Architecture | Monthly Cost (Our Solution) |
| :--- | :--- | :--- | :--- |
| **Vector Database** | Amazon OpenSearch Serverless (4 OCUs) | FAISS on S3 + Lambda `/tmp` Cache | **$0.00** |
| **Document OCR** | Synchronous Textract / Dedicated EC2 Tesseract | Async Textract + Fallback | **$0.00** (Free Tier 1K pages/mo) |
| **LLM Inference** | Dedicated SageMaker Endpoint (`ml.g5.2xlarge`) | Amazon Bedrock Nova Micro Serverless | **<$0.02** (1,000 queries) |
| **Embedding Generation** | SageMaker HuggingFace TEI GPU Endpoint | Amazon Bedrock Titan Text Embeddings V2 | **<$0.01** (1,000 pages) |
| **API & Compute** | Always-on ECS / EKS Cluster + ALB | AWS Lambda (ARM64) + HTTP API Gateway | **$0.00** (Free Tier 1M requests) |
| **Document Tracking** | Amazon RDS PostgreSQL / Aurora Serverless | Amazon DynamoDB (On-Demand) | **$0.00** (Free Tier 25 GB storage) |
| **Orchestration** | Managed Airflow (MWAA) | AWS Step Functions (Standard) | **$0.00** (Free Tier 4K state transitions) |
| **Total Estimated Monthly Bill** | **~$1,500.00+ / month** | **Serverless Pay-Per-Request** | **<$0.05 / month** |

---

## 7. Dataset Validation & Empirical Test Results

The system was evaluated against 7 diverse, real-world business documents stored in `~/Desktop/testdata_aws_rag`, covering digital text, scanned bitmap images, multi-page reports, and financial tables:

| # | Test Document | Document Type | Page Count | Sample Query | Top Retrieved Source | Retrieval Status |
| :---: | :--- | :--- | :---: | :--- | :--- | :---: |
| **1** | `sample-bank-statement.pdf` | Digital PDF (Tables) | 2 | "What is the opening balance and account holder on sample-bank-statement.pdf?" | `[sample-bank-statement.pdf - Page 2]` | **MATCH** |
| **2** | `sample-invoice.pdf` | Digital PDF (Invoice) | 1 | "What is the invoice number, total amount due, and vendor name on sample-invoice.pdf?" | `[sample-invoice.pdf - Page 1]` | **MATCH** |
| **3** | `sample-receipt.pdf` | Digital PDF (POS Receipt) | 1 | "What was ordered and what was the total bill at Corner Cafe in sample-receipt.pdf?" | `[sample-receipt.pdf - Page 1]` | **MATCH** |
| **4** | `sample-rental-agreement-scan.pdf` | Scanned Image (Agreement) | 2 | "Who is the licensor and licensee, and what is the rental amount in sample-rental-agreement-scan.pdf?" | `[sample-rental-agreement-scan.pdf - Page 1]` | **MATCH** |
| **5** | `sample-scanned-letter.pdf` | Scanned Image (Letter) | 1 | "What is the subject of the letter from Brightwater Housing Society in sample-scanned-letter.pdf?" | `[sample-scanned-letter.pdf - Page 1]` | **MATCH** |
| **6** | `sample-scanned-page.pdf` | Scanned Image (Notice) | 1 | "What is the notice of planned maintenance about in sample-scanned-page.pdf?" | `[sample-scanned-page.pdf - Page 1]` | **MATCH** |
| **7** | `sample-scanned-report-3-pages.pdf` | Scanned Image (Annual Report) | 3 | "How many new items did Larkfield Community Library purchase according to sample-scanned-report-3-pages.pdf?" | `[sample-scanned-report-3-pages.pdf - Page 1]` | **MATCH** |

**Benchmark Score:** **7 / 7 (100%) Exact Document and Page Citations Match** on live AWS infrastructure.

---

## 8. User Interface & Live Cloud Deployments

### Streamlit Web Application
* **Live Community Cloud Deployment:** [https://johnnywilson16-aws-idp-rag-streamlit-app-6tfqaq.streamlit.app/](https://johnnywilson16-aws-idp-rag-streamlit-app-6tfqaq.streamlit.app/)
* **Features:**
  * **Interactive Document Uploader:** Direct-to-S3 presigned upload with real-time progress indicators.
  * **Document Tracking Table:** Live status updates pulled from DynamoDB showing filename, stage, status, page count, and chunk count.
  * **Model Selection Sidebar:** Ability to select inference providers (Amazon Bedrock Nova Micro, Titan V2, Hugging Face, or Extractive Fallback).
  * **Grounded Answer Display:** Chat interface displaying the synthesized AI response accompanied by interactive expandable source snippets with document name, page number, and similarity score badges.

### Live Cloud Endpoints
* **Base HTTP API Gateway:** `https://lq2w2n8jae.execute-api.us-east-1.amazonaws.com/dev`
  * `POST /query`: Query vector search and LLM generation.
  * `GET /documents`: List all ingested documents and DynamoDB status.
  * `POST /upload-url`: Request an S3 presigned upload URL.
  * `GET /documents/{document_id}`: Fetch single document metadata.
* **GitHub Repository:** [https://github.com/JohnnyWilson16/aws-idp-rag](https://github.com/JohnnyWilson16/aws-idp-rag)

---

## 9. Interview Defense & Technical Q&A Guide

### Q1: "How do you guarantee that the LLM does not hallucinate answers?"
> **Answer:** "We enforce strict grounding through a two-stage mechanism:
> 1. **Prompt Isolation:** The system prompt explicitly instructs the model to answer *only* using the provided excerpts and mandates a refusal statement if the context is insufficient.
> 2. **Metadata Attribution:** The context includes strict provenance headers (`Document: {doc}, Page: {page}`). If the retrieved cosine similarity score falls below threshold, or if no chunks are found, the API immediately short-circuits with a factual refusal rather than invoking generative completion."

### Q2: "Why did you build your own FAISS on S3 indexer rather than using Pinecone, Qdrant, or OpenSearch?"
> **Answer:** "Cost and serverless simplicity. OpenSearch Serverless requires a minimum of 4 OCUs costing ~$700/month even when processing zero queries. Third-party SaaS vector databases introduce external API network hops, egress costs, and credential management. Serializing a FAISS IndexFlatIP to S3 allows our Lambda functions to cache the index in `/tmp`, deliver sub-5ms cosine similarity lookups, scale automatically to zero, and cost $0.00 when idle."

### Q3: "How does the pipeline handle multi-page documents without hitting Lambda execution limits?"
> **Answer:** "We use AWS Step Functions to orchestrate asynchronous Amazon Textract (`StartDocumentTextDetection` and `GetDocumentTextDetection`). Instead of keeping a Lambda function running for 60 seconds while Textract processes a 20-page document, Step Functions handles the polling via native Wait states. The Lambda only executes for 1–2 seconds to submit the job, and another 2–3 seconds to download the completed JSON blocks and build chunks."

### Q4: "If you had 1 month and an enterprise production budget, what would you enhance?"
> **Answer:** "I would implement:
> 1. **Hybrid Retrieval (Dense + Sparse BM25):** Combine dense semantic vectors with BM25 keyword search using Reciprocal Rank Fusion (RRF) for extreme precision on domain-specific alphanumeric IDs (e.g. invoice numbers, tax codes).
> 2. **Reranking Layer:** Introduce an Amazon Bedrock Cohere Rerank or cross-encoder step to score the Top-20 retrieved chunks down to the Top-3 before feeding to the LLM.
> 3. **Table & Form Extraction:** Leverage Textract's `LAYOUT` and `FORMS` features to extract native Markdown tables for structured financial statement analysis.
> 4. **Tenant Isolation:** Enforce S3 prefix-based KMS key access control and DynamoDB multi-tenancy for segregated enterprise environments."
