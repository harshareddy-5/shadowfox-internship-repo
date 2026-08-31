<<<<<<< HEAD
# DocuMind AI — Production RAG Knowledge Assistant

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1.0+-purple.svg)](https://github.com/langchain-ai/langgraph)
[![FAISS](https://img.shields.io/badge/FAISS-CPU-orange.svg)](https://github.com/facebookresearch/faiss)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://www.docker.com/)

**DocuMind AI** is an end-to-end, production-style Retrieval-Augmented Generation (RAG) system built for the **ShadowFox AI Engineer Internship Advanced Level Task**.

Unlike basic chatbots or single-file prototypes, DocuMind AI demonstrates enterprise AI engineering principles: modular architecture, multi-step state graph orchestration (LangGraph), persistent vector search (FAISS), neural reranking (`BAAI/bge-reranker-base`), strict context quality filtering, groundedness verification, document-scoped retrieval, full observability, and containerized deployment.

---

## 📸 Interface Overview

```
+-----------------------------------------------------------------------------------+
|  DocuMind AI Knowledge Assistant                                                  |
|  Grounded Document Question Answering powered by LangGraph, FAISS & Google Gemini |
+-----------------------------------------------------------------------------------+
|  Question: "What is supervised learning?"                                         |
|  [Ask Question]                                                                   |
+-----------------------------------------------------------------------------------+
|  Groundedness: [GROUNDED] | Context Quality: [HIGH] | Retrived Chunks: 4 | 0.8s     |
|                                                                                   |
|  Answer:                                                                          |
|  Supervised learning is a paradigm where models are trained on labeled datasets   |
|  containing input-output pairs [Document: ML_Intro.pdf, Page 3].                  |
|                                                                                   |
|  Source Citations:                                                                |
|  * ML_Intro.pdf | Page 3 | Chunk ID: doc1_c2 | Score: 0.8842                     |
|                                                                                   |
|  [> Inspect Retrieved Chunks & Scores (Observability)]                            |
+-----------------------------------------------------------------------------------+
```

---

## 1. Problem Statement & Objectives

Standard Large Language Models (LLMs) suffer from hallucinations, outdated parametric knowledge, and an inability to query private enterprise files. Naive RAG implementations often retrieve irrelevant noise or force answers when uploaded documents lack sufficient context.

**DocuMind AI solves this by delivering:**
1. **Grounded Question Answering**: Enforces strict context-only answers with inline citations.
2. **Document-Scoped Search**: Allows users to filter retrieval to specific uploaded files.
3. **Multi-Step LangGraph Orchestration**: Executes query rewriting, neural reranking, quality filtering, and post-generation groundedness audits.
4. **Hallucination Protection**: Refuses to answer safely when context is insufficient.
5. **Observability**: Displays similarity scores, rerank scores, retrieved chunks, and graph execution metrics.

---

## 2. System Architecture & Workflow

```mermaid
flowchart TD
    subgraph Frontend [Streamlit UI]
        UI[Streamlit Application :8501]
    end

    subgraph Backend [FastAPI Application :8000]
        API[FastAPI Endpoints /api/v1]
        DS[DocumentService]
        RET[DocumentRetriever]
        RER[RerankerService]
        EMB[EmbeddingService: BAAI/bge-small-en-v1.5]
    end

    subgraph LangGraphWorkflow [LangGraph RAG Workflow]
        N1[validate_query] --> N2[rewrite_query]
        N2 --> N3[retrieve]
        N3 --> N4[rerank]
        N4 --> N5[filter_context]
        N5 --> N6[generate_answer]
        N6 --> N7[groundedness_check]
        N7 -- Grounded --> N8[final_response]
        N7 -- Ungrounded & Retry < 2 --> N6
        N7 -- Ungrounded & Max Retries --> N8
    end

    subgraph Storage [Persistent Storage]
        FAISS_DB[(FAISS Vector Index & Metadata)]
    end

    UI <--> API
    API --> DS
    API --> LangGraphWorkflow
    LangGraphWorkflow <--> FAISS_DB
    LangGraphWorkflow <--> Gemini[Google Gemini API]
=======
<<<<<<< HEAD
# 📚 DocuMind – AI Document Question Answering Assistant

DocuMind is an intermediate-level, production-ready **Retrieval-Augmented Generation (RAG)** web application built with **Streamlit**, **LangChain**, **PyMuPDF**, **sentence-transformers**, **FAISS**, and the **Groq LLM API** (Llama 3.3 / 3.1).

Unlike simple LLM API integrations, DocuMind implements a complete, end-to-end vector search retrieval pipeline to extract, chunk, embed, index, retrieve, and ground answers strictly within user-uploaded PDF and TXT documents.

---

## 🎯 Problem Statement

Standard Large Language Models (LLMs) suffer from two primary limitations when querying private or custom domain documents:
1. **Knowledge Cutoffs & Lack of Private Context**: LLMs cannot access private user files or real-time document context.
2. **Hallucination**: When asked about specific un-trained facts, LLMs tend to generate plausible-sounding but incorrect information.

**DocuMind solves this** by dynamically retrieving relevant document chunks from a local vector database and feeding only the verified context to the LLM under strict grounding constraints.

---

## ✨ Features

- 📁 **Multi-Document Support**: Upload single or multiple PDF and TXT files simultaneously.
- 📄 **Page-Aware PDF Extraction**: High-performance PDF parsing using PyMuPDF (`fitz`), preserving exact page numbers and filenames.
- ✂️ **Smart Text Chunking**: Configurable recursive character chunking with context-preserving overlaps.
- 🧠 **Local Vector Embeddings**: Generate 384-dimensional dense vector representations using `sentence-transformers/all-MiniLM-L6-v2` cached in memory.
- ⚡ **Local FAISS Indexing**: Blazing fast similarity search without third-party vector database subscriptions.
- 🎯 **Grounded Answer Generation**: Powered by Groq's high-speed Llama models (`llama-3.3-70b-versatile`) with zero-hallucination prompt enforcement.
- 🔍 **Source Citation Transparency**: Expandable source views showing exact document filename, page number, similarity distance, and chunk text.
- 🛡️ **No-Answer & Relevance Filtering**: Automatically detects when requested information is absent from documents and notifies the user instead of hallucinating.

---

## 🏗️ Architecture & RAG Pipeline

### Sequence Workflow

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as Streamlit App
    participant Loader as Document Loader (PyMuPDF)
    participant Chunker as Text Chunker
    participant Embedder as Sentence Transformer
    participant VectorStore as Local FAISS Store
    participant LLM as Groq API (Llama-3.3)

    User->>UI: Upload PDF / TXT & Click "Process Documents"
    UI->>Loader: Extract text page-by-page
    Loader-->>UI: Raw text + Metadata (Source, Page)
    UI->>Chunker: Split text into overlapping chunks
    Chunker-->>UI: Chunked Documents
    UI->>Embedder: Generate 384d Embeddings
    Embedder-->>UI: Vector Arrays
    UI->>VectorStore: Build & store FAISS Index
    VectorStore-->>UI: Index Ready Notification

    User->>UI: Ask Natural Language Question
    UI->>Embedder: Convert Question to Embedding Vector
    Embedder-->>UI: Query Vector
    UI->>VectorStore: Search Top-K Nearest Neighbors
    VectorStore-->>UI: Top Chunks + Distance Scores
    UI->>UI: Filter Relevance & Construct Context Block
    UI->>LLM: Send Context + Question + Strict Grounding Prompt
    LLM-->>UI: Grounded Answer Text
    UI-->>User: Display Answer + Expandable Sources & Citations
>>>>>>> be79986e90c8ea7dbe607f02acb5d68fb39347e1
```

---

<<<<<<< HEAD
## 3. Technology Stack

| Component | Technology | Purpose / Rationale |
| :--- | :--- | :--- |
| **Language** | Python 3.11 | Primary language for AI engineering |
| **Backend Framework** | FastAPI | High-performance async REST API & automatic OpenAPI docs |
| **Frontend UI** | Streamlit | Responsive dashboard with real-time updates and observability |
| **Workflow Engine** | LangGraph | Stateful multi-step RAG graph execution & retry loops |
| **Vector Index** | FAISS (`faiss-cpu`) | Ultra-fast local similarity search with disk persistence |
| **Embeddings** | `BAAI/bge-small-en-v1.5` | Top-performing 384-dim dense embedding model |
| **Reranker** | `BAAI/bge-reranker-base` | Neural CrossEncoder for precision context re-ordering |
| **Document Parsers** | PyMuPDF (`fitz`), TXT, MD | Native page-aware PDF text extraction and MD section parsing |
| **LLM Provider** | Google Gemini (`google-genai`) | Modern default LLM generation provider |
| **Validation** | Pydantic v2 | Strict request/response payload validation |
| **Containerization** | Docker & Docker Compose | Production multi-container orchestration |

---

## 4. LangGraph Multi-Step RAG Workflow

The core RAG engine is modeled as a stateful graph in `app/graph/workflow.py`:

```
START -> validate_query -> rewrite_query -> retrieve -> rerank -> filter_context -> generate_answer -> groundedness_check -> (condition) -> final_response -> END
```

1. **`validate_query`**: Validates input string length and checks for empty queries.
2. **`rewrite_query`**: Uses LLM to reformulate ambiguous or pronoun-heavy questions into standalone search queries.
3. **`retrieve`**: Performs vector search on FAISS index with document-scoped filtering (`selected_document_ids`).
4. **`rerank`**: Reranks top-10 candidates using CrossEncoder (`bge-reranker-base`) down to top-4.
5. **`filter_context`**: Evaluates relevance scores against a threshold (`0.35`). Rates context quality (`HIGH`, `MEDIUM`, `LOW`, `INSUFFICIENT`).
6. **`generate_answer`**: Generates grounded answer using strict system prompt instructions.
7. **`groundedness_check`**: Runs an audit check to verify answer facts against retrieved context. Triggers a retry loop (max 2) if ungrounded.
8. **`final_response`**: Formats inline citations (`[Document: filename, Page: X]`) and returns execution state.

---

## 5. Repository Structure

```
documind-ai/
├── app/
│   ├── main.py                     # FastAPI entry point
│   ├── api/                        # REST API routes
│   │   ├── dependencies.py
│   │   ├── routes_ingestion.py     # File upload & document management
│   │   └── routes_query.py         # RAG execution & streaming endpoints
│   ├── core/                       # Config, logging, exceptions
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── exceptions.py
│   ├── schemas/                    # Pydantic v2 schemas
│   │   ├── document.py
│   │   ├── query.py
│   │   └── response.py
│   ├── ingestion/                  # Parsers & chunker
│   │   ├── loaders.py              # PyMuPDF PDF, TXT, MD loader
│   │   ├── parser.py               # Section & whitespace parser
│   │   └── chunker.py              # Recursive chunker with overlap
│   ├── embeddings/                 # Embeddings service
│   │   └── embedding_service.py    # BAAI/bge-small-en-v1.5 wrapper
│   ├── retrieval/                  # FAISS store & reranker
│   │   ├── faiss_store.py          # Persistent FAISS index & metadata DB
│   │   ├── retriever.py            # Document-scoped retriever
│   │   └── reranker.py             # BAAI/bge-reranker-base CrossEncoder
│   ├── generation/                 # LLM interface & prompts
│   │   ├── prompts.py
│   │   └── llm.py                  # Gemini LLM provider abstraction
│   ├── graph/                      # LangGraph state machine
│   │   ├── state.py
│   │   ├── nodes.py
│   │   └── workflow.py
│   └── services/                   # Business logic coordination
│       └── document_service.py
├── frontend/
│   └── streamlit_app.py            # Streamlit UI dashboard
├── data/
│   ├── uploads/                    # Archived uploaded raw files
│   └── indexes/                    # FAISS binary index & metadata JSON
├── tests/                          # Pytest test suite
│   ├── conftest.py
│   ├── test_ingestion.py
│   ├── test_faiss.py
│   ├── test_graph.py
│   └── test_api.py
├── docs/
│   └── architecture.md             # Detailed system architecture document
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
=======
## 🛠️ Technology Stack & Selection Rationale

| Layer | Technology | Rationale |
| :--- | :--- | :--- |
| **Frontend UI** | Streamlit | Rapid, interactive, Python-native Web UI with reactive state management. |
| **PDF Extraction** | PyMuPDF (`fitz`) | 10x faster than PyPDF/pdfplumber, highly accurate text & layout extraction with page tracking. |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` | Respects natural paragraph (`\n\n`), sentence (`\n`), and word boundaries. |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Lightweight (80MB), fast, 384-dimensional vector space optimized for semantic search. |
| **Vector Index** | FAISS (`faiss-cpu`) | Facebook AI Similarity Search offers zero-latency, local, in-memory vector retrieval without API limits. |
| **LLM Inference** | Groq API (`llama-3.3-70b-versatile`) | Ultra-low latency inference engine delivering >300 tokens/sec for Llama 3 models. |
| **Environment** | `python-dotenv` | Clean security management for API keys. |

---

## 🔬 Technical Deep-Dive

### 1. Chunking Strategy & Overlap Utility
- **Why Chunking is Necessary**: Large documents exceed LLM context windows and degrade attention precision. Chunking breaks documents into micro-contexts.
- **Why Overlap is Useful**: Splitting text strictly by character count risks breaking sentences in half. A **chunk overlap of 150 characters** ensures that boundary sentences appear in both adjacent chunks, maintaining continuity.
- **Configuration**:
  - `chunk_size = 800` characters (~120-150 words)
  - `chunk_overlap = 150` characters (~20-25 words)

### 2. Embeddings & FAISS Indexing
- The model `all-MiniLM-L6-v2` converts text into normalized 384-dimensional vectors.
- Similarity is computed using **L2 (Euclidean) Distance**:
  $$\text{Distance} = \sqrt{\sum_{i=1}^{n} (q_i - d_i)^2}$$
- Smaller L2 distance values indicate higher semantic similarity.

### 3. Grounding & Hallucination Mitigation
DocuMind employs a **two-tier defense** against hallucinations:
1. **Distance Thresholding**: Chunks with L2 distance $> 1.35$ are flagged as low relevance.
2. **System Prompt Enforcement**:
   > *"Answer the user's question using ONLY the provided document context... If the answer cannot be determined from the provided context, clearly state that the information is not available in the uploaded documents."*

---

## 📂 Project Architecture

```
documind-rag/
│
├── app.py                     # Streamlit web application interface
├── requirements.txt           # Python dependencies
├── .env                       # Local environment file (API keys)
├── .env.example               # Template environment configuration
├── .gitignore                 # Git safety & exclusion rules
├── README.md                  # Complete project documentation
│
├── src/                       # Modular RAG core package
│   ├── __init__.py            # Package initialization
│   ├── document_loader.py     # PDF & TXT text extraction + metadata
│   ├── chunker.py             # Recursive text splitting & stats
│   ├── embeddings.py          # Cached sentence-transformers model loader
│   ├── vector_store.py        # FAISS vector store creation & persistence
│   ├── retriever.py           # Similarity search & context construction
│   ├── llm.py                 # Groq API client & grounded system prompt
│   └── rag_pipeline.py        # Orchestration pipeline
│
└── sample_documents/          # Sample documents for evaluation
    ├── company_policy.txt
    ├── generate_sample_pdf.py # Helper script to build sample PDF
    └── ai_research_report.pdf
>>>>>>> be79986e90c8ea7dbe607f02acb5d68fb39347e1
```

---

<<<<<<< HEAD
## 6. Installation & Execution

### Prerequisites
- Python 3.11+
- Google Gemini API Key (`GEMINI_API_KEY`)

### Environment Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/documind-ai.git
   cd documind-ai
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and set your `GEMINI_API_KEY`:
   ```env
   GEMINI_API_KEY=AIzaSy...your_real_key
   ```

---

### Option A: Local Execution

1. **Start FastAPI Backend**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   * API Documentation available at: `http://localhost:8000/api/v1/docs`

2. **Start Streamlit Frontend** (in a second terminal):
   ```bash
   streamlit run frontend/streamlit_app.py --server.port=8501
   ```
   * Open browser at: `http://localhost:8501`

---

### Option B: Docker Execution

Run the complete multi-container stack with a single command:

```bash
docker-compose up --build
```

- **Streamlit Frontend**: `http://localhost:8501`
- **FastAPI Backend & Swagger**: `http://localhost:8000/api/v1/docs`

---

## 7. API Summary & Example Usage

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | `GET` | System status, indexed document count, and vector telemetry |
| `/api/v1/documents/upload` | `POST` | Upload PDF, TXT, or MD document for chunking & indexing |
| `/api/v1/documents` | `GET` | List all indexed document metadata |
| `/api/v1/documents/{document_id}` | `GET` | Retrieve single document metadata |
| `/api/v1/documents/{document_id}/chunks` | `GET` | Retrieve all chunks of a document |
| `/api/v1/documents/{document_id}` | `DELETE` | Delete a document and purge vector index |
| `/api/v1/query` | `POST` | Execute full multi-step LangGraph RAG query |
| `/api/v1/query/stream` | `POST` | Execute RAG query with streaming token response |

### Example cURL Request (`POST /api/v1/query`)

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "What is supervised learning?",
           "selected_document_ids": ["doc_123"],
           "top_k": 10,
           "enable_reranking": true
         }'
```

---

## 8. Testing Suite

Run unit and integration tests using `pytest`:

```bash
pytest tests/ -v
```

**Tested Functionality:**
- File validation (unsupported extensions, empty files, size limits).
- Text loading & PDF page preservation.
- Recursive chunking and overlap logic.
- FAISS vector indexing, persistent disk load/save, and document deletion.
- Document-scoped retrieval filtering.
- Context quality filtering and ungrounded refusal logic.
- Full FastAPI endpoint integration using `TestClient`.

---

## 9. Hallucination Reduction Strategy

1. **Strict Context-Only System Prompt**: System instructions explicitly forbid using outside facts.
2. **Quality Relevance Filter**: Drops chunks below relevance threshold (`0.35`). If all chunks are dropped, returns an immediate safe refusal.
3. **Groundedness Audit Node**: After answer generation, an LLM evaluator verifies whether every claim is supported by the retrieved context snippets.
4. **Controlled Retry Loop**: If the answer is ungrounded, it retries answer generation with stricter instructions up to 2 times before defaulting to a safe fallback.

---

## 10. Internship Evaluation Checklist

- [x] Python 3.11 modular project architecture.
- [x] FastAPI REST API backend with Pydantic v2 schemas.
- [x] Streamlit interactive frontend dashboard.
- [x] LangGraph multi-step state graph RAG workflow.
- [x] FAISS vector store with disk persistence.
- [x] PyMuPDF PDF page extraction and Markdown section parsing.
- [x] `BAAI/bge-small-en-v1.5` embeddings & `BAAI/bge-reranker-base` neural reranking.
- [x] Google Gemini LLM default generation provider.
- [x] Document-scoped retrieval filter.
- [x] Hallucination refusal & groundedness verification loop.
- [x] Docker & Docker Compose setup.
- [x] Comprehensive pytest test suite.
- [x] Professional interviewer-ready documentation.

---

## 11. GitHub Submission & Video Demonstration Guide

For your **ShadowFox AI Engineer Internship Submission**:

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "feat: complete production-style DocuMind AI RAG system"
   git branch -M main
   git remote add origin https://github.com/yourusername/documind-ai.git
   git push -u origin main
   ```
2. **Demonstration Video Outline (3-5 Minutes)**:
   - **0:00 - 0:45**: Introduction & Architecture Overview (Explain FastAPI + LangGraph + FAISS + Gemini stack).
   - **0:45 - 1:45**: Ingestion Demo (Upload a PDF/MD document, demonstrate chunking & FAISS index creation).
   - **1:45 - 2:45**: Grounded Q&A & Document-Scoped Search (Ask a document question, show inline citations, inspect retrieved chunks & rerank scores).
   - **2:45 - 3:30**: Hallucination Defense & Refusal Demo (Ask a question NOT in the uploaded document, demonstrate safe refusal response).
   - **3:30 - 4:15**: Code Walkthrough & Docker Execution (Show `app/graph/workflow.py`, run `docker-compose up`, show pytest execution).

---

### License
MIT License. Developed for ShadowFox AI Engineer Internship.
=======
## 🚀 Installation & Local Setup Guide

### Step 1: Clone Repository & Navigate
```bash
git clone https://github.com/your-username/documind-rag.git
cd documind-rag
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Groq API Key
1. Obtain a free Groq API key from [https://console.groq.com](https://console.groq.com).
2. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
3. Edit `.env` and set your key:
```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

### Step 5: (Optional) Generate Sample PDF Test Document
```bash
python sample_documents/generate_sample_pdf.py
```

### Step 6: Launch Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Testing Plan & Validation Matrix

To verify the RAG system performance during evaluation, perform the following 6 tests:

| Test Case | Inputs / Action | Expected Result | Pass Criteria |
| :--- | :--- | :--- | :--- |
| **Test 1: In-Document Query** | Upload `company_policy.txt`, ask: *"How many days of paid vacation do employees receive?"* | Answers **20 days**, citing `company_policy.txt` | Grounded answer with exact source citation |
| **Test 2: Out-of-Document Query** | Ask: *"What is the capital of France?"* or *"What is DocuMind's stock price?"* | Responds: *"I couldn't find sufficient information in the uploaded documents..."* | Zero hallucination, strict fallback message |
| **Test 3: Multi-Document Retrieval** | Upload both `company_policy.txt` and `ai_research_report.pdf`. Ask about RAG benchmarks. | Retrieves from `ai_research_report.pdf` page 2 | Multi-doc citation accuracy |
| **Test 4: Empty Question** | Click "Ask Question" with blank text box | Displays warning: *"Please enter a question."* | Input validation triggered |
| **Test 5: Unsupported File** | Attempt uploading `.docx` or `.png` file | Rejects upload with friendly error message | Graceful error handling |
| **Test 6: Pre-Processing Query** | Ask question before clicking "Process Documents" | Displays warning: *"Please process documents first."* | Pipeline state check |

---

## ☁️ Deployment Guide (Streamlit Community Cloud)

1. Push your completed project repository to **GitHub** (ensure `.env` is **NOT** committed).
2. Log into [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New App"** and select your repository, branch (`main`), and main file path (`app.py`).
4. Go to **Advanced Settings -> Secrets** and paste your API key:
   ```toml
   GROQ_API_KEY = "gsk_your_actual_groq_api_key_here"
   ```
5. Click **Deploy**. Your RAG assistant will be live online!

---

## 🎓 Internship Interview Defense & Explanation Guide

When presenting DocuMind in an internship interview, walk the evaluator through these 4 key talking points:

1. **Architecture Separation (`src/`)**: Explain how document ingestion (`document_loader.py`), chunking (`chunker.py`), indexing (`vector_store.py`), search (`retriever.py`), and LLM invocation (`llm.py`) are decoupled so each layer can be tested or replaced independently (e.g., swapping FAISS for ChromaDB or sentence-transformers for OpenAI embeddings).
2. **Metadata Retention**: Emphasize how `source` filename and `page` numbers are attached at step 1 during PyMuPDF parsing and preserved all the way through chunking into FAISS metadata payloads to provide verifiable UI citations.
3. **Hallucination Prevention Mechanics**: Explain how combining dense vector distance thresholding with strict system prompt boundaries prevents the model from generating external facts when query context is absent.
4. **Efficiency**: Point out the use of `@st.cache_resource` for the 80MB embedding model to prevent memory leaks and redundant disk reads during Streamlit app re-renders.

---

## 🔮 Limitations & Future Improvements

- **Scanned PDFs**: Current PyMuPDF extraction handles native text PDFs; integrating OCR (Tesseract / EasyOCR) would enable scanned image PDF parsing.
- **Hybrid Retrieval**: Combining dense FAISS vector search with sparse keyword search (BM25) would improve exact match keyword queries (e.g., product IDs, SKU numbers).
- **Reranking**: Adding a Cross-Encoder reranker model (e.g., `bge-reranker-large`) after initial FAISS retrieval to further improve top-K chunk precision.
=======
# shadowfox-internship-repo
this is my project developed during internship
>>>>>>> ed89d76d1e8d748893e9a4c1081037fe1a482053
>>>>>>> be79986e90c8ea7dbe607f02acb5d68fb39347e1
