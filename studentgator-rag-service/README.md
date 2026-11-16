# StudentGator Production RAG Advisory Microservice

Production-grade asynchronous microservice for multi-jurisdiction international university admissions, visa compliance, and practical employment regulations (Germany, United States, France, Spain, and StudentGator platform guidelines).

---

## 1. Directory Structure

```text
studentgator-rag-service/
├── requirements.txt            # Microservice runtime dependencies
├── package.json                # npm / pnpm run scripts
├── .env                        # Local model keys & persistence settings
├── start.bat                   # 1-click startup (FastAPI server + auto-opens Swagger UI)
├── chat.bat                    # 1-click interactive multi-turn terminal chat
├── compare_rag.bat             # 1-click benchmark (Naive RAG vs Proper Advanced RAG)
├── test_client.py              # CLI client for SSE token-streaming chat
├── static/
│   └── index.html              # Embedded web chat interface with live SSE token rendering
├── data/
│   ├── chroma_db/              # Persistent ChromaDB vector index (1,100 deduplicated chunks)
│   ├── documents_manifest.json # Institutional metadata registry
│   └── *.pdf                   # 8 official institutional immigration & admissions handbooks
└── src/
    ├── main.py                 # FastAPI ASGI application & SSE streaming endpoints
    ├── config.py               # Pydantic v2 application settings
    ├── metadata.py             # Strongly-typed document metadata schemas & filters
    ├── retriever.py            # Hybrid RRF (Dense Chroma + Sparse BM25) & Cross-Encoder
    ├── ingestion.py            # Offline document chunking & vector indexing pipeline
    ├── schemas.py              # Request / Response schemas
    └── graph/
        ├── state.py            # Centralized LangGraph state schema
        ├── router.py           # Pure decision gates (greeting / out-of-scope / rag)
        ├── nodes.py            # State-mutating worker nodes & LLM generator
        └── workflow.py         # Compiled StateGraph with MemorySaver checkpointer
```

---

## 2. Quickstart (Local Development)

### Prerequisites
- Python 3.11+
- Virtual environment with dependencies installed:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\activate
  pip install -r requirements.txt
  ```

### Running the Microservice
- **Windows (1-Click)**: Double-click **`start.bat`**
- **Manual (PowerShell / Bash)**:
  ```powershell
  python src/main.py
  ```
  The server boots on `http://localhost:8000` and automatically launches your browser to:
  - **Swagger API Docs**: `http://localhost:8000/docs`
  - **Interactive Web Chat**: `http://localhost:8000/chat`
  - **Healthcheck**: `http://localhost:8000/api/v1/health`

### Interactive Terminal Chat
```powershell
chat.bat
```

### Naive vs Proper RAG Benchmark
```powershell
compare_rag.bat
```

---

## 3. Architectural Highlights

1. **Decoupled Search vs. Generation (Hierarchical Chunking)**:
   Overlapping 400-char child chunks are indexed into ChromaDB and BM25 for maximum retrieval precision.
2. **Hybrid Retrieval with Reciprocal Rank Fusion (RRF)**:
   Fuses dense semantic embeddings (`sentence-transformers/all-MiniLM-L6-v2`) with sparse lexical frequency (`BM25`) at a 60/40 ratio to capture both conceptual synonyms and exact statutory codes (e.g. `§16b AufenthG`).
3. **Two-Stage Cross-Encoder Re-Ranking**:
   Applies `cross-encoder/ms-marco-MiniLM-L-6-v2` across all candidates for full token-to-token cross-attention, accurately resolving negations (*"without limit"*), restrictions, and legal exceptions.
4. **LangGraph v2 Multi-Turn State Machine**:
   Maintains conversation state with memory checkpointing, routing off-topic queries in 0ms without hitting the vector database.
5. **Real-time Server-Sent Events (SSE)**:
   Streams tokens directly into the browser or test client via `StreamingResponse` wrapping `astream_events(..., version="v2")`.

---

## 4. Docker Containerization

The microservice utilizes a multi-stage `Dockerfile` optimizing build speed, final image footprint, and container security:

- **Stage 1 (`builder`)**: Installs CPU-only PyTorch wheels (~180MB vs 540MB GPU CUDA) and production dependencies from `requirements.txt`.
- **Stage 2 (`runner`)**: Uses `python:3.11-slim`, sets up dedicated non-root execution user `appuser`, configures unbuffered logging, points HuggingFace cache to writable `/tmp`, copies the pre-built virtual environment and local vector database (`data/chroma_db`), and executes `uvicorn src.main:app --host 0.0.0.0 --port ${PORT}`.

### Run with Docker Compose
```powershell
docker compose up --build
```
Access Swagger UI at `http://localhost:8080/docs` and web chat at `http://localhost:8080/`.

### Run with Docker Directly
```powershell
docker build -t studentgator-rag-service .
docker run -p 8080:8080 -e PORT=8080 -e GEMINI_API_KEY="your-key-here" studentgator-rag-service
```

---

## 5. Automated CI/CD Pipeline (GitHub Actions)

The workflow at `.github/workflows/deploy.yml` triggers automatically on every push to `main` touching `studentgator-rag-service/**`:

1. **Code Quality (`lint`)**: Enforces style and syntax hygiene with Ruff (`ruff check --select E,F`).
2. **Bytecode Compilation (`build`)**: Verifies Python syntax across all modules with `python -m compileall`.
3. **Unit Test Matrix (`test`)**: Parallel testing across Python 3.11, 3.12, and 3.13 verifying Pydantic request contracts, relational metadata filters, and LangGraph routing.
4. **RAG Ingestion Tests (`ingest`)**: Validates PDF document extraction, recursive chunking, and isolated vector store indexing.
5. **Container Packaging (`build-and-push`)**: Builds the multi-stage Docker image and pushes tags `:${{ github.sha }}` and `:latest` to GitHub Container Registry (`ghcr.io/hamshary99/studentgator-rag-service`).
6. **Container Verification (`verify-container`)**: Pulls the published container image, spins up an isolated smoke test container, and polls `/api/v1/health` until `{"status":"healthy"}` is confirmed.

---

## 6. Cloud Deployment & Hosting Cost Analysis

### Render Cloud Deployment
- **Runtime**: Docker (auto-detected from `Dockerfile`)
- **Instance Type**: Free Web Service (0.1 CPU, 512 MB RAM)
- **Port Binding**: Automatically detected via `${PORT}` environment variable
- **Offline Embeddings**: Automatically loads pre-built vector collections in `data/chroma_db` using local `all-MiniLM-L6-v2`, making 0 external embedding API calls during startup.

### Cost Trade-off: Render vs. Google Cloud Run

| Feature / Dimension | Render (Free Tier) | Google Cloud Run (Fully Managed) |
| :--- | :--- | :--- |
| **Monthly Base Cost** | $0.00 / month | $0.00 / month (within free tier) |
| **Scale to Zero** | Yes (spins down after 15 min inactivity) | Yes (spins down to 0 instances instantly) |
| **Cold Start Latency** | ~30 - 45 seconds on wake-up | ~4 - 8 seconds with container image streaming |
| **Credit Card / Wallet** | Not required for Free tier | Mandatory billing setup and project linkage |
| **Production Cost (100k requests/mo)** | $7.00/mo (Starter 0.5 CPU / 512MB) | ~$2.10 - $3.40/mo (Per-request vCPU-sec billing) |
| **Persistent Disk** | Paid tiers only ($0.25/GB/mo) | Cloud Storage FUSE / Cloud SQL integration |

**Architecture Decision**: For zero-wallet student testing and internship capstone demonstration, Render Free Tier delivers immediate public accessibility with automated Git-push deployments. For enterprise SLA workloads exceeding 100,000 monthly queries, Cloud Run is more cost-efficient due to per-millisecond billing and rapid concurrency autoscaling.
