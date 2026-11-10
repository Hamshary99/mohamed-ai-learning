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
