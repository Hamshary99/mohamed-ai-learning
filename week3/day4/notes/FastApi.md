# FastAPI for Production AI Backends

## Why LLMs Need an API Gateway

In modern AI engineering, Large Language Models and agentic frameworks such as LangGraph cannot be exposed directly to end-user clients. A production AI gateway must:

* Handle client authentication
* Enforce strict request and response contracts
* Manage connection pooling
* Balance concurrent traffic
* Stream token-by-token outputs in real time

**FastAPI** has emerged as the industry standard framework for AI backends due to three core capabilities:

1. **Native Asynchronous Support:** Handles high-concurrency I/O (waiting on LLM APIs and vector databases) without blocking the server.
2. **Tight Integration with Pydantic:** Validates and sanitizes incoming and outgoing data structures automatically.
3. **Internal Dependency Injection Engine:** Manages shared resources (database connections, retrievers, auth checks) cleanly across endpoints.

---

## Pydantic v2 Data Contracts

### 1. Why It Exists (Protecting the Model from Malformed Requests)

At the gate of any AI microservice, data validation must prevent malformed payloads from consuming model tokens.

* **The Problem Without Validation:** If a frontend bug or malicious user sends an empty string (`"   "`), a 200,000-character text dump, or mistyped JSON keys directly into LangGraph, the system wastes expensive LLM tokens, crashes mid-graph, or leaks internal errors.
* **The Solution (The Gatekeeper Contract):** Pydantic models define the exact shape and constraints of incoming requests and outgoing responses. If an incoming JSON payload violates any rule, FastAPI immediately rejects it at the border with an **HTTP `422 Unprocessable Entity`** error *before* a single model token or database query is executed.

---

### 2. Production Code Reference: Request & Response Contracts

```python
from typing import Optional, List
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict


# =====================================================================
# 1. INCOMING REQUEST CONTRACT (Protects the AI Pipeline Input)
# =====================================================================
class ChatMessageRequest(BaseModel):
    # Global model rules:
    # - extra="forbid": Rejects requests containing unknown/unexpected JSON fields
    #   (prevents payload injection and catches frontend typos immediately).
    # - str_strip_whitespace=True: Strips leading/trailing spaces before validation
    #   (turns "   " into "", which then gets caught by min_length=1).
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # Stateful Thread ID:
    # - Must be a valid UUID format (blocks arbitrary strings or SQL/NoSQL injection).
    # - default_factory=uuid4: Calls uuid4() dynamically per request if the client
    #   omits thread_id, creating a fresh session for LangGraph checkpointing.
    thread_id: UUID = Field(
        default_factory=uuid4,
        description="Stateful conversation thread identifier for checkpointing."
    )

    # User Prompt Validation:
    # - ... (Ellipsis): Marks this field as strictly required.
    # - min_length=1: Prevents empty prompts from triggering wasted LLM calls.
    # - max_length=4000: Caps prompt length to prevent context-window overflow
    #   and token-exhaustion attacks.
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The raw user prompt submitted to the agentic workflow."
    )

    # Optional Metadata Pre-Filter:
    # - Defaults to None if omitted, or passes a clean string to filter ChromaDB.
    category_filter: Optional[str] = Field(
        default=None,
        description="Optional metadata pre-filter for institutional retrieval scoping."
    )

    # Streaming Toggle:
    # - Coerces and validates boolean flag for SSE streaming vs. blocking JSON.
    stream: bool = Field(
        default=True,
        description="Toggle between chunked Server-Sent Events streaming and blocking JSON."
    )


# =====================================================================
# 2. OUTGOING RESPONSE CONTRACT (Guarantees Predictable Client Output)
# =====================================================================
class CitationMetadata(BaseModel):
    """Schema for a single retrieved source document citation."""
    source: str
    page: Optional[int] = None
    form_code: Optional[str] = None
    relevance_score: float


class ChatMessageResponse(BaseModel):
    """
    Strict outbound contract returned to the client.
    Ensures the frontend always receives a predictable structure and strips out
    any internal LangGraph state or sensitive system variables.
    """
    thread_id: UUID
    answer: str
    # default_factory=list creates a new empty list per response if no citations exist
    citations: List[CitationMetadata] = Field(default_factory=list)
    total_tokens: int
    execution_time_ms: float
```

---

### 3. How the Contract Blocks Malformed Requests in Practice

| Malformed Client Payload | Which Pydantic Rule Catches It | What Happens |
| :--- | :--- | :--- |
| `{"message": "     "}` (Whitespace only) | `str_strip_whitespace=True` + `min_length=1` | Trimmed to `""`, fails `min_length=1`, blocked with `422` (0 tokens spent). |
| `{"message": "A" * 50000}` (Prompt bomb) | `max_length=4000` | Blocked at the gateway before hitting the embedding model or LLM. |
| `{"message": "Hi", "admin": true}` (Extra field) | `extra="forbid"` | Rejects the unapproved `"admin"` key immediately instead of ignoring it silently. |
| `{"thread_id": "123-bad-id", "message": "Hi"}` | `thread_id: UUID` | Fails UUID format parsing; prevents corrupting the LangGraph checkpointer DB. |
| `{}` (Missing `"message"` key) | `Field(...)` (Required Ellipsis) | Rejects payload immediately because the required field is missing. |

---

## The Single-Threaded Event Loop (ASGI & Uvicorn)

FastAPI runs on an **ASGI (Asynchronous Server Gateway Interface)** server like **Uvicorn**, using a single-threaded `asyncio` event loop per worker process.

* **Cooperative Task Scheduling:** When a task starts a slow network I/O operation (calling an external LLM API, querying PostgreSQL, or searching a vector DB), it hits `await` and pauses itself—handing control back to the event loop.
* **Zero Idle Waiting:** While that network request is in flight, the event loop immediately switches to handle other active requests instead of sitting idle.

```
Incoming Requests ──> [ Uvicorn Worker Process ]
                             │
                             ▼
              [ Single-Threaded asyncio Event Loop ]
                │
                ├── Task 1: Awaiting external LLM API (I/O) ──> [Paused]
                ├── Task 2: Parsing incoming HTTP JSON ───────> [Active]
                └── Task 3: Streaming SSE chunk to Client ────> [Active]
```

* **Why This Matters for AI Backends:** An LLM API call spends **95%+ of its time just waiting** on remote token generation. Using `async def` endpoints lets a single CPU core juggle hundreds of concurrent user chats at once without spawning or thrashing heavy OS threads.

### The Problem of Blocking the Event Loop

* **The Rule:** Never run synchronous blocking I/O (`time.sleep()`, `requests.get()`, local disk reads) or heavy CPU math inside an `async def` endpoint.
* **What Happens:** Because there is only **one thread**, a blocking call without `await` freezes the entire event loop—new requests cannot be accepted, active SSE streams stall, and container health checks time out.

```python
# Anti-pattern: freezes the entire Uvicorn worker for 2 seconds
@app.post("/blocking-chat")
async def blocking_chat(payload: ChatMessageRequest):
    time.sleep(2)  # No 'await' -> hijacks the single thread and blocks ALL users
    return {"status": "delayed"}
```

### Starlette's `run_in_threadpool` and the 40-Worker Thread Limit

FastAPI handles route definitions differently based on whether you use `async def` or plain `def`:

| Route Declaration | Where It Runs | Developer Rule |
| :--- | :--- | :--- |
| **`async def`** | Directly on the main `asyncio` event loop. | Every I/O call **must** be non-blocking (`await`). |
| **`def`** | Offloaded to Starlette's internal thread pool (`anyio.to_thread.run_sync` / `run_in_threadpool`). | Safe for legacy blocking libraries (`time.sleep`, `requests`), because the main event loop stays free. |

```python
# Synchronous route: safely offloaded to a background worker thread
@app.post("/sync-safe-chat")
def sync_safe_chat(payload: ChatMessageRequest):
    time.sleep(2)  # Blocks only 1 worker thread; main event loop keeps running
    return {"status": "success"}
```

* **The 40-Thread Ceiling:** Starlette's thread pool is capped at **40 worker tokens** by default. If 40 users trigger a slow 5-second `def` route simultaneously, the **41st request queues in memory** waiting for a thread to free up.
* **AI Production Rule:** Because LLM inference is slow and high-concurrency, `def` thread pools saturate quickly. Always write AI pipelines natively with `async def` and async clients (e.g., `httpx.AsyncClient`).

---

## Dependency Injection (`Depends`)

Instead of copy-pasting complex setup logic inside every route call (like opening a database/vector store connection or validating an auth token), **Dependency Injection (DI)** decouples that setup from the route's business logic—preparing those resources in a standalone function and handing them directly to the endpoint via `Depends()`.

* **Purpose:** Separates infrastructure setup (DB connections, auth checks, loading ChromaDB/retrievers) from route business logic.
* **What It Does:** Runs the setup function automatically before the endpoint starts and injects the ready-to-use result via `Depends()`.
* **Why Use It:** Reusability across dozens of routes, zero duplicated connection code, automatic cleanup (via `yield`), and 1-line mock replacement during unit testing (`app.dependency_overrides`).

### Before vs. After Dependency Injection

```python
from fastapi import FastAPI, Depends, Header, HTTPException

app = FastAPI()


# =====================================================================
# BEFORE DI (Bad: Setup & Auth logic copy-pasted inside every route)
# =====================================================================
@app.post("/chat-without-di")
async def chat_without_di(question: str, api_key: str = Header(None)):
    # 1. Repeated Auth Check (have to copy-paste this into every route!)
    if api_key != "secret-sg-key":
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # 2. Repeated DB / Vector Store connection setup
    vector_db = connect_to_chroma(collection="visa_docs")

    # 3. Actual Business Logic finally starts here
    docs = await vector_db.asearch(question)
    return {"answer": docs}


# =====================================================================
# AFTER DI (Clean: Setup is decoupled and handed in via Depends)
# =====================================================================
# 1. Standalone reusable setup functions
async def verify_api_key(api_key: str = Header(None)) -> str:
    if api_key != "secret-sg-key":
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key


def get_vector_db():
    return connect_to_chroma(collection="visa_docs")


# 2. Route only contains pure business logic; FastAPI hands in the resources!
@app.post("/chat-with-di")
async def chat_with_di(
    question: str,
    _auth: str = Depends(verify_api_key),     # Runs auth check automatically
    vector_db = Depends(get_vector_db),       # Hands ready-to-use DB to route
):
    docs = await vector_db.asearch(question)
    return {"answer": docs}
```

---

## Real-Time Streaming Protocols: Server-Sent Events (SSE) for LLMs

* **The Standard HTTP Bottleneck:** In a normal HTTP request-response cycle, the server cannot stream chunks as they are generated—it has to buffer the **entire** LLM output in memory before sending a single response payload back to the client.
* **Why This Hurts UX:** Because LLMs generate tokens one by one at **15 to 80 tokens/second**, a full **500-token answer** forces the user to stare at a loading spinner for **8 to 25 seconds** waiting for the complete response to finish before seeing a single word (rather than streaming tokens immediately as they arrive).

### 1. Time-to-First-Token (TTFT) vs. Total Latency

Users do not judge an AI app by how long the entire 500-token essay takes to finish (**Total Latency**). They judge it by how fast the **first word** appears on screen (**Time-to-First-Token / TTFT**):

$$\text{Perceived Latency (TTFT)} = t_{\text{network}} + t_{\text{retrieval}} + t_{\text{prefill}}$$

* **Without Streaming:** Perceived latency = **Total Latency** (~15,000 ms of staring at a frozen screen).
* **With Streaming:** Perceived latency = **TTFT** (**< 400 ms** to see the first word, followed by a live typing effect while the user reads).

---

### 2. Server-Sent Events (SSE) vs. WebSockets

For LLM token streaming, **SSE** is preferred over **WebSockets** because chat generation is fundamentally one-way once the prompt is sent:

| Feature | WebSockets | Server-Sent Events (SSE) |
| :--- | :--- | :--- |
| **Direction** | **Bi-directional** (Client <-> Server at the same time). | **Uni-directional** (Client sends 1 `POST`, Server pushes a stream back). |
| **Protocol** | Custom TCP framing (`ws://` / `wss://`). | Standard HTTP/1.1 or HTTP/2 (`Content-Type: text/event-stream`). |
| **Infrastructure** | Harder to load-balance, bypasses HTTP caching, often blocked by strict corporate firewalls. | Works out-of-the-box with standard HTTP proxies, auth headers, and firewalls. |
| **Best Use Case** | Live multiplayer games, two-way voice agents. | **LLM text streaming** (ChatGPT / StudentGator advisor chat). |

```
Client                                      FastAPI Gateway
  │                                                │
  ├─── POST /api/chat (stream=True) ──────────────>│
  │<── HTTP/1.1 200 OK (text/event-stream) ────────┤
  │                                                ├──> 1. Retrieve RAG Chunks
  │<── data: {"type": "citation", ...}\n\n ────────┤
  │                                                ├──> 2. LLM Prefill & Generate
  │<── data: {"type": "token", "content": "Form"}──┤
  │<── data: {"type": "token", "content": " I"}────┤
  │<── data: {"type": "token", "content": "-20"}───┤
  │<── data: {"type": "done"}\n\n ─────────────────┤
```

---

### 3. The SSE Message Anatomy

SSE sends plain-text blocks over an open HTTP connection. Every message block **must end with two newlines (`\n\n`)** so the browser knows the event is complete:

```text
event: message
id: 1042
retry: 5000
data: {"type": "token", "content": "travel signature"}

```

* **`data:`** The actual payload (usually a JSON string containing the token, citation, or status).
* **`event:`** Optional label (`token`, `citation`, `error`, `done`) so the frontend knows how to render the chunk.
* **`id:`** Event counter—if the user's Wi-Fi drops briefly, the browser sends `Last-Event-ID: 1042` to resume without losing tokens.
* **`retry:`** Tells the browser how many milliseconds (e.g., `5000` = 5s) to wait before reconnecting if the stream drops.

---

### 4. Production Stream Hardening (3 Things That Break in Prod)

1. **Reverse Proxy Buffering (`"X-Accel-Buffering": "no"`):**
   * *Problem:* NGINX and cloud load balancers try to be helpful by collecting small response chunks in a buffer before sending them out—which ruins live token streaming.
   * *Fix:* Send the header `"X-Accel-Buffering": "no"` so the proxy flushes every single token to the user immediately.
2. **Client Disconnect Detection (`await request.is_disconnected()`):**
   * *Problem:* If a user closes their browser tab after 2 seconds, the backend will keep generating the remaining 450 tokens—wasting GPU compute and paid LLM API credits for nobody.
   * *Fix:* Check `if await request.is_disconnected(): break` inside the token loop to cancel the LLM call immediately.
3. **In-Stream Error Signaling (`data: {"type": "error", ...}`):**
   * *Problem:* Once the stream starts, the `HTTP 200 OK` status code is already locked in and sent to the browser. If the LLM provider crashes on token #150, FastAPI **cannot** retroactively send an `HTTP 500` status code.
   * *Fix:* Catch exceptions inside the generator and yield a structured error event (`data: {"type": "error", "message": "Provider timeout"}\n\n`) so the frontend UI can display a clean warning.

---

## Background Task Execution: Native `BackgroundTasks` vs. Distributed Queues (Celery + Redis)

* **The Core Problem:** When a student sends a chat message, the backend often needs to do extra chores after generating the answer (saving token analytics to Postgres, sending an email alert, or indexing a document). Making the user wait for those follow-up chores makes the API feel slow.
* **The Goal:** Return the HTTP response to the user **immediately**, and run non-critical or heavy jobs in the background.

### 1. Option A: FastAPI Native `BackgroundTasks` (In-Memory, Same Server)

* **How It Works:** Built directly into FastAPI/Starlette. The route sends the HTTP response back to the user first, and then runs the background function inside the **same FastAPI container**.
* **The Catch (No Persistence):** Tasks live in the FastAPI server's local RAM. If the container restarts or crashes mid-task, **the job is lost forever**, and CPU-heavy tasks will still steal resources from the main API server.

```python
@app.post("/chat")
async def chat(query: str, bg: BackgroundTasks):
    response = await run_rag(query)

    # Runs AFTER the HTTP response is returned to the user
    bg.add_task(log_chat_to_analytics_db, query, response)

    return {"answer": response}  # User gets this immediately
```

### 2. Option B: Distributed Task Queues (Celery + Redis / RabbitMQ)

* **How It Works:** FastAPI does **not** run the job itself—it drops a task ticket into an external message broker (**Redis** or **RabbitMQ**). Separate worker machines running **Celery** pick up the ticket and execute the job independently.
* **Why It Wins at Scale:**
  1. **Zero API Slowdown:** Heavy PDF parsing or batch embedding runs on separate machines, leaving FastAPI 100% free for user traffic.
  2. **Crash-Proof (Persistence & Retries):** Because the ticket is saved in Redis, if a worker crashes, the job is not lost and retries automatically.
  3. **Independent Scaling:** Can spin up 10 extra Celery workers during peak application deadlines without touching the FastAPI web containers.

### 3. Trade-Off Comparison: When to Use Which

| Feature | FastAPI Native `BackgroundTasks` | Distributed Queue (Celery + Redis) |
| :--- | :--- | :--- |
| **Where It Runs** | In-memory on the **same FastAPI server**. | On **separate worker containers/machines**. |
| **Setup Complexity** | **Zero** (built into FastAPI). | **High** (requires Redis/RabbitMQ + Celery workers). |
| **Crash Safety** | **Low:** Tasks vanish if the server restarts. | **High:** Persisted in Redis with automatic retries. |
| **Best Use Cases** | Quick 1–2s fire-and-forget chores (token logging, audit trails, simple webhook pings). | Heavy or critical jobs (ingesting a 300-page university handbook PDF, batch embeddings, billing reports). |

> **Rule of Thumb:** Use **`BackgroundTasks`** for fast, non-critical logging after a chat response; use **Celery + Redis** for anything CPU-heavy, slow (> 3 seconds), or too important to lose if a container restarts.

---

## Production Infrastructure Optimization: Caching & Cost Reduction

Calling a commercial LLM API for every single user message gets expensive and slow at scale. Production AI backends use **two layers of caching** to cut token costs and latency:

### 1. App-Level Caching: Exact-Match vs. Semantic Vector Caching

Before calling the LLM at all, FastAPI checks if another student has already asked the same question:

| Cache Type | How It Works | Simple Example | Result |
| :--- | :--- | :--- | :--- |
| **Exact-Match Cache** (Key-Value / SHA-256) | Hashes the exact text string and looks for an identical character-for-character match. | **User 1:** `"What is Form I-20?"`<br>**User 2:** `"What is an I-20 form?"` | **Cache Miss.** One extra word (`"an"`) changes the SHA-256 hash completely, forcing a full paid LLM call. |
| **Semantic Vector Cache** (Redis Vector / GPTCache) | Embeds the query (`all-MiniLM-L6-v2`) and checks if a past query has `Cosine Similarity >= 0.95`. | **User 1:** `"What is Form I-20?"`<br>**User 2:** `"What is an I-20 form?"` | **Cache Hit (`0.98 >= 0.95`).** Same meaning -> returns the saved answer in **< 15 ms** at **$0 LLM cost**. |

$$\text{Semantic Cache Hit Rule:}\quad \text{Cosine Similarity}(V_{\text{new}}, V_{\text{cached}}) \ge 0.95$$

---

### 2. Provider Prompt Caching (KV Cache Reuse)

When a student asks a **brand-new question** (so our Semantic Cache misses) and we *must* call the LLM, we can still save money using **Provider Prompt Caching** (supported by OpenAI, Anthropic, and Gemini).

* **What It Is:** Even when the user's question is new, the heavy part of the prompt—like a **4,000-token System Prompt + Visa Handbook Rules + Tool Schemas**—stays identical across requests. The LLM provider saves the pre-computed Transformer **Key-Value (KV) attention states** on their GPUs so they do not have to re-process those 4,000 tokens every time.
* **The Catch (Prefix-Only Rule):** Provider caching checks for an **exact match starting from character 0**. The moment a single word changes, caching stops for everything after that point.
* **Engineering Rule:** Always put **static content at the very top** and **dynamic content (chat history, user question) at the very bottom**.

#### Simple Example: Bad vs. Good Prompt Ordering

```text
# BAD PROMPT ORDER (0% Cache Hit - Breaks at Line 1!)
# Because the student's question is at the top, character 0 changes on every request.
# The LLM is forced to re-calculate all 4,000 handbook tokens from scratch every time.
[Dynamic] User Question: "Can I work 20 hours in Germany?"
[Static]  System Rules & 4,000-Token Visa Handbook: "You are a StudentGator advisor..."


# GOOD PROMPT ORDER (Cached KV Hit - Discounts & Speeds Up 4,000 Tokens!)
# The first 4,000 tokens are 100% identical across all users.
# The provider reuses its GPU KV cache for the top block and only computes the last line.
[Static]  System Rules & 4,000-Token Visa Handbook: "You are a StudentGator advisor..."
[Dynamic] User Question: "Can I work 20 hours in Germany?"
```

---

## End-to-End Capstone Implementation: Exposing LangGraph via FastAPI Streaming

This production reference ties every concept together into a single FastAPI gateway that runs a compiled **LangGraph** RAG workflow over **Server-Sent Events (`/chat/stream`)**, protected by **Pydantic v2 contracts**, **Dependency Injection (`Depends`)**, and **Background Tasks**.

```python
"""
StudentGator Production AI Gateway
Framework: FastAPI, LangGraph, Pydantic v2, Server-Sent Events (SSE)
"""

import asyncio
import json
from typing import AsyncGenerator, Annotated, Dict, Any, List
from uuid import UUID, uuid4

from fastapi import FastAPI, Depends, Header, HTTPException, Request, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ConfigDict

# LangGraph & Core Types
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document


# =========================================================================
# 1. Pydantic Schemas (Data Contracts)
# =========================================================================

class ChatStreamRequest(BaseModel):
    # Reject unknown fields and strip whitespace before checking length
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    thread_id: UUID = Field(default_factory=uuid4)
    message: str = Field(..., min_length=1, max_length=2000)
    category_filter: str | None = Field(default=None)


# Structured SSE Event Schemas (serialized into `data: {...}\n\n` blocks)
class TokenEvent(BaseModel):
    type: str = "token"
    content: str


class CitationEvent(BaseModel):
    type: str = "citation"
    source: str
    relevance_score: float


class DoneEvent(BaseModel):
    type: str = "done"
    total_tokens: int


# =========================================================================
# 2. LangGraph State Machine Definition
# =========================================================================

class AgentState(TypedDict):
    question: str
    category: str | None
    documents: List[Document]
    generation: str


def mock_retrieval_node(state: AgentState) -> Dict[str, Any]:
    """Simulates Day 2 Hybrid Search + Cross-Encoder Re-ranking."""
    doc = Document(
        page_content="Form I-20 requires a valid travel endorsement signature on page 2.",
        metadata={"source": "visa_policy.pdf", "score": 0.94}
    )
    return {"documents": [doc]}


def mock_generator_node(state: AgentState) -> Dict[str, Any]:
    """Generates the final grounded response from retrieved context."""
    return {
        "generation": "To leave the country, your Form I-20 must be signed on page 2 by a DSO."
    }


graph_builder = StateGraph(AgentState)
graph_builder.add_node("retrieve", mock_retrieval_node)
graph_builder.add_node("generate", mock_generator_node)
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)

compiled_agent = graph_builder.compile()


# =========================================================================
# 3. Dependency Injection Layer
# =========================================================================

async def verify_auth_header(
    authorization: Annotated[str | None, Header()] = None
) -> str:
    """Security Dependency: Blocks unauthorized requests before hitting the LLM."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header"
        )
    return "authorized_user"


# Reusable type alias for injecting auth checks into any endpoint
AuthDep = Annotated[str, Depends(verify_auth_header)]


# =========================================================================
# 4. Background Task Operations
# =========================================================================

def record_audit_event(thread_id: str, query: str, token_count: int):
    """Fire-and-forget audit logging executed AFTER the stream finishes."""
    print(f"[AUDIT LOG] Thread: {thread_id} | Tokens: {token_count} | Query: {query}")


# =========================================================================
# 5. FastAPI Application & Streaming Route
# =========================================================================

app = FastAPI(
    title="StudentGator AI Gateway",
    version="1.0.0",
    docs_url="/docs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def sse_token_generator(
    payload: ChatStreamRequest,
    raw_request: Request
) -> AsyncGenerator[str, None]:
    """
    Asynchronous generator executing LangGraph steps and yielding
    Server-Sent Events formatted as `data: <json>\n\n`.
    """
    try:
        # 1. Execute the compiled LangGraph workflow asynchronously
        initial_state: AgentState = {
            "question": payload.message,
            "category": payload.category_filter,
            "documents": [],
            "generation": "",
        }
        result = await compiled_agent.ainvoke(initial_state)

        # 2. Emit Citation Events first so the UI can show sources immediately
        for doc in result.get("documents", []):
            citation = CitationEvent(
                source=doc.metadata.get("source", "unknown"),
                relevance_score=doc.metadata.get("score", 0.0),
            )
            yield f"data: {citation.model_dump_json()}\n\n"

        # 3. Stream generated answer token-by-token
        words = result["generation"].split(" ")
        token_count = 0

        for word in words:
            # Stop generating if the user closed their browser tab
            if await raw_request.is_disconnected():
                break

            token_event = TokenEvent(content=f"{word} ")
            yield f"data: {token_event.model_dump_json()}\n\n"
            token_count += 1
            await asyncio.sleep(0.05)  # Simulates token generation cadence

        # 4. Emit final completion event
        done_event = DoneEvent(total_tokens=token_count)
        yield f"data: {done_event.model_dump_json()}\n\n"

    except Exception as exc:
        # In-stream error signaling (since HTTP 200 OK was already sent)
        error_payload = json.dumps({"type": "error", "message": str(exc)})
        yield f"data: {error_payload}\n\n"


@app.post("/chat/stream")
async def chat_stream_endpoint(
    payload: ChatStreamRequest,
    raw_request: Request,
    background_tasks: BackgroundTasks,
    user: AuthDep,  # Runs verify_auth_header before entering route
) -> StreamingResponse:
    # Schedule non-blocking audit log to run after the response completes
    background_tasks.add_task(
        record_audit_event,
        thread_id=str(payload.thread_id),
        query=payload.message,
        token_count=len(payload.message.split()),
    )

    return StreamingResponse(
        sse_token_generator(payload, raw_request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disables NGINX chunk buffering
        },
    )
```





