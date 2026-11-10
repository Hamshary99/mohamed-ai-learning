# Week 3 Quick Revision Guide: RAG, LangChain, LangGraph & FastAPI

This revision guide consolidates every core architectural concept, trade-off matrix, mental model, and production implementation pattern across Week 3 notes. The master index below provides one-click internal navigation to any topic within this revision document. Inside each topic section, a direct link in front of the heading opens the corresponding source note file at the exact line in VS Code.

---

## Master Table of Contents

1. [Day 1: Native RAG & LangChain Core](#day-1-native-rag--langchain-core)
   * [1.1 Why RAG Exists (The 3 Flaws of Standalone LLMs)](#11-why-rag-exists)
   * [1.2 Query Time vs. Training Time (RAG vs. Fine-Tuning)](#12-query-vs-training-time)
   * [1.3 Chunking Mechanics & Isolation Rules](#13-chunking-mechanics)
   * [1.4 Overlap Mechanics (Context Bridges Across Splits)](#14-overlap-mechanics)
   * [1.5 Chunk Size Sweet Spot](#15-chunk-size-sweet-spot)
   * [1.6 Keyword Match vs. Semantic Search](#16-keyword-vs-semantic)
   * [1.7 Cosine Similarity vs. Dot Product](#17-cosine-vs-dot-product)
   * [1.8 The Production Normalization Shortcut](#18-normalization-shortcut)
   * [1.9 Vector Databases & ANN Indexing](#19-vector-indexing)
   * [1.10 Vector Engine Comparison (Chroma vs. Qdrant vs. Pinecone)](#110-vector-engines)
   * [1.11 Caching in RAG (Exact Redis vs. Semantic GPTCache)](#111-caching-in-rag)
   * [1.12 LangChain Core Building Blocks & LCEL](#112-langchain-core-lcel)
   * [1.13 End-to-End LCEL RAG Implementation](#113-day1-code-reference)

2. [Day 2: Advanced Retrieval & Re-Ranking](#day-2-advanced-retrieval--re-ranking)
   * [2.1 Query Expansion (Multi-Query Variation)](#21-query-expansion)
   * [2.2 Multi-Query Decomposition (Sub-Questions)](#22-query-decomposition)
   * [2.3 Parent-Child Retriever (Hierarchical Indexing)](#23-parent-child-retriever)
   * [2.4 The Chunk Size Dilemma & Decoupled Architecture](#24-chunk-size-dilemma)
   * [2.5 Hybrid Search: Semantic Dense + Sparse BM25](#25-hybrid-search)
   * [2.6 BM25 Scoring Mechanics (TF, IDF, Length Norm)](#26-bm25-scoring)
   * [2.7 Reciprocal Rank Fusion (RRF Formula & Analogy)](#27-rrf-fusion)
   * [2.8 Metadata Filtering & Confidence Threshold Gates](#28-metadata-filtering)
   * [2.9 Cross-Jurisdiction Leak Bug & Fallback Gates](#29-cross-jurisdiction-leak)
   * [2.10 Bi-Encoder vs. Cross-Encoder Mechanics](#210-bi-vs-cross-encoder)
   * [2.11 The Two-Stage Re-Ranking Retrieval Funnel](#211-two-stage-funnel)
   * [2.12 Ensemble Retrieval & Weighted Domain Strategies](#212-ensemble-retrieval)
   * [2.13 Modular Python Code References (7 Standalone Scripts)](#213-day2-code-references)

3. [Day 3: LangGraph, Agentic Workflows & LangSmith](#day-3-langgraph-agentic-workflows--langsmith)
   * [3.1 LCEL Conveyor Belt vs. LangGraph State Machine](#31-lcel-vs-langgraph)
   * [3.2 State Schema, Nodes & Reducers](#32-state-nodes-reducers)
   * [3.3 Linear Execution Graphs vs. Conditional Router Edges](#33-linear-vs-conditional)
   * [3.4 Pregel Super-Step Execution (Turn-Based Board Game)](#34-pregel-execution)
   * [3.5 Checkpointers, Persistence & Human-in-the-Loop](#35-checkpointers-persistence)
   * [3.6 Agentic RAG (Self-Correction Reflection Loops)](#36-agentic-rag-loops)
   * [3.7 Multi-Agent Architectures (Supervisor vs. Networks)](#37-multi-agent-architectures)
   * [3.8 LangSmith Observability & Tracing](#38-langsmith-observability)
   * [3.9 Runnable LangGraph Production Agent Reference](#39-day3-code-reference)

4. [Day 4: FastAPI Production Gateway, Streaming & Caching](#day-4-fastapi-production-gateway-streaming--caching)
   * [4.1 Why LLMs Need an API Gateway](#41-why-api-gateway)
   * [4.2 Pydantic v2 Gatekeeper Contracts](#42-pydantic-contracts)
   * [4.3 Malformed Payload Rejection Matrix (HTTP 422 Protection)](#43-malformed-payload-matrix)
   * [4.4 ASGI Single-Threaded Event Loop & Uvicorn Workers](#44-asgi-event-loop)
   * [4.5 The Event Loop Blocking Anti-Pattern](#45-event-loop-blocking)
   * [4.6 Starlette 40-Worker Thread Ceiling](#46-starlette-thread-pool)
   * [4.7 Dependency Injection Engine (Depends)](#47-dependency-injection)
   * [4.8 Real-Time Streaming: Server-Sent Events (SSE)](#48-sse-streaming)
   * [4.9 TTFT vs. Total Response Latency](#49-ttft-latency)
   * [4.10 Production Stream Hardening (NGINX, Disconnects, In-Stream Errors)](#410-stream-hardening)
   * [4.11 Native BackgroundTasks vs. Distributed Queues (Celery)](#411-background-tasks-celery)
   * [4.12 Production Caching & Provider KV Attention Reuse](#412-prompt-caching-kv)
   * [4.13 The Prefix-Only Prompt Ordering Rule](#413-prefix-only-rule)
   * [4.14 Capstone Streaming Gateway Implementation](#414-day4-code-reference)

5. [Whiteboard & Interview Talking Points](#whiteboard--interview-talking-points)

---

## Day 1: Native RAG & LangChain Core

<a id="11-why-rag-exists"></a>
### [[RAG.md#L3](./day1/notes/RAG.md#L3)] 1.1 Why RAG Exists (The 3 Flaws of Standalone LLMs)
* **Mental Model:** Standalone LLMs operate like students taking a closed-book exam strictly from memory. RAG converts this into an **open-book exam** by fetching exact source pages at query time.
* **The 3 Flaws Solved:**
  1. *No Source Citations:* LLMs cannot reference their sources; RAG injects document metadata (`source`, `page`) directly into prompt context.
  2. *Outdated Knowledge:* Training stops at cutoff dates; RAG updates dynamically whenever a new handbook or policy is uploaded.
  3. *Hallucinations:* Restricts the generation model to facts grounded strictly in retrieved context.

<a id="12-query-vs-training-time"></a>
### [[RAG.md#L23](./day1/notes/RAG.md#L23)] 1.2 Query Time vs. Training Time (RAG vs. Fine-Tuning)
* Indexing fresh documents into a vector DB at query time costs cents and is instant. Fine-tuning takes days, costs thousands, and still does not guarantee factual recall or eliminate hallucinations.

<a id="13-chunking-mechanics"></a>
### [[RAG.md#L29](./day1/notes/RAG.md#L29)] 1.3 Chunking Mechanics & Isolation Rules
* **Isolation Rule:** Chunking occurs document-by-document. Documents are never merged across boundaries prior to splitting; each chunk retains its isolated file origin metadata.

<a id="14-overlap-mechanics"></a>
### [[RAG.md#L37](./day1/notes/RAG.md#L37)] 1.4 Overlap Mechanics (Context Bridges Across Splits)
* Prevents splitting critical sentences down the middle. Repeating 10%–15% across adjacent chunk boundaries creates a continuity bridge so thoughts are preserved across cuts.

<a id="15-chunk-size-sweet-spot"></a>
### [[RAG.md#L44](./day1/notes/RAG.md#L44)] 1.5 Chunk Size Sweet Spot
* Recommended baseline: 1 to 2 paragraphs (~100–250 words / 400–800 characters). Overly small chunks lose references; overly large chunks dilute vector embeddings.

<a id="16-keyword-vs-semantic"></a>
### [[RAG.md#L56](./day1/notes/RAG.md#L56)] 1.6 Keyword Match vs. Semantic Search
* Keyword search (exact match) fails on synonyms and paraphrasing (*"tuition fees"* vs. *"college cost"*). Semantic search projects text into coordinate space where similar meanings cluster together regardless of exact wording.

<a id="17-cosine-vs-dot-product"></a>
### [[RAG.md#L74](./day1/notes/RAG.md#L74)] 1.7 Cosine Similarity vs. Dot Product
* *Dot Product:* Multiplies coordinates directly. Sensitive to vector magnitude (long paragraphs unfairly outscore short, direct answers).
* *Cosine Similarity:* Measures only the angle between vectors (`-1.0` to `+1.0`), ignoring vector length.

<a id="18-normalization-shortcut"></a>
### [[RAG.md#L89](./day1/notes/RAG.md#L89)] 1.8 The Production Normalization Shortcut
* When embeddings are L2-normalized to unit vectors ($|V| = 1.0$), **Cosine Similarity is mathematically identical to Dot Product**, allowing vector DBs to run high-speed hardware dot products without square-root divisions.

<a id="19-vector-indexing"></a>
### [[RAG.md#L96](./day1/notes/RAG.md#L96)] 1.9 Vector Databases & ANN Indexing
* Approximate Nearest Neighbor (ANN) indexes vectors into topical neighborhoods/clusters. Queries jump directly to the relevant cluster, skipping 90%+ of irrelevant vectors.

<a id="110-vector-engines"></a>
### [[RAG.md#L124](./day1/notes/RAG.md#L124)] 1.10 Vector Engine Comparison
* ChromaDB (in-memory/local prototyping), Qdrant (Rust-based production hybrid search), Pinecone (fully managed cloud).

<a id="111-caching-in-rag"></a>
### [[RAG.md#L131](./day1/notes/RAG.md#L131)] 1.11 Caching in RAG (Exact Redis vs. Semantic GPTCache)
* *Exact Query Cache:* Redis Key-Value hashes queries for 2ms responses at $0 cost on identical strings.
* *Semantic Vector Cache:* Embeds queries and evaluates Cosine Similarity $\ge 0.95$ against answered prompts, catching synonyms.

<a id="112-langchain-core-lcel"></a>
### [[LangChain.md#L36](./day1/notes/LangChain.md#L36)] 1.12 LangChain Core Building Blocks & LCEL
* Standardizes RAG into Ingestion, Chunking, Storage, and Generation. LCEL pipes components using `|`:
  `{"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | parser`

<a id="113-day1-code-reference"></a>
### [[LangChain.md#L81](./day1/notes/LangChain.md#L81)] 1.13 End-to-End LCEL RAG Implementation
* Complete runnable 6-step LCEL pipeline code connecting `TextLoader`, `RecursiveCharacterTextSplitter`, `Chroma`, `ChatPromptTemplate`, and `ChatOpenAI`.

---

## Day 2: Advanced Retrieval & Re-Ranking

<a id="21-query-expansion"></a>
### [[Advanced Rag.md#L15](./day2/notes/Advanced%20Rag.md#L15)] 2.1 Query Expansion (Multi-Query Variation)
* Uses an LLM to generate 3–5 prompt variations with identical intent, querying vector storage in parallel to overcome synonym blind spots.

<a id="22-query-decomposition"></a>
### [[Advanced Rag.md#L31](./day2/notes/Advanced%20Rag.md#L31)] 2.2 Multi-Query Decomposition (Sub-Questions)
* Breaks compound questions (*"Difference between CPT and OPT?"*) into independent atomic sub-questions, searching in parallel and merging contexts.

<a id="23-parent-child-retriever"></a>
### [[Advanced Rag.md#L78](./day2/notes/Advanced%20Rag.md#L78)] 2.3 Parent-Child Retriever (Hierarchical Indexing)
* Solves the chunk size dilemma by storing small child chunks (150 tokens) in ChromaDB strictly for search, while full parent chunks (800–1000 tokens) are stored in an underlying DocStore and injected into the prompt.

<a id="24-chunk-size-dilemma"></a>
### [[Advanced Rag.md#L80](./day2/notes/Advanced%20Rag.md#L80)] 2.4 The Chunk Size Dilemma & Decoupled Architecture
* Small chunks maximize search precision but starve the generator of narrative context; large chunks preserve context but dilute vector coordinates. Decoupling search from generation resolves this compromise.

<a id="25-hybrid-search"></a>
### [[Advanced Rag.md#L140](./day2/notes/Advanced%20Rag.md#L140)] 2.5 Hybrid Search: Semantic Dense + Sparse BM25
* Dense vectors smooth alphanumeric terms into conceptual coordinates, confusing exact course codes (`CS-101` vs `CS-102`) or form codes (`Form I-20`). Hybrid search runs both dense and lexical BM25 retrievers in parallel.

<a id="26-bm25-scoring"></a>
### [[Advanced Rag.md#L154](./day2/notes/Advanced%20Rag.md#L154)] 2.6 BM25 Scoring Mechanics (TF, IDF, Length Norm)
* Uses Term Frequency saturation (caps keyword spam), Inverse Document Frequency (massive rarity bonus for terms like `I-20`), and document length normalization.

<a id="27-rrf-fusion"></a>
### [[Advanced Rag.md#L222](./day2/notes/Advanced%20Rag.md#L222)] 2.7 Reciprocal Rank Fusion (RRF Formula & Analogy)
* Eliminates incompatible score scales (bounded Cosine vs. unbounded BM25) by scoring based purely on position rank:
  $$\text{RRF Score}(d) = \sum_{m \in M} \frac{1}{k + \text{rank}_m(d)} \quad (k = 60)$$

<a id="28-metadata-filtering"></a>
### [[Advanced Rag.md#L246](./day2/notes/Advanced%20Rag.md#L246)] 2.8 Metadata Filtering & Confidence Threshold Gates
* Relational pre-filtering (`WHERE department == 'visas' AND year == 2026`) executed before vector distance calculation begins, slashing compute and preventing cross-department noise.

<a id="29-cross-jurisdiction-leak"></a>
### [[Advanced Rag.md#L249](./day2/notes/Advanced%20Rag.md#L249)] 2.9 Cross-Jurisdiction Leak Bug & Fallback Gates
* Failure mode where generic words (*"work"*, *"student"*) caused US F-1 regulations to leak into German student inquiries. Production gates evaluate candidate scores (`score >= 0.0`); below-threshold results trigger deterministic fallbacks.

<a id="210-bi-vs-cross-encoder"></a>
### [[Advanced Rag.md#L322](./day2/notes/Advanced%20Rag.md#L322)] 2.10 Bi-Encoder vs. Cross-Encoder Mechanics
* *Bi-Encoder:* Independent query/document passes. Fast $O(1)$ lookups, pre-computed offline, but blind to token-level interaction and negation nuances (*"with"* vs *"without"* authorization).
* *Cross-Encoder:* Concatenates `[CLS] Q [SEP] D [SEP]` into a single Transformer pass. Full cross-attention between all tokens outputs an exact relevance scalar (`0.0` to `1.0`).

<a id="211-two-stage-funnel"></a>
### [[Advanced Rag.md#L373](./day2/notes/Advanced%20Rag.md#L373)] 2.11 The Two-Stage Re-Ranking Retrieval Funnel
* Stage 1 (Bi-Encoder / Hybrid) pulls top 50 candidates (High Recall); Stage 2 (Cross-Encoder) re-ranks and filters down to the top 3 chunks (High Precision).

<a id="212-ensemble-retrieval"></a>
### [[Advanced Rag.md#L431](./day2/notes/Advanced%20Rag.md#L431)] 2.12 Ensemble Retrieval & Weighted Domain Strategies
* Scales retriever influence based on domain (e.g. 70% Dense / 30% BM25 for conversational student forums; 60% BM25 / 40% Dense for legal and regulatory codes).

<a id="213-day2-code-references"></a>
### [[Advanced Rag.md#L519](./day2/notes/Advanced%20Rag.md#L519)] 2.13 Modular Python Code References (7 Standalone Scripts)
* Python code references for Query Expansion, Multi-Query Decomposition, Parent-Child Retriever, Chroma Metadata Pre-Filtering, Pure Python RRF, Cross-Encoder Re-Ranking, and LangChain `EnsembleRetriever`.

---

## Day 3: LangGraph, Agentic Workflows & LangSmith

<a id="31-lcel-vs-langgraph"></a>
### [[LangGraph.md#L3](./day3/notes/LangGraph.md#L3)] 3.1 LCEL Conveyor Belt vs. LangGraph State Machine
* Standard LCEL is strictly a forward Directed Acyclic Graph (DAG). Real-world agents require reflection loops to critique retrieved chunks, call tools, and retry.
* Mental Model: LCEL is a factory conveyor belt; LangGraph is an air traffic control state machine that routes planes, holds in circles (cycles), and halts for pilot clearances (interrupts).

<a id="32-state-nodes-reducers"></a>
### [[LangGraph.md#L32](./day3/notes/LangGraph.md#L32)] 3.2 State Schema, Nodes & Reducers
* State is defined via `TypedDict`. Reducers like `Annotated[list[BaseMessage], operator.add]` dictate how node outputs update state, appending messages rather than overwriting history.

<a id="33-linear-vs-conditional"></a>
### [[LangGraph.md#L63](./day3/notes/LangGraph.md#L63)] 3.3 Linear Execution Graphs vs. Conditional Router Edges
* Linear graphs handle fixed workflows ($A \to B \to C$). Conditional router edges inspect runtime state and dynamically route to different specialized nodes (e.g. visa advisor vs. housing advisor).

<a id="34-pregel-execution"></a>
### [[LangGraph.md#L115](./day3/notes/LangGraph.md#L115)] 3.4 Pregel Super-Step Execution (Turn-Based Board Game)
* Active nodes receive an immutable frozen snapshot of state, execute concurrently, and buffer their outputs. LangGraph atomically merges all buffered updates via reducers at the super-step boundary, eliminating race conditions.

<a id="35-checkpointers-persistence"></a>
### [[LangGraph.md#L139](./day3/notes/LangGraph.md#L139)] 3.5 Checkpointers, Persistence & Human-in-the-Loop
* Attaching `MemorySaver` or `PostgresSaver` records state snapshots under a `thread_id` for fault-tolerant resumes, human-in-the-loop approval gates (`interrupt`), and time-travel debugging.

<a id="36-agentic-rag-loops"></a>
### [[LangGraph.md#L74](./day3/notes/LangGraph.md#L74)] 3.6 Agentic RAG (Self-Correction Reflection Loops)
* Moves beyond single-pass naive RAG: Retrieve $\to$ Grade candidates $\to$ Route (generate if relevant, rewrite query if irrelevant, or call search tools if missing).

<a id="37-multi-agent-architectures"></a>
### [[LangGraph.md#L87](./day3/notes/LangGraph.md#L87)] 3.7 Multi-Agent Architectures (Supervisor vs. Networks)
* Supervisor pattern uses a central router LLM to delegate to sub-agent nodes; Peer-to-Peer networks collaborate sequentially or via shared message buses.

<a id="38-langsmith-observability"></a>
### [[LangGraph.md#L93](./day3/notes/LangGraph.md#L93)] 3.8 LangSmith Observability & Tracing
* Zero-code distributed tracing configured via environment variables (`LANGSMITH_TRACING=true`). Visualizes graph DAG run trees, token usage, latency per node, and failure points.

<a id="39-day3-code-reference"></a>
### [[LangGraph.md#L174](./day3/notes/LangGraph.md#L174)] 3.9 Runnable LangGraph Production Agent Reference
* Complete Python implementation defining state schema, tools, agent node, tool node, conditional edges, cyclic loops, and `MemorySaver` persistence.

---

## Day 4: FastAPI Production Gateway, Streaming & Caching

<a id="41-why-api-gateway"></a>
### [[FastApi.md#L3](./day4/notes/FastApi.md#L3)] 4.1 Why LLMs Need an API Gateway
* Manages client authentication, enforces request/response contracts, handles connection pooling, balances concurrent traffic, and streams real-time tokens.

<a id="42-pydantic-contracts"></a>
### [[FastApi.md#L21](./day4/notes/FastApi.md#L21)] 4.2 Pydantic v2 Gatekeeper Contracts
* Validates incoming payloads at the gateway, returning HTTP `422 Unprocessable Entity` before triggering expensive LLM tokens or vector queries.
* Rules: `ConfigDict(extra="forbid", str_strip_whitespace=True)`, `min_length=1`, `max_length=4000`, and `UUID` thread parsing.

<a id="43-malformed-payload-matrix"></a>
### [[FastApi.md#L114](./day4/notes/FastApi.md#L114)] 4.3 Malformed Payload Rejection Matrix
* Catches whitespace-only prompts, 50,000-character prompt bombs, unapproved schema fields, and corrupted thread IDs before they hit the model.

<a id="44-asgi-event-loop"></a>
### [[FastApi.md#L126](./day4/notes/FastApi.md#L126)] 4.4 ASGI Single-Threaded Event Loop & Uvicorn Workers
* LLM generation spends 95%+ of time waiting on network I/O. `async def` endpoints yield control via `await`, allowing a single thread to manage hundreds of concurrent connections.

<a id="45-event-loop-blocking"></a>
### [[FastApi.md#L146](./day4/notes/FastApi.md#L146)] 4.5 The Event Loop Blocking Anti-Pattern
* Calling synchronous blocking functions (`time.sleep()`, `requests.get()`) freezes the event loop for all users across the worker process.

<a id="46-starlette-thread-pool"></a>
### [[FastApi.md#L159](./day4/notes/FastApi.md#L159)] 4.6 Starlette 40-Worker Thread Ceiling
* Normal synchronous `def` routes run on a worker thread pool capped at 40 threads. High-concurrency AI workloads must use native `async def` with async clients (`httpx`).

<a id="47-dependency-injection"></a>
### [[FastApi.md#L180](./day4/notes/FastApi.md#L180)] 4.7 Dependency Injection Engine (Depends)
* Decouples infrastructure setup (auth checks, database connections, vector retrievers) from route business logic, allowing clean test overrides.

<a id="48-sse-streaming"></a>
### [[FastApi.md#L241](./day4/notes/FastApi.md#L241)] 4.8 Real-Time Streaming: Server-Sent Events (SSE)
* LLM text streaming is uni-directional. SSE runs over standard HTTP (`text/event-stream`), easily traversing corporate proxies, load balancers, and firewalls.

<a id="49-ttft-latency"></a>
### [[FastApi.md#L248](./day4/notes/FastApi.md#L248)] 4.9 TTFT vs. Total Response Latency
* Users judge AI applications by Time-To-First-Token (<400ms via SSE) rather than total response latency (15s for full generation).

<a id="410-stream-hardening"></a>
### [[FastApi.md#L303](./day4/notes/FastApi.md#L303)] 4.10 Production Stream Hardening
* Header `"X-Accel-Buffering": "no"` disables NGINX buffering, `await request.is_disconnected()` aborts generation when users close browser tabs, and structured in-stream error JSON handles mid-stream provider failures.

<a id="411-background-tasks-celery"></a>
### [[FastApi.md#L317](./day4/notes/FastApi.md#L317)] 4.11 Native BackgroundTasks vs. Distributed Queues (Celery)
* *FastAPI Native `BackgroundTasks`:* In-memory, single container. Best for fast 1–2s chores (logging audit records, token metrics). Unsafe for critical work.
* *Distributed Queues (Celery + Redis):* External message broker and workers. Persistent, crash-proof, horizontally scalable. Best for heavy PDF ingestion or batch embeddings.

<a id="412-prompt-caching-kv"></a>
### [[FastApi.md#L359](./day4/notes/FastApi.md#L359)] 4.12 Production Caching & Provider KV Attention Reuse
* Providers reuse pre-computed attention keys/values on GPUs for identical prompt prefixes across requests.

<a id="413-prefix-only-rule"></a>
### [[FastApi.md#L384](./day4/notes/FastApi.md#L384)] 4.13 The Prefix-Only Prompt Ordering Rule
* Prefix matching starts strictly at character 0. Always place static rules and handbooks at the very top, and dynamic conversation history / user questions at the bottom.

<a id="414-day4-code-reference"></a>
### [[FastApi.md#L403](./day4/notes/FastApi.md#L403)] 4.14 Capstone Streaming Gateway Implementation
* Complete FastAPI microservice exposing a compiled LangGraph workflow over Server-Sent Events (`/chat/stream`), secured with Pydantic contracts and dependency injection.

---

## Whiteboard & Interview Talking Points

| Review Topic | Core Talking Point | Direct Route |
| :--- | :--- | :--- |
| **Why RAG vs Fine-Tuning?** | Fine-Tuning bakes facts into weights during training (slow, expensive, hallucination-prone). RAG provides facts at query time from verified documents (instant updates, cheap, cited). | [[RAG.md#L23](./day1/notes/RAG.md#L23)] |
| **Why Dot Product equals Cosine?** | When vector embeddings are L2-normalized ($|V|=1$), Cosine Similarity equals Dot Product. Hardware dot products avoid expensive square root division. | [[RAG.md#L89](./day1/notes/RAG.md#L89)] |
| **Why Hybrid Search (BM25 + Vector)?** | Vector search models semantic concepts but fails on exact alphanumeric codes (`Form I-20`, `CS-101`). BM25 enforces exact keyword matches via IDF rarity scoring. | [[Advanced Rag.md#L140](./day2/notes/Advanced%20Rag.md#L140)] |
| **Why RRF over Score Addition?** | Vector cosine scores (`0.0–1.0`) and BM25 scores (`0–50+`) have incompatible scales. RRF fuses lists using position rank ($\frac{1}{60 + \text{rank}}$). | [[Advanced Rag.md#L222](./day2/notes/Advanced%20Rag.md#L222)] |
| **Bi-Encoder vs Cross-Encoder?** | Bi-encoders encode queries and docs independently (fast $O(1)$, Stage 1 recall). Cross-encoders run full joint cross-attention on $(Q, D)$ pairs (slow $O(N)$, Stage 2 precision). | [[Advanced Rag.md#L322](./day2/notes/Advanced%20Rag.md#L322)] |
| **LCEL vs LangGraph?** | LCEL is a forward DAG conveyor belt. LangGraph is a stateful air traffic control state machine supporting cycles, reducers, and checkpointed persistence. | [[LangGraph.md#L17](./day3/notes/LangGraph.md#L17)] |
| **What is Pregel Execution?** | Graphs execute in synchronized super-steps. Nodes run concurrently against an immutable state snapshot, merging buffered updates atomically at step boundaries. | [[LangGraph.md#L115](./day3/notes/LangGraph.md#L115)] |
| **Event Loop Blocking Rule?** | Never invoke blocking calls (`time.sleep`) inside `async def` endpoints on single-threaded event loops; always use `await` or offload blocking sync code. | [[FastApi.md#L146](./day4/notes/FastApi.md#L146)] |
| **Why SSE over WebSockets?** | LLM text generation is fundamentally uni-directional. SSE runs over standard HTTP (`text/event-stream`), natively passing reverse proxies and firewalls. | [[FastApi.md#L257](./day4/notes/FastApi.md#L257)] |
| **KV Cache Prefix Rule?** | Provider prompt caching matches prefixes starting from index 0. Static handbooks must sit at the top; dynamic user questions must sit at the bottom. | [[FastApi.md#L376](./day4/notes/FastApi.md#L376)] |
