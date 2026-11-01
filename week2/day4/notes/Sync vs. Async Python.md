# Asynchronous Python & Concurrency in AI Pipelines

## 1. The Core Bottleneck: I/O-Bound vs. CPU-Bound

In modern AI and web applications, the CPU spends over **95% of its time completely idle**, waiting on network latency:

* **I/O-Bound Tasks:** The process is **waiting for an external system** to respond (e.g., calling OpenAI/Anthropic cloud APIs, querying ChromaDB/PostgreSQL, or reading files from disk).
  * *Solution:* **`asyncio`** (non-blocking event loop). A single thread can handle hundreds of concurrent requests by eliminating idle wait times.
* **CPU/GPU-Bound Tasks:** The hardware processor is **actively crunching numbers** without pausing (e.g., matrix multiplication, tokenization, image processing, local model training).
  * *Solution:* **`multiprocessing`** or distributed worker queues (Celery, Ray), because Python's Global Interpreter Lock (GIL) prevents standard threads from running CPU math in parallel.

---

## 2. Concurrency vs. Parallelism (Systems Engineering View)

* **Concurrency (Task Interleaving / Cooperative Multitasking):**
  * **Hardware level:** Runs on a **single CPU core / single thread**.
  * **Mechanism:** When a task awaits a network socket read, the event loop yields CPU control to process another task.
  * **Core concept:** Managing multiple in-flight tasks by overlapping wait times. Best for **I/O-bound** workloads.
* **Parallelism (Simultaneous Physical Execution):**
  * **Hardware level:** Runs across **multiple physical CPU / GPU cores**.
  * **Mechanism:** Instructions execute at the exact same physical clock cycle across separate OS processes, bypassing the Python GIL.
  * **Core concept:** Multiplying raw compute throughput. Best for **CPU-bound** workloads.

> **Key Distinction:**
> * **Concurrency** is about *dealing with* lots of things at once (smart scheduling during I/O waits).
> * **Parallelism** is about *doing* lots of things at once (simultaneous hardware execution).

---

## 3. The Engine: Python Event Loop vs. Node.js

Under the hood, Python's `asyncio` works like the Node.js event loop: a single-threaded cooperative loop polling OS I/O events (`epoll` on Linux, `kqueue` on macOS) and switching tasks at I/O boundaries.

* **What is a Coroutine?**
  * Declared using `async def`.
  * **JavaScript (Eager) vs. Python (Lazy) Execution:**
    * **Node.js (Eager):** Calling `fetchData()` immediately executes synchronous code on the call stack until reaching the first `await`.
    * **Python (Lazy):** Calling `fetch_data()` executes **zero lines of code**. It only returns a dormant `coroutine` object.
    * **The Node.js Trap:** In JS, calling `logEvent()` without `await` runs it in the background. In Python, calling `log_event()` without `await` or `asyncio.create_task()` never executes and triggers `RuntimeWarning: coroutine was never awaited`.
    * **How to run in Python:** Explicitly schedule via `await coro`, `asyncio.create_task(coro)` (for background tasks), or `asyncio.run(coro)` (entry point).
* **The Event Loop Cycle:**
  1. Executes Task A until it reaches an `await` statement.
  2. Task A pauses and yields control back to the event loop.
  3. The event loop immediately executes Task B.
  4. When the OS signals that network packets arrived for Task A, the loop wakes Task A up and resumes execution right where it paused.
* **The #1 Trap (Blocking the Loop):**
  * Calling synchronous blocking code (like `time.sleep()`, standard `requests.get()`, or a heavy local loop) **freezes the entire event loop** for all users.
  * *Fix:* Use async equivalents (`httpx.AsyncClient`, `aiohttp`, `asyncio.sleep()`) or offload blocking sync code to a threadpool via `asyncio.to_thread(blocking_func)`.

---

## 4. Syntax & Rules: `async def`, `await`, and `asyncio.gather`

* **`async def`:** Declares a coroutine function and unlocks the use of `await` inside it.
* **`await`:** Can **only** be used inside an `async def` function. It marks the exact pause point where execution yields back to the event loop.
* **`asyncio.gather()`:** Takes multiple coroutines, registers all of them onto the event loop, and runs them concurrently in flight.

### Code Benchmark: Sequential (6s) vs. Concurrent (2s)

#### Synchronous (Slow: 3 calls take 6.0 seconds)
```python
import time

def call_api(name: str) -> str:
    print(f"Calling {name}...")
    time.sleep(2)  # Freezes the entire thread for 2 seconds!
    return f"{name} done"

def main():
    start = time.perf_counter()
    res1 = call_api("API 1")
    res2 = call_api("API 2")
    res3 = call_api("API 3")
    print(f"Total time: {time.perf_counter() - start:.2f}s")  # ~6.0s

main()
```

#### Asynchronous with `asyncio.gather` (Fast: 3 calls take 2.0 seconds)
```python
import asyncio
import time

async def call_api(name: str) -> str:
    print(f"Calling {name}...")
    await asyncio.sleep(2)  # Non-blocking: yields control back to the event loop!
    return f"{name} done"

async def main():
    start = time.perf_counter()
    # Runs all 3 calls concurrently on the single-threaded event loop:
    results = await asyncio.gather(
        call_api("API 1"),
        call_api("API 2"),
        call_api("API 3"),
    )
    print(f"Total time: {time.perf_counter() - start:.2f}s")  # ~2.0s!

asyncio.run(main())
```

---

## 5. Production AI Applications & FastAPI Integration

### 1. High-Throughput Streaming Backends (FastAPI + ASGI)
* **The Reality:** LLM text generation takes 2 to 5 seconds per request.
* In a synchronous server (like Flask or Django WSGI), 100 concurrent requests would block 100 separate worker processes or queue up sequentially.
* In an **async FastAPI backend**, a single worker process effortlessly handles hundreds of concurrent SSE (Server-Sent Events) token-streaming connections because it yields execution while waiting on downstream API/GPU token generation.

### 2. Parallel Hybrid RAG Retrieval
In production RAG systems, a user query often queries multiple search engines at once:
* A Vector DB similarity search (`ChromaDB`).
* A Keyword lexical search (`BM25`).
* A Web Search API call (e.g., Tavily or Google Search).

```python
# All 3 network calls execute concurrently:
vector_res, bm25_res, web_res = await asyncio.gather(
    vector_search(query),
    bm25_search(query),
    web_search(query)
)
# Total retrieval latency = max(t1, t2, t3) instead of their sum (t1 + t2 + t3)!
```

---

## 6. Architecture Decision Matrix

| Workload Type | Concrete Examples | Optimal Python Tool |
| :--- | :--- | :--- |
| **I/O-Bound (Concurrency)** | Calling cloud LLM APIs, Vector DB queries, web scraping, streaming UI tokens | `asyncio`, `httpx.AsyncClient`, FastAPI |
| **CPU/GPU-Bound (Parallelism)** | Local PyTorch model inference, tokenization, heavy NumPy array calculations | `multiprocessing`, NumPy/Torch vectorization, Celery |
| **Legacy Sync in Async App** | Running a synchronous SDK or blocking file I/O inside an async route | `asyncio.to_thread()` |
