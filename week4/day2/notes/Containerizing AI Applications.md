# Containerizing AI Applications & Multi-Stage Builds

Day 2 scope: Containerizing FastAPI apps, ML models, and RAG services, multi-stage Docker builds, and production container patterns implemented in `studentgator-rag-service`.

---

## 1. What Is It & Why Does It Exist?

### The Core Problem
Shipping standard Python web APIs in Docker is straightforward. Shipping **AI and RAG applications** is not:
1. **Dependency Bloat:** Installing PyTorch or HuggingFace often pulls 1GB to 3GB+ of CUDA libraries by default, even if our service only runs inference on CPU.
2. **Build Tool Clutter:** C++ compilers (`gcc`, `g++`) and pip caches get baked into the production image, wasting bandwidth, slowing cold starts, and introducing security vulnerabilities.
3. **State vs. Stateless Mismatch:** RAG applications require both stateless request handling (FastAPI + LangGraph) and persistent state (local ChromaDB vector stores or cached embeddings).
4. **Security Risks:** Containers that run as `root` give compromised web applications full control over container namespaces and mounted host volumes.

### The Mental Model
* **Multi-Stage Build:** A factory assembly line. We use a messy workshop (`builder`) with all the heavy tools and scrap metal to assemble the machine. Once finished, we crate only the final clean machine into the shipping container (`runner`) and leave all the tools and dust behind.
* **CPU-Only Wheels:** If our cloud host (Cloud Run) has no GPU, shipping Nvidia CUDA libraries is like carrying a spare plane propeller inside a car trunk: dead weight.

---

## 2. How It Works (The Mechanics)

### 2.1 Multi-Stage Pipeline Mechanics

Instead of a single `FROM` instruction that keeps everything in one image, multi-stage builds declare multiple `FROM` blocks. Only the **last stage** becomes the final image.

```
[ Stage 1: builder (python:3.11-slim) ]
  ├── Create clean virtual environment (/opt/venv)
  ├── Upgrade pip & install CPU-only PyTorch (--index-url .../cpu)
  └── Install heavy requirements (-r requirements.txt)
                     │
                     │  (COPY --from=builder /opt/venv /opt/venv)
                     ▼
[ Stage 2: runner (python:3.11-slim) ]  <-- Final Image (~400MB vs 2GB+)
  ├── Add system user & group (appuser)
  ├── Configure cache directories (/tmp/huggingface, /tmp/torch)
  ├── Copy application source code (src/, static/, data/)
  ├── Set permissions (chown to appuser)
  └── Start server via exec uvicorn (PID 1 signal forwarding)
```

### 2.2 Key Mechanics in Containerizing AI Services

1. **Python Environment Isolation (`/opt/venv`):**
   * Installing packages into a dedicated virtual environment in `builder` allows clean copying to `runner` via `COPY --from=builder /opt/venv /opt/venv` without dragging system package manager artifacts.
2. **CPU-Only PyTorch Optimization:**
   * Standard `pip install torch` grabs GPU/CUDA binaries (~540MB–1.2GB).
   * Pointing to `https://download.pytorch.org/whl/cpu` installs the CPU-optimized wheel (~180MB), immediately cutting hundreds of megabytes. Subsequent packages (like `sentence-transformers`) reuse this CPU wheel.
3. **Environment Flags for Python Containers:**
   * `PYTHONDONTWRITEBYTECODE=1`: Prevents writing `.pyc` files to disk during execution.
   * `PYTHONUNBUFFERED=1`: Flushes stdout/stderr immediately so logs appear in real-time in Docker and Cloud Run logging consoles.
4. **Non-Root Execution (`USER appuser`):**
   * Running as non-root mitigates container breakout attacks. If an exploit compromises FastAPI, the attacker cannot modify system files or mount points.
5. **Signal Forwarding with `exec`:**
   * Running `CMD exec uvicorn ...` makes Python process ID 1 (`PID 1`). When Docker or Cloud Run stops the container, the `SIGTERM` signal routes directly to Uvicorn for graceful shutdown rather than getting swallowed by a shell.

---

## 3. Trade-Offs & Comparisons (When to Use What)

### 3.1 Single-Stage vs. Multi-Stage Docker Builds

| Dimension | Single-Stage Build | Multi-Stage Build |
|---|---|---|
| **Image Size** | Massive (1GB–3GB+) due to compilers & build caches | Compact (~350MB–500MB); contains only runtime code and compiled wheels |
| **Build Speed** | Faster on first simple script, but slower image transfers | Better cache reusability across builds; fast registry push/pull |
| **Security Surface** | High (build tools, compiler packages left in image) | Minimal (only runtime dependencies present) |
| **Maintenance** | Single simple Dockerfile | Slightly more Dockerfile syntax to maintain |
| **Verdict** | Fine for simple toy scripts | **Mandatory standard for production AI services** |

### 3.2 Persistent Data: Volume Mount vs. Baking into Image

| Approach | How It Works | Best Used For | Production Caveat |
|---|---|---|---|
| **Baked into Image** (`COPY data/ ./data/`) | Files exist inside image layers | Static FAQ documents, default lookup tables | Updates require full image rebuild and redeployment. |
| **Docker Bind Mount** (`./data:/app/data`) | Local host folder mapped directly into container | Local dev with ChromaDB; instant document ingestion feedback | Tied to host filesystem path; does not scale horizontally across cloud nodes. |
| **Managed DB / Object Store** (Cloud Storage / Cloud SQL) | Remote service over private network | Production vector indexes and large model checkpoints | Requires network calls; need fallback or local caching. |

---

## 4. Production Code Reference

### 4.1 Production Multi-Stage `Dockerfile` (From `studentgator-rag-service`)

```dockerfile
# Stage 1: builder - temporary environment to compile and install dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Prevent .pyc clutter and ensure unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create clean virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 1. Install lightweight CPU-only PyTorch using BuildKit cache mount
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --timeout 100 --retries 10 torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install remaining RAG dependencies using BuildKit cache mount
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 100 --retries 10 -r requirements.txt


# Stage 2: runner - minimal production runtime
FROM python:3.11-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8080 \
    HF_HOME=/tmp/huggingface \
    TRANSFORMERS_CACHE=/tmp/huggingface \
    TORCH_HOME=/tmp/torch

# Grab only the compiled venv from builder
COPY --from=builder /opt/venv /opt/venv

# Run as non-root user for security (container best practice)
RUN addgroup --system appgroup && adduser --system --home /home/appuser --group appuser && \
    mkdir -p /tmp/huggingface /tmp/torch && \
    chown -R appuser:appgroup /tmp/huggingface /tmp/torch

# Copy app code, static chat UI, and vector DB directly with appuser ownership (no duplicate layer)
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup data/ ./data/
COPY --chown=appuser:appgroup static/ ./static/

USER appuser

EXPOSE 8080

# exec forwards SIGTERM directly to uvicorn for graceful shutdown
CMD exec uvicorn src.main:app --host 0.0.0.0 --port ${PORT}
```

### 4.2 Keeping Build Context Lean: `.dockerignore`

Before Docker runs `docker build`, it sends the root folder context to the Docker daemon. Omitting unwanted files keeps build start times instantaneous:

```gitignore
.git
.github
.venv
__pycache__
*.pyc
.env
tests/
*.bat
benchmark_results.md
compare_rag.py
test_client.py
```

### 4.3 Local Orchestration with `docker-compose.yml`

Wraps environment injection and volume persistence so developers do not need to memorize verbose `docker run` flags:

```yaml
services:
  rag-service:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: studentgator-rag-service
    ports:
      - "8080:8080"
    # Load API keys from local .env (never baked into image)
    env_file:
      - .env
    environment:
      - PORT=8080
    volumes:
      # Mount data folder so Chroma vector DB changes persist across container restarts
      - ./data:/app/data
    restart: unless-stopped
```

---

## 5. Whiteboard / Interview Talking Points

* **Why use multi-stage builds for a RAG / FastAPI container?**
  Multi-stage builds decouple the build environment from the runtime environment. By installing dependencies and compiling wheels in a `builder` stage and copying only `/opt/venv` to a bare `runner` stage, we strip out compiler tools, headers, and package caches. This shrinks our image from over 2GB to roughly 400MB, speeding up CI/CD pushes, slashing cold-start download times, and reducing attack surface.

* **How do we optimize PyTorch and Transformers image sizes when running on CPU?**
  Standard PyTorch pulls massive Nvidia CUDA packages by default. We explicitly point pip to the official CPU wheel index (`--index-url https://download.pytorch.org/whl/cpu`). This drops PyTorch from ~540MB to ~180MB. Because downstream packages like `sentence-transformers` resolve dependencies against the already-installed CPU PyTorch, the entire embedding pipeline stays lean.

* **Why must we run containers as a non-root user and use `exec` in `CMD`?**
  By default, containers execute as `root`. If a vulnerability in an endpoint allows arbitrary file writes or command execution, an attacker gains root privileges inside the container. We define a locked-down `appuser` and grant write access only to specific working paths (`/tmp/huggingface`, `/app/data`). We prefix `CMD` with `exec` so Uvicorn runs as PID 1, allowing the process to receive OS `SIGTERM` signals for graceful request draining during deployments.
