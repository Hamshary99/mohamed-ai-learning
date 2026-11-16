# Docker Architecture & Fundamentals

## 1. What Is It & Why Does It Exist?

### What is Docker?
Docker is an open source platform that simplifies shipping, developing and running applications into isolated environments.

It helps with the "it works on machine dilemma" instead of having to setup the whole dependencies to run one application, in different OS systems too, docker ensures portability.

*Reinforcing intuition:*
* Docker packages application code, runtime libraries, and environment configs into an immutable image, eliminating environment drift between local dev and cloud production.
* Physical analogy: Standardized shipping containers. The cargo carrier (host/cloud) does not care what is packed inside (FastAPI, ChromaDB, Nginx); it only needs to know how to load and run the standard container.

### Containers vs. Virtual Machines (VMs)
Virtual Machines and Containers represent two fundamentally distinct approaches to isolation.

#### Virtual Machines (Hardware-Level Virtualization)
Every single VM independently runs and boots their own OS kernel, system drivers and daemons.

**Trade-offs:**
* Heavy footprint (gigabytes per VM).
* Slow boot times (minutes) because an entire OS must initialize.
* **Benefit:** Very strong, hard isolation boundary; allows running completely different OS kernels concurrently on the same machine (e.g., Linux and Windows on one physical host).

#### Containers (Operating-System-Level Isolation)
**How it works:** Instead of virtualizing physical hardware, containers perform isolation at the OS user-space level, with shared OS kernels.

**Trade-offs:**
* Extremely lightweight (megabytes) and spawns in milliseconds to seconds.
* Near-native CPU and execution performance (no hypervisor translation layer).
* **Limitation:** Tied to the host kernel family; a vulnerability in the shared host kernel exposes adjacent containers.

---

## 2. How It Works (The Mechanics)

### 2.1 Docker Architecture (Client-Server Model)

The Docker Engine is structured as a decoupled **Client-Server Architecture**:

```
+------------------------------------+
|        Docker Client (CLI)         |  <-- 'docker run', 'docker build'
+------------------------------------+
                  |  (REST API calls over /var/run/docker.sock)
                  v
+------------------------------------+
|       Docker Host (Daemon)         |
|  [dockerd]                         |  <-- Coordinates images, volumes, networks
|      |                             |
|      v                             |
|  [containerd]                      |  <-- OCI runtime supervisor
|      |                             |
|      v                             |
|    [runc]                          |  <-- Spawns kernel namespaces & cgroups
+------------------------------------+
                  ^
                  |  (pull / push layers)
+------------------------------------+
|          Docker Registry           |  <-- Docker Hub, private cloud registries
+------------------------------------+
```

Deconstructing the components:

1. **Docker Client (`docker` CLI):**
   * The command-line interface tool we interact with directly (e.g., `docker run`, `docker ps`).
   * It does not execute containers itself; it converts commands into structured **REST API HTTP requests** and transmits them to the daemon over `/var/run/docker.sock`.
2. **Docker Host & Daemon (`dockerd`):**
   * A persistent background service listening on a UNIX domain socket (`/var/run/docker.sock`) or network socket.
   * Coordinates high-level Docker objects (images, storage volumes, virtual networks).
   * Delegates container lifecycle management to **containerd**, which in turn calls low-level execution runtimes like **runc** to configure kernel `namespaces` and `cgroups`.
3. **Container Runtime (`containerd` & `runc`):**
   * `containerd`: Supervises container lifecycles, manages image distribution (pulling/pushing), and mounts storage.
   * `runc`: Lightweight OCI runtime that makes system calls to create isolated namespaces and cgroup limits, starts the container's PID 1, and exits.
4. **Docker Registry:**
   * A centralized storage repository for versioned container images (e.g., Docker Hub, Google Cloud Artifact Registry).
   * The daemon pulls images when they are not available locally and pushes built images for deployment.

### 2.2 Concrete Practical Example: Running an Isolated Process

To see this entire architecture in action, run a minimal Nginx web server:
```bash
docker run -d --name test-web -p 8080:80 nginx:alpine
```

What happens behind the scenes:
1. **Client -> Daemon:** The `docker` CLI sends a REST API `POST` request to `dockerd` over the local UNIX socket (`/var/run/docker.sock`).
2. **Registry check:** `dockerd` checks if `nginx:alpine` exists locally; if not, it downloads the read-only layers from Docker Hub.
3. **Process Isolation:** `containerd` and `runc` instruct the host Linux kernel to create isolated namespaces (`pid`, `net`, `mnt`) and configure `cgroups` for resource governance.
4. **Networking:** Docker binds host port 8080 to container port 80 using virtual ethernet (`veth`) and iptables, routing external web traffic into the container's isolated network space.
5. **Clean teardown:** Running `docker stop test-web` and `docker rm test-web` cleanly terminates the isolated process tree (sending `SIGTERM` to PID 1) and removes its ephemeral writable layer without leaving leftover files or software clutter on the host machine.

### 2.3 Images vs. Containers & Copy-on-Write (CoW)

**Docker image:**
An immutable read only package containing the application code and all its dependencies. It is created using a Dockerfile.
Image is contained of read only filesystem layers Each layer records a specific delta or change (e.g., installing a package, adding files).

**Docker container:**
A runnable instance of a Docker image. Docker mounts all read only layers into a view via union file system(AUFS, overlay2) and adds an ephemeral read-write layer on top.

**How Copy-on-Write (CoW) works:**
* Mental model: A stack of transparent tracing sheets placed over a printed master blueprint. We can read the blueprint directly through the sheets, and any edits get drawn only on the blank top sheet without touching the original.
* **Reading:** When a container reads a file, it reads straight from the shared read-only image layer below.
* **Modifying:** When a process tries to edit an existing file, the union filesystem (`overlay2`) copies that file *up* into the container's top writable layer first, then applies the edit there.
* **Sharing:** Because the base layers never change, 10 containers started from the same image share the exact same disk space for those read-only layers without interfering with one another.
* **Ephemeral state:** When a container is deleted (`docker rm`), its top writable layer disappears and all runtime changes are lost unless stored outside the container (volumes, covered later).

```
 [ writable container layer ]  <-- Copy-on-Write happens here (deleted on docker rm)
 ----------------------------
 [ layer 3: COPY app code   ]  \
 [ layer 2: RUN pip install ]   } read-only image layers (shared across containers)
 [ layer 1: FROM python     ]  /
```

### 2.4 How Dockerfile Instructions Build Layers

A `Dockerfile` is a recipe script containing sequential instructions that Docker runs top-to-bottom to assemble an image layer by layer. Each instruction that changes files (`RUN`, `COPY`, `ADD`) commits a new cached layer.

1. **`FROM` (Base Image):**
   * Initializes the build and sets the starting parent filesystem layer (e.g., `FROM python:3.11-slim`). Every `Dockerfile` must start with `FROM`.
2. **`WORKDIR` (Working Directory):**
   * Sets the current directory inside the image for all subsequent `RUN`, `COPY`, `CMD`, and `ENTRYPOINT` instructions (and creates the folder if missing).
   * Clean alternative to fragile commands like `RUN cd /app && ...` which reset on the next layer.
3. **`COPY` vs. `ADD` (Bringing Files In):**
   * Transfers local project files from the build context directory into the image filesystem.
   * `COPY` does plain file copying and nothing else. `ADD` has hidden side effects (auto-extracting `.tar` archives and fetching remote URLs), which makes builds less predictable.
4. **`RUN` (Build-Time Execution):**
   * Executes shell commands *while building the image* (e.g., `apt-get install`, `pip install`) and commits the filesystem changes as a new read-only layer.
   * Because layers are permanent once committed, cleanup must happen in the **same `RUN` instruction** using `&&` and flags like `--no-cache-dir`—deleting cache files in a *later* `RUN` instruction hides them from the final view, but the earlier layer still holds their bytes.
5. **`EXPOSE` (Port Documentation):**
   * Acts purely as metadata documenting which port the application inside the container listens on (e.g., `EXPOSE 8000`).
   * **Crucial fact:** `EXPOSE` does **not** actually publish or open the port to the host machine. We still must explicitly pass `-p 8080:8000` at `docker run` time to forward host traffic into the container.
6. **`CMD` vs. `ENTRYPOINT` (Runtime Startup Command):**
   * **`ENTRYPOINT`:** Locks in the main executable that always runs when the container starts (e.g., `ENTRYPOINT ["uvicorn"]`). CLI arguments appended to `docker run <image> ...` get passed as arguments *to* `ENTRYPOINT`.
   * **`CMD`:** Sets the default command or default arguments passed to `ENTRYPOINT` (e.g., `CMD ["app.main:app", "--host", "0.0.0.0", "--port", "8000"]`). Unlike `ENTRYPOINT`, `CMD` is easily overridden if we append a new command at the end of `docker run <image> bash`.
   * **Exec Form vs. Shell Form:**
     * *Exec form* (`CMD ["uvicorn", "main:app"]`): Runs the process directly as **PID 1**, so `SIGTERM` from `docker stop` reaches the app cleanly.
     * *Shell form* (`CMD uvicorn main:app`): Wraps the command inside `/bin/sh -c`, making the shell PID 1 instead of our app. The shell often swallows `SIGTERM`, forcing Docker to wait 10 seconds and `SIGKILL` the container.

### 2.5 Layer Caching & Build Optimization

Docker builds a `Dockerfile` top-to-bottom and computes a checksum hash for each instruction and its input files. If the instruction and its inputs match a previous build, Docker skips execution and reuses the cached layer.

**The Invalidation Rule:**
* If any layer changes (e.g., a file copied into that layer is edited), that layer and **every single layer below it** are invalidated and rebuilt from scratch.

**Why Dependencies Go Before Source Code:**
* `requirements.txt` changes rarely; application source code (`./src` or `./app`) changes constantly.
* Copying `requirements.txt` and running `pip install` *before* copying source code ensures Docker reuses the heavy dependency layer. Editing Python code only invalidates the final `COPY` layer, keeping rebuilds at 1–2 seconds instead of minutes.

```dockerfile
# GOOD: Least-frequently changed at top, most-frequently changed at bottom
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src
```

**`.dockerignore` (Controlling the Build Context):**
* Running `docker build` tells the CLI to tar and upload the local build context folder to `dockerd`.
* Adding a `.dockerignore` file prevents local artifacts (`__pycache__/`, `.git`, `.venv`, `.env`) from being uploaded to the daemon or baked into the image—cutting build latency and preventing secret leaks (such as API keys in `.env`).

### 2.6 Environment Variables: Decoupling Config from Images

A container image should be **immutable and generic**, the exact same image should rerun without rebuilding it. To change db credentials or api keys we inject them as **environment variables** at runtime.

* **Security rule:** Never hardcode secrets or `COPY .env` into the image—anyone who pulls the image can inspect its layers and read the credentials.
* **Three ways to pass environment variables at runtime:**
  1. **Inline CLI flag:** `docker run -e DB_HOST=postgres-db my-app`
  2. **Environment file:** `docker run --env-file .env my-app`
  3. **Docker Compose:** Set inside the `environment` (or `env_file`) block in `compose.yaml`.

### 2.7 Persistent Storage: Volumes vs. Bind Mounts

When a container is deleted (`docker rm`), its top writable layer is permanently destroyed. If a container runs PostgreSQL or stores StudentGator's ChromaDB vector embeddings inside that writable layer, all data vanishes with the container. Docker solves this with **external persistent mounts** that bypass the ephemeral writable layer and write straight to the host disk.

1. **Named Volumes (Production Default):**
   * **What it is:** A managed storage folder that Docker creates and tracks inside its own host directory (`/var/lib/docker/volumes/` on Linux). We refer to it by a short name (like `db_data`) instead of a hardcoded laptop path.
   * **Why use it:** Survives container restarts and deletions, delivers fast direct disk I/O, is isolated from accidental edits on the host, and works identically across different servers.
   * **Syntax:** `docker run -v db_data:/var/lib/postgresql/data postgres`

2. **Bind Mounts (Development Default):**
   * **What it is:** Mapping an exact folder path from our local machine directly into a folder inside the container.
   * **Why use it:** Ideal for **local hot-reloading**. Mounting our local `./src` folder into `/app/src` means edits saved in our editor immediately appear inside the running container with zero image rebuilds.
   * **Syntax:** `docker run -v $(pwd)/src:/app/src my-api`

### 2.8 Networking & Port Mapping

When containers run, they attach to a virtual network bridge inside Docker:

* **Default Bridge Network:** Containers communicate only if they know each other's temporarily dynamic IP addresses, can't resolve each other by name.
* **User-Defined Bridge Network (Production & Compose):** Docker activates an embedded internal DNS server (`127.0.0.11`). Containers attached to the same custom network can resolve and call one another **directly by container or service name** (e.g., `http://postgres_db:5432`), even if their internal IP addresses change on restart.

**Port Mapping (`-p HOST:CONTAINER`):**
* Because every container runs inside its own isolated `net` namespace, its ports are private and unreachable from the outside host by default.
* **Mental model:** `-p 8080:80` = `[Outside Door on Host] : [Inside Door in Container]`. External traffic arriving on host port `8080` is forwarded across the virtual bridge to port `80` inside the container.
* **Why `0.0.0.0` matters inside the app:** Apps like Uvicorn/FastAPI must bind to `0.0.0.0` (all interfaces) rather than `127.0.0.1` inside the container. Binding to `127.0.0.1` restricts the server to the container's own internal loopback, blocking traffic forwarded from the host bridge.

### 2.9 Docker Compose: Multi-Container Orchestration

**The problem:** Real systems (like StudentGator's FastAPI + PostgreSQL/ChromaDB stack) need multiple containers running together. Typing long `docker run` commands with matching network flags, volume mounts, port mappings, and environment variables every time is brittle and error-prone.

**What Docker Compose is:**
* A declarative blueprint (`compose.yaml`) and CLI (`docker compose`) that defines and runs an entire multi-container application stack as one unit.
* Inside `compose.yaml`, we declare **services** (containers), **volumes** (persistent storage), and **networks** (custom bridges).
* Running `docker compose up -d` automatically creates the shared user-defined network (enabling DNS by service name), provisions named volumes, builds images if needed, and starts every service. Running `docker compose down` stops and removes the containers and network while keeping named volumes intact.

**Startup Ordering (`depends_on` & `healthcheck`):**
* Plain `depends_on` only waits for a database container *process* to start—not until the database is actually ready to accept queries, which can cause an API container to crash on boot.
* Adding a `healthcheck` (e.g., running `pg_isready` inside Postgres every 5 seconds) paired with `condition: service_healthy` forces `api-service` to wait until the database passes its readiness check before starting.

---

## 3. Trade-Offs & Comparisons (When to Use What)

### 3.1 VM vs. Container

| Feature | Virtual Machine (VM) | Container (Docker) |
|---|---|---|
| **Isolation Level** | Hardware-level (Hypervisor) | OS user-space level (Kernel namespaces & cgroups) |
| **Guest OS** | Full dedicated OS per VM | None (shares host Linux kernel) |
| **Startup Time** | Minutes (full OS boot) | Milliseconds to seconds (process spawn) |
| **Footprint** | Gigabytes per VM | Megabytes to hundreds of MBs |
| **Performance** | Overhead from hypervisor translation | Near-native bare-metal speed |
| **OS Flexibility** | Run Linux and Windows on one host | Linux containers require Linux kernel |

### 3.2 `python:3.11-slim` vs. `python:3.11-alpine` (Python & AI Services)

| Trade-Off | `slim` (Debian-based) | `alpine` (Alpine Linux) |
|---|---|---|
| **C Library** | Standard `glibc` | Lightweight `musl libc` |
| **Image Size** | ~150 MB | ~50 MB |
| **Python / AI Wheels** | Installs pre-compiled wheels (`numpy`, `torch`, `chromadb`) immediately | Often lacks pre-built `musl` wheels; forces slow compilation from C/C++ source or fails |
| **When to Use** | Default choice for Python ML, RAG, and FastAPI services | Minimal Go/Node/Nginx utilities with zero complex C-extensions |

### 3.3 Dockerfile & Daemon Trade-Offs

* **`COPY` vs. `ADD`:** Always use `COPY` for local files so layer contents stay transparent and auditable; reserve `ADD` strictly for explicit local tar extraction.
* **Docker (`dockerd`) vs. Daemonless Runtimes (Podman):** `dockerd` gives a unified standard API over `/var/run/docker.sock`, but socket access equals root access on the host. Daemonless tools run rootless without a single daemon process, reducing privilege escalation risk on shared servers.

### 3.4 Named Volumes vs. Bind Mounts

| Trade-Off | Named Volume (`db_data:/path`) | Bind Mount (`./src:/app/src`) |
|---|---|---|
| **Who Manages It** | Docker manages storage under `/var/lib/docker/volumes/` | Host OS path chosen manually |
| **Portability** | High (works across Linux, macOS, Windows without path edits) | Low (tied to local folder layout and OS permissions) |
| **Performance** | High-throughput native disk I/O | Slower on macOS/Windows due to VM file-sharing overhead |
| **When to Use** | Production databases (PostgreSQL), ChromaDB vector stores | Local development hot-reloading (VS Code edits reflected instantly) |

### 3.5 Default Bridge vs. User-Defined Bridge Network

* **Default `bridge`:** No automatic DNS—containers must hardcode dynamic IPs that break on restart. Use only for quick isolated one-off tests.
* **User-Defined Bridge (`docker network create` / Compose):** Embedded DNS (`127.0.0.11`) lets services reach each other by container/service name reliably. Always use this in multi-container stacks.

---

## 4. Production Code Reference

### 4.1 StudentGator FastAPI `Dockerfile` & `.dockerignore`

`Dockerfile`:
```dockerfile
# 1. Base Image: Lean Debian Linux with Python runtime (glibc for pre-compiled AI wheels)
FROM python:3.11-slim

# 2. Establish working directory inside container
WORKDIR /app

# 3. Cache Optimization: Install dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy source code (invalidates only this layer when code changes)
COPY ./src ./src

# 5. Document listening port (metadata only; runtime still needs -p)
EXPOSE 8000

# 6. Exec form startup command (PID 1 graceful shutdown)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`.dockerignore`:
```text
.git
.venv
__pycache__/
*.pyc
.env
```

Build and run commands:
```bash
# Build the image and tag it as 'studentgator-api:v1' using current directory (.)
docker build -t studentgator-api:v1 .

# Run detached, mapping host port 8000 to container port 8000 with cgroup limits
docker run -d --name api-service -p 8000:8000 --memory=512m --cpus=1.0 studentgator-api:v1
```

### 4.2 Inspecting the Daemon Socket & Managing Containers in Python

```bash
# Verify the Docker daemon directly over the local REST socket
sudo curl --unix-socket /var/run/docker.sock http://localhost/version
```

```python
import docker
from docker.errors import DockerException, ImageNotFound

def run_isolated_service() -> None:
    """Spawns an isolated container with explicit memory and CPU bounds."""
    try:
        # Connect to the local Docker daemon socket via REST client
        docker_client = docker.from_env()

        container = docker_client.containers.run(
            image="python:3.11-slim",
            name="studentgator-service",
            command=["python", "-c", "import time; time.sleep(30)"],
            mem_limit="256m",       # cgroups memory cap
            nano_cpus=1000000000,   # cgroups CPU cap: 1.0 core
            detach=True,
            remove=False
        )

        print(f"Started container: {container.short_id} (Status: {container.status})")

        # Clean teardown removes the container and its writable layer
        container.stop(timeout=5)
        container.remove()
        print("Container stopped and writable layer deleted.")

    except (ImageNotFound, DockerException) as error:
        print(f"Docker operation failed: {error}")

if __name__ == "__main__":
    run_isolated_service()
```

### 4.3 Multi-Container Stack: FastAPI + PostgreSQL (`compose.yaml`)

This `compose.yaml` connects all four runtime concepts—environment variables, named volumes vs. bind mounts, custom bridge DNS, and `healthcheck` startup ordering—for a StudentGator API + database stack:

```yaml
services:
  # 1. Web API Service
  api-service:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: fastapi_api
    ports:
      # Forward host 8000 -> container 8000
      - "8000:8000"
    environment:
      # Connects directly using the service name 'postgres_db' via embedded DNS
      - DATABASE_URL=postgresql://user:secret@postgres_db:5432/app_db
    volumes:
      # Bind Mount for hot-reloading local Python code during development
      - ./src:/app/src
    depends_on:
      postgres_db:
        condition: service_healthy
    networks:
      - app_network

  # 2. Database Service
  postgres_db:
    image: postgres:16-alpine
    container_name: postgres_db
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: app_db
    volumes:
      # Named Volume so database rows survive container restarts and deletions
      - db_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d app_db"]
      interval: 5s
      timeout: 5s
      retries: 5
    networks:
      - app_network

# Top-level managed persistent storage
volumes:
  db_data:

# Top-level user-defined bridge network (automatic service-name DNS)
networks:
  app_network:
    driver: bridge
```

Common Compose lifecycle commands:
```bash
# Build images if needed and start all services in the background
docker compose up -d

# Stop and remove containers + network while preserving the named volume (db_data)
docker compose down
```

---

## 5. Whiteboard / Interview Talking Points

- **Client-Server Decoupling:** The `docker` CLI is a stateless HTTP client talking over `/var/run/docker.sock` to `dockerd`, which coordinates high-level objects and delegates execution to `containerd` and `runc` to create kernel namespaces and cgroups.
- **Image Layers & Copy-on-Write:** An image is a stack of shared, read-only delta layers; a running container simply mounts one thin writable layer on top via `overlay2`. Copying `requirements.txt` and running `pip install` before copying source code maximizes layer cache reuse during builds.
- **Layer Caching & `.dockerignore`:** Docker hashes every build instruction and invalidates all downstream layers if an earlier layer changes. Ordering `COPY requirements.txt` and `RUN pip install` before `COPY ./src ./src` keeps Python code rebuilds down to seconds, while `.dockerignore` prevents `.venv`, `__pycache__`, and `.env` secrets from entering the build context.
- **Immutability, Persistence & Networking:** Images stay generic by injecting credentials via runtime environment variables rather than baking secrets in. Stateful stores (PostgreSQL, ChromaDB) require **named volumes** so data survives `docker rm`, while local dev uses **bind mounts** for hot-reloading. User-defined bridge networks enable automatic DNS resolution by service name (`postgres_db`) instead of brittle dynamic IPs.
- **Why `slim` + Exec Form + `healthcheck` Matter for StudentGator:** Using `python:3.11-slim` (`glibc`) avoids hours of compiling PyTorch/NumPy/ChromaDB dependencies that break on Alpine (`musl`); using `CMD [...]` in exec form ensures FastAPI/Uvicorn runs as PID 1 for graceful `SIGTERM` shutdown on Cloud Run; and pairing `depends_on` with `condition: service_healthy` in `compose.yaml` ensures the API waits until PostgreSQL is actually ready for queries before booting.
