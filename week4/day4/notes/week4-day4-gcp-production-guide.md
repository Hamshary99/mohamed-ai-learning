# Week 4 Day 4: GCP Cloud Architecture & Production AI Deployment Guide

A concise, note-friendly reference covering Google Cloud Platform fundamentals, storage topologies, container logistics, and production serving architectures for machine learning models and Retrieval-Augmented Generation (RAG) systems.

---

## Part 1: GCP Fundamentals (Compute Engine, IAM, and Networking)

### 1. Resource Hierarchy: The Governance Tree
Google Cloud organizes all cloud assets into an inherited organizational tree:

* **Organization**: The root domain node representing the enterprise. Company-wide security and compliance policies attach here and flow downward.
* **Folders**: Logical groupings for business units, teams, or environments (e.g., Engineering, Staging, Production).
* **Projects**: The core administrative, billing, and API enablement boundary. Every virtual machine, storage bucket, or Cloud Run service must belong to exactly one project.
* **Resources**: The concrete instances running inside a project (e.g., a Compute Engine VM, a Cloud Storage bucket, or a Cloud SQL database).

```
Organization
  └── Folders (Departments / Environments)
        └── Projects (Billing & API Boundaries)
              └── Resources (VMs, Buckets, Cloud Run)
```

---

### 2. Identity and Access Management (IAM)
IAM enforces authentication and authorization under the principle of least privilege:

* **Principals**:
  * Human accounts (Google Accounts, Google Workspace, or federated enterprise identity providers).
  * **Service Accounts**: Special non-human identities assigned to applications and VMs. They allow services to talk to Google APIs without embedding hardcoded credentials in application code.
* **Roles**:
  * **Basic / Primitive Roles**: Legacy broad permissions (`Owner`, `Editor`, `Viewer`). Avoid in production—`Editor` grants expansive write and delete access across almost all services.
  * **Predefined Roles**: Fine-grained, job-specific roles maintained by Google (e.g., `roles/run.invoker`, `roles/storage.objectViewer`, `roles/artifactregistry.writer`).
  * **Custom Roles**: Tailored permission bundles created when predefined roles grant more access than necessary.
* **Policy Bindings**: Granting a role to a principal on a specific resource node. Because permissions are inherited, granting a role at the Folder level applies to every project and resource inside that folder.

---

### 3. Compute Engine Core Abstractions
Compute Engine provides Infrastructure as a Service (IaaS) by running virtual machines on Google physical server fleet:

* **Machine Families**:
  * **General-Purpose (`E2`, `N2`, `C3`)**: Balanced CPU and memory for web servers, dev environments, and microservices. The `E2` family uses cost-efficient resource scheduling (`e2-micro` has 2 vCPUs, 1 GB RAM, suitable for lightweight tests).
  * **Compute-Optimized (`C2`, `C2D`)**: High CPU clock speeds for intensive computation and mathematical modeling.
  * **Memory-Optimized (`M1`, `M2`, `M3`)**: Massive RAM-to-vCPU ratios for large in-memory databases and caching.
  * **Accelerator-Optimized (`A2`, `A3`, `G2`)**: Attached NVIDIA GPUs (L4, A100, H100) for deep learning model training, LLM fine-tuning, and large-scale model inference.
* **Provisioning Models**:
  * **Standard VMs**: Dedicated, persistent runtime capacity governed by standard Service Level Agreements (SLAs).
  * **Spot VMs (Preemptible)**: Excess Google capacity offered at significant discounts (60% to 91% below on-demand prices). The trade-off: Google can reclaim and terminate Spot instances at any time when capacity is needed elsewhere. They are suitable strictly for stateless, fault-tolerant batch processing or parallel training checkpoints—never for production databases or live API endpoints.

---

### 4. Networking Basics (VPC, Subnets, and Firewalls)
* **Global Virtual Private Cloud (VPC)**: Unlike traditional clouds where virtual networks are bound to a single data center, a Google Cloud VPC is global. One VPC spans worldwide regions, containing regional subnets that talk to each other privately without VPN tunnels.
* **Internal vs. External IPs**:
  * **Internal IPs**: Private addresses assigned from the subnet range. Traffic between instances in the same VPC stays on Google private global fiber network.
  * **External IPs**: Public, internet-routable addresses. In production, backend servers and databases omit external IPs to stay isolated from internet scanning, accessing outbound package updates via Cloud NAT.
* **VPC Firewall Rules**: Distributed, stateful packet filters applied directly at each VM virtual network interface (NIC). They evaluate ingress/egress direction, source IP ranges (CIDRs), protocols/ports (e.g., TCP port 8000), and target VMs via network tags. Because rules are stateful, return traffic for an allowed incoming connection is automatically permitted.
* **VPC Network Peering**: Connects two distinct VPC networks privately using internal RFC 1918 addresses. Traffic traverses Google internal backbone with high bandwidth and low latency, bypassing the public internet entirely.

---

### Part 1 Summary Table
| Concept | Definition | Production Rule of Thumb |
| :--- | :--- | :--- |
| **Project** | Administrative & billing container | One project per environment (Dev, Staging, Prod). |
| **Service Account** | Non-human application identity | Use attached service accounts; never hardcode credentials. |
| **Predefined Roles** | Granular Google-managed permissions | Always use predefined roles instead of legacy `Editor`/`Owner`. |
| **Spot VMs** | Cheap, preemptible compute | Use for batch jobs and training; never for live web APIs. |
| **VPC Firewall** | Stateful NIC-level packet filtering | Apply rules using network tags; restrict ingress to required ports. |

---

## Part 2: Storage & Cost Management (Cloud Storage, Persistent Disk, and Billing)

### 1. Cloud Storage (GCS): Object Storage
Google Cloud Storage (GCS) is a fully managed, highly durable object storage system designed for storing unstructured files, media assets, training datasets, and serialized machine learning model checkpoints:

* **Buckets and Objects**: Data is stored as immutable objects organized into globally unique, named containers called buckets. GCS provides a flat namespace; forward slashes in object keys emulate folder hierarchies, but objects remain directly addressable by their URI (`gs://bucket-name/models/v1/model.pkl`).
* **Storage Classes & Access Frequencies**:
  * **Standard Storage**: For frequently accessed data (live website assets, active ML model weights, streaming media). No minimum retention period.
  * **Nearline Storage**: For infrequently accessed data read less than once per month (monthly reports, periodic backups). Minimum 30-day storage commitment.
  * **Coldline Storage**: For rarely accessed data read less than once per quarter (quarterly compliance archives, disaster recovery images). Minimum 90-day storage commitment.
  * **Archive Storage**: For digital preservation accessed less than once per year (multi-year regulatory records). Lowest storage cost per gigabyte, with a 365-day storage commitment.
* **Object Lifecycle Management**: Automated bucket rules that inspect object age or versioning status, automatically downgrading older files (e.g., moving data from Standard to Nearline after 30 days) or deleting temporary artifacts to prevent cost accumulation.
* **Signed URLs**: Cryptographic URLs generated by service accounts that grant temporary, time-limited read or write access to a specific object without requiring the user to have a Google account or IAM permissions.

---

### 2. Persistent Disk (PD): Block Storage for Compute
Persistent Disks provide durable, network-attached block devices mounted to Compute Engine VMs and GKE container nodes:

* **Network-Attached Architecture via Colossus**:
  Persistent Disks are not physical disks inside the VM chassis. They are virtualized block devices attached over Google high-speed data center network. The storage backend is powered by **Colossus**, Google distributed file system. Colossus automatically shards and replicates data across multiple physical disks and racks, ensuring durability without requiring software RAID configuration.
* **Persistent Disk Types**:
  * **Standard Persistent Disk (`pd-standard`)**: Backed by magnetic hard disk drives (HDD). Optimized for large, sequential I/O workloads (batch data processing, bulk log storage).
  * **Balanced Persistent Disk (`pd-balanced`)**: Backed by solid-state drives (SSD). Balances IOPS, throughput, and cost for general-purpose applications. Default boot disk in the Cloud Console.
  * **SSD Persistent Disk (`pd-ssd`)**: High-performance SSD storage designed for transactional enterprise databases, vector indexing engines, and applications demanding low latency and high random IOPS.
  * **Extreme Persistent Disk (`pd-extreme`)**: Ultra high-performance storage allowing provisioned IOPS configuration for mission-critical database workloads.
* **Zonal vs. Regional Persistent Disks**:
  * **Zonal Persistent Disk**: Replicated across physical hardware within a single zone. If that specific data center zone fails, the disk is temporarily unavailable.
  * **Regional Persistent Disk**: Synchronously replicated across two availability zones in the same region. If the primary zone fails, the disk can failover and attach to a standby VM in the secondary zone, maintaining high-availability database architectures.
* **Lifecycle Independence & Live Resizing**:
  Because Persistent Disks exist as independent services on Colossus, their lifecycle is decoupled from the compute instance. A VM can be deleted while its data disk is retained (`keep-disk` policy). Persistent Disks can also be dynamically resized online while mounted to a running VM, instantly scaling provisioned throughput and IOPS without rebooting.

---

### 3. Cloud Billing & Cost Management
Production cloud engineering requires proactive cost governance to prevent unexpected billing spikes:

* **Cloud Billing Account**: Root-level financial entity linked to a payment profile. One billing account can be linked to multiple Google Cloud projects, accumulating all charges into a consolidated monthly invoice. If a project is not linked to an active billing account, all paid compute resources and APIs are disabled.
* **Billing Reports & Analysis Views**:
  * **Billing Report**: Interactive console dashboards filtering usage costs by project, service, SKU, region, and labels over custom time ranges.
  * **Cost Table**: Detailed tabular breakdown of invoice charges including project numbers, service IDs, and SKU identifiers, exportable to CSV.
  * **Cost Breakdown**: Visual waterfall chart showing on-demand list pricing, contracted discounts, and usage credits (Sustained Use Discounts, Free Tier).
* **Budgets and Alert Thresholds**:
  * **Budgets**: Spending targets established per month or quarter, set to fixed monetary amounts (e.g., $50/month) or based on previous month spend.
  * **Alert Thresholds**: Automated triggers firing when actual or forecasted spending crosses percentage milestones (typically 50%, 90%, and 100%).
  * **Notification Channels**: Alerts are delivered via automated email notifications to billing administrators or published to Cloud Pub/Sub topics. Cloud Functions can listen to Pub/Sub alerts to execute programmatic cost-capping actions (e.g., shutting down non-essential development VMs).
* **BigQuery Billing Export**: Enables automated daily export of detailed billing data into a BigQuery dataset, allowing data teams to run SQL queries analyzing cost drivers, anomalous spikes, and unit economics per user.

---

### Part 2 Summary Table
| Storage / Billing Primitive | Key Attribute | Primary Use Case |
| :--- | :--- | :--- |
| **GCS Standard** | Unstructured object storage | Model weights (`model.pkl`), web assets, raw datasets. |
| **GCS Nearline / Coldline** | Low-cost storage with minimum duration | Backups, snapshots, compliance archives. |
| **Persistent Disk (`pd-balanced`)** | Colossus-backed network block storage | VM boot disks, database data directories. |
| **Regional Persistent Disk** | Dual-zone synchronous replication | High-availability stateful database failover. |
| **Budget Alerts** | 50%, 90%, 100% threshold notifications | Proactive warning system to prevent unexpected billing overruns. |

---

## Part 3: VM Operations, Docker on GCP, Artifact Registry & Cloud Run

### 1. Compute Engine VM Setup & Secure SSH Operations
Managing and connecting to Compute Engine virtual machines follows standardized cloud administrative patterns:

* **Connecting to Linux VMs**:
  * **SSH-in-Browser**: One-click terminal window in the Google Cloud console. Compute Engine generates an ephemeral SSH key pair and automatically writes the public key to instance metadata.
  * **Command Line via `gcloud compute ssh`**: The standard developer tool. It authenticates with Google Cloud credentials, generates/verifies SSH keys in `~/.ssh/google_compute_engine`, pushes public keys to VM metadata, and connects over TCP port 22:
    ```bash
    gcloud compute ssh instance-name --zone=us-central1-a
    ```
  * **OS Login (Recommended)**: Links SSH access directly to the user IAM identity. It eliminates manual key management, enforces multi-factor authentication, and records audit logs of every interactive SSH session.
* **Startup Scripts**: User data scripts executed automatically during initial VM boot. Used to automate package installations and runtime configuration.

---

### 2. Docker on GCP & Artifact Registry
Artifact Registry is Google Cloud modern, universal package manager for storing, managing, and securing container images and build artifacts (replacing legacy Container Registry `gcr.io`):

* **Repository Structure**: Follows a standard regional naming hierarchy:
  `LOCATION-docker.pkg.dev/PROJECT_ID/REPOSITORY_NAME/IMAGE_NAME:TAG`
* **Docker Authentication Workflow**:
  Before pushing or pulling images, the local Docker daemon must be configured with Google Cloud credentials:
  ```bash
  # 1. Create a dedicated Docker repository
  gcloud artifacts repositories create ml-repo \
      --repository-format=docker \
      --location=us-central1 \
      --description="Production container images"

  # 2. Configure local Docker daemon helper for the regional registry domain
  gcloud auth configure-docker us-central1-docker.pkg.dev

  # 3. Tag and push container image
  docker tag my-api:v1 us-central1-docker.pkg.dev/my-project/ml-repo/my-api:v1.0.0
  docker push us-central1-docker.pkg.dev/my-project/ml-repo/my-api:v1.0.0
  ```

---

### 3. Cloud Run: Serverless Container Architecture
Cloud Run is a fully managed serverless platform that runs stateless Docker containers directly in response to incoming HTTP requests, gRPC calls, or Pub/Sub events:

```
[ Incoming HTTP Requests ]
            |
            v
[ Cloud Run HTTPS Ingress ]
            |
    +-------+-------+
    |               |
    v               v
[ Container 1 ]  [ Container 2 ]  (Autoscaling: 0 to N instances)
(Concurrency=80) (Concurrency=80)
```

* **Stateless Execution**: Containers are ephemeral. Any file written to the local container filesystem lives in RAM (`tmpfs`) and vanishes when the instance terminates. Persistent state must be stored in external databases or Cloud Storage.
* **Scale-to-Zero**: When there is no incoming traffic, Cloud Run automatically scales active container instances down to zero. You pay zero dollars while idle, unlike Compute Engine VMs which incur continuous hourly charges.
* **Cold Starts & Min-Instances**: When a request arrives while scaled to zero, Cloud Run pulls the image, initializes the container, and boots the application before serving the request. For heavy Python AI containers loading large model weights into RAM, cold starts can introduce latency. Setting `--min-instances=1` keeps a baseline instance warm at all times.
* **Request Concurrency**: Unlike classic Function-as-a-Service platforms that process only one request per container instance, Cloud Run allows a single container instance to process multiple requests concurrently (configurable from 1 to 1000, default is 80). For asynchronous Python applications built with FastAPI and `asyncio`, high concurrency maximizes CPU/RAM utilization and prevents unnecessary container sprawl.
* **Immutable Revisions & Traffic Splitting**: Every deployment produces an immutable revision. Cloud Run allows gradual canary releases and A/B testing by splitting traffic between revisions (e.g., 90% to `v1.0`, 10% to `v1.1`) without service downtime.

---

### Part 3 Summary Table
| Component | Functionality | Key Production Practice |
| :--- | :--- | :--- |
| **`gcloud compute ssh`** | Secure CLI shell connection | Use OS Login to manage SSH access via IAM identities. |
| **Artifact Registry** | Regional container registry | Use regional `*.pkg.dev` domains; authenticate via `gcloud auth configure-docker`. |
| **Cloud Run Scale-to-Zero** | Zero idle infrastructure cost | Perfect for variable web traffic and microservices. |
| **Cloud Run Concurrency** | Multiple requests per container instance | Leverage FastAPI `async def` routes to handle up to 80 concurrent calls. |
| **Traffic Splitting** | Canary deployments | Roll out new model revisions to 10% traffic before full release. |

---

## Part 4: Deploying ML APIs, RAG Systems, and AI Chatbots

### 1. The Decoupled Model Architecture Pattern
A critical mistake in machine learning deployment is baking multi-gigabyte model weights directly inside the Docker image:

$$\text{Anti-Pattern}: \text{Docker Image} = \text{Application Code} + \text{Model Weights (5GB)}$$

* **Why Baking Weights Fails**:
  * Massive container images (5GB+) lead to slow CI/CD builds, prolonged push/pull times, and severe Cloud Run cold start delays.
  * Every model fine-tune or retraining run requires rebuilding, re-scanning, and redeploying the entire Docker image.
* **The Modern Decoupled Solution**:
  Keep the Docker container lean (~300MB containing only code and runtime libraries). Store serialized model checkpoints (`model.pkl`, ONNX files, or transformer weights) in a versioned Cloud Storage bucket.

```
Docker Image (Artifact Registry)    ---> Application Code & FastAPI (~300MB)
Model Checkpoints (Cloud Storage)  ---> gs://bucket/models/classifier/v1/model.pkl
```

When Cloud Run starts:
1. The container boots rapidly due to its small image size.
2. The application reads `GCS_MODEL_PATH` from environment variables.
3. The model file is downloaded from GCS (or mounted via Cloud Storage FUSE) and loaded into RAM once during server startup.
4. Future model updates require only uploading new weights to GCS and updating the environment variable on Cloud Run—no image rebuild required:
   ```bash
   gcloud run services update rag-api \
       --set-env-vars=GCS_MODEL_PATH=gs://my-bucket/models/v2/model.pkl
   ```

---

### 2. Production Web Server Architecture: Gunicorn + Uvicorn Workers
Development commands like `python main.py` or `uvicorn main:app --reload` are single-process tools unsuitable for production. Production containers on Cloud Run should use **Gunicorn** as a process supervisor managing **Uvicorn worker classes**:

```dockerfile
CMD exec gunicorn \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --threads 8 \
    --timeout 0 \
    --worker-class uvicorn.workers.UvicornWorker \
    src.main:app
```

* **Why `--workers 1` with `--threads 8` for ML/AI Containers?**
  * Large ML models and vector embedding engines consume significant memory when loaded into RAM. Spawning 4 separate Gunicorn worker processes creates 4 independent copies of the model in memory, exhausting the container RAM allocation and triggering Out-Of-Memory (OOM) crashes.
  * Using a single master worker running 8 worker threads allows the process to hold **one shared copy** of the model weights in RAM while handling concurrent I/O-bound requests asynchronously.
* **Why `--timeout 0`?**
  * Cloud Run manages request timeouts externally (up to 60 minutes for streaming endpoints). Disabling Gunicorn internal 30-second timeout prevents the process supervisor from killing long-running LLM generation streams or complex RAG retrieval operations prematurely.
* **Why `exec`?**
  * Prefixing the launch command with `exec` replaces the shell process with the Gunicorn process, ensuring Gunicorn becomes Process ID 1 (`PID 1`) and receives operating system termination signals (`SIGTERM`) for graceful connection draining.

---

### 3. Deploying RAG Systems & Conversational AI
Production RAG systems—such as the LangGraph state machine with hybrid search developed in Week 3—require connecting stateless compute with external persistence:

* **Vector Database Architecture**:
  * While lightweight vector indexes like ChromaDB can run in-memory for local testing, production multi-instance Cloud Run services require a centralized vector database.
  * Google Cloud SQL for PostgreSQL with the `pgvector` extension provides a fully managed, enterprise-grade vector store. Cloud Run connects securely to Cloud SQL over private IP using Serverless VPC Access or the Cloud SQL Python Connector without exposing database ports to the public internet.
* **Workload Segmentation (Cloud Run Services vs. Cloud Run Jobs)**:
  * **Cloud Run Services**: Dedicated to interactive, low-latency, request-driven workloads (FastAPI endpoints, chat interfaces, streaming responses).
  * **Cloud Run Jobs**: Dedicated to run-to-completion, batch processing tasks. Heavy document parsing, PDF chunking, and text embedding indexing should run as Cloud Run Jobs, preventing background data processing from degrading live conversational inference.
* **Vertex AI & Gemini Integration**:
  * Cloud Run containers authenticate seamlessly to Google Vertex AI using the attached runtime service account. The application uses the Google Cloud client library to invoke foundational models (e.g., `gemini-1.5-pro` or `textembedding-gecko`) without managing manual API keys.

---

### 4. Step-by-Step Hands-On Deployment Commands
Follow this exact sequence to deploy a containerized AI service to Cloud Run:

```bash
# 1. Set environment variables
export PROJECT_ID="studentgator-prod"
export REGION="us-central1"
export BUCKET_NAME="${PROJECT_ID}-ml-assets"
export REPO_NAME="ai-repo"
export SERVICE_NAME="rag-service"

# 2. Configure project & enable APIs
gcloud config set project $PROJECT_ID
gcloud services enable \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    storage.googleapis.com \
    cloudbuild.googleapis.com

# 3. Create GCS bucket & upload model checkpoint
gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_NAME/
gsutil cp model.pkl gs://$BUCKET_NAME/models/v1.0.0/model.pkl

# 4. Create Artifact Registry repository & authenticate Docker
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \
    --location=$REGION
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# 5. Build & push container image (specifying linux/amd64 platform)
docker build --platform linux/amd64 -t \
    ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:v1.0.0 .
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:v1.0.0

# 6. Deploy service to Cloud Run
gcloud run deploy $SERVICE_NAME \
    --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:v1.0.0 \
    --region=$REGION \
    --allow-unauthenticated \
    --set-env-vars=GCS_MODEL_PATH=gs://${BUCKET_NAME}/models/v1.0.0/model.pkl \
    --memory=2Gi \
    --cpu=2 \
    --concurrency=80 \
    --min-instances=0 \
    --max-instances=10

# 7. Test live endpoint
export SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")
curl -X POST "${SERVICE_URL}/predict" \
    -H "Content-Type: application/json" \
    -d '{"query": "What are the rules for Form I-20 travel endorsement?"}'
```

---

## Part 5: Checkpoint #4 Mentorship Review & Architectural Defense

The following four questions represent the core architectural topics evaluated during your Checkpoint Meeting #4 review with mentor Mohammed Orabi:

### Question 1: Why deploy an AI/RAG inference API on Cloud Run instead of hosting it directly on a Compute Engine VM?
> **Answer**: Compute Engine operates on an unmanaged IaaS model: virtual machines run continuously, requiring engineering teams to manually configure OS security patching, scaling policies, load balancers, and SSL certificates. Compute Engine instances incur 24/7 infrastructure costs regardless of incoming traffic. Cloud Run provides a serverless, fully managed container platform that natively scales to zero when idle, eliminating idle compute expenses. Furthermore, Cloud Run automatically provisions HTTPS endpoints, handles request-based horizontal auto-scaling in seconds, and provides immutable revision tracking with instant traffic-splitting capabilities.

### Question 2: Why must machine learning model binaries and vector indices be decoupled from the Docker image in production deployments?
> **Answer**: Baking multi-gigabyte model weights directly into a container image creates bloated artifacts (5GB+), drastically slowing down CI/CD build pipelines, image registry uploads, and container cold start download times. Furthermore, any retrained model iteration forces a complete rebuild of the application image. Decoupling storage—by preserving code in Artifact Registry and hosting model binaries in Cloud Storage (GCS)—allows the container image to remain lean (~300MB). When a model is updated, the artifact is uploaded to GCS, and the Cloud Run service simply points to the new URI via environment variable adjustments without requiring image rebuilding.

### Question 3: What is the purpose of Gunicorn with Uvicorn workers in an AI container, and why is `--workers 1` paired with multiple `--threads`?
> **Answer**: Raw Uvicorn or development scripts (`python app.py`) lack multi-threaded process supervision, connection draining, and robust signal handling. Gunicorn acts as a master process manager that coordinates worker lifecycles. In AI applications, loading large deep learning weights or vector models consumes significant RAM. Setting multiple worker processes (such as 4 workers) forces the operating system to duplicate the model across 4 independent process memory spaces, causing Out-Of-Memory container termination. Using `--workers 1` with multiple `--threads` allows the container to retain a single shared copy of the model in RAM while handling concurrent I/O requests asynchronously via Uvicorn event loop.

### Question 4: How do Google Cloud IAM roles and Service Accounts eliminate hardcoded secrets in production AI services?
> **Answer**: Hardcoding API keys or database service account JSON files inside code repositories or Docker images introduces severe security risks. Google Cloud implements metadata-based identity inheritance. When a Cloud Run service or Compute Engine VM is assigned a dedicated Service Account, the cloud runtime automatically provisions short-lived OAuth access tokens through the local instance metadata server. Application client libraries (such as Google Cloud Storage or Vertex AI SDKs) detect these Application Default Credentials (ADC) automatically, eliminating the need to store long-lived credentials in configuration files.
