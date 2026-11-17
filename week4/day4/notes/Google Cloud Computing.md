# Google Cloud Computing

Day 4 scope: GCP fundamentals (Compute Engine, IAM, Cloud Storage, Artifact Registry, Cloud Run) and deploying a containerized ML API on Cloud Run.

---

## 1. Cloud Computing Fundamentals

### What Is Cloud Computing?
Renting IT resources (compute, storage, networking) over the internet and paying for what we use, instead of buying and babysitting physical servers.

**The 5 core traits (NIST):**
1. **On-demand self-service:** We spin things up from the console, CLI, or API. No human needs to approve it.
2. **Broad network access:** Reachable over normal protocols (HTTPS, SSH, REST) from anywhere.
3. **Resource pooling:** Many customers share the same hardware, and we don't care where it physically sits.
4. **Rapid elasticity:** Scales up when traffic spikes and back down when it drops.
5. **Measured service:** Billed on actual usage (CPU-seconds, GB-hours, egress). Stop using it, stop paying.

* **Scalability vs. Elasticity:** Scalability is how high the ceiling is (an 8-lane highway). Elasticity is growing *and shrinking* automatically with load (a rubber band).
* **CapEx vs. OpEx:** CapEx = buy hardware upfront. OpEx = pay as we consume.
* **StudentGator angle:** The RAG service gets hammered near university application deadlines and is almost idle otherwise. Pay-per-use plus scale-to-zero means no money wasted on idle servers.

---

## 2. GCP Resource Hierarchy

Everything sits in a 4-level tree. Policies and IAM permissions flow **downward**. If a parent grants access, a child can't take it away.

```
[ Organization ]   company root (Google Workspace / Cloud Identity)
       |
   [ Folders ]     optional grouping by team or environment (dev, prod)
       |
   [ Projects ]    the main boundary: billing, APIs, IAM
       |
  [ Resources ]    VMs, buckets, Cloud Run services, etc.
```

* **Organization:** Top node. Company-wide policies live here.
* **Folders:** Group projects by department or environment so we don't repeat the same policy everywhere.
* **Projects:** The unit we actually work in. Every resource belongs to exactly one project.
  * `Project ID`: globally unique, can't change. This is what we use in `gcloud` and code.
  * `Project Name`: just a label, can change.
  * `Project Number`: assigned by Google, used internally.
* **Resources:** The real stuff (VMs, buckets, containers).
* **StudentGator setup:** `studentgator-dev` for experiments and `studentgator-prod` for the locked-down deployment with its own budget alerts. One project per environment.

---

## 3. IAM (Identity and Access Management)

IAM answers one question: **who** can do **what** on **which resource**.

```
Principal (user / service account)  --granted-->  Role (permissions)  --bound on-->  Resource
```

* **Principal (who):** A Google account, a group, a domain, or a **Service Account**. A service account is an identity for an app (like a Cloud Run container), not a person.
* **Role (what):** A bundle of permissions such as `storage.objects.get` or `compute.instances.start`.
* Policies inherit downward, and deny rules are checked before allow rules.

### The 3 role types

| Type | What it is | When we use it | Catch |
|---|---|---|---|
| **Basic** | `Viewer`, `Editor`, `Owner` across the whole project | Solo sandbox only | Way too broad for production. Breaks least privilege. |
| **Predefined** | Per-service roles from Google (`roles/storage.objectViewer`, `roles/run.invoker`) | The production default | Can be bound at org, folder, project, or resource level. |
| **Custom** | We pick the exact permissions | When predefined gives too much | High upkeep. Can't be created at folder level (only project or org). |

* **Least privilege:** Never give `Editor` or `Owner` to an app. The RAG API on Cloud Run runs under its own service account with just what it needs, e.g. `roles/storage.objectViewer` to read model files plus logging.
* **Why service accounts matter:** We attach one to the Cloud Run service and Google hands the app short-lived credentials automatically (Application Default Credentials). No API keys or JSON key files in the repo or the image.

---

## 4. Compute Engine (IaaS VMs)

Compute Engine rents us virtual machines: virtual CPUs, RAM, and disks, with full control of the OS.

* **VM lifecycle:** create, start, stop, change machine type, delete.
* **Machine families (just the idea):**
  * General purpose (E2, N2): web servers, small services.
  * Compute optimized (C2): CPU-heavy work.
  * Memory optimized (M-series): huge in-memory data.
  * Accelerator optimized (A2, A3, G2): GPUs for training and heavy inference.
* **Spot VMs:** 60-91% cheaper, but Google can take them back any time. Fine for batch jobs and training with checkpoints. Never for a live API.
* **IAM on VMs:**
  * `roles/compute.instanceAdmin`: full control of VMs.
  * Custom example: an `instanceOperator` role that can start and stop VMs but not reconfigure or delete them.
* **Connecting:** `gcloud compute ssh my-vm --zone=us-central1-a`. OS Login is the better way since SSH access is tied to IAM identity.

### Compute Engine vs. Cloud Run

| | Compute Engine | Cloud Run |
|---|---|---|
| Model | IaaS, we manage the OS, patches, scaling, SSL | Serverless, Google manages all of that |
| Cost | Pays 24/7 while the VM is running | Scales to zero, pay per request |
| Use when | Need full OS control, GPUs, or long-running stateful stuff | Stateless HTTP APIs like our FastAPI RAG service |

---

## 5. Cloud Storage (GCS)

Object storage for unstructured files: datasets, model checkpoints, documents.

* **Buckets and objects:** A bucket is a globally uniquely named container. Files inside are objects, addressed like `gs://bucket-name/models/v1/model.pkl`. The "folders" are just slashes in the name.
* **Storage classes** (only the access pattern and minimum storage time change):

| Class | Access pattern | Min storage |
|---|---|---|
| Standard | Frequent (live model weights, active data) | none |
| Nearline | Less than once a month (backups) | 30 days |
| Coldline | Less than once a quarter | 90 days |
| Archive | Less than once a year (compliance records) | 365 days |

* **Lifecycle rules:** Automatically move old objects to a cheaper class or delete them after N days. Keeps the bill from creeping up.
* **Signed URLs:** A temporary link to one object. Anyone with it can read it until it expires, no Google account needed.
* **Why we care:** Model weights go here, not inside the Docker image (see section 8).
* **Cloud Storage vs. Persistent Disk:** GCS is object storage reached over an API and shared by anything. Persistent Disk is a block disk attached to a VM, like its hard drive. For Cloud Run we use GCS since containers have no persistent disk.

---

## 6. Artifact Registry

The place to store our Docker images on GCP. It replaces the old Container Registry (`gcr.io`).

* **Image path format:**
  `LOCATION-docker.pkg.dev/PROJECT_ID/REPOSITORY/IMAGE:TAG`
* **Workflow:** create a repo, let Docker authenticate, tag the image, push.

```bash
# create a Docker repository
gcloud artifacts repositories create ml-repo \
    --repository-format=docker \
    --location=us-central1

# let local Docker authenticate to this registry
gcloud auth configure-docker us-central1-docker.pkg.dev

# tag and push
docker tag my-api:v1 us-central1-docker.pkg.dev/my-project/ml-repo/my-api:v1.0.0
docker push us-central1-docker.pkg.dev/my-project/ml-repo/my-api:v1.0.0
```

* Use specific version tags (`v1.0.0`), not just `latest`, so we always know what is running and can roll back.
* The CI/CD pipeline from Day 3 pushes to this registry, and Cloud Run pulls from it.
* Build with `--platform linux/amd64`. Building on an ARM laptop and deploying to Cloud Run is a classic cause of "container failed to start".

---

## 7. Cloud Run

Runs a stateless Docker container and gives us an HTTPS URL. No servers to manage.

```
HTTP requests --> Cloud Run HTTPS endpoint
                      |
            +---------+---------+
            |                   |
       [ Container 1 ]    [ Container 2 ]    autoscaling 0 to N
       (concurrency 80)   (concurrency 80)
```

* **Stateless:** Local files vanish when the instance dies. Anything that must survive goes to GCS or a database.
* **Scale to zero:** No traffic means no instances and no cost.
* **Cold start:** The first request after idle has to pull the image and boot the app. For an ML app loading a big model this hurts. `--min-instances=1` keeps one instance warm, which costs money but removes the delay. This is a latency vs. cost trade-off.
* **Concurrency:** One instance handles many requests at once (default 80). That fits async FastAPI well since most of the time is spent waiting on I/O (LLM calls, DB).
* **Revisions and traffic splitting:** Every deploy creates an immutable revision. We can send 10% of traffic to the new one first (canary) and roll back instantly if it breaks.
* **Services vs. Jobs:** Services answer requests (the chat API). Jobs run to completion (batch embedding of documents) so heavy indexing doesn't slow down live users.

---

## 8. Deploying a Containerized ML API on Cloud Run

### Don't bake the model into the image
Putting a 5 GB model inside the Docker image means slow builds, slow pushes, slow cold starts, and a full rebuild every time we retrain.

* **Better:** Keep the image small (code and libraries only) and store the model in a GCS bucket.
* On startup the app reads `GCS_MODEL_PATH` from an env var, downloads the model once, and keeps it in memory.
* New model version means upload to GCS and update the env var. No image rebuild.

```bash
gcloud run services update rag-service \
    --set-env-vars=GCS_MODEL_PATH=gs://my-bucket/models/v2/model.pkl
```

### Gunicorn with Uvicorn workers
`uvicorn --reload` is for development. In the container we use Gunicorn as the process manager running Uvicorn workers.

```dockerfile
CMD exec gunicorn \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --threads 8 \
    --timeout 0 \
    --worker-class uvicorn.workers.UvicornWorker \
    src.main:app
```

* `--workers 1`: each worker loads its own copy of the model. 4 workers means 4 copies in RAM and an out-of-memory crash.
* `--threads 8`: concurrency without copying the model.
* `--timeout 0`: Cloud Run enforces its own request timeout, and Gunicorn's default 30s would kill long LLM streams.
* `exec`: makes Gunicorn PID 1 so it receives SIGTERM and shuts down gracefully.
* Cloud Run gives the port through the `$PORT` env var, so the app must listen on it.

### RAG-specific notes
* **Vector DB:** Chroma in memory is fine locally, but with multiple Cloud Run instances each would have its own copy. Production needs a shared store like Cloud SQL Postgres with `pgvector`.
* **LLM access:** Calling Vertex AI / Gemini works through the attached service account, so no API keys. Other secrets go in Secret Manager (the Day 3 secrets idea).

### Deployment steps

```bash
# 1. Variables
export PROJECT_ID="studentgator-prod"
export REGION="us-central1"
export REPO_NAME="ai-repo"
export SERVICE_NAME="rag-service"

# 2. Select project and enable the APIs we need
gcloud config set project $PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com storage.googleapis.com

# 3. Upload the model to a bucket
gcloud storage buckets create gs://${PROJECT_ID}-ml-assets --location=$REGION
gcloud storage cp model.pkl gs://${PROJECT_ID}-ml-assets/models/v1.0.0/model.pkl

# 4. Registry and Docker auth
gcloud artifacts repositories create $REPO_NAME --repository-format=docker --location=$REGION
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# 5. Build and push
IMAGE=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:v1.0.0
docker build --platform linux/amd64 -t $IMAGE .
docker push $IMAGE

# 6. Deploy
gcloud run deploy $SERVICE_NAME \
    --image=$IMAGE \
    --region=$REGION \
    --set-env-vars=GCS_MODEL_PATH=gs://${PROJECT_ID}-ml-assets/models/v1.0.0/model.pkl \
    --memory=2Gi --cpu=2 \
    --concurrency=80 \
    --min-instances=0 --max-instances=10

# 7. Test
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")
curl -X POST "$SERVICE_URL/predict" -H "Content-Type: application/json" \
    -d '{"query": "What are the rules for Form I-20 travel endorsement?"}'
```

* `--allow-unauthenticated` makes the API public. Only use it if it should be public; otherwise grant `roles/run.invoker` to specific callers.
* `--max-instances` is a cost safety cap.
* Set a billing budget with alerts (50%, 90%, 100%) so a mistake doesn't turn into a surprise bill.

---

## 9. Interview Talking Points

* **Why Cloud Run instead of a VM for the RAG API?**
  A VM runs 24/7 and we'd have to handle patching, scaling, load balancing, and SSL ourselves. Cloud Run scales to zero, gives HTTPS out of the box, scales on requests, and has revisions with traffic splitting. Since our traffic is spiky around deadlines, it fits well. We'd pick a VM only for GPUs or full OS control.

* **Why keep the model out of the Docker image?**
  A multi-GB image slows CI/CD and cold starts, and every retrain forces a rebuild. With the image at around 300 MB and the model in GCS, a new model is just a new file and an env var change.

* **Why Gunicorn with `--workers 1 --threads 8`?**
  Gunicorn supervises the process and handles shutdown signals. Each worker process loads its own copy of the model, so multiple workers risk running out of memory. One worker with threads shares one copy and still handles concurrent requests.

* **How do service accounts avoid hardcoded secrets?**
  The service gets an attached service account, and the runtime gives the app short-lived tokens through the metadata server. Google client libraries pick them up automatically, so there are no key files in the repo or image. We grant only predefined roles scoped to what the app needs.
