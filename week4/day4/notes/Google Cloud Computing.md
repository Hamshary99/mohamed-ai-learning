# Google Cloud Computing

## 1. Cloud Computing Fundamentals

### What Is Cloud Computing?
Renting IT resources (compute, storage, and networking) over the internet with pay-as-you-go pricing instead of buying physical servers.

**The 5 Core Traits (NIST):**
1. **On-demand self-service:** Spin up resources via web console, CLI, or API with zero human intervention.
2. **Broad network access:** Accessible over standard internet protocols (HTTPS, SSH, REST) from anywhere.
3. **Resource pooling:** Multi-tenant physical hardware dynamically pooled across customers (economies of scale; physical location is abstracted).
4. **Rapid elasticity:** Automatically scales up with demand surges and scales back down when traffic drops.
5. **Measured service:** Pay strictly for what is consumed or reserved (CPU-seconds, GB-hours, network egress); stop using it, stop paying.

* **Scalability vs. Elasticity:** Scalability is capacity ceiling (building an 8-lane highway); Elasticity is dynamic adaptation that expands *and shrinks* in real time to match load (a rubber band).
* **CapEx vs. OpEx:** CapEx is buying physical hardware upfront; OpEx is operational pay-as-you-go for actual consumption.
* **StudentGator relevance:** Our RAG service auto-scales during university application deadlines and scales to zero off-peak to eliminate idle server costs.

---

## 2. GCP Resource Hierarchy

Resources are organized into a 4-level downward hierarchy. Policies and IAM permissions **inherit downward** (a permission granted at a parent level cannot be revoked by a child level).

```
[ Organization Node ]   (Company root via Google Workspace / Cloud Identity)
        │
    [ Folders ]         (Optional: groups projects by dept or env: dev, prod)
        │
   [ Projects ]         (Core boundary: billing, APIs, IAM access, Resource Manager)
        │
  [ Resources ]         (VMs, Cloud Storage buckets, Cloud Run services, BigQuery)
```

* **Organization Node:** Topmost root representing the whole company. Centralizes governance via *Organization Policy Administrator* and restricts project creation/spending via *Project Creator*.
* **Folders:** Groups projects by department (`Engineering`, `Data`) or environment (`dev`, `prod`). Eliminates duplicate policy management; child projects inherit folder permissions automatically.
* **Projects:** Base operational unit. Every resource belongs to **exactly one** project.
  * `Project ID`: Globally unique, immutable string. Used in `gcloud` and code.
  * `Project Name`: User-defined display name, mutable, does not need to be unique.
  * `Project Number`: Google-assigned immutable integer (used internally by GCP).
  * *Resource Manager API:* Programmatically manages project lifecycle (create, delete, recover).
* **Resources:** The actual assets (Compute Engine VMs, Cloud Storage buckets, Cloud Run containers).
* **StudentGator setup:** Separate projects for `studentgator-dev` (sandbox experiments) and `studentgator-prod` (locked-down deployment with dedicated billing alerts).

---

## 3. Identity and Access Management (IAM)

IAM governs access: **Who** (principal) can do **what** (role) on **which resource**.

```
[ Principal (User / SA) ]  ──(granted)──>  [ Role (Permissions) ]  ──(bound on)──>  [ Resource Scope ]
```

* **Principal (Who):** Identified by email. Google account, Google group, Cloud Identity domain, or **Service Account** (an identity for applications/containers, e.g. Cloud Run, not humans).
* **Role (What):** A bundle of permissions (e.g. `storage.objects.get`, `compute.instances.start`).
* **Inheritance & Deny Rules:** Policies inherit downward. **Deny rules are evaluated before Allow rules.**

### The 3 Role Types

| Type | Scope & Granularity | When to Use | Production Caveat |
|---|---|---|---|
| **Basic (Primitive)** | Project-wide: `Viewer`, `Editor`, `Owner`, `Billing Administrator` | Solo sandbox / learning | **Too broad for production.** Violates least privilege across all services in the project. |
| **Predefined** | Service-specific permission bundles managed by Google (e.g. `roles/compute.instanceAdmin`, `roles/storage.objectViewer`) | Production standard | Bound at Org, Folder, Project, or Resource level. |
| **Custom** | User-curated list of exact permissions | Strict **least privilege** | High upkeep; **cannot be applied at Folder level** (only Project or Org). |

* **StudentGator rule (Least Privilege):** Never grant `Editor`/`Owner`. Our RAG API on Cloud Run runs under a dedicated Service Account with minimal predefined roles (`roles/storage.objectViewer` to load vector embeddings, plus logging).

---

## 4. Compute Engine (IaaS Virtual Machines)

Compute Engine is Google Cloud's Infrastructure as a Service (IaaS) product providing **Virtual Machines (VMs)** on demand.

* **What It Provides:** Raw virtualized hardware (vCPUs, RAM, Persistent Disks) with full operating system control (Linux/Windows).
* **Core VM Lifecycle:** Creating, starting, stopping, modifying machine configuration, and deleting instances.
* **IAM Integration:**
  * *Predefined Role:* `roles/compute.instanceAdmin` grants full lifecycle management of VM instances in a project or folder.
  * *Custom Role Use Case:* An `instanceOperator` role that allows starting and stopping instances for maintenance without permission to reconfigure or delete them.
* **Compute Engine vs. Cloud Run (StudentGator trade-off):**
  * *Compute Engine:* Full OS control, custom GPU acceleration, or running persistent stateful Docker daemons.
  * *Cloud Run (serverless):* Deploying our stateless FastAPI RAG containers with zero server maintenance and auto-scaling to zero.
