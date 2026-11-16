# What Is a Container? (Kernel, Layers, Namespaces, cgroups)

## 1. What Is It & Why Does It Exist?

**A container is just a normal process on the host, wrapped in isolation so it believes it owns the machine.** It is not a tiny VM. There is no second operating system booting inside it.

### The problem with VMs

One issue with a VM is the number of steps to set it up. The VM architecture has several layers:

```
Application
Dependencies
Guest OS            <- a full OS (its own kernel + user space) per VM
Virtual hardware
```

Every VM boots a whole OS, which costs RAM, disk, boot time and an OS license/patch cycle per VM.

### Kernel vs user space (the key idea)

An OS is really two parts:

| Part | What it is | Runs in |
|---|---|---|
| **Kernel** | The core. Talks to the hardware (physical or virtual), manages memory, CPU time, files, network. | Kernel mode (privileged CPU mode) |
| **User space** | Everything else: shell, libraries, package manager, `/bin`, `/etc`, plus our app and its dependencies. | User mode |

Our goal: **one kernel shared by all containers**. Each container only carries its own user space (the files, libraries and dependencies it needs) and uses the host's kernel to do the real work.

- Containers start in milliseconds (no OS boot) and use far less memory.
- No full guest OS per app, so less to patch and less to license.

> Mental model: an apartment building. The kernel is the building (plumbing, electricity, structure) shared by everyone. Each container is an apartment with its own furniture (user space), a locked door (namespaces) and its own utility meter with a cap (cgroups).

> Common trap: containers do not remove user space. Each container still has its own. What gets removed is the duplicated **kernel**. Sharing a kernel also makes isolation **weaker** than a VM, not stronger (see trade-offs).

---

## 2. How It Works (The Mechanics)

### 2.1 Images and layers (how we avoid copying a huge OS every time)

- A **base image** (parent) is a template that holds a lot of common stuff (for example a web/Ubuntu template).
- The parent can have several child images. Each child only carries what it exactly needs on top, which is the **differencing** idea: store only the difference instead of copying the whole large parent.
- **Every layer is a differencing file.** Each layer is a modification of the layer below it. The group of layers together is the **image**.
- Layers are **read-only and shared**. Ten containers from the same image reuse the same layers on disk.
- When a container runs, Docker adds one thin **writable layer** on top (copy-on-write: changing a file copies it up into this layer).
- Snapshot in case we mess up our differencing image: with containers there is no manual snapshot. Image layers are never modified, so if a container gets messed up we delete it and start a fresh one from the same image.

```
 [ writable layer ]   <- the running container (disappears when container is removed)
 [ app code       ]   \
 [ dependencies   ]    }  read-only image layers (shared)
 [ ubuntu user space ]/
 -------------------
 [ host kernel ]       <- shared by every container
```

### 2.2 Matching the OS to the kernel

The image's user space must speak the language of the host kernel (system calls). So:

- Ubuntu, Red Hat, Fedora, Alpine are all **Linux distributions**: same Linux kernel, different user space. A container "being Ubuntu" only means it carries Ubuntu's files and libraries.
- Common trap: the kernel shared across distros is the **Linux kernel**, not a Windows one. Windows has its own (NT) kernel, so Windows containers need a Windows host.
- Docker Desktop on Windows/Mac runs a small hidden **Linux VM** (WSL2 on Windows) so Linux containers still have a Linux kernel to share.

### 2.3 Namespaces: what a container can SEE

A **namespace** gives a process its own private view of a system resource. Same kernel, different "window".

| Namespace | What it isolates | Effect inside a container |
|---|---|---|
| `pid` | Process IDs | Container sees only its own processes; its first process is PID 1 |
| `net` | Network stack | Own IP, ports, interfaces |
| `mnt` | Mount points / filesystem | Sees only its image's filesystem as `/` |
| `uts` | Hostname | Own hostname |
| `ipc` | Shared memory, queues | Cannot talk to other containers' memory |
| `user` | User/group IDs | `root` inside can map to a normal user outside |
| `cgroup` | cgroup view | Sees its own cgroup as the root |

Analogy: namespaces are blinders. The process cannot see what it is not allowed to see, but nothing stops it from using as much CPU or RAM as it likes. That is where cgroups come in.

### 2.4 cgroups (control groups): what a container can USE

A **cgroup** is a kernel feature that groups processes and **limits and measures** their resources: CPU, memory, disk I/O, and number of processes.

- Without cgroups, one greedy container could eat all the RAM and starve the rest.
- If a container goes over its memory limit, the kernel kills it (OOM kill).
- Docker flags map directly to cgroup settings:

```bash
# Limit this container to half a CPU core and 256 MB of RAM
docker run --rm -it --cpus=0.5 --memory=256m ubuntu bash
```

Where to see it (Linux):

```bash
# Inside the container: its own cgroup limits (cgroup v2)
cat /sys/fs/cgroup/memory.max     # 268435456 = 256 MB
cat /proc/1/cgroup                # which cgroup PID 1 belongs to

# On the host: the same process lives in a docker-<id>.scope cgroup
cat /proc/<host_pid>/cgroup
```

> Namespaces = what a process can see. cgroups = what a process can use. Together with image layers, that is what a container is.

### 2.5 PID 1 (the "proc 1" part)

- Inside a container's `pid` namespace, the **first process started is PID 1**. That is the command from `CMD`/`ENTRYPOINT`.
- The same process also has a normal, large PID on the host. It is one process seen through two views.

```bash
docker run --rm -it ubuntu bash
ps -ef                 # inside: bash is PID 1

docker top <container> # on the host: the same bash with a real host PID
```

Why PID 1 matters in production:
- **When PID 1 exits, the container stops.** The container lives exactly as long as its main process.
- PID 1 gets no default signal handling, so `docker stop` (SIGTERM) can be ignored by an app that does not handle it, and Docker waits then force-kills it. Use exec form `CMD ["python", "app.py"]` so the app itself is PID 1, and consider `docker run --init`.
- PID 1 is also responsible for cleaning up orphaned child processes (zombies).

### 2.6 Build / pull / run lifecycle

Image and container flow (build creates the image, run creates the container, pull/push move images to and from a registry):

```
Dockerfile --build--> image --run--> container
                        ^   |
                  pull  |   |  push
                        |   v
                      registry (Docker Hub, Artifact Registry)
```

VM equivalent, for comparison:

```
running VM --export--> VM template --import--> new running VM
```

Note: a VM template is basically one big disk image. The "stack of independent layers where each modifies the last" idea belongs to **container images**, not VM templates.

---

## 3. Trade-Offs & Comparisons (When to Use What)

| | Virtual Machine | Container |
|---|---|---|
| Isolation | Strong (own kernel, hardware-level) | Weaker (shared host kernel) |
| Startup | Seconds to minutes | Milliseconds to seconds |
| Size | GBs | MBs to hundreds of MBs |
| OS | Any OS per VM | Must match host kernel type (Linux containers need Linux kernel) |
| Density | Few per host | Many per host |
| Best for | Running different OSes, strong tenant separation | Packaging and scaling one app and its dependencies |

### 3.1 The "Shared Security" Trade-Off (Management vs. Blast Radius)

The shared kernel creates a two-sided security reality that explains why tutorials highlight it while security audits treat it with caution:

1. **Where sharing helps (Centralized Defense & Minimal Surface):**
   - **One kernel to patch:** Updating and hardening the host kernel (plus applying AppArmor, SELinux, and seccomp filters) automatically protects all running containers. There is no need to patch and audit 30 distinct guest OS kernels.
   - **Reduced attack surface:** Unlike full VMs that run background daemons, cron jobs, and SSH servers by default, minimal container images (Alpine, distroless) strip out compilers, shells, and network tools (`curl`, `wget`), leaving an attacker with fewer tools if they gain code execution.
   - **Standardized baselines:** Securing a single base image ensures that every child image inherits those verified security controls.

2. **Where sharing is dangerous (The Blast Radius):**
   - **Container breakout:** Because containers share the host kernel, any unpatched kernel vulnerability allows an attacker to escape container isolation and compromise the entire host and all neighboring containers. In a VM, hypervisor escapes are far rarer.
   - **Shared crash domain:** If a container triggers a kernel panic or exhausts kernel-level resources (file descriptors, routing tables), the entire host crashes, taking down all services.
   - **User-space attacks remain:** App-level bugs (vulnerable dependencies, prompt injection, SQL injection) run in user space inside the container. The shared kernel provides no protection against them.

Edge cases:
- A kernel vulnerability can affect every container on the host, since they share it. Untrusted multi-tenant code should get a VM or lightweight microVM (e.g., Firecracker, gVisor).
- Containers are not security boundaries by default: running as `root` inside a container with no `user` namespace mapping is risky.
- Container filesystem is temporary. Data needs **volumes** (covered later).

---

## 4. Production Code Reference

```bash
# Run an API container with explicit resource limits
docker run -d --name api \
  --memory=512m \
  --cpus=1.0 \
  --init \
  my-fastapi-image

# See the isolation (namespaces) from the host
docker inspect -f '{{.State.Pid}}' api   # host PID of the container's PID 1
sudo ls -l /proc/<that_pid>/ns           # one entry per namespace

# See live usage measured via cgroups
docker stats api
```

---

## 5. Whiteboard / Interview Talking Points

- **Container vs VM:** A VM virtualizes hardware and runs a full OS with its own kernel. A container virtualizes the OS: it is an isolated process sharing the host kernel, which is why it is smaller and faster but less isolated.
- **What makes a container:** Namespaces limit what a process can see, cgroups limit what it can use, and image layers give it its filesystem. There is no special "container" object in the kernel.
- **Why we care in production:** For StudentGator, packaging the FastAPI app with its exact dependencies means it runs the same on our laptop and on Cloud Run, and cgroup limits stop one noisy service from starving the rest during deadline traffic spikes.

---

## Common Misconceptions (Quick Review)

| Misconception | Reality |
|---|---|
| Containers remove user mode | Each container keeps its own user space; only the kernel is shared |
| Containers are more secure than VMs | Shared kernel means weaker isolation than VMs |
| All containers share one security setup | They share the kernel, so a kernel bug affects all of them; per-container settings (user, capabilities) still differ |
| Licensing disappears | Linux containers need no guest OS per app; Windows containers still have licensing rules |
| Snapshots protect a broken container | Image layers are read-only; recreate the container from the image |
| One kernel is shared across all Windows OS | Ubuntu, Red Hat and Fedora share the Linux kernel; Windows has its own |
| Container --build--> image --pull--> | Dockerfile builds the image, the image runs as a container |
| VM templates are stacks of layers | Layers are a container-image concept; a VM template is basically one disk image |
<details>
<summary>Raw notes from the video (unedited)</summary>

```
container archetecture

One issue with vm is the amount of steps to set it up
The vm architecrue contains severl layers
Virtual Hawrdware machine
OS
dependencies
Application

our issue with the OS

OS goes into both kernel mode to verify the hw and user mode 
user mode isnt relevant here, only kernel whether virtual or physical hw
so our goal is to get one kernel for all the containers to be shared and since we knocked out the user mode we protected ourselves from several service attacks and from the licensing of several os, so all vms might share the same security measures, licensing and so

but how?

with base disk or base image that provides a template (for example web template) that holds tons of stuff? 
The parent image can have several child images and each child image carries what we exactly need, which is called differencing image. Instead of copying the whole large parent image
Then we take a snapshot of the child to make another vm copy in case if we fuck up our vm differncing image

We also somehow can make the os specs to meet the kernel specs? now we can fill only the needed os specs so we move on to the application and its deps.

Kernel is usually the same across all windows os and then we fill what we need from the user mode? to the vm and now we get either ubuntu, red hat, fedora or whatever

every layer is a differencial file, the group of layers reach us to installation that image, huh?

at the end we get container image that turns to a container

container --- build ---> image-> |
|                               | 
|---------pull------------------| 


running vm --- export ---> vm template |
                                       | 
        |---------import-------------------| 

vm template contains layers of file systems that are independent and each layer is a modification for the last layer, differncing.
```

</details>
