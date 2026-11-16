# CI/CD & GitHub Actions

## 1. What Is It & Why Does It Exist?

**CI/CD** is a set of automatic checks and delivery steps that run between `git push` and production.

**Problems it solves:**
* "Works on my laptop": my code passes locally only because of a package or file that is missing from `requirements.txt`. It breaks once merged.
* Manual releases: running tests, building the image and deploying by hand is slow and easy to get wrong.

**Mental model:** a factory line with an inspector. Every new part (`git push`) is tested on a clean bench (CI). If it passes, it gets packaged (Docker image) and checked in a staging area (CD) before it reaches customers.

### CI, Continuous Delivery, Continuous Deployment

| Term | What it does | Last step to production |
|---|---|---|
| **CI** (Continuous Integration) | On every push, lint and test the code in an isolated, fresh cloud machine | none, only validates |
| **Continuous Delivery** | CI + build the image + verify it in a production-like environment (Docker/staging) | a human approves |
| **Continuous Deployment** | Same as Delivery | automatic, no human |

Note: CD means Delivery or Deployment, not "Continuous Development" as I first wrote.

```
git push --> [CI: fresh VM -> install -> lint -> test]
         --> [CD: build image -> verify in staging -> push to registry]
         --> approve (Delivery)  or  auto (Deployment) --> PRODUCTION
```

Why CI runs on a fresh machine: it can't rely on files that only exist on my laptop, so hidden dependencies show up immediately.

### GitHub Actions

* GitHub = cloud platform for Git and collaboration.
* GitHub Actions = workflows that run automatically when a **trigger** (event) happens.
* It is more than CI/CD: any GitHub event works, for example opening an Issue can auto-label it.

**Why it matters for StudentGator:** before application deadlines the RAG service gets many small fixes. Every one of them is linted and tested automatically before it can reach students.

---

## 2. How It Works (The Mechanics)

### 2.1 Anatomy of a workflow

A workflow is a `.yml` file in `.github/workflows/` (plural `workflows`, otherwise GitHub ignores it).

| Part | Keyword | Meaning |
|---|---|---|
| Workflow | the `.yml` file | the whole pipeline |
| Trigger | `on:` | event that starts it |
| Job | `jobs:` | group of steps, gets its own fresh VM |
| Runner | `runs-on:` | the VM, usually `ubuntu-latest` |
| Step | `steps:` | one action or command |

```yaml
name: First Workflow
on:
  push:
    branches: [main]
jobs:
  example-job:
    runs-on: ubuntu-latest
    steps:
      - run: echo "no repo files yet"        # VM is empty here
      - uses: actions/checkout@v7.0.1        # downloads the repo onto the VM
      - run: ls -la                          # now it lists the repo files
```

The order matters: the runner starts blank, so `checkout` is almost always step 1.

### 2.2 Jobs vs steps (parallel vs sequential)

* **Jobs run in parallel by default**, each on its own VM (files are not shared between jobs).
* **`needs:`** makes a job wait for another job to pass. (In the video it sounds like "meets".)
* **Steps run top to bottom** on the same VM and share files. If one fails, the rest are skipped.

**Why build/test/deploy must not run side by side:** if they all start together I can't tell where the failure began, and `deploy` might ship code that `test` is about to reject. With `needs:` the pipeline stops at the failing job, the later jobs show as *skipped*, and the root cause is obvious. It also saves runner minutes.

```
        [build]
        /     \          both have needs: build -> run in parallel
    [test]   [ingest]
        \     /          needs: [test, ingest] -> waits for both
        [deploy]
```

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7.0.1
      - run: python -m py_compile studentgator-rag-service/src/*.py
  test:
    needs: build
    runs-on: ubuntu-latest
    steps: [...]
  ingest:
    needs: build
    runs-on: ubuntu-latest
    steps: [...]
  deploy:
    needs: [test, ingest]
    runs-on: ubuntu-latest
    steps: [...]
```

### 2.3 Triggers

**Webhook events (react to GitHub activity).** GitHub notifies the workflow the moment the event happens, so nothing has to poll.

```yaml
on:
  push:
    branches: [main, "release/**"]
    paths: ["studentgator-rag-service/**"]   # only when this folder changed
  pull_request:                              # note: pull_request, not "pull"
    branches: [main]
    types: [opened, synchronize, reopened]   # these are the defaults
  issues:
    types: [opened]
```

* `branches` / `branches-ignore`: which branches.
* `paths` / `paths-ignore`: skip runs when only notes or docs changed.
* `types`: sub-events of one event.

**Scheduled (`schedule`, cron).** Runs on a timer, always in **UTC**. Cron has 5 fields:

```
┌ minute (0-59)
│ ┌ hour (0-23, UTC)
│ │ ┌ day of month (1-31)
│ │ │ ┌ month (1-12)
│ │ │ │ ┌ day of week (0-6, 0 = Sunday)
* * * * *
```

| Symbol | Meaning | Example |
|---|---|---|
| `*` | every value | `* * * * *` every minute |
| `/` | every N | `*/15 * * * *` every 15 min |
| `-` | range | `0 9 * * 1-5` 09:00 Mon-Fri |
| `,` | list | `0 8,20 * * *` 08:00 and 20:00 |

| Cron | Runs |
|---|---|
| `'*/5 * * * *'` | every 5 min (minimum allowed) |
| `'0 0 * * *'` | daily at midnight UTC |
| `'30 2 * * *'` | daily 02:30 UTC (04:30 Cairo), good for nightly RAG re-ingestion |
| `'0 0 * * 0'` | every Sunday midnight |

```yaml
on:
  schedule:
    - cron: '30 2 * * *'     # always quote it, a bare * breaks YAML
```

Rules: quote the string, UTC only, minimum 5 minutes, and the file must be on the default branch.

**Manual (`workflow_dispatch`, `repository_dispatch`).** Production is sensitive, so a manual trigger works as the human approval before a deploy.

| | `workflow_dispatch` | `repository_dispatch` |
|---|---|---|
| How | "Run workflow" button in GitHub, or `gh workflow run manual.yml` | HTTP `POST /repos/{owner}/{repo}/dispatches` from an outside system (Slack bot, Jira, monitoring) |
| Input form | `inputs:` (choice, boolean, text) | JSON in `github.event.client_payload` |

```yaml
on:
  workflow_dispatch:
    inputs:
      environment:
        type: choice
        options: [staging, production]
        default: staging
      verify_rag_ingestion:
        type: boolean
        default: true
```

### 2.4 Actions (`uses`, `with`, `env`)

An **action** is a reusable step, so I don't copy-paste long scripts (install Python, cache, log in to a registry).

```
uses: actions/setup-python@v5
      \_____/ \__________/ \/
       owner      repo      ref (tag, branch or commit SHA)
```

* `actions/...` = made by GitHub itself. Other owners (`docker/`, `google-github-actions/`) = partners or community.
* Pin the `@ref` so an upstream change can't silently break my pipeline.

| Keyword | Role | Like |
|---|---|---|
| `uses` | which action to run | calling a function |
| `with` | inputs for that action | function arguments |
| `env` | environment variables for the step | `export KEY=value` |

```yaml
- uses: actions/setup-python@v5
  with:
    python-version: "3.11"
    cache: "pip"
  env:
    PIP_DISABLE_PIP_VERSION_CHECK: "1"
```

**Docker and Actions (two ways):**

```yaml
jobs:
  hybrid_job:                         # 1. one step inside a container
    runs-on: ubuntu-latest
    steps:
      - uses: docker://alpine:3.14    # public image only
        with:
          entrypoint: /bin/sh
          args: -c "echo inside alpine"
      - run: echo "back on the Ubuntu VM"

  dockerized_job:                     # 2. whole job inside a container
    runs-on: ubuntu-latest
    container:
      image: node:20-alpine
      # private image: image: ghcr.io/org/image:latest
      # credentials: { username: ${{ secrets.DOCKER_USERNAME }}, password: ${{ secrets.DOCKER_PASSWORD }} }
    steps:
      - run: node --version
```

### 2.5 Matrix (`strategy`)

Run the same job for several values without copy-pasting it.

| Keyword | Default | Effect |
|---|---|---|
| `matrix` | none | one parallel job per value, read with `${{ matrix.key }}` |
| `fail-fast` | `true` | if one job fails, cancel the rest. `false` lets all finish so I see every result |
| `max-parallel` | unlimited | cap how many matrix jobs run at once (rate limits, quota) |

```yaml
test:
  runs-on: ubuntu-latest
  strategy:
    fail-fast: false
    max-parallel: 2
    matrix:
      python-version: ["3.11", "3.12", "3.13"]
  steps:
    - uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
```

### 2.6 Expressions, `if:` and `env:` scopes

`${{ ... }}` is evaluated at runtime. It supports `==`, `!=`, `&&`, `||`, `!` and functions like `contains()`, `startsWith()`, `success()`, `failure()`, `always()`.

**`if:`** on a job or step skips it when false:

```yaml
- name: Production gate
  if: ${{ inputs.environment == 'production' && github.ref_name == 'main' }}
  run: echo "approved by ${{ github.actor }}"
- name: Alert
  if: ${{ failure() }}        # only when an earlier step failed
  run: echo "failed on ${{ github.ref_name }}"
- name: Cleanup
  if: ${{ always() }}         # runs even after a failure
  run: echo "cleanup"
```

**`env:` scopes.** The narrowest scope wins.

```yaml
env:                          # workflow: every job and step
  SERVICE_DIR: studentgator-rag-service
jobs:
  test:
    runs-on: ubuntu-latest
    env:                      # job: all steps in this job
      LOG_LEVEL: INFO
    steps:
      - env:                  # step: this step only
          STEP_TAG: unit-test
        run: echo "$SERVICE_DIR $LOG_LEVEL ${{ env.STEP_TAG }}"
```

**Useful contexts:**

| Expression | Value |
|---|---|
| `inputs.<name>` | value chosen in `workflow_dispatch` |
| `matrix.<key>` | current matrix value |
| `github.event_name` | `push`, `pull_request`, `workflow_dispatch` |
| `github.ref_name` | branch or tag name |
| `github.actor` | user who triggered the run |
| `github.sha` | commit hash (handy as a Docker image tag) |
| `runner.os` | `Linux` |

### 2.7 Workflow commands and passing values between steps

`echo "::command::message"` talks to the runner. It creates annotations in the run summary (and on the file/line in a PR).

| Command | Result |
|---|---|
| `::warning::` | yellow warning |
| `::error::` | red error (add `exit 1` to fail the step) |
| `::notice::` | blue info |
| `::group::title` ... `::endgroup::` | collapsible log section |
| `::add-mask::value` | hides the value as `***` in logs |
| `::debug::` | only shown when step debug is on |

```bash
echo "::warning file=studentgator-rag-service/src/schemas.py,line=54,title=Deprecated::old alias still used"
```

**Passing a value to the next step.** Each `run:` step starts a new shell, so a plain variable set in step 1 is gone in step 2. I write it to `$GITHUB_ENV` (or `$GITHUB_OUTPUT`) instead.

```yaml
- run: echo "deprecated=true" >> $GITHUB_ENV
- run: |
    if [[ "$deprecated" == "true" ]]; then
      echo "::warning::deprecated code found in step 1"
    fi
```

| Method | Set in step 1 | Read in step 2 |
|---|---|---|
| plain variable | `x=1` | not available |
| `$GITHUB_ENV` | `echo "x=1" >> $GITHUB_ENV` | `$x` or `${{ env.x }}` |
| `$GITHUB_OUTPUT` | `echo "x=1" >> $GITHUB_OUTPUT` (step needs `id: s1`) | `${{ steps.s1.outputs.x }}` |

### 2.8 Secrets management

**Problem:** the pipeline needs API keys (Gemini, OpenAI, LangSmith) and cloud credentials. If they are committed in the repo (`.env`, a hardcoded key, a service account JSON), anyone with repo access, every fork and the git history can read them.

**Mental model:** a safe in the office wall. Workflows can use what is inside, but nobody can read it back out.

* **GitHub Secrets** are stored encrypted. Set them in *Settings > Secrets and variables > Actions*. Read them with `${{ secrets.NAME }}`.
* GitHub **masks** a secret as `***` if it appears in logs. `::add-mask::` does the same for values created at runtime.
* Secrets are **not passed to workflows triggered from forked PRs**, so strangers can't steal them by opening a PR.
* **Variables** (`${{ vars.NAME }}`) are for non-sensitive config such as a region or project id.

| Scope | Visible to | Use for |
|---|---|---|
| Repository secret | all workflows in the repo | `GEMINI_API_KEY`, `LANGSMITH_API_KEY` |
| Environment secret (`staging`, `production`) | only jobs with `environment: <name>` | production credentials, can require reviewer approval |
| Organization secret | selected repos in the org | shared keys |

```yaml
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production          # uses production secrets and waits for required reviewers
    steps:
      - name: Use the secret through env, not inline in the command
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
        run: python -m src.ingestion
```

Rules:
* Pass a secret through `env:` instead of writing it inside `run:`.
* Never `echo` a secret, and never commit `.env` (keep it in `.gitignore`, keep `.env.example` with fake values).
* A GitHub Environment with **required reviewers** gives the manual approval gate from 2.3 without writing extra logic.
* For GCP, a service account JSON stored as a secret works but is a long-lived key. **Workload Identity Federation** (`permissions: id-token: write`) lets GitHub log in with a short-lived token and no stored key. That is the better option.
* At runtime on Cloud Run, keep keys in **Secret Manager** and mount them as environment variables (`--set-secrets`), not baked into the image.

### 2.9 Deployment strategies: rolling vs blue/green

**Problem:** a deploy replaces the running version. If the new version is broken, users see errors. The strategy decides how risky the switch is.

**Rolling:** replace instances one by one (or in small batches) with the new version. Old and new run side by side for a while.

```
v1 v1 v1 v1  ->  v2 v1 v1 v1  ->  v2 v2 v1 v1  ->  v2 v2 v2 v2
```

**Blue/Green:** keep the live environment (blue) untouched. Deploy v2 to a full copy (green) and test it there. When it looks good, switch all traffic to green in one step. Rolling back is switching traffic back to blue.

```
users -> [BLUE v1]  (live)         users -> [BLUE v1]  (idle, kept for rollback)
         [GREEN v2] (tested)  ==>          [GREEN v2] (live)
```

| | Rolling | Blue/Green |
|---|---|---|
| Extra infrastructure | little, reuses the same instances | a full second environment during the switch (roughly double cost) |
| Downtime | none if health checks pass | none, traffic switches at once |
| Rollback | slow, roll the update back batch by batch | instant, point traffic back to blue |
| Risk | v1 and v2 serve users together, so versions must be compatible (API, DB schema) | new version is fully tested before users touch it |
| Use for | frequent small releases of stateless services | risky releases, big changes, when instant rollback matters |

**On Cloud Run:** every deploy creates a new **revision**. I can deploy it with no traffic, test it on its own URL (blue/green style), then move traffic, or split it gradually (for example 10% then 100%, close to a canary/rolling rollout).

```bash
gcloud run deploy rag-api --image $IMAGE --no-traffic --tag green        # deploy green, 0% traffic
gcloud run services update-traffic rag-api --to-tags green=100           # switch traffic to green
gcloud run services update-traffic rag-api --to-revisions LATEST=10      # or send only 10% first
```

For StudentGator: a rolling or gradual rollout for small fixes, blue/green when the RAG pipeline changes (new chunking, new embedding model), because a bad answer set before deadlines is costly and rollback must be instant. A new embedding model also means the vector index must match, which makes mixing v1 and v2 unsafe.

### 2.10 Monitoring pipelines

A pipeline nobody watches fails silently. Monitoring means seeing that runs pass, being alerted when they fail, and checking the deployed app after release.

| What to watch | How |
|---|---|
| Run status and history | *Actions* tab, `gh run list`, `gh run view <id> --log-failed` |
| Status at a glance | status badge in the README |
| Failure alerts | a step with `if: ${{ failure() }}` that sends a Slack/email message |
| Stuck or slow runs | `timeout-minutes` on a job, `concurrency` to cancel superseded runs |
| Deeper logs | re-run with debug logging, `::group::` for readable logs |
| Deployed app health | smoke test step after deploy, plus a scheduled health check |

```yaml
# README badge
# ![CI](https://github.com/Hamshary99/mohamed-ai-learning/actions/workflows/push-pr.yml/badge.svg)

jobs:
  deploy:
    runs-on: ubuntu-latest
    timeout-minutes: 15                       # fail instead of hanging
    steps:
      - run: echo "deploy..."
      - name: Smoke test the new version
        run: curl --fail --retry 5 --retry-delay 5 https://<service-url>/api/v1/health
      - name: Alert on failure
        if: ${{ failure() }}
        run: echo "::error::Deploy failed for ${{ github.sha }}"   # replace with a Slack/email action

# scheduled uptime check (separate workflow)
on:
  schedule:
    - cron: '*/30 * * * *'
```

* If the smoke test fails, the pipeline is red right away and I can roll back (blue/green: switch traffic back).
* Pipeline monitoring watches the CI/CD run. Application monitoring (Cloud Logging, LangSmith traces) watches the live service. I need both.

---

## 3. Trade-Offs & Comparisons (When to Use What)

**Delivery vs Deployment**

| | Continuous Delivery | Continuous Deployment |
|---|---|---|
| Production release | human approves | automatic |
| Pick it when | production is sensitive or regulated | test suite is strong and I want fast releases |

**Which trigger for which job**

| Trigger | Fires | Use for |
|---|---|---|
| `pull_request` | before merge | CI gate: lint and tests, no deploy |
| `push` to `main` | after merge | build image and release |
| `schedule` | on a timer | nightly RAG re-ingestion, health checks |
| `workflow_dispatch` | button / CLI | manual approval before production |
| `repository_dispatch` | outside API call | external systems starting a workflow |

**Steps vs jobs**

| | Steps in one job | Separate jobs |
|---|---|---|
| VM | one, installs once | one each, repeat checkout and install |
| Order | always sequential | parallel, or sequential with `needs:` |
| Use for | lint then pytest (same environment) | independent checks, or stages that must gate each other (test -> build -> deploy) |

**`uses` vs `run`:** `uses` for setup that is long to script (checkout, setup-python, registry login). `run` for my own commands (`pytest`, `ruff`) so CI runs the same commands I run locally.

**Step-level vs job-level Docker:** `uses: docker://` isolates one step (public image only). `container:` runs the whole job in the image (public or private).

**Where to keep credentials**

| Option | Good | Bad | Use for |
|---|---|---|---|
| Committed `.env` / hardcoded | easy | leaked to everyone and stays in git history | never |
| GitHub repository secret | encrypted, masked in logs | same value for every workflow in the repo | API keys for CI |
| GitHub environment secret + reviewers | scoped to production, adds approval | slightly more setup | production deploys |
| Workload Identity Federation | no stored key, short-lived token | needs one-time GCP setup | logging in to GCP from Actions |
| Secret Manager (Cloud Run) | key never baked into the image, rotatable | GCP setup, small cost | runtime secrets of the deployed app |

---

## 4. Production Code Reference

One workflow that combines the pieces above for `studentgator-rag-service` (tests live in `studentgator-rag-service/tests/test_service.py`). The Docker and Artifact Registry parts use placeholder names until the Dockerfile and GCP project exist (Day 2 and Day 4).

Pipeline order: `lint` and `build` in parallel -> `test` and `ingest` in parallel -> `build-push` (image to Artifact Registry) -> `deploy` (manual, with secrets and a smoke test).

```yaml
name: StudentGator CI/CD

on:
  push:
    branches: [main]
    paths: ["studentgator-rag-service/**"]
  pull_request:
    branches: [main]
  schedule:
    - cron: '30 2 * * *'            # nightly, UTC
  workflow_dispatch:
    inputs:
      environment:
        type: choice
        options: [staging, production]
        default: staging

env:
  SERVICE_DIR: studentgator-rag-service

jobs:
  lint:                              # linting, runs in parallel with build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7.0.1
      - run: pip install ruff
      - run: ruff check $SERVICE_DIR

  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7.0.1
      - run: python -m py_compile $SERVICE_DIR/src/*.py

  test:                              # runs in parallel with ingest
    needs: [build, lint]
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v7.0.1
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install pytest pydantic langchain-core langgraph
      - run: pytest $SERVICE_DIR/tests/test_service.py -k "not rag" -v

  ingest:                            # PDF load, chunk, index test
    needs: [build, lint]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7.0.1
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install pytest pydantic pydantic-settings langchain-core langchain-community langchain-text-splitters langgraph pypdf
      - run: pytest $SERVICE_DIR/tests/test_service.py -k "rag" -v

  build-push:                        # build the image and push it to Artifact Registry
    needs: [test, ingest]
    if: ${{ github.event_name == 'push' && github.ref_name == 'main' }}   # never from PRs
    runs-on: ubuntu-latest
    outputs:
      image: ${{ steps.meta.outputs.image }}   # hand the image name to the deploy job
    steps:
      - uses: actions/checkout@v7.0.1
      - uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}    # or Workload Identity Federation
      - uses: google-github-actions/setup-gcloud@v2
      - run: gcloud auth configure-docker ${{ vars.GCP_REGION }}-docker.pkg.dev
      - id: meta
        run: echo "image=${{ vars.GCP_REGION }}-docker.pkg.dev/${{ vars.GCP_PROJECT }}/studentgator/rag-api:${{ github.sha }}" >> $GITHUB_OUTPUT
      - run: |
          docker build -t ${{ steps.meta.outputs.image }} $SERVICE_DIR
          docker push ${{ steps.meta.outputs.image }}

  deploy:                            # only when I trigger it manually (approval gate)
    needs: build-push
    if: ${{ github.event_name == 'workflow_dispatch' }}
    runs-on: ubuntu-latest
    environment: ${{ inputs.environment }}   # environment secrets and optional required reviewers
    timeout-minutes: 15
    steps:
      - uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}
      - uses: google-github-actions/setup-gcloud@v2
      - name: Deploy new revision with no traffic (blue/green style)
        run: |
          gcloud run deploy rag-api \
            --image ${{ needs.build-push.outputs.image }} \
            --region ${{ vars.GCP_REGION }} \
            --set-secrets GEMINI_API_KEY=gemini-api-key:latest \
            --no-traffic --tag green
      - name: Smoke test the green revision
        run: curl --fail --retry 5 --retry-delay 5 https://green---rag-api-<hash>-uc.a.run.app/api/v1/health
      - name: Switch traffic to green
        run: gcloud run services update-traffic rag-api --region ${{ vars.GCP_REGION }} --to-tags green=100
      - name: Alert on failure
        if: ${{ failure() }}
        run: echo "::error::Deploy of ${{ github.sha }} failed, traffic still on the previous revision"
```

What each added part covers: `lint` (ruff), `build-push` (docker build and push), `deploy` (manual approval, secrets, blue/green switch, smoke test, failure alert). The image name travels between jobs through `outputs` and `needs.<job>.outputs`, because jobs run on different VMs.

---

## 5. Whiteboard / Interview Talking Points

* **CI vs Delivery vs Deployment:** CI lints and tests every push on a clean VM. Delivery also packages the build and verifies it in a production-like environment, then waits for approval. Deployment removes the approval and releases automatically.
* **Jobs vs steps:** jobs run in parallel on separate VMs unless chained with `needs:`. Steps run in order on one VM. Chaining test -> build -> deploy makes failures easy to locate and stops bad code from reaching deploy.
* **Triggers in production:** `pull_request` gates merges, `push` to `main` releases, `schedule` handles nightly RAG ingestion, and `workflow_dispatch` is the manual approval before touching production.
* **Secrets:** never in the repo or the image. GitHub Secrets (masked, not given to fork PRs) for CI, environment secrets with required reviewers for production, Workload Identity Federation instead of long-lived GCP keys, and Secret Manager for the running Cloud Run service.
* **Rolling vs blue/green:** rolling replaces instances gradually, cheap but slow to roll back and both versions serve users at once. Blue/green keeps the old environment live while the new one is tested, then switches traffic, so rollback is instant but costs a second environment. I'd use blue/green when the RAG pipeline or embedding model changes.
* **Monitoring:** watch the pipeline (run history, failure alerts, `timeout-minutes`, status badge) and the result (post-deploy smoke test and a scheduled health check on `/api/v1/health`).

---

## Quick Review: Common Traps

| Trap | Reality |
|---|---|
| CD = Continuous Development | CD = Delivery or Deployment |
| `.github/workflow/` works | folder must be `workflows` |
| `on: pull:` | the event is `pull_request` |
| Jobs share files | each job has its own VM, only steps in one job share files |
| Runner already has my code | it is empty, run `actions/checkout` first |
| A variable set in step 1 exists in step 2 | each step has its own shell, use `$GITHUB_ENV` |
| Cron runs in my local time | cron is UTC |
| Unquoted `*/5 * * * *` | quote it, or YAML fails |
| Put the API key in the workflow or `.env` in git | use `${{ secrets.NAME }}`, keep `.env` out of git |
| Blue/green and rolling are the same | rolling swaps instances gradually, blue/green runs two full environments and switches traffic once |
| Pipeline is green so the app is healthy | add a smoke test after deploy and a scheduled health check |

## Day 3 Checklist (from the learning plan)

| Learning plan item | Where |
|---|---|
| What is CI/CD, integration vs deployment | 1 (CI, Delivery, Deployment) |
| GitHub Actions: automated testing | 2.1, 2.2, 4 (`test`, `ingest`) |
| GitHub Actions: linting | 4 (`lint` job with ruff) |
| GitHub Actions: build and deployment pipelines | 2.2, 4 (`build-push`, `deploy`) |
| Secrets management | 2.8, 3 |
| Rolling vs blue/green deployment | 2.9, 3 |
| Monitoring pipelines | 2.10 |
| Deliverable: pipeline that lints, tests, builds and pushes to Artifact Registry | 4 |

Extra topics beyond the plan: triggers, cron, matrix, expressions, workflow commands.

<details>
<summary>Raw notes from the lecture (unedited)</summary>

```text
What's CI/CD?

CI - Contiunous Integration
Checks the code quality against the rest of the application, in isolation. Triggerd automatically on every git push. In brief it ensures that the code works in isolation or in the local env.

CD - Continous Developement
Ensure that the code can be delivered to encironment that's like the production env not the prod env itself, like in docker or any kernel before pushing to prod

What else can github actions can do? more than just ci/cd
Github is a cloud based platform for version control and collaboration. 

Github actions? some pipeline functions or workflow tasks that are ready to go according to a trigger
so any yml file in the path .github/workflow will be executed for every push, workflow works by a trigger which is usually a push command
another trigger could be issue creation, workdflow got some "jobs" and they execute steps or whatever. Jobs run in parallel so maybe we have to tell what jobs work in sequence
Steps are what run on sequence, 
```

</details>
