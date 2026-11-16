# GitHub Actions Reference Examples (Week 4 Day 3)

This directory gathers the 7 concept workflows built during Week 4 Day 3 study. They serve as isolated syntax references for triggers, jobs, expressions, and workflow commands.

| Workflow File | Core Concepts Demonstrated | Key Keywords |
|---|---|---|
| `actions.yml` | Marketplace actions, step parameters, environment variables | `uses:`, `with:`, `env:` |
| `push-pr.yml` | Push and pull request triggers, branch filtering, path filtering | `on: push:`, `pull_request:`, `paths:` |
| `scheduled.yml` | Cron syntax and scheduled automation (UTC) | `on: schedule: cron:` |
| `manual.yml` | Manual execution with dynamic user inputs (strings, booleans, choices) | `workflow_dispatch: inputs:` |
| `jobs.yml` | Multi-job execution, sequential dependencies, matrix strategies | `jobs:`, `needs:`, `strategy: matrix:` |
| `commands.yml` | Workflow commands, log grouping, annotations, inter-step persistence | `::group::`, `::warning::`, `$GITHUB_ENV` |
| `main.yml` | Minimal hello-world workflow structure | `name:`, `on:`, `jobs:`, `steps:` |

Full explanations and trade-offs are documented in `week4/day3/notes/Github Actions.md`.
