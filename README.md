# mohamed-ai-learning
## StudentGator AI Engineering Internship: From ML to Production

A 4-week learning program covering Machine Learning Foundations, Deep Learning, Transformers, RAG, FastAPI, Docker, CI/CD, and GCP.

Organized by **week → day** (4 weeks × 5 days). Shared Python env stays at the root.

## Layout

```
ML/
  .venv/                      # one env for everything
  requirements.txt
  week1/
    day1/                     # current work (Ch1 + Ch2)
      notebooks/
      notes/
      data/
    day2/ … day5/             # empty — ready
  week2/ … week4/
    day1/ … day5/
```

## How to run notebooks

1. Open a notebook under `weekX/dayY/notebooks/`.
2. Kernel → **ML Sandbox (.venv)**.
3. Run with `Shift+Enter`.

From a notebook, day’s data is:

```python
pd.read_csv("../data/your_file.csv")
```

## What goes where

| When | Put it in |
|------|-----------|
| New day | `weekN/dayM/notebooks/` |
| Notes | `weekN/dayM/notes/` |
| Datasets | `weekN/dayM/data/` |
