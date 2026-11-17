# Stage 1: builder - temporary environment to install dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Stop python from writing .pyc files and keep logs unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create clean virtual environment (all modern ML packages have pre-built wheels, no gcc needed)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install lightweight CPU-only PyTorch using BuildKit cache mount
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --timeout 100 --retries 10 torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining libraries with cache mount
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
    chown -R appuser:appgroup /tmp/huggingface /tmp/torch /app

# Copy app code, static chat UI, and vector DB directly with appuser ownership (no duplicate layer)
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup data/ ./data/
COPY --chown=appuser:appgroup static/ ./static/

USER appuser

EXPOSE 8080

# exec forwards OS signals (SIGTERM) directly to uvicorn so Cloud Run can shut down cleanly
CMD exec uvicorn src.main:app --host 0.0.0.0 --port ${PORT}
