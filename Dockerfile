# ---------- Builder ----------
FROM python:3.10-slim AS builder
WORKDIR /app

# Build tools + BLAS for FAISS / numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libblas3 liblapack3 libopenblas-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Virtual-env
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ---------- Runtime ----------
FROM python:3.10-slim

WORKDIR /app

# ---- ENV defaults (ALL inside /app) ----
ENV DATA_DIR=/app/data \
    MANUALS_DIR=/app/data/manuals \
    INDEX_DIR=/app/vector_store/support_index \
    LOG_LEVEL=INFO \
    PORT=8000

# ---- Non-root user (created early) ----
RUN useradd --create-home --shell /bin/bash appuser

# ---- Create *all* required directories and give ownership ----
RUN mkdir -p ${DATA_DIR} ${MANUALS_DIR} ${INDEX_DIR} \
    && chown -R appuser:appuser /app

# ---- Copy venv (owned by appuser) ----
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ---- Application code ----
COPY --chown=appuser:appuser backend/. .

# ---- Static data (PDF manuals) ----
COPY --chown=appuser:appuser data/manuals/ ${MANUALS_DIR}/

# ---- Pre-built index (optional – if you ship it) ----
COPY --chown=appuser:appuser vector_store/support_index/ ${INDEX_DIR}/

# ---- Switch to non-root ----
USER appuser

EXPOSE ${PORT}

# ---- Health-check (optional but recommended) ----
HEALTHCHECK CMD curl -f http://localhost:${PORT}/health || exit 1

# ---- Entrypoint ----
CMD ["sh", "-c", "gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:${PORT}"]
