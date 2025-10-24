# ---------- Builder ----------
FROM python:3.10-slim AS builder
WORKDIR /app

# Build-time dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libblas3 libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python deps
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (as a package)
COPY backend backend/
COPY backend/data/manuals/ /app/backend/data/manuals/

# Build index + cache HF model
COPY build_index.py .
RUN python build_index.py

# ---------- Runtime ----------
FROM python:3.10-slim
WORKDIR /app

# ----- ENV (NO PORT – Cloud Run injects it) -----
ENV DATA_DIR=/app/backend/data \
    MANUALS_DIR=/app/backend/data/manuals \
    INDEX_DIR=/app/backend/vector_store/support_index

# Non-root user
RUN useradd -m appuser

# Create required directories
RUN mkdir -p ${DATA_DIR} ${MANUALS_DIR} ${INDEX_DIR} \
    && chown -R appuser:appuser /app

# Copy venv from builder
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the whole backend package
COPY --chown=appuser:appuser backend backend/

# Copy manuals (runtime copy – optional, already in builder)
COPY --chown=appuser:appuser backend/data/manuals/ ${MANUALS_DIR}/

# Copy pre-built FAISS index
COPY --from=builder --chown=appuser:appuser \
     /app/backend/vector_store/support_index/ ${INDEX_DIR}/

# Copy cached HuggingFace model
COPY --from=builder --chown=appuser:appuser \
     /root/.cache/huggingface /home/appuser/.cache/huggingface

USER appuser
EXPOSE 8000
# Health-check (uses $PORT, which will be 8000)
HEALTHCHECK CMD curl -f http://localhost:${PORT}/health || exit 1

# Gunicorn – binds to whatever $PORT is (8000 on Cloud Run)
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", \
     "backend.main:app", "--bind", "0.0.0.0:8000"]
