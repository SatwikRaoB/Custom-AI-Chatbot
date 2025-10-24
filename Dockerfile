# ---------- Builder ----------
FROM python:3.10-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libblas3 libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code + data
COPY backend/. .
COPY backend/data/manuals/ /app/backend/data/manuals/

# Build index (downloads + caches model)
COPY build_index.py .
RUN python build_index.py

# ---------- Runtime ----------
FROM python:3.10-slim
WORKDIR /app

ENV DATA_DIR=/app/backend/data \
    MANUALS_DIR=/app/backend/data/manuals \
    INDEX_DIR=/app/backend/vector_store/support_index \
    PORT=8000

RUN useradd -m appuser

RUN mkdir -p ${DATA_DIR} ${MANUALS_DIR} ${INDEX_DIR} \
    && chown -R appuser:appuser /app

# Copy venv
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy app + data
COPY --chown=appuser:appuser backend/. .
COPY --chown=appuser:appuser backend/data/manuals/ ${MANUALS_DIR}/

# Copy pre-built index
COPY --from=builder --chown=appuser:appuser /app/backend/vector_store/support_index/ ${INDEX_DIR}/

# --- COPY HF CACHE (critical!) ---
COPY --from=builder --chown=appuser:appuser /root/.cache/huggingface /home/appuser/.cache/huggingface

USER appuser
EXPOSE ${PORT}

CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "backend.main:app", "--bind", "0.0.0.0:${PORT}"]
