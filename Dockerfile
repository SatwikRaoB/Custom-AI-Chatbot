# ---------- Runtime ----------
FROM python:3.10-slim
WORKDIR /app

# ENV (no PORT — Cloud Run injects 8080)
ENV DATA_DIR=/app/backend/data \
    MANUALS_DIR=/app/backend/data/manuals \
    INDEX_DIR=/app/backend/vector_store/support_index

RUN useradd -m appuser

RUN mkdir -p ${DATA_DIR} ${MANUALS_DIR} ${INDEX_DIR} \
    && chown -R appuser:appuser /app

COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CRITICAL: Copy backend/ as a folder
COPY --chown=appuser:appuser backend backend/

# Copy manuals into the correct path
COPY --chown=appuser:appuser backend/data/manuals/ ${MANUALS_DIR}/

# Copy pre-built index
COPY --from=builder --chown=appuser:appuser /app/backend/vector_store/support_index/ ${INDEX_DIR}/

# Copy HF cache
COPY --from=builder --chown=appuser:appuser /root/.cache/huggingface /home/appuser/.cache/huggingface

USER appuser

HEALTHCHECK CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "backend.main:app", "--bind", "0.0.0.0:${PORT}"]
