# --- Stage 1: Build Dependencies ---
# Use the slim image for a much smaller base layer and faster downloads/copies
FROM python:3.10-slim as builder

WORKDIR /app

# Install OS dependencies necessary for compilation (build-essential, linear algebra libs for numpy/faiss)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libblas3 \
    liblapack3 \
    libopenblas-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy ONLY the requirements file to leverage Docker layer caching
COPY backend/requirements.txt .

# Install Python packages into the venv
RUN pip install --no-cache-dir -r requirements.txt


# --- Stage 2: Final Production Image ---
FROM python:3.10-slim

WORKDIR /app

# ... (Standard ENV vars)

# Application-specific env vars (Define ABSOLUTE paths)
ENV DATA_DIR /app/data
ENV MANUALS_DIR /app/data/manuals
ENV INDEX_DIR /app/vector_store/support_index

# Copy the venv from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Explicitly set PATH and PYTHONPATH
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/opt/venv/lib/python3.10/site-packages"

# --- CRITICAL FIX: Ensure ALL directories exist and are absolute, created by root ---
RUN mkdir -p ${DATA_DIR}
RUN mkdir -p ${MANUALS_DIR}
RUN mkdir -p ${INDEX_DIR}

# Copy your application code
COPY backend/. .

# Copy data/index files into the absolute directories
COPY data/manuals/ ${MANUALS_DIR}/
COPY vector_store/support_index/ ${INDEX_DIR}/

# Set log level
ENV LOG_LEVEL=INFO

# --- CRITICAL FIX: Grant Permissions to the Non-Root User ---
RUN useradd --create-home --shell /bin/bash appuser
# This fixes PermissionError [Errno 13] by granting ownership of everything under /app
RUN chown -R appuser:appuser /app
# Switch to the non-root user
USER appuser

EXPOSE ${PORT}

# The command to run the app.
CMD ["sh", "-c", "/opt/venv/bin/gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:${PORT}"]
