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

# Standard environment variables for Python and Cloud Run
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT 8080

# Application-specific env vars
ENV DATA_DIR /app/data
ENV MANUALS_DIR /app/data/manuals
ENV INDEX_DIR /app/vector_store/support_index

# Copy the venv from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Explicitly set PATH to the venv bin directory
ENV PATH="/opt/venv/bin:$PATH"

# Copy your application code and data
COPY backend/. .
COPY data/manuals/ ./data/manuals/

# --- Index Copy (As requested by the user) ---
# This assumes the index is pre-built locally and .dockerignore has been updated
COPY vector_store/support_index/ /app/vector_store/support_index/
# --- End Index Copy ---

# Set log level
ENV LOG_LEVEL=INFO

# --- Security Best Practice ---
# Create a new, non-root user
RUN useradd --create-home --shell /bin/bash appuser
# Give ownership to the new user
RUN chown -R appuser:appuser /app
# Switch to this non-root user
USER appuser

EXPOSE ${PORT}

# The command to run the app in production.
# Explicitly uses /opt/venv/bin/gunicorn for robustness.
CMD ["sh", "-c", "/opt/venv/bin/gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:${PORT}"]
