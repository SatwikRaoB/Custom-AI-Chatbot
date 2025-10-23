# --- Stage 1: Build Dependencies ---
# Use the python:3.10 base image
FROM python:3.10 as builder

WORKDIR /app

# Install build-essential for any C extensions (like in faiss-cpu)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy ONLY the requirements file from your backend folder
COPY backend/requirements.txt .

# Install Python dependencies into the venv
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


# --- Stage 2: Final Production Image ---
FROM python:3.10

WORKDIR /app

# Set env vars for Python and GCP Cloud Run
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Cloud Run injects the PORT env var (default 8080)
ENV PORT 8080

# Set env vars for your application, matching main.py
# These paths are now explicit inside the container
ENV DATA_DIR /app/data
ENV MANUALS_DIR /app/data/manuals
ENV INDEX_DIR /app/vector_store/support_index

# Copy the virtual environment from the builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy your application code from the 'backend' folder
# This brings in main.py
COPY backend/. .

# --- PDF and Index Setup ---

# Copy your PDF manuals from the 'data' folder
COPY data/manuals/ ./data/manuals/

# Create the directory for the index
RUN mkdir -p ${INDEX_DIR}

# --- Pre-build the FAISS index ---
# This runs 'load_vector_store_sync' during the build
# This is CRITICAL for fast startups on Cloud Run
ENV LOG_LEVEL=INFO
RUN echo "--- Building FAISS index for Docker image ---" && \
    python -c " \
import logging; \
logging.basicConfig(level='INFO'); \
from main import load_vector_store_sync; \
vs = load_vector_store_sync(); \
assert vs is not None, 'FAISS Index build FAILED. Check PDFs in data/manuals.'; \
print('--- FAISS index built successfully ---')"

# --- Security: Run as non-root user ---
# Create a dedicated user
RUN useradd --create-home --shell /bin/bash appuser
# Give that user ownership of the app directory
RUN chown -R appuser:appuser /app
# Switch to the new user
USER appuser

# Expose the port Cloud Run will use
EXPOSE ${PORT}

# Run the application using Gunicorn (as in your old file)
# This command correctly uses the $PORT variable provided by Cloud Run
CMD ["sh", "-c", "gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:${PORT}"]
