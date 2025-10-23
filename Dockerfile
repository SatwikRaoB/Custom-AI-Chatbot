# --- Stage 1: Build Dependencies ---
# Use the python:3.10 base image
FROM python:3.10-slim as builder

WORKDIR /app

# Install 'build-essential' for packages that need to compile C code (like faiss-cpu)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /opt/venv

# Add the venv to the PATH for this build stage
ENV PATH="/opt/venv/bin:$PATH"

# Copy ONLY the requirements file from your 'backend' folder
COPY backend/requirements.txt .

# Upgrade pip within the venv
RUN pip install --no-cache-dir --upgrade pip
# Install all Python packages into the venv
RUN pip install --no-cache-dir -r requirements.txt


# --- Stage 2: Final Production Image ---

# Start from a fresh python:3.10 image
FROM python:3.10-slim

# Set the final working directory
WORKDIR /app

# Standard Python environment variables for production containers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Set the default port. Cloud Run will inject its own $PORT variable.
ENV PORT 8080

# Set the application-specific env vars to match your main.py
ENV DATA_DIR /app/data
ENV MANUALS_DIR /app/data/manuals
ENV INDEX_DIR /app/vector_store/support_index

# Copy the entire venv (with all packages) from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Add the venv to the PATH for the final image
ENV PATH="/opt/venv/bin:$PATH"

# Force Python to look for modules in the venv's site-packages
ENV PYTHONPATH="/opt/venv/lib/python3.10/site-packages"

# Copy your application code from the 'backend' folder into /app
COPY backend/. .

# Copy your PDF files from the 'data' folder into the image
COPY data/manuals/ ./data/manuals/

# --- THIS IS THE CHANGED SECTION ---
# As requested, copy the local vector store instead of building it.
# This assumes you have 'vector_store/support_index' in your project root.
COPY vector_store/support_index/ /app/vector_store/support_index/
# --- END CHANGE ---

# Set log level for the app
ENV LOG_LEVEL=INFO

# --- Security Best Practice ---
# Create a new, non-root user to run the application
RUN useradd --create-home --shell /bin/bash appuser
# Give this user ownership of all app files (code, data, and index)
RUN chown -R appuser:appuser /app
# Switch to this non-root user
USER appuser

# Expose the port the container will listen on
EXPOSE ${PORT}

# The command to run the app.
CMD ["sh", "-c", "/opt/venv/bin/gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:${PORT}"]
