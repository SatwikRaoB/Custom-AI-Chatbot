# --- Stage 1: Build Dependencies ---

# Use the python:3.10 base image
FROM python:3.10 as builder

# Set the working directory for this build stage
WORKDIR /app

# Install 'build-essential' for packages that need to compile C code (like faiss-cpu)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /opt/venv

# Add the venv to the PATH for this stage
ENV PATH="/opt/venv/bin:$PATH"

# Copy ONLY the requirements file from your 'backend' folder
COPY backend/requirements.txt .

# Upgrade pip within the venv
RUN pip install --no-cache-dir --upgrade pip
# Install all Python packages into the venv
RUN pip install --no-cache-dir -r requirements.txt


# --- Stage 2: Final Production Image ---

# Start from a fresh python:3.10 image
FROM python:3.10

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

# Copy your application code from the 'backend' folder into /app
COPY backend/. .

# --- PDF and Index Setup ---

# Copy your PDF files from the 'data' folder into the image
COPY data/manuals/ ./data/manuals/

# Create the directory where the FAISS index will be built
RUN mkdir -p ${INDEX_DIR}

# Set the log level for the build script
ENV LOG_LEVEL=INFO

# --- KEY FIX #1 (ModuleNotFoundError) ---
# Pre-build the FAISS index *during the build*.
# We explicitly use '/opt/venv/bin/python' to ensure it finds all installed packages.
RUN echo "--- Building FAISS index for Docker image ---" && \
    /opt/venv/bin/python -c "import logging; logging.basicConfig(level='INFO'); from main import load_vector_store_sync; vs = load_vector_store_sync(); assert vs is not None, 'FAISS Index build FAILED. Check PDFs in data/manuals.'; print('--- FAISS index built successfully ---')"

# --- Security Best Practice ---

# Create a new, non-root user to run the application
RUN useradd --create-home --shell /bin/bash appuser
# Give this user ownership of the entire /app directory
# (This includes the code, the data, and the newly built index)
RUN chown -R appuser:appuser /app
# Switch to this non-root user
USER appuser

# Expose the port the container will listen on
EXPOSE ${PORT}

# --- KEY FIX #2 (Robustness) ---
# The command to run the app.
# We use 'sh -c' so that the $PORT variable is correctly used.
# We explicitly use '/opt/venv/bin/gunicorn' to be 100% sure it's found.
CMD ["sh", "-c", "/opt/venv/bin/gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:${PORT}"]
