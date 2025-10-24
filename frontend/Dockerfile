# ---------- Frontend Dockerfile ----------
FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . /app/

# Streamlit uses $PORT injected by Cloud Run
ENV PORT=8080
ENV BACKEND_URL=${BACKEND_URL}

# Enable Cloud Logging (stdout + stderr are automatically captured by Cloud Run)
ENV STREAMLIT_LOG_LEVEL=info

EXPOSE 8080

# Run Streamlit on the injected port
CMD ["sh", "-c", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"]
