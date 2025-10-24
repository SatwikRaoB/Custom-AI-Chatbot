# ---------- Streamlit Frontend ----------
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY frontend/ /app/

# Environment for Streamlit + Cloud Run
ENV PORT=8000 \
    STREAMLIT_SERVER_PORT=8000 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8080

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Launch Streamlit app
CMD ["sh", "-c", "streamlit run app.py --server.port $PORT --server.address 0.0.0.0"]
