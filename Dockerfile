# 1. Start with an official Python image
FROM python:3.10-slim

# 2. Set the working directory for the application code
WORKDIR /app

# 3. Copy only the requirements file first to cache packages
# Assumes requirements.txt is in the 'backend' folder relative to the Dockerfile (at root)
COPY backend/requirements.txt .

# 4. Install all the Python packages
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the Python application code 
# from the 'backend' folder into the container's /app directory
COPY backend/. .

# --- NEW: Copy Data and Vector Store ---
# 6. Copy the PDF manuals into the container. 
# Create the /app/data/manuals directory structure inside the container.
COPY data/manuals/ /app/data/manuals/

# 7. Copy the pre-built FAISS index into the container.
# Create the /app/vector_store/support_index directory structure.
COPY vector_store/support_index/ /app/vector_store/support_index/
# --- End NEW Section ---

# 8. Expose the port your app will run on
EXPOSE 8000 # Or 8080 if you switched back

# 9. The command to run your app
# Gunicorn runs from /app, finds main:app, paths in main.py are relative to /app
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"] # Or :${PORT:-8080}

